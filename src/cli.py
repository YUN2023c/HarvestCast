# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""命令行入口。

用法（在项目根目录）：
    python -m src.cli init                          # 首次：拉取全部品种初始行情
    python -m src.cli init --variety corn,sugar     # 只拉指定品种
    python -m src.cli init --allow-demo             # 无网络时的离线演示（合成数据）
    python -m src.cli predict --variety corn
    python -m src.cli predict-all
    python -m src.cli prompt --variety corn --as-of 2026-09-30
    python -m src.cli extract --variety corn --json-file data/output/pasted.json
    python -m src.cli serve --port 8848
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import config, timeutil
from .data import ingest
from .llm import client as llm_client
from .llm import extract as llm_extract
from .llm import prompt_builder as pb
from .models import forecast as fc

MARKET_SOURCES = ["auto", "sina", "akshare", "demo"]


def _fmt_money(v, unit=""):
    try:
        return f"{float(v):,.0f}{unit}"
    except (TypeError, ValueError):
        return "-"


def _pick_varieties(text: str):
    if not text or text == "all":
        return config.variety_list()
    wanted = [x.strip() for x in text.split(",") if x.strip()]
    out, unknown = [], []
    for w in wanted:
        try:
            out.append(config.get_variety(w))
        except KeyError:
            unknown.append(w)
    if unknown:
        print(f"[warn] 未知品种已忽略: {', '.join(unknown)}", file=sys.stderr)
    return out


def cmd_init(args) -> None:
    config.ensure_dirs()
    varieties = _pick_varieties(args.variety)
    if not varieties:
        print("没有可处理的品种", file=sys.stderr)
        raise SystemExit(2)
    demo = bool(args.allow_demo or getattr(args, "source", "auto") == "demo")
    if demo:
        print("[demo] 使用合成数据（仅离线演示，禁止用于真实决策）")
    else:
        print(f"[init] 获取初始行情：{len(varieties)} 个品种（新浪通道，失败自动回退 AKShare）")
    if args.force:
        print("[init] --force：已存在的数据也会重新拉取")

    res = ingest.bootstrap(
        varieties, min_months=args.min_months, force=args.force,
        allow_demo=demo, horizon=args.horizon,
        retries_sina=getattr(args, "source", "auto") != "akshare",
        progress=lambda name: print(f"  ... {name}", flush=True),
    )

    print("-" * 68)
    print(f"{'品种':<10}{'状态':<8}{'月数':>6}{'来源':>10}  说明")
    for it in res["items"]:
        status = {"fetched": "已获取", "exists": "已存在", "error": "失败"}.get(it["status"], it["status"])
        print(f"{it['name']:<10}{status:<8}{it.get('rows', 0):>6}{str(it.get('source') or '-'):>10}  "
              f"{str(it.get('message') or '')[:54]}")
    print("-" * 68)
    print(f"成功 {res['ok']} / 跳过 {res['skipped']} / 失败 {res['failed']}（共 {res['total']}）")
    if res.get("hint"):
        print(res["hint"])
    if res["ok"] == 0 and res["skipped"] == 0:
        raise SystemExit(1)


def cmd_demo(args) -> None:
    """显式生成合成数据（离线演示/单测），不做真实拉取。"""
    config.ensure_dirs()
    varieties = _pick_varieties(args.variety)
    print("[demo] 生成合成数据（非真实行情，禁止用于真实决策）")
    for v in varieties:
        out = ingest.save_demo(v, years=args.years, seed=args.seed)
        print(f"  {v['name_cn']:<8}{v['id']:<12} -> {out['market']}")
    print(f"完成 {len(varieties)} 个品种")


def cmd_ingest(args) -> None:
    """兼容旧命令：等价于 init 单品种。"""
    args.allow_demo = args.source == "demo"
    cmd_init(args)


def cmd_prompt(args) -> None:
    config.ensure_dirs()
    v = config.get_variety(args.variety)
    as_of = args.as_of or timeutil.ym_str(timeutil.today_ym()) + "-01"
    out = llm_extract.export_prompt(v, as_of, config.OUTPUT_DIR / "prompts")
    print(f"提示词已导出: {out['path']}")
    print(f"模板 SHA256: {out['hash']}  schema: {out['schema_version']}")
    print("把它粘贴到任意联网大模型，要求返回 JSON；再用 extract 子命令回填。")


def cmd_extract(args) -> None:
    settings = config.load_settings()
    text = Path(args.json_file).read_text(encoding="utf-8") if args.json_file else (args.json_text or "")
    if not text:
        print("请用 --json-file 或 --json-text 提供 JSON", file=sys.stderr)
        raise SystemExit(2)
    res = llm_extract.import_payload(text, args.variety, settings)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_api(args) -> None:
    settings = config.load_settings()
    v = config.get_variety(args.variety)
    as_of = args.as_of or timeutil.ym_str(timeutil.today_ym()) + "-01"
    rendered = pb.render(v, as_of)
    info = llm_client.preflight(settings)
    print(f"[api] {info}")
    obj, meta = llm_client.call_chat(rendered, settings)
    if obj is None:
        print(f"[api] 失败: {meta}", file=sys.stderr)
        raise SystemExit(1)
    res = llm_extract.ingest(obj, v["id"], settings)
    print(json.dumps({"meta": meta, "ingest": res}, ensure_ascii=False, indent=2))


def cmd_predict(args) -> None:
    settings = config.load_settings()
    res = fc.run_forecast(args.variety, settings, with_backtest=not args.no_backtest,
                          max_origins=args.origins, horizon=args.horizon)
    path = fc.save_result(res)
    h = res["headline"]
    print("=" * 62)
    print(f"{res['variety']['name']}（{res['variety']['ticker']}） 截止 {res['as_of_month']}")
    print(f"目标期      {', '.join(h['target_months'] or [])}")
    print(f"预测价格    {_fmt_money(h['price'], res['variety']['unit'])}"
          f"   [{_fmt_money(h['lo'])}, {_fmt_money(h['hi'])}] ({int(h['interval_level']*100)}%)")
    print(f"当前/远月   {_fmt_money(h['base_price'])} / {_fmt_money(res['far_close'])}")
    print(f"上涨概率    {h['prob_up']}%   预期变动 {h['expected_change_pct']:+.2f}%")
    print(f"模型权重    {h['weights']}")
    for s in res["scenarios"]:
        print(f"  情景 {s['label']:<3} {_fmt_money(s['price'])}  上涨概率 {s['prob_up']}%")
    print(f"逻辑校验    {res['validation']['status']}  "
          f"(warn={res['validation']['counts']['warn']}, error={res['validation']['counts']['error']})")
    for c in res["validation"]["checks"]:
        if c["level"] != "info":
            print(f"   [{c['level']}] {c['title']}: {c['message']}")
    print(f"耗时        {res['runtime_seconds']}s")
    print(f"结果已保存  {path}")
    print("=" * 62)


def cmd_predict_all(args) -> None:
    settings = config.load_settings()
    rows = []
    for v in config.variety_list():
        st = ingest.market_status(v["id"])
        if st["rows"] < 24:
            rows.append((v["name_cn"], "-", "-", "-", "-", "无数据（先 init）"))
            continue
        try:
            res = fc.run_forecast(v["id"], settings, with_backtest=not args.no_backtest,
                                  max_origins=args.origins)
            fc.save_result(res)
            h = res["headline"]
            rows.append((v["name_cn"], h["price"], h["lo"], h["hi"], h["prob_up"],
                         res["validation"]["status"]))
        except Exception as exc:  # noqa: BLE001
            rows.append((v["name_cn"], "-", "-", "-", "-", f"失败: {str(exc)[:40]}"))
    print(f"{'品种':<8}{'预测价':>10}{'区间下':>10}{'区间上':>10}{'涨概率':>8}  校验")
    for r in rows:
        print(f"{r[0]:<8}{str(r[1]):>10}{str(r[2]):>10}{str(r[3]):>10}{str(r[4]):>8}  {r[5]}")


def cmd_serve(args) -> None:
    from .server import serve
    serve(host=args.host, port=args.port)


def cmd_selftest(args) -> None:
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harvestcast", description="农产品中长期价格预测系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    it = sub.add_parser("init", help="获取初始行情（没有数据时先跑这个）")
    it.add_argument("--variety", default="all", help="品种 id，逗号分隔；all 表示全部")
    it.add_argument("--min-months", type=int, default=60)
    it.add_argument("--horizon", type=int, default=12, help="挑选远月合约时使用的跨度（月）")
    it.add_argument("--force", action="store_true", help="已有数据也重新拉取")
    it.add_argument("--allow-demo", action="store_true", help="无网络时生成合成数据（仅演示）")
    it.add_argument("--source", choices=MARKET_SOURCES, default="auto", help=argparse.SUPPRESS)
    it.set_defaults(func=cmd_init)

    d = sub.add_parser("demo", help="生成离线合成数据（仅演示）")
    d.add_argument("--variety", default="all")
    d.add_argument("--years", type=int, default=10)
    d.add_argument("--seed", type=int, default=20261121)
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("ingest", help="兼容命令：等于 init 单品种")
    i.add_argument("--variety", required=True)
    i.add_argument("--source", choices=MARKET_SOURCES, default="auto")
    i.add_argument("--min-months", type=int, default=60)
    i.add_argument("--horizon", type=int, default=12)
    i.add_argument("--force", action="store_true")
    i.add_argument("--allow-demo", action="store_true")
    i.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("prompt", help="导出结构化调研提示词（手工模式）")
    pr.add_argument("--variety", required=True)
    pr.add_argument("--as-of", default="")
    pr.set_defaults(func=cmd_prompt)

    e = sub.add_parser("extract", help="回填手工模式 JSON")
    e.add_argument("--variety", required=True)
    e.add_argument("--json-file", default="")
    e.add_argument("--json-text", default="")
    e.set_defaults(func=cmd_extract)

    a = sub.add_parser("api", help="API 自动模式抽取")
    a.add_argument("--variety", required=True)
    a.add_argument("--as-of", default="")
    a.set_defaults(func=cmd_api)

    f = sub.add_parser("predict", help="训练+回测+预测并落盘")
    f.add_argument("--variety", required=True)
    f.add_argument("--origins", type=int, default=8)
    f.add_argument("--horizon", type=int, default=0, help="指定单一跨度（月），0 表示用配置")
    f.add_argument("--no-backtest", action="store_true")
    f.set_defaults(func=cmd_predict)

    fa = sub.add_parser("predict-all", help="全品种批量预测（跳过无数据品种）")
    fa.add_argument("--origins", type=int, default=6)
    fa.add_argument("--no-backtest", action="store_true")
    fa.set_defaults(func=cmd_predict_all)

    s = sub.add_parser("serve", help="启动本地 Web 界面")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8848)
    s.set_defaults(func=cmd_serve)

    st = sub.add_parser("selftest", help="运行单元测试")
    st.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config.ensure_dirs()
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
