# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""本地 Web 服务（仅标准库 http.server）。

职责边界：只做「读文件 + 返回 JSON + 触发一次后台任务」，不做任何数值计算。
所有数字都能在 data/output/*.json 或 data/*.csv 里找到原始出处，
前端不参与建模，避免"界面上的数与模型不一致"这类不可审计的问题。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .core import config
from .core.config import DATA_DIR, OUTPUT_DIR, WEB_DIR
from .data import ingest
from .llm import extract as llm_extract
from .llm import jsonschema_lite, prompt_builder
from .models import forecast as fc

TASKS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _varieties() -> list:
    out = []
    for v in config.variety_list():
        f = OUTPUT_DIR / f"forecast_{v['id']}.json"
        res = _read_json(f, None)
        out.append({
            "id": v["id"], "name": v["name_cn"], "exchange": v["exchange"],
            "ticker": v["ticker"], "unit": v["unit"], "category": v.get("category", "其他"),
            "harvest_months": v["harvest_months"],
            "has_forecast": res is not None,
            "has_fundamentals": bool((res or {}).get("factor_quality", {}).get("count")),
            "headline": (res or {}).get("headline"),
            "generated_at": (res or {}).get("generated_at"),
            "validation": (res or {}).get("validation", {}).get("status"),
        })
    return out


def api_state() -> Dict[str, Any]:
    with LOCK:
        tasks = {k: dict(v) for k, v in TASKS.items()}
    vs = _varieties()
    status = {v["id"]: ingest.market_status(v["id"]) for v in config.variety_list()}
    for item in vs:
        st = status.get(item["id"], {})
        item["has_market"] = bool(st.get("has_data"))
        item["market_rows"] = st.get("rows", 0)
        item["market_source"] = st.get("source", "")
        item["market_last"] = st.get("last", "")
    ready = [v for v in vs if v.get("has_market") and v["market_rows"] >= 24]
    return {
        "varieties": vs,
        "tasks": tasks,
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings_path": config.load_settings().get("_settings_path"),
        "data": {
            "total": len(vs),
            "ready": len(ready),
            "missing": [{"id": v["id"], "name": v["name"]} for v in vs if not v.get("has_market")],
            "sources": sorted({v.get("market_source") for v in vs if v.get("market_source")}),
            "fundamentals_ready": sum(1 for v in vs if v.get("has_fundamentals")),
        },
        "demo_ready": any(v.get("market_source") == "demo" for v in vs),
    }


def api_init(payload: Dict[str, Any]) -> Dict[str, Any]:
    """获取初始行情（异步）。payload: {variety: 'all'|'corn,sugar', allow_demo: bool, force: bool}"""
    raw = (payload.get("variety") or "all").strip()
    if raw == "all":
        varieties = config.variety_list()
    else:
        varieties = []
        for vid in [x.strip() for x in raw.split(",") if x.strip()]:
            try:
                varieties.append(config.get_variety(vid))
            except KeyError:
                pass
    allow_demo = bool(payload.get("allow_demo"))
    force = bool(payload.get("force"))
    key = f"init-{int(time.time())}"
    with LOCK:
        TASKS[key] = {"state": "running", "kind": "init", "variety": raw,
                      "started": time.strftime("%H:%M:%S"), "message": "准备拉取初始行情",
                      "done": 0, "total": len(varieties)}

    def work() -> None:
        try:
            def progress(name: str) -> None:
                with LOCK:
                    t = TASKS.get(key, {})
                    t["message"] = f"正在获取 {name}"
                    t["done"] = t.get("done", 0) + 1
            res = ingest.bootstrap(varieties, force=force, allow_demo=allow_demo,
                                   progress=progress)
            with LOCK:
                TASKS[key].update({"state": "done", "result": res,
                                   "message": f"成功 {res['ok']} / 失败 {res['failed']} / 跳过 {res['skipped']}",
                                   "finished": time.strftime("%H:%M:%S")})
        except Exception as exc:  # noqa: BLE001
            with LOCK:
                TASKS[key].update({"state": "error", "message": str(exc),
                                   "trace": traceback.format_exc()[-800:],
                                   "finished": time.strftime("%H:%M:%S")})

    threading.Thread(target=work, daemon=True).start()
    return {"task": key, "state": "running", "total": len(varieties)}


def api_data(variety: str) -> Dict[str, Any]:
    res = _read_json(OUTPUT_DIR / f"forecast_{variety}.json", {})
    from .data import ingest
    from .data.schema import FactorRow
    rows, factors = ingest.load_local(variety)
    # 因子预览：按因子聚合，给出最新值/单位/方向/置信度/来源数
    latest: Dict[str, FactorRow] = {}
    counts: Dict[str, int] = {}
    for f in factors:
        counts[f.factor_id] = counts.get(f.factor_id, 0) + 1
        if f.factor_id not in latest or f.as_of > latest[f.factor_id].as_of:
            latest[f.factor_id] = f
    factor_rows = [{
        "factor_id": k, "name": k, "value": v.value, "unit": v.unit, "period": v.period,
        "as_of": v.as_of, "direction": v.direction, "confidence": v.confidence,
        "source_title": v.source_title, "source_url": v.source_url,
        "evidence": v.evidence, "records": counts.get(k, 0),
    } for k, v in sorted(latest.items())]
    hist = res.get("history") or [
        {"month": r.month, "close": r.close, "far_close": r.far_close} for r in rows
    ]
    return {
        "variety": variety,
        "history": hist,
        "factors": factor_rows,
        "logs": res.get("logs", []),
        "data_summary": res.get("data_summary", {}),
        "market_rows": len(rows),
        "factor_rows": len(factors),
        "as_of_month": res.get("as_of_month"),
    }


def api_models(variety: str) -> Dict[str, Any]:
    res = _read_json(OUTPUT_DIR / f"forecast_{variety}.json", {})
    return {
        "variety": variety,
        "horizons": res.get("horizons", {}),
        "folds": (res.get("backtest") or {}).get("folds", []),
        "residual": res.get("residual", {}),
        "sensitivity": res.get("sensitivity", []),
        "seasonal_profile": res.get("seasonal_profile", []),
        "elasticity": res.get("elasticity", {}),
        "feature_importance": next((m.get("diagnostics", {}).get("feature_importance")
                                    for m in (res.get("headline") or {}).get("models", [])
                                    if m.get("name") == "gbm"), {}),
        "headline_diagnostics": [{"name": m.get("name"), "diagnostics": m.get("diagnostics")}
                                 for m in (res.get("headline") or {}).get("models", [])],
    }


def api_extraction(variety: str) -> Dict[str, Any]:
    settings = config.load_settings()
    v = config.get_variety(variety)
    as_of = time.strftime("%Y-%m-%d")
    rendered = prompt_builder.render(v, as_of)
    return {
        "variety": variety,
        "mode": settings["llm"].get("mode", "manual"),
        "schema": jsonschema_lite.summarize(rendered["schema"]),
        "schema_raw": rendered["schema"],
        "prompt": rendered["full"],
        "prompt_hash": rendered["hash"],
        "schema_version": rendered["schema_version"],
        "queries": rendered["queries"],
        "llm": {k: settings["llm"].get(k) for k in
                ("model", "base_url", "api_style", "enable_web_search", "min_confidence", "require_source")},
    }


def api_settings() -> Dict[str, Any]:
    settings = config.load_settings()
    lic = _read_json(config.ROOT / "third_party_licenses.json", {})
    deps = {}
    for name, mod in (("statsmodels", "statsmodels"), ("lightgbm", "lightgbm"),
                      ("akshare", "akshare"), ("pandas", "pandas"), ("numpy", "numpy")):
        try:
            __import__(mod)
            deps[name] = True
        except Exception:  # noqa: BLE001
            deps[name] = False
    return {
        "settings": {k: v for k, v in settings.items() if k != "_settings_path"},
        "settings_path": settings.get("_settings_path"),
        "api_key_configured": bool(__import__("os").environ.get(
            settings["llm"].get("api_key_env", "HARVESTCAST_API_KEY"), "")),
        "third_party": lic,
        "optional_deps": deps,
        "python": __import__("sys").version.split()[0],
    }


def api_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    variety = payload.get("variety", "corn")
    origins = int(payload.get("origins", 8))
    with_backtest = bool(payload.get("backtest", True))
    key = f"{variety}-{int(time.time())}"
    with LOCK:
        TASKS[key] = {"state": "running", "variety": variety, "started": time.strftime("%H:%M:%S"),
                      "message": "正在清洗数据并训练模型"}

    def work() -> None:
        try:
            settings = config.load_settings()
            res = fc.run_forecast(variety, settings, with_backtest=with_backtest, max_origins=origins)
            path = fc.save_result(res)
            with LOCK:
                TASKS[key].update({"state": "done", "message": f"完成，用时 {res['runtime_seconds']}s",
                                   "path": path, "price": res["headline"]["price"],
                                   "validation": res["validation"]["status"],
                                   "finished": time.strftime("%H:%M:%S")})
        except Exception as exc:  # noqa: BLE001
            with LOCK:
                TASKS[key].update({"state": "error", "message": str(exc),
                                   "trace": traceback.format_exc()[-1200:],
                                   "finished": time.strftime("%H:%M:%S")})

    threading.Thread(target=work, daemon=True).start()
    return {"task": key, "state": "running"}


class Handler(BaseHTTPRequestHandler):
    server_version = "HarvestCast/1.0"

    def log_message(self, fmt: str, *args) -> None:  # 静音默认日志
        return

    # -- helpers ---------------------------------------------------------
    def _json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._json({"error": "not found", "path": str(path)}, 404)
            return
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
                 ".svg": "image/svg+xml", ".png": "image/png"}.get(path.suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        variety = (q.get("variety") or ["corn"])[0]
        try:
            if u.path in ("/", "/index.html"):
                self._file(WEB_DIR / "index.html")
            elif u.path == "/api/state":
                self._json(api_state())
            elif u.path == "/api/forecast":
                self._json(_read_json(OUTPUT_DIR / f"forecast_{variety}.json", {"status": "missing"}))
            elif u.path == "/api/data":
                self._json(api_data(variety))
            elif u.path == "/api/models":
                self._json(api_models(variety))
            elif u.path == "/api/extraction":
                self._json(api_extraction(variety))
            elif u.path == "/api/settings":
                self._json(api_settings())
            else:
                target = (WEB_DIR / u.path.lstrip("/")).resolve()
                if str(target).startswith(str(WEB_DIR.resolve())):
                    self._file(target)
                else:
                    self._json({"error": "bad path"}, 400)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc), "trace": traceback.format_exc()[-800:]}, 500)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid json"}, 400)
            return
        try:
            if u.path == "/api/init":
                self._json(api_init(payload))
            elif u.path == "/api/run":
                self._json(api_run(payload))
            elif u.path == "/api/extraction/prompt":
                v = config.get_variety(payload.get("variety", "corn"))
                out = llm_extract.export_prompt(v, payload.get("as_of") or time.strftime("%Y-%m-%d"),
                                                OUTPUT_DIR / "prompts")
                self._json(out)
            elif u.path == "/api/extraction/import":
                settings = config.load_settings()
                self._json(llm_extract.import_payload(payload.get("text", ""),
                                                      payload.get("variety", "corn"), settings))
            elif u.path == "/api/extraction/api":
                from .llm import client as llm_client
                settings = config.load_settings()
                v = config.get_variety(payload.get("variety", "corn"))
                rendered = prompt_builder.render(v, payload.get("as_of") or time.strftime("%Y-%m-%d"))
                obj, meta = llm_client.call_chat(rendered, settings)
                if obj is None:
                    self._json({"status": "error", "meta": meta}, 502)
                else:
                    self._json({"status": "ok", "meta": meta,
                                "ingest": llm_extract.ingest(obj, v["id"], settings)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc), "trace": traceback.format_exc()[-800:]}, 500)


def serve(host: str = "127.0.0.1", port: int = 8848) -> None:
    config.ensure_dirs()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"HarvestCast 已启动: http://{host}:{port}")
    print("首次使用：先 python -m src.cli demo 生成示例数据，再在页面点「重跑预测」")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
