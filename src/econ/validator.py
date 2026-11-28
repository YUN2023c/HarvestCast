# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""第四层：经济学逻辑校验。

模型算出来的数不一定在经济上讲得通。这一层把「农产品定价常识」写成可执行规则，
对预测结果做六类检查，产出告警/告警阻断标记，绝不静默放行异常结果。

规则编号与文档 docs/ARCHITECTURE.md 的「校验矩阵」一一对应。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..core import numeric, timeutil

LEVEL_ORDER = {"info": 0, "warn": 1, "error": 2}


def _chk(cid: str, level: str, title: str, message: str, value: Any = None) -> Dict[str, Any]:
    return {"id": cid, "level": level, "title": title, "message": message, "value": value}


def validate_forecast(variety: Dict[str, Any], forecast: Dict[str, Any],
                      ds_summary: Dict[str, Any], seasonal_profile: Sequence[Dict[str, Any]],
                      scenarios: Sequence[Dict[str, Any]], backtest_metrics: Dict[str, Any],
                      cfg: Dict[str, Any], factor_quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    price = forecast.get("price") or 0.0
    lo, hi = forecast.get("lo") or 0.0, forecast.get("hi") or 0.0
    prob_up = forecast.get("prob_up")
    harvest = variety.get("harvest_months", [])
    band = variety.get("sanity_price_band", [0, 1e9])

    # 1. 价格带
    if price and not (band[0] * 0.7 <= price <= band[1] * 1.3):
        checks.append(_chk("price_band", "error", "价格越界",
                           f"预测价 {price:.0f} 超出 {variety['name_cn']} 合理带 "
                           f"[{band[0]}, {band[1]}] 的 30% 容差，疑似数据或量纲错误。", price))
    else:
        checks.append(_chk("price_band", "info", "价格带", "预测价位于历史合理区间内。", price))

    # 2. 区间宽度
    if price and hi > lo:
        width = (hi - lo) / price
        max_w = float(cfg.get("max_interval_width", 0.75))
        if width > max_w:
            checks.append(_chk("interval_width", "warn", "区间过宽",
                               f"80% 区间宽度 {width*100:.1f}% 超过阈值 {max_w*100:.0f}%，"
                               "说明不确定性极高，建议补数据或缩短预测跨度。", round(width, 3)))
        elif width < 0.02:
            checks.append(_chk("interval_width", "warn", "区间过窄",
                               f"区间宽度仅 {width*100:.1f}%，模型可能过度自信。", round(width, 3)))
        else:
            checks.append(_chk("interval_width", "info", "区间宽度",
                               f"区间宽度 {width*100:.1f}%，处于可接受范围。", round(width, 3)))

    # 3. 概率边界
    if prob_up is not None:
        p = float(prob_up) / 100.0   # 统一口径：上游一律传百分数
        lo_p, hi_p = float(cfg.get("min_probability", 0.03)), 1 - float(cfg.get("min_probability", 0.03))
        if not (lo_p <= p <= hi_p):
            checks.append(_chk("probability", "warn", "概率极端",
                               f"上涨概率 {p*100:.1f}% 位于极端区间，长跨度预测不应给出如此确定的方向。",
                               round(p, 3)))
        else:
            checks.append(_chk("probability", "info", "概率合理",
                               f"上涨概率 {p*100:.1f}%。", round(p, 3)))

    # 4. 远月定价一致性
    far = forecast.get("far_close")
    tol = float(cfg.get("far_month_tolerance", 0.18))
    if far:
        dev = price / far - 1
        if abs(dev) > tol:
            checks.append(_chk("far_month", "warn", "偏离远月定价",
                               f"预测价与收获季远月合约相差 {dev*100:+.1f}%，超过 ±{tol*100:.0f}%。"
                               "若模型无误，说明存在可关注的期现/跨期机会；否则应复核基本面输入。",
                               round(dev, 3)))
        else:
            checks.append(_chk("far_month", "info", "远月定价",
                               f"预测价与远月合约偏离 {dev*100:+.1f}%，在容差内。", round(dev, 3)))

    # 5. 季节形状：收获月是否确实为相对低点
    if seasonal_profile and harvest:
        hm = [p for p in seasonal_profile if p["month"] in harvest]
        om = [p for p in seasonal_profile if p["month"] not in harvest]
        if hm and om and numeric.mean([p["index"] for p in hm]) > numeric.mean([p["index"] for p in om]):
            checks.append(_chk("season_shape", "warn", "季节形状异常",
                               "模型给出的收获月季节性高于非收获月，与'收获集中上市压价'的常识相反，"
                               "请检查目标月份是否对齐错误。", None))
        else:
            checks.append(_chk("season_shape", "info", "季节形状",
                               "收获月季节性低于非收获月，符合供需季节性。", None))

    # 6. 情景单调性：丰产价 < 平产价 < 减产价
    if not scenarios:
        checks.append(_chk("scenario", "info", "情景分析不可用",
                           "缺少基本面因子，无法估计供给/需求弹性，本次不输出丰产/减产情景。", None))
    elif len(scenarios) == 3:
        by = {s["key"]: s["price"] for s in scenarios}
        if not (by.get("bumper", 0) < by.get("base", 0) < by.get("shortfall", 0)):
            checks.append(_chk("scenario_order", "error", "情景方向不自洽",
                               f"丰产/平产/减产价格为 {by.get('bumper')}/{by.get('base')}/"
                               f"{by.get('shortfall')}，未满足'供给增→价格降'的单调性，"
                               "模型对产量因子的响应方向错误。", by))
        else:
            gap = (by["shortfall"] - by["bumper"]) / by["base"] * 100 if by.get("base") else 0
            checks.append(_chk("scenario_order", "info", "情景方向",
                               f"三情景单调性成立，丰产与减产价差 {gap:.1f}%（供给弹性）。",
                               round(gap, 2)))

    # 7. 回测是否跑赢季节朴素基准
    m = backtest_metrics or {}
    best = m.get("_best")
    ens_m = m.get("ensemble") if isinstance(m.get("ensemble"), dict) else None
    if ens_m and ens_m.get("mase") is not None:
        if ens_m["mase"] >= 1.0:
            checks.append(_chk("mase", "warn", "未跑赢季节基准",
                               f"集成模型 MASE={ens_m['mase']}（≥1 表示不优于季节朴素），"
                               "该预测的解释力有限，不宜单独作为决策依据。", ens_m["mase"]))
        else:
            checks.append(_chk("mase", "info", "回测有效",
                               f"集成模型 MASE={ens_m['mase']} < 1，优于季节朴素基准。", ens_m["mase"]))

    # 8. 数据质量
    cov = ds_summary.get("coverage") or {}
    weak = [k for k, v in cov.items() if v < 0.6]
    if weak:
        checks.append(_chk("coverage", "warn", "因子覆盖不足",
                           f"{len(weak)} 个因子在样本期内可见比例 <60%：{', '.join(weak[:5])}。", cov))
    if factor_quality and factor_quality.get("count", 1) == 0:
        checks.append(_chk("fundamentals", "warn", "缺少基本面因子",
                           "当前只用价格结构与季节特征预测；建议先跑 LLM 抽取或用 demo 数据演示完整链路。",
                           0))
    elif factor_quality:
        if factor_quality.get("mean_confidence", 1) < 0.6:
            checks.append(_chk("confidence", "warn", "来源置信度偏低",
                               f"入模因子平均置信度 {factor_quality['mean_confidence']:.2f}，"
                               "建议补充权威来源后重跑。", factor_quality.get("mean_confidence")))
        if factor_quality.get("dropped", 0) > 0:
            checks.append(_chk("dropped", "info", "低质记录已剔除",
                               f"{factor_quality['dropped']} 条低置信/无来源记录未进入训练集。",
                               factor_quality.get("dropped")))

    # 9. 样本长度
    n = ds_summary.get("n_months", 0)
    if n < 60:
        checks.append(_chk("sample", "warn", "样本偏短",
                           f"仅 {n} 个月样本，长跨度预测的统计可靠性不足（建议 ≥60）。", n))
    else:
        checks.append(_chk("sample", "info", "样本长度", f"{n} 个月样本，满足最低要求。", n))

    worst = max((LEVEL_ORDER[c["level"]] for c in checks), default=0)
    status = {0: "pass", 1: "warn", 2: "fail"}[worst]
    blocked = status == "fail" and bool(cfg.get("block_on_error", False))
    return {"status": status, "blocked": blocked,
            "counts": {lv: sum(1 for c in checks if c["level"] == lv) for lv in ("info", "warn", "error")},
            "checks": checks}


def sensitivity(base_log: float, sigma: float, deltas: Sequence[float] = (-0.1, -0.05, 0.05, 0.1)) -> List[Dict[str, Any]]:
    """价格对关键驱动的一阶灵敏度（Δ 输入 → Δ 价格）。"""
    out = []
    for d in deltas:
        out.append({"input_shift_pct": round(d * 100, 1),
                    "price": round(numeric.exp_series([base_log + d])[0], 2),
                    "prob_up": round((1 - numeric.norm_cdf((base_log - (base_log + d)) / max(sigma, 1e-6))) * 100, 1)})
    return out
