"""集成融合。

权重不用固定值，也不用「在测试集上挑最优」，而是：
1. 由滚动回测的各模型 MSE 反比定权（inverse-MSE），并设权重下限防止单模型垄断；
2. 计算每个回测起点时，权重只用**该起点之前**的误差，避免信息回流（leave-future-out）；
3. 区间由「模型内方差」与「模型间分歧」两部分合成，分歧越大区间越宽。

这样输出的区间既反映单模型不确定性，也反映模型对同一时点的看法差异。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from ..core import numeric
from .base import Forecast, clamp_sigma


def inverse_mse_weights(mse: Dict[str, float], floor: float = 0.05) -> Dict[str, float]:
    ok = {k: v for k, v in mse.items() if v is not None and v == v and v > 1e-9}
    if not ok:
        n = len(mse) or 1
        return {k: 1.0 / n for k in mse}
    raw = {k: 1.0 / v for k, v in ok.items()}
    s = sum(raw.values())
    w = {k: v / s for k, v in raw.items()}
    # 权重下限：抬升后重新归一，迭代到所有成员都不低于 floor。
    # （单次「抬升 + 归一」会让被抬升者又掉回 floor 以下，必须迭代收敛。）
    for _ in range(20):
        w = {k: max(floor, v) for k, v in w.items()}
        s2 = sum(w.values())
        w = {k: v / s2 for k, v in w.items()}
        if min(w.values()) >= floor - 1e-12:
            break
    return w


def combine(forecasts: Sequence[Forecast], weights: Dict[str, float],
            level: float = 0.8) -> Forecast:
    usable = [f for f in forecasts if f.model in weights]
    if not usable:
        raise ValueError("无可融合的模型输出")
    wsum = sum(weights[f.model] for f in usable) or 1.0
    w = {f.model: weights[f.model] / wsum for f in usable}
    mean_log = sum(w[f.model] * f.mean_log for f in usable)
    # 模型内方差（加权）+ 模型间分歧（加权平方偏差）
    within = sum((w[f.model] ** 2) * (f.sigma ** 2) for f in usable)
    between = sum(w[f.model] * (f.mean_log - mean_log) ** 2 for f in usable)
    sigma = clamp_sigma(math.sqrt(within + between))
    diag = {
        "weights": {k: round(v, 4) for k, v in w.items()},
        "members": {f.model: {"mean_price": round(f.mean_price, 2), "sigma": round(f.sigma, 4)}
                    for f in usable},
        "sigma_within": round(math.sqrt(within), 4),
        "sigma_between": round(math.sqrt(max(between, 0.0)), 4),
        "dispersion_pct": round(math.exp(math.sqrt(max(between, 0.0))) - 1, 4) * 100,
    }
    detail: Dict[str, Any] = {"interval_level": level}
    for f in usable:
        if f.detail.get("target_months"):
            detail["target_months"] = f.detail["target_months"]
            break
    return Forecast(model="ensemble", mean_log=mean_log, sigma=sigma, horizon=usable[0].horizon,
                    diagnostics=diag, detail=detail)


def probability_up(mean_log: float, sigma: float, base_log: float) -> float:
    """P(目标价 > 当前价)，对数正态口径。"""
    if sigma <= 0:
        return 1.0 if mean_log > base_log else 0.0
    return 1.0 - numeric.norm_cdf((base_log - mean_log) / sigma)


def probability_above(mean_log: float, sigma: float, threshold_log: float) -> float:
    if sigma <= 0:
        return 1.0 if mean_log > threshold_log else 0.0
    return 1.0 - numeric.norm_cdf((threshold_log - mean_log) / sigma)


def scenario_table(gbm: Any, anchor_log: float, sigma: float,
                   factors: Sequence[str], gbm_base_log: Optional[float] = None) -> List[Dict[str, Any]]:
    """丰产 / 平产 / 减产三情景。

    用产量与库存因子的 ±1σ 位移驱动 GBM 重推理。为了让情景与「主预测」同口径，
    只取 GBM 的情景**相对差异**，再叠加到集成模型的锚点价格上：
        price_scenario = anchor + (gbm_scenario - gbm_base)
    这样「平产」情景与首页建议价一致，不会出现两套数字打架。
    """
    rows: List[Dict[str, Any]] = []
    # 情景沿「供给松紧」单一维度移动，各因子同向联动，避免单因子互斥造成的伪结论
    specs = [
        ("bumper", "丰产", {"production_est": 1.0, "inventory_stock": 0.8,
                            "weather_anomaly": -0.8, "import_volume": 0.5}),
        ("base", "平产", {}),
        ("shortfall", "减产", {"production_est": -1.0, "inventory_stock": -0.8,
                               "weather_anomaly": 0.8, "import_volume": -0.5}),
    ]
    for key, label, shift in specs:
        shifted = {k: v for k, v in shift.items() if k in factors}
        if gbm is None or not hasattr(gbm, "scenario_predict"):
            pred = anchor_log
        else:
            out = gbm.scenario_predict(shifted)
            if out is None:
                pred = anchor_log
            else:
                base_g = gbm_base_log if gbm_base_log is not None else out
                pred = anchor_log + (out - base_g)
        rows.append({
            "key": key,
            "label": label,
            "shift": shifted,
            "mean_log": pred,
            "price": round(numeric.exp_series([pred])[0], 2),
            "prob_up": round(probability_up(pred, sigma, anchor_log) * 100, 1),
        })
    return rows
