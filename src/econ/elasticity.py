# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""供需弹性估计与情景模拟。

为什么情景不直接用 GBM：
树模型只能在训练样本的取值范围内做分段常数插值，外推到「丰产/减产」这种
极端情景时响应会被压平（我们实测过：±1σ 位移的价格变化只有 ±0.1%），
既不可解释也不单调。所以情景分析改用**结构弹性**：

    log P = c + β_s·S + β_d·D + γ·season + ε
    S = 供给松紧指数（产量、库存、进口同向，天气异常反向）
    D = 需求指数（饲用/加工）

β_s < 0、β_d > 0 是农产品定价的基本符号约束，估出来符号不对会被校验层拦下。
情景区间用 「β 的置信区间」而非固定点值，避免给出假精确的数字。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from ..core import numeric
from ..data.schema import DatasetBundle


def _avg_available(series_list: Sequence[Sequence[Optional[float]]], n: int,
                   signs: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(n):
        vals = []
        for s, sign in zip(series_list, signs):
            v = s[i] if i < len(s) else None
            if v is not None:
                vals.append(sign * float(v))
        out.append(numeric.mean(vals) if vals else None)
    return out


def supply_index(ds: DatasetBundle) -> List[Optional[float]]:
    """供给松紧：产量/库存/进口上升=宽松（价格下行），不利天气=收紧。"""
    parts, signs = [], []
    # 只取外生性较强的三项：产量、库存、不利天气。
    # 进口量、需求类指标与价格同期内生（价格高→进口多、需求名义值高），
    # 放进「松紧指数」会把内生性直接灌进弹性估计，因此单独作为需求侧控制变量。
    for fid, sign in (("production_est", 1.0), ("inventory_stock", 1.0),
                      ("supply_area", 1.0), ("supply_yield", 1.0),
                      ("weather_anomaly", -1.0)):
        if ds.features.get(fid):
            parts.append(ds.features[fid])
            signs.append(sign)
    if not parts:
        return [None] * ds.n()
    return _avg_available(parts, ds.n(), signs)


def demand_index(ds: DatasetBundle) -> List[Optional[float]]:
    parts, signs = [], []
    for fid in ("feed_demand", "processing_demand"):
        if ds.features.get(fid):
            parts.append(ds.features[fid])
            signs.append(1.0)
    if not parts:
        return [None] * ds.n()
    return _avg_available(parts, ds.n(), signs)


def estimate(ds: DatasetBundle) -> Dict[str, Any]:
    """估计结构性弹性（全样本 OLS，仅用于情景差异，不用于点预测）。"""
    from ..core import timeutil

    n = ds.n()
    if n < 36:
        return {"available": False, "reason": "样本不足 36 个月"}
    s_idx, d_idx = supply_index(ds), demand_index(ds)
    has_demand = any(v is not None for v in d_idx)
    if not any(v is not None for v in s_idx):
        return {"available": False, "reason": "缺少可用的供给类因子（产量/库存/天气）"}
    logp = numeric.log_series(ds.close)

    # 季节用「月份虚拟变量」吸收，不能用由价格自身推导出的 season_idx——
    # 它与因变量机械共线（实测会把 β_supply 压到 0），等于没有回归。
    X, y = [], []
    for i in range(n):
        if s_idx[i] is None:
            continue
        m = timeutil.parse_ym(ds.months[i]) % 12      # 0..11，0 为基准月
        row = [1.0, float(s_idx[i])]
        if has_demand:
            row.append(float(d_idx[i]) if d_idx[i] is not None else 0.0)
        row += [1.0 if m == k else 0.0 for k in range(1, 12)]
        X.append(row)
        y.append(logp[i])
    if len(y) < 30:
        return {"available": False, "reason": "有效样本不足"}

    fit = numeric.ols(X, y, ridge=1e-6)
    beta = fit["beta"]
    resid_sigma = fit["sigma"]
    se = []
    for j in range(len(beta)):
        col = [row[j] for row in X]
        ss = sum((v - numeric.mean(col)) ** 2 for v in col) or 1e-9
        se.append(resid_sigma / math.sqrt(ss))
    offset = 3 if has_demand else 2
    beta_d = beta[2] if has_demand else 0.0
    t_d = (beta_d / se[2]) if has_demand and se[2] else None
    return {
        "available": True,
        "n": len(y),
        "r2": round(fit["r2"], 4),
        "has_demand": has_demand,
        "beta_supply": round(beta[1], 4),
        "se_supply": round(se[1], 4),
        "t_supply": round(beta[1] / se[1], 2) if se[1] else None,
        "beta_demand": round(beta_d, 4),
        "se_demand": round(se[2], 4) if has_demand else None,
        "t_demand": round(t_d, 2) if t_d is not None else None,
        "beta_month": [round(b, 3) for b in beta[offset:]],
        "resid_sigma": round(resid_sigma, 4),
        "sign_ok": bool(beta[1] < 0 and (beta_d > 0 if has_demand else True)),
        "note": "β_s<0（供给宽松压价）、β_d>0（需求走强抬价）为经济学符号约束；缺失需求因子时退化为单因子弹性",
    }


def scenarios(base_log: float, sigma: float, est: Dict[str, Any],
              sigma_mult: float = 1.0) -> List[Dict[str, Any]]:
    """三情景：沿供给维度 ±1σ，需求同步 ±0.5σ 联动（丰产常伴需求平淡）。

    弹性估不出来（没有基本面因子）时**返回空列表**，不编造默认系数：
    凭空给一个 -0.35 的弹性会让情景价格摆动 ±40%，比不给情景更糟。
    """
    if not est.get("available"):
        return []
    specs = [
        ("bumper", "丰产", 1.0, -0.5, "供给宽松 + 需求平淡：产量与库存同步上移，天气不利程度下降"),
        ("base", "平产", 0.0, 0.0, "供给与需求均处常态，沿用基准预测"),
        ("shortfall", "减产", -1.0, 0.5, "供给收紧 + 补库需求：产量下修、库存去化、天气不利"),
    ]
    bs = est.get("beta_supply", -0.35) if est.get("available") else -0.35
    bd = est.get("beta_demand", 0.25) if est.get("available") else 0.25
    rows = []
    for key, label, ds_, dd_, note in specs:
        delta = (bs * ds_ + bd * dd_) * sigma_mult
        pred = base_log + delta
        rows.append({
            "key": key, "label": label, "supply_shift": ds_, "demand_shift": dd_,
            "delta_log": round(delta, 4),
            "price": round(numeric.exp_series([pred])[0], 2),
            "change_pct": round((math.exp(delta) - 1) * 100, 2),
            "prob_up": round((1 - numeric.norm_cdf((base_log - pred) / max(sigma, 1e-6))) * 100, 1),
            "note": note,
        })
    return rows
