# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""特征工程：全部按「当时可获得」原则构建。

防未来函数的三个硬措施：
1. 基本面因子只有在其「所属周期结束」且「实际发布日期」都早于当前月时，才对当前月可见；
2. 所有滚动统计一律使用 expanding 窗口（截至 t），不用全样本均值/标准差；
3. 目标值只用 t 之后的数据构造，特征与目标在时间上严格分离，且训练集按时间切分。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..core import numeric, timeutil
from .schema import DatasetBundle, FactorRow, MarketRow

BASE_FEATURES = ["log_close", "basis", "trend12", "mom3", "yoy", "vol6", "season_idx"]


def _expanding_z(series: List[Optional[float]], min_obs: int = 12) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    seen: List[float] = []
    for v in series:
        if v is None:
            out.append(None)
            continue
        seen.append(v)
        if len(seen) < min_obs:
            out.append(None)
            continue
        m = numeric.mean(seen)
        s = numeric.stdev(seen)
        out.append((v - m) / s if s else 0.0)
    return out



def _detrend_ratio(series: List[Optional[float]], window: int = 24, min_obs: int = 12) -> List[Optional[float]]:
    """把量纲型因子转成「相对近两年均值的对数偏离」。

    直接对产量/库存这类带长期增长的序列做 z 分数，会把「趋势」误当成「信号」，
    甚至让产量与价格呈现伪正相关。这里先除掉慢变趋势，再标准化，
    得到的是「当期供需相对常态的偏离」，经济含义明确，也便于情景模拟。
    """
    out: List[Optional[float]] = []
    seen: List[float] = []
    for v in series:
        if v is not None and v > 0:
            seen.append(v)
        hist = seen[-window:]
        if v is None or v <= 0 or len(hist) < min_obs:
            out.append(None)
            continue
        base = numeric.mean(hist)
        out.append(math.log(v / base) if base > 0 else None)
    return out


def _expanding_seasonal(log_prices: List[float]) -> List[float]:
    """扩张窗口季节指数：当月对数价 / 截至当月的整体均值 - 1。"""
    out: List[float] = []
    seen: List[float] = []
    for i, p in enumerate(log_prices):
        seen.append(p)
        m = numeric.mean(seen)
        out.append((p - m) if m else 0.0)
    return out


def _factor_monthly(factors: Sequence[FactorRow], months: List[int]) -> Tuple[List[Optional[float]], int]:
    """把长表因子转成月度可见序列。

    可见条件：period 所属月 <= 当前月 且 as_of 发布月 <= 当前月。
    同一时间多值取 as_of 最新者。
    """
    events: List[Tuple[int, int, float]] = []  # (visible_from, asof_idx, value)
    for f in factors:
        if f.value is None:
            continue
        try:
            period_idx = timeutil.parse_ym(f.period)
        except Exception:
            continue
        try:
            asof_idx = timeutil.parse_ym(f.as_of[:7])
        except Exception:
            asof_idx = period_idx
        visible = max(period_idx, asof_idx)
        events.append((visible, asof_idx, float(f.value)))
    events.sort(key=lambda e: (e[0], e[1]))

    series: List[Optional[float]] = []
    latest: Optional[float] = None
    j = 0
    hit = 0
    for m in months:
        while j < len(events) and events[j][0] <= m:
            latest = events[j][2]
            j += 1
        series.append(latest)
        if latest is not None:
            hit += 1
    return series, hit


def build_dataset(variety_id: str, market: Sequence[MarketRow], factors: Sequence[FactorRow],
                  harvest_months: Sequence[int], min_conf: float = 0.55,
                  require_source: bool = True) -> DatasetBundle:
    months = [timeutil.parse_ym(r.month) for r in market]
    close = [r.close for r in market]
    far = [r.far_close for r in market]
    ds = DatasetBundle(variety=variety_id, months=[timeutil.ym_str(m) for m in months],
                       close=close, far_close=far)
    if not months:
        ds.warnings.append("无行情数据")
        return ds

    log_p = numeric.log_series(close)
    feats: Dict[str, List[Optional[float]]] = {}
    feats["log_close"] = log_p
    feats["basis"] = [math.log(f) - math.log(c) for f, c in zip(far, close)]
    feats["trend12"] = [None if i < 11 else log_p[i] - numeric.mean(log_p[i - 11:i + 1])
                        for i in range(len(log_p))]
    feats["mom3"] = [None if i < 3 else log_p[i] - log_p[i - 3] for i in range(len(log_p))]
    feats["yoy"] = [None if i < 12 else log_p[i] - log_p[i - 12] for i in range(len(log_p))]
    vol6: List[Optional[float]] = []
    rets = [0.0] + [log_p[i] - log_p[i - 1] for i in range(1, len(log_p))]
    for i in range(len(log_p)):
        vol6.append(None if i < 6 else numeric.stdev(rets[i - 5:i + 1]))
    feats["vol6"] = vol6
    feats["season_idx"] = _expanding_seasonal(log_p)

    # 基本面因子
    grouped: Dict[str, List[FactorRow]] = {}
    for f in factors:
        if float(f.confidence) < min_conf:
            continue
        if require_source and (not f.source_url or f.source_url.strip().upper() == "N/A"):
            continue
        grouped.setdefault(f.factor_id, []).append(f)
    for fid, rows in grouped.items():
        raw, hit = _factor_monthly(rows, months)
        ds.coverage[fid] = round(hit / len(months), 3)
        unit = (rows[0].unit or "").strip()
        if unit in ("万吨", "万公顷", "元/吨", "吨/公顷"):   # 量纲型：先去趋势
            raw = _detrend_ratio(raw)
        z = _expanding_z(raw)
        feats[fid] = z
        if hit == 0:
            ds.warnings.append(f"因子 {fid} 在样本期内不可见，已忽略")

    ds.features = feats
    ds.feature_meta = {
        "base": {"features": BASE_FEATURES, "note": "价格结构类特征，全部 expanding 计算"},
        "fundamental": {"features": [k for k in feats if k not in BASE_FEATURES],
                        "note": "LLM 抽取因子，按发布日可见后取扩张窗口 z 分数"},
    }
    return ds


def design_matrix(ds: DatasetBundle, columns: Sequence[str]) -> List[List[float]]:
    """取特征矩阵；缺失统一填 0（z 分数口径下 0 = 样本均值）。"""
    rows: List[List[float]] = []
    for i in range(ds.n()):
        row: List[float] = []
        for c in columns:
            series = ds.features.get(c)
            v = series[i] if series and i < len(series) else None
            row.append(0.0 if v is None else float(v))
        rows.append(row)
    return rows


def available_columns(ds: DatasetBundle, max_fund: int = 8) -> List[str]:
    """价格结构特征 + 覆盖率达标的基本面因子（最多 max_fund 个）。"""
    cols = [c for c in BASE_FEATURES if ds.features.get(c)]
    fund = [(k, v) for k, v in ds.coverage.items() if v >= 0.5 and k in ds.features]
    fund.sort(key=lambda kv: -kv[1])
    cols += [k for k, _ in fund[:max_fund]]
    return cols


def build_targets(ds: DatasetBundle, harvest_months: Sequence[int], horizon: int
                  ) -> List[Dict[str, float]]:
    """构造 t -> 收获窗口均价 的有监督样本。

    y = 目标收获窗口内月度收盘均值（对数），并记录目标期首末月，便于展示区间。
    """
    out: List[Dict[str, float]] = []
    idxs = [timeutil.parse_ym(m) for m in ds.months]
    log_p = numeric.log_series(ds.close)
    for t in range(ds.n()):
        window = timeutil.harvest_window(harvest_months, idxs[t], horizon)
        if not window:
            continue
        if window[-1] > idxs[-1] or window[0] < idxs[0]:
            continue
        vals = []
        for w in window:
            if w in idxs:
                vals.append(log_p[idxs.index(w)])
        if len(vals) != len(window):
            continue
        out.append({
            "t": t,
            "month": ds.months[t],
            "target_month": timeutil.ym_str(window[len(window) // 2]),
            "target_first": timeutil.ym_str(window[0]),
            "target_last": timeutil.ym_str(window[-1]),
            "y": numeric.mean(vals),
            "y_price": math.exp(numeric.mean(vals)),
        })
    return out
