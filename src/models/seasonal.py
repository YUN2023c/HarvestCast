# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""季节基准模型（Seasonal Baseline）。

用途：作为「必须被打败的下限」。任何复杂模型若无法稳定优于该基准，
按方法论要求就不该上线。

形式：log P_t = 趋势_t + 季节指数_月 + ε
- 趋势：12 个月因果移动平均（只回看，不看未来）
- 季节指数：按日历月对「对数价 - 趋势」取多年均值
- 外推：趋势按最近一年斜率外推，带 0.5 阻尼（防止长期外推发散）
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..core import numeric, timeutil
from ..data.schema import DatasetBundle
from .base import Forecast, Forecaster, clamp_sigma, residual_sigma


class SeasonalBaseline(Forecaster):
    name = "seasonal_baseline"

    def __init__(self) -> None:
        self._log: List[float] = []
        self._trend: List[float] = []
        self._seasonal: Dict[int, float] = {}
        self._drift = 0.0
        self._sigma = 0.08
        self._n = 0
        self._harvest: Sequence[int] = []
        self._horizon = 6
        self._midx: List[int] = []

    def fit(self, ds: DatasetBundle, harvest_months: Sequence[int], horizon: int,
            cols: Sequence[str], settings: Dict[str, Any]) -> "SeasonalBaseline":
        self._log = numeric.log_series(ds.close)
        self._n = len(self._log)
        self._harvest = harvest_months
        self._horizon = horizon
        self._midx = [timeutil.parse_ym(m) for m in ds.months]

        trend: List[float] = []
        for t in range(self._n):
            lo = max(0, t - 11)
            trend.append(numeric.mean(self._log[lo:t + 1]))
        self._trend = trend

        buckets: Dict[int, List[float]] = {}
        for t, lp in enumerate(self._log):
            m = timeutil.parse_ym(ds.months[t]) % 12 + 1
            buckets.setdefault(m, []).append(lp - trend[t])
        self._seasonal = {m: numeric.mean(v) for m, v in buckets.items()}
        # 季节项中心化，避免整体水平被季节均值抬高
        if self._seasonal:
            center = numeric.mean(list(self._seasonal.values()))
            self._seasonal = {m: v - center for m, v in self._seasonal.items()}

        if self._n >= 24:
            self._drift = (trend[-1] - trend[-13]) / 12.0 * 0.5
        else:
            self._drift = 0.0

        fitted = [trend[t] + self._seasonal.get(timeutil.parse_ym(ds.months[t]) % 12 + 1, 0.0)
                  for t in range(self._n)]
        self._sigma = clamp_sigma(residual_sigma(self._log, fitted, k=2))
        return self

    def forecast_at(self, origin_t: int) -> Forecast:
        if self._n == 0:
            raise RuntimeError("模型未训练")
        origin_t = max(0, min(origin_t, self._n - 1))
        base = self._trend[origin_t]
        origin_idx = self._midx[origin_t]
        horizon = self._horizon
        window = timeutil.harvest_window(self._harvest, origin_idx, horizon)
        preds: List[float] = []
        detail_months: List[str] = []
        for idx in window:
            steps = idx - origin_idx
            m = idx % 12 + 1
            preds.append(base + self._drift * steps + self._seasonal.get(m, 0.0))
            detail_months.append(timeutil.ym_str(idx))
        mean_log = numeric.mean(preds)
        # 外推越远越不可靠：按跨度放大方差
        span = max(1, window[-1] - origin_idx)
        sigma = clamp_sigma(self._sigma * (1.0 + 0.06 * span))
        return Forecast(model=self.name, mean_log=mean_log, sigma=sigma, horizon=horizon,
                        diagnostics={"drift": round(self._drift, 6),
                                     "sigma": round(self._sigma, 4),
                                     "n": self._n},
                        detail={"target_months": detail_months})

    def season_profile(self, ds: DatasetBundle) -> List[Dict[str, Any]]:
        out = []
        for m in range(1, 13):
            out.append({"month": m, "index": round(self._seasonal.get(m, 0.0), 4),
                        "pct": round((numeric.exp_series([self._seasonal.get(m, 0.0)])[0] - 1) * 100, 2)})
        return out
