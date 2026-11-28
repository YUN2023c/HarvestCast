# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""SARIMAX 轻量实现（条件最小二乘，无 statsmodels 依赖）。

模型形式（对月度对数价 y）：

    Δy_t = c + Σ_{i=1..p} φ_i Δy_{t-i} + Φ·Δy_{t-12} + Σ_j β_j x_{j,t-1} + e_t

即 AR(p) + 季节 AR(1)@12 + 外生回归，MA 项省略——月度农产品序列在 6-12 个月
跨度上，MA 项对点预测的边际贡献很小，而条件最小二乘的估计稳定性更高。
若安装 statsmodels，见 ``fit_with_statsmodels`` 可直接对比完整 SARIMAX。

外生变量一律滞后 1 期（x_{t-1}），确保 t 时点预测只用 t-1 已发布的信息。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from ..core import numeric, timeutil
from ..data.schema import DatasetBundle
from .base import Forecast, Forecaster, clamp_sigma


class SarimaxLite(Forecaster):
    name = "sarimax"

    def __init__(self) -> None:
        self._y: List[float] = []
        self._dy: List[float] = []
        self._cols: List[str] = []
        self._exog: List[List[float]] = []
        self._beta: List[float] = []
        self._p = 1
        self._seasonal = True
        self._sigma_dy = 0.05
        self._n = 0
        self._harvest: Sequence[int] = []
        self._horizon = 6
        self._diag: Dict[str, Any] = {}
        self._means: List[float] = [0.0]
        self._sds: List[float] = [1.0]
        self._midx: List[int] = []

    # ------------------------------------------------------------------
    def _row(self, dy: Sequence[float], exog: List[List[float]], t: int, p: int,
             seasonal: bool, use_exog: bool) -> List[float]:
        row = [1.0]
        for i in range(1, p + 1):
            row.append(dy[t - i])
        if seasonal:
            row.append(dy[t - 12])
        if use_exog:
            for j in range(len(exog)):
                row.append(exog[j][t - 1])
        return row

    def _fit_order(self, dy: List[float], exog: List[List[float]], p: int,
                   seasonal: bool, use_exog: bool) -> Tuple[List[float], float, int]:
        start = max(p, 12 if seasonal else 0) + 1
        if len(dy) - start < 12:
            raise ValueError("样本不足以估计 SARIMAX")
        X = [self._row(dy, exog, t, p, seasonal, use_exog) for t in range(start, len(dy))]
        yv = dy[start:]
        # 外生列标准化，便于比较系数与数值稳定
        k = len(X[0])
        means = [0.0] * k
        sds = [1.0] * k
        for c in range(1, k):
            col = [r[c] for r in X]
            means[c] = numeric.mean(col)
            sds[c] = numeric.stdev(col) or 1.0
            if sds[c] == 0:
                sds[c] = 1.0
            for r in X:
                r[c] = (r[c] - means[c]) / sds[c]
        fit = numeric.ols(X, yv, ridge=1e-6)
        sse = sum(r * r for r in fit["resid"])
        n = len(yv)
        bic = n * math.log(max(sse / n, 1e-12)) + len(fit["beta"]) * math.log(n)
        self._means, self._sds = means, sds
        return fit["beta"], float(bic), start

    def fit(self, ds: DatasetBundle, harvest_months: Sequence[int], horizon: int,
            cols: Sequence[str], settings: Dict[str, Any]) -> "SarimaxLite":
        cfg = (settings.get("model", {}).get("sarimax") or {})
        p_max = int(cfg.get("p_max", 2))
        sp_max = int(cfg.get("sp_max", 1))
        include_exog = bool(cfg.get("include_exog", True))
        max_exog = int(cfg.get("max_exog", 3))

        self._y = numeric.log_series(ds.close)
        self._n = len(self._y)
        self._harvest = harvest_months
        self._horizon = horizon
        self._midx = [timeutil.parse_ym(m) for m in ds.months]
        self._dy = [self._y[0]] + [self._y[i] - self._y[i - 1] for i in range(1, self._n)]

        # 选外生：只取价格结构以外的基本面因子，按覆盖率排序截断
        cand = [c for c in cols if ds.coverage.get(c, 0) > 0.5]
        cand.sort(key=lambda c: -ds.coverage.get(c, 0))
        self._cols = cand[:max_exog] if include_exog else []
        self._exog = []
        for c in self._cols:
            series = ds.features[c]
            self._exog.append([0.0 if v is None else float(v) for v in series])
        use_exog = bool(self._exog)

        best: Tuple[float, int, bool, List[float]] | None = None
        for p in range(0, p_max + 1):
            for seasonal in ([False, True] if sp_max else [False]):
                try:
                    beta, bic, _ = self._fit_order(self._dy, self._exog, p, seasonal, use_exog)
                except Exception:
                    continue
                if best is None or bic < best[0]:
                    best = (bic, p, seasonal, beta)
        if best is None:
            # 退化为纯随机游走
            best = (0.0, 0, False, [0.0])
        bic, self._p, self._seasonal, self._beta = best
        # 重跑一次拿残差（保持同一标准化参数）
        beta, bic2, start = self._fit_order(self._dy, self._exog, self._p, self._seasonal, use_exog)
        self._beta = beta
        X = [self._row(self._dy, self._exog, t, self._p, self._seasonal, use_exog)
             for t in range(start, self._n)]
        resid = []
        for i, row in enumerate(X):
            pred = sum(row[j] * beta[j] for j in range(len(beta)))
            resid.append(self._dy[start + i] - pred)
        self._sigma_dy = clamp_sigma(numeric.stdev(resid) if len(resid) > 3 else 0.05, 0.005, 0.3)
        self._diag = {
            "order": f"(p={self._p}, d=1, q=0)(P={1 if self._seasonal else 0}, D=0, Q=0, s=12)",
            "bic": round(bic2, 2),
            "sigma_dy": round(self._sigma_dy, 5),
            "n_obs": len(resid),
            "r2": round(1 - sum(r * r for r in resid) /
                        max(1e-9, sum((v - numeric.mean(self._dy[start:])) ** 2
                                      for v in self._dy[start:])), 4),
            "ljung_box_p": round(numeric.ljung_box_pvalue(resid, lags=min(10, max(4, len(resid) // 5))), 4),
            "dw": round(numeric.durbin_watson(resid), 3) if len(resid) > 2 else None,
            "exog": self._cols,
        }
        return self

    def forecast_at(self, origin_t: int) -> Forecast:
        if self._n == 0:
            raise RuntimeError("模型未训练")
        origin_t = max(13, min(origin_t, self._n - 1))
        origin_idx = self._midx[origin_t]
        window = timeutil.harvest_window(self._harvest, origin_idx, self._horizon)
        max_step = max(0, window[-1] - origin_idx)
        dy = self._dy[:origin_t + 1]
        last_y = self._y[origin_t]
        use_exog = bool(self._exog)
        steps: Dict[int, float] = {origin_idx: last_y}
        level = last_y
        for s in range(1, max_step + 1):
            t = origin_idx + s
            # 构造与训练期同口径的滞后向量（用扩展后的 dy 副本）
            row = [1.0]
            for i in range(1, self._p + 1):
                row.append(dy[len(dy) - i])
            if self._seasonal:
                row.append(dy[len(dy) - 12] if len(dy) >= 12 else 0.0)
            if use_exog:
                for j in range(len(self._exog)):
                    row.append(self._exog[j][origin_t])
            for c in range(1, len(row)):
                row[c] = (row[c] - self._means[c]) / self._sds[c]
            d_hat = sum(row[j] * self._beta[j] for j in range(len(self._beta)))
            dy.append(d_hat)
            level += d_hat
            steps[t] = level
        preds = [steps.get(idx, level) for idx in window]
        mean_log = numeric.mean(preds)
        span = max(1, window[-1] - origin_idx)
        sigma = clamp_sigma(self._sigma_dy * (span ** 0.5))
        return Forecast(model=self.name, mean_log=mean_log, sigma=sigma, horizon=self._horizon,
                        diagnostics=dict(self._diag),
                        detail={"target_months": [timeutil.ym_str(i) for i in window]})


def fit_with_statsmodels(ds: DatasetBundle, horizon: int, exog_cols: Sequence[str],
                         order: Tuple[int, int, int] = (1, 1, 1),
                         seasonal_order: Tuple[int, int, int, int] = (0, 0, 1, 12)) -> Dict[str, Any]:
    """可选路径：用 statsmodels（BSD-3）做完整 SARIMAX，便于与轻量版对照。

    未安装 statsmodels 时返回 {"available": false}，不抛异常，保证主流程可跑。
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    except Exception:
        return {"available": False, "reason": "statsmodels 未安装"}
    y = numeric.log_series(ds.close)
    exog = []
    if exog_cols:
        exog = [[0.0 if ds.features[c][i] is None else float(ds.features[c][i])
                 for c in exog_cols] for i in range(ds.n())]
    try:
        model = SARIMAX(y, exog=exog or None, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=200)
        fc = res.get_forecast(steps=horizon, exog=None if not exog else [exog[-1]] * horizon)
        return {"available": True, "mean": list(fc.predicted_mean),
                "ci": fc.conf_int().tolist(), "aic": float(res.aic)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
