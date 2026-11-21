"""模型公共契约。

统一接口的价值：回测、集成、前端展示都只认 ``Forecast``，
换模型不需要改调用方，也方便逐模型替换为 statsmodels / LightGBM 实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from ..core import numeric
from ..data.schema import DatasetBundle


@dataclass
class Forecast:
    model: str
    mean_log: float
    sigma: float
    horizon: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_price(self) -> float:
        return numeric.exp_series([self.mean_log])[0]

    def interval(self, level: float = 0.8) -> Dict[str, float]:
        z = numeric.norm_ppf(0.5 + level / 2)
        lo = self.mean_log - z * self.sigma
        hi = self.mean_log + z * self.sigma
        return {"lo_log": lo, "hi_log": hi,
                "lo": numeric.exp_series([lo])[0],
                "hi": numeric.exp_series([hi])[0],
                "level": level}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean_price"] = round(self.mean_price, 2)
        return d


class Forecaster:
    """子类实现 fit / forecast_at。"""

    name = "base"
    needs_cols: bool = False

    def fit(self, ds: DatasetBundle, harvest_months: Sequence[int], horizon: int,
            cols: Sequence[str], settings: Dict[str, Any]) -> "Forecaster":
        raise NotImplementedError

    def forecast_at(self, origin_t: int) -> Forecast:
        raise NotImplementedError

    # -- 供回测复用 -------------------------------------------------------
    def forecast_last(self) -> Forecast:
        return self.forecast_at(self._n - 1)


def slice_dataset(ds: DatasetBundle, end_t: int) -> DatasetBundle:
    """截取 [0, end_t] 的样本，模拟「站在历史上某个月做预测」。"""
    end_t = max(0, min(end_t, ds.n() - 1))
    sub = DatasetBundle(
        variety=ds.variety,
        months=ds.months[:end_t + 1],
        close=ds.close[:end_t + 1],
        far_close=ds.far_close[:end_t + 1],
        features={k: v[:end_t + 1] for k, v in ds.features.items()},
        feature_meta=ds.feature_meta,
        coverage=ds.coverage,
        warnings=list(ds.warnings),
    )
    return sub


def residual_sigma(actual: Sequence[float], pred: Sequence[float], k: int = 2) -> float:
    """小样本残差标准差：自由度修正，避免过度自信。"""
    n = len(actual)
    if n <= k:
        return 0.10
    s = numeric.stdev([a - p for a, p in zip(actual, pred)])
    return s if s == s and s > 1e-6 else 0.08


def clamp_sigma(sigma: float, lo: float = 0.015, hi: float = 0.45) -> float:
    return max(lo, min(hi, sigma))
