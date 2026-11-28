# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""滚动起点回测（rolling-origin / walk-forward）。

为什么不用随机 K 折：
时间序列随机切分会把未来信息混进训练集，指标虚高。这里固定用「扩张窗口 +
按时间向前推进」：每个起点只用该起点之前的数据训练，预测之后 6/9/12 个月的
收获窗口均价，再与真实值比对。

同时输出五类指标，覆盖不同关注点：
- MAE / RMSE：绝对误差（同量纲）
- MAPE：相对误差（便于跨品种比较）
- MASE：相对季节朴素基准，< 1 才说明模型有价值
- 方向命中率：涨跌判断是否正确
- 区间覆盖率：80% 区间实际覆盖比例是否接近 80%（校准度）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..core import numeric, timeutil
from ..data import features as feat
from ..data.schema import DatasetBundle
from . import ensemble as ens
from .base import Forecast, Forecaster, slice_dataset
from .gbm import GbmForecaster
from .sarimax import SarimaxLite
from .seasonal import SeasonalBaseline

MODEL_FACTORY = {
    "seasonal_baseline": SeasonalBaseline,
    "sarimax": SarimaxLite,
    "gbm": GbmForecaster,
}


def fit_models(ds: DatasetBundle, harvest_months: Sequence[int], horizon: int,
               cols: Sequence[str], settings: Dict[str, Any]) -> Dict[str, Forecaster]:
    models: Dict[str, Forecaster] = {}
    for name, cls in MODEL_FACTORY.items():
        try:
            models[name] = cls().fit(ds, harvest_months, horizon, cols, settings)
        except Exception as exc:  # noqa: BLE001
            models[name] = _Failed(name, str(exc))
    return models


class _Failed(Forecaster):
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        self._n = 0

    def forecast_at(self, origin_t: int) -> Forecast:
        raise RuntimeError(f"{self.name} 不可用: {self.reason}")


def _targets_map(ds: DatasetBundle, harvest: Sequence[int], horizon: int) -> Dict[int, Dict[str, Any]]:
    return {t["t"]: t for t in feat.build_targets(ds, harvest, horizon)}


def rolling_backtest(ds: DatasetBundle, harvest_months: Sequence[int],
                     horizons: Sequence[int], cols: Sequence[str],
                     settings: Dict[str, Any], max_origins: int = 16,
                     level: float = 0.8) -> Dict[str, Any]:
    out: Dict[str, Any] = {"variety": ds.variety, "level": level, "horizons": {}}
    for h in horizons:
        tmap = _targets_map(ds, harvest_months, h)
        origins = [t for t in sorted(tmap) if t <= ds.n() - 1]
        origins = origins[-max_origins:] if max_origins else origins
        folds: List[Dict[str, Any]] = []
        for t in origins:
            sub = slice_dataset(ds, t)
            models = fit_models(sub, harvest_months, h, cols, settings)
            rec: Dict[str, Any] = {"origin": ds.months[t], "origin_t": t,
                                   "target_month": tmap[t]["target_month"],
                                   "actual": round(tmap[t]["y_price"], 2),
                                   "base": round(ds.close[t], 2), "models": {}}
            forecasts: Dict[str, Forecast] = {}
            for name, m in models.items():
                try:
                    f = m.forecast_at(t)
                except Exception as exc:  # noqa: BLE001
                    rec["models"][name] = {"error": str(exc)}
                    continue
                forecasts[name] = f
                iv = f.interval(level)
                rec["models"][name] = {
                    "mean": round(f.mean_price, 2), "lo": round(iv["lo"], 2),
                    "hi": round(iv["hi"], 2), "sigma": round(f.sigma, 4),
                }
            # 集成权重：只用此前折的误差
            if forecasts:
                prev = folds[:]
                if len(prev) >= 3:
                    mse: Dict[str, float] = {}
                    for name in forecasts:
                        errs = []
                        for f in prev:
                            src = f["models"].get(name)
                            if src and "mean" in src:
                                errs.append(src["mean"] - f["actual"])
                        mse[name] = numeric.mean([e * e for e in errs]) if errs else 1e9
                    weights = ens.inverse_mse_weights(
                        mse, settings.get("model", {}).get("ensemble", {}).get("floor_weight", 0.05))
                else:
                    weights = {n: 1.0 / len(forecasts) for n in forecasts}
                combined = ens.combine(list(forecasts.values()), weights, level)
                civ = combined.interval(level)
                rec["ensemble"] = {"mean": round(combined.mean_price, 2),
                                   "lo": round(civ["lo"], 2), "hi": round(civ["hi"], 2),
                                   "sigma": round(combined.sigma, 4),
                                   "weights": {k: round(v, 3) for k, v in weights.items()}}
            folds.append(rec)
        target_series = [tmap[t]["y_price"] for t in sorted(tmap)]
        out["horizons"][str(h)] = {"folds": folds,
                                   "metrics": _metrics(folds, level, ds, h, harvest_months, target_series)}
    out["n_folds"] = {k: len(v["folds"]) for k, v in out["horizons"].items()}
    return out


def _metrics(folds: List[Dict[str, Any]], level: float, ds: DatasetBundle,
             h: int, harvest: Sequence[int], target_series: Sequence[float]) -> Dict[str, Any]:
    names = set()
    for f in folds:
        names |= set(f["models"].keys())
        if "ensemble" in f:
            names.add("ensemble")
    out: Dict[str, Any] = {}
    for name in sorted(names):
        pts = []
        for f in folds:
            src = f.get("ensemble") if name == "ensemble" else f["models"].get(name)
            if not src or "mean" not in src:
                continue
            pts.append((f["actual"], src["mean"], src.get("lo"), src.get("hi"), f["base"]))
        if len(pts) < 3:
            out[name] = {"status": "样本不足", "n": len(pts)}
            continue
        actual = [p[0] for p in pts]
        pred = [p[1] for p in pts]
        base = [p[4] for p in pts]
        cover = [1.0 if (p[2] is not None and p[2] <= p[0] <= p[3]) else 0.0 for p in pts]
        widths = [(p[3] - p[2]) / p[0] for p in pts if p[2] and p[3] and p[0]]
        out[name] = {
            "n": len(pts),
            "mae": round(numeric.mae(actual, pred), 2),
            "rmse": round(numeric.rmse(actual, pred), 2),
            "mape": round(numeric.mape(actual, pred), 2),
            "mase": round(numeric.mase(actual, pred, list(target_series)), 3),
            "dir_acc": round(numeric.direction_accuracy(actual, pred, base), 1),
            "coverage": round(numeric.mean(cover) * 100, 1),
            "avg_width": round(numeric.mean(widths) * 100, 1) if widths else None,
            "bias": round(numeric.mean([p - a for a, p in zip(actual, pred)]), 2),
        }
    ranked = [(k, v["rmse"]) for k, v in out.items() if "rmse" in v]
    out["_rank"] = [k for k, _ in sorted(ranked, key=lambda kv: kv[1])]
    if ranked:
        out["_best"] = out["_rank"][0]
    return out


def residual_report(folds: List[Dict[str, Any]], model: str = "ensemble") -> Dict[str, Any]:
    """残差诊断：分布、自相关、系统偏差。"""
    res: List[float] = []
    for f in folds:
        src = f.get("ensemble") if model == "ensemble" else f["models"].get(model)
        if src and "mean" in src:
            res.append(src["mean"] - f["actual"])
    if len(res) < 6:
        return {"n": len(res)}
    return {
        "n": len(res),
        "mean": round(numeric.mean(res), 2),
        "sd": round(numeric.stdev(res), 2),
        "q10": round(numeric.quantile(res, 0.1), 2),
        "q90": round(numeric.quantile(res, 0.9), 2),
        "min": round(min(res), 2),
        "max": round(max(res), 2),
        "dw": round(numeric.durbin_watson(res), 3),
        "acf1": round(numeric.acf(res, 1), 3),
        "ljung_box_p": round(numeric.ljung_box_pvalue(res, lags=min(5, len(res) // 3)), 4),
        "series": [round(v, 2) for v in res],
        "hit_rate": round(sum(1 for r in res if abs(r) < numeric.stdev(res)) / len(res) * 100, 1),
    }
