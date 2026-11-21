"""预测编排：清洗 → 特征 → 三模型 → 集成 → 情景 → 经济学校验 → 落盘。

一次 ``run_forecast`` 产出前端需要的全部数据，UI 不做任何计算，只做展示，
保证「页面上的每个数字都能在 JSON 里找到出处」。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..core import config, numeric, timeutil
from ..data import clean, features as feat, ingest
from ..econ import elasticity as elast
from ..econ import validator as econ
from . import backtest as bt
from . import ensemble as ens
from .base import Forecast

MAX_HISTORY_POINTS = 96


def _weights_from_backtest(metrics: Dict[str, Any]) -> Dict[str, float]:
    mse = {}
    for name, m in metrics.items():
        if name.startswith("_") or name == "ensemble" or not isinstance(m, dict):
            continue   # ensemble 是评价对象，不能作为自己的融合成员
        if m.get("rmse") is not None:
            mse[name] = float(m["rmse"]) ** 2
    if not mse:
        return {"seasonal_baseline": 0.34, "sarimax": 0.33, "gbm": 0.33}
    return ens.inverse_mse_weights(mse)


def load_dataset(variety_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    variety = config.get_variety(variety_id)
    status = ingest.market_status(variety_id)
    if status["rows"] < 24:
        raise RuntimeError(
            f"{variety['name_cn']} 尚无可用行情（当前 {status['rows']} 个月）。"
            f"请先获取初始数据：python -m src.cli init --variety {variety_id}")
    mrows, frows = ingest.load_local(variety_id)
    rows, logs_m = clean.to_monthly(mrows, settings["data"].get("monthly_agg", "last"),
                                    float(settings["data"].get("outlier_mad_k", 4.0)),
                                    int(settings["data"].get("max_forward_fill_months", 2)))
    fkept, logs_f = clean.dedupe_factors(frows, float(settings["llm"]["min_confidence"]),
                                        bool(settings["llm"]["require_source"]))
    ds = feat.build_dataset(variety_id, rows, fkept, variety["harvest_months"],
                            float(settings["llm"]["min_confidence"]),
                            bool(settings["llm"]["require_source"]))
    return {"variety": variety, "ds": ds, "market_rows": rows, "factors": fkept,
            "logs": logs_m + logs_f}


def run_forecast(variety_id: str, settings: Dict[str, Any], with_backtest: bool = True,
                 max_origins: int = 10, horizon: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.time()
    ctx = load_dataset(variety_id, settings)
    variety, ds = ctx["variety"], ctx["ds"]
    if ds.n() < 24:
        raise RuntimeError(f"{variety_id} 样本仅 {ds.n()} 个月，无法建模（至少 24 个月）")

    cols = feat.available_columns(ds)
    horizons = [horizon] if horizon else list(settings["model"]["horizons_months"])
    primary = horizons[0]
    level = float(settings["model"].get("interval", {}).get("level", 0.8))

    # 1) 回测（可选）→ 得到权重与评价指标
    bt_res: Dict[str, Any] = {"horizons": {}}
    if with_backtest:
        bt_res = bt.rolling_backtest(ds, variety["harvest_months"], horizons, cols, settings,
                                     max_origins=max_origins, level=level)

    # 2) 全样本重训 → 产出未来预测
    out_h: Dict[str, Any] = {}
    gbm_main = None
    gbm_base_log: Optional[float] = None
    seasonal_profile: List[Dict[str, Any]] = []
    for h in horizons:
        models = bt.fit_models(ds, variety["harvest_months"], h, cols, settings)
        forecasts: List[Forecast] = []
        for name, m in models.items():
            try:
                forecasts.append(m.forecast_at(ds.n() - 1))
            except Exception:  # noqa: BLE001
                continue
        if not forecasts:
            continue
        metrics = (bt_res.get("horizons", {}).get(str(h), {}) or {}).get("metrics", {})
        weights = _weights_from_backtest(metrics) if metrics else {
            n: 1.0 / len(forecasts) for n in [f.model for f in forecasts]}
        weights = {k: v for k, v in weights.items() if any(f.model == k for f in forecasts)}
        if not weights:
            weights = {f.model: 1.0 / len(forecasts) for f in forecasts}
        combo = ens.combine(forecasts, weights, level)
        iv = combo.interval(level)
        base_price = ds.close[-1]
        base_log = numeric.log_series([base_price])[0]
        prob_up = ens.probability_up(combo.mean_log, combo.sigma, base_log)
        out_h[str(h)] = {
            "horizon_months": h,
            "target_months": combo.detail.get("target_months"),
            "price": round(combo.mean_price, 2),
            "lo": round(iv["lo"], 2),
            "hi": round(iv["hi"], 2),
            "interval_level": level,
            "sigma": round(combo.sigma, 4),
            "base_price": round(base_price, 2),
            "prob_up": round(prob_up * 100, 1),
            "prob_down": round((1 - prob_up) * 100, 1),
            "expected_change_pct": round((combo.mean_price / base_price - 1) * 100, 2),
            "weights": {k: round(v, 3) for k, v in weights.items()},
            "models": [{"name": f.model, "price": round(f.mean_price, 2),
                        "lo": round(f.interval(level)["lo"], 2),
                        "hi": round(f.interval(level)["hi"], 2),
                        "sigma": round(f.sigma, 4),
                        "weight": round(weights.get(f.model, 0), 3),
                        "diagnostics": f.diagnostics} for f in forecasts],
            "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
            "best_model": metrics.get("_best"),
            "ranking": metrics.get("_rank", []),
        }
        if h == primary and "gbm" in models:
            gbm_main = models["gbm"]
            for f in forecasts:
                if f.model == "gbm":
                    gbm_base_log = f.mean_log
        if h == primary:
            sb = models.get("seasonal_baseline")
            if sb is not None and hasattr(sb, "season_profile"):
                seasonal_profile = sb.season_profile(ds)

    if not out_h:
        raise RuntimeError("所有模型均不可用，请检查数据与依赖")

    main = out_h[str(primary)]

    # 3) 情景分析：用结构弹性而非树模型外推（树无法外推，极端情景响应会被压平）
    est = elast.estimate(ds)
    scenarios = elast.scenarios(numeric.log_series([main["price"]])[0], main["sigma"], est)

    # 4) 经济学校验
    factor_quality = {
        "mean_confidence": round(numeric.mean([f.confidence for f in ctx["factors"]]), 3) if ctx["factors"] else 0,
        "count": len(ctx["factors"]),
        "dropped": sum(1 for l in ctx["logs"] if "置信度" in l.get("msg", "") or "来源" in l.get("msg", "")),
    }
    validation = econ.validate_forecast(
        variety, {"price": main["price"], "lo": main["lo"], "hi": main["hi"],
                  "prob_up": main["prob_up"], "far_close": ds.far_close[-1]},
        ds.summary(), seasonal_profile, scenarios, main["metrics"],
        settings["econ"], factor_quality)

    # 5) 残差诊断与灵敏度
    folds = (bt_res.get("horizons", {}).get(str(primary), {}) or {}).get("folds", [])
    residual = bt.residual_report(folds, "ensemble") if folds else {}
    sens = econ.sensitivity(numeric.log_series([main["price"]])[0], main["sigma"])

    result = {
        "schema_version": "1.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 1),
        "variety": {"id": variety["id"], "name": variety["name_cn"],
                    "exchange": variety["exchange"], "ticker": variety["ticker"],
                    "unit": variety["unit"], "harvest_months": variety["harvest_months"]},
        "as_of_month": ds.last_month(),
        "far_close": round(ds.far_close[-1], 2),
        "headline": out_h[str(primary)],
        "horizons": out_h,
        "scenarios": scenarios,
        "elasticity": est,
        "seasonal_profile": seasonal_profile,
        "validation": validation,
        "residual": residual,
        "sensitivity": sens,
        "data_summary": ds.summary(),
        "factor_quality": factor_quality,
        "logs": ctx["logs"][-40:],
        "backtest": {"folds": folds, "n_folds": len(folds)} if folds else {"folds": []},
        "history": [
            {"month": ds.months[i], "close": round(ds.close[i], 2),
             "far_close": round(ds.far_close[i], 2)}
            for i in range(max(0, ds.n() - MAX_HISTORY_POINTS), ds.n())
        ],
    }
    return result


def save_result(result: Dict[str, Any]) -> str:
    from ..data import store
    path = config.OUTPUT_DIR / f"forecast_{result['variety']['id']}.json"
    store.write_json(path, result)
    store.audit("forecast_saved", {"variety": result["variety"]["id"], "path": str(path),
                                   "price": result["headline"]["price"]})
    return str(path)
