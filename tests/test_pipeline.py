# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""核心链路单元测试（纯标准库 unittest，零依赖可跑）。

    python -m src.cli selftest
    python -m unittest tests.test_pipeline -v
"""

from __future__ import annotations

import json
import unittest

from src.core import numeric, timeutil
from src.data import clean, features as feat, ingest
from src.data.schema import FactorRow, MarketRow
from src.econ import elasticity, validator
from src.llm import extract as llm_extract
from src.llm import jsonschema_lite, prompt_builder
from src.models import backtest, ensemble, gbm as gbm_mod, sarimax, seasonal


class TestTimeMath(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(timeutil.ym_str(timeutil.parse_ym("2026-09")), "2026-09")
        self.assertEqual(timeutil.ym_str(timeutil.parse_ym("202609")), "2026-09")
        self.assertEqual(timeutil.month_of_year(timeutil.parse_ym("2026-01")), 1)

    def test_add_months_across_year(self):
        idx = timeutil.parse_ym("2026-11") + 6
        self.assertEqual(timeutil.ym_str(idx), "2027-05")

    def test_harvest_window_lands_on_harvest_month(self):
        start = timeutil.parse_ym("2026-11")
        win = timeutil.harvest_window([9, 10, 11], start, 6)
        self.assertTrue(all(timeutil.month_of_year(w) in (9, 10, 11) for w in win))
        self.assertTrue(all(timeutil.ym_str(w).startswith("2027") for w in win))

    def test_harvest_window_all_year_uses_target(self):
        start = timeutil.parse_ym("2026-03")
        win = timeutil.harvest_window(list(range(1, 13)), start, 6)
        self.assertEqual(len(win), 1)
        self.assertEqual(timeutil.ym_str(win[0]), "2026-09")

    def test_harvest_window_snaps_to_next_harvest(self):
        """11 月起点 + 6 个月落在 5 月（非收获月），应吸附到最近收获季 9 月。"""
        win = timeutil.harvest_window([9, 10, 11], timeutil.parse_ym("2026-11"), 6)
        self.assertEqual([timeutil.ym_str(w) for w in win], ["2027-09", "2027-10", "2027-11"])


class TestNumeric(unittest.TestCase):
    def test_norm_ppf_roundtrip(self):
        for p in (0.1, 0.5, 0.8, 0.95):
            self.assertAlmostEqual(numeric.norm_cdf(numeric.norm_ppf(p)), p, places=6)

    def test_ols_recovers_line(self):
        X = [[1.0, float(i)] for i in range(30)]
        y = [3.0 + 2.0 * i for i in range(30)]
        fit = numeric.ols(X, y)
        self.assertAlmostEqual(fit["beta"][0], 3.0, places=6)
        self.assertAlmostEqual(fit["beta"][1], 2.0, places=6)
        self.assertGreater(fit["r2"], 0.999)

    def test_ljung_box_white_noise(self):
        import random
        rnd = random.Random(7)
        resid = [rnd.gauss(0, 1) for _ in range(200)]
        self.assertGreater(numeric.ljung_box_pvalue(resid, lags=10), 0.01)

    def test_metrics(self):
        self.assertAlmostEqual(numeric.mae([1, 2], [2, 4]), 1.5)
        self.assertAlmostEqual(numeric.rmse([1, 2], [2, 4]), (2.5) ** 0.5)
        self.assertEqual(numeric.direction_accuracy([11, 9], [10.5, 9.5], [10, 10]), 100.0)


class TestClean(unittest.TestCase):
    def test_unit_normalize(self):
        v, unit, _ = clean.normalize_unit(1.0, "元/斤")
        self.assertAlmostEqual(v, 2000.0)
        self.assertEqual(unit, "元/吨")
        v2, unit2, logs = clean.normalize_unit(3.0, "未知单位")
        self.assertEqual(v2, 3.0)
        self.assertTrue(logs)

    def test_monthly_aggregation_and_fill(self):
        rows = [MarketRow(month="2026-01", close=100, far_close=101),
                MarketRow(month="2026-01", close=110, far_close=111),
                MarketRow(month="2026-03", close=130, far_close=131)]
        out, logs = clean.to_monthly(rows, "last", mad_k=99, max_ffill=2)
        months = [r.month for r in out]
        self.assertEqual(months, ["2026-01", "2026-02", "2026-03"])
        self.assertEqual(out[0].close, 110)   # 同月取最后一条
        self.assertEqual(out[1].close, 110)   # 前值填充
        self.assertTrue(any("填充" in l["msg"] for l in logs))

    def test_outlier_replaced_not_dropped(self):
        prices = [100, 101, 99, 100, 1000, 101, 100, 99, 100, 101, 100, 99]
        rows = [MarketRow(month=timeutil.ym_str(timeutil.parse_ym("2024-01") + i),
                          close=p, far_close=p) for i, p in enumerate(prices)]
        out, logs = clean.to_monthly(rows, "last", mad_k=3.0)
        self.assertEqual(len(out), 12)                       # 一行不少
        self.assertLess(out[4].close, 200)                   # 异常被修正
        self.assertTrue(any("异常" in l["msg"] for l in logs))

    def test_dedupe_keeps_newest_and_drops_low_quality(self):
        base = dict(variety="corn", factor_id="production_est", unit="万吨",
                    period="2026-08", direction="bearish")
        rows = [
            FactorRow(as_of="2026-08-10", value=100, confidence=0.8,
                      source_url="http://a", **base),
            FactorRow(as_of="2026-09-10", value=105, confidence=0.9,
                      source_url="http://b", **base),
            FactorRow(as_of="2026-09-11", value=999, confidence=0.2,
                      source_url="http://c", **base),
            FactorRow(as_of="2026-09-12", value=888, confidence=0.9,
                      source_url="N/A", **base),
        ]
        kept, logs = clean.dedupe_factors(rows, min_conf=0.55, require_source=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].value, 105)
        self.assertTrue(any("置信度" in l["msg"] for l in logs))


class TestFeaturesNoLookahead(unittest.TestCase):
    def _ds(self, extra: float = 0.0):
        market, _ = ingest.generate_demo(
            {"id": "corn", "name_cn": "玉米", "harvest_months": [9, 10, 11],
             "key_factors": [], "ticker": "C"}, years=4, seed=11)
        if extra:
            market = market[:-1] + [MarketRow(month=market[-1].month,
                                              close=market[-1].close * (1 + extra),
                                              far_close=market[-1].far_close)]
        return feat.build_dataset("corn", market, [], [9, 10, 11])

    def test_future_price_does_not_change_past_features(self):
        a = self._ds(0.0)
        b = self._ds(0.9)
        for name in ("trend12", "yoy", "season_idx", "vol6"):
            for i in range(a.n() - 1):
                va, vb = a.features[name][i], b.features[name][i]
                if va is None or vb is None:
                    continue
                self.assertAlmostEqual(va, vb, places=9, msg=f"{name}[{i}] 受未来数据影响")

    def test_target_is_after_origin(self):
        ds = self._ds()
        targets = feat.build_targets(ds, [9, 10, 11], 6)
        for t in targets[:5]:
            self.assertGreater(t["target_first"], t["month"])


def _demo_dataset(variety=None, years=10):
    v = variety or {"id": "corn", "name_cn": "玉米", "harvest_months": [9, 10, 11],
                    "key_factors": [], "ticker": "C"}
    market, factors = ingest.generate_demo(v, years=years, seed=2026)
    ds = feat.build_dataset(v["id"], market, factors, v["harvest_months"], 0.5, True)
    return ds, v, factors


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds, cls.v, cls.factors = _demo_dataset()
        cls.cols = feat.available_columns(cls.ds)
        cls.settings = json.loads((__import__("pathlib").Path("config/settings.example.json")
                                   .read_text(encoding="utf-8")))

    def test_dataset_has_fundamentals(self):
        self.assertGreater(self.ds.n(), 100)
        self.assertTrue(self.ds.features.get("production_est"))
        self.assertGreater(self.ds.coverage.get("production_est", 0), 0.8)

    def test_seasonal_baseline_forecast_sane(self):
        m = seasonal.SeasonalBaseline().fit(self.ds, [9, 10, 11], 6, self.cols, self.settings)
        f = m.forecast_at(self.ds.n() - 1)
        iv = f.interval(0.8)
        self.assertTrue(iv["lo"] < f.mean_price < iv["hi"])
        self.assertTrue(500 < f.mean_price < 6000)
        self.assertEqual(len(f.detail["target_months"]), len(timeutil.harvest_window([9, 10, 11], timeutil.parse_ym(self.ds.months[-1]), 6)))

    def test_sarimax_fits_and_forecasts(self):
        m = sarimax.SarimaxLite().fit(self.ds, [9, 10, 11], 6, self.cols, self.settings)
        f = m.forecast_at(self.ds.n() - 1)
        self.assertTrue(0.1 < f.sigma < 0.6)
        self.assertTrue(f.diagnostics["order"].startswith("(p="))
        self.assertGreater(f.mean_price, 500)

    def test_gbm_learns_supply_relation(self):
        m = gbm_mod.GbmForecaster().fit(self.ds, [9, 10, 11], 6, self.cols, self.settings)
        f = m.forecast_at(self.ds.n() - 1)
        self.assertGreater(f.mean_price, 500)
        # 供给宽松（产量+1σ）应压价：方向必须为负
        if "production_est" in self.cols:
            base = m.predict_vector(m._last_x)
            loose = m.scenario_predict({"production_est": 1.0})
            self.assertLess(loose, base + 1e-6)
        self.assertIn("backend", f.diagnostics)

    def test_ensemble_weights_and_probability(self):
        fs = [seasonal.SeasonalBaseline().fit(self.ds, [9, 10, 11], 6, self.cols, self.settings)
              .forecast_at(self.ds.n() - 1)]
        w = {"seasonal_baseline": 1.0}
        c = ensemble.combine(fs, w)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertGreaterEqual(c.sigma, fs[0].sigma * 0.99)
        p = ensemble.probability_up(c.mean_log, c.sigma, c.mean_log)
        self.assertAlmostEqual(p, 0.5, places=6)

    def test_elasticity_sign(self):
        est = elasticity.estimate(self.ds)
        self.assertTrue(est["available"])
        self.assertLess(est["beta_supply"], 0)   # 供给宽松 → 价格下行
        scen = elasticity.scenarios(7.8, 0.1, est)
        prices = {s["key"]: s["price"] for s in scen}
        self.assertLess(prices["bumper"], prices["base"])
        self.assertLess(prices["base"], prices["shortfall"])

    def test_rolling_backtest_metrics(self):
        res = backtest.rolling_backtest(self.ds, [9, 10, 11], [6], self.cols, self.settings,
                                        max_origins=4)
        m = res["horizons"]["6"]["metrics"]
        self.assertIn("ensemble", m)
        self.assertEqual(m["ensemble"]["n"], 4)
        self.assertGreater(m["ensemble"]["rmse"], 0)
        self.assertTrue(0 <= m["ensemble"]["coverage"] <= 100)


class TestEconValidation(unittest.TestCase):
    def setUp(self):
        self.v = {"id": "corn", "name_cn": "玉米", "harvest_months": [9, 10, 11],
                  "sanity_price_band": [1400, 3600]}
        self.cfg = {"far_month_tolerance": 0.18, "max_interval_width": 0.75,
                    "min_probability": 0.03, "block_on_error": False}

    def test_detects_inverted_scenarios(self):
        scen = [{"key": "bumper", "price": 2600}, {"key": "base", "price": 2500},
                {"key": "shortfall", "price": 2400}]
        out = validator.validate_forecast(
            self.v, {"price": 2500, "lo": 2300, "hi": 2700, "prob_up": 50},
            {"n_months": 120, "coverage": {}}, [], scen, {}, self.cfg)
        self.assertEqual(out["status"], "fail")
        self.assertTrue(any(c["id"] == "scenario_order" and c["level"] == "error" for c in out["checks"]))

    def test_flags_price_out_of_band(self):
        out = validator.validate_forecast(
            self.v, {"price": 9000, "lo": 8000, "hi": 10000, "prob_up": 50},
            {"n_months": 120, "coverage": {}}, [], [], {}, self.cfg)
        self.assertTrue(any(c["id"] == "price_band" and c["level"] == "error" for c in out["checks"]))

    def test_flags_short_sample_and_wide_interval(self):
        out = validator.validate_forecast(
            self.v, {"price": 2500, "lo": 1000, "hi": 4000, "prob_up": 50},
            {"n_months": 20, "coverage": {}}, [], [], {}, self.cfg)
        ids = {c["id"] for c in out["checks"] if c["level"] == "warn"}
        self.assertIn("sample", ids)
        self.assertIn("interval_width", ids)


class TestLLMSchema(unittest.TestCase):
    def _payload(self):
        return {
            "schema_version": "1.0.0", "variety_id": "corn",
            "as_of_date": "2026-09-30", "research_window": "2026-01-01/2026-09-30",
            "factors": [{
                "factor_id": "production_est", "value": 28840, "unit": "万吨",
                "period": "2026-09", "as_of_date": "2026-08-12",
                "direction": "bearish", "confidence": 0.82,
                "source_title": "供需形势分析", "source_url": "http://example.gov.cn/a",
                "source_date": "2026-08-12", "evidence_quote": "产量 28840 万吨。", "notes": "",
            }],
            "events": [], "data_gaps": [], "refusal": False, "refusal_reason": None,
            "meta": {"model": "test", "generated_at": "2026-09-30T10:00:00",
                     "search_queries": ["玉米 产量"], "mode": "manual"},
        }

    def test_valid_payload_passes(self):
        schema = prompt_builder.load_schema()
        ok, errs = jsonschema_lite.validate(self._payload(), schema)
        self.assertTrue(ok, errs)

    def test_missing_field_rejected(self):
        p = self._payload()
        del p["meta"]
        ok, errs = jsonschema_lite.validate(p, prompt_builder.load_schema())
        self.assertFalse(ok)
        self.assertTrue(any("meta" in e for e in errs))

    def test_extra_field_rejected(self):
        p = self._payload()
        p["price_forecast"] = 3000
        ok, errs = jsonschema_lite.validate(p, prompt_builder.load_schema())
        self.assertFalse(ok)

    def test_bad_enum_rejected(self):
        p = self._payload()
        p["factors"][0]["direction"] = "up"
        ok, errs = jsonschema_lite.validate(p, prompt_builder.load_schema())
        self.assertFalse(ok)

    def test_business_check_rejects_future_and_price_prediction(self):
        p = self._payload()
        p["factors"][0]["as_of_date"] = "2026-12-01"
        ok, errs = llm_extract.business_checks(p, "corn")
        self.assertFalse(ok)
        p2 = self._payload()
        p2["factors"][0]["evidence_quote"] = "机构预计涨到 3200 元/吨"
        ok2, errs2 = llm_extract.business_checks(p2, "corn")
        self.assertFalse(ok2)
        self.assertTrue(any("价格预测" in e for e in errs2))

    def test_prompt_render_has_no_placeholder_left(self):
        v = {"id": "corn", "name_cn": "玉米", "exchange": "DCE", "ticker": "C",
             "unit": "元/吨", "harvest_months": [9, 10, 11],
             "key_factors": ["production_est"], "sanity_price_band": [1400, 3600]}
        r = prompt_builder.render(v, "2026-09-30")
        self.assertNotIn("{{", r["user"])
        self.assertEqual(len(r["hash"]), 64)
        self.assertIn("玉米", r["user"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
