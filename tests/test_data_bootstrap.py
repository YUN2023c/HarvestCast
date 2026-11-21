"""初始数据获取链路的单元测试。

重点验证「没有初始数据时先获取，且绝不用合成数据顶替真实行情」：
- 有网络通道时写入真实行情并记录 meta；
- 通道全失败且未显式允许 demo 时，必须报错且不落盘任何行情；
- 显式 --allow-demo 时才生成合成数据；
- 真实行情落地时，残留的合成因子必须被隔离（改名而非删除）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.core import config
from src.data import ingest
from src.data.schema import FactorRow, MarketRow


def _fake_rows(n: int = 80, base: float = 2400.0):
    rows = []
    start = 2020 * 12
    for i in range(n):
        from src.core import timeutil
        rows.append(MarketRow(month=timeutil.ym_str(start + i),
                              close=base + i, far_close=base + i + 20,
                              volume=1000, open_interest=2000, contract="C0"))
    return rows


def _fake_fetch(variety, horizon=12, **kw):
    return _fake_rows(), {"source": "sina", "far_symbol": "C2709", "note": "test"}


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.market = root / "market"
        self.fund = root / "fundamentals"
        self.market.mkdir()
        self.fund.mkdir()
        self.patches = [
            mock.patch.object(ingest, "MARKET_DIR", self.market),
            mock.patch.object(ingest, "FUND_DIR", self.fund),
            mock.patch.object(ingest, "META_PATH", self.market / "_meta.json"),
        ]
        for p in self.patches:
            p.start()
        self.variety = config.get_variety("corn")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_fetch_writes_market_and_meta(self):
        with mock.patch.object(ingest, "fetch_market_sina", side_effect=_fake_fetch):
            res = ingest.bootstrap([self.variety], min_months=60)
        self.assertEqual(res["ok"], 1, res)
        self.assertTrue((self.market / "corn.csv").exists())
        self.assertEqual(len(ingest.store.read_csv(self.market / "corn.csv")), 80)
        meta = ingest.store.read_json(self.market / "_meta.json", {})
        self.assertEqual(meta["corn"]["source"], "sina")
        st = ingest.market_status("corn")
        self.assertTrue(st["has_data"])
        self.assertEqual(st["rows"], 80)

    def test_second_run_skips_existing(self):
        with mock.patch.object(ingest, "fetch_market_sina", side_effect=_fake_fetch):
            ingest.bootstrap([self.variety], min_months=60)
            res = ingest.bootstrap([self.variety], min_months=60)
        self.assertEqual(res["ok"], 0)
        self.assertEqual(res["skipped"], 1)

    def test_failure_does_not_fabricate_data(self):
        """两条真实通道都失败时必须如实报错，且不写任何行情文件。"""
        with mock.patch.object(ingest, "fetch_market_sina", side_effect=ingest.FetchError("net down")), \
             mock.patch.object(ingest, "fetch_market_akshare", side_effect=ImportError("no akshare")):
            res = ingest.bootstrap([self.variety], min_months=60)
        self.assertEqual(res["failed"], 1)
        self.assertEqual(res["ok"], 0)
        self.assertFalse((self.market / "corn.csv").exists())
        self.assertIn("net down", res["items"][0]["message"])

    def test_allow_demo_is_explicit(self):
        res = ingest.bootstrap([self.variety], min_months=60, allow_demo=True)
        self.assertEqual(res["ok"], 1)
        self.assertEqual(res["items"][0]["source"], "demo")
        self.assertTrue((self.fund / "corn.csv").exists())

    def test_stale_demo_factors_are_quarantined(self):
        # 先放一份「合成」因子
        demo = FactorRow(as_of="2026-09-10", variety="corn", factor_id="production_est",
                         value=28000, unit="万吨", period="2026-08", direction="bearish",
                         confidence=0.8, source_title="示例", source_url="https://example.gov.cn/demo",
                         evidence="synthetic")
        ingest.store.write_csv(self.fund / "corn.csv", [demo.to_dict()], ingest.FUND_FIELDS)
        with mock.patch.object(ingest, "fetch_market_sina", side_effect=_fake_fetch):
            res = ingest.bootstrap([self.variety], min_months=60)
        self.assertEqual(res["ok"], 1)
        self.assertFalse((self.fund / "corn.csv").exists(), "合成因子应被移走")
        backups = list(self.fund.glob("corn.csv.demo-*.bak"))
        self.assertEqual(len(backups), 1, "只改名不删除")

    def test_real_factors_are_kept(self):
        real = FactorRow(as_of="2026-09-10", variety="corn", factor_id="production_est",
                         value=28000, unit="万吨", period="2026-08", direction="bearish",
                         confidence=0.8, source_title="农业农村部", source_url="http://www.moa.gov.cn/x",
                         evidence="真实来源")
        ingest.store.write_csv(self.fund / "corn.csv", [real.to_dict()], ingest.FUND_FIELDS)
        with mock.patch.object(ingest, "fetch_market_sina", side_effect=_fake_fetch):
            ingest.bootstrap([self.variety], min_months=60)
        self.assertTrue((self.fund / "corn.csv").exists(), "真实因子不应被隔离")


class TestJsonp(unittest.TestCase):
    def test_parse_prefix_and_parens(self):
        text = '/*<script>location.href="//sina.com";</script>*/\nvar _C0=([{"d":"2026-01-05","c":"2400"}]);'
        rows = ingest.parse_jsonp_rows(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["c"], "2400")

    def test_parse_empty(self):
        self.assertEqual(ingest.parse_jsonp_rows("var _X=();"), [])

    def test_contract_symbol(self):
        v = config.get_variety("corn")
        self.assertEqual(ingest.contract_symbol(v, 2027, 9), "C2709")
        self.assertEqual(ingest.contract_symbol(config.get_variety("rubber"), 2027, 1), "RU2701")


class TestVarietyCatalog(unittest.TestCase):
    def test_catalog_is_expanded_and_consistent(self):
        doc = config.load_varieties()
        vs = doc["varieties"]
        self.assertGreaterEqual(len(vs), 20, "品种数量应显著扩展")
        self.assertEqual(len({v["id"] for v in vs}), len(vs), "id 必须唯一")
        self.assertEqual(len({v["ticker"] for v in vs}), len(vs), "ticker 必须唯一")
        cats = set(doc["categories"])
        factors = set(doc["factors"].keys())
        for v in vs:
            self.assertIn(v["category"], cats, v["id"])
            self.assertTrue(set(v["harvest_months"]) <= set(range(1, 13)), v["id"])
            self.assertTrue(set(v["contract_cycle"]) <= set(range(1, 13)), v["id"])
            self.assertLess(v["sanity_price_band"][0], v["sanity_price_band"][1], v["id"])
            self.assertTrue(set(v["key_factors"]) <= factors, v["id"])
            self.assertTrue(v["ticker"].isalpha() and v["ticker"].isupper(), v["id"])

    def test_all_categories_populated(self):
        vs = config.variety_list()
        cats = {v["category"] for v in vs}
        self.assertGreaterEqual(len(cats), 4)
        for c in cats:
            self.assertGreaterEqual(sum(1 for v in vs if v["category"] == c), 2, c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
