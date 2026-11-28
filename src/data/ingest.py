# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""数据接入层：初始行情获取 + 离线合成样本。

三条通道，优先级从高到低：

1. ``fetch_market_sina``    —— 新浪财经期货日线（**纯标准库 urllib，无需安装任何包**）。
   主力连续 `<ticker>0` 取月度收盘；收获季远月合约（如 C2709）取远月价。
2. ``fetch_market_akshare`` —— AKShare（MIT）适配器，作为备用通道。
3. ``generate_demo``        —— 合成数据，**只用于离线演示与单测**，必须显式 `--allow-demo`。

⚠️ 数据授权：新浪/交易所行情仅用于个人研究与演示；商用前必须获得数据源授权。
   本项目不随附任何真实行情数据，首次运行需主动拉取。
"""

from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import timeutil
from ..core.config import FUND_DIR, MARKET_DIR
from . import store
from .schema import FUND_FIELDS, MARKET_FIELDS, FactorRow, MarketRow

SINA_KLINE = ("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
              "var%20_{sym}=/InnerFuturesNewService.getDailyKLine?symbol={sym}")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HarvestCast/2.0"
META_PATH = MARKET_DIR / "_meta.json"
MIN_MONTHS_DEFAULT = 60


class FetchError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# 1. 新浪期货日线（标准库）
# --------------------------------------------------------------------------


def _http_get(url: str, timeout: int = 20, retries: int = 2) -> str:
    last = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Referer": "https://finance.sina.com.cn/",
                "Accept": "*/*",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
            time.sleep(0.6 * (attempt + 1))
    raise FetchError(f"请求失败: {url} ({last})")


def parse_jsonp_rows(text: str) -> List[Dict[str, Any]]:
    """从 JSONP 文本中取出数组。容错处理前缀脚本与外层括号。"""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise FetchError(f"JSONP 解析失败: {exc}") from exc
    return data if isinstance(data, list) else []


def fetch_sina_daily(symbol: str, timeout: int = 20) -> List[Dict[str, Any]]:
    """取单个合约的日线。symbol 形如 C0 / C2709 / SR2701。"""
    text = _http_get(SINA_KLINE.format(sym=symbol), timeout=timeout)
    rows = parse_jsonp_rows(text)
    out = []
    for r in rows:
        try:
            out.append({
                "date": str(r["d"])[:10],
                "open": float(r.get("o") or 0),
                "high": float(r.get("h") or 0),
                "low": float(r.get("l") or 0),
                "close": float(r["c"]),
                "settle": float(r.get("s") or r["c"]),
                "volume": float(r.get("v") or 0),
                "open_interest": float(r.get("p") or 0),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def contract_symbol(variety: Dict[str, Any], year: int, month: int) -> str:
    return f"{variety['ticker']}{year % 100:02d}{month:02d}"


def harvest_contracts(variety: Dict[str, Any], start_month: int, horizon: int,
                      max_n: int = 5) -> List[str]:
    """生成「收获季合约」候选串（按时间升序），覆盖近 4 年与未来 2 年。

    远月价不是只取一个合约：为了让基差特征有历史，需要按年份各取一个收获季合约，
    再按「离目标期最近且尚未到期」逐月选出当月远月价。
    """
    y0, _ = timeutil.from_index(start_month)
    tick = variety["ticker"]
    cycle = variety.get("contract_cycle") or [1, 5, 9]
    harvest = variety["harvest_months"]
    syms: List[str] = []
    for year in range(y0 - 4, y0 + 3):
        months = [m for m in cycle if m in harvest] or list(cycle)
        # 取离收获季最近的合约月
        target_m = harvest[0] if harvest else months[0]
        m = min(months, key=lambda x: min(abs(x - target_m), 12 - abs(x - target_m)))
        sym = contract_symbol(variety, year, m)
        idx = timeutil.ym_index(year, m)
        # 不排除已到期合约：历史月份的远月价正是由这些合约提供的。
        # 只排除过远的未来合约（尚未挂牌或几乎没有成交）。
        if idx > start_month + horizon + 18:
            continue
        if sym not in syms:
            syms.append(sym)
    return syms[-max_n:]


def fetch_market_sina(variety: Dict[str, Any], horizon: int = 12,
                      timeout: int = 20, sleep: float = 0.35,
                      max_contracts: int = 4) -> Tuple[List[MarketRow], Dict[str, Any]]:
    """拉取真实行情并聚合为月度。返回 (rows, meta)。"""
    symbol = f"{variety['ticker']}0"
    daily = fetch_sina_daily(symbol, timeout=timeout)
    if len(daily) < 200:
        raise FetchError(f"{variety['name_cn']} {symbol} 日线仅 {len(daily)} 条，样本不足")
    time.sleep(sleep)
    start_month = timeutil.parse_ym(daily[-1]["date"][:7])

    # 多个收获季合约 -> 逐月挑「离目标期最近且未到期」的远月价
    far_series: Dict[str, Dict[str, float]] = {}
    contract_idx: Dict[str, int] = {}
    far_note = ""
    for sym in harvest_contracts(variety, start_month, horizon, max_n=max_contracts):
        try:
            rows = fetch_sina_daily(sym, timeout=timeout)
        except FetchError:
            continue
        if len(rows) < 5:
            continue
        tail = sym[len(variety["ticker"]):]
        contract_idx[sym] = timeutil.ym_index(2000 + int(tail[:2]), int(tail[2:]))
        far_series[sym] = {r["date"][:7]: r["close"] for r in rows if r["close"] > 0}
        time.sleep(sleep)
    if not far_series:
        far_note = "远月合约均不可用，基差按 0 处理"

    by_month: Dict[str, List[Dict[str, Any]]] = {}
    for r in daily:
        by_month.setdefault(r["date"][:7], []).append(r)

    def far_for(month: str) -> Tuple[float, str]:
        midx = timeutil.parse_ym(month)
        best: Tuple[int, float, str] | None = None
        for sym, series in far_series.items():
            if contract_idx[sym] <= midx + 1:      # 已到期，不能作为远月
                continue
            val = series.get(month)
            if val is None:
                continue
            d = abs(contract_idx[sym] - (midx + horizon))
            if best is None or d < best[0]:
                best = (d, val, sym)
        return (best[1], best[2]) if best else (0.0, "")

    rows: List[MarketRow] = []
    basis_months = 0
    last_far_symbol = ""
    for month in sorted(by_month):
        group = sorted(by_month[month], key=lambda x: x["date"])
        last = group[-1]
        close = last["close"] if last["close"] > 0 else last["settle"]
        if close <= 0:
            continue
        far, sym = far_for(month)
        if far and abs(far / close - 1) < 0.6:      # 量级校验，防错配合约
            basis_months += 1
            last_far_symbol = sym
        else:
            far = close
        rows.append(MarketRow(month=month, close=close, far_close=far,
                              volume=sum(g["volume"] for g in group),
                              open_interest=last["open_interest"], contract=symbol))
    meta = {
        "source": "sina",
        "symbol": symbol,
        "far_symbol": last_far_symbol,
        "contracts": sorted(far_series.keys()),
        "far_contract_months": basis_months,
        "rows": len(rows),
        "first": rows[0].month if rows else "",
        "last": rows[-1].month if rows else "",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": far_note or f"远月价逐月取自 {len(far_series)} 个收获季合约",
    }
    return rows, meta


# --------------------------------------------------------------------------
# 2. AKShare 备用通道
# --------------------------------------------------------------------------


def fetch_market_akshare(variety_id: str, start_date: str = "20060101") -> List[MarketRow]:
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:
        raise ImportError("未安装 akshare，请 pip install akshare") from exc
    from ..core.config import get_variety
    variety = get_variety(variety_id)
    symbol = f"{variety['ticker']}0"
    df = ak.futures_zh_daily_sina(symbol=symbol)
    rows: List[MarketRow] = []
    for _, r in df.iterrows():
        d = str(r["date"])[:10]
        if d.replace("-", "") < start_date:
            continue
        try:
            close = float(r["close"])
            settle = float(r.get("settle", close)) or close
        except Exception:
            continue
        rows.append(MarketRow(month=d[:7], close=close, far_close=settle,
                              volume=float(r.get("volume", 0) or 0),
                              open_interest=float(r.get("hold", 0) or 0), contract=symbol))
    return rows


# --------------------------------------------------------------------------
# 3. 初始数据引导（没有数据时先获取，不静默造假）
# --------------------------------------------------------------------------


def _market_path(variety_id: str):
    return MARKET_DIR / f"{variety_id}.csv"


def _read_meta() -> Dict[str, Any]:
    return store.read_json(META_PATH, {}) or {}


def _write_meta(meta: Dict[str, Any]) -> None:
    store.write_json(META_PATH, meta)


def market_status(variety_id: str) -> Dict[str, Any]:
    rows = store.read_csv(_market_path(variety_id))
    meta = _read_meta().get(variety_id, {})
    return {
        "variety": variety_id,
        "has_data": len(rows) > 0,
        "rows": len(rows),
        "first": rows[0]["date"] if rows else "",
        "last": rows[-1]["date"] if rows else "",
        "source": meta.get("source", "unknown"),
        "far_symbol": meta.get("far_symbol", ""),
        "fetched_at": meta.get("fetched_at", ""),
        "note": meta.get("note", ""),
    }


def quarantine_demo_factors(variety_id: str) -> Optional[str]:
    """真实行情落地时，隔离残留的合成因子，避免「真价格 + 假基本面」混用。

    只改名不删除（符合不删用户数据的约束），返回隔离后的文件名。
    """
    path = FUND_DIR / f"{variety_id}.csv"
    rows = store.read_csv(path)
    if not rows:
        return None
    if "example.gov.cn/demo" not in (rows[0].get("source_url") or ""):
        return None
    target = FUND_DIR / f"{variety_id}.csv.demo-{time.strftime('%Y%m%d%H%M%S')}.bak"
    try:
        path.rename(target)
        store.audit("demo_factors_quarantined", {"variety": variety_id, "to": str(target)})
        return target.name
    except OSError:
        return None


def bootstrap(varieties: Sequence[Dict[str, Any]], min_months: int = MIN_MONTHS_DEFAULT,
              force: bool = False, allow_demo: bool = False, horizon: int = 12,
              retries_sina: bool = True,
              progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """为缺失数据的品种拉取初始行情。已有且足量者默认跳过。

    allow_demo=False（默认）时**绝不用合成数据顶替真实行情**：
    拉取失败就如实报错，让用户决定装依赖、换网络，或显式选择离线演示。

    单品失败不阻断整体；返回逐品种报告，供 CLI 与界面展示。
    """
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    meta_all = _read_meta()
    items: List[Dict[str, Any]] = []
    ok = failed = skipped = 0

    for v in varieties:
        vid = v["id"]
        path = _market_path(vid)
        existing = store.read_csv(path)
        if existing and not force and len(existing) >= min_months:
            skipped += 1
            items.append({"variety": vid, "name": v["name_cn"], "status": "exists",
                          "rows": len(existing),
                          "source": meta_all.get(vid, {}).get("source", "unknown"),
                          "message": "本地已有足够样本，跳过"})
            continue
        if progress:
            progress(f"{v['name_cn']}({vid})")
        try:
            if allow_demo:
                rows, factors = generate_demo(v)
                store.write_csv(FUND_DIR / f"{vid}.csv", [r.to_dict() for r in factors], FUND_FIELDS)
                meta = {"source": "demo", "rows": len(rows),
                        "first": rows[0].month, "last": rows[-1].month,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "note": "合成数据，仅用于离线演示，禁止用于真实决策"}
            else:
                rows, meta = None, {}
                err_sina = None
                if retries_sina:
                    try:
                        rows, meta = fetch_market_sina(v, horizon=horizon)
                    except Exception as exc:  # noqa: BLE001
                        err_sina = str(exc)[:160]
                if rows is None:
                    try:
                        rows = fetch_market_akshare(vid)
                        meta = {"source": "akshare", "rows": len(rows),
                                "first": rows[0].month if rows else "",
                                "last": rows[-1].month if rows else "",
                                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "note": f"新浪通道失败({err_sina})，改用 AKShare" if err_sina else "AKShare"}
                    except Exception as exc:  # noqa: BLE001
                        raise FetchError(f"新浪: {err_sina or '未尝试'}; AKShare: {exc}") from exc
            if len(rows) < min_months:
                raise FetchError(f"仅 {len(rows)} 个月，低于门槛 {min_months}")
            store.write_csv(path, [r.to_dict() for r in rows], MARKET_FIELDS)
            quarantined = None
            if meta.get("source") != "demo":
                quarantined = quarantine_demo_factors(vid)
            meta_all[vid] = {**meta, "variety": vid, "rows": len(rows),
                             "first": rows[0].month, "last": rows[-1].month}
            _write_meta(meta_all)
            ok += 1
            items.append({"variety": vid, "name": v["name_cn"], "status": "fetched",
                          "rows": len(rows), "source": meta.get("source"),
                          "far_symbol": meta.get("far_symbol", ""),
                          "first": rows[0].month, "last": rows[-1].month,
                          "message": (meta.get("note", "") +
                                      ("；已隔离合成因子 " + quarantined if quarantined else ""))})
            store.audit("market_bootstrapped", {"variety": vid, "rows": len(rows),
                                                "source": meta.get("source")})
        except Exception as exc:  # noqa: BLE001
            failed += 1
            items.append({"variety": vid, "name": v["name_cn"], "status": "error",
                          "rows": len(existing), "message": str(exc)[:220]})
        time.sleep(0.2)

    return {"ok": ok, "failed": failed, "skipped": skipped, "total": len(varieties),
            "items": items,
            "hint": ("全部失败：检查网络后重试，或 pip install -r requirements-optional.txt；"
                     "完全离线可用 python -m src.cli init --allow-demo 生成合成演示数据")
            if ok == 0 and skipped == 0 else ""}


def save_demo(variety: Dict, years: int = 10, seed: int = 20261121) -> Dict[str, str]:
    """显式写入合成数据（离线演示/单测）。"""
    market, factors = generate_demo(variety, years, seed)
    mpath = _market_path(variety["id"])
    fpath = FUND_DIR / f"{variety['id']}.csv"
    store.write_csv(mpath, [r.to_dict() for r in market], MARKET_FIELDS)
    store.write_csv(fpath, [r.to_dict() for r in factors], FUND_FIELDS)
    meta = _read_meta()
    meta[variety["id"]] = {"source": "demo", "rows": len(market),
                           "first": market[0].month, "last": market[-1].month,
                           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "note": "合成数据，仅用于离线演示"}
    _write_meta(meta)
    store.audit("demo_generated", {"variety": variety["id"], "rows": len(market),
                                   "factors": len(factors)})
    return {"market": str(mpath), "fundamentals": str(fpath)}


# --------------------------------------------------------------------------
# 4. 本地读取
# --------------------------------------------------------------------------


def load_local(variety_id: str) -> Tuple[List[MarketRow], List[FactorRow]]:
    mrows = [MarketRow.from_dict(r) for r in store.read_csv(_market_path(variety_id))]
    frows: List[FactorRow] = []
    for r in store.read_csv(FUND_DIR / f"{variety_id}.csv"):
        try:
            frows.append(FactorRow(
                as_of=r["as_of"], variety=r["variety"], factor_id=r["factor_id"],
                value=float(r["value"]) if r.get("value") not in ("", None) else None,
                unit=r.get("unit", ""), period=r.get("period", ""),
                direction=r.get("direction", "neutral"),
                confidence=float(r.get("confidence") or 0),
                source_title=r.get("source_title", ""), source_url=r.get("source_url", ""),
                evidence=r.get("evidence", ""),
            ))
        except (KeyError, ValueError):
            continue
    return mrows, frows


# --------------------------------------------------------------------------
# 5. 合成数据生成器（离线演示 / 单测）
# --------------------------------------------------------------------------

DEMO_PROFILE = {
    "corn": (2380, 0.012, 0.085, 0.055), "cornstarch": (2700, 0.010, 0.075, 0.050),
    "soybean": (4900, 0.010, 0.075, 0.060), "soybean2": (3800, 0.008, 0.070, 0.062),
    "soymeal": (3100, 0.012, 0.090, 0.070), "soyoil": (8200, 0.008, 0.080, 0.065),
    "palm": (7200, 0.010, 0.100, 0.080), "rapemeal": (2600, 0.010, 0.085, 0.068),
    "rapeseedoil": (9200, 0.008, 0.080, 0.065), "rapeseed": (5600, 0.006, 0.070, 0.060),
    "peanut": (9400, 0.008, 0.090, 0.070), "sugar": (5900, 0.006, 0.110, 0.070),
    "cotton": (15800, 0.008, 0.095, 0.075), "rubber": (14000, 0.010, 0.120, 0.090),
    "apple": (8300, 0.004, 0.150, 0.090), "jujube": (11500, 0.004, 0.120, 0.085),
    "hog": (16500, 0.020, 0.180, 0.120), "egg": (3900, 0.010, 0.150, 0.090),
    "rice_j": (3500, 0.004, 0.045, 0.030), "rice_l": (2800, 0.004, 0.050, 0.035),
    "rice_e": (2600, 0.004, 0.050, 0.035), "wheat": (2900, 0.005, 0.060, 0.040),
}


def generate_demo(variety: Dict, years: int = 10, seed: int = 20261121,
                  end_ym: Optional[int] = None) -> Tuple[List[MarketRow], List[FactorRow]]:
    """生成带季节性、趋势、供需冲击的月度行情 + 配套基本面因子。"""
    rnd = random.Random(seed + sum(ord(c) for c in variety["id"]))
    base, drift, amp, vol = DEMO_PROFILE.get(variety["id"], (3000, 0.01, 0.08, 0.06))
    end = end_ym or timeutil.today_ym()
    n = years * 12
    start = end - (n - 1)
    harvest = set(variety["harvest_months"])

    market: List[MarketRow] = []
    factors: List[FactorRow] = []
    tight = 0.0
    for k in range(n):
        idx = start + k
        month = timeutil.month_of_year(idx)
        phase = (month - (min(harvest) if harvest else 10)) / 12.0 * 2 * math.pi
        seasonal = -amp * math.cos(phase)
        tight = 0.55 * tight + rnd.gauss(0, 0.045)
        if rnd.random() < 0.04:
            tight += rnd.gauss(0, 0.10)
        logp = (math.log(base) + drift * k / 12.0 + seasonal
                + 0.9 * tight + rnd.gauss(0, vol * 0.45))
        close = round(math.exp(logp), 1)
        carry = 0.012 * max(0, (min(harvest) - month) % 12) / 12.0
        far = round(close * math.exp(carry - 0.5 * amp * math.cos(phase) + rnd.gauss(0, 0.012)), 1)
        market.append(MarketRow(month=timeutil.ym_str(idx), close=close, far_close=far,
                                volume=round(80000 + rnd.gauss(0, 15000)),
                                open_interest=round(220000 + rnd.gauss(0, 30000)),
                                contract=f"{variety['ticker']}0"))

        prod = 1000 * (1 + 0.006 * k / 12) * math.exp(-0.75 * tight + rnd.gauss(0, 0.02))
        stock = 420 * math.exp(-1.0 * tight + rnd.gauss(0, 0.03))
        demand = 780 * math.exp(0.25 * (logp - math.log(base)) + rnd.gauss(0, 0.03))
        imp = 160 * math.exp(0.35 * tight + rnd.gauss(0, 0.05))
        weather = max(0.0, 0.85 * tight + rnd.gauss(0, 0.15))
        fert = 2600 * math.exp(0.01 * k / 12 + rnd.gauss(0, 0.03))

        asof = timeutil.ym_str(idx + 1) + "-10"
        period = timeutil.ym_str(idx)
        rows = [
            ("production_est", round(prod, 1), "万吨", -1),
            ("inventory_stock", round(stock, 1), "万吨", -1),
            ("feed_demand", round(demand, 1), "万吨", 1),
            ("import_volume", round(imp, 1), "万吨", -1),
            ("weather_anomaly", round(weather, 2), "σ", 1),
            ("fertilizer_cost", round(fert, 1), "元/吨", 1),
        ]
        for fid, val, unit, sign in rows:
            if fid not in variety.get("key_factors", []) and fid not in ("production_est", "inventory_stock"):
                continue
            conf = round(min(0.95, max(0.5, rnd.gauss(0.78, 0.08))), 2)
            factors.append(FactorRow(
                as_of=asof, variety=variety["id"], factor_id=fid, value=val, unit=unit,
                period=period, direction=("bearish" if sign < 0 else "bullish"),
                confidence=conf, source_title="示例数据 · 月度供需简报",
                source_url="https://example.gov.cn/demo",
                evidence=f"{period} {variety['name_cn']} {fid} 为 {val}{unit}（合成样例，非真实数据）",
            ))
    return market, factors
