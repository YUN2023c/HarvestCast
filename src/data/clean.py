"""清洗层：统一频率、去重、补缺、去噪、单位归一。

原则：
1. 一律先排序再计算，任何聚合都不依赖输入顺序；
2. 异常值只做「替换/标记」，绝不静默删除整行；
3. 每个处理动作都产出可读日志，供数据管理页展示与复盘。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..core import numeric, timeutil
from .schema import FactorRow, MarketRow

# 常用单位换算到「元/吨」「万吨」
UNIT_ALIASES = {
    "元/吨": ("price", 1.0),
    "元／吨": ("price", 1.0),
    "yuan/t": ("price", 1.0),
    "元/斤": ("price", 2000.0),
    "元/公斤": ("price", 1000.0),
    "元/千克": ("price", 1000.0),
    "吨": ("mass", 1e-4),      # -> 万吨
    "万吨": ("mass", 1.0),
    "千吨": ("mass", 0.1),
    "万吨/年": ("mass", 1.0),
    "亿斤": ("mass", 50.0),    # 1 亿斤 = 5 万吨
    "万公顷": ("area", 1.0),
    "千公顷": ("area", 0.1),
    "吨/公顷": ("yield", 1.0),
    "公斤/亩": ("yield", 15.0),  # 1 公斤/亩 = 15 公斤/公顷 ... 仅作近似，日志提示
    "指数": ("index", 1.0),
    "百分数": ("pct", 1.0),
    "%": ("pct", 1.0),
    "σ": ("sigma", 1.0),
}


def normalize_unit(value: Optional[float], unit: str) -> Tuple[Optional[float], str, List[str]]:
    """把数值换算到内部标准单位，返回 (值, 标准单位, 日志)。"""
    logs: List[str] = []
    if value is None:
        return None, unit, logs
    key = (unit or "").strip()
    if key not in UNIT_ALIASES:
        logs.append(f"未登记单位 {unit!r}，按原值使用，请人工确认量纲")
        return value, unit, logs
    kind, factor = UNIT_ALIASES[key]
    std = {"price": "元/吨", "mass": "万吨", "area": "万公顷", "yield": "吨/公顷",
           "index": "指数", "pct": "百分数", "sigma": "σ"}[kind]
    if factor != 1.0:
        logs.append(f"单位换算 {key} -> {std} (x{factor})")
    return value * factor, std, logs


def to_monthly(rows: Sequence[MarketRow], agg: str = "last",
               mad_k: float = 4.0, max_ffill: int = 2) -> Tuple[List[MarketRow], List[Dict]]:
    """日/周频行情聚合为月度，并做补缺与去噪。"""
    logs: List[Dict] = []
    if not rows:
        return [], [{"level": "error", "msg": "行情为空，无法清洗"}]

    buckets: Dict[int, List[MarketRow]] = {}
    bad = 0
    for r in rows:
        try:
            idx = timeutil.parse_ym(r.month)
        except Exception:
            bad += 1
            continue
        if not r.close or r.close <= 0:
            bad += 1
            continue
        buckets.setdefault(idx, []).append(r)
    if bad:
        logs.append({"level": "warn", "msg": f"丢弃 {bad} 条非法日期/价格记录"})

    idxs = sorted(buckets)
    out: List[MarketRow] = []
    for idx in idxs:
        group = sorted(buckets[idx], key=lambda r: r.month)
        pick = group[-1] if agg == "last" else group[0]
        closes = [g.close for g in group]
        out.append(MarketRow(
            month=timeutil.ym_str(idx),
            close=pick.close if agg == "last" else sum(closes) / len(closes),
            far_close=(pick.far_close if agg == "last" else sum(g.far_close for g in group) / len(group)),
            volume=sum(g.volume for g in group),
            open_interest=pick.open_interest,
            contract=pick.contract,
        ))
    logs.append({"level": "info", "msg": f"聚合为 {len(out)} 个月度观测"})

    # 补缺：仅在缺口 <= max_ffill 时前值填充，且标记来源
    full = timeutil.fill_month_gaps([timeutil.parse_ym(r.month) for r in out])
    by_idx = {timeutil.parse_ym(r.month): r for r in out}
    filled: List[MarketRow] = []
    gap_run = 0
    last: Optional[MarketRow] = None
    for idx in full:
        if idx in by_idx:
            filled.append(by_idx[idx])
            last = by_idx[idx]
            gap_run = 0
        else:
            gap_run += 1
            if last is not None and gap_run <= max_ffill:
                filled.append(MarketRow(month=timeutil.ym_str(idx), close=last.close,
                                        far_close=last.far_close, volume=0.0,
                                        open_interest=last.open_interest,
                                        contract=f"{last.contract}(fill)"))
            else:
                logs.append({"level": "warn", "msg": f"{timeutil.ym_str(idx)} 缺口超过 {max_ffill} 个月，未填充"})
    if len(filled) != len(out):
        logs.append({"level": "info", "msg": f"前值填充 {len(filled) - len(out)} 个月"})

    # 去噪：对数收益率的 MAD 稳健 z 分数
    logs_all = [timeutil.parse_ym(r.month) for r in filled]
    prices = [r.close for r in filled]
    if len(prices) > 8:
        rets = [0.0] + [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        med = numeric.median(rets)
        scale = numeric.mad(rets) * 1.4826
        if scale > 0:
            for i, r in enumerate(rets):
                if abs(r - med) / scale > mad_k:
                    lo = prices[i - 1] if i > 0 else prices[i]
                    hi = prices[i + 1] if i < len(prices) - 1 else prices[i]
                    replacement = math.sqrt(max(lo * hi, 1e-9))
                    logs.append({"level": "warn",
                                 "msg": f"{timeutil.ym_str(logs_all[i])} 价格 {prices[i]:.1f} 疑似异常，"
                                        f"收益 z={((r - med) / scale):.1f}，以相邻几何均值 {replacement:.1f} 替换"})
                    filled[i].close = replacement
                    filled[i].contract = (filled[i].contract or "") + "(outlier-fixed)"
    return filled, logs


def dedupe_factors(rows: Sequence[FactorRow], min_conf: float = 0.55,
                   require_source: bool = True) -> Tuple[List[FactorRow], List[Dict]]:
    """同一 (factor_id, period) 只留最新发布的一条；过滤低置信/无来源记录。"""
    logs: List[Dict] = []
    kept: Dict[Tuple[str, str], FactorRow] = {}
    dropped_conf = dropped_src = 0
    for r in rows:
        errs = r.validate()
        if errs:
            logs.append({"level": "warn", "msg": f"字段非法丢弃: {r.factor_id} {r.period} -> {'; '.join(errs)}"})
            continue
        if not r.usable(min_conf, require_source):
            if r.value is None:
                pass
            elif float(r.confidence) < min_conf:
                dropped_conf += 1
            else:
                dropped_src += 1
            continue
        key = (r.factor_id, r.period)
        prev = kept.get(key)
        if prev is None or r.as_of >= prev.as_of:
            kept[key] = r
    if dropped_conf:
        logs.append({"level": "warn", "msg": f"{dropped_conf} 条记录置信度低于 {min_conf}，未进入训练集"})
    if dropped_src:
        logs.append({"level": "warn", "msg": f"{dropped_src} 条记录缺少有效来源，未进入训练集"})
    logs.append({"level": "info", "msg": f"保留 {len(kept)} 条可用因子观测"})
    return sorted(kept.values(), key=lambda r: (r.factor_id, r.period)), logs
