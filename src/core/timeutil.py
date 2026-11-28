# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""月份口径工具。

全部内部计算统一使用「月序号」整数（year*12 + month-1），
避免 datetime 时区/月末天数差异带来的对齐错误。
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable, List, Sequence, Tuple


def ym_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def parse_ym(text: str) -> int:
    """'2026-09' / '2026/9' / '202609' -> 月序号。"""
    t = str(text).strip().replace("/", "-")
    if len(t) == 6 and t.isdigit():
        return ym_index(int(t[:4]), int(t[4:]))
    parts = t.split("-")
    if len(parts) < 2:
        raise ValueError(f"无法解析月份: {text!r}")
    return ym_index(int(parts[0]), int(parts[1]))


def from_index(idx: int) -> Tuple[int, int]:
    return idx // 12, idx % 12 + 1


def ym_str(idx: int) -> str:
    y, m = from_index(idx)
    return f"{y:04d}-{m:02d}"


def month_of_year(idx: int) -> int:
    return idx % 12 + 1


def add_months(idx: int, n: int) -> int:
    return idx + n


def months_between(a: int, b: int) -> int:
    return b - a


def parse_date(text: str) -> _dt.date:
    return _dt.date.fromisoformat(str(text)[:10])


def today_ym() -> int:
    d = _dt.date.today()
    return ym_index(d.year, d.month)


def month_range(start: int, end: int) -> List[int]:
    return list(range(start, end + 1))


def fill_month_gaps(indices: Iterable[int]) -> List[int]:
    s = sorted(set(indices))
    if not s:
        return []
    return list(range(s[0], s[-1] + 1))


def harvest_window(harvest_months: Sequence[int], start_idx: int, horizon_months: int) -> List[int]:
    """返回「不少于 horizon 个月的最近一个收获窗口」。

    规则（按优先级）：
    1. 全年收获品种（生猪等）无季节窗口概念，直接取目标月；
    2. 目标月 ±3 个月内存在收获月 → 取该窗口内的全部收获月；
    3. 否则从目标月向未来找最近的收获月（最多 12 个月），并展开为连续收获月簇。

    第 3 条的关键是「只向未来找」：若向过去找，会把跨度压到 horizon 以内
    （例如 11 月预测白糖 6 个月，回头选到 1 月合约，实际只剩 2 个月），
    与「6–12 个月后收获季」的业务定义冲突。
    """
    target = start_idx + horizon_months
    months = sorted({int(m) for m in harvest_months if 1 <= int(m) <= 12})
    if len(months) == 12:
        return [target]

    near = [target + d for d in range(-3, 4) if month_of_year(target + d) in months]
    if near:
        year = from_index(near[0])[0]
        same_year = [i for i in near if from_index(i)[0] == year]
        return same_year or near

    for d in range(4, 16):
        cand = target + d
        if month_of_year(cand) in months:
            lo = cand
            while month_of_year(lo - 1) in months and cand - lo < 3:
                lo -= 1
            hi = cand
            while month_of_year(hi + 1) in months and hi - cand < 3:
                hi += 1
            return list(range(lo, hi + 1))
    return [target]


def seasonal_position(idx: int, harvest_months: Sequence[int]) -> str:
    """相对收获季的位置：pre_harvest / harvest / post_harvest。"""
    m = month_of_year(idx)
    if m in harvest_months:
        return "harvest"
    hm = sorted(harvest_months)
    if not hm:
        return "harvest"
    # 找到下一个收获月
    for k in range(0, 12):
        if month_of_year(idx + k) in hm:
            ahead = k
            break
    else:
        ahead = 0
    return "pre_harvest" if 0 < ahead <= 4 else "post_harvest"
