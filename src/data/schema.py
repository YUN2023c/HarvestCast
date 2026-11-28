# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""数据契约（dataclass + 校验）。

只做结构校验与单位归一，不做业务判断；业务判断放在 econ 层，
这样数据层可以独立替换为别的数据源而不影响模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..core import timeutil

MARKET_FIELDS = ["date", "close", "far_close", "volume", "open_interest", "contract"]
FUND_FIELDS = ["as_of", "variety", "factor_id", "value", "unit", "period",
               "direction", "confidence", "source_title", "source_url", "evidence"]

DIRECTIONS = {"bullish", "bearish", "neutral"}


@dataclass
class MarketRow:
    month: str          # YYYY-MM
    close: float        # 主力连续收盘（现货代理）
    far_close: float    # 收获季对应远月合约收盘
    volume: float = 0.0
    open_interest: float = 0.0
    contract: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketRow":
        return cls(
            month=d.get("date") or d.get("month"),
            close=float(d["close"]),
            far_close=float(d.get("far_close") or d["close"]),
            volume=float(d.get("volume") or 0),
            open_interest=float(d.get("open_interest") or 0),
            contract=str(d.get("contract") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"date": self.month, **{k: v for k, v in asdict(self).items() if k != "month"}}


@dataclass
class FactorRow:
    as_of: str
    variety: str
    factor_id: str
    value: Optional[float]
    unit: str
    period: str
    direction: str
    confidence: float
    source_title: str = ""
    source_url: str = ""
    evidence: str = ""

    def validate(self) -> List[str]:
        errs: List[str] = []
        try:
            timeutil.parse_date(self.as_of)
        except Exception:
            errs.append(f"as_of 非法: {self.as_of!r}")
        if self.direction not in DIRECTIONS:
            errs.append(f"direction 非法: {self.direction!r}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            errs.append(f"confidence 越界: {self.confidence}")
        if self.value is not None and not isinstance(self.value, (int, float)):
            errs.append("value 必须是数字或 null")
        if not self.unit:
            errs.append("unit 不得为空")
        return errs

    def usable(self, min_conf: float, require_source: bool) -> bool:
        """是否允许进入训练集：置信度足够 + 有来源 + 有数值。"""
        if self.value is None:
            return False
        if float(self.confidence) < min_conf:
            return False
        if require_source and (not self.source_url or self.source_url.strip().upper() == "N/A"):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetBundle:
    """一次建模所需的完整输入。"""
    variety: str
    months: List[str]
    close: List[float]
    far_close: List[float]
    features: Dict[str, List[Optional[float]]] = field(default_factory=dict)
    feature_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def n(self) -> int:
        return len(self.months)

    def last_month(self) -> str:
        return self.months[-1] if self.months else ""

    def summary(self) -> Dict[str, Any]:
        return {
            "variety": self.variety,
            "n_months": self.n(),
            "start": self.months[0] if self.months else "",
            "end": self.last_month(),
            "features": list(self.features.keys()),
            "coverage": self.coverage,
            "warnings": self.warnings,
        }
