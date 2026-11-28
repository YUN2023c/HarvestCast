# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""LLM 产出落地：业务校验 → 归一化 → 合并入库。

LLM 只产出「事实」，这里把它转成模型可吃的长表，并把不可信的部分挡在门外：
- 结构已由 jsonschema_lite 校验；
- 本层再查：日期不得晚于 as_of、不得含价格预测表述、品种必须匹配、重复项去重。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from ..core import timeutil
from ..core.config import FUND_DIR
from ..data import store
from ..data.clean import normalize_unit
from ..data.schema import FUND_FIELDS, FactorRow
from . import prompt_builder as pb

PRICE_PREDICTION_PATTERNS = [
    r"目标价", r"预计(涨|跌)到", r"将(上涨|下跌)至", r"看(涨|跌)至", r"价格将达",
    r"target price", r"price target", r"forecast(ed)? price",
]


def business_checks(obj: Dict[str, Any], variety_id: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    as_of = obj.get("as_of_date", "")
    if obj.get("variety_id") != variety_id:
        errs.append(f"品种不匹配：产出为 {obj.get('variety_id')}，期望 {variety_id}")
    seen = set()
    for i, f in enumerate(obj.get("factors") or []):
        if f.get("as_of_date", "") > as_of:
            errs.append(f"factors[{i}] as_of_date={f.get('as_of_date')} 晚于信息截止日 {as_of}")
        key = (f.get("factor_id"), f.get("period"))
        if key in seen:
            errs.append(f"factors[{i}] 重复的 (factor_id, period)={key}")
        seen.add(key)
        quote = (f.get("evidence_quote") or "") + " " + (f.get("notes") or "")
        for pat in PRICE_PREDICTION_PATTERNS:
            if re.search(pat, quote, flags=re.I):
                errs.append(f"factors[{i}] 含价格预测性表述（命中 {pat}），应剔除")
                break
    if obj.get("refusal") and not obj.get("refusal_reason"):
        errs.append("refusal=true 但未给 refusal_reason")
    return (not errs), errs


def to_factor_rows(obj: Dict[str, Any], min_conf: float, require_source: bool
                   ) -> Tuple[List[FactorRow], List[Dict[str, Any]]]:
    rows: List[FactorRow] = []
    dropped: List[Dict[str, Any]] = []
    for f in obj.get("factors") or []:
        raw_val = f.get("value")
        val, std_unit, logs = normalize_unit(raw_val, f.get("unit", ""))
        row = FactorRow(
            as_of=f.get("as_of_date", ""), variety=obj.get("variety_id", ""),
            factor_id=f.get("factor_id", ""), value=val, unit=std_unit,
            period=f.get("period", ""), direction=f.get("direction", "neutral"),
            confidence=float(f.get("confidence") or 0.0),
            source_title=(f.get("source_title") or "")[:120],
            source_url=(f.get("source_url") or "").strip(),
            evidence=(f.get("evidence_quote") or "")[:400],
        )
        errs = row.validate()
        if errs:
            dropped.append({"factor": row.factor_id, "period": row.period, "reason": "; ".join(errs)})
            continue
        if not row.usable(min_conf, require_source):
            reason = ("置信度不足" if float(row.confidence) < min_conf
                      else ("无有效来源" if (require_source and (not row.source_url or row.source_url.upper() == "N/A"))
                            else "无数值"))
            dropped.append({"factor": row.factor_id, "period": row.period,
                            "reason": f"{reason}（confidence={row.confidence}, value={raw_val}）"})
            continue
        rows.append(row)
    return rows, dropped


def quality(obj: Dict[str, Any], rows: Sequence[FactorRow]) -> Dict[str, Any]:
    factors = obj.get("factors") or []
    confs = [float(f.get("confidence") or 0) for f in factors] or [0.0]
    with_src = sum(1 for f in factors if (f.get("source_url") or "").strip() not in ("", "N/A"))
    by_factor: Dict[str, int] = {}
    for r in rows:
        by_factor[r.factor_id] = by_factor.get(r.factor_id, 0) + 1
    return {
        "records_in": len(factors),
        "records_kept": len(rows),
        "records_dropped": len(factors) - len(rows),
        "mean_confidence": round(sum(confs) / len(confs), 3),
        "source_ratio": round(with_src / len(factors), 3) if factors else 0.0,
        "factors": by_factor,
        "gaps": obj.get("data_gaps") or [],
        "events": len(obj.get("events") or []),
        "as_of_date": obj.get("as_of_date"),
        "mode": (obj.get("meta") or {}).get("mode"),
        "model": (obj.get("meta") or {}).get("model"),
    }


def merge_into_store(variety_id: str, new_rows: Sequence[FactorRow]) -> Dict[str, Any]:
    """与本地长表合并：同一 (factor_id, period) 保留 as_of 最新的一条。"""
    path = FUND_DIR / f"{variety_id}.csv"
    existing: List[FactorRow] = []
    for r in store.read_csv(path):
        try:
            existing.append(FactorRow(
                as_of=r["as_of"], variety=r["variety"], factor_id=r["factor_id"],
                value=float(r["value"]) if r.get("value") not in ("", None) else None,
                unit=r.get("unit", ""), period=r.get("period", ""),
                direction=r.get("direction", "neutral"),
                confidence=float(r.get("confidence") or 0),
                source_title=r.get("source_title", ""), source_url=r.get("source_url", ""),
                evidence=r.get("evidence", "")))
        except (KeyError, ValueError):
            continue
    merged: Dict[Tuple[str, str], FactorRow] = {(r.factor_id, r.period): r for r in existing}
    for r in new_rows:
        key = (r.factor_id, r.period)
        prev = merged.get(key)
        if prev is None or r.as_of >= prev.as_of:
            merged[key] = r
    out_rows = sorted(merged.values(), key=lambda r: (r.factor_id, r.period))
    store.write_csv(path, [r.to_dict() for r in out_rows], FUND_FIELDS)
    store.audit("factors_merged", {"variety": variety_id, "added": len(new_rows),
                                   "total": len(out_rows)})
    return {"path": str(path), "total": len(out_rows), "added": len(new_rows)}


def ingest(obj: Dict[str, Any], variety_id: str, settings: Dict[str, Any],
           save_raw: bool = True) -> Dict[str, Any]:
    ok, errs = business_checks(obj, variety_id)
    if not ok:
        return {"status": "rejected", "errors": errs}
    llm = settings.get("llm", {})
    rows, dropped = to_factor_rows(obj, float(llm.get("min_confidence", 0.55)),
                                  bool(llm.get("require_source", True)))
    q = quality(obj, rows)
    q["dropped_detail"] = dropped
    if not rows:
        return {"status": "empty", "quality": q,
                "message": "没有通过质量门槛的记录，未写入训练集"}
    merged = merge_into_store(variety_id, rows)
    if save_raw:
        raw_path = FUND_DIR / f"raw_{variety_id}_{obj.get('as_of_date')}.json"
        store.write_json(raw_path, obj)
        merged["raw"] = str(raw_path)
    store.audit("llm_ingested", {"variety": variety_id, "kept": len(rows),
                                 "mode": q["mode"], "model": q["model"]})
    return {"status": "ok", "quality": q, "store": merged}


def export_prompt(variety: Dict[str, Any], as_of: str, out_dir) -> Dict[str, Any]:
    """手工模式：把渲染好的提示词落盘并返回全文，供复制粘贴。"""
    rendered = pb.render(variety, as_of)
    path = pb.save_rendered(variety["id"], as_of, rendered, out_dir)
    store.audit("prompt_exported", {"variety": variety["id"], "as_of": as_of, "path": str(path)})
    return {"path": str(path), "prompt": rendered["full"], "hash": rendered["hash"],
            "schema_version": rendered["schema_version"], "queries": rendered["queries"]}


def import_payload(text: str, variety_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """手工模式：接收粘贴/上传的 JSON，走与 API 模式完全相同的校验链路。"""
    from .jsonschema_lite import validate
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"status": "rejected", "errors": [f"JSON 解析失败: {exc}"]}
    schema = pb.load_schema()
    ok, errs = validate(obj, schema)
    if not ok:
        return {"status": "rejected", "errors": errs[:12]}
    obj.setdefault("meta", {})
    obj["meta"]["mode"] = "manual"
    return ingest(obj, variety_id, settings)
