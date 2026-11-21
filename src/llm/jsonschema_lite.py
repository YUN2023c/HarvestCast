"""极简 JSON Schema 校验器（覆盖本项目 schema 用到的关键字）。

为什么不用 jsonschema 库：目标是零依赖可跑。本项目 schema 只用到
type / required / properties / additionalProperties / enum / pattern /
minimum / maximum / items / maxItems / $schema 这些关键字，够用且可审计。

校验分三层，与设计文档一致：
- 结构层：字段是否齐全、类型是否正确、有无多余字段；
- 约束层：枚举、范围、正则、数组长度；
- 业务层（见 llm/extract.py）：日期不得超过 as_of、禁止价格预测字段等。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_TYPE_MAP = {
    "object": dict, "array": list, "string": str,
    "number": (int, float), "integer": int, "boolean": bool, "null": type(None),
}


def _type_ok(value: Any, spec: Any) -> bool:
    if isinstance(spec, list):
        return any(_type_ok(value, s) for s in spec)
    py = _TYPE_MAP.get(spec)
    if py is None:
        return True
    if spec in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, py)


def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> Tuple[bool, List[str]]:
    errs: List[str] = []

    def walk(node: Any, sch: Dict[str, Any], where: str) -> None:
        t = sch.get("type")
        if t is not None and not _type_ok(node, t):
            errs.append(f"{where}: 类型应为 {t}，实际 {type(node).__name__}")
            return
        if "enum" in sch and node not in sch["enum"]:
            errs.append(f"{where}: 值 {node!r} 不在枚举 {sch['enum']} 中")
        if isinstance(node, str):
            if "pattern" in sch and not re.search(sch["pattern"], node):
                errs.append(f"{where}: 不匹配正则 {sch['pattern']}")
            if "minLength" in sch and len(node) < sch["minLength"]:
                errs.append(f"{where}: 长度小于 {sch['minLength']}")
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            if "minimum" in sch and node < sch["minimum"]:
                errs.append(f"{where}: {node} < 最小值 {sch['minimum']}")
            if "maximum" in sch and node > sch["maximum"]:
                errs.append(f"{where}: {node} > 最大值 {sch['maximum']}")
        if isinstance(node, list):
            if "maxItems" in sch and len(node) > sch["maxItems"]:
                errs.append(f"{where}: 数组长度 {len(node)} 超过 {sch['maxItems']}")
            item_sch = sch.get("items")
            if isinstance(item_sch, dict):
                for i, item in enumerate(node):
                    walk(item, item_sch, f"{where}[{i}]")
        if isinstance(node, dict):
            props = sch.get("properties", {})
            for key in sch.get("required", []):
                if key not in node:
                    errs.append(f"{where}: 缺少必填字段 {key}")
            if sch.get("additionalProperties") is False:
                for key in node:
                    if key not in props:
                        errs.append(f"{where}: 存在未定义字段 {key}")
            for key, sub in props.items():
                if key in node:
                    walk(node[key], sub, f"{where}.{key}")

    walk(instance, schema, path)
    return (not errs), errs


def summarize(schema: Dict[str, Any]) -> Dict[str, Any]:
    """给前端「Schema 预览」用：拍平成字段表。"""
    rows: List[Dict[str, Any]] = []

    def walk(sch: Dict[str, Any], prefix: str, depth: int) -> None:
        if depth > 3:
            return
        for key, sub in (sch.get("properties") or {}).items():
            t = sub.get("type")
            rows.append({
                "path": f"{prefix}{key}",
                "type": "/".join(t) if isinstance(t, list) else str(t),
                "required": key in (sch.get("required") or []),
                "enum": sub.get("enum"),
                "desc": (sub.get("description") or "")[:80],
            })
            if sub.get("type") == "array" and isinstance(sub.get("items"), dict):
                walk(sub["items"], f"{prefix}{key}[].", depth + 1)
            elif sub.get("type") == "object":
                walk(sub, f"{prefix}{key}.", depth + 1)

    walk(schema, "", 0)
    return {"title": schema.get("title"), "draft": schema.get("$schema"),
            "fields": rows, "field_count": len(rows)}
