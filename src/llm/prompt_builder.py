"""提示词渲染器（API 模式与手工模式共用同一份产物）。

同一份渲染结果 + 同一个 schema 版本 + 同一个校验器，才能保证「双模式 100% 等价」。
渲染结果的 SHA256 会写进产出元数据，导入时对比，模板被改过会直接报错。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..core.config import PROMPT_DIR

TEMPLATE_PATH = PROMPT_DIR / "extraction_prompt.md"
SCHEMA_PATH = PROMPT_DIR / "schema.json"


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def schema_version() -> str:
    return load_schema().get("properties", {}).get("schema_version", {}).get("enum", ["1.0.0"])[0]


def build_search_queries(variety: Dict[str, Any], as_of: str) -> List[str]:
    year = as_of[:4]
    name = variety["name_cn"]
    hm = "、".join(str(m) for m in variety.get("harvest_months", []))
    base = [
        f"{name} {year} 产量 预估 农业农村部",
        f"{name} 库存 港口 仓单 {year}",
        f"{name} 进口 出口 关税 配额 {year}",
        f"{name} 需求 开工率 利润 {year}",
        f"{name} 主产区 天气 墒情 干旱 {year}",
        f"{name} 期货 远月 {hm}月合约 基差",
    ]
    return base


def render(variety: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    """渲染提示词，返回 {system, user, full, hash, queries, schema}。"""
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    schema = load_schema()
    queries = build_search_queries(variety, as_of)
    hm = "、".join(str(m) for m in variety.get("harvest_months", []))
    mapping = {
        "{{VARIETY_NAME}}": variety["name_cn"],
        "{{VARIETY_ID}}": variety["id"],
        "{{EXCHANGE}}": variety.get("exchange", ""),
        "{{TICKER}}": variety.get("ticker", ""),
        "{{UNIT}}": variety.get("unit", ""),
        "{{YEAR}}": as_of[:4],
        "{{HARVEST_MONTHS}}": hm,
        "{{AS_OF_DATE}}": as_of,
        "{{KEY_FACTORS}}": "、".join(variety.get("key_factors", [])),
        "{{DEMAND_HINT}}": "饲料/深加工/纺织/食品等下游" if variety["id"] != "hog" else "养殖补栏与出栏节奏",
        "{{SEARCH_QUERIES}}": "\n".join(f"- {q}" for q in queries),
        "{{SCHEMA_JSON}}": json.dumps(schema, ensure_ascii=False, indent=2),
    }
    full = tpl
    for k, v in mapping.items():
        full = full.replace(k, str(v))
    # 拆分 SYSTEM / USER 两段，便于按 chat 结构调用
    system, user = full, ""
    marker = "## USER"
    if marker in full:
        head, tail = full.split(marker, 1)
        system = head.replace("## SYSTEM", "", 1).strip()
        user = tail.strip()
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()
    return {"system": system, "user": user, "full": full, "hash": digest,
            "queries": queries, "schema": schema, "schema_version": schema_version(),
            "rendered_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def save_rendered(variety_id: str, as_of: str, rendered: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"prompt_{variety_id}_{as_of}.md"
    header = (f"<!-- harvestcast prompt\n  variety: {variety_id}\n  as_of: {as_of}\n"
              f"  schema_version: {rendered['schema_version']}\n  sha256: {rendered['hash']}\n-->\n\n")
    path.write_text(header + rendered["full"], encoding="utf-8")
    return path


def check_hash(expected: str, actual_text: str) -> Tuple[bool, str]:
    """导入时用同一模板重渲染并比对 hash，防止「改了模板却复用旧结果」。"""
    got = hashlib.sha256(actual_text.encode("utf-8")).hexdigest()
    return got == expected, got
