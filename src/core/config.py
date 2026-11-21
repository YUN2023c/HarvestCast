"""配置加载：路径、settings、品种主数据。

设计取舍：只用 JSON，不引入 YAML/pydantic，保证零依赖启动。
settings.json 缺失时自动回退 settings.example.json，避免首次运行即报错。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
FUND_DIR = DATA_DIR / "fundamentals"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUT_DIR = DATA_DIR / "output"
WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_settings() -> Dict[str, Any]:
    path = CONFIG_DIR / "settings.json"
    if not path.exists():
        path = CONFIG_DIR / "settings.example.json"
    settings = _read_json(path)
    # 允许用环境变量覆盖敏感项与常用项
    if os.environ.get("HARVESTCAST_BASE_URL"):
        settings["llm"]["base_url"] = os.environ["HARVESTCAST_BASE_URL"]
    if os.environ.get("HARVESTCAST_MODEL"):
        settings["llm"]["model"] = os.environ["HARVESTCAST_MODEL"]
    if os.environ.get("HARVESTCAST_PORT"):
        settings["server"]["port"] = int(os.environ["HARVESTCAST_PORT"])
    settings["_settings_path"] = str(path)
    return settings


def api_key(settings: Dict[str, Any]) -> str:
    llm = settings.get("llm", {})
    env_name = llm.get("api_key_env", "HARVESTCAST_API_KEY")
    return os.environ.get(env_name, "") or llm.get("api_key", "") or ""


def load_varieties() -> Dict[str, Any]:
    return _read_json(CONFIG_DIR / "varieties.json")


def variety_list() -> List[Dict[str, Any]]:
    return load_varieties()["varieties"]


def get_variety(variety_id: str) -> Dict[str, Any]:
    for v in variety_list():
        if v["id"] == variety_id:
            return v
    raise KeyError(f"未知品种: {variety_id}")


def factor_meta() -> Dict[str, Any]:
    return load_varieties()["factors"]


def ensure_dirs() -> None:
    for d in (MARKET_DIR, FUND_DIR, DERIVED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
