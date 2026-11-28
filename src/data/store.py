# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""文件式存储层（无数据库）。

约定：
- data/market/{variety}.csv      行情（月度收盘 + 远月收盘）
- data/fundamentals/{variety}.csv 基本面因子（长表，逐条带来源）
- data/derived/*.json            派生特征/模型/回测结果
- data/output/*.json             最终预测产物
- data/output/audit.log          追加式审计日志（每次写入记一行）

所有写入均为「临时文件 + 原子替换」，覆盖前自动留 .bak，
符合「绝不删除用户数据」的硬约束。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..core.config import OUTPUT_DIR


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except OSError:
            pass


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent, default=str)
    os.replace(tmp, path)


def audit(event: str, detail: Dict[str, Any]) -> None:
    """追加式审计日志：谁、何时、对哪份文件做了什么。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **detail},
                      ensure_ascii=False)
    with (OUTPUT_DIR / "audit.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def list_derived(pattern: str = "*.json") -> List[Path]:
    from ..core.config import DERIVED_DIR
    return sorted(DERIVED_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
