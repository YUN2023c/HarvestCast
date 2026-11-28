# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""给全部源码文件加版权头（一次性脚本，可重跑去重）。

`//` 不是 Python 注释、也不是 cmd 注释，所以按语言适配注释符：
- Python → #
- cmd/bat → @REM（@ 抑制回显，避免 @echo off 之前的两行被打印）
- HTML → <!-- -->
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
COPYRIGHT = "Copyright (c) 2026 李硕"
SPDX = "SPDX-License-Identifier: MIT"

HEADERS = {
    ".py": (f"# {COPYRIGHT}\n# {SPDX}\n", ("# " + COPYRIGHT, "# " + SPDX)),
    ".bat": (f"@REM {COPYRIGHT}\n@REM {SPDX}\n", ("@REM " + COPYRIGHT, "@REM " + SPDX)),
    ".html": (f"<!-- {COPYRIGHT} -->\n<!-- {SPDX} -->\n", ("<!-- " + COPYRIGHT, "<!-- " + SPDX)),
}


def targets():
    for folder in ("src", "tests", "scripts"):
        for f in (ROOT / folder).rglob("*.py"):
            if "__pycache__" not in f.parts:
                yield f
    yield ROOT / "start.bat"
    yield ROOT / "web" / "index.html"


def main() -> None:
    changed, skipped = 0, 0
    for path in targets():
        conf = HEADERS.get(path.suffix.lower())
        if not conf:
            continue
        header, markers = conf
        text = path.read_text(encoding="utf-8")
        first = text.splitlines()[:2]
        if first == list(markers):
            skipped += 1
            continue
        path.write_text(header + text, encoding="utf-8")
        changed += 1
        print(f"  + {path.relative_to(ROOT)}")
    print(f"版权头：新增 {changed}，已存在 {skipped}")


if __name__ == "__main__":
    main()
