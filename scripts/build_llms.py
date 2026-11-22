"""生成 llms-full.txt：把主文档按 LLM 友好格式拼接为单文件。

llms.txt 是精选索引（人手维护）；llms-full.txt 是全文合集（脚本生成），
供需要整段上下文的 AI 客户端一次抓取。文档更新后重跑本脚本即可。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/DATA_SOURCES.md",
    "docs/RESEARCH.md",
]
OUT = ROOT / "llms-full.txt"


def main() -> None:
    parts = [
        "# HarvestCast · 完整文档合集（llms-full）\n\n"
        "> 本文件由 scripts/build_llms.py 自动生成，内容 = 主文档全文拼接，"
        "供 LLM 一次性读取。更新文档后请重跑：python scripts/build_llms.py\n"
    ]
    for rel in FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        parts.append(f"\n\n---\n\n<!-- 来源文件：{rel} -->\n\n{text}")
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"llms-full.txt: {OUT.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
