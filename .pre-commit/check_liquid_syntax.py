#!/usr/bin/env python3
"""Reject Liquid delimiters in pages that the GitHub Pages build renders.

GitHub Pages builds docs/ with Jekyll, which runs every Markdown page (and any
file with front matter) through Liquid, including fenced code. Jinja quoted
from a chat template looks like Liquid to it:
  - an unknown tag such as `{% set ... %}` fails the whole Pages build;
  - an expression such as `{{ message.content }}` renders silently as empty
    text, so the page loses content and the build still passes.
The pages-build CI job catches the first; only this check catches the second.

Text between `{% raw %}` and `{% endraw %}` is allowed, but those tags show up
literally on github.com, so rewording is usually better.

Usage: check_liquid_syntax.py FILE... (the files pre-commit passes).
Read-only. Exit 0 on clean, exit 1 on any finding.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DELIMITER_RE = re.compile(r"\{[{%]")
RAW_BLOCK_RE = re.compile(r"\{%-?\s*raw\s*-?%\}.*?\{%-?\s*endraw\s*-?%\}", re.DOTALL)
LIQUID_SUFFIXES = {".md", ".markdown"}


def is_rendered(path: Path, text: str) -> bool:
    """Jekyll renders Markdown always, and any other file only with front matter."""
    return path.suffix.lower() in LIQUID_SUFFIXES or text.startswith("---\n")


def findings(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not is_rendered(path, text):
        return []
    # Blank out raw blocks but keep their newlines, so line numbers stay right.
    text = RAW_BLOCK_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if DELIMITER_RE.search(line):
            out.append(f"{path}:{lineno}: Liquid delimiter: {line.strip()}")
    return out


def main(argv: list[str]) -> int:
    problems = [f for arg in argv for f in findings(Path(arg))]
    for problem in problems:
        print(problem)
    if problems:
        print(
            "\nGitHub Pages renders these files with Liquid: '{%' fails the "
            "build and '{{' renders as empty text. Reword the passage (e.g. "
            "quote Jinja without its braces), or wrap it in "
            "{% raw %}...{% endraw %}."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
