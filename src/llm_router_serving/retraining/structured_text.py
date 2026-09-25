"""Versioned structure-preserving question normalization."""

from __future__ import annotations

import re
import unicodedata


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    lines = [line.rstrip() for line in text.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
    lines = text.split("\n")
    if (
        lines
        and lines[0].casefold().strip().rstrip(".")
        == "please answer with the letter of the correct answer"
    ):
        lines.pop(0)
    if lines and re.fullmatch(
        r"print\s+only\s+a\s+single\s+choice\s+from\s+.+without\s+explanation\.?",
        lines[-1].strip(),
        re.I,
    ):
        lines.pop()
    return "\n".join(lines).strip("\n")


def structured_duplicate_key(text: str) -> str:
    # Preserve punctuation, case, line boundaries and leading whitespace.
    # Word-only matching conflates x+y/x-y and differently scoped Python code.
    return clean_text(text)
