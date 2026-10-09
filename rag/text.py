"""Description cleaning shared by every chunk builder."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\s*\(Citation:[^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_CODE_FENCE = re.compile(r"<code>|</code>")
_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(value: str | None) -> str:
    """Strip ATT&CK citation markers and inline Markdown, keeping the prose."""
    if not value:
        return ""
    out = _CITATION.sub("", value)
    out = _MD_LINK.sub(r"\1", out)
    out = _CODE_FENCE.sub("", out)
    out = _TRAILING_WS.sub("", out)
    out = _BLANK_RUN.sub("\n\n", out)
    return out.strip()
