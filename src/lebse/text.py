"""Pure text utilities (no I/O, no heavy deps) — unit-tested."""

from __future__ import annotations

import re

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, lo: int = 40, hi: int = 350) -> list[str]:
    """Split text into sentences, keeping only those with ``lo <= len <= hi`` that contain a letter.

    Splits within each newline-delimited line so that opinions whose newlines were not collapsed
    still segment sensibly. Deterministic; used for SimCSE training sentence sampling.
    """
    out: list[str] = []
    for line in text.split("\n"):
        for s in _SENT_BOUNDARY.split(line):
            s = s.strip()
            if lo <= len(s) <= hi and any(c.isalpha() for c in s):
                out.append(s)
    return out
