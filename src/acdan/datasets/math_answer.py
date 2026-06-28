"""Math answer extraction and equivalence helpers.

The real math benchmarks contain numeric, symbolic, and LaTeX answers.  The old
fallback of comparing the last number in a string can silently mark malformed
solutions correct, so these helpers only use numeric comparison when the
extracted answer itself is numeric.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Optional


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+(?:\.\d*)?)?$")


def _balanced_brace_content(text: str, start: int) -> Optional[str]:
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                begin = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i]
    return None


def _last_boxed(text: str) -> Optional[str]:
    matches = list(_BOXED_RE.finditer(text))
    for match in reversed(matches):
        content = _balanced_brace_content(text, match.end() - 1)
        if content:
            return content
    return None


def _after_answer_marker(text: str) -> Optional[str]:
    patterns = [
        r"(?:the\s+)?final\s+answer\s+is\s*[:\-]?\s*(.+)$",
        r"(?:the\s+)?answer\s+is\s*[:\-]?\s*(.+)$",
        r"answer\s*[:\-]\s*(.+)$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[-1]
    return None


def _clean_marked_answer(text: str) -> str:
    first = str(text).splitlines()[0]
    boxed = _last_boxed(first)
    if boxed:
        return _strip_outer_noise(boxed)
    numeric = re.match(
        r"\s*([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d[\d,]*(?:\.\d*)?)?)",
        first,
    )
    if numeric:
        return _strip_outer_noise(numeric.group(1))
    return _strip_outer_noise(first)


def _strip_outer_noise(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"^[\s$]+|[\s$]+$", "", s)
    s = s.strip(" .,:;")
    while s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    while s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return s.strip(" .,:;")


def _latex_normalize(text: str) -> str:
    s = _strip_outer_noise(text)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\infty", "infty")
    s = re.sub(r"\\mathrm\s*\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _numeric_value(text: str) -> Optional[Decimal]:
    s = _latex_normalize(text).replace(",", "")
    if not _NUMBER_RE.match(s):
        return None
    try:
        if "/" in s:
            frac = Fraction(s)
            return Decimal(frac.numerator) / Decimal(frac.denominator)
        return Decimal(s)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def extract_final_answer(text: str) -> str:
    """Extract the most likely final answer without last-number guessing."""

    s = str(text).strip()
    if not s:
        return ""

    boxed = _last_boxed(s)
    if boxed:
        return _strip_outer_noise(boxed)

    marked = _after_answer_marker(s)
    if marked:
        return _clean_marked_answer(marked)

    lines = [line.strip() for line in s.splitlines() if line.strip()]
    tail = lines[-1] if lines else s
    return _strip_outer_noise(tail[-160:])


def answers_equivalent(pred: str, gold: str) -> bool:
    """Return whether two extracted math answers should count as equivalent."""

    p = extract_final_answer(pred)
    g = extract_final_answer(gold)
    if not p or not g:
        return False

    if _latex_normalize(p) == _latex_normalize(g):
        return True

    pn = _numeric_value(p)
    gn = _numeric_value(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= Decimal("1e-9")

    return False
