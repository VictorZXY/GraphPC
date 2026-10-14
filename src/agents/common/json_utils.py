"""Backbone-portable JSON extraction for model responses.

The canonical multi-parser JSON fallback: it handles fenced code
blocks, prose-wrapped JSON, trailing commas, and Python-dict-style literals
(single quotes, True/False/None).
Every backbone shares this single implementation instead of drifting separate
copies.

The multi-parser fallback chain is intentionally progressive — each parser is
strictly more permissive than the last, so the strict cases keep going through
`json.loads` while the messy cases recover instead of crashing the task on its
first reasoning call. Failure logs the raw payload (bounded) so future drift is
diagnosable from runner.log without re-running the task.
"""

from __future__ import annotations

import ast
import json
import logging
import re

from typing import Any, Optional

logger = logging.getLogger("desktopenv.agent")


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


class JsonParseExhausted(ValueError):
    """Raised when ``extract_json_object`` exhausts every parser without
    yielding a dict — the marker of a MODEL-OUTPUT parse failure. Call sites
    soft-degrade on exactly this class (the reasoning and grounding degrades)
    while any other ``ValueError`` escaping a backbone (e.g. the deliberate
    pattern-bearing-schema guard raise) stays loud. Subclasses ``ValueError``
    so broad ``ValueError`` catches still see it."""



def strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before a closing brace/bracket."""
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def extract_json_object(text: str, *, log_label: str = "Model") -> dict[str, Any]:
    """Parse a model response into a dict using a progressive parser chain.

    Candidates: the stripped body, and the substring between the first `{` and
    last `}` (in case the model wrapped JSON in prose).

    Parsers (each tried against every candidate, in order):
      1. strict json.loads
      2. json.loads after trailing-comma cleanup
      3. ast.literal_eval (Python-dict style: single quotes, True/False/None)

    The first parser that yields a dict wins. If all combinations fail, raises
    ValueError. The `log_label` is interpolated into the warning + error so the
    failure is traceable to the calling backbone.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1]
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
        stripped = stripped.strip()

    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        brace_slice = stripped[start : end + 1]
        if brace_slice != stripped:
            candidates.append(brace_slice)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        for parse in (
            lambda c: json.loads(c),
            lambda c: json.loads(strip_trailing_commas(c)),
            lambda c: ast.literal_eval(c),
        ):
            try:
                payload = parse(candidate)
            except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                last_error = exc
                continue
            if isinstance(payload, dict):
                return payload
            last_error = ValueError(
                f"parser produced non-dict {type(payload).__name__}"
            )

    snippet = (
        text
        if len(text) <= 8192
        else text[:4096] + "\n...[truncated]...\n" + text[-4096:]
    )
    logger.warning(
        "%s JSON parse failed; all fallbacks exhausted. last_error=%s. raw response:\n%s",
        log_label,
        last_error,
        snippet,
    )
    raise JsonParseExhausted(
        f"{log_label} response did not contain a parseable JSON object; last error: {last_error}"
    )
