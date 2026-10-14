"""Pure infeasibility-declaration matcher shared by the benchmark adapters.

The shared terminal-claim contract (WebArena's) has no give-up channel: a
model that finds the objective genuinely impossible decides ``done`` with a
brief statement to that effect (a model-facing infeasibility token acts as a
give-up cue: its mere availability draws give-ups on feasible tasks); on
OSWorld 2.0 the give-up is the ``fail`` action and this matcher is the fallback
for one that still arrives as a ``done``. Each benchmark scores infeasible
tasks on its own dialect (WebArena: the literal
"N/A" answer; OSWorld 2.0: the ``FAIL`` env token), so each adapter converts a
``done`` whose carried text is a PURE infeasibility declaration at the
boundary, and audits every conversion. This module is the matcher both
adapters share; the policy around it (identity skip, audit record, the
dialect emitted) stays in the adapter.

Precision-first by construction:

- Whole-answer matches only — never substring/contains matching.
- R2 requires the grammatical subject to be the TASK itself ("the task is
  impossible"), never content ("There is no airport within 5 km" is a
  legitimate ANSWER — a literal WebArena gold — and must never fire).
- The matchable language is finite, and none of its strings can score
  against any exact/must_include gold in WebArena's 812 config set, so a
  conversion can never destroy a would-be win there (re-check against the 812
  configs' exact/must_include golds when editing R1 or R2).
- Rejected variants (do NOT re-add without a validated-benefit case): bare
  "na" (token collision); "N/A - <reason>" prefix and declaration+reason
  forms (reason text can contain a must_include gold substring, and
  converting reasoned answers bypasses the benchmark's own reason-judge).

Pure module: no state, no benchmark imports.
"""

from __future__ import annotations

import re
from typing import Optional

# R1: the whole answer is an N/A spelling variant ("N/A", "n/a", "n / a",
# "N/A.", "N.A."). Bare "na" is deliberately excluded.
_LITERAL_NA = re.compile(r"^n\s*/\s*a[.!]?$|^n\.a\.?$", re.IGNORECASE)

# R2: the whole answer is a task-level infeasibility declaration. Task-subject
# only; no digits, no free text, no trailing clauses.
_TASK_DECLARATION = re.compile(
    r"^(?:the\s+|this\s+)?(?:task|objective|request|goal)\s+"
    r"(?:is\s+|was\s+)?"
    r"(?:impossible|unachievable|infeasible|"
    r"not\s+(?:possible|achievable|feasible)|"
    r"cannot\s+be\s+(?:completed|done|achieved|accomplished|performed))"
    r"[.!]?$",
    re.IGNORECASE,
)


# The OSWorld variant of R2, selected by ``allow_trailing_explanation``. TWO
# DIFFERENCES from the pattern above, both scoped to that benchmark:
#
#   * NO END ANCHOR. A give-up typically arrives as a declaration followed by an
#     explanation (the contracts ask for a brief statement of why), which the
#     anchored rule never matches: it matches only when the declaration is the
#     ENTIRE answer.
#   * A WIDER PREDICATE: ``not achieved`` / ``not completed`` / ``could not be
#     ...``, the other common phrasings of the same declaration.
#
# THE SUBJECT ANCHOR IS KEPT, so a declaration that puts the task in OBJECT
# position — "Unable to complete the task as specified", "I can't fully complete
# this task" — does not match. Admitting those means relaxing R2's core
# precision rule rather than widening a vocabulary. Not done here; a known
# undercount.
#
# WHY THIS CANNOT BE THE DEFAULT. The docstring above records declaration+reason
# forms as DELIBERATELY REJECTED for WebArena, because reason text can contain a
# must_include gold substring and converting reasoned answers bypasses that
# benchmark's own reason-judge. That the matchable strings cannot score
# against any gold in the 812 config set is a guarantee that holds only
# because the language is FINITE, which dropping the end anchor destroys. So the
# default stays anchored and OSWorld opts in.
_TASK_DECLARATION_TRAILING = re.compile(
    r"^(?:the\s+|this\s+)?(?:task|objective|request|goal)\s+"
    r"(?:is\s+|was\s+)?"
    r"(?:impossible|unachievable|infeasible|"
    r"not\s+(?:possible|achievable|feasible|achieved|completed)|"
    r"(?:cannot|could\s+not)\s+be\s+(?:completed|done|achieved|accomplished|performed))",
    re.IGNORECASE,
)


def _clean(answer: str) -> str:
    """Mirror of the WebArena evaluator's clean_answer minus lowercasing (the
    rules are IGNORECASE): strip, then strip ONE layer of matching quotes."""
    answer = answer.strip()
    for quote in ("'", '"'):
        if len(answer) >= 2 and answer.startswith(quote) and answer.endswith(quote):
            return answer[1:-1].strip()
    return answer


def match_infeasible_declaration(
    answer: str, *, allow_trailing_explanation: bool = False,
) -> Optional[str]:
    """Return the matched rule name (``"literal_na"`` / ``"task_declaration"``)
    when ``answer`` is a pure infeasibility declaration, else ``None``.

    Matching only — the calling adapter owns the policy around it (identity
    skip, the emitted dialect, and the audit record).

    ``allow_trailing_explanation`` selects the OSWorld variant of R2 (see the
    pattern above): a declaration may be followed by explanatory text, and the
    predicate vocabulary is wider. Benchmark-scoped, default False, which is
    WebArena's behaviour — the no-collision guarantee above rests on the
    anchored rule and its finite matchable language.
    """
    if not isinstance(answer, str):
        return None
    cleaned = _clean(answer)
    if _LITERAL_NA.match(cleaned):
        return "literal_na"
    declaration = _TASK_DECLARATION_TRAILING if allow_trailing_explanation else _TASK_DECLARATION
    if declaration.match(cleaned):
        return "task_declaration"
    return None
