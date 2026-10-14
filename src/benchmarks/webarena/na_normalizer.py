"""Adapter-layer N/A-answer dialect for WebArena.

WebArena's evaluator keys infeasible-task ("UA pool") answers on the literal
string "N/A" (StringEvaluator: exact match on the cleaned answer, with an
LLM reason-judge fallback). The stock baseline prompt TEACHES that dialect;
the GraphPC contract deliberately does not — any model-facing N/A channel
acts as a give-up cue, drawing N/A answers on feasible tasks. This
module is the adapter-side replacement: when the agent's terminal answer is a
PURE infeasibility declaration, the WebArena shim rewrites the submitted string
to "N/A" — the model never sees the mechanism, and every conversion is audited
on ``_na_normalized_events``.

The matcher itself (``match_infeasible_declaration`` and its rules) lives in
``src/agents/common/infeasible_declaration.py`` since OSWorld 2.0 shares it
(its dialect is the ``FAIL`` env token); the shim imports it from there. The
precision rationale and the rejected variants are documented there.
"""

from __future__ import annotations

__all__ = ["NA_ANSWER"]

# The evaluator-accepted literal (uppercase canonical form).
NA_ANSWER = "N/A"
