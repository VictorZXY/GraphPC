from __future__ import annotations

import re
from typing import Any


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric+underscore tokenization (``instruction_tokens``)."""
    return set(re.findall(r"[a-z0-9_]+", str(text).lower()))


# Stopword filter for instruction tokens — used by WebArena's accessibility-tree
# relevance selection (benchmarks/webarena/a11y_parser.py), which scores node names
# against the instruction's tokens. Filters function words so generic overlap
# between a11y trees and any English instruction doesn't count as relevance.
_DEFAULT_STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "with",
    "and", "or", "but", "if", "is", "are", "be", "by", "as", "from",
    "this", "that", "it", "its", "i", "you", "we", "they", "your",
    "my", "our", "their", "do", "does", "did",
})


def instruction_tokens(text: str) -> set[str]:
    """Tokenize an instruction and remove stopwords for relevance scoring."""
    return _tokenize(text) - _DEFAULT_STOPWORDS


def compact_graph(payload: dict | None) -> dict:
    """Project the agent's own running record for the deciding prompt.

    The reasoning call emits no node/edge topology, so the projection is
    exactly the running natural-language ``graph_summary``; the current
    intent is NOT duplicated here — it already reaches the model through
    ``recent_history`` (intended_actions / next_intent). Read by the deciding
    prompt only — no action path imports this.
    """
    if not isinstance(payload, dict):
        return {"graph_summary": ""}

    graph_summary = payload.get("graph_summary")
    return {"graph_summary": graph_summary if isinstance(graph_summary, str) else ""}


_REASONING_TEXT_LIMIT = 240
_NEXT_INTENT_LIMIT = 240
# The same limit as the two above, applied to each retained intended action's
# string values. Uniform over every string rather than a field list: the closed
# vocabularies (``action_class``, ``predicted_effect_class``) are far under it and
# cannot be damaged, and the free-text ones (``intent``, ``target``, ``text``,
# ``value``) are the unbounded tail — a single typed paragraph would otherwise sit
# in the deciding prompt once per retained step.
_INTENDED_ACTION_TEXT_LIMIT = 240


def _bound_intended_actions(intended_actions: list, cap: int) -> list:
    """Keep the first ``cap`` intended actions, each with its string values bounded.

    WHY A COUNT IS THE RIGHT BOUND HERE. The model emits a whole forward plan in
    ``intended_actions`` — often many actions per reply — but the action stage
    grounds ``intended[:max_actions]`` and nothing beyond that is ever executed,
    and the next step re-emits a fresh plan regardless. So actions 2..N are a
    superseded speculation, and they are most of a history item: characters
    spent on what the agent decided NOT to do yet. The surviving carrier of
    "what remains" is the ``plan`` field and the running ``graph_summary``,
    which the deciding prompt already reads.

    WHAT IS PRESERVED, checked against the consumers rather than assumed:
    ``_prior_predicted_effect_class`` (orchestrator) is the only code that reads
    ``intended_actions`` back out of ``recent_history``, and it takes the FIRST
    non-empty ``predicted_effect_class`` of the LAST item — action 1's. At any
    ``cap >= 1`` the predict/verify contract is untouched. The WebArena grounding
    agents and ``MiniMaxActionAgent`` read the LIVE reasoning message, never history.
    """
    bounded = []
    for action in intended_actions[:cap]:
        if not isinstance(action, dict):
            bounded.append(action)
            continue
        bounded.append({
            key: (value[:_INTENDED_ACTION_TEXT_LIMIT] if isinstance(value, str) else value)
            for key, value in action.items()
        })
    return bounded


def compact_history_item(item: dict, *, intended_actions_cap: int | None = None) -> dict:
    """
    Compact one ``recent_history`` entry before it is appended in the
    orchestrator. Projects reasoning to the fields the next reasoning/action
    turn actually needs (the two short texts, the intended actions and the
    actual effect label); drops retrieval/monitor ``raw``
    payloads but keeps the selected ids and progress status that downstream
    stages reference.

    The full content is still on the live AgentMessages within the
    current step — only the history-projection is trimmed.

    ``intended_actions_cap`` is benchmark-scoped and defaults to ``None`` = the
    unbounded copy, WebArena's default. OSWorld 2.0 passes its own value from its
    ``build_default``: on a 500-step benchmark the un-shrunk forward plan makes a
    history item many times the size of the control's per-turn context, which
    would force the deciding call's history window down.
    See ``_bound_intended_actions``.
    """
    if not isinstance(item, dict):
        return {}

    compacted: dict[str, Any] = {}
    for key in ("step_idx", "stage_idx", "current_stage", "status"):
        if key in item:
            compacted[key] = item[key]

    actions = item.get("actions")
    if actions is not None:
        compacted["actions"] = actions

    reasoning = item.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning_text = reasoning.get("reasoning_text") or ""
        next_intent = reasoning.get("next_intent") or ""
        projected_reasoning: dict[str, Any] = {
            "reasoning_text": str(reasoning_text)[:_REASONING_TEXT_LIMIT],
            "next_intent": str(next_intent)[:_NEXT_INTENT_LIMIT],
        }
        # Verification signals (Verification) must survive compaction so the next step's
        # orchestrator can resolve predicted-vs-actual (``_prior_predicted_effect_class``).
        # The intended_actions array carries each action's predicted_effect_class.
        intended_actions = reasoning.get("intended_actions")
        if isinstance(intended_actions, list):
            projected_reasoning["intended_actions"] = (
                intended_actions if intended_actions_cap is None
                else _bound_intended_actions(intended_actions, intended_actions_cap)
            )
        actual_effect_class = reasoning.get("actual_effect_class")
        if actual_effect_class is not None:
            projected_reasoning["actual_effect_class"] = actual_effect_class
        compacted["reasoning"] = projected_reasoning
    elif reasoning is not None:
        compacted["reasoning"] = None

    retrieval = item.get("retrieval")
    if isinstance(retrieval, dict):
        compacted["retrieval"] = {
            "selected_entry_ids": list(retrieval.get("selected_entry_ids") or []),
        }
    elif retrieval is not None:
        compacted["retrieval"] = None

    monitor = item.get("monitor")
    if isinstance(monitor, dict):
        compacted["monitor"] = {
            "progress_status": monitor.get("progress_status"),
        }
    elif monitor is not None:
        compacted["monitor"] = None

    # No graph blob is carried per step: ``graph_summary`` is the only graph
    # artifact, and it lives on RuntimeState.content as a single current
    # snapshot.
    return compacted
