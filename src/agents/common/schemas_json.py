"""Strict JSON schemas for OpenAI Responses API ``text.format=json_schema`` mode.

Strict JSON-schema output for the GPT backbone's ``complete_json()``
(``src/agents/backbone/gpt/main.py``). Each call site
passes the schema matching its expected output shape; the OpenAI Responses API
server-side enforces ``additionalProperties=False`` + required-key + enum
constraints, eliminating the "model returned markdown / extra prose / missing
key" class of failures.

Defense-in-depth: the multi-parser JSON fallback in
``src/agents/common/json_utils.py`` STAYS in place. It is not only a fallback:
on the Claude, DeepSeek, MiniMax and Qwen backbones, which do not enforce the
schema server-side, post-parsing the reply with it is the primary deciding route.

Backbone-portable: lives in ``common/`` so both GPT and Claude (and future
backbones) can import the same schema dicts. The GPT backbone wraps each
schema in the OpenAI-specific ``{"type": "json_schema", "name": ..., "schema":
..., "strict": True}`` envelope at request-build time.
"""

from __future__ import annotations

import copy

from .effect_class import EFFECT_CLASSES


# Nullable effect-class enum for the prior-step actual-effect label (OpenAI
# strict mode: every property must be in ``required``, so optional attrs are
# typed nullable). Null at step 0, where there is no prior action to close.
_EFFECT_CLASS_OR_NULL = {
    "type": ["string", "null"],
    "enum": [*EFFECT_CLASSES, None],
}


# ---------------------------------------------------------------------------
# Reasoning call site schemas (graph-reasoning, design.md Workflow/Effect-class vocabulary/Verification)
# ---------------------------------------------------------------------------
#
# Modes: "decide" (single-pass: one call closes the prior effect, rewrites the
# running graph_summary, and decides the action), "initial"/"refine" (two-pass:
# initial closes the prior effect + emits a retrieval request without deciding;
# refine decides the action). Every mode emits the fused summary update. The deciding modes
# (decide/refine) carry the predicted effect class inside each intended action —
# the Verification field-contract guard (the prediction is REQUIRED, never silently lost).

_EFFECT_CLASS = {"type": "string", "enum": list(EFFECT_CLASSES)}

_PLAN_OR_NULL = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "required": ["goal", "current_stage", "stages"],
    "properties": {
        "goal": {"type": "string"},
        "current_stage": {"type": "string"},
        "stages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
        },
    },
}

# The reasoning -> action handoff (Workflow): a structured intended action carrying the
# action_class + the SEMANTIC target (by label/role, not coordinates) + params +
# intent + the predicted effect class. The action stage grounds the target onto
# concrete coordinates/operations; reasoning never emits coordinates.
_INTENDED_ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action_class", "target", "text", "keys", "value", "intent", "predicted_effect_class"],
    "properties": {
        # Free-form by default, and deliberately so: this is WebArena's schema,
        # and on the GPT path the schema is enforced server-side, so
        # constraining it here would change what a WebArena run may emit as
        # well as what it is told. A benchmark that wants
        # a closed vocabulary (design.md §3: enumerated on OSWorld 2.0) passes its
        # own names through `reasoning_schema_for_mode(action_classes=...)` below;
        # OSWorld 2.0 is the only caller that does.
        "action_class": {"type": "string"},
        "target": {"type": "string"},
        "text": {"type": ["string", "null"]},
        "keys": {"type": ["string", "null"]},
        "value": {"type": ["string", "null"]},
        "intent": {"type": ["string", "null"]},
        # field-contract guard (Verification): required, non-null in every deciding mode.
        "predicted_effect_class": _EFFECT_CLASS,
    },
}

_RETRIEVAL_REQUEST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scope", "query", "k"],
    "properties": {
        "scope": {"type": "string", "enum": ["task", "stage", "recovery", "none"]},
        "query": {"type": ["string", "null"]},
        "k": {"type": "integer"},
    },
}


# Single-pass deciding call: closes the prior effect (actual_effect_class, null
# at step 0), (re)plans when needed (plan nullable), rewrites the running
# graph_summary, decides the action(s).
REASONING_DECIDE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "reasoning_text", "plan", "actual_effect_class", "next_intent",
        "intended_actions", "graph_summary",
    ],
    "properties": {
        "reasoning_text": {"type": "string"},
        "plan": _PLAN_OR_NULL,
        "actual_effect_class": _EFFECT_CLASS_OR_NULL,
        "next_intent": {"type": "string"},
        "intended_actions": {"type": "array", "items": _INTENDED_ACTION_SCHEMA},
        "graph_summary": {"type": "string"},
    },
}


# Two-pass initial call: closes the prior effect + rewrites the running
# graph_summary + emits a retrieval request; does NOT decide the current action.
REASONING_INITIAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "reasoning_text", "plan", "actual_effect_class", "retrieval_request",
        "graph_summary",
    ],
    "properties": {
        "reasoning_text": {"type": "string"},
        "plan": _PLAN_OR_NULL,
        "actual_effect_class": _EFFECT_CLASS_OR_NULL,
        "retrieval_request": _RETRIEVAL_REQUEST_SCHEMA,
        "graph_summary": {"type": "string"},
    },
}


# Two-pass refine call: consumes retrieval + the running summary, decides the
# action(s).
REASONING_REFINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "reasoning_text", "next_intent", "intended_actions",
        "graph_summary",
    ],
    "properties": {
        "reasoning_text": {"type": "string"},
        "next_intent": {"type": "string"},
        "intended_actions": {"type": "array", "items": _INTENDED_ACTION_SCHEMA},
        "graph_summary": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# Selector helper for reasoning-mode-based schema routing
# ---------------------------------------------------------------------------


def _with_action_class_enum(schema: dict, action_classes: tuple[str, ...]) -> dict:
    """``schema`` with ``intended_actions[].action_class`` closed to
    ``action_classes``; the schema unchanged when it declares no intended
    actions.

    The unchanged return is the two-pass "initial" mode, which decides nothing
    and therefore has no ``action_class`` to constrain — not a silent miss. A
    copy is built rather than the constant mutated, so the module constants stay
    the free-form ones every other call site reads.
    """
    items = schema.get("properties", {}).get("intended_actions", {}).get("items")
    if items is None or "action_class" not in items.get("properties", {}):
        return schema
    constrained = copy.deepcopy(schema)
    constrained["properties"]["intended_actions"]["items"]["properties"]["action_class"] = {
        "type": "string", "enum": list(action_classes),
    }
    return constrained


def reasoning_schema_for_mode(
    reasoning_mode: str, *, action_classes: tuple[str, ...] | None = None,
) -> tuple[dict, str]:
    """Return (schema, schema_name) for a graph-reasoning mode.

    "initial"/"refine" are the two-pass modes; anything else (incl. the
    single-pass "decide") routes to the decide schema, the deciding contract.

    ``action_classes`` closes the ``action_class`` field to a benchmark's own
    vocabulary. The default is None and returns the module constant ITSELF, so a
    caller that passes nothing sends the shared default schema bytes unchanged —
    WebArena's schema. Benchmark-scoped on the same
    discipline as the reasoning agent's other scoped settings; OSWorld 2.0's
    build_default is the only caller that supplies a vocabulary, and what it
    supplies is its grounding TOOLS block's own action set (design.md §3: the
    field is enumerated there; left free-form, the deciding call invents names
    such as `type_text`, `typewrite` and `keypress`, none of which any consumer
    keyed on `action_class` can match).
    """
    if reasoning_mode == "initial":
        schema, name = REASONING_INITIAL_SCHEMA, "reasoning_initial"
    elif reasoning_mode == "refine":
        schema, name = REASONING_REFINE_SCHEMA, "reasoning_refine"
    else:
        schema, name = REASONING_DECIDE_SCHEMA, "reasoning_decide"
    if action_classes is None:
        return schema, name
    return _with_action_class_enum(schema, action_classes), name
