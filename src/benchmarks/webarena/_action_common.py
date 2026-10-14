"""Shared building blocks for the WebArena action agent (``action_agent.py``,
one class shared by every backbone).

Every backbone emits the same payload shape (WebArena id-based action
strings + a reasoning_text); the only thing that differs is which backbone's
``complete_json`` builds the request. Kept here so any contract change
happens once.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from src.agents.common.schema import Action, ActionBatch

from .url_mappings import map_url_to_local


# Strict output schema for every backbone — the model emits a short reasoning
# trace + one or more id-based action strings (e.g. "click [5]",
# "type [3] [hello] [1]", "stop [answer]"). The translator parses them
# server-side through ``create_id_based_action`` later (``action_translator.py``).
#
# The action_strings `pattern` enforces the WebArena id-based grammar at
# emission (design.md: closed vocabulary enforced at emission — natively where
# the backbone supports schema-constrained output). GPT enforces it
# server-side (OpenAI strict json_schema accepts `pattern` on array items —
# verified live); the MiniMax, Qwen and Claude backbones send the schema as a
# forced tool's parameters, where the pattern is advisory, with WebArena's own
# parser and the adapter's degrade path as the backstop.
#
# The pattern is a strict SUBSET of what WebArena's create_id_based_action
# accepts (browser_env/actions.py id-based regexes), so anything that matches
# here parses there:
#   * `type` requires the explicit [0|1] enter-flag — WebArena auto-appends [1]
#     when omitted, but requiring it is a safe subset.
#   * `stop \[.*\]` is safe: WebArena's stop regex failing to match just yields
#     answer="" (never raises).
#   * `noop` never reaches WebArena's parser (our translator maps wait ->
#     create_none_action).
#   * `type` text is PRINTABLE ASCII `[ -~]+` (space 0x20 .. tilde 0x7E): WebArena
#     tokenizes typed text through an ASCII-only keyboard map (_keys2ids,
#     browser_env/actions.py) that raises KeyError on any non-ASCII char (e.g. a
#     curly apostrophe), so printable-ASCII is the env's true input alphabet —
#     constraining it forces the model to pick ASCII spellings at decode time.
#     (Newlines are excluded too, consistent with WebArena's own re.search.) The
#     range covers `[`/`]`/space, so bracketed/spaced text still matches; the
#     `]` is inside the range (0x5D) so no separate escaping is needed.
#     stop `[.*]` is intentionally NOT restricted — its answer goes to the
#     evaluator, never the keyboard.
# press/goto middles exclude newline as well as `]` ([^\]\r\n]) — WebArena's
# `press ?\[(.+)\]` / `goto ?\[(.+)\]` use Python `re` where `.` excludes newline
# (and `$` matches before a trailing newline under ECMA262), so allowing a newline
# here would let an action string match ours but be REJECTED by WebArena, breaking
# the strict-subset property.
_WEBARENA_ACTION_STRING_PATTERN = (
    r"^(click \[\d+\]|hover \[\d+\]|type \[\d+\] \[[ -~]+\] \[[01]\]|"
    r"press \[[^\]\r\n]+\]|scroll \[(?:up|down)\]|goto \[[^\]\r\n]+\]|new_tab|"
    r"go_back|go_forward|close_tab|tab_focus \[\d+\]|stop \[.*\]|stop|noop)$"
)
WEBARENA_ACTION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reasoning_text", "action_strings"],
    "properties": {
        "reasoning_text": {"type": "string"},
        "action_strings": {
            "type": "array",
            "items": {"type": "string", "pattern": _WEBARENA_ACTION_STRING_PATTERN},
        },
    },
}


_VERB_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")

# Map the WebArena verb to our internal action_type. ``stop`` ends the
# trajectory — we route to ``done`` so the OSWorld-shaped completion-gate path
# applies uniformly (the translator turns ``done`` back into a stop_action
# at the WebArena boundary).
_VERB_TO_INTERNAL: dict[str, str] = {
    "stop": "done",
    "noop": "wait",
}


def _system_prompt() -> str:
    """Shared system prompt — task-agnostic; the per-step user prompt supplies
    the observation + intended actions."""
    return (
        "You are the GraphPC WebArena action agent. You ground the upstream "
        "intended_actions onto WebArena's id-based browser action grammar "
        "against the current accessibility tree. Do not plan or re-decide — "
        "locate each intended action's semantic target as an element id on "
        "the current AXTree and emit the corresponding action string.\n"
        "\n"
        "Allowed action strings (one per element in `action_strings`):\n"
        "  click [id]\n"
        "  hover [id]\n"
        "  type [id] [text] [flag]   (flag 0 = no enter, 1 = press enter; default 1)\n"
        "  press [key_comb]          (e.g. press [Ctrl+Enter])\n"
        "  scroll [direction]        (direction = up | down)\n"
        "  goto [url]\n"
        "  go_back\n"
        "  go_forward\n"
        "  new_tab\n"
        "  tab_focus [index]\n"
        "  close_tab\n"
        "  stop [answer]             (task completion; answer may be empty)\n"
        "  noop                       (no-op step boundary)\n"
        "\n"
        "Return strict JSON with `reasoning_text` and `action_strings`. Only "
        "use element ids that appear on the supplied AXTree."
    )


def _build_user_payload(action_input, intended_actions: list[dict[str, Any]]) -> dict[str, Any]:
    observation_text = ""
    if hasattr(action_input.observation, "content") and isinstance(
        action_input.observation.content, dict
    ):
        observation_text = action_input.observation.content.get("observation_text", "") or ""
    # observation_text is the projected AXTree (relevance-selected,
    # bracketed ids preserved) — the action agent grounds against it.
    return {
        "instruction": action_input.instruction,
        "current_stage": action_input.runtime_state.current_stage,
        "observation_text": observation_text,
        "intended_actions": intended_actions,
        "max_actions": action_input.max_actions,
    }


def parse_action_verb(action_string: str) -> tuple[str, str]:
    """Pull the verb out of an id-based action string. Returns
    ``(internal_action_type, verb)`` so the orchestrator's gating logic sees
    "done"/"wait" rather than the WebArena-specific "stop"/"noop"."""
    match = _VERB_RE.match(action_string or "")
    if not match:
        return ("wait", "noop")
    verb = match.group(1).lower()
    internal = _VERB_TO_INTERNAL.get(verb, verb)
    return internal, verb


def project_intended_actions(reasoning_message) -> list[dict[str, Any]]:
    """Slim handoff: read ONLY intended_actions from the reasoning message
    (the §1 invariant we enforce)."""
    if reasoning_message is None:
        return []
    content = reasoning_message.content or {}
    intended = content.get("intended_actions")
    return deepcopy(intended) if isinstance(intended, list) else []


def build_action_batch(
    *,
    raw_response: Any,
    response_json: dict[str, Any],
    intended_actions: list[dict[str, Any]],
    action_input,
    tool_name: str,
    url_mappings: dict[str, str],
) -> ActionBatch:
    """Common ``ActionBatch`` assembly for every backbone — wraps each
    action_string into one of our ``Action``s with the WebArena payload on
    ``raw["input"]["command"]`` (where the translator picks it up), the
    ``action_type`` mapped to our internal vocabulary so the orchestrator's
    gates (done/wait) work uniformly, and ``predicted_effect_class`` mirrored
    from the intended action via positional pairing (the effect-class
    stamping convention).

    Each action string goes through the control's ``map_url_to_local`` over the
    WHOLE string, as upstream's ``extract_action`` does before
    ``create_id_based_action``: alias hosts in a ``goto``, a ``type`` text or a
    ``stop`` answer become the local hosts. It is mapped HERE so history, the
    completion gate, ``_executed_actions`` and the submitted action all read the
    one mapped string (``command`` and its documented mirror
    ``webarena_action_string``). ``raw_response`` is never rewritten: it is the
    reply as the model sent it, so the issued alias survives there, and history
    carries it. Each rewrite is recorded as an event on the BATCH's metadata,
    which reaches no prompt. It is never recorded on ``Action.metadata``, which
    history copies into the next deciding prompt.
    """
    action_strings = response_json.get("action_strings") or []
    if not action_strings:
        action_strings = ["noop"]

    actions: list[Action] = []
    mapping_events: list[dict[str, Any]] = []
    for index, action_string in enumerate(action_strings[: action_input.max_actions]):
        issued = str(action_string).strip()
        s = map_url_to_local(issued, url_mappings)
        if s != issued:
            mapping_events.append({"action_index": index, "issued": issued, "mapped": s})
        internal_type, verb = parse_action_verb(s)
        metadata: dict[str, Any] = {
            "tool_name": tool_name,
            "webarena_verb": verb,
            "webarena_action_string": s,
            "raw_response": raw_response,
            "action_space": "webarena_id_based",
            "action_index": index,
        }
        if index < len(intended_actions):
            predicted = (intended_actions[index] or {}).get("predicted_effect_class")
            if predicted is not None:
                metadata["predicted_effect_class"] = predicted
        actions.append(
            Action(
                action_type=internal_type,
                parameters={"webarena_verb": verb, "command": s},
                raw={
                    "name": tool_name,
                    "input": {"command": s, "verb": verb},
                    "raw_response": raw_response,
                },
                metadata=metadata,
            )
        )

    batch = ActionBatch(
        actions=actions,
        content={"reasoning_text": response_json.get("reasoning_text", "")},
        raw={"response_json": response_json, "raw_response": raw_response},
    )
    if mapping_events:
        batch.metadata["url_mapping_events"] = mapping_events
    return batch
