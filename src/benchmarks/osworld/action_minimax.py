"""MiniMax action agent for OSWorld 2.0 — grounds intended actions into the
OSWorld-V2 M3 ``computer`` tool vocabulary with ``[0, 1000]`` relative
coordinates (design.md §1 "Action boundary").

Lives under the benchmark, like ``src/benchmarks/webarena/action_agent.py``:
the vocabulary it emits is OSWorld-V2's, not a backbone-generic one.

Grounding call = OSWorld-V2's NATIVE format (copy-first applied to the
agent-side output format). System prompt = a copy
of V2's ``<TOOLS>`` block plus the ``[0, 1000]`` relative-coordinate rule
(``M3_GROUNDING_SYSTEM_PROMPT``, provenance below); user turn = the
screenshot + one short imperative line rendered from the intended action;
``backbone.complete_text(grounding=True, stop=["</tool_call>"])`` —
non-thinking, the grounding sampling pin (``grounding_temperature``; OSWorld's
build_default passes 0.6 = the V2 M3 runner's non-thinking default — a
documented instantiation, not a parity pin: the baseline arm grounds
thinking-adaptive at the forced temperature 1),
NO tools, NO response_format.
The model writes its trained ``Action: ...`` line + ``<tool_call>`` JSON,
which ``parse_m3_tool_calls`` (V2's own scanner + BPE recovery) turns into
``(action, arguments)``. Forced function-calling pushes M3 off that
distribution: on the same endpoint, image and prompt its forced calls come
back malformed or off target, where its trained free-text format grounds well.

Slim handoff (design.md Workflow, the same rule as the WebArena action
agents): the action stage sees ONLY the structured ``intended_actions`` —
never the reasoning trace, plan, graph, or retrieved experience — and the
grounding line carries the target by label/role only: coordinates the deciding
call wrote into ``target`` / ``intent`` are stripped from it
(``strip_grounding_coordinates``; payload fields are never touched).

Terminal classes (``done``, ``fail``) in the intended action are emitted
WITHOUT a model call: ending the episode is the reasoning call's decision,
and grounding must not be able to re-decide it — which is why the ``done``
entry and the termination rules are removed from the copied prompt, and why
a grounding-emitted ``done`` / ``fail`` / ``terminate`` / ``call_user`` (or
the ``[INFEASIBLE]`` token) never becomes an ``Action`` (rejected below).
Both are emitted as action type ``done`` with ``metadata["terminal_intent"]``
naming which, so the orchestrator's completion gate — which acts on ``done`` —
evaluates a give-up exactly as it evaluates a success claim (see
``_terminal_action``); the adapter emits ``FAIL`` for an admitted ``fail``.
There is NO ask-user channel on this profile (parity by omission: the
baseline prompt forbids asking).

The four bypassed COORDINATE-FREE classes (``key``, ``type``, ``wait``,
``screenshot``; ``_BYPASS_CLASSES`` — ``hold_key`` stays on the grounding
route, see there) are emitted without a model call for a
different reason: they take no coordinate, so the tool call is fully
determined by the intended action and the grounding request has nothing to
ground — it can acquire no information from the screenshot that the intent
does not already carry, and in practice it loses much of what it is given
(a ``type`` intent sent through grounding often comes back as a click).
Building the call in Python also removes
the escaping failure class by construction: a payload of shell text with
quotes and pipes never has to survive a model emitting JSON. Every other
intended class still goes to the grounding model as a computer action.

Degrade contract (design.md §1): a grounding response that cannot become an
action — no ``computer`` tool call, a rejected terminal / ask name, the
``[INFEASIBLE]`` token — yields an EMPTY batch carrying
``grounding_parse_error`` in ``content``. All three kinds are re-sampled
first, up to ``model_call_attempts`` attempts — 3 on this benchmark, the retry
budget of V2's M3 agent, passed by build_default: each yields no
executable action FOR OUR PIPELINE, and the grounding route has no termination
authority — termination is decided on the reasoning route and verified by the
completion gate. Residual difference from the baseline, to be named in any
verdict: on a terminal-name reply V2's M3 agent emits DONE/FAIL and ends its
episode, while this route re-samples it (up to two extra model calls, no step
cost). A well-formed call with invalid arguments is the translator's
``ValueError`` instead: a consumed no-op step, never re-sampled, as in V2's
M3 agent. The adapter emits a counted no-op
and surfaces the failure to the next reasoning call via the monitor feedback
channel (the contract ``src/benchmarks/webarena/action_agent.py`` follows).
Argument defects are the translator's ``ValueError`` (its only argument error), also a counted
no-op at the adapter. A coordinate-free intent carrying no usable payload
takes the SAME empty-batch route (``M3CoordinateFreeArgumentMissing``), but
is not re-sampled: there is no model call to repeat. A rejection mid-loop returns the EMPTY batch and
discards actions already grounded in the same call — moot at ``max_actions=1``
(the OSWorld setting); a ``max_actions > 1`` caller would have
to decide whether a partial batch is wanted.
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

from src.agents.action.base import ActionAgent, ActionInput
from src.agents.backbone.minimax.main import MiniMaxAgent
from src.agents.common.resample import call_with_resample
from src.agents.common.schema import Action, ActionBatch

from .m3_action_translator import (
    _ARGUMENT_CONTRACT,
    M3GroundingRejected,
    is_unset_argument,
    parse_m3_tool_calls,
)


logger = logging.getLogger("desktopenv.agent")


# PROVENANCE — OSWorld-V2/mm_agents/m3/prompts.py, tag v2026.08.08 (d578d2d),
# ``M3_SYSTEM_PROMPT_TEMPLATE`` with ``{{``/``}}`` rendered:
#   * the ``* All `coordinate` values ...`` line of <SYSTEM_CAPABILITY> (its
#     other lines — platform / home directory / sudo password / date facts and
#     the TASK FEASIBILITY section — are NOT copied: environment facts reach
#     the reasoning call through the environment brief, and feasibility
#     is the reasoning call's decision);
#   * the <TOOLS> block VERBATIM except the termination content, removed
#     because reasoning owns termination and the parser rejects these
#     emissions: the ``* `done` ...`` entry and the two Rules lines
#     "When the task has been completed ... action `done` ..." and
#     "For impossible/infeasible tasks ... `[INFEASIBLE]` ...".
M3_GROUNDING_SYSTEM_PROMPT = """<SYSTEM_CAPABILITY>
* All `coordinate` values in `computer` tool calls are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen, regardless of the underlying screen resolution.
</SYSTEM_CAPABILITY>

<TOOLS>
You have access to the `computer` tool for interacting with the desktop GUI. Output one tool call per turn.

For each tool call, return a json object with name and arguments inside <tool_call></tool_call> XML tags:
<tool_call>
{"name": "computer", "arguments": {"action": "<action>", ...}}
</tool_call>

Supported `action` values include:
* `key`        — args: {"text": "<key combo, e.g. ctrl+s>"}
* `type`       — args: {"text": "<text to type>"}
* `mouse_move` — args: {"coordinate": [x, y]}
* `left_click` / `right_click` / `middle_click` / `double_click` / `triple_click`
                 — args: {"coordinate": [x, y]}, optional {"text": "<modifier keys>"}
* `left_click_drag` — args: {"coordinate": [x, y]}, optional {"start_coordinate": [x, y]}
* `scroll`     — args: {"coordinate": [x, y], "scroll_direction": "up|down|left|right", "scroll_amount": <int>}
* `wait`       — args: {"duration": <seconds>}
* `screenshot` — args: {}
* `hold_key`   — args: {"text": "<keys>", "duration": <seconds>}
* `left_mouse_down` / `left_mouse_up` — args: {"coordinate": [x, y]}

Rules:
- Output exactly one short imperative `Action: <text>` line followed by exactly one <tool_call>...</tool_call> block.
- Coordinates are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen.
</TOOLS>"""

# V2's stop sequence for the tool-call block (mm_agents/m3/agent.py); the
# copied parser recovers the missing closing tag either way.
M3_GROUNDING_STOP = ["</tool_call>"]

# THE PROMPT-FACING ACTION CONTRACT, DERIVED from the translator, not retyped.
#
# `m3_action_translator._ARGUMENT_CONTRACT` is the table the runtime actually
# validates against, so anything disagreeing with it is wrong by definition. This is
# its PROMPT-FACING subset, and it is computed rather than written out: a hand-copied
# table drifts from the translator's (an optional argument such as `mouse_move`'s
# `duration` silently missing), which is the whole failure mode.
#
# THE EXCLUSIONS ARE THE ONLY JUDGEMENT HERE, so they are named rather than implied:
#   * `left_press` — accepted by the translator (it is in the model's `_VALID_ACTIONS`)
#     but deliberately kept out of upstream's <TOOLS> block; advertising it would
#     expose a name upstream chose to hide.
#   * `done` / `fail` — the reasoning stage owns them and they take no tool arguments.
#     They rejoin in the vocabulary below, which is a different question.
#
# Deriving means a new action in the translator enters the vocabulary and the schema
# enum automatically. That is the safe direction, but it could enter WITHOUT the
# grounding prompt advertising it: when the translator gains an action, check every
# name here against `M3_GROUNDING_SYSTEM_PROMPT` before syncing the change.
_PROMPT_FACING_EXCLUSIONS: frozenset[str] = frozenset({"left_press", "done", "fail"})
M3_ACTION_ARGUMENTS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = tuple(
    (name, required, optional)
    for name, (required, optional) in _ARGUMENT_CONTRACT.items()
    if name not in _PROMPT_FACING_EXCLUSIONS
)

# The closed vocabulary `action_class` may take on this benchmark: the prompt-facing
# action set above plus the two terminal classes the reasoning stage owns. ONE
# constant with three consumers — the schema enum (passed to the reasoning agent by
# build_default), the normaliser's target set below, and the bypass membership test
# — so the three cannot drift. NOT design.md's list, which is WebArena-shaped (goto,
# hover, tab ops).
M3_ACTION_CLASS_VOCABULARY: tuple[str, ...] = tuple(
    name for name, _required, _optional in M3_ACTION_ARGUMENTS
) + ("done", "fail")

# Surface variants of a vocabulary name, normalised before anything keys on the
# field. Modelled on `src/agents/common/effect_class.py`'s synonym map and kept to
# the same standard: ONLY obvious, unambiguous variants of a name that already
# exists. Deciding replies do carry off-vocabulary `action_class` names; this table
# resolves the common variants at zero model cost, and the residue falls through to
# the grounding model — the UNKNOWN-sentinel analogue, unresolved and handled
# conservatively.
#
# THIS TABLE STOPS BEING SAFE THE MOMENT SOMEONE ADDS A NAME WHOSE MEANING IS NOT
# UNIQUE. Names deliberately LEFT OUT, each because the name does not determine the
# action: `keyboard` (it arrives with a key combo in `keys` or with a shell command in
# `text`, so a name-to-name map would try to press `ls -la ~/Desktop/` as a key
# combo), `key_hold`, and the names of tools which do not exist at all (`bash`,
# `open_app`, `open_terminal`, `open_file`).
_ACTION_CLASS_ALIASES: dict[str, str] = {
    "type_text": "type", "typewrite": "type", "write": "type",
    "key_press": "key", "keypress": "key", "hotkey": "key",
    "press_key": "key", "key_combo": "key",
    "computer_screenshot": "screenshot", "take_screenshot": "screenshot",
    "click": "left_click", "leftclick": "left_click",
    "double_left_click": "double_click",
    "sleep": "wait",
    "move_mouse": "mouse_move",
    "mouse_drag": "left_click_drag", "drag": "left_click_drag",
}

# Intended-action classes that bypass the model. TWO SETS, KEPT APART ON PURPOSE:
# they bypass for different reasons and collapsing them would hide that.
#
#   _TERMINAL_CLASSES  — grounding has no AUTHORITY here. Ending the episode is the
#                        reasoning call's decision and the grounding route must not
#                        be able to re-decide it.
#   _BYPASS_CLASSES    — grounding has nothing to GROUND. For these four the tool
#                        call is a total function of the intent alone, so the
#                        screenshot supplies no argument the intent does not already
#                        carry. A call that cannot acquire information cannot lose
#                        any when removed — which is also why this structurally
#                        removes the escaping failure class: the call is built in
#                        Python, so no model ever has to emit valid JSON containing a
#                        payload of shell text, quotes and pipes.
#
# `wait` is here because the duration is DISCARDED: `tool_action_to_pyautogui`
# emits `pyautogui.sleep(0.5)` for any wait, with or without a duration argument
# (upstream's `mm_agents/m3/parser.py` is byte-identical), so it is as constant a
# tool call as `screenshot` and there is nothing for a model to parse.
#
# `hold_key` is NOT here, and the reason is an UPSTREAM LATENT DEFECT rather than
# anything about the bypass. The copied branch emits `keyDown` for each part and
# NEVER a `keyUp`, and it never reads `duration` — so any `hold_key` leaves the key
# held for the rest of the episode, modifying every later keystroke. That is true in
# BOTH arms. It is not ours to repair: fixing it on our side only would
# be the one-sided repair refused everywhere else here. Bypassing the class would
# make our arm the one that exercises the defect, so it stays on the grounding
# route, where the model's own refusal is the thing standing in front of it.
#
# `scroll` is not here because its tool signature requires a coordinate.
_TERMINAL_CLASSES: frozenset[str] = frozenset({"done", "fail"})
_BYPASS_CLASSES: frozenset[str] = frozenset({"type", "key", "screenshot", "wait"})

# THE FIELD PRECEDENCE, per bypassed class, stated rather than inferred and MEASURED
# rather than guessed. `_INTENDED_ACTION_SCHEMA` carries `text`, `keys` AND `value`
# as separate nullable fields while the tool takes one `text` argument, so a bypass
# has to decide an ambiguity a grounding model would absorb by re-reading the
# imperative — and a wrong choice here is silent corruption, not a visible failure:
# `type {"text": "find /etc"}` types the string, while `key {"text": "find /etc"}`
# emits `keyDown('find /etc')`, which presses nothing.
#
# As executed intents carry them: `type` resolves from `text`, `key` from `keys`
# then `text`. There is NO `value` fallback anywhere: a bypassed-class intent with an
# empty payload does not carry its argument in `value` instead. An empty tuple
# means the class takes no argument from the intent at all.
_BYPASS_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "type": ("text",),
    "key": ("keys", "text"),
    "screenshot": (),
    "wait": (),
}

# The classes whose payload has to NAME something rather than BE something. The
# structural guard below runs for these only: a `type` payload legitimately contains
# newlines and spaces, and refusing those would reject almost every typed string.
_KEYSTROKE_CLASSES: frozenset[str] = frozenset({"key"})


# Grounding-emitted action names that must never become an Action: the
# reasoning call owns termination and there is no ask-user channel.
_REJECTED_GROUNDING_ACTIONS: frozenset[str] = frozenset({"done", "fail", "terminate", "call_user"})


class M3GroundingZeroActions(M3GroundingRejected):
    """The grounding reply parsed to zero actions — no ``computer`` tool call,
    malformed tool-call JSON, a wrong tool name, a missing ``action`` (the
    parser skips all of these). Re-sampled (common/resample.py); V2's M3
    agent re-samples this class as well."""


class M3GroundingTerminalRejected(M3GroundingRejected):
    """The grounding reply named a terminal / ask action (``done``, ``fail``,
    ``terminate``, ``call_user``). From the pipeline's point of view this is
    ALSO zero executable actions — the grounding route has no termination
    authority (termination is decided on the reasoning route and verified by
    the completion gate) — so it is re-sampled like a no-tool-call reply.
    Kept as its own class so the per-step
    ``grounding_parse_error`` marker names which kind occurred. (The
    ``[INFEASIBLE]`` token is the same case; the parse layer raises the base
    class for it, re-sampled as well.)"""


class M3BypassPayloadRejected(M3GroundingRejected):
    """A bypassed intended action whose payload cannot become an executable call.

    Two kinds:

      * NOT A KEYSTROKE — the payload is a shell command or two combos crammed into
        one field, so pressing it would be a silent no-op: shell commands labelled
        ``key`` (``ls\n``, ``pip install ... | tail -5\n``, a bare ``\n``, long
        ``echo ... | sudo -S apt install ...`` compounds) or ``shift+Tab alt+Down``.
        This is the whole reason the guard exists.
      * MISSING — the named fields are all empty. PURELY DEFENSIVE: a missing
        required argument must not become a silent no-op, so the branch stays; it is
        expected to fire rarely, if ever, and is interesting only when it does.

    THE FIRST KIND IS WHY "the grounding model cannot acquire information" NEEDS A
    QUALIFIER. It holds where the payload is a keystroke. Handed
    ``key {"text": "ls\n"}`` the grounding model declines and substitutes — a
    PROTECTIVE refusal, not the corruption the bypass exists to prevent — so without the
    guard the bypass would turn a visible substitution into a silent no-op. Read any
    substitution count with that distinction in mind; protective refusals and dropped
    good intents are not one number.

    WHAT THE GUARD DELIBERATELY DOES NOT DO is test the payload against pyautogui's
    key set. ``ctrl+grave`` and ``super`` are unpressable or wrongly mapped in the
    COPIED translator, so the control hits the same wall on the same records;
    refusing them here would convert a symmetric no-op into a one-sided WAIT for our
    arm, which is a delta in our favour rather than a repair.

    Degrades through the route that already exists: empty batch,
    ``grounding_parse_error`` in ``content``, a counted ``WAIT`` at the adapter with
    this class named, and ``grounding_parse_failures`` advanced. It also carries
    ``intended_action_defect``, which the adapter puts on the next step's feedback so
    the reasoning call is TOLD what was wrong — without that the WAIT is silent and
    the model re-issues the same intent.

    NOT re-sampled, and that is the one behavioural difference from the grounding
    route's rejections: re-sampling means repeating a model call, and here there is
    no call to repeat. The intent is already final."""


# COORDINATES NEVER CROSS THE GROUNDING HANDOFF (design.md §1: the reasoning call
# names a target by label/role; grounding alone finds it on the screen). The
# deciding call is not asked for coordinates and gets no field for them, but it writes
# them into `target` / `intent` prose anyway, in no consistent unit and mostly copied
# from history — and grounding copies them, so a stale coordinate misplaces the
# action, while with the coordinates stripped grounding finds the named target on
# the screen. On a line that carries none, the strip is a byte no-op.
#
# The patterns: a "(x, y)" / "[x, y]" pair with its lead-in ("at coordinates", "at
# approximately", "around approximate coordinates", ...), an "x=…, y=…" pair (=, ≈,
# ~ or :), and a lone "x=…" / "y=…", where
#   * a number may carry a leading "~" / "≈" — "(~403, 546)", "(≈170,243)";
#   * the lone form may carry a range tail — "x≈275-410".
# A number is an INTEGER: the coordinates the deciding call writes are integers,
# while the decimal pairs in intents are document measurements — "moved to (11, 0.5)"
# (slide inches), "(15.97, 9.03) cm" — which the strip must leave alone.
# The lone form needs a word boundary before the x / y, so "Budget cap=48000" and
# "months=5" are untouched. Applied to `target` and `intent` ONLY: `text` / `keys` /
# `value` are payloads — what gets typed or pressed — and a "(1, 2)" inside one is
# content, not a location.
_COORDINATE_NUMBER = r"[~≈]?\s*\d{1,4}(?!\.\d)"
_COORDINATE_LEAD_IN = (r"(?:\s+(?:at|around|near|to))?(?:\s+(?:approximate|approximately|approx\.?|about|roughly))?"
                       r"(?:\s+coordinates?)?")
_COORDINATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(_COORDINATE_LEAD_IN + r"\s*[\(\[]\s*" + _COORDINATE_NUMBER + r"\s*,\s*" + _COORDINATE_NUMBER
               + r"\s*[\)\]]"),
    re.compile(_COORDINATE_LEAD_IN + r"\s*\(?\s*x\s*[=≈~:]\s*" + _COORDINATE_NUMBER
               + r"\s*,?\s*(?:and\s+)?y\s*[=≈~:]\s*" + _COORDINATE_NUMBER + r"\s*\)?", re.I),
    re.compile(_COORDINATE_LEAD_IN + r"\s*\b[xy]\s*[=≈~]\s*" + _COORDINATE_NUMBER
               + r"(?:\s*[-–]\s*\d{1,4}(?!\.\d))?\b", re.I),
)
# The intended-action fields the strip applies to; everything else is left byte for byte.
_COORDINATE_STRIPPED_FIELDS = ("target", "intent")


def strip_coordinates(text: str) -> tuple[str, list[str]]:
    """``text`` with every coordinate span removed, and the removed spans in the
    order removed (pattern by pattern)."""
    removed: list[str] = []
    for pattern in _COORDINATE_PATTERNS:
        removed.extend(match.group(0) for match in pattern.finditer(text))
        text = pattern.sub("", text)
    return text, removed


def strip_grounding_coordinates(intended_action: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(intended action for the grounding line, audit record)``. A field that the
    strip would EMPTY keeps its original text — a line with no target at all is a
    worse input than one with a coordinate in it — and the record says so
    (``fallback``). The record: ``stripped`` (anything removed and kept removed),
    ``removed`` ({field: [spans]}), ``fallback`` ([fields])."""
    cleaned = dict(intended_action)
    removed: dict[str, list[str]] = {}
    fallback: list[str] = []
    for field in _COORDINATE_STRIPPED_FIELDS:
        value = intended_action.get(field)
        if not isinstance(value, str):
            continue
        text, spans = strip_coordinates(value)
        if not spans:
            continue
        if not text.strip():
            fallback.append(field)
            continue
        cleaned[field] = text
        removed[field] = spans
    return cleaned, {"stripped": bool(removed), "removed": removed, "fallback": fallback}


def render_intended_action(intended_action: dict[str, Any]) -> str:
    """The one imperative line the grounding model sees: action_class +
    semantic target, then the parameters and intent it carries, e.g.
    ``click Save button (text: "Q3 report"; intent: "save the file")``."""
    action_class = str(intended_action.get("action_class") or "act").strip()
    target = str(intended_action.get("target") or "").strip()
    line = f"{action_class} {target}".strip()
    details = [
        f"{key}: {json.dumps(str(intended_action[key]))}"
        for key in ("text", "keys", "value", "intent")
        if intended_action.get(key) not in (None, "")
    ]
    if details:
        line += " (" + "; ".join(details) + ")"
    return line


def normalize_action_class(raw: Any) -> str:
    """The emitted ``action_class``, reduced to a vocabulary name where the variant
    is unambiguous; otherwise the token unchanged.

    Modelled on ``src/agents/common/effect_class.py:normalize_effect_class`` — same
    token handling, same deliberately-small synonym table, same fall-through — and
    placed HERE rather than in the shared reasoning path on purpose. It exists only
    to make grounding and bypass keying reliable, grounding is benchmark-side by
    design, and putting it here makes the class-two quarantine STRUCTURAL: the code
    is not on WebArena's path at all, rather than being guarded by a parameter that
    defaults off there.

    Normalisation rather than a schema refusal is the same choice
    ``normalize_effect_class`` already makes for the other closed vocabulary. On this
    benchmark the schema cannot enforce anything anyway — MiniMax's deciding route is
    ``response_format=json_object`` with the schema appended as advisory prompt text
    (backbone/minimax/main.py, module docstring and ``_invoke_reasoning``), and
    ``require_decision_shape`` checks container shape only — so the alternative was
    re-sampling the deciding call, the expensive one carrying the screenshot and the
    history, for every variant.
    """
    token = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ACTION_CLASS_ALIASES.get(token, token)


def coerce_intent_field(field: str, value: Any) -> str | None:
    """One intent field as the ``str`` the tool argument needs, or None when it is
    not populated. Raises ``M3BypassPayloadRejected`` for a shape with no unambiguous
    reading.

    THE FIELDS ARE POLYMORPHIC AND THE SCHEMA DOES NOT STOP IT. ``schemas_json.py``
    declares ``keys`` / ``text`` / ``value`` as string-or-null, and the model does
    violate that, because MiniMax does not enforce the schema server-side — it is appended to the prompt as advisory text. That is the
    same fact behind the enum being advisory, and it means every consumer must handle
    what the model actually sends rather than what the contract says.

    A list ``keys`` JOINS WITH ``+`` (``['ctrl','shift','j']`` -> ``ctrl+shift+j``):
    the shape arrives on key-family intents, all inside the bypass set, and the
    translator's existing conversion handles every result. That is a SHAPE coercion,
    not an intent guess. A list ``text`` has no unambiguous join — prose is not a
    combo — so it raises; it is not expected, but a sibling field produces the shape
    and the schema permits it. An int ``value`` coerces with ``str``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, list) and field == "keys":
        parts = [str(part) for part in value if isinstance(part, (str, int, float))]
        if len(parts) != len(value) or not parts:
            raise M3BypassPayloadRejected(f"the {field!r} list {value!r} is not a key combo")
        return "+".join(parts)
    if isinstance(value, int) and not isinstance(value, bool) and field == "value":
        return str(value)
    raise M3BypassPayloadRejected(
        f"the {field!r} field carries {type(value).__name__} {value!r}, which has no unambiguous "
        f"reading as a tool argument"
    )


def reject_non_keystroke_payload(action_class: str, payload: str) -> None:
    """Raise unless ``payload`` is STRUCTURALLY a keystroke. No-op for a class whose
    payload is not a key name.

    STRUCTURAL, NOT A KEY-SET MEMBERSHIP TEST, and that is the whole design. The
    isolation that makes this safe is that the control has no intent stage: its model
    writes the tool call directly and would have written ``type`` for a shell
    command, so a ``key`` call carrying one is uniquely ours. Testing against
    pyautogui's key set instead would refuse ``ctrl+grave`` and mis-handle ``super``,
    which the COPIED translator gets wrong in both arms — converting a symmetric
    no-op into a one-sided WAIT.

    THE ORDER IS LOAD-BEARING AND IT IS EASY TO GET WRONG. The newline test runs on
    the payload BEFORE any stripping, because the translator itself does
    ``k.strip().lower()`` on each part: ``'ls\n'`` strips to ``'ls'``, which has no
    internal whitespace, passes a strip-first check, and is then pressed as a key
    named ``ls``. Running this rule strip-first misses ``'ls\n'`` and ``'\n'``.
    Do not normalise and then check.

    No length threshold either: length is a proxy, and the two properties below are
    the actual ones. A single long token with no whitespace would pass; inventing a
    cut-off would be the guess this rule exists to avoid.
    """
    if action_class not in _KEYSTROKE_CLASSES:
        return
    if "\n" in payload or "\r" in payload:
        raise M3BypassPayloadRejected(
            f"the {action_class!r} payload {payload[:80]!r} contains a newline, so it is a command "
            f"or a line of text rather than a key combo; it needs a different action class"
        )
    for part in payload.split("+"):
        stripped = part.strip(" \t")
        if any(character.isspace() for character in stripped):
            raise M3BypassPayloadRejected(
                f"the {action_class!r} payload {payload[:80]!r} has a component {stripped[:40]!r} "
                f"containing internal whitespace, so it is not a single key name"
            )


def bypass_arguments(action_class: str, intended_action: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """``(tool arguments, the intent field that supplied them)`` for a bypassed class,
    or ``M3BypassPayloadRejected``.

    THREE STAGES IN THIS ORDER — coerce, resolve, guard — and the order is the point:

      1. COERCE each candidate field to a ``str`` (``coerce_intent_field``), so a
         list-valued ``keys`` becomes a combo before anything reads it. Resolving
         first would take the list as "populated" and hand a list to the translator.
      2. RESOLVE by the measured precedence above, and require what won to be
         non-empty — checked on the value that actually feeds ``text``, not by
         inspecting the intent, because a ``keys`` of ``""`` does not look empty from
         outside and yields ``keyDown('')``.
      3. GUARD what won, structurally (``reject_non_keystroke_payload``), on the
         string BEFORE the translator strips it.

    The second element of the return is an anti-masking record:
    a bypass translates whatever the intent holds, faithfully and silently, so which
    field it read has to be visible or an upstream defect has nowhere to surface.
    """
    fields = _BYPASS_TEXT_FIELDS[action_class]
    if not fields:
        return {}, None
    for field in fields:
        value = coerce_intent_field(field, intended_action.get(field))
        if value:
            reject_non_keystroke_payload(action_class, value)
            return {"text": value}, field
    seen = {field: intended_action.get(field) for field in fields}
    raise M3BypassPayloadRejected(
        f"the {action_class!r} intent carries no text to act on: {seen!r} "
        f"(checked {list(fields)} in that order)"
    )


class MiniMaxActionAgent(ActionAgent):
    """``model_call_attempts`` is the TOTAL grounding attempts per intended action
    (1 = no re-sample). It defaults to 1 — the shared default, which is WebArena's
    control behaviour (``--max_retry`` 1, one parse failure and it degrades) — and
    OSWorld's ``build_default`` passes 3, which is PARITY with this benchmark's own
    control: V2's M3 agent loops ``range(self.max_llm_retries + 1)`` with
    ``M3_MAX_LLM_RETRIES`` defaulting to 2 (mm_agents/m3/agent.py:283, :647). The
    budget is a per-benchmark parameter rather than a shared constant precisely
    because the same number is parity on one benchmark and a threefold advantage on
    the other — see src/agents/common/resample.py."""

    def __init__(
        self, *, backbone: MiniMaxAgent, tool_name: str = "computer", model_call_attempts: int = 1,
    ) -> None:
        self.backbone = backbone
        self.tool_name = tool_name
        self.model_call_attempts = int(model_call_attempts)

    def act(self, action_input: ActionInput) -> ActionBatch:
        intended = self._intended_actions(action_input)
        if not intended:
            # Nothing to ground (e.g. reasoning soft-degraded to an empty
            # decision on a parse failure): empty batch, no grounding call.
            return ActionBatch(actions=[], content={}, raw={})

        actions: list[Action] = []
        raw_responses: list[Any] = []
        instructions: list[str] = []
        # Re-samples across the batch's grounded actions (one per intended
        # action; max_actions=1 in the shipped configuration).
        resamples_total = 0
        strip_records: list[dict[str, Any]] = []
        for index, intended_action in enumerate(intended[: action_input.max_actions]):
            intended_action = intended_action or {}
            # NORMALISE FIRST, then test membership. One table, not two: the alias
            # map and the bypass set are otherwise the same knowledge written twice,
            # and two tables that must agree is a defect waiting to happen. The
            # exclusions (`scroll`, `wait`, `keyboard`, the tools that do not exist)
            # are then consequences of this membership test rather than a second list.
            raw_action_class = str(intended_action.get("action_class") or "")
            action_class = normalize_action_class(raw_action_class)
            if action_class in _TERMINAL_CLASSES:
                actions.append(self._terminal_action(
                    action_type=action_class, intended_action=intended_action, index=index,
                ))
                continue
            if action_class in _BYPASS_CLASSES:
                # NO GROUNDING REQUEST IS ISSUED. The tool call is built in Python
                # from the intent, so the text to type never has to survive a round
                # trip through a model emitting JSON. A payload that is missing or
                # not executable degrades through the SAME route a rejected grounding
                # reply takes.
                try:
                    action = self._validated_bypass_action(
                        action_name=action_class, raw_action_class=raw_action_class,
                        intended_action=intended_action, index=index,
                    )
                except M3GroundingRejected as exc:
                    logger.warning("bypassed intent cannot be executed: %s; degrading to empty batch", exc)
                    return self._rejected_batch(exc, resamples_total=resamples_total)
                actions.append(action)
                continue
            # Prompt rendered ONCE and closed over: every re-sample sends
            # identical request bytes (see common/resample.py). Every parse
            # layer rejection is re-sampled — no tool call / malformed call
            # (M3GroundingZeroActions), a terminal or ask name
            # (M3GroundingTerminalRejected) and the [INFEASIBLE] token (the
            # base class): each yields zero EXECUTABLE actions, and the
            # grounding route cannot terminate (see the classes above). A
            # well-formed call with invalid arguments is not a parse rejection:
            # it reaches the translator, whose ValueError is a consumed no-op
            # step at the adapter, as in V2's M3 agent.
            # Coordinates the deciding call wrote into its prose never reach the
            # grounding line (see strip_grounding_coordinates).
            grounding_intent, strip_record = strip_grounding_coordinates(intended_action)
            strip_records.append(strip_record)
            user_prompt = render_intended_action(grounding_intent)
            try:
                (text, instruction, action_name, arguments, tool_calls), resamples = call_with_resample(
                    lambda: self._ground_once(
                        user_prompt=user_prompt, observation=action_input.observation,
                    ),
                    attempts=self.model_call_attempts,
                    retry_on=(M3GroundingRejected,),
                    on_retry=lambda attempt, exc: logger.warning(
                        "grounding response parsed to zero actions (attempt %d): %s; re-sampling", attempt, exc,
                    ),
                )
            except M3GroundingRejected as exc:
                # Deliberately NARROW (the WebArena precedent): only the parse
                # layer's rejection degrades — zero actions after the re-sample
                # budget, a terminal / ask name or [INFEASIBLE] at once; any
                # other exception is one of OUR bugs or an API error and
                # propagates loudly.
                logger.warning("grounding response rejected: %s; degrading to empty batch", exc)
                batch = self._rejected_batch(exc, resamples_total=resamples_total)
                batch.metadata["grounding_input_strip"] = strip_records
                return batch
            resamples_total += resamples
            if len(tool_calls) > 1:
                logger.info("grounding response carried %d tool calls; using the first", len(tool_calls))
            raw_responses.append(text)
            instructions.append(instruction)
            actions.append(self._grounded_action(
                action_name=action_name, arguments=arguments, raw_response=text,
                intended_action=intended_action, index=index,
            ))

        metadata: dict[str, Any] = {"resample_attempts": resamples_total}
        if strip_records:
            # One audit record per GROUNDED intended action (its re-samples send the
            # same line); absent on a batch that made no grounding call.
            metadata["grounding_input_strip"] = strip_records
        return ActionBatch(
            actions=actions,
            content={"reasoning_text": "\n".join(instructions)},
            raw={"minimax_responses": raw_responses},
            metadata=metadata,
        )

    def _ground_once(self, *, user_prompt: str, observation: Any):
        """One grounding request + parse. Returns ``(text, instruction,
        action_name, arguments, tool_calls)``. Raises ``M3GroundingZeroActions``
        when the reply parses to no usable ``computer`` tool call,
        ``M3GroundingTerminalRejected`` on a terminal / ask name, and lets the
        parse layer's base-class ``[INFEASIBLE]`` rejection through — all three
        ARE re-sampled like the zero-actions case (``retry_on`` at the call site
        in ``act`` is the base class ``M3GroundingRejected``). Any other
        exception propagates."""
        response = self.backbone.complete_text(
            system_prompt=M3_GROUNDING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            observation=observation,
            grounding=True,
            stop=M3_GROUNDING_STOP,
        )
        text = response["text"]
        instruction, tool_calls = parse_m3_tool_calls(text)
        if not tool_calls:
            raise M3GroundingZeroActions(f"no `computer` tool call in the grounding response: {text[:200]!r}")
        action_name, arguments = tool_calls[0]
        action_name = str(action_name).strip().lower()
        if action_name in _REJECTED_GROUNDING_ACTIONS:
            raise M3GroundingTerminalRejected(
                f"the grounding response emitted the terminal/ask action {action_name!r}"
            )
        return text, instruction, action_name, arguments, tool_calls

    # ------------------------------------------------------------------
    def _grounded_action(
        self, *, action_name: str, arguments: dict[str, Any], raw_response: Any,
        intended_action: dict[str, Any], index: int,
    ) -> Action:
        # Tool arguments minus "action", with unused optionals DROPPED so the
        # translator sees the same sparse dict V2's parser hands it (a model
        # can fill an unused key with null, "" or [] — e.g.
        # ``start_coordinate: []``).
        parameters = {
            key: value for key, value in arguments.items()
            if key != "action" and not is_unset_argument(value)
        }
        metadata: dict[str, Any] = {
            "tool_name": self.tool_name,
            "action_index": index,
            "action_space": "pyautogui",
            "coordinate_type": "relative",
            "raw_response": raw_response,
        }
        predicted = intended_action.get("predicted_effect_class")
        if predicted is not None:
            metadata["predicted_effect_class"] = predicted
        return Action(
            action_type=action_name,
            parameters=parameters,
            raw={
                "name": self.tool_name,
                "input": {"action": action_name, **parameters},
                "raw_response": raw_response,
            },
            metadata=metadata,
        )

    def _rejected_batch(self, exc: M3GroundingRejected, *, resamples_total: int) -> ActionBatch:
        """The degrade contract, in one place: an EMPTY batch naming the rejection.

        Shared by the grounding route (rejected after its re-sample budget) and the
        coordinate-free route (an intent with no usable payload, which has no model
        call to repeat). The adapter reads ``grounding_parse_error`` and emits a
        counted ``WAIT`` carrying the class name, and ``grounding_parse_failures``
        advances — one path, so a bypass failure cannot become invisible by being
        reported differently."""
        content: dict[str, Any] = {"grounding_parse_error": f"{type(exc).__name__}: {exc}"}
        if isinstance(exc, M3BypassPayloadRejected):
            # The one thing the existing degrade route does not carry: WHAT was wrong
            # with the intent. The adapter puts this on the next step's feedback, so
            # the reasoning call is told rather than left to re-issue the same intent
            # into the same silent WAIT.
            content["intended_action_defect"] = str(exc)
        return ActionBatch(
            actions=[],
            content=content,
            raw={},
            metadata={"resample_attempts": resamples_total + getattr(exc, "resample_attempts", 0)},
        )

    def _validated_bypass_action(
        self, *, action_name: str, raw_action_class: str, intended_action: dict[str, Any],
        index: int,
    ) -> Action:
        """Coerce, resolve and guard the payload, then build the Action."""
        arguments, source_field = bypass_arguments(action_name, intended_action)
        return self._bypassed_action(
            action_name=action_name, arguments=arguments, source_field=source_field,
            raw_action_class=raw_action_class, intended_action=intended_action, index=index,
        )

    def _bypassed_action(
        self, *, action_name: str, arguments: dict[str, Any], source_field: str | None,
        raw_action_class: str, intended_action: dict[str, Any], index: int,
    ) -> Action:
        """A tool call built from the intent, with no model call.

        Same ``Action`` shape the grounding route produces — the translator and the
        adapter cannot tell the two apart, which is the point — except that
        ``raw_response`` is None and the metadata records how it was built:
        ``model_call: False`` (as ``_terminal_action`` already does),
        ``intended_action_field`` (the field that supplied ``text``), and
        ``intended_action_class_raw`` (the emitted name BEFORE normalisation).

        All three are measurement integrity rather than decoration. The field is the
        only place a corrupted intent would show, since this route translates
        whatever it is given without complaint; the raw class is the only way to tell
        later whether the schema's advisory vocabulary is actually shrinking the alias
        tail, which is the only way that hint ever gets evaluated."""
        metadata: dict[str, Any] = {
            "tool_name": self.tool_name,
            "action_index": index,
            "action_space": "pyautogui",
            "coordinate_type": "relative",
            "raw_response": None,
            "model_call": False,
            "intended_action_field": source_field,
            "intended_action_class_raw": raw_action_class,
        }
        predicted = intended_action.get("predicted_effect_class")
        if predicted is not None:
            metadata["predicted_effect_class"] = predicted
        return Action(
            action_type=action_name,
            parameters=dict(arguments),
            raw={
                "name": self.tool_name,
                "input": {"action": action_name, **arguments},
                "raw_response": None,
            },
            metadata=metadata,
        )

    def _terminal_action(
        self, *, action_type: str, intended_action: dict[str, Any], index: int,
    ) -> Action:
        """``done`` / ``fail`` emitted without a model call. ``parameters["text"]``
        carries the intended action's text — the first non-empty of ``text``,
        ``value``, ``intent``, ``target`` — which the adapter reads (e.g. a
        ``done`` whose text is an infeasibility declaration is normalised to
        ``"FAIL"`` there).

        BOTH ARE EMITTED AS ``done``, with ``metadata["terminal_intent"]`` naming
        the class, and that is what keeps a give-up gated. The completion gate
        (orchestrator ``_apply_completion_gate``) acts on ``done`` only, and a
        give-up the gate blocks keeps the episode going. Emitting ``fail`` as its own
        action type would let it walk past the gate; emitting it as a ``done`` that
        says it is a ``fail`` gives a give-up the gate's coverage, its admit rule and
        its two-strike budget, shared by both terminal forms. The adapter reads the
        intent back and emits ``FAIL`` for an admitted one."""
        text = next(
            (str(intended_action[key]) for key in ("text", "value", "intent", "target")
             if intended_action.get(key)),
            "",
        )
        parameters = {"text": text}
        metadata: dict[str, Any] = {
            "tool_name": self.tool_name,
            "action_index": index,
            "action_space": "pyautogui",
            "coordinate_type": "relative",
            "raw_response": None,
            "model_call": False,
            "terminal_intent": action_type,
        }
        predicted = intended_action.get("predicted_effect_class")
        if predicted is not None:
            metadata["predicted_effect_class"] = predicted
        return Action(
            action_type="done",
            parameters=parameters,
            raw={
                "name": self.tool_name,
                "input": {"action": "done", **parameters},
                "raw_response": None,
            },
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _project_reasoning_message(self, reasoning_message) -> dict:
        """Slim handoff (Workflow/Implementation invariants): the action stage
        sees ONLY the structured intended_actions payload — never the reasoning
        trace, graph, plan, or retrieved experience."""
        if reasoning_message is None:
            return {}
        content = reasoning_message.content or {}
        intended = content.get("intended_actions")
        if not isinstance(intended, list):
            return {}
        return {"intended_actions": deepcopy(intended)}

    def _intended_actions(self, action_input: ActionInput) -> list:
        projected = self._project_reasoning_message(action_input.reasoning_message)
        return projected.get("intended_actions", [])
