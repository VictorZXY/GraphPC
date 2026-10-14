"""DeepSeek action agent for OSWorld 2.0 — grounds an intended action into the M3
``computer`` tool vocabulary with ``[0, 1000]`` relative coordinates through a NATIVE
forced function call, validated and translated by the M3 translator
(``m3_action_translator.translate_m3_action``, the adapter's call).

WHAT IS MINIMAX'S, UNCHANGED (``action_minimax.py``; this class subclasses
``MiniMaxActionAgent`` so the Action shapes it emits are byte for byte the ones the
translator and the adapter already read): the slim handoff (only ``intended_actions``
reach this stage), the intended-action line (``render_intended_action``; the
coordinates stripped first are this model's question, below), the terminal
classes emitted without a model call as ``done`` carrying ``terminal_intent``, the
coordinate-free bypass (``type``, ``key``, ``screenshot``, ``wait``) built from the
intent and guarded, the sparse-argument rule (unset optionals dropped), and the degrade
contract (an EMPTY batch naming the rejection; a counted WAIT at the adapter). What a
well-formed call with invalid arguments does is also MiniMax's: the translator's
``ValueError`` at the adapter, a consumed no-op step, never a repair.

WHAT IS THIS MODEL'S — THE CALL. DeepSeek ignores M3's text tool grammar and answers a
forced native function call reliably, so the grounding request is the backbone's
``complete_grounding``: non-thinking, temperature 0.6, the ``computer`` tool declared
as a function (``GROUNDING_TOOL``) and forced by ``tool_choice``, the current screen and
the intended-action line. The system prompt (``DEEPSEEK_GROUNDING_SYSTEM_PROMPT``) is
M3's ``<SYSTEM_CAPABILITY>`` coordinate bullet (taken off ``M3_GROUNDING_SYSTEM_PROMPT``,
not retyped) and a ``<TOOLS>`` block that names the function; the action list and the
argument shapes live in the tool definition, where the native format puts them.

THE TOOL DEFINITION IS SHARED WITH THE CONTROL (one prompt text on both arms, edited
only where compatibility differs). ``COMPUTER_TOOL_DESCRIPTION`` and
``COMPUTER_TOOL_SCHEMA`` are the sentence and the parameter schema the DeepSeek control
(upstream M3 with the native-tool deltas) declares; the arm declares
them WITHOUT ``done`` — ``COMPUTER_TOOL_DESCRIPTION_WITHOUT_DONE`` (the sentence without
its final ``done`` clause) and ``COMPUTER_TOOL_SCHEMA_WITHOUT_DONE`` (the enum without
``done``), both DERIVED from the shared constants — because termination is the deciding
call's and a grounding reply naming ``done`` / ``fail`` / ``terminate`` / ``call_user`` is
rejected (``action_minimax._REJECTED_GROUNDING_ACTIONS``), as on MiniMax. The control's
``<TOOLS>`` block carries the two termination rules; the arm's block carries neither.
Keep the two copies (the control's in
``scripts/osworld/deepseek_control/mm_agents/deepseek/prompts.py``) equal when editing
either, and the description and the enum in step with the translator's table
(``M3_ACTION_ARGUMENTS``), as ``M3_GROUNDING_SYSTEM_PROMPT`` is.

THE CLASS-FAMILY RULE (ON for this profile; adapted from
``action_qwen.py``, whose constants — the degrade marker, the batch metadata keys, the
repair text and the exhaustion line — are imported, not retyped, because the adapter
keys on exactly those strings). For a decided class in ``GROUNDED_CLASSES`` — every
class that reaches grounding on this route: the five clicks, ``left_click_drag``,
``mouse_move``, ``scroll``, ``hold_key``, ``left_mouse_down`` / ``left_mouse_up`` — a reply
whose action is outside the decided class's family (``CLASS_FAMILIES``: the click family
— the five clicks and ``left_click_drag`` — and every other grounded class alone) or acts
on nothing (``ALWAYS_REJECTED``: ``screenshot`` / ``wait``) is not run but REPAIRED:
the rejected call rendered as an assistant TEXT turn (``{"name":
"computer", "arguments": {...}}`` — never a ``tool_calls`` message, so the forced-function
request stays single-shot) and the user turn "{decided} was decided but the reply is
{got}; reply with a {decided}", after the original request; up to ``model_call_attempts``
attempts in all (3); exhausted, the batch degrades to ``grounding_class_mismatch`` (the
adapter's WAIT reason and line). A substitution within the click family executes, as on
Qwen: the repair enforces the class, not the location. The rule is checked LAST, after
the terminal-name rejection. The repair turn renders the call as the rule judged it, so a
call named after its action (below) is rendered with that action in its arguments. Every
batch carries the decided class, normalised (``decided_action_class``), and the repairs
the rule caused (``class_mismatch_repairs``), which the adapter keeps out of its grounding
sampling counters. No composite: M3's ``scroll`` and ``left_click_drag`` carry their own
coordinate.

THE ACTION IN THE FUNCTION NAME. DeepSeek sometimes names the forced call after the action
and leaves ``action`` out of the arguments (``left_click`` with ``{"coordinate": [516,
136]}``, named after its decided class), which would otherwise be rejected and re-sampled.
The backbone takes such a call as the ``computer`` call with that action when the name is one of
``GROUNDING_TOOL``'s ``action`` values (``backbone_deepseek.function_name_as_action``), and
everything above then applies to the action as to one written in the arguments: the
terminal-name rejection, the class-family rule, the translator's argument check at the
adapter. Visible at every level: the per-call record's ``action_from_function_name``, the
batch's count (``actions_from_function_name``, the episode's
``grounding_actions_from_function_name`` in the telemetry), the run's active_config
(``ACTION_FROM_FUNCTION_NAME``).

THE GROUNDING LINE CARRIES NO COORDINATE. The deciding call writes the point it computed
into the handoff in forms MiniMax's strip (``strip_grounding_coordinates``: bracketed
pairs and ``x=`` forms in ``target`` / ``intent``) leaves: unit-tagged prose — "(screen
~1494,470)", "(~px 1223,296)", "normalized 639,280 ≈ px 1227,302" — and the ``value``
field ("825,470", "[635,296]"), which that strip never reads and the line renders. The
non-thinking grounder copies them, so its precision gives way to the deciding call's
arithmetic. This profile's strip (``strip_deepseek_grounding_coordinates``)
reads ``target``, ``intent`` and ``value`` with MiniMax's patterns and three of its own
(``DEEPSEEK_COORDINATE_PATTERNS``), plus a ``value`` that is only integers
(``DEEPSEEK_VALUE_POINT``); ``text`` and ``keys`` are payloads (modifier keys, key names)
and are not read. A field the strip leaves with no letter or digit is dropped, except
that ``target`` / ``intent`` keep their text when neither would otherwise remain (the
audit record's ``fallback``). The audit record is MiniMax's shape, on the batch as on MiniMax
and on every per-call record of the step (``grounding_input_strip``); the run's
active_config carries the field and pattern sets (``GROUNDING_COORDINATE_STRIP``). What
the deciding call sees is untouched. A MARKED PAIR ("click ≈557,720", "image ≈680,725 →
click ≈358,725", "about 103,537"), which the grounder would otherwise copy as its point,
goes too: ``DEEPSEEK_MARKED_PAIR_PATTERNS`` run on ``target`` and
``intent`` after the patterns above and remove two integers (or integer ranges) of 2 to 4
digits that a marker chain stands right before — ≈, ~, about, approx(.), approximately,
around, near, at, click, point, coordinate(s), pixel(s), px, image, screen, x= / y= —
bare or filling a bracket. A pair with no marker before it ("960,400 (Week view …)") is
left, as are thousands separators, dates and times; ``value`` keeps its rule.

THE CLICK TEXT RULE. On a click-family action the translator turns ``text`` into key
presses: ``pyautogui.keyDown('<text>')`` per ``+``-token, unescaped. The deciding call can
write descriptions into ``text`` ("pigeon on the small building's roof ridge, image ≈725,652"),
the line renders them, and the grounder copies them into its reply: with an apostrophe the
guest's program is a SyntaxError and the click never runs (while the step is recorded as
executed), without one a bogus key is held. A ``text`` is
kept only when every ``+``-token, stripped and lower-cased, is a modifier
(``MODIFIER_KEYS``); otherwise it is dropped and the action kept — from the grounding line
on every grounded class but ``hold_key`` (whose ``text`` is the key it holds;
``LINE_TEXT_CLASSES``, ``left_click_drag`` and ``mouse_move`` among them, whose ``text``
the translator refuses), and from an accepted reply, after the terminal and class-family
rules, on the classes whose ``text`` becomes key presses (``REPLY_TEXT_CLASSES``).
All-or-nothing: "ctrl+click X" becomes a plain click. Each grounding call's record says
what was dropped (``click_text_dropped``: ``{"line", "reply"}``), each batch counts it.

THE GUEST'S PROGRAM IS CHECKED BEFORE IT IS HANDED OVER. The profile's translator is
``translate_deepseek_action``: the M3 translator, then the exact program the controller runs
(``PYAUTOGUI_PKGS_PREFIX`` with the command) compiled, and every statement of the command
held to the translator's shape — a call of one of its pyautogui verbs with literal
arguments, or ``left_press``'s ``import time as _t; _t.sleep(1)`` pair. A program that
fails is the translator's ``ValueError``: the adapter's counted WAIT (``action_parse_error``)
with its reason on the next request, never executed. It covers every action the
translator sees, the coordinate-free bypass included; ``DONE`` / ``FAIL`` / ``WAIT`` pass.

A GROUNDING STEP WITH NO USABLE REPLY (every rejection but the class-family rule's, after
the attempts) degrades as MiniMax's does, and its batch also carries
``grounding_reply_unusable`` (attempts, the last kind, the decided class): the adapter's
WAIT line is then ``GROUNDING_REPLY_UNUSABLE_LINE``, which names the decided action that did
not run and puts the fault on the grounding model's output — not the exception text, which
the deciding call reads as its own fault (after "GroundingReplyRejected: the tool call's
arguments are not JSON" it re-decides the same click).

EVERY OTHER REJECTION — no tool call, a function name that is not one of the tool's actions
or stands beside an explicit ``action``, arguments that are not a JSON object (after the
backbone's surplus-closer repair, ``backbone_deepseek.repair_surplus_closers``), a missing
``action`` on a call named ``computer``, a terminal / ask name, a reply cut at the output
cap, an interrupted or filtered reply, a 200 with no choices — is RE-SAMPLED with the
identical request within the same attempts (``action_minimax.py``'s semantics; the control's
parse-empty budget), each attempt on the per-call record with its rejection, then the
batch degrades to ``grounding_parse_error``. The ``[INFEASIBLE]`` token is not looked for:
under a forced function the content beside the tool call is normally empty, and the
grounding route has no termination authority either way — a token there would be
ignored and the tool call executed (MiniMax's text parser re-samples it instead).
"""

from __future__ import annotations

import ast
import json
import logging
import re
from copy import deepcopy
from typing import Any, Optional

from src.agents.common.schema import Action, ActionBatch

from .action_minimax import (
    _BYPASS_CLASSES,
    _COORDINATE_LEAD_IN,
    _COORDINATE_PATTERNS,
    _REJECTED_GROUNDING_ACTIONS,
    _TERMINAL_CLASSES,
    M3_ACTION_ARGUMENTS,
    M3_GROUNDING_SYSTEM_PROMPT,
    M3GroundingRejected,
    MiniMaxActionAgent,
    normalize_action_class,
    render_intended_action,
)
from .action_qwen import (
    CLASS_MISMATCH,
    CLASS_MISMATCH_KIND,
    CLASS_MISMATCH_LINE,
    CLASS_MISMATCH_REPAIR_TEXT,
    DECIDED_CLASS_KEY,
    GROUNDING_CLASS_CHECK_KEY,
    MISMATCH_REPAIRS_KEY,
)
from .backbone_deepseek import (
    ARGUMENTS_REPAIR_RULE,
    ARGUMENTS_REPAIRED_FIELD,
    FUNCTION_NAME_FIELD,
    GroundingReplyRejected,
    UNPARSEABLE,
)
from .m3_action_translator import is_unset_argument, translate_m3_action


logger = logging.getLogger("desktopenv.agent")

TOOL_NAME = "computer"

# The rejection kind of a grounding reply naming a terminal / ask action (re-sampled).
TERMINAL = "terminal"

# THE SHARED TOOL DEFINITION (module docstring): the control's, byte for byte.
# The action order is the sentence's.
COMPUTER_TOOL_ACTIONS: tuple[str, ...] = (
    "key", "type", "mouse_move", "left_click", "right_click", "middle_click", "double_click", "triple_click",
    "left_click_drag", "scroll", "wait", "screenshot", "hold_key", "left_mouse_down", "left_mouse_up",
)
_DONE_CLAUSE = "; `done`: declare the task has been completed successfully."
COMPUTER_TOOL_DESCRIPTION_WITHOUT_DONE = (
    "Interact with the desktop GUI. Supported `action` values: `key` (text: key combo, e.g. ctrl+s); `type` "
    "(text: text to type); `mouse_move` (coordinate); `left_click` / `right_click` / `middle_click` / "
    "`double_click` / `triple_click` (coordinate, optional text: modifier keys); `left_click_drag` (coordinate, "
    "optional start_coordinate); `scroll` (coordinate, scroll_direction, scroll_amount); `wait` (duration); "
    "`screenshot`; `hold_key` (text, duration); `left_mouse_down` / `left_mouse_up` (coordinate)."
)
# The control's sentence: the arm's with the `done` clause in place of its final full stop.
COMPUTER_TOOL_DESCRIPTION = COMPUTER_TOOL_DESCRIPTION_WITHOUT_DONE[:-1] + _DONE_CLAUSE

COORDINATE_DESCRIPTION = ("[x, y], normalized integers in [0, 1000]; (0, 0) top-left, (1000, 1000) bottom-right of "
                          "the screen.")
_COORDINATE_FIELD = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                     "description": COORDINATE_DESCRIPTION}
COMPUTER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [*COMPUTER_TOOL_ACTIONS, "done"]},
        "coordinate": deepcopy(_COORDINATE_FIELD),
        "start_coordinate": deepcopy(_COORDINATE_FIELD),
        "text": {"type": "string",
                 "description": "Key combo (key/hold_key), text to type (type), or modifier keys (clicks)."},
        "scroll_direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
        "scroll_amount": {"type": "integer"},
        "duration": {"type": "number"},
    },
    "required": ["action"],
}
COMPUTER_TOOL_SCHEMA_WITHOUT_DONE: dict[str, Any] = deepcopy(COMPUTER_TOOL_SCHEMA)
COMPUTER_TOOL_SCHEMA_WITHOUT_DONE["properties"]["action"]["enum"] = list(COMPUTER_TOOL_ACTIONS)

# What the arm declares (backbone_deepseek.OSWorldDeepSeekAgent.complete_grounding's ``tool``).
GROUNDING_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": COMPUTER_TOOL_DESCRIPTION_WITHOUT_DONE,
    "parameters": COMPUTER_TOOL_SCHEMA_WITHOUT_DONE,
}

# The arm's <TOOLS> block: the function named, the action list
# and argument shapes in the tool definition, no text-format rules, no termination rules.
GROUNDING_NATIVE_TOOLS_BLOCK = (
    "<TOOLS>\n"
    "You have access to the `computer` tool for interacting with the desktop GUI, provided as a function. "
    "Call it exactly once per turn.\n"
    "</TOOLS>"
)
# M3's <SYSTEM_CAPABILITY> coordinate bullet, taken off the MiniMax grounding prompt.
_SYSTEM_CAPABILITY_BLOCK = M3_GROUNDING_SYSTEM_PROMPT[: M3_GROUNDING_SYSTEM_PROMPT.index("\n\n<TOOLS>")]
DEEPSEEK_GROUNDING_SYSTEM_PROMPT = f"{_SYSTEM_CAPABILITY_BLOCK}\n\n{GROUNDING_NATIVE_TOOLS_BLOCK}"

# THE CLASS-FAMILY RULE'S TABLES, on M3's names (module docstring): every prompt-facing class that
# is neither bypassed nor terminal reaches grounding — `hold_key` among them, a family of its own (a
# keyboard action; any other class answering it is a mismatch), which the Qwen vocabulary does not have.
GROUNDED_CLASSES: tuple[str, ...] = tuple(
    name for name, _required, _optional in M3_ACTION_ARGUMENTS if name not in _BYPASS_CLASSES
)
_CLICK_FAMILY: frozenset[str] = frozenset({"left_click", "right_click", "middle_click", "double_click",
                                           "triple_click", "left_click_drag"})
CLASS_FAMILIES: dict[str, frozenset[str]] = {
    name: frozenset({name}) if name not in _CLICK_FAMILY else _CLICK_FAMILY for name in GROUNDED_CLASSES
}
ALWAYS_REJECTED: frozenset[str] = frozenset({"screenshot", "wait"})
# The rule's method surface as the run's active_config records it (the key action_qwen.py records
# its own under; the launch preflight holds the built agent to the pre-registration's copy).
GROUNDING_CLASS_CHECK: dict[str, Any] = {
    "families": {name: sorted(CLASS_FAMILIES[name]) for name in GROUNDED_CLASSES},
    "always_rejected": sorted(ALWAYS_REJECTED),
    "repair_text": CLASS_MISMATCH_REPAIR_TEXT,
    "exhaustion_line": CLASS_MISMATCH_LINE,
    "composite_classes": [],
}
# The action-in-the-function-name rule (module docstring) as the run's active_config records it: the names the
# backbone takes as an action are the declared tool's `action` values.
ACTION_FROM_FUNCTION_NAME_KEY = "grounding_action_from_function_name"
ACTION_FROM_FUNCTION_NAME: dict[str, Any] = {
    "names": list(GROUNDING_TOOL["parameters"]["properties"]["action"]["enum"]),
    "rule": "a grounding call named after one of these actions, with no `action` argument, is the `computer` call "
            "with that action; the terminal-name and class-family rules then apply to it",
}
# The batch metadata key counting the grounded actions taken from a function's name.
FUNCTION_NAME_ACTIONS_KEY = "actions_from_function_name"

# THE COORDINATE STRIP (module docstring), built from the forms the deciding call writes on this profile: every span
# it removes is a point, and a line with no point is unchanged. A number in the profile's own patterns has 2 to 4
# digits and is an integer, and a pair in prose needs a unit word (px, pixel(s), screen, normalized / normalised,
# norm) or must fill a parenthesis with its markers (~, ≈, approx.), so "~7 px", "10–25 px", "1920x1080", "17:01",
# "localhost:8080" and "12,500 rows" stay. The patterns run before MiniMax's, so a tagged pair goes with its tags, and
# match in any case ("(Screen ~1494,470)", "(PX 1223,296)"); the flag is inline, so the recorded pattern carries it.
_DS_NUMBER = r"(?<![\d.])\d{1,4}(?!\.?\d)"
_DS_TAGGED_NUMBER = r"(?<![\d.])\d{2,4}(?!\.?\d)"
_DS_BARE_PAIR = _DS_TAGGED_NUMBER + r"\s*,\s*" + _DS_TAGGED_NUMBER
_DS_PAIR = rf"(?:\(\s*{_DS_BARE_PAIR}\s*\)|\[\s*{_DS_BARE_PAIR}\s*\]|{_DS_BARE_PAIR})"
_DS_UNIT = r"\b(?:px|pixels?|screen|normali[sz]ed|norm)(?![A-Za-z])"
_DS_APPROX = r"(?:[~≈]|\bapprox\.?(?![A-Za-z]))"
_DS_MARK = rf"(?:{_DS_UNIT}|{_DS_APPROX})"
_DS_UNIT_AFTER = r"\s*(?:px|pixels?|normali[sz]ed|norm)(?![A-Za-z])"
_DS_TAGGED = rf"(?:(?:{_DS_MARK}\s*)+{_DS_PAIR}(?:{_DS_UNIT_AFTER})?|{_DS_PAIR}{_DS_UNIT_AFTER})"
_DS_MORE_TAGGED = rf"(?:\s*[,;≈=→]?\s*{_DS_TAGGED})*"
_DS_UNIT_TAGGED = (rf"(?:(?:{_DS_APPROX}\s*)*{_DS_UNIT}\s*(?:{_DS_MARK}\s*)*{_DS_PAIR}(?:{_DS_UNIT_AFTER})?"
                   rf"|(?:{_DS_MARK}\s*)*{_DS_PAIR}{_DS_UNIT_AFTER})")
_DS_KEYED = (r"""['"](?:start_)?coordinates?['"]\s*:\s*\[\s*""" + _DS_NUMBER + r"\s*,\s*" + _DS_NUMBER + r"\s*\]"
             + r"""|['"]x['"]\s*:\s*""" + _DS_NUMBER + r"""\s*,\s*['"]y['"]\s*:\s*""" + _DS_NUMBER)
DEEPSEEK_COORDINATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # A point keyed in a dict — "'coordinate': [500, 556]", '"start_coordinate": [636, 262]', "'x': 441, 'y': 625" —
    # with its trailing comma, or with the comma before it when it is the dict's last key.
    re.compile(rf"(?i)(?:{_DS_KEYED})\s*,\s*|\s*,\s*(?:{_DS_KEYED})(?=\s*[}}\]])|(?:{_DS_KEYED})"),
    # A parenthesis tagged pairs fill, with its lead-in: " (~px 1223,296)", " (normalized 639,280 ≈ px 1227,302)",
    # " (~659,575 px ≈ normalized)", " (approx. 740, 122)".
    re.compile(r"(?i)" + _COORDINATE_LEAD_IN + r"\s*\(\s*" + _DS_TAGGED + _DS_MORE_TAGGED
               + rf"(?:\s*[≈=→]\s*(?:{_DS_MARK}\s*)+)?" + r"\s*\)"),
    # A pair tagged with a unit word anywhere else: ", ~px (1222,296) → normalized (636,274)", " at screen ~835,615".
    re.compile(r"(?i),?" + _COORDINATE_LEAD_IN + r"\s*" + _DS_UNIT_TAGGED + _DS_MORE_TAGGED),
    *_COORDINATE_PATTERNS,
)
# A `value` that is two integers (a point) or four (a start and an end) and nothing else: "825,470", "648, 130",
# "[182, 146, 335, 342]". A value of "10,000" would read the same way; the `value` of a grounded intent does not carry
# a number of that shape in practice.
DEEPSEEK_VALUE_POINT = re.compile(r"^\s*[\[\(]?\s*" + _DS_NUMBER + r"\s*,\s*" + _DS_NUMBER + r"(?:\s*,\s*" + _DS_NUMBER
                                  + r"\s*,\s*" + _DS_NUMBER + r")?\s*[\]\)]?\s*$")
DEEPSEEK_STRIPPED_FIELDS: tuple[str, ...] = ("target", "intent", "value")
# THE MARKED PAIR (module docstring), chosen by measurement on recorded grounding lines: what it removes beyond the
# patterns above is a point or a screen region ("(1075–1090, 845)", "screen ≈165–240, 245–300"). A number has 2 to 4
# digits, does not start with 0 and is not part of a decimal, a time, a date or a word, so "10,000", "~1,500 rows",
# "$3,899.00", "20:00", "04/03/2026", "CX884, 02Dec2025" stay; a pair followed by a third number is a list ("7, 8, 18")
# and stays; the marker chain must stand right before the pair (white space, "=" / ":" and an opening bracket
# between), so "HKD 5,256" and an unmarked "960,400" stay. Run on target and intent only, after the patterns above.
_DS_MARK_WORD = (r"(?:[≈~]|\babout\b|\bapprox(?:imately\b|\.|\b)|\baround\b|\bnear\b|\bat\b|\bclick\b|\bpoint\b"
                 r"|\bcoordinates?\b|\bpixels?\b|\bpx\b|\bimage\b|\bscreen\b|\b[xy]\s*=)")
_DS_MARKED_NUMBER = r"(?<![\w.:/])[1-9]\d{1,3}(?![\w:/]|\.\d)"
_DS_MARKED_VALUE = rf"{_DS_MARKED_NUMBER}(?:\s*[–-]\s*{_DS_MARKED_NUMBER})?(?!\s*[–-]\s*\d)"
_DS_MARKED_PAIR = (rf"(?:[\(\[]\s*[~≈]?\s*{_DS_MARKED_VALUE}\s*,\s*[~≈]?\s*{_DS_MARKED_VALUE}\s*[\)\]]"
                   rf"|{_DS_MARKED_VALUE}\s*,\s*{_DS_MARKED_VALUE}(?!\s*,\s*\d))")
DEEPSEEK_MARKED_PAIR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?i),?\s*(?:{_DS_MARK_WORD}\s*[:=]?\s*)+{_DS_MARKED_PAIR}"),
)
DEEPSEEK_MARKED_PAIR_FIELDS: tuple[str, ...] = ("target", "intent")
# The strip as the run's active_config records it.
GROUNDING_COORDINATE_STRIP_KEY = "grounding_coordinate_strip"
GROUNDING_COORDINATE_STRIP: dict[str, Any] = {
    "fields": list(DEEPSEEK_STRIPPED_FIELDS),
    "patterns": [pattern.pattern for pattern in DEEPSEEK_COORDINATE_PATTERNS],
    "value_point_pattern": DEEPSEEK_VALUE_POINT.pattern,
    "marked_pair_fields": list(DEEPSEEK_MARKED_PAIR_FIELDS),
    "marked_pair_patterns": [pattern.pattern for pattern in DEEPSEEK_MARKED_PAIR_PATTERNS],
    "emptied_field": "dropped; an emptied target or intent keeps its text when neither would otherwise remain",
}
# The per-call record's copy of the step's audit record.
STRIP_RECORD_FIELD = "grounding_input_strip"


def strip_deepseek_grounding_coordinates(intended_action: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(intended action for the grounding line, audit record)`` on this profile (module docstring): every
    point removed from ``target``, ``intent`` and ``value`` (a marked pair from ``target`` and ``intent``); a field
    left with no letter or digit dropped, unless neither ``target`` nor ``intent`` would remain, when the emptied
    ones keep their text (``fallback``). The record is MiniMax's: ``stripped``, ``removed`` ({field: [spans]}),
    ``fallback`` ([fields])."""
    cleaned = dict(intended_action)
    removed: dict[str, list[str]] = {}
    emptied: list[str] = []
    for field in DEEPSEEK_STRIPPED_FIELDS:
        text = intended_action.get(field)
        if not isinstance(text, str):
            continue
        spans: list[str] = []
        for pattern in (*DEEPSEEK_COORDINATE_PATTERNS, *((DEEPSEEK_VALUE_POINT,) if field == "value" else ()),
                        *(DEEPSEEK_MARKED_PAIR_PATTERNS if field in DEEPSEEK_MARKED_PAIR_FIELDS else ())):
            spans.extend(match.group(0) for match in pattern.finditer(text))
            text = pattern.sub("", text)
        if not spans:
            continue
        removed[field] = spans
        if any(character.isalnum() for character in text):
            cleaned[field] = text
        else:
            cleaned[field] = None
            emptied.append(field)
    fallback: list[str] = []
    if not any(isinstance(cleaned.get(field), str) and cleaned[field].strip() for field in ("target", "intent")):
        fallback = [field for field in ("target", "intent") if field in emptied]
        for field in fallback:
            cleaned[field] = intended_action[field]
            del removed[field]
    return cleaned, {"stripped": bool(removed), "removed": removed, "fallback": fallback}


# THE CLICK TEXT RULE (module docstring). The modifiers are written out here (the click branch of the translator
# applies no alias: only .strip().lower()); the line classes are every grounded class but hold_key, the reply classes
# those whose `text` the translator presses as keys.
MODIFIER_KEYS: frozenset[str] = frozenset({"ctrl", "control", "shift", "alt", "super", "cmd", "command", "win", "meta"})
LINE_TEXT_CLASSES: tuple[str, ...] = tuple(name for name in GROUNDED_CLASSES if name != "hold_key")
REPLY_TEXT_CLASSES: tuple[str, ...] = ("left_click", "right_click", "middle_click", "double_click", "triple_click",
                                       "left_mouse_down", "left_mouse_up", "scroll")
CLICK_TEXT_RULE_KEY = "grounding_click_text_rule"
CLICK_TEXT_RULE: dict[str, Any] = {
    "modifiers": sorted(MODIFIER_KEYS),
    "line_classes": list(LINE_TEXT_CLASSES),
    "reply_classes": list(REPLY_TEXT_CLASSES),
    "rule": "a `text` is kept only when every `+`-separated token, stripped and lower-cased, is one of the modifiers; "
            "otherwise it is dropped and the action kept: from the grounding line on the line classes, from an "
            "accepted reply on the reply classes",
}
# The per-call record's {"line": the text dropped from the step's grounding line, "reply": the text dropped from
# this reply} (each None when nothing was), and the batch metadata keys counting them.
CLICK_TEXT_DROPPED_FIELD = "click_text_dropped"
LINE_TEXTS_DROPPED_KEY = "line_texts_dropped"
REPLY_TEXTS_DROPPED_KEY = "reply_texts_dropped"
# The argument repair (backbone_deepseek.repair_surplus_closers) as the run's active_config records it, and the batch
# metadata key counting the accepted replies it repaired.
ARGUMENTS_REPAIR_KEY = "grounding_arguments_repair"
ARGUMENTS_REPAIR: dict[str, Any] = {"rule": ARGUMENTS_REPAIR_RULE}
REPLIES_REPAIRED_KEY = "replies_repaired"
# THE UNUSABLE-REPLY LINE (module docstring): decider-visible method surface, beside the class-family rule's
# exhaustion line. The batch content key the adapter renders it from.
GROUNDING_REPLY_UNUSABLE = "grounding_reply_unusable"
GROUNDING_REPLY_UNUSABLE_LINE = ("the decided {decided} was not executed: none of the {n} grounding replies was a usable "
                                 "tool call (a fault in the grounding model's output, not in your decision)")
GROUNDING_REPLY_UNUSABLE_LINE_KEY = "grounding_reply_unusable_line"


def is_modifier_combination(text: str) -> bool:
    """Whether every ``+``-separated token of ``text``, stripped and lower-cased, is in ``MODIFIER_KEYS``."""
    return all(token.strip().lower() in MODIFIER_KEYS for token in text.split("+"))


def dropped_text(value: Any) -> Any:
    """The ``text`` the click text rule drops: ``value`` itself when it is set and not a modifier combination; None
    when it is kept (a modifier combination) or unset (null, "", []: dropped as every unset argument is)."""
    if is_unset_argument(value) or (isinstance(value, str) and is_modifier_combination(value)):
        return None
    return value


def drop_line_text(intended_action: dict[str, Any], action_class: str) -> tuple[dict[str, Any], Any]:
    """``(the intended action for the grounding line, the text dropped from it or None)``: on a line class, a
    ``text`` that is not a modifier combination is left off the line (module docstring)."""
    text = dropped_text(intended_action.get("text")) if action_class in LINE_TEXT_CLASSES else None
    if text is None:
        return intended_action, None
    return {**intended_action, "text": None}, text


# THE GUEST'S PROGRAM (module docstring). OSWorld-V2's controller runs a pyautogui command as
# `python -c PYAUTOGUI_PKGS_PREFIX.format(command=...)`; the prefix is copied verbatim from OSWorld-V2
# desktop_env/controllers/python.py (tag v2026.08.08, commit d578d2d), PythonController's default pkgs_prefix.
PYAUTOGUI_PKGS_PREFIX = (
    "import pyautogui; import time; import platform; "
    "pyautogui.FAILSAFE = False; "
    "_osworld_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:\"<>?'; "
    "_osworld_linux_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:\">?'; "
    "pyautogui.isShiftCharacter = lambda character: character.isupper() or "
    "character in (_osworld_linux_shift_chars if platform.system() == 'Linux' else _osworld_shift_chars); "
    "{command}"
)
# The pyautogui verbs the M3 translator emits (m3_action_translator.tool_action_to_pyautogui), the tokens it hands the
# driver as they are, and left_press's pause: the one statement pair of its own the translator writes.
GROUNDED_PROGRAM_VERBS: frozenset[str] = frozenset({
    "click", "rightClick", "doubleClick", "middleClick", "tripleClick", "mouseDown", "mouseUp", "moveTo", "dragTo",
    "keyDown", "keyUp", "press", "scroll", "hscroll", "sleep",
})
_PASSED_THROUGH_TOKENS: frozenset[str] = frozenset({"DONE", "FAIL", "WAIT"})
_LEFT_PRESS_PAUSE: tuple[str, ...] = tuple(ast.dump(statement)
                                           for statement in ast.parse("import time as _t; _t.sleep(1)").body)
GROUNDED_PROGRAM_CHECK_KEY = "grounded_program_check"
GROUNDED_PROGRAM_CHECK: dict[str, Any] = {
    "prefix": PYAUTOGUI_PKGS_PREFIX,
    "verbs": sorted(GROUNDED_PROGRAM_VERBS),
    "passed_through": sorted(_PASSED_THROUGH_TOKENS),
    "rule": "the translated command, after the controller's prefix, must compile, and each statement of the command "
            "must be a call of one of these pyautogui verbs with literal arguments, or left_press's "
            "`import time as _t; _t.sleep(1)`; otherwise it is the translator's ValueError, a counted WAIT "
            "(action_parse_error), never executed",
}


def _is_translator_call(statement: ast.stmt) -> bool:
    """Whether ``statement`` is ``pyautogui.<verb>(<literals>)`` for one of ``GROUNDED_PROGRAM_VERBS``, every
    argument and keyword value a literal (``ast.literal_eval``: a negative scroll amount is one)."""
    call = statement.value if isinstance(statement, ast.Expr) else None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "pyautogui"
            and call.func.attr in GROUNDED_PROGRAM_VERBS and all(keyword.arg for keyword in call.keywords)):
        return False
    try:
        for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
            ast.literal_eval(argument)
    except (TypeError, ValueError):
        return False
    return True


def grounded_program_defect(command: str) -> Optional[str]:
    """Why the guest cannot run ``command`` as the translator writes it, or None (module docstring): the controller's
    whole program must compile, and the command's statements must be the translator's shape."""
    try:
        compile(PYAUTOGUI_PKGS_PREFIX.format(command=command), "<grounded action>", "exec")
        statements = ast.parse(command).body
    except (SyntaxError, ValueError) as exc:
        return f"the program does not compile ({exc.msg if isinstance(exc, SyntaxError) else exc})"
    index = 0
    while index < len(statements):
        pair = tuple(ast.dump(statement) for statement in statements[index:index + 2])
        if pair == _LEFT_PRESS_PAUSE:
            index += 2
            continue
        if not _is_translator_call(statements[index]):
            return f"statement {index + 1} is not a pyautogui call with literal arguments"
        index += 1
    return None


def translate_deepseek_action(action: Action, *, screen_size: tuple[int, int]) -> str:
    """The profile's translator: ``translate_m3_action``, then the guest's program checked (module docstring); a
    program the guest cannot run is a ``ValueError``, as every untranslatable action is."""
    command = translate_m3_action(action, screen_size=screen_size)
    if command in _PASSED_THROUGH_TOKENS:
        return command
    defect = grounded_program_defect(command)
    if defect is not None:
        raise ValueError(f"the translated command cannot run in the guest: {defect}")
    return command


def is_class_mismatch(decided: str | None, got: str) -> bool:
    """Whether a reply of class ``got`` is sent back for the decided class: for a decided class in
    ``GROUNDED_CLASSES``, a reply that acts on nothing or is outside its family; never for a
    decided class outside the list (an off-vocabulary class keeps the first-call rule)."""
    if decided not in GROUNDED_CLASSES:
        return False
    return got in ALWAYS_REJECTED or got not in CLASS_FAMILIES[decided]


class GroundingClassMismatch(GroundingReplyRejected):
    """A grounding reply whose call the class-family rule sends back; the message is the text
    the repair turn quotes, ``arguments`` the call's arguments as the rule judged them."""

    def __init__(self, decided: str, got: str, arguments: dict[str, Any]) -> None:
        super().__init__(CLASS_MISMATCH_REPAIR_TEXT.format(decided=decided, got=got), kind=CLASS_MISMATCH_KIND)
        self.decided, self.got, self.arguments = decided, got, arguments


def repair_turns(arguments: dict[str, Any], repair: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The two turns a class-mismatch repair appends after the original request (module
    docstring): the rejected call as an assistant TEXT message, then the user repair."""
    rendered = json.dumps({"name": TOOL_NAME, "arguments": arguments}, ensure_ascii=False)
    return ({"role": "assistant", "content": rendered}, {"role": "user", "content": repair})


class DeepSeekActionAgent(MiniMaxActionAgent):
    """``MiniMaxActionAgent`` with the native grounding call and the class-family rule (module
    docstring). ``model_call_attempts`` is the TOTAL grounding attempts per intended action (3 on
    this benchmark, the control's budget); ``method_surface``: the keys this agent adds to the
    run's active_config (``build_default``)."""

    method_surface: dict[str, Any] = {GROUNDING_CLASS_CHECK_KEY: GROUNDING_CLASS_CHECK,
                                      ACTION_FROM_FUNCTION_NAME_KEY: ACTION_FROM_FUNCTION_NAME,
                                      GROUNDING_COORDINATE_STRIP_KEY: GROUNDING_COORDINATE_STRIP,
                                      ARGUMENTS_REPAIR_KEY: ARGUMENTS_REPAIR,
                                      CLICK_TEXT_RULE_KEY: CLICK_TEXT_RULE,
                                      GROUNDING_REPLY_UNUSABLE_LINE_KEY: GROUNDING_REPLY_UNUSABLE_LINE,
                                      GROUNDED_PROGRAM_CHECK_KEY: GROUNDED_PROGRAM_CHECK}

    # The constructor is MiniMaxActionAgent's (backbone, tool_name="computer", model_call_attempts=1), inherited:
    # an override forwarding `model_call_attempts` would be a second call site passing a benchmark-scoped value
    # outside the factory (build_default), the one place those values are set.

    def act(self, action_input):
        """Every batch, degraded or empty included, carries the decided class, normalised (None
        without an intent), the class-family rule's repairs, the actions taken from a
        function's name, the repaired replies and the texts the click text rule dropped."""
        batch = self._act(action_input)
        intended = self._intended_actions(action_input)
        first = intended[0] if intended and isinstance(intended[0], dict) else {}
        batch.metadata[DECIDED_CLASS_KEY] = normalize_action_class(first.get("action_class")) or None
        for key in (MISMATCH_REPAIRS_KEY, FUNCTION_NAME_ACTIONS_KEY, REPLIES_REPAIRED_KEY, LINE_TEXTS_DROPPED_KEY,
                    REPLY_TEXTS_DROPPED_KEY):
            batch.metadata.setdefault(key, 0)
        return batch

    def _act(self, action_input) -> ActionBatch:
        intended = self._intended_actions(action_input)
        if not intended:
            return ActionBatch(actions=[], content={}, raw={})
        actions = []
        raw_responses: list[Any] = []
        resamples_total = 0
        mismatch_repairs_total = 0
        function_name_actions = 0
        replies_repaired = 0
        line_texts_dropped = 0
        reply_texts_dropped = 0
        strip_records: list[dict[str, Any]] = []
        for index, intended_action in enumerate(intended[: action_input.max_actions]):
            intended_action = intended_action or {}
            raw_action_class = str(intended_action.get("action_class") or "")
            action_class = normalize_action_class(raw_action_class)
            if action_class in _TERMINAL_CLASSES:
                actions.append(self._terminal_action(action_type=action_class, intended_action=intended_action,
                                                     index=index))
                continue
            if action_class in _BYPASS_CLASSES:
                try:
                    actions.append(self._validated_bypass_action(
                        action_name=action_class, raw_action_class=raw_action_class, intended_action=intended_action,
                        index=index,
                    ))
                except M3GroundingRejected as exc:
                    logger.warning("bypassed intent cannot be executed: %s; degrading to empty batch", exc)
                    return self._rejected_batch(exc, resamples_total=resamples_total)
                continue
            grounding_intent, line_text = drop_line_text(intended_action, action_class)
            line_texts_dropped += line_text is not None
            grounding_intent, strip_record = strip_deepseek_grounding_coordinates(grounding_intent)
            strip_records.append(strip_record)
            try:
                result, resamples, mismatch_repairs = self._ground(
                    user_prompt=render_intended_action(grounding_intent), observation=action_input.observation,
                    decided_class=action_class, strip_record=strip_record, line_text=line_text,
                )
            except GroundingReplyRejected as exc:
                logger.warning("grounding reply rejected (%s): %s; degrading to empty batch", exc.kind, exc)
                batch = self._degraded(exc, resamples=resamples_total + exc.resamples, decided=action_class)
                batch.metadata["grounding_input_strip"] = strip_records
                batch.metadata[MISMATCH_REPAIRS_KEY] = mismatch_repairs_total + exc.mismatch_repairs
                batch.metadata[LINE_TEXTS_DROPPED_KEY] = line_texts_dropped
                return batch
            resamples_total += resamples
            mismatch_repairs_total += mismatch_repairs
            function_name_actions += bool(result.get(FUNCTION_NAME_FIELD))
            replies_repaired += result.get(ARGUMENTS_REPAIRED_FIELD) is not None
            reply_texts_dropped += (result.get(CLICK_TEXT_DROPPED_FIELD) or {}).get("reply") is not None
            action_name, arguments = result["parsed"]
            raw_responses.append(result["text"])
            actions.append(self._grounded_action(
                action_name=action_name, arguments=arguments, raw_response=result["text"],
                intended_action=intended_action, index=index,
            ))
        metadata: dict[str, Any] = {"resample_attempts": resamples_total, MISMATCH_REPAIRS_KEY: mismatch_repairs_total,
                                    FUNCTION_NAME_ACTIONS_KEY: function_name_actions,
                                    REPLIES_REPAIRED_KEY: replies_repaired, LINE_TEXTS_DROPPED_KEY: line_texts_dropped,
                                    REPLY_TEXTS_DROPPED_KEY: reply_texts_dropped}
        if strip_records:
            metadata["grounding_input_strip"] = strip_records
        return ActionBatch(
            actions=actions,
            # The native route carries no `Action:` line: nothing to show as the grounding text.
            content={"reasoning_text": ""},
            raw={"deepseek_responses": raw_responses},
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _ground(self, *, user_prompt: str, observation: Any, decided_class: str, strip_record: dict[str, Any],
                line_text: Any = None):
        """``(result, resamples, mismatch_repairs)``: the grounding call, re-sampled identically on
        any rejection but a class mismatch, which is repaired (module docstring); or
        ``GroundingReplyRejected`` (carrying ``resamples`` and ``mismatch_repairs``) when the
        attempts run out. Every attempt's per-call record carries the step's ``strip_record``
        and the text dropped from its line (``line_text``)."""
        turns: tuple[dict[str, Any], ...] = ()
        mismatch_repairs = 0
        for attempt in range(1, self.model_call_attempts + 1):
            try:
                result = self.backbone.complete_grounding(
                    system_prompt=DEEPSEEK_GROUNDING_SYSTEM_PROMPT, user_prompt=user_prompt, observation=observation,
                    tool=GROUNDING_TOOL, turns=turns,
                    validate=lambda arguments: self._parse(arguments, decided_class, line_text),
                    facts={STRIP_RECORD_FIELD: strip_record,
                           CLICK_TEXT_DROPPED_FIELD: {"line": line_text, "reply": None}},
                )
                return result, attempt - 1, mismatch_repairs
            except GroundingReplyRejected as exc:
                exc.resamples, exc.mismatch_repairs = attempt - 1, mismatch_repairs
                if attempt == self.model_call_attempts:
                    raise
                if exc.kind == CLASS_MISMATCH_KIND:
                    mismatch_repairs += 1
                    logger.warning("repairing grounding reply (attempt %d): %s", attempt, exc)
                    turns = repair_turns(exc.arguments, str(exc))
                else:
                    logger.warning("grounding reply rejected (attempt %d, %s): %s; re-sampling", attempt, exc.kind,
                                   exc)
                    turns = ()
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse(arguments: dict[str, Any], decided_class: str | None,
               line_text: Any = None) -> tuple[tuple[str, dict[str, Any]], dict[str, Any]]:
        """``((the action name, its arguments), facts)`` for the first tool call's arguments, or
        ``GroundingReplyRejected``: a missing ``action`` (``unparseable``), a terminal / ask name
        (``terminal``), a class outside the decided class's family (``class_mismatch``). On an
        accepted reply of a reply class, a ``text`` that is not a modifier combination is
        dropped (the click text rule); the facts say what was dropped from the line
        (``line_text``) and from the reply. The arguments' validity is the translator's
        question, at the adapter."""
        action = arguments.get("action")
        if not isinstance(action, str) or not action.strip():
            raise GroundingReplyRejected("the tool call carries no `action`", kind=UNPARSEABLE)
        action_name = action.strip().lower()
        if action_name in _REJECTED_GROUNDING_ACTIONS:
            raise GroundingReplyRejected(f"the grounding reply emitted the terminal/ask action {action_name!r}",
                                         kind=TERMINAL)
        if is_class_mismatch(decided_class, action_name):
            raise GroundingClassMismatch(decided_class, action_name, arguments)
        arguments = {**arguments, "action": action_name}
        reply_text = dropped_text(arguments.get("text")) if action_name in REPLY_TEXT_CLASSES else None
        if reply_text is not None:
            del arguments["text"]
        return (action_name, arguments), {CLICK_TEXT_DROPPED_FIELD: {"line": line_text, "reply": reply_text}}

    @staticmethod
    def _degraded(exc: GroundingReplyRejected, *, resamples: int, decided: str) -> ActionBatch:
        """The degrade contract, as MiniMax's: an EMPTY batch naming the rejection. A last reply
        that failed the class-family rule is named apart, with what the adapter's WAIT line
        states: the attempts made, the last reply's class and the decided one. Any other
        exhaustion also carries what the unusable-reply line states (``GROUNDING_REPLY_UNUSABLE``):
        the attempts, the last reply's rejection kind and the decided class."""
        if isinstance(exc, GroundingClassMismatch):
            return ActionBatch(actions=[], content={CLASS_MISMATCH: {"attempts": exc.resamples + 1, "got": exc.got,
                                                                     "decided": exc.decided}},
                               raw={}, metadata={"resample_attempts": resamples})
        return ActionBatch(actions=[], content={"grounding_parse_error": f"{type(exc).__name__}: {exc}",
                                                GROUNDING_REPLY_UNUSABLE: {"attempts": exc.resamples + 1,
                                                                           "kind": exc.kind, "decided": decided}},
                           raw={}, metadata={"resample_attempts": resamples})
