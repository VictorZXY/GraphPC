"""OSWorld-V2 M3 ``computer`` tool grounding-response parsing and action ->
pyautogui translation for the GraphPC OSWorld adapter (benchmark-side;
copy-first rule, AGENTS.md).

PROVENANCE
  Source:  OSWorld-V2/mm_agents/m3/parser.py
  Tag:     v2026.08.08  (commit d578d2d)
  Copied VERBATIM:
    * ``tool_action_to_pyautogui`` (source lines 72-252) — the single-action
      translator V2's own M3 agent runs every emitted tool call through;
    * ``_VALID_ACTIONS`` (lines 34-45) — the accepted action names the
      BPE-artifact recovery consults;
    * the ``coordinate_type="relative"`` branch of the ``scaled_xy`` closure
      inside ``parse_m3_response`` (``int(x * original_width / 1000),
      int(y * original_height / 1000)``), lifted into ``_relative_scaler``.
  Copied WITH DOCUMENTED EDITS — ``parse_m3_response`` (lines 255-392) as
  ``parse_m3_tool_calls``; everything not listed here is verbatim:
    E1 signature ``(response) -> (low_level_instruction, [(action, args)])``
       and docstring (the width/height/coordinate_type parameters are gone:
       scaling happens in ``translate_m3_action``);
    E2 the ``[INFEASIBLE]`` short-circuit raises ``M3GroundingRejected``
       instead of returning ``["FAIL"]`` (a grounding call may not end the
       episode);
    E3 the ``scaled_xy`` closure is removed (see E1);
    E4 the accumulator is ``tool_calls`` instead of ``pyautogui_code``;
    E5 in ``emit``, everything after the BPE recovery — the truncated/short-
       action no-op, the ``tool_action_to_pyautogui`` call, the invalid-
       action no-op, the code join — is replaced by
       ``tool_calls.append((action, args))``: the ValueError-only argument contract and
       the terminal-name rejection decide what happens to the call, loudly,
       instead of V2's silent ``pyautogui.sleep(0.1)`` downgrade;
    E6 the return value follows E4;
    E7 two inserted isinstance guards (a non-object tool call before the
       name check, non-object arguments before the action lookup) so model
       text can never raise AttributeError — the copied code assumed dicts.
  Not copied: ``wrap_for_history`` (chat-history rendering; we send no
  history) and ``_NORMALIZE_KEY`` (unused upstream too).

Public API
  ``parse_m3_tool_calls(response) -> (low_level_instruction, [(action, args)])``
    The ``Action:`` line and every ``computer`` tool call in the grounding
    text, BPE-repaired, in order; raises ``M3GroundingRejected`` on the
    ``[INFEASIBLE]`` token. The action agent takes the first call and
    rejects terminal / ask names itself.
  ``translate_m3_action(action, *, screen_size) -> str``
    * pyautogui source (multi-statement code joined with ``"\n"`` so
      ``desktop_env`` runs the whole action atomically — V2's ``emit`` does the
      same join) for an executable action;
    * the literal ``"DONE"`` / ``"FAIL"`` for the terminal actions the
      REASONING call decided (the action agent emits those without a model
      call; a grounding-emitted done/fail never becomes an Action);
    * ``ValueError`` — and ONLY ``ValueError`` — for any defect, so the adapter
      can degrade to a counted ``"WAIT"`` step: an action outside the
      vocabulary (incl. ``call_user`` / ``terminate`` — there is no ask-user
      channel and grounding may not end the episode), a missing required
      argument (a click / move / drag without a coordinate, ``key`` /
      ``type`` / ``hold_key`` without text, ``scroll`` without direction +
      amount), a wrong-typed argument (a coordinate that is not a pair of
      numbers, a non-integer scroll amount, ...), or an action that
      translates to no code at all (V2's scanner drops that silently, which
      would read as ASK_USER on the driver). The copied function itself
      indexes coordinates before validating and would raise ``IndexError``
      / ``TypeError`` on the same inputs, and clicks with no coordinate at
      wherever the pointer happens to be — the wrapper's argument contract
      runs first so neither can happen.
  ``M3GroundingRejected(ValueError)``
    Raised by the parse layer for a grounding response that must not become
    an action; the action agent degrades on exactly this class.
  ``is_unset_argument(value)``
    The one definition of "the model did not use this argument": null, an
    empty string, an empty list (a required-but-unused key can come back as
    any of the three). The action agent drops such keys; the wrapper drops
    them again so the contract never depends on the caller.
  ``pointer_xy_from_pyautogui(command) -> (x, y) | None``
    Where the emitted source leaves the mouse pointer, in screen pixels;
    ``None`` when it moves no pointer. The inverse of the emission above, and
    it lives here so the two cannot disagree about which verbs carry a
    coordinate or where in the argument list it sits. Read by the monitor's
    pointer mask (see graphpc_agent.pointer_positions_for_mask).
"""

from __future__ import annotations

import logging
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agents.common.schema import Action

logger = logging.getLogger("desktopenv.agent.m3.parser")


class M3GroundingRejected(ValueError):
    """A grounding response that must not become an action (see module docstring)."""


# --- BEGIN verbatim copy: OSWorld-V2/mm_agents/m3/parser.py::_VALID_ACTIONS ---
_VALID_ACTIONS = frozenset({
    "click", "left click", "right click",
    "left_click", "right_click", "double_click", "middle_click",
    "left_press", "triple_click",
    "left_mouse_down", "left_mouse_up",
    "mouse_move", "left_click_drag",
    "hold_key", "key", "type",
    "scroll",
    "wait", "screenshot", "fail", "done", "call_user",
    "terminate", "cursor_position",
    "zoom", "zoom_in",
})
# --- END verbatim copy ---


# --- BEGIN verbatim copy: OSWorld-V2/mm_agents/m3/parser.py::tool_action_to_pyautogui ---
def tool_action_to_pyautogui(action: str, args: Dict, scaled_xy) -> List[str]:
    """Translate one computer-use tool action to pyautogui code.

    ``scaled_xy(x, y)`` is a callback that maps the model's emitted
    (x, y) to actual screen pixels.

    Returns a list of code strings; special tokens (``"WAIT"``, ``"DONE"``,
    ``"FAIL"``, ``"CALL_USER"``) are returned as their own one-element list
    so the caller can keep them as standalone strings.
    """
    code: List[str] = []

    action_conversion = {"left click": "click", "right click": "right_click"}
    action = action_conversion.get(action, action)
    if action == "click":
        action = "left_click"

    text = args.get("text")
    coordinate = args.get("coordinate")
    start_coordinate = args.get("start_coordinate")
    scroll_direction = args.get("scroll_direction")
    scroll_amount = args.get("scroll_amount")
    duration = args.get("duration")

    if coordinate is not None:
        coordinate = list(scaled_xy(coordinate[0], coordinate[1]))
    if start_coordinate is not None:
        start_coordinate = list(scaled_xy(start_coordinate[0], start_coordinate[1]))

    if action in ("left_mouse_down", "left_mouse_up"):
        if text:
            for k in text.split('+'):
                code.append(f"pyautogui.keyDown('{k.strip().lower()}')")
        if coordinate is not None:
            x, y = coordinate
            code.append(
                f"pyautogui.mouseDown({x}, {y})" if action == "left_mouse_down"
                else f"pyautogui.mouseUp({x}, {y})"
            )
        else:
            code.append(
                "pyautogui.mouseDown()" if action == "left_mouse_down"
                else "pyautogui.mouseUp()"
            )
        if text:
            for k in reversed(text.split('+')):
                code.append(f"pyautogui.keyUp('{k.strip().lower()}')")
    elif action == "hold_key":
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        for key in text.split('+'):
            code.append(f"pyautogui.keyDown('{key.strip().lower()}')")
    elif action in ("mouse_move", "left_click_drag"):
        if coordinate is None:
            raise ValueError(f"coordinate is required for {action}")
        if text is not None:
            raise ValueError(f"text is not accepted for {action}")
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError(f"{coordinate} must be a tuple of length 2")
        if not all(isinstance(i, int) for i in coordinate):
            raise ValueError(f"{coordinate} must be a tuple of ints")
        x, y = coordinate[0], coordinate[1]
        if action == "mouse_move":
            code.append(f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})")
        else:
            if start_coordinate:
                if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                    raise ValueError(f"{start_coordinate} must be a tuple of length 2")
                if not all(isinstance(i, int) for i in start_coordinate):
                    raise ValueError(f"{start_coordinate} must be a tuple of ints")
                sx, sy = start_coordinate
                code.append(f"pyautogui.moveTo({sx}, {sy}, duration={duration or 0.5})")
            code.append(f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})")
    elif action in ("key", "type"):
        if text is None:
            raise ValueError(f"text is required for {action}")
        if coordinate is not None:
            raise ValueError(f"coordinate is not accepted for {action}")
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        if action == "key":
            key_conversion = {
                "page_down": "pagedown",
                "page_up": "pageup",
                "super_l": "win",
                "super": "command",
                "escape": "esc",
            }
            keys = [key_conversion.get(k.strip().lower(), k.strip().lower()) for k in text.split('+')]
            for k in keys:
                code.append(f"pyautogui.keyDown('{k}')")
            for k in reversed(keys):
                code.append(f"pyautogui.keyUp('{k}')")
        else:
            for char in text:
                if char == '\n':
                    code.append("pyautogui.press('enter')")
                elif char == "'":
                    code.append('pyautogui.press("\'")')
                elif char == '\\':
                    code.append("pyautogui.press('\\\\')")
                elif char == '"':
                    code.append("pyautogui.press('\"')")
                else:
                    code.append(f"pyautogui.press('{char}')")
    elif action == "scroll":
        if text is not None:
            for k in text.split('+'):
                code.append(f"pyautogui.keyDown('{k.strip().lower()}')")
        if coordinate is None:
            if scroll_direction in ("up", "down"):
                code.append(f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount})")
            elif scroll_direction in ("left", "right"):
                code.append(f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount})")
        else:
            x, y = coordinate[0], coordinate[1]
            if scroll_direction in ("up", "down"):
                code.append(f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount}, {x}, {y})")
            elif scroll_direction in ("left", "right"):
                code.append(f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount}, {x}, {y})")
        if text is not None:
            for k in reversed(text.split('+')):
                code.append(f"pyautogui.keyUp('{k.strip().lower()}')")
    elif action in ("left_click", "right_click", "double_click", "middle_click", "left_press", "triple_click"):
        if text:
            for k in text.split('+'):
                code.append(f"pyautogui.keyDown('{k.strip().lower()}')")
        if coordinate is not None:
            x, y = coordinate
            mapping = {
                "left_click":   f"pyautogui.click({x}, {y})",
                "right_click":  f"pyautogui.rightClick({x}, {y})",
                "double_click": f"pyautogui.doubleClick({x}, {y})",
                "middle_click": f"pyautogui.middleClick({x}, {y})",
                "triple_click": f"pyautogui.tripleClick({x}, {y})",
            }
            if action == "left_press":
                code.append(f"pyautogui.mouseDown({x}, {y})")
                code.append("import time as _t; _t.sleep(1)")
                code.append(f"pyautogui.mouseUp({x}, {y})")
            else:
                code.append(mapping[action])
        else:
            mapping_no_xy = {
                "left_click":   "pyautogui.click()",
                "right_click":  "pyautogui.rightClick()",
                "double_click": "pyautogui.doubleClick()",
                "middle_click": "pyautogui.middleClick()",
                "triple_click": "pyautogui.tripleClick()",
            }
            if action == "left_press":
                code.append("pyautogui.mouseDown()")
                code.append("import time as _t; _t.sleep(1)")
                code.append("pyautogui.mouseUp()")
            else:
                code.append(mapping_no_xy[action])
        if text:
            for k in reversed(text.split('+')):
                code.append(f"pyautogui.keyUp('{k.strip().lower()}')")
    elif action == "wait":
        code.append("pyautogui.sleep(0.5)")
    elif action == "fail":
        code.append("FAIL")
    elif action == "done":
        code.append("DONE")
    elif action == "call_user":
        code.append("CALL_USER")
    elif action == "screenshot":
        code.append("pyautogui.sleep(0.1)")
    elif action == "terminate":
        status = str(args.get("status", "success")).lower()
        code.append("FAIL" if status in ("failure", "fail") else "DONE")
    elif action == "cursor_position":
        code.append("pyautogui.sleep(0.1)")
    elif action in ("zoom", "zoom_in"):
        logger.info(f"action {action!r} → no-op sleep (env doesn't expose native zoom)")
        code.append("pyautogui.sleep(0.1)")
    else:
        raise ValueError(f"Invalid action: {action}")

    return code
# --- END verbatim copy ---


# --- BEGIN edited copy: OSWorld-V2/mm_agents/m3/parser.py::parse_m3_response (edits E1-E7, see header) ---
def parse_m3_tool_calls(response: str) -> Tuple[str, List[Tuple[str, Dict]]]:
    """Parse a single M3 grounding response into
    ``(low_level_instruction, [(action, arguments), ...])``.

    EDITED COPY of ``parse_m3_response`` (OSWorld-V2/mm_agents/m3/parser.py,
    tag v2026.08.08): the line scanner (``Action:`` line, ``<tool_call>`` /
    ``</tool_call>`` blocks, bare JSON lines, stop-sequence recovery of a
    missing closing tag) and the BPE-artifact recovery are verbatim; the
    edits are listed in the module provenance header (E1-E7). The tool call
    is returned as ``(action, arguments)`` for ``translate_m3_action``
    instead of being turned into pyautogui code here, so the argument
    contract and the terminal-name rejection run before any code is emitted.
    """
    if not response or not response.strip():
        return "", []
    if "[INFEASIBLE]" in response:
        raise M3GroundingRejected("the grounding response carries the [INFEASIBLE] token")

    low_level_instruction = ""
    tool_calls: List[Tuple[str, Dict]] = []

    def emit(json_str: str) -> None:
        try:
            tool_call = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool_call JSON: {e} — body: {json_str[:200]}")
            return
        if not isinstance(tool_call, dict):
            logger.warning(f"tool_call is not an object — skipping: {json_str[:200]}")
            return
        if tool_call.get("name") != "computer":
            logger.warning(f"Expected name='computer', got {tool_call.get('name')!r} — skipping")
            return
        args = tool_call.get("arguments") or tool_call.get("input") or {}
        if not isinstance(args, dict):
            logger.warning(f"tool_call arguments are not an object: {tool_call}")
            return
        action = args.get("action")
        if not action:
            logger.warning(f"tool_call missing 'action': {tool_call}")
            return

        # BPE tokenizer artifact recovery (M3 quirks observed in production).
        if isinstance(action, str) and "_ " in action:
            fixed = action.replace("_ ", "_")
            logger.info(f"[parser] BPE-fix action: {action!r} → {fixed!r}")
            action = fixed
            args = dict(args, action=fixed)
        if isinstance(args, dict):
            renamed = {}
            changed = False
            for k, v in args.items():
                if isinstance(k, str) and "_ " in k:
                    nk = k.replace("_ ", "_")
                    logger.info(f"[parser] BPE-fix arg key: {k!r} → {nk!r}")
                    renamed[nk] = v
                    changed = True
                else:
                    renamed[k] = v
            if changed:
                args = renamed
        if isinstance(action, str) and "_lick" in action and action not in _VALID_ACTIONS:
            candidate = action.replace("_lick", "_click")
            if candidate in _VALID_ACTIONS:
                logger.info(f"[parser] BPE click-recover: {action!r} → {candidate!r}")
                action = candidate
                args = dict(args, action=candidate)
        tool_calls.append((action, args))

    inside = False
    buffer: List[str] = []
    for line in response.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("action:"):
            if not low_level_instruction:
                low_level_instruction = stripped.split(":", 1)[-1].strip()
            continue
        if stripped.startswith("<tool_call>"):
            inside = True
            continue
        if stripped.startswith("</tool_call>"):
            if buffer:
                emit("\n".join(buffer))
                buffer = []
            inside = False
            continue
        if inside:
            buffer.append(stripped)
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    emit(stripped)
            except json.JSONDecodeError:
                pass

    if inside and buffer:
        emit("\n".join(buffer))

    return low_level_instruction, tool_calls
# --- END edited copy ---


_TERMINAL_TOKENS = frozenset({"DONE", "FAIL"})

# The same alias normalisation the copied function applies first, so the
# contract is looked up under the canonical name.
_ACTION_ALIASES = {"left click": "left_click", "right click": "right_click", "click": "left_click"}


def is_unset_argument(value: Any) -> bool:
    return value is None or value == "" or value == []


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    # An integral float (``5.0`` — JSON from a model can carry it) counts; it
    # is cast to int before translation so the emitted code reads ``scroll(-5``.
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def _is_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value)


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _is_direction(value: Any) -> bool:
    return value in ("up", "down", "left", "right")


_ARGUMENT_CHECKS: Dict[str, Callable[[Any], bool]] = {
    "coordinate": _is_pair,
    "start_coordinate": _is_pair,
    "text": _is_text,
    "scroll_direction": _is_direction,
    "scroll_amount": _is_integer,
    "duration": _is_number,
}

# (required, optional) argument names per supported action — the arguments
# the copied function actually reads for that action. The prompt-advertised
# grounding vocabulary (action_minimax.M3_GROUNDING_SYSTEM_PROMPT) plus the
# reasoning-owned terminal classes is listed; everything else is outside the
# contract.
_CLICK_ARGUMENTS = (("coordinate",), ("text",))
_ARGUMENT_CONTRACT: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "left_click": _CLICK_ARGUMENTS,
    "right_click": _CLICK_ARGUMENTS,
    "middle_click": _CLICK_ARGUMENTS,
    "double_click": _CLICK_ARGUMENTS,
    "triple_click": _CLICK_ARGUMENTS,
    "left_press": _CLICK_ARGUMENTS,  # in the model's vocabulary (_VALID_ACTIONS), not in the prompt
    "mouse_move": (("coordinate",), ("duration",)),
    "left_click_drag": (("coordinate",), ("start_coordinate", "duration")),
    "key": (("text",), ()),
    "type": (("text",), ()),
    "scroll": (("scroll_direction", "scroll_amount"), ("coordinate", "text")),
    "wait": ((), ("duration",)),
    "screenshot": ((), ()),
    "hold_key": (("text",), ("duration",)),
    "left_mouse_down": ((), ("coordinate", "text")),
    "left_mouse_up": ((), ("coordinate", "text")),
    "done": ((), ("text",)),
    "fail": ((), ("text",)),
}
# The copied function's ``call_user`` -> "CALL_USER" and ``terminate`` -> DONE/FAIL
# branches are dead by construction: both names raise ValueError above first.


def _validate_arguments(action: str, parameters: Dict[str, Any]) -> None:
    """Raise ``ValueError`` unless every required argument is present and every
    present KNOWN argument has the right shape — known arguments are checked
    whether or not the action uses them, because the copied function scales
    ``coordinate`` / ``start_coordinate`` for every action before branching
    (a malformed pair on a ``key`` or ``done`` would still crash it). Unknown
    extra keys are ignored, as the copied function ignores them."""
    if action not in _ARGUMENT_CONTRACT:
        raise ValueError(f"unsupported action {action!r}")
    required, _optional = _ARGUMENT_CONTRACT[action]
    for key in required:
        if key not in parameters:
            raise ValueError(f"{action}: required argument {key!r} is missing")
    for key, check in _ARGUMENT_CHECKS.items():
        if key in parameters and not check(parameters[key]):
            raise ValueError(f"{action}: argument {key!r} has an invalid value {parameters[key]!r}")


def _relative_scaler(screen_size: Tuple[int, int]) -> Callable[[float, float], Tuple[int, int]]:
    """The ``coordinate_type="relative"`` branch of V2's ``scaled_xy``: the
    model's ``[0, 1000]`` space mapped onto the screen's pixel dimensions."""
    original_width, original_height = screen_size

    def scaled_xy(x, y):
        return int(x * original_width / 1000), int(y * original_height / 1000)

    return scaled_xy


# Reading the emission back: which pyautogui verbs move the pointer, and the
# index of the FIRST coordinate argument in each call. Taken off
# `tool_action_to_pyautogui` above rather than from the model's action names,
# because the executed source is what actually moved the cursor — post-scaling,
# post-translation, in screen pixels. Note `scroll`/`hscroll` put the amount
# first, so their coordinate starts at index 1; every other verb starts at 0.
_PYAUTOGUI_CALL = re.compile(r"pyautogui\.(\w+)\(([^)]*)\)")
_POINTER_VERB_COORDINATE_INDEX: Dict[str, int] = {
    "click": 0, "rightClick": 0, "doubleClick": 0, "middleClick": 0, "tripleClick": 0,
    "moveTo": 0, "dragTo": 0, "mouseDown": 0, "mouseUp": 0,
    "scroll": 1, "hscroll": 1,
}


def pointer_xy_from_pyautogui(command: Any) -> Optional[Tuple[int, int]]:
    """The pointer position an executed step left the cursor at, in SCREEN
    PIXELS — or ``None`` for a step that moved no pointer.

    ``command`` is the exact source the driver executed, as the adapter recorded
    it on ``runtime_state.content["_executed_actions"]``. A step can emit several
    statements (a modifier keyDown, the click, the keyUp; a drag's moveTo then
    dragTo), so the LAST coordinate-bearing call wins — that is where the cursor
    ends up. Returns ``None`` for a key press, a typed string, an env token
    (``WAIT`` / ``DONE`` / ``FAIL``), and for the coordinate-less forms of
    ``scroll`` / ``mouseDown`` / ``mouseUp``, all of which leave the pointer
    where it already was. Never raises: unparseable source simply has no
    position."""
    if not isinstance(command, str):
        return None
    position: Optional[Tuple[int, int]] = None
    for match in _PYAUTOGUI_CALL.finditer(command):
        index = _POINTER_VERB_COORDINATE_INDEX.get(match.group(1))
        if index is None:
            continue
        arguments = [token.strip() for token in match.group(2).split(",")]
        if len(arguments) <= index + 1:
            continue
        try:
            position = (int(arguments[index]), int(arguments[index + 1]))
        except ValueError:
            continue
    return position


def translate_m3_action(action: Action, *, screen_size: Tuple[int, int]) -> str:
    """Translate one ``MiniMaxActionAgent`` action (``action_type`` = the M3
    action name, ``parameters`` = the tool arguments minus ``action``, relative
    coordinates) into what V2's driver executes. See the module docstring for
    the return / ValueError contract."""
    name = str(action.action_type)
    name = _ACTION_ALIASES.get(name, name)
    parameters = {
        key: value for key, value in dict(action.parameters or {}).items()
        if not is_unset_argument(value)
    }
    _validate_arguments(name, parameters)
    if isinstance(parameters.get("scroll_amount"), float):
        parameters["scroll_amount"] = int(parameters["scroll_amount"])
    code = tool_action_to_pyautogui(name, parameters, _relative_scaler(screen_size))
    if not code:
        raise ValueError(f"action {name!r} translated to no pyautogui code: {parameters!r}")
    if len(code) == 1 and code[0] in _TERMINAL_TOKENS:
        return code[0]
    return "\n".join(code)
