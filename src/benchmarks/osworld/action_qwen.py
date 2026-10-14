"""Qwen action agent for OSWorld 2.0 — grounds an intended action into the Qwen
control's own ``computer_use`` XML tool call (``osworld-v2.1:mm_agents/qwen``), parsed,
validated and translated by the control's own code (``qwen_upstream/``).

THE GROUNDING PROMPT is the control's system prompt (``build_system_prompt`` over
``build_tool_schema``, batching off), derived here from the copied builders by exactly
these edits:
  * ``terminate`` and ``call_user`` are removed from the action enum and its
    description, and ``status`` (terminate's only parameter) from the schema; ``text``
    reads "Text for type." — termination and asking are the deciding call's, as on
    MiniMax;
  * the Rules keep "Required parameters…", "Optional reasoning…" and the batch-off
    line "Return exactly one tool call per turn.".
EACH CONTROL CLAUSE IS IN EXACTLY ONE GRAPHPC CALL, BY FUNCTION. The tool
description (incl. "You have no terminal access; …", the 1000x1000 coordinate space,
"Consult the latest screenshot …") is here, in grounding, beside the tool. The
task-strategy rules (no macros/scripts/terminals/APIs, verify before terminating),
the terminate / call_user descriptions and the old-screenshot rule are in the deciding
suffix (``prompts.QWEN_DECIDING_INSTRUCTION_SUFFIX``); the date is in the deciding
call's environment brief.

THE CALL: the backbone's ``complete_grounding`` — reasoning off, the control's
sampling and output cap, no tools, no ``stop`` (the copied parser requires the closing
tag). The user turn is the current screen and the model-neutral intended-action line
(``action_minimax.render_intended_action``, coordinates stripped from target/intent
first), as on MiniMax.

REPAIR, NOT RE-SAMPLE (the control's, ``main.py:_parse_with_repairs``): a reply that
cannot become the action is sent back — the original request, the reply's content as
an assistant turn, and the control's own repair turn (``recovery.repair_payload``:
"Your previous computer_use response could not execute: {error}. …") — up to
``model_call_attempts`` attempts in all (3: the control's 2 repairs). What is repaired
is what the control repairs: the copied parser's rejections, the copied translator's
validation (dry-run at the real screen size, so the adapter's translation cannot fail
on it), and two cases the control never meets in grounding: a reply with no tool call
(the control sends that to its judge; this call has no other outcome) and a
``terminate`` / ``call_user`` (outside the offered enum: the control's "Unsupported
tool action" wording) — plus GraphPC's own class-family rule below. With several
tool calls the FIRST complete one is taken, with one exception, the composite below. A
reply cut at the output cap before any content is never repaired. Exhausted, the batch
degrades: empty, ``grounding_parse_error`` (``grounding_class_mismatch`` when the last
reply failed the class-family rule), a counted WAIT at the adapter.

THE COMPOSITE. The control's ``scroll`` / ``hscroll`` takes no coordinate —
it scrolls wherever the pointer is — so the model spells "scroll there" as two calls,
``mouse_move`` then ``scroll``, which the control runs in one turn. Taking only the
first would run a pointer move instead of the decided scroll. Its ``left_click_drag`` has no
start point — it drags from wherever the pointer is — so "drag from here to there" is
[``mouse_move``, ``left_click_drag``] likewise. So when the deciding intent's class is
``scroll``, ``hscroll`` or ``left_click_drag`` AND the reply's complete calls are
exactly [``mouse_move``, that class's own call], the action is that call carrying the
pointer move in ``metadata["grounding_composite"]``, and ``translate_qwen_action``
emits the control's own translation of the pair, ``moveTo`` then the scroll or the
drag, as one command. A pair of that shape the control's translator refuses is repaired
with the translator's text, as the control would repair it. Any other intent, shape or
axis keeps the first-call rule.

THE CLASS-FAMILY RULE. For a decided class in
``GROUNDED_CLASSES``, a valid first call is not run
but repaired when its class is outside the decided class's family (``CLASS_FAMILIES``:
the click family — left, right, double, triple and middle click and ``left_click_drag``
— and every other grounded class alone, ``scroll`` and ``hscroll`` each on its own
axis) or acts on nothing (``screenshot`` / ``wait``, ``ALWAYS_REJECTED``). The repair
quotes "{decided} was decided but the reply is {got}; reply with a {decided}" ({got}:
the first complete call's action). A substitution within the click family executes:
a repair enforces the class, not the location. The rule is checked LAST —
after the parse, the composite, the ``terminate`` / ``call_user`` rejection and the
translator's validation — so each earlier rejection keeps its own kind and the control's
own text. A decided class outside the list (a bypass or terminal class never reaches
grounding; an off-vocabulary class does, and keeps the first-call rule) is not checked.
GraphPC therefore repairs replies the control would execute; that is disclosed. Every
batch carries the decided class, normalised (``decided_action_class``), which the adapter
writes on the step's executed-action row, and the repairs the rule caused
(``class_mismatch_repairs``), which the adapter keeps out of its grounding sampling
counters.

WITHOUT A GROUNDING CALL, as on MiniMax: ``done`` / ``fail`` (the reasoning call owns
termination; emitted as ``done`` with ``terminal_intent``, so the completion gate sees
a give-up), ``call_user`` (the ask channel; emitted with its question for the adapter,
no model call) and the coordinate-free classes — ``type``, ``key``, ``key_down``,
``key_up``, ``wait``, ``screenshot``. THE COORDINATE-FREE BYPASS IS GRAPHPC'S METHOD,
NOT A MODEL SETTING: these classes' tool call is
a total function of the intent, so a grounding call can add nothing and can lose what
the intent carries (action_minimax.py, module docstring); built here from the intent,
guarded as MiniMax's is (a newline or internal whitespace in a key payload is not a key),
and validated by the copied translator. A payload that cannot become the call degrades
the same way, carrying ``intended_action_defect``.

THE VOCABULARY. The internal action names are M3's (graphpc_agent.OSWorldModelProfile)
and the control's overlap them for 14 actions; the control-only ``key_down``, ``key_up``
and ``hscroll`` are internal names of their own, ``terminate`` is ``done`` / ``fail``,
and ``call_user`` is the ask channel's class. An Action's type is therefore the
control's action name and its parameters the control's arguments, which is what the
translator (``translate_qwen_action``) reads back.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

from src.agents.action.base import ActionAgent, ActionInput
from src.agents.common.schema import Action, ActionBatch

from .action_minimax import (
    M3BypassPayloadRejected,
    coerce_intent_field,
    normalize_action_class,
    reject_non_keystroke_payload,
    render_intended_action,
    strip_grounding_coordinates,
)
from .backbone_qwen import GroundingReplyRejected
from .reasoning import LENGTH_EMPTY
from .qwen_upstream.actions import BASE_ACTIONS, ActionValidationError, translate_tool_calls
from .qwen_upstream.images import ProcessedScreenshot
from .qwen_upstream.parser import (
    ToolCall,
    ToolCallParseError,
    looks_repeated_think,
    natural_language_text,
    parse_tool_calls,
)
from .qwen_upstream.prompts import ACTION_DESCRIPTIONS, build_system_prompt, build_tool_schema
from .qwen_upstream.recovery import repair_payload


logger = logging.getLogger("desktopenv.agent")

TOOL_NAME = "computer_use"
ASK_USER_CLASS = "call_user"
_TERMINAL_CLASSES: frozenset[str] = frozenset({"done", "fail"})
# The control's own actions the deciding call owns: never offered to grounding.
_DECIDING_ONLY = ("terminate", ASK_USER_CLASS)
GROUNDING_ACTIONS: tuple[str, ...] = tuple(name for name in BASE_ACTIONS if name not in _DECIDING_ONLY)

# The deciding schema's closed vocabulary on the Qwen profile: the control's actions,
# `terminate` as the two terminal classes the reasoning stage owns.
QWEN_ACTION_CLASS_VOCABULARY: tuple[str, ...] = tuple(
    name for name in BASE_ACTIONS if name != "terminate"
) + tuple(sorted(_TERMINAL_CLASSES))

# Coordinate-free classes, built from the intent without a grounding call, and the
# intent fields that supply their one argument, in precedence order (MiniMax's
# measured precedence: `keys` then `text` for a key, `text` for a type).
_BYPASS_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "type": ("text", ("text",)),
    "key": ("keys", ("keys", "text")),
    "key_down": ("keys", ("keys", "text")),
    "key_up": ("keys", ("keys", "text")),
    "wait": ("", ()),
    "screenshot": ("", ()),
}

# Every class that reaches grounding, in the control's action names: the classes the
# class-family rule checks (module docstring).
GROUNDED_CLASSES: tuple[str, ...] = (
    "left_click", "right_click", "double_click", "triple_click", "middle_click",
    "left_click_drag", "left_mouse_down", "left_mouse_up",
    "mouse_move", "scroll", "hscroll",
)
# The class-family rule's families, per decided class: the click family, every other
# grounded class alone; and the reply classes that act on nothing, never run for a
# grounded intent.
_CLICK_FAMILY: frozenset[str] = frozenset({"left_click", "right_click", "double_click", "triple_click",
                                           "middle_click", "left_click_drag"})
CLASS_FAMILIES: dict[str, frozenset[str]] = {
    name: _CLICK_FAMILY if name in _CLICK_FAMILY else frozenset({name}) for name in GROUNDED_CLASSES
}
ALWAYS_REJECTED: frozenset[str] = frozenset({"screenshot", "wait"})

# The intents the composite serves, and the reply shape it takes (module docstring).
_COMPOSITE_CLASSES: frozenset[str] = frozenset({"scroll", "hscroll", "left_click_drag"})
COMPOSITE_KEY = "grounding_composite"

# The class-family rule's degrade marker (the adapter's WAIT reason) and rejection kind, and
# the batch metadata the adapter reads (module docstring).
CLASS_MISMATCH = "grounding_class_mismatch"
CLASS_MISMATCH_KIND = "class_mismatch"
DECIDED_CLASS_KEY = "decided_action_class"
MISMATCH_REPAIRS_KEY = "class_mismatch_repairs"
# Its two decider-visible texts: what the repair
# quotes, and the line the adapter's exhausted WAIT carries into the next deciding request.
CLASS_MISMATCH_REPAIR_TEXT = "{decided} was decided but the reply is {got}; reply with a {decided}"
CLASS_MISMATCH_LINE = "the decided {decided} was not executed: the last of {n} grounding replies was {got}"
# The rule's method surface as the run's active_config records it, DERIVED from the
# constants above: method_surface_qwen.grounding_class_check in the pre-registration
# declares the same value, and the host launch preflight holds the built agent to it.
GROUNDING_CLASS_CHECK_KEY = "grounding_class_check"
GROUNDING_CLASS_CHECK: dict[str, Any] = {
    "families": {name: sorted(CLASS_FAMILIES[name]) for name in GROUNDED_CLASSES},
    "always_rejected": sorted(ALWAYS_REJECTED),
    "repair_text": CLASS_MISMATCH_REPAIR_TEXT,
    "exhaustion_line": CLASS_MISMATCH_LINE,
    "composite_classes": sorted(_COMPOSITE_CLASSES),
}

# The error a repair quotes when the reply held no tool call at all (the control sends
# such a reply to its terminal judge, so it has no wording for it).
NO_TOOL_CALL_ERROR = "the response contains no tool call"
# The control's own wording for a call_user without its question (actions.py).
CALL_USER_WITHOUT_TEXT = "call_user requires text"


def _grounding_system_prompt() -> str:
    """The control's system prompt with the edits the module docstring lists. The date
    rule is dropped with the other deciding-call rules, so the date passed is inert."""
    schema = build_tool_schema(coordinate_type="relative", model_width=0, model_height=0, enable_zoom=False)
    properties = schema["function"]["parameters"]["properties"]
    properties["action"]["enum"] = list(GROUNDING_ACTIONS)
    properties["action"]["description"] = "\n".join(
        f"* `{name}`: {ACTION_DESCRIPTIONS[name]}" for name in GROUNDING_ACTIONS
    )
    del properties["status"]
    properties["text"]["description"] = "Text for type."
    rendered = build_system_prompt(schema, collapsed_screenshot_text="", task_current_date="-",
                                   enable_batch_actions=False)
    moved = ("- Do not use macros", "- Verify the visible", "- Current date:", "- Old screenshots")
    return "\n".join(line for line in rendered.split("\n") if not line.startswith(moved))


QWEN_GROUNDING_SYSTEM_PROMPT = _grounding_system_prompt()


def _screen(screen_size: tuple[int, int]) -> ProcessedScreenshot:
    """The control translator's view of the screen, relative coordinates: only the
    original size scales a point; the model size is not read in relative mode."""
    width, height = int(screen_size[0]), int(screen_size[1])
    return ProcessedScreenshot(data_url="", coordinate_type="relative", original_width=width,
                               original_height=height, model_width=width, model_height=height)


def _translate(*calls: ToolCall, screen_size: tuple[int, int]):
    return translate_tool_calls(list(calls), screenshot=_screen(screen_size), coordinate_type="relative",
                                enable_zoom=False)


def is_class_mismatch(decided: str | None, got: str) -> bool:
    """The class-family rule's one predicate (module docstring): whether a valid reply call
    of class ``got`` is sent back for the decided class — for a decided class in
    ``GROUNDED_CLASSES``, a call that acts on nothing or is outside its family; never for a
    decided class outside the list."""
    if decided not in GROUNDED_CLASSES:
        return False
    return got in ALWAYS_REJECTED or got not in CLASS_FAMILIES[decided]


def _is_composite(calls: list[ToolCall], decided_class: str | None) -> bool:
    return (len(calls) == 2 and calls[0].action == "mouse_move" and decided_class in _COMPOSITE_CLASSES
            and calls[1].action == decided_class)


class GroundingClassMismatch(GroundingReplyRejected):
    """A valid grounding reply whose first call the class-family rule sends back (module
    docstring); the message is the text the repair quotes."""

    def __init__(self, decided: str, got: str) -> None:
        super().__init__(CLASS_MISMATCH_REPAIR_TEXT.format(decided=decided, got=got), kind=CLASS_MISMATCH_KIND)
        self.decided, self.got = decided, got


_OPENING, _CLOSING = re.compile(r"<tool_call>", re.IGNORECASE), re.compile(r"</tool_call>", re.IGNORECASE)


def _dropped(text: str, executed: int) -> int:
    """The reply's tool-call blocks minus the calls executed (``_parse``); the tags are matched
    without regard to case, as the control's parser matches them."""
    return max(0, max(len(_OPENING.findall(text)), len(_CLOSING.findall(text))) - executed)


def translate_qwen_action(action: Action, *, screen_size: tuple[int, int]) -> str:
    """One ``QwenActionAgent`` action (``action_type`` = the control's action name,
    ``parameters`` = its arguments) -> what V2's driver executes, by the control's
    translator; ``ValueError`` (the control's ``ActionValidationError`` is one) for any
    defect. Commands the control would run as consecutive steps of one turn are joined
    into one source — a composite's pointer move, then the scroll or the drag; ``wait``
    is V2's ``WAIT`` token."""
    composite = (action.metadata or {}).get(COMPOSITE_KEY)
    calls = [ToolCall({"action": "mouse_move", **composite["mouse_move"]}, 0, 0)] if composite else []
    calls.append(ToolCall({"action": str(action.action_type), **dict(action.parameters or {})}, 0, 0))
    batch = _translate(*calls, screen_size=screen_size)
    if batch.question is not None or batch.terminal_status is not None or not batch.commands:
        raise ValueError(f"action {action.action_type!r} is not an executable computer action")
    return "\n".join(batch.commands)


class QwenActionAgent(ActionAgent):
    """``model_call_attempts`` is the TOTAL grounding attempts per intended action
    (3 on this benchmark: the control's ``tool_repair_retries`` 2 + 1). ``method_surface``:
    the keys this agent adds to the run's active_config (``build_default``)."""

    method_surface: dict[str, Any] = {GROUNDING_CLASS_CHECK_KEY: GROUNDING_CLASS_CHECK}

    def __init__(self, *, backbone: Any, model_call_attempts: int = 1) -> None:
        self.backbone = backbone
        self.model_call_attempts = int(model_call_attempts)

    def act(self, action_input: ActionInput) -> ActionBatch:
        """Every batch, degraded or empty included, carries the decided class, normalised
        (None without an intent), and the class-family rule's repairs (module docstring)."""
        batch = self._act(action_input)
        intended = self._intended_actions(action_input)
        first = intended[0] if intended and isinstance(intended[0], dict) else {}
        batch.metadata[DECIDED_CLASS_KEY] = normalize_action_class(first.get("action_class")) or None
        batch.metadata.setdefault(MISMATCH_REPAIRS_KEY, 0)
        return batch

    def _act(self, action_input: ActionInput) -> ActionBatch:
        intended = self._intended_actions(action_input)
        if not intended:
            return ActionBatch(actions=[], content={}, raw={})
        screen_size = self._screen_size(action_input)
        actions: list[Action] = []
        responses: list[str] = []
        instructions: list[str] = []
        repairs_total = 0
        mismatch_repairs_total = 0
        strip_records: list[dict[str, Any]] = []
        for index, intended_action in enumerate(intended[: action_input.max_actions]):
            intended_action = intended_action or {}
            raw_action_class = str(intended_action.get("action_class") or "")
            action_class = normalize_action_class(raw_action_class)
            if action_class in _TERMINAL_CLASSES:
                actions.append(self._terminal_action(action_type=action_class, intended_action=intended_action,
                                                     index=index))
                continue
            if action_class == ASK_USER_CLASS or action_class in _BYPASS_FIELDS:
                try:
                    actions.append(self._built_action(action_class=action_class, raw_action_class=raw_action_class,
                                                      intended_action=intended_action, index=index,
                                                      screen_size=screen_size))
                except (M3BypassPayloadRejected, ActionValidationError) as exc:
                    logger.warning("intent cannot become a %s call: %s; degrading to empty batch", action_class, exc)
                    return self._degraded(exc, repairs=repairs_total, defect=str(exc))
                continue
            grounding_intent, strip_record = strip_grounding_coordinates(intended_action)
            strip_records.append(strip_record)
            try:
                # The decided class gates the composite and names the family the reply must be in.
                result, repairs, mismatch_repairs = self._ground(
                    user_prompt=render_intended_action(grounding_intent), observation=action_input.observation,
                    screen_size=screen_size, decided_class=action_class,
                )
            except GroundingReplyRejected as exc:
                logger.warning("grounding reply rejected (%s): %s; degrading to empty batch", exc.kind, exc)
                batch = self._degraded(exc, repairs=repairs_total + exc.repairs)
                batch.metadata["grounding_input_strip"] = strip_records
                batch.metadata[MISMATCH_REPAIRS_KEY] = mismatch_repairs_total + exc.mismatch_repairs
                return batch
            repairs_total += repairs
            mismatch_repairs_total += mismatch_repairs
            text = result["text"]
            responses.append(text)
            instructions.append(natural_language_text(text))
            call, pointer_move = result["parsed"]
            actions.append(self._grounded_action(call=call, pointer_move=pointer_move, raw_response=text,
                                                 intended_action=intended_action, index=index))

        metadata: dict[str, Any] = {"resample_attempts": repairs_total, MISMATCH_REPAIRS_KEY: mismatch_repairs_total}
        if strip_records:
            metadata["grounding_input_strip"] = strip_records
        return ActionBatch(
            actions=actions,
            content={"reasoning_text": "\n".join(instructions)},
            raw={"qwen_responses": responses},
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _ground(self, *, user_prompt: str, observation: Any, screen_size: tuple[int, int],
                decided_class: str):
        """``(result, repairs, mismatch_repairs)``: the grounding call, repaired as the
        control repairs, and how many of its repairs the class-family rule caused; or
        ``GroundingReplyRejected`` (carrying ``repairs`` and ``mismatch_repairs``) when the
        attempts run out or the reply was cut at the output cap. ``decided_class``: the
        intent's normalised class, whose family the reply must be in and whose composite
        it may be (``_parse``)."""
        turns: tuple[dict[str, Any], ...] = ()
        mismatch_repairs = 0
        for attempt in range(1, self.model_call_attempts + 1):
            try:
                result = self.backbone.complete_grounding(
                    system_prompt=QWEN_GROUNDING_SYSTEM_PROMPT, user_prompt=user_prompt, observation=observation,
                    turns=turns, validate=lambda text: self._parse(text, screen_size, decided_class),
                )
                return result, attempt - 1, mismatch_repairs
            except GroundingReplyRejected as exc:
                exc.repairs, exc.mismatch_repairs = attempt - 1, mismatch_repairs
                if exc.kind == LENGTH_EMPTY or attempt == self.model_call_attempts:
                    raise
                mismatch_repairs += exc.kind == CLASS_MISMATCH_KIND
                logger.warning("repairing grounding reply (attempt %d): %s", attempt, exc)
                turns = tuple(repair_payload({"messages": []}, exc.reply_text, str(exc))["messages"])
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse(text: str, screen_size: tuple[int, int],
               decided_class: str | None = None) -> tuple[tuple[ToolCall, ToolCall | None], dict[str, Any]]:
        """``((the call to execute, the pointer move before it or None), facts)``: the
        first complete tool call, validated by the control's translator and, for a
        ``decided_class`` in ``GROUNDED_CLASSES``, in its family — or, when the complete
        calls are exactly [``mouse_move``, ``decided_class``] for a scroll, hscroll or
        drag, the pair as the control's translator accepts it: that call with the move
        (the composite, module docstring). Checked in this order, each rejection with
        its own kind: parse (``unparseable`` / ``no_tool_call``), composite (an invalid
        pair: ``invalid_action``), ``terminal``, validation (``invalid_action``), class
        family (``class_mismatch``). The facts are the per-call record's: how many
        complete calls the reply held (``tool_call_count``; the grounding prompt asks for
        one) and ``tool_calls_dropped``, the reply's tool-call blocks minus the calls
        executed — dropped, not repaired. A block is counted by its ``<tool_call>``
        opening or its ``</tool_call>`` closing, whichever the reply has more of, so a
        trailing block that is not a complete call (e.g. calls chained with ``<then>``:
        one opening, two closings) counts as dropped too. Else ``GroundingReplyRejected``
        with the control's error text for it, or the rule's own for a class mismatch."""
        try:
            if looks_repeated_think(text):
                raise ToolCallParseError("repeated thinking block without a tool call")
            calls = parse_tool_calls(text)
        except ToolCallParseError as exc:
            raise GroundingReplyRejected(str(exc), kind="unparseable") from exc
        if not calls:
            raise GroundingReplyRejected(NO_TOOL_CALL_ERROR, kind="no_tool_call")
        if _is_composite(calls, decided_class):
            try:
                _translate(*calls, screen_size=screen_size)
            except ActionValidationError as exc:
                # The decided class's own pair, refused: repaired as the control would repair it.
                raise GroundingReplyRejected(str(exc), kind="invalid_action") from exc
            return (calls[1], calls[0]), {"tool_call_count": 2, "tool_calls_dropped": _dropped(text, 2)}
        call = calls[0]
        if call.action in _DECIDING_ONLY:
            raise GroundingReplyRejected(f"Unsupported tool action: {call.action}", kind="terminal")
        try:
            _translate(call, screen_size=screen_size)
        except ActionValidationError as exc:
            raise GroundingReplyRejected(str(exc), kind="invalid_action") from exc
        if is_class_mismatch(decided_class, call.action):
            raise GroundingClassMismatch(decided_class, call.action)
        if len(calls) > 1:
            logger.info("grounding reply carried %d tool calls; using the first", len(calls))
        return (call, None), {"tool_call_count": len(calls), "tool_calls_dropped": _dropped(text, 1)}

    def _built_action(self, *, action_class: str, raw_action_class: str, intended_action: dict[str, Any],
                      index: int, screen_size: tuple[int, int]) -> Action:
        """A call_user or coordinate-free action from the intent alone (no model call),
        validated by the control's translator."""
        if action_class == ASK_USER_CLASS:
            question = coerce_intent_field("text", intended_action.get("text")) or ""
            if not question.strip():
                raise ActionValidationError(CALL_USER_WITHOUT_TEXT)
            arguments: dict[str, Any] = {"text": question}
            source_field: str | None = "text"
        else:
            argument, fields = _BYPASS_FIELDS[action_class]
            arguments, source_field = {}, None
            for field in fields:
                value = coerce_intent_field(field, intended_action.get(field))
                if value:
                    if argument == "keys":
                        reject_non_keystroke_payload("key", value)
                    arguments, source_field = {argument: value}, field
                    break
            if fields and not arguments:
                seen = {field: intended_action.get(field) for field in fields}
                raise M3BypassPayloadRejected(
                    f"the {action_class!r} intent carries no text to act on: {seen!r} (checked {list(fields)} in "
                    f"that order)"
                )
        call = ToolCall({"action": action_class, **arguments}, 0, 0)
        _translate(call, screen_size=screen_size)
        metadata: dict[str, Any] = {
            "tool_name": TOOL_NAME,
            "action_index": index,
            "action_space": "pyautogui",
            "coordinate_type": "relative",
            "raw_response": None,
            "model_call": False,
            "intended_action_field": source_field,
            "intended_action_class_raw": raw_action_class,
        }
        return self._action(action_class, arguments, raw_response=None, metadata=metadata,
                            intended_action=intended_action)

    def _grounded_action(self, *, call: ToolCall, pointer_move: ToolCall | None, raw_response: str,
                         intended_action: dict[str, Any], index: int) -> Action:
        """The grounded call as the action; a composite's pointer move rides in its
        metadata, not its parameters: a lone scroll may carry a coordinate of its own,
        which the control's translator ignores, and must stay that scroll."""
        parameters = {key: value for key, value in call.parameters.items() if key != "action"}
        metadata = {"tool_name": TOOL_NAME, "action_index": index, "action_space": "pyautogui",
                    "coordinate_type": "relative", "raw_response": raw_response}
        if pointer_move is not None:
            metadata[COMPOSITE_KEY] = {
                "mouse_move": {key: value for key, value in pointer_move.parameters.items() if key != "action"},
            }
        return self._action(call.action, parameters, raw_response=raw_response, metadata=metadata,
                            intended_action=intended_action)

    def _terminal_action(self, *, action_type: str, intended_action: dict[str, Any], index: int) -> Action:
        """``done`` / ``fail`` without a model call, both as ``done`` carrying
        ``terminal_intent`` (MiniMaxActionAgent._terminal_action's contract: the
        completion gate acts on ``done`` only, so a give-up is gated the same way)."""
        text = next((str(intended_action[key]) for key in ("text", "value", "intent", "target")
                     if intended_action.get(key)), "")
        metadata = {"tool_name": TOOL_NAME, "action_index": index, "action_space": "pyautogui",
                    "coordinate_type": "relative", "raw_response": None, "model_call": False,
                    "terminal_intent": action_type}
        return self._action("done", {"text": text}, raw_response=None, metadata=metadata,
                            intended_action=intended_action)

    @staticmethod
    def _action(name: str, parameters: dict[str, Any], *, raw_response: Any, metadata: dict[str, Any],
                intended_action: dict[str, Any]) -> Action:
        predicted = intended_action.get("predicted_effect_class")
        if predicted is not None:
            metadata["predicted_effect_class"] = predicted
        return Action(
            action_type=name,
            parameters=dict(parameters),
            raw={"name": TOOL_NAME, "input": {"action": name, **parameters}, "raw_response": raw_response},
            metadata=metadata,
        )

    @staticmethod
    def _degraded(exc: Exception, *, repairs: int, defect: str | None = None) -> ActionBatch:
        """The degrade contract, as MiniMax's: an EMPTY batch naming the rejection; an
        intent-side defect also rides to the next step's feedback. A last reply that
        failed the class-family rule is named apart, with what the adapter's WAIT line
        states: the attempts made, the last reply's class and the decided one."""
        if isinstance(exc, GroundingClassMismatch):
            return ActionBatch(actions=[], content={CLASS_MISMATCH: {"attempts": exc.repairs + 1, "got": exc.got,
                                                                     "decided": exc.decided}},
                               raw={}, metadata={"resample_attempts": repairs})
        content: dict[str, Any] = {"grounding_parse_error": f"{type(exc).__name__}: {exc}"}
        if defect:
            content["intended_action_defect"] = defect
        return ActionBatch(actions=[], content=content, raw={}, metadata={"resample_attempts": repairs})

    @staticmethod
    def _screen_size(action_input: ActionInput) -> tuple[int, int]:
        size = (getattr(action_input.observation, "metadata", None) or {}).get("screen_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])
        return 1920, 1080

    @staticmethod
    def _intended_actions(action_input: ActionInput) -> list:
        """Slim handoff: the structured intended_actions payload only."""
        reasoning = action_input.reasoning_message
        intended = (reasoning.content or {}).get("intended_actions") if reasoning is not None else None
        return deepcopy(intended) if isinstance(intended, list) else []

