"""The OSWorld 2.0 deciding call — the graph-reasoning agent, specialised, and the
timeline route a backbone carries (``TimelineRoute``; MiniMax's is
``OSWorldMiniMaxAgent``).

WHY SUBCLASSES AND NOT PARAMETERS. Every change here renders the DECIDING
request differently, and ``src/agents/reasoning/graph_reasoning.py`` and
``src/agents/backbone/minimax/main.py`` are shared with WebArena, whose request
bytes must not move with OSWorld's. A defaulted parameter in shared code would put an
OSWorld branch into WebArena's code path; subclasses that only OSWorld's
``build_default`` constructs cannot execute there at all — class two by
construction (design.md §10). The shared classes are used unchanged
for everything these do not override: the grounding route, usage accounting, the
retry policy, the re-sample and degrade contract, the message the orchestrator
reads back.

THE REQUEST THIS RENDERS (see ``timeline.py`` for the user message):

    SYSTEM (static for the episode) = the deciding instruction + this benchmark's
        suffix (prompts.py) + the environment facts
    USER = the episode timeline, the current state, the JSON-schema block, then
        the CURRENT SCREEN last

The environment facts sit in the system prompt, where the control keeps them.
The schema block is the last text before the model answers, because with the
schema at the end of the system prompt instead the model sometimes returns the
schema itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time

from typing import Any, Callable, Optional, Protocol

from src.agents.backbone.base import Backbone
from src.agents.backbone.minimax.main import MiniMaxAgent
from src.agents.common.json_utils import JsonParseExhausted
from src.agents.common.resample import call_with_resample
from src.agents.common.schema import AgentMessage
from src.agents.reasoning.base import ReasoningInput, empty_decision_message, require_decision_shape
from src.agents.reasoning.graph_reasoning import GraphReasoningAgent

from .prompts import OSWORLD_REQUIREMENTS_INSTRUCTION
from .timeline import (
    REQUIREMENT_FIELDS,
    REQUIREMENT_STATUSES,
    REQUIREMENT_TEXT_LIMIT,
    REQUIREMENTS_RENDER_ITEMS,
    EpisodeTimeline,
    render_state_text,
)


logger = logging.getLogger("desktopenv.agent")

# The `facts` field this benchmark's deciding schema adds (the ledger and its
# bounds are in timeline.py). Advisory like the rest of the schema on MiniMax.
_FACTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "source"],
        "properties": {"value": {"type": "string"}, "source": {"type": "string"}},
    },
}

# What the ledger is FOR, stated as scope. Left unscoped, the ledger fills with screen state and action
# outcomes ("the modal is still open", "step 179 navigated to page 2") and progress summaries — all of
# which the step history and the plan already carry — and the values the task depends on (rules read in
# a chat channel, the records the task works on) are pushed out of the rendered list by the rest. This
# paragraph is OSWorld's own scaffolding for its own field, not a parity clause: the control has no
# ledger. Neutral wording — it says what belongs in the field, and nothing about how long to work or
# what to verify.
# On a desktop every value is read off the screen, so `source` is asked to name the document or record
# itself: the render keeps entries whose source names one ahead of the rest (timeline.fact_class), and
# an entry whose source is "screenshot" or "screen" ranks with screen state. The examples occur in none
# of the release's task classes, so no example names a task's own asset.
_FACTS_INSTRUCTION = (
    "FACTS: in `facts`, record only values you READ FROM A SOURCE — a document, message, "
    "email, record, file or web page — that later steps will need: names, numbers, dates, "
    "paths, e-mail addresses, IDs, rules, requirements. Copy each value verbatim. In "
    "`source`, name the document, message, email, record, file or page itself (for example "
    "\"handbook.pdf\", \"email from the finance team, 12 Feb\", \"customer record #4471\"), "
    "not the screen or screenshot you read it on; [] when there is none. Do not record screen "
    "state, what an action did, or progress: the step history and the plan already carry "
    "those. Every entry is kept for the whole "
    "episode and shown back to you as `facts` under STATE, so do not repeat an entry; to "
    "correct one, add the corrected entry and say in its `source` which it corrects."
)

# The `requirements` field a profile with the requirement ledger adds to the deciding schema (the DeepSeek profile;
# prompts.OSWORLD_REQUIREMENTS_INSTRUCTION says what it is for, timeline.py holds its bounds and render). Flat items, the
# bounds written into the schema as the instruction states them; advisory like the rest of the schema on this route:
# the reply's full list is stored and judged whatever its length, and only the render is clipped.
REQUIREMENTS_SCHEMA = {
    "type": "array",
    "maxItems": REQUIREMENTS_RENDER_ITEMS,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIREMENT_FIELDS),
        "properties": {
            "id": {"type": "string"},
            "requirement": {"type": "string", "maxLength": REQUIREMENT_TEXT_LIMIT},
            "status": {"type": "string", "enum": list(REQUIREMENT_STATUSES)},
            "evidence_step": {"type": ["integer", "null"]},
            "evidence": {"type": "string", "maxLength": REQUIREMENT_TEXT_LIMIT},
            "reason": {"type": "string", "maxLength": REQUIREMENT_TEXT_LIMIT},
        },
    },
}

# The degrade reason when every attempt came back without a decision; the adapter
# reads it to say so on the step's WAIT line.
DECISION_MISSING_REASON = "reasoning reply carried no decision"

# The rejection kind of a reply cut at the output cap before any content: never
# repaired or re-sampled, the step degrades at once (a backbone raises it from
# ``TimelineRoute._check_reply``).
LENGTH_EMPTY = "length_empty"

class DecisionMissing(JsonParseExhausted):
    """A deciding reply that PARSES but carries no decision: no list-typed
    ``intended_actions``, or the JSON Schema itself echoed back. ``kind`` says
    which shape it was: ``schema_echo``, ``native_tool_call`` (the model's own
    tool-call format, e.g. ``{"name": "computer", "input": {...}}`` after its
    ``<tool_call>`` tag) or ``other``.

    WHY IT IS RE-SAMPLED. The control re-samples a non-empty reply that parses to
    zero actions (mm_agents/m3/agent.py ``predict``, outcome ``parse_empty``),
    within the same budget this benchmark's calls get (MODEL_CALL_ATTEMPTS). The
    shared shape check (``require_decision_shape``) checks only what the consumers
    would crash on, so such a reply would otherwise pass and run as a WAIT; the
    model does return schema echoes and native tool calls in place of a decision.

    A subclass of ``JsonParseExhausted`` so ``call_with_resample`` retries it and the
    shared degrade runs unchanged on exhaustion; counted APART from unparseable
    replies (``decision_missing_attempts`` / ``decision_missing_kinds`` on the
    message metadata -> the adapter's ``reasoning_decision_missing``), and the kind
    is the per-call record's ``rejection``."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _missing_decision_kind(response: dict[str, Any]) -> str:
    body, text = response["json"], str(response.get("text") or "")
    if "<tool_call>" in text or "function_calls" in body or body.get("type") == "function" \
            or ("name" in body and ("input" in body or "arguments" in body)):
        return "native_tool_call"
    if "properties" in body or (body.get("type") == "object" and "required" in body):
        return "schema_echo"
    return "other"


def require_decision(mode: str, response: dict[str, Any]) -> dict[str, Any]:
    """``response`` unchanged when its reply is a decision; else ``DecisionMissing``.
    Runs after ``require_decision_shape``, inside the re-sampled call, in the
    DECIDING modes only (decide/refine): the two-pass ``initial`` call decides
    nothing and is never rejected here. ``intended_actions: []`` is a decision —
    list-typed — and passes; the adapter runs it as a WAIT like any empty batch."""
    if mode not in GraphReasoningAgent.DECIDING_MODES:
        return response
    kind = _missing_decision_kind(response)
    if kind == "schema_echo" or not isinstance(response["json"].get("intended_actions"), list):
        raise DecisionMissing(f"the reply carries no decision ({kind}): no list-typed intended_actions", kind=kind)
    return response


# The deciding route's name in the per-call record (the adapter files it under
# the "reasoning" stage, as it does every route not starting with "grounding").
TIMELINE_ROUTE = "reasoning_timeline"

# Encoded frames kept for re-sending: the window's peak is 30 images (the opening
# frame, 28 history frames and the current one), plus headroom for the step where
# the sawtooth resets.
_ENCODED_FRAME_CACHE = 32


class OSWorldBackbone(Backbone, Protocol):
    """What OSWorld's adapter and its timeline route need from a backbone, beyond
    the shared ``Backbone`` contract. Documentation, like ``Backbone``: not
    ``runtime_checkable``, and nothing subclasses it. A backbone for a new model is
    its client class with ``TimelineRoute`` placed before it in the bases (as
    ``OSWorldMiniMaxAgent`` is) and the three per-model hooks below implemented.

    FROM THE CLIENT CLASS: the grounding route the profile's action agent calls
    (MiniMax: ``complete_text(grounding=True, stop=[...])``; Qwen and DeepSeek ground
    through their OSWorld subclass's own ``complete_grounding``), the request builders
    ``_base_request(messages, max_tokens)`` and ``_extra_body(use_thinking)``,
    ``_encode_screenshot`` (the route memoises it), ``_create_with_backoff``
    returning an OpenAI chat-completions response (``choices[0].message.content``,
    ``usage.model_dump()``), ``_extract_json_object``, ``_notify_observer`` with a
    ``call_observer`` slot the adapter sets, ``reset_session``, and
    ``system_prompt_suffix``.

    PER-MODEL HOOKS, written in the backbone subclass: ``IMAGE_URL_PREFIX`` (the
    data-URL prefix its encoder's output is sent under), ``_record_response_usage``
    (the backbone's usage counters from one response) and ``_thinking_text`` (the
    thinking the provider returned beside the content, or None).

    COUNTERS the adapter reads for telemetry with ``getattr`` defaults
    (``graphpc_agent.OSWorldGraphPCAgent._collect_token_usage`` / ``_model_name``):
    ``model_name``, ``call_count``, ``total_input_tokens``, ``total_output_tokens``.
    A backbone without them silently records zeros and no model name.
    """

    model_name: str
    call_count: int
    total_input_tokens: int
    total_output_tokens: int
    system_prompt_suffix: str
    call_observer: Optional[Callable[[dict[str, Any]], None]]
    IMAGE_URL_PREFIX: str

    def reset_session(self) -> None: ...

    def _encode_screenshot(self, image_bytes: bytes) -> str: ...

    def _base_request(self, messages: list[dict[str, Any]], max_tokens: Optional[int]) -> dict[str, Any]: ...

    def _extra_body(self, use_thinking: bool) -> dict[str, Any]: ...

    def _create_with_backoff(self, request: dict[str, Any]) -> Any: ...

    def _extract_json_object(self, text: str) -> dict[str, Any]: ...

    def _notify_observer(self, result: dict[str, Any]) -> dict[str, Any]: ...

    def _record_response_usage(self, response: Any) -> None: ...

    def _thinking_text(self, message: Any) -> Optional[str]: ...


class TimelineRoute:
    """The deciding route that sends a pre-rendered timeline, and the per-call
    record, for any backbone. A mixin placed BEFORE the backbone class in the
    bases; what it needs from that class is ``OSWorldBackbone``.

    The request is the shared reasoning route's, part for part — thinking pinned,
    ``response_format=json_object``, usage recorded, the observer notified — except
    that the user message is the renderer's list of text and image parts, and
    nothing is appended to it: the schema is already in the system prompt, and the
    current screen must stay the last content. The backbone's grounding route is
    inherited, and reaches the same memoised frame encoding.

    THE PER-CALL RECORD. Every result handed to the observer carries a
    ``request_record`` built from the request as SENT (captured at the transport):
    the system prompt, the user content with each image replaced by its slot — the
    step label the renderer gave it, the predict call it was observed at, and the
    sha1 of its SOURCE bytes, which is the sha1 of the step PNG the driver saved —
    plus the cached prompt tokens and any thinking the provider returned. That is
    what lets a one-task run PROVE what a deciding call saw, rather than infer it
    from prompt sizes. Diagnostics only: nothing here changes a request byte, and
    the record is built after the call returned.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # A frame is re-sent for up to thirty steps; encoding it once per step would
        # cost ~38.5 ms per frame per call on a 1920x1080 guest frame, ~387 s of CPU
        # over a 500-step episode. Least-recently-USED eviction, because the opening
        # frame is re-sent on every call but inserted first.
        self._encoded_frames: dict[bytes, str] = {}
        # encoded frame -> sha1 of the bytes it was encoded from, for the slot map.
        self._encoded_sources: dict[str, str] = {}
        self._last_request: Optional[dict[str, Any]] = None
        self._last_response: Any = None

    def reset_session(self) -> None:
        super().reset_session()
        self._encoded_frames.clear()
        self._encoded_sources.clear()

    def _encode_screenshot(self, image_bytes: bytes) -> str:
        """The backbone's own encoding, memoised, and remembered against the
        source bytes' sha1. This class sits BEFORE the backbone in the MRO, so the
        backbone's own grounding route reaches this memo too — same bytes out,
        encoded once."""
        frame = bytes(image_bytes)
        cached = self._encoded_frames.pop(frame, None)
        if cached is None:
            cached = super()._encode_screenshot(frame)
            if len(self._encoded_frames) >= _ENCODED_FRAME_CACHE:
                evicted = self._encoded_frames.pop(next(iter(self._encoded_frames)))
                self._encoded_sources.pop(evicted, None)
            self._encoded_sources[cached] = hashlib.sha1(frame).hexdigest()
        self._encoded_frames[frame] = cached  # most recently used goes last
        return cached

    def _create_with_backoff(self, request: dict[str, Any]):
        self._last_request = request
        self._last_response = None
        response = super()._create_with_backoff(request)
        self._last_response = response
        return response

    def _check_reply(self, response: Any) -> None:
        """A backbone's rejection of a reply before it is parsed (a ``JsonParseExhausted``
        carrying ``kind``); none here."""
        return None

    def _notify_observer(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.call_observer is not None and "request_record" not in result:
            try:
                result["request_record"] = self._request_record(labels=None)
            except Exception:  # noqa: BLE001 — diagnostics never fail a scored call
                logger.exception("per-call request record failed; the call result is unaffected")
        return super()._notify_observer(result)

    def _request_record(self, *, labels: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
        """The last request as sent, images replaced by slots. ``labels`` are the
        renderer's, one per image in order; without them (the grounding route) the
        request's single image is the current screen."""
        request = self._last_request or {}
        system = ""
        user: list[dict[str, Any]] = []
        slots: list[dict[str, Any]] = []
        for message in request.get("messages") or []:
            if message.get("role") == "system":
                system = str(message.get("content") or "")
                continue
            content = message.get("content")
            if isinstance(content, str):
                user.append({"type": "text", "text": content})
                continue
            for part in content or []:
                if part.get("type") != "image_url":
                    user.append({"type": "text", "text": part.get("text", "")})
                    continue
                url = (part.get("image_url") or {}).get("url", "")
                encoded = url[len(self.IMAGE_URL_PREFIX):] if url.startswith(self.IMAGE_URL_PREFIX) else url
                label = labels[len(slots)] if labels is not None and len(slots) < len(labels) else {}
                slot = {
                    "slot": len(slots),
                    "label": label.get("label", "current"),
                    "predict_call": label.get("predict_call"),
                    "sha1": self._encoded_sources.get(encoded),
                    "encoded_chars": len(encoded),
                }
                slots.append(slot)
                user.append({"type": "image", **slot})
        response = self._last_response
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        message = response.choices[0].message if response is not None and getattr(response, "choices", None) else None
        thinking = self._thinking_text(message) if message is not None else None
        return {
            "system": system,
            "user": user,
            "image_slots": slots,
            "cached_tokens": int(cached) if isinstance(cached, (int, float)) else None,
            "thinking": thinking,
        }

    def complete_timeline_json(
        self, *, system_prompt: str, user_parts: list[dict[str, Any]], use_thinking: bool = True,
        max_tokens: Optional[int] = None, validate: Optional[Callable[[dict[str, Any]], Any]] = None,
        turns: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        """The deciding call. ``validate`` (the reasoning agent's reply checks) runs on
        the parsed result BEFORE the observer is told, so the per-call record says
        whether this reply was rejected and why (``rejection``); a rejection then
        propagates to the caller's re-sample loop unchanged, carrying the rejected
        reply as ``reply_text``. ``turns`` follow the user message — a profile that
        repairs a rejected reply sends it back with its repair request there, the one
        place anything comes after the current screen."""
        instructions = system_prompt
        if self.system_prompt_suffix:
            instructions = f"{instructions}\n\n{self.system_prompt_suffix}"
        content: list[dict[str, Any]] = []
        for part in user_parts:
            if part["type"] == "image":
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"{self.IMAGE_URL_PREFIX}{self._encode_screenshot(part['image'])}"},
                })
            else:
                content.append({"type": "text", "text": part["text"]})
        req = self._base_request(
            [{"role": "system", "content": instructions}, {"role": "user", "content": content}, *turns], max_tokens,
        )
        req["extra_body"] = self._extra_body(use_thinking)
        req["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        resp = self._create_with_backoff(req)
        self._record_response_usage(resp)
        text = resp.choices[0].message.content or ""
        images = [part for part in user_parts if part["type"] == "image"]
        result: dict[str, Any] = {
            "text": text,
            "raw_response": text,
            "response_output": [],
            "usage": resp.usage.model_dump() if resp.usage else {},
            "diagnostics": {
                "route": TIMELINE_ROUTE,
                "thinking": bool(use_thinking),
                "model": self.model_name,
                "system_chars": len(system_prompt),
                "user_chars": sum(len(part["text"]) for part in user_parts if part["type"] == "text"),
                "image_bytes": len(images[-1]["image"]) if images else 0,
                "image_count": len(images),
                "latency_s": round(time.perf_counter() - started, 3),
            },
        }
        if self.call_observer is not None:
            try:
                result["request_record"] = self._request_record(labels=[
                    {"label": part.get("label"), "predict_call": part.get("predict_call")} for part in images
                ])
            except Exception:  # noqa: BLE001 — diagnostics never fail a scored call
                logger.exception("per-call request record failed; the call result is unaffected")
        # Every reply reaches the record, the rejected ones with the reason — they
        # are the ones a re-sample follows.
        try:
            self._check_reply(resp)
            result["json"] = self._extract_json_object(text)
            if validate is not None:
                validate(result)
        except JsonParseExhausted as exc:
            kind = getattr(exc, "kind", None) or ("wrong_shape" if "json" in result else "unparseable")
            result["rejection"] = {"kind": kind, "detail": str(exc)[:300]}
            self._notify_observer(result)
            exc.reply_text, exc.rejection = text, result["rejection"]
            raise
        result["rejection"] = None
        self._notify_observer(result)
        return result


def call_with_repair(
    call: Callable[[tuple[dict[str, Any], ...]], Any],
    *,
    attempts: int,
    repair: Callable[[Exception], str],
    on_retry: Callable[[int, BaseException], None],
) -> tuple[Any, int]:
    """``call_with_resample``'s contract with a REPAIR in place of the identical
    re-sample: ``call(turns)`` is invoked up to ``attempts`` times, first with no turns,
    then with the rejected reply's content (``exc.reply_text``) as an assistant turn and
    ``repair(exc)`` as a user turn — each after the ORIGINAL request, never accumulated
    (the control's ``recovery.repair_payload``). Returns ``(result, repairs)``. On
    exhaustion, or at once for a ``LENGTH_EMPTY`` rejection, the last exception is
    re-raised with ``exc.resample_attempts`` = the repairs made."""
    turns: tuple[dict[str, Any], ...] = ()
    for attempt in range(attempts):
        try:
            return call(turns), attempt
        except JsonParseExhausted as exc:
            if getattr(exc, "kind", None) == LENGTH_EMPTY or attempt + 1 >= attempts:
                exc.resample_attempts = attempt  # type: ignore[attr-defined]
                raise
            on_retry(attempt + 1, exc)
            turns = (
                {"role": "assistant", "content": [{"type": "text", "text": getattr(exc, "reply_text", "")}]},
                {"role": "user", "content": [{"type": "text", "text": repair(exc)}]},
            )
    raise AssertionError("unreachable")  # pragma: no cover


class OSWorldMiniMaxAgent(TimelineRoute, MiniMaxAgent):
    """``MiniMaxAgent`` with the timeline route (``TimelineRoute``) and MiniMax's
    three hooks: frames go out as JPEG data URLs (the shared encoder's output),
    usage comes off ``response.usage``, and thinking comes out of the
    ``reasoning_split`` fields. The constructor is ``MiniMaxAgent``'s."""

    IMAGE_URL_PREFIX = "data:image/jpeg;base64,"

    def _record_response_usage(self, response: Any) -> None:
        self._record_usage(response.usage)

    @staticmethod
    def _thinking_text(message: Any) -> Optional[str]:
        """The thinking the provider returned beside the content, if any. With
        ``reasoning_split`` MiniMax separates it into ``reasoning_content`` or
        ``reasoning_details`` (backbone/minimax/main.py); the shared backbone never
        reads it. Best-effort: a shape this does not know reads as None."""
        text = getattr(message, "reasoning_content", None)
        if isinstance(text, str) and text:
            return text
        details = getattr(message, "reasoning_details", None)
        if isinstance(details, list):
            pieces = [d.get("text") if isinstance(d, dict) else getattr(d, "text", None) for d in details]
            joined = "".join(piece for piece in pieces if isinstance(piece, str))
            return joined or None
        return None


class OSWorldReasoningAgent(GraphReasoningAgent):
    """``GraphReasoningAgent`` whose deciding request is the episode timeline.

    Constructed only by ``OSWorldGraphPCAgent.build_default``, with the
    ``EpisodeTimeline`` the adapter writes. ``reason`` is overridden whole because
    the request has a different SHAPE (system + a list of parts, sent through the
    backbone's ``complete_timeline_json``, ``TimelineRoute``); what it does with the reply —
    the re-sample budget, the degrade to an empty decision, the message the
    orchestrator reads — is the shared agent's, kept identical below.

    It never reads ``recent_history``: the timeline is built from the adapter's
    record of what was EXECUTED, so ``max_trajectory_length`` and the base
    ``_build_user_prompt`` are not on this path at all.

    ``repair`` is the model profile's (``graphpc_agent.OSWorldModelProfile``): None
    re-samples an unusable reply with the identical request (M3's control); otherwise
    the reply is REPAIRED as the profile's control repairs it — the original request,
    the reply's content as an assistant turn and ``repair(exc)`` as a user turn, within
    the same attempt budget (``call_with_repair``).

    ``requirement_ledger`` and ``step_budget_line`` are the profile's too, off by
    default and set by the DeepSeek profile only: the ledger adds its instruction block
    to the system prompt, its field to the schema, its block to the STATE text and its
    reply field to the message the orchestrator reads; the budget line (a template over
    ``{step}`` and ``{max_steps}``, ``max_steps`` the arm's budget) is one line of the
    STATE text. Off, the request carries none of them.
    """

    def __init__(self, *, timeline: EpisodeTimeline, repair: Optional[Callable[[Exception], str]] = None,
                 requirement_ledger: bool = False, step_budget_line: Optional[str] = None,
                 max_steps: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.timeline = timeline
        self.repair = repair
        self.requirement_ledger = bool(requirement_ledger)
        if step_budget_line is not None and max_steps is None:
            raise ValueError("a step budget line needs the arm's max_steps")
        self.step_budget_line = step_budget_line
        self.max_steps = max_steps

    def _deciding_system_prompt(self, mode: str, *, environment_brief: str) -> str:
        """Static for the episode (the brief changes only with the task), so it is a
        stable cacheable prefix. The control's coordinate line (upstream :49) is NOT
        here: it has no measured effect on the deciding call, grounding — the call that
        emits coordinates — carries it verbatim
        (action_minimax.py:M3_GROUNDING_SYSTEM_PROMPT), and coordinates never cross
        the handoff (strip_grounding_coordinates)."""
        sections = [super()._system_prompt(mode), _FACTS_INSTRUCTION]
        if self.requirement_ledger:
            sections.append(OSWORLD_REQUIREMENTS_INSTRUCTION)
        if environment_brief:
            sections.append(f"ENVIRONMENT: {environment_brief}")
        return "\n\n".join(sections)

    @staticmethod
    def _schema_block(json_schema: dict) -> str:
        """The reply instruction and the schema, identical at every step, placed
        right before the CURRENT SCREEN header so the frame stays the last content."""
        return ("Return a single JSON object conforming to this schema (JSON only, no prose):\n"
                + json.dumps(json_schema))

    def _schema_for_mode(self, mode: str) -> tuple[dict, str]:
        """The shared schema (vocabulary applied) plus the required ``facts`` field, and
        the required ``requirements`` field on a profile with the requirement ledger.
        A copy: the shared module constants stay the ones WebArena sends."""
        schema, name = super()._schema_for_mode(mode)
        schema = copy.deepcopy(schema)
        schema["properties"]["facts"] = _FACTS_SCHEMA
        schema["required"] = [*schema["required"], "facts"]
        if self.requirement_ledger:
            schema["properties"]["requirements"] = REQUIREMENTS_SCHEMA
            schema["required"] = [*schema["required"], "requirements"]
        return schema, name

    def _state(self, reasoning_input: ReasoningInput) -> dict[str, Any]:
        """The dynamic state the step history does not carry. The facts ledger is
        rendered here and nowhere else in the request."""
        return {
            "current_stage": reasoning_input.runtime_state.current_stage,
            "plan": self._serialize_plan(reasoning_input.plan),
            "graph_summary": reasoning_input.runtime_state.content.get("graph_summary") or "",
            "facts": self.timeline.facts_for_prompt(),
            "retrieved_memory": self._serialize_retrieved_memory(reasoning_input.retrieved_memory),
        }

    def render_request(self, reasoning_input: ReasoningInput) -> tuple[str, list[dict[str, Any]]]:
        """``(system_prompt, user_parts)`` for this deciding call, without calling
        the model — split out so a check can read exactly what would be sent."""
        mode = reasoning_input.reasoning_mode
        json_schema, _ = self._schema_for_mode(mode)
        observation = reasoning_input.observation.content \
            if isinstance(reasoning_input.observation.content, dict) else {}
        current = self.timeline.current_predict_call
        screenshot = observation.get("screenshot")
        if screenshot is not None and self.timeline.frames.get(current) != screenshot:
            # The frame the adapter recorded for this step IS the observation being
            # decided on; anything else is a misaligned timeline, and a misaligned
            # timeline shows the model a screen it is not looking at.
            raise RuntimeError(
                f"the timeline's current frame (predict call {current}) is not the observation's screenshot"
            )
        system_prompt = self._deciding_system_prompt(mode, environment_brief=observation.get("environment_brief") or "")
        budget_line = (self.step_budget_line.format(step=current, max_steps=self.max_steps)
                       if self.step_budget_line is not None else None)
        requirements = (self.timeline.requirements_for_prompt(reasoning_input.runtime_state.content.get("requirements"))
                        if self.requirement_ledger else None)
        user_parts = self.timeline.render(
            instruction=reasoning_input.instruction,
            state_text=render_state_text(current, self._state(reasoning_input), budget_line=budget_line,
                                         requirements=requirements) + "\n\n"
            + self._schema_block(json_schema),
        )
        return system_prompt, user_parts

    def reason(self, reasoning_input: ReasoningInput) -> AgentMessage:
        mode = reasoning_input.reasoning_mode
        # Rendered ONCE and closed over: every re-sample sends the identical request
        # (common/resample.py).
        system_prompt, user_parts = self.render_request(reasoning_input)
        # Attempts rejected as DecisionMissing (not unparseable), by kind, for the telemetry.
        decision_missing: list[str] = []

        def on_retry(attempt: int, exc: BaseException) -> None:
            if isinstance(exc, DecisionMissing):
                decision_missing.append(exc.kind)
            logger.warning("reasoning output unusable (%s mode, attempt %d): %s; re-sampling", mode, attempt, exc)

        try:
            if self.repair is None:
                response, resamples = call_with_resample(
                    lambda: self.backbone.complete_timeline_json(
                        system_prompt=system_prompt, user_parts=user_parts,
                        validate=lambda result: require_decision(mode, require_decision_shape(result)),
                    ),
                    attempts=self.model_call_attempts,
                    on_retry=on_retry,
                )
            else:
                response, resamples = call_with_repair(
                    lambda turns: self.backbone.complete_timeline_json(
                        system_prompt=system_prompt, user_parts=user_parts,
                        validate=lambda result: require_decision(mode, require_decision_shape(result)),
                        turns=turns,
                    ),
                    attempts=self.model_call_attempts,
                    repair=self.repair,
                    on_retry=on_retry,
                )
        except JsonParseExhausted as exc:
            # The shared degrade (GraphReasoningAgent.reason): an empty decision, a
            # noop step, the marker surfaced to the next step.
            if isinstance(exc, DecisionMissing):
                decision_missing.append(exc.kind)
            reason = DECISION_MISSING_REASON if isinstance(exc, DecisionMissing) else "reasoning output unparseable"
            logger.warning("%s (%s mode): %s; degrading to a noop step", reason, mode, exc)
            message = empty_decision_message(mode, reason=reason)
            message.metadata["resample_attempts"] = getattr(exc, "resample_attempts", 0)
            message.metadata["decision_missing_attempts"] = len(decision_missing)
            message.metadata["decision_missing_kinds"] = decision_missing
            return message
        message = self._decision_message(mode, response, resamples)
        if self.requirement_ledger:
            # The reply's ledger as it came (the orchestrator normalises, stores and judges it), and a graph_update on
            # every reply, so the orchestrator's graph hook runs on each one and the completion gate judges the
            # claim's own ledger. A reply without a ledger carries none here: never a reason to re-sample.
            if "requirements" in response["json"]:
                message.content["requirements"] = response["json"]["requirements"]
            message.content.setdefault("graph_update", {})
        message.metadata["decision_missing_attempts"] = len(decision_missing)
        message.metadata["decision_missing_kinds"] = decision_missing
        return message

    @staticmethod
    def _decision_message(mode: str, response: dict[str, Any], resamples: int) -> AgentMessage:
        """The reply as the orchestrator reads it — GraphReasoningAgent.reason's
        extraction, key for key, plus this benchmark's ``facts``."""
        response_json = response["json"]
        content: dict[str, Any] = {"reasoning_text": response_json.get("reasoning_text", "")}
        plan_payload = response_json.get("plan")
        if plan_payload is not None:
            content["plan"] = plan_payload
        for key in ("actual_effect_class", "next_intent", "intended_actions", "retrieval_request"):
            if key in response_json:
                content[key] = response_json[key]
        # This benchmark's addition, read by the adapter into the facts ledger. The
        # orchestrator ignores it and compaction drops it from recent_history, so it
        # reaches the next request through the ledger alone.
        if "facts" in response_json:
            content["facts"] = response_json["facts"]
        if "graph_summary" in response_json:
            content["graph_update"] = {"graph_summary": response_json["graph_summary"]}
            content["graph_summary"] = response_json["graph_summary"]
        return AgentMessage(
            sender="reasoning",
            message_type="plan" if plan_payload is not None else "local_reasoning",
            content=content,
            raw={"gpt_response": response["raw_response"], "response_json": response_json},
            metadata={"reasoning_mode": mode, "resample_attempts": resamples},
        )
