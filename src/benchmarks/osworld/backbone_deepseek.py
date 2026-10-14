"""The DeepSeek (``deepseek-flash``, V4.1-Flash over https://api.deepseek.com) backbone
on OSWorld 2.0: ``DeepSeekAgent`` with the timeline route (``reasoning.TimelineRoute``),
screenshot messages, a native forced-function grounding route and its own transport.

WHY A SUBCLASS HERE. ``src/agents/backbone/deepseek/main.py`` is the WebArena DeepSeek
backbone and stays unchanged for it; everything this benchmark does differently lives in
OSWorld's own modules, which only OSWorld's ``build_default`` constructs (the same rule
as ``backbone_qwen.py``).

THE TWO ROUTES (both on DeepSeek's OpenAI-compatible endpoint):

  * DECIDING (``TimelineRoute.complete_timeline_json``): thinking ON at the pinned
    effort ``high`` (the WebArena backbone's ``_set_thinking`` form; here the effort rides
    in ``_extra_body``, because the timeline route sets only ``extra_body`` — the wire
    body is the same, the SDK merges ``extra_body`` into it), ``response_format``
    ``json_object``, no ``max_tokens`` unless the runner passes one, no sampling
    parameters (thinking mode ignores them). The thinking the provider returns beside
    the content (``message.reasoning_content``) is the per-call record's ``thinking``.
  * GROUNDING (``complete_grounding``): NON-thinking, a FORCED native function call —
    ``tools=[the computer tool]``, ``tool_choice`` naming it — the grounding sampling
    pin (``grounding_temperature``; the profile passes 0.6), no ``max_tokens``, no
    ``strict``; the reply is ``tool_calls[0].function.arguments``, a JSON object, handed to the
    action agent's validation; a second tool call is counted (``tool_calls_dropped``),
    never executed. A call NAMED AFTER ITS ACTION — the function named ``left_click``
    rather than ``computer``, no ``action`` in its arguments, named after the decided
    class — is taken as the ``computer`` call with that action (``function_name_as_action``)
    when the name is one of the declared tool's ``action`` values; the per-call record
    says so (``action_from_function_name``), and the action agent's rules then apply
    to the action as to one written in the arguments. This is the WebArena DeepSeek
    backbone's own grounding shape (``_invoke_grounding``: non-thinking + a forced
    named function) carried to OSWorld with the ``computer`` tool; the M3 text grammar
    is not used on this model (it does not follow it).
    A SURPLUS CLOSER in the arguments — ``{"action": "left_click", "coordinate": [25,
    11]]}``, typically a single ``]`` before the final ``}``, near-deterministic per
    target, so re-sampling repeats it — is repaired, at both places the arguments are read (the ``computer``
    call and ``function_name_as_action``), only after ``json.loads`` has failed:
    ``repair_surplus_closers`` removes the closing brackets and braces that have no
    opener to close, when all of them stand in the trailing run of closers (so no key
    or value is moved between containers), and the text is parsed again; a JSON object
    then goes through every rule as any other reply. Nothing else is repaired. The
    per-call record says so (``arguments_repaired``: the removed characters and their
    positions, or None) on every reply whose arguments were read; ``reply_text`` stays
    the arguments as they came.

FRAMES go out as the MiniMax control encodes them (``MiniMaxAgent._encode_screenshot``:
JPEG quality 90 at the native size), as ``data:image/jpeg;base64,`` content parts, image
part before text part, images in user turns only.

TRANSPORT (modelled on ``backbone_qwen.py``, the one OSWorld transport with the lost-
episode semantics this profile uses). The client is built with ``max_retries=0`` and the
profile's request timeout (500 s), and this class's own retry loop is the ONLY one, so
``transport_attempts`` is exact: ``MAX_ATTEMPTS`` = 15 attempts — PARITY WITH THE
CONTROL'S EXPOSURE: the control's Anthropic client makes 4 SDK
retries inside each of M3's 3 attempts, 15 HTTP attempts per step, after which its step
is a no-op; here the 15th failure is infrastructure and the episode is re-run (a
disclosed asymmetry) — sleep ``min(15, 3a)`` s after failed attempt ``a``,
retrying ``APIConnectionError`` (``APITimeoutError`` is one), ``RateLimitError`` and
``InternalServerError`` (every 5xx), and an HTTP 200 the SDK cannot decode
(``UndecodableResponseError``); exhausted, all of these are infrastructure
(``is_infra_error``). Not the Qwen transport's 50: that mirrored the OpenRouter
control's own client. An HTTP 400 in no known form, or a 200 carrying a
body-level ``error``, is an UNRECOGNISED provider failure (``failure_kind``): its
attempts within one call are counted apart, up to ``UNRECOGNISED_ATTEMPTS`` (3), then
raised as ``UnrecognisedProviderError``, infrastructure, its text carrying the
provider's in full. Raised at once, never retried: 401, 402, 404 and 408
(infrastructure: the account or the route) and 403, 413, 422 and a 400 whose text names
a context overflow (the agent's: our request). Any other status leaves unwrapped and is
the agent's. The WebArena backbone's ``_DS_RETRY_POLICY`` is never called.

WHAT IS NOT CARRIED FROM THE WEBARENA DEEPSEEK BACKBONE: its "a failed call is
infrastructure and the task is excluded" rule (``DeepSeekCallError``, raised by
``_validate_response``). On OSWorld a failed call costs a step on both arms,
so no path here calls ``_validate_response``: a 200 with no choices, a provider-
interrupted finish (``insufficient_system_resource`` / ``aborted``), a ``content_filter``
finish, a ``length`` finish with nothing in it and an empty content are REJECTION KINDS
(``NO_CHOICES``, ``INTERRUPTED``, ``CONTENT_FILTER``, ``LENGTH_EMPTY``,
``EMPTY_CONTENT``; on grounding also ``NO_TOOL_CALL`` and ``UNPARSEABLE``), recorded on
the per-call record (``rejection``) and re-sampled within the profile's attempts — the
deciding route through ``call_with_resample`` (``DeepSeekReplyRejected`` is a
``JsonParseExhausted``), grounding through the action agent's loop
(``GroundingReplyRejected``) — then a counted WAIT at the adapter. ``NO_CHOICES`` is the
one kind raised by the transport itself, because the timeline route indexes
``choices[0]`` before it asks ``_check_reply``; ``OSWorldDeepSeekAgent.complete_timeline_json``
catches it around the shared route and records it like any other rejection, so every
attempt of a deciding chain is on the per-call record.

EVERY RESULT ALSO CARRIES THE REPLY'S PROVENANCE: ``finish_reason``, the
response's ``model`` echo (``response_model``: the id that was SENT is the record's
``model``) and ``system_fingerprint`` (the alias-move detector), read off the last response
the transport received, no-choices and in-band replies included. The adapter copies the
three keys only when a result carries them, so the MiniMax and Qwen records are unchanged.

The class has no ``complete_text`` route: the action agent calls ``complete_grounding``,
the deciding route ``complete_timeline_json``. An ``openai.APIError``
that is neither a status nor a connection error (the SDK's ``APIResponseValidationError``
on a body it cannot validate) is not in the transport's retried classes: it leaves raw and
``predict`` ends the episode as an agent error — disclosed, not handled.

METERING, EXACTLY ONCE PER HTTP RESPONSE: the transport loop runs the WebArena
backbone's ``_record_usage`` (its ``deepseek usage:`` line, the cache split, the finish
reason, the model echo and ``system_fingerprint``) on every response it receives — an
in-band failure and a no-choices reply included, before they are raised — and
``_record_response_usage``, the hook ``TimelineRoute`` calls after the call, is a no-op.

EXPOSURE, ON THE PER-CALL RECORD: ``transport_attempts``, ``transport_last_error`` and
``transport_unrecognised`` ride on every result this backbone hands its observer, as on
the Qwen profile.

MRO. ``OSWorldDeepSeekAgent(TimelineRoute, _DeepSeekOSWorldTransport)``: the transport
sits BELOW ``TimelineRoute``, so the route's request bookkeeping and frame memo wrap it.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any, Callable, Optional

import openai
from PIL import Image

from src.agents.backbone.deepseek.main import (
    _FLASH_MODEL,
    _FLASH_REASONING_EFFORT,
    _INTERRUPTED_FINISH_REASONS,
    DeepSeekAgent,
    _reply_is_empty,
)
from src.agents.common.json_utils import JsonParseExhausted
from src.agents.retrieval.embedding import TextArmUnavailableError

from .reasoning import LENGTH_EMPTY, TIMELINE_ROUTE, TimelineRoute
from .timeline import ScreenshotCaptureFailed


logger = logging.getLogger("desktopenv.agent")

# The transient transport classes, as they reach the openai client; APITimeoutError is an
# APIConnectionError, InternalServerError is every 5xx.
_RETRYABLE_EXCEPTIONS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)

# Statuses raised at once (carried from the Qwen profile's table): 401 / 402 (the
# account), 404 / 408 (the route) are infrastructure; 403 / 413 / 422 are properties of our request.
_INFRA_AT_ONCE = frozenset({401, 402, 404, 408})
_AGENT_AT_ONCE = frozenset({403, 413, 422})
# What is_infra_error counts as infrastructure among statuses: the at-once ones plus 429, which the
# loop retries and which is infrastructure once exhausted.
_INFRA_STATUS_CODES = _INFRA_AT_ONCE | {429}

# A 400 that names a context overflow is the agent's (our request), at once. The substrings are
# the Qwen control's (qwen_upstream/recovery.py:looks_like_context_error); DeepSeek's own overflow
# text has not been observed on this endpoint, so the classification rests on them.
_CONTEXT_OVERFLOW_FRAGMENTS = ("context length", "context_length", "maximum context", "prompt is too long",
                               "too many tokens")

# The rejection kinds (the per-call record's ``rejection.kind``); LENGTH_EMPTY is reasoning.py's.
NO_CHOICES = "no_choices"
INTERRUPTED = "interrupted"
CONTENT_FILTER = "content_filter"
EMPTY_CONTENT = "empty_content"
NO_TOOL_CALL = "no_tool_call"
UNPARSEABLE = "unparseable"

# The per-call record's mark on a grounding reply whose action came from its function's name (function_name_as_action):
# True or False on every reply whose arguments were read.
FUNCTION_NAME_FIELD = "action_from_function_name"
# The per-call record's mark on a grounding reply whose arguments parsed only after the surplus-closer repair
# (repair_surplus_closers): {"removed": [characters], "positions": [indices in the reply's arguments]}, or None, on
# every reply whose arguments were read.
ARGUMENTS_REPAIRED_FIELD = "arguments_repaired"
# The repair as the run's active_config records it (action_deepseek's method surface carries it).
ARGUMENTS_REPAIR_RULE = ("when the arguments of a grounding tool call are not JSON, the closing brackets and braces "
                         "that close no opener (none open, or the innermost open one is of the other kind; string "
                         "literals skipped) are removed when every one of them stands in the trailing run of closers, "
                         "and the text is parsed again; only a JSON object is taken, and it meets every rule a reply "
                         "meets; nothing else is repaired")

# The route labels of the per-call record; the adapter files every route not starting with
# "grounding" under the reasoning stage.
GROUNDING_ROUTE = "grounding_tool"
# The reply provenance keys every result carries (module docstring), before any response arrives.
_NO_REPLY: dict[str, Any] = {"finish_reason": None, "response_model": None, "system_fingerprint": None}


_OPENER_OF = {"]": "[", "}": "{"}


def repair_surplus_closers(text: str) -> Optional[tuple[str, dict[str, Any]]]:
    """``(text without its surplus closers, {"removed": [characters], "positions": [indices]})`` (module
    docstring), or None when ``text`` has no surplus closer or one stands before its trailing run of closers. A
    closer is surplus when no bracket or brace is open, or the innermost open one is of the other kind; string
    literals (escapes honoured) are skipped, and nothing is added."""
    stack: list[str] = []
    surplus: list[int] = []
    in_string = escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            if stack and stack[-1] == _OPENER_OF[character]:
                stack.pop()
            else:
                surplus.append(index)
    # The trailing run: the closers (and white space) after the last other character. A closer removed before it
    # would move the keys and values that follow it into another container.
    tail = len(text)
    while tail and text[tail - 1] in "]} \t\r\n":
        tail -= 1
    if not surplus or surplus[0] < tail:
        return None
    removed = set(surplus)
    repaired = "".join(character for index, character in enumerate(text) if index not in removed)
    return repaired, {"removed": [text[index] for index in surplus], "positions": surplus}


def load_tool_arguments(reply_text: str) -> tuple[Any, Optional[dict[str, Any]]]:
    """``(the tool call's arguments parsed, the repair record or None)``: ``json.loads``, and when that fails the
    surplus-closer repair (``repair_surplus_closers``), taken only when the repaired text is a JSON object; else the
    first parse's error is raised."""
    try:
        return json.loads(reply_text), None
    except (TypeError, ValueError) as error:
        first_error = error
    repair = repair_surplus_closers(reply_text) if isinstance(reply_text, str) else None
    if repair is not None:
        try:
            repaired = json.loads(repair[0])
        except ValueError:
            repaired = None
        if isinstance(repaired, dict):
            return repaired, repair[1]
    raise first_error


def function_name_as_action(name: Any, reply_text: str,
                            tool: dict[str, Any]) -> Optional[tuple[dict[str, Any], Optional[dict[str, Any]]]]:
    """``(the arguments of a call named after its action as the declared tool's call, the repair record or
    None)``: ``{"action": name, **the rest}`` when ``name`` is one of ``tool``'s ``action`` values (its
    parameter schema's enum) and the arguments are a JSON object (``load_tool_arguments``, read only after the
    name passed) with no ``action`` (absent, null or blank); else None, and the call is rejected as any other
    wrongly named one. An ``action`` beside the name is not reconciled with it."""
    properties = (tool.get("parameters") or {}).get("properties") or {}
    if name not in ((properties.get("action") or {}).get("enum") or ()):
        return None
    try:
        arguments, repaired = load_tool_arguments(reply_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(arguments, dict):
        return None
    action = arguments.get("action")
    if action is not None and not (isinstance(action, str) and not action.strip()):
        return None
    return {"action": name, **{key: value for key, value in arguments.items() if key != "action"}}, repaired


def reply_provenance(response: Any) -> dict[str, Any]:
    """``finish_reason`` (of the first choice; None without choices), the response's ``model`` echo and
    its ``system_fingerprint``, as the response carries them."""
    choices = getattr(response, "choices", None) or []
    return {
        "finish_reason": getattr(choices[0], "finish_reason", None) if choices else None,
        "response_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
    }


class UndecodableResponseError(RuntimeError):
    """An HTTP 200 the SDK could not turn into a completion, at the provider call itself: a JSON
    content type whose body does not decode (the SDK's ``json.JSONDecodeError``), or a non-JSON
    content type, for which ``create()`` returns the body as a ``str``. Transient, retried in the
    transport's loop, infrastructure once exhausted (the Qwen profile's class). Its text
    starts with ``UNDECODABLE_PREFIX``."""


class UnrecognisedProviderError(RuntimeError):
    """An HTTP 400, or a 200 carrying a body-level ``error``, in no known form (``failure_kind``),
    still failing after the transport's ``UNRECOGNISED_ATTEMPTS``: infrastructure, so an unseen
    provider message is never scored as the agent's (the Qwen profile's class). Its text is
    ``UNRECOGNISED_PREFIX``, the original exception's type and the provider's failure in full."""


UNDECODABLE_PREFIX = f"{UndecodableResponseError.__name__}: "
UNRECOGNISED_PREFIX = f"{UnrecognisedProviderError.__name__}: "


class InBandProviderError(RuntimeError):
    """A 200 whose body carries an ``error`` object (the SDK builds a completion around it). Not
    documented on this endpoint; treated as an unrecognised failure (``failure_kind``). ``error`` is
    the object as the response carried it."""

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(f"the provider's HTTP 200 carries an error: {json.dumps(error, ensure_ascii=False)[:500]}")
        self.error = error


class DeepSeekReplyRejected(JsonParseExhausted):
    """A reply the deciding route cannot use (``reply_rejection``), or a 200 with no choices
    (raised by the transport, any route). A ``JsonParseExhausted`` so ``call_with_resample``
    re-samples it and the route's rejection path records it (``kind``)."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class GroundingReplyRejected(ValueError):
    """A grounding reply that cannot become the one action asked for. ``kind`` names why (the
    per-call record's ``rejection``); the message is the text a repair turn quotes."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:  # noqa: BLE001 — a shape this does not know reads as empty
            return {}
    return {}


def _status_of(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None) if isinstance(exc, openai.APIStatusError) else None
    return int(status) if isinstance(status, int) else None


def looks_like_context_overflow(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(fragment in text for fragment in _CONTEXT_OVERFLOW_FRAGMENTS)


def failure_kind(exc: BaseException) -> str:
    """How the transport treats a failure it catches (module docstring): ``"transient"`` (retried for
    the full budget), ``"unrecognised"`` (retried for ``UNRECOGNISED_ATTEMPTS``, then
    ``UnrecognisedProviderError``), ``"infra"`` or ``"agent"`` (raised at once)."""
    if isinstance(exc, (*_RETRYABLE_EXCEPTIONS, UndecodableResponseError)):
        return "transient"
    if isinstance(exc, InBandProviderError):
        return "unrecognised"  # a 200 carrying an error object: no known form
    status = _status_of(exc)
    if status in _INFRA_AT_ONCE:
        return "infra"
    if status in _AGENT_AT_ONCE:
        return "agent"
    if status == 400:
        return "agent" if looks_like_context_overflow(exc) else "unrecognised"
    return "agent"


def reply_rejection(response: Any) -> Optional[tuple[str, str]]:
    """``(kind, detail)`` when a reply with choices cannot be used by either route, else None:
    a provider-interrupted finish, a ``content_filter`` finish, a ``length`` finish with
    nothing in it (no content and no tool-call arguments). An empty content with a tool
    call is a normal grounding reply, so ``EMPTY_CONTENT`` is the deciding route's own
    check (``OSWorldDeepSeekAgent._check_reply``)."""
    choices = getattr(response, "choices", None) or []
    choice = choices[0]
    finish = getattr(choice, "finish_reason", None)
    message = getattr(choice, "message", None)
    if finish in _INTERRUPTED_FINISH_REASONS:
        return INTERRUPTED, f"the provider interrupted the generation: finish_reason={finish!r}"
    if finish == "content_filter":
        return CONTENT_FILTER, "the provider's content filter ended the generation"
    if finish == "length" and _reply_is_empty(message):
        return LENGTH_EMPTY, "the reply reached the output cap before any content"
    return None


def _encode_jpeg_q90(image_bytes: bytes) -> str:
    """Raw screenshot bytes -> base64 of a JPEG quality-90 re-encode at the native size
    (RGB-converted; ``optimize`` on): ``MiniMaxAgent._encode_screenshot``, the encoding
    OSWorld-V2's M3 agent uses, which the DeepSeek control (M3 with the native-tool deltas)
    keeps."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class _DeepSeekOSWorldTransport(DeepSeekAgent):
    """``DeepSeekAgent`` with screenshot messages, the grounding sampling pin, the observer
    hooks and the transport of the module docstring. Below ``TimelineRoute`` in
    ``OSWorldDeepSeekAgent``'s MRO."""

    IMAGE_URL_PREFIX = "data:image/jpeg;base64,"
    MAX_ATTEMPTS = 15  # the control's 4 SDK retries x M3's 3 attempts (module docstring)
    UNRECOGNISED_ATTEMPTS = 3
    RETRY_SLEEP_BASE_S = 3.0
    RETRY_SLEEP_CAP_S = 15.0

    def __init__(self, *args: Any, grounding_temperature: Optional[float] = None, request_timeout: float,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Grounding-route-only sampling pin (MiniMaxAgent's slot); None = send nothing.
        self.grounding_temperature = grounding_temperature
        self.request_timeout = float(request_timeout)
        # Per-call observer (the OSWorld adapter): the completed result dict of EVERY model call,
        # after the call returned. Diagnostics only; None = nobody listening.
        self.call_observer: Optional[Callable[[dict[str, Any]], None]] = None
        # The last call's HTTP attempts, the error of its last failed attempt (None) and its count
        # of attempts that failed in no known form.
        self._transport: dict[str, Any] = {"transport_attempts": 0, "transport_last_error": None,
                                           "transport_unrecognised": 0}
        # The last response's provenance (module docstring), None until one arrives.
        self._reply: dict[str, Any] = dict(_NO_REPLY)

    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.api_key)
        # DeepSeekAgent's key rule (explicit, read at call time), with no SDK retries (this class's
        # loop is the only one) and the profile's timeout.
        api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
        return openai.OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.request_timeout, max_retries=0)

    def _retry_sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    @staticmethod
    def _route_of(request: dict[str, Any]) -> str:
        """The usage line's route, read off the request sent: a forced tool is grounding."""
        return "grounding" if request.get("tools") else "deciding"

    @staticmethod
    def _wire_view(request: dict[str, Any]) -> dict[str, Any]:
        """The request as the SDK serialises it — ``extra_body``'s keys merged into the body — so
        the usage line reads the effort that was sent (``_record_usage`` reads it top-level)."""
        view = dict(request)
        view.update({key: value for key, value in (request.get("extra_body") or {}).items() if key != "thinking"})
        return view

    def _create_with_backoff(self, request: dict[str, Any]):
        self._transport = {"transport_attempts": 0, "transport_last_error": None, "transport_unrecognised": 0}
        self._reply = dict(_NO_REPLY)
        unrecognised = 0
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            self._transport["transport_attempts"] = attempt
            try:
                try:
                    response = self._build_client().chat.completions.create(**request)
                except json.JSONDecodeError as decode_error:
                    raise UndecodableResponseError(
                        f"{UNDECODABLE_PREFIX}the provider's HTTP 200 body is not JSON: {decode_error}") from decode_error
                if isinstance(response, (str,)):
                    raise UndecodableResponseError(f"{UNDECODABLE_PREFIX}the provider's HTTP 200 carries a "
                                                   f"non-JSON content type ({len(response)} characters)")
                # Meter first (a billed reply still costs), then judge it.
                self._record_usage(getattr(response, "usage", None), route=self._route_of(request),
                                   request=self._wire_view(request), response=response)
                self._reply = reply_provenance(response)
                error = _as_mapping(getattr(response, "error", None))
                if error:
                    raise InBandProviderError(error)
                if not getattr(response, "choices", None):
                    rejected = DeepSeekReplyRejected("the reply carries no choices", kind=NO_CHOICES)
                    rejected.usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
                    raise rejected
                return response
            except (*_RETRYABLE_EXCEPTIONS, UndecodableResponseError, InBandProviderError,
                    openai.APIStatusError) as exc:
                kind = failure_kind(exc)
                if kind in ("agent", "infra"):
                    raise
                if kind == "unrecognised":
                    unrecognised += 1
                    self._transport["transport_unrecognised"] = unrecognised
                    self._transport["transport_last_error"] = UnrecognisedProviderError.__name__
                    if unrecognised == self.UNRECOGNISED_ATTEMPTS or attempt == self.MAX_ATTEMPTS:
                        raise UnrecognisedProviderError(
                            f"{UNRECOGNISED_PREFIX}an HTTP 400 or in-band failure in no known form "
                            f"({type(exc).__name__}): {exc}") from exc
                else:
                    self._transport["transport_last_error"] = type(exc).__name__
                    if attempt == self.MAX_ATTEMPTS:
                        raise
                logger.warning("deepseek API error (attempt %d/%d): %s", attempt, self.MAX_ATTEMPTS, exc)
                self._retry_sleep(min(self.RETRY_SLEEP_CAP_S, self.RETRY_SLEEP_BASE_S * attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def _notify_observer(self, result: dict[str, Any]) -> dict[str, Any]:
        """Every result the observer sees carries this call's transport exposure and the reply's
        provenance (module docstring)."""
        result.update(self._transport)
        result.update(self._reply)
        if self.call_observer is not None:
            try:
                self.call_observer(result)
            except Exception:  # noqa: BLE001 — diagnostics never fail a scored call
                logger.exception("deepseek call observer failed; the call result is unaffected")
        return result

    def _diagnostics(
        self, *, route: str, thinking: bool, system_prompt: str, user_prompt: str, observation: Any, started: float,
    ) -> dict[str, Any]:
        """The per-call record's request-side facts (sizes, not content) and the wall latency of
        the provider call (MiniMaxAgent's)."""
        screenshot = self._screenshot_bytes(observation)
        return {
            "route": route,
            "thinking": bool(thinking),
            "model": self.model_name,
            "system_chars": len(system_prompt),
            "user_chars": len(user_prompt),
            "image_bytes": len(screenshot) if screenshot else 0,
            "latency_s": round(time.perf_counter() - started, 3),
        }

    @staticmethod
    def _screenshot_bytes(observation: Any) -> Optional[bytes]:
        """``observation.content["screenshot"]`` (an ``Observation``) or ``observation["screenshot"]``
        (a bare dict); None when absent — the lookup every backbone uses."""
        if observation is None:
            return None
        if hasattr(observation, "content") and isinstance(observation.content, dict):
            return observation.content.get("screenshot")
        if isinstance(observation, dict):
            return observation.get("screenshot")
        return None

    @staticmethod
    def _encode_screenshot(image_bytes: bytes) -> str:
        return _encode_jpeg_q90(image_bytes)

    def _messages(self, system_prompt: str, user_prompt: str, observation: Any = None) -> list[dict[str, Any]]:
        """DeepSeekAgent's message shape with the frame as a content part before the text;
        without a frame, the text-only shape."""
        instructions = system_prompt
        if self.system_prompt_suffix:
            instructions = f"{instructions}\n\n{self.system_prompt_suffix}"
        screenshot = self._screenshot_bytes(observation)
        if screenshot is None:
            user_content: Any = user_prompt
        else:
            user_content = [
                {"type": "image_url", "image_url": {"url": f"{self.IMAGE_URL_PREFIX}{self._encode_screenshot(screenshot)}"}},
                {"type": "text", "text": user_prompt},
            ]
        return [{"role": "system", "content": instructions}, {"role": "user", "content": user_content}]

    def _extra_body(self, use_thinking: bool) -> dict[str, Any]:
        """The thinking toggle, always pinned explicitly, and with thinking on the effort pin of
        ``DeepSeekAgent._set_thinking`` (``deepseek-flash`` only), carried here because the
        timeline route sets only ``extra_body``; the SDK merges it into the body."""
        body: dict[str, Any] = {"thinking": {"type": "enabled" if use_thinking else "disabled"}}
        if use_thinking and self.model_name == _FLASH_MODEL:
            body["reasoning_effort"] = _FLASH_REASONING_EFFORT
        return body

    def _apply_grounding_sampling(self, req: dict[str, Any]) -> None:
        """Sampling on the GROUNDING (non-thinking) route only; the explicit pin wins over the
        generic constructor value (MiniMaxAgent's rule)."""
        temperature = self.grounding_temperature if self.grounding_temperature is not None else self.temperature
        if temperature is not None:
            req["temperature"] = temperature
        if self.top_p is not None:
            req["top_p"] = self.top_p


class OSWorldDeepSeekAgent(TimelineRoute, _DeepSeekOSWorldTransport):
    """``DeepSeekAgent`` with the timeline route and the three per-model hooks: frames as JPEG
    data URLs, no usage hook (the transport meters every response itself), and thinking off
    ``message.reasoning_content``. The constructor is ``DeepSeekAgent``'s plus
    ``grounding_temperature`` and ``request_timeout``."""

    def _record_response_usage(self, response: Any) -> None:
        """No-op: ``_create_with_backoff`` metered this response already."""
        return None

    def complete_timeline_json(
        self, *, system_prompt: str, user_parts: list[dict[str, Any]], use_thinking: bool = True,
        max_tokens: Optional[int] = None, validate: Optional[Callable[[dict[str, Any]], Any]] = None,
        turns: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        """The shared deciding route, with the transport's one rejection (a 200 with no choices,
        raised before the route can index the reply) recorded like the route's own — the same
        result shape, ``rejection`` of kind ``no_choices``, the request record — before it
        propagates to the re-sample loop."""
        started = time.perf_counter()
        try:
            return super().complete_timeline_json(system_prompt=system_prompt, user_parts=user_parts,
                                                  use_thinking=use_thinking, max_tokens=max_tokens,
                                                  validate=validate, turns=turns)
        except DeepSeekReplyRejected as exc:
            if exc.kind != NO_CHOICES:
                raise
            images = [part for part in user_parts if part["type"] == "image"]
            result: dict[str, Any] = {
                "text": "", "raw_response": "", "response_output": [], "usage": getattr(exc, "usage", {}),
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
                "rejection": {"kind": exc.kind, "detail": str(exc)[:300]},
            }
            if self.call_observer is not None:
                try:
                    result["request_record"] = self._request_record(labels=[
                        {"label": part.get("label"), "predict_call": part.get("predict_call")} for part in images
                    ])
                except Exception:  # noqa: BLE001 — diagnostics never fail a scored call
                    logger.exception("per-call request record failed; the call result is unaffected")
            self._notify_observer(result)
            exc.reply_text, exc.rejection = "", result["rejection"]
            raise

    def _check_reply(self, response: Any) -> None:
        """The deciding route's rejections (module docstring), before the reply is parsed."""
        rejection = reply_rejection(response)
        if rejection is None and not (response.choices[0].message.content or "").strip():
            rejection = (EMPTY_CONTENT, "the reply carries no content")
        if rejection is not None:
            kind, detail = rejection
            raise DeepSeekReplyRejected(detail, kind=kind)

    def complete_grounding(
        self, *, system_prompt: str, user_prompt: str, observation: Any, tool: dict[str, Any],
        turns: tuple[dict[str, Any], ...] = (),
        validate: Optional[Callable[[dict[str, Any]], tuple[Any, dict[str, Any]]]] = None,
        facts: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """The grounding call (module docstring): thinking off, the sampling pin, ``tool`` (the
        function definition: ``name``, ``description``, ``parameters``) declared and forced, no
        cap. ``turns`` follow the user message (the action agent's repair turn). The reply's
        first tool call's arguments, a JSON object, go to ``validate``, which returns ``(the
        caller's action, facts for the per-call record)`` — ``result["parsed"]`` and merged
        into ``result`` — or raises ``GroundingReplyRejected``; every rejection reaches the
        observer BEFORE it propagates, carrying ``reply_text`` (the arguments text, or the
        content when there was no call). A call named after one of ``tool``'s actions goes
        to ``validate`` with that action (``function_name_as_action``); arguments that are not
        JSON are read after the surplus-closer repair (``load_tool_arguments``), and
        ``reply_text`` stays the arguments as they came. ``facts`` ride on the result whatever becomes of the reply
        (the action agent's coordinate-strip record)."""
        request = self._base_request([*self._messages(system_prompt, user_prompt, observation), *turns], None)
        request["extra_body"] = self._extra_body(False)
        self._apply_grounding_sampling(request)
        request["tools"] = [{"type": "function", "function": dict(tool)}]
        request["tool_choice"] = {"type": "function", "function": {"name": tool["name"]}}
        started = time.perf_counter()
        result: dict[str, Any] = {
            "text": "", "raw_response": "", "response_output": [], "usage": {}, **(facts or {}),
        }
        try:
            response = self._create_with_backoff(request)
        except DeepSeekReplyRejected as exc:
            # The transport's one rejection (no choices): recorded like the route's own.
            result["usage"] = getattr(exc, "usage", {})
            result["diagnostics"] = self._diagnostics(
                route=GROUNDING_ROUTE, thinking=False, system_prompt=system_prompt, user_prompt=user_prompt,
                observation=observation, started=started)
            result["rejection"] = {"kind": exc.kind, "detail": str(exc)[:300]}
            self._notify_observer(result)
            rejected = GroundingReplyRejected(str(exc), kind=exc.kind)
            rejected.reply_text = ""
            raise rejected from exc
        message = response.choices[0].message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        reply_text = tool_calls[0].function.arguments if tool_calls else (message.content or "")
        reply_text = reply_text or ""
        result.update({
            "text": reply_text, "raw_response": reply_text,
            "response_output": [call.model_dump() for call in tool_calls],
            "usage": response.usage.model_dump() if response.usage else {},
            "diagnostics": self._diagnostics(
                route=GROUNDING_ROUTE, thinking=False, system_prompt=system_prompt, user_prompt=user_prompt,
                observation=observation, started=started,
            ),
        })
        try:
            rejection = reply_rejection(response)
            if rejection is not None:
                raise GroundingReplyRejected(rejection[1], kind=rejection[0])
            if not tool_calls:
                raise GroundingReplyRejected("the reply carries no tool call", kind=NO_TOOL_CALL)
            name = getattr(tool_calls[0].function, "name", None)
            if name != tool["name"]:
                named = function_name_as_action(name, reply_text, tool)
                if named is None:
                    raise GroundingReplyRejected(f"the tool call names {name!r}, not {tool['name']!r}",
                                                 kind=NO_TOOL_CALL)
                arguments, repaired = named
            else:
                try:
                    arguments, repaired = load_tool_arguments(reply_text)
                except (TypeError, ValueError) as exc:
                    raise GroundingReplyRejected(f"the tool call's arguments are not JSON: {exc}",
                                                 kind=UNPARSEABLE) from exc
                if not isinstance(arguments, dict):
                    raise GroundingReplyRejected("the tool call's arguments are not a JSON object", kind=UNPARSEABLE)
            result[FUNCTION_NAME_FIELD] = name != tool["name"]
            result[ARGUMENTS_REPAIRED_FIELD] = repaired
            result["tool_call_count"] = len(tool_calls)
            result["tool_calls_dropped"] = len(tool_calls) - 1
            if validate is not None:
                result["parsed"], facts = validate(arguments)
                result.update(facts)
            else:
                result["parsed"] = arguments
        except GroundingReplyRejected as exc:
            result["rejection"] = {"kind": exc.kind, "detail": str(exc)[:300]}
            self._notify_observer(result)
            exc.reply_text = reply_text
            raise
        result["rejection"] = None
        return self._notify_observer(result)

    @staticmethod
    def _thinking_text(message: Any) -> Optional[str]:
        """The thinking DeepSeek returned beside the content: ``message.reasoning_content`` (a
        string), else None. Read as an attribute: the SDK's message model admits the extra
        field."""
        try:
            text = message.reasoning_content
        except AttributeError:
            return None
        return text if isinstance(text, str) and text else None


def is_infra_error(exc: BaseException) -> bool:
    """Whether an exception leaving ``predict`` on the DeepSeek profile is infrastructure —
    re-raised, so the task has no result and is re-run — rather than an agent error, which ends
    the episode evaluated (graphpc_agent.OSWorldGraphPCAgent.predict). The Qwen profile's
    table. Infrastructure, AFTER this backbone's own retries: a transport failure
    or timeout; HTTP 5xx, 429, 402, 408, 401 and 404; an HTTP 200 the SDK could not decode
    (``UndecodableResponseError``); an HTTP 400 or in-band error in no known form after its
    attempts (``UnrecognisedProviderError``); a dead retrieval text arm
    (``TextArmUnavailableError``); host memory exhaustion (``MemoryError``); a failed screenshot
    capture (``ScreenshotCaptureFailed``, which ``predict`` raises on this profile). Everything
    else is an agent error: a 4xx from our own request (403, 413, 422, a context-overflow 400),
    a defect in the agent's code."""
    if isinstance(exc, (openai.APIConnectionError, TextArmUnavailableError, MemoryError, ScreenshotCaptureFailed,
                        UndecodableResponseError, UnrecognisedProviderError)):
        return True
    status = _status_of(exc)
    return status is not None and (status >= 500 or status in _INFRA_STATUS_CODES)
