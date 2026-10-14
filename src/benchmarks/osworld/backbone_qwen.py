"""The Qwen (qwen3.7-plus) backbone on OSWorld 2.0: ``QwenAgent`` with the timeline
route (``reasoning.TimelineRoute``) and the control's frame encoding, sampling and
transport.

WHY A SUBCLASS HERE. ``src/agents/backbone/qwen/main.py`` is the WebArena Qwen
backbone and stays unchanged for it; everything this benchmark's Qwen control does
differently lives in OSWorld's own modules, which only OSWorld's ``build_default``
constructs.

WHAT FOLLOWS THE CONTROL (``osworld-v2.1:mm_agents/qwen``, the Qwen control):

  * FRAMES go out as the control encodes them, by the control's own code
    (``qwen_upstream/images.py:process_screenshot``: ``smart_resize(factor=32,
    max_pixels=16*16*4*12800)``, a LANCZOS resize and a JPEG at the quality the
    control's delta D9 sets there, PIL defaults otherwise — D9 replaces upstream's lossless
    PNG because 20 PNG frames exceed OpenRouter's 30 MB image cap; 1920x1080 -> 1920x1088),
    sent as ``data:image/jpeg;base64,``. QwenAgent's own encoder is the MiniMax one
    (JPEG q90, native size).
  * SAMPLING: temperature 0.0 and top_p 0.9 on both routes, the control runner's
    defaults (``run_multienv_qwen.py``), which its client sends on every call,
    thinking on included. QwenAgent sends nothing on its reasoning route.
  * TRANSPORT (``client.py``): the client is built with ``max_retries=0`` and this
    class's own retry loop is the ONLY one — 50 attempts, sleep ``min(15, 3a)`` s after
    failed attempt ``a``, retrying the control's classes: ``APIConnectionError``
    (``APITimeoutError`` is one), ``RateLimitError``, ``InternalServerError`` — plus
    OpenRouter's transient in-band errors (QwenAgent's ``_RETRYABLE_ERROR_TYPES``), the
    control's three provider faults in either envelope, in-band or an HTTP 400 (``transient_rule``),
    and an HTTP 200 whose body the SDK cannot decode (``UndecodableResponseError``); exhausted,
    all of these are infrastructure (``is_infra_error``).
    WHERE IT DEPARTS FROM THE CONTROL (``failure_kind``): an HTTP 400,
    or an in-band error, in none of those forms is an UNRECOGNISED provider failure, not the
    agent's. Its attempts within one call are counted apart, up to ``UNRECOGNISED_ATTEMPTS`` (3);
    then it is raised as ``UnrecognisedProviderError``, infrastructure, its text carrying the
    provider's in full. Two failures stay the agent's, raised at once: a context
    overflow (the control's own patterns, on a 400 or an in-band code 400 only) and a 200 with no
    choices. The control keeps an unrecognised failure the agent's: its context-overflow rescue
    needs the context 400 to leave its client at once. 403, 413 and 422 (our request) are the
    agent's and 401, 402, 404 and 408 infrastructure, each raised at once; an in-band
    error's code follows the same map (403 / 413 / 422 the agent's, 401 / 402 infrastructure).
    Every result carries ``transport_unrecognised``, the call's count of unrecognised attempts.
    The control's other two classes cannot arise here: ``requests``' ``SSLError``
    belongs to a client this backbone does not use (an SSL failure under the openai
    client is an ``APIConnectionError``), and ``IncompleteStreamError`` belongs to
    streaming, which this backbone does not do. ``QwenAgent._create_with_backoff``
    (and its nested ``_QW_RETRY_POLICY``) is never called.
  * The request timeout is the profile's (``profile_qwen.QWEN_REQUEST_TIMEOUT_S``,
    recorded in ``active_config``): 900 s. The rule ``max(500, 1.5 x output cap / r)``,
    with r = 57.01 tokens per second, the slowest measured output rate in the deciding
    call's regime, gives 862 s, rounded up to 900 for margin; either way the output
    cap binds before the timeout does.

EXPOSURE, ON THE PER-CALL RECORD. How many HTTP attempts a call took and the error
the last failed one raised ride on every result this backbone hands its observer
(``transport_attempts``, ``transport_last_error``), so the arm's provider-fault exposure
is countable beside the control's per-attempt records.

METERING, EXACTLY ONCE PER HTTP RESPONSE. On every response the loop runs
``_record_usage`` and then validates (QwenAgent's order: a billed partial is metered
before it raises), and ``_record_response_usage`` — the hook ``TimelineRoute`` calls
after ``_create_with_backoff`` — is a no-op, where MiniMax's meters. Both re-meter
paths would double-count every deciding call.

A ``length`` stop with empty content is NOT raised by the transport (QwenAgent's
validation raises it as a failed call): it is returned, and each route rejects it as
a reply of kind ``length_empty`` (``LengthEmptyReply`` on the deciding route,
``GroundingReplyRejected`` on grounding), recorded on the per-call record; the step
degrades at once, never repaired: the same request would spend the same cap.

MRO. ``OSWorldQwenAgent(TimelineRoute, _QwenControlTransport)``: the transport sits
BELOW ``TimelineRoute``, so the route's request bookkeeping and frame memo wrap it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional

import openai

from src.agents.backbone.qwen.main import (
    _API_KEY_ENV,
    _BASE_URL,
    _RETRYABLE_ERROR_TYPES,
    OpenRouterCallError,
    QwenAgent,
    _as_mapping,
    _validate_response,
)
from src.agents.common.json_utils import JsonParseExhausted
from src.agents.retrieval.embedding import TextArmUnavailableError

from .qwen_upstream.images import process_screenshot
from .qwen_upstream.recovery import looks_like_context_error
from .reasoning import LENGTH_EMPTY, TimelineRoute
from .timeline import ScreenshotCaptureFailed


logger = logging.getLogger("desktopenv.agent")

# The control's transport classes (client.py:_retryable_exceptions), as they reach the
# openai client; APITimeoutError is an APIConnectionError.
_RETRYABLE_EXCEPTIONS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)

# Three provider faults that arrive typed invalid_request: a provider failure carrying any
# fragment is transient whatever its envelope. The control's client.py (D2) matches the
# same tuple; keep the two copies equal when editing either.
_TRANSIENT_ERROR_MESSAGES = ("JSON decode error: unexpected EOF", "Download multimodal file timed out",
                             "Failed to download multimodal content")


class UndecodableResponseError(RuntimeError):
    """An HTTP 200 the SDK could not turn into a completion, at the provider call itself:
    a JSON content type whose body does not decode (the SDK's ``json.JSONDecodeError``), or a
    non-JSON content type, for which ``create()`` returns the body as a ``str``. Transient, retried in the
    transport's loop, infrastructure once exhausted. The control's client.py (D2) defines the same class.
    A decode error anywhere else — parsing the model's reply, validating a body — is not this. Its text
    starts with ``UNDECODABLE_PREFIX``, on both arms, so either arm's records can be searched for it."""


class UnrecognisedProviderError(RuntimeError):
    """An HTTP 400, or an in-band error (a body-level ``error``, a choice's ``error`` or ``finish_reason``
    "error"), in no known form (``failure_kind``), still failing after the transport's
    ``UNRECOGNISED_ATTEMPTS``: infrastructure, so an unseen provider message is
    never scored as the agent's. Its text is ``UNRECOGNISED_PREFIX``, the original exception's type and the
    provider's failure in full. GraphPC only; the control has no such class."""


UNDECODABLE_PREFIX = f"{UndecodableResponseError.__name__}: "
UNRECOGNISED_PREFIX = f"{UnrecognisedProviderError.__name__}: "
# _validate_response's text for a 200 that decodes but carries no choices (``{}``): the agent's,
# never this fallback's.
_NO_CHOICES = "openrouter response carried no choices"


# An in-band error's code means what the same HTTP status means: 403, 413 and
# 422 are properties of our request, the agent's; 401 and 402 concern the account, infrastructure. Neither is
# retried.
_IN_BAND_AGENT_CODES = frozenset({"403", "413", "422"})
_IN_BAND_INFRA_CODES = frozenset({"401", "402"})


def _in_band_status(exc: BaseException) -> Optional[str]:
    """The code the transport read off the response an ``OpenRouterCallError`` came from (``in_band_code``), as a
    string; None for any other exception or an error raised elsewhere."""
    code = getattr(exc, "in_band_code", None) if isinstance(exc, OpenRouterCallError) else None
    return None if code is None else str(code)


def failure_kind(exc: BaseException) -> str:
    """How the transport treats a failure it catches, decided in this order:
    an in-band error whose code is in the status map, raised at once — ``"agent"`` for 403 / 413 / 422,
    ``"infra"`` for 401 / 402, as the same HTTP statuses are; ``"transient"``, a known transient form
    (``_RETRYABLE_EXCEPTIONS``, ``UndecodableResponseError``, ``transient_rule``), retried for the full budget;
    ``"agent"``, raised at once — a context overflow, by the control's own patterns
    (``qwen_upstream/recovery.py:looks_like_context_error``, upstream's bytes), on an HTTP 400 or an in-band error
    with code 400 only, and a 200 with no choices; ``"unrecognised"``, any other HTTP 400 or in-band error,
    retried for ``UNRECOGNISED_ATTEMPTS``. An in-band error's code is the one the transport read off the response
    (``in_band_code``)."""
    status = _in_band_status(exc)
    if status in _IN_BAND_AGENT_CODES:
        return "agent"
    if status in _IN_BAND_INFRA_CODES:
        return "infra"
    if isinstance(exc, (*_RETRYABLE_EXCEPTIONS, UndecodableResponseError)) or transient_rule(exc) is not None:
        return "transient"
    if str(exc) == _NO_CHOICES:
        return "agent"
    on_400 = isinstance(exc, openai.BadRequestError) or str(getattr(exc, "in_band_code", None)) == "400"
    if on_400 and looks_like_context_error(exc):
        return "agent"
    return "unrecognised"


def _in_band_code(response: Any) -> Any:
    """The code of the failure a 200 carries: its body-level ``error``'s, else its first choice's."""
    error = _as_mapping(getattr(response, "error", None))
    if error:
        return error.get("code")
    choices = getattr(response, "choices", None) or []
    return _as_mapping(getattr(choices[0], "error", None)).get("code") if choices else None


def _carries_transient_message(text: Any) -> bool:
    return isinstance(text, str) and any(fragment in text for fragment in _TRANSIENT_ERROR_MESSAGES)


def transient_rule(exc: BaseException) -> Optional[str]:
    """Why a PROVIDER failure is transient, or None. Reads only two classes, never the text of
    any other exception: an in-band ``OpenRouterCallError`` is transient by its typed
    ``error_type`` (``"error_type"``) or because its message — which quotes the provider
    error's ``message`` field and nothing else of the body — carries a provider-fault fragment
    (``"fragment"``); an HTTP 400 ``BadRequestError`` only by a fragment in its error body's
    ``message`` or ``metadata.raw``. The transport's retry and ``is_infra_error`` both ask it."""
    if isinstance(exc, OpenRouterCallError):
        if exc.error_type in _RETRYABLE_ERROR_TYPES:
            return "error_type"
        return "fragment" if _carries_transient_message(str(exc)) else None
    if isinstance(exc, openai.BadRequestError) and exc.status_code == 400:
        body = exc.body if isinstance(exc.body, dict) else {}
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        raw = metadata.get("raw")
        if _carries_transient_message(body.get("message")) or (
                raw is not None and _carries_transient_message(str(raw))):
            return "fragment"
    return None


_JPEG_DATA_URL_PREFIX = "data:image/jpeg;base64,"


def encode_control_frame(image_bytes: bytes) -> str:
    """Raw screenshot bytes -> base64 of the frame the control sends, by its own code
    (relative coordinates, no fixed model size: the control runner's defaults)."""
    data_url = process_screenshot(bytes(image_bytes), coordinate_type="relative").data_url
    if not data_url.startswith(_JPEG_DATA_URL_PREFIX):
        raise ValueError(f"the control's frame encoder returned a {data_url[:30]!r} data URL, not a JPEG one")
    return data_url[len(_JPEG_DATA_URL_PREFIX):]


def is_length_empty(response: Any) -> bool:
    """A reply cut at the output cap before any content: ``finish_reason: "length"``
    with empty or whitespace-only content, and no error envelope (QwenAgent's
    validation raises on an error first, and so does this backbone)."""
    if _as_mapping(getattr(response, "error", None)):
        return False
    choices = getattr(response, "choices", None)
    if not choices:
        return False
    choice = choices[0]
    if _as_mapping(getattr(choice, "error", None)):
        return False
    message = getattr(choice, "message", None)
    return getattr(choice, "finish_reason", None) == "length" and not (getattr(message, "content", None) or "").strip()


class LengthEmptyReply(JsonParseExhausted):
    """A reply cut at the output cap before any content (``is_length_empty``). A
    ``JsonParseExhausted`` so the deciding route's rejection path records it
    (``kind``), but not repaired: the degrade runs at once."""

    kind = LENGTH_EMPTY


class GroundingReplyRejected(ValueError):
    """A grounding reply that cannot become the one action asked for. ``kind`` names
    why (the per-call record's ``rejection``); the message is the control's error
    text for it, which a repair request quotes. ``length_empty`` is never repaired."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class _QwenControlTransport(QwenAgent):
    """``QwenAgent`` with the control's encoding, sampling and transport (module
    docstring). Below ``TimelineRoute`` in ``OSWorldQwenAgent``'s MRO."""

    IMAGE_URL_PREFIX = _JPEG_DATA_URL_PREFIX
    TEMPERATURE = 0.0
    TOP_P = 0.9
    MAX_ATTEMPTS = 50
    UNRECOGNISED_ATTEMPTS = 3
    RETRY_SLEEP_BASE_S = 3.0
    RETRY_SLEEP_CAP_S = 15.0

    def __init__(self, *args: Any, request_timeout: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.request_timeout = float(request_timeout)
        # The last call's HTTP attempts and the error of its last failed attempt (None).
        self._transport: dict[str, Any] = {"transport_attempts": 0, "transport_last_error": None}

    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.api_key)
        # QwenAgent's key rule (explicit, read at call time), with the control's client
        # settings: no SDK retries (this class's loop is the only one) and its timeout.
        api_key = self.api_key or os.environ.get(_API_KEY_ENV)
        return openai.OpenAI(api_key=api_key, base_url=_BASE_URL, timeout=self.request_timeout, max_retries=0)

    def _retry_sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _create_with_backoff(self, request: dict[str, Any]):
        # transport_unrecognised: this call's attempts that failed in no known form (``failure_kind``), on every
        # record, so a call the fallback recovered stays visible whatever failed after it.
        self._transport = {"transport_attempts": 0, "transport_last_error": None, "transport_unrecognised": 0}
        unrecognised = 0
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            self._transport["transport_attempts"] = attempt
            try:
                try:
                    response = self._build_client().chat.completions.create(**request)
                except json.JSONDecodeError as exc:
                    raise UndecodableResponseError(
                        f"{UNDECODABLE_PREFIX}the provider's HTTP 200 body is not JSON: {exc}") from exc
                if isinstance(response, str):
                    raise UndecodableResponseError(f"{UNDECODABLE_PREFIX}the provider's HTTP 200 carries a "
                                                   f"non-JSON content type ({len(response)} characters)")
                # QwenAgent's order: meter first (a billed partial still costs), then validate.
                self._record_usage(response)
                if not is_length_empty(response):
                    try:
                        _validate_response(response)
                    except OpenRouterCallError as exc:
                        exc.in_band_code = _in_band_code(response)
                        raise
                return response
            except (*_RETRYABLE_EXCEPTIONS, UndecodableResponseError, OpenRouterCallError, openai.BadRequestError) as exc:
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
                logger.warning("qwen API error (attempt %d/%d): %s", attempt, self.MAX_ATTEMPTS, exc)
                self._retry_sleep(min(self.RETRY_SLEEP_CAP_S, self.RETRY_SLEEP_BASE_S * attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def _notify_observer(self, result: dict[str, Any]) -> dict[str, Any]:
        """Every result the observer sees carries this call's transport exposure."""
        result.update(self._transport)
        return super()._notify_observer(result)

    def _encode_screenshot(self, image_bytes: bytes) -> str:
        return encode_control_frame(image_bytes)

    def _base_request(self, messages: list[dict[str, Any]], max_tokens: Optional[int]) -> dict[str, Any]:
        """QwenAgent's request plus the control's sampling, on every route."""
        request = super()._base_request(messages, max_tokens)
        request["temperature"] = self.TEMPERATURE
        request["top_p"] = self.TOP_P
        return request

    def _apply_grounding_sampling(self, req: dict[str, Any]) -> None:
        """The control's sampling is already on every request (``_base_request``)."""
        return None

    def _messages(self, system_prompt: str, user_prompt: str, observation: Any = None) -> list[dict[str, Any]]:
        """QwenAgent's message shape, with the frame under this encoder's prefix."""
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


class OSWorldQwenAgent(TimelineRoute, _QwenControlTransport):
    """``QwenAgent`` with the timeline route and the three per-model hooks: frames as
    JPEG data URLs (the control's encoding), no usage hook (the transport meters every
    response itself), and thinking off OpenRouter's ``reasoning`` fields. The
    constructor is ``QwenAgent``'s plus ``request_timeout``."""

    def _record_response_usage(self, response: Any) -> None:
        """No-op: ``_create_with_backoff`` metered this response already."""
        return None

    def _check_reply(self, response: Any) -> None:
        if is_length_empty(response):
            raise LengthEmptyReply("the reply reached the output cap before any content")

    def complete_grounding(
        self, *, system_prompt: str, user_prompt: str, observation: Any, turns: tuple[dict[str, Any], ...] = (),
        validate: Optional[Callable[[str], tuple[Any, dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """The grounding call in the control's shape: reasoning off, its sampling and
        output cap, no tools, no ``stop`` (the copied parser requires the closing tag,
        which a stop string would strip). ``turns`` follow the user message: the
        rejected reply and its repair request (action_qwen.py). ``validate`` turns the
        content into ``(the caller's action, facts for the per-call record)`` —
        ``result["parsed"]`` and merged into ``result`` — or raises
        ``GroundingReplyRejected``, BEFORE the observer is told, so the per-call record
        carries the rejection; the exception then propagates carrying ``reply_text``."""
        request = self._base_request([*self._messages(system_prompt, user_prompt, observation), *turns], None)
        request["extra_body"] = self._extra_body(False)
        started = time.perf_counter()
        response = self._create_with_backoff(request)
        content = response.choices[0].message.content or ""
        result: dict[str, Any] = {
            "text": content, "raw_response": content, "response_output": [],
            "usage": response.usage.model_dump() if response.usage else {},
            "diagnostics": self._diagnostics(
                route="grounding_text", thinking=False, system_prompt=system_prompt, user_prompt=user_prompt,
                observation=observation, started=started,
            ),
        }
        try:
            if is_length_empty(response):
                raise GroundingReplyRejected("the reply reached the output cap before any content",
                                             kind=LENGTH_EMPTY)
            if validate is not None:
                result["parsed"], facts = validate(content)
                result.update(facts)
        except GroundingReplyRejected as exc:
            result["rejection"] = {"kind": exc.kind, "detail": str(exc)[:300]}
            self._notify_observer(result)
            exc.reply_text = content
            raise
        result["rejection"] = None
        return self._notify_observer(result)

    @staticmethod
    def _thinking_text(message: Any) -> Optional[str]:
        """The thinking OpenRouter returned beside the content: ``message.reasoning``
        (a string), else the text of ``message.reasoning_details`` entries. Best-effort:
        a shape this does not know reads as None."""
        text = getattr(message, "reasoning", None)
        if isinstance(text, str) and text:
            return text
        details = getattr(message, "reasoning_details", None)
        if isinstance(details, list):
            pieces = []
            for detail in details:
                detail = detail if isinstance(detail, dict) else _as_mapping(detail)
                piece = detail.get("text") or detail.get("summary")
                if isinstance(piece, str):
                    pieces.append(piece)
            return "".join(pieces) or None
        return None


# The lost-episode cause table's infrastructure rows: an HTTP status
# that says the route, the account or the provider failed, never the request.
_INFRA_STATUS_CODES = frozenset({401, 402, 404, 408, 429})


def is_infra_error(exc: BaseException) -> bool:
    """Whether an exception leaving ``predict`` on the Qwen profile is infrastructure —
    re-raised, so the task has no result and is re-run — rather than an agent error,
    which ends the episode evaluated (graphpc_agent.OSWorldGraphPCAgent.predict).

    Infrastructure, AFTER this backbone's own retries: a transport failure or timeout
    (``APIConnectionError``, ``APITimeoutError`` among them); HTTP 5xx, 429, 402, 408,
    401 and a 404 (on this route a 404 is the routing pin or the slug going away); a
    transient in-band OpenRouter error or an HTTP 400 carrying a provider-fault fragment
    (``transient_rule``); a dead retrieval text arm
    (``TextArmUnavailableError``: an episode on a degraded retriever is never scored); an HTTP 200
    the SDK could not decode at the provider call (``UndecodableResponseError``; a raw
    ``json.JSONDecodeError`` from anywhere else is an agent error); an HTTP 400 or an in-band
    error in no known form, after its 3 attempts (``UnrecognisedProviderError``);
    host memory exhaustion (``MemoryError``; a repeat on the same task's re-run is
    escalated, an ops rule); a failed screenshot capture (``ScreenshotCaptureFailed``, which
    ``predict`` raises on this profile as the control raises its ``ValueError``).
    Everything else is an agent error: a
    deterministic 4xx from our own request (403, 413, 422), a context-overflow 400, a 200 with no
    choices, a defect in the agent's code. An in-band error the transport read a code off follows
    the status map: 403 / 413 / 422 the agent's, 401 / 402 infrastructure. A raw non-transient
    in-band error or 400 is otherwise classed agent_error here; the transport lets one leave
    unwrapped only when ``failure_kind`` makes it the agent's (or, for 401 / 402, infrastructure),
    and wraps every other as an ``UnrecognisedProviderError``. The table deliberately differs from WebArena's
    ``_classify_episode_error``, which counts 403 and every ``OpenRouterCallError``."""
    if isinstance(exc, (openai.APIConnectionError, TextArmUnavailableError, MemoryError, ScreenshotCaptureFailed,
                        UndecodableResponseError, UnrecognisedProviderError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status is not None and (status >= 500 or status in _INFRA_STATUS_CODES):
            return True
        return transient_rule(exc) is not None
    if isinstance(exc, OpenRouterCallError):
        status = _in_band_status(exc)
        if status in _IN_BAND_AGENT_CODES or status in _IN_BAND_INFRA_CODES:
            return status in _IN_BAND_INFRA_CODES
        return transient_rule(exc) is not None
    return False
