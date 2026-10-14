"""Qwen (qwen3.7-plus) backbone (openai SDK over the OpenRouter route).

Satisfies ``Backbone`` (``src/agents/backbone/base.py``; the shared reasoning /
action agents call ``complete_json`` and read ``result["json"]`` +
``result["raw_response"]``), so those agents are REUSED unchanged with a
``QwenAgent`` instance. Structural mirror of the MiniMax backbone
(``src/agents/backbone/minimax/main.py``); only the provider facts differ.

Provider facts below are tagged **[V]** — stated by OpenRouter's model catalog
and docs — or **[A]** — an assumption that only a live call can settle.
Re-check both against the live catalog before a run on this route rather than
inheriting them.

  * Route: ``https://openrouter.ai/api/v1`` with
    ``Authorization: Bearer $OPENROUTER_API_KEY`` [V]. There is NO base-URL
    override and NO second provider: OpenRouter is the only route this backbone
    has, so there is nothing to fall back to and no env knob for one. The key
    is passed EXPLICITLY (with a custom base URL the openai SDK would
    otherwise fall back to ``OPENAI_API_KEY``) and is read at CALL time, since
    the ``__init__`` default binds at import, before the runner loads ``.env``.
    PINNED: the control-arm shim (``src/benchmarks/webarena/_openai_compat.py``)
    resolves the identical env var and host; the two arms' values must stay
    equal, so change them together.
  * Model slug ``qwen/qwen3.7-plus``; the response ``model`` echo returns the
    slug that was sent, not a canonical dated form, so it detects a rename
    rather than a silent re-point of the alias. It is logged on every usage
    line below.
  * One endpoint serves the slug: provider tag ``alibaba``, context 1,000,000,
    ``max_prompt_tokens`` 983,616, ``supports_tool_choice.function: true`` [V].
    Mind the two senses of the name: the endpoint's CATALOG field
    ``max_completion_tokens`` is 131,072 and is the CEILING (``_MAX_TOKENS_CEILING``
    below); the REQUEST parameter the endpoint accepts is ``max_tokens`` —
    ``supported_parameters`` lists ``max_tokens`` and not
    ``max_completion_tokens`` [V]. This backbone therefore sends ``max_tokens``
    and never ``max_completion_tokens``, and the GraphPC arm sends no cap at all.
  * Route pin, byte-identical on EVERY request of both arms:
    ``provider={"only": ["alibaba"], "allow_fallbacks": false,
    "require_parameters": true}``. OpenRouter defaults to
    ``allow_fallbacks: true`` plus price-based load balancing [V], so an
    unpinned request may be served by a different endpoint of the same slug —
    an unmeasured model change mid-run. ``require_parameters`` is deliberate:
    it converts "this endpoint no longer supports ``tools`` /
    ``response_format``" into a 503 instead of a silent parameter drop, which
    would otherwise surface as free text where a tool call belongs. The pin is
    falsifiable, never silent: an unavailable pinned endpoint returns 503 ("no
    available model provider that meets your routing requirements"), a
    conflict with an account ceiling returns 404 [V] — and both already
    classify as infra in the runner (``>= 500`` / ``_INFRA_STATUS_CODES``).
  * Thinking is pinned EXPLICITLY on every request through the ``reasoning``
    field, never the provider default. The openai SDK has no ``reasoning``
    kwarg, so it rides in ``extra_body``. Deciding / reasoning route:
    ``{"enabled": true}`` — enable at medium effort. Grounding route:
    ``{"effort": "none"}`` — the only form the docs state disables reasoning
    entirely [V-doc, A-behaviour: the catalog publishes no ``supported_efforts``
    for this model, so only a live call shows that ``none`` is honoured rather
    than mapped to the nearest supported level].
    Two forms are FORBIDDEN here, and the reason each is forbidden:
      - ``{"exclude": true}`` and the legacy ``include_reasoning: false`` do
        NOT disable reasoning. The model still reasons, the tokens are still
        billed and still count against ``max_tokens``; they are merely hidden
        from the response [V] — which also destroys the only evidence that a
        chain-of-thought happened at all.
      - a bare ``{}`` defers to ``default_enabled``, which this model reports
        as ``true`` [V]. That is a provider default, and a provider default is
        exactly what a pinned regime may not rest on (measurement integrity).
  * Thinking lands on ``choices[].message.reasoning`` (string|null) and
    ``choices[].message.reasoning_details`` (a typed array) [V] — off
    ``message.content``, so there is nothing to strip. An inline ``<think>``
    block inside ``content`` is undocumented but not excluded [A]; if one
    appears on the deciding route, REPORT it (never add a stripper).
  * Reasoning billing: billed as output, and counted against ``max_tokens``
    "on most providers" — Alibaba is never named, so [A] here. The GraphPC
    arm sends NO cap (``--max_tokens`` default None) and therefore inherits an
    unknown provider default; a deciding call that finishes on ``length`` with
    empty content (the documented signature of a reasoning budget spent
    entirely on reasoning) is a failed call (``_validate_response``).
  * Cap ceiling on this endpoint: 131,072 [V] (``_MAX_TOKENS_CEILING`` below).
    Any cap a caller passes must lie at or under it.
  * Usage is ALWAYS returned (``usage:{include:true}`` is a deprecated no-op)
    and carries ``cost``, ``cost_details``,
    ``completion_tokens_details.reasoning_tokens`` and
    ``prompt_tokens_details.cached_tokens`` [V]. This route has NO
    ``system_fingerprint`` field, so the metering line carries ``cost=``
    instead — see ``_record_usage``.
  * Route identity is NOT on a successful response: which endpoint actually
    served a request is visible only when the request sends the
    ``X-OpenRouter-Metadata: enabled`` header [V]. Production requests stay
    minimal — no metadata header, no ``debug`` upstream echo (that one is
    streaming-only and not for production anyway) — so the per-call ``model=``
    echo is the route record here; reading the routing metadata takes a
    separate, diagnostic request.
  * Structured output: ``response_format={"type": "json_object"}`` is a
    supported parameter on this endpoint [V]; strict server-side
    ``json_schema`` is NOT used (mirror: the reasoning schema rides in the
    prompt as an advisory hint + ``extract_json_object``). Forced
    function-calling is supported (``supports_tool_choice.function: true`` [V])
    and ``strict: True`` is accepted at the API edge, but its provider
    semantics are unspecified — it is a HINT [A], so the returned arguments are
    validated downstream against the grammar. A 400 naming ``tools`` /
    ``tool_choice`` STOPS the run and is reported, never worked around.
  * A regex ``pattern`` inside the tool parameters is ASSUMED advisory as on
    MiniMax (not enforcement-tested): closed-vocabulary closure relies on the
    action prompt's grammar listing + downstream post-parse normalization /
    the translator's degrade path (design.md's sanctioned fallback).
  * Price: the per-token price triples in BOTH directions once the prompt
    exceeds 256k tokens [V]; a lane tripwire (run_lanes.py) can key on the
    ``input=`` field of the usage line below to catch that cliff.
  * Rate limits: no platform RPM cap on paid models; upstream 429s still occur
    [V]. Nothing here lowers the openai SDK's 600 s default timeout, which is
    the property this route relies on.
  * Vision: an ``image_url`` content part with a base64 data URI; the endpoint's
    modality is ``text+image->text`` [V]. Per-image token accounting is not
    published for this endpoint [A]; images are billed at the input price.

A 200 can carry a failure, and an unvalidated one becomes a scored 0.0.
OpenRouter has two documented 200-carrying-failure shapes [V]: (1) a body with
an ``error`` object and no ``choices`` key at all; (2) a ``choices[0]`` holding
partial content plus its own ``error`` and ``finish_reason: "error"``. So EVERY
response is validated before use (``_validate_response``): a body-level
``error``, missing/empty ``choices``, ``finish_reason: "error"``, or ``length``
with empty (or whitespace-only) content is a FAILED call and raises
``OpenRouterCallError``, which carries the typed
``error.metadata.error_type`` and the upstream ``provider_code``. The runner
classifies that class by NAME (``_INFRA_EXC_NAMES`` in
``scripts/webarena/run_graphpc.py``) so the episode is EXCLUDED rather than
scored zero; it deliberately carries no ``status_code``, because synthesising
one to slip past the runner's status branch would be falsified telemetry.
Only the transient ``error_type`` values retry (``_RETRYABLE_ERROR_TYPES``).

Two shapes that look like failures but are NOT, because raising on them would
exclude a real agent outcome from the denominator: ``finish_reason:
"content_filter"`` (on this route a model's own refusal is a COMPLETED response
carrying ``message.refusal``, and this endpoint reports ``is_moderated:
false``), and ``length`` with non-empty content (a truncated reply). Both flow
into the normal path, where ``extract_json_object`` fails on them and the step
degrades as ONE counted no-op — the degrade contract. A refusal is logged so it
is never silent.

Screenshot attachment (OSWorld 2.0, screenshot-primary): when the ``observation``
carries ``content["screenshot"]`` bytes (an ``Observation`` or a bare dict),
the user message becomes a content-part LIST with the image part FIRST and
the text part second — ``{"type": "image_url", "image_url": {"url":
"data:image/jpeg;base64,..."}}`` — on both routes and on ``complete_text``.
The bytes are re-encoded as JPEG quality 90 at native resolution (no resize),
the MiniMax encoding. With NO screenshot the user message stays the plain
string it is on the text-only (WebArena) request shape.

Two call shapes, routed inside ``complete_json`` by the EXPLICIT ``grounding``
flag the caller passes (the grounding/action call sites pass
``grounding=True``) — the DeepSeek / MiniMax routing, kept identical:

  * GROUNDING (``grounding=True`` — the action call): reasoning DISABLED +
    FORCED function-calling with the schema as the tool's ``parameters``
    (``strict:True``). The tool-call arguments are the emitted JSON
    (``pattern`` advisory, see above). Grounding is mechanical; deliberation
    belongs to the deciding call. The two halves are INSEPARABLE — see
    ``_invoke_grounding``.
  * REASONING (``grounding=False`` — the default): reasoning ENABLED
    (per ``use_thinking``) + ``response_format=json_object`` + post-parse via
    ``extract_json_object`` (the schema is injected into the prompt as an
    advisory hint; closed enums are normalised downstream). Reasoning schemas
    are unchanged.

Routing is on the explicit flag, NOT on whether the schema happens to carry a
``pattern`` — an implicit pattern-presence inference would silently flip any
future reasoning schema that gains a string ``pattern`` into non-thinking
forced-FC. ``_schema_has_pattern`` is retained only as a defensive assertion
(grounding without a pattern → warn; a pattern-bearing schema routed as
reasoning → raise, because grammar emission would then silently not happen).

``complete_text`` is a plain text route (reasoning per ``use_thinking``, NO
tools and NO response_format); WebArena's call sites use ``complete_json``, and
the OSWorld Qwen action agent grounds through ``OSWorldQwenAgent.complete_grounding``
(``src/benchmarks/osworld/backbone_qwen.py``).

Sampling regime: ``grounding_temperature`` is the explicit pin for the
GROUNDING (reasoning-disabled) routes only, kept for build_default symmetry;
it is sent only when not None, so a WebArena request (which passes nothing)
carries no sampling params. The generic ``temperature`` / ``top_p`` kwargs are
grounding-only too, and ``grounding_temperature`` takes precedence over
``temperature`` when both are set. NOTHING is ever sent on the reasoning-
enabled route: a pinned value there would silently change the deciding call's
regime.

Usage counters mirror the GPT backbone: input ← ``prompt_tokens``, output ←
``completion_tokens``; ``completion_tokens_details.reasoning_tokens`` and the
call's ``cost`` are surfaced in the log line.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import os
import time
from typing import Any, Callable, Optional

import openai
from PIL import Image

from src.agents.common.json_utils import extract_json_object
from src.agents.common.retry_policy import RetryPolicy

logger = logging.getLogger("desktopenv.agent")

# The OpenRouter OpenAI-compatible endpoint — this backbone's only route.
# PINNED: the control-arm shim (src/benchmarks/webarena/_openai_compat.py)
# carries the identical literal so both arms resolve the same host; change the
# two together.
_BASE_URL = "https://openrouter.ai/api/v1"
# The OpenRouter model slug. The response `model` echo returns the slug sent and
# rides on every usage line (it detects a rename, not a silent alias re-point).
_DEFAULT_MODEL = "qwen/qwen3.7-plus"
# The only credential this backbone reads; resolved at CALL time.
_API_KEY_ENV = "OPENROUTER_API_KEY"
# The route pin (module docstring). Sent on EVERY request, byte-identical on
# both arms; copied per request so no caller can mutate it.
_PROVIDER_PIN = {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}
# The two reasoning pins. `{"enabled": true}` = enable at medium effort;
# `{"effort": "none"}` = the only documented form that disables reasoning
# entirely. `exclude` / `include_reasoning: false` / `{}` are FORBIDDEN — see
# the module docstring for why each one is not a disable.
_REASONING_ENABLED = {"enabled": True}
_REASONING_DISABLED = {"effort": "none"}
# The endpoint's catalog `max_completion_tokens` — the CEILING for any cap a
# caller passes. The request parameter itself is `max_tokens` (see
# _base_request).
_MAX_TOKENS_CEILING = 131072
# The typed `error_type` values that a retry can plausibly clear. Everything
# else (invalid_request, content_policy_violation, a spent reasoning budget)
# returns the same result on the next attempt, so it is raised straight away.
_RETRYABLE_ERROR_TYPES = frozenset({
    "rate_limit_exceeded", "provider_overloaded", "provider_unavailable", "timeout", "server",
})


class OpenRouterCallError(RuntimeError):
    """A call that returned HTTP 200 but carries a provider failure.

    Carries the typed ``error.metadata.error_type`` and the upstream
    ``provider_code`` as attributes (either may be None: a 500 masks
    ``provider_code``, and a ``length``-with-empty-content failure is not an
    error envelope at all). Deliberately NOT an ``openai`` exception and
    deliberately WITHOUT a ``status_code``: the runner's classifier has a
    status branch, and synthesising a status to slip past it would be
    falsified telemetry. The runner classifies this by class NAME instead
    (``_INFRA_EXC_NAMES`` in scripts/webarena/run_graphpc.py), so the episode
    is EXCLUDED as infra rather than counted as a scored 0.0."""

    def __init__(
        self, message: str, *, error_type: Optional[str] = None, provider_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.provider_code = provider_code


def _as_mapping(value: Any) -> dict[str, Any]:
    """A provider sub-object as a plain dict. OpenRouter's extra fields reach us
    either as dicts or as SDK models (openai-python builds response objects
    leniently, so unknown fields survive in whichever shape they arrived);
    never raises, because this feeds response validation."""
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001 — a malformed payload is "no metadata", not a crash.
            return {}
    return {}


def _validate_response(response: Any) -> None:
    """Raise ``OpenRouterCallError`` when a 200 carries a provider failure.

    Called on EVERY response before anything reads ``choices[0]`` — an
    unvalidated failure shape becomes a scored 0.0 (shape (1) carries no
    ``choices`` key at all and would otherwise be an IndexError/AttributeError
    deep in the call site). The failed-call set is: a body-level ``error``;
    missing/empty ``choices``; ``finish_reason: "error"``; and ``length`` with
    empty or whitespace-only content.

    ``content_filter`` is deliberately NOT in that set — see below."""
    error = _as_mapping(getattr(response, "error", None))
    if error:
        # Shape (1) — and also the mid-stream shape, where a top-level error
        # rides alongside a stub choice; taking it first gets the typed code.
        metadata = _as_mapping(error.get("metadata"))
        raise OpenRouterCallError(
            f"openrouter returned a body-level error: code={error.get('code')!r} "
            f"message={error.get('message')!r}",
            error_type=metadata.get("error_type"),
            provider_code=metadata.get("provider_code"),
        )
    choices = getattr(response, "choices", None)
    if not choices:
        raise OpenRouterCallError("openrouter response carried no choices")
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    choice_error = _as_mapping(getattr(choice, "error", None))
    if choice_error or finish_reason == "error":
        # Shape (2) — a billed partial. The usage line is already out (see
        # _create_with_backoff), so the cost stays in the record.
        #
        # `content_filter` is NOT part of this condition. On this route a
        # model's own refusal comes back as a COMPLETED response carrying
        # message.refusal with finish_reason "content_filter", and this
        # endpoint reports is_moderated: false, so a provider-side moderation
        # block is not the mechanism. A refusal is the MODEL's behaviour on a
        # scored step: raising here would classify it infra_error and drop a
        # real agent outcome out of the denominator — the same bias as the
        # `length` + non-empty case below. It flows into the normal path
        # instead, where extract_json_object fails on it and the step degrades
        # as ONE counted no-op. A `content_filter` that DOES carry a choice
        # error object still raises, via choice_error above.
        metadata = _as_mapping(choice_error.get("metadata"))
        raise OpenRouterCallError(
            f"openrouter choice failed: finish_reason={finish_reason!r} "
            f"error={choice_error.get('message')!r}",
            error_type=metadata.get("error_type"),
            provider_code=metadata.get("provider_code"),
        )
    message = getattr(choice, "message", None)
    if finish_reason == "content_filter":
        # Not a failure, but never silent: the run log names it so a refusal is
        # visible as a counted degradation rather than an unexplained no-op.
        logger.info(
            "qwen refusal: finish_reason='content_filter' refusal=%r — the model declined; the step "
            "degrades as a counted no-op (not an infra failure)",
            getattr(message, "refusal", None),
        )
    if finish_reason == "length" and not (getattr(message, "content", None) or "").strip():
        # ONLY the empty half is infra. `length` with EMPTY content — or
        # whitespace only, which the control arm also treats as empty, so the
        # two arms agree — is the documented signature of a token budget spent
        # entirely on reasoning: nothing was produced, so there is nothing to
        # score. `length` with
        # NON-EMPTY content is a truncated reply: extract_json_object fails on
        # it and the step degrades as a COUNTED no-op, which is what the degrade
        # contract requires. Widening this raise to cover that case would
        # silently exclude real degradations from the denominator — the exact
        # bias _classify_episode_error exists to prevent.
        #
        # Retrying does not help either way, so error_type stays None and
        # _qw_is_retryable declines it. (An empty content with
        # finish_reason="tool_calls" is the NORMAL grounding response and is
        # untouched here.)
        raise OpenRouterCallError(
            "openrouter returned finish_reason='length' with empty content "
            "(the token budget was spent before any reply was emitted)"
        )


def _qw_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, OpenRouterCallError):
        return exc.error_type in _RETRYABLE_ERROR_TYPES
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


def _qw_is_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, openai.RateLimitError)


# Reuse the shared retry policy (retry_policy.py) — same predicates the GPT /
# DeepSeek / MiniMax backbones use; OpenRouter's compatible route surfaces the
# same openai SDK exception types, plus the 200-carrying-failure class above.
_QW_RETRY_POLICY = RetryPolicy(is_retryable=_qw_is_retryable, is_rate_limit=_qw_is_rate_limit)


def _schema_has_pattern(schema: Any) -> bool:
    """True if the JSON schema carries a regex ``pattern`` anywhere — the marker
    of a GROUNDING (grammar-bearing) schema vs a reasoning schema."""
    if isinstance(schema, dict):
        if "pattern" in schema:
            return True
        return any(_schema_has_pattern(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_schema_has_pattern(v) for v in schema)
    return False


class QwenAgent:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = os.environ.get(_API_KEY_ENV),
        system_prompt_suffix: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        client_factory: Optional[Callable[[Optional[str]], Any]] = None,
        grounding_temperature: Optional[float] = None,
    ) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.system_prompt_suffix = system_prompt_suffix
        self.temperature = temperature
        self.top_p = top_p
        # Grounding-route-only sampling pin (see module docstring); None = send nothing.
        self.grounding_temperature = grounding_temperature
        # reasoning_effort accepted for build_default symmetry but IGNORED —
        # this backbone pins the two reasoning forms itself (deciding =
        # {"enabled": true}, grounding = {"effort": "none"}); a caller-chosen
        # effort would break that pin. Log a one-line notice when a value is
        # actually supplied so the config surface never silently no-ops.
        self.reasoning_effort = reasoning_effort
        if reasoning_effort is not None:
            logger.info(
                "qwen backbone: reasoning_effort=%r is ignored; the per-route reasoning pin "
                "governs deliberation (deciding=%s, grounding=%s).",
                reasoning_effort, _REASONING_ENABLED, _REASONING_DISABLED,
            )
        self.client_factory = client_factory

        # The route echo: one line per CONSTRUCTION (not per run) naming what
        # this arm sends — a lane that builds the agent in the preflight and
        # again in the runner emits it twice, and two echoes are two objects,
        # not two routes. It is NOT evidence of what served either: a successful
        # OpenRouter response carries no provider field, and production requests
        # do not ask for the routing metadata. The per-call `model=` echo on the
        # usage line is the route record.
        logger.info(
            "qwen backbone config echo — what we send, not evidence of what served: "
            "base_url=%s model=%s reasoning(deciding)=%s reasoning(grounding)=%s provider=%s",
            _BASE_URL, self.model_name, json.dumps(_REASONING_ENABLED, sort_keys=True),
            json.dumps(_REASONING_DISABLED, sort_keys=True), json.dumps(_PROVIDER_PIN, sort_keys=True),
        )

        # Telemetry counters (same names the runner reads off the backbone).
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        # Per-call observer (OSWorld adapter): called with the completed result
        # dict of EVERY model call, after the call returned. Diagnostics only —
        # the request bytes are untouched and an observer failure never fails
        # the call. None = nobody listening (WebArena).
        self.call_observer: Optional[Callable[[dict[str, Any]], None]] = None

    # ------------------------------------------------------------------
    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.api_key)
        # Read OPENROUTER_API_KEY at CALL time (the __init__ default binds at
        # import, before the runner loads .env). The key must be explicit: with
        # a custom base_url the openai SDK would otherwise fall back to
        # OPENAI_API_KEY.
        api_key = self.api_key or os.environ.get(_API_KEY_ENV)
        return openai.OpenAI(api_key=api_key, base_url=_BASE_URL)

    def _create_with_backoff(self, request: dict[str, Any]):
        def _attempt():
            response = self._build_client().chat.completions.create(**request)
            # ORDER IS LOAD-BEARING: meter first, validate second. A 200 that
            # carries a provider failure can still be a billed partial (failure
            # shape (2)), and skipping the usage line would drop real cost from
            # the run record. An UNBILLED failure emits nothing — _record_usage
            # returns early when the response carries no usage payload at all.
            # Validation runs INSIDE the retried attempt so that
            # _qw_is_retryable's transient set actually governs the retry — a
            # provider_overloaded body is retried exactly like an HTTP 503.
            self._record_usage(response)
            _validate_response(response)
            return response
        return _QW_RETRY_POLICY.run(_attempt)

    def reset_session(self) -> None:
        # Qwen is stateless per call (no server-side response chaining like
        # the GPT Responses API); nothing to reset. Present so the WebArena shim's
        # reset() finds it.
        return None

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        return extract_json_object(text, log_label="Qwen")

    def _record_usage(self, response: Any) -> None:
        # Telemetry only — NEVER raises: a malformed provider usage payload
        # must not fail a scored call (the baseline shim's _log_qwen_usage
        # wraps its extraction the same way). Extraction completes before any
        # counter mutation, so on failure the counters simply do not advance
        # for this call and response processing continues untouched. Takes the
        # whole response (not just ``usage``) because the ``model`` echo rides
        # on the response envelope — the slug sent, so a rename shows on the
        # metering line. Called once per
        # ATTEMPT, before validation, so every billed attempt is metered —
        # including a retried one and a failed partial.
        #
        # NO usage payload -> NO line and NO counter movement, matching the
        # control arm. A zeros line would be invented telemetry, and under
        # OpenRouter's zero-completion insurance a response with no completion
        # tokens and an error finish was not billed at all, so there is nothing
        # to record.
        #
        # Key names are shared with the control arm's line (both start with the
        # literal ``qwen usage: ``, so one reader of the run's log parses both
        # arms). ``cost`` comes from
        # the usage payload, which this route always returns on a billed call;
        # there is no ``system_fingerprint`` here. A lane tripwire can key on
        # ``input=`` to catch the 256k-prompt 3x price cliff.
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0
            cost = getattr(usage, "cost", None)
            model = getattr(response, "model", None) or "none"
            self.call_count += 1
            self.total_input_tokens += prompt
            self.total_output_tokens += completion
            logger.info(
                "qwen usage: call=%d input=%d output=%d reasoning=%d cached=%d cost=%s model=%s "
                "| cumulative input=%d output=%d",
                self.call_count, prompt, completion, reasoning, cached,
                "none" if cost is None else cost, model,
                self.total_input_tokens, self.total_output_tokens,
            )
        except Exception:
            logger.debug("qwen usage line skipped (malformed usage payload)", exc_info=True)

    def _diagnostics(
        self, *, route: str, thinking: bool, system_prompt: str, user_prompt: str, observation: Any, started: float,
    ) -> dict[str, Any]:
        """The per-call record's request-side facts (sizes, not content) and the
        wall latency of the provider call; the caller attaches it as ``result["diagnostics"]``."""
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

    def _notify_observer(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.call_observer is not None:
            try:
                self.call_observer(result)
            except Exception:  # noqa: BLE001 — diagnostics never fail a scored call
                logger.exception("qwen call observer failed; the call result is unaffected")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _screenshot_bytes(observation: Any) -> Optional[bytes]:
        """``observation.content["screenshot"]`` (an ``Observation``) or
        ``observation["screenshot"]`` (a bare dict); None when absent — the
        same lookup the Claude / GPT / MiniMax backbones use."""
        if observation is None:
            return None
        if hasattr(observation, "content") and isinstance(observation.content, dict):
            return observation.content.get("screenshot")
        if isinstance(observation, dict):
            return observation.get("screenshot")
        return None

    @staticmethod
    def _encode_screenshot(image_bytes: bytes) -> str:
        """Raw screenshot bytes -> base64 of a JPEG quality-90 re-encode at the
        image's native size (RGB-converted; ``optimize`` on) — the MiniMax
        backbone's encoding (OSWorld-V2's M3 agent defaults). No resize:
        coordinate parity rests on the model seeing the full-resolution frame."""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _messages(
        self, system_prompt: str, user_prompt: str, observation: Any = None,
    ) -> list[dict[str, Any]]:
        instructions = system_prompt
        if self.system_prompt_suffix:
            instructions = f"{instructions}\n\n{self.system_prompt_suffix}"
        screenshot = self._screenshot_bytes(observation)
        if screenshot is None:
            # No image: the plain-string user message — the text-only (WebArena)
            # request shape.
            user_content: Any = user_prompt
        else:
            # Image part BEFORE the text part (OpenAI content-part list).
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._encode_screenshot(screenshot)}"},
                },
                {"type": "text", "text": user_prompt},
            ]
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _extra_body(use_thinking: bool) -> dict[str, Any]:
        """The per-request ``extra_body``: the reasoning pin and the route pin,
        both EXPLICIT on every request.

        The reasoning pin is never the provider default (which is ON) and never
        ``exclude`` / ``include_reasoning: false`` / ``{}`` — none of those
        disables reasoning; see the module docstring for what each one actually
        does. The route pin keeps every request on the one pinned endpoint.
        Fresh copies, so nothing a caller does can mutate the module-level pins,
        which must stay equal across the arms."""
        return {
            "reasoning": dict(_REASONING_ENABLED if use_thinking else _REASONING_DISABLED),
            "provider": copy.deepcopy(_PROVIDER_PIN),
        }

    def _apply_grounding_sampling(self, req: dict[str, Any]) -> None:
        """Sampling params live on the GROUNDING (reasoning-disabled) routes
        ONLY — never with reasoning enabled; the explicit grounding pin wins
        over the generic build_default kwarg."""
        temperature = self.grounding_temperature if self.grounding_temperature is not None else self.temperature
        if temperature is not None:
            req["temperature"] = temperature
        if self.top_p is not None:
            req["top_p"] = self.top_p

    def _base_request(self, messages: list[dict[str, Any]], max_tokens: Optional[int]) -> dict[str, Any]:
        req: dict[str, Any] = {"model": self.model_name, "messages": messages}
        cap = max_tokens if max_tokens is not None else self.max_tokens
        # max_tokens stays None by default on every route: the GraphPC runner
        # default is None (--max_tokens default None -> self.max_tokens None -> no
        # cap sent -> the endpoint's own default). The request parameter is
        # max_tokens — never max_completion_tokens, which on this endpoint is a
        # CATALOG field naming the ceiling (_MAX_TOKENS_CEILING), not an accepted
        # parameter; reasoning tokens are billed as output and counted
        # against max_tokens "on most providers" (Alibaba unnamed — only a live
        # call settles it), so a cap may bound the chain-of-thought as well as the
        # reply. An explicit cap is the caller's responsibility, and one above the
        # endpoint's ceiling is rejected HERE rather than as a provider 400 — the
        # ceiling is a guarantee, not a docstring claim. The launch regime is
        # unaffected: the arm sends no cap, so the success path is untouched.
        if cap is not None:
            if cap > _MAX_TOKENS_CEILING:
                raise ValueError(
                    f"qwen max_tokens={cap} exceeds the endpoint ceiling {_MAX_TOKENS_CEILING}"
                )
            req["max_tokens"] = cap
        # Sampling params are set on the grounding path only, mirroring the
        # MiniMax template (this backbone never sends them with reasoning on).
        return req

    def _invoke_grounding(
        self, *, system_prompt: str, user_prompt: str, observation: Any, max_tokens: Optional[int],
        json_schema: dict[str, Any], schema_name: Optional[str],
    ) -> dict[str, Any]:
        """Reasoning DISABLED + forced function-calling with the grammar schema.

        The two halves are INSEPARABLE and must ride in the SAME request: a
        forced ``tool_choice`` is only honoured with thinking off, so a later
        edit that sets one without the other turns grounding into free prose
        where a tool call belongs — silently, since the backbone would fall
        back to ``msg.content``. Keep them together."""
        req = self._base_request(self._messages(system_prompt, user_prompt, observation), max_tokens)
        req["extra_body"] = self._extra_body(False)
        req["tools"] = [{
            "type": "function",
            "function": {
                "name": schema_name or "emit",
                "description": "Emit the structured action payload.",
                "parameters": json_schema,
                "strict": True,
            },
        }]
        req["tool_choice"] = {"type": "function", "function": {"name": schema_name or "emit"}}
        self._apply_grounding_sampling(req)

        started = time.perf_counter()
        resp = self._create_with_backoff(req)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        arguments = tool_calls[0].function.arguments if tool_calls else (msg.content or "{}")
        parsed = self._extract_json_object(arguments)
        return self._notify_observer({
            "text": arguments,
            "raw_response": arguments,
            "response_output": [tc.model_dump() for tc in tool_calls] if tool_calls else [],
            "usage": resp.usage.model_dump() if resp.usage else {},
            "json": parsed,
            "diagnostics": self._diagnostics(
                route="grounding_tool", thinking=False, system_prompt=system_prompt, user_prompt=user_prompt,
                observation=observation, started=started,
            ),
        })

    def _invoke_reasoning(
        self, *, system_prompt: str, user_prompt: str, observation: Any, max_tokens: Optional[int],
        use_thinking: bool, json_schema: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Reasoning enabled (per use_thinking) + json_object + post-parse. The
        schema is injected into the prompt as an advisory hint (server
        json_schema is not used, see module docstring); closed enums are
        normalised downstream."""
        user = user_prompt
        if json_schema is not None:
            user = (
                f"{user_prompt}\n\nReturn a single JSON object conforming to this schema "
                f"(JSON only, no prose):\n{json.dumps(json_schema)}"
            )
        elif "json" not in user.lower():
            user = f"{user}\n\nReturn JSON only."
        req = self._base_request(self._messages(system_prompt, user, observation), max_tokens)
        req["extra_body"] = self._extra_body(use_thinking)
        req["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        resp = self._create_with_backoff(req)
        content = resp.choices[0].message.content or ""
        return self._notify_observer({
            "text": content,
            "raw_response": content,
            "response_output": [],
            "usage": resp.usage.model_dump() if resp.usage else {},
            "json": self._extract_json_object(content),
            "diagnostics": self._diagnostics(
                route="reasoning_json", thinking=use_thinking, system_prompt=system_prompt, user_prompt=user,
                observation=observation, started=started,
            ),
        })

    # ------------------------------------------------------------------
    # Consumer-facing contract (``Backbone``, backbone/base.py).
    # ------------------------------------------------------------------

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        observation: Any = None,  # screenshot attached when content["screenshot"] is present (OSWorld); text-only otherwise (WebArena).
        max_tokens: Optional[int] = None,
        use_thinking: bool = True,
        json_schema: Optional[dict[str, Any]] = None,
        schema_name: Optional[str] = None,
        grounding: bool = False,
    ) -> dict[str, Any]:
        # Route on the EXPLICIT grounding flag (the action call sites pass
        # grounding=True). _schema_has_pattern is only a defensive assertion here.
        has_pattern = json_schema is not None and _schema_has_pattern(json_schema)
        if grounding:
            if not has_pattern:
                logger.warning(
                    "qwen grounding=True but the schema carries no grammar pattern; "
                    "forced function-calling will run without a regex constraint."
                )
            return self._invoke_grounding(
                system_prompt=system_prompt, user_prompt=user_prompt, observation=observation,
                max_tokens=max_tokens, json_schema=json_schema, schema_name=schema_name,
            )
        if has_pattern:
            raise ValueError(
                "qwen complete_json: a pattern-bearing (grounding) schema was passed "
                "with grounding=False — grammar emission would silently not happen. "
                "Pass grounding=True at the action/grounding call site."
            )
        return self._invoke_reasoning(
            system_prompt=system_prompt, user_prompt=user_prompt, observation=observation,
            max_tokens=max_tokens, use_thinking=use_thinking, json_schema=json_schema,
        )

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        observation: Any = None,
        max_tokens: Optional[int] = None,
        use_thinking: bool = True,
    ) -> dict[str, Any]:
        req = self._base_request(self._messages(system_prompt, user_prompt, observation), max_tokens)
        req["extra_body"] = self._extra_body(use_thinking)
        started = time.perf_counter()
        resp = self._create_with_backoff(req)
        content = resp.choices[0].message.content or ""
        return self._notify_observer({
            "text": content, "raw_response": content, "response_output": [],
            "usage": resp.usage.model_dump() if resp.usage else {},
            "diagnostics": self._diagnostics(
                route="text", thinking=use_thinking,
                system_prompt=system_prompt, user_prompt=user_prompt, observation=observation, started=started,
            ),
        })
