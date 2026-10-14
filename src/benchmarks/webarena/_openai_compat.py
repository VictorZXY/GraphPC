"""openai 0.27 → 2.x compatibility shim (the seam between two SDK generations).

The problem this solves lives entirely in one process. Our GPT backbone
(``src/agents/backbone/gpt/main.py``) targets the openai **2.x** Responses API
(``openai.OpenAI().responses``, ``openai.RateLimitError``, …). WebArena's LLM
scorer, however, was written against openai **0.27**: ``evaluation_harness``
imports ``llm_fuzzy_match`` / ``llm_ua_match``, which call
``llms/providers/openai_utils.generate_from_openai_chat_completion`` — and that
module does ``import openai.error`` at module load and uses a retry decorator
whose *default argument* ``errors=(openai.error.RateLimitError,)`` is evaluated
at import time. Under openai 2.x there is no ``openai.error`` submodule, so the
WebArena module crashes on **import**, before any scorer call is made.

We converge on a single openai 2.x install and re-create — additively — the
tiny 0.27 surface WebArena actually touches. Nothing here removes or shadows
2.x symbols (``openai.OpenAI``, ``openai.responses``, ``openai.RateLimitError``,
…); the GPT backbone keeps working unchanged. WebArena stays strictly
read-only — the compatibility work lives here, in our tree.

Surface re-created (only what WebArena uses):
  * ``openai.error`` — a real submodule (registered in ``sys.modules`` so
    ``import openai.error`` resolves) exposing exception *classes* aliased to
    the nearest 2.x class, so ``except openai.error.RateLimitError`` and the
    ``errors=(openai.error.RateLimitError,)`` default arg both bind to a class.
  * ``openai.ChatCompletion.create(**kwargs)`` — routes to the 2.x client and
    returns a plain dict (``resp.model_dump()``) so 0.27-style subscripting
    ``response["choices"][0]["message"]["content"]`` works. (On the qwen route
    only, a 200 that carries a provider failure raises instead of returning —
    see ``_qwen_call_failure``; so does an unusable 200 on the exact id
    ``deepseek-flash`` — see ``_deepseek_call_failure``. The exact id ``claude-sonnet-5`` is called on
    the native anthropic Messages API instead and mapped back to that dict —
    see ``_claude_chat_completion``.)
  * ``openai.Completion.create(**kwargs)`` — same shape, best-effort (not used
    by the evaluators, kept so the surface is consistent).

``openai.api_key`` / ``openai.organization`` assignment already works on the
2.x module object (plain attributes) — no shim needed for those.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from datetime import datetime, timezone

import httpx
import openai

_LOGGER = logging.getLogger(__name__)

# Sentinel so the shim is idempotent (safe to call twice in one process).
_INSTALLED_FLAG = "_graphpc_openai_027_compat_installed"

# WebArena's LLM judge (evaluation_harness/helper_functions.py) hardcodes the
# now-retired ``gpt-4-1106-preview`` model, passed as ``model=`` into
# ``generate_from_openai_chat_completion`` — whose calls land in the
# ``ChatCompletion.create`` shim below. WebArena is read-only, so we remap that
# exact id to a reachable successor here. Default is the closest successor,
# ``gpt-4-turbo-2024-04-09``; override via ``WEBARENA_JUDGE_MODEL``. Only this
# one retired id is remapped — every other model name passes through unchanged.
# MEASUREMENT CAVEAT: any judge != gpt-4-1106-preview shifts scores vs the
# published WebArena numbers; since 1106-preview is retired this is unavoidable,
# and the default minimizes drift.
_RETIRED_JUDGE_MODEL = "gpt-4-1106-preview"
_DEFAULT_JUDGE_MODEL = "gpt-4-turbo-2024-04-09"

# Module-level guard so the remap is logged once, not on every judge call.
_judge_remap_logged = False


def _remap_judge_model(kwargs: dict) -> dict:
    """Narrowly remap ONLY the retired WebArena judge model to a reachable
    successor, returning updated kwargs (any other model is returned as-is)."""
    global _judge_remap_logged
    if kwargs.get("model") != _RETIRED_JUDGE_MODEL:
        return kwargs
    target = os.environ.get("WEBARENA_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
    if not _judge_remap_logged:
        _LOGGER.info(
            "WebArena judge model %s -> %s (WEBARENA_JUDGE_MODEL)",
            _RETIRED_JUDGE_MODEL,
            target,
        )
        _judge_remap_logged = True
    return {**kwargs, "model": target}


# Reasoning-effort symmetry for the baseline arm. GraphPC (our arm) runs the
# Responses API with reasoning_effort=high; WebArena's bare-control PromptAgent
# reaches gpt-5* through this chat shim and passes no effort, so it gets the
# provider default (medium). That asymmetry would contaminate Δ attribution.
# When ``WEBARENA_AGENT_REASONING_EFFORT`` is set we inject it so control and
# treatment share the same reasoning budget. Explicit + off by default.
_REASONING_EFFORT_ENV = "WEBARENA_AGENT_REASONING_EFFORT"
# gpt-5* is the reasoning-capable family that accepts reasoning_effort on the
# chat path (verified live). The remapped judge (gpt-4-turbo-*) must NEVER get
# the parameter — the prefix check excludes it.
_REASONING_MODEL_PREFIX = "gpt-5"
# gpt-6-sol joins the family by its EXACT id, not by a "gpt-6" prefix: it is the
# only gpt-6 id verified against the live API, and a prefix would silently adapt an
# unverified sibling (e.g. drop a top_p it accepts). An unlisted sibling instead
# fails loudly — a 400 on max_tokens on its first call.
_REASONING_MODEL_IDS = ("gpt-6-sol",)

# Reasoning-model param compatibility on the 0.27 chat path. WebArena's
# generate_from_openai_chat_completion hard-codes 0.27-era kwargs
# (max_tokens=384, top_p=0.9). Live findings against gpt-5.4-mini:
#   * WITHOUT reasoning_effort: temperature/top_p are accepted (any value).
#   * WITH reasoning_effort (the baseline config, since we inject it): the model
#     requires max_completion_tokens instead of max_tokens, and rejects
#     non-default temperature/top_p (400 unsupported_parameter). top_p=0.9 (the
#     run.py default) is non-default → rejected.
# gpt-6-sol (verified against the live API) is stricter in one respect and the
# same in the rest: it rejects max_tokens even WITHOUT reasoning_effort; under
# effort it rejects top_p and accepts only the default temperature 1. The
# adaptation below produces exactly the shape it accepts.
# So _adapt_reasoning_model_params rewrites max_tokens->max_completion_tokens and
# drops temperature/top_p when reasoning_effort is in play, for the reasoning
# family only (never the gpt-4-turbo judge). WebArena is read-only, so this lives
# here. (_REASONING_EFFORT_ENV / _REASONING_MODEL_PREFIX / _REASONING_MODEL_IDS
# are defined once above.)

# Module-level guards so each adaptation is logged once, mirroring the judge-remap.
_reasoning_effort_injection_logged = False
_reasoning_param_adapt_logged = False


def _is_reasoning_capable_model(model) -> bool:
    return isinstance(model, str) and (
        model.startswith(_REASONING_MODEL_PREFIX) or model in _REASONING_MODEL_IDS
    )


def _inject_reasoning_effort(kwargs: dict) -> dict:
    """Inject ``reasoning_effort`` into the baseline PromptAgent's reasoning-family
    (gpt-5*, gpt-6-sol) chat calls when ``WEBARENA_AGENT_REASONING_EFFORT`` is
    set, for effort-symmetry with the GraphPC arm. Gated three ways: env must be
    set; the (post-remap) model must be in the reasoning family (never the
    gpt-4-turbo judge); and the caller must not have already passed
    reasoning_effort. Any gate unmet → kwargs unchanged."""
    global _reasoning_effort_injection_logged
    effort = os.environ.get(_REASONING_EFFORT_ENV)
    if not effort:
        return kwargs
    if not _is_reasoning_capable_model(kwargs.get("model")):
        return kwargs
    if kwargs.get("reasoning_effort") is not None:
        return kwargs
    if not _reasoning_effort_injection_logged:
        _LOGGER.info(
            "WebArena baseline agent calls: injecting reasoning_effort=%s (%s)",
            effort,
            _REASONING_EFFORT_ENV,
        )
        _reasoning_effort_injection_logged = True
    return {**kwargs, "reasoning_effort": effort}


def _adapt_reasoning_model_params(kwargs: dict) -> dict:
    """Make WebArena's 0.27-era chat kwargs compatible with a reasoning-family
    model (see the module NOTE above). Reasoning family only (gpt-5*, gpt-6-sol)
    — the gpt-4-turbo judge is excluded and passes through unchanged. Rewrites
    max_tokens -> max_completion_tokens; and when reasoning_effort is present,
    drops temperature/top_p (the model rejects non-default values there)."""
    global _reasoning_param_adapt_logged
    if not _is_reasoning_capable_model(kwargs.get("model")):
        return kwargs
    adapted = dict(kwargs)
    changes: list[str] = []
    if "max_tokens" in adapted and "max_completion_tokens" not in adapted:
        adapted["max_completion_tokens"] = adapted.pop("max_tokens")
        changes.append("max_tokens->max_completion_tokens")
    if adapted.get("reasoning_effort") is not None:
        for param in ("temperature", "top_p"):
            if param in adapted:
                adapted.pop(param)
                changes.append(f"drop {param}")
    if changes and not _reasoning_param_adapt_logged:
        _LOGGER.info(
            "WebArena gpt-5* chat calls: adapting reasoning-model params (%s)",
            ", ".join(changes),
        )
        _reasoning_param_adapt_logged = True
    return adapted


def _build_error_module() -> types.ModuleType:
    """Construct the ``openai.error`` submodule WebArena imports.

    Each name is aliased to the nearest openai 2.x exception *class* so that
    ``except openai.error.X`` catches the same object our 2.x code raises. The
    two names without a 2.x equivalent (``ServiceUnavailableError`` /
    ``InvalidRequestError``) fall back to the generic ``openai.APIError``.
    """
    error_mod = types.ModuleType("openai.error")
    # run.py's test() catches ``openai.error.OpenAIError`` at runtime (the 0.27
    # base class); alias it to the 2.x base so that except clause binds.
    error_mod.OpenAIError = openai.OpenAIError
    error_mod.RateLimitError = openai.RateLimitError
    error_mod.APIError = openai.APIError
    error_mod.APIConnectionError = openai.APIConnectionError
    error_mod.Timeout = openai.APITimeoutError
    error_mod.AuthenticationError = openai.AuthenticationError
    # No exact 2.x equivalent — alias to the generic API error class.
    error_mod.ServiceUnavailableError = openai.APIError
    error_mod.InvalidRequestError = openai.APIError
    return error_mod


# Baseline model routing: DeepSeek agent-model calls (deepseek-*) go to the
# DeepSeek endpoint; MiniMax agent-model calls (minimax-*, case-insensitive —
# the official id is "MiniMax-M3") go to the MiniMax OpenAI-compatible endpoint;
# Qwen agent-model calls (qwen-prefixed, case-insensitive — the slug is
# "qwen/qwen3.7-plus") go to OpenRouter; everything else (incl. the gpt-4-turbo
# judge and any gpt-* agent) stays on OpenAI. So a DeepSeek/MiniMax/Qwen
# baseline PromptAgent run reaches its provider for the agent calls while the
# LLM judge still hits OpenAI.
_DEEPSEEK_PREFIX = "deepseek"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_deepseek_thinking_logged = False
# deepseek-flash (V4.1-Flash). Its id and its effort pin are written out here
# for the control arm's standalone import graph; the GraphPC arm pins the same
# literals in src/agents/backbone/deepseek/main.py, and the two copies must stay
# equal, so change them together. EXACT id, never the prefix: validation and the
# effort pin apply to deepseek-flash; other model ids (deepseek-v4-flash,
# deepseek-v4-pro) take the default route, unmodified.
_DEEPSEEK_FLASH_MODEL = "deepseek-flash"
_DEEPSEEK_FLASH_REASONING_EFFORT = "high"
# Finish reasons the provider documents for a generation it interrupted itself.
_DEEPSEEK_INTERRUPTED_FINISH_REASONS = frozenset({"insufficient_system_resource", "aborted"})
# Per-process metering counters, so the control line carries the GraphPC line's
# call= counter and cumulative tail (one parser for both arms).
_deepseek_call_count = 0
_deepseek_total_input = 0
_deepseek_total_output = 0
_MINIMAX_PREFIX = "minimax"
_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
_minimax_thinking_logged = False
_QWEN_PREFIX = "qwen"
# The Qwen route runs over OPENROUTER — one host, one key var, one model slug,
# written out identically on both arms (the GraphPC arm pins the same literals
# in src/agents/backbone/qwen/main.py). The control arm keeps a STANDALONE
# import graph on purpose: it is the bare baseline, so it must not import our
# agent code to share a constant. The two arms' copies must stay equal, so
# change them together.
_QWEN_BASE_URL = "https://openrouter.ai/api/v1"
_QWEN_API_KEY_ENV = "OPENROUTER_API_KEY"
# The slug both arms send, and the one this arm ENFORCES: any other qwen slug
# raises on its first call (see ``_pin_qwen_route``), because the routing
# predicate is a prefix test and a real neighbour like ``qwen/qwen3-max`` would
# otherwise be pinned, metered and answered with a 200 while the GraphPC arm
# ran a different model — two arms, two models, both green. OpenRouter echoes
# the slug that was sent back in ``model`` (not a dated form), so that echo — the
# ``model=`` field of the metering line below — detects a rename, not a silent
# re-point of the alias.
_QWEN_MODEL = "qwen/qwen3.7-plus"
# Route pin, sent on EVERY qwen request of BOTH arms, byte-identical, and
# independent of the thinking injection: ``allow_fallbacks`` defaults to TRUE
# and OpenRouter load-balances across endpoints on price, so without this a
# request can be served by a different endpoint of the same slug.
# ``require_parameters`` is deliberate — it turns "this endpoint no longer
# supports tools / response_format" into a 503 instead of a silent parameter
# drop. Both failure modes are loud and falsifiable: an unavailable pinned
# endpoint is a 503, a conflict with an account-wide allowed-providers ceiling
# a 404. There is no silent reroute.
_QWEN_PROVIDER_PIN = {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}
# Thinking pin for the control arm's deciding calls. ``{"enabled": true}`` is
# "reasoning on, medium effort". The three forms that look equivalent are
# FORBIDDEN here: ``exclude: true`` and ``include_reasoning: false`` do NOT turn
# thinking off — the model still reasons, the tokens are still billed, they
# still count against max_tokens, and the only evidence it happened is removed;
# ``{}`` means "whatever the endpoint's default_enabled says", a provider
# default this route does not accept (never rely on provider defaults).
_QWEN_REASONING_ENABLED = {"enabled": True}
_qwen_thinking_logged = False
_qwen_route_pin_logged = False


def _is_minimax_model(model) -> bool:
    return isinstance(model, str) and model.lower().startswith(_MINIMAX_PREFIX)


def _is_qwen_model(model) -> bool:
    return isinstance(model, str) and model.lower().startswith(_QWEN_PREFIX)


def _make_client(model: Any = None):
    """Build a 2.x client for the given model. ``deepseek-*`` → the DeepSeek
    endpoint (DEEPSEEK_API_KEY); minimax-prefixed (case-insensitive) → the
    MiniMax endpoint (MINIMAX_API_KEY); qwen-prefixed (case-insensitive) →
    OpenRouter (OPENROUTER_API_KEY, read per call, passed explicitly so the SDK
    cannot fall back to OPENAI_API_KEY under a custom base URL — there is no
    base-URL override env var on this route); otherwise OpenAI (OPENAI_API_KEY /
    OPENAI_ORGANIZATION; empty org → None so no empty header is sent)."""
    if isinstance(model, str) and model.startswith(_DEEPSEEK_PREFIX):
        return openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=_DEEPSEEK_BASE_URL)
    if _is_minimax_model(model):
        return openai.OpenAI(api_key=os.environ.get("MINIMAX_API_KEY"), base_url=_MINIMAX_BASE_URL)
    if _is_qwen_model(model):
        return openai.OpenAI(
            api_key=os.environ.get(_QWEN_API_KEY_ENV),
            base_url=_QWEN_BASE_URL,
        )
    return openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=(os.environ.get("OPENAI_ORGANIZATION") or None),
    )


def _inject_deepseek_thinking(kwargs: dict) -> dict:
    """Effort-symmetry for a DeepSeek baseline: when
    ``WEBARENA_AGENT_REASONING_EFFORT`` is set and the model is ``deepseek-*``,
    run the baseline agent's deciding calls in THINKING mode (the DeepSeek analog
    of the GraphPC deciding calls' thinking). Never the gpt-4-turbo judge (not
    deepseek). Respects a caller-supplied ``extra_body``.

    Settled (.claude/docs/benchmarks/webarena.md, "Per-backbone route and launch
    facts"): this enables thinking but does NOT strip WebArena's temperature/top_p,
    which DeepSeek accepts under thinking (so no param-drop mirror exists), and it
    does not lift run.py's max_tokens=384, which is unsafe under thinking — a
    DeepSeek control launches with a larger --max_tokens (65536 on deepseek-flash:
    the GraphPC deciding call's effective cap, the provider default under
    thinking; a 16384 cap loses calls to an empty ``length`` finish). Not
    enforced here: run.py's --max_tokens is passed through."""
    global _deepseek_thinking_logged
    if not os.environ.get(_REASONING_EFFORT_ENV):
        return kwargs
    model = kwargs.get("model")
    if not (isinstance(model, str) and model.startswith(_DEEPSEEK_PREFIX)):
        return kwargs
    if "extra_body" in kwargs:
        return kwargs
    if not _deepseek_thinking_logged:
        _LOGGER.info(
            "WebArena baseline deepseek calls: enabling thinking mode (%s)", _REASONING_EFFORT_ENV
        )
        _deepseek_thinking_logged = True
    injected = {**kwargs, "extra_body": {"thinking": {"type": "enabled"}}}
    # deepseek-flash only: the pinned effort, the GraphPC deciding call's
    # literal. The env value stays an on-switch, as on every DeepSeek id.
    if model == _DEEPSEEK_FLASH_MODEL and kwargs.get("reasoning_effort") is None:
        injected["reasoning_effort"] = _DEEPSEEK_FLASH_REASONING_EFFORT
    return injected


class DeepSeekBaselineCallError(openai.APIError):
    """A deepseek-flash control call that came back HTTP 200 without a usable
    reply: no choices, a provider-interrupted finish, or ``length`` with empty
    content (``_deepseek_call_failure``). An ``openai.APIError`` subclass, like
    ``OpenRouterBaselineCallError``: run.py records the task as ``[OpenAI Error]``
    — provider infra, excluded — instead of handing PromptAgent an empty reply
    that would become a scored step. The GraphPC arm raises
    ``DeepSeekCallError`` on the same shapes."""

    def __init__(self, message, *, finish_reason=None, body=None):
        super().__init__(message, httpx.Request("POST", f"{_DEEPSEEK_BASE_URL}/chat/completions"), body=body)
        self.finish_reason = finish_reason


def _deepseek_call_failure(response: dict):
    """``(reason, finish_reason)`` when a deepseek-flash reply is unusable, else
    None — the GraphPC backbone's ``_validate_response`` rule, written out for
    this arm. A ``length`` reply with text in it is a truncated reply and flows
    on (counted where it fails to parse); only an EMPTY one is a failed call."""
    choices = response.get("choices")
    if not choices:
        return "no choices in the response body", None
    choice = choices[0] or {}
    finish = choice.get("finish_reason")
    if finish in _DEEPSEEK_INTERRUPTED_FINISH_REASONS:
        return f"generation interrupted by the provider: finish_reason={finish!r}", finish
    content = (choice.get("message") or {}).get("content") or ""
    if finish == "length" and not content.strip():
        return ("finish_reason='length' with empty content — reasoning spent the whole max_tokens budget; "
                "retrying does not help, the cap has to rise"), finish
    return None


def _log_deepseek_usage(response: dict, sent: dict) -> None:
    """One ``deepseek usage:`` line per deepseek-flash control call (the exact
    id; other DeepSeek ids take the default route and log no such line),
    with the GraphPC arm's key names and order (route=control), every per-call
    key before the ``|``, so one anchored parser reads both arms. ``thinking_sent`` /
    ``effort_sent`` are read off the kwargs sent (``absent`` when missing);
    ``model`` is the response's echo and ``fingerprint`` its
    ``system_fingerprint`` — the alias-move detector; ``ts`` prices the
    call (peak and off-peak rates). Logging only — never raises; a skipped line
    is a WARNING because its call went unrecorded."""
    global _deepseek_call_count, _deepseek_total_input, _deepseek_total_output
    try:
        usage = response.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        choice = (response.get("choices") or [{}])[0] or {}
        content = (choice.get("message") or {}).get("content") or ""
        call_input = int(usage.get("prompt_tokens") or 0)
        call_output = int(usage.get("completion_tokens") or 0)
        fields = (
            call_input, call_output, int(details.get("reasoning_tokens") or 0),
            ((sent.get("extra_body") or {}).get("thinking") or {}).get("type") or "absent",
            sent.get("reasoning_effort") or "absent",
            int(usage.get("prompt_cache_hit_tokens") or 0), int(usage.get("prompt_cache_miss_tokens") or 0),
            choice.get("finish_reason") or "none", "true" if not content.strip() else "false",
            response.get("model") or "none", response.get("system_fingerprint") or "none",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    except Exception:
        _LOGGER.warning("deepseek usage line skipped (malformed response)", exc_info=True)
        return
    _deepseek_call_count += 1
    _deepseek_total_input += call_input
    _deepseek_total_output += call_output
    _LOGGER.info(
        "deepseek usage: call=%d input=%d output=%d (reasoning=%d) route=control thinking_sent=%s effort_sent=%s "
        "cache_hit=%d cache_miss=%d finish=%s empty=%s model=%s fingerprint=%s ts=%s | cumulative input=%d output=%d",
        _deepseek_call_count, *fields, _deepseek_total_input, _deepseek_total_output,
    )


def _inject_minimax_thinking(kwargs: dict) -> dict:
    """Effort-symmetry for a MiniMax baseline: when
    ``WEBARENA_AGENT_REASONING_EFFORT`` is set and the model is minimax-prefixed
    (case-insensitive — the official id is ``MiniMax-M3``), run the baseline
    agent's deciding calls in THINKING mode. Like DeepSeek, MiniMax thinking is
    binary (no effort tiers), so the effort VALUE is ignored — the env var acts
    as an on-switch only. ``{"type": "adaptive"}`` == thinking ON for M3; the
    provider default is also on, but we pin it explicitly (measurement
    integrity: never rely on provider defaults). Never the gpt-4-turbo judge
    (not minimax). Respects a caller-supplied ``extra_body``.

    ``reasoning_split: True`` is REQUIRED on this path: without it MiniMax
    returns the thinking as ``<think>...</think>`` tags INSIDE
    ``message.content``, which breaks PromptAgent's action parsing. With it the
    thinking arrives in separate ``reasoning_content`` / ``reasoning_details``
    message fields and ``content`` stays clean (the shim's ``model_dump()``
    dict just carries the extra keys; 0.27-style subscripting ignores them).

    Settled (.claude/docs/benchmarks/webarena.md, "Per-backbone route and launch
    facts"): like deepseek, this enables thinking but does NOT strip WebArena's
    temperature/top_p or lift run.py's max_tokens=384
    (``_adapt_reasoning_model_params`` is gpt-5*-only, so raw kwargs reach MiniMax
    unmodified). MiniMax accepts the sampling parameters under thinking; 384
    returns ``finish_reason=length`` with empty content, so the control launches
    with --max_tokens 16384."""
    global _minimax_thinking_logged
    if not os.environ.get(_REASONING_EFFORT_ENV):
        return kwargs
    if not _is_minimax_model(kwargs.get("model")):
        return kwargs
    if "extra_body" in kwargs:
        return kwargs
    if not _minimax_thinking_logged:
        _LOGGER.info(
            "WebArena baseline minimax calls: enabling thinking mode + reasoning_split (%s)",
            _REASONING_EFFORT_ENV,
        )
        _minimax_thinking_logged = True
    return {**kwargs, "extra_body": {"thinking": {"type": "adaptive"}, "reasoning_split": True}}


def _log_minimax_usage(response: dict) -> None:
    """Per-call METERED cost telemetry for minimax-routed baseline calls.

    MiniMax exposes no balance endpoint to pay-as-you-go keys
    (``/v1/account/balance`` → 403 token_type_mismatch), so there is no
    DeepSeek-style balance-delta cost readout — run cost must be metered from
    per-call usage. One parseable line per successful create() call, via the
    module logger, so it lands in the lane run.log files, from which run cost is
    metered (without it, only a call-count estimate).
    Absent fields default to 0. Logging only — never raises (a malformed usage
    payload must not kill a scored call)."""
    try:
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        _LOGGER.info(
            "minimax usage: prompt=%s completion=%s reasoning=%s cached=%s",
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
            completion_details.get("reasoning_tokens") or 0,
            prompt_details.get("cached_tokens") or 0,
        )
    except Exception:
        _LOGGER.debug("minimax usage line skipped (malformed usage payload)", exc_info=True)


def _inject_qwen_thinking(kwargs: dict) -> dict:
    """Effort-symmetry for a Qwen baseline: when
    ``WEBARENA_AGENT_REASONING_EFFORT`` is set and the model is qwen-prefixed
    (case-insensitive — the slug is ``qwen/qwen3.7-plus``), run the baseline
    agent's deciding calls with reasoning ON, via
    ``extra_body={"reasoning": {"enabled": True}}`` (the SDK has no ``reasoning``
    kwarg). This endpoint's reasoning control has no effort scale and no token
    budget, so like DeepSeek/MiniMax the effort VALUE is ignored — the env var
    acts as an on-switch only. The endpoint's default is also on, but we pin it
    explicitly (measurement integrity: never rely on provider defaults). Never
    the gpt-4-turbo judge (not qwen).

    Only the ``reasoning`` key respects a caller-supplied ``extra_body``: a
    caller that passed its own ``reasoning`` keeps it, any other caller key is
    merged, and the route pin is applied regardless (see ``_pin_qwen_route``).

    The thinking arrives in the ``message.reasoning`` string and the
    ``message.reasoning_details`` array, not in ``content``, so ``content``
    stays clean for PromptAgent's action parsing; the shim's ``model_dump()``
    dict just carries the extra keys. [A] inline ``<think>`` tags in ``content``
    are undocumented but not excluded; if they appear, REPORT it, do not add a
    stripper.

    Settled (.claude/docs/benchmarks/webarena.md, "Qwen ``qwen3.7-plus`` over
    OpenRouter"): like deepseek/minimax, this enables thinking but does NOT strip
    WebArena's temperature/top_p (``_adapt_reasoning_model_params`` is
    gpt-5*-only, so raw kwargs reach the endpoint unmodified). Reasoning tokens
    are billed as output and bounded by ``max_tokens`` on this route, so the
    control launches with --max_tokens 16384 (not enforced here: run.py's
    --max_tokens is passed through). The REQUEST parameter this endpoint accepts
    is ``max_tokens``; its catalog ``max_completion_tokens`` (131,072) is the
    ceiling a launch cap must stay under, and the gpt-5*
    max_tokens->max_completion_tokens rewrite above is prefix-gated on "gpt-5"
    and never fires for a qwen slug."""
    global _qwen_thinking_logged
    if not os.environ.get(_REASONING_EFFORT_ENV):
        return kwargs
    if not _is_qwen_model(kwargs.get("model")):
        return kwargs
    extra_body = dict(kwargs.get("extra_body") or {})
    if "reasoning" in extra_body:
        return kwargs
    if not _qwen_thinking_logged:
        _LOGGER.info(
            "WebArena baseline qwen calls: enabling thinking mode (%s)", _REASONING_EFFORT_ENV
        )
        _qwen_thinking_logged = True
    extra_body["reasoning"] = dict(_QWEN_REASONING_ENABLED)
    return {**kwargs, "extra_body": extra_body}


def _pin_qwen_route(kwargs: dict) -> dict:
    """Pin every qwen request to the one pinned (``alibaba``) endpoint.

    UNCONDITIONAL and MERGED, which is the whole point: the pin rides in
    ``extra_body`` alongside whatever the caller already put there, it is never
    skipped because a caller supplied an ``extra_body``, and it overwrites a
    caller-supplied ``provider`` key. An unpinned request still returns HTTP
    200 — it is just served by whichever endpoint the router picked — so a
    skipped pin is a silent measurement failure, the exact thing the pin exists
    to prevent. Route pinning and thinking are separate concerns: the pin goes
    on every qwen call, whether or not the thinking injection fired.

    Also the arm's slug guard. Routing is a PREFIX test on purpose — narrowing
    it would send a mistyped qwen slug to OpenAI with ``OPENAI_API_KEY``, a
    worse failure — so a real neighbour such as ``qwen/qwen3-max`` would reach
    OpenRouter, be pinned, be metered and answer 200 while the GraphPC arm ran
    ``qwen/qwen3.7-plus``: two arms on two models, both green, and the
    same-model pairing quietly gone. Any qwen slug that is not the pinned one
    therefore raises a plain configuration error on the FIRST call of every
    task.

    What that guard does and does not do: WebArena's ``run.py`` wraps each task
    in a blanket ``except Exception``, so this raise is caught there, logged
    ``[Unhandled Error]`` and the loop moves to the next task. It therefore
    stops a wrong slug from being SCORED — no task produces a ``[Result]`` line,
    so every one is excluded — but it does NOT stop the run, which walks the
    whole list raising once per task. The half that stops the run is the
    launcher check in ``scripts/webarena/run_baseline.py``, before the first
    task starts. ``ValueError`` and not ``OpenRouterBaselineCallError``: the
    blanket handler excuses every exception class alike, so the class changes
    only the label in the log — and the honest label for our own
    misconfiguration is not "provider infra".

    Also emits the arm's config echo once, naming what the control arm SENDS.
    That is not evidence of what served: the served endpoint is only visible
    with the ``X-OpenRouter-Metadata`` header, which production requests do not
    send (they stay minimal), and per-call the response's ``model`` echo is the
    route record. Non-qwen models pass through untouched."""
    global _qwen_route_pin_logged
    model = kwargs.get("model")
    if not _is_qwen_model(model):
        return kwargs
    if model != _QWEN_MODEL:
        raise ValueError(
            f"qwen baseline configuration error: this arm sends {_QWEN_MODEL!r}, but the call "
            f"asked for {model!r}. Every qwen slug routes to OpenRouter, so a wrong one would "
            "be pinned, metered and answered normally while the GraphPC arm ran the pinned "
            "model — the two arms would score different models. Fix the launch's --model."
        )
    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body["provider"] = dict(_QWEN_PROVIDER_PIN)
    pinned = {**kwargs, "extra_body": extra_body}
    if not _qwen_route_pin_logged:
        _LOGGER.info(
            "config echo — what we send, not evidence of what served: qwen route host=%s "
            "model=%s extra_body=%s",
            _QWEN_BASE_URL,
            pinned.get("model"),
            extra_body,
        )
        _qwen_route_pin_logged = True
    return pinned


class OpenRouterBaselineCallError(openai.APIError):
    """A qwen call that came back HTTP 200 carrying a provider failure.

    Subclass of ``openai.APIError`` on purpose: ``_build_error_module`` aliases
    ``openai.APIError`` / ``openai.OpenAIError`` into the re-created
    ``openai.error`` surface, so WebArena's ``except openai.error.OpenAIError``
    binds and run.py records the step as ``[OpenAI Error]`` — provider/API
    infra, excluded — instead of KeyError-ing into ``[Unhandled Error]``, which
    the runner reads as env/eval infra. Carries the provider's typed
    ``error_type`` / ``provider_code`` when the body supplies them; they are
    None when the body reports the failure only through ``finish_reason``. No
    status code is invented — the transport status really was 200."""

    def __init__(self, message, *, error_type=None, provider_code=None, body=None):
        super().__init__(
            message,
            httpx.Request("POST", f"{_QWEN_BASE_URL}/chat/completions"),
            body=body,
        )
        self.error_type = error_type
        self.provider_code = provider_code


def _qwen_error_fields(error) -> tuple:
    """(error_type, provider_code) out of an OpenRouter error object — the typed
    code lives in ``error.metadata``; ``provider_code`` is omitted on 500s."""
    if not isinstance(error, dict):
        return None, None
    metadata = error.get("metadata") or {}
    return metadata.get("error_type"), metadata.get("provider_code")


def _qwen_call_failure(response: dict):
    """Decide whether a 200 from the qwen route is a usable answer. Returns None
    when it is, else ``(reason, error_type, provider_code)``.

    A 200 here is not a success: if the provider fails after the headers are
    sent, OpenRouter answers 200 with a body that holds only an ``error`` object
    and NO ``choices``, and a provider error that interrupts generation lands as
    a choice carrying its own error object. Four shapes are failures: a
    body-level ``error``; missing/empty ``choices``; a choice with an ``error``
    envelope OR ``finish_reason == "error"``; and ``finish_reason == "length"``
    with empty content (no non-whitespace characters), which is reasoning having
    spent the whole ``max_tokens`` budget — retrying does not help. The error
    envelope is a trigger in its own right, not just a place to read
    ``error_type`` from: it is the provider saying generation failed, whatever
    finish reason rides with it, and treating it as a completed answer would
    feed partial or empty content into PromptAgent and score it. The GraphPC
    arm's validator applies the same rule, so the two arms classify the same
    body the same way.

    Two neighbouring shapes are deliberately NOT failures, because calling them
    provider infra would take a real agent outcome out of the denominator.
    ``length`` with NON-empty content is an ordinary truncated reply.
    ``finish_reason == "content_filter"`` is how this API returns the MODEL's
    own refusal (with ``message.refusal``) as a completed response — this
    endpoint is not moderated, so it is the model's behaviour on a scored step,
    not a provider-side block. It flows through the normal path and is recorded
    instead (``_log_qwen_content_filter``). The line runs between the refusal
    and the error envelope: a bare ``content_filter`` is counted, a
    ``content_filter`` that ALSO carries a choice-level ``error`` is a failed
    call, because then the provider is reporting a failure and not relaying a
    refusal."""
    body_error = response.get("error")
    if body_error:
        error_type, provider_code = _qwen_error_fields(body_error)
        message = body_error.get("message") if isinstance(body_error, dict) else None
        return (f"body-level error: {message or repr(body_error)}", error_type, provider_code)
    choices = response.get("choices")
    if not choices:
        return ("no choices in the response body", None, None)
    choice = choices[0] or {}
    choice_error = choice.get("error")
    error_type, provider_code = _qwen_error_fields(choice_error)
    finish_reason = choice.get("finish_reason")
    if choice_error or finish_reason == "error":
        detail = ("with a choice-level error envelope" if choice_error
                  else "without an error envelope")
        return (f"finish_reason={finish_reason!r} {detail}", error_type, provider_code)
    content = (choice.get("message") or {}).get("content") or ""
    if finish_reason == "length" and not content.strip():
        return ("finish_reason='length' with empty content — reasoning spent the whole "
                "max_tokens budget; retrying does not help, the cap has to rise",
                error_type, provider_code)
    return None


def _log_qwen_content_filter(response: dict) -> None:
    """Record a model refusal so it is visible in the run log instead of silent.

    ``finish_reason == "content_filter"`` is not treated as a failed call (see
    ``_qwen_call_failure``): the step continues down the normal path and WebArena
    fails it its own way, which is a counted outcome. But a refusal is worth
    seeing when reading a lane log, so one line names the finish reason and
    whether the model supplied a ``refusal`` payload. Emitted only for calls
    that PASSED validation — a ``content_filter`` carrying an error envelope is
    a failed call, and recording it as a counted refusal would contradict the
    raise. Logging only — never raises."""
    try:
        choice = (response.get("choices") or [{}])[0] or {}
        if choice.get("finish_reason") != "content_filter":
            return
        message = choice.get("message") or {}
        _LOGGER.info(
            "qwen refusal: finish_reason='content_filter' refusal=%s model=%s — the model "
            "declined; counted as an agent outcome, not provider infra",
            "present" if message.get("refusal") else "absent",
            response.get("model") or "none",
        )
    except Exception:
        _LOGGER.debug("qwen refusal line skipped (malformed response)", exc_info=True)


def _log_qwen_usage(response: dict) -> None:
    """Per-call METERED cost telemetry for qwen-routed baseline calls, the
    ``_log_minimax_usage`` pattern: one parseable line per create() call, via
    the module logger, so it lands in the lane run.log files (a reader keys on
    the literal ``qwen usage:`` prefix). The key names are the ones the GraphPC
    arm's line uses, so a single parser reads both arms. ``cost=`` is what
    OpenRouter charged for the call (``usage.cost``, always returned) and
    ``model=`` is the response's model echo — the slug that was sent, so it
    detects a rename, not a silent re-point of the alias.
    ``input=`` is also what a lane tripwire can key on to catch the 3x price
    step above 256k prompt tokens. Absent numeric fields default to 0; an absent
    cost prints ``none``
    (a cost of 0 is a real value and prints as 0). Logging only — never raises
    (a malformed usage payload must not kill a scored call)."""
    try:
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        cost = usage.get("cost")
        _LOGGER.info(
            "qwen usage: input=%s output=%s reasoning=%s cached=%s cost=%s model=%s",
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
            completion_details.get("reasoning_tokens") or 0,
            prompt_details.get("cached_tokens") or 0,
            cost if cost is not None else "none",
            response.get("model") or "none",
        )
    except Exception:
        _LOGGER.debug("qwen usage line skipped (malformed usage payload)", exc_info=True)


def _log_gpt_usage(response: dict, sent: dict) -> None:
    """Per-call metering AND the effort record for reasoning-family control calls.

    One parseable line per create() call, the ``_log_qwen_usage`` pattern, with
    every per-call key before any ``|`` so one anchored parser reads this line
    and the GraphPC arm's ``gpt usage:`` line alike. ``effort_sent=`` is the
    ``reasoning_effort`` in the kwargs actually sent (``absent`` when none was):
    Chat Completions echoes no effort, so what was sent is the only per-call
    record of it. ``cached=`` / ``cache_write=`` are billed at 0.1x / 1.25x, so
    token counts alone misprice the call. ``finish=`` / ``refusal=`` /
    ``empty=`` make the two silent outcomes visible — a refusal and a ``length``
    stop with empty content both reach PromptAgent as unparseable text and are
    counted there. ``model=`` is the response echo. Logging only — never raises;
    a skipped line is logged at WARNING because it is a call whose effort went
    unrecorded."""
    try:
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        choice = (response.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        _LOGGER.info(
            "gpt usage: input=%s output=%s reasoning=%s cached=%s cache_write=%s model=%s "
            "effort_sent=%s finish=%s refusal=%s empty=%s",
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
            completion_details.get("reasoning_tokens") or 0,
            prompt_details.get("cached_tokens") or 0,
            prompt_details.get("cache_write_tokens") or 0,
            response.get("model") or "none",
            sent.get("reasoning_effort") or "absent",
            choice.get("finish_reason") or "none",
            "present" if message.get("refusal") else "absent",
            "true" if not (message.get("content") or "").strip() else "false",
        )
    except Exception:
        _LOGGER.warning("gpt usage line skipped (malformed response)", exc_info=True)


# Claude Sonnet 5 control route.
# The EXACT id only (a sibling id is not adapted: it keeps the OpenAI path and fails
# loudly there), called on the NATIVE anthropic Messages API — never an
# OpenAI-compatible endpoint. The control arm keeps a STANDALONE import graph on
# purpose (it is the bare baseline), so it writes the route literals out for
# itself; the GraphPC arm pins the same values in
# src/agents/backbone/claude/main.py, and the two copies must stay equal, so
# change them together.
_CLAUDE_MODEL = "claude-sonnet-5"
_CLAUDE_API_KEY_ENV = "ANTHROPIC_API_KEY"
_CLAUDE_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
# Client literals: the SDK is the one retry layer, and the explicit timeout keeps the
# SDK's client-side "streaming is required" guard out of the way of a large cap.
_CLAUDE_TIMEOUT_SECONDS = 600
_CLAUDE_MAX_RETRIES = 4
# Thinking pin for the control arm's calls, sent together with the effort.
_CLAUDE_THINKING = {"type": "adaptive"}
# The two billing markers inside an error body (see _claude_billing_failure).
_CLAUDE_SPEND_LIMIT_CODE = "enforced_spend_limit_reached"
_CLAUDE_USAGE_LIMIT_TEXT = "specified api usage limits"
_claude_injection_logged = False
# Per-process metering counters, so the control line carries the GraphPC line's
# call= counter and cumulative tail (one parser, one reconciliation for both arms).
_claude_call_count = 0
_claude_total_input = 0
_claude_total_output = 0


class ClaudeBaselineCallError(openai.APIError):
    """Any failure of a claude-sonnet-5 control call: an anthropic SDK error, a
    client-side error inside the SDK, or a malformed reply.

    Subclass of ``openai.APIError`` on purpose, like ``OpenRouterBaselineCallError``:
    run.py catches ``openai.error.OpenAIError`` and records ``[OpenAI Error]`` —
    provider/API infra, excluded — while a ``ValueError`` escaping ``next_action``
    would be turned into a stop action and SCORED (run.py:296-301). Never a
    subclass of ``openai.RateLimitError``: WebArena's retry decorator retries that
    class, which would add a second retry layer on this arm only. The message
    names the original class and carries its message."""

    def __init__(self, message: str) -> None:
        super().__init__(message, httpx.Request("POST", _CLAUDE_MESSAGES_URL), body=None)


class ClaudeBillingError(ClaudeBaselineCallError):
    """A billing refusal: 402, the organisation spend-cap 429
    (``enforced_spend_limit_reached``), or the user-set usage-limit 400. The same
    class name the GraphPC arm raises, so one lane tripwire (run_lanes.py) can
    match both arms' lines; the message carries the status, the error type and
    the error code."""


def _claude_billing_failure(status, body):
    """``(message, error_type, error_code)`` when an error response is a billing
    refusal, else None — the GraphPC backbone's rule, written out for this arm."""
    error = body.get("error") if isinstance(body, dict) else None
    error = error if isinstance(error, dict) else {}
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    error_type, error_code = error.get("type"), details.get("error_code")
    message = str(error.get("message") or "")
    if not (status == 402
            or (status == 429 and error_code == _CLAUDE_SPEND_LIMIT_CODE)
            or (status == 400 and _CLAUDE_USAGE_LIMIT_TEXT in message.lower())):
        return None
    return (f"claude billing failure: status={status} type={error_type} error_code={error_code or 'none'} "
            f"message={message!r}", error_type, error_code)


def _is_claude_model(model) -> bool:
    return model == _CLAUDE_MODEL


def _claude_request(kwargs: dict) -> dict:
    """WebArena's chat kwargs -> the Messages API request.

    PromptAgent sends its intro and its few-shot examples as ``role: system``
    messages (the examples carry ``name: example_user`` / ``example_assistant``);
    the Messages API has one ``system`` slot and no ``name``. They are joined with
    "\n" into ``system``, in order — the mapping Anthropic documents for its own
    OpenAI-format layer — and the ``name`` labels are lost (a disclosure). The
    other messages keep their role and content. ``temperature`` is kept (run.py
    passes Sonnet 5's only accepted value, 1.0; the launcher refuses any other);
    ``top_p`` is dropped (any value is a 400); ``max_tokens`` is the launch cap;
    ``stop`` becomes ``stop_sequences`` only when non-empty (the chat path always
    passes None). Thinking and effort are injected from
    ``WEBARENA_AGENT_REASONING_EFFORT`` exactly when the other routes' injections
    fire — set by run_baseline.py on every launch — and ride in ``extra_body``,
    the GraphPC arm's encoding."""
    global _claude_injection_logged
    messages = kwargs.get("messages") or []
    system = "\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system")
    request = {"model": kwargs.get("model"), "max_tokens": kwargs.get("max_tokens")}
    if system:
        request["system"] = system
    request["messages"] = [{"role": message.get("role"), "content": message.get("content")}
                           for message in messages if message.get("role") != "system"]
    if kwargs.get("temperature") is not None:
        request["temperature"] = kwargs["temperature"]
    stop = kwargs.get("stop")
    if stop:
        request["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    effort = os.environ.get(_REASONING_EFFORT_ENV)
    if effort:
        request["extra_body"] = {"thinking": dict(_CLAUDE_THINKING), "output_config": {"effort": effort}}
        if not _claude_injection_logged:
            _LOGGER.info(
                "WebArena baseline claude calls: injecting thinking=%s effort=%s (%s)",
                _CLAUDE_THINKING["type"], effort, _REASONING_EFFORT_ENV,
            )
            _claude_injection_logged = True
    return request


def _log_claude_usage(response, request: dict) -> None:
    """One ``claude usage:`` line per call, with the GraphPC arm's key names and
    order (route=control), every per-call key before the ``|``, so one anchored
    parser reads both arms. ``effort_sent`` / ``thinking_sent`` are read off the
    request dict handed to the SDK, ``absent`` when missing — Claude echoes
    neither. ``stop`` / ``refusal`` / ``empty`` make a refusal and a
    ``max_tokens`` stop with no text visible; both reach PromptAgent as
    unparseable text and are counted there. Logging only — never raises; a
    skipped line is a WARNING because its call's regime went unrecorded."""
    global _claude_call_count, _claude_total_input, _claude_total_output
    try:
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None) or {}
        thinking = details.get("thinking_tokens") if isinstance(details, dict) else getattr(
            details, "thinking_tokens", 0)
        call_input = int(usage.input_tokens or 0)
        call_output = int(usage.output_tokens or 0)
        extra_body = request.get("extra_body") or {}
        empty = not any(getattr(block, "type", None) == "text" and getattr(block, "text", "")
                        for block in response.content or [])
        fields = (
            (extra_body.get("output_config") or {}).get("effort") or "absent",
            (extra_body.get("thinking") or {}).get("type") or "absent",
            call_input,
            call_output,
            int(thinking or 0),
            int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            response.model or "none",
            response.stop_reason or "none",
            "present" if response.stop_reason == "refusal" else "absent",
            "true" if empty else "false",
        )
    except Exception:
        _LOGGER.warning("claude usage line skipped (malformed response)", exc_info=True)
        return
    _claude_call_count += 1
    _claude_total_input += call_input
    _claude_total_output += call_output
    _LOGGER.info(
        "claude usage: call=%d route=control effort_sent=%s thinking_sent=%s input=%d output=%d thinking=%d "
        "cache_read=%d cache_write=%d model=%s stop=%s refusal=%s empty=%s | cumulative input=%d output=%d",
        _claude_call_count, *fields, _claude_total_input, _claude_total_output,
    )
    if response.stop_reason == "refusal":
        stop_details = getattr(response, "stop_details", None) or {}
        _LOGGER.info(
            "claude refusal: route=control category=%s explanation=%r (counted as an agent outcome)",
            stop_details.get("category") if isinstance(stop_details, dict) else None,
            stop_details.get("explanation") if isinstance(stop_details, dict) else None,
        )


def _claude_chat_completion(kwargs: dict) -> dict:
    """The claude-sonnet-5 control call: native ``messages.create``, then the reply
    mapped back to the 0.27 dict PromptAgent subscripts. ``content`` is the text
    blocks joined — never a thinking block, and ``""`` (not None, which PromptAgent
    would read as the string "None") when there are none. Every failure leaves as a
    ``ClaudeBaselineCallError`` (``ClaudeBillingError`` for billing)."""
    import anthropic

    request = _claude_request(kwargs)
    try:
        client = anthropic.Anthropic(api_key=os.environ.get(_CLAUDE_API_KEY_ENV),
                                     timeout=_CLAUDE_TIMEOUT_SECONDS, max_retries=_CLAUDE_MAX_RETRIES)
        response = client.messages.create(**request)
    except anthropic.APIStatusError as exc:
        billing = _claude_billing_failure(exc.status_code, exc.body)
        if billing is not None:
            raise ClaudeBillingError(billing[0]) from exc
        raise ClaudeBaselineCallError(f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — a ValueError here would be scored by run.py
        raise ClaudeBaselineCallError(f"{type(exc).__name__}: {exc}") from exc
    _log_claude_usage(response, request)
    try:
        content = "\n".join(block.text for block in response.content or []
                            if getattr(block, "type", None) == "text" and getattr(block, "text", None))
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": response.model,
            "choices": [{"index": 0, "finish_reason": response.stop_reason,
                         "message": {"role": "assistant", "content": content}}],
            "usage": response.usage.model_dump(),
        }
    except Exception as exc:  # noqa: BLE001
        raise ClaudeBaselineCallError(f"malformed claude reply: {type(exc).__name__}: {exc}") from exc


class _ChatCompletionShim:
    """0.27 ``openai.ChatCompletion`` facade over the 2.x client.

    ``create`` forwards model / messages / temperature / max_tokens / top_p /
    stop unchanged (0.27 passes ``stop=None`` when there is no stop token — the
    2.x SDK tolerates that) and returns a dict so downstream 0.27-style
    subscripting keeps working.
    """

    @staticmethod
    def create(**kwargs):
        kwargs = _remap_judge_model(kwargs)
        if _is_claude_model(kwargs.get("model")):
            return _claude_chat_completion(kwargs)      # claude-sonnet-5 only; native Messages API
        # After remap so the judge (now gpt-4-turbo-*) is excluded by the
        # reasoning-family gate and never receives reasoning_effort / param
        # adaptation. The adapt step runs last so it sees the injected
        # reasoning_effort.
        kwargs = _inject_reasoning_effort(kwargs)      # gpt-5* / gpt-6-sol only
        kwargs = _adapt_reasoning_model_params(kwargs)  # gpt-5* / gpt-6-sol only
        kwargs = _inject_deepseek_thinking(kwargs)      # deepseek-* only
        kwargs = _inject_minimax_thinking(kwargs)       # minimax-* only
        kwargs = _inject_qwen_thinking(kwargs)          # qwen-* only
        kwargs = _pin_qwen_route(kwargs)                # qwen-* only; unconditional
        resp = _make_client(kwargs.get("model")).chat.completions.create(**kwargs)
        response = resp.model_dump()
        if _is_reasoning_capable_model(kwargs.get("model")):
            _log_gpt_usage(response, kwargs)            # gpt-5* / gpt-6-sol only; logging only
        if kwargs.get("model") == _DEEPSEEK_FLASH_MODEL:
            # Validate, meter, then raise: a billed failure still reaches the
            # record, and nothing reaches PromptAgent's response["choices"][0]
            # on an unusable body. deepseek-flash only (exact id).
            failure = _deepseek_call_failure(response)
            _log_deepseek_usage(response, kwargs)
            if failure is not None:
                reason, finish_reason = failure
                raise DeepSeekBaselineCallError(
                    f"deepseek-flash call returned HTTP 200 without a usable reply: {reason}",
                    finish_reason=finish_reason, body=response,
                )
        if _is_minimax_model(kwargs.get("model")):
            _log_minimax_usage(response)                # minimax-* only; logging only
        if _is_qwen_model(kwargs.get("model")):
            # Order is load-bearing: validate, THEN meter, THEN raise. A
            # provider failure that still carries usage is a billed partial, and
            # skipping its metering line would drop real cost from the record;
            # raising before the dict is returned is what keeps WebArena away
            # from response["choices"][0] on a body that has no choices.
            failure = _qwen_call_failure(response)
            if failure is None or response.get("usage"):
                _log_qwen_usage(response)
            if failure is None:
                _log_qwen_content_filter(response)      # a refusal is recorded, not raised
            if failure is not None:
                reason, error_type, provider_code = failure
                raise OpenRouterBaselineCallError(
                    f"qwen call returned HTTP 200 carrying a provider failure: {reason} "
                    f"(error_type={error_type!r}, provider_code={provider_code!r})",
                    error_type=error_type,
                    provider_code=provider_code,
                    body=response,
                )
        return response


class _CompletionShim:
    """0.27 ``openai.Completion`` facade (best-effort — not used by the
    evaluators, kept so the re-created surface is consistent)."""

    @staticmethod
    def create(**kwargs):
        resp = _make_client().completions.create(**kwargs)
        return resp.model_dump()


def install_openai_027_compat() -> None:
    """Additively patch the live (2.x) ``openai`` module with the 0.27 surface
    WebArena's evaluator needs. Idempotent; call before WebArena /
    ``evaluation_harness`` / ``llms`` is imported anywhere in the process."""
    if getattr(openai, _INSTALLED_FLAG, False):
        return

    error_mod = _build_error_module()
    # Register both ways: sys.modules so ``import openai.error`` resolves, and
    # the attribute so ``openai.error.X`` works without the import machinery.
    sys.modules["openai.error"] = error_mod
    openai.error = error_mod

    openai.ChatCompletion = _ChatCompletionShim
    openai.Completion = _CompletionShim

    setattr(openai, _INSTALLED_FLAG, True)
