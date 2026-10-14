"""MiniMax-M3 backbone (openai SDK over https://api.minimax.io/v1).

Satisfies ``Backbone`` (``src/agents/backbone/base.py``; the shared reasoning /
action agents call ``complete_json`` and read ``result["json"]`` +
``result["raw_response"]``), so those agents are REUSED unchanged with a
``MiniMaxAgent`` instance. Structural mirror of the DeepSeek backbone
(``src/agents/backbone/deepseek/main.py``); only the provider facts differ.

Structured-output support (MiniMax-M3):

  * ``response_format={"type": "json_schema", ...}`` is NOT supported — the
    request succeeds but the schema is silently ignored (a strict json_schema
    request returns free prose/markdown, no error). Worst case: a silent
    degrade, so it is never used here.
  * ``response_format={"type": "json_object"}`` IS honored — content comes
    back as a clean JSON object.
  * Forced function-calling (``tools`` + forced ``tool_choice``,
    ``strict:True``) works and returns well-formed JSON arguments with the
    schema's keys — and unlike DeepSeek it is accepted in thinking mode too.
    However a regex ``pattern`` inside the tool parameters is NOT enforced
    server-side (an un-bracketed string such as "click 5" comes through
    unchanged), so on MiniMax the WebArena grammar
    pattern is advisory: closed-vocabulary closure relies on the action
    prompt's grammar listing + downstream post-parse normalization / the
    translator's degrade path (design.md's sanctioned fallback — the
    Claude-path precedent).

Screenshot attachment (OSWorld 2.0, screenshot-primary): when the
``observation`` carries ``content["screenshot"]`` bytes (an ``Observation`` or a
bare dict), the user message becomes a content-part LIST with the image part
FIRST and the text part second — ``{"type": "image_url", "image_url": {"url":
"data:image/jpeg;base64,..."}}`` — on both routes and on ``complete_text``.
The bytes are re-encoded as JPEG quality 90 at native resolution (no resize),
exactly what OSWorld-V2's M3 agent sends (mm_agents/m3/agent.py
``_encode_screenshot``), so the model sees the same pixels as the baseline
arm. With NO screenshot the user message is the plain string, so a WebArena
(text-only) request carries no image part and no content-part list.

Two call shapes, routed inside ``complete_json`` by the EXPLICIT ``grounding``
flag the caller passes (the grounding/action call sites pass
``grounding=True``) — the DeepSeek routing, kept identical:

  * GROUNDING (``grounding=True`` — the action call): NON-thinking mode +
    FORCED function-calling with the schema as the tool's ``parameters``
    (``strict:True``). The tool-call arguments are the emitted JSON
    (structurally well-formed; ``pattern`` advisory, see above). Forced
    ``tool_choice`` is also accepted under thinking on MiniMax, but
    grounding stays non-thinking to mirror the DeepSeek routing —
    grounding is mechanical; deliberation belongs to the deciding call.
  * REASONING (``grounding=False`` — the default): thinking mode (per
    ``use_thinking``) + ``response_format=json_object`` + post-parse via
    ``extract_json_object`` (the schema is injected into the prompt as an
    advisory hint — no server json_schema on MiniMax; closed enums are
    normalised downstream). Reasoning schemas are unchanged.

Routing is on the explicit flag, NOT on whether the schema happens to carry a
``pattern`` — an implicit pattern-presence inference would silently flip any
future reasoning schema that gains a string ``pattern`` into non-thinking
forced-FC. ``_schema_has_pattern`` is retained only as a defensive assertion
(grounding without a pattern → warn; a pattern-bearing schema routed as
reasoning → raise, because grammar emission would then silently not happen).

OSWorld grounding route = ``complete_text(grounding=True)``: non-thinking,
the grounding sampling pin, an optional ``stop`` list (the action agent passes
``["</tool_call>"]``, V2's stop sequence), NO tools and NO response_format —
the model writes its trained free-text ``<tool_call>`` block, parsed
downstream (``src/benchmarks/osworld/m3_action_translator.py``). Forced
function-calling pushes M3 off that distribution and degrades its coordinate
grounding, while free text matches its trained distribution, so the forced-FC
``complete_json(grounding=True)`` route below stays the WebArena grammar
route only.

Sampling regime: ``grounding_temperature`` is the explicit pin for the
GROUNDING (non-thinking) routes only. OSWorld's ``build_default`` passes 0.6
= the V2 M3 runner's default ``--temperature`` (its non-thinking value) —
a documented instantiation, NOT a parity pin: the baseline arm runs
``--thinking_mode adaptive``, under which V2's M3Agent forces temperature 1
and drops top_p (mm_agents/m3/agent.py ``_build_request_body``). It is sent
only when not None, so a WebArena request (which passes nothing) carries no
sampling parameter. The generic ``temperature`` / ``top_p`` kwargs
(build_default symmetry across backbones) are grounding-only on this
backbone too and reach the native grounding route through the same helper
(OSWorld's build_default passes neither, so
nothing is sent); ``grounding_temperature`` takes precedence over
``temperature`` when both are set. NOTHING is ever sent on the thinking
route: MiniMax forces temperature under thinking, and a pinned value there
would silently change the deciding call's regime.

Thinking is binary (no effort tiers): ``{"type": "adaptive"}`` = on,
``{"type": "disabled"}`` = off (both accepted by the provider). It is pinned
EXPLICITLY via ``extra_body`` on every request — never the provider default (measurement
integrity). ``reasoning_split: True`` rides in every ``extra_body`` too: it
separates thinking into ``reasoning_content``/``reasoning_details`` so
``message.content`` stays clean — without it ``<think>`` tags pollute content
and break ``extract_json_object``. Usage counters mirror the GPT backbone:
input ← ``prompt_tokens``, output ← ``completion_tokens`` (reasoning tokens
are folded into completion_tokens; surfaced in the log line via
``completion_tokens_details.reasoning_tokens``), plus
``prompt_tokens_details.cached_tokens`` → ``total_cached_input_tokens``, the
part of the input the provider served from its implicit prefix cache at a
fifth of the price. That one is a SUBSET of the input total, not an addition
to it, and it is recorded because otherwise a run cannot say from its own
logs whether caching happened.
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

from src.agents.common.json_utils import extract_json_object
from src.agents.common.retry_policy import RetryPolicy

logger = logging.getLogger("desktopenv.agent")

# NOTE the /v1: MiniMax documents the versioned path (unlike DeepSeek's bare host).
_BASE_URL = "https://api.minimax.io/v1"
_DEFAULT_MODEL = "MiniMax-M3"


def _mm_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


def _mm_is_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, openai.RateLimitError)


# Reuse the shared retry policy (retry_policy.py) — same predicates the GPT /
# DeepSeek backbones use; MiniMax surfaces the same openai SDK exception types.
_MM_RETRY_POLICY = RetryPolicy(is_retryable=_mm_is_retryable, is_rate_limit=_mm_is_rate_limit)


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


class MiniMaxAgent:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = os.environ.get("MINIMAX_API_KEY"),
        system_prompt_suffix: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        base_url: str = _BASE_URL,
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
        # MiniMax-M3 controls reasoning via the binary thinking toggle, not an
        # effort level (deciding calls run thinking; grounding runs non-thinking).
        # Log a one-line notice when a value is actually supplied so the config
        # surface never silently no-ops.
        self.reasoning_effort = reasoning_effort
        if reasoning_effort is not None:
            logger.info(
                "minimax backbone: reasoning_effort=%r is ignored; the thinking-mode "
                "toggle governs deliberation (deciding=thinking, grounding=non-thinking).",
                reasoning_effort,
            )
        self.base_url = base_url
        self.client_factory = client_factory

        # Telemetry counters (same names the runner reads off the backbone).
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        # Of `total_input_tokens`, how much the provider served from its implicit
        # prefix cache at a fifth of the input price. A SUBSET of the input total,
        # never added to it — MiniMax reports `prompt_tokens` inclusive of the
        # cached part, so summing the two would double-count. Recorded because
        # caching is invisible otherwise: whether a run's leading frames are being
        # cached is the difference between a cheap run and an expensive one, and
        # without this counter that question cannot be answered from a run's logs
        # at all, only guessed at.
        self.total_cached_input_tokens = 0
        # Per-call observer (OSWorld adapter): called with the completed result
        # dict of EVERY model call, after the call returned. Diagnostics only —
        # the request bytes are untouched and an observer failure never fails
        # the call. None = nobody listening (WebArena).
        self.call_observer: Optional[Callable[[dict[str, Any]], None]] = None

    # ------------------------------------------------------------------
    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.api_key)
        # Read MINIMAX_API_KEY at CALL time (the __init__ default binds at import,
        # before the runner loads .env). Must be explicit: with a custom base_url
        # the openai SDK would otherwise fall back to OPENAI_API_KEY.
        api_key = self.api_key or os.environ.get("MINIMAX_API_KEY")
        return openai.OpenAI(api_key=api_key, base_url=self.base_url)

    def _create_with_backoff(self, request: dict[str, Any]):
        def _attempt():
            return self._build_client().chat.completions.create(**request)
        return _MM_RETRY_POLICY.run(_attempt)

    def reset_session(self) -> None:
        # MiniMax is stateless per call (no server-side response chaining like
        # the GPT Responses API); nothing to reset. Present so the WebArena shim's
        # reset() finds it.
        return None

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        return extract_json_object(text, log_label="MiniMax")

    def _record_usage(self, usage: Any) -> None:
        # Telemetry only — NEVER raises: a malformed provider usage payload
        # must not fail a scored call (the baseline shim's _log_minimax_usage
        # wraps its extraction the same way). Extraction completes before any
        # counter mutation, so on failure the counters simply do not advance
        # for this call and response processing continues untouched.
        try:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            # The implicit-prefix-cache hit, as the OpenAI-compatible route reports
            # it (the Anthropic route the control runs on calls the same quantity
            # `cache_read_input_tokens`). Absent on a route or a response that does
            # not cache, which reads as 0 — the same shape as a genuine miss, and
            # the honest one: both mean nothing was served from cache.
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0
            self.call_count += 1
            self.total_input_tokens += prompt
            self.total_output_tokens += completion
            self.total_cached_input_tokens += cached
            logger.info(
                "minimax usage: call=%d input=%d (cached=%d) output=%d (reasoning=%d) "
                "| cumulative input=%d (cached=%d) output=%d",
                self.call_count, prompt, cached, completion, reasoning,
                self.total_input_tokens, self.total_cached_input_tokens, self.total_output_tokens,
            )
        except Exception:
            logger.debug("minimax usage line skipped (malformed usage payload)", exc_info=True)

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
                logger.exception("minimax call observer failed; the call result is unaffected")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _screenshot_bytes(observation: Any) -> Optional[bytes]:
        """``observation.content["screenshot"]`` (an ``Observation``) or
        ``observation["screenshot"]`` (a bare dict); None when absent — the
        same lookup the Claude / GPT backbones use."""
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
        image's native size (RGB-converted; ``optimize`` on) — the encoding
        OSWorld-V2's M3 agent uses (mm_agents/m3/agent.py ``_encode_screenshot``
        with its JPEG/90 defaults). No resize: coordinate parity with the
        baseline arm rests on the model seeing the full-resolution frame."""
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
            # No image: the plain-string user message — byte-identical to the
            # text-only (WebArena) request shape.
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
        """The per-request extra_body — ALWAYS pins the thinking mode explicitly
        (never the provider default; measurement integrity) and reasoning_split
        so thinking is separated out of message.content."""
        return {
            "thinking": {"type": "adaptive" if use_thinking else "disabled"},
            "reasoning_split": True,
        }

    def _apply_grounding_sampling(self, req: dict[str, Any]) -> None:
        """Sampling params live on the GROUNDING (non-thinking) routes ONLY —
        never under thinking; the explicit grounding pin wins over the generic
        build_default kwarg."""
        temperature = self.grounding_temperature if self.grounding_temperature is not None else self.temperature
        if temperature is not None:
            req["temperature"] = temperature
        if self.top_p is not None:
            req["top_p"] = self.top_p

    def _base_request(self, messages: list[dict[str, Any]], max_tokens: Optional[int]) -> dict[str, Any]:
        req: dict[str, Any] = {"model": self.model_name, "messages": messages}
        cap = max_tokens if max_tokens is not None else self.max_tokens
        # max_tokens must stay GENEROUS-or-None on the reasoning path: thinking
        # mode folds reasoning tokens into completion_tokens, so a tight cap starves
        # the visible JSON and yields empty/truncated content that extract_json_object
        # can't parse. The GraphPC runner default is None (--max_tokens default None
        # -> self.max_tokens None -> no cap sent -> MiniMax's provider default), so no
        # cap is applied unless a caller explicitly asks for one. Only omit when None.
        if cap is not None:
            req["max_tokens"] = cap
        # Sampling params are set on the grounding path only, mirroring the
        # DeepSeek template (this backbone never sends them under thinking).
        return req

    def _invoke_grounding(
        self, *, system_prompt: str, user_prompt: str, observation: Any, max_tokens: Optional[int],
        json_schema: dict[str, Any], schema_name: Optional[str],
    ) -> dict[str, Any]:
        """NON-thinking + forced function-calling with the grammar schema."""
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
        self._record_usage(resp.usage)
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
        """Thinking (per use_thinking) + json_object + post-parse. The schema is
        injected into the prompt as an advisory hint (no server json_schema on
        MiniMax — silently ignored, see module docstring); closed enums are
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
        self._record_usage(resp.usage)
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
                    "minimax grounding=True but the schema carries no grammar pattern; "
                    "forced function-calling will run without a regex constraint."
                )
            return self._invoke_grounding(
                system_prompt=system_prompt, user_prompt=user_prompt, observation=observation,
                max_tokens=max_tokens, json_schema=json_schema, schema_name=schema_name,
            )
        if has_pattern:
            raise ValueError(
                "minimax complete_json: a pattern-bearing (grounding) schema was passed "
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
        grounding: bool = False,
        stop: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        req = self._base_request(self._messages(system_prompt, user_prompt, observation), max_tokens)
        if grounding:
            # OSWorld grounding route (module docstring): non-thinking + the
            # grounding sampling pin; use_thinking is ignored here on purpose —
            # grounding is mechanical, deliberation belongs to the deciding call.
            req["extra_body"] = self._extra_body(False)
            self._apply_grounding_sampling(req)
        else:
            req["extra_body"] = self._extra_body(use_thinking)
        if stop is not None:
            req["stop"] = list(stop)
        started = time.perf_counter()
        resp = self._create_with_backoff(req)
        self._record_usage(resp.usage)
        content = resp.choices[0].message.content or ""
        return self._notify_observer({
            "text": content, "raw_response": content, "response_output": [],
            "usage": resp.usage.model_dump() if resp.usage else {},
            "diagnostics": self._diagnostics(
                route="grounding_text" if grounding else "text", thinking=(False if grounding else use_thinking),
                system_prompt=system_prompt, user_prompt=user_prompt, observation=observation, started=started,
            ),
        })
