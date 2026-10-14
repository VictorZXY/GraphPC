"""DeepSeek V4 backbone (openai SDK over https://api.deepseek.com).

Satisfies ``Backbone`` (``src/agents/backbone/base.py``; the shared reasoning and
action agents call ``complete_json`` and read ``result["json"]`` +
``result["raw_response"]``), so those agents are REUSED unchanged with a
``DeepSeekAgent`` instance.

DeepSeek V4 is a hybrid thinking model. Two call shapes, routed inside
``complete_json`` by the EXPLICIT ``grounding`` flag the caller passes (the
grounding/action call sites pass ``grounding=True``):

  * GROUNDING (``grounding=True`` — the action call): NON-thinking mode + FORCED
    function-calling with the schema as the tool's ``parameters``
    (``strict:True``; forced ``tool_choice`` is only accepted in non-thinking
    mode). ``strict`` is a hint, not an enforcement: on ``deepseek-flash`` it
    enforces neither the grammar ``pattern`` nor an ``enum`` (see below), so the
    tool-call arguments are validated downstream.
  * REASONING (``grounding=False`` — the default): thinking mode (per
    ``use_thinking``) + ``response_format=json_object`` + post-parse via
    ``extract_json_object`` (the Claude-path precedent — no server json_schema on
    DeepSeek; the schema is injected into the prompt as an advisory hint and
    closed enums are normalised downstream). Reasoning schemas are unchanged.

Routing is on the explicit flag, NOT on whether the schema happens to carry a
``pattern`` — an implicit pattern-presence inference would silently flip any
future reasoning schema that gains a string ``pattern`` into non-thinking
forced-FC. ``_schema_has_pattern`` is retained only as a defensive assertion
(grounding without a pattern → warn; a pattern-bearing schema routed as
reasoning → raise, because grammar enforcement would then silently not happen).

Thinking is toggled per call via ``extra_body={"thinking": {"type": ...}}`` on the
single ``deepseek-v4-flash`` id (the ``deepseek-chat``/``-reasoner`` aliases
are deprecated and not used). Usage counters mirror the GPT backbone:
input ← ``prompt_tokens``, output ← ``completion_tokens`` (reasoning tokens are
folded into completion_tokens; surfaced in the log line when present).

The id ``deepseek-flash`` (V4.1-Flash) gets two adaptations keyed on that
EXACT id; other model ids take the default route, unmodified. Every thinking-on
request also sends ``reasoning_effort: "high"`` (the documented effort form; the
provider default is high too, but the route pins it rather than relying on the
default), and every response is validated before use (``_validate_response``):
no choices, a provider-interrupted finish (``insufficient_system_resource`` /
``aborted``) or a ``length`` finish with nothing in it is a FAILED call and raises
``DeepSeekCallError``, which the runner classifies as infra by class name. The
control arm's shim applies the same rule. ``strict`` on the grounding tool is a
hint only: for ``deepseek-flash``, neither the standard URL nor ``/beta``
enforces a ``pattern`` or an ``enum``, so an off-grammar action string is caught
downstream exactly as on the routes without ``strict``.

Every metered call logs one ``deepseek usage:`` line, for every DeepSeek id: the
counters above plus the cache split, the finish reason, whether the reply was
empty, the response's ``model`` echo and ``system_fingerprint`` (the alias-move
detector: the echo names the id that was SENT, the fingerprint the backend that
served it) and a UTC timestamp (the price has peak and off-peak hours). Logging
only; no request byte depends on it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import openai

from src.agents.common.json_utils import extract_json_object
from src.agents.common.retry_policy import RetryPolicy

logger = logging.getLogger("desktopenv.agent")

_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-v4-flash"
# The deepseek-flash id and its effort pin, both arms. EXACT id, never a
# prefix: validation and the effort pin apply to deepseek-flash; other model ids
# (deepseek-v4-flash, deepseek-v4-pro) take the default route, unmodified. The
# control arm's shim (src/benchmarks/webarena/_openai_compat.py) writes the same
# literals out for itself; the two copies must stay equal, so change them together.
_FLASH_MODEL = "deepseek-flash"
_FLASH_REASONING_EFFORT = "high"
# Finish reasons the provider documents for a generation it interrupted itself.
_INTERRUPTED_FINISH_REASONS = frozenset({"insufficient_system_resource", "aborted"})


class DeepSeekCallError(RuntimeError):
    """A ``deepseek-flash`` call that returned HTTP 200 without a usable reply:
    no choices, a provider-interrupted finish, or ``length`` with nothing in it
    (thinking spent the whole budget; a retry does not help). Deliberately not
    an ``openai`` exception and without a ``status_code``: the runner classifies
    it by class NAME (``_INFRA_EXC_NAMES`` in scripts/webarena/run_graphpc.py),
    so the episode is EXCLUDED as infra rather than scored 0.0."""

    def __init__(self, message: str, *, finish_reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason


def _ds_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


def _ds_is_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, openai.RateLimitError)


# Reuse the shared retry policy (retry_policy.py) — same predicates the GPT
# backbone uses; DeepSeek surfaces the same openai SDK exception types.
_DS_RETRY_POLICY = RetryPolicy(is_retryable=_ds_is_retryable, is_rate_limit=_ds_is_rate_limit)


def _schema_has_pattern(schema: Any) -> bool:
    """True if the JSON schema carries a regex ``pattern`` anywhere — the marker
    of a GROUNDING (grammar-enforced) schema vs a reasoning schema."""
    if isinstance(schema, dict):
        if "pattern" in schema:
            return True
        return any(_schema_has_pattern(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_schema_has_pattern(v) for v in schema)
    return False


def _reply_is_empty(message: Any) -> bool:
    """True when a reply message carries neither text nor tool-call arguments."""
    if (getattr(message, "content", None) or "").strip():
        return False
    for tool_call in getattr(message, "tool_calls", None) or []:
        if (getattr(getattr(tool_call, "function", None), "arguments", None) or "").strip():
            return False
    return True


class DeepSeekAgent:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = os.environ.get("DEEPSEEK_API_KEY"),
        system_prompt_suffix: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        base_url: str = _BASE_URL,
        client_factory: Optional[Callable[[Optional[str]], Any]] = None,
    ) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.system_prompt_suffix = system_prompt_suffix
        self.temperature = temperature
        self.top_p = top_p
        # reasoning_effort accepted for build_default symmetry but IGNORED —
        # DeepSeek V4 controls reasoning via the thinking toggle, not an effort
        # level (deciding calls run thinking; grounding runs non-thinking). Log a
        # one-line notice when a value is actually supplied so the config surface
        # never silently no-ops.
        self.reasoning_effort = reasoning_effort
        if reasoning_effort is not None:
            logger.info(
                "deepseek backbone: reasoning_effort=%r is ignored; the thinking-mode "
                "toggle governs deliberation (deciding=thinking, grounding=non-thinking).",
                reasoning_effort,
            )
        self.base_url = base_url
        self.client_factory = client_factory

        # Telemetry counters (same names the runner reads off the backbone).
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    # ------------------------------------------------------------------
    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.api_key)
        # Read DEEPSEEK_API_KEY at CALL time (the __init__ default binds at import,
        # before the runner loads .env). Must be explicit: with a custom base_url
        # the openai SDK would otherwise fall back to OPENAI_API_KEY.
        api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
        return openai.OpenAI(api_key=api_key, base_url=self.base_url)

    def _create_with_backoff(self, request: dict[str, Any]):
        def _attempt():
            return self._build_client().chat.completions.create(**request)
        return _DS_RETRY_POLICY.run(_attempt)

    def reset_session(self) -> None:
        # DeepSeek is stateless per call (no server-side response chaining like
        # the GPT Responses API); nothing to reset. Present so the WebArena shim's
        # reset() finds it.
        return None

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        return extract_json_object(text, log_label="DeepSeek")

    def _record_usage(self, usage: Any, *, route: str = "none", request: Optional[dict[str, Any]] = None,
                      response: Any = None) -> None:
        # Telemetry only — NEVER raises: a malformed provider usage payload
        # must not fail a scored call (the baseline shim's _log_minimax_usage
        # wraps its extraction the same way). Extraction completes before any
        # counter mutation, so on failure the counters simply do not advance
        # for this call and response processing continues untouched.
        # Every per-call key sits before the "|" (one anchored parser reads this
        # line and the control arm's); ``route`` / ``thinking_sent`` /
        # ``effort_sent`` are read off the request sent, the rest off the reply.
        try:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            cache_hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
            cache_miss = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
            request = request or {}
            thinking = ((request.get("extra_body") or {}).get("thinking") or {}).get("type") or "absent"
            choices = getattr(response, "choices", None) or []
            choice = choices[0] if choices else None
            finish = getattr(choice, "finish_reason", None) or "none"
            empty = "true" if _reply_is_empty(getattr(choice, "message", None)) else "false"
            self.call_count += 1
            self.total_input_tokens += prompt
            self.total_output_tokens += completion
            logger.info(
                "deepseek usage: call=%d input=%d output=%d (reasoning=%d) route=%s thinking_sent=%s "
                "effort_sent=%s cache_hit=%d cache_miss=%d finish=%s empty=%s model=%s fingerprint=%s ts=%s "
                "| cumulative input=%d output=%d",
                self.call_count, prompt, completion, reasoning, route, thinking,
                request.get("reasoning_effort") or "absent", cache_hit, cache_miss, finish, empty,
                getattr(response, "model", None) or "none", getattr(response, "system_fingerprint", None) or "none",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                self.total_input_tokens, self.total_output_tokens,
            )
        except Exception:
            # WARNING, as the control shim's: a skipped line is a call whose model,
            # fingerprint and finish went unrecorded, and the run's tripwire reads INFO+.
            logger.warning("deepseek usage line skipped (malformed usage payload)", exc_info=True)

    def _validate_response(self, response: Any) -> None:
        """Raise ``DeepSeekCallError`` when a ``deepseek-flash`` reply is unusable
        (class docstring). Every other id returns untouched: there these shapes
        count as a degraded step downstream. A ``length``
        reply that carries text or tool-call arguments is a truncated reply, not
        a failed call: it flows on and is counted where it fails to parse."""
        if self.model_name != _FLASH_MODEL:
            return
        choices = getattr(response, "choices", None)
        if not choices:
            raise DeepSeekCallError("deepseek-flash returned no choices")
        finish = getattr(choices[0], "finish_reason", None)
        if finish in _INTERRUPTED_FINISH_REASONS:
            raise DeepSeekCallError(f"deepseek-flash generation interrupted by the provider: "
                                    f"finish_reason={finish!r}", finish_reason=finish)
        if finish == "length" and _reply_is_empty(getattr(choices[0], "message", None)):
            raise DeepSeekCallError("deepseek-flash finish_reason='length' with an empty reply — reasoning "
                                    "spent the whole token budget; retrying does not help",
                                    finish_reason=finish)

    def _set_thinking(self, req: dict[str, Any], enabled: bool) -> None:
        """The thinking toggle, and for ``deepseek-flash`` with thinking on the
        pinned effort (both in the documented chat form)."""
        req["extra_body"] = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        if enabled and self.model_name == _FLASH_MODEL:
            req["reasoning_effort"] = _FLASH_REASONING_EFFORT

    def _create_and_record(self, req: dict[str, Any], route: str):
        """The call, its metering line, then the validation — in that order, so a
        failed call that was billed still reaches the record."""
        resp = self._create_with_backoff(req)
        self._record_usage(resp.usage, route=route, request=req, response=resp)
        self._validate_response(resp)
        return resp

    # ------------------------------------------------------------------
    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        instructions = system_prompt
        if self.system_prompt_suffix:
            instructions = f"{instructions}\n\n{self.system_prompt_suffix}"
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_prompt},
        ]

    def _base_request(self, messages: list[dict[str, str]], max_tokens: Optional[int]) -> dict[str, Any]:
        req: dict[str, Any] = {"model": self.model_name, "messages": messages}
        cap = max_tokens if max_tokens is not None else self.max_tokens
        # max_tokens must stay GENEROUS-or-None on the reasoning path: thinking
        # mode folds reasoning tokens into completion_tokens, so a tight cap starves
        # the visible JSON and yields empty/truncated content that extract_json_object
        # can't parse. The GraphPC runner default is None (--max_tokens default None
        # -> self.max_tokens None -> no cap sent -> DeepSeek's provider default), so no
        # cap is applied unless a caller explicitly asks for one. Only omit when None.
        if cap is not None:
            req["max_tokens"] = cap
        # Sampling params are accepted in non-thinking mode only; set on the
        # grounding path (this backbone never sends them under thinking).
        return req

    def _invoke_grounding(
        self, *, system_prompt: str, user_prompt: str, max_tokens: Optional[int],
        json_schema: dict[str, Any], schema_name: Optional[str],
    ) -> dict[str, Any]:
        """NON-thinking + forced function-calling with the grammar schema."""
        req = self._base_request(self._messages(system_prompt, user_prompt), max_tokens)
        self._set_thinking(req, False)
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
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.top_p is not None:
            req["top_p"] = self.top_p

        resp = self._create_and_record(req, "grounding")
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        arguments = tool_calls[0].function.arguments if tool_calls else (msg.content or "{}")
        parsed = self._extract_json_object(arguments)
        return {
            "text": arguments,
            "raw_response": arguments,
            "response_output": [tc.model_dump() for tc in tool_calls] if tool_calls else [],
            "usage": resp.usage.model_dump() if resp.usage else {},
            "json": parsed,
        }

    def _invoke_reasoning(
        self, *, system_prompt: str, user_prompt: str, max_tokens: Optional[int],
        use_thinking: bool, json_schema: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Thinking (per use_thinking) + json_object + post-parse. The schema is
        injected into the prompt as an advisory hint (no server json_schema on
        DeepSeek); closed enums are normalised downstream."""
        user = user_prompt
        if json_schema is not None:
            user = (
                f"{user_prompt}\n\nReturn a single JSON object conforming to this schema "
                f"(JSON only, no prose):\n{json.dumps(json_schema)}"
            )
        elif "json" not in user.lower():
            user = f"{user}\n\nReturn JSON only."
        req = self._base_request(self._messages(system_prompt, user), max_tokens)
        self._set_thinking(req, use_thinking)
        req["response_format"] = {"type": "json_object"}

        resp = self._create_and_record(req, "deciding")
        content = resp.choices[0].message.content or ""
        return {
            "text": content,
            "raw_response": content,
            "response_output": [],
            "usage": resp.usage.model_dump() if resp.usage else {},
            "json": self._extract_json_object(content),
        }

    # ------------------------------------------------------------------
    # Consumer-facing contract (``Backbone``, backbone/base.py).
    # ------------------------------------------------------------------

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        observation: Any = None,  # noqa: ARG002 — WebArena a11y mode has no image; accepted for signature parity.
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
                    "deepseek grounding=True but the schema carries no grammar pattern; "
                    "forced function-calling will run without a regex constraint."
                )
            return self._invoke_grounding(
                system_prompt=system_prompt, user_prompt=user_prompt,
                max_tokens=max_tokens, json_schema=json_schema, schema_name=schema_name,
            )
        if has_pattern:
            raise ValueError(
                "deepseek complete_json: a pattern-bearing (grounding) schema was passed "
                "with grounding=False — grammar enforcement would silently not happen. "
                "Pass grounding=True at the action/grounding call site."
            )
        return self._invoke_reasoning(
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, use_thinking=use_thinking, json_schema=json_schema,
        )
