"""Claude Sonnet 5 backbone (anthropic SDK, first-party Messages API).

Satisfies ``Backbone`` (``src/agents/backbone/base.py``; the shared reasoning /
action agents call ``complete_json`` and read ``result["json"]`` +
``result["raw_response"]``), so those agents are REUSED unchanged with a
``ClaudeAgent`` instance: the DeepSeek / MiniMax / Qwen pattern.

  * Route: ``POST https://api.anthropic.com/v1/messages`` through
    ``client.messages.create``. No beta surface and no ``anthropic-beta``
    header. ``ANTHROPIC_API_KEY`` is read at CALL time, because the runner loads
    ``.env`` after import. ``timeout=600`` and ``max_retries`` are literals on
    the client. The SDK is the ONE retry layer (connection errors, 408, 409,
    429, 5xx), with no ``RetryPolicy`` on top. The explicit timeout is also what
    lets a non-streaming request carry a cap above 21,333: at the SDK's default
    timeout the SDK raises ``ValueError`` client-side before sending. PINNED:
    the control arm's native branch in ``src/benchmarks/webarena/_openai_compat.py``
    writes the same literals out for itself; the two arms' copies must stay
    equal, so change them together.
  * Thinking and effort are pinned EXPLICITLY on every request, through
    ``extra_body``: SDK 0.75's ``messages.create`` has no ``output_config``
    kwarg and its thinking type lacks ``adaptive``, and the SDK merges
    ``extra_body`` into the top level of the wire body. Deciding:
    ``thinking {type: adaptive}`` + ``output_config {effort}``. Grounding:
    ``thinking {type: disabled}`` + the same effort. On Claude, effort shapes all
    output and not only thinking, so the grounding call states it instead of
    inheriting the provider default. The response echoes neither, so the usage
    line records what was SENT, read off the request dict handed to the SDK.
  * Deciding route (``grounding=False``): the schema is appended to the user text
    as an advisory hint, with MiniMax's suffix byte for byte, and the reply's text
    blocks are post-parsed with ``extract_json_object``. There is no server-side
    ``output_config.format``: it adds input tokens to every call and needs a
    Claude-only rewrite of the shared schema.
  * Grounding route (``grounding=True``): a FORCED tool call
    (``tool_choice {type: tool}``) carrying the caller's schema UNCHANGED as the
    tool's ``input_schema``, with ``strict`` omitted, as on the MiniMax and Qwen
    backbones. A regex ``pattern`` is advisory there. A strict tool rejects
    the WebArena grammar ("Schema is too complex"), and the only pattern it does
    enforce, one top-level character class, would also rewrite non-ASCII stop
    answers. The tool input is the parsed JSON. A reply with no ``tool_use``
    block, or with a required array field that did not arrive as a list (e.g. a
    JSON-encoded string), raises ``JsonParseExhausted``:
    the step degrades as ONE counted no-op, never a silent one.
  * Caps: Anthropic requires ``max_tokens``, and thinking counts against it.
    Deciding 32000, grounding 4096, unless the caller or the constructor names
    one.
  * Sampling: Sonnet 5 accepts ``temperature`` only at 1.0 and rejects every
    ``top_p``. Nothing is sent, and a constructed value raises rather than
    silently doing nothing.
  * ``text`` is the reply's text blocks only. ``raw_response`` is the same
    joined text (deciding) or ``json.dumps`` of the tool input (grounding, the
    analogue of MiniMax's tool-call ``arguments``). Thinking blocks never reach
    either one: ``raw_response`` rides into the next deciding prompts as history.
  * A billing failure (402, the organisation spend-cap 429
    ``enforced_spend_limit_reached``, or the user-set usage-limit 400) raises
    ``ClaudeBillingError``. The runner classifies that class by NAME as infra,
    so an episode it ends is excluded, not a scored 0.0. Every other SDK error
    propagates unchanged (the runner classifies anthropic names and statuses).
  * One ``claude usage:`` line per call, every per-call key before the ``|``, so
    one anchored parser reads it and the control arm's line alike.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os

from typing import Any, Callable, Optional

from PIL import Image

from src.agents.common.json_utils import JsonParseExhausted, extract_json_object


logger = logging.getLogger("desktopenv.agent")

# The one model id this backbone runs, as the first-party API names it.
# PINNED: the control shim writes the same id out for itself (_CLAUDE_MODEL).
_DEFAULT_MODEL = "claude-sonnet-5"
# The only credential this backbone reads; resolved at CALL time.
_API_KEY_ENV = "ANTHROPIC_API_KEY"
# Client literals, equal on both arms (the control shim carries the same values).
_TIMEOUT_SECONDS = 600
_MAX_RETRIES = 4
# Per-route caps when neither the caller nor the constructor names one.
_DECIDING_MAX_TOKENS = 32000
_GROUNDING_MAX_TOKENS = 4096
# The effort values Sonnet 5 accepts; anything else (incl. "none") is a 400.
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# The two billing markers inside an error body (see ``_billing_failure``).
_SPEND_LIMIT_CODE = "enforced_spend_limit_reached"
_USAGE_LIMIT_TEXT = "specified api usage limits"


class ClaudeBillingError(RuntimeError):
    """A request the provider refused for billing: 402, the organisation spend-cap
    429 (``enforced_spend_limit_reached``), or the user-set usage-limit 400.

    Not an anthropic exception class on purpose. The runner classifies it by NAME
    (``_INFRA_EXC_NAMES`` in scripts/webarena/run_graphpc.py), so the episode is
    excluded as infra and can be re-run, instead of the 400 case scoring a 0.0.
    The message carries the status, the error type and the error code verbatim,
    which a lane tripwire (run_lanes.py) can key on."""

    def __init__(self, message: str, *, status_code: Optional[int], error_type: Optional[str],
                 error_code: Optional[str]) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code


def _billing_failure(status: Optional[int], body: Any) -> Optional[tuple[str, Optional[str], Optional[str]]]:
    """``(message, error_type, error_code)`` when an error response is a billing
    refusal, else None. The control shim applies the same rule to the same
    fields; change the two together."""
    error = body.get("error") if isinstance(body, dict) else None
    error = error if isinstance(error, dict) else {}
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    error_type, error_code = error.get("type"), details.get("error_code")
    message = str(error.get("message") or "")
    if not (status == 402
            or (status == 429 and error_code == _SPEND_LIMIT_CODE)
            or (status == 400 and _USAGE_LIMIT_TEXT in message.lower())):
        return None
    return (f"claude billing failure: status={status} type={error_type} error_code={error_code or 'none'} "
            f"message={message!r}", error_type, error_code)


def _schema_has_pattern(schema: Any) -> bool:
    """True if the JSON schema carries a regex ``pattern`` anywhere: the marker
    of a GROUNDING (grammar-bearing) schema vs a reasoning schema."""
    if isinstance(schema, dict):
        if "pattern" in schema:
            return True
        return any(_schema_has_pattern(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_schema_has_pattern(v) for v in schema)
    return False


def _unlisted_arrays(schema: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """The REQUIRED top-level properties ``schema`` declares as arrays that
    ``payload`` does not carry as a list. Read off the schema the caller passed,
    so no caller's field names live here."""
    properties = schema.get("properties") or {}
    return [
        f"{name}={payload.get(name)!r:.120} ({type(payload.get(name)).__name__}, not a list)"
        for name in schema.get("required") or []
        if (properties.get(name) or {}).get("type") == "array" and not isinstance(payload.get(name), list)
    ]


class ClaudeAgent:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        system_prompt_suffix: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: str = "high",
        client_factory: Optional[Callable[[Optional[str]], Any]] = None,
    ) -> None:
        # temperature / top_p are accepted for build_default symmetry across
        # backbones (the runner passes None). Sonnet 5 rejects every top_p and any
        # temperature but 1.0, so a value is refused here rather than ignored.
        if temperature is not None or top_p is not None:
            raise ValueError(
                f"claude backbone: temperature={temperature!r} / top_p={top_p!r} are refused; Sonnet 5 "
                "rejects top_p and any temperature but its default, and this backbone sends neither"
            )
        if reasoning_effort not in _EFFORTS:
            raise ValueError(
                f"claude backbone: reasoning_effort={reasoning_effort!r} is not one of {_EFFORTS} "
                "(the API rejects anything else, 'none' included, with a 400)"
            )
        self.model_name = model
        # None = the per-route caps; a value caps both routes (a per-call value wins).
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.system_prompt_suffix = system_prompt_suffix
        self.reasoning_effort = reasoning_effort
        self.client_factory = client_factory

        # Telemetry counters (same names the runner reads off the backbone).
        # Anthropic's input_tokens EXCLUDES cache reads and writes; this backbone
        # sends no cache_control, so both are 0 and every line records them.
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    # ------------------------------------------------------------------
    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.api_key)
        # Imported here, not at module top.
        import anthropic

        return anthropic.Anthropic(
            api_key=self.api_key or os.environ.get(_API_KEY_ENV),
            timeout=_TIMEOUT_SECONDS,
            max_retries=_MAX_RETRIES,
        )

    def reset_session(self) -> None:
        # Stateless per call; present so the WebArena shim's reset() finds it.
        return None

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        return extract_json_object(text, log_label="Claude")

    # ------------------------------------------------------------------
    # Request building.
    # ------------------------------------------------------------------

    def _prepare_screenshot(self, screenshot_bytes: bytes) -> bytes:
        screenshot_image = Image.open(io.BytesIO(screenshot_bytes))
        resampling = getattr(Image, "Resampling", Image)
        resized_image = screenshot_image.resize((1280, 720), resampling.LANCZOS)

        output_buffer = io.BytesIO()
        resized_image.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    def _messages(self, user_prompt: str, observation: Any) -> list[dict[str, Any]]:
        screenshot_bytes = None
        if observation is not None:
            if hasattr(observation, "content") and isinstance(observation.content, dict):
                screenshot_bytes = observation.content.get("screenshot")
            elif isinstance(observation, dict):
                screenshot_bytes = observation.get("screenshot")
        if screenshot_bytes is None:
            # No image (WebArena's accessibility-tree mode): the plain-string user message.
            content: Any = user_prompt
        else:
            # Image mode: the 1280x720 PNG first, then the text.
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(self._prepare_screenshot(screenshot_bytes)).decode("utf-8"),
                    },
                },
                {"type": "text", "text": user_prompt},
            ]
        return [{"role": "user", "content": content}]

    def _base_request(self, system_prompt: str, user_prompt: str, observation: Any, cap: int) -> dict[str, Any]:
        system = system_prompt
        if self.system_prompt_suffix:
            system = f"{system}\n\n{self.system_prompt_suffix}"
        return {
            "model": self.model_name,
            "max_tokens": cap,
            "system": system,
            "messages": self._messages(user_prompt, observation),
        }

    def _cap(self, max_tokens: Optional[int], route_default: int) -> int:
        if max_tokens is not None:
            return max_tokens
        return self.max_tokens if self.max_tokens is not None else route_default

    def _extra_body(self, *, thinking: bool) -> dict[str, Any]:
        """Thinking and effort, EXPLICIT on every request (never the provider default)."""
        return {
            "thinking": {"type": "adaptive" if thinking else "disabled"},
            "output_config": {"effort": self.reasoning_effort},
        }

    def _deciding_request(
        self, *, system_prompt: str, user_prompt: str, observation: Any, max_tokens: Optional[int],
        use_thinking: bool, json_schema: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        # MiniMax's advisory suffix (backbone/minimax/main.py _invoke_reasoning), byte for byte.
        user = user_prompt
        if json_schema is not None:
            user = (
                f"{user_prompt}\n\nReturn a single JSON object conforming to this schema "
                f"(JSON only, no prose):\n{json.dumps(json_schema)}"
            )
        elif "json" not in user.lower():
            user = f"{user}\n\nReturn JSON only."
        request = self._base_request(system_prompt, user, observation, self._cap(max_tokens, _DECIDING_MAX_TOKENS))
        request["extra_body"] = self._extra_body(thinking=use_thinking)
        return request

    def _grounding_request(
        self, *, system_prompt: str, user_prompt: str, observation: Any, max_tokens: Optional[int],
        json_schema: dict[str, Any], schema_name: Optional[str],
    ) -> dict[str, Any]:
        """Thinking disabled + a FORCED tool carrying the caller's schema unchanged, ``strict`` omitted."""
        name = schema_name or "emit"
        request = self._base_request(system_prompt, user_prompt, observation,
                                     self._cap(max_tokens, _GROUNDING_MAX_TOKENS))
        request["tools"] = [{
            "name": name,
            "description": "Emit the structured action payload.",
            "input_schema": json_schema,
        }]
        request["tool_choice"] = {"type": "tool", "name": name}
        request["extra_body"] = self._extra_body(thinking=False)
        return request

    # ------------------------------------------------------------------
    # The call and its metering.
    # ------------------------------------------------------------------

    def _create(self, request: dict[str, Any], *, route: str):
        import anthropic

        try:
            response = self._build_client().messages.create(**request)
        except anthropic.APIStatusError as exc:
            billing = _billing_failure(exc.status_code, exc.body)
            if billing is not None:
                message, error_type, error_code = billing
                raise ClaudeBillingError(message, status_code=exc.status_code, error_type=error_type,
                                         error_code=error_code) from exc
            raise
        self._record_usage(route, request, response)
        return response

    def _record_usage(self, route: str, request: dict[str, Any], response: Any) -> None:
        # One line per call, every per-call key before the "|". effort_sent /
        # thinking_sent are read off the request dict handed to the SDK ("absent"
        # when the key is missing): Claude echoes neither, so what was sent is the
        # only per-call record. stop / refusal / empty make the outcomes that
        # otherwise surface only as a parse failure visible. Logging only: a
        # malformed response must not fail the call, and a skipped line is a
        # WARNING because that call's regime goes unrecorded.
        try:
            usage = response.usage
            details = getattr(usage, "output_tokens_details", None) or {}
            thinking = details.get("thinking_tokens") if isinstance(details, dict) else getattr(
                details, "thinking_tokens", 0)
            call_input = int(usage.input_tokens or 0)
            call_output = int(usage.output_tokens or 0)
            extra_body = request.get("extra_body") or {}
            blocks = list(response.content or [])
            if route == "grounding":
                empty = not any(getattr(block, "type", None) == "tool_use" for block in blocks)
            else:
                empty = not any(getattr(block, "type", None) == "text" and getattr(block, "text", "")
                                for block in blocks)
            fields = (
                route,
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
            logger.warning("claude usage line skipped (call=%d): malformed response", self.call_count, exc_info=True)
            return
        self.call_count += 1
        self.total_input_tokens += call_input
        self.total_output_tokens += call_output
        logger.info(
            "claude usage: call=%d route=%s effort_sent=%s thinking_sent=%s input=%d output=%d thinking=%d "
            "cache_read=%d cache_write=%d model=%s stop=%s refusal=%s empty=%s | cumulative input=%d output=%d",
            self.call_count, *fields, self.total_input_tokens, self.total_output_tokens,
        )
        if response.stop_reason == "refusal":
            stop_details = getattr(response, "stop_details", None) or {}
            logger.info(
                "claude refusal: route=%s category=%s explanation=%r (the step degrades as a counted no-op)",
                route, stop_details.get("category") if isinstance(stop_details, dict) else None,
                stop_details.get("explanation") if isinstance(stop_details, dict) else None,
            )

    @staticmethod
    def _text(response: Any) -> str:
        """The reply's text blocks only; thinking blocks never reach the text."""
        return "\n".join(
            block.text for block in response.content or []
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ).strip()

    def _invoke_deciding(self, **request_inputs: Any) -> dict[str, Any]:
        response = self._create(self._deciding_request(**request_inputs), route="deciding")
        text = self._text(response)
        return {
            "text": text,
            "raw_response": text,
            "usage": response.usage.model_dump(),
            "stop_reason": response.stop_reason,
            "json": self._extract_json_object(text),
        }

    def _invoke_grounding(self, *, json_schema: dict[str, Any], **request_inputs: Any) -> dict[str, Any]:
        response = self._create(self._grounding_request(json_schema=json_schema, **request_inputs),
                                route="grounding")
        tool_use = next((block for block in response.content or [] if getattr(block, "type", None) == "tool_use"),
                        None)
        if tool_use is None:
            raise JsonParseExhausted(
                f"Claude grounding reply carried no tool_use block (stop_reason={response.stop_reason!r}, "
                f"text={self._text(response)[:200]!r})"
            )
        payload = tool_use.input
        if not isinstance(payload, dict):
            raise JsonParseExhausted(f"Claude grounding tool input is {type(payload).__name__}, not an object")
        unlisted = _unlisted_arrays(json_schema or {}, payload)
        if unlisted:
            raise JsonParseExhausted(f"Claude grounding tool input breaks its schema's container types: {unlisted}")
        return {
            "text": self._text(response),
            "raw_response": json.dumps(payload),
            "usage": response.usage.model_dump(),
            "stop_reason": response.stop_reason,
            "json": payload,
        }

    # ------------------------------------------------------------------
    # Consumer-facing contract (``Backbone``, backbone/base.py).
    # ------------------------------------------------------------------

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        observation: Any = None,
        max_tokens: Optional[int] = None,
        use_thinking: bool = True,
        json_schema: Optional[dict[str, Any]] = None,
        schema_name: Optional[str] = None,
        grounding: bool = False,
    ) -> dict[str, Any]:
        # Route on the EXPLICIT grounding flag (the action call sites pass
        # grounding=True); _schema_has_pattern is only a defensive assertion, as
        # on the MiniMax / Qwen backbones.
        has_pattern = json_schema is not None and _schema_has_pattern(json_schema)
        if grounding:
            if not has_pattern:
                logger.warning(
                    "claude grounding=True but the schema carries no grammar pattern; the forced tool "
                    "call will run without one."
                )
            return self._invoke_grounding(
                system_prompt=system_prompt, user_prompt=user_prompt, observation=observation,
                max_tokens=max_tokens, json_schema=json_schema, schema_name=schema_name,
            )
        if has_pattern:
            raise ValueError(
                "claude complete_json: a pattern-bearing (grounding) schema was passed with grounding=False; "
                "pass grounding=True at the action/grounding call site."
            )
        return self._invoke_deciding(
            system_prompt=system_prompt, user_prompt=user_prompt, observation=observation,
            max_tokens=max_tokens, use_thinking=use_thinking, json_schema=json_schema,
        )
