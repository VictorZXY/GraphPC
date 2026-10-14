from __future__ import annotations

import json
import logging
import os
import re

from typing import Any, Callable, Optional

import openai

from src.agents.common.json_utils import JsonParseExhausted, extract_json_object
from src.agents.common.retry_policy import RetryPolicy

from .utils import build_client, encode_image, get_field, model_dump

logger = logging.getLogger("desktopenv.agent")


# OpenAI's RateLimitError message embeds the optimal retry delay:
# "...Please try again in 8.808s. Visit https://...". The RetryPolicy uses this
# to sleep exactly that long instead of guessing via exponential backoff (which
# often over- or under-shoots the actual rolling-TPM-window recovery time).
_GPT_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


def _gpt_is_retryable(exc: BaseException) -> bool:
    # Order matters: RateLimitError IS-A APIStatusError in the openai SDK, so it
    # must be checked first to avoid being filtered by the status<500 branch.
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


def _gpt_is_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, openai.RateLimitError)


# Module-level singleton: configured once with GPT-specific predicates and the
# OpenAI retry-after regex. GPTAgent methods are thin delegators so any future
# OpenAI-wrapped backbone can reuse the same policy without re-deriving it.
_GPT_RETRY_POLICY = RetryPolicy(
    is_retryable=_gpt_is_retryable,
    is_rate_limit=_gpt_is_rate_limit,
    # A long retry window, and one API key: keys of one organisation share the
    # same per-org-per-model TPM bucket, so rotating to a backup key buys nothing
    # a larger retry budget does not. The long window also absorbs the
    # "rapid-refill miss" (a re-429 right after a hinted wait).
    max_attempts=20,
    exp_cap=60.0,
    hint_cap=120.0,
    retry_after_regex=_GPT_RETRY_AFTER_RE,
    label="OpenAI",
    # A 5s floor on retry sleeps. OpenAI hints sometimes under-estimate
    # the actual bucket-refill window (caller observes a 1s hint then immediately
    # re-429s); floor smooths the small-hint thrash without capping the long ones.
    floor_seconds=5.0,
)


class GPTAgent:
    """
    Shared GPT backbone wrapper used by the internal GraphPC agents.

    The wrapper keeps the GPT-specific handling local: Responses API requests,
    screenshot formatting and JSON extraction. Satisfies ``Backbone``
    (``src/agents/backbone/base.py``), the contract the shared agents call.
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = os.environ.get("OPENAI_API_KEY"),
        system_prompt_suffix: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: str = "xhigh",
        client_factory: Optional[Callable[[Optional[str]], Any]] = None,
        grounding_reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.system_prompt_suffix = system_prompt_suffix
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.client_factory = client_factory
        # Effort for grounding calls (complete_json(grounding=True)). None (the
        # default): grounding sends the deciding effort. The other backbones
        # ground with thinking off; "none" is GPT's equivalent.
        self.grounding_reasoning_effort = grounding_reasoning_effort

        # Cumulative token-usage telemetry. Field names match
        # the OpenAI Responses-API ``usage`` payload (input_tokens/output_tokens).
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0

    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.api_key)
        return build_client(api_key=self.api_key)

    def _create_with_backoff(self, request):
        # Single-tier retry via RetryPolicy (retry-after hint, 5s floor,
        # exp-backoff fallback, max 20 attempts). One API key: keys of one
        # organisation share the same TPM bucket, so a backup key would only be
        # equivalent to a larger max_attempts. The client is rebuilt per attempt
        # inside the closure.

        def _primary_attempt():
            return self._build_client().responses.create(**request)

        return _GPT_RETRY_POLICY.run(_primary_attempt)

    def _build_reasoning(self, *, use_thinking: bool, grounding: bool = False) -> dict[str, Any]:
        # Precedence: an explicit grounding effort on a grounding call, then
        # use_thinking=False -> "low", then the deciding effort.
        if grounding and self.grounding_reasoning_effort is not None:
            effort = self.grounding_reasoning_effort
        else:
            effort = self.reasoning_effort if use_thinking else "low"
        return {
            "effort": effort,
            "summary": "concise",
        }

    def _build_input_content(
        self,
        *,
        user_prompt: str,
        observation: Any = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": user_prompt,
            }
        ]

        screenshot_bytes = None
        if observation is not None:
            if hasattr(observation, "content") and isinstance(observation.content, dict):
                screenshot_bytes = observation.content.get("screenshot")
            elif isinstance(observation, dict):
                screenshot_bytes = observation.get("screenshot")

        if screenshot_bytes is not None:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encode_image(screenshot_bytes)}",
                    "detail": "original",
                }
            )

        return content

    def _build_request_input(
        self,
        *,
        user_prompt: str,
        observation: Any = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": self._build_input_content(
                    user_prompt=user_prompt,
                    observation=observation,
                ),
            }
        ]

    def _extract_output_text_blocks(self, output_items: list[dict[str, Any]]) -> list[str]:
        text_blocks: list[str] = []

        for item in output_items:
            item_type = item.get("type")
            if item_type == "message":
                for content in item.get("content", []) or []:
                    if content.get("type") == "output_text" and content.get("text"):
                        text_blocks.append(str(content["text"]))
            elif item_type == "reasoning":
                for summary in item.get("summary", []) or []:
                    if summary.get("text"):
                        text_blocks.append(str(summary["text"]))

        return text_blocks

    def _extract_answer_text_blocks(self, output_items: list[dict[str, Any]]) -> list[str]:
        # The answer only: `message` output_text parts, without the reasoning
        # summaries `_extract_output_text_blocks` also collects.
        return [
            str(content["text"])
            for item in output_items
            if item.get("type") == "message"
            for content in item.get("content", []) or []
            if content.get("type") == "output_text" and content.get("text")
        ]

    def _log_usage(
        self,
        *,
        grounding: bool,
        request: dict[str, Any],
        response: Any,
        usage: dict[str, Any],
        output_items: list[dict[str, Any]],
        call_input: int,
        call_output: int,
    ) -> None:
        # One line per call, every per-call key before the "|" so one anchored
        # parser reads this line and the control arm's `gpt usage:` line alike.
        # effort_sent is what the request carried, effort_echo what the server
        # says it ran (response.reasoning.effort). status / incomplete / refusal
        # make the outcomes that otherwise surface only as a parse failure
        # visible. Logging only — a malformed response must not kill the call;
        # a skipped line is a WARNING because that call's effort goes unrecorded.
        try:
            input_details = usage.get("input_tokens_details") or {}
            output_details = usage.get("output_tokens_details") or {}
            refusal = any(
                content.get("type") == "refusal"
                for item in output_items
                if item.get("type") == "message"
                for content in item.get("content", []) or []
            )
            logger.info(
                "gpt usage: call=%d route=%s input=%d output=%d effort_sent=%s effort_echo=%s "
                "reasoning=%s cached=%s cache_write=%s model=%s status=%s incomplete=%s refusal=%s "
                "| cumulative input=%d output=%d",
                self.call_count,
                "grounding" if grounding else "deciding",
                call_input,
                call_output,
                request["reasoning"]["effort"],
                get_field(get_field(response, "reasoning"), "effort") or "absent",
                output_details.get("reasoning_tokens") or 0,
                input_details.get("cached_tokens") or 0,
                input_details.get("cache_write_tokens") or 0,
                get_field(response, "model") or "none",
                get_field(response, "status") or "none",
                get_field(get_field(response, "incomplete_details"), "reason") or "none",
                "present" if refusal else "absent",
                self.total_input_tokens,
                self.total_output_tokens,
            )
        except Exception:
            logger.warning(
                "gpt usage line skipped (call=%d): malformed response", self.call_count, exc_info=True,
            )

    def _invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        observation: Any = None,
        max_tokens: Optional[int] = None,
        use_thinking: bool = True,
        json_mode: bool = False,
        json_schema: Optional[dict[str, Any]] = None,
        schema_name: Optional[str] = None,
        grounding: bool = False,
    ) -> dict[str, Any]:
        instructions = system_prompt
        if self.system_prompt_suffix:
            instructions = f"{instructions}\n\n{self.system_prompt_suffix}"

        # Strict schema mode takes precedence over the json_object fallback
        # mode. When a schema is supplied, the OpenAI
        # Responses API server-side enforces additionalProperties=False +
        # required-keys + enum constraints — eliminating extra-prose /
        # missing-key / wrong-type failures at the source rather than
        # patching them in the parser.
        use_schema = json_schema is not None
        # json_object mode requires the literal word "json" in user content
        # (not instructions); strict-schema mode does not. The prefix is added
        # only when falling back to json_object.
        if json_mode and not use_schema:
            user_prompt = "Reply in JSON format only.\n\n" + user_prompt

        request = {
            "model": self.model_name,
            "instructions": instructions,
            "input": self._build_request_input(user_prompt=user_prompt, observation=observation),
            "tools": [],
            "truncation": "auto",
            "reasoning": self._build_reasoning(use_thinking=use_thinking, grounding=grounding),
        }
        if use_schema:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name or "response_schema",
                    "schema": json_schema,
                    "strict": True,
                }
            }
        elif json_mode:
            request["text"] = {"format": {"type": "json_object"}}
        output_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if output_tokens is not None:
            request["max_output_tokens"] = output_tokens

        response = self._create_with_backoff(request)
        error = get_field(get_field(response, "error", {}), "message")
        if error:
            raise RuntimeError(error)
        if get_field(response, "status") == "failed":
            raise RuntimeError("OpenAI Responses API request failed.")

        output_items = model_dump(get_field(response, "output", []) or [])
        text_blocks = self._extract_output_text_blocks(output_items)

        usage = model_dump(get_field(response, "usage", {}) or {})
        call_input = usage.get("input_tokens", 0) or 0
        call_output = usage.get("output_tokens", 0) or 0
        self.call_count += 1
        self.total_input_tokens += call_input
        self.total_output_tokens += call_output
        self._log_usage(
            grounding=grounding,
            request=request,
            response=response,
            usage=usage,
            output_items=output_items,
            call_input=call_input,
            call_output=call_output,
        )

        return {
            "text": "\n".join(text_blocks).strip(),
            "text_blocks": text_blocks,
            "raw_response": json.dumps(output_items),
            "response_output": output_items,
            "usage": usage,
        }

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
        # `grounding` routes the effort (see _build_reasoning) and labels the
        # usage line; GPT's strict json_schema path enforces the grammar either way.
        # Strict-schema mode is the primary path: callers supply a JSON
        # Schema describing the expected response, and the Responses API
        # enforces it server-side (no markdown / no extra keys / typed
        # values). When ``json_schema`` is None, fall back to the
        # text.format=json_object mode so callers without a schema still
        # work. The multi-parser JSON fallback in
        # ``json_utils.extract_json_object`` stays in place as
        # defense-in-depth.
        result = self._invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            observation=observation,
            max_tokens=max_tokens,
            use_thinking=use_thinking,
            json_mode=json_schema is None,
            json_schema=json_schema,
            schema_name=schema_name,
            grounding=grounding,
        )
        # Parse the answer, not the reasoning: the candidate is the `message`
        # output_text only, so a brace in a reasoning summary cannot break the
        # first-`{`-to-last-`}` slice. `text` (summaries included) stays the
        # record and is what the failure line shows. (The line's wording avoids
        # "GPT JSON parse failed", which json_utils already logs once per failure.)
        answer_text = "\n".join(self._extract_answer_text_blocks(result["response_output"])).strip()
        try:
            result["json"] = self._extract_json_object(answer_text)
        except JsonParseExhausted:
            logger.warning(
                "GPT parse candidate was the answer text only; full output text "
                "(reasoning summaries included):\n%s",
                result["text"],
            )
            raise
        return result

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        # Thin delegator to the backbone-portable extractor in src/agents/common/json_utils.py.
        # Kept as a method (not a static delegate): every backbone exposes
        # `_extract_json_object`, and call sites bind to `self._extract_json_object`.
        return extract_json_object(text, log_label="GPT")
