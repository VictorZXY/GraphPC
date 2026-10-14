# GraphPC vendored copy of OSWorld-V2 mm_agents/qwen/client.py
# at tag osworld-v2.1 (commit 3d778a3c9a34a079316f70df023b166700445792), blob 3a87f94cafe9a83f1e28c41e79eb2b3598a6a9e9.
# The OSWorld Qwen cell's CONTROL is upstream's own code: this file is byte-identical to
# that blob except this header and the deltas listed below. Each delta is marked in place
# (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`) and carries the upstream lines it
# replaces as `#|` lines, so the file reverses mechanically to upstream's; the Qwen mode
# of scripts/osworld/verify_runner_deltas.py holds it to that. The delta list with its
# reasons is in the header of scripts/osworld/run_multienv_qwen_baseline.py.
# Deltas carried: D1 (the OpenRouter route: base, key, provider pin, explicit reasoning form),
# D2 (OpenRouter failures carried inside an HTTP 200; a provider fault carried by an HTTP 400; an HTTP 200
# whose body the SDK cannot decode),
# D4 (one record per HTTP attempt).
# end of GraphPC header
"""OpenAI-compatible Qwen client with streaming and bounded retries."""

from __future__ import annotations

# >>> GraphPC D1/D4: imports for the pin copy and the per-call records
import copy
import gzip
import hashlib
import json
import sys
# <<< GraphPC D1/D4
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import openai
import requests
# >>> GraphPC D1: the cell's OpenRouter route, written out here (the control imports nothing of ours)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
# Sent on every chat request by `complete` below, whatever the caller built.
PROVIDER_PIN = {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}
# The only two reasoning forms a request may carry: enabled for the thinking calls,
# the documented disable form for the judge and the summarizer.
REASONING_ENABLED = {"enabled": True}
REASONING_DISABLED = {"effort": "none"}
# <<< GraphPC D1
# >>> GraphPC D2: OpenRouter failures carried inside an HTTP 200, provider faults carried by an HTTP 400, and an
# HTTP 200 whose body the SDK cannot decode
# The error.metadata.error_type values a retry can plausibly clear. The GraphPC arm's
# backbone retries the same set (src/agents/backbone/qwen/main.py:_RETRYABLE_ERROR_TYPES);
# keep the two copies in step when editing either.
TRANSIENT_ERROR_TYPES = frozenset(
    {"rate_limit_exceeded", "provider_overloaded", "provider_unavailable", "timeout", "server"}
)
# Three provider faults that arrive typed invalid_request: a provider failure whose error carries
# any fragment is transient, in either envelope — an in-band error (its `message`) or an HTTP 400
# (its `message` or `metadata.raw`). The GraphPC arm's backbone matches the same tuple
# (src/benchmarks/osworld/backbone_qwen.py:_TRANSIENT_ERROR_MESSAGES); keep the two copies in step
# when editing either.
TRANSIENT_ERROR_MESSAGES = ("JSON decode error: unexpected EOF", "Download multimodal file timed out",
                            "Failed to download multimodal content")


def _carries_transient_message(text: Any) -> bool:
    return isinstance(text, str) and any(fragment in text for fragment in TRANSIENT_ERROR_MESSAGES)


class InBandError(RuntimeError):
    """A provider failure inside an HTTP 200: a body or stream chunk carrying `error`, a
    choice with `finish_reason: "error"`, or a non-streamed body without choices."""

    def __init__(self, message: str, *, error_type: str | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.body = body


class TransientInBandError(InBandError):
    """An in-band failure with a transient error_type, or whose `message` carries a
    provider-fault fragment: retried like IncompleteStreamError."""


class TransientStatusError(RuntimeError):
    """An HTTP 400 whose error carries a provider-fault fragment: retried like a transient
    in-band failure. Keeps the status code and the error body for the per-call record."""

    def __init__(self, message: str, *, status_code: int, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class UndecodableResponseError(RuntimeError):
    """An HTTP 200 the SDK could not turn into a completion, at the provider call: a JSON body or
    stream chunk that does not decode (the SDK's json.JSONDecodeError), or a
    non-JSON content type, for which a non-streamed create() returns the body as a str. Retried like an
    incomplete stream; infra once exhausted (D7). The GraphPC arm's backbone defines the same class, and on
    both arms its text starts with "UndecodableResponseError: ", so both arms' records can be searched for it."""


def _typed_status(exc: openai.BadRequestError) -> Exception:
    """`exc` as the retry loop sees it: TransientStatusError when its error body's `message`
    or `metadata.raw` carries a provider-fault fragment, else `exc` itself."""
    body = exc.body if isinstance(exc.body, dict) else {}
    raw = _field(_field(body, "metadata"), "raw")
    if exc.status_code != 400 or not (
        _carries_transient_message(_field(body, "message"))
        or (raw is not None and _carries_transient_message(str(raw)))
    ):
        return exc
    return TransientStatusError(str(exc), status_code=exc.status_code, body=exc.body)


def _typed_in_band(where: str, error: Any, detail: str) -> InBandError:
    error_type = _field(_field(error, "metadata"), "error_type")
    transient = error_type in TRANSIENT_ERROR_TYPES or _carries_transient_message(_field(error, "message"))
    kind = TransientInBandError if transient else InBandError
    return kind(
        f"OpenRouter failure inside an HTTP 200 ({where}): code={_field(error, 'code')!r} "
        f"message={_field(error, 'message')!r} error_type={error_type!r} {detail}",
        error_type=error_type,
        body=error.model_dump() if hasattr(error, "model_dump") else error,
    )


def _in_band_error(where: str, body: Any, choice: Any, *, require_choice: bool) -> InBandError | None:
    error = _field(body, "error") or _field(choice, "error")
    finish_reason = _field(choice, "finish_reason")
    if not error and finish_reason != "error" and not (require_choice and choice is None):
        return None
    return _typed_in_band(
        where,
        error,
        f"finish_reason={finish_reason!r} choice={'present' if choice is not None else 'absent'}",
    )


class _InBandStream:
    """The stream as the SDK yields it, except that the SDK's own raise for a chunk whose
    data carries a top-level `error` — a bare openai.APIError holding that error as its
    body, raised before the chunk is yielded — becomes the typed in-band class above."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __iter__(self) -> Any:
        try:
            yield from self._stream
        except openai.APIError as exc:
            if type(exc) is not openai.APIError or not isinstance(exc.body, dict):
                raise
            raise _typed_in_band("a stream chunk", exc.body, "raised by the SDK") from exc
        except json.JSONDecodeError as exc:
            raise UndecodableResponseError(
                f"UndecodableResponseError: the provider's HTTP 200 stream carries a chunk that is not JSON: {exc}"
            ) from exc

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()
# <<< GraphPC D2


@dataclass(frozen=True)
class Completion:
    reasoning: str
    content: str
    generation_timed_out: bool = False
    estimated_tokens: int | None = None

    def rendered(self, include_thinking: bool) -> str:
        if include_thinking:
            return f"<think>\n{self.reasoning}\n</think>\n\n{self.content}".strip()
        return self.content


class IncompleteStreamError(RuntimeError):
    pass


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    if isinstance(value, dict):
        return value.get(name)
    if hasattr(value, "model_dump"):
        return value.model_dump().get(name)
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text(_field(part, "text")) for part in value)
    return str(value)


class QwenClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key: str | None,
        stream: bool = True,
        max_retries: int = 50,
        retry_sleep_base: float = 3.0,
        retry_sleep_cap: float = 15.0,
        request_timeout: float = 500.0,
        generation_timeout: float = 200.0,
        top_k: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        # >>> GraphPC D1: OpenRouter base and key, no OPENAI_* / localhost / "dummy" fallback
#|        self.base_url = base_url or os.environ.get(
#|            "OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"
#|        )
#|        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                f"the Qwen control reads its key from {OPENROUTER_API_KEY_ENV}, which is unset"
            )
        # <<< GraphPC D1
        self.stream = bool(stream)
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep_base = max(0.0, float(retry_sleep_base))
        self.retry_sleep_cap = max(0.0, float(retry_sleep_cap))
        self.request_timeout = max(1.0, float(request_timeout))
        self.generation_timeout = max(1.0, float(generation_timeout))
        self.top_k = top_k
        self.logger = logger or logging.getLogger("desktopenv.qwen")
        self._client: openai.OpenAI | None = None
        # >>> GraphPC D4: per-call records (set by the agent's set_api_log_dir; off when None)
        self.api_log_dir: str | None = None
        self.api_log_count = 0
        # The agent's step and context state, stamped on every record (QwenAgent sets it).
        self.record_context: Any = None
        self._record: dict[str, Any] | None = None
        self._last_record: tuple[str, dict[str, Any]] | None = None
        self._tokenize_record: dict[str, Any] = {}
        # <<< GraphPC D4

    def _openai_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.request_timeout,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _retryable_exceptions() -> tuple[type[Exception], ...]:
        candidates = [
            requests.exceptions.SSLError,
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "InternalServerError", None),
            IncompleteStreamError,
            # >>> GraphPC D2: a transient in-band failure, a 400 carrying a provider-fault
            # fragment, and an HTTP 200 the SDK cannot decode, retry like an incomplete stream
            TransientInBandError,
            TransientStatusError,
            UndecodableResponseError,
            # <<< GraphPC D2
        ]
        return tuple(item for item in candidates if isinstance(item, type))

    def _consume_stream(self, stream: Any, generation_timeout: float) -> Completion:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        started = time.monotonic()
        finished = False
        try:
            for chunk in stream:
                choice = (getattr(chunk, "choices", None) or [None])[0]
                # >>> GraphPC D4: every chunk's id / model / provider / usage / finish is recorded
                self._note_response(chunk, choice)
                # <<< GraphPC D4
                # >>> GraphPC D2: an error chunk ends the attempt (a final usage chunk has no choice)
                in_band = _in_band_error("a stream chunk", chunk, choice, require_choice=False)
                if in_band is not None:
                    raise in_band
                # <<< GraphPC D2
                delta = _field(choice, "delta")
                reasoning = _field(delta, "reasoning_content")
                if reasoning is None:
                    reasoning = _field(delta, "reasoning")
                reasoning_parts.append(_text(reasoning))
                content_parts.append(_text(_field(delta, "content")))
                if _field(choice, "finish_reason") is not None:
                    finished = True
                if time.monotonic() - started >= generation_timeout:
                    text = "".join(reasoning_parts) + "".join(content_parts)
                    return Completion(
                        "".join(reasoning_parts),
                        "".join(content_parts),
                        generation_timed_out=True,
                        estimated_tokens=self.count_tokens(text),
                    )
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        if not finished:
            raise IncompleteStreamError("stream ended without finish_reason")
        return Completion("".join(reasoning_parts), "".join(content_parts))

    # >>> GraphPC D4: the tokenize POST (the one non-chat route, unpinned) is recorded as its own kind
    def count_tokens(self, text: str) -> int:
        record = self._start_record("tokenize", 1, None, current=False)
        record["request"] = {
            "endpoint": self.base_url.rstrip("/") + "/tokenize",
            "model": self.model,
            "prompt_chars": len(text),
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        self._tokenize_record = record
        count = self._count_tokens(text)
        record["estimated_tokens"] = count
        self._finish_record(record)
        return count

#|    def count_tokens(self, text: str) -> int:
    def _count_tokens(self, text: str) -> int:
    # <<< GraphPC D4
        endpoint = self.base_url.rstrip("/") + "/tokenize"
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "prompt": text},
                timeout=30.0,
            )
            # >>> GraphPC D4
            self._tokenize_record["response"] = {"status_code": response.status_code}
            # <<< GraphPC D4
            response.raise_for_status()
            payload = response.json()
            count = payload.get("count")
            if count is None and isinstance(payload.get("tokens"), list):
                count = len(payload["tokens"])
            return max(0, int(count))
        except Exception:
            # >>> GraphPC D4: the fallback estimate is recorded as such, with what failed
            failure = sys.exc_info()[1]
            self._tokenize_record["fallback_estimate"] = True
            self._tokenize_record["exception"] = {
                "type": type(failure).__name__,
                "message": str(failure)[:300],
            }
            # <<< GraphPC D4
            return max(1, (len(text) + 3) // 4)

    # >>> GraphPC D4: the caller names the call kind; `log_extra` rides on its records
#|    def complete(self, payload: dict[str, Any]) -> Completion:
    def complete(
        self,
        payload: dict[str, Any],
        *,
        call_kind: str = "main",
        log_extra: dict[str, Any] | None = None,
    ) -> Completion:
    # <<< GraphPC D4
        retryable = self._retryable_exceptions()
        last_error: Exception | None = None
        # >>> GraphPC D1: every chat request states its reasoning form explicitly
        reasoning_form = (payload.get("extra_body") or {}).get("reasoning")
        if reasoning_form not in (REASONING_ENABLED, REASONING_DISABLED):
            raise ValueError(
                f"a {call_kind!r} request carries reasoning={reasoning_form!r}; the route "
                f"requires {REASONING_ENABLED!r} or {REASONING_DISABLED!r}"
            )
        # <<< GraphPC D1
        for attempt in range(1, self.max_retries + 1):
            # >>> GraphPC D4
            record = self._start_record(call_kind, attempt, log_extra)
            # <<< GraphPC D4
            try:
                extra_body = dict(payload.get("extra_body") or {})
                if self.top_k is not None:
                    extra_body.setdefault("top_k", self.top_k)
                # >>> GraphPC D1: the provider pin, on every chat request
                extra_body["provider"] = copy.deepcopy(PROVIDER_PIN)
                # <<< GraphPC D1
                kwargs: dict[str, Any] = {
                    "model": payload.get("model", self.model),
                    "messages": payload["messages"],
                    "max_tokens": payload.get("max_tokens", 32768),
                    "temperature": payload.get("temperature", 0.0),
                    "top_p": payload.get("top_p", 0.9),
                    "stream": payload.get("stream", self.stream),
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                # >>> GraphPC D4
                record["request"] = _loggable_request(kwargs)
                record["generation_timeout"] = payload.get(
                    "generation_timeout", self.generation_timeout
                )
                # <<< GraphPC D4
                # >>> GraphPC D2: an HTTP 400 carrying a provider-fault fragment is typed transient, and
                # so is an HTTP 200 the SDK cannot decode: a JSON body that does not parse, or a non-JSON
                # content type (create() then returns the body as a str)
#|                response = self._openai_client().chat.completions.create(**kwargs)
                try:
                    response = self._openai_client().chat.completions.create(**kwargs)
                except openai.BadRequestError as exc:
                    typed = _typed_status(exc)
                    if typed is exc:
                        raise
                    raise typed from exc
                except json.JSONDecodeError as exc:
                    raise UndecodableResponseError(
                        f"UndecodableResponseError: the provider's HTTP 200 body is not JSON: {exc}"
                    ) from exc
                if isinstance(response, str):
                    raise UndecodableResponseError(
                        f"UndecodableResponseError: the provider's HTTP 200 carries a non-JSON content type "
                        f"({len(response)} characters)"
                    )
                # <<< GraphPC D2
                if kwargs["stream"]:
                    # >>> GraphPC D2/D4: the SDK's in-band raise is typed (D2); the completion
                    # is recorded before it is returned (D4)
#|                    return self._consume_stream(
#|                        response,
#|                        float(
#|                            payload.get("generation_timeout", self.generation_timeout)
#|                        ),
#|                    )
                    completion = self._consume_stream(
                        _InBandStream(response),
                        float(
                            payload.get("generation_timeout", self.generation_timeout)
                        ),
                    )
                    self._finish_record(record, completion=completion)
                    return completion
                    # <<< GraphPC D2/D4
                # >>> GraphPC D4
                body_choices = _field(response, "choices") or []
                self._note_response(response, body_choices[0] if body_choices else None)
                # <<< GraphPC D4
                # >>> GraphPC D2: a non-streamed body is validated before choices[0] is read
                in_band = _in_band_error(
                    "a response body",
                    response,
                    body_choices[0] if body_choices else None,
                    require_choice=True,
                )
                if in_band is not None:
                    raise in_band
                # <<< GraphPC D2
                message = response.choices[0].message
                reasoning = _field(message, "reasoning_content")
                if reasoning is None:
                    reasoning = _field(message, "reasoning")
                # >>> GraphPC D4: the completion is recorded before it is returned
#|                return Completion(_text(reasoning), _text(_field(message, "content")))
                completion = Completion(_text(reasoning), _text(_field(message, "content")))
                self._finish_record(record, completion=completion)
                return completion
                # <<< GraphPC D4
            except retryable as exc:
                # >>> GraphPC D4
                self._finish_record(record, error=exc)
                # <<< GraphPC D4
                last_error = exc
                self.logger.warning(
                    "Qwen API error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    delay = min(
                        self.retry_sleep_cap,
                        self.retry_sleep_base * max(1, attempt),
                    )
                    if delay:
                        time.sleep(delay)
            # >>> GraphPC D4: a failure that is not retried is recorded, then raised as before
            except Exception as exc:
                self._finish_record(record, error=exc)
                raise
            # <<< GraphPC D4
        if last_error is not None:
            raise last_error
        raise RuntimeError("Qwen request failed without an exception")
    # >>> GraphPC D4: the per-call record helpers
    #
    # One JSON file per HTTP attempt, <api_log_dir>/NNN.json in attempt order, and beside it the
    # attempt's request, gzipped, as NNN.request.json.gz (the NNN.json names it in `request_file` and
    # keeps a `request_summary`: every scalar field and the message count). Observer-only: nothing
    # here changes a request byte. Fields: call_kind (main / judge / summarizer /
    # repair / endless / timeout-retry / tokenize / smoke), attempt and max_attempts, the
    # agent's `step` and `context_hard_truncation` (record_context), the caller's log_extra
    # (a judge call's `judge_candidate`, a context-error retry's `context_error_rescue`),
    # `request` (images replaced by the sha256 of their data URL), `response` (id,
    # model and provider echo, finish_reason, usage — a stream's final usage chunk
    # included — and any in-band error), `content`, `reasoning`, `generation_timed_out`,
    # `estimated_tokens`, `exception` (type, message cut at 300, status_code, error_type,
    # in_band_error, and status_body: an HTTP status failure's whole error body) and the
    # request/response times in ms. The agent may amend the last record of a given kind (a
    # judge verdict, a summarizer or judge exception) via amend_last_record.

    def _start_record(
        self,
        call_kind: str,
        attempt: int,
        log_extra: dict[str, Any] | None,
        *,
        current: bool = True,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "call_kind": call_kind,
            "attempt": attempt,
            "max_attempts": self.max_retries if call_kind != "tokenize" else 1,
            "request_time": int(time.time() * 1000),
            "response": {},
        }
        if callable(self.record_context):
            record.update(self.record_context())
        record.update(log_extra or {})
        if current:
            self._record = record
        return record

    def _note_response(self, body: Any, choice: Any) -> None:
        if self._record is None:
            return
        seen = self._record["response"]
        for name in ("id", "model", "provider"):
            value = _field(body, name)
            if value is not None and name not in seen:
                seen[name] = value
        usage = _field(body, "usage")
        if usage is not None:
            seen["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        finish_reason = _field(choice, "finish_reason")
        if finish_reason is not None:
            seen["finish_reason"] = finish_reason
        error = _field(body, "error") or _field(choice, "error")
        if error:
            seen["in_band_error"] = error.model_dump() if hasattr(error, "model_dump") else error
            seen["error_type"] = _field(_field(error, "metadata"), "error_type")

    def _finish_record(
        self,
        record: dict[str, Any],
        *,
        completion: Completion | None = None,
        error: BaseException | None = None,
    ) -> None:
        record["response_time"] = int(time.time() * 1000)
        if completion is not None:
            record["content"] = completion.content
            record["reasoning"] = completion.reasoning
            record["generation_timed_out"] = completion.generation_timed_out
            record["estimated_tokens"] = completion.estimated_tokens
        if error is not None:
            record["exception"] = {
                "type": type(error).__name__,
                "message": str(error)[:300],
                "status_code": getattr(error, "status_code", None),
                "error_type": getattr(error, "error_type", None),
                "in_band_error": getattr(error, "body", None) if isinstance(error, InBandError) else None,
                # An HTTP status failure's error body, whole (the message above is cut at 300).
                "status_body": getattr(error, "body", None)
                if isinstance(error, (openai.APIStatusError, TransientStatusError)) else None,
            }
        self._write_record(record)

    def _write_record(self, record: dict[str, Any]) -> None:
        if not self.api_log_dir:
            return
        self.api_log_count += 1
        stem = f"{self.api_log_count:03d}"
        path = os.path.join(self.api_log_dir, f"{stem}.json")
        summary = dict(record)
        request = summary.pop("request", None)
        try:
            if request is not None:
                target = os.path.join(self.api_log_dir, f"{stem}.request.json.gz")
                with gzip.open(target, "wt", encoding="utf-8") as handle:
                    json.dump(request, handle, ensure_ascii=False, default=str)
                summary["request_file"] = f"{stem}.request.json.gz"
                summary["request_summary"] = {
                    **{k: v for k, v in request.items() if k != "messages"},
                    "message_count": len(request.get("messages") or []),
                }
            with open(path, "w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, default=str)
            self._last_record = (path, summary)
        except Exception:
            self.logger.exception("Failed to write the per-call record %s", path)

    def amend_last_record(self, call_kind: str, **fields: Any) -> None:
        if self._last_record is None or self._last_record[1]["call_kind"] != call_kind:
            return
        path, record = self._last_record
        record.update(fields)
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(record, file, ensure_ascii=False, default=str)
        except Exception:
            self.logger.exception("Failed to amend the per-call record %s", path)


def _loggable_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request as sent, with each image data URL replaced by its sha256 (the slot hash)."""
    request = copy.deepcopy(kwargs)
    for message in request.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else []:
            image = part.get("image_url") if isinstance(part, dict) else None
            if isinstance(image, dict) and isinstance(image.get("url"), str):
                digest = hashlib.sha256(image["url"].encode("utf-8")).hexdigest()
                image["url"] = f"sha256:{digest}"
    return request
# <<< GraphPC D4
