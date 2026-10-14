"""Re-sample a model call whose OUTPUT failed to parse, before the step degrades.

Parity with OSWorld-V2's M3 agent (mm_agents/m3/agent.py ``predict``): on a
reply that parses to nothing it re-calls the model, up to 3 attempts total, at
no step cost, and only then falls through to its no-op. Without this helper a
GraphPC call site re-parses the SAME reply with fallback parsers and then
degrades to a counted no-op step — zero re-samples. This helper gives a
deciding or grounding call site that OPTS IN the same retry budget. The
re-sampled CLASS is each site's own: the OSWorld grounding site re-samples
every reply that yields no executable action for our pipeline, including a
terminal-name reply that V2's M3 agent would execute as DONE/FAIL — a residual
difference recorded in design.md §1, not a claim of the identical rule.

**Benchmark scoping — why the default is ONE attempt (no re-sample).** The
budget is not uniform across benchmarks, because each benchmark's own control
arm sets it: OSWorld 2.0's M3 control re-samples up to 3 attempts (matching it
is parity), WebArena's ``run.py --agent_type prompt`` control does not re-sample
at all (matching it means 1). So the default here is the WebArena control's
budget — a call site that passes nothing re-samples nothing and makes one
request per call — and the OSWorld construction site passes ``attempts=3``
explicitly. Benchmark-scoped (design.md §10): the shared default is the
behaviour of the benchmark that must not change, and the benchmark that needs
more asks for it by name.

Scope is deliberately narrow:

  * it retries ONLY on the exception classes in ``retry_on`` — the markers of a
    MODEL-OUTPUT parse failure (``JsonParseExhausted`` by default; the OSWorld
    grounding route passes its own ``M3GroundingRejected``);
  * transport / API errors are NOT retried here — each backbone already wraps
    its request in ``RetryPolicy`` — and any other exception propagates on the
    first occurrence (our bugs stay loud);
  * no sleep; each attempt is a fresh request built by ``call`` from the same
    inputs (call sites compute their prompts ONCE and close over them, so every
    attempt sends identical bytes).

After the last attempt the last exception is re-raised unchanged in class, with
``resample_attempts`` (the number of re-samples made) set on it, so the
call site's EXISTING degrade path runs and can record the count. An exception
that is NOT retried but arrives after one or more re-samples (e.g. a terminal
rejection on the second sample) is propagated unchanged in class with the
re-samples spent so far stamped on it the same way — the count must never be
lost to telemetry; a first-occurrence pass-through is not touched at all.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Type

from .json_utils import JsonParseExhausted

# Total attempts, not re-samples: 1 means the first parse failure degrades,
# WebArena's default. OSWorld 2.0 passes 3 (its control's budget) from its own
# construction site.
DEFAULT_RESAMPLE_ATTEMPTS = 1


def call_with_resample(
    call: Callable[[], Any],
    *,
    attempts: int = DEFAULT_RESAMPLE_ATTEMPTS,
    retry_on: Tuple[Type[BaseException], ...] = (JsonParseExhausted,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> Tuple[Any, int]:
    """Invoke ``call`` up to ``attempts`` times; return ``(result, resamples)``
    where ``resamples`` is the number of re-samples that preceded the
    successful attempt (0 when the first attempt parsed).

    ``on_retry(attempt_number, exc)`` is invoked before each re-sample (attempt
    numbers 1..attempts-1 name the attempt that just failed). On exhaustion the
    last exception is re-raised with ``exc.resample_attempts = attempts - 1``;
    a non-retried exception raised after ``k >= 1`` re-samples propagates with
    ``exc.resample_attempts = k`` (untouched when it is the first attempt).
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    for attempt in range(attempts):
        try:
            return call(), attempt
        except retry_on as exc:
            if attempt + 1 >= attempts:
                exc.resample_attempts = attempt  # type: ignore[attr-defined]
                raise
            if on_retry is not None:
                on_retry(attempt + 1, exc)
        except BaseException as exc:
            # Not retried: propagate at once, but the re-samples already spent
            # are real calls the telemetry must still see.
            if attempt:
                exc.resample_attempts = attempt  # type: ignore[attr-defined]
            raise
    raise AssertionError("unreachable")  # pragma: no cover


def resample_attempts_of(message: Any) -> int:
    """The ``resample_attempts`` a reasoning message / action batch carries on
    its ``metadata`` (0 when absent or not an int) — the adapter-side reader."""
    metadata = getattr(message, "metadata", None)
    if not isinstance(metadata, dict):
        return 0
    value = metadata.get("resample_attempts")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0
