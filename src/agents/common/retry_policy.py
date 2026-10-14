"""Backbone-portable retry / backoff loop with optional retry-after-hint parsing.

A reusable retry-loop class: each backbone supplies its predicates instead of
duplicating the retry-loop boilerplate.

Design: `RetryPolicy` is a pure-logic class: no provider imports. Callers supply
predicates (`is_retryable`, `is_rate_limit`) keyed against their own SDK's
exception classes and a `retry_after_regex` for message-embedded hints (OpenAI's
"Please try again in X.Xs" is the canonical case).

The hint branch logs `"retry-after hint: sleeping"` and the fallback branch
`"exp backoff fallback"`.
"""

from __future__ import annotations

import logging
import random
import re
import time

from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

logger = logging.getLogger("desktopenv.agent")

T = TypeVar("T")


def _default_predicate(_exc: BaseException) -> bool:
    return False


@dataclass
class RetryPolicy:
    """Provider-portable retry/backoff loop with optional retry-after-hint parsing.

    Wraps a `request_fn` closure so the caller owns client construction and
    request shape. The policy only owns: which exceptions retry, which of those
    are rate-limits (eligible for hint-parsing), how long to sleep, and how
    many attempts to make.

    Attributes
    ----------
    is_retryable : Callable[[BaseException], bool]
        Predicate identifying transient errors. Non-retryable exceptions are
        re-raised immediately on the first attempt.
    is_rate_limit : Callable[[BaseException], bool]
        Predicate identifying rate-limit errors among retryable ones. Only
        rate-limit errors are eligible for retry-after-hint parsing.
    max_attempts : int
        Total attempts (inclusive of the first). Default 5.
    exp_cap : float
        Upper bound on exponential-backoff sleep when no hint is available.
    hint_cap : float
        Upper bound on hint-driven sleep (safety against malformed/huge hints).
    retry_after_regex : Optional[re.Pattern[str]]
        Regex with one capture group for the suggested delay in seconds. If
        None, hint parsing is disabled and the policy always uses exponential
        backoff for rate-limit retries.
    label : str
        Short identifier interpolated into log messages (e.g. "OpenAI").
    floor_seconds : float
        Lower bound on the pre-jitter sleep, applied to BOTH the hint-driven
        and exp-backoff branches. Useful when the provider's hint sometimes
        under-estimates the bucket-refill time (callers see a 1s hint then
        immediately re-429); a 5s floor smooths small-hint thrash without
        capping the long ones. Default 0.0 (no floor) for callers that don't
        opt in.
    """

    is_retryable: Callable[[BaseException], bool] = _default_predicate
    is_rate_limit: Callable[[BaseException], bool] = _default_predicate
    max_attempts: int = 5
    exp_cap: float = 60.0
    hint_cap: float = 120.0
    retry_after_regex: Optional[re.Pattern[str]] = None
    label: str = "call"
    floor_seconds: float = 0.0

    def parse_retry_after(self, error: BaseException) -> Optional[float]:
        """Extract the optimal retry delay from a rate-limit error message.

        Returns None if no regex is configured, no match is found, or the
        captured group is not a valid float.
        """
        if self.retry_after_regex is None:
            return None
        match = self.retry_after_regex.search(str(error))
        if not match:
            return None
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None

    def run(self, request_fn: Callable[[], T]) -> T:
        """Execute `request_fn` with retry + backoff.

        Each attempt invokes `request_fn` fresh — callers can rebuild clients
        inside the closure if defensive per-attempt rebuilds are desired.
        Non-retryable exceptions propagate immediately.
        """
        for attempt in range(self.max_attempts):
            try:
                return request_fn()
            except BaseException as exc:
                if not self.is_retryable(exc):
                    raise
                error = exc
            if attempt + 1 >= self.max_attempts:
                raise error
            suggested = (
                self.parse_retry_after(error) if self.is_rate_limit(error) else None
            )
            if suggested is not None:
                sleep_s = min(
                    self.hint_cap,
                    max(self.floor_seconds, suggested) + random.uniform(0, 1),
                )
                logger.warning(
                    "%s retry-after hint: sleeping %.2fs (suggested %.2fs, attempt %d/%d)",
                    self.label,
                    sleep_s,
                    suggested,
                    attempt + 1,
                    self.max_attempts,
                )
            else:
                sleep_s = min(
                    self.exp_cap,
                    max(self.floor_seconds, 2 ** attempt) + random.uniform(0, 1),
                )
                logger.warning(
                    "Retrying %s after %s: attempt %d/%d, sleeping %.2fs (exp backoff fallback)",
                    self.label,
                    type(error).__name__,
                    attempt + 1,
                    self.max_attempts,
                    sleep_s,
                )
            time.sleep(sleep_s)
        raise RuntimeError(f"{self.label} retry loop exhausted unexpectedly")
