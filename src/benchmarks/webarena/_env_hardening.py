"""Runtime env-hardening patch for WebArena's flaky observation processing.

WebArena's ``TextObervationProcessor.process`` fetches browser layout info via
``fetch_browser_info``; on transient zero-width root layouts (a snapshot taken
while the page is still settling) it hits ``bounds[0][2] == 0`` and raises
``ZeroDivisionError``. Its own internal recovery
(``page.wait_for_load_state("load", timeout=500)`` then a single re-fetch) is
broken for this case: the page already fired ``load``, so the wait necessarily
times out (Playwright ``TimeoutError``), and the episode dies. It happens even
with ``sleep_after_execution=2.0``, often enough to lose a measurable share of
a run's episodes.

This module installs a bounded re-fetch: call the original ``process``; on a
``ZeroDivisionError`` or Playwright ``TimeoutError``, sleep and retry the
ORIGINAL up to two more times (3 attempts total). All three failing re-raises
the LAST exception unchanged, so ``episode_error`` accounting stays honest.

Design + comparability notes:
  (a) This is a DOCUMENTED harness-robustness deviation from stock WebArena,
      which swallows these as task-level ``[Unhandled Error]`` (lost episodes).
  (b) It is installed IDENTICALLY in both the GraphPC and bare-control runners,
      so the environment behaves the same for both arms — the Δ between them is
      unaffected. Symmetry is the requirement.
  (c) Because flaky-page episodes that upstream would drop are instead rescued,
      absolute numbers on flaky pages may be slightly more favorable than
      published upstream runs — a comparability caveat for absolute (not Δ)
      claims.

Runtime patch only — mirrors ``_openai_compat.py``; zero WebArena file edits.
It does NOT change method/agent behavior; its only effect is a bounded re-fetch
of an observation for an action that has already executed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("desktopenv.experiment")

# Idempotency guard so a double-install in one process is a no-op.
_INSTALLED = False

_MAX_ATTEMPTS = 3
_RETRY_SLEEP_SECONDS = 1.0

# Verdict-level telemetry: how often the patch fired. ``retries`` counts retry
# attempts scheduled; ``rescues`` counts calls that succeeded after >=1 retry.
OBS_RETRY_STATS: dict[str, int] = {"retries": 0, "rescues": 0}

# Separate idempotency guard + telemetry for the page.client recreate patch.
_PAGE_CLIENT_INSTALLED = False
PAGE_CLIENT_STATS: dict[str, int] = {"client_recreated": 0}


def _playwright_timeout_cls() -> Any:
    """Resolve Playwright's ``TimeoutError`` class defensively — the import path
    has moved across versions. Returns ``None`` if Playwright isn't importable
    (dev/CI), in which case the name/module match in ``_is_retryable`` covers
    it. Both known paths resolve to the same class."""
    try:
        from playwright.sync_api import TimeoutError as PWTimeout

        return PWTimeout
    except Exception:  # noqa: BLE001
        try:
            from playwright._impl._api_types import TimeoutError as PWTimeout

            return PWTimeout
        except Exception:  # noqa: BLE001
            return None


_PW_TIMEOUT = _playwright_timeout_cls()


def _is_retryable(exc: BaseException) -> bool:
    """Retry only the two transient obs-processing failures: the zero-width
    layout ``ZeroDivisionError`` and Playwright's ``TimeoutError`` (from the
    broken internal recovery). Everything else propagates unchanged."""
    if isinstance(exc, ZeroDivisionError):
        return True
    if _PW_TIMEOUT is not None and isinstance(exc, _PW_TIMEOUT):
        return True
    # Defensive fallback if the Playwright import path changed: match Playwright's
    # TimeoutError by name + module (NOT builtins.TimeoutError).
    if type(exc).__name__ == "TimeoutError" and "playwright" in (type(exc).__module__ or ""):
        return True
    return False


def install_obs_retry_patch() -> None:
    """Wrap ``browser_env.processors.TextObervationProcessor.process`` with the
    bounded re-fetch retry. Idempotent. Call once per runner process, after the
    WebArena checkout is importable and BEFORE the env is constructed."""
    global _INSTALLED
    if _INSTALLED:
        return

    from browser_env import processors  # lazy: needs the WebArena checkout on path.

    original_process = processors.TextObervationProcessor.process

    def _process_with_retry(self: Any, page: Any, client: Any) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                result = original_process(self, page, client)
                if attempt > 0:
                    OBS_RETRY_STATS["rescues"] += 1
                return result
            except Exception as exc:  # noqa: BLE001
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    OBS_RETRY_STATS["retries"] += 1
                    logger.warning(
                        "obs processing failed (%s); retrying fetch (%d/%d)",
                        exc, attempt + 1, _MAX_ATTEMPTS - 1,
                    )
                    time.sleep(_RETRY_SLEEP_SECONDS)
        # All attempts exhausted — re-raise the LAST exception unchanged.
        assert last_exc is not None  # only reachable via the retryable path
        raise last_exc

    processors.TextObervationProcessor.process = _process_with_retry
    _INSTALLED = True
    logger.info(
        "Installed WebArena obs-retry patch (%d attempts, %.1fs backoff).",
        _MAX_ATTEMPTS, _RETRY_SLEEP_SECONDS,
    )


def install_page_client_patch() -> None:
    """Patch ``ScriptBrowserEnv.get_page_client`` to recreate a missing per-page CDP
    ``client`` on demand. Idempotent. Call once per runner process, after the
    WebArena checkout is importable and BEFORE the env is constructed.

    WebArena attaches ``page.client`` (a ``context.new_cdp_session``) at reset
    (``envs.py``) and on the ``new_tab`` action (``actions.py``), but NOT on a
    ``page_focus`` to — or a JS/``window.open`` popup that becomes — the active page.
    When such a page becomes ``env.page``, ``get_page_client(page)`` (``return
    page.client``) — called by ``_get_obs`` every step AND by ``run.py``'s evaluator
    call ``env.get_page_client(env.page)`` — raises ``AttributeError("'Page' object
    has no attribute 'client'")`` and the task is eaten upstream as
    ``[Unhandled Error]``. Agents that open tabs often hit it most.

    The patch returns an existing ``client`` unchanged, and when absent recreates
    the CDP session on demand — ``page.context.new_cdp_session(page)`` (+
    ``Accessibility.enable`` in a11y mode, mirroring the env's own setup) — caches it
    back on the page, and counts it in ``PAGE_CLIENT_STATS["client_recreated"]``. If
    the recreate ITSELF fails, it falls back to the original accessor (return
    ``page.client``, re-raising the AttributeError) WITHOUT caching or counting — so
    failure is explicitly the status quo (the task is still eaten upstream, but the
    chunk/run does not crash), never a half-attached session.

    Comparability (same policy as the obs-retry patch above): a DOCUMENTED harness
    deviation from stock WebArena, which drops these tasks as ``[Unhandled Error]``.
    Installed IDENTICALLY in both arms, so the ENVIRONMENT's response to a missing
    client is the same for both — it removes a confound (the defect no longer depends
    on which arm hit it). It is NOT literally outcome-neutral, though: the rescue's
    FIRING RATE is agent-behaviour-dependent (tab-opening frequency differs by agent),
    so absolute numbers on tab-opening tasks shift by an arm-specific amount vs
    upstream. A comparability caveat for absolute claims; the Δ is de-confounded,
    not guaranteed untouched.
    """
    global _PAGE_CLIENT_INSTALLED
    if _PAGE_CLIENT_INSTALLED:
        return

    from browser_env import envs  # lazy: needs the WebArena checkout on path.

    def _get_page_client_safe(self: Any, page: Any) -> Any:
        client = getattr(page, "client", None)
        if client is not None:
            return client
        # Client-less page (a tab the env attached no CDP session to). Recreate it
        # on demand, mirroring ScriptBrowserEnv's own setup, and cache it back. Only
        # cache + count AFTER a full success, so a failed recreate leaves no
        # half-attached/uncounted session and falls back to the status quo.
        try:
            new_client = page.context.new_cdp_session(page)
            if getattr(self, "text_observation_type", None) == "accessibility_tree":
                new_client.send("Accessibility.enable")
        except Exception:  # noqa: BLE001
            logger.warning(
                "page.client recreate failed; falling back to the original accessor "
                "(task eaten as before, run does not crash)", exc_info=True,
            )
            return page.client  # re-raises the original AttributeError semantics
        page.client = new_client
        PAGE_CLIENT_STATS["client_recreated"] += 1
        logger.warning("recreated missing page.client (CDP session) for a client-less page")
        return new_client

    envs.ScriptBrowserEnv.get_page_client = _get_page_client_safe
    _PAGE_CLIENT_INSTALLED = True
    logger.info("Installed WebArena page.client recreate-on-demand patch.")
