"""Single boundary helper for all WebArena optional imports.

Keeps the try/except dance for WebArena's optional dependencies in ONE
place so the rest of the package stays clean (no scattered import guards).
Every other module here imports through this helper, except
``_env_hardening.py``, which imports ``browser_env`` directly inside its
install functions (they run only on a live WebArena run).

When the WebArena checkout is installed and on the path, the real types /
factories resolve. When it isn't (dev / CI env), the fallbacks return
``None`` for factory callables and a stub ``Agent`` base that exposes the
expected methods so subclassing still works without a checkout. Live WebArena
runs need the real package on the path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("desktopenv.agent")


# ---------------------------------------------------------------------------
# Agent base
# ---------------------------------------------------------------------------

try:  # pragma: no cover — exercised at deploy time, not in this env.
    from agent.agent import Agent as _RuntimeAgentBase  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    class _RuntimeAgentBase:  # type: ignore[no-redef]
        """Fallback base used when no WebArena checkout is on the path.

        Exposes the canonical method names so subclassing succeeds at import
        time; raises if called without an override (subclasses always
        override). Mirrors the runtime contract of WebArena's ``Agent``
        (``next_action`` + ``reset``).
        """

        def next_action(self, trajectory, intent, meta_data):  # noqa: D401
            raise NotImplementedError

        def reset(self, test_config_file):  # noqa: D401
            raise NotImplementedError


WebArenaAgentBase = _RuntimeAgentBase


# ---------------------------------------------------------------------------
# Action factories (browser_env.actions.*)
# ---------------------------------------------------------------------------


def load_factories() -> dict[str, Optional[Callable[..., Any]]]:
    """Lazy load the three WebArena action factories.

    Returns a dict with keys ``id_based`` / ``stop`` / ``none`` — values are
    the callables when WebArena is installed, ``None`` otherwise. Callers
    decide whether ``None`` is a hard failure or a degrade path (the action
    translator raises ``RuntimeError`` when a needed factory is missing; a
    caller can inject factories directly to bypass this loader).
    """
    try:
        from browser_env.actions import (  # type: ignore[import-not-found]
            create_id_based_action,
            create_none_action,
            create_stop_action,
        )
        return {
            "id_based": create_id_based_action,
            "stop": create_stop_action,
            "none": create_none_action,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("WebArena factories unavailable (lazy import): %s", exc)
        return {"id_based": None, "stop": None, "none": None}


def load_action_parsing_error() -> Optional[type]:
    """Lazy load ``browser_env.actions.ActionParsingError`` — the exception the
    id-based action factory raises on a malformed action string.

    Returned so the shim can catch exactly this type and degrade to a noop
    (the episode CONTINUES, mirroring WebArena's own PromptAgent, which falls
    back to ``create_none_action`` on a parse failure) instead of letting the
    parse error crash the whole episode. Returns ``None`` when no WebArena
    checkout is on the path."""
    try:
        from browser_env.actions import ActionParsingError  # type: ignore[import-not-found]
        return ActionParsingError
    except Exception as exc:  # noqa: BLE001
        logger.debug("WebArena ActionParsingError unavailable (lazy import): %s", exc)
        return None


def load_stop_support() -> tuple[Any, Optional[Callable[..., Any]]]:
    """Lazy load ``(ActionTypes.STOP, create_stop_action)`` for the live loop.

    The run loop needs both: ``ActionTypes.STOP`` to detect the agent's own
    terminal action (break WITHOUT stepping the env), and ``create_stop_action``
    to synthesize the terminal stop that must be the last trajectory element
    when the env terminates. Returns ``(None, None)`` when no WebArena checkout
    is on the path (dry-run / CI use their own ``"stop"`` marker instead).
    """
    try:
        from browser_env.actions import (  # type: ignore[import-not-found]
            ActionTypes,
            create_stop_action,
        )
        return ActionTypes.STOP, create_stop_action
    except Exception as exc:  # noqa: BLE001
        logger.debug("WebArena stop-support unavailable (lazy import): %s", exc)
        return None, None
