"""Translate our grounded ``ActionBatch.actions`` into WebArena ``Action``s
through the factory functions.

We do NOT hand-build WebArena's ``Action`` TypedDict — its ``text`` field holds
token-ids and many other fields must be filled — so all conversions route
through ``browser_env.actions.create_id_based_action`` /
``create_stop_action`` / ``create_none_action``. The WebArena action agent
(``action_agent.py``) produces the id-based action string on each
``Action.raw["input"]["command"]``; this translator just calls the right
factory.

WebArena imports are LAZY so this module imports cleanly in an env without a
WebArena checkout — the translation functions raise on use in that case unless
the caller injects factories (``factories=``).
"""

from __future__ import annotations

from typing import Any, Optional

from src.agents.common.schema import Action
from src.agents.common.stop_answer import done_answer, stop_command

from ._webarena_imports import load_factories as _load_factories


# Action types we own internally that map to factory shortcuts rather than
# to a parseable id-based action string.
_TERMINAL_DONE = "done"
_NOOP_WAIT = "wait"


def translate_action(action: Action, *, factories: Optional[dict[str, Any]] = None) -> Any:
    """Convert one of our grounded ``Action``s into a WebArena ``Action``.

    Routing:
      - ``done``  -> ``create_stop_action(answer)``. The answer is extracted from
        the ``stop [answer]`` command with the SAME shared parser
        (``common/stop_answer``) the completion gate uses, so the submitted answer
        and the gate's rule-1 grounding can never disagree — NOT via a second
        ``create_id_based_action`` parse (which could diverge on bracketed answers).
        ``action.parameters['answer']`` is used only as the fallback when there is
        no command at all (dry-run stub / OSWorld-style state done).
      - ``wait``  -> ``create_none_action()`` (WebArena has no explicit wait;
        emitting a no-op preserves the step boundary).

    There is no ``fail`` branch: the WebArena action grammar has no ``fail``
    verb, and nothing produces a synthetic ``fail`` decision (there is no
    progress-gated stop), so a ``fail`` action_type never reaches this
    translator (the completion gate emits ``wait``; on OSWorld 2.0 the give-up is
    the model's own ``fail`` action, handled by that benchmark's adapter).
      - any other action_type -> the action agent placed the id-based action
        STRING on ``action.raw["input"]["command"]``; we call
        ``create_id_based_action(command)``.

    ``factories`` lets callers inject factory functions (e.g. mocks). The
    default lazy-loads from ``browser_env.actions``; raises ``RuntimeError``
    when the factory required for the action_type isn't available.
    """
    facs = factories if factories is not None else _load_factories()
    action_type = (action.action_type or "").lower()

    if action_type == _TERMINAL_DONE:
        if facs.get("stop") is None:
            raise RuntimeError("WebArena create_stop_action unavailable")
        # When the done carries a `stop [answer]` command (the WebArena action
        # agent's output), extract the answer with the shared parser — the same
        # one the completion gate reads. Only when there is NO command (dry-run
        # stub / OSWorld-style done) fall back to parameters['answer'].
        if stop_command(action):
            answer = done_answer(action)
        else:
            answer = str(action.parameters.get("answer") or "") if isinstance(action.parameters, dict) else ""
        return facs["stop"](answer)

    if action_type == _NOOP_WAIT:
        if facs.get("none") is None:
            raise RuntimeError("WebArena create_none_action unavailable")
        return facs["none"]()

    # Default: the action agent emitted the id-based action string.
    raw = action.raw if isinstance(action.raw, dict) else {}
    tool_input = raw.get("input") if isinstance(raw.get("input"), dict) else {}
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(
            f"Action.raw['input']['command'] missing for action_type={action.action_type!r}; "
            "the WebArena action agent must emit the id-based action string."
        )
    if facs.get("id_based") is None:
        raise RuntimeError("WebArena create_id_based_action unavailable")
    return facs["id_based"](command.strip())
