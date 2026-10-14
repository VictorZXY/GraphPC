"""The backbone contract the shared method agents call.

``GraphReasoningAgent`` and ``WebArenaActionAgent`` annotate ``backbone: Backbone``:
what they need from a backbone is this one method, whichever model is behind it.
The contract is documentation, not enforcement. ``Backbone`` is not
``runtime_checkable`` and no backbone subclasses it, so it changes no MRO, no
isinstance test and no request byte. It is still a runtime import of the shared
reasoning module, which is why this file is in the OSWorld sync list.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class Backbone(Protocol):
    """One model call returning a JSON object: all the shared agents call.

    RESULT. A dict carrying at least ``"json"`` (the reply parsed as an object) and
    ``"raw_response"`` (the reply as the provider sent it, kept for the record).
    The shared agents read those two keys and nothing else; the other keys differ
    per backbone.

    ERRORS. ``JsonParseExhausted`` (``src/agents/common/json_utils.py``) means the
    reply could not be read as the requested object. It is the one exception the
    callers catch: the trigger for a re-sample and, once the budget is spent, for
    the degrade to a no-op step. Every other exception propagates.

    ``grounding=True`` marks the grounding call (the WebArena action agent's).
    Claude, DeepSeek, MiniMax and Qwen route on it (thinking off, a forced tool
    call carrying the schema); on GPT it only selects the reasoning effort and the
    usage label.

    THREE DIVERGENCES, documented and deliberately not unified:
      * a ``json_schema`` carrying a regex ``pattern`` with ``grounding=False``
        raises ``ValueError`` on Claude, DeepSeek, MiniMax and Qwen; GPT has no
        such guard (its strict schema route enforces the grammar either way);
      * DeepSeek ignores ``observation`` (text-only); the others attach its
        screenshot when it carries one;
      * ``schema_name`` names the strict response format on GPT, on both routes;
        elsewhere it names the forced tool on the grounding route and is unused on
        the deciding route.

    OPTIONAL HOOKS, outside the Protocol; the adapters probe for them with
    ``hasattr``: ``reset_session()`` (every backbone but GPT; called at an episode
    reset) and ``call_observer`` (MiniMax and Qwen; only OSWorld's adapter sets it,
    and no WebArena component holds one).
    ``complete_text`` and the hooks of OSWorld's timeline route are an OSWorld-side
    contract, not part of this one.
    """

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
        ...
