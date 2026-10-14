"""WebArena action agent — emits id-based browser action strings.

The grounding agent for every WebArena backbone. Uses ``complete_json`` with
``WEBARENA_ACTION_OUTPUT_SCHEMA``, which each backbone ships its own way (a
strict server-side schema on GPT, a forced tool call elsewhere) so the model
cannot return off-shape text. WebArena id-based actions are NOT a backbone-
native tool call (unlike OSWorld's computer tool), so this path is plain
JSON-emission.
"""

from __future__ import annotations

import json
import logging

from src.agents.action.base import ActionAgent, ActionInput
from src.agents.backbone.base import Backbone
from src.agents.common.json_utils import JsonParseExhausted
from src.agents.common.resample import call_with_resample
from src.agents.common.schema import ActionBatch

from ._action_common import (
    WEBARENA_ACTION_OUTPUT_SCHEMA,
    _build_user_payload,
    _system_prompt,
    build_action_batch,
    project_intended_actions,
)


logger = logging.getLogger("desktopenv.agent")


class WebArenaActionAgent(ActionAgent):
    """``url_mappings`` is the site-URL table (``url_mappings.py``), required and
    with no default: the same object ``build_default`` hands the observation
    adapter, used here in the other direction (``build_action_batch``)."""

    def __init__(self, *, backbone: Backbone, url_mappings: dict[str, str],
                 tool_name: str = "webarena_browser") -> None:
        self.backbone = backbone
        self.url_mappings = url_mappings
        self.tool_name = tool_name

    def act(self, action_input: ActionInput) -> ActionBatch:
        intended = project_intended_actions(action_input.reasoning_message)
        if not intended:
            # Nothing to ground (e.g. reasoning soft-degraded to an empty decision
            # on a parse failure): return an empty batch so the runner's
            # empty-batch defensive path emits a noop — no wasted grounding call.
            return ActionBatch(actions=[], content={}, raw={})
        payload = _build_user_payload(action_input, intended)
        # Prompt built ONCE and closed over: every re-sample sends identical
        # request bytes (see common/resample.py).
        system_prompt = _system_prompt()
        user_prompt = json.dumps(payload, indent=2)
        try:
            response, resamples = call_with_resample(
                lambda: self.backbone.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    observation=action_input.observation,
                    json_schema=WEBARENA_ACTION_OUTPUT_SCHEMA,
                    schema_name="webarena_action",
                    grounding=True,  # explicit grounding call; every backbone routes on it.
                ),
                on_retry=lambda attempt, exc: logger.warning(
                    "grounding output unparseable (attempt %d): %s; re-sampling", attempt, exc,
                ),
            )
        except JsonParseExhausted as exc:
            # Unparseable GROUNDING response (extract_json_object exhausted its
            # parsers) after the re-sample budget — the action-side analog of
            # the reasoning path's soft-degrade: return an empty batch
            # carrying the marker so the shim emits a COUNTED noop and surfaces
            # the failure upstream next step; malformed grounding responses
            # cost a step, never the episode (design.md degrade contract).
            # Deliberately NARROW: any other ValueError out of the backbone is
            # one of OUR bugs and propagates loudly (an agent_error in the runner's
            # episode-error classification); network/API errors are not ValueError
            # either and propagate (an infra_error there).
            logger.warning("grounding output unparseable: %s; degrading to noop", exc)
            return ActionBatch(
                actions=[],
                content={"grounding_parse_error": f"{type(exc).__name__}: {exc}"},
                raw={},
                metadata={"resample_attempts": getattr(exc, "resample_attempts", 0)},
            )
        batch = build_action_batch(
            raw_response=response["raw_response"],
            response_json=response["json"],
            intended_actions=intended,
            action_input=action_input,
            tool_name=self.tool_name,
            url_mappings=self.url_mappings,
        )
        batch.metadata["resample_attempts"] = resamples
        return batch
