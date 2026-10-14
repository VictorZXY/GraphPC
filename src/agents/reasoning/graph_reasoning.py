from __future__ import annotations

import json
import logging

from typing import Any, Optional

from .base import (
    TERMINAL_CLAIM_CONTRACT,
    ReasoningAgent,
    ReasoningInput,
    empty_decision_message,
    require_decision_shape,
)
from ..backbone.base import Backbone
from ..common.compaction import compact_graph
from ..common.json_utils import JsonParseExhausted
from ..common.resample import DEFAULT_RESAMPLE_ATTEMPTS, call_with_resample
from ..common.effect_class import EFFECT_CLASSES
from ..common.schema import AgentMessage
from ..common.schemas_json import reasoning_schema_for_mode

logger = logging.getLogger("desktopenv.agent")


class GraphReasoningAgent(ReasoningAgent):
    """Graph-reasoning call, shared by every backbone (design.md Workflow).

    One model call emits both the reasoning (plan / intent / decided actions)
    AND the fused graph update — that is what keeps per-step model calls low.
    Three modes:

      * ``decide`` — single-pass: closes the prior step's actual effect class,
        (re)plans when needed, rewrites the running graph_summary, and decides
        the action(s).
      * ``initial`` — two-pass first call: closes the prior effect + rewrites
        the running summary + emits a ``retrieval_request``; does NOT decide an
        action.
      * ``refine`` — two-pass second call: consumes the retrieved experience +
        the running summary and decides the action(s).

    The deciding modes (decide/refine) carry the predicted effect class inside
    each intended action — the Verification field-contract guard (required, non-null, in
    every pre-action mode's strict schema) so the prediction is never silently
    lost.
    """

    DECIDING_MODES = frozenset({"decide", "refine"})

    def __init__(
        self,
        *,
        backbone: Backbone,
        max_trajectory_length: int = 3,
        model_call_attempts: int = DEFAULT_RESAMPLE_ATTEMPTS,
        deciding_instruction_suffix: str = TERMINAL_CLAIM_CONTRACT,
        action_class_vocabulary: tuple[str, ...] | None = None,
    ) -> None:
        self.backbone = backbone
        # Deciding-prompt text-history depth, in items. Benchmark-scoped
        # (design.md §10), like the settings below: the default, 3, is WebArena's;
        # OSWorld 2.0 passes its own larger cap from its build_default, because
        # its control's text history is unbounded and a 3-item window there is a
        # confound rather than a design choice. Sized in items rather than
        # characters; `_history_window` records the step where it binds, so a
        # capped prompt is never a silent one.
        self.max_trajectory_length = max_trajectory_length
        # TOTAL attempts at the deciding call when the reply fails to parse (1 =
        # no re-sample). Benchmark-scoped: the default is the WebArena control's
        # budget, one request per call; OSWorld 2.0 passes 3 from its own
        # build_default to match V2's M3 control. See common/resample.py for the
        # full rationale.
        self.model_call_attempts = model_call_attempts
        # The block appended to every DECIDING-mode instruction. Benchmark-scoped
        # on the same discipline as the budget above: the default is the shared
        # terminal-claim contract, WebArena's block; OSWorld 2.0 passes its own
        # wider block from its build_default (src/benchmarks/osworld/prompts.py),
        # which adds the operating rules and anti-give-up guardrails its control's
        # prompt carries. Named for what it HOLDS rather than for the
        # shared default's content, because on OSWorld it holds more than a
        # terminal claim.
        self.deciding_instruction_suffix = deciding_instruction_suffix
        # The closed vocabulary `action_class` may take, or None for the free-form
        # field. Benchmark-scoped on the same discipline as the three settings above:
        # None is WebArena's default, and it reaches the schema helper as the
        # module constant itself, so a WebArena run sends the shared schema.
        # OSWorld 2.0 passes its grounding TOOLS block's own action set.
        #
        # IT ADVERTISES; IT DOES NOT ENFORCE, and nothing here pretends otherwise.
        # Server-side enforcement exists on the GPT backbone only; MiniMax's deciding
        # route is `response_format=json_object` with the schema appended to the
        # prompt as advisory text (backbone/minimax/main.py, module docstring and
        # ``_invoke_reasoning``), and
        # `require_decision_shape` checks container shape alone. What the enum buys
        # on that route is that the model is TOLD the legal names, where the
        # free-form field tells it none. Holding it to them is normalisation's job, and that lives
        # benchmark-side in `src/benchmarks/osworld/action_minimax.py` — the same
        # division `normalize_effect_class` already makes for the other closed
        # vocabulary.
        self.action_class_vocabulary = action_class_vocabulary

    def _schema_for_mode(self, mode: str) -> tuple[dict, str]:
        """The schema THIS agent routes for `mode`, vocabulary applied.

        Split out of `reason` so what the agent would actually send can be read
        without making a model call, and so a benchmark subclass can extend it
        (``src/benchmarks/osworld/reasoning.py``). Reading the built
        object is route-independent; comparing the module constant is not.
        """
        return reasoning_schema_for_mode(mode, action_classes=self.action_class_vocabulary)

    def reason(self, reasoning_input: ReasoningInput) -> AgentMessage:
        mode = reasoning_input.reasoning_mode
        json_schema, schema_name = self._schema_for_mode(mode)
        # Prompts are built ONCE and closed over: every re-sample sends the
        # identical request bytes (see common/resample.py).
        system_prompt = self._system_prompt(mode)
        user_prompt = self._build_user_prompt(reasoning_input)
        history_window = self._history_window(reasoning_input)
        try:
            response, resamples = call_with_resample(
                lambda: require_decision_shape(self.backbone.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    observation=reasoning_input.observation,
                    json_schema=json_schema,
                    schema_name=schema_name,
                )),
                attempts=self.model_call_attempts,
                on_retry=lambda attempt, exc: logger.warning(
                    "reasoning output unparseable (%s mode, attempt %d): %s; re-sampling", mode, attempt, exc,
                ),
            )
        except JsonParseExhausted as exc:
            # Unparseable deciding response (extract_json_object exhausted its
            # parsers, or require_decision_shape refused the container shape)
            # after the re-sample budget — soft-degrade to an empty
            # decision so the malformed responses yield a noop step, not a
            # 0.0 episode. Deliberately NARROW: any other ValueError out of the
            # backbone (e.g. the pattern-bearing-schema guard raise) is one of
            # OUR contract bugs and propagates loudly (an agent_error in the WebArena
            # runner's episode-error classification); network/API errors are not
            # ValueError either and propagate (an infra_error there).
            logger.warning("reasoning output unparseable (%s mode): %s; degrading to a noop step", mode, exc)
            message = empty_decision_message(mode, reason="reasoning output unparseable")
            message.metadata["resample_attempts"] = getattr(exc, "resample_attempts", 0)
            if history_window is not None:
                message.metadata["history_window"] = history_window
            return message
        response_json = response["json"]

        content: dict[str, Any] = {
            "reasoning_text": response_json.get("reasoning_text", ""),
        }
        # plan (nullable in decide/initial; absent in refine)
        plan_payload = response_json.get("plan")
        if plan_payload is not None:
            content["plan"] = plan_payload
        # prior-step closure (decide/initial)
        if "actual_effect_class" in response_json:
            content["actual_effect_class"] = response_json["actual_effect_class"]
        # deciding-mode handoff to action (decide/refine) — carries the prediction
        if "next_intent" in response_json:
            content["next_intent"] = response_json["next_intent"]
        if "intended_actions" in response_json:
            content["intended_actions"] = response_json["intended_actions"]
        # demand-driven retrieval request (initial only)
        if "retrieval_request" in response_json:
            content["retrieval_request"] = response_json["retrieval_request"]

        # Graph stage fused into the reasoning call (design.md §1): the one graph
        # artifact, graph_summary, is surfaced as the graph_update payload the
        # orchestrator writes back; there is no separate graph call.
        graph_update: dict[str, Any] = {}
        if "graph_summary" in response_json:
            graph_update["graph_summary"] = response_json["graph_summary"]
        if graph_update:
            content["graph_update"] = graph_update
            if "graph_summary" in graph_update and "graph_summary" not in content:
                content["graph_summary"] = graph_update["graph_summary"]

        message_type = "plan" if plan_payload is not None else "local_reasoning"
        metadata: dict[str, Any] = {"reasoning_mode": mode, "resample_attempts": resamples}
        if history_window is not None:
            metadata["history_window"] = history_window
        return AgentMessage(
            sender="reasoning",
            message_type=message_type,
            content=content,
            raw={
                "gpt_response": response["raw_response"],
                "response_json": response_json,
            },
            metadata=metadata,
        )

    def _history_window(self, reasoning_input: ReasoningInput) -> Optional[dict[str, Any]]:
        """``{cap, available, dropped, step_idx}`` when ``max_trajectory_length``
        actually BOUND this call's prompt; ``None`` when every item fitted.

        Present so the bound is a recorded fact rather than a code read. Something
        bounds the deciding prompt whatever we do, and the alternative bound is the
        model's context window, which does not announce itself: it returns an error,
        the re-sample budget burns, and the step degrades to a noop. A tail of silent
        noops on exactly the longest episodes is indistinguishable in the telemetry
        from an agent that stopped making progress — a degraded path contaminating
        the measurement. A cap that fires visibly is the honest version of the same
        limit. Absent (not zero) when it never fired, so a reader can tell "never
        bound" from "not recorded"."""
        available = len(reasoning_input.runtime_state.recent_history)
        dropped = available - self.max_trajectory_length
        if dropped <= 0:
            return None
        return {
            "cap": self.max_trajectory_length,
            "available": available,
            "dropped": dropped,
            "step_idx": reasoning_input.runtime_state.step_idx,
        }

    # ---- prompt construction --------------------------------------------------

    def _graph_instruction(self) -> str:
        return (
            "Maintain a running natural-language graph_summary of the procedure so far: "
            "what has been accomplished, the current sub-goal, and what remains. Keep it "
            "compact (a few sentences) and rewrite it each step rather than appending. "
            "Do not emit node/edge structure — the causal record is carried by each "
            "action's predicted effect_class and the actual_effect_class you label it "
            "with. Do not invent GUI state or actions not supported by the inputs."
        )

    def _decide_instruction(self) -> str:
        return (
            "DECIDE mode (single-pass): in one call, (1) label the prior action's actual "
            "effect_class against the resulting observation (set actual_effect_class; null at "
            "step 0 with no prior action), (2) (re)plan when needed (emit plan, else null), "
            "(3) update the running graph_summary, and (4) decide the next intended_actions. Each intended "
            "action MUST carry a predicted_effect_class drawn from the closed vocabulary "
            f"({'|'.join(EFFECT_CLASSES)}) — this is the prediction that will be verified next step. "
            + self.deciding_instruction_suffix
        )

    def _initial_instruction(self) -> str:
        return (
            "INITIAL mode (two-pass first call): (1) label the prior action's actual "
            "effect_class against the resulting observation (null at step 0), (2) (re)plan when "
            "needed (else null), (3) update the running graph_summary, and (4) emit retrieval_request "
            "{scope ∈ task|stage|recovery|none, query (nullable rewritten string), k}. Do NOT "
            "decide a current-step action — that is the refine call. ``scope=none`` skips "
            "retrieval when the agent is confident."
        )

    def _refine_instruction(self) -> str:
        return (
            "REFINE mode (two-pass second call): consume the retrieved experience + the running "
            "summary and decide the next intended_actions. Each intended action MUST carry a predicted_effect_class "
            f"({'|'.join(EFFECT_CLASSES)}) — the prediction that will be verified next step. "
            + self.deciding_instruction_suffix
        )

    def _system_prompt(self, mode: str) -> str:
        if mode == "initial":
            mode_instr = self._initial_instruction()
        elif mode == "refine":
            mode_instr = self._refine_instruction()
        else:
            mode_instr = self._decide_instruction()
        return (
            "You are the GraphPC graph-reasoning agent. "
            + mode_instr
            + " "
            + self._graph_instruction()
        )

    def _build_user_prompt(self, reasoning_input: ReasoningInput) -> str:
        observation_content = reasoning_input.observation.content \
            if isinstance(reasoning_input.observation.content, dict) else {}
        observation_text = observation_content.get("observation_text", "")
        existing_graph = {
            key: reasoning_input.runtime_state.content.get(key)
            for key in ("graph_summary",)
            if reasoning_input.runtime_state.content.get(key)
        }
        # `observation_text` is sent even when EMPTY: it is "" on every step of an
        # `--observation_type image` run and on any page whose tree parses to zero
        # nodes.
        payload: dict[str, Any] = {
            "reasoning_mode": reasoning_input.reasoning_mode,
            "instruction": reasoning_input.instruction,
            "current_stage": reasoning_input.runtime_state.current_stage,
            "step_idx": reasoning_input.runtime_state.step_idx,
            "observation_text": observation_text,
            "plan": self._serialize_plan(reasoning_input.plan),
            "recent_history": reasoning_input.runtime_state.recent_history[-self.max_trajectory_length:],
        }
        payload["existing_graph"] = compact_graph(existing_graph) if existing_graph else {}
        payload["retrieved_memory"] = self._serialize_retrieved_memory(reasoning_input.retrieved_memory)
        return json.dumps(payload, indent=2)

    def _serialize_retrieved_memory(self, retrieved_memory) -> dict[str, Any]:
        if retrieved_memory is None:
            return {}
        return {
            "scope": retrieved_memory.scope,
            "items": retrieved_memory.items,
            "content": retrieved_memory.content,
        }

    def _serialize_plan(self, plan) -> Optional[dict[str, Any]]:
        if plan is None:
            return None
        return {
            "goal": plan.goal,
            "current_stage": plan.current_stage,
            "stages": plan.stages,
            "content": plan.content,
        }
