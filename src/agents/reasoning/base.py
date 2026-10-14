from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..common.json_utils import JsonParseExhausted
from ..common.schema import AgentMessage, Observation, PlanState, RetrievalResult, RuntimeState


# Shared terminal-claim contract appended to every DECIDING-mode instruction
# (decide / refine) across backbones. Backbone- and benchmark-agnostic: it
# mirrors the INTENT of the benchmarks' own stop-instruction without
# benchmark-specific wording. Rationale: with no progress-gated stop,
# termination is the model's own explicit decision, so the deciding prompt must
# cue it — an uncued model runs every episode to the step cap and never submits
# an answer. This contract makes the terminal `done` an explicit, contracted
# decision. It coexists with the M2 completion gate:
# an answer-bearing `done` is scored by rule 1 (answer grounding) — hence
# "grounded in what you actually observed", which helps legitimate wins pass — and
# an empty/state-change `done` by rule 2.
TERMINAL_CLAIM_CONTRACT = (
    "TERMINAL-CLAIM CONTRACT: termination is YOUR explicit decision — do not run out "
    "the step budget. When the task objective is ACHIEVED, decide the terminal `done` "
    "action now. For an information/answer-seeking task the `done` MUST carry the final "
    "answer itself — exact and concise, grounded in what you actually observed (the "
    "answer text only, no explanation). For a state-change task the `done` carries no "
    "answer text. If the objective is genuinely impossible/unachievable, decide `done` "
    "with a brief statement to that effect — never fabricate an answer."
)


@dataclass
class ReasoningInput:
    """
    reasoning_mode: "decide" (single-pass), or "initial" / "refine" (two-pass).
    """

    reasoning_mode: str
    instruction: str
    observation: Observation
    runtime_state: RuntimeState

    plan: Optional[PlanState] = None
    retrieved_memory: Optional[RetrievalResult] = None

    content: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def empty_decision_message(reasoning_mode: str, *, reason: str) -> AgentMessage:
    """Soft-degrade payload for an unparseable deciding response: a
    reasoning message that carries NO intended_actions and an EMPTY graph_update,
    so downstream the action stage produces an empty batch (the runner's
    empty-batch defensive path emits a noop) and the graph record is left as it
    was — instead of the ``extract_json_object`` ``ValueError``
    propagating and failing the whole episode 0.0. The ``reasoning_parse_error``
    marker is read by the benchmark adapter and surfaced upstream on the next step
    via the same feedback channel as a grounding failure. Backbone-agnostic:
    every reasoning agent returns this on a parse failure."""
    return AgentMessage(
        sender="reasoning",
        message_type="local_reasoning",
        content={
            "reasoning_text": f"{reason}; degraded to a noop step",
            "intended_actions": [],
            "graph_update": {},
            "reasoning_parse_error": reason,
        },
        raw={"reasoning_parse_error": reason},
        metadata={"reasoning_mode": reasoning_mode, "reasoning_parse_error": reason},
    )


def _shape_message(field: str, value: Any) -> str:
    shown = repr(value)
    if len(shown) > 200:
        shown = shown[:197] + "..."
    return (
        f"reasoning output has the wrong shape: {field} is {type(value).__name__}, "
        f"not the contracted container ({shown})"
    )


def require_decision_shape(response: Dict[str, Any]) -> Dict[str, Any]:
    """Container-shape check on a parsed deciding response, run INSIDE the
    re-sampled call (``lambda: require_decision_shape(backbone.complete_json(...))``)
    so a wrong shape is handled exactly like an unparseable reply: re-sampled
    under the site's existing budget, then degraded to ``empty_decision_message``.
    Returns ``response`` unchanged when it passes.

    Checks ONLY what the consumers dereference without a guard: ``plan`` must be
    absent, None or a dict (``orchestrator._extract_plan_state`` subscripts it)
    and ``intended_actions`` must be absent or a list whose every item is a dict
    (the grounding agents call ``.get`` on each item). Unchecked, either
    failure would escape ``predict`` as a TypeError / AttributeError and end the
    episode unscored instead of costing one degraded step. These are the schema's own
    container constraints (``schemas_json.py``: ``_PLAN_OR_NULL``, the
    ``intended_actions`` items), so a schema-valid reply passes untouched — the
    GPT backbone enforces them server-side already; the Claude, MiniMax and
    DeepSeek routes do not. A ``null`` or dict ``intended_actions`` and a
    ``[None]`` item do not crash the consumers, but they are outside the
    contract too and are re-sampled the same way (failure path only). Nothing
    else is validated here.

    Raises ``JsonParseExhausted`` — the exact class ``call_with_resample``
    retries — naming the field, the observed type and a bounded repr, because
    on a shape failure nothing else logs the reply (``extract_json_object`` does
    so only on a parse failure)."""
    response_json = response["json"]
    plan = response_json.get("plan")
    if plan is not None and not isinstance(plan, dict):
        raise JsonParseExhausted(_shape_message("plan", plan))
    if "intended_actions" in response_json:
        intended_actions = response_json["intended_actions"]
        if not isinstance(intended_actions, list):
            raise JsonParseExhausted(_shape_message("intended_actions", intended_actions))
        for index, item in enumerate(intended_actions):
            if not isinstance(item, dict):
                raise JsonParseExhausted(_shape_message(f"intended_actions[{index}]", item))
    return response


class ReasoningAgent(ABC):
    @abstractmethod
    def reason(self, reasoning_input: ReasoningInput) -> AgentMessage:
        """
        One model call, reasoning and graph update fused (``GraphReasoningAgent``):

            AgentMessage(
                sender="reasoning",
                message_type="plan" when the reply carries a plan, else "local_reasoning",
                content={
                    "reasoning_text": "...",
                    "plan": {...},                     # when (re)planned
                    "actual_effect_class": "...",      # the prior step's label (decide / initial)
                    "intended_actions": [...],         # each with its predicted_effect_class (decide / refine)
                    "retrieval_request": {...},        # initial only
                    "graph_update": {"graph_summary": "..."},
                }
            )

        An unparseable reply returns ``empty_decision_message``.
        """
        raise NotImplementedError
