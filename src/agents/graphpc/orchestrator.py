from __future__ import annotations

import json
import re

from copy import deepcopy
from typing import Any, Dict, Optional, Protocol

from ..action.base import ActionAgent, ActionInput
from ..common.compaction import compact_history_item
from ..common.stop_answer import done_answer
from ..common.effect_class import (
    POSITIVE_PROGRESS_EFFECTS,
    UNKNOWN,
    effect_family,
    effects_match,
    is_progress_effect,
    is_valid_effect_class,
    normalize_effect_class,
)
from ..common.schema import (
    Action,
    ActionBatch,
    AgentMessage,
    GraphPCDecision,
    GraphPCRuntimeUpdate,
    Observation,
    PlanState,
    RetrievalResult,
    RuntimeState,
)
from ..monitor.base import MonitorAgent, MonitorInput
from ..reasoning.base import ReasoningAgent, ReasoningInput
from ..retrieval.base import RetrievalAgent, RetrievalInput

ObservationInput = Observation | Dict[str, Any]


class PerceptionProcessor(Protocol):
    """
    Optional observation processing hook.

    This stays lightweight on purpose: some backbones may need an explicit
    perception step, while others can consume the raw observation directly.
    """

    def __call__(
        self,
        observation_input: ObservationInput,
        runtime_state: RuntimeState,
    ) -> Observation:
        raise NotImplementedError


# Observation keys holding the raw structured text: truncated in the compact
# variant that retrieval embeds, and promoted untruncated into ``observation_text``
# for the deciding reasoning call (see ``_reasoning_observation``).
_HEAVY_OBSERVATION_KEYS = ("a11y_tree", "accessibility_tree")

# Completion-gate (Mechanism 2, design.md §6) constants — fixed a priori as part of the
# mechanism, NOT sweep knobs.
# Positive-progress effects EXCLUDING navigation: navigation alone never admits a
# done (a navigate-then-done ending is the loss the gate exists to block).
_NON_NAV_POSITIVE_EFFECTS: frozenset[str] = POSITIVE_PROGRESS_EFFECTS - {"navigation"}
# Answer-grounding: admit an answer-bearing stop iff informative-token overlap
# with the recent observations is at least this fraction.
_ANSWER_OVERLAP_THRESHOLD: float = 0.5
# Blocked-done becomes a submitted done after this many corrective blocks in one
# episode (the evaluator is the final judge; the gate only forces evidence).
_MAX_BLOCKED_DONE_CORRECTIONS: int = 2
# Answers equal to these (case-insensitive) are treated as empty (route to rule 2).
_EMPTY_ANSWER_SENTINELS: frozenset[str] = frozenset({"", "n/a", "na", "none"})
_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMERIC_RE = re.compile(r"\d[\d.,]*")
# Small English stopword set for informative-token extraction (answer grounding).
_ANSWER_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "at", "for", "and", "or", "but", "if", "then", "with",
    "as", "by", "from", "that", "this", "these", "those", "it", "its", "there",
    "here", "you", "your", "i", "we", "he", "she", "they", "them", "his", "her",
    "not", "no", "yes", "do", "does", "did", "has", "have", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "so", "than",
    "into", "out", "up", "down", "over", "about",
})


def _injected_chars(items: Any) -> Optional[int]:
    """Size of the exemplars this retrieval put in front of the model, as the
    character length of their compact JSON.

    A stable PROXY for injected volume, not a byte count of the prompt: the
    prompt re-serializes the same objects with indentation inside a larger
    payload. Measured after the injection diet, since that is what is injected.
    ``default=str`` keeps a telemetry field from being able to raise on a stored
    value that is not JSON-native; ``None`` when there is nothing to size.
    """
    if not items:
        return 0 if isinstance(items, list) else None
    return len(json.dumps(items, default=str))


class GraphPCOrchestrator:
    """
    Orchestrator for the GraphPC workflow (design.md Workflow).

    Two workflow shapes, selectable by the ``two_pass`` flag:

      * **single-pass (default)** — retrieval[deterministic] → graph-reasoning
        ("decide") → action. The single graph-reasoning call closes the prior
        step's actual effect, (re)plans, rewrites the running graph_summary, and
        decides the action(s) — all fused.
      * **two-pass (a backup design, not the default)** —
        graph-reasoning("initial") → retrieval [deterministic with agent-emitted
        request] → graph-reasoning("refine") → action. ``initial`` closes the
        prior effect + rewrites the running summary + emits a
        ``retrieval_request``; ``refine`` consumes retrieval and decides the
        action. Both shapes route experience into the deciding
        reasoning call (Workflow) and use the slim ``intended_actions`` handoff into
        the action stage.

    Observation views (design.md §2): the DECIDING reasoning call reads the
    RAW environment observation (the full accessibility tree promoted into
    ``observation_text``, untruncated — ``_reasoning_observation``); retrieval
    embeds the compact projection (structured text truncated to
    ``compact_observation_chars`` — ``_make_compact_observation``); the action stage reads the full observation
    as delivered (the element-id projection) for grounding.
    """

    def __init__(
        self,
        *,
        reasoning_agent: ReasoningAgent,
        retrieval_agent: RetrievalAgent,
        action_agent: ActionAgent,
        monitor_agent: Optional[MonitorAgent] = None,
        perception_processor: Optional[PerceptionProcessor] = None,
        default_backend: str = "custom",
        max_actions: int = 1,
        two_pass: bool = False,
        compact_observation_chars: int = 6000,
        enable_completion_gate: bool = True,
        completion_gate_window: int = 3,
        history_intended_actions_cap: Optional[int] = None,
        completion_gate_answer_rule: bool = True,
    ) -> None:
        self.reasoning_agent = reasoning_agent
        self.retrieval_agent = retrieval_agent
        self.action_agent = action_agent
        self.monitor_agent = monitor_agent
        self.perception_processor = perception_processor
        self.default_backend = default_backend
        self.max_actions = max_actions
        self.two_pass = two_pass
        self.compact_observation_chars = compact_observation_chars
        # How many of a step's intended_actions survive into recent_history.
        # Benchmark-scoped: None = the unbounded copy, WebArena's default.
        # OSWorld 2.0 passes its own value from its build_default; see
        # common/compaction.py:_bound_intended_actions for why the un-shrunk
        # forward plan dominates a history item there.
        self.history_intended_actions_cap = history_intended_actions_cap
        # Whether the completion gate's RULE 1 (answer-bearing stop, grounded against
        # observation text) is available at all. Benchmark-scoped on the same
        # discipline as the settings above: True is WebArena's default; OSWorld 2.0
        # passes False from its build_default.
        #
        # WHY IT IS OFF THERE, and it is structural rather than a tuning choice. Rule 1
        # grounds the answer against recent observation TEXT, and OSWorld's observation
        # text is constantly empty by construction (`observation_adapter.py` sets
        # `observation_text = ""` on a screenshot-only benchmark). So rule 1 cannot
        # ever PASS there — and, because it ends in `return False` with NO fallthrough
        # to rule 2, making it reachable would turn every answer-bearing done into a
        # permanent block: a done carrying "Task cannot be completed: ..." would be
        # blocked on every attempt under a reachable rule 1 instead of admitted by
        # rule 2 for the right reason.
        #
        # The gate on OSWorld is therefore rule 2 BY DESIGN rather than by accident.
        self.completion_gate_answer_rule = completion_gate_answer_rule
        # Completion gate (Mechanism 2; design.md §6). There is no
        # progress-gated stop — the max-steps cap is the sole budget bound.
        self.enable_completion_gate = enable_completion_gate
        self.completion_gate_window = completion_gate_window

    # ------------------------------------------------------------------
    # Public boundary (stable inner contract — see common/schema.py types)
    # ------------------------------------------------------------------

    def decide_next_action(
        self,
        *,
        instruction: str,
        observation_input: ObservationInput,
        runtime_state: Optional[RuntimeState] = None,
    ) -> GraphPCDecision:
        state = self._prepare_runtime_state(instruction=instruction, runtime_state=runtime_state)
        full_observation = self._prepare_observation(observation_input=observation_input, runtime_state=state)
        compact_observation = self._make_compact_observation(full_observation)
        # design.md §2: the deciding reasoning call reads the
        # RAW accessibility tree (the same field the monitor diffs), not the
        # 200-node observation_text projection. The action stage keeps the full
        # (projection) observation for its element-id space.
        reasoning_observation = self._reasoning_observation(full_observation)

        if self.two_pass:
            deciding_msg, retrieval_result, initial_msg = self._run_two_pass(
                instruction=instruction, compact_obs=compact_observation,
                reasoning_obs=reasoning_observation, state=state,
            )
        else:
            deciding_msg, retrieval_result, initial_msg = self._run_single_pass(
                instruction=instruction, compact_obs=compact_observation,
                reasoning_obs=reasoning_observation, state=state,
            )

        plan_state = self._extract_plan_state(
            plan_payload=deciding_msg.content.get("plan"),
            instruction=instruction,
        )
        if plan_state is not None:
            state.plan = plan_state
            state.current_stage = plan_state.current_stage

        # Graph stage. The deciding reasoning call is fused (see Workflow in
        # design.md) and is the only producer of the running summary; a call that
        # emits no graph_update leaves the accumulated summary in place.
        if "graph_update" in deciding_msg.content:
            self._apply_graph_update_from_reasoning(state=state, reasoning_message=deciding_msg)

        state.content["need_retrieval"] = False
        state.content["last_reasoning_mode"] = "refine" if self.two_pass else "decide"
        state.content["last_retrieval_scope"] = retrieval_result.scope

        # Verification: close the prior step's prediction by reading the
        # actual effect_class the deciding-mode call just labelled, compare to
        # the stored prior prediction, and record the verdict (external-grounded
        # telemetry + effect-class reliability audit data; also feeds
        # completion-gate rule 2).
        actual_effect = self._extract_actual_effect_class(initial_msg, deciding_msg)
        prior_predicted = self._prior_predicted_effect_class(state)
        verdict = self._record_verification(
            state=state, prior_predicted=prior_predicted, actual=actual_effect
        )

        # No progress-gated stop (design.md §6): a stretch without progress is
        # not evidence that the task is over — successful episodes routinely
        # pass through one. Episodes end only via the agent's own stop/done
        # (subject to the completion gate) or the runner's max-steps cap. No
        # early short-circuit here.

        # Action call: FULL observation for grounding (design.md §2), slim handoff (Workflow) —
        # action sees only the intended_actions inside the reasoning_message,
        # NOT the retrieved experience or reasoning trace.
        action_batch = self.action_agent.act(
            ActionInput(
                instruction=instruction,
                observation=full_observation,
                runtime_state=state,
                plan=deepcopy(state.plan),
                reasoning_message=deciding_msg,
                retrieved_memory=None,
                max_actions=self.max_actions,
            )
        )

        # Completion gate (Mechanism 2): a ``done`` claim is
        # admitted only on external-grounded evidence (answer grounding / a
        # monitor-confirmed non-navigation state change); else blocked into a
        # corrective noop with 2-strikes-then-submit.
        completion_blocked = self._apply_completion_gate(
            action_batch=action_batch, state=state, current_observation=full_observation,
        )

        planning_message = deciding_msg if deciding_msg.message_type == "plan" else None
        decision_content: Dict[str, Any] = {
            "perception_applied": self.perception_processor is not None,
            "reasoning_mode": "refine" if self.two_pass else "decide",
            "two_pass": self.two_pass,
            "retrieval_scope": retrieval_result.scope,
            "verification": verdict,
        }
        if completion_blocked:
            decision_content["completion_gate_blocked"] = True
        graph_update_payload = self._select_graph_update_payload(reasoning_message=deciding_msg)
        if graph_update_payload is not None:
            decision_content["graph_update"] = graph_update_payload
        if initial_msg is not None:
            decision_content["initial_reasoning"] = deepcopy(initial_msg.content)

        return GraphPCDecision(
            observation=full_observation,
            action_batch=action_batch,
            runtime_state=state,
            planning=planning_message,
            reasoning=deciding_msg,
            retrieval=retrieval_result,
            content=decision_content,
        )

    def update_runtime(
        self,
        *,
        decision: GraphPCDecision,
        current_observation_input: ObservationInput,
    ) -> GraphPCRuntimeUpdate:
        # NOTE (DEFERRED): this deep-copies the whole runtime_state each
        # decide/update step, including the per-step telemetry lists on .content
        # (_obs_text_history etc.) — O(n^2) copying across an episode. Bounded by
        # max_steps (30 on WebArena, 500 on OSWorld 2.0): a synthetic 500-step
        # state costs ~1.5 MB / ~16 ms per deepcopy, two copies per step (here
        # and _prepare_runtime_state) ≈ 10 s per episode — a minor per-episode
        # cost, not a correctness issue; left as-is until it shows up as a
        # measured hotspot.
        state = deepcopy(decision.runtime_state)
        current_observation = self._prepare_observation(
            observation_input=current_observation_input,
            runtime_state=state,
        )

        monitor_message = None
        if self.monitor_agent is not None:
            monitor_message = self.monitor_agent.evaluate(
                MonitorInput(
                    instruction=state.instruction,
                    prev_observation=decision.observation,
                    current_observation=current_observation,
                    action_batch=decision.action_batch,
                    runtime_state=state,
                    plan=deepcopy(state.plan),
                )
            )
            self._apply_monitor_feedback(
                state=state, monitor_message=monitor_message,
                current_observation=current_observation,
            )

        # No graph stage here: the graph update is fused into the deciding call,
        # which already wrote it onto state during decide_next_action.

        state.step_idx += 1
        history_item = {
            "step_idx": state.step_idx,
            "stage_idx": state.stage_idx,
            "current_stage": state.current_stage,
            "actions": [
                {
                    "action_type": action.action_type,
                    "parameters": deepcopy(action.parameters),
                    "metadata": deepcopy(action.metadata),
                }
                for action in decision.action_batch.actions
            ],
            "reasoning": deepcopy(decision.reasoning.content) if decision.reasoning else None,
            "retrieval": deepcopy(decision.retrieval.content) if decision.retrieval else None,
            "monitor": deepcopy(monitor_message.content) if monitor_message else None,
            "status": state.status,
        }

        state.recent_history.append(compact_history_item(
            history_item, intended_actions_cap=self.history_intended_actions_cap,
        ))

        update_content: Dict[str, Any] = {
            "monitor_evaluated": self.monitor_agent is not None,
            "status": state.status,
        }
        graph_update_payload = self._select_graph_update_payload(reasoning_message=decision.reasoning)
        if graph_update_payload is not None:
            update_content["graph_update"] = graph_update_payload

        return GraphPCRuntimeUpdate(
            previous_observation=decision.observation,
            current_observation=current_observation,
            action_batch=decision.action_batch,
            runtime_state=state,
            monitor=monitor_message,
            content=update_content,
        )

    # ------------------------------------------------------------------
    # Workflow shapes
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        *,
        instruction: str,
        compact_obs: Observation,
        reasoning_obs: Observation,
        state: RuntimeState,
    ) -> tuple[AgentMessage, RetrievalResult, Optional[AgentMessage]]:
        """retrieval[deterministic] → graph-reasoning("decide") → (action).

        design.md §2: the DECIDING reasoning call reads the RAW accessibility
        tree (``reasoning_obs``: the full tree the baseline sees), not the 200-node
        projection — a compact view drops decision-relevant rows. Retrieval keeps
        the compact observation."""
        scope = self._select_retrieval_scope(state)
        retrieval_result = self.retrieval_agent.retrieve(
            RetrievalInput(
                retrieval_scope=scope,
                instruction=instruction,
                observation=compact_obs,
                runtime_state=state,
                plan=deepcopy(state.plan),
            )
        )
        self._record_retrieval_event(state=state, retrieval_result=retrieval_result)
        deciding_msg = self.reasoning_agent.reason(
            ReasoningInput(
                reasoning_mode="decide",
                instruction=instruction,
                observation=reasoning_obs,
                runtime_state=state,
                plan=deepcopy(state.plan),
                retrieved_memory=retrieval_result,
            )
        )
        return deciding_msg, retrieval_result, None

    def _run_two_pass(
        self,
        *,
        instruction: str,
        compact_obs: Observation,
        reasoning_obs: Observation,
        state: RuntimeState,
    ) -> tuple[AgentMessage, RetrievalResult, AgentMessage]:
        """initial → retrieval(request-or-deterministic) → refine → (action).

        design.md §2: both reasoning passes (initial + the deciding refine)
        read the RAW accessibility tree (``reasoning_obs``) — they are the same
        fused graph-reasoning call family. Retrieval keeps the compact observation.
        """
        initial_msg = self.reasoning_agent.reason(
            ReasoningInput(
                reasoning_mode="initial",
                instruction=instruction,
                observation=reasoning_obs,
                runtime_state=state,
                plan=deepcopy(state.plan),
                retrieved_memory=None,
            )
        )

        # Apply initial's plan + graph update so refine sees a consistent state.
        initial_plan = self._extract_plan_state(
            plan_payload=initial_msg.content.get("plan"),
            instruction=instruction,
        )
        if initial_plan is not None:
            state.plan = initial_plan
            state.current_stage = initial_plan.current_stage
        if "graph_update" in initial_msg.content:
            self._apply_graph_update_from_reasoning(state=state, reasoning_message=initial_msg)

        # Demand-driven retrieval scope/query; deterministic fallback when
        # the request is missing/empty. ``scope=none`` skips retrieval.
        request = initial_msg.content.get("retrieval_request") or {}
        scope = request.get("scope") or self._select_retrieval_scope(state)
        query = request.get("query")
        k = int(request.get("k") or 2)

        if scope == "none":
            retrieval_result = RetrievalResult(
                scope="none",
                items=[],
                content={"skipped": True, "reason": "agent_scope_none"},
            )
        else:
            retrieval_result = self.retrieval_agent.retrieve(
                RetrievalInput(
                    retrieval_scope=scope,
                    instruction=instruction,
                    observation=compact_obs,
                    runtime_state=state,
                    plan=deepcopy(state.plan),
                    query=query,
                    top_k=k,
                )
            )
        self._record_retrieval_event(state=state, retrieval_result=retrieval_result)

        deciding_msg = self.reasoning_agent.reason(
            ReasoningInput(
                reasoning_mode="refine",
                instruction=instruction,
                observation=reasoning_obs,
                runtime_state=state,
                plan=deepcopy(state.plan),
                retrieved_memory=retrieval_result,
            )
        )
        return deciding_msg, retrieval_result, initial_msg

    # ------------------------------------------------------------------
    # Observation views (design.md §2): compact for retrieval, raw for reasoning
    # ------------------------------------------------------------------

    def _make_compact_observation(self, full: Observation) -> Observation:
        """Compact projection of an Observation for the RETRIEVAL stage (the
        deciding reasoning call reads the raw tree instead): structured-text
        fields truncated to ``compact_observation_chars``
        (a11y trees especially are the dominant token cost); screenshot and
        other content keys preserved unchanged. The full observation is reserved
        for the action stage's grounding pass (design.md §2).
        """
        if not isinstance(full.content, dict):
            return deepcopy(full)
        content = dict(full.content)
        limit = self.compact_observation_chars
        observation_text = content.get("observation_text")
        if isinstance(observation_text, str) and len(observation_text) > limit:
            content["observation_text"] = observation_text[:limit]
        for key in _HEAVY_OBSERVATION_KEYS:
            value = content.get(key)
            if isinstance(value, str) and len(value) > limit:
                content[key] = value[:limit]
        return Observation(
            backend=full.backend,
            content=content,
            raw=full.raw,
            metadata=deepcopy(full.metadata),
        )

    def _reasoning_observation(self, full: Observation) -> Observation:
        """The observation the DECIDING reasoning call reads (design.md §2):
        the RAW accessibility tree promoted into
        ``observation_text``, so the reasoner sees the same full-fidelity tree the
        baseline does — NOT the 200-node projection that ``observation_text``
        otherwise carries (``a11y_tree`` is the raw tree; ``observation_text`` is
        the ``parse_axtree(max_nodes=200)`` projection the ACTION stage keeps for
        its element-id space). The reasoning agent reads ``observation_text``, so
        the raw tree is promoted into that field. No truncation (raw trees are
        genuinely longer — a context-length error must surface, not be clipped).
        Falls back to the existing observation_text when no raw-tree field is
        present (non-WebArena backends / image mode)."""
        if not isinstance(full.content, dict):
            return full
        raw_tree = next(
            (full.content[key] for key in _HEAVY_OBSERVATION_KEYS
             if isinstance(full.content.get(key), str) and full.content.get(key)),
            None,
        )
        if raw_tree is None:
            return full
        content = dict(full.content)
        content["observation_text"] = raw_tree
        return Observation(
            backend=full.backend,
            content=content,
            raw=full.raw,
            metadata=deepcopy(full.metadata),
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _prepare_runtime_state(
        self,
        *,
        instruction: str,
        runtime_state: Optional[RuntimeState],
    ) -> RuntimeState:
        if runtime_state is None:
            return RuntimeState(instruction=instruction)
        return deepcopy(runtime_state)

    def _prepare_observation(
        self,
        *,
        observation_input: ObservationInput,
        runtime_state: RuntimeState,
    ) -> Observation:
        if self.perception_processor is not None:
            observation = self.perception_processor(observation_input, runtime_state)
            if not isinstance(observation, Observation):
                raise TypeError("perception_processor must return Observation")
            return observation

        if isinstance(observation_input, Observation):
            return deepcopy(observation_input)

        if isinstance(observation_input, dict):
            payload = deepcopy(observation_input)
            return Observation(
                backend=self.default_backend,
                content=payload,
                raw=payload,
            )

        raise TypeError("observation_input must be Observation or dict")

    def _record_retrieval_event(self, *, state: RuntimeState, retrieval_result: RetrievalResult) -> None:
        """Retrieval-event telemetry: append an injection-level retrieval event to
        ``_retrieval_events`` on state.content — ``selected_entry_ids``,
        per-entry ``text_scores``, and ``injected`` (post relevance gate: whether
        any entry survived). Purely additive telemetry; nothing in the decision
        path reads it. Ranking is text-only, so each entry carries the one score
        there is.

        No degraded-arm counter: an unavailable text arm RAISES
        (``TextArmUnavailableError`` -> ``infra_error``) rather than serving a
        degraded ranking, so there is no "ran, but degraded" state left to count.
        A permanent 0 here would read to a scorer as "verified healthy" — the
        opposite of what it would mean.

        Built from ``selected`` ALONE, three different states would serialize
        identically here — an empty store, a warm store whose gate rejected
        every candidate, and a warm store that ranked but scored poorly — which
        would make "retrieval contributed nothing" unattributable after the fact.
        The store size and the arm's health are already computed and sitting on
        the retrieval result, so they are copied onto the event (``text_arm_enabled``
        is None under ``--retrieval_off``, where no arm ran at all and the event
        also carries ``retrieval_bypassed``). ``injected_chars`` sizes what was
        actually put in front of the model, so injected volume per step is
        readable from the record rather than inferred from token bills.

        SCOPE — stated precisely, because it differs from the benchmark-scoped
        settings. This one is NOT scoped off on WebArena and has
        no parameter: it RUNS on both benchmarks and adds three keys to every
        recorded event on both. The claim is the weaker but true one — **no
        behaviour change on WebArena**, not "cannot execute there". It holds
        because ``state.content['_retrieval_events']`` reaches no prompt: the
        reasoning prompt takes an explicit key allow-list from
        ``runtime_state.content`` (``graph_summary`` only — ``reasoning/graph_reasoning.py``),
        the WebArena action prompt reads only
        ``runtime_state.current_stage``, and both adapters' ``_build_graph_artifacts``
        persist the same one-key allow-list, so the added fields cannot reach the
        support set and return inside a warm exemplar either. The underscore keys
        are read only by the adapters' end-of-episode telemetry dump. No prompt
        byte depends on them; the recorded telemetry carries the three keys,
        deliberately.
        """
        content = retrieval_result.content or {}
        selected = list(content.get("selected_entry_ids") or [])
        scores = content.get("scores") or {}
        text_scores = {eid: (scores.get(eid) or {}).get("text_score") for eid in selected}
        events = list(state.content.get("_retrieval_events") or [])
        event: Dict[str, Any] = {
            "step_idx": state.step_idx,
            "scope": retrieval_result.scope,
            "selected_entry_ids": selected,
            "text_scores": text_scores,
            "injected": bool(selected),
            "candidate_count": content.get("candidate_count"),
            "text_arm_enabled": content.get("text_arm_enabled"),
            "injected_chars": _injected_chars(retrieval_result.items),
        }
        # Echo the retrieval-off operational mode when it is active (read from
        # metadata, which — unlike content — never reaches a prompt). Absent key
        # means the default configuration, so the event shape is unchanged when
        # retrieval is on.
        if (retrieval_result.metadata or {}).get("retrieval_bypassed"):
            event["retrieval_bypassed"] = True
        events.append(event)
        state.content["_retrieval_events"] = events

    def _select_retrieval_scope(self, runtime_state: RuntimeState) -> str:
        if runtime_state.content.get("need_retrieval"):
            return "recovery"
        if runtime_state.plan is None or runtime_state.step_idx == 0:
            return "task"
        return "stage"

    def _extract_plan_state(
        self,
        *,
        plan_payload: Any,
        instruction: str,
    ) -> Optional[PlanState]:
        if plan_payload is None:
            return None

        if isinstance(plan_payload, PlanState):
            return deepcopy(plan_payload)

        if not isinstance(plan_payload, dict):
            raise TypeError("reasoning plan payload must be PlanState or dict")

        return PlanState(
            goal=plan_payload.get("goal") or instruction,
            current_stage=plan_payload.get("current_stage"),
            stages=deepcopy(plan_payload.get("stages", [])),
            content=deepcopy(plan_payload.get("content", {})),
            raw=deepcopy(plan_payload.get("raw", {})),
            metadata=deepcopy(plan_payload.get("metadata", {})),
        )

    def _apply_monitor_feedback(
        self,
        *,
        state: RuntimeState,
        monitor_message: AgentMessage,
        current_observation: Optional[Observation] = None,
    ) -> None:
        progress_content = monitor_message.content

        state.content["need_retrieval"] = bool(progress_content.get("need_retrieval", False))
        state.content["progress_status"] = progress_content.get("progress_status", "UNKNOWN")

        # Mechanism 1: thread the monitor's external text-diff status forward so the
        # NEXT decide's _record_verification can reconcile the model's actual-effect
        # label against it. update_runtime CLOSES the just-executed action, so this
        # is the monitor status for exactly the action that _record_verification
        # closes on the next decide (the documented monitor[k] ↔ verdict[k+1] offset).
        # Overwritten each update_runtime; additive on state.content.
        state.content["_last_monitor_status"] = {
            "changed": bool(progress_content.get("changed", False)),
            "progress_status": progress_content.get("progress_status", "UNKNOWN"),
        }

        # Telemetry-only (design principle #2 measurement-instrumentation
        # exemption): accumulate the per-step external monitor signal so the
        # effect-class reliability audit can cross-check the model's
        # actual_effect_class labels offline. NOTHING
        # in the decision path reads _monitor_history — only progress_status
        # (set above) drives control flow. Mirrors how _actual_effects
        # accumulates (list copy + append).
        #
        # Alignment for the audit (verified against the call sites): _monitor_history
        # is appended once per monitor-enabled update_runtime, which CLOSES a
        # decision — so there is NO entry for the initial pre-action decide (no
        # prior decision to close). It is NOT positionally 1:1 with _actual_effects,
        # which decide_next_action appends CONDITIONALLY (only when the model emits
        # a valid actual_effect_class) — a filtered subset. The robust positional
        # companion is _verification_verdicts (appended UNCONDITIONALLY once per
        # decide): _monitor_history[k] corresponds to _verification_verdicts[k+1]
        # (offset by one — the monitor closes step k while that transition's
        # predicted/actual verdict is recorded when deciding step k+1).
        monitor_history = list(state.content.get("_monitor_history") or [])
        monitor_history.append({
            "progress_status": progress_content.get("progress_status", "UNKNOWN"),
            "observed_change": progress_content.get("observed_change", ""),
        })
        state.content["_monitor_history"] = monitor_history

        # Telemetry-only (design principle #2 measurement-instrumentation
        # exemption): persist the CURRENT compact observation text the monitor
        # scored, so the audit's gold sample can annotate each transition against
        # its before/after observation pair. NOTHING in the decision path reads
        # _obs_text_history. Appended at the SAME call site as _monitor_history,
        # so the alignment is identical: _obs_text_history[k] is the post-action
        # ("after") observation the monitor scored at _monitor_history[k] — i.e.
        # the "after" of step k's transition; the "before" is
        # _obs_text_history[k-1]. The initial pre-action observation is NOT
        # captured (update_runtime closes a decision), so the k=0 transition's
        # "before" is absent. Truncated to 4000 chars defensively (~2KB/step;
        # can reach ~100KB/task — intentionally kept in the telemetry dump).
        obs_text = ""
        if current_observation is not None and isinstance(current_observation.content, dict):
            obs_text = str(current_observation.content.get("observation_text", ""))[:4000]
        obs_text_history = list(state.content.get("_obs_text_history") or [])
        obs_text_history.append(obs_text)
        state.content["_obs_text_history"] = obs_text_history

        if progress_content.get("stage_completed"):
            state.stage_idx += 1

        if progress_content.get("task_completed"):
            state.status = "DONE"
        elif progress_content.get("task_infeasible"):
            state.status = "FAIL"

    def _apply_graph_update_from_reasoning(
        self,
        *,
        state: RuntimeState,
        reasoning_message: AgentMessage,
    ) -> None:
        """Write the fused call's graph update onto state.

        The reasoning call emits no node/edge topology (design.md §3), so the
        running natural-language ``graph_summary`` is the only artifact there is
        to carry forward.
        """
        graph_update = reasoning_message.content.get("graph_update") or {}
        if "graph_summary" in graph_update:
            state.content["graph_summary"] = graph_update["graph_summary"]

    def _select_graph_update_payload(
        self,
        *,
        reasoning_message: Optional[AgentMessage],
    ) -> Optional[Dict[str, Any]]:
        """
        Surface the fused call's graph_update dict for the decision/runtime
        envelope so the compulsory-graph invariant stays observable downstream.
        """
        if reasoning_message is not None and "graph_update" in reasoning_message.content:
            return deepcopy(reasoning_message.content["graph_update"])
        return None

    # ------------------------------------------------------------------
    # Verification + gates (design.md Verification)
    # ------------------------------------------------------------------

    def _extract_actual_effect_class(
        self,
        initial_msg: Optional[AgentMessage],
        deciding_msg: Optional[AgentMessage],
    ) -> Optional[str]:
        """Shape-aware getter for the just-emitted prior-step actual effect.

        ``actual_effect_class`` is closed by the *first* reasoning call of the
        step:
          - single-pass: the deciding ``decide`` call carries it.
          - two-pass: the ``initial`` call carries it (``refine`` does not — it
            decides the current action only).
        Returns the normalized closed-vocab value, or None when no actual was
        emitted (e.g. step 0 with no prior action to close).
        """
        msg = initial_msg if initial_msg is not None else deciding_msg
        if msg is None:
            return None
        raw = msg.content.get("actual_effect_class")
        if raw is None:
            return None
        return normalize_effect_class(raw)

    def _prior_predicted_effect_class(self, state: RuntimeState) -> Optional[str]:
        """Recover the prior step's predicted effect class from recent_history.

        First the prior step's ``Action.metadata.predicted_effect_class``
        (mirrored by the action stage on the slim handoff), then fall back to
        the prior reasoning's ``intended_actions[].predicted_effect_class``
        (kept by ``compact_history_item``). Returns the normalized closed-vocab
        value, or None when no prior prediction exists (step 0).
        """
        if not state.recent_history:
            return None
        last = state.recent_history[-1]
        if not isinstance(last, dict):
            return None
        actions = last.get("actions") or []
        for action in actions:
            if isinstance(action, dict):
                metadata = action.get("metadata")
                if isinstance(metadata, dict):
                    value = metadata.get("predicted_effect_class")
                    if value:
                        return normalize_effect_class(value)
        reasoning = last.get("reasoning") or {}
        intended = reasoning.get("intended_actions") if isinstance(reasoning, dict) else None
        if isinstance(intended, list):
            for item in intended:
                if isinstance(item, dict):
                    value = item.get("predicted_effect_class")
                    if value:
                        return normalize_effect_class(value)
        return None

    def _reconcile_actual_effect(
        self,
        *,
        actual_model: Optional[str],
        monitor_status: Optional[Dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        """Mechanism 1: reconcile the model's actual-effect label against the
        monitor's external text-diff. Returns ``(actual_recorded, override_tag)``.

        On a STALL step (monitor reports NO observation change) a positive-progress
        claim, an ``UNKNOWN``, or a missing label all mean "no external evidence of
        change" → recorded ``no_change`` (external-grounded telemetry + effect-class
        reliability audit data; also feeds completion-gate rule 2). On a CHANGED
        step the model stays authoritative for WHICH class occurred (incl.
        ``UNKNOWN`` staying ``UNKNOWN``); we never manufacture ``no_change`` out of
        a page that did change (the safe-direction rule). At
        step 0 / with no monitor there is no external signal → no reconciliation.
        """
        if monitor_status is None:
            return actual_model, None
        if bool(monitor_status.get("changed")):
            return actual_model, None  # page changed: model authoritative
        # STALL — no observation change.
        if actual_model is None or actual_model == UNKNOWN or is_progress_effect(actual_model):
            return "no_change", "stall_reconciled"
        # Model already reports explicit non-progress (error/no_change) → agrees.
        return actual_model, None

    def _record_verification(
        self,
        *,
        state: RuntimeState,
        prior_predicted: Optional[str],
        actual: Optional[str],
    ) -> Dict[str, Any]:
        """Append the actual effect to ``_actual_effects`` + record a verdict.

        Mechanism 1: the effect appended and verified is the monitor-RECONCILED
        value (external-grounded), not the raw model label; the verdict keeps both
        (``actual_model`` / ``actual_recorded`` / ``override`` / ``monitor_changed``)
        so verdict-level telemetry is preserved and doubles as effect-class
        reliability audit data.
        ``_actual_effects`` is telemetry only (there is no progress-gated stop
        to consume it); completion-gate rule 2 reads the verdicts.
        Resolution-level telemetry (design.md §7).
        """
        monitor_status = state.content.get("_last_monitor_status")
        actual_recorded, override = self._reconcile_actual_effect(
            actual_model=actual, monitor_status=monitor_status
        )
        monitor_changed = None if monitor_status is None else bool(monitor_status.get("changed"))

        if actual_recorded is not None and is_valid_effect_class(actual_recorded):
            actuals = list(state.content.get("_actual_effects") or [])
            actuals.append(actual_recorded)
            state.content["_actual_effects"] = actuals

        if prior_predicted is None or actual is None:
            verdict: Dict[str, Any] = {
                "resolved": False,
                "reason": "no_prior_prediction" if prior_predicted is None else "no_actual",
                "actual_model": actual,
                "actual_recorded": actual_recorded,
                "override": override,
                "monitor_changed": monitor_changed,
            }
        else:
            # Verify the prediction against the RECONCILED (external-grounded) actual.
            matches = effects_match(prior_predicted, actual_recorded)
            verdict = {
                "resolved": True,
                "predicted": prior_predicted,
                "actual": actual_recorded,
                "actual_model": actual,
                "actual_recorded": actual_recorded,
                "override": override,
                "matches": matches,
                "monitor_changed": monitor_changed,
            }

        verdicts = list(state.content.get("_verification_verdicts") or [])
        verdicts.append(verdict)
        state.content["_verification_verdicts"] = verdicts
        return verdict

    def _apply_completion_gate(
        self,
        *,
        action_batch: ActionBatch,
        state: RuntimeState,
        current_observation: Optional[Observation] = None,
    ) -> bool:
        """Completion gate on external-grounded evidence (Mechanism 2, design.md §6).

        A ``done`` claim is admitted only when its evidence is grounded in the
        recent observations:
          1. answer-bearing stop → the answer's informative tokens overlap the
             recent observation text (≥0.5) OR every numeric token in the answer
             appears there;
          2. empty/"N/A" stop → a NON-navigation positive effect in the recent
             window whose monitor status confirms the page changed (label + the
             external signal agree; navigation alone never admits).
        A blocked done is CORRECTIVE, not terminal: the batch is marked blocked
        (the WebArena adapter emits a noop + surfaces which rule failed via
        _env_feedback), and after ``_MAX_BLOCKED_DONE_CORRECTIONS`` blocks in one
        episode the next done stands and is submitted (the benchmark evaluator is
        the final judge). Returns True iff the gate fired (blocked).

        (There is no any-progress-effect admit: a bare navigation never admits a
        done. Evidence is external-grounded answer/monitor agreement, not a
        model self-label, so an all-``UNKNOWN`` trajectory is still decidable
        via Mechanism-1 reconciliation.)
        """
        if not self.enable_completion_gate:
            return False
        if not any(a.action_type == "done" for a in action_batch.actions):
            return False

        # 2-strikes escape: the gate forces evidence-gathering, it does not veto
        # the final submission. Bounds the burn-to-cap risk both ways.
        blocked_count = int(state.content.get("_blocked_done_count") or 0)
        if blocked_count >= _MAX_BLOCKED_DONE_CORRECTIONS:
            return False

        admit, failed_rule = self._completion_admits(
            action_batch=action_batch, state=state, current_observation=current_observation,
        )
        if admit:
            return False

        state.content["_blocked_done_count"] = blocked_count + 1
        # Replace the done with a corrective NOOP (not a give-up fail): the step
        # executes a noop so a possibly-valid answer survives for re-submission,
        # AND the closed action the next monitor/verification sees is the noop —
        # never the blocked done (which would otherwise be read as task-completed).
        # The adapter reads the content marker to emit the noop + surface feedback.
        action_batch.actions = [
            Action(
                action_type="wait",
                parameters={"webarena_verb": "noop", "command": "noop"},
                raw={"name": "computer", "input": {"command": "noop", "verb": "noop"}},
                metadata={"reason": "completion_gate_corrective_noop", "tool_name": "computer"},
            )
        ]
        action_batch.content = {
            **action_batch.content,
            "completion_gate_blocked": True,
            "completion_gate_failed_rule": failed_rule,
        }
        return True

    def _completion_admits(
        self,
        *,
        action_batch: ActionBatch,
        state: RuntimeState,
        current_observation: Optional[Observation],
    ) -> tuple[bool, str]:
        """The Mechanism-2 admit decision. Returns ``(admit, failed_rule)``."""
        done_action = next((a for a in action_batch.actions if a.action_type == "done"), None)
        answer = self._done_answer(done_action)
        # Rule 1 is unavailable where the benchmark has no observation text to ground
        # an answer against (see __init__). Skipping it here — rather than letting it
        # run and fail — is the difference between "rule 2 by design" and "rule 1
        # blocks everything", because rule 1 does not fall through.
        if self.completion_gate_answer_rule and answer.strip().lower() not in _EMPTY_ANSWER_SENTINELS:
            # Rule 1 — answer-bearing stop: the answer must be grounded in obs.
            obs_union = self._recent_observation_text(state=state, current_observation=current_observation)
            if self._answer_grounded(answer, obs_union):
                return True, ""
            return False, "answer_not_grounded"
        # Rule 2 — empty / N-A stop: require a monitor-confirmed non-navigation
        # positive-effect step in the recent window.
        if self._has_confirmed_non_nav_progress(state):
            return True, ""
        return False, "no_confirmed_state_change"

    @staticmethod
    def _done_answer(done_action: Optional[Action]) -> str:
        """Extract the stop answer from a done action via the SHARED parser
        (common/stop_answer.done_answer) — the SAME one the WebArena translator
        uses to build the submitted stop, so the gate's rule-1 grounding and the
        submitted answer can never disagree. Empty for a bare/other-benchmark done."""
        return "" if done_action is None else done_answer(done_action)

    def _recent_observation_text(
        self, *, state: RuntimeState, current_observation: Optional[Observation]
    ) -> str:
        """Union of the last ``completion_gate_window`` (default 3, the design's
        "last 3") compact observation texts (the monitor-scored _obs_text_history)
        + the current full observation text — the recency window the ANSWER is
        grounded against (rule 1). Recency matters here: the answer should be
        visible in what is currently/recently on screen."""
        window = self.completion_gate_window if self.completion_gate_window > 0 else len(state.content.get("_obs_text_history") or [])
        texts = list(state.content.get("_obs_text_history") or [])[-window:] if window else []
        if current_observation is not None and isinstance(current_observation.content, dict):
            texts.append(str(current_observation.content.get("observation_text", "")))
        return " ".join(t for t in texts if t)

    @staticmethod
    def _informative_tokens(text: str) -> frozenset[str]:
        return frozenset(
            t for t in _WORD_RE.findall(text.lower())
            if len(t) > 1 and t not in _ANSWER_STOPWORDS
        )

    @staticmethod
    def _numeric_tokens(text: str) -> frozenset[str]:
        return frozenset(m.group(0).rstrip(".,") for m in _NUMERIC_RE.finditer(text))

    def _answer_grounded(self, answer: str, obs_text: str) -> bool:
        """Deterministic answer-grounding (Mechanism 2 rule 1): informative-token
        overlap ≥ threshold, OR (the answer has numeric tokens AND every one
        appears in the observation union). The numeric clause is gated on there
        being at least one numeric token, so a text-only answer never admits
        vacuously — it must clear the overlap bar."""
        answer_tokens = self._informative_tokens(answer)
        if answer_tokens:
            obs_tokens = self._informative_tokens(obs_text)
            overlap = len(answer_tokens & obs_tokens) / len(answer_tokens)
            if overlap >= _ANSWER_OVERLAP_THRESHOLD:
                return True
        numeric = self._numeric_tokens(answer)
        if numeric:
            obs_lower = obs_text.lower()
            if all(n in obs_lower for n in numeric):
                return True
        return False

    def _has_confirmed_non_nav_progress(self, state: RuntimeState) -> bool:
        """Mechanism 2 rule 2: ANY verdict across the trajectory whose RECONCILED
        actual effect is a non-navigation positive class AND whose monitor status
        confirmed the page changed (label + external signal agree).

        Whole-trajectory (not a last-N window): "did a real, monitor-confirmed
        state change happen during this task" is a trajectory-level property. A
        last-N window wrongly blocks a legitimate completion whose state change
        happened early and was followed by navigation/verification (e.g. a
        toggle at step 0, then navigation). Navigation is excluded, so a
        navigate-then-done with no genuine state change never admits (the loss
        this gate targets)."""
        verdicts = state.content.get("_verification_verdicts") or []
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                continue
            recorded = verdict.get("actual_recorded")
            if recorded in _NON_NAV_POSITIVE_EFFECTS and verdict.get("monitor_changed") is True:
                return True
        return False


def reconcile_effect_labels(recent_history: Any, verdicts: Any) -> list[Dict[str, Any]]:
    """Episode-close relabel write-back (design.md §6).

    The reasoning call emits no topology, so the conservative rule runs over the
    effect-label channel: each ``recent_history`` step's
    ``reasoning.actual_effect_class``. That field is the per-step actual-effect
    claim (the graph's outcome node, design.md §3), and it is what the
    support-set entry persists, so the stored experience carries
    externally-grounded labels rather than the model's raw self-reports.

    For each step claiming a positive-progress effect class C: if the episode's
    M1-reconciled verdicts contain ZERO monitor-confirmed effects whose FAMILY
    matches ``family(C)`` ANYWHERE in the episode, the claim is externally
    unsupported -> relabel to ``no_change`` and tag provenance
    (``actual_effect_relabeled: "m1-episode"``). If ANY confirmed same-family
    instance exists, LEAVE it (prefer missed corrections over wrong ones; no
    fragile per-step alignment). Non-progress claims are untouched. A
    zero-confirmed-families guard applies (below).

    MUTATES the history items in place; the M1 verdict records are read-only.
    Returns the ``_relabel_events`` list (step_idx / from / to).
    """
    if not isinstance(recent_history, list):
        return []
    confirmed_families = {
        effect_family(verdict.get("actual_recorded"))
        for verdict in (verdicts or [])
        if isinstance(verdict, dict)
        and verdict.get("monitor_changed") is True
        and isinstance(verdict.get("actual_recorded"), str)
        and is_valid_effect_class(verdict.get("actual_recorded"))
    }
    # Zero-confirmed-families guard (conservative): with NO monitor-confirmed
    # effect of ANY family in the episode there is no positive external signal to
    # judge against — relabeling every positive claim would be the MAXIMAL wrong
    # correction (a genuine win degraded to a no-progress husk), against the
    # prefer-missed-over-wrong stance. Subsumes the empty-verdicts case.
    if not confirmed_families:
        return []
    events: list[Dict[str, Any]] = []
    for item in recent_history:
        if not isinstance(item, dict):
            continue
        reasoning = item.get("reasoning")
        if not isinstance(reasoning, dict):
            continue
        claimed = reasoning.get("actual_effect_class")
        if not is_progress_effect(normalize_effect_class(claimed)):
            continue  # non-progress claim (no_change/error/unknown) -> untouched
        if effect_family(claimed) in confirmed_families:
            continue  # a same-family effect was externally confirmed -> leave it
        events.append({"step_idx": item.get("step_idx"), "from": claimed, "to": "no_change"})
        reasoning["actual_effect_class"] = "no_change"
        reasoning["actual_effect_relabeled"] = "m1-episode"
    return events
