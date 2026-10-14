"""WebArena-facing GraphPC agent (the agent plug the WebArena runner drives).

Translation layer — OSWorld-shaped inside, WebArena-shaped outside:

  Inward (toward our orchestrator): mirrors ``OSWorldGraphPCAgent`` —
  ``next_action`` reads ``trajectory[-1]`` (StateInfo), converts it to an obs
  dict, calls the stable inner boundary (``decide_next_action`` /
  ``update_runtime``) ONLY (never internals), turns the resulting
  ``ActionBatch`` into a WebArena ``Action`` through the factory, and persists
  successful trajectories + graph artifacts to the support set.

  Outward (toward WebArena): subclasses ``agent.agent.Agent`` so the WebArena
  runner can drive it via the canonical ``next_action(trajectory, intent,
  meta_data) -> Action`` + ``reset(test_config_file)`` interface. Action
  emission goes through the WebArena factories (``create_id_based_action`` /
  ``create_stop_action`` / ``create_none_action``); we never hand-build the
  ``Action`` TypedDict.

WebArena imports are LAZY — this module imports cleanly when no WebArena
checkout is on the path; the ``Agent`` base resolves to a
no-op fallback that exposes the required methods so the runner-side type/MRO
checks still succeed against this class. Live runs require a real WebArena
checkout.
"""

from __future__ import annotations

import logging

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Protocol

from src.agents.action.base import ActionAgent
from src.agents.backbone.claude.main import ClaudeAgent
from src.agents.backbone.deepseek.main import DeepSeekAgent
from src.agents.backbone.minimax.main import MiniMaxAgent
from src.agents.backbone.qwen.main import QwenAgent
from src.agents.backbone.gpt.main import GPTAgent
from src.agents.common.infeasible_declaration import match_infeasible_declaration
from src.agents.common.resample import resample_attempts_of
from src.agents.common.schema import (
    Action as InternalAction,
    GraphPCDecision,
    GraphPCRuntimeUpdate,
    RuntimeState,
)
from src.agents.common.stop_answer import done_answer, stop_command
from src.agents.graphpc.orchestrator import GraphPCOrchestrator, reconcile_effect_labels
from src.agents.monitor.base import MonitorAgent
from src.agents.monitor.heuristic import HeuristicMonitorAgent
from src.agents.reasoning.base import ReasoningAgent
from src.agents.reasoning.graph_reasoning import GraphReasoningAgent
from src.agents.retrieval.base import RetrievalAgent
from src.agents.retrieval.bypass import BypassRetrievalAgent
from src.agents.retrieval.deterministic import DeterministicRetrievalAgent
from src.agents.retrieval.embedding import get_default_embedder
from src.agents.retrieval.support_set import LocalSupportSetStore

from ._webarena_imports import (
    WebArenaAgentBase as _WebArenaAgent,
    load_action_parsing_error,
)
from .action_agent import WebArenaActionAgent
from .action_translator import translate_action
from .na_normalizer import NA_ANSWER
from .observation_adapter import WebArenaObservationAdapter
from .prompts import WEBARENA_DECIDING_INSTRUCTION_SUFFIX
from .url_mappings import url_mappings_from_env


logger = logging.getLogger("desktopenv.agent")


def _traceback_originates_in_webarena_actions(exc: BaseException) -> bool:
    """True when any frame in ``exc``'s traceback is WebArena's action factory
    module (``browser_env/actions.py``). Used to classify a translate_action
    failure as an env-side grounding rejection (degrade) vs one of our own bugs
    (re-raise), since the factory's exception surface is broader than
    ActionParsingError (KeyError from the ASCII keyboard map, etc.)."""
    tb = exc.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename.replace("\\", "/")
        if filename.endswith("browser_env/actions.py"):
            return True
        tb = tb.tb_next
    return False


def _action_command(action: InternalAction) -> str:
    """The concrete WebArena action string carried on one of our grounded
    actions — ``parameters['command']`` (where ``build_action_batch`` puts it)
    else ``raw['input']['command']``. ``""`` when the action carries neither
    (e.g. an OSWorld-style done with only ``parameters['answer']``)."""
    parameters = getattr(action, "parameters", None)
    if isinstance(parameters, dict) and parameters.get("command"):
        return str(parameters["command"])
    raw = getattr(action, "raw", None)
    if isinstance(raw, dict):
        return str((raw.get("input") or {}).get("command") or "")
    return ""


class _OrchestratorProtocol(Protocol):
    """The stable inner boundary the shim depends on. Anything else is
    internals — never reach across this protocol."""

    def decide_next_action(self, *, instruction, observation_input, runtime_state) -> GraphPCDecision: ...
    def update_runtime(self, *, decision, current_observation_input) -> GraphPCRuntimeUpdate: ...


class WebArenaGraphPCAgent(_WebArenaAgent):
    """OSWorld-shaped inside, WebArena-shaped outside. Holds graph-side
    state ourselves (the WebArena ``Trajectory`` only carries WebArena
    state/action records, not our internal artifacts).
    """

    def __init__(
        self,
        *,
        reasoning_agent: ReasoningAgent,
        retrieval_agent: RetrievalAgent,
        action_agent: ActionAgent,
        support_set_store: LocalSupportSetStore,
        monitor_agent: Optional[MonitorAgent] = None,
        perception_processor: WebArenaObservationAdapter,
        max_actions: int = 1,
        two_pass: bool = False,
        enable_completion_gate: bool = True,
        completion_gate_window: int = 3,
        retrieval_off: bool = False,
        orchestrator: Optional[_OrchestratorProtocol] = None,
    ) -> None:
        # Retrieval-off operational mode, adapter half: the store-WRITE side of
        # the bypass (the read side is BypassRetrievalAgent). Persistence lives
        # here, so this is where a run is made to leave the store untouched in
        # both directions.
        self.retrieval_off = retrieval_off
        self.reasoning_agent = reasoning_agent
        self.retrieval_agent = retrieval_agent
        self.action_agent = action_agent
        self.support_set_store = support_set_store
        self.monitor_agent = monitor_agent or HeuristicMonitorAgent()
        # Required, with no default: a fallback adapter built here would carry no
        # site-URL table, and passing None would make the orchestrator skip
        # perception altogether. Either way the deciding view would lose its URL line.
        # So an explicit None is refused too.
        if perception_processor is None:
            raise TypeError(
                "WebArenaGraphPCAgent needs a perception_processor (the WebArena observation adapter "
                "with its site-URL table); None would make the orchestrator skip perception, and the "
                "deciding view would lose its URL line"
            )
        self.perception_processor = perception_processor
        # The orchestrator is injectable so a stub orchestrator (webarena.md,
        # Decoupling invariant) can be swapped in without touching this shim.
        self.orchestrator: _OrchestratorProtocol = orchestrator or GraphPCOrchestrator(
            reasoning_agent=self.reasoning_agent,
            retrieval_agent=self.retrieval_agent,
            action_agent=self.action_agent,
            monitor_agent=self.monitor_agent,
            perception_processor=self.perception_processor,
            default_backend="webarena",
            max_actions=max_actions,
            two_pass=two_pass,
            enable_completion_gate=enable_completion_gate,
            completion_gate_window=completion_gate_window,
        )

        self.runtime_state: Optional[RuntimeState] = None
        self.last_decision: Optional[GraphPCDecision] = None
        self.last_update: Optional[GraphPCRuntimeUpdate] = None
        self.task_instruction: Optional[str] = None
        self.task_id: Optional[str] = None
        # Upstream env-feedback signal stashed for the NEXT step (surfaced via
        # _env_feedback, read by the monitor pass). A dict = the `info` payload,
        # keyed by failure kind so the monitor can describe it distinctly:
        #   {"action_parse_error": ...}      grounding failure
        #   {"reasoning_parse_error": ...}   unparseable deciding response (soft-degrade)
        #   {"completion_gate_blocked": ...} blocked done, corrective (Mechanism 2)
        #   {"grounding_parse_error": ...}   unparseable grounding response (soft-degrade)
        self._pending_env_feedback: Optional[dict] = None
        # Per-episode count of grounding-failure degradations, read by the
        # runner's telemetry dump (motivated by the effect-class reliability audit).
        self.action_parse_failure_count: int = 0
        # Per-episode count of reasoning-parse soft-degrades: an unparseable
        # DECIDING response degraded to a noop step instead of failing the episode.
        # Kept distinct from action_parse_failure_count so telemetry stays honest.
        self.reasoning_parse_failure_count: int = 0
        # Per-episode count of grounding-parse soft-degrades: the action agent's
        # own complete_json hit parse exhaustion (unparseable GROUNDING response)
        # and degraded to a noop step instead of killing the episode. Distinct
        # from action_parse_failure_count (env-side translation rejects of a
        # parsed-but-invalid action string) so telemetry stays honest.
        self.grounding_parse_failure_count: int = 0
        # Free re-samples of unparseable model output (common/resample.py):
        # per-episode totals read off each decision's reasoning message /
        # action batch metadata; the pair for the CURRENT step rides on that
        # step's executed-action record (only when non-zero — additive).
        self.reasoning_resample_count: int = 0
        self.grounding_resample_count: int = 0
        # Steps where all attempts failed and the degrade path ran (noop). The
        # *_parse_failure_count pair above counts unparseable samples per
        # ATTEMPT (rescued or not) = resamples + exhausted steps, so the two
        # families are not duplicates.
        self.resample_exhausted_reasoning: int = 0
        self.resample_exhausted_grounding: int = 0
        self._step_resamples: tuple[int, int] = (0, 0)

    def ablation_switch_state(self) -> dict[str, Any]:
        """Resolved retrieval mode, read back off the constructed objects (not off
        the CLI args) so the echo cannot disagree with what is actually running.
        Written into each episode's telemetry and asserted by the launch preflight.

        Text-only ranking and exemplar-graph stripping are fixed behaviour
        rather than switches, so echoing them would report a constant.
        ``retrieval_off`` is a documented operational mode, which is what this
        reports."""
        return {
            "retrieval_off": bool(self.retrieval_off),
            "retrieval_agent": type(self.retrieval_agent).__name__,
        }

    # ------------------------------------------------------------------
    # build_default — symmetric with OSWorldGraphPCAgent.build_default.
    # ------------------------------------------------------------------

    @classmethod
    def build_default(
        cls,
        *,
        model: str = "claude-sonnet-5",
        backbone_type: str = "claude",
        support_set_root: str | Path = "artifacts/support_set",
        observation_type: str = "accessibility_tree",
        max_actions: int = 1,
        max_trajectory_length: int = 3,
        two_pass: bool = False,
        enable_completion_gate: bool = True,
        completion_gate_window: int = 3,
        retrieval_off: bool = False,
        **backbone_kwargs,
    ) -> "WebArenaGraphPCAgent":
        """Builds the agent with the WebArena backend on the support set + the
        WebArena-shaped action agent for the configured backbone. Mirrors
        ``OSWorldGraphPCAgent.build_default``; only the action agent + the
        support-set backend tag differ. ``observation_type`` must match the
        ``ScriptBrowserEnv`` setting so the perception adapter only forwards a
        screenshot into ``content`` in image mode.

        ``retrieval_off`` is the one retrieval switch — an operational mode, not
        a diagnostic. Text-only ranking and exemplar-graph stripping are fixed
        behaviour, not parameters.

        The site-URL table is resolved HERE, once, from the site variables, as
        ``browser_env.env_config`` builds it. A missing or empty table aborts
        construction, before the first episode. Both directions read the one
        object: the adapter's URL line and the action agent's mapping.

        Every backbone's reasoning agent gets WebArena's deciding suffix
        (``prompts.py``): the shared terminal-claim contract, unchanged, followed
        by the control's "Homepage" paragraph. The shared default in
        ``src/agents/reasoning/base.py`` is the contract alone, so this text
        cannot reach OSWorld.
        """
        url_mappings = url_mappings_from_env()
        # url_mappings_from_env keeps upstream's assert verbatim, and `python -O`
        # strips it. This check stays.
        empty = [alias for local, alias in url_mappings.items() if not local]
        if empty:
            raise ValueError(
                f"the site-URL table has an empty local URL for {empty}: set REDDIT, SHOPPING, "
                "SHOPPING_ADMIN, GITLAB, WIKIPEDIA, MAP and HOMEPAGE (the repo .env) before building the agent"
            )
        support_set_store = LocalSupportSetStore(
            root_dir=support_set_root, backend="webarena", embedder=get_default_embedder()
        )
        perception_processor = WebArenaObservationAdapter(
            observation_type=observation_type, url_mappings=url_mappings,
        )
        # One retrieval agent for every backbone branch: retrieval-off swaps the
        # whole agent out (the orchestrator requires one, so bypass is an agent,
        # not a branch).
        retrieval_agent: RetrievalAgent = (
            BypassRetrievalAgent() if retrieval_off
            else DeterministicRetrievalAgent(
                support_set_store=support_set_store,
                embedder=get_default_embedder(),
            )
        )
        if backbone_type == "gpt":
            backbone = GPTAgent(model=model, **backbone_kwargs)
            return cls(
                reasoning_agent=GraphReasoningAgent(
                    backbone=backbone, max_trajectory_length=max_trajectory_length,
                    deciding_instruction_suffix=WEBARENA_DECIDING_INSTRUCTION_SUFFIX,
                ),
                retrieval_agent=retrieval_agent,
                action_agent=WebArenaActionAgent(backbone=backbone, url_mappings=url_mappings),
                support_set_store=support_set_store,
                perception_processor=perception_processor,
                max_actions=max_actions,
                two_pass=two_pass,
                enable_completion_gate=enable_completion_gate,
                completion_gate_window=completion_gate_window,
                retrieval_off=retrieval_off,
            )
        if backbone_type == "deepseek":
            # DeepSeek V4 backbone. REUSES the shared reasoning/graph/action agents
            # unchanged (they call complete_json + read json/raw_response; the
            # DeepSeekAgent routes grounding vs reasoning internally — grounding =
            # non-thinking forced function-calling with the grammar pattern,
            # reasoning = thinking + json_object). One shared backbone instance.
            backbone = DeepSeekAgent(model=model, **backbone_kwargs)
            return cls(
                reasoning_agent=GraphReasoningAgent(
                    backbone=backbone, max_trajectory_length=max_trajectory_length,
                    deciding_instruction_suffix=WEBARENA_DECIDING_INSTRUCTION_SUFFIX,
                ),
                retrieval_agent=retrieval_agent,
                action_agent=WebArenaActionAgent(backbone=backbone, url_mappings=url_mappings),
                support_set_store=support_set_store,
                perception_processor=perception_processor,
                max_actions=max_actions,
                two_pass=two_pass,
                enable_completion_gate=enable_completion_gate,
                completion_gate_window=completion_gate_window,
                retrieval_off=retrieval_off,
            )
        if backbone_type == "minimax":
            # MiniMax-M3 backbone. REUSES the shared reasoning/action agents
            # unchanged (they call complete_json + read json/raw_response; the
            # MiniMaxAgent routes grounding vs reasoning internally — grounding =
            # non-thinking forced function-calling with the grammar schema,
            # reasoning = adaptive thinking + json_object). One shared backbone
            # instance. Mirrors the deepseek branch.
            backbone = MiniMaxAgent(model=model, **backbone_kwargs)
            return cls(
                reasoning_agent=GraphReasoningAgent(
                    backbone=backbone, max_trajectory_length=max_trajectory_length,
                    deciding_instruction_suffix=WEBARENA_DECIDING_INSTRUCTION_SUFFIX,
                ),
                retrieval_agent=retrieval_agent,
                action_agent=WebArenaActionAgent(backbone=backbone, url_mappings=url_mappings),
                support_set_store=support_set_store,
                perception_processor=perception_processor,
                max_actions=max_actions,
                two_pass=two_pass,
                enable_completion_gate=enable_completion_gate,
                completion_gate_window=completion_gate_window,
                retrieval_off=retrieval_off,
            )
        if backbone_type == "qwen":
            # Qwen (qwen3.7-plus over OpenRouter) backbone. REUSES the shared
            # reasoning/action agents unchanged (they call complete_json + read
            # json/raw_response; the QwenAgent routes grounding vs reasoning
            # internally — grounding = reasoning {"effort": "none"} + forced
            # function-calling with the grammar schema, reasoning =
            # reasoning {"enabled": true} + json_object). One shared backbone
            # instance. Mirrors the minimax branch.
            backbone = QwenAgent(model=model, **backbone_kwargs)
            return cls(
                reasoning_agent=GraphReasoningAgent(
                    backbone=backbone, max_trajectory_length=max_trajectory_length,
                    deciding_instruction_suffix=WEBARENA_DECIDING_INSTRUCTION_SUFFIX,
                ),
                retrieval_agent=retrieval_agent,
                action_agent=WebArenaActionAgent(backbone=backbone, url_mappings=url_mappings),
                support_set_store=support_set_store,
                perception_processor=perception_processor,
                max_actions=max_actions,
                two_pass=two_pass,
                enable_completion_gate=enable_completion_gate,
                completion_gate_window=completion_gate_window,
                retrieval_off=retrieval_off,
            )
        if backbone_type == "claude":
            # Claude Sonnet 5 backbone (first-party Messages API). REUSES the shared
            # reasoning/action agents unchanged (they call complete_json + read
            # json/raw_response; the ClaudeAgent routes grounding vs reasoning
            # internally — grounding = thinking disabled + a forced tool call with
            # the grammar schema, reasoning = adaptive thinking + effort with the
            # schema as an advisory hint). One shared backbone instance. Mirrors
            # the qwen branch.
            backbone = ClaudeAgent(model=model, **backbone_kwargs)
            return cls(
                reasoning_agent=GraphReasoningAgent(
                    backbone=backbone, max_trajectory_length=max_trajectory_length,
                    deciding_instruction_suffix=WEBARENA_DECIDING_INSTRUCTION_SUFFIX,
                ),
                retrieval_agent=retrieval_agent,
                action_agent=WebArenaActionAgent(backbone=backbone, url_mappings=url_mappings),
                support_set_store=support_set_store,
                perception_processor=perception_processor,
                max_actions=max_actions,
                two_pass=two_pass,
                enable_completion_gate=enable_completion_gate,
                completion_gate_window=completion_gate_window,
                retrieval_off=retrieval_off,
            )
        raise NotImplementedError(f"Backbone type {backbone_type} not supported")

    # ------------------------------------------------------------------
    # WebArena ``Agent`` contract — the outward boundary.
    # ------------------------------------------------------------------

    def reset(self, test_config_file: str | None = None) -> None:
        self.runtime_state = None
        self.last_decision = None
        self.last_update = None
        self.task_instruction = None
        self._pending_env_feedback = None
        self.action_parse_failure_count = 0
        self.reasoning_parse_failure_count = 0
        self.grounding_parse_failure_count = 0
        self.reasoning_resample_count = 0
        self.grounding_resample_count = 0
        self.resample_exhausted_reasoning = 0
        self.resample_exhausted_grounding = 0
        self._step_resamples = (0, 0)
        self.task_id = test_config_file  # opaque id; runner-supplied per task
        for backbone_holder in (self.reasoning_agent, self.action_agent):
            backbone = getattr(backbone_holder, "backbone", None)
            if backbone is not None and hasattr(backbone, "reset_session"):
                backbone.reset_session()

    def next_action(self, trajectory: Any, intent: str, meta_data: Any = None) -> Any:
        """One step of the WebArena run loop:

        1. If a prior decision is outstanding, close it via ``update_runtime``
           against the latest state in the trajectory (the env executed our
           prior action and produced this state).
        2. Build an observation dict from ``trajectory[-1]`` (a ``StateInfo``)
           and call the stable inner boundary's ``decide_next_action``.
        3. Translate the resulting ``ActionBatch.actions[0]`` to a WebArena
           ``Action`` via the factories.

        Holds the decision + runtime state on ``self`` so the next call can
        close it. The WebArena ``Trajectory`` is reserved for WebArena state
        and actions only — none of our internal artifacts ride on it.
        """
        self.task_instruction = intent
        current_state_info = self._latest_state_info(trajectory)
        current_obs_dict = self._state_info_to_obs_dict(current_state_info, meta_data=meta_data)

        # Surface a prior-step failure upstream (the WebArena analog of OSWorld's
        # observe() _env_feedback): the adapter pops _env_feedback into
        # Observation.metadata["env_feedback"], where the monitor's update pass
        # reads the info dict and reports it as observed_change. Consumed once,
        # then cleared. The info dict's key distinguishes the failure kind.
        if self._pending_env_feedback is not None:
            current_obs_dict["_env_feedback"] = {
                "reward": 0,
                "done": False,
                "info": dict(self._pending_env_feedback),
            }
            self._pending_env_feedback = None

        # (1) close prior step.
        if self.last_decision is not None:
            self.last_update = self.orchestrator.update_runtime(
                decision=self.last_decision,
                current_observation_input=current_obs_dict,
            )
            self.runtime_state = self.last_update.runtime_state

        # (2) decide.
        self.last_decision = self.orchestrator.decide_next_action(
            instruction=intent,
            observation_input=current_obs_dict,
            runtime_state=self.runtime_state,
        )
        # decide_next_action returned a fresh state for this step; carry it
        # forward so reset/replay don't drop the trajectory accumulation.
        self.runtime_state = self.last_decision.runtime_state
        self._step_resamples = (
            resample_attempts_of(self.last_decision.reasoning),
            resample_attempts_of(self.last_decision.action_batch),
        )
        self.reasoning_resample_count += self._step_resamples[0]
        self.grounding_resample_count += self._step_resamples[1]
        # Every re-sample was an unparseable sample: the per-attempt failure
        # counters advance by the same amount (an exhausted step adds its final
        # failed attempt at its degrade marker below).
        self.reasoning_parse_failure_count += self._step_resamples[0]
        self.grounding_parse_failure_count += self._step_resamples[1]
        # Recorded before any degrade return below, so every step has its record.
        self._record_url_telemetry()

        # Reasoning soft-degrade: an unparseable DECIDING response returns an
        # empty decision carrying this marker. Surface it upstream on the NEXT step
        # via the same _env_feedback channel as a grounding failure, and emit a
        # noop this step so one malformed response does not fail the episode.
        reasoning_msg = self.last_decision.reasoning
        if reasoning_msg is not None and reasoning_msg.content.get("reasoning_parse_error"):
            # Surfaced under a DISTINCT info key so
            # the monitor describes it as a reasoning failure, not "action failed to ground".
            self._pending_env_feedback = {"reasoning_parse_error": reasoning_msg.content["reasoning_parse_error"]}
            self.reasoning_parse_failure_count += 1
            self.resample_exhausted_reasoning += 1
            return self._noop_action("reasoning_parse_error")

        # (3) translate the first action of the batch to a WebArena Action.
        action_batch = self.last_decision.action_batch

        # Completion-gate corrective block (Mechanism 2): a done claim lacking
        # grounded evidence is blocked → emit a noop this step (NOT a give-up stop,
        # so a possibly-valid answer survives) and surface which rule failed upstream
        # so the next reasoning step gathers the missing evidence. The gate's
        # 2-strikes escape (in the orchestrator) lets a persistent done through.
        if action_batch.content.get("completion_gate_blocked"):
            failed_rule = action_batch.content.get("completion_gate_failed_rule") or "no_grounded_evidence"
            self._pending_env_feedback = {"completion_gate_blocked": failed_rule}
            return self._noop_action("completion_gate_blocked")

        # Grounding-parse soft-degrade: the action agent's complete_json hit
        # parse exhaustion (unparseable grounding response) and returned an
        # empty batch carrying this marker (see action_agent.py).
        # Counted noop + DISTINCT upstream signal — the action-side mirror of
        # the reasoning soft-degrade, so a malformed grounding response costs a
        # step, never the episode. Checked BEFORE the generic empty-batch
        # defensive path so the failure is counted, not silently absorbed.
        if action_batch.content.get("grounding_parse_error"):
            self._pending_env_feedback = {
                "grounding_parse_error": action_batch.content["grounding_parse_error"]
            }
            self.grounding_parse_failure_count += 1
            self.resample_exhausted_grounding += 1
            return self._noop_action("grounding_parse_error")

        if not action_batch.actions:
            # Defensive: orchestrator always emits ≥1 action, but never crash
            # the runner — emit a noop.
            return self._noop_action("empty_action_batch")

        # N/A-answer adapter parity:
        # when the terminal answer is a PURE infeasibility declaration, submit
        # WebArena's literal "N/A" dialect instead. Adapter layer only — the
        # contract text is untouched, so no model-facing give-up cue exists.
        # The answer is read with the SAME shared parser the completion
        # gate and translator use (no second parser); an already-exact "N/A"
        # passes through untouched with no event (identity skip); every actual
        # rewrite is audited on _na_normalized_events. The rewrite targets
        # parameters["command"] — the first field stop_command() reads — so the
        # translator needs no change. Conversion is coupled to auditability:
        # no runtime_state, no rewrite (never a silent unaudited conversion).
        first_action = action_batch.actions[0]
        if (first_action.action_type or "").lower() == "done":
            answer = done_answer(first_action)
            na_rule = match_infeasible_declaration(answer)
            if na_rule is not None and answer != NA_ANSWER and self.runtime_state is not None:
                first_action = deepcopy(first_action)
                if not isinstance(first_action.parameters, dict):
                    first_action.parameters = {}
                first_action.parameters["command"] = f"stop [{NA_ANSWER}]"
                self.runtime_state.content.setdefault("_na_normalized_events", []).append(
                    {"task_id": self.task_id, "original_answer": answer, "rule": na_rule}
                )
            # _terminal_answer rider: record the SUBMITTED terminal answer
            # (post-normalization) on every done that reaches translation —
            # without it the record holds no terminal answer, and the recall /
            # precision of submitted answers is unmeasurable. The
            # answer selection MIRRORS translate_action's done branch (shared
            # parser for command-bearing dones; parameters['answer'] fallback
            # for command-less dry-run/OSWorld-style dones) so the record can
            # never disagree with what create_stop_action receives. Absence of
            # the key = the episode never submitted a done (cap-exit etc.).
            if self.runtime_state is not None:
                if stop_command(first_action):
                    submitted_answer = done_answer(first_action)
                elif isinstance(first_action.parameters, dict):
                    submitted_answer = str(first_action.parameters.get("answer") or "")
                else:
                    submitted_answer = ""
                self.runtime_state.content["_terminal_answer"] = submitted_answer

        # NOTE (DEFERRED): on this degrade the un-translatable decision
        # stays as last_decision, so the NEXT update_runtime records the *intended*
        # (un-executed) action into recent_history/graph as if it ran and the
        # verifier closes it. Control flow is fine (episode continues, monitor sees
        # action_parse_error); only the effect-class reliability audit's telemetry
        # slightly overstates what the agent actually did. Left until that audit's
        # fidelity is being tightened.
        #
        # An action string WebArena's factory rejects is a grounding failure,
        # not a runner crash: degrade to a noop (episode CONTINUES) and signal
        # the failure upstream on the NEXT step via _env_feedback — the reasoning
        # loop is our recovery path, so no blind LLM retry (decomposition tax).
        # The factory's exception surface is broader than ActionParsingError
        # (e.g. type text with a non-ASCII char -> KeyError from _keys2ids), so
        # we degrade on ANY exception whose origin is env-side — ActionParsingError
        # OR a traceback originating in browser_env/actions.py — and re-raise
        # everything else, keeping OUR translator bugs loud.
        parse_error_cls = load_action_parsing_error()
        try:
            webarena_action = translate_action(first_action)
        except Exception as exc:  # noqa: BLE001
            is_env_side = (
                (parse_error_cls is not None and isinstance(exc, parse_error_cls))
                or _traceback_originates_in_webarena_actions(exc)
            )
            if not is_env_side:
                raise
            # Include the exception TYPE in the upstream signal so the model
            # learns the cause (e.g. "KeyError: '’'" -> pick an ASCII spelling).
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("action grounding failure (%s); degrading to noop", reason)
            self._pending_env_feedback = {"action_parse_error": reason}
            self.action_parse_failure_count += 1
            return self._noop_action("action_parse_error")
        # Recorded only after the factory accepted the string, so the per-step
        # action record reflects what WebArena actually received (a rejected
        # string is recorded by the degrade path above as a noop instead).
        self._record_executed_action(first_action, status="submitted")
        return webarena_action

    def _record_url_telemetry(self) -> None:
        """Page-URL telemetry. It records this step's page URL, both raw and as the
        deciding view's ``URL:`` line shows it, in ``_page_url_history``. The record
        also carries the sha256 of the raw tree text before the URL line, so a URL-only
        transition (same tree, different URL) can be counted exactly. It also
        records every alias the grounding reply wrote that the action agent mapped
        back to local, in ``_url_mapping_events``.

        Both lists live on ``runtime_state.content``: episode telemetry, which the
        prompt builders' key allow-lists never read. Neither goes on
        ``Action.metadata``, which history carries into the next prompt.

        The adapter sets the URL pair on every observation or raises
        ``MissingPageUrlError``. The pair is absent only when an injected
        orchestrator never ran the adapter (the dry run, a stub orchestrator)."""
        decision, state = self.last_decision, self.runtime_state
        observation_metadata = decision.observation.metadata or {}
        if "page_url" in observation_metadata:
            state.content.setdefault("_page_url_history", []).append({
                "step_idx": state.step_idx,
                "page_url": observation_metadata["page_url"],
                "displayed_url": observation_metadata["displayed_url"],
                "tree_sha256": observation_metadata["tree_sha256"],
            })
        for event in decision.action_batch.metadata.get("url_mapping_events") or []:
            state.content.setdefault("_url_mapping_events", []).append({"step_idx": state.step_idx, **event})

    def _noop_action(self, reason: str) -> Any:
        """The defensive noop WebArena action. Used both when the orchestrator
        emits an empty batch and when a grounding failure degrades the step —
        the episode continues and the failure is signaled upstream next step.
        ``reason`` names which of those paths fired, for the per-step action
        record (it is telemetry only; nothing reads it back)."""
        action = InternalAction(action_type="wait", parameters={}, raw={"input": {"command": "noop"}})
        self._record_executed_action(action, status="noop", reason=reason)
        return translate_action(action)

    def _record_executed_action(
        self, action: InternalAction, *, status: str, reason: Optional[str] = None
    ) -> None:
        """Append this step's concrete WebArena action string to
        ``_executed_actions`` on the runtime state (episode telemetry; the runner's
        dump carries every runtime_content key). Purely additive; it is what
        makes loop-rate / repeat-action comparisons against the baseline
        computable.
        Recorded HERE because this is the boundary where the string that WebArena
        will actually receive is in hand (the internal ``Action`` still holds it;
        the translated WebArena action is an opaque TypedDict)."""
        if self.runtime_state is None:
            return
        record: dict[str, Any] = {
            "step_idx": self.runtime_state.step_idx,
            "action_type": action.action_type,
            "command": _action_command(action),
            "status": status,
        }
        if reason is not None:
            record["reason"] = reason
        # Re-samples this step spent before the recorded outcome (absent = 0).
        if self._step_resamples[0]:
            record["reasoning_resamples"] = self._step_resamples[0]
        if self._step_resamples[1]:
            record["grounding_resamples"] = self._step_resamples[1]
        self.runtime_state.content.setdefault("_executed_actions", []).append(record)

    # ------------------------------------------------------------------
    # Persistence — runner calls this after evaluation.
    # ------------------------------------------------------------------

    def finalize_task(
        self,
        *,
        task_id: str,
        success: bool,
        result: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Persist the trajectory + graph artifacts to the support set when the
        task succeeded. Mirrors ``OSWorldGraphPCAgent.finalize_task``."""
        if self.retrieval_off:
            # The retrieval-off arm: no store write, and say so on the episode
            # record rather than skipping silently.
            if self.runtime_state is not None:
                self.runtime_state.content["_store_write_skipped"] = "retrieval_off"
            logger.info("store write skipped for %s (retrieval_off)", task_id)
            return None
        if not success or self.runtime_state is None:
            return None
        # Episode-close relabel write-back (design.md §6): reconcile the graph's
        # outcome labels (recent_history's actual_effect_class) against the
        # episode's M1-reconciled verdicts BEFORE persistence — a conservative
        # per-step support check so the support set stores externally-grounded
        # effect labels, not the model's raw self-reports. Mutates the
        # history in place BEFORE it is deepcopied into the entry below, so the
        # persisted recipe log carries the corrected labels; the relabel ledger
        # rides on _relabel_events (telemetry). Applied at episode close so in-run
        # prompts only see corrections via retrieved experience.
        relabel_events = reconcile_effect_labels(
            self.runtime_state.recent_history,
            self.runtime_state.content.get("_verification_verdicts") or [],
        )
        if relabel_events:
            self.runtime_state.content["_relabel_events"] = relabel_events
        entry: dict[str, Any] = {
            "task_id": task_id,
            "instruction": self.task_instruction,
            "success": True,
            "result": result,
            "current_stage": self.runtime_state.current_stage,
            "plan": self._serialize_plan(self.runtime_state.plan),
            "recent_history": deepcopy(self.runtime_state.recent_history),
            "observation": self._build_support_observation_payload(),
            "metadata": deepcopy(metadata or {}),
        }
        graph_artifacts = self._build_graph_artifacts()
        if graph_artifacts:
            entry["graph_artifacts"] = graph_artifacts
        return self.support_set_store.add_success_entry(entry)

    def _build_support_observation_payload(self) -> dict[str, Any]:
        """Compact-observation text to embed candidate-side for the retrieval
        text arm. Mirrors OSWorld's payload shape so ``_entry_observation_text``
        in ``support_set.py`` reads both backends the same way."""
        if self.last_update is None:
            return {}
        content = self.last_update.current_observation.content
        if not isinstance(content, dict):
            return {}
        return {"observation_text": content.get("observation_text", "")}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_state_info(trajectory: Any) -> Any:
        """``trajectory[-1]`` may be either a ``StateInfo`` (dict-like with
        ``observation``/``info`` keys) or an ``Action`` (when the runner
        appends an action before the next state). Walk back to the most recent
        state-shaped entry to be robust to either calling convention.
        """
        if trajectory is None:
            return {}
        try:
            iterator = list(trajectory)
        except TypeError:
            return {}
        for entry in reversed(iterator):
            if isinstance(entry, dict) and "observation" in entry:
                return entry
        return iterator[-1] if iterator else {}

    @staticmethod
    def _state_info_to_obs_dict(state_info: Any, *, meta_data: Any = None) -> dict[str, Any]:
        """Flatten ``StateInfo {observation, info}`` into the observation dict
        the orchestrator + perception adapter expect. Unknown extra fields
        survive on ``raw`` so nothing is lost.
        """
        if not isinstance(state_info, dict):
            return {}
        observation = state_info.get("observation") or {}
        info = state_info.get("info") or {}
        payload: dict[str, Any] = {}
        if isinstance(observation, dict):
            for key in ("text", "image", "html"):
                if key in observation:
                    payload[key] = observation[key]
        else:
            # observation can be a single ndarray (image-only mode).
            payload["image"] = observation
        if info:
            payload["info"] = info
            # observation_metadata is carried through as cheap benign context;
            # it does NOT feed an id-validator (none is wired — see a11y_parser).
            obs_meta = info.get("observation_metadata") if isinstance(info, dict) else None
            if obs_meta:
                payload["observation_metadata"] = obs_meta
        if meta_data:
            payload["meta_data"] = meta_data
        return payload

    @staticmethod
    def _serialize_plan(plan) -> Optional[dict[str, Any]]:
        if plan is None:
            return None
        return {
            "goal": plan.goal,
            "current_stage": plan.current_stage,
            "stages": deepcopy(plan.stages),
            "content": deepcopy(plan.content),
        }

    def _build_graph_artifacts(self) -> dict[str, Any]:
        """``graph_summary`` is the only graph artifact the run produces: the
        reasoning call emits no node/edge topology (design.md §3)."""
        if self.runtime_state is None or "graph_summary" not in self.runtime_state.content:
            return {}
        return {"graph_summary": deepcopy(self.runtime_state.content["graph_summary"])}
