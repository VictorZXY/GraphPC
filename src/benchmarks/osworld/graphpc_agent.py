"""OSWorld-V2-facing GraphPC agent (the outer benchmark boundary).

Translation layer — GraphPC-shaped inside, OSWorld-V2-shaped outside:

  Inward (toward our orchestrator): calls ONLY the stable inner boundary
  (``decide_next_action`` / ``update_runtime``) over the schema types. The
  decision for step k is closed by ``update_runtime`` at the START of
  ``predict`` k+1 with the new observation (the same pattern as the WebArena
  adapter — the V2 driver exposes no post-step hook to the agent), so the
  monitor / verification / graph write-back all run inside the driver's own
  loop without any driver modification.

  Outward (toward OSWorld-V2's ``lib_run_single.run_single_example``, used
  UNMODIFIED): the M3 agent contract —
  ``reset(runtime_logger=None, **kw)``; ``predict(instruction, obs) ->
  (response_text, actions)`` where ``actions`` is a ONE-element list holding
  a pyautogui source string or one of the literal env tokens ``"DONE"`` /
  ``"FAIL"`` / ``"WAIT"`` — never empty, except on a profile with an ask-user
  channel (Qwen): a decided ``call_user`` returns ``(question, [])``, the
  driver's ASK_USER path, and the answer arrives on the next
  ``obs["user_response"]``; the driver may
  ``setattr(agent, "task_current_date", ...)`` (consumed into the environment
  brief); the runner calls ``set_api_log_dir(path)`` before each task and
  ``finalize_task(...)`` after the episode (our post-episode hook in the
  vendored runner — the driver itself never calls the agent after the loop).
  Runner call order per task: set_api_log_dir -> driver setattr -> reset, so
  ``reset`` clears per-episode state only.

One action per decision (``max_actions=1``). Degrade contract: an
unparseable reasoning / grounding response, an empty batch, a blocked
completion claim, or an action the translator rejects each becomes the
env-accepted no-op ``"WAIT"`` (counts a driver step) with the failure surfaced
to the next reasoning call through the monitor feedback channel
(``_env_feedback``), exactly as the WebArena adapter does with its noop.

Infeasibility (the give-up route is set by the terminal contract in ``prompts.py``):
the deciding contract's give-up channel is the ``fail`` action. The action
agent hands it over as a ``done`` carrying ``terminal_intent: "fail"``, so the
completion gate evaluates it as it evaluates a success claim; once admitted it
is emitted as the profile's give-up token (V2's ``FAIL``; ``DONE`` on Qwen, whose
control never emits ``FAIL``). A ``done`` whose carried text is nonetheless
an infeasibility declaration is still emitted as that token at this boundary —
a fallback the prompt does not offer as a route — audited on
``_infeasible_normalized_events``.
"""

from __future__ import annotations

import dataclasses
import datetime
import gzip
import json
import logging
import math
import os
import posixpath

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from src.agents.common.infeasible_declaration import match_infeasible_declaration
from src.agents.common.resample import resample_attempts_of
from src.agents.common.schema import Action, GraphPCDecision, GraphPCRuntimeUpdate, RuntimeState
from src.agents.graphpc.orchestrator import reconcile_effect_labels
from src.agents.monitor.heuristic import HeuristicMonitorAgent, ScreenshotDiffConfig
from src.agents.retrieval.base import RetrievalAgent
from src.agents.retrieval.bypass import BypassRetrievalAgent
from src.agents.retrieval.deterministic import DeterministicRetrievalAgent, InjectedHistoryDiet
from src.agents.retrieval.embedding import get_default_embedder
from src.agents.retrieval.relevance import (
    PREREGISTERED_RELEVANCE_PERCENTILE,
    PercentileRelevanceGate,
    RelevanceCalibrationError,
)
from src.agents.retrieval.support_set import LocalSupportSetStore

from .action_minimax import (
    M3_ACTION_CLASS_VOCABULARY,
    MiniMaxActionAgent,
    normalize_action_class,
    strip_grounding_coordinates,
)
from .m3_action_translator import pointer_xy_from_pyautogui, translate_m3_action
from .observation_adapter import (
    DEFAULT_CLIENT_PASSWORD,
    TASK_CURRENT_DATE_KEY,
    OSWorldObservationAdapter,
    environment_brief,
    screenshot_size,
)
from .orchestrator import (
    FAIL_RESTS_ON_STEP_BUDGET, NAMED_APP_BYPASSED_BY_COMMAND_LINE, OSWORLD_COMPLETION_GATE_RULES,
    REQUIREMENTS_STRIKES, REQUIREMENTS_UNVERIFIED, SYNTHETIC_MEDIA_DELIVERABLE, UNVERIFIED_BARRIER,
    USE_FILE_CLAIMED_DONE, OSWorldOrchestrator,
)
from .prompts import OSWORLD_DECIDING_INSTRUCTION_SUFFIX, OSWORLD_REQUIREMENTS_INSTRUCTION, OSWORLD_TYPING_NOTE
from .reasoning import DECISION_MISSING_REASON, REQUIREMENTS_SCHEMA, OSWorldMiniMaxAgent, OSWorldReasoningAgent
# The image-retention arithmetic and the timeline's text bounds live with the renderer;
# this module records them in `active_config`.
from .timeline import (
    EXECUTED_TEXT_LIMIT,
    FACT_ENTRY_LIMIT,
    FACT_SOURCE_LIMIT,
    FACTS_RENDER_HEAD,
    FACTS_RENDER_LIMIT,
    FACTS_RENDER_TAIL,
    M3_IMAGE_RETENTION,
    OUT_OF_WINDOW_LINE_LIMIT,
    REASONING_TEXT_LIMIT,
    REQUIREMENT_TEXT_LIMIT,
    REQUIREMENTS_RENDER_ITEMS,
    EpisodeTimeline,
    ImageRetention,
    ScreenshotCaptureFailed,
)


logger = logging.getLogger("desktopenv.agent")

# The env tokens ``DesktopEnv.step`` recognises without executing code.
DONE_TOKEN = "DONE"
FAIL_TOKEN = "FAIL"
WAIT_TOKEN = "WAIT"

# The ask channel (a profile's ``ask_user_channel``): the action class a question is
# decided as, and the executed-action record's command for it — the label V2's driver
# writes on its ASK_USER trajectory row.
ASK_USER_ACTION = "call_user"
ASK_USER_COMMAND = "ASK_USER"

TELEMETRY_FILENAME = "graphpc_telemetry.json"
# Per-model-call diagnostics (parity with the baseline's api_logs/): one JSON per
# call under <task result dir>/graphpc_calls/NNN_<stage>.json while a task's
# api_log_dir is set, and beside it NNN_<stage>.request.json.gz — the request's
# text parts as sent, each image replaced by its slot. Text only — never the
# screenshot bytes; a slot's sha1 is that of the step PNG the driver saved.
CALL_LOG_DIRNAME = "graphpc_calls"

# Sampling temperature of the GROUNDING route (non-thinking) on this benchmark,
# for the MiniMax profile (V2's native <tool_call> text format:
# ``complete_text(grounding=True)``) and the DeepSeek profile, which passes the
# same value — the OSWorld instantiation: the V2 M3
# runner's non-thinking default ``--temperature`` (0.6). NOT a parity pin: the baseline arm runs
# ``--thinking_mode adaptive``, under which V2's M3Agent forces temperature 1.
# Pinned here, not exposed on the runner; the deciding (thinking) route never
# receives it (see the backbone module docstring).
GROUNDING_TEMPERATURE = 0.6

# Keys a done action may carry its terminal statement under (the action
# agent's parameters / raw input, then the reasoning's intended action), in
# precedence order. ``intent`` is LAST: it is consulted only when every text
# key is empty, so a substantive done (answer text present) can never be
# converted through its intent; the reasoning contract has no dedicated field
# for the "brief statement", and a model may put it there.
_TERMINAL_TEXT_KEYS = ("text", "answer", "message", "statement", "value", "intent")

# ----------------------------------------------------------------------
# The benchmark-scoped settings this adapter hands to shared components.
#
# Each has a shared default that is WebArena's behaviour, and naming the OSWorld
# value here is what keeps it out of a WebArena run BY CONSTRUCTION — not by an
# accident of episode length, observation type or record format (design.md §10).
# They are constants rather than runner flags on purpose:
# they are part of what this benchmark's configuration is, and a flag would let
# a scored run differ from the run that was smoked.
# ----------------------------------------------------------------------

# Total attempts per model call, deciding and grounding alike (1 = no re-sample).
# 3 is PARITY with this benchmark's own control: V2's M3 agent loops
# `range(self.max_llm_retries + 1)` with M3_MAX_LLM_RETRIES defaulting to 2
# (mm_agents/m3/agent.py:283, :647). The shared default is 1, which is WebArena's
# control behaviour — so the same budget that is parity here would be a
# threefold advantage there, which is exactly why it is not a shared constant.
MODEL_CALL_ATTEMPTS = 3

# Within-step compression of an injected exemplar (the injection path only; the
# store keeps the full record). `recent_history` is most of an injected
# exemplar and these three rows are most of it — the verbatim model reply whose
# parsed action already sits beside it, the per-step narration, and the loop
# bookkeeping. Every step survives; only what each step SAYS gets shorter. This
# is the injection diet's own "recipe, not diary" rule applied one level down,
# and it matters here because a 500-step budget makes a stored winner an order of
# magnitude longer than a WebArena one.
OSWORLD_INJECTED_HISTORY_DIET = InjectedHistoryDiet(
    drop_action_raw_response=True,
    drop_reasoning_text=True,
    drop_bookkeeping_keys=True,
)

# How many of a step's intended_actions survive into `recent_history` (the shared
# default is None = the unbounded copy WebArena keeps). 1 is
# `max_actions` on this benchmark, i.e. exactly what was DECIDED and executed: the
# action stage grounds `intended[:max_actions]` and the model re-emits a fresh plan
# next step, so actions 2..N are a superseded speculation that would take most of a
# history item. See common/compaction.py:_bound_intended_actions for the consumer audit.
OSWORLD_INTENDED_ACTIONS_CAP = 1

# The episode step budget the pre-registration declares and the launch script passes as
# --max_steps (osworld_cell_config.sh MAX_STEPS). The orchestrator uses it only to place
# its give-up budget rule (orchestrator.FAIL_BUDGET_RULE_BEFORE). No MiniMax or Qwen
# request shows it; on the DeepSeek profile the reasoning agent's budget line states it at
# every step (the profile's step_budget_line). A smoke launched with a smaller MAX_STEPS
# still places the rule against 500 (and the budget line states 500).
OSWORLD_MAX_STEPS = 500
# Each GraphPC arm's budget, by backbone: what runner delta 15 holds the launch to and what the
# orchestrator places the rule on. The Qwen GraphPC arm runs an 800-step budget with the stuck-stop: the Qwen
# control executes several tool calls per step (its --batch_actions default), so within its 500 steps it can carry
# out more than 500 real actions, while a GraphPC step executes one. Every other arm, and every control, runs 500.
# The launch preflight holds each value to the pre-registration (its method surface's max_steps,
# method_surface_qwen.max_steps for Qwen). The controls are not built here.
OSWORLD_MAX_STEPS_BY_BACKBONE: dict[str, int] = {"minimax": OSWORLD_MAX_STEPS, "qwen": 800, "deepseek": OSWORLD_MAX_STEPS}
# THE STUCK-STOP, by backbone like the budget: the Qwen GraphPC arm only, at its 800-step budget; MiniMax and
# DeepSeek have none. Once more than `min_step` predict calls of a phase have been made, it ends the episode
# with the profile's give-up token, without a model call, when the episode spins in place (stuck_stop_fire): `repeat`, one executed command at least K times among
# the last `window` executed steps with at most `max_changed` of them changing the screen; or `wait`, `run`
# consecutive transparent WAITs with at most `max_changed` changing it. "Changing" is the live monitor's masked
# diff at the monitor's own floor. The transparent steps (a decided wait, a grounding_class_mismatch exhaustion
# WAIT) are skipped by `repeat`; every other step counts, a degrade WAIT as the command "WAIT". The value is the
# run's active_config key `stuck_stop` and method_surface_qwen.stuck_stop in the pre-registration.
OSWORLD_STUCK_STOP_BY_BACKBONE: dict[str, Optional[dict[str, Any]]] = {
    "minimax": None,
    "qwen": {"repeat": {"K": 50, "window": 55, "max_changed": 11}, "wait": {"run": 50, "max_changed": 10},
             "diff_floor_px": ScreenshotDiffConfig().changed_min_pixels, "min_step": 500,
             "transparent": ["decided_wait", "grounding_class_mismatch"]},
    "deepseek": None,
}
STUCK_STOP_REASON = "stuck_stop"


def stuck_step(row: dict[str, Any], diff_pixel_count: Optional[int]) -> dict[str, Any]:
    """An executed-action row as the stuck-stop reads it: its command, whether it is transparent (a decided wait,
    or a grounding_class_mismatch exhaustion WAIT), and its masked diff (None: unobserved)."""
    transparent = ((row.get("action_type") == "wait" and row.get("status") == "submitted")
                   or row.get("reason") == "grounding_class_mismatch")
    return {"command": row.get("command"), "transparent": transparent, "diff_pixel_count": diff_pixel_count}


def stuck_stop_fire(history: Sequence[dict[str, Any]], predict_calls: int,
                    rule: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The stuck-stop's verdict at predict call ``predict_calls`` over the phase's steps so far, in order
    (``stuck_step``): the clause that holds, with its window counts, or None. A missing diff reads as unchanged."""
    if predict_calls <= rule["min_step"]:
        return None

    def changed(step: dict[str, Any]) -> bool:
        return (step["diff_pixel_count"] or 0) >= rule["diff_floor_px"]

    repeat = rule["repeat"]
    window = [step for step in history if not step["transparent"]][-repeat["window"]:]
    if window:
        command, identical = Counter(step["command"] for step in window).most_common(1)[0]
        moved = sum(changed(step) for step in window)
        if identical >= repeat["K"] and moved <= repeat["max_changed"]:
            return {"clause": "repeat", "command": command, "identical": identical, "changed": moved,
                    "window": len(window)}
    wait = rule["wait"]
    run = list(history[-wait["run"]:])
    if len(run) == wait["run"] and all(step["transparent"] for step in run):
        moved = sum(changed(step) for step in run)
        if moved <= wait["max_changed"]:
            return {"clause": "wait", "run": len(run), "changed": moved}
    return None


# The requirements_unverified block's parenthesis on the blocked step's line (orchestrator.py: the rule and the fact it
# records). Decider-visible method surface of the DeepSeek profile: the ids and texts of the items not admitted, each
# with why, the first REQUIREMENTS_BLOCK_ITEMS of them named with their text clipped to REQUIREMENTS_BLOCK_TEXT_LIMIT,
# and how many more there are.
REQUIREMENTS_BLOCK_ITEMS = 5
REQUIREMENTS_BLOCK_TEXT_LIMIT = 60
REQUIREMENTS_MISSING_TEXT = "(the reply carries no requirement ledger)"
REQUIREMENTS_EMPTY_TEXT = "(the requirement ledger is empty)"
REQUIREMENT_PROBLEM_TEXTS = {
    "malformed": "not a well-formed item (an id and a status of open, verified or unmet)",
    "open": "open",
    "dropped": "dropped from the ledger",
    "no_reason": "unmet without a reason",
    "no_evidence_step": "it has no evidence_step",
    "evidence_step": "its evidence_step {step} is not an executed step before this one",
    "stale_evidence": "its evidence_step {step} is before step {last_open}, where you last listed it as open",
}


def requirements_block_text(fact: dict[str, Any]) -> str:
    """The parenthesis a requirements_unverified block's line carries, from the fact the rule recorded."""
    if fact.get("ledger") == "missing":
        return REQUIREMENTS_MISSING_TEXT
    if fact.get("ledger") == "empty":
        return REQUIREMENTS_EMPTY_TEXT
    problems = fact.get("problems") or []
    named = []
    for problem in problems[:REQUIREMENTS_BLOCK_ITEMS]:
        text = " ".join(str(problem.get("requirement") or "").split())
        if len(text) > REQUIREMENTS_BLOCK_TEXT_LIMIT:
            text = text[: REQUIREMENTS_BLOCK_TEXT_LIMIT - 1] + "…"
        why = REQUIREMENT_PROBLEM_TEXTS.get(problem.get("why"), str(problem.get("why"))).format(
            step=problem.get("evidence_step"), last_open=problem.get("last_open"))
        named.append(f"{problem.get('id')} {json.dumps(text, ensure_ascii=False)}: {why}" if text
                     else f"{problem.get('id')}: {why}")
    more = (f"; and {len(problems) - REQUIREMENTS_BLOCK_ITEMS} more"
            if len(problems) > REQUIREMENTS_BLOCK_ITEMS else "")
    return (f"({len(problems)} of your requirements {'is' if len(problems) == 1 else 'are'} not verified: "
            f"{'; '.join(named)}{more})")


def infeasible_declaration_rule(statement: str) -> Optional[str]:
    """The OSWorld variant of the infeasibility matcher: the rule a `done` statement matches
    as a give-up, or None. Scoped because WebArena deliberately rejects declaration+reason
    forms (see infeasible_declaration.py). Its two readers are ``predict``'s done→FAIL
    conversion and the orchestrator's give-up rules, fail_rests_on_step_budget and
    unverified_barrier (handed over by build_default), so both test a `done` by the one
    setting."""
    return match_infeasible_declaration(statement, allow_trailing_explanation=True)


# ----------------------------------------------------------------------
# The model profile: everything on this benchmark that varies with the model.
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OSWorldModelProfile:
    """What varies with the model on this benchmark. Each model's control is its own
    upstream agent, so a profile follows that agent; the rest
    of this adapter (the ``predict`` flow and telemetry, the timeline, the facts
    ledger, the per-call record, the give-up contract) is the same for every model.
    The typing note, the completion gate's rules and the grounding line's coordinate
    strip vary only through the profile (``typing_note``; ``requirement_ledger``,
    which adds the requirements_unverified rule; ``action_agent_class``, whose
    agent strips coordinates its own way).

    TWO PER-MODEL SETTINGS SIT OUTSIDE THE PROFILE, as backbone-keyed module tables:
    ``OSWORLD_MAX_STEPS_BY_BACKBONE`` (the episode step budget: 800 for the Qwen
    GraphPC arm, 500 otherwise) and ``OSWORLD_STUCK_STOP_BY_BACKBONE`` (the Qwen
    GraphPC arm's stuck-stop; None otherwise). The budget is read where no profile
    is built: the vendored runner imports the table to hold ``--max_steps`` to it
    (delta 15), and the launch preflight and ``verify_runner_deltas.py`` read it from
    this file's source. The stuck-stop is keyed the same way, beside it.

    AS CLASSES, where the variation is code:

      * ``backbone_class`` — the model's client with ``reasoning.TimelineRoute``
        before it in the bases; ``reasoning.OSWorldBackbone`` says what it provides.
      * ``action_agent_class`` — grounds an intended action into the model's own tool
        call: owns the grounding prompt, the reply parser and the terminal mapping (a
        give-up is a ``done`` carrying ``terminal_intent: "fail"``). Built as
        ``action_agent_class(backbone=..., model_call_attempts=...)``.
      * ``translate_action`` — ``(action, *, screen_size) ->`` what V2's driver
        executes (pyautogui source, ``DONE`` or ``FAIL``), raising ``ValueError`` for
        any defect, which the adapter degrades to a counted ``WAIT``. The adapter
        calls it, not the action agent, so it sits here beside the classes.

    AS VALUES, where it is data: ``action_classes`` (the deciding schema's closed
    ``action_class`` vocabulary), ``deciding_suffix`` (the block appended to every
    deciding instruction, matched to the control's prompt), ``environment_brief``
    (renders the environment facts of the control's system prompt),
    ``call_attempts`` (total attempts per model call, deciding and grounding: the
    control's retry budget), ``grounding_temperature`` (sampling on the
    non-thinking grounding route), ``typing_note`` (the sentence ending the
    environment brief: how the model's translator types, "" when it raises no
    hazard), ``give_up_token`` (the env token an admitted give-up is emitted as),
    ``max_tokens`` (the output cap on both calls, winning over the runner's; None =
    the runner's value), ``deciding_repair`` (None: an unusable deciding reply is
    re-sampled with the identical request; else the repair turn the reply is sent back
    with, ``reasoning.call_with_repair``), ``request_timeout`` (the backbone's
    request timeout in seconds; None = the backbone's own, which takes no such
    argument), ``give_up_block_text`` (None: a blocked give-up's line names the
    completion-gate rule; else this text replaces it for the two give-up rules),
    ``ask_user_channel`` (whether a decided ``call_user`` is handed to the driver's
    ASK_USER path), ``infra_error`` (None: an exception leaving ``predict`` ends
    the episode unscored; else ``predict`` re-raises only what this
    classifies as infrastructure and ends the episode EVALUATED on anything else,
    recording it — ``_lost_episode``) and ``capture_failure_raises`` (False: a step
    whose screenshot capture failed is a hole in the timeline and the episode goes on;
    True: ``predict`` raises ``ScreenshotCaptureFailed`` before anything else, as the
    control does). ``image_retention`` is the timeline's (``timeline.py``):
    which earlier frames a deciding request re-sends, since each control folds its
    image history with its own function, not with different numbers.

    TWO GRAPHPC METHOD FIELDS, off by default and set on the DeepSeek profile only:
    ``requirement_ledger`` (the deciding reply keeps a requirement
    ledger — prompts.OSWORLD_REQUIREMENTS_INSTRUCTION, the schema's ``requirements``, its
    block in STATE — and the completion gate's requirements_unverified rule judges a
    success claim by it, orchestrator.py) and ``step_budget_line`` (a template over
    ``{step}`` and ``{max_steps}``: one STATE line stating the true budget at every step,
    prompts.OSWORLD_STEP_BUDGET_LINE; None: no request shows the budget). Appended with
    their defaults, so a profile that does not name them builds without them.

    EACH PROFILE NAMES ITS OWN ACTION VOCABULARY, used on both sides of the handoff:
    the deciding schema's classes (``action_classes``), the names
    ``normalize_action_class`` reduces a deciding reply to (``predict`` reads the
    terminal class that way), and the action types the profile's action agent
    emits. MiniMax and DeepSeek use M3's normalised names
    (``M3_ACTION_CLASS_VOCABULARY``); Qwen uses its control's
    (``QWEN_ACTION_CLASS_VOCABULARY``: M3's without ``hold_key``, plus ``call_user``,
    ``hscroll``, ``key_down`` and ``key_up``). A model whose tool calls use other
    names maps them into its profile's names in its action agent. The timeline's
    idle and held-type rules and the ``typed_text`` record read those names, and the
    timeline keys the ask channel on ``call_user``.

    The field names differ from the scoped parameter names they feed —
    ``call_attempts``, ``action_classes`` and ``deciding_suffix``, not
    ``model_call_attempts``, ``action_class_vocabulary`` and
    ``deciding_instruction_suffix`` — so a scoped parameter name marks a call site
    that passes it, never a profile field."""

    backbone_class: type
    action_agent_class: type
    translate_action: Callable[..., str]
    action_classes: tuple[str, ...]
    deciding_suffix: str
    environment_brief: Callable[..., str]
    call_attempts: int
    grounding_temperature: Optional[float]
    typing_note: str
    image_retention: ImageRetention
    give_up_token: str
    max_tokens: Optional[int]
    deciding_repair: Optional[Callable[[Exception], str]]
    request_timeout: Optional[float]
    give_up_block_text: Optional[str]
    ask_user_channel: bool
    infra_error: Optional[Callable[[BaseException], bool]]
    capture_failure_raises: bool
    requirement_ledger: bool = False
    step_budget_line: Optional[str] = None


def _minimax_m3_profile() -> OSWorldModelProfile:
    """MiniMax-M3, against V2's ``mm_agents/m3`` control. Built at CALL time from this
    module's globals, so a lever that patches one of them reaches the next build."""
    return OSWorldModelProfile(
        backbone_class=OSWorldMiniMaxAgent,
        action_agent_class=MiniMaxActionAgent,
        translate_action=translate_m3_action,
        action_classes=M3_ACTION_CLASS_VOCABULARY,
        deciding_suffix=OSWORLD_DECIDING_INSTRUCTION_SUFFIX,
        environment_brief=environment_brief,
        call_attempts=MODEL_CALL_ATTEMPTS,
        grounding_temperature=GROUNDING_TEMPERATURE,
        typing_note=OSWORLD_TYPING_NOTE,
        image_retention=M3_IMAGE_RETENTION,
        give_up_token=FAIL_TOKEN,
        max_tokens=None,
        deciding_repair=None,
        request_timeout=None,
        give_up_block_text=None,
        ask_user_channel=False,
        infra_error=None,
        capture_failure_raises=False,
    )


def _qwen_profile() -> OSWorldModelProfile:
    """Qwen qwen3.7-plus over OpenRouter, against osworld-v2.1's ``mm_agents/qwen``
    control (``profile_qwen.qwen_model_profile``). Imported at call time, so the MiniMax
    build imports nothing of the Qwen control."""
    from .profile_qwen import qwen_model_profile

    return qwen_model_profile()


def _deepseek_profile() -> OSWorldModelProfile:
    """DeepSeek deepseek-flash on its own endpoint, against upstream M3 with the native-tool
    deltas as the control (``profile_deepseek.deepseek_model_profile``). Imported at call
    time, like the Qwen profile."""
    from .profile_deepseek import deepseek_model_profile

    return deepseek_model_profile()


# backbone_type -> the zero-argument builder of its profile. Adding a model to this
# adapter is one entry here plus the classes and values its profile names.
OSWORLD_MODEL_PROFILES: dict[str, Callable[[], OSWorldModelProfile]] = {
    "minimax": _minimax_m3_profile,
    "qwen": _qwen_profile,
    "deepseek": _deepseek_profile,
}


def model_profile(backbone_type: str) -> OSWorldModelProfile:
    """The profile of ``backbone_type``, built now; ``NotImplementedError`` for a
    model without one."""
    builder = OSWORLD_MODEL_PROFILES.get(backbone_type)
    if builder is None:
        raise NotImplementedError(f"Backbone type {backbone_type} not supported on OSWorld 2.0 "
                                  f"(supported: {', '.join(sorted(OSWORLD_MODEL_PROFILES))})")
    return builder()


# The pre-registration, and the installed task corpus, both inside the benchmark
# checkout. Checkout-relative and resolved from `task_loader.__file__`, which sits
# at the checkout root — NOT from cwd, and NOT from a runner flag: a path passed as
# argv lets the launch preflight and the run resolve DIFFERENT files, which is the
# drift the whole rule exists to remove, reintroduced at the CLI.
SUBSET_PREREG_FILENAME = "subset_prereg.json"
TASK_CONFIG_BASE_DIRNAME = "evaluation_examples"
# Field names the pre-registration declares. All four are REQUIRED here — see
# `resolve_relevance_gate`.
_PREREG_SUBSET_IDS = "subset_ids"
_PREREG_PERCENTILE = "relevance_threshold_percentile"
_PREREG_RESOLVED = "relevance_threshold_resolved"
_PREREG_TOLERANCE = "relevance_threshold_tolerance"
# REQUIRED, like the four above. It is the one declared quantity that travels with
# the THRESHOLD rather than with the task list, so it is what detects a
# pre-registration whose task set and whose threshold were written at different
# times. The rule module accepts it as optional — correctly, since WebArena has no
# pre-registration at all — but leaving it optional HERE would arm that check only
# if a file this code does not own happens to carry a field nothing requires, and
# would be the one soft failure in a function where a missing loader and every other
# absent field are hard ones. Absent, the staleness it catches is caught only if it
# happens to move the percentile past the tolerance, which at the measured margin is
# luck rather than construction.
_PREREG_POPULATION_SIZE = "relevance_threshold_population_size"
# REQUIRED for the same reason, and it closes what neither of the other two can see:
# the ids match and a task's TEXT was edited underneath — exactly what a benchmark
# point release does. That case matters under this design because the corpus is not
# an artefact this repository controls: it is whatever the checkout's loader returns.
#
# AND THE TAG DOES NOT HOLD THE TEXT. `evaluation_examples/task_class/task_*.py` is in the
# OSWorld-V2 checkout's OWN `.gitignore`; the task classes ship separately from a
# gated HuggingFace dataset. So `git describe --tags --exact-match` and a clean
# `git status` say NOTHING about instruction text, and a re-download inside one tag
# changes it with no git signal at all. Drift inside a release is the blind case,
# not the safe one, and this digest is the only thing watching it.
_PREREG_CORPUS_DIGEST = "relevance_threshold_corpus_digest"
# REQUIRED. The ids the digest does NOT cover, because they have no installed
# instruction to hash — their class mints a value per construction and interpolates
# it. Required rather than optional so that the record must state the scope of its
# own digest; an empty list is the ordinary case and says "the digest covers
# everything".
_PREREG_DIGEST_EXCLUDED_IDS = "relevance_threshold_corpus_digest_excluded_ids"


def _task_instruction(task: Any) -> Optional[str]:
    """The instruction off a loaded task. V2's ``BaseTask`` subclasses ``dict`` and
    populates it in ``__init__``, and a JSON task is a plain dict, so mapping access
    covers both; the class attribute is the fallback.

    RAW, deliberately: no strip, no Unicode normalisation, no whitespace collapsing.
    The corpus digest hashes exactly what this returns, and the pre-registration's
    digest was computed from the same loader output — real instructions in the
    installed corpus carry trailing spaces and newlines, and 11 of 28 contain
    newlines outright. Any normalisation added here changes every digest and would
    surface on the host as a corpus mismatch. Empty is treated as absent, which the
    gate's identity check then refuses by name."""
    value = None
    if isinstance(task, dict):
        value = task.get("instruction")
    if not value:
        value = getattr(task, "instruction", None)
    return str(value) if value else None


def _task_loader_and_base_dir() -> tuple[Any, Path]:
    """The benchmark checkout's own ``task_loader`` module and the task-config
    directory under it — the ONE derivation of where the corpus and the
    pre-registration live, shared by every reader in this module. Imported
    inside the function on purpose: this module must import outside a checkout,
    and a check can swap ``sys.modules["task_loader"]`` around the call."""
    try:
        import task_loader  # noqa: PLC0415 — only importable inside the benchmark checkout
    except ImportError as exc:
        raise RelevanceCalibrationError(
            "the benchmark's task_loader is not importable, so the installed task corpus "
            "cannot be read and the pre-registered relevance rule cannot be substantiated. "
            "This agent is built to run inside the OSWorld-V2 checkout, where the runner's "
            f"`import _repo_path` puts the checkout root on sys.path ({exc})."
        ) from exc
    checkout_root = Path(task_loader.__file__).resolve().parent
    return task_loader, checkout_root / TASK_CONFIG_BASE_DIRNAME


def load_instruction_corpus(task_ids: Sequence[str]) -> dict[str, str]:
    """The instruction text of each task id, read through the checkout's OWN
    loader — the single corpus reader. The worker's verify-mode gate
    (``resolve_relevance_gate``) obtains its corpus here and nowhere else. No
    ``base_dir`` parameter: the root is
    derived here, from ``task_loader.__file__``, so no caller can pass a
    different one.

    Per-task load failures and missing instructions are logged individually and
    the id is left out, so the caller's corpus-identity check refuses with the
    authoritative message naming every id. The extractor is
    ``_task_instruction`` (raw, no strip): the digest hashes exactly what it
    returns."""
    task_loader, base_dir = _task_loader_and_base_dir()
    corpus: dict[str, str] = {}
    for task_id in task_ids:
        try:
            task = task_loader.load_task_config(
                None, task_id=task_id, base_dir=str(base_dir), eval_version="v2", prefer_class=True,
            )
        except Exception as exc:  # noqa: BLE001 — diagnosed here, refused by the gate
            logger.error("relevance corpus: task %s failed to load from %s: %s", task_id, base_dir, exc)
            continue
        instruction = _task_instruction(task)
        if instruction is None:
            logger.error("relevance corpus: task %s loaded but carries no instruction", task_id)
            continue
        corpus[task_id] = instruction
    return corpus


def resolve_relevance_gate() -> PercentileRelevanceGate:
    """The OSWorld relevance gate, in VERIFY mode, with the corpus read from the
    benchmark checkout's OWN task loader.

    WHY THIS EXISTS AT ALL. Without it `LocalSupportSetStore` defaults
    ``relevance_gate=None`` and resolves to the fixed 0.15 — which, on this
    benchmark's distribution, admits essentially everything and injects a stored
    entry into nearly every later task. The percentile module can
    be present and correct and still be completely inert; the rule is the call site.

    WHY VERIFY AND NOT CALIBRATE. The disagreement recomputation catches — installed
    task set or encoder not the one the pre-registration was written against — is a
    RUN-LEVEL static property. Recomputing here would pay n*(n-1) encodes per worker,
    again on every environment recycle, to re-derive a constant, and would discover a
    launch blocker mid-run after VM time is spent. Calibration belongs before the run:
    calibrate mode is run offline, before any worker (no shipped script runs it: call
    ``resolve_relevance_threshold`` with a ``PercentileRelevanceGate(..., recompute=True)``
    on the installed corpus to re-derive the pinned value); workers run verify mode (and
    the OSWorld runner refuses a retrieval-on launch). The worker confirms the corpus is
    the declared one and applies the pinned value, at zero encodes.

    WHY IT RAISES AND NEVER FALLS BACK. Everything below is a hard failure, before
    any VM is provisioned. A worker that cannot substantiate the rule must die rather
    than gate at 0.15 while its telemetry records ``rule="percentile"`` — a record
    claiming more work than was done is the degraded-path-as-primary failure in its
    purest form, and this call site is where that property holds or is lost.

    Per-task load failures are logged individually (``load_instruction_corpus``) and
    the task is simply left out of the corpus, because the gate's own corpus-identity
    check then refuses with the authoritative message naming every id. The diagnosis
    is ours; the refusal is its.
    """
    _, base_dir = _task_loader_and_base_dir()
    prereg_path = base_dir / SUBSET_PREREG_FILENAME
    try:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RelevanceCalibrationError(
            f"the pre-registration is unreadable at {prereg_path}: {exc}. It is a hand-copied "
            "host file, so a fresh checkout or a host rebuild will not have it."
        ) from exc

    missing_fields = [
        name for name in (_PREREG_SUBSET_IDS, _PREREG_PERCENTILE, _PREREG_RESOLVED,
                          _PREREG_TOLERANCE, _PREREG_POPULATION_SIZE, _PREREG_CORPUS_DIGEST,
                          _PREREG_DIGEST_EXCLUDED_IDS)
        if prereg.get(name) is None
    ]
    if missing_fields:
        raise RelevanceCalibrationError(
            f"the pre-registration at {prereg_path} declares no {', '.join(missing_fields)}. "
            "The installed copy is hand-maintained and drifts from the repository's; refresh it "
            "rather than supplying the numbers here — a threshold typed at a call site is the "
            "hard-coded constant this rule replaces."
        )

    # PRESENT IS NOT THE SAME AS USABLE. The `is None` pass above only says the keys exist; every
    # number below then goes through float()/int() and every id through a for-loop, so a field that
    # is present but the wrong SHAPE — a quoted "74.5", a null-like string, a subset_ids that is a
    # string rather than a list — left this function with a bare ValueError or TypeError naming no
    # file and inviting no fix, in a function whose whole contract is that every failure here is a
    # typed hard failure that names the record. Strict types, not coercion: this record is JSON
    # written by a script, so a quoted number is a defect in the record, not a spelling of it.
    bad_shape = [
        f"{name}={prereg[name]!r}"
        for name in (_PREREG_PERCENTILE, _PREREG_RESOLVED, _PREREG_TOLERANCE, _PREREG_POPULATION_SIZE)
        if isinstance(prereg[name], bool) or not isinstance(prereg[name], (int, float))
    ]
    if not isinstance(prereg[_PREREG_CORPUS_DIGEST], str):
        bad_shape.append(f"{_PREREG_CORPUS_DIGEST}={prereg[_PREREG_CORPUS_DIGEST]!r} (want a string)")
    if not isinstance(prereg[_PREREG_SUBSET_IDS], list) or not prereg[_PREREG_SUBSET_IDS]:
        bad_shape.append(f"{_PREREG_SUBSET_IDS}={prereg[_PREREG_SUBSET_IDS]!r} (want a non-empty list)")
    # EMPTY IS LEGITIMATE and must not be confused with absent: a corpus whose instructions are all
    # fixed strings declares [], and that is the ordinary case. Only a non-list is a defect.
    if not isinstance(prereg[_PREREG_DIGEST_EXCLUDED_IDS], list):
        bad_shape.append(f"{_PREREG_DIGEST_EXCLUDED_IDS}={prereg[_PREREG_DIGEST_EXCLUDED_IDS]!r} "
                         f"(want a list, possibly empty)")
    if bad_shape:
        raise RelevanceCalibrationError(
            f"the pre-registration at {prereg_path} declares {'; '.join(bad_shape)}. Those fields are "
            "read as numbers, a digest string and an id list; the file is generated, so a wrong shape "
            "means it was hand-edited or written by something else. Refresh it — do not coerce here, "
            "because a value this function has to repair is a value nobody pre-registered."
        )

    # The percentile is THE pre-registered parameter, and the file that declares it
    # is a hand-copied host artefact. Without this comparison a coherently-edited
    # pre-registration (p50, p90) would pass every guard without complaint, because each
    # guard checks the file against ITSELF — the pinned threshold really is that
    # corpus's p50, so the arithmetic agrees and the rule silently becomes a
    # different rule. Binding it to the code constant is what makes "pre-registered"
    # mean fixed-in-advance rather than whatever the installed file says today.
    declared_percentile = float(prereg[_PREREG_PERCENTILE])
    if declared_percentile != PREREGISTERED_RELEVANCE_PERCENTILE:
        raise RelevanceCalibrationError(
            f"the pre-registration at {prereg_path} declares "
            f"{_PREREG_PERCENTILE}={declared_percentile}, but the pre-registered percentile is "
            f"{PREREGISTERED_RELEVANCE_PERCENTILE} (src/agents/retrieval/relevance.py). The "
            "percentile is fixed in advance and applied unchanged to every benchmark; an "
            "installed file that moves it is stale or edited. Refresh the file — do not change "
            "the constant to match it, and do not launch on the file's value."
        )

    expected_task_ids = [str(task_id) for task_id in prereg[_PREREG_SUBSET_IDS]]
    corpus = load_instruction_corpus(expected_task_ids)

    return PercentileRelevanceGate(
        corpus=corpus,
        expected_task_ids=expected_task_ids,
        percentile=float(prereg[_PREREG_PERCENTILE]),
        pinned_threshold=float(prereg[_PREREG_RESOLVED]),
        pinned_population_size=int(prereg[_PREREG_POPULATION_SIZE]),
        pinned_corpus_digest=str(prereg[_PREREG_CORPUS_DIGEST]),
        digest_excluded_task_ids=[str(task_id) for task_id in prereg[_PREREG_DIGEST_EXCLUDED_IDS]],
        tolerance=float(prereg[_PREREG_TOLERANCE]),
        # VERIFY: calibration is an offline step (no shipped script runs it); the worker confirms and applies.
        recompute=False,
    )


def pointer_positions_for_mask(monitor_input: Any) -> tuple[tuple[int, int], ...]:
    """The monitor's pointer-mask locator for this benchmark: the screen-pixel
    positions the mouse cursor occupies in the two frames being diffed, or an
    empty tuple when the mask must not fire.

    WHY THE MASK EXISTS. The guest's screenshot includes the mouse cursor, so
    moving the pointer changes pixels and every step reads as "something
    changed", including a step where nothing happened. That silently disables the three
    checks the diff feeds: whether the model's claim about its own action was
    truthful, whether it has earned the right to declare the task finished, and
    whether an unsupported claim is corrected before the episode is stored.

    WHY TWO POSITIONS. The frames bracket the action: the cursor composites at
    the position it held while the earlier frame was captured and at this step's
    in the later one, so the artifact appears twice and both must be masked.
    Both are read from the executed pyautogui source, i.e. what was actually
    sent to the environment in screen pixels, not from the model's normalised
    intent.

    WHY THE PREDECESSOR IS SEARCHED FOR, NOT COUNTED BACK TO. The earlier
    frame's cursor sits wherever the last action that ACTUALLY MOVED THE POINTER
    left it — which is not in general the immediately preceding record. On
    `click -> type -> click`, or on any step following a key press or a WAIT,
    the record at k-1 carries no coordinate and the sprite in the earlier frame
    is still at the k-2 click. Taking the last two records and letting the
    predecessor parse to None would mask only the current sprite on exactly those
    steps: on a synthesised true no-op that leaves 540 px unmasked, that removes
    half and returns 270 px — still over four times the floor, still `changed`.
    So the predecessor is the NEAREST EARLIER record whose source parses to a
    position, scanning back as far as it takes. There is no bound on the scan
    because there is no bound on how long a cursor may sit still: a click
    followed by ten typed steps still has its sprite at that click.

    WHY ONE POSITION ON THE FIRST CLOSED STEP. When no earlier record moved the
    pointer, the earlier frame's cursor is at the position the guest booted it
    to, which is never recorded and cannot be recovered from anything the
    adapter holds. Masking the current sprite alone is strictly better than
    masking neither, and inventing a coordinate — a screen-centre guess, a
    replay of the current point — would blank a region the cursor may never have
    occupied and could hide a real change there. So this returns one point and
    accepts that a pointer-only no-op can still read `changed` on the first
    closed step of an episode.

    WHY EMPTY ON A COORDINATE-FREE STEP. If this step's action moves no pointer
    (a key press, a typed string, a WAIT), the cursor is drawn identically in
    both frames and contributes nothing — and masking the PREVIOUS click anyway
    would eat the characters of a step that types into the field it clicked on
    the step before. That is the one way this mask could hide a real change, so
    it is closed here rather than mitigated.

    WHY A CALLABLE. The WebArena adapter writes `_executed_actions` too, with
    browser action strings; a coordinate parser living in the shared monitor
    would be inert there only by accident. Supplying the locator from this
    benchmark's build_default makes the mask unreachable anywhere else."""
    content = getattr(getattr(monitor_input, "runtime_state", None), "content", None)
    records = [record for record in ((content or {}).get("_executed_actions") or [])
               if isinstance(record, dict)]
    if not records:
        return ()
    # This step. A coordinate-free action here masks nothing at all (above).
    current = pointer_xy_from_pyautogui(records[-1].get("command"))
    if current is None:
        return ()
    # The earlier frame's sprite: the nearest earlier record that moved the
    # pointer, however many coordinate-free steps sit in between.
    for record in reversed(records[:-1]):
        previous = pointer_xy_from_pyautogui(record.get("command"))
        if previous is not None:
            return (previous, current)
    return (current,)


class _RecordingPointerLocator:
    """``pointer_positions_for_mask`` with a memory: it returns exactly what the
    locator returned and keeps that answer, stamped with the step it answered
    for, so the adapter can write the masked region into the episode's record.

    WHY THE POINTS AND NOT A BOOLEAN. The mask's quiet failure mode is INERT
    MASKING — the sprite missed through wrong coordinates or wrong scaling. A
    boolean cannot show that: it is ``True`` for
    a mask that landed on the cursor and ``True`` for one that landed 500 px
    away, and it cannot tell the degraded one-point path (no earlier record moved
    the pointer) from the normal two-point one, which the measurement-integrity
    rule requires telemetry to distinguish.

    WHY RECORDED LIVE AND NOT RE-DERIVED. ``pointer_positions_for_mask`` is the
    function under suspicion. Re-running it offline over the same
    ``_executed_actions`` would reproduce a coordinate bug identically and agree
    with itself; what makes the record evidence is that it holds the coordinates
    the run actually masked, judgeable against the step's screenshot without
    calling the suspect function again.

    WHY A WRAPPER RATHER THAN A NEW MONITOR FIELD. The monitor is shared code, and
    a new content key there would change WebArena's records too; the locator is
    already this benchmark's own object, handed over by ``build_default``, so
    wrapping it keeps the whole mechanism inside ``src/benchmarks/osworld/``.

    READING THE RECORD. These are the points the locator OFFERED. The monitor
    honours at most ``POINTER_MASK_MAX_POINTS`` of them and skips any point
    outside the frame, so the recorded list is an upper bound on what was masked
    — an exact one, because ``pointer_positions_for_mask`` never returns
    more than two, which is why the monitor's own ``pointer_mask_applied`` is
    kept beside it as a second, independently computed witness."""

    def __init__(self, locator: Callable[[Any], Sequence[tuple[int, int]]] = pointer_positions_for_mask) -> None:
        self._locator = locator
        # The step the stamped answer belongs to. The monitor is consulted once
        # per update_runtime, BEFORE step_idx advances, so a record written for
        # step k must find k here; anything else is a stale read and is recorded
        # as absent rather than as a coordinate that was never masked.
        self.last_step_idx: Optional[int] = None
        self.last_points: tuple[tuple[int, int], ...] = ()

    def __call__(self, monitor_input: Any) -> tuple[tuple[int, int], ...]:
        points = tuple(self._locator(monitor_input) or ())
        state = getattr(monitor_input, "runtime_state", None)
        self.last_step_idx = getattr(state, "step_idx", None)
        self.last_points = points
        return points

    def take(self, step_idx: int) -> Optional[list[list[int]]]:
        """The points recorded FOR ``step_idx``, and empty the slot. ``None`` when
        the slot is empty or stamped for a different step; ``[]`` when the locator
        ran on this step and deliberately masked nothing.

        READ-ONCE, DELIBERATELY, and it is what makes a cross-episode leak
        impossible rather than merely unlikely. The runner rebuilds the agent only
        when a task's env spec changes, so ONE instance of this class spans many
        episodes; ``step_idx`` restarts from 0 every episode, so a stamp left by a
        previous episode can equal the current episode's step number. The monitor
        also emits its mask keys whenever a locator is CONFIGURED, not only on the
        steps where it consults one — so a step whose screenshot was missing takes
        the text branch, never calls this object, and still asks for a masked
        region. Without it, episode B's step 0 would record episode A's
        coordinates, with ``pointer_mask_applied`` False beside them. That is a
        confident false statement about a region the episode never masked, which
        is worse than the gap ``None`` records, and it defeats the point of the
        field — evidence independent of the function under suspicion.

        The PRIMARY guard is at the call site, which asks for a region only where
        the monitor reports it took the screenshot branch — i.e. where the locator
        definitely ran this step. Emptying the slot on read is the second,
        independent one: it makes a second read of the same answer impossible
        whatever the caller does, and it is a property of this object rather than
        of a branch in shared code."""
        recorded, stamp = self.last_points, self.last_step_idx
        self.clear()
        if stamp is None or stamp != step_idx:
            return None
        return [[int(x), int(y)] for x, y in recorded]

    def clear(self) -> None:
        """Drop any recorded answer. Called at the adapter's per-episode reset:
        this object outlives the episode, so its state has to be reset with the
        rest of the per-episode state rather than left to the next read."""
        self.last_step_idx = None
        self.last_points = ()


class _OrchestratorProtocol(Protocol):
    """The stable inner boundary the adapter depends on. Anything else is
    internals — never reach across this protocol."""

    def decide_next_action(self, *, instruction, observation_input, runtime_state) -> GraphPCDecision: ...
    def update_runtime(self, *, decision, current_observation_input) -> GraphPCRuntimeUpdate: ...


def _json_safe(value: Any) -> Any:
    """Telemetry sanitiser: bytes (a screenshot that leaked into a record)
    become a placeholder instead of a multi-megabyte repr."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return "<bytes omitted>"
    return value


def _is_full_score(score: Optional[float]) -> bool:
    """Support-set persistence criterion: binary completion, i.e. the
    final score is 1.0 (a partial-score evaluator sums weighted criteria, so
    a full pass is compared with a float tolerance rather than ``==``)."""
    if score is None:
        return False
    try:
        return math.isclose(float(score), 1.0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _first_text(mapping: Any) -> tuple[str, Optional[str]]:
    """``(text, key)`` of the first non-empty terminal-text key in ``mapping``;
    ``("", None)`` when there is none."""
    if not isinstance(mapping, dict):
        return "", None
    for key in _TERMINAL_TEXT_KEYS:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    return "", None


class OSWorldGraphPCAgent:
    def __init__(
        self,
        *,
        orchestrator: _OrchestratorProtocol,
        support_set_store: Optional[LocalSupportSetStore],
        screen_size: tuple[int, int] = (1920, 1080),
        retrieval_off: bool = False,
        backbone: Any = None,
        config: Optional[dict[str, Any]] = None,
        pointer_mask_recorder: Optional[_RecordingPointerLocator] = None,
        timeline: Optional[EpisodeTimeline] = None,
        backbone_type: str = "minimax",
    ) -> None:
        self.orchestrator = orchestrator
        # The model this adapter runs, i.e. its profile (OSWORLD_MODEL_PROFILES). The
        # profile is resolved where a per-model piece is used, as build_default
        # resolves it, and not cached: a patched module global reaches it. The
        # default is build_default's own.
        self.backbone_type = backbone_type
        # None under retrieval_off (no store and no embedder are
        # constructed in that mode; the read side is BypassRetrievalAgent
        # inside the orchestrator, the write side is suppressed here).
        self.support_set_store = support_set_store
        self.screen_size = (int(screen_size[0]), int(screen_size[1]))
        self.retrieval_off = retrieval_off
        # The shared backbone, held ONLY for telemetry (model name + cumulative
        # usage counters) and ``reset_session`` — every model call goes through
        # the orchestrator. Optional so the adapter can be built with a stub
        # orchestrator and no backbone at all.
        self.backbone = backbone
        # Per-call diagnostics: every OSWorld backbone reports each completed call
        # to ``call_observer``; records are numbered per task (reset at the
        # task boundary) and the attempt index per stage per step (reset per
        # predict). A backbone without the hook (a fake) is left alone.
        self._call_records = 0
        self._stage_calls_this_step: dict[str, int] = {}
        if backbone is not None and hasattr(backbone, "call_observer"):
            backbone.call_observer = self._on_model_call
        # The monitor's pointer locator, wrapped so each step's masked region is
        # recorded rather than re-derivable only by calling the suspect function
        # again (_RecordingPointerLocator). None where no mask is configured —
        # an adapter built with a stub orchestrator.
        self._pointer_mask_recorder = pointer_mask_recorder
        # The deciding call's episode history (timeline.py): this adapter writes it —
        # it is called once per driver step and holds what was EXECUTED, the frame
        # and the monitor verdict — and the OSWorld reasoning agent build_default
        # constructs with the SAME instance reads it. A private one when none is
        # handed over (a stub orchestrator reads nothing).
        self.timeline = timeline if timeline is not None else EpisodeTimeline()
        # Active configuration echoed into every telemetry dump and support-set
        # entry (measurement integrity: the record names what actually ran).
        self.config: dict[str, Any] = dict(config or {})

        self.runtime_state: Optional[RuntimeState] = None
        self.last_decision: Optional[GraphPCDecision] = None
        self.last_update: Optional[GraphPCRuntimeUpdate] = None
        self.task_instruction: Optional[str] = None
        # Set by the V2 driver via setattr before reset(); handed to the
        # observation adapter each step for the environment brief.
        # Survives reset.
        self.task_current_date = None
        # Set by the runner before each task; survives reset.
        self._api_log_dir: Optional[str] = None
        # Upstream env-feedback signal stashed for the NEXT step (surfaced via
        # _env_feedback, read by the monitor pass). A dict = the `info` payload,
        # keyed by failure kind so the monitor can describe it distinctly:
        #   {"action_parse_error": ...}      untranslatable action (env-side reject)
        #   {"reasoning_parse_error": ...}   unparseable deciding response (soft-degrade)
        #   {"completion_gate_blocked": ...} blocked done, corrective (Mechanism 2)
        #   {"grounding_parse_error": ...}   unparseable grounding response
        #   {"grounding_class_mismatch": ...} grounding replies outside the decided class's family (Qwen)
        self._pending_env_feedback: Optional[dict[str, Any]] = None
        # Translation rejects (an action the M3 translator refused): per step.
        self.action_parse_failure_count = 0
        # Unparseable model SAMPLES, counted per ATTEMPT (every unparseable
        # reply, whether a later re-sample rescued the step or not) — so they
        # are not a duplicate of the exhausted-step counts below.
        self.reasoning_parse_failure_count = 0
        self.grounding_parse_failure_count = 0
        # Deciding SAMPLES that parsed but carried no decision — the schema echoed
        # back, or no list-typed intended_actions (reasoning.py:DecisionMissing) —
        # counted per attempt and APART from the unparseable ones above; they are
        # re-sampled the same way.
        self.reasoning_decision_missing_count = 0
        # The same samples by shape (schema_echo / native_tool_call / other).
        self.reasoning_decision_missing_kinds: dict[str, int] = {}
        # Free re-samples of unusable model output (common/resample.py):
        # extra samples beyond the first call, summed over the task's steps,
        # read off each decision's reasoning message / action batch metadata.
        self.reasoning_resample_count = 0
        self.grounding_resample_count = 0
        # Steps where all attempts failed and the degrade path ran (WAIT).
        self.resample_exhausted_reasoning = 0
        self.resample_exhausted_grounding = 0
        # THE DENOMINATOR FOR EVERY GROUNDING-SIDE COUNTER ABOVE. The bypass removes
        # most grounding calls, so `grounding_parse_failures` falls by roughly the
        # same proportion for reasons that have nothing to do with sampling quality —
        # and set beside a control number whose scope did not move, it would be a
        # number whose denominator moved silently, reported as if it had not. These
        # two make the rate computable, so the comparison is of like with like.
        self.grounding_calls_made = 0
        self.grounding_calls_bypassed = 0
        # Coordinates stripped from the grounding line (action_minimax.py:
        # strip_grounding_coordinates): grounded intents that had any removed, and
        # fields left unstripped because stripping would have emptied them.
        self.grounding_inputs_stripped = 0
        self.grounding_strip_fallbacks = 0
        # Grounded actions the backbone took from a call's function name (the DeepSeek profile:
        # action_deepseek.py, "THE ACTION IN THE FUNCTION NAME"); recorded where the profile
        # carries the rule (_function_name_count).
        self.grounding_actions_from_function_name = 0
        # The DeepSeek profile's grounding repairs (action_deepseek.py): accepted replies whose
        # arguments parsed after the surplus-closer repair, and the click texts dropped from the
        # grounding line and from accepted replies; recorded where the profile carries the rules
        # (_grounding_repair_counts).
        self.grounding_arguments_repaired = 0
        self.grounding_line_texts_dropped = 0
        self.grounding_reply_texts_dropped = 0
        # Driver-visible step count (one per predict).
        self.predict_calls = 0
        # Per-step monitor diff telemetry: signal / changed / pixel counts
        # read off each ``update_runtime``'s monitor message; ``recent_history``
        # only keeps the monitor's progress_status.
        self.monitor_signals: list[dict[str, Any]] = []
        # Multi-phase tasks: the driver resets the agent per phase,
        # so each phase is a fresh runtime state. The outgoing phase's record is
        # stashed at the next reset and dumped with the task's telemetry
        # (finalize covers the last phase's live state).
        self.phase_records: list[dict[str, Any]] = []
        # The ask step whose answer the next predict's obs carries and the questions
        # asked (the ask channel), and the episode's lost-episode records (the profile's
        # infra_error).
        self._ask_step: Optional[int] = None
        self.ask_user_count = 0
        self.agent_errors: list[dict[str, Any]] = []
        self._finalized = True
        self._token_start: dict[str, Any] = self._collect_token_usage()

    # ------------------------------------------------------------------
    # build_default — the wiring, per model profile (OSWORLD_MODEL_PROFILES: MiniMax-M3, Qwen, DeepSeek).
    # ------------------------------------------------------------------

    @classmethod
    def build_default(
        cls,
        backbone_type: str = "minimax",
        *,
        model: str,
        support_set_root: str | Path,
        retrieval_off: bool = False,
        two_pass: bool = False,
        enable_completion_gate: bool = True,
        completion_gate_window: int = 3,
        screen_size: tuple[int, int] = (1920, 1080),
        client_password: str = DEFAULT_CLIENT_PASSWORD,
        **backbone_kwargs,
    ) -> "OSWorldGraphPCAgent":
        """The agent for ``backbone_type``'s model profile (``OSWORLD_MODEL_PROFILES``;
        ``NotImplementedError`` for a model without one), with one shared backbone
        (the profile's ``backbone_class``) for the deciding and grounding calls:
        deciding = ``OSWorldReasoningAgent``, the graph-reasoning agent with this
        benchmark's deciding-prompt additions (``reasoning.py``), which renders the
        episode timeline and sends it through the backbone's
        ``complete_timeline_json`` (thinking + json_object); grounding = the
        profile's action agent (MiniMax-M3: non-thinking, V2's native
        ``<tool_call>`` text format via ``complete_text(grounding=True)``, emitting
        the V2-M3 ``computer`` vocabulary with [0,1000] relative coordinates). Retrieval
        is deterministic over the ``osworld`` support-set backend, or — under
        ``retrieval_off`` — bypassed with NO store and NO embedder constructed
        (the orchestrator requires a retrieval agent, so bypass is an
        agent, not a branch). ``client_password`` is the guest's sudo password
        for the environment brief (the runner passes ``env.client_password``).
        This is also the ONLY site that names OSWorld's values for the
        benchmark-scoped settings in shared code — the model-call budget (the
        profile's), the within-step injection diet, the monitor's pointer mask, the
        percentile relevance gate, the closed ``action_class`` vocabulary (the
        profile's, against WebArena's shared default, the free-form field), the
        guarded deciding-instruction suffix (the
        profile's; run-invariant, so it is passed here rather than per step, unlike
        the environment brief) and how much of a step's intended-action plan survives
        into ``recent_history`` (``OSWORLD_INTENDED_ACTIONS_CAP``, against
        WebArena's shared default, the unbounded copy)
        (retrieval-event telemetry is unconditional and takes no parameter). The
        deciding request itself — the episode timeline, the environment facts and
        the schema in the system prompt — is rendered by ``OSWorldReasoningAgent``
        over the backbone's ``TimelineRoute`` (``reasoning.py``), constructed only
        here and sharing one ``EpisodeTimeline`` with the adapter, so it takes no
        parameter in shared code at all. Every shared default is WebArena's
        behaviour, so a WebArena run cannot execute any of these paths.
        Constructing the gate is what makes the percentile rule RUN: the
        store's default is the fixed 0.15, so a missing
        call here leaves the whole calibrated rule inert with nothing saying so.
        It resolves the corpus from the checkout and RAISES rather than degrading,
        so the failure lands before any VM is provisioned.
        A model is added to this benchmark by adding its profile."""
        profile = model_profile(backbone_type)
        max_steps = OSWORLD_MAX_STEPS_BY_BACKBONE[backbone_type]
        if retrieval_off:
            support_set_store = None
            retrieval_agent: RetrievalAgent = BypassRetrievalAgent()
        else:
            embedder = get_default_embedder()
            support_set_store = LocalSupportSetStore(
                root_dir=support_set_root, backend="osworld", embedder=embedder,
                # The percentile rule. Without this the store defaults to the fixed
                # 0.15 and the whole calibrated rule is inert while nothing says so.
                relevance_gate=resolve_relevance_gate(),
            )
            retrieval_agent = DeterministicRetrievalAgent(
                support_set_store=support_set_store,
                embedder=embedder,
                injected_history_diet=OSWORLD_INJECTED_HISTORY_DIET,
            )
        if profile.max_tokens is not None:
            backbone_kwargs["max_tokens"] = profile.max_tokens
        if profile.request_timeout is not None:
            backbone_kwargs["request_timeout"] = profile.request_timeout
        backbone = profile.backbone_class(
            model=model, grounding_temperature=profile.grounding_temperature,
            **backbone_kwargs,
        )
        # The locator the monitor calls AND the thing the adapter reads the masked
        # region back off: one object, so the record cannot disagree with the mask.
        pointer_mask_recorder = _RecordingPointerLocator()
        # The deciding call's episode history: the adapter writes it, the reasoning
        # agent reads it — one object, the same pattern as the locator above.
        timeline = EpisodeTimeline(retention=profile.image_retention)
        observation_adapter = OSWorldObservationAdapter(
            screen_size=screen_size, client_password=client_password, environment_note=profile.typing_note,
            environment_brief=profile.environment_brief,
        )
        orchestrator = OSWorldOrchestrator(
            timeline=timeline,
            max_steps=max_steps,
            infeasible_declaration_rule=infeasible_declaration_rule,
            requirement_ledger=profile.requirement_ledger,
            reasoning_agent=OSWorldReasoningAgent(
                backbone=backbone, timeline=timeline, model_call_attempts=profile.call_attempts,
                repair=profile.deciding_repair,
                deciding_instruction_suffix=profile.deciding_suffix,
                # NO item cap: the timeline renders every executed step and never reads
                # `recent_history`, which is what this parameter slices. None rather
                # than the shared 3, so the built object says what runs.
                max_trajectory_length=None,
                # The grounding TOOLS block's own action set (the profile's). Passed
                # here rather than defaulted in shared code because the schema is sent
                # to the model: constraining the shared default would change WebArena's
                # schema bytes.
                action_class_vocabulary=profile.action_classes,
                # The profile's two GraphPC method fields (off but on DeepSeek): the requirement
                # ledger and the step-budget line, which states this arm's budget.
                requirement_ledger=profile.requirement_ledger,
                step_budget_line=profile.step_budget_line,
                max_steps=max_steps,
            ),
            retrieval_agent=retrieval_agent,
            action_agent=profile.action_agent_class(backbone=backbone, model_call_attempts=profile.call_attempts),
            monitor_agent=HeuristicMonitorAgent(
                screenshot_diff_config=ScreenshotDiffConfig(pointer_locator=pointer_mask_recorder),
            ),
            perception_processor=observation_adapter,
            default_backend="osworld",
            max_actions=1,
            two_pass=two_pass,
            enable_completion_gate=enable_completion_gate,
            completion_gate_window=completion_gate_window,
            history_intended_actions_cap=OSWORLD_INTENDED_ACTIONS_CAP,
            # Rule 1 of the completion gate grounds a stop ANSWER against observation
            # text, which does not exist on a screenshot-only benchmark. It also does
            # not fall through to rule 2, so leaving it reachable would block every
            # answer-bearing done permanently. Off here, by design and on the record.
            completion_gate_answer_rule=False,
        )
        config = {
            "backbone_type": backbone_type,
            "model": model,
            "retrieval_off": bool(retrieval_off),
            "retrieval_agent": type(retrieval_agent).__name__,
            "two_pass": bool(two_pass),
            "enable_completion_gate": bool(enable_completion_gate),
            "completion_gate_window": int(completion_gate_window),
            "max_actions": 1,
            "screen_size": [int(screen_size[0]), int(screen_size[1])],
            "observation": "screenshot",
            # Instruction-only retrieval text is the OSWorld instantiation of
            # design.md §5 (no observation text on this benchmark); named here
            # so the telemetry states it rather than implying it.
            "retrieval_text": "instruction_only",
            "support_set_root": None if retrieval_off else str(support_set_root),
            # The benchmark-scoped settings, stated in the telemetry so a run's
            # active configuration is a recorded fact and not a code read.
            "model_call_attempts": profile.call_attempts,
            # The history settings, stated so a run's record names them. The deciding
            # request is the episode timeline (timeline.py): every step rendered, so
            # the text history has NO item cap (null here); the
            # per-step text bounds are recorded beside it. The image pair is the
            # control's MECHANISM (budget + truncation threshold), not a frame count:
            # the count sawtooths and no single number describes it.
            "deciding_request": "timeline",
            "max_trajectory_length": None,
            "timeline_text_limits": {
                "reasoning_text": REASONING_TEXT_LIMIT,
                "out_of_window_line": OUT_OF_WINDOW_LINE_LIMIT,
                "executed_text": EXECUTED_TEXT_LIMIT,
                "fact_entry": FACT_ENTRY_LIMIT,
                "fact_source": FACT_SOURCE_LIMIT,
                "facts_render": [FACTS_RENDER_LIMIT, FACTS_RENDER_HEAD, FACTS_RENDER_TAIL],
            },
            "history_intended_actions_cap": OSWORLD_INTENDED_ACTIONS_CAP,
            "orchestrator": "OSWorldOrchestrator",
            # The rules it can block with, and the typing sentence the environment brief
            # ends with, as this build carries them: a score read off this record is
            # self-certifying about which completion-gate rule set produced it.
            "completion_gate_rules": sorted(OSWORLD_COMPLETION_GATE_RULES),
            "typing_note": observation_adapter.environment_note,
            "max_steps": max_steps,
            "retained_images": dict(profile.image_retention.config),
            # DERIVED from the dataclass, not a third hand-written copy of its field
            # names: a fourth switch would otherwise need adding here, in the diet
            # itself and in the preflight's assertion, with nothing tying the three
            # together. Same bind as the pre-registered percentile to its constant.
            "injected_history_diet": {
                field.name: getattr(OSWORLD_INJECTED_HISTORY_DIET, field.name)
                for field in dataclasses.fields(OSWORLD_INJECTED_HISTORY_DIET)
            },
            "pointer_mask": True,
            # The relevance rule AS RESOLVED, read back off the store rather than
            # restated from intent: this is the field that distinguishes "the
            # percentile rule ran" from "the module is present and the fixed 0.15
            # is what actually gated". Under retrieval_off there is no store and
            # no rule, and the block says so rather than implying a default.
            "relevance": None if support_set_store is None else {
                "rule": support_set_store.resolved_relevance.rule,
                "resolution": support_set_store.resolved_relevance.resolution,
                "threshold": support_set_store.resolved_relevance.value,
                "percentile": support_set_store.resolved_relevance.percentile,
                "corpus_size": support_set_store.resolved_relevance.corpus_size,
                "population_size": support_set_store.resolved_relevance.population_size,
                "computed_value": support_set_store.resolved_relevance.computed_value,
                "pinned": support_set_store.resolved_relevance.pinned,
            },
        }
        # Settings only some profiles set, recorded where they are set: a profile without
        # them (MiniMax) records none of them.
        if profile.max_tokens is not None:
            config["max_tokens"] = profile.max_tokens
        if profile.request_timeout is not None:
            config["request_timeout_s"] = profile.request_timeout
        if profile.deciding_repair is not None:
            config["unusable_deciding_reply"] = "repaired"
        if OSWORLD_STUCK_STOP_BY_BACKBONE.get(backbone_type) is not None:
            config["stuck_stop"] = deepcopy(OSWORLD_STUCK_STOP_BY_BACKBONE[backbone_type])
        # The profile's two GraphPC method fields (the DeepSeek profile only): the ledger's rule joins the
        # completion-gate rules this build can block with, and the ledger's text, schema, bounds and rule, and the
        # budget line's template, are the method surface the pre-registration holds the run to.
        if profile.requirement_ledger:
            config["completion_gate_rules"] = sorted((*OSWORLD_COMPLETION_GATE_RULES, REQUIREMENTS_UNVERIFIED))
            config["requirement_ledger"] = {
                "instruction": OSWORLD_REQUIREMENTS_INSTRUCTION,
                "schema": deepcopy(REQUIREMENTS_SCHEMA),
                "render_items": REQUIREMENTS_RENDER_ITEMS,
                "text_limit": REQUIREMENT_TEXT_LIMIT,
                "rule": REQUIREMENTS_UNVERIFIED,
                "strikes": REQUIREMENTS_STRIKES,
                "block_line": {"missing": REQUIREMENTS_MISSING_TEXT, "empty": REQUIREMENTS_EMPTY_TEXT,
                               "problems": dict(REQUIREMENT_PROBLEM_TEXTS), "named_items": REQUIREMENTS_BLOCK_ITEMS,
                               "text_limit": REQUIREMENTS_BLOCK_TEXT_LIMIT},
            }
        if profile.step_budget_line is not None:
            config["step_budget_line"] = profile.step_budget_line
        # Method surface the profile's action agent carries of its own (the Qwen grounding
        # class-family rule, action_qwen.py), recorded where it is carried.
        config.update(deepcopy(getattr(profile.action_agent_class, "method_surface", None) or {}))
        return cls(
            orchestrator=orchestrator,
            support_set_store=support_set_store,
            screen_size=screen_size,
            retrieval_off=retrieval_off,
            backbone=backbone,
            config=config,
            pointer_mask_recorder=pointer_mask_recorder,
            timeline=timeline,
            backbone_type=backbone_type,
        )

    # ------------------------------------------------------------------
    # V2 agent contract — the outward boundary.
    # ------------------------------------------------------------------

    def reset(self, runtime_logger=None, **kwargs) -> None:
        """Per-episode state only: ``_api_log_dir`` and ``task_current_date``
        are set by the runner / driver BEFORE reset and must survive.
        A reset that closes an un-finalized episode (the driver's per-phase
        reset inside a multi-phase task) stashes the outgoing phase's record."""
        if not self._finalized and self.runtime_state is not None:
            self.phase_records.append(self._phase_record())
        self._finalized = False
        self.runtime_state = None
        self.last_decision = None
        self.last_update = None
        self.task_instruction = None
        self._pending_env_feedback = None
        self.action_parse_failure_count = 0
        self.reasoning_parse_failure_count = 0
        self.grounding_parse_failure_count = 0
        self.reasoning_decision_missing_count = 0
        self.reasoning_decision_missing_kinds = {}
        self.reasoning_resample_count = 0
        self.grounding_resample_count = 0
        self.resample_exhausted_reasoning = 0
        self.resample_exhausted_grounding = 0
        self.grounding_calls_made = 0
        self.grounding_calls_bypassed = 0
        self.grounding_inputs_stripped = 0
        self.grounding_strip_fallbacks = 0
        self.grounding_actions_from_function_name = 0
        self.grounding_arguments_repaired = 0
        self.grounding_line_texts_dropped = 0
        self.grounding_reply_texts_dropped = 0
        self.predict_calls = 0
        self.monitor_signals = []
        self._ask_step = None
        self.ask_user_count = 0
        self.agent_errors = []
        # Per-EPISODE, and per-PHASE within a multi-phase task: the control rebuilds
        # its screenshot list on every reset too, and a frame from the previous phase
        # would be re-sent as "this episode's opening screen".
        self.timeline.reset()
        # THE POINTER-MASK RECORDER IS DELIBERATELY NOT CLEARED HERE. It outlives the episode
        # (the runner rebuilds the agent only when a task's env spec changes, and one
        # VOLUME_SIZE default covers 23 of the subset's 28 tasks, so one recorder
        # spans most of a run), which makes clearing it look obviously right. It is
        # not: a clear here CANNOT CHANGE ANY RECORDED VALUE. `_masked_points` returns
        # before touching the recorder on any step where the monitor did not consult
        # it, and on the steps where it does read, the locator has just written the
        # slot — so the slot is empty or fresh at every read, and an episode boundary
        # cannot make it otherwise.
        #
        # The guards that DO fire are the branch gate in `_masked_points` and, under a
        # change to the shared monitor, the stamp and read-once inside
        # `_RecordingPointerLocator.take`. An inert guard standing beside those makes
        # the next reader think they are weaker than they are.
        # Snapshot the cumulative usage counters at episode START so the
        # per-episode delta can be recorded (the counters never reset).
        self._token_start = self._collect_token_usage()
        if self.backbone is not None and hasattr(self.backbone, "reset_session"):
            self.backbone.reset_session()

    def set_api_log_dir(self, path: str) -> None:
        """Runner hook (duck-typed, called before each task): the per-task
        model-call log location. The per-task telemetry is written next to it
        (its parent = the task result dir) at ``finalize_task``, and the
        per-call diagnostics go to ``<parent>/graphpc_calls/`` (the baseline's
        ``api_logs/`` counterpart; numbering restarts here).

        This call is also the TASK BOUNDARY: the runner makes it before the
        driver's reset, and its except branch never calls ``finalize_task``, so
        an episode that died mid-task is still live here. Its record is dumped
        into ITS OWN result dir (the previous ``api_log_dir``'s parent) as an
        orphan and the per-task accumulators are cleared, so nothing of it is
        attributed to the task that follows."""
        if not self._finalized and self.runtime_state is not None:
            self._close_orphaned_episode()
        self.phase_records = []
        self._finalized = True
        self._api_log_dir = str(path)
        self._call_records = 0
        self._stage_calls_this_step = {}

    def close_episode(self, *, reason: str, result_dir: Optional[str] = None) -> bool:
        """Runner hook (delta 8): dump the LIVE episode's record now, as an
        orphan (``persist: "orphaned_<reason>"``, no score), exactly as the
        ``set_api_log_dir`` task boundary would — for the runner's per-task
        except branch (the episode died: VM, evaluator, driver crash) and the
        worker's exit, where no next task boundary will ever come (a one-task
        smoke, the last task of a worker). Idempotent: a no-op after
        ``finalize_task``, after an earlier close, or when no episode is live.
        Returns whether a record was written. ``result_dir`` is the fallback
        location when no api_log_dir was ever set."""
        if self._finalized or self.runtime_state is None:
            return False
        self._close_orphaned_episode(reason=reason, result_dir=result_dir)
        self._finalized = True
        self.phase_records = []
        return True

    def predict(self, instruction: str, obs: dict[str, Any]) -> tuple[str, list[str]]:
        """One step of the V2 driver loop (``_predict``). With the profile's
        ``infra_error`` set, an exception that would leave it is classified: one the
        classifier calls infrastructure is re-raised (the driver skips the evaluation,
        the task has no result and is re-run); anything else ends the episode as V2's
        ``DONE``, so the unmodified driver evaluates the environment as it stands, and
        is recorded (``agent_errors``). Without it every exception leaves."""
        infra_error = model_profile(self.backbone_type).infra_error
        if infra_error is None:
            return self._predict(instruction, obs)
        try:
            return self._predict(instruction, obs)
        except Exception as exc:  # noqa: BLE001 — classified; infrastructure is re-raised
            if infra_error(exc):
                raise
            return self._lost_episode(exc)

    def _lost_episode(self, exc: Exception) -> tuple[str, list[str]]:
        """The agent_error record and the ``DONE`` that ends the episode evaluated."""
        message = str(exc)[:300]
        self.agent_errors.append({"class": "agent_error", "exception_type": type(exc).__name__, "message": message,
                                  "step": self.predict_calls})
        logger.warning("agent_error at predict call %d (%s: %s); ending the episode as DONE",
                       self.predict_calls, type(exc).__name__, message)
        return f"agent_error: {type(exc).__name__}: {message}", [DONE_TOKEN]

    def _predict(self, instruction: str, obs: dict[str, Any]) -> tuple[str, list[str]]:
        """One step of the V2 driver loop:

        1. If a prior decision is outstanding, close it via ``update_runtime``
           against ``obs`` (the env executed our prior action and produced it),
           with the previous step's failure (if any) riding on ``_env_feedback``
           so the monitor names it.
        2. Call ``decide_next_action`` on ``obs``; carry its runtime state.
        3. Translate ``ActionBatch.actions[0]`` to the driver's one-element
           action list.
        """
        screenshot = obs.get("screenshot")
        if model_profile(self.backbone_type).capture_failure_raises and not isinstance(screenshot, (bytes, bytearray)):
            raise ScreenshotCaptureFailed(
                f"the screenshot capture failed: obs['screenshot'] is {type(screenshot).__name__}, not bytes")
        self.predict_calls += 1
        self.task_instruction = instruction
        self._stage_calls_this_step = {}
        # Never mutate the driver's dict.
        obs_dict: dict[str, Any] = dict(obs)
        obs_dict[TASK_CURRENT_DATE_KEY] = self.task_current_date
        # The monitor reads metadata["env_feedback"]; V2's driver hands us only
        # obs, so the envelope is synthesized here and the previous step's
        # failure keys are merged into its info.
        obs_dict["_env_feedback"] = {
            "reward": 0,
            "done": False,
            "info": dict(self._pending_env_feedback or {}),
        }
        self._pending_env_feedback = None
        self._write_initial_screenshot(obs.get("screenshot"))
        # The deciding call's history: this step's frame is the screen after the
        # previous step, keyed by predict call so a failed capture is a hole in its
        # own slot rather than a shift of every later frame.
        self.timeline.record_frame(self.predict_calls, obs.get("screenshot"))
        # The driver's answer to the previous step's question (the ask channel): it
        # rides on this obs, and the next deciding request shows it with that step.
        if self._ask_step is not None:
            answer = obs.get("user_response")
            if answer is not None and str(answer).strip():
                self.timeline.record_user_response(self._ask_step, str(answer))
            self._ask_step = None

        # (1) close the prior step.
        if self.last_decision is not None:
            self.last_update = self.orchestrator.update_runtime(
                decision=self.last_decision,
                current_observation_input=obs_dict,
            )
            self.runtime_state = self.last_update.runtime_state
            self._record_monitor_signal(self.last_update)
            if self.last_update.monitor is not None:
                self.timeline.record_screen_change(self.predict_calls - 1, self.last_update.monitor.content)

        # The stuck-stop (OSWORLD_STUCK_STOP_BY_BACKBONE): read off the steps closed so far, the last one's diff
        # included, before any model call.
        stop = self._stuck_stop_verdict()
        if stop is not None:
            return self._stuck_stop(stop)

        # (2) decide; carry the fresh state forward so finalize can
        # persist a DONE episode (the terminal decision is never closed by
        # update_runtime — the driver stops calling predict).
        self.last_decision = self.orchestrator.decide_next_action(
            instruction=instruction,
            observation_input=obs_dict,
            runtime_state=self.runtime_state,
        )
        self.runtime_state = self.last_decision.runtime_state
        # The deciding call just labelled the previous step; the orchestrator's
        # verdict is that label reconciled against the monitor. The values it read
        # go into the episode's facts ledger, which every later request shows.
        self.timeline.record_label(self.predict_calls - 1, self.last_decision.content.get("verification"))
        if self.last_decision.reasoning is not None:
            self.timeline.record_facts(self.predict_calls, self.last_decision.reasoning.content.get("facts"))
        response_text = self._extract_response_text(self.last_decision)
        # Re-samples this step spent (extra samples beyond the first call); each
        # one was a rejected sample, so the per-attempt failure counters advance by
        # the same amount — on the deciding side split into the samples that carried
        # no decision (counted apart) and the unparseable rest, the final failed
        # attempt of an exhausted step included. Grounding's exhausted step adds its
        # final attempt below, at the degrade marker.
        reasoning = self.last_decision.reasoning
        reasoning_resamples = resample_attempts_of(reasoning)
        # The Qwen class-family rule's repairs (action_qwen.py) are none of these counters' samples:
        # they repair replies the control would execute, so they stay out of the counters printed
        # beside the control's, and their exhausted step is a WAIT of its own reason below.
        grounding_resamples = resample_attempts_of(self.last_decision.action_batch) - int(
            (getattr(self.last_decision.action_batch, "metadata", None) or {}).get("class_mismatch_repairs") or 0)
        reasoning_exhausted = int(bool(reasoning is not None and reasoning.content.get("reasoning_parse_error")))
        decision_missing = int((reasoning.metadata or {}).get("decision_missing_attempts") or 0) \
            if reasoning is not None else 0
        self.reasoning_resample_count += reasoning_resamples
        self.grounding_resample_count += grounding_resamples
        self.reasoning_decision_missing_count += decision_missing
        for kind in (reasoning.metadata or {}).get("decision_missing_kinds") or [] if reasoning is not None else []:
            self.reasoning_decision_missing_kinds[kind] = self.reasoning_decision_missing_kinds.get(kind, 0) + 1
        self.reasoning_parse_failure_count += reasoning_resamples + reasoning_exhausted - decision_missing
        self.grounding_parse_failure_count += grounding_resamples
        # Which route produced this step's action. Read off the batch rather than
        # inferred from the class, so a bypass that stops firing shows up here rather
        # than being assumed. A bypassed batch carries no resample attempts, so this
        # cannot double-count against the counters just advanced.
        self._count_grounding_route(self.last_decision.action_batch)
        for record in (self.last_decision.action_batch.metadata or {}).get("grounding_input_strip") or []:
            self.grounding_inputs_stripped += bool(record.get("stripped"))
            self.grounding_strip_fallbacks += len(record.get("fallback") or [])
        self.grounding_actions_from_function_name += int(
            (self.last_decision.action_batch.metadata or {}).get("actions_from_function_name") or 0)
        batch_metadata = self.last_decision.action_batch.metadata or {}
        self.grounding_arguments_repaired += int(batch_metadata.get("replies_repaired") or 0)
        self.grounding_line_texts_dropped += int(batch_metadata.get("line_texts_dropped") or 0)
        self.grounding_reply_texts_dropped += int(batch_metadata.get("reply_texts_dropped") or 0)

        # (3) translate.
        reasoning_msg = self.last_decision.reasoning
        if reasoning_msg is not None and reasoning_msg.content.get("reasoning_parse_error"):
            self._pending_env_feedback = {"reasoning_parse_error": reasoning_msg.content["reasoning_parse_error"]}
            self.resample_exhausted_reasoning += 1
            return response_text, self._wait("reasoning_parse_error")

        action_batch = self.last_decision.action_batch
        if action_batch.content.get("completion_gate_blocked"):
            failed_rule = action_batch.content.get("completion_gate_failed_rule") or "no_grounded_evidence"
            self._pending_env_feedback = {"completion_gate_blocked": failed_rule}
            return response_text, self._wait("completion_gate_blocked")
        if action_batch.content.get("grounding_parse_error"):
            self._pending_env_feedback = {"grounding_parse_error": action_batch.content["grounding_parse_error"]}
            # A bypassed intent that could not be executed also says WHICH field was
            # missing or WHICH value was unpressable. Without it the WAIT is silent
            # and the reasoning call re-issues the same broken intent; with it this
            # is strictly better than a grounding call, where an empty `type` intent
            # comes back as a confident click.
            defect = action_batch.content.get("intended_action_defect")
            if defect:
                self._pending_env_feedback["intended_action_defect"] = defect
            # A grounding step none of whose replies was usable says so on its line, where the
            # action agent records it (the DeepSeek profile: action_deepseek.GROUNDING_REPLY_UNUSABLE).
            unusable = action_batch.content.get("grounding_reply_unusable")
            if unusable:
                self._pending_env_feedback["grounding_reply_unusable"] = unusable
            self.grounding_parse_failure_count += 1
            self.resample_exhausted_grounding += 1
            return response_text, self._wait("grounding_parse_error")
        if action_batch.content.get("grounding_class_mismatch"):
            # The Qwen grounding class-family rule ran out of attempts (action_qwen.py): its own
            # reason, apart from grounding_parse_error, and its own line naming the class that came
            # back instead of the decided one.
            self._pending_env_feedback = {"grounding_class_mismatch": action_batch.content["grounding_class_mismatch"]}
            return response_text, self._wait("grounding_class_mismatch")
        if not action_batch.actions:
            return response_text, self._wait("empty_action_batch")

        action = action_batch.actions[0]
        action_name = str(action.action_type or "").lower()
        if action_name == ASK_USER_ACTION and model_profile(self.backbone_type).ask_user_channel:
            # The driver's ASK_USER path: an empty action list with the question as the
            # response; its answer comes back on the next obs (``user_response``).
            question = str((action.parameters or {}).get("text") or "")
            self._record_executed_action(action, command=ASK_USER_COMMAND, status="ask")
            self._ask_step = self.predict_calls
            self.ask_user_count += 1
            return question, []
        if action_name == "done" and (action.metadata or {}).get("terminal_intent") == "fail":
            # A give-up the completion gate admitted (it reaches here as a `done`
            # so the gate sees it; MiniMaxActionAgent._terminal_action), emitted as the
            # profile's give-up token.
            give_up_token = model_profile(self.backbone_type).give_up_token
            self._record_executed_action(action, command=give_up_token, status="terminal")
            return response_text, [give_up_token]
        if action_name == "done":
            # A done that is a pure infeasibility declaration is the FAIL
            # dialect on this benchmark (V2 scores infeasible tasks only on
            # FAIL; a DONE scores 0). Audited; the decision itself is untouched.
            statement, source = self._terminal_text(self.last_decision, action)
            # THE OSWORLD VARIANT, and the ONE place a give-up that arrives as a
            # `done` is converted (the gate's give-up rules read the same rule). The deciding
            # prompt does not offer that route — infeasibility is the `fail` action
            # (prompts.py) — so this is a FALLBACK and an audit: a give-up typically arrives
            # as a declaration followed by an explanation, which the anchored rule never
            # matches. Anything that needs to COUNT give-ups reads the FAIL this produces;
            # it does not re-run the matcher.
            rule = infeasible_declaration_rule(statement)
            if rule is not None:
                self.runtime_state.content.setdefault("_infeasible_normalized_events", []).append(
                    {"step_idx": self.runtime_state.step_idx, "text": statement, "rule": rule, "source": source}
                )
                give_up_token = model_profile(self.backbone_type).give_up_token
                self._record_executed_action(
                    action, command=give_up_token, status="terminal", reason="infeasible_declaration",
                )
                return response_text, [give_up_token]
            self._record_executed_action(action, command=DONE_TOKEN, status="terminal")
            return response_text, [DONE_TOKEN]

        screen_size = screenshot_size(obs.get("screenshot")) or self.screen_size
        translate_action = model_profile(self.backbone_type).translate_action
        try:
            command = translate_action(action, screen_size=screen_size)
        except ValueError as exc:
            # An action the model's translator rejects is a grounding failure, not a
            # runner crash: degrade to WAIT (episode CONTINUES) and signal the
            # failure upstream on the NEXT step — the reasoning loop is our
            # recovery path, so no blind LLM retry (decomposition tax).
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("action grounding failure (%s); degrading to WAIT", reason)
            self._pending_env_feedback = {"action_parse_error": reason}
            self.action_parse_failure_count += 1
            return response_text, self._wait("action_parse_error", attempted=action)
        if not isinstance(command, str) or not command:
            # Interface contract: the translator returns pyautogui
            # source or a literal token, never empty/None. Loud here rather
            # than as an opaque failure inside the guest controller.
            raise TypeError(f"{translate_action.__name__} returned {command!r} for action {action.action_type!r}")
        self._record_executed_action(action, command=command, status="submitted")
        return response_text, [command]

    # ------------------------------------------------------------------
    # Persistence + telemetry — the runner's post-episode hook calls this.
    # ------------------------------------------------------------------

    def finalize_task(
        self,
        *,
        task_id: str,
        score: Optional[float],
        result_payload: Optional[dict[str, Any]] = None,
        result_dir: Optional[str] = None,
    ) -> Optional[Path]:
        """Persist the trajectory + graph artifacts to the support set iff the
        episode scored full completion and is single-phase, after the
        relabel write-back; then dump the per-task telemetry regardless of
        outcome. ``result_payload`` is the runner's read of the result files:
        ``{"score", "result_json", "checkpoint_results", "phase_results"}``;
        it decides admission and is recorded in the telemetry, but is
        NOT stored on the entry — the entry's ``metadata`` is injected into
        later prompts (see the write site). A multi-phase task
        (``phase_results`` not None) is never persisted — the live state is
        the last phase only and the score is the whole-task sum. Returns the
        persisted entry path (``None`` when nothing was stored)."""
        payload = result_payload or {}
        entry_path: Optional[Path] = None
        if self.retrieval_off:
            # No store write, and say so on the episode record rather than
            # skipping silently.
            if self.runtime_state is not None:
                self.runtime_state.content["_store_write_skipped"] = "retrieval_off"
            persist = "skipped_retrieval_off"
            logger.info("store write skipped for %s (retrieval_off)", task_id)
        elif payload.get("phase_results") is not None:
            if self.runtime_state is not None:
                self.runtime_state.content["_store_write_skipped"] = "multi_phase"
            persist = "skipped_multi_phase"
            logger.info("store write skipped for %s (multi-phase task)", task_id)
        elif score is None:
            persist = "no_score"
        elif not _is_full_score(score):
            persist = "score_below_full"
        elif self.runtime_state is None:
            persist = "no_runtime_state"
        elif self.support_set_store is None:
            persist = "no_store"
        else:
            # Relabel write-back (design.md §6): reconcile the episode's
            # actual-effect labels against its M1-reconciled verdicts BEFORE
            # persistence — mirrors WebArenaGraphPCAgent.finalize_task so the
            # cross-benchmark grounding quality doesn't diverge. Mutates the
            # history in place before the deepcopy below; ledger on
            # _relabel_events.
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
                "result": score,
                "current_stage": self.runtime_state.current_stage,
                "plan": self._serialize_plan(self.runtime_state.plan),
                "recent_history": deepcopy(self.runtime_state.recent_history),
                "observation": self._build_support_observation_payload(),
                # Everything here rides into the DECIDING PROMPT of every later
                # task: the injection diet drops `graph_artifacts`, `observation`
                # and `signatures` but not `metadata`. So this dict is a prompt
                # surface, not a record, and only recipe-shaped facts belong in
                # it. `result_payload` (the runner's read of result.json /
                # checkpoint_results.json / phase_results.json — the BENCHMARK
                # EVALUATOR'S own per-criterion output for a previous task) and
                # the whole `config` dict are kept out for that reason; both are
                # written in full to graphpc_telemetry.json
                # (`result_payload`, `active_config`), so nothing is lost for
                # analysis. Kept out HERE rather than in the shared drop list
                # because what is at stake is what this adapter chooses to store —
                # a WebArena entry's `metadata` is small — and widening the shared
                # list would change WebArena prompts for no reason.
                "metadata": {
                    "score": score,
                    "model": self._model_name(),
                    "retrieval_off": bool(self.retrieval_off),
                    "screen_size": list(self.screen_size),
                    # A FAIL on an infeasible task scores 1.0, so a persisted
                    # recipe can be a give-up: name the terminal action.
                    "terminal_action": self._terminal_action(),
                },
            }
            graph_artifacts = self._build_graph_artifacts()
            if graph_artifacts:
                entry["graph_artifacts"] = graph_artifacts
            entry_path = self.support_set_store.add_success_entry(entry)
            persist = "persisted"

        self._write_telemetry(
            task_id=task_id, score=score, result_payload=payload,
            result_dir=result_dir, entry_path=entry_path, persist=persist,
        )
        self._finalized = True
        self.phase_records = []
        return entry_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stuck_stop_verdict(self) -> Optional[dict[str, Any]]:
        """``stuck_stop_fire`` over this phase's executed-action rows joined by step_idx to the monitor's masked
        diffs, on a backbone that has the rule; None otherwise."""
        rule = OSWORLD_STUCK_STOP_BY_BACKBONE.get(self.backbone_type)
        if rule is None or self.runtime_state is None or self.predict_calls <= rule["min_step"]:
            return None
        diffs = {signal.get("step_idx"): signal.get("diff_pixel_count") for signal in self.monitor_signals}
        history = [stuck_step(row, diffs.get(row.get("step_idx")))
                   for row in self.runtime_state.content.get("_executed_actions") or []]
        return stuck_stop_fire(history, self.predict_calls, rule)

    def _stuck_stop(self, stop: dict[str, Any]) -> tuple[str, list[str]]:
        """End the episode with the profile's give-up token, without a model call and past the completion gate: a
        controller decision, not a model claim. The step's executed-action row (status terminal, reason
        stuck_stop, the step and the window counts; no decided class) is written before the return, so the
        telemetry records it."""
        token = model_profile(self.backbone_type).give_up_token
        facts = {"step": self.predict_calls, **stop}
        self.last_decision = None   # nothing was decided at this step
        self._record_executed_action(
            Action(action_type="done", parameters={}, raw={},
                   metadata={"decided_action_class": None, STUCK_STOP_REASON: facts}),
            command=token, status="terminal", reason=STUCK_STOP_REASON,
        )
        logger.warning("stuck_stop at predict call %d: %s; ending the episode as %s", self.predict_calls, stop, token)
        return f"{STUCK_STOP_REASON}: {json.dumps(facts, ensure_ascii=False)}", [token]

    def _wait(self, reason: str, *, attempted: Optional[Action] = None) -> list[str]:
        """The env-accepted no-op (counts a driver step); ``reason`` names the
        degrade path for the per-step action record, and ``attempted`` is the
        action the environment did NOT get, where there was one."""
        self._record_executed_action(
            Action(action_type="wait", parameters={}, raw={}), command=WAIT_TOKEN, status="noop", reason=reason,
            attempted=attempted,
        )
        return [WAIT_TOKEN]

    def _record_executed_action(
        self, action: Action, *, command: str, status: str, reason: Optional[str] = None,
        attempted: Optional[Action] = None,
    ) -> None:
        """Append this step's concrete driver action (pyautogui source or env
        token) to ``_executed_actions`` on the runtime state — episode
        telemetry; the dump carries every runtime_content key — and to the
        deciding call's timeline, as the line later steps are shown. Recorded
        HERE because this is the boundary where the string the driver will
        actually execute is in hand."""
        reasoning = self.last_decision.reasoning if self.last_decision is not None else None
        content = reasoning.content if reasoning is not None else {}
        typed = (action.parameters or {}).get("text") if action.action_type == "type" and status != "noop" else None
        self.timeline.record_step(
            self.predict_calls,
            executed=self._executed_text(action, status=status, reason=reason, attempted=attempted),
            # What a `type` action typed: the orchestrator's named_app_bypassed_by_command_line
            # and synthetic_media_deliverable rules read it.
            typed_text=typed if isinstance(typed, str) else "",
            # What kind of action executed: the timeline's held-type rule reads it.
            action_type=str(action.action_type or ""),
            # A degraded decision's reasoning_text is the harness's own degrade note,
            # not the model's reasoning; the WAIT line already says what happened.
            reasoning_text="" if content.get("reasoning_parse_error") else str(content.get("reasoning_text") or ""),
            ask=status == "ask",
        )
        if self.runtime_state is None:
            return
        record: dict[str, Any] = {
            "step_idx": self.runtime_state.step_idx,
            "action_type": action.action_type,
            "command": command,
            "status": status,
        }
        if reason is not None:
            record["reason"] = reason
        # MEASUREMENT INTEGRITY: telemetry has to distinguish the model-grounded path
        # from the bypassed one, and a bypass that is invisible per step cannot be
        # audited. Three facts, only on the steps that have them, so the row shape is
        # unchanged everywhere else: whether a model call produced this action, which
        # intent field supplied its argument (the only place a corrupted intent would
        # show, since the bypass translates whatever it is given without complaint),
        # and the action_class as EMITTED, before normalisation — which is the only
        # way to tell later whether the schema's advisory vocabulary is shrinking the
        # alias tail, and so the only way that hint ever gets evaluated.
        for key in ("model_call", "intended_action_field", "intended_action_class_raw",
                    "terminal_intent", "grounding_composite", "decided_action_class",
                    "stuck_stop"):
            if key in (action.metadata or {}):
                record[key] = action.metadata[key]
        # The coordinate-strip audit of this step's grounding line (what was removed,
        # what fell back), on the steps that made a grounding call — WAITs from a
        # rejected grounding included.
        batch = self.last_decision.action_batch if self.last_decision is not None else None
        strip = (getattr(batch, "metadata", None) or {}).get("grounding_input_strip")
        if strip:
            record["grounding_input_strip"] = strip
        # The Qwen class-family rule's record (action_qwen.py): the step's decided class, normalised,
        # beside the executed action_type, on every row (a composite is marked by its
        # grounding_composite, copied above); the batch carries it, so only that profile's rows have
        # the key.
        if "decided_action_class" in (getattr(batch, "metadata", None) or {}):
            record["decided_action_class"] = batch.metadata["decided_action_class"]
        self.runtime_state.content.setdefault("_executed_actions", []).append(record)

    def _action_text(self, action: Action) -> str:
        """An executed action as the model knows actions: the M3 tool call (name +
        arguments), plus the target the deciding call named for it — without the
        target a line outside the image window says only WHERE something was
        clicked, never WHAT.

        The target is the one the grounding line carried: coordinates the deciding
        call wrote into it are stripped by the same function
        (``strip_grounding_coordinates``), so the line describes what was handed on
        and the history does not feed prose coordinates back to the next decision."""
        text = str(action.action_type)
        if action.parameters:
            text += " " + json.dumps(action.parameters, ensure_ascii=False)
        # A scroll or drag that carries its own pointer move (Qwen's composite, action_qwen.py)
        # says where it scrolled ("at") or where the drag started ("from"), in the coordinates a
        # mouse_move line shows.
        composite = (action.metadata or {}).get("grounding_composite")
        if composite:
            text += (" from " if action.action_type == "left_click_drag" else " at ") \
                + json.dumps(composite["mouse_move"].get("coordinate"), ensure_ascii=False)
        reasoning = self.last_decision.reasoning if self.last_decision is not None else None
        intended = (reasoning.content if reasoning is not None else {}).get("intended_actions")
        target = strip_grounding_coordinates(intended[0])[0].get("target") \
            if isinstance(intended, list) and intended and isinstance(intended[0], dict) else None
        if isinstance(target, str) and target.strip():
            text += f" on {json.dumps(target.strip(), ensure_ascii=False)}"
        return text

    def _executed_text(
        self, action: Action, *, status: str, reason: Optional[str], attempted: Optional[Action],
    ) -> str:
        """What the environment executed at this step, as the timeline shows it. A
        WAIT says WHY the decided action did not run — built from the same feedback
        dict the monitor reads (``_pending_env_feedback``, set before ``_wait``), so
        the degrade the reasoner is told about is the degrade that happened. Without
        it, only the monitor's progress status would reach the deciding prompt."""
        if status == "ask":
            return f"asked the user: {(action.parameters or {}).get('text') or ''}"
        if status != "noop":
            return self._action_text(action)
        feedback = self._pending_env_feedback or {}
        if reason == "completion_gate_blocked":
            # `done` or `fail`: both reach the gate as a done; the decision says which. A
            # profile's give-up block text stands in for the two give-up rules' lines.
            rule = feedback.get("completion_gate_blocked")
            block_text = model_profile(self.backbone_type).give_up_block_text
            rule_text = (f"({block_text})" if block_text and rule in (FAIL_RESTS_ON_STEP_BUDGET, UNVERIFIED_BARRIER)
                         else self._gate_rule_text(rule))
            detail = f"your `{self._decided_terminal_class()}` was blocked by the completion gate {rule_text}"
        elif reason == "action_parse_error":
            tried = self._action_text(attempted) if attempted is not None else "the action"
            detail = f"{tried} could not be executed: {feedback.get('action_parse_error')}"
        elif reason == "grounding_parse_error" and feedback.get("grounding_reply_unusable"):
            # The fixed line (decider-visible method surface of the DeepSeek profile),
            # from the one profile that records it, imported at call time as model_profile does.
            from .action_deepseek import GROUNDING_REPLY_UNUSABLE_LINE  # noqa: PLC0415
            unusable = feedback.get("grounding_reply_unusable") or {}
            detail = GROUNDING_REPLY_UNUSABLE_LINE.format(n=unusable.get("attempts"), decided=unusable.get("decided"))
        elif reason == "grounding_parse_error":
            detail = f"grounding produced no executable action: {feedback.get('grounding_parse_error')}"
            defect = feedback.get("intended_action_defect")
            if defect and defect not in detail:
                detail += f" ({defect})"
        elif reason == "grounding_class_mismatch":
            # The fixed line (decider-visible method surface of the Qwen profile),
            # from the one profile that degrades this way, imported at call time as model_profile does.
            from .action_qwen import CLASS_MISMATCH_LINE  # noqa: PLC0415
            mismatch = feedback.get("grounding_class_mismatch") or {}
            detail = CLASS_MISMATCH_LINE.format(n=mismatch.get("attempts"), got=mismatch.get("got"),
                                                decided=mismatch.get("decided"))
        elif reason == "reasoning_parse_error" and feedback.get("reasoning_parse_error") == DECISION_MISSING_REASON:
            detail = "your reply carried no decision (the schema itself, or no intended_actions)"
        elif reason == "reasoning_parse_error":
            detail = "your reply could not be parsed as the JSON object the schema asks for"
        elif reason == "empty_action_batch":
            detail = "no action was decided (intended_actions was empty)"
        else:
            detail = str(reason)
        return f"WAIT — {detail}"

    def _gate_rule_text(self, rule: Any) -> str:
        """The parenthesis the blocked step's line carries: a shared rule's name, or the
        fact an OSWorld rule (orchestrator.py) recorded — never the step cap or the steps
        remaining (which no MiniMax or Qwen request shows; on DeepSeek the STATE block's
        budget line states them, never a blocked step's line)."""
        if rule == FAIL_RESTS_ON_STEP_BUDGET:
            return "(running out of steps is not one of the infeasibility conditions)"
        if rule == UNVERIFIED_BARRIER:
            return ("(a barrier that has not been verified is not one of the infeasibility conditions; "
                    "try reasonable approaches first)")
        facts = (self.runtime_state.content.get("_completion_gate_facts")
                 if self.runtime_state is not None else None) or []
        fact = facts[-1] if facts and facts[-1].get("rule") == rule else {}
        if rule == NAMED_APP_BYPASSED_BY_COMMAND_LINE:
            # What was detected, and nothing more: the deliverable, the CLI, and the app
            # the instruction directs the use of (orchestrator.directed_applications). The
            # wording is the one a replay of the blocked step tested; the named input stays in the fact.
            return (f"({posixpath.basename(str(fact.get('path') or ''))} was written by {fact.get('tool')}; "
                    f"the instruction says to use {' and '.join(fact.get('applications') or [])})")
        if rule == USE_FILE_CLAIMED_DONE:
            # The replay-chosen line (orchestrator.py, use_file_claimed_done).
            return (f"(the instruction says to use {fact.get('file')} to complete the {fact.get('object')}; "
                    f"{fact.get('file')} is an input, not the {fact.get('object')})")
        if rule == SYNTHETIC_MEDIA_DELIVERABLE:
            # The record, and nothing more: the deliverable, its tool and inputs, who drew
            # each input at which step, and the first application the instruction directs.
            drawn = " and ".join(f"{source} was drawn by {producer} at step {step}"
                                 for source, producer, step in fact.get("drawn") or [])
            return (f"({posixpath.basename(str(fact.get('path') or ''))} was written by {fact.get('tool')} from "
                    f"{' and '.join(fact.get('inputs') or [])} only, and {drawn}; the instruction says to use "
                    f"{fact.get('application')})")
        if rule == REQUIREMENTS_UNVERIFIED:
            return requirements_block_text(fact)
        return f"({rule})"

    def _decided_terminal_class(self) -> str:
        """The terminal class this step's decision named (``done`` or ``fail``), read
        off the deciding call's first intended action; ``done`` when unreadable."""
        reasoning = self.last_decision.reasoning if self.last_decision is not None else None
        intended = (reasoning.content if reasoning is not None else {}).get("intended_actions")
        first = intended[0] if isinstance(intended, list) and intended and isinstance(intended[0], dict) else {}
        return "fail" if normalize_action_class(first.get("action_class")) == "fail" else "done"

    def _write_initial_screenshot(self, screenshot: Any) -> None:
        """Write the episode's opening frame as ``step_0_initial.png``, once.

        PARITY WITH THE CONTROL, and it is a gap in the RECORD rather than in
        behaviour. Upstream's AGENT writes this, not the driver
        (``mm_agents/m3/agent.py``, guarded on ``_api_call_count == 0`` and an
        api-log dir), so every control task directory has it. Frames are written
        POST-action, so without an opening frame the FIRST action of every episode
        has no predecessor and its effect cannot be measured at all — on the control
        it can.

        Condition and filename follow upstream exactly, including the already-exists
        check. BEST-EFFORT BY CONSTRUCTION: a disk or path problem writing a forensic
        artefact must never take down a scored run, which is why this swallows
        everything, as upstream's does.

        WHY IT CANNOT LAND UNDER THE PREVIOUS TASK'S DIRECTORY. ``predict_calls`` is
        zeroed in ``reset``, which the driver calls per episode, and incremented at
        the top of ``predict``, so ``== 1`` is the first observation of THIS episode
        rather than a counter carried across tasks; the runner calls
        ``set_api_log_dir`` per task, so the path is the current one. And if
        ``_api_log_dir`` were ever stale, the target already exists from that episode
        and the write is a no-op — the already-exists check covers the residual case,
        which is a second reason to keep it rather than only upstream fidelity."""
        if self.predict_calls != 1 or not self._api_log_dir or not screenshot:
            return
        try:
            target = Path(os.path.dirname(self._api_log_dir)) / "step_0_initial.png"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bytes(screenshot))
        except Exception:  # noqa: BLE001 — a forensic artefact never fails an episode
            logger.exception("failed to write step_0_initial.png")

    def _count_grounding_route(self, action_batch) -> None:
        """One tally per grounded/bypassed action in the batch.

        A terminal action is neither: ``done`` / ``fail`` never had a grounding call
        to make, so counting them would inflate the bypassed side with steps the
        bypass did not cause. A question (``call_user``) is neither for the same
        reason. The completion gate's corrective no-op is neither too: it replaces a
        blocked terminal claim, so no grounding call was sent and no bypass caused it."""
        if (getattr(action_batch, "content", None) or {}).get("completion_gate_blocked"):
            return
        for action in getattr(action_batch, "actions", None) or []:
            if str(action.action_type or "").lower() in ("done", "fail", ASK_USER_ACTION):
                continue
            if (action.metadata or {}).get("model_call") is False:
                self.grounding_calls_bypassed += 1
            else:
                self.grounding_calls_made += 1

    def _record_monitor_signal(self, update: GraphPCRuntimeUpdate) -> None:
        """One row per closed step, into ``graphpc_telemetry.json``.

        THE POINTER MASK'S AUDIT PAIR LANDS HERE OR NOWHERE. The monitor computes
        the unmasked count beside the masked one on every screenshot step, but
        nothing downstream kept it: ``recent_history`` compacts the monitor to its
        ``progress_status`` and the orchestrator's ``_monitor_history`` keeps two
        fields, so without this row a run's record would hold the MASKED count
        alone. The mask's failure modes are detectable afterwards but NOT
        recoverable, so an episode whose record cannot produce the pair cannot be
        judged on its mask at all.

        The mask keys are written only where the monitor emitted them, i.e. where
        a pointer locator is configured, so the row shape is unchanged on a
        benchmark without one.

        KNOWN AND DELIBERATE: ``monitor_signals`` reaches disk at episode close,
        so an episode killed mid-flight leaves no mask record at all. Those
        episodes are unscored and drop out of the paired comparison, so nothing is
        restructured for them — but no check may read an absent record as a clean
        one."""
        monitor = update.monitor
        if monitor is None:
            return
        content = monitor.content or {}
        # The decision just closed; update_runtime advanced step_idx after the
        # monitor ran, so this is also the index the locator was consulted under.
        closed_step = update.runtime_state.step_idx - 1
        signal: dict[str, Any] = {
            "step_idx": closed_step,
            "signal": content.get("signal"),
            "changed": content.get("changed"),
            "diff_pixel_count": content.get("diff_pixel_count"),
            "diff_fraction": content.get("diff_fraction"),
            "progress_status": content.get("progress_status"),
            "observed_change": content.get("observed_change"),
        }
        bypass = self._closed_step_bypass(closed_step)
        if bypass is not None:
            signal.update(bypass)
            # WHAT IS RECORDED HERE. `grounding_bypassed` is the one
            # fact about this step that CANNOT be recovered after the run — the saved
            # frames do not say which route produced the action.
        if "diff_pixel_count_unmasked" in content:
            signal["diff_pixel_count_unmasked"] = content.get("diff_pixel_count_unmasked")
            # The monitor's own witness that the mask ran, kept next to the
            # recorder's witness of WHAT it was: the two are computed on different
            # sides of the call, so a disagreement between them is itself a finding.
            signal["pointer_mask_applied"] = content.get("pointer_mask_applied")
            # ASK FOR A REGION ONLY WHERE THE LOCATOR RAN THIS STEP, and only where
            # the monitor's own witness agrees with what came back.
            signal["pointer_mask_points"] = self._masked_points(closed_step, content)
        self.monitor_signals.append(signal)

    def _closed_step_bypass(self, step_idx: int) -> Optional[dict[str, Any]]:
        """The bypass provenance of the action executed at ``step_idx``, or None.

        Read back off ``_executed_actions`` rather than threaded through the monitor,
        because that record is written at the boundary where the executed string is in
        hand and already carries the fields. Returns None for a step with no such
        record (a degraded WAIT), so the row shape is unchanged there."""
        if self.runtime_state is None:
            return None
        for record in reversed(self.runtime_state.content.get("_executed_actions") or []):
            if not isinstance(record, dict) or record.get("step_idx") != step_idx:
                continue
            if "model_call" not in record:
                return None
            return {
                "grounding_bypassed": record["model_call"] is False,
                "bypassed_action_type": record.get("action_type"),
                "intended_action_field": record.get("intended_action_field"),
            }
        return None

    def _masked_points(self, step_idx: int, monitor_content: dict[str, Any]) -> Optional[list[list[int]]]:
        """The coordinates the mask actually covered on ``step_idx``. ``None`` means
        "not recorded"; ``[]`` means the locator ran and deliberately masked nothing
        (a coordinate-free step). The two must not collapse into each other — one is
        a gap in the record, the other is a recorded fact about the step — and
        neither may become a coordinate belonging to some other step.

        THE QUESTION IS "DID THE LOCATOR RUN ON THIS STEP", not "when did it last
        run". The monitor emits its mask keys whenever a locator is CONFIGURED, but
        calls one only inside its screenshot branch, so a step whose capture failed
        (the errno-28 defect ``--volume_size`` exists to prevent) reaches here having
        consulted nothing — and a stamp comparison cannot tell, because ``step_idx``
        restarts every episode and one recorder spans most of a run.

        Both of the monitor's witnesses are used, and neither alone is enough:

        * ``signal == "screenshot"`` is the branch that calls the locator. It is the
          gate, because it is TRUE on a coordinate-free step where the locator ran
          and returned nothing — which must record ``[]``.
        * ``pointer_mask_applied`` is ``bool(pointer_points)`` computed on the
          monitor's side of the call. Alone it would be the wrong gate: it is
          ``False`` both for a step that never consulted the locator AND for that
          coordinate-free step, so gating on it collapses ``[]`` into ``None``. Used
          as a CROSS-CHECK it earns its place — the two values are computed on
          opposite sides of the call, so a disagreement means the recorder and the
          monitor are not describing the same step, and the honest record of that is
          "not recorded" rather than a region nothing corroborates."""
        recorder = self._pointer_mask_recorder
        if recorder is None or monitor_content.get("signal") != "screenshot":
            return None
        points = recorder.take(step_idx)
        if points is None or bool(points) != bool(monitor_content.get("pointer_mask_applied")):
            return None
        return points

    @staticmethod
    def _terminal_text(decision: GraphPCDecision, action: Action) -> tuple[str, Optional[str]]:
        """``(statement, source)`` for a done: the action's parameters, its raw
        tool input, else the reasoning's first intended action (where the
        contract puts the "brief statement"). ``source`` names the holder and
        key (e.g. ``"parameters.text"``, ``"intended_action.intent"``) for the
        audit event; ``("", None)`` when the done carries no text."""
        text, key = _first_text(action.parameters)
        if text:
            return text, f"parameters.{key}"
        if isinstance(action.raw, dict):
            text, key = _first_text(action.raw.get("input"))
            if text:
                return text, f"raw_input.{key}"
        if decision.reasoning is not None:
            intended = decision.reasoning.content.get("intended_actions")
            if isinstance(intended, list) and intended:
                text, key = _first_text(intended[0])
                if text:
                    return text, f"intended_action.{key}"
        return "", None

    def _close_orphaned_episode(self, *, reason: str = "no_finalize", result_dir: Optional[str] = None) -> None:
        """An episode that ended without ``finalize_task`` (the runner's except
        branch: VM death, evaluator crash; or the worker exiting). Its telemetry
        goes into its OWN result dir with ``persist: "orphaned_<reason>"`` and
        no score, so it stays visible for forensics but is never attributed to
        the next task. Nothing is written when no result dir is known (no prior
        ``set_api_log_dir`` and no ``result_dir`` given)."""
        own_dir = os.path.dirname(self._api_log_dir) if self._api_log_dir else result_dir
        if own_dir:
            task_id = "/".join(Path(own_dir).parts[-2:])
        else:
            task_id = "unknown"
        logger.warning(
            "episode %s ended without finalize_task (%s; %d predict calls); recording it as an orphan",
            task_id, reason, self.predict_calls,
        )
        self._write_telemetry(
            task_id=task_id, score=None, result_payload={}, result_dir=result_dir,
            entry_path=None, persist=f"orphaned_{reason}",
        )

    def _on_model_call(self, result: dict[str, Any]) -> None:
        """Backbone observer: one JSON per model call under
        ``<task result dir>/graphpc_calls/NNN_<stage>.json`` — stage
        (reasoning | grounding, from the backbone's route), the attempt index
        within this step's stage (2+ = a re-sample in single-pass), prompt sizes,
        the full response text, usage and latency — and, from the backbone's
        ``request_record``, what the request CARRIED: the image slot map (step
        label, predict call and source sha1 per image, in request order), the
        cached prompt tokens, any thinking the provider returned, and, on a
        deciding call, the line it carried about the previous step (the feedback
        the model was given). The request's text parts go to the gzipped sibling
        ``NNN_<stage>.request.json.gz``. Best-effort: never raises."""
        try:
            diagnostics = dict(result.get("diagnostics") or {})
            route = str(diagnostics.get("route") or "")
            stage = "grounding" if route.startswith("grounding") else "reasoning"
            self._stage_calls_this_step[stage] = self._stage_calls_this_step.get(stage, 0) + 1
            self._call_records += 1
            if not self._api_log_dir:
                return
            record = {
                "call": self._call_records,
                "stage": stage,
                "attempt": self._stage_calls_this_step[stage],
                "predict_call": self.predict_calls,
                "route": route,
                "thinking": diagnostics.get("thinking"),
                "model": diagnostics.get("model"),
                "prompt_chars": {"system": diagnostics.get("system_chars"), "user": diagnostics.get("user_chars")},
                "image_bytes": diagnostics.get("image_bytes"),
                "latency_s": diagnostics.get("latency_s"),
                "usage": result.get("usage"),
                "response_text": result.get("raw_response"),
                "tool_calls": result.get("response_output") or [],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            target_dir = Path(os.path.dirname(self._api_log_dir)) / CALL_LOG_DIRNAME
            stem = f"{self._call_records:03d}_{stage}"
            request = result.get("request_record")
            if isinstance(request, dict):
                slots = list(request.get("image_slots") or [])
                for slot in slots:
                    # The grounding route's one image is this predict call's frame.
                    if slot.get("predict_call") is None:
                        slot["predict_call"] = self.predict_calls
                thinking = request.get("thinking")
                record.update({
                    "image_count": len(slots),
                    # The current screen is the last image of a deciding request.
                    "history_image_count": max(0, len(slots) - 1) if stage == "reasoning" else 0,
                    "image_slots": slots,
                    "cached_tokens": request.get("cached_tokens"),
                    "thinking": None if not thinking else {"chars": len(thinking), "text": thinking},
                    "request_file": f"{stem}.request.json.gz",
                })
                target_dir.mkdir(parents=True, exist_ok=True)
                with gzip.open(target_dir / f"{stem}.request.json.gz", "wt", encoding="utf-8") as handle:
                    json.dump({"system": request.get("system"), "user": request.get("user")}, handle,
                              ensure_ascii=False)
            if stage == "reasoning":
                record["feedback_text"] = self.timeline.feedback_line()
            # Whether this reply was used, and if not why — the call a re-sample or a
            # repair follows. Absent on a route that does not validate its reply.
            if "rejection" in result:
                record["rejection"] = result.get("rejection")
            # How many complete tool calls an accepted grounding reply held, and how many
            # of them were not executed, where the action agent counts them (the first is
            # used; Qwen's scroll composite uses both of its two).
            for key in ("tool_call_count", "tool_calls_dropped"):
                if key in result:
                    record[key] = result[key]
            # Whether the reply's action came from its function's name, where the backbone
            # takes one from there (the DeepSeek profile).
            if "action_from_function_name" in result:
                record["action_from_function_name"] = result["action_from_function_name"]
            # What the coordinate strip removed from the line this call grounded, where the action
            # agent hands it to the call (the DeepSeek profile).
            if "grounding_input_strip" in result:
                record["grounding_input_strip"] = result["grounding_input_strip"]
            # The surplus-closer repair of the reply's arguments and the click texts dropped from the
            # line and the reply, where the profile records them (the DeepSeek profile).
            for key in ("arguments_repaired", "click_text_dropped"):
                if key in result:
                    record[key] = result[key]
            # The call's HTTP attempts, its last failed attempt's error and its count of
            # attempts that failed in no known form, where the backbone reports them (its
            # own retry loop; the Qwen profile's unrecognised-failure fallback).
            for key in ("transport_attempts", "transport_last_error", "transport_unrecognised"):
                if key in result:
                    record[key] = result[key]
            # The reply's provenance — its finish reason, the response's model echo (the record's
            # `model` is the id SENT) and the backend fingerprint — where the backbone reports them
            # (the DeepSeek profile).
            for key in ("finish_reason", "response_model", "system_fingerprint"):
                if key in result:
                    record[key] = result[key]
            target = target_dir / f"{stem}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(_json_safe(record), handle, indent=2, default=str)
        except Exception:  # noqa: BLE001 — diagnostics never break a step
            logger.exception("failed to write the per-call diagnostics record")

    def _terminal_action(self) -> Optional[str]:
        if self.runtime_state is None:
            return None
        records = self.runtime_state.content.get("_executed_actions") or []
        for record in reversed(records):
            if isinstance(record, dict) and record.get("status") == "terminal":
                return record.get("command")
        return None

    @staticmethod
    def _extract_response_text(decision: GraphPCDecision) -> str:
        if decision.reasoning is not None:
            reasoning_text = decision.reasoning.content.get("reasoning_text")
            if reasoning_text:
                return str(reasoning_text)
        return str(decision.action_batch.content.get("reasoning_text", ""))

    def _model_name(self) -> Optional[str]:
        return getattr(self.backbone, "model_name", None) or self.config.get("model")

    def _collect_token_usage(self) -> dict[str, Any]:
        """The shared backbone's cumulative counters (zeros / ``None`` model
        without a backbone)."""
        backbone = self.backbone
        return {
            "model": getattr(backbone, "model_name", None),
            "call_count": getattr(backbone, "call_count", 0),
            "total_input_tokens": getattr(backbone, "total_input_tokens", 0),
            "total_output_tokens": getattr(backbone, "total_output_tokens", 0),
        }

    def _phase_record(self) -> dict[str, Any]:
        """Compact record of the live episode state (a finished phase of a
        multi-phase task) for the task telemetry."""
        content = self.runtime_state.content if self.runtime_state is not None else {}
        return {
            "instruction": self.task_instruction,
            "predict_calls": self.predict_calls,
            "closed_steps": self.runtime_state.step_idx if self.runtime_state is not None else 0,
            "action_parse_failures": self.action_parse_failure_count,
            "reasoning_parse_failures": self.reasoning_parse_failure_count,
            "reasoning_decision_missing": self.reasoning_decision_missing_count,
            "reasoning_decision_missing_kinds": dict(self.reasoning_decision_missing_kinds),
            "grounding_parse_failures": self.grounding_parse_failure_count,
            "reasoning_resamples": self.reasoning_resample_count,
            "grounding_resamples": self.grounding_resample_count,
            "resample_exhausted_reasoning": self.resample_exhausted_reasoning,
            "resample_exhausted_grounding": self.resample_exhausted_grounding,
            # The denominator for the grounding-side counters above (see __init__).
            "grounding_calls_made": self.grounding_calls_made,
            "grounding_calls_bypassed": self.grounding_calls_bypassed,
            "grounding_inputs_stripped": self.grounding_inputs_stripped,
            "grounding_strip_fallbacks": self.grounding_strip_fallbacks,
            "monitor_signals": list(self.monitor_signals),
            "model_call_records": self._call_records,
            # The deciding call's facts ledger, whole (the request shows it bounded:
            # timeline.py), and how often the bound had to elide part of it.
            "facts_ledger": list(self.timeline.facts),
            "facts_elision": {
                "renders": self.timeline.facts_elided_renders,
                "first_step": self.timeline.facts_first_elision_step,
                "max_elided_entries": self.timeline.facts_max_elided,
                "last_render": self.timeline.facts_last_render,
            },
            # Entries the record step did not append, by why (timeline.record_facts).
            "facts_dropped": dict(self.timeline.facts_dropped),
            **self._function_name_count(),
            **self._grounding_repair_counts(),
            **self._requirements_render(),
            **self._profile_episode_counts(),
            "runtime_content": dict(content),
        }

    def _profile_episode_counts(self) -> dict[str, Any]:
        """The profile-gated episode keys, the same in the task telemetry and in each finished phase's record
        (``reset`` zeroes both counts per phase): ``ask_user_count``, the questions asked, on a profile that can ask
        (each is also an _executed_actions row with status "ask"); ``agent_errors``, each lost episode the profile
        ended evaluated, on a profile with an ``infra_error`` classifier, [] when none happened (the key's presence
        says the episode could have one)."""
        profile = model_profile(self.backbone_type)
        counts: dict[str, Any] = {}
        if profile.ask_user_channel:
            counts["ask_user_count"] = self.ask_user_count
        if profile.infra_error is not None:
            counts["agent_errors"] = deepcopy(self.agent_errors)
        return counts

    def _requirements_render(self) -> dict[str, Any]:
        """The requirement ledger's render clips, keyed for the telemetry, on a profile with the ledger (its
        active_config key); nothing on the others, whose records keep their shape. The ledger, its per-claim records
        and its counts live on the runtime state (orchestrator.py) and reach the record through runtime_content."""
        if "requirement_ledger" not in self.config:
            return {}
        return {"requirements_render": {"clipped_renders": self.timeline.requirements_clipped_renders,
                                        "max_clipped": self.timeline.requirements_max_clipped}}

    def _function_name_count(self) -> dict[str, int]:
        """The episode's grounded actions taken from a call's function name, keyed for the
        telemetry, on a profile whose action agent carries the rule (its active_config key,
        action_deepseek.ACTION_FROM_FUNCTION_NAME_KEY); nothing on the others, whose records
        keep their shape."""
        if "grounding_action_from_function_name" not in self.config:
            return {}
        return {"grounding_actions_from_function_name": self.grounding_actions_from_function_name}

    def _grounding_repair_counts(self) -> dict[str, int]:
        """The episode's repaired grounding replies and dropped click texts, keyed for the
        telemetry, on a profile whose action agent carries the rules (their active_config keys,
        action_deepseek.ARGUMENTS_REPAIR_KEY and CLICK_TEXT_RULE_KEY); nothing on the others,
        whose records keep their shape."""
        counts: dict[str, int] = {}
        if "grounding_arguments_repair" in self.config:
            counts["grounding_arguments_repaired"] = self.grounding_arguments_repaired
        if "grounding_click_text_rule" in self.config:
            counts["grounding_line_texts_dropped"] = self.grounding_line_texts_dropped
            counts["grounding_reply_texts_dropped"] = self.grounding_reply_texts_dropped
        return counts

    def _collect_telemetry(
        self,
        *,
        task_id: str,
        score: Optional[float],
        result_payload: dict[str, Any],
        entry_path: Optional[Path],
        persist: str,
    ) -> dict[str, Any]:
        """The per-task resolution-level record: retrieval events, verification
        verdicts (with the monitor's reconciled ``monitor_changed``), the
        per-step monitor diff signals, degrade counters, usage (cumulative +
        this episode's delta) and the active configuration.

        Sampling counters — ALWAYS present, ints, 0 included (a reader takes an
        absent key as "not recorded"):
          * ``reasoning_resamples`` / ``grounding_resamples``: extra samples
            beyond the first call, summed over the task's steps;
          * ``resample_exhausted_reasoning`` / ``resample_exhausted_grounding``:
            steps where all attempts failed and the degrade path ran (WAIT);
          * ``reasoning_parse_failures`` / ``grounding_parse_failures``:
            unparseable samples counted per ATTEMPT (every unparseable reply,
            rescued or not) = resamples + exhausted steps — not a duplicate of
            the exhausted counts — minus, on the deciding side, the samples
            counted in ``reasoning_decision_missing``;
          * ``reasoning_decision_missing``: deciding samples that parsed but
            carried no decision (the schema echoed, or no list-typed
            intended_actions), per ATTEMPT, re-sampled like the unparseable ones
            and counted apart from them; ``reasoning_decision_missing_kinds``
            splits them by shape (schema_echo / native_tool_call / other), and
            each such call's record carries its ``rejection``;
          * ``action_parse_failures``: per step, an action the translator refused.
        None counts a reply the Qwen class-family rule sent back (action_qwen.py): its repairs
        are taken out of the grounding re-samples and failures, and its exhausted step is a
        ``grounding_class_mismatch`` WAIT, not a ``resample_exhausted_grounding`` one.
        """
        usage = self._collect_token_usage()
        telemetry: dict[str, Any] = {
            "task_id": task_id,
            "score": score,
            "success": _is_full_score(score),
            "result_payload": deepcopy(result_payload),
            "predict_calls": self.predict_calls,
            "closed_steps": self.runtime_state.step_idx if self.runtime_state is not None else 0,
            "action_parse_failures": self.action_parse_failure_count,
            "reasoning_parse_failures": self.reasoning_parse_failure_count,
            "reasoning_decision_missing": self.reasoning_decision_missing_count,
            "reasoning_decision_missing_kinds": dict(self.reasoning_decision_missing_kinds),
            "grounding_parse_failures": self.grounding_parse_failure_count,
            "reasoning_resamples": self.reasoning_resample_count,
            "grounding_resamples": self.grounding_resample_count,
            "resample_exhausted_reasoning": self.resample_exhausted_reasoning,
            "resample_exhausted_grounding": self.resample_exhausted_grounding,
            # The denominator for the grounding-side counters above (see __init__).
            "grounding_calls_made": self.grounding_calls_made,
            "grounding_calls_bypassed": self.grounding_calls_bypassed,
            "grounding_inputs_stripped": self.grounding_inputs_stripped,
            "grounding_strip_fallbacks": self.grounding_strip_fallbacks,
            "monitor_signals": list(self.monitor_signals),
            "model_call_records": self._call_records,
            # The deciding call's facts ledger, whole (the request shows it bounded:
            # timeline.py), and how often the bound had to elide part of it.
            "facts_ledger": list(self.timeline.facts),
            "facts_elision": {
                "renders": self.timeline.facts_elided_renders,
                "first_step": self.timeline.facts_first_elision_step,
                "max_elided_entries": self.timeline.facts_max_elided,
                "last_render": self.timeline.facts_last_render,
            },
            # Entries the record step did not append, by why (timeline.record_facts).
            "facts_dropped": dict(self.timeline.facts_dropped),
            **self._function_name_count(),
            **self._grounding_repair_counts(),
            **self._requirements_render(),
            "support_set_entry": str(entry_path) if entry_path is not None else None,
            "persist": persist,
            "active_config": {
                **deepcopy(self.config),
                "retrieval_off": bool(self.retrieval_off),
                "screen_size": list(self.screen_size),
                "model": self._model_name(),
            },
            # This episode's share: the backbone's counters run across every episode the
            # worker runs, so the raw totals would mix earlier tasks into this one.
            "token_usage": {
                "model": usage["model"],
                **{key: usage[key] - self._token_start.get(key, 0)
                   for key in ("call_count", "total_input_tokens", "total_output_tokens")},
            },
            "episode_calls": usage["call_count"] - self._token_start.get("call_count", 0),
            "episode_input_tokens": usage["total_input_tokens"] - self._token_start.get("total_input_tokens", 0),
            "episode_output_tokens": usage["total_output_tokens"] - self._token_start.get("total_output_tokens", 0),
        }
        if self.phase_records:
            # Multi-phase task: the earlier phases' records (the live state
            # below is the last phase).
            telemetry["phase_records"] = deepcopy(self.phase_records)
        telemetry.update(self._profile_episode_counts())
        content = self.runtime_state.content if self.runtime_state is not None else None
        if isinstance(content, dict):
            # Every content key: _retrieval_events, _verification_verdicts,
            # _monitor_history, _obs_text_history, _executed_actions,
            # _infeasible_normalized_events, _blocked_done_count,
            # graph_summary, progress_status, ...
            telemetry["runtime_content"] = dict(content)
            telemetry["retrieval_events"] = list(content.get("_retrieval_events") or [])
            telemetry["verification_verdicts"] = list(content.get("_verification_verdicts") or [])
        else:
            telemetry["runtime_content"] = None
        return telemetry

    def _write_telemetry(
        self,
        *,
        task_id: str,
        score: Optional[float],
        result_payload: dict[str, Any],
        result_dir: Optional[str],
        entry_path: Optional[Path],
        persist: str,
    ) -> None:
        """Best-effort dump to ``<parent of api_log_dir>/graphpc_telemetry.json``
        (else ``<result_dir>/``). Wrapped so a dump failure never masks the
        real episode outcome (measurement integrity)."""
        target_dir = os.path.dirname(self._api_log_dir) if self._api_log_dir else result_dir
        if not target_dir:
            return
        try:
            telemetry = self._collect_telemetry(
                task_id=task_id, score=score, result_payload=result_payload,
                entry_path=entry_path, persist=persist,
            )
            target = Path(target_dir) / TELEMETRY_FILENAME
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(_json_safe(telemetry), handle, indent=2, default=str)
        except Exception:  # noqa: BLE001 — telemetry must never break the run.
            logger.exception("failed to write episode telemetry for %s", task_id)

    def _build_support_observation_payload(self) -> dict[str, Any]:
        """Observation text to embed candidate-side for the retrieval text arm.
        Same shape as the WebArena payload so ``_entry_observation_text`` in
        ``support_set.py`` reads both backends the same way; on this
        screenshot-primary benchmark it is empty, so candidate text is the
        instruction (the OSWorld instantiation of design.md §5)."""
        if self.last_update is None:
            return {}
        content = self.last_update.current_observation.content
        if not isinstance(content, dict):
            return {}
        return {"observation_text": content.get("observation_text", "")}

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
        """``graph_summary`` is the only graph artifact the run produces."""
        if self.runtime_state is None or "graph_summary" not in self.runtime_state.content:
            return {}
        return {"graph_summary": deepcopy(self.runtime_state.content["graph_summary"])}
