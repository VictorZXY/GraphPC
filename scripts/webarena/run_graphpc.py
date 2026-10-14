"""Vendored WebArena runner for GraphPC (`design.md` §12 Benchmark integration boundary +
`.claude/docs/benchmarks/webarena.md`). Drives WebArena's ``ScriptBrowserEnv`` +
``evaluation_harness`` through ``WebArenaGraphPCAgent`` without touching any
WebArena core file (strict read-only). The baseline runs WebArena's own
``run.py --agent_type prompt`` loop through ``run_baseline.py`` — this file is
GraphPC-only.

Layout intent: WebArena's own ``run.py`` loop shape (the OSWorld-side runner is
a copy of OSWorld-V2's own M3 runner, so the two are not mirrors). CLI flags
are snake_case parallel where possible; WebArena-specific flags
(``--config_file`` / ``--webarena_root`` / ``--observation_type``) are added.

DEPLOYMENT PREREQUISITES (see ``.claude/docs/benchmarks/webarena.md`` for
the full checklist):
  1. Clone WebArena to ``<repo>/../WebArena`` (or pass ``--webarena_root``)
     and install per their README (Playwright + per-domain env URLs).
  2. ``pip install -r <repo>/requirements.txt`` (it lists sentence-transformers) + cache the
     ``Qwen/Qwen3-Embedding-0.6B`` weights. The text arm IS the
     ranking, so this is a HARD REQUIREMENT for any retrieval-on run: without
     it every episode aborts (``TextArmUnavailableError`` -> infra_error).
     Only ``--retrieval_off`` runs are unaffected.
  3. The baseline run is WebArena's own ``run.py --agent_type prompt`` loop and
     agent, launched through ``scripts/webarena/run_baseline.py``, which installs
     our openai-0.27 compat shim (``_openai_compat``) and the environment
     hardening (``_env_hardening``) that this runner installs too.

In an env without a WebArena checkout, ``--dry_run`` walks the runner's
construction + agent contract via a stub orchestrator + mocked env, so the
runner is exercisable in dev/CI. The LIVE end-to-end check (real env +
evaluator) is the deployment-step verification — this file is the artifact
that exercises it once WebArena is provisioned.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Ensure the repo root is importable when invoked as `python scripts/webarena/run_graphpc.py …`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("desktopenv.experiment")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Vendored WebArena runner for GraphPC. Constructs "
            "WebArenaGraphPCAgent and drives WebArena's run loop "
            "(read-only on WebArena core). See README.md."
        )
    )
    # backbone / model
    p.add_argument("--backbone", type=str, default="claude", choices=["claude", "gpt", "deepseek", "minimax", "qwen"])
    p.add_argument("--model", type=str, required=True,
                   help="e.g. claude-sonnet-5  |  gpt-6-sol  |  deepseek-flash  |  MiniMax-M3  |  qwen/qwen3.7-plus")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--reasoning_effort", type=str,
                   choices=["none", "low", "medium", "high", "xhigh"], default="high",
                   help="GPT and Claude. On Claude both calls send it (deciding with "
                        "adaptive thinking, grounding with thinking disabled); 'none' "
                        "is refused there.")
    p.add_argument("--grounding_reasoning_effort", type=str,
                   choices=["none", "low", "medium", "high", "xhigh"], default=None,
                   help="GPT-only. Effort for the grounding call; omitted = the "
                        "--reasoning_effort value.")

    # workflow shape + gates (parallel to scripts/osworld/run_multienv_graphpc.py)
    p.add_argument("--two_pass", action="store_true",
                   help="Two-pass workflow (a backup design, not in the current plan); default single-pass.")
    p.add_argument("--completion_gate_off", action="store_true",
                   help="Disable the completion-gate (Verification).")
    p.add_argument("--completion_gate_window", type=int, default=3)
    p.add_argument("--max_actions", type=int, default=1)
    p.add_argument("--max_trajectory_length", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=30,
                   help="Per-episode step cap (independent of WebArena's own cap).")

    # WebArena specifics
    p.add_argument("--webarena_root", type=str,
                   default=str(_REPO_ROOT.parent / "WebArena"),
                   help="WebArena checkout directory (analogous to the OSWorld-V2 staging clone).")
    p.add_argument("--config_file", type=str, default=None,
                   help="One WebArena task config JSON to run.")
    p.add_argument("--config_dir", type=str, default=None,
                   help="Directory of WebArena task config JSONs (alternative to --config_file).")
    p.add_argument("--observation_type", type=str, default="accessibility_tree",
                   choices=["accessibility_tree", "html", "image"])
    p.add_argument("--result_dir", type=str, default="./webarena_results")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Skip configs whose task already has a SCORED result in the "
                        "canonical summary.json (errored eval_error/infra_error rows "
                        "re-run — the casualty policy; run_baseline.py-parallel). "
                        "Default on.")
    p.add_argument("--no_resume", dest="resume", action="store_false",
                   help="Force re-running every config.")
    p.add_argument("--support_set_root", type=str, default="./artifacts/support_set")
    p.add_argument("--viewport_width", type=int, default=1280)
    p.add_argument("--viewport_height", type=int, default=720)
    p.add_argument("--env_chunk_size", type=int, default=15,
                   help="Recreate the ScriptBrowserEnv every N episodes (planned "
                        "Playwright-driver lifecycle boundary; mitigates driver-side "
                        "heap OOM on long sequential runs).")

    # Retrieval mode. The resolved state is echoed into every episode's telemetry
    # and into the run config dump, and the launch preflight asserts it.
    p.add_argument("--retrieval_off", action="store_true",
                   help="Run with retrieval fully off — no support-set reads and no "
                        "support-set writes. The intended mode for task sets of "
                        "UNIQUE templates, where no stored sibling exists to retrieve. "
                        "Leave OFF (i.e. retrieval on) for family-rich sets, where "
                        "stored siblings exist.")

    # Exerciseability without WebArena installed.
    p.add_argument("--dry_run", action="store_true",
                   help=(
                       "Exercise the runner without touching the real WebArena env "
                       "(no Playwright, no evaluator). Constructs the agent, walks a "
                       "stub trajectory, and verifies the boundary contract holds. "
                       "Used for in-repo / CI smoke tests; the LIVE end-to-end check "
                       "is the deployment step."
                   ))
    p.add_argument("--log_level", type=str,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default="INFO")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    config_file: str
    success: bool
    score: Optional[float]
    steps: int
    errored: bool = False
    # Error taxonomy for contamination accounting (shared with the baseline wrapper).
    # EXCLUDED from the scored denominator (errored=True): "eval_error" (the
    # evaluator/judge raised; agent loop completed) and "infra_error" (browser /
    # env / driver / network / provider-API — external to agent logic, symmetric
    # with the baseline's [OpenAI Error] exclusion). COUNTED as a real 0.0
    # (errored=False, score=0.0): "agent_error" (an exception from our
    # orchestrator/backbone code — symmetric with the baseline's ValueError ->
    # stop -> [Result] FAIL). None for cleanly scored tasks.
    error_type: Optional[str] = None
    # Additive (resume): the payload-derived task id — stable across the
    # ordinal-prefixed symlink config names a launch may use, unlike the config
    # path. The resume skip-set and the canonical-summary merge key on it.
    task_id: Optional[str] = None


def _validate_webarena_root(path: Path) -> None:
    """Mirror of OSWorld's `_validate_osworld_root`: expect WebArena's
    ``run.py`` at the top so we fail fast with a clear error if the user
    pointed at the wrong dir."""
    expected = path / "run.py"
    if not expected.exists():
        raise FileNotFoundError(
            f"Expected a WebArena checkout with {expected}. "
            "Clone WebArena to ../WebArena (or pass --webarena_root). "
            "See .claude/docs/benchmarks/webarena.md."
        )


def _ensure_webarena_on_path(root: Path) -> None:
    """Put the WebArena root on ``sys.path`` so its top-level packages
    (``browser_env``, ``evaluation_harness``, ``agent``) are importable.
    The runner is the only place we mutate sys.path; the adapter package
    itself stays import-clean."""
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_config_files(args: argparse.Namespace) -> list[Path]:
    if args.config_file:
        return [Path(args.config_file)]
    if args.config_dir:
        found = sorted(Path(args.config_dir).glob("*.json"))
        if not found:
            # A missing/empty dir must fail loudly, not "succeed" with 0 tasks
            # (a relative --config_dir resolves against the subprocess cwd,
            # which the parallel launcher sets to the WebArena checkout).
            raise ValueError(f"--config_dir has no *.json configs: {args.config_dir}")
        return found
    raise ValueError("Provide --config_file or --config_dir.")


def _task_id_from_config(config_path: Path, config_payload: dict[str, Any]) -> str:
    """A stable, human-readable task id for persistence + result files.
    WebArena configs typically carry ``task_id``/``id``; fall back to the
    stem of the config filename."""
    for key in ("task_id", "id"):
        if config_payload.get(key) is not None:
            return str(config_payload[key])
    return config_path.stem


def _intent_from_config(config_payload: dict[str, Any]) -> str:
    """Read the natural-language goal from a WebArena task config."""
    for key in ("intent", "goal", "instruction", "question"):
        v = config_payload.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


# ---------------------------------------------------------------------------
# Live run (real WebArena env)
# ---------------------------------------------------------------------------


# WebArena harness contract this loop is written against (confirm against the
# pinned checkout at deploy time — mirrors WebArena's own ``run.py``):
#   * ``env.reset(options={"config_file": <path>}) -> (obs, info)``
#   * ``env.step(action) -> (obs, reward, terminated, truncated, info)``  (5-tuple)
#   * ``trajectory`` is ``list[StateInfo | Action]`` interleaved: initial state,
#     then (action, state) per step; when the agent emits ``ActionTypes.STOP``
#     the loop breaks WITHOUT stepping the env, leaving the stop action last.
#   * ``evaluator(trajectory=, config_file=, page=env.page,
#     client=env.get_page_client(env.page)) -> float``
def _is_stop_action(action: Any, stop_type: Any) -> bool:
    """True when ``action`` is a terminal STOP.

    Live: WebArena actions are dicts carrying an ``action_type`` enum; compare
    against ``ActionTypes.STOP`` (``stop_type``). Dry-run / mock: no WebArena
    checkout, so the stub stop factory marks its dict with
    ``action_type == "stop"`` and the same break path runs.
    """
    if not isinstance(action, dict):
        return False
    action_type = action.get("action_type")
    if stop_type is not None and action_type == stop_type:
        return True
    return action_type == "stop"


def _terminal_stop(stop_factory) -> Any:
    """The terminal stop appended when the ENV reports ``terminated`` (so the
    stop action is always the trajectory's last element). Uses WebArena's
    ``create_stop_action("")`` when available; falls back to a recognizable
    marker for the dry-run / mock path where no factory is loaded. Empty answer
    mirrors run.py's ``terminated`` path (``create_stop_action("")``)."""
    if stop_factory is not None:
        return stop_factory("")
    return {"action_type": "stop", "answer": ""}


def _cap_exit_stop(stop_factory, max_steps: int) -> Any:
    """The stop appended on a --max_steps cap-exit. BYTE-MIRRORS run.py's early_stop
    answer (run.py:169 ``f"Reach max steps {max_steps}"`` -> run.py:293
    ``create_stop_action(f"Early stop: {stop_info}")``). The answer MUST be
    non-empty: an empty answer destabilizes the fuzzy_match LLM judge (asserts on
    "" / can hallucinate a false 1.0) and differs from the baseline on must_include
    substring edges — so the cap-exit uses run.py's exact non-empty text, not the
    empty terminated-path stop. Falls back to a marker in the checkout-less path."""
    answer = f"Early stop: Reach max steps {max_steps}"
    if stop_factory is not None:
        return stop_factory(answer)
    return {"action_type": "stop", "answer": answer}


# Everything on state.content is small telemetry and IS dumped, so new
# telemetry keys are captured automatically rather than silently dropped.
def _collect_token_usage(agent) -> dict[str, Any]:
    """Read the shared backbone's cumulative token counters. For GPT, the
    reasoning + action agents share one ``GPTAgent`` instance (see the agent's
    ``reset``), so reading either yields the same totals — record once. The other
    backbones share one instance the same way; a backbone without the counters
    (the dry-run stub) degrades to zeros / ``None`` model."""
    backbone = None
    for holder_name in ("reasoning_agent", "action_agent"):
        candidate = getattr(getattr(agent, holder_name, None), "backbone", None)
        if candidate is not None:
            backbone = candidate
            break
    return {
        "model": getattr(backbone, "model_name", None),
        "call_count": getattr(backbone, "call_count", 0),
        "total_input_tokens": getattr(backbone, "total_input_tokens", 0),
        "total_output_tokens": getattr(backbone, "total_output_tokens", 0),
    }


def _collect_episode_telemetry(
    *,
    agent,
    task_id: str,
    config_file: str,
    score: Optional[float],
    success: bool,
    steps: int,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    token_start: Optional[dict[str, Any]] = None,
    final_page_url: Optional[str] = None,
    error_class: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the per-episode resolution-level telemetry record (design
    principles #6/#9). The WebArena path otherwise persists nothing for failed
    episodes; this is the record the effect-class reliability audit reads.

    ``final_page_url`` is the raw URL of the page the evaluator read. It is
    absent when the episode crashed before evaluation. ``error_class`` is the type name of the exception behind an
    errored record. A missing page URL (``MissingPageUrlError``) and other infra
    failures share ``error_type == "infra_error"``, and this is what tells them
    apart without parsing the message.

    ``token_start`` is the backbone's cumulative counter snapshot taken at this
    episode's START; when provided, the per-episode DELTA is recorded alongside
    the run-cumulative token_usage. The backbone counters are never reset (other
    consumers read them), so summing the cumulative across a run's telemetry
    files double-counts — sum the ``episode_*`` deltas instead."""
    telemetry: dict[str, Any] = {
        "task_id": task_id,
        "config_file": config_file,
        "score": score,
        "success": success,
        "steps": steps,
        "action_parse_failures": getattr(agent, "action_parse_failure_count", 0),
        "reasoning_parse_failures": getattr(agent, "reasoning_parse_failure_count", 0),
        "grounding_parse_failures": getattr(agent, "grounding_parse_failure_count", 0),
        # Sampling counters (common/resample.py), always present as ints:
        # *_resamples = extra samples beyond the first call, summed over steps;
        # resample_exhausted_* = steps where all attempts failed and the noop
        # degrade ran; the *_parse_failures pair above counts unparseable
        # samples per ATTEMPT (rescued or not) = resamples + exhausted steps.
        "reasoning_resamples": getattr(agent, "reasoning_resample_count", 0),
        "grounding_resamples": getattr(agent, "grounding_resample_count", 0),
        "resample_exhausted_reasoning": getattr(agent, "resample_exhausted_reasoning", 0),
        "resample_exhausted_grounding": getattr(agent, "resample_exhausted_grounding", 0),
    }
    if error is not None:
        telemetry["error"] = error
    if error_type is not None:
        telemetry["error_type"] = error_type
    if final_page_url is not None:
        telemetry["final_page_url"] = final_page_url
    if error_class is not None:
        telemetry["error_class"] = error_class

    runtime_state = getattr(agent, "runtime_state", None)
    content = getattr(runtime_state, "content", None) if runtime_state is not None else None
    if isinstance(content, dict):
        # Dump every content key. This carries _actual_effects +
        # _verification_verdicts (the resolved verdicts hold the
        # predicted/actual/matches pairs — the predicted-vs-actual source for
        # the effect-class reliability audit), plus need_retrieval /
        # progress_status / last_reasoning_mode / last_retrieval_scope /
        # graph_summary / _monitor_history, and the
        # per-step _obs_text_history (the audit gold sample's before/after
        # observation context — ~2KB/step, up to ~100KB/task, INTENTIONALLY kept
        # in the dump). graph_summary is the only graph artifact on state, so
        # no key needs a size exclusion.
        telemetry["runtime_content"] = dict(content)
        if "_verification_verdicts" not in content:
            # Predicted/actual pairs are only recoverable from the verdicts;
            # note their absence rather than inventing a new mechanism.
            telemetry["predicted_actual_note"] = (
                "no _verification_verdicts on runtime_state.content"
            )
    else:
        telemetry["runtime_content"] = None
        telemetry["predicted_actual_note"] = (
            "runtime_state (or its content) was None; no per-step telemetry"
        )

    # Ablation switch state, read back off the constructed agent (not off the CLI
    # args) so the echo names what actually ran. Present on every episode record
    # whether or not a switch is set — an absent block would be indistinguishable
    # from an old telemetry file.
    switch_state = getattr(agent, "ablation_switch_state", None)
    if callable(switch_state):
        telemetry["ablation_switches"] = switch_state()

    usage = _collect_token_usage(agent)
    telemetry["token_usage"] = usage  # run-cumulative (monotone check)
    if token_start is not None:
        # Per-episode delta — the correct quantity to sum across a run (the
        # cumulative counters never reset, so summing them double-counts).
        telemetry["episode_calls"] = usage["call_count"] - token_start.get("call_count", 0)
        telemetry["episode_input_tokens"] = (
            usage["total_input_tokens"] - token_start.get("total_input_tokens", 0)
        )
        telemetry["episode_output_tokens"] = (
            usage["total_output_tokens"] - token_start.get("total_output_tokens", 0)
        )
    # Snapshot of the env-hardening counters at this point in the run (cumulative
    # across episodes) so flaky-page rescues + client recreations are visible per-dump.
    from src.benchmarks.webarena._env_hardening import OBS_RETRY_STATS, PAGE_CLIENT_STATS
    telemetry["obs_retry_stats"] = dict(OBS_RETRY_STATS)
    telemetry["page_client_stats"] = dict(PAGE_CLIENT_STATS)
    return telemetry


def _write_episode_telemetry(
    *,
    agent,
    result_dir: Path,
    task_id: str,
    config_file: str,
    score: Optional[float],
    success: bool,
    steps: int,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    token_start: Optional[dict[str, Any]] = None,
    final_page_url: Optional[str] = None,
    error_class: Optional[str] = None,
) -> None:
    """Best-effort per-episode telemetry dump to
    ``<result_dir>/telemetry_<task_id>.json``. Wrapped so a dump failure never
    masks the real episode outcome / error (measurement integrity)."""
    try:
        telemetry = _collect_episode_telemetry(
            agent=agent, task_id=task_id, config_file=config_file,
            score=score, success=success, steps=steps, error=error,
            error_type=error_type, token_start=token_start, final_page_url=final_page_url,
            error_class=error_class,
        )
        safe_task_id = str(task_id).replace("/", "_").replace("\\", "_")
        target = result_dir / f"telemetry_{safe_task_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, default=str)
    except Exception:  # noqa: BLE001 — telemetry must never break the run.
        logger.exception("failed to write episode telemetry for %s", task_id)


def run_episode_live(
    *,
    agent,
    env,
    evaluator,
    config_path: Path,
    config_payload: dict[str, Any],
    max_steps: int,
    result_dir: Optional[Path] = None,
) -> EpisodeResult:
    """One episode against the real WebArena env. Mirrors WebArena's run-loop
    shape (reset -> loop[next_action -> stop-check -> env.step] -> evaluator ->
    finalize) without copying their ``run.py``. Strict read-only on WebArena
    core. See the contract comment above for the assumed env/evaluator API.

    ``result_dir`` enables the per-episode telemetry dump; ``None`` (the
    dry-run path) skips it."""
    intent = _intent_from_config(config_payload)
    task_id = _task_id_from_config(config_path, config_payload)
    config_file = str(config_path)
    agent.reset(config_file)
    # Snapshot the backbone's cumulative token counters at episode START so the
    # per-episode delta can be recorded (the counters never reset, so the
    # cumulative alone double-counts when summed across a run).
    token_start = _collect_token_usage(agent)

    # ActionTypes.STOP + create_stop_action via the centralized lazy helper
    # (None/None in a checkout-less env; the stub marker covers that path).
    # Imported here — not at module load — so the WebArena checkout is already
    # on the path (added by main()) before ``_webarena_imports`` resolves its
    # Agent base; a top-level import would pin the stub base for the live agent.
    from src.benchmarks.webarena._webarena_imports import load_stop_support
    stop_type, stop_factory = load_stop_support()

    obs, info = env.reset(options={"config_file": config_file})  # type: ignore[union-attr]
    trajectory: list[Any] = [{"observation": obs, "info": info or {}}]
    steps = 0

    # --max_steps is our extra guard on top of the agent-stop / env-terminated
    # exits (WebArena's own cap is separate).
    while steps < max_steps:
        action = agent.next_action(trajectory, intent, meta_data={})
        trajectory.append(action)
        steps += 1
        if _is_stop_action(action, stop_type):
            # Agent decided it's done: break WITHOUT stepping the env, leaving
            # the stop action as the final trajectory element.
            break
        obs, _reward, terminated, _truncated, info = env.step(action)  # type: ignore[union-attr]
        trajectory.append({"observation": obs, "info": info or {}})
        if terminated:
            trajectory.append(_terminal_stop(stop_factory))
            break

    # Our --max_steps cap can exit the loop with the trajectory ending on a
    # non-stop element (the last observation). WebArena's StringEvaluator then
    # KeyErrors on 'answer', which we'd tag eval_error → score=None → EXCLUDED,
    # inflating our arm's rate vs the baseline (asymmetric taxonomy). Mirror
    # run.py's early_stop path EXACTLY: append a stop whose answer is run.py's cap
    # text ("Early stop: Reach max steps N") — a non-empty answer the fuzzy judge
    # can process, so the episode scores a real 0.0 like the baseline.
    if not (trajectory and _is_stop_action(trajectory[-1], stop_type)):
        trajectory.append(_cap_exit_stop(stop_factory, max_steps))

    # The raw URL of the page the evaluator is about to read.
    # The agent's per-step record stops one page short of it on every exit: a
    # stop is never stepped, and a cap exit never calls next_action again.
    final_page_url = env.page.url  # type: ignore[union-attr]

    # WebArena's evaluator needs the page + CDP client alongside the trajectory.
    # The evaluator (esp. the LLM judge) is its own error class: a judge crash
    # is NOT an agent/env failure. Catch it here, tag it "eval_error", and
    # return a non-re-raising errored record so the run continues and the
    # contamination accounting can attribute it correctly.
    try:
        score = float(evaluator(  # type: ignore[misc]
            trajectory=trajectory,
            config_file=config_file,
            page=env.page,  # type: ignore[union-attr]
            client=env.get_page_client(env.page),  # type: ignore[union-attr]
        ))
    except Exception as exc:  # noqa: BLE001 — evaluator/judge error, not agent.
        logger.exception("Evaluator failed for %s", config_file)
        # finalize with success=False so no support-set write occurs.
        agent.finalize_task(task_id=task_id, success=False, result=None)
        if result_dir is not None:
            _write_episode_telemetry(
                agent=agent, result_dir=result_dir, task_id=task_id,
                config_file=config_file, score=None, success=False, steps=steps,
                error=str(exc), error_type="eval_error", token_start=token_start,
                final_page_url=final_page_url, error_class=type(exc).__name__,
            )
        return EpisodeResult(
            config_file=config_file, success=False, score=None, steps=steps,
            errored=True, error_type="eval_error", task_id=task_id,
        )

    agent.finalize_task(task_id=task_id, success=score > 0, result=score)
    if result_dir is not None:
        _write_episode_telemetry(
            agent=agent, result_dir=result_dir, task_id=task_id,
            config_file=config_file, score=score, success=score > 0, steps=steps,
            token_start=token_start, final_page_url=final_page_url,
        )
    return EpisodeResult(
        config_file=config_file, success=score > 0, score=score, steps=steps, task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Dry run (no WebArena env required)
# ---------------------------------------------------------------------------


class _DryRunEnv:
    """In-repo stand-in for ``ScriptBrowserEnv`` so the runner can exercise
    the agent + loop end-to-end without a WebArena checkout. Matches the real
    env's call shape: ``reset(options={"config_file": ...}) -> (obs, info)``,
    ``step(action) -> (obs, reward, terminated, truncated, info)`` (5-tuple),
    plus ``page`` + ``get_page_client(page)`` for the evaluator call. Returns
    deterministic canned StateInfo and terminates after a fixed number of
    steps. The agent never sees the difference."""

    def __init__(self, *, max_steps: int = 3):
        self._max = max_steps
        self._steps = 0
        # WebArena's DetachedPage shape: the runner reads the final page's URL.
        self.page = types.SimpleNamespace(url="http://dry-run.invalid/", content="")

    def reset(self, *, options: dict[str, Any]):  # noqa: ARG002
        self._steps = 0
        return (
            {"text": "RootWebArea 'dry_run'\n\t[1] button 'next'\n"},
            {"page": self.page, "observation_metadata": {}},
        )

    def step(self, _action):
        self._steps += 1
        terminated = self._steps >= self._max
        return (
            {"text": f"RootWebArea 'dry_run step={self._steps}'\n\t[1] button 'next'\n"},
            0.0,
            terminated,
            False,  # truncated
            {"page": self.page, "observation_metadata": {}},
        )

    def get_page_client(self, page):  # noqa: ARG002
        return "dry_run_client"


def _dry_evaluator(*, trajectory, config_file, page, client):  # noqa: ARG001
    return 0.0  # dry-run never succeeds; finalize_task no-ops on score==0


def run_dry(args: argparse.Namespace) -> list[EpisodeResult]:
    """Construct the agent, walk a short trajectory against ``_DryRunEnv``,
    and assert the boundary contract held. No WebArena imports needed.

    Dry-run deliberately does NOT instantiate real backbone clients (which
    would require API keys / SDKs); ``_build_dry_run_agent`` wires the agent
    to a stub orchestrator and stub action factories instead.
    """
    agent = _build_dry_run_agent(args)
    _write_run_config(args=args, agent=agent, result_dir=Path(args.result_dir))
    env = _DryRunEnv(max_steps=min(args.max_steps, 3))
    results: list[EpisodeResult] = []
    config_payload = {"intent": "dry-run task", "task_id": "dry_run/synthetic"}
    config_path = Path("dry_run/synthetic.json")
    res = run_episode_live(
        agent=agent,
        env=env,
        evaluator=_dry_evaluator,
        config_path=config_path,
        config_payload=config_payload,
        max_steps=min(args.max_steps, 3),
        # Dump telemetry here too, so --dry_run exercises the episode-telemetry
        # writer (including the ablation-switch echo) at the launcher layer.
        result_dir=Path(args.result_dir),
    )
    results.append(res)
    logger.info("[dry_run] completed %d step(s); contract held.", res.steps)
    return results


def _build_dry_run_agent(args: argparse.Namespace):
    """For ``--dry_run``, build a WebArenaGraphPCAgent with a stub
    orchestrator + stub action translation so no real backbone / WebArena
    factory is touched."""
    from unittest.mock import MagicMock

    from src.agents.common.schema import (
        Action as InternalAction,
        ActionBatch,
        GraphPCDecision,
        GraphPCRuntimeUpdate,
        Observation,
        RuntimeState,
    )
    from src.benchmarks.webarena import action_translator as at
    from src.benchmarks.webarena.graphpc_agent import WebArenaGraphPCAgent

    class _StubOrchestrator:
        def decide_next_action(self, *, instruction, observation_input, runtime_state):
            state = runtime_state or RuntimeState(instruction=instruction)
            # Emit a terminal ``done`` once a step has elapsed so the runner's
            # stop-break path is exercised (mirrors an agent that decides it is
            # finished); the first step is a normal click so a step is stepped.
            if state.step_idx >= 1:
                action = InternalAction(
                    action_type="done", parameters={"answer": "dry"}, raw={},
                )
            else:
                action = InternalAction(
                    action_type="click", parameters={},
                    raw={"input": {"command": "click [1]"}},
                )
            return GraphPCDecision(
                observation=Observation(backend="webarena", content={}),
                action_batch=ActionBatch(actions=[action]),
                runtime_state=deepcopy(state),
            )

        def update_runtime(self, *, decision, current_observation_input):
            state = deepcopy(decision.runtime_state)
            state.step_idx += 1
            return GraphPCRuntimeUpdate(
                previous_observation=decision.observation,
                current_observation=Observation(backend="webarena", content={}),
                action_batch=decision.action_batch,
                runtime_state=state,
            )

    # Stub the factories so translate_action doesn't import browser_env. The
    # stop factory marks its dict with ``action_type == "stop"`` so the runner's
    # stop-detection (``_is_stop_action``) recognizes it as terminal — the same
    # break path the live ``ActionTypes.STOP`` check drives.
    at._load_factories = lambda: {  # type: ignore[assignment]
        "id_based": lambda s: {"factory": "id_based", "command": s},
        "stop": lambda a: {"factory": "stop", "action_type": "stop", "answer": a},
        "none": lambda: {"factory": "none"},
    }
    # The retrieval mode is carried by a REAL object here (a mock would make
    # ablation_switch_state() report mock attributes as set), so --dry_run
    # exercises the switch-echo + config-dump path at the launcher layer. The
    # stub orchestrator never calls the retrieval agent, so no store is touched.
    from src.agents.retrieval.bypass import BypassRetrievalAgent
    from src.agents.retrieval.deterministic import DeterministicRetrievalAgent
    from src.agents.retrieval.support_set import LocalSupportSetStore

    class _DryRunEmbedder:
        """Available stand-in for the text arm: the retrieval agent asserts a
        live embedder at construction (a retrieval-on run must not start on a
        dead arm), and the dry run — like its stub reasoning/action agents —
        never makes a real retrieval call, so no model is loaded here."""

        available = True

        def similarity(self, query, candidates):  # pragma: no cover - never called by the stub orchestrator
            raise AssertionError("dry-run embedder must not be called")

    retrieval_agent = BypassRetrievalAgent() if args.retrieval_off else DeterministicRetrievalAgent(
        support_set_store=LocalSupportSetStore(root_dir=args.support_set_root, backend="webarena"),
        embedder=_DryRunEmbedder(),
    )
    return WebArenaGraphPCAgent(
        reasoning_agent=MagicMock(),
        retrieval_agent=retrieval_agent,
        action_agent=MagicMock(),
        support_set_store=MagicMock(),
        # The stub orchestrator never runs perception, so the adapter is stubbed like
        # the other collaborators it replaces. The dry run also runs where no site
        # variables exist, so it resolves no site-URL table. It is a stub-orchestrator
        # path that cannot produce a scored row.
        perception_processor=MagicMock(),
        retrieval_off=args.retrieval_off,
        orchestrator=_StubOrchestrator(),
    )


# ---------------------------------------------------------------------------
# Browser lifecycle (driver-heap-OOM mitigation)
# ---------------------------------------------------------------------------
#
# WebArena's ScriptBrowserEnv.reset() already tears down + recreates the
# Playwright driver per EPISODE (context_manager.__exit__ + setup). But a long
# sequential run still accumulates driver-side memory (WebArena's forced
# save_trace_enabled contributes), and the Node driver OOMs (V8 "heap out of
# memory"). We add a PLANNED lifecycle boundary: recycle the whole env every N
# episodes (hard close + fresh object) and give the Node driver heap headroom.
# This is a deterministic every-N-tasks driver recycle that is part of the run
# config, applied identically to both runners — NOT an unplanned mid-run
# crash-restart.


def _set_node_heap_headroom() -> None:
    """Give the Playwright Node driver more V8 heap. ``setdefault`` so an
    explicit user NODE_OPTIONS wins. Must run before the driver subprocess
    spawns (i.e. before env construction)."""
    os.environ.setdefault("NODE_OPTIONS", "--max-old-space-size=8192")


def _make_script_browser_env(args: argparse.Namespace):
    """Construct a fresh ScriptBrowserEnv with the standard forced settings."""
    from browser_env import ScriptBrowserEnv  # type: ignore[import-not-found]

    return ScriptBrowserEnv(
        # Hard-forced headless: symmetry with the baseline, which runs
        # headless (run.py forces render=False -> headless=True). A headed
        # GraphPC run against a headless baseline is a real environment
        # difference (rendering / timing / bot-heuristics) that would contaminate
        # Δ. There is no --headless flag, so symmetry cannot depend on
        # remembering it.
        headless=True,
        observation_type=args.observation_type,
        viewport_size={"width": args.viewport_width, "height": args.viewport_height},
        sleep_after_execution=2.0,
        current_viewport_only=True,
    )


def _maybe_recycle_env(env, *, episode_idx: int, chunk_size: int, make_env):
    """Recycle the env at each planned chunk boundary (every ``chunk_size``
    episodes): hard-close the old driver and build a fresh one. Returns the env
    to use for this episode. Best-effort close so a teardown error never aborts
    the run."""
    if chunk_size > 0 and episode_idx > 0 and episode_idx % chunk_size == 0:
        try:
            env.close()
        except Exception:  # noqa: BLE001 — never abort the run on teardown.
            logger.exception("env.close() failed at recycle boundary (episode %d)", episode_idx)
        logger.info(
            "env recycled at episode %d (planned lifecycle boundary, chunk_size=%d)",
            episode_idx, chunk_size,
        )
        return make_env()
    return env


# ---------------------------------------------------------------------------
# Error taxonomy — infra vs agent errors (shared with the baseline wrapper)
# ---------------------------------------------------------------------------

# Frames from these modules mean an env/driver failure, not our reasoning.
_INFRA_TRACEBACK_MARKERS = ("browser_env", "playwright")
# Provider / network exception class NAMES treated as infra (name-based so it
# covers openai + anthropic + builtins without importing them; the two backbones
# raise identically-named classes). Provider 4xx (BadRequestError,
# UnprocessableEntityError, ...) are our malformed request -> deliberately absent
# so they classify as agent_error and count as a 0.0.
_INFRA_EXC_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "APIError",
    "RateLimitError", "InternalServerError", "ServiceUnavailableError",
    "AuthenticationError", "PermissionDeniedError",
    # A provider failure delivered inside an HTTP 200 body (OpenRouter's two
    # documented 200-carrying-failure shapes), raised by the qwen backbone as
    # OpenRouterCallError. It is a provider-availability failure like the
    # names above — external to agent logic — but it carries NO status_code,
    # deliberately (faking one to reach the status branch below would be
    # falsified telemetry), so the name is the only handle. Inert for every
    # other backbone: no other module raises a class with this name.
    "OpenRouterCallError",
    # A billing refusal on the claude backbone (402, the organisation spend-cap 429
    # `enforced_spend_limit_reached`, the user-set usage-limit 400), raised as
    # ClaudeBillingError. The 400 would otherwise be counted as a 0.0 while the
    # control arm excludes the same refusal, so the name is the handle here too.
    # Inert for every other backbone: no other module raises a class with this name.
    "ClaudeBillingError",
    # A deepseek-flash reply that is HTTP 200 but unusable (no choices, a
    # provider-interrupted finish, or `length` with nothing in it), raised by the
    # deepseek backbone as DeepSeekCallError. The control arm excludes the same
    # shapes as [OpenAI Error]; without the name this arm would count them 0.0.
    # Inert for every other backbone and every other DeepSeek id.
    "DeepSeekCallError",
})
# Provider API HTTP status codes that are infra (availability / billing / rate),
# NOT an agent-logic bug — caught even when the SDK raises a generic
# ``APIStatusError`` with no specific subclass name (e.g. DeepSeek 402
# "Insufficient Balance"; openai maps 402 to the base APIStatusError). 5xx is
# handled separately (>= 500). 401/403 (auth) are covered by name above. 400/422
# (our malformed request) are deliberately EXCLUDED -> agent_error.
_INFRA_STATUS_CODES = frozenset({402, 408, 429})


def _classify_episode_error(exc: BaseException) -> str:
    """Classify an exception escaping the episode loop: "infra_error"
    (browser / env / driver / network / provider-API — external to agent logic,
    EXCLUDED, symmetric with the baseline's [OpenAI Error] exclusion) vs
    "agent_error" (our orchestrator/backbone logic — a real 0.0, COUNTED). Biased
    toward COUNTING: only a clear infra signal excludes, so agent-logic bugs are
    not silently dropped (the denominator bias this taxonomy removes)."""
    # A run whose retrieval text arm died is a GPU/resource failure external to
    # agent logic — excluded and retriable by the lane script, never a scored 0.0
    # (scoring it is exactly the defect this class exists to prevent). Ranking
    # is text-only, so the store raises on ANY retrieval-on run served
    # without the arm, and THIS classification is the degradation signal: the
    # episode aborts before it could reach a scored outcome.
    from src.agents.retrieval.embedding import TextArmUnavailableError
    from src.benchmarks.webarena.observation_adapter import MissingPageUrlError

    if isinstance(exc, TextArmUnavailableError):
        return "infra_error"
    # A state with no page URL cannot be rendered the way the control renders it
    # (the URL line has no fallback), and the defect is on the env side. So the
    # episode is excluded and retried, never scored 0.0. It is matched by TYPE,
    # because the adapter raises it from our own frames, which the traceback
    # markers below would read as agent_error.
    if isinstance(exc, MissingPageUrlError):
        return "infra_error"
    tb = exc.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if any(marker in filename for marker in _INFRA_TRACEBACK_MARKERS):
            return "infra_error"
        tb = tb.tb_next
    if type(exc).__name__ in _INFRA_EXC_NAMES or isinstance(exc, (ConnectionError, TimeoutError)):
        return "infra_error"
    # Provider API status errors (openai/anthropic APIStatusError family) carry a
    # numeric status_code even when the specific subclass name isn't in the set
    # above — classify billing/rate/availability statuses (402/408/429/5xx) as infra.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status in _INFRA_STATUS_CODES):
        return "infra_error"
    return "agent_error"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Resume + summary accounting (run_baseline.py-parallel)
# ---------------------------------------------------------------------------


def _scored_task_ids(result_dir: Path) -> set[str]:
    """Resume skip-set: task ids with a SCORED result. Scored = ``errored``
    false AND ``score`` present — errored rows (eval_error / infra_error, the
    casualty policy's re-run class) do NOT count as done and re-run on resume;
    agent_error rows are scored 0.0 and DO count. Basis: the MERGED view
    (canonical seed ∪ per-invocation files), NOT the canonical file alone —
    after a non-clean stop (SIGTERM / crash / reap) the end-of-invocation
    canonical rebuild never ran, but the per-episode-persisted invocation file
    already carries every completed episode, and a relaunch must skip exactly
    those (re-running them would duplicate support-store writes and corrupt
    the cold-run store trajectory). Keyed by ``task_id`` (payload-derived, so
    ordinal-prefixed symlink config names cannot break matching); rows
    without one fall back to the config filename stem — a fallback mismatch
    re-runs the task (never wrongly skips)."""
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        try:
            json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # The merge silently skips an unreadable file; say so loudly —
            # its rows are missing from the skip-set until repaired.
            logger.warning(
                "resume: unreadable %s (%s) — its rows are excluded from the skip-set",
                summary_path, exc,
            )
    done: set[str] = set()
    for row in _merge_invocation_summaries(result_dir):
        if row.get("errored") or row.get("score") is None:
            continue
        key = row.get("task_id") or Path(str(row.get("config_file") or "")).stem
        if key:
            done.add(str(key))
    return done


def _merge_invocation_summaries(result_dir: Path) -> list[dict[str, Any]]:
    """Rebuild the canonical summary as the union of the existing canonical
    ``summary.json`` (the seed, so batches that predate per-invocation files
    are never dropped) and every per-invocation ``summary_<ts>.json``, later
    files winning — run_baseline.py's per-invocation summary accounting, with the merge key
    switched to ``task_id`` (``config_file`` fallback for rows without one) because
    config paths go through per-launch ordinal-symlink sets and are not stable
    across invocations. The ``summary_*.json`` glob excludes the canonical
    ``summary.json`` (no underscore)."""
    rows_by_key: dict[str, dict[str, Any]] = {}
    for path in [result_dir / "summary.json", *sorted(result_dir.glob("summary_*.json"))]:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            rows_by_key[str(row.get("task_id") or row.get("config_file"))] = row
    return list(rows_by_key.values())


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.grounding_reasoning_effort is not None and args.backbone != "gpt":
        # Only GPTAgent takes the kwarg; the other backbones' constructors would TypeError.
        parser.error("--grounding_reasoning_effort is GPT-only (--backbone gpt)")
    if args.backbone == "claude":
        # Each would reach Sonnet 5 as a 400 on every call, i.e. every episode an
        # agent_error counted 0.0: it rejects any top_p, any temperature but 1.0,
        # and effort "none".
        for flag, value in (("--temperature", args.temperature), ("--top_p", args.top_p)):
            if value is not None:
                parser.error(f"{flag} is refused for --backbone claude (Sonnet 5 rejects it; the backbone "
                             "sends no sampling parameter)")
        if args.reasoning_effort == "none":
            parser.error("--reasoning_effort none is refused for --backbone claude (the API accepts "
                         "low|medium|high|xhigh|max)")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    os.makedirs(args.result_dir, exist_ok=True)

    # Load the repo-root .env so WebArena's site URLs (SHOPPING/REDDIT/...) are
    # in os.environ before WebArena is imported — its env_config asserts them at
    # import time (a bare `python run_graphpc.py` otherwise crashes on import).
    # Mirrors the OSWorld runner. Best-effort: no-op if python-dotenv is absent
    # (e.g. the --dry_run CI env) and never overrides an already-exported var.
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001 — export still works if this fails.
        pass

    if args.dry_run:
        results = run_dry(args)
        _write_results(results, Path(args.result_dir) / "dry_run_summary.json")
        logger.info("dry_run results: %s", [r.__dict__ for r in results])
        return 0

    # Live path: WebArena must be installed.
    webarena_root = Path(args.webarena_root)
    _validate_webarena_root(webarena_root)
    _ensure_webarena_on_path(webarena_root)

    # Re-create the openai 0.27 surface WebArena's LLM scorer expects on top of
    # the openai 2.x install our GPT backbone needs. MUST run before any
    # WebArena / evaluation_harness / llms import — ``llms/providers/openai_utils``
    # does ``import openai.error`` and evaluates a ``errors=(openai.error.…,)``
    # default arg at module load, which would otherwise crash under 2.x. See
    # src/benchmarks/webarena/_openai_compat.py.
    from src.benchmarks.webarena._openai_compat import install_openai_027_compat

    install_openai_027_compat()

    # Lazy imports — only when we actually intend to run live. (ScriptBrowserEnv
    # is imported inside _make_script_browser_env, which builds + recycles it.)
    from evaluation_harness.evaluators import evaluator_router  # type: ignore[import-not-found]
    from src.benchmarks.webarena.graphpc_agent import WebArenaGraphPCAgent

    # Env-hardening (installed IDENTICALLY in the baseline runner so both arms see
    # the same env behaviour, Δ unaffected): (1) bounded re-fetch of flaky obs
    # processing (zero-width layout ZeroDivisionError / Playwright TimeoutError);
    # (2) recreate a missing per-page CDP client on demand (tab-opening tasks that
    # upstream eats as [Unhandled Error] AttributeError 'Page'.'client').
    from src.benchmarks.webarena._env_hardening import (
        install_obs_retry_patch,
        install_page_client_patch,
    )

    install_obs_retry_patch()
    install_page_client_patch()

    # Passed only when set (GPT-only, enforced above), so an omitted flag passes
    # nothing and every backbone keeps its own default.
    grounding_kwargs = (
        {"grounding_reasoning_effort": args.grounding_reasoning_effort}
        if args.grounding_reasoning_effort is not None else {}
    )
    agent = WebArenaGraphPCAgent.build_default(
        model=args.model,
        backbone_type=args.backbone,
        support_set_root=args.support_set_root,
        observation_type=args.observation_type,
        max_actions=args.max_actions,
        max_trajectory_length=args.max_trajectory_length,
        two_pass=args.two_pass,
        enable_completion_gate=not args.completion_gate_off,
        completion_gate_window=args.completion_gate_window,
        retrieval_off=args.retrieval_off,
        temperature=args.temperature if args.temperature is not None else None,
        top_p=args.top_p if args.top_p is not None else None,
        max_tokens=args.max_tokens if args.max_tokens is not None else None,
        reasoning_effort=args.reasoning_effort if args.backbone in ("gpt", "claude") else None,
        **grounding_kwargs,
    )
    if args.backbone == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        # The backbone reads the key at call time; without one every episode would
        # die at its first call as a client-side TypeError, counted 0.0.
        raise SystemExit("FATAL: --backbone claude needs ANTHROPIC_API_KEY (repo .env or environment); "
                         "nothing was run.")

    _write_run_config(args=args, agent=agent, result_dir=Path(args.result_dir))

    # Node driver heap headroom before the first driver spawns.
    _set_node_heap_headroom()
    # Mirror WebArena's own run.py forced env settings (run.py:416/435):
    # sleep_after_execution=2.0 avoids the post-action obs snapshot racing
    # navigation (zero-width layout -> ZeroDivisionError); current_viewport_only
    # keeps the AXTree viewport-scoped like the baseline. The env is recycled
    # every --env_chunk_size episodes (planned driver-heap-OOM boundary); only
    # the env recycles — the agent (support set, counters) persists.
    env = _make_script_browser_env(args)

    # Resume (run_baseline-parallel): compute the skip-set once, up front.
    completed = _scored_task_ids(Path(args.result_dir)) if args.resume else set()
    if completed:
        logger.info(
            "resume: %d task(s) already scored in %s — skipping them (--no_resume to force)",
            len(completed), str(Path(args.result_dir) / "summary.json"),
        )

    # The invocation's summary file is stamped ONCE (microsecond-precision, so
    # rapid successive invocations can never collide on a name) and persisted
    # after EVERY episode: a non-clean stop must never lose completed rows —
    # the resume skip-set reads the merged view, so a relaunch skips exactly
    # what completed.
    invocation_summary_path = Path(args.result_dir) / (
        f"summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )

    results: list[EpisodeResult] = []
    skipped = 0
    executed = 0  # env-recycle cadence counts EXECUTED episodes only
    for config_path in _load_config_files(args):
        with config_path.open("r", encoding="utf-8") as f:
            config_payload = json.load(f)
        task_id = _task_id_from_config(config_path, config_payload)
        if task_id in completed:
            # Resume skip happens BEFORE any env / evaluator / agent work: a
            # skipped task re-executes NO side effect — no model call, no store
            # write (the cold-store trajectory must not take duplicate entries
            # from re-runs; the store's add-time dedup is only a second belt,
            # resume must not rely on it).
            skipped += 1
            logger.info("[%s] resume-skip: task %s already scored", config_path.name, task_id)
            continue
        env = _maybe_recycle_env(
            env, episode_idx=executed, chunk_size=args.env_chunk_size,
            make_env=lambda: _make_script_browser_env(args),
        )
        executed += 1
        evaluator = evaluator_router(str(config_path))
        result_dir = Path(args.result_dir)
        # Token-counter snapshot for THIS episode's delta on the crash path
        # (the counters don't reset on agent.reset, so this pre-call value is the
        # correct start; the normal path snapshots the same value inside
        # run_episode_live after reset).
        episode_token_start = _collect_token_usage(agent)
        try:
            res = run_episode_live(
                agent=agent, env=env, evaluator=evaluator,
                config_path=config_path, config_payload=config_payload,
                max_steps=args.max_steps, result_dir=result_dir,
            )
            results.append(res)
            # Crash-safety: the row is on disk BEFORE the next episode starts.
            _write_results(results, invocation_summary_path)
            logger.info("[%s] success=%s score=%s steps=%s",
                        config_path.name, res.success, res.score, res.steps)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Episode failed: %s", config_path)
            # Cross-arm rule: an agent-logic failure is a real 0.0 (COUNTED,
            # symmetric with the baseline's ValueError -> stop -> [Result] FAIL);
            # only infra (browser/env/driver/network/provider-API) is EXCLUDED,
            # symmetric with the baseline's [OpenAI Error] exclusion. Either way
            # the task is named, never silently dropped (measurement integrity).
            error_type = _classify_episode_error(exc)
            is_agent = error_type == "agent_error"
            score = 0.0 if is_agent else None
            # Still dump telemetry for the crashed episode (failure-path coverage).
            # Best-effort inside the writer so it never masks the real error.
            _write_episode_telemetry(
                agent=agent, result_dir=result_dir,
                task_id=task_id,
                config_file=str(config_path), score=score, success=False,
                steps=0, error=str(exc), error_type=error_type,
                token_start=episode_token_start, error_class=type(exc).__name__,
            )
            results.append(EpisodeResult(
                config_file=str(config_path), success=False, score=score, steps=0,
                errored=not is_agent, error_type=error_type, task_id=task_id,
            ))
            _write_results(results, invocation_summary_path)

    # Per-invocation summary accounting (mirrors run_baseline): this invocation's results go to an
    # immutable summary_<ts>.json; the canonical summary.json is rebuilt as the
    # union across invocations, so a resumed invocation MERGES into the earlier
    # batches instead of clobbering them.
    result_dir = Path(args.result_dir)
    # Final write covers the zero-episode invocation (nothing executed, so the
    # per-episode writes never fired); otherwise idempotent.
    _write_results(results, invocation_summary_path)
    merged = _merge_invocation_summaries(result_dir)
    summary_path = result_dir / "summary.json"
    if not merged and summary_path.exists() and summary_path.stat().st_size > len("[]"):
        # Belt (run_baseline-parallel): never clobber a non-empty canonical
        # summary with an empty merged set.
        logger.warning("Refusing to overwrite non-empty %s with an empty result set.", summary_path)
    else:
        _dump_json_atomic(merged, summary_path)

    # Report on the canonical MERGED view so the logged numbers match summary.json.
    scored = [r for r in merged if not r.get("errored")]
    errored = [r for r in merged if r.get("errored")]
    n_scored, n_err = len(scored), len(errored)
    # Name each errored (EXCLUDED) task WITH its error class so contamination
    # accounting (eval_error / infra_error) can attribute correctly. agent_error
    # rows are NOT here — they are counted 0.0 in `scored`.
    errored_tagged = [(r.get("config_file"), r.get("error_type")) for r in errored]
    if n_scored:
        avg = sum(float(r.get("score") or 0.0) for r in scored) / n_scored
        logger.info("Scored %d task(s): avg=%.4f | %d errored (excluded from avg): %s | "
                    "%d resume-skipped this invocation",
                    n_scored, avg, n_err, errored_tagged, skipped)
    elif merged:
        logger.info("All %d task(s) errored (no scored tasks; no avg): %s",
                    n_err, errored_tagged)
    from src.benchmarks.webarena._env_hardening import OBS_RETRY_STATS, PAGE_CLIENT_STATS
    logger.info("obs_retry_stats: %s | page_client_stats: %s", OBS_RETRY_STATS, PAGE_CLIENT_STATS)
    return 0


def _write_run_config(*, args: argparse.Namespace, agent, result_dir: Path) -> None:
    """Full resolved launch config into the run record.
    Records the parsed CLI args, the switch state read back off the constructed
    agent, and the measurement-integrity facts a later reader would otherwise
    have to infer — whether the retrieval text arm is genuinely live, and the env
    knobs that change scoring or embedding."""
    switch_state = getattr(agent, "ablation_switch_state", None)
    embedder = getattr(getattr(agent, "retrieval_agent", None), "embedder", None)
    payload = {
        "argv": list(sys.argv),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "ablation_switches": switch_state() if callable(switch_state) else None,
        "retrieval_text_arm": {
            "embedder": type(embedder).__name__ if embedder is not None else None,
            "available": bool(getattr(embedder, "available", False)),
            "model_name": getattr(embedder, "model_name", None),
        },
        "env": {
            key: os.environ.get(key)
            for key in ("GRAPHPC_EMBED_BATCH_SIZE", "WEBARENA_JUDGE_MODEL", "NODE_OPTIONS")
        },
        "backbone_effort": _backbone_effort_readback(agent),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _backbone_effort_readback(agent) -> Optional[dict[str, Any]]:
    """The GPT, Claude or DeepSeek backbone's effort settings and what each call
    type will SEND, read back off the built backbone through its own request
    builder (not off the CLI args, which is what ``args`` above already records).
    None for any other backbone."""
    from src.agents.backbone.claude.main import ClaudeAgent
    from src.agents.backbone.deepseek.main import DeepSeekAgent
    from src.agents.backbone.gpt.main import GPTAgent

    backbone = getattr(getattr(agent, "reasoning_agent", None), "backbone", None)
    if isinstance(backbone, DeepSeekAgent):
        resolved = {}
        for route, thinking in (("deciding", True), ("grounding", False)):
            request: dict[str, Any] = {}
            backbone._set_thinking(request, thinking)
            resolved[route] = {"thinking": request["extra_body"]["thinking"]["type"],
                               "effort": request.get("reasoning_effort"), "base_url": backbone.base_url}
        return {"model_name": backbone.model_name, "reasoning_effort": backbone.reasoning_effort,
                "resolved": resolved}
    if isinstance(backbone, ClaudeAgent):
        built = {
            "deciding": backbone._deciding_request(system_prompt="", user_prompt="", observation=None,
                                                   max_tokens=None, use_thinking=True, json_schema=None),
            "grounding": backbone._grounding_request(system_prompt="", user_prompt="", observation=None,
                                                     max_tokens=None, json_schema={}, schema_name=None),
        }
        return {
            "model_name": backbone.model_name,
            "reasoning_effort": backbone.reasoning_effort,
            "resolved": {
                route: {"thinking": request["extra_body"]["thinking"]["type"],
                        "effort": request["extra_body"]["output_config"]["effort"],
                        "max_tokens": request["max_tokens"]}
                for route, request in built.items()
            },
        }
    if not isinstance(backbone, GPTAgent):
        return None
    return {
        "model_name": backbone.model_name,
        "reasoning_effort": backbone.reasoning_effort,
        "grounding_reasoning_effort": backbone.grounding_reasoning_effort,
        "resolved": {
            "deciding": backbone._build_reasoning(use_thinking=True)["effort"],
            "grounding": backbone._build_reasoning(use_thinking=True, grounding=True)["effort"],
        },
    }


def _dump_json_atomic(rows: list[dict[str, Any]], target: Path) -> None:
    """Atomic JSON write: dump to a temp file in the SAME dir, then os.replace
    (atomic rename on POSIX — the support store's write pattern). A kill -9 at
    any moment leaves the previous complete file or the new complete file,
    never a torn one, so the resume skip-set / summary merge can always parse
    what they find. Chosen over a per-episode JSONL append because it keeps
    the summary format and merge untouched and makes torn records impossible
    by construction (rewriting an <=812-row array per episode is trivial next
    to an episode). The pid-tagged ``.tmp`` suffix keeps concurrent writers
    apart and keeps the temp out of the ``summary_*.json`` merge glob."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp_path, target)


def _write_results(results: Iterable[EpisodeResult], target: Path) -> None:
    _dump_json_atomic([r.__dict__ for r in results], target)


if __name__ == "__main__":
    raise SystemExit(main())
