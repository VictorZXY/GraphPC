"""Vendored runner for the WebArena baseline (bare-control PromptAgent).

The baseline arm is WebArena's OWN ``run.py --agent_type prompt`` loop — we do
NOT reimplement it. This wrapper only sets up the process so that loop runs
under our stack, then calls WebArena's own ``prepare`` / ``construct_agent`` /
``test`` verbatim (WebArena stays strictly read-only — we import and call).

Two problems it solves:
  (a) plain ``python run.py`` crashes under openai 2.x — our 0.27 compat shim
      (``_openai_compat.install_openai_027_compat``) must be installed BEFORE
      any WebArena ``llms`` / ``evaluation_harness`` import.
  (b) run.py's __main__ builds its task list from a CONTIGUOUS index range
      (--test_start_idx/--test_end_idx); Lite-165 is a non-contiguous id set, so
      we build ``test_file_list`` from our id list instead and hand it to
      ``test`` directly.

gpt-5* reasoning-model compatibility (all handled at OUR seam, WebArena
read-only — verified live against gpt-5.4-mini):
  * reasoning-effort symmetry with the GraphPC arm via
    ``WEBARENA_AGENT_REASONING_EFFORT`` injection in the chat shim.
  * the chat shim rewrites max_tokens -> max_completion_tokens and drops
    temperature/top_p for gpt-5* under reasoning_effort (the model rejects them).
  * ``openai.error.OpenAIError`` is aliased in the shim (run.py catches it).
  * a tiktoken fallback (o200k_base) for gpt-5* — WebArena's Tokenizer calls
    ``tiktoken.encoding_for_model`` which KeyErrors on gpt-5* ids.

NOTE: unlike ``run_graphpc.py`` there is NO dry/live split — this wrapper is
baseline-only and always drives the real WebArena loop; its pure helpers
(task-list construction, forced-override application, summary assembly) run
without importing WebArena.

Scores: WebArena's ``test()`` keeps per-task scores in a LOCAL list and never
returns them; it only logs ``[Result] (PASS|FAIL) <config_file>`` per task (plus
a final average). We attach a logging handler to run.py's ``logger`` to capture
those lines and assemble ``<result_dir>/summary.json`` in the same shape
``run_graphpc.py`` writes, so downstream Δ tooling reads both arms identically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("desktopenv.experiment")

# run.py logs one line per task: "[Result] (PASS) config_files/5.json".
_RESULT_LINE_RE = re.compile(r"\[Result\]\s+\((PASS|FAIL)\)\s+(.+)$")
# run.py logs "[Config file]: <path>" at the START of each task, before any
# error line — used to attribute a subsequent [OpenAI Error]/[Unhandled Error]
# (which carry no task id) to the current task.
_CONFIG_LINE_RE = re.compile(r"\[Config file\]:\s*(.+)$")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the WebArena baseline (PromptAgent) by driving "
            "WebArena's own run.py loop over a non-contiguous task-id set. "
            "Read-only on WebArena core."
        )
    )
    p.add_argument("--model", type=str, required=True, help="e.g. claude-sonnet-5")
    p.add_argument("--result_dir", type=str, required=True)
    p.add_argument(
        "--reasoning_effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="high",
        help="Sets WEBARENA_AGENT_REASONING_EFFORT for effort-symmetry with the "
             "GraphPC arm (injected into gpt-5* chat calls, and with adaptive "
             "thinking into claude-sonnet-5 calls, by the compat shim).",
    )
    # Task set (non-contiguous). Exactly one of these.
    p.add_argument("--task_ids_json", type=str, default=None,
                   help='JSON file with a "task_ids" list (e.g. the Lite-165 set).')
    p.add_argument("--task_ids", type=str, default=None,
                   help="Comma-separated task ids (smoke), e.g. 4,7,15.")

    # WebArena location + resume.
    p.add_argument("--webarena_root", type=str,
                   default=str(_REPO_ROOT.parent / "WebArena"))
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Skip already-finished tasks (WebArena get_unfinished). Default on.")
    p.add_argument("--no_resume", dest="resume", action="store_false",
                   help="Re-run every task even if a prior result html exists.")

    # Mirrored run.py agent/model args (their defaults, unless noted).
    p.add_argument("--instruction_path", type=str,
                   default="agent/prompts/jsons/p_cot_id_actree_2s.json",
                   help="PromptAgent instruction json (generated by prepare()). "
                        "run.py's own config() default points at a path absent "
                        "from the checkout, so we default to the canonical CoT "
                        "id-actree prompt.")
    p.add_argument("--temperature", type=float, default=1.0)   # run.py default
    p.add_argument("--top_p", type=float, default=0.9)         # run.py default
    p.add_argument("--max_tokens", type=int, default=384,      # run.py default
                   help="Per-call completion cap. NOTE: run.py's 384 default was "
                        "sized for non-reasoning models; for gpt-5* at high "
                        "effort the reasoning tokens share this budget — raise if "
                        "answers truncate.")
    p.add_argument("--max_steps", type=int, default=30)        # run.py default
    # run.py early_stop thresholds (run.py:104/110). Threaded through to run.py's
    # OWN config()/test()/early_stop via _build_run_args — this is run.py's
    # official parameter surface, so setting them here disables/relaxes the
    # self-kill with ZERO WebArena edits. Default = run.py's default (3/3); a
    # full-budget control arm passes e.g. 999/999.
    p.add_argument("--parsing_failure_th", type=int, default=3,
                   help="Consecutive parse failures before run.py early_stop fires "
                        "(run.py default 3). Pass a high value (e.g. 999) for a "
                        "full-budget baseline arm.")
    p.add_argument("--repeating_action_failure_th", type=int, default=3,
                   help="Consecutive repeated actions before run.py early_stop fires "
                        "(run.py default 3). Pass a high value (e.g. 999) for a "
                        "full-budget baseline arm.")
    p.add_argument("--env_chunk_size", type=int, default=15,
                   help="Run WebArena's test() over the task list in chunks of N "
                        "(planned Playwright-driver lifecycle boundary; test() "
                        "creates + closes a fresh env per call, so this recycles "
                        "the driver every N tasks and mitigates heap OOM).")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


# ---------------------------------------------------------------------------
# Pure helpers (no WebArena import)
# ---------------------------------------------------------------------------


def build_test_file_list(
    *, task_ids_json: Optional[str], task_ids: Optional[str]
) -> list[str]:
    """Build the WebArena ``config_files/<id>.json`` list from OUR id set.

    Paths are relative to the WebArena root (the wrapper chdirs there so
    config storage_state ``./.auth/...`` paths resolve). Exactly one source:
    a JSON file's ``task_ids`` key, or a comma-separated ``--task_ids`` list."""
    if task_ids_json:
        with open(task_ids_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ids = payload["task_ids"]
    elif task_ids:
        ids = [tok.strip() for tok in task_ids.split(",") if tok.strip()]
    else:
        raise ValueError("Provide --task_ids_json or --task_ids.")
    return [f"config_files/{str(i)}.json" for i in ids]


def apply_webarena_main_overrides(ns: argparse.Namespace) -> None:
    """Apply the SAME forced overrides WebArena's run.py __main__ applies, so
    the baseline runs under the exact conditions their harness intends. Values
    are literal mirrors of run.py (sleep_after_execution=2.0 [run.py:416];
    render=False / render_screenshot=True / save_trace_enabled=True [431-433];
    current_viewport_only=True [435]). ``render_screenshot`` is NOT a config()
    arg — run.py adds it in __main__, so we set it here too."""
    ns.sleep_after_execution = 2.0
    ns.render = False
    ns.render_screenshot = True
    # DOCUMENTED DEVIATION from run.py:433 (save_trace_enabled=True): Playwright
    # trace recording accumulates in the Node driver and OOM-kills it on long
    # 30-step tasks (under the 4GB default heap and under an 8GB NODE_OPTIONS
    # heap alike — trace serialization on context close is the cause). Traces
    # are debug artifacts, not measurement
    # inputs (scores unaffected); the GraphPC arm records none, so disabling
    # them here IMPROVES arm symmetry. Resume markers are the render HTMLs
    # (render_screenshot above), which stay on.
    ns.save_trace_enabled = False
    ns.current_viewport_only = True


def _task_id_from_config(config_file: str) -> str:
    """WebArena config basename stem is the task id (``config_files/5.json`` ->
    ``5``). Robust to the temp-dir rewrite run.py does for auth tasks."""
    return os.path.basename(config_file).split(".")[0]


def assemble_summary(
    config_files: list[str],
    results_by_task_id: dict[str, bool],
    error_by_task_id: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Build the per-task summary from captured ``[Result]`` verdicts, in the
    same shape ``run_graphpc.py`` writes. A task with a PASS/FAIL verdict is
    scored (binary — the float never leaves run.py's test()); a task WITHOUT a
    verdict was dropped by WebArena's own generic error handling and stays
    EXCLUDED, labelled by its captured infra class:
    ``baseline_openai_error`` (provider/API) / ``baseline_unhandled_error``
    (env/eval) where the log showed one, else ``baseline_error`` (unexplained
    no-[Result] — the residual "not distinguishable" case, kept excluded)."""
    error_by_task_id = error_by_task_id or {}
    summary: list[dict[str, Any]] = []
    for config_file in config_files:
        task_id = _task_id_from_config(config_file)
        if task_id in results_by_task_id:
            passed = results_by_task_id[task_id]
            summary.append({
                "config_file": config_file,
                "score": 1.0 if passed else 0.0,
                "success": passed,
                "steps": None,   # not exposed by run.py's loop
                "errored": False,
                "error_type": None,
            })
        else:
            summary.append({
                "config_file": config_file,
                "score": None,
                "success": False,
                "steps": None,
                "errored": True,
                "error_type": error_by_task_id.get(task_id, "baseline_error"),
            })
    return summary


def _merge_invocation_summaries(result_dir: Path) -> list[dict[str, Any]]:
    """Union every per-invocation ``summary_<ts>.json``, keyed by
    config_file with later files winning, so the canonical ``summary.json``
    reflects ALL tasks run across resumed invocations — not just the last batch.
    The ``summary_*.json`` glob excludes the canonical ``summary.json`` (no
    underscore), so this rebuilds purely from the immutable per-invocation files."""
    rows_by_key: dict[str, dict[str, Any]] = {}
    for path in sorted(result_dir.glob("summary_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                rows = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            rows_by_key[str(row.get("config_file"))] = row
    return list(rows_by_key.values())


class _ResultCapture(logging.Handler):
    """Capture run.py's per-task log lines (test() logs but never returns scores):
    ``[Result] (PASS|FAIL) <config_file>`` -> ``{task_id: passed}``, and the
    error classes that yield NO [Result] -> ``{task_id: error_type}``.
    Error lines carry no task id, so they are attributed to the most recent
    ``[Config file]:`` line."""

    def __init__(self) -> None:
        super().__init__()
        self.results_by_task_id: dict[str, bool] = {}
        self.error_by_task_id: dict[str, str] = {}
        self._current_task_id: Optional[str] = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        config_match = _CONFIG_LINE_RE.search(msg)
        if config_match:
            self._current_task_id = _task_id_from_config(config_match.group(1).strip())
            return
        result_match = _RESULT_LINE_RE.search(msg)
        if result_match:
            verdict, config_file = result_match.group(1), result_match.group(2).strip()
            self.results_by_task_id[_task_id_from_config(config_file)] = verdict == "PASS"
            return
        # No [Result] classes. [OpenAI Error] is provider/API infra; [Unhandled
        # Error] is env/eval infra in practice (WebArena's PromptAgent routes its
        # own logic failures through ValueError -> [Result] FAIL, so a non-ValueError
        # escape is not agent-logic). Both stay EXCLUDED, labelled by class rather
        # than as a blanket baseline_error.
        if self._current_task_id is not None:
            if "[OpenAI Error]" in msg:
                self.error_by_task_id[self._current_task_id] = "baseline_openai_error"
            elif "[Unhandled Error]" in msg:
                self.error_by_task_id[self._current_task_id] = "baseline_unhandled_error"


def _install_tiktoken_fallback() -> None:
    """WebArena's Tokenizer calls ``tiktoken.encoding_for_model`` which KeyErrors
    on gpt-5* ids. Fall back to o200k_base (the gpt-4o/gpt-5 family encoding) for
    unknown models only; known models keep their real encoding. WebArena is
    read-only, so this compat lives here."""
    import tiktoken

    original = tiktoken.encoding_for_model

    def _with_fallback(model_name: str):
        try:
            return original(model_name)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")

    tiktoken.encoding_for_model = _with_fallback


# ---------------------------------------------------------------------------
# Browser lifecycle (driver-heap-OOM mitigation)
# ---------------------------------------------------------------------------
#
# WebArena's run.py test() creates its ScriptBrowserEnv INSIDE the function and
# close()s it at the end (envs.py), and env.reset() already recreates the
# Playwright driver per task. But a long sequential run still OOMs the Node
# driver (V8 "heap out of memory"; forced save_trace_enabled contributes). We call
# test() in CHUNKS: each call gets a fresh env object + a hard close at chunk end
# (a PLANNED every-N-tasks driver-lifecycle boundary, part of the run config —
# NOT an unplanned mid-run crash-restart). Composes with --resume
# (get_unfinished filters BEFORE chunking). Same default N in both runners.


def _set_node_heap_headroom() -> None:
    """Give the Playwright Node driver more V8 heap. ``setdefault`` so an explicit
    user NODE_OPTIONS wins. Must run before the driver subprocess spawns."""
    os.environ.setdefault("NODE_OPTIONS", "--max-old-space-size=8192")


def _run_test_in_chunks(run_module, run_args, agent, test_file_list, *, chunk_size):
    """Call WebArena's ``test()`` over the (already resume-filtered) task list in
    slices of ``chunk_size`` — a fresh env per chunk. A chunk crash is logged and
    the run continues to the next chunk (its tasks retry on the next --resume)."""
    n = len(test_file_list)
    step = chunk_size if chunk_size > 0 else n
    for start in range(0, n, step):
        chunk = test_file_list[start:start + step]
        logger.info("env chunk [%d:%d) of %d (planned fresh-env lifecycle boundary)",
                    start, start + len(chunk), n)
        try:
            run_module.test(run_args, agent, chunk)
        except Exception:  # noqa: BLE001 — a chunk crash must not abort the rest.
            logger.exception("chunk starting at %d failed; continuing to next chunk", start)


# ---------------------------------------------------------------------------
# Main (drives the real WebArena loop)
# ---------------------------------------------------------------------------


def _build_run_args(run_module, args: argparse.Namespace, *, result_dir: Path) -> argparse.Namespace:
    """Reuse run.py's own ``config()`` (its defaults + validation) by feeding a
    controlled argv, then apply the __main__ forced overrides. Faithful to their
    parser so we don't drift from their defaults."""
    argv = [
        "run_baseline",
        "--agent_type", "prompt",
        "--provider", "openai",
        "--model", args.model,
        "--result_dir", str(result_dir),
        "--instruction_path", args.instruction_path,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--max_tokens", str(args.max_tokens),
        "--max_steps", str(args.max_steps),
        # run.py's own early_stop threshold args — run.py config() defines both,
        # test() reads them into early_stop_thresholds (run.py:226-227) and passes
        # them to early_stop() (run.py:288). Passing them here is the official,
        # WebArena-read-only way to relax the self-kill for a full-budget arm.
        "--parsing_failure_th", str(args.parsing_failure_th),
        "--repeating_action_failure_th", str(args.repeating_action_failure_th),
        "--observation_type", "accessibility_tree",
        "--action_set_tag", "id_accessibility_tree",
    ]
    saved_argv = sys.argv
    try:
        sys.argv = argv
        run_args = run_module.config()
    finally:
        sys.argv = saved_argv
    apply_webarena_main_overrides(run_args)
    return run_args


def _check_qwen_model_pinned(model: str) -> None:
    """Stop a Qwen baseline launch that would score a model the other arm is not
    running. No-op for every other backbone, so their launch path is unchanged.

    The compat shim routes every qwen-prefixed slug to OpenRouter — narrowing
    that would send a mistyped slug to OpenAI with OPENAI_API_KEY, a worse
    failure — and raises on a slug that is not the pinned one. But that raise
    lands inside run.py's per-task ``except Exception``, so it only keeps the
    wrong model from being SCORED: the loop walks the whole list, raising once
    per task. Stopping the RUN has to happen here, before the first task starts.
    """
    from src.benchmarks.webarena import _openai_compat

    if not _openai_compat._is_qwen_model(model):
        return
    if model != _openai_compat._QWEN_MODEL:
        raise SystemExit(
            f"FATAL: --model {model!r} is not the slug this cell's Qwen arms pin "
            f"({_openai_compat._QWEN_MODEL!r}). Both arms must run the same model or the paired "
            "comparison means nothing, and every qwen slug routes to OpenRouter, so this one "
            "would have been answered normally. Fix --model and relaunch."
        )


def _check_claude_launch(model: str, reasoning_effort: str, temperature: float) -> None:
    """Stop a Claude baseline launch that would turn every task into an
    ``[OpenAI Error]`` before the first task starts. No-op for every other model,
    so their launch path is unchanged. Called after ``.env`` is loaded.

    The compat shim adapts the EXACT id ``claude-sonnet-5`` only, so a sibling
    claude id would reach OpenAI and 404 on every call; Sonnet 5 answers effort
    ``none`` and any temperature but 1.0 with a 400; and a missing key fails every
    call client-side. The shim raises each of these per task, inside run.py's
    per-task handler, so it cannot stop the run — this does."""
    from src.benchmarks.webarena import _openai_compat

    if not (isinstance(model, str) and model.lower().startswith("claude")):
        return
    problems = []
    if model != _openai_compat._CLAUDE_MODEL:
        problems.append(f"--model {model!r} is not the one claude id the shim routes "
                        f"({_openai_compat._CLAUDE_MODEL!r}); it would go to OpenAI")
    if reasoning_effort == "none":
        problems.append("--reasoning_effort none is a 400 on Sonnet 5 (low|medium|high|xhigh)")
    if temperature != 1.0:
        problems.append(f"--temperature {temperature} is a 400 on Sonnet 5 (only 1.0 is accepted)")
    if not os.environ.get(_openai_compat._CLAUDE_API_KEY_ENV):
        problems.append(f"{_openai_compat._CLAUDE_API_KEY_ENV} is not set (repo .env or environment)")
    if problems:
        raise SystemExit("FATAL: claude baseline launch refused, nothing was run: " + "; ".join(problems))


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    # Resolve output + input paths to ABSOLUTE before we chdir into WebArena
    # (config storage_state paths are CWD-relative, so we must chdir there).
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    task_ids_json = str(Path(args.task_ids_json).resolve()) if args.task_ids_json else None

    # Load the repo .env (site URLs + OPENAI_API_KEY) before WebArena import.
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass

    # Effort-symmetry: set the env var BEFORE the shim's first call so the
    # injection (compat shim) applies to the baseline's gpt-5* chat calls.
    os.environ["WEBARENA_AGENT_REASONING_EFFORT"] = args.reasoning_effort

    # A Claude launch that would fail every task stops here. No-op otherwise.
    _check_claude_launch(args.model, args.reasoning_effort, args.temperature)

    # Install the 0.27 compat shim BEFORE importing anything WebArena.
    from src.benchmarks.webarena._openai_compat import install_openai_027_compat

    install_openai_027_compat()

    # A Qwen launch must send the slug both arms pin; the shim's own guard fires
    # inside run.py's per-task handler and cannot stop the run. No-op otherwise.
    _check_qwen_model_pinned(args.model)

    # WebArena must be installed; chdir into it (config + auth paths are relative).
    webarena_root = Path(args.webarena_root).resolve()
    if not (webarena_root / "run.py").exists():
        raise FileNotFoundError(
            f"Expected a WebArena checkout with {webarena_root / 'run.py'}. "
            "Clone WebArena to ../WebArena (or pass --webarena_root)."
        )
    if str(webarena_root) not in sys.path:
        sys.path.insert(0, str(webarena_root))
    os.chdir(webarena_root)

    # run.py's test() renews cookies per task via `subprocess.run(["python",
    # "browser_env/auto_login.py", ...])` (run.py:262) — a BARE `python`,
    # assuming an activated env. Outside one, that resolves to a system python
    # without playwright, the subprocess fails silently (no check=True), and
    # run.py:271's `assert os.path.exists(storage_state)` kills every
    # login-required task. Prepend THIS interpreter's bin dir so the child
    # `python` is the same env we're running in.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

    _install_tiktoken_fallback()

    import run as run_module  # WebArena's run.py; safe to import (loop is under __main__).

    # Env-hardening — installed IDENTICALLY to the GraphPC runner so both arms see
    # the same env (Δ safe): (1) bounded re-fetch of flaky obs processing; (2)
    # recreate a missing per-page CDP client on demand (tab-opening tasks upstream
    # eats as [Unhandled Error] AttributeError 'Page'.'client').
    from src.benchmarks.webarena._env_hardening import (
        install_obs_retry_patch,
        install_page_client_patch,
    )

    install_obs_retry_patch()
    install_page_client_patch()

    run_args = _build_run_args(run_module, args, result_dir=result_dir)
    run_module.prepare(run_args)  # generates prompt jsons + result dirs

    test_file_list = build_test_file_list(
        task_ids_json=task_ids_json, task_ids=args.task_ids
    )
    if args.resume:
        test_file_list = run_module.get_unfinished(test_file_list, str(result_dir))

    _set_node_heap_headroom()  # before the first Playwright driver spawns (inside test()).
    capture = _ResultCapture()
    run_module.logger.addHandler(capture)
    try:
        if not test_file_list:
            logger.info("No task left to run (all finished; use --no_resume to force).")
        else:
            logger.info("Running %d baseline task(s) in chunks of %d.",
                        len(test_file_list), args.env_chunk_size)
            agent = run_module.construct_agent(run_args)
            _run_test_in_chunks(run_module, run_args, agent, test_file_list,
                                chunk_size=args.env_chunk_size)
    except Exception:  # noqa: BLE001 — still write the partial summary below.
        logger.exception("Baseline run raised; writing partial summary.")
    finally:
        run_module.logger.removeHandler(capture)

    # Summary over the tasks executed THIS invocation (a resumed run covers only
    # the remaining tasks — WebArena persists per-task scores only in per-run logs).
    summary = assemble_summary(test_file_list, capture.results_by_task_id, capture.error_by_task_id)

    # Write this invocation's summary to a distinct, never-clobbered file,
    # then rebuild the canonical summary.json as the union across all invocations.
    # Overwriting summary.json with THIS invocation's (resume-filtered) set would,
    # on a re-invocation after everything finished, write [] over the complete
    # file; this way a fully-resumed rerun's empty batch adds nothing to the
    # merge, and the canonical file survives intact.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with (result_dir / f"summary_{timestamp}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    merged = _merge_invocation_summaries(result_dir)
    summary_path = result_dir / "summary.json"
    if not merged and summary_path.exists() and summary_path.stat().st_size > len("[]"):
        # Belt-and-suspenders: never clobber a non-empty canonical summary with an
        # empty merged set (e.g. per-invocation files deleted but summary.json kept).
        logger.warning("Refusing to overwrite non-empty %s with an empty result set.", summary_path)
        with summary_path.open("r", encoding="utf-8") as f:
            merged = json.load(f)
    else:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    # Report on the canonical MERGED view so the logged numbers match summary.json.
    scored = [r for r in merged if not r["errored"]]
    passed = [r for r in scored if r["success"]]
    errored = [r for r in merged if r["errored"]]
    if scored:
        logger.info(
            "Baseline (merged across invocations): %d scored, %d passed (%.4f), %d errored -> %s",
            len(scored), len(passed), len(passed) / len(scored),
            len(errored), str(summary_path),
        )
    else:
        logger.info("Baseline (merged across invocations): 0 scored, %d errored -> %s",
                    len(errored), str(summary_path))
    from src.benchmarks.webarena._env_hardening import OBS_RETRY_STATS, PAGE_CLIENT_STATS
    logger.info("obs_retry_stats: %s | page_client_stats: %s", OBS_RETRY_STATS, PAGE_CLIENT_STATS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
