"""Site-partitioned PARALLEL launcher for ``run_graphpc.py`` on Lite-165 cold runs.

A cold run executes the site-pure partitions in PARALLEL (5 subprocesses, a
fraction of the sequential wall-clock), then the cross-site tasks SEQUENTIALLY
after a barrier.

DOCUMENTED DEVIATION: all partitions share ONE
support_set_root, so within-run support-set accumulation / retrieval ordering is
perturbed vs a fully sequential run — entries land in wall-clock order across
partitions rather than global task order. This is second-order for a COLD run
(the support set starts empty) and is the documented cost of the parallel
speed-up. Phase-2 cross-site tasks run AFTER the phase-1 barrier so they see the
accumulated (if unordered) support set.

Each subprocess is a normal ``run_graphpc.py`` invocation, so the env-lifecycle
knobs (--env_chunk_size / NODE_OPTIONS heap headroom) apply per subprocess
unchanged. Subprocess CWD = webarena_root so the config ``./.auth/...``
storage_state paths resolve; per-partition configs are symlinks with ABSOLUTE
targets into ``<webarena_root>/config_files/<id>.json`` so they resolve
regardless of CWD.

Merge: every partition's + phase-2's ``summary.json`` is concatenated into
``<run_root>/summary.json`` (same shape run_graphpc writes) with a rollup log.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # the pre-spawn sweep imports src.*
    sys.path.insert(0, str(_REPO_ROOT))
_RUN_GRAPHPC = _REPO_ROOT / "scripts" / "webarena" / "run_graphpc.py"
_DEFAULT_PLAN = _REPO_ROOT / "data" / "webarena" / "lite165_site_partitions.json"

logger = logging.getLogger("desktopenv.experiment")

_PHASE2_NAME = "cross_site"


# ---------------------------------------------------------------------------
# Plan + layout (pure)
# ---------------------------------------------------------------------------


def load_plan(path: str | Path) -> tuple[dict[str, list], list]:
    """Return (phase1_partitions {site: [ids]}, phase2_cross_site [ids])."""
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return plan["phase1_parallel_partitions"], plan["phase2_sequential_cross_site"]


def partition_paths(run_root: str | Path, name: str) -> tuple[Path, Path]:
    """(config_dir, result_dir) for a partition (or the phase-2 group)."""
    base = Path(run_root) / "partitions" / name
    return base / "configs", base / "results"


def create_config_symlinks(config_dir: Path, ids: list, webarena_root: Path) -> list[Path]:
    """Symlink ``<config_dir>/<id>.json`` -> ``<webarena_root>/config_files/<id>.json``
    with ABSOLUTE targets (resolve regardless of the subprocess CWD). Idempotent:
    an existing correct link is left; a stale one is replaced."""
    config_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(webarena_root).resolve() / "config_files"
    links: list[Path] = []
    for task_id in ids:
        target = source_root / f"{task_id}.json"
        link = config_dir / f"{task_id}.json"
        if link.is_symlink() or link.exists():
            if link.is_symlink() and Path(link).resolve() == target.resolve():
                links.append(link)
                continue
            link.unlink()
        link.symlink_to(target)
        links.append(link)
    return links


def build_run_graphpc_command(
    *,
    python_executable: str,
    config_dir: Path,
    result_dir: Path,
    support_set_root: Path,
    model: str,
    backbone: str,
    webarena_root: Path,
    log_level: str = "INFO",
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """The run_graphpc.py argv for one partition / the phase-2 group."""
    cmd = [
        python_executable, str(_RUN_GRAPHPC),
        "--backbone", backbone,
        "--model", model,
        # run_graphpc is always headless (baseline symmetry) — no --headless flag to pass.
        "--config_dir", str(config_dir),
        "--result_dir", str(result_dir),
        "--support_set_root", str(support_set_root),
        "--webarena_root", str(webarena_root),
        "--log_level", log_level,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


# ---------------------------------------------------------------------------
# Launch (side-effecting; subprocess.Popen injectable for tests)
# ---------------------------------------------------------------------------

PopenFactory = Callable[..., Any]


def _launch(cmd: list[str], *, cwd: Path, log_path: Path, popen: PopenFactory):
    """Start one subprocess, streaming stdout+stderr to ``log_path``. Returns
    (process, open_log_file) — the caller waits + closes the file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 — closed by caller
    proc = popen(cmd, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def run_phase1(
    partitions: dict[str, list],
    *,
    run_root: Path,
    webarena_root: Path,
    support_set_root: Path,
    model: str,
    backbone: str,
    python_executable: str,
    extra_args: Optional[list[str]] = None,
    popen: PopenFactory = subprocess.Popen,
) -> dict[str, int]:
    """Spawn ALL site partitions in parallel, then wait on all (the barrier).
    A failed partition does NOT abort the others — its rc is recorded. Returns
    {site: returncode}."""
    running: dict[str, tuple[Any, Any]] = {}
    for site, ids in partitions.items():
        config_dir, result_dir = partition_paths(run_root, site)
        create_config_symlinks(config_dir, ids, webarena_root)
        cmd = build_run_graphpc_command(
            python_executable=python_executable, config_dir=config_dir,
            result_dir=result_dir, support_set_root=support_set_root, model=model,
            backbone=backbone, webarena_root=webarena_root, extra_args=extra_args,
        )
        log_path = Path(run_root) / "partitions" / site / "run.log"
        logger.info("phase1 spawn partition %s (%d tasks)", site, len(ids))
        running[site] = _launch(cmd, cwd=webarena_root, log_path=log_path, popen=popen)

    returncodes: dict[str, int] = {}
    for site, (proc, log_file) in running.items():
        rc = proc.wait()
        log_file.close()
        returncodes[site] = rc
        level = logging.INFO if rc == 0 else logging.ERROR
        logger.log(level, "phase1 partition %s exited rc=%s", site, rc)
    return returncodes


def run_phase2(
    cross_site_ids: list,
    *,
    run_root: Path,
    webarena_root: Path,
    support_set_root: Path,
    model: str,
    backbone: str,
    python_executable: str,
    extra_args: Optional[list[str]] = None,
    popen: PopenFactory = subprocess.Popen,
) -> int:
    """Run the cross-site tasks SEQUENTIALLY through ONE run_graphpc invocation
    (after the phase-1 barrier), sharing the same support set. Returns rc."""
    if not cross_site_ids:
        return 0
    config_dir, result_dir = partition_paths(run_root, _PHASE2_NAME)
    create_config_symlinks(config_dir, cross_site_ids, webarena_root)
    cmd = build_run_graphpc_command(
        python_executable=python_executable, config_dir=config_dir,
        result_dir=result_dir, support_set_root=support_set_root, model=model,
        backbone=backbone, webarena_root=webarena_root, extra_args=extra_args,
    )
    log_path = Path(run_root) / "partitions" / _PHASE2_NAME / "run.log"
    logger.info("phase2 spawn cross_site (%d tasks) after barrier", len(cross_site_ids))
    proc, log_file = _launch(cmd, cwd=webarena_root, log_path=log_path, popen=popen)
    rc = proc.wait()
    log_file.close()
    logger.log(logging.INFO if rc == 0 else logging.ERROR, "phase2 cross_site exited rc=%s", rc)
    return rc


# ---------------------------------------------------------------------------
# Merge (pure)
# ---------------------------------------------------------------------------


# Error types excluded from the scored denominator (infra / evaluator classes
# that GraphPC telemetry can carry). agent_error and None (a scored result) are
# COUNTED, so they are deliberately absent; "episode_error" is the undivided infra/agent
# tag, excluded so telemetry carrying it still classifies.
_EXCLUDED_ERROR_TYPES = frozenset({"eval_error", "infra_error", "episode_error"})


def _rows_from_telemetry(partition_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct summary rows from the per-episode ``telemetry_<id>.json`` that
    survive a crash. A partition OOM-killed mid-run never reaches its
    end-of-run ``summary.json`` write, so without this its completed tasks vanish
    from the rollup denominator (the exact failure the env-chunking mitigates).
    ``errored`` is derived from ``error_type`` with the same rule the summary uses,
    and rows are tagged ``source="telemetry_fallback"`` for auditability."""
    rows: list[dict[str, Any]] = []
    for tpath in sorted(partition_dir.glob("telemetry_*.json")):
        try:
            with tpath.open("r", encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        error_type = rec.get("error_type")
        rows.append({
            "config_file": rec.get("config_file"),
            "score": rec.get("score"),
            "success": bool(rec.get("success")),
            "steps": rec.get("steps"),
            "errored": error_type in _EXCLUDED_ERROR_TYPES,
            "error_type": error_type,
            "source": "telemetry_fallback",
        })
    return rows


def merge_summaries(summary_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Concatenate every existing ``summary.json`` and compute the rollup: scored
    / success counts + errored counts by error_type. For a partition whose
    ``summary.json`` is absent (crashed before the end-of-run write), fall back to
    its surviving ``telemetry_*.json`` records so its completed tasks
    still count; only a partition with neither summary nor telemetry is recorded
    as ``missing``."""
    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in summary_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                merged.extend(json.load(f))
            continue
        recovered = _rows_from_telemetry(path.parent)
        if recovered:
            merged.extend(recovered)
        else:
            missing.append(str(path))

    errored = [r for r in merged if r.get("errored")]
    scored = [r for r in merged if not r.get("errored")]
    passed = [r for r in scored if r.get("success")]
    rollup = {
        "total": len(merged),
        "scored": len(scored),
        "success": len(passed),
        "errored": len(errored),
        "errored_by_type": dict(Counter(r.get("error_type") for r in errored)),
        "missing_summaries": missing,
        "recovered_from_telemetry": sum(1 for r in merged if r.get("source") == "telemetry_fallback"),
    }
    return merged, rollup


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------


def _presweep_support_set(support_set_root: Path) -> list[dict]:
    """One-shot keep-best dedup sweep on the shared support store before
    any partition spawns. Idempotent; writes ``_dedup_events.json`` beside the
    store for post-hoc decomposition. Best-effort — never aborts the run."""
    from src.agents.retrieval.support_set import LocalSupportSetStore

    if not (Path(support_set_root) / "webarena").exists():
        return []
    try:
        events = LocalSupportSetStore(root_dir=support_set_root, backend="webarena").keep_best_sweep()
    except Exception:  # noqa: BLE001 — dedup hygiene must never block a run.
        logger.exception("pre-spawn keep-best sweep failed; continuing without dedup")
        return []
    if events:
        out = Path(support_set_root) / "_dedup_events.json"
        try:
            with out.open("w", encoding="utf-8") as f:
                json.dump(events, f, indent=2)
        except OSError:
            pass
        logger.info("pre-spawn keep-best sweep: %d task_id group(s) deduped -> %s", len(events), out)
    return events


def run_all(
    *,
    partitions: dict[str, list],
    cross_site_ids: list,
    run_root: Path,
    webarena_root: Path,
    support_set_root: Path,
    model: str,
    backbone: str,
    python_executable: str,
    extra_args: Optional[list[str]] = None,
    popen: PopenFactory = subprocess.Popen,
) -> dict[str, Any]:
    """Phase 1 (parallel) -> BARRIER -> phase 2 (sequential) -> merge."""
    # Keep-best dedup sweep on the SHARED store PRE-SPAWN (after any seed
    # copy, before any partition starts — store quiescent, single process,
    # idempotent). No-op on a cold run (task_ids unique per partition); removes
    # warm seed duplicates that would otherwise crowd the injection slots.
    _presweep_support_set(support_set_root)
    phase1_rc = run_phase1(
        partitions, run_root=run_root, webarena_root=webarena_root,
        support_set_root=support_set_root, model=model, backbone=backbone,
        python_executable=python_executable, extra_args=extra_args, popen=popen,
    )
    phase2_rc = run_phase2(
        cross_site_ids, run_root=run_root, webarena_root=webarena_root,
        support_set_root=support_set_root, model=model, backbone=backbone,
        python_executable=python_executable, extra_args=extra_args, popen=popen,
    )
    summary_paths = [partition_paths(run_root, site)[1] / "summary.json" for site in partitions]
    if cross_site_ids:
        summary_paths.append(partition_paths(run_root, _PHASE2_NAME)[1] / "summary.json")
    merged, rollup = merge_summaries(summary_paths)

    out = Path(run_root) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    failed = [s for s, rc in phase1_rc.items() if rc != 0]
    return {
        "phase1_returncodes": phase1_rc,
        "phase2_returncode": phase2_rc,
        "failed_partitions": failed,
        "rollup": rollup,
        "summary_path": str(out),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Site-partitioned parallel launcher for run_graphpc.py (Lite-165 cold runs).")
    p.add_argument("--partitions_json", type=str, default=str(_DEFAULT_PLAN))
    p.add_argument("--run_root", type=str, required=True, help="the run directory, e.g. runs/<id>")
    p.add_argument("--webarena_root", type=str, default=str(_REPO_ROOT.parent / "WebArena"))
    p.add_argument("--model", type=str, default="claude-sonnet-5")
    p.add_argument("--backbone", type=str, default="claude", choices=["gpt", "claude", "deepseek", "qwen"])
    p.add_argument("--support_set_root", type=str, default=None,
                   help="Shared support set (default <run_root>/support_set).")
    p.add_argument("--python_executable", type=str, default=sys.executable,
                   help="Python for the run_graphpc subprocesses (must have "
                        "WebArena + GraphPC deps; defaults to the launcher's own).")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    # Resolve to absolute BEFORE building subprocess argv: the run_graphpc
    # subprocesses run with cwd=webarena_root, so a relative --run_root would
    # make them resolve config/result/support paths inside the WebArena
    # checkout (empty config glob -> silent 0-task "success").
    run_root = Path(args.run_root).resolve()
    webarena_root = Path(args.webarena_root).resolve()
    support_set_root = (Path(args.support_set_root).resolve()
                        if args.support_set_root else run_root / "support_set")
    partitions, cross_site_ids = load_plan(args.partitions_json)

    result = run_all(
        partitions=partitions, cross_site_ids=cross_site_ids, run_root=run_root,
        webarena_root=webarena_root, support_set_root=support_set_root,
        model=args.model, backbone=args.backbone,
        python_executable=args.python_executable,
    )
    r = result["rollup"]
    logger.info(
        "Parallel rollup: %d total | %d scored | %d success | %d errored %s | failed partitions: %s -> %s",
        r["total"], r["scored"], r["success"], r["errored"], r["errored_by_type"],
        result["failed_partitions"] or "none", result["summary_path"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
