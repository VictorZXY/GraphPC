"""Paired partial-score comparison of two OSWorld 2.0 arms over one task list.

OSWorld-V2's show_result.py cannot read the V2 layout (it hard-codes the 1.0
domain names), so this is our scorer over the files the V2 driver writes per task
(``<arm>/<action_space>/<observation_type>/<model>/<domain>/<task_id>/``):

* ``result.txt``               — the authoritative final score, ``"<float>\\n"``;
* ``result.json``              — the evaluator's dict (``score`` + ``partial_scores``), when returned;
* ``checkpoint_results.json``  — inline checkpoint records (``step``, ``status``, ``score``);
* ``phase_results.json``       — present only for a multi-phase task, which has no checkpoints
                                 (none of the 28 subset tasks is one; the handling serves any
                                 other task list).

Checkpoint columns are NA (never 0) for multi-phase tasks and for checkpoints whose
record is ``status: "skipped"`` (``intermediate_eval_safe=False``, 7 of the 28:
034, 035, 053, 067, 079, 092, 104, recorded as ``checkpoint_unsafe`` in the
pre-registration; the final score at max_steps is still recorded for every one of them) or
``"error"``; a checkpoint mean is computed only over tasks with a successful value
at that step in BOTH arms, so the per-checkpoint ``n`` is the same for both arms.
Ties: ``|delta| < COMPLETION_EPSILON`` counts as a tie in the sign test.

A task listed in the manifest but without ``result.txt`` in an arm is UNSCORED and
reported loudly (with the error row from ``<arm>/summary/results.json`` when one
exists); it is never counted as 0. So is a ``result.txt`` that is not a finite score
in [0, 1] — ``float()`` accepts ``nan`` and ``inf``, and a NaN would otherwise count as
a TIE in the exact sign test. Unless ``--allow_missing`` is given the script exits 1 in
that case, after printing everything it could compute.

    python scripts/osworld/tools/score_paired.py \\
        --baseline_root <control result tree> \\
        --graphpc_root <GraphPC result tree> \\
        --task_list data/osworld_v2/prereg_task_list.json \\
        --output_json <output json>

The Qwen qwen3.7-plus cell is scored the same way over its own trees (``baseline_qwen37plus``,
``graphpc_qwen37plus``). Both of its runners sanitize the model directory (``qwen/qwen3.7-plus`` ->
``qwen_qwen3.7-plus``; the control's delta D5), so the chain below resolves exactly as it does for
MiniMax-M3.

Each ``--*_root`` is the runner's ``--result_dir`` (the scorer resolves the unique
``<action_space>/<observation_type>/<model>`` chain beneath it) or, when results
were copied around, the ``<model>`` directory that holds the domain folders.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from pathlib import Path
from typing import Any, Optional

# A result tree may keep its support set inside it as support_set/, beside the action-space
# directory; like summary/ and logs/ it is not part of the task layout.
ARM_LAYOUT_SKIP = {"summary", "logs", "support_set"}
COMPLETION_EPSILON = 1e-9


def load_task_list(path: Path) -> list[tuple[str, str]]:
    """Manifest ``{"<domain>": ["001", ...]}`` -> ordered (domain, task_id) pairs."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"{path}: expected a non-empty {{domain: [task ids]}} object")
    pairs: list[tuple[str, str]] = []
    for domain, task_ids in manifest.items():
        for task_id in task_ids:
            pairs.append((str(domain), str(task_id)))
    return pairs


def resolve_arm_dir(root: Path, domains: set[str]) -> tuple[Path, Optional[Path]]:
    """Return (dir holding the domain folders, summary/results.json path or None).

    ``root`` is either the runner's ``--result_dir`` (descend through the unique
    ``<action_space>/<observation_type>/<model>`` chain) or already the ``<model>``
    directory. Ambiguity is an error, never a guess.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"arm root is not a directory: {root}")
    if all((root / domain).is_dir() for domain in domains):
        summary = root.parents[2] / "summary" / "results.json" if len(root.parents) > 2 else None
        return root, summary if summary is not None and summary.exists() else None
    current = root
    for level in ("action_space", "observation_type", "model"):
        candidates = sorted(
            child for child in current.iterdir()
            if child.is_dir() and child.name not in ARM_LAYOUT_SKIP
        )
        if len(candidates) != 1:
            names = ", ".join(child.name for child in candidates) or "<none>"
            raise ValueError(
                f"{root}: cannot resolve a unique <{level}> directory under {current} "
                f"(found: {names}); pass the <model> directory directly"
            )
        current = candidates[0]
    missing = [domain for domain in sorted(domains) if not (current / domain).is_dir()]
    if missing:
        raise FileNotFoundError(f"{current}: no directory for domain(s) {missing}")
    summary = root / "summary" / "results.json"
    return current, summary if summary.exists() else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_error_rows(summary_path: Optional[Path]) -> tuple[dict[str, str], Optional[str]]:
    """task_id -> latest error message from summary/results.json (error rows only),
    AND why that mapping may be incomplete.

    An unreadable summary is not returned as a silent {}: that would make every
    unscored task's reason read as a bare "no result.txt", dropping the harness's
    own account of why. The problem travels with the rows so the report can say the
    file was not read instead of implying it held nothing."""
    if summary_path is None:
        return {}, "summary/results.json not found under the arm root"
    try:
        rows = _read_json(summary_path)
    except (OSError, ValueError) as exc:
        return {}, f"{summary_path} is unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(rows, list):
        return {}, f"{summary_path} is not a JSON list (got {type(rows).__name__})"
    errors: dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("status") == "error":
            errors[str(row.get("task_id"))] = str(row.get("err_message", ""))
    return errors, None


def read_task_result(task_dir: Path, summary_errors: dict[str, str]) -> dict[str, Any]:
    """One arm's record for one task. ``score`` is None when unscored."""
    task_id = task_dir.name
    record: dict[str, Any] = {
        "score": None,
        "unscored_reason": None,
        "partial_scores": None,
        "multi_phase": False,
        "phase_results": None,
        "checkpoints": {},
        "checkpoint_status": {},
    }
    result_txt = task_dir / "result.txt"
    if not result_txt.exists():
        reason = "no result.txt" if task_dir.is_dir() else "no task directory"
        if task_id in summary_errors:
            reason += f"; summary/results.json error row: {summary_errors[task_id]}"
        record["unscored_reason"] = reason
        return record
    text = result_txt.read_text(encoding="utf-8").strip()
    try:
        score = float(text)
    except ValueError:
        record["unscored_reason"] = f"result.txt is not a float: {text!r}"
        return record
    # float() accepts "nan" and "inf" and there is no range in the parse, so without this check an
    # impossible score would enter the comparison. A NaN is the dangerous one: `nan > eps` and
    # `nan < -eps` are both False, so it would count as a TIE and silently enter the denominator of
    # the exact sign test — the comparison's p-value. Out of range is a corrupt or mis-scaled
    # evaluator result, and > 1.0 would also count as a binary completion. Same loud UNSCORED path.
    # The bound carries COMPLETION_EPSILON, the tolerance this module already uses for ties and
    # completions, so the harness's float noise around 1.0 stays a legitimate perfect score.
    if not math.isfinite(score) or not -COMPLETION_EPSILON <= score <= 1.0 + COMPLETION_EPSILON:
        record["unscored_reason"] = (f"result.txt is not a score in [0, 1]: {text!r} — an impossible value is never "
                                     f"a tie and never a completion; treat this task as unscored and investigate")
        return record
    record["score"] = score
    result_json = task_dir / "result.json"
    if result_json.exists():
        payload = _read_json(result_json)
        if isinstance(payload, dict):
            record["partial_scores"] = payload.get("partial_scores")
    phase_file = task_dir / "phase_results.json"
    if phase_file.exists():
        record["multi_phase"] = True
        record["phase_results"] = _read_json(phase_file)
        return record  # the driver runs multi-phase tasks final-only: no checkpoints (NA)
    checkpoint_file = task_dir / "checkpoint_results.json"
    if checkpoint_file.exists():
        for item in _read_json(checkpoint_file) or []:
            if not isinstance(item, dict) or item.get("step") is None:
                continue
            step = int(item["step"])
            status = str(item.get("status"))
            record["checkpoint_status"][step] = status
            record["checkpoints"][step] = (
                float(item["score"]) if status == "success" and item.get("score") is not None else None
            )
    return record


def exact_sign_test(graphpc_wins: int, baseline_wins: int) -> Optional[float]:
    """Two-sided exact sign test p-value over the discordant pairs (None if none)."""
    n = graphpc_wins + baseline_wins
    if n == 0:
        return None
    k = graphpc_wins
    total = 2 ** n
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / total
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / total
    return min(1.0, 2.0 * min(lower, upper))


def _is_complete(score: Optional[float]) -> bool:
    return score is not None and score >= 1.0 - COMPLETION_EPSILON


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def score_paired(
    baseline_root: Path,
    graphpc_root: Path,
    task_list: Path,
) -> dict[str, Any]:
    pairs = load_task_list(task_list)
    domains = {domain for domain, _ in pairs}
    baseline_dir, baseline_summary = resolve_arm_dir(baseline_root, domains)
    graphpc_dir, graphpc_summary = resolve_arm_dir(graphpc_root, domains)
    baseline_errors, baseline_summary_problem = _summary_error_rows(baseline_summary)
    graphpc_errors, graphpc_summary_problem = _summary_error_rows(graphpc_summary)

    rows: list[dict[str, Any]] = []
    unscored: list[dict[str, str]] = []
    for domain, task_id in pairs:
        base = read_task_result(baseline_dir / domain / task_id, baseline_errors)
        graphpc = read_task_result(graphpc_dir / domain / task_id, graphpc_errors)
        row = {
            "domain": domain,
            "task_id": task_id,
            "baseline": base,
            "graphpc": graphpc,
            "delta": (
                graphpc["score"] - base["score"]
                if base["score"] is not None and graphpc["score"] is not None
                else None
            ),
        }
        rows.append(row)
        for arm, record in (("baseline", base), ("graphpc", graphpc)):
            if record["score"] is None:
                unscored.append({"arm": arm, "domain": domain, "task_id": task_id, "reason": record["unscored_reason"]})

    paired = [row for row in rows if row["delta"] is not None]
    # |delta| below the float tolerance is a tie, not a discordant pair.
    graphpc_wins = sum(1 for row in paired if row["delta"] > COMPLETION_EPSILON)
    baseline_wins = sum(1 for row in paired if row["delta"] < -COMPLETION_EPSILON)
    ties = len(paired) - graphpc_wins - baseline_wins
    checkpoint_steps = sorted(
        {step for row in rows for arm in ("baseline", "graphpc") for step in row[arm]["checkpoints"]}
    )

    def _checkpoint_paired_rows(step: int) -> list[dict[str, Any]]:
        # A checkpoint column is paired only where BOTH arms have a successful
        # value at that step, so the two arms' means cover the same tasks (n equal).
        return [
            row for row in paired
            if row["baseline"]["checkpoints"].get(step) is not None
            and row["graphpc"]["checkpoints"].get(step) is not None
        ]

    def arm_summary(arm: str) -> dict[str, Any]:
        scores = [row[arm]["score"] for row in paired]
        all_scored = [row[arm]["score"] for row in rows if row[arm]["score"] is not None]
        per_checkpoint = {}
        for step in checkpoint_steps:
            values = [row[arm]["checkpoints"][step] for row in _checkpoint_paired_rows(step)]
            per_checkpoint[str(step)] = {"n": len(values), "mean_partial_score": _mean(values)}
        return {
            "n_scored": len(all_scored),
            "mean_partial_score_all_scored": _mean(all_scored),
            "mean_partial_score_paired": _mean(scores),
            "binary_completions_paired": sum(1 for score in scores if _is_complete(score)),
            "checkpoints_paired": per_checkpoint,
        }

    return {
        "task_list": str(task_list),
        "multi_phase_tasks": [f"{row['domain']}/{row['task_id']}" for row in rows
                              if row["baseline"]["multi_phase"] or row["graphpc"]["multi_phase"]],
        "arms": {
            # summary_problem is None when the harness's own results.json was read. Anything
            # else means the unscored_reason strings below carry no error-row detail because
            # the file could not be read -- not because the harness recorded none.
            "baseline": {"root": str(baseline_root), "resolved_dir": str(baseline_dir),
                         "summary_problem": baseline_summary_problem},
            "graphpc": {"root": str(graphpc_root), "resolved_dir": str(graphpc_dir),
                         "summary_problem": graphpc_summary_problem},
        },
        "n_listed": len(rows),
        "n_paired": len(paired),
        "summary": {
            "baseline": arm_summary("baseline"),
            "graphpc": arm_summary("graphpc"),
            "mean_delta_paired": _mean([row["delta"] for row in paired]),
            "discordant": {"graphpc_better": graphpc_wins, "baseline_better": baseline_wins, "ties": ties},
            "sign_test_p_two_sided": exact_sign_test(graphpc_wins, baseline_wins),
        },
        "checkpoint_steps": checkpoint_steps,
        "rows": rows,
        "unscored": unscored,
    }


def _fmt(value: Optional[float], width: int = 7, na: str = "NA") -> str:
    return f"{value:{width}.3f}" if value is not None else f"{na:>{width}}"


def render_table(report: dict[str, Any]) -> str:
    steps = report["checkpoint_steps"]
    header = ["task", "base", "graphpc", "delta"]
    for arm in ("base", "gpc"):
        header.extend(f"{arm}@{step}" for step in steps)
    lines = ["  ".join(f"{name:>9}" for name in header)]
    for row in report["rows"]:
        cells = [
            f"{row['domain']}/{row['task_id']}",
            _fmt(row["baseline"]["score"], na="UNSCORED"),
            _fmt(row["graphpc"]["score"], na="UNSCORED"),
            _fmt(row["delta"], na="-"),
        ]
        for arm in ("baseline", "graphpc"):
            cells.extend(_fmt(row[arm]["checkpoints"].get(step)) for step in steps)
        lines.append("  ".join(f"{cell:>9}" for cell in cells))
    summary = report["summary"]
    lines.append("")
    lines.append(f"listed {report['n_listed']}, paired (scored in both arms) {report['n_paired']}")
    for arm in ("baseline", "graphpc"):
        arm_summary = summary[arm]
        lines.append(
            f"{arm:>9}: scored {arm_summary['n_scored']}, "
            f"mean partial score (paired) {_fmt(arm_summary['mean_partial_score_paired'], 6)}, "
            f"binary completions (paired) {arm_summary['binary_completions_paired']}"
        )
        for step, entry in arm_summary["checkpoints_paired"].items():
            lines.append(f"{'':>9}  @{step}: n={entry['n']} mean {_fmt(entry['mean_partial_score'], 6)}")
    discordant = summary["discordant"]
    lines.append(
        f"mean delta (graphpc - baseline) {_fmt(summary['mean_delta_paired'], 6)}; "
        f"graphpc better {discordant['graphpc_better']}, baseline better {discordant['baseline_better']}, "
        f"ties {discordant['ties']}; exact sign test p = {_fmt(summary['sign_test_p_two_sided'], 6)}"
    )
    if steps:
        lines.append(
            "checkpoint columns: NA = no successful inline checkpoint at that step "
            "(multi-phase task, intermediate_eval_safe=False, or checkpoint error); never counted as 0; "
            "the per-arm @step means cover only tasks with a value in BOTH arms (same n)"
        )
    if report["multi_phase_tasks"]:
        lines.append(f"multi-phase (final-only, no checkpoints): {', '.join(report['multi_phase_tasks'])}")
    # This scorer reads result.txt, which cannot distinguish the two ways a task reaches 0.00.
    # Printed unconditionally for the same reason as the block below: a caveat that only appears
    # when something already looks wrong is a caveat nobody reads when it matters.
    lines.append("")
    lines.append("NOTE: a 0.00 the EVALUATOR could not substantiate (a mocked service down at evaluation time) "
                 "is indistinguishable here from a 0.00 the agent earned — result.json records nothing of the "
                 "former. Inspect the final-evaluation window of each 0.00 task's eval.log (and its "
                 "model_eval_file_log/) for evaluator-side failures and adjudicate those before quoting any delta.")
    # Printed whether or not anything is unscored: an unread summary is a fact about the
    # report's completeness, and it must not be discoverable only by opening the JSON.
    unread = [(arm, report["arms"][arm]["summary_problem"]) for arm in ("baseline", "graphpc")
              if report["arms"][arm].get("summary_problem")]
    if unread:
        lines.append("")
        lines.append("SUMMARY NOT READ (the harness's own error rows are unavailable for this arm, so an "
                     "unscored task's reason below carries no error-row detail — that is missing evidence, "
                     "not an absence of errors):")
        for arm, problem in unread:
            lines.append(f"  {arm:>9}: {problem}")
    if report["unscored"]:
        lines.append("")
        lines.append("UNSCORED (listed in the manifest, no result.txt in that arm; not counted as 0):")
        for item in report["unscored"]:
            lines.append(f"  {item['arm']:>9} {item['domain']}/{item['task_id']}: {item['reason']}")
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--graphpc_root", type=Path, required=True)
    parser.add_argument("--task_list", type=Path, required=True, help='{"tasks": [ids]} manifest given to both runners')
    parser.add_argument("--output_json", type=Path, default=None, help="Write the full report (rows + summary) here")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Exit 0 even when a listed task is unscored in an arm (it is still reported, never counted as 0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    report = score_paired(args.baseline_root, args.graphpc_root, args.task_list)
    print(render_table(report))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport written to {args.output_json}")
    if report["unscored"] and not args.allow_missing:
        print(
            f"\nERROR: {len(report['unscored'])} listed task(s) unscored; pass --allow_missing to accept a partial comparison.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
