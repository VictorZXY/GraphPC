from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OSWORLD_ROOT = REPO_ROOT.parent / "OSWorld-V2"

# Everything the sync writes into the OSWorld-V2 checkout; --clean removes exactly these.
SYNCED_TARGETS = (
    Path("mm_agents/graphpc"),
    Path("scripts/python/run_multienv_graphpc.py"),
    Path("scripts/python/run_multienv_m3_baseline.py"),
    Path("scripts/python/verify_runner_deltas.py"),
    Path("scripts/bash/run_multienv_graphpc.sh"),
    Path("scripts/bash/run_multienv_m3_baseline.sh"),
    Path("scripts/python/verify_minimax_vision.py"),
    Path("scripts/python/verify_qwen_vision.py"),
    Path("scripts/python/verify_deepseek_vision.py"),
    Path("scripts/bash/osworld_cell_config.sh"),
    Path("scripts/python/launch_preflight.py"),
    Path("scripts/python/solve_subset_run_order.py"),
    Path("mm_agents/qwen"),
    Path("scripts/python/run_multienv_qwen_baseline.py"),
    Path("scripts/bash/run_multienv_qwen_baseline.sh"),
    Path("mm_agents/deepseek"),
    Path("scripts/python/run_multienv_deepseek_baseline.py"),
    Path("scripts/bash/run_multienv_deepseek_baseline.sh"),
)

# The vendored snapshot: the transitive import closure of the OSWorld adapter.
# The backbone package __init__ re-exports nothing, so only the backbones the adapter's profiles
# build (MiniMax, Qwen, DeepSeek) are here.
SYNC_SOURCE_TARGETS = (
    ("src/agents/__init__.py", "mm_agents/graphpc/agents/__init__.py"),
    ("src/agents/action/__init__.py", "mm_agents/graphpc/agents/action/__init__.py"),
    ("src/agents/action/base.py", "mm_agents/graphpc/agents/action/base.py"),
    ("src/agents/backbone/__init__.py", "mm_agents/graphpc/agents/backbone/__init__.py"),
    ("src/agents/backbone/base.py", "mm_agents/graphpc/agents/backbone/base.py"),
    ("src/agents/backbone/deepseek/__init__.py", "mm_agents/graphpc/agents/backbone/deepseek/__init__.py"),
    ("src/agents/backbone/deepseek/main.py", "mm_agents/graphpc/agents/backbone/deepseek/main.py"),
    ("src/agents/backbone/minimax/__init__.py", "mm_agents/graphpc/agents/backbone/minimax/__init__.py"),
    ("src/agents/backbone/minimax/main.py", "mm_agents/graphpc/agents/backbone/minimax/main.py"),
    ("src/agents/backbone/qwen/__init__.py", "mm_agents/graphpc/agents/backbone/qwen/__init__.py"),
    ("src/agents/backbone/qwen/main.py", "mm_agents/graphpc/agents/backbone/qwen/main.py"),
    ("src/agents/common/__init__.py", "mm_agents/graphpc/agents/common/__init__.py"),
    ("src/agents/common/compaction.py", "mm_agents/graphpc/agents/common/compaction.py"),
    ("src/agents/common/effect_class.py", "mm_agents/graphpc/agents/common/effect_class.py"),
    ("src/agents/common/infeasible_declaration.py", "mm_agents/graphpc/agents/common/infeasible_declaration.py"),
    ("src/agents/common/json_utils.py", "mm_agents/graphpc/agents/common/json_utils.py"),
    ("src/agents/common/resample.py", "mm_agents/graphpc/agents/common/resample.py"),
    ("src/agents/common/retry_policy.py", "mm_agents/graphpc/agents/common/retry_policy.py"),
    ("src/agents/common/schema.py", "mm_agents/graphpc/agents/common/schema.py"),
    ("src/agents/common/schemas_json.py", "mm_agents/graphpc/agents/common/schemas_json.py"),
    ("src/agents/common/stop_answer.py", "mm_agents/graphpc/agents/common/stop_answer.py"),
    ("src/agents/graphpc/__init__.py", "mm_agents/graphpc/agents/graphpc/__init__.py"),
    ("src/agents/graphpc/orchestrator.py", "mm_agents/graphpc/agents/graphpc/orchestrator.py"),
    ("src/agents/monitor/__init__.py", "mm_agents/graphpc/agents/monitor/__init__.py"),
    ("src/agents/monitor/base.py", "mm_agents/graphpc/agents/monitor/base.py"),
    ("src/agents/monitor/heuristic.py", "mm_agents/graphpc/agents/monitor/heuristic.py"),
    ("src/agents/reasoning/__init__.py", "mm_agents/graphpc/agents/reasoning/__init__.py"),
    ("src/agents/reasoning/base.py", "mm_agents/graphpc/agents/reasoning/base.py"),
    ("src/agents/reasoning/graph_reasoning.py", "mm_agents/graphpc/agents/reasoning/graph_reasoning.py"),
    ("src/agents/retrieval/__init__.py", "mm_agents/graphpc/agents/retrieval/__init__.py"),
    ("src/agents/retrieval/base.py", "mm_agents/graphpc/agents/retrieval/base.py"),
    ("src/agents/retrieval/bypass.py", "mm_agents/graphpc/agents/retrieval/bypass.py"),
    ("src/agents/retrieval/deterministic.py", "mm_agents/graphpc/agents/retrieval/deterministic.py"),
    ("src/agents/retrieval/embedding.py", "mm_agents/graphpc/agents/retrieval/embedding.py"),
    ("src/agents/retrieval/relevance.py", "mm_agents/graphpc/agents/retrieval/relevance.py"),
    ("src/agents/retrieval/support_set.py", "mm_agents/graphpc/agents/retrieval/support_set.py"),
    ("src/benchmarks/osworld/__init__.py", "mm_agents/graphpc/benchmarks/osworld/__init__.py"),
    ("src/benchmarks/osworld/action_deepseek.py", "mm_agents/graphpc/benchmarks/osworld/action_deepseek.py"),
    ("src/benchmarks/osworld/action_minimax.py", "mm_agents/graphpc/benchmarks/osworld/action_minimax.py"),
    ("src/benchmarks/osworld/action_qwen.py", "mm_agents/graphpc/benchmarks/osworld/action_qwen.py"),
    ("src/benchmarks/osworld/backbone_deepseek.py", "mm_agents/graphpc/benchmarks/osworld/backbone_deepseek.py"),
    ("src/benchmarks/osworld/backbone_qwen.py", "mm_agents/graphpc/benchmarks/osworld/backbone_qwen.py"),
    ("src/benchmarks/osworld/graphpc_agent.py", "mm_agents/graphpc/benchmarks/osworld/graphpc_agent.py"),
    ("src/benchmarks/osworld/m3_action_translator.py", "mm_agents/graphpc/benchmarks/osworld/m3_action_translator.py"),
    ("src/benchmarks/osworld/observation_adapter.py", "mm_agents/graphpc/benchmarks/osworld/observation_adapter.py"),
    ("src/benchmarks/osworld/orchestrator.py", "mm_agents/graphpc/benchmarks/osworld/orchestrator.py"),
    ("src/benchmarks/osworld/profile_deepseek.py", "mm_agents/graphpc/benchmarks/osworld/profile_deepseek.py"),
    ("src/benchmarks/osworld/profile_qwen.py", "mm_agents/graphpc/benchmarks/osworld/profile_qwen.py"),
    ("src/benchmarks/osworld/prompts.py", "mm_agents/graphpc/benchmarks/osworld/prompts.py"),
    # The Qwen control's own code the Qwen profile calls (headers name each upstream blob).
    *((f"src/benchmarks/osworld/qwen_upstream/{name}.py", f"mm_agents/graphpc/benchmarks/osworld/qwen_upstream/{name}.py")
      for name in ("__init__", "actions", "history", "images", "parser", "prompts", "qwen_vl_utils", "recovery")),
    ("src/benchmarks/osworld/reasoning.py", "mm_agents/graphpc/benchmarks/osworld/reasoning.py"),
    ("src/benchmarks/osworld/timeline.py", "mm_agents/graphpc/benchmarks/osworld/timeline.py"),
    # Runners + launch scripts land next to OSWorld-V2's own (scripts/python/_repo_path.py
    # resolves the checkout root for both vendored runners). The baseline runner is a copy of
    # upstream's run_multienv_m3.py with one delta: --volume_size, ported verbatim from
    # upstream's own run_multienv_gpt_response_api.py (argparse :206-211, per-task lookup
    # :429-436), carried identically by both arms (see each runner's header); its agent is
    # upstream's mm_agents/m3, unmodified.
    ("scripts/osworld/run_multienv_graphpc.py", "scripts/python/run_multienv_graphpc.py"),
    ("scripts/osworld/run_multienv_m3_baseline.py", "scripts/python/run_multienv_m3_baseline.py"),
    ("scripts/osworld/osworld_cell_config.sh", "scripts/bash/osworld_cell_config.sh"),
    ("scripts/osworld/run_multienv_graphpc.sh", "scripts/bash/run_multienv_graphpc.sh"),
    ("scripts/osworld/run_multienv_m3_baseline.sh", "scripts/bash/run_multienv_m3_baseline.sh"),
    # Standalone loud-fail gates run on the host before provisioning (uv run ...).
    ("scripts/osworld/verify_minimax_vision.py", "scripts/python/verify_minimax_vision.py"),
    ("scripts/osworld/verify_qwen_vision.py", "scripts/python/verify_qwen_vision.py"),
    ("scripts/osworld/verify_deepseek_vision.py", "scripts/python/verify_deepseek_vision.py"),
    ("scripts/osworld/verify_runner_deltas.py", "scripts/python/verify_runner_deltas.py"),
    ("scripts/osworld/launch_preflight.py", "scripts/python/launch_preflight.py"),
    # Imported by launch_preflight for run_order_gaps(), so the declared run order's twin-pair
    # separation is COMPUTED on the host rather than read back out of the pre-registration. It is
    # also the instrument that derives the order in the first place; without it on the host, the
    # one check that catches a corrupted run_order would not exist on the machine that runs the
    # preflight.
    ("scripts/osworld/solve_subset_run_order.py", "scripts/python/solve_subset_run_order.py"),
    # The Qwen cell's CONTROL: upstream's osworld-v2.1 Qwen agent and runner, which the pinned
    # v2026.08.08 tag lacks, vendored with the deltas their headers declare. The agent keeps its
    # upstream import path mm_agents.qwen; none of these files holds an import the rewrite
    # below touches, so they land byte for byte.
    ("scripts/osworld/qwen_control/mm_agents/qwen/README.md", "mm_agents/qwen/README.md"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/__init__.py", "mm_agents/qwen/__init__.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/actions.py", "mm_agents/qwen/actions.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/client.py", "mm_agents/qwen/client.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/context.py", "mm_agents/qwen/context.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/history.py", "mm_agents/qwen/history.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/images.py", "mm_agents/qwen/images.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/main.py", "mm_agents/qwen/main.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/parser.py", "mm_agents/qwen/parser.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/prompts.py", "mm_agents/qwen/prompts.py"),
    ("scripts/osworld/qwen_control/mm_agents/qwen/recovery.py", "mm_agents/qwen/recovery.py"),
    ("scripts/osworld/run_multienv_qwen_baseline.py", "scripts/python/run_multienv_qwen_baseline.py"),
    ("scripts/osworld/run_multienv_qwen_baseline.sh", "scripts/bash/run_multienv_qwen_baseline.sh"),
    # The DeepSeek cell's CONTROL: upstream's v2026.08.08 M3 agent vendored as mm_agents/deepseek with the
    # marked deltas its files declare (the computer tool declared natively, the action read from the tool_use
    # block, the tool protocol in the history), its own runner (upstream's run_multienv_m3.py plus the
    # --volume_size port and the control-agent import) and launcher. None of these files holds an import the
    # rewrite below touches, so they land byte for byte; the MiniMax control's run_multienv_m3_baseline.py and
    # mm_agents/m3 are not involved.
    ("scripts/osworld/deepseek_control/mm_agents/deepseek/README.md", "mm_agents/deepseek/README.md"),
    ("scripts/osworld/deepseek_control/mm_agents/deepseek/__init__.py", "mm_agents/deepseek/__init__.py"),
    ("scripts/osworld/deepseek_control/mm_agents/deepseek/agent.py", "mm_agents/deepseek/agent.py"),
    ("scripts/osworld/deepseek_control/mm_agents/deepseek/parser.py", "mm_agents/deepseek/parser.py"),
    ("scripts/osworld/deepseek_control/mm_agents/deepseek/prompts.py", "mm_agents/deepseek/prompts.py"),
    ("scripts/osworld/run_multienv_deepseek_baseline.py", "scripts/python/run_multienv_deepseek_baseline.py"),
    ("scripts/osworld/run_multienv_deepseek_baseline.sh", "scripts/bash/run_multienv_deepseek_baseline.sh"),
)

GENERATED_TARGETS = {
    "mm_agents/graphpc/__init__.py": 'from .benchmarks.osworld.graphpc_agent import OSWorldGraphPCAgent\n\n__all__ = ["OSWorldGraphPCAgent"]\n',
    "mm_agents/graphpc/benchmarks/__init__.py": "",
}

# Written by every sync: the list of every path the sync wrote (derived from the tuples
# above) plus the source provenance. The host-side preflight checks the snapshot against
# it, so a new sync entry is covered without editing any hand-written list.
VENDORED_MANIFEST_TARGET = "mm_agents/graphpc/vendored_manifest.json"


def vendored_target_paths() -> list[str]:
    """Every checkout-relative path the sync writes: generated package files, each
    SYNC_SOURCE_TARGETS target, and the manifest itself."""
    paths = set(GENERATED_TARGETS) | {target for _, target in SYNC_SOURCE_TARGETS} | {VENDORED_MANIFEST_TARGET}
    return sorted(paths)


def missing_exclude_entries(osworld_root: str | Path) -> list[str]:
    """Vendored paths NOT covered by the checkout's ``.git/info/exclude``, in the
    form the file wants them.

    READ ONLY — this never writes into the checkout. The snapshot is untracked by
    design and the exclude file is what keeps it out of `git status`; the launch
    preflight FAILs on any untracked path it does not cover, so a sync that adds a
    file without a matching exclude line produces a red gate at launch, which is
    the moment nobody wants to be editing git plumbing. The list is hand-maintained
    on the host, so the sync reports the drift rather than leaving the operator to
    remember. A path counts as covered when it, or any ancestor directory, is
    listed."""
    exclude_path = Path(osworld_root) / ".git" / "info" / "exclude"
    patterns: set[str] = set()
    if exclude_path.is_file():
        for line in exclude_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                patterns.add(entry.strip("/"))
    missing = []
    for target in vendored_target_paths():
        segments = target.split("/")
        if not any("/".join(segments[:count]) in patterns for count in range(1, len(segments) + 1)):
            missing.append(f"/{target}")
    return missing


def _source_provenance() -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/osworld"], cwd=str(REPO_ROOT),
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "synced_sources_dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "synced_sources_dirty": None}


def write_vendored_manifest(destination_root: Path) -> Path:
    manifest_path = destination_root / VENDORED_MANIFEST_TARGET
    payload = {
        "targets": vendored_target_paths(),
        "synced_targets": [str(path) for path in SYNCED_TARGETS],
        "source": _source_provenance(),
        "synced_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def sync_to_osworld(
    osworld_root: str | Path = DEFAULT_OSWORLD_ROOT,
    *,
    clean: bool = False,
    clean_only: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    destination_root = Path(osworld_root).resolve()
    _validate_osworld_root(destination_root)

    touched_paths: list[Path] = []
    if clean or clean_only:
        touched_paths.extend(remove_synced_targets(destination_root, dry_run=dry_run))

    if clean_only:
        return touched_paths

    for relative_target, content in GENERATED_TARGETS.items():
        target_path = destination_root / relative_target
        touched_paths.append(target_path)
        if dry_run:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")

    for source_relative, target_relative in SYNC_SOURCE_TARGETS:
        source_path = REPO_ROOT / source_relative
        target_path = destination_root / target_relative
        touched_paths.append(target_path)
        if dry_run:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".py":
            target_path.write_text(_rewrite_python_imports(source_path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(source_path, target_path)

    manifest_path = destination_root / VENDORED_MANIFEST_TARGET
    touched_paths.append(manifest_path)
    if not dry_run:
        write_vendored_manifest(destination_root)

    return touched_paths


def remove_synced_targets(osworld_root: str | Path, *, dry_run: bool = False) -> list[Path]:
    destination_root = Path(osworld_root).resolve()
    removed_paths: list[Path] = []

    for relative_target in SYNCED_TARGETS:
        target_path = destination_root / relative_target
        if not target_path.exists():
            continue
        removed_paths.append(target_path)
        if dry_run:
            continue
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    return removed_paths


def _rewrite_python_imports(source_text: str) -> str:
    rewritten = source_text.replace("from src.", "from mm_agents.graphpc.")
    rewritten = rewritten.replace("import src.", "import mm_agents.graphpc.")
    return rewritten


def _validate_osworld_root(osworld_root: Path) -> None:
    # Both markers exist only in OSWorld-V2 (a 1.0 checkout has neither).
    expected_markers = (
        osworld_root / "task_loader.py",
        osworld_root / "scripts" / "python" / "run_multienv_m3.py",
    )
    missing = [str(path) for path in expected_markers if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Expected an OSWorld-V2 checkout; missing marker(s): " + ", ".join(missing)
        )
    # Every synced path must be ours, i.e. untracked in the checkout: --clean removes them
    # wholesale, and a later release tracks at least one of them (osworld-v2.1 ships
    # mm_agents/qwen/). A root that is not a git checkout tracks nothing.
    listed = subprocess.run(
        ["git", "-C", str(osworld_root), "ls-files", "--", *(str(path) for path in SYNCED_TARGETS)],
        capture_output=True, text=True,
    )
    tracked = listed.stdout.split() if listed.returncode == 0 else []
    if tracked:
        raise FileExistsError(
            "The checkout TRACKS path(s) the sync writes and --clean removes: " + ", ".join(tracked)
            + " — this is not the pinned tag the vendored files were written for."
        )
