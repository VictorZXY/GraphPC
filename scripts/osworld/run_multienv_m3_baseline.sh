#!/usr/bin/env bash
# Baseline arm launcher for OSWorld 2.0 — a copy of OSWorld-V2 scripts/bash/run_multienv_m3.sh
# (tag v2026.08.08). The agent is upstream's, unmodified (mm_agents/m3). The runner is a
# VERBATIM vendored copy of upstream's scripts/python/run_multienv_m3.py with exactly ONE
# delta — scripts/python/run_multienv_m3_baseline.py, from
# scripts/osworld/run_multienv_m3_baseline.py. The delta adds --volume_size, ported verbatim
# from upstream's own sibling runner run_multienv_gpt_response_api.py at the same tag
# (argparse :206-211, per-task lookup :429-436); a harness setting applied identically in
# both arms; see that file's header. WHY: the V2 client AMI ships a ~29.5 GB
# root partition on a 40 GB volume with neither growpart nor cloud-init, so the root
# filesystem starts near 100% full, snapd's overdue auto-refresh fills it, and the guest
# server's /screenshot handler dies writing its temp PNG (errno 28) ~5 min 35 s after
# env.reset on a fully idle VM, in both arms alike. A non-None volume_size is what
# makes DesktopEnv run its guest-side expansion (growpart + resize2fs) -- on EVERY task, at
# the price of an in-guest apt-get update + cloud-guest-utils install per episode, whose
# failure fails env construction for that task. --volume_size is a HARNESS flag: it comes
# from osworld_cell_config.sh, identical in both arms.
# This script only fixes the invocation so the two arms are paired:
#   * OSWORLD_CELL=minimax, set here and never read from the environment: the launch guard holds
#     this launch to the MiniMax-M3 cell's scored paths;
#   * no inline `export` block: host settings come from <checkout>/.env via the runner's
#     load_dotenv() (exporting upstream's placeholders here would shadow the .env);
#   * every harness flag comes from osworld_cell_config.sh (shared with the GraphPC arm);
#     this script adds only the M3Agent flags;
#   * MiniMax is pinned explicitly — the pinned tag has no MiniMax defaults:
#     --base_url https://api.minimax.io/anthropic --model MiniMax-M3 --thinking_mode adaptive,
#     and --api_key is the MiniMax key (the runner's default is ANTHROPIC_API_KEY, which on
#     the host is the evaluator's Anthropic key, so it must be passed explicitly).
# The MiniMax key is passed ONLY through --api_key; it is never exported as ANTHROPIC_API_KEY
# (the Sonnet evaluator and the user simulator in the same process read that name). A misrouted
# key does not error: every call 401s, the agent returns no actions, and the task SCORES 0.0
# -- so the one-task smoke must show step 1 of traj.jsonl is a pyautogui action.
# Resume hazard (upstream, unfixed at the tag): get_unfinished os.remove()s every entry of an
# unfinished task dir and crashes on the api_logs/ and model_eval_file_log*/ DIRECTORIES the
# previous attempt left there. This script removes exactly those directories from task dirs
# that have no result.txt before launching (logged), so a relaunch resumes cleanly.
# Run from the OSWorld-V2 checkout root on the host: bash scripts/bash/run_multienv_m3_baseline.sh
# DRY_RUN=1 prints the fully resolved command (the key shown as <MINIMAX_API_KEY>) and exits 0
# without running anything and without the resume pre-clean.
# A SCORED LAUNCH APPENDS NOTHING. A one-task smoke of this arm must declare itself and use a
# throwaway result tree, which is what keeps the unscoped pre-clean below safe:
#   OSWORLD_UNSCORED_LAUNCH=1 RESULT_DIR=./results/osworld_v2/smoke_baseline \
#     bash scripts/bash/run_multienv_m3_baseline.sh --specific_task_id 004 --max_steps 15 --checkpoint_steps 5,10
set -euo pipefail

OSWORLD_CELL=minimax
# shellcheck source=osworld_cell_config.sh
source "$(dirname "${BASH_SOURCE[0]}")/osworld_cell_config.sh"

MODEL="${MODEL:-MiniMax-M3}"
BASE_URL="${BASE_URL:-https://api.minimax.io/anthropic}"
THINKING_MODE="${THINKING_MODE:-adaptive}"
RESULT_DIR="${RESULT_DIR:-./results/osworld_v2/baseline_minimax_m3}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
# No embedder on this arm; workers mostly wait on the VM / API. Default 4 = one per vCPU on a
# 4 vCPU run host (the vCPU quota permits far more; the host's CPU, shared with the GraphPC
# arm's 4 workers, is the practical bound).
NUM_ENVS="${NUM_ENVS:-4}"

# Appended flags the PREFLIGHT CANNOT SEE, and the scored result tree an unscored launch must not
# touch: both are refused by the shared guard in osworld_cell_config.sh (which carries the full
# argument). This is the arm where the hazard bites hardest — the pre-clean below and upstream's
# get_unfinished are both scoped by RESULT_DIR and by nothing else, and get_unfinished ignores
# --specific_task_id / --domain here. Placed BEFORE the pre-clean, so a refused launch removes
# nothing. A scored run appends nothing, so the guard returns at once and the argv is unchanged.
osworld_cell_guard_launch_args baseline --result_dir "${RESULT_DIR}" -- "$@"

osworld_cell_require_task_list

# Remove the directories upstream's get_unfinished cannot handle from unfinished task dirs.
#
# DELIBERATELY UNSCOPED IN THE TASK DIMENSION, and it must stay that way. It exists to stop
# upstream's get_unfinished raising IsADirectoryError, and that function walks EVERY task dir under
# its target regardless of --specific_task_id / --domain (the GraphPC arm's runner carries a
# scoping delta, this one is upstream verbatim). Narrowing this loop to the task
# list would leave exactly the dirs the wipe still visits uncleaned, reintroducing the crash it
# prevents. Narrowing it also protects nothing: what this removes from an unfinished task dir is a
# STRICT SUBSET of what get_unfinished is about to os.remove from that same dir anyway. The
# destructive step is the runner's wipe, not this.
#
# IT REACHES ONE LEVEL FURTHER THAN THAT WIPE IN THE MODEL DIMENSION. get_unfinished builds
# result_dir/action_space/observation_type/USE_MODEL and walks only that subtree; the glob below is
# ${model_root}/*/*/*/ = model/domain/task, so its first * iterates every MODEL directory in the
# result tree. Consequence, bounded: an unfinished task dir belonging to a DIFFERENT model in the
# same tree can lose its api_logs/ and model_eval_file_log*/ — no result.txt is touched and no
# scored data exists in an unfinished dir, so nothing scored is at risk. It is left unscoped
# rather than scoped here because the model DIRECTORY NAME is
# not ${MODEL}: the runner writes _model_for_paths(args), i.e. the model with [^A-Za-z0-9._-]+
# collapsed to _, so scoping here would mean a second implementation of a runner-side path rule
# inside bash — the kind of duplicate that drifts, and MiniMax-M3 being a fixed point of that
# sanitiser is exactly what would make the duplicate look correct until it stopped being.
# verify_runner_deltas.py asserts the fact this rests on (both arms' get_unfinished stay inside
# their own model directory), so the asymmetry is checked rather than asserted.
#
# KNOWN RESIDUAL, ACCEPTED, and this is the same fact from the operator's side:
# `MODEL=Other bash scripts/bash/run_multienv_m3_baseline.sh` with the default RESULT_DIR appends
# NOTHING, so osworld_cell_guard_launch_args has nothing to refuse — and this loop then reaches the
# scored MiniMax-M3 subtree, removing api_logs/ and model_eval_file_log*/ from its unfinished task
# dirs. No result.txt is touched and an unfinished dir holds no scored data, so nothing scored is
# lost; the rule that would close it ("refuse when MODEL differs while RESULT_DIR is scored") is a
# third launcher rule for a hazard that cannot cost a score. If you are changing MODEL, change
# RESULT_DIR with it.
#
# What keeps this safe is the containment osworld_cell_guard_launch_args ENFORCES above, at the
# point of launch:
# the arms default to distinct result trees, and any launch that appends an argument must declare
# itself unscored and must not name either arm's scored tree.
# scripts/osworld/verify_runner_deltas.py also checks the two static facts (the defaults are
# distinct, the pre-clean is CALLED), but a static check runs before launch and cannot see what an
# operator types at it.
osworld_baseline_resume_preclean() {
    local model_root="${RESULT_DIR}/${ACTION_SPACE}/${OBSERVATION_TYPE}"
    local task_dir sub removed=0
    [[ -d "${model_root}" ]] || { echo "resume pre-clean: no result tree at ${model_root}; nothing to do"; return 0; }
    for task_dir in "${model_root}"/*/*/*/; do
        [[ -d "${task_dir}" ]] || continue
        [[ -f "${task_dir}result.txt" ]] && continue
        for sub in "${task_dir}api_logs" "${task_dir}"model_eval_file_log*; do
            [[ -d "${sub}" ]] || continue
            echo "resume pre-clean: removing ${sub} (unfinished task dir; upstream get_unfinished cannot remove directories)"
            rm -rf "${sub}"
            removed=$((removed + 1))
        done
    done
    echo "resume pre-clean: removed ${removed} director$([[ ${removed} -eq 1 ]] && echo y || echo ies) under ${model_root}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
    MINIMAX_API_KEY_ARG="<MINIMAX_API_KEY>"
    echo "DRY_RUN: resume pre-clean skipped (would scan ${RESULT_DIR}/${ACTION_SPACE}/${OBSERVATION_TYPE})"
else
    # The MiniMax key: from the environment, else from the same .env the runner reads.
    if [[ -z "${MINIMAX_API_KEY:-}" ]]; then
        MINIMAX_API_KEY="$(uv run --no-sync python -c 'from dotenv import dotenv_values; print(dotenv_values(".env").get("MINIMAX_API_KEY") or "")')"
    fi
    if [[ -z "${MINIMAX_API_KEY}" ]]; then
        echo "ERROR: MINIMAX_API_KEY is not set and not present in .env" >&2
        exit 1
    fi
    MINIMAX_API_KEY_ARG="${MINIMAX_API_KEY}"
    osworld_baseline_resume_preclean
fi

osworld_cell_harness_args

osworld_cell_launch uv run --no-sync python scripts/python/run_multienv_m3_baseline.py \
    "${OSWORLD_CELL_HARNESS_ARGS[@]}" \
    --base_url "${BASE_URL}" \
    --api_key "${MINIMAX_API_KEY_ARG}" \
    --thinking_mode "${THINKING_MODE}" \
    --model "${MODEL}" \
    --max_tokens "${MAX_TOKENS}" \
    --result_dir "${RESULT_DIR}" \
    --num_envs "${NUM_ENVS}" \
    "$@"
