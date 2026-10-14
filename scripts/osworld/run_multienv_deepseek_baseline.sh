#!/usr/bin/env bash
# Control arm launcher for the OSWorld 2.0 DeepSeek cell: the sibling of run_multienv_m3_baseline.sh (itself
# a copy of OSWorld-V2 scripts/bash/run_multienv_m3.sh at tag v2026.08.08), differing from it only where the
# model differs. The agent is the vendored DeepSeek control, mm_agents/deepseek: upstream's mm_agents/m3 plus
# the marked deltas its files declare (the computer tool declared natively, the action read from the tool_use
# block, the tool protocol in the history), because DeepSeek-V4.1-Flash does not
# write M3's <tool_call> text grammar (its replies mostly fail upstream M3's parser).
# The runner is scripts/python/run_multienv_deepseek_baseline.py (from
# scripts/osworld/run_multienv_deepseek_baseline.py): upstream's run_multienv_m3.py with two marked deltas,
# the --volume_size port both arms carry and the control-agent import (see its header); the MiniMax cell's
# run_multienv_m3_baseline.py is not touched.
# This script only fixes the invocation so the two arms are paired:
#   * OSWORLD_CELL=deepseek, set here and never read from the environment: the launch guard holds this launch
#     to the DeepSeek cell's scored paths;
#   * no inline `export` block: host settings come from <checkout>/.env via the runner's load_dotenv()
#     (exporting upstream's placeholders here would shadow the .env);
#   * every harness flag comes from osworld_cell_config.sh (shared with the GraphPC arm); this script adds
#     only the control agent's flags;
#   * the model is pinned explicitly on DeepSeek's Anthropic-format endpoint:
#     --base_url https://api.deepseek.com/anthropic --model deepseek-flash, and --api_key is the DeepSeek key
#     (the runner's default is ANTHROPIC_API_KEY, which on the host is the evaluator's Anthropic key, so it
#     must be passed explicitly);
#   * THINKING ON, in one explicit form: --thinking_budget alone, which the
#     agent sends as thinking {"type": "enabled", "budget_tokens": N}; the endpoint accepts the form and ignores
#     the budget. NO --thinking_mode is passed. The runner's --thinking_mode default reads M3_THINKING_MODE from
#     the environment, the checkout's .env included (load_dotenv() runs before argparse), so that variable MUST
#     NOT be set on the host for this cell: `adaptive` would move the request form and `disabled` would turn
#     thinking off, with no change here. Holding that on the host is the launch preflight's (its pinned
#     control flags), not this script's;
#   * --max_tokens 65536: thinking and the reply share the cap, which is kept at one value for the
#     cell's regime table;
# The DeepSeek key is passed ONLY through --api_key; it is never exported as ANTHROPIC_API_KEY (the Sonnet
# evaluator and the user simulator in the same process read that name). A misrouted key does not error:
# every call 401s, the agent returns no actions, and the task SCORES 0.0 -- so the one-task smoke must show
# step 1 of traj.jsonl is a pyautogui action and api_logs/001_*.json carries response_body.model deepseek-flash
# with a tool_use block.
# Resume hazard (upstream, unfixed at the tag): get_unfinished os.remove()s every entry of an
# unfinished task dir and crashes on the api_logs/ and model_eval_file_log*/ DIRECTORIES the
# previous attempt left there. This script removes exactly those directories from task dirs
# that have no result.txt before launching (logged), so a relaunch resumes cleanly.
# Run from the OSWorld-V2 checkout root on the host: bash scripts/bash/run_multienv_deepseek_baseline.sh
# DRY_RUN=1 prints the fully resolved command (the key shown as <DEEPSEEK_API_KEY>) and exits 0
# without running anything and without the resume pre-clean.
# A SCORED LAUNCH APPENDS NOTHING. A one-task smoke of this arm must declare itself and use a
# throwaway result tree, which is what keeps the unscoped pre-clean below safe:
#   OSWORLD_UNSCORED_LAUNCH=1 RESULT_DIR=./results/osworld_v2/smoke_baseline_deepseek \
#     bash scripts/bash/run_multienv_deepseek_baseline.sh --specific_task_id 004 --max_steps 15 --checkpoint_steps 5,10
set -euo pipefail

OSWORLD_CELL=deepseek
# shellcheck source=osworld_cell_config.sh
source "$(dirname "${BASH_SOURCE[0]}")/osworld_cell_config.sh"

MODEL="${MODEL:-deepseek-flash}"
BASE_URL="${BASE_URL:-https://api.deepseek.com/anthropic}"
# Thinking on: the explicit `enabled` form (the budget is ignored by the endpoint); see the header.
THINKING_BUDGET="${THINKING_BUDGET:-4096}"
RESULT_DIR="${RESULT_DIR:-./results/osworld_v2/baseline_deepseek_flash}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
# No embedder on this arm; workers mostly wait on the VM / API. Default 4, as on the other controls.
NUM_ENVS="${NUM_ENVS:-4}"

# Appended flags the PREFLIGHT CANNOT SEE, and the scored result tree an unscored launch must not
# touch: both are refused by the shared guard in osworld_cell_config.sh (which carries the full
# argument). Placed BEFORE the pre-clean, so a refused launch removes nothing. A scored run appends
# nothing, so the guard returns at once and the argv is unchanged.
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
# inside bash — the kind of duplicate that drifts, and deepseek-flash being a fixed point of that
# sanitiser is exactly what would make the duplicate look correct until it stopped being.
# This runner's get_unfinished is upstream's verbatim — verify_runner_deltas.py --cell deepseek
# (DS2) holds the runner to upstream's file, and the default mode asserts upstream's wipe stays
# inside its own model directory — so the asymmetry is checked rather than asserted.
#
# KNOWN RESIDUAL, ACCEPTED, and this is the same fact from the operator's side:
# `MODEL=Other bash scripts/bash/run_multienv_deepseek_baseline.sh` with the default RESULT_DIR appends
# NOTHING, so osworld_cell_guard_launch_args has nothing to refuse — and this loop then reaches the
# scored deepseek-flash subtree, removing api_logs/ and model_eval_file_log*/ from its unfinished task
# dirs. No result.txt is touched and an unfinished dir holds no scored data, so nothing scored is
# lost; the rule that would close it ("refuse when MODEL differs while RESULT_DIR is scored") is a
# third launcher rule for a hazard that cannot cost a score. If you are changing MODEL, change
# RESULT_DIR with it.
#
# What keeps this safe is the containment osworld_cell_guard_launch_args ENFORCES above, at the
# point of launch:
# the arms default to distinct result trees, and any launch that appends an argument must declare
# itself unscored and must not name either arm's scored tree.
# scripts/osworld/verify_runner_deltas.py checks those two static facts (distinct defaults, the
# pre-clean CALLED) for run_multienv_m3_baseline.sh, whose pre-clean this copies; for this launcher
# nothing checks them statically, and the guard above is what holds the containment.
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
    DEEPSEEK_API_KEY_ARG="<DEEPSEEK_API_KEY>"
    echo "DRY_RUN: resume pre-clean skipped (would scan ${RESULT_DIR}/${ACTION_SPACE}/${OBSERVATION_TYPE})"
else
    # The DeepSeek key: from the environment, else from the same .env the runner reads.
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
        DEEPSEEK_API_KEY="$(uv run --no-sync python -c 'from dotenv import dotenv_values; print(dotenv_values(".env").get("DEEPSEEK_API_KEY") or "")')"
    fi
    if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
        echo "ERROR: DEEPSEEK_API_KEY is not set and not present in .env" >&2
        exit 1
    fi
    DEEPSEEK_API_KEY_ARG="${DEEPSEEK_API_KEY}"
    osworld_baseline_resume_preclean
fi

osworld_cell_harness_args

osworld_cell_launch uv run --no-sync python scripts/python/run_multienv_deepseek_baseline.py \
    "${OSWORLD_CELL_HARNESS_ARGS[@]}" \
    --base_url "${BASE_URL}" \
    --api_key "${DEEPSEEK_API_KEY_ARG}" \
    --thinking_budget "${THINKING_BUDGET}" \
    --model "${MODEL}" \
    --max_tokens "${MAX_TOKENS}" \
    --result_dir "${RESULT_DIR}" \
    --num_envs "${NUM_ENVS}" \
    "$@"
