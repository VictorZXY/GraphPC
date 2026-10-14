#!/usr/bin/env bash
# GraphPC arm launcher for OSWorld 2.0 — a copy of OSWorld-V2 scripts/bash/run_multienv_m3.sh
# (tag v2026.08.08) with four adaptations:
#   * no inline `export` block: every host setting (AWS_*, WEBSITE_HOST_SUFFIX,
#     OSWORLD_FILE_BASE_URL, OSWORLD_EVAL_MODEL_*, OSWORLD_USER_SIM_*, ANTHROPIC_API_KEY,
#     MINIMAX_API_KEY, OPENROUTER_API_KEY, ...) comes from <checkout>/.env, which the runner reads via
#     load_dotenv(). load_dotenv never overrides a variable that is already set, so
#     exporting upstream's placeholder values here would shadow the .env — hence none.
#     The required variable list is in .claude/docs/benchmarks/osworld_v2.md (Per-run agent steps).
#   * every harness flag comes from osworld_cell_config.sh (shared with the baseline arm);
#     this script adds only the GraphPC agent flags;
#   * runner = scripts/python/run_multienv_graphpc.py (the vendored GraphPC copy of
#     run_multienv_m3.py) with its flags; the M3Agent-only flags are gone;
#   * the cell (OSWORLD_CELL, REQUIRED: minimax, qwen or deepseek, no default; osworld_cell_config.sh
#     refuses a launch without it) picks the backbone, the model and the scored defaults:
#     minimax -> --backbone minimax, MiniMax-M3, the graphpc_minimax_m3 trees; qwen ->
#     --backbone qwen, qwen/qwen3.7-plus, the graphpc_qwen37plus trees; deepseek -> --backbone deepseek,
#     deepseek-flash, the graphpc_deepseek_flash trees.
# Run from the OSWorld-V2 checkout root on the host, naming the cell:
#   OSWORLD_CELL=minimax bash scripts/bash/run_multienv_graphpc.sh
#   OSWORLD_CELL=qwen bash scripts/bash/run_multienv_graphpc.sh
#   OSWORLD_CELL=deepseek bash scripts/bash/run_multienv_graphpc.sh
# DRY_RUN=1 prints the fully resolved command and exits 0 without running anything.
# A SCORED LAUNCH APPENDS NOTHING — osworld_cell_guard_launch_args refuses any appended argument
# unless the launch declares itself unscored, and refuses a declared-unscored launch into either
# arm's scored result tree. A wiring smoke is therefore:
#   OSWORLD_CELL=minimax OSWORLD_UNSCORED_LAUNCH=1 RESULT_DIR=./results/osworld_v2/smoke \
#   SUPPORT_SET_ROOT=./results/osworld_v2/smoke_support_set \
#     bash scripts/bash/run_multienv_graphpc.sh --specific_task_id 004 --checkpoint_steps 5,10
# (no --max_steps: runner delta 15 refuses any budget but the adapter's for the backbone, 500, and 800
# on the Qwen GraphPC arm, so a GraphPC smoke is kept short by stopping it, not by lowering the budget; the checkpoint
# override makes it touch inline checkpoint evaluation early).
# RETRIEVAL IS OFF on every launch of this script, every cell: every GraphPC run on the 28-task subset
# runs --retrieval_off. The flag is written into the command below, not
# read from a variable, so a launch that never ran the preflight still runs retrieval off; the
# launch preflight requires it on the resolved argv. Under it build_default opens no store and loads
# no embedder; --support_set_root is still passed (the guard holds the path) and nothing is written there.
set -euo pipefail

# shellcheck source=osworld_cell_config.sh
source "$(dirname "${BASH_SOURCE[0]}")/osworld_cell_config.sh"

# The Qwen and DeepSeek cells' defaults are set here, first, so the MiniMax defaults below leave them
# as they are. The MiniMax lines stay unindented: verify_runner_deltas.py reads the MiniMax cell's
# RESULT_DIR and SUPPORT_SET_ROOT defaults out of them.
BACKBONE=minimax
if [[ "${OSWORLD_CELL}" == "deepseek" ]]; then
    BACKBONE=deepseek
    MODEL="${MODEL:-deepseek-flash}"
    RESULT_DIR="${RESULT_DIR:-./results/osworld_v2/graphpc_deepseek_flash}"
    SUPPORT_SET_ROOT="${SUPPORT_SET_ROOT:-./results/osworld_v2/graphpc_deepseek_flash_support_set}"
fi
if [[ "${OSWORLD_CELL}" == "qwen" ]]; then
    BACKBONE=qwen
    MODEL="${MODEL:-qwen/qwen3.7-plus}"
    RESULT_DIR="${RESULT_DIR:-./results/osworld_v2/graphpc_qwen37plus}"
    SUPPORT_SET_ROOT="${SUPPORT_SET_ROOT:-./results/osworld_v2/graphpc_qwen37plus_support_set}"
    # The arm's step budget is 800: the Qwen control executes several tool calls per step (its --batch_actions
    # default), so within its 500 steps it can carry out more than 500 real actions, while a GraphPC step executes
    # one. The control runs 500.
    MAX_STEPS="${MAX_STEPS:-${OSWORLD_MAX_STEPS_QWEN_GRAPHPC}}"
fi
MODEL="${MODEL:-MiniMax-M3}"
RESULT_DIR="${RESULT_DIR:-./results/osworld_v2/graphpc_minimax_m3}"
# Default 4 workers, as on every control; more than 4 is refused unless FORCE_NUM_ENVS=1. The launch
# preflight's memory check charges a per-worker cost sized on a 16 GB / 4 vCPU host with each worker
# holding the embedder, where four fit; under --retrieval_off a worker holds less, so that bound is
# conservative, and it says how many fit after a host resize. --env_start_delay 30 (below) starts the
# workers 30 s apart.
NUM_ENVS="${NUM_ENVS:-4}"
if [[ "${NUM_ENVS}" -gt 4 && "${FORCE_NUM_ENVS:-0}" != "1" ]]; then
    echo "ERROR: NUM_ENVS=${NUM_ENVS} > 4 on the GraphPC arm (the cells' run shape); set FORCE_NUM_ENVS=1 to override." >&2
    exit 1
fi
# Support set lives OUTSIDE the result task tree: the runner's get_unfinished wipes
# unfinished task dirs on resume, and a store under <result_dir>/.../<model>/ would
# be walked as a domain directory.
SUPPORT_SET_ROOT="${SUPPORT_SET_ROOT:-./results/osworld_v2/graphpc_minimax_m3_support_set}"

# Appended flags the PREFLIGHT CANNOT SEE, and the scored result tree an unscored launch must not
# touch: both are refused by the shared guard in osworld_cell_config.sh (which carries the full
# argument). A scored run appends nothing, so only the guard's unconditional half applies to it (no
# write into the other arm's paths, another cell's or a locked tree).
osworld_cell_guard_launch_args graphpc \
    --result_dir "${RESULT_DIR}" --support_set_root "${SUPPORT_SET_ROOT}" -- "$@"

osworld_cell_require_task_list
osworld_cell_harness_args

osworld_cell_launch uv run --no-sync python scripts/python/run_multienv_graphpc.py \
    "${OSWORLD_CELL_HARNESS_ARGS[@]}" \
    --backbone "${BACKBONE}" \
    --model "${MODEL}" \
    --result_dir "${RESULT_DIR}" \
    --support_set_root "${SUPPORT_SET_ROOT}" \
    --retrieval_off \
    --num_envs "${NUM_ENVS}" \
    --env_start_delay 30 \
    "$@"
