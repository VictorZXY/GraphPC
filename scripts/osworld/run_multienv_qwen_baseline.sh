#!/usr/bin/env bash
# Control-arm launcher of the OSWorld 2.0 Qwen qwen3.7-plus cell. Upstream ships no launcher for
# its Qwen runner, so this one follows run_multienv_m3_baseline.sh. The agent and the runner are
# upstream's osworld-v2.1 Qwen control, vendored with the deltas D1-D9 their headers declare
# (scripts/osworld/qwen_control/mm_agents/qwen/ -> mm_agents/qwen/, and
# scripts/osworld/run_multienv_qwen_baseline.py -> scripts/python/run_multienv_qwen_baseline.py).
# This script only fixes the invocation so the two arms are paired:
#   * OSWORLD_CELL=qwen: the launch guard holds this launch to the Qwen cell's scored paths;
#   * every harness flag comes from osworld_cell_config.sh (shared with the GraphPC arm); this
#     script adds only the control's agent flags;
#   * the route is the runner's own (delta D1): OpenRouter, the provider pin and an explicit
#     reasoning form on every request. The key is OPENROUTER_API_KEY from <checkout>/.env, read by
#     the runner's load_dotenv(); it is never passed on the command line, and the runner refuses to
#     start without it (delta D8);
#   * thinking is pinned explicitly (--enable_thinking, upstream's flag name; D1 maps it to the
#     route's reasoning form);
#   * --generation_timeout 300: upstream's own runner flag, whose default is 200 s. A stream cut
#     at this clock is re-sent only when its chars/4 token estimate (this route has no tokenize
#     endpoint) is <= 14,000, about 16,600-17,000 true tokens at the route's measured 3.29-3.37
#     characters per token: 278-285 s at its ~60 tokens/s, rounded up to 300.
#     No vendored file changes; the launch preflight refuses a scored launch without exactly this value.
#   The sampling, the output cap and every other agent flag are upstream's defaults.
# Resume: upstream's runner shutil.rmtree()s every SELECTED task dir without result.txt before it
# re-runs the task, which would destroy the failed attempt's api_logs/ (its per-call records and
# any agent_error record). So before a real launch this script MOVES those dirs out of the result
# tree, to <RESULT_DIR>_quarantine/<UTC timestamp>/<domain>/<task_id>, choosing them with the
# runner's own task selection on the same argv; then the runner finds nothing to delete.
# Run from the OSWorld-V2 checkout root on the host: bash scripts/bash/run_multienv_qwen_baseline.sh
# DRY_RUN=1 prints the fully resolved command and exits 0 without running anything and without the
# quarantine.
# A SCORED LAUNCH APPENDS NOTHING. A one-task smoke must declare itself and use a throwaway tree:
#   OSWORLD_UNSCORED_LAUNCH=1 RESULT_DIR=./results/osworld_v2/smoke_qwen_baseline \
#     bash scripts/bash/run_multienv_qwen_baseline.sh --specific_task_id 004 --max_steps 15 --checkpoint_steps 5,10
set -euo pipefail

OSWORLD_CELL=qwen
# shellcheck source=osworld_cell_config.sh
source "$(dirname "${BASH_SOURCE[0]}")/osworld_cell_config.sh"

MODEL="${MODEL:-qwen/qwen3.7-plus}"
RESULT_DIR="${RESULT_DIR:-./results/osworld_v2/baseline_qwen37plus}"
# No embedder on this arm; workers mostly wait on the VM / API. Same default as the MiniMax control.
NUM_ENVS="${NUM_ENVS:-4}"

# Appended flags the preflight cannot see, and scored paths an unscored launch must not touch: both
# refused by the shared guard in osworld_cell_config.sh. Placed before the quarantine, so a refused
# launch moves nothing.
osworld_cell_guard_launch_args baseline --result_dir "${RESULT_DIR}" -- "$@"

osworld_cell_require_task_list

osworld_cell_harness_args
QWEN_BASELINE_ARGV=(
    "${OSWORLD_CELL_HARNESS_ARGS[@]}"
    --model "${MODEL}"
    --enable_thinking
    --generation_timeout 300
    --result_dir "${RESULT_DIR}"
    --num_envs "${NUM_ENVS}"
    "$@"
)

# Moves every selected unfinished task dir out of the result tree before the runner's resume would
# delete it. The selection and the model directory name are the runner's own (_select_tasks with the
# deletion step replaced, _model_for_paths), parsed from the exact argv the launch passes, so this
# never re-implements a runner path rule in bash.
osworld_qwen_baseline_resume_quarantine() {
    uv run --no-sync python - "${QWEN_BASELINE_ARGV[@]}" <<'PY'
import datetime
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join("scripts", "python"))
import run_multienv_qwen_baseline as runner  # noqa: E402  (imports only; main() never runs)

args = runner.config(sys.argv[1:])
with open(args.test_all_meta_path, encoding="utf-8") as file:
    metadata = json.load(file)
runner._filter_unfinished = lambda selected, _args: selected
selected = runner._select_tasks(metadata, args)
root = os.path.join(args.result_dir, args.action_space, args.observation_type, runner._model_for_paths(args))
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
quarantine = os.path.join(os.path.normpath(args.result_dir) + "_quarantine", stamp)
moved = 0
for domain, task_ids in selected.items():
    for task_id in task_ids:
        task_dir = os.path.join(root, domain, str(task_id))
        if not os.path.lexists(task_dir) or os.path.isfile(os.path.join(task_dir, "result.txt")):
            continue
        target = os.path.join(quarantine, domain, str(task_id))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(task_dir, target)
        moved += 1
        print(f"resume quarantine: moved {task_dir} -> {target}")
print(f"resume quarantine: moved {moved} unfinished task dir(s) out of {root}")
PY
}

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: resume quarantine skipped (would move unfinished selected task dirs out of ${RESULT_DIR})"
else
    osworld_qwen_baseline_resume_quarantine
fi

osworld_cell_launch uv run --no-sync python scripts/python/run_multienv_qwen_baseline.py "${QWEN_BASELINE_ARGV[@]}"
