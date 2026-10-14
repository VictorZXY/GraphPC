#!/usr/bin/env bash
# Shared OSWorld 2.0 cell configuration: the ONE source of truth for every HARNESS flag both
# arms pass to their runner. Sourced by run_multienv_graphpc.sh (GraphPC arm),
# run_multienv_m3_baseline.sh (bare MiniMax-M3 control), run_multienv_qwen_baseline.sh (bare
# Qwen control) and run_multienv_deepseek_baseline.sh (the DeepSeek control, upstream's M3 agent with the
# native-tool deltas); each arm's script adds only its agent flags on top of OSWORLD_CELL_HARNESS_ARGS.
# The harness flags are the same in every cell; what differs per cell (OSWORLD_CELL) is only the
# set of scored paths the launch guard holds a launch to and the Qwen GraphPC arm's step budget
# (below). scripts/osworld/launch_preflight.py runs both scripts under DRY_RUN=1 and fails if any
# other harness flag differs between the arms.
# Environment overrides exist for smokes only (e.g. MAX_STEPS=15); a scored run uses the defaults.
# Synced to <OSWorld-V2>/scripts/bash/osworld_cell_config.sh next to the launch scripts.

# The task list, relative to the checkout root, copied to the host by hand. Every launch runs the
# pre-registered 28 (OSWORLD_TASK_LIST_SUBSET). The default is applied when the task list is required
# (below), after each launcher has set its own.
OSWORLD_TASK_LIST_SUBSET="evaluation_examples/prereg_task_list.json"
TEST_ALL_META_PATH="${TEST_ALL_META_PATH:-}"
# The step budget: 500 on every arm except the Qwen GraphPC arm, which runs 800 (method_surface_qwen.max_steps
# in the pre-registration: the Qwen control executes several tool calls per step, its --batch_actions default, so
# within its 500 steps it can carry out more than 500 real actions, while a GraphPC step executes one) and whose
# launcher defaults to OSWORLD_MAX_STEPS_QWEN_GRAPHPC. Applied with the harness flags (below), after each launcher
# has set its own; runner delta 15 refuses any other value for the backbone.
OSWORLD_MAX_STEPS_DEFAULT=500
OSWORLD_MAX_STEPS_QWEN_GRAPHPC=800
MAX_STEPS="${MAX_STEPS:-}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-150,300}"
PROVIDER_NAME="${PROVIDER_NAME:-aws}"
REGION="${REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
OBSERVATION_TYPE="${OBSERVATION_TYPE:-screenshot}"
ACTION_SPACE="${ACTION_SPACE:-pyautogui}"
SCREEN_WIDTH="${SCREEN_WIDTH:-1920}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-1080}"
CLIENT_PASSWORD="${CLIENT_PASSWORD:-osworld-public-evaluation}"
# Root volume size in GB for every task that does not set its own volume_size. Non-None is what
# makes DesktopEnv run its guest-side expansion (growpart + resize2fs, self-installing
# cloud-guest-utils): the V2 client AMI ships a ~29.5 GB root partition on a 40 GB volume with
# neither growpart nor cloud-init, so the root filesystem starts near 100% full, snapd's overdue
# auto-refresh fills it, and the guest server's /screenshot handler dies writing its temp PNG
# (errno 28) ~5 min 35 s after env.reset -- on a fully idle VM, in both arms alike. Harness flag: identical in both arms.
# Cost: ~75 s per EPISODE (at least ~70 min over the 56 episodes of a 28-task pair — a FLOOR: setup retries revert up to 5x per task and the env is also recreated whenever the env spec changes, and instance types differ inside the 28). The expansion
# re-runs on every task -- _finalize_volume closes _start_emulator, reset calls it after every
# revert, and on AWS the revert launches a NEW instance at this size -- so every episode does an
# apt-get update + cloud-guest-utils install inside the guest, unguarded under `set -euo pipefail`:
# a mirror failure fails env construction for that task. On a live VM: 59 GB / 31 GB free / 46%
# used vs 29 GB / 0 free / 100% unfixed; the snap refresh still runs but is no longer fatal. Subset
# tasks 048/057/064/107/108 declare their own size in their task classes (64/80/50/80/80) and keep
# it -- this is a default, not a floor, so 064 runs BELOW this value and the other four above it --
# and the remaining 23 take this value.
VOLUME_SIZE="${VOLUME_SIZE:-60}"
# Concurrency is a run-shape setting, not a harness flag, so each script sets its own NUM_ENVS
# default and passes --num_envs itself.
DRY_RUN="${DRY_RUN:-0}"

# EVERY PATH THE SCORED CELL OWNS, per arm. The launch guard below needs both halves: an arm may
# name its OWN scored paths (that is what a scored run is) but never the other arm's, and a launch
# that declares itself unscored may name neither. The GraphPC arm owns two — its result tree and
# its support-set store, which is a scored artefact in its own right: whenever retrieval is on,
# `finalize_task` writes an entry for an episode that scores 1.0, so an entry written by a smoke
# would become retrieval evidence.
#
# Each launcher keeps its own `RESULT_DIR:-<default>` line — that is what an operator reads and
# what the arms-identity check parses — and verify_runner_deltas.py asserts each launcher's
# defaults are members of these lists, so the duplication cannot drift unnoticed.
#
# THE SCORED PATHS ARE ANCHORED TO THE CHECKOUT, THE CANDIDATES ARE NOT, and the asymmetry is the
# point. `realpath -m` resolves a relative path against the CALLER's cwd, so with relative constants
# the guard would recognise either arm's scored tree only when the launcher is invoked from the
# checkout root: from $HOME, a declared-unscored launch (`--specific_task_id 004`) into an arm's own
# scored tree would pass as "DECLARED-UNSCORED LAUNCH ... into RESULT_DIR=<the arm's own scored
# tree>". Two levels up from this file is the checkout root in both layouts — scripts/osworld/ in
# the repo, scripts/bash/ on the host — the same trick verify_runner_deltas.py's _launcher() uses.
# The CANDIDATES stay cwd-relative because cwd is what the resume pre-clean and the runner will
# themselves resolve against: anchoring those too would refuse a launch from elsewhere whose
# default ./results/... lands somewhere harmless, which is a false refusal in the other direction.
OSWORLD_CELL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OSWORLD_SCORED_PATHS_GRAPHPC=(
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/graphpc_minimax_m3"
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/graphpc_minimax_m3_support_set"
)
OSWORLD_SCORED_PATHS_BASELINE=(
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/baseline_minimax_m3"
)
# THE CELL. The two lists above are the MiniMax-M3 cell's;
# the Qwen qwen3.7-plus cell and the DeepSeek deepseek-flash cell have their own, below, with the same
# shape — the GraphPC store is a sibling of its result tree, outside it. A launch holds to its OWN
# cell's lists: every other cell's scored paths are foreign to it exactly as the other arm's are.
# OSWORLD_CELL has NO default: the GraphPC launcher serves every cell, and a launch
# that forgot the variable would run another cell's model into that cell's scored tree, which the
# guard below accepts because the tree is that arm's own. Each control launcher sets its own cell;
# the GraphPC launch names it. A launch without it, or with any other value, is refused here.
OSWORLD_CELL="${OSWORLD_CELL:-}"
if [[ "${OSWORLD_CELL}" != "minimax" && "${OSWORLD_CELL}" != "qwen" && "${OSWORLD_CELL}" != "deepseek" ]]; then
    echo "ERROR: OSWORLD_CELL must name this launch's cell, minimax, qwen or deepseek; got '${OSWORLD_CELL}'." >&2
    echo "       There is no default. The GraphPC launcher serves every cell, so name the cell:" >&2
    echo "         OSWORLD_CELL=minimax bash scripts/bash/run_multienv_graphpc.sh   (MiniMax-M3)" >&2
    echo "         OSWORLD_CELL=qwen bash scripts/bash/run_multienv_graphpc.sh      (qwen3.7-plus)" >&2
    echo "         OSWORLD_CELL=deepseek bash scripts/bash/run_multienv_graphpc.sh  (deepseek-flash)" >&2
    echo "       The control launchers set their own cell." >&2
    exit 1
fi
# The Qwen qwen3.7-plus cell.
OSWORLD_SCORED_PATHS_GRAPHPC_QWEN=(
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/graphpc_qwen37plus"
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/graphpc_qwen37plus_support_set"
)
OSWORLD_SCORED_PATHS_BASELINE_QWEN=(
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/baseline_qwen37plus"
)
# The DeepSeek deepseek-flash cell (both arms thinking on).
OSWORLD_SCORED_PATHS_GRAPHPC_DEEPSEEK=(
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/graphpc_deepseek_flash"
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/graphpc_deepseek_flash_support_set"
)
OSWORLD_SCORED_PATHS_BASELINE_DEEPSEEK=(
    "${OSWORLD_CELL_ROOT}/results/osworld_v2/baseline_deepseek_flash"
)
# LOCKED TREES no launch of any cell or arm may write to: result trees (or stores) an operator keeps
# frozen, e.g. a finished run kept for reference under another name. No arm owns them, so the guard
# treats them as foreign to every launch. Empty by default; to lock a tree, add its path anchored to
# ${OSWORLD_CELL_ROOT} like the lists above.
OSWORLD_LOCKED_PATHS=()

# The options whose appended value REDIRECTS WHERE THIS LAUNCH WRITES. Matched by PREFIX, not by
# exact spelling: both runners leave argparse's `allow_abbrev` at its default, so `--resu` resolves
# to `--result_dir` and would walk straight past an exact-match `case`. Over-matching is the safe direction — an ambiguous prefix argparse would itself
# reject costs an operator one clear refusal, while under-matching costs a scored arm its episodes.
OSWORLD_WRITE_PATH_OPTIONS=(--result_dir --support_set_root)

# THE COMPLETE RULE FOR A LAUNCH, in two parts.
#
# PART ONE, CHECKED ON EVERY LAUNCH INCLUDING A BARE ONE: an arm never writes into the OTHER arm's
# scored paths, another cell's, or a locked tree. This half is unconditional because the most ordinary invocation form there is —
# no flags, one environment variable — can reach the catastrophe the guard exists to prevent:
#   DRY_RUN=1 RESULT_DIR=./results/osworld_v2/graphpc_minimax_m3 bash scripts/bash/run_multienv_m3_baseline.sh
# resolves with `--result_dir ./results/osworld_v2/graphpc_minimax_m3` and appends nothing, so a
# guard that returned at `$# -eq 0` would compare nothing. Both arms default to the same model, action
# space and observation type, so that path collides exactly with the GraphPC arm's tree: the
# baseline pre-clean walks it and upstream's `get_unfinished` empties every unfinished task dir
# under it. Each arm's own default passes this check, so the scored argv is unchanged.
#
# PART TWO, WHEN ANYTHING IS APPENDED: a scored launch appends nothing. Both launchers end in "$@"
# and the launch preflight resolves them BARE, so no preflight verdict covers anything typed after
# the script name — and argparse is last-wins, so an appended `--max_steps` / `--observation_type` /
# `--test_all_meta_path` silently breaks the harness-flag identity the preflight asserts. A
# blocklist of flag NAMES can only refuse the flags someone remembered. So: anything appended requires
# OSWORLD_UNSCORED_LAUNCH=1, and a launch that declares itself unscored may not name ANY scored
# path of either arm.
#
# Usage: osworld_cell_guard_launch_args <arm> <option> <value> [<option> <value> ...] -- "$@"
# where the option/value pairs are the write paths the script itself passes.
osworld_cell_guard_launch_args() {
    local arm="$1"
    shift
    local -A script_paths=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        if [[ $# -lt 2 ]]; then
            echo "ERROR: osworld_cell_guard_launch_args: write paths come in <option> <value> pairs." >&2
            exit 1
        fi
        script_paths["$1"]="$2"
        shift 2
    done
    if [[ "${1:-}" != "--" ]]; then
        echo "ERROR: osworld_cell_guard_launch_args was invoked without its -- separator, so it does" >&2
        echo "       not know which arguments were appended at launch. Fix the call site." >&2
        exit 1
    fi
    shift
    local -a appended=("$@")

    # Owned: this arm's lists in this launch's cell. Foreign: every other scored list, of either arm
    # in either cell, and the locked trees.
    local -a owned=() foreign=()
    case "${OSWORLD_CELL}:${arm}" in
        minimax:graphpc)
            owned=("${OSWORLD_SCORED_PATHS_GRAPHPC[@]}")
            foreign=("${OSWORLD_SCORED_PATHS_BASELINE[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_QWEN[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_QWEN[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_DEEPSEEK[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_DEEPSEEK[@]}")
            ;;
        minimax:baseline)
            owned=("${OSWORLD_SCORED_PATHS_BASELINE[@]}")
            foreign=("${OSWORLD_SCORED_PATHS_GRAPHPC[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_QWEN[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_QWEN[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_DEEPSEEK[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_DEEPSEEK[@]}")
            ;;
        qwen:graphpc)
            owned=("${OSWORLD_SCORED_PATHS_GRAPHPC_QWEN[@]}")
            foreign=("${OSWORLD_SCORED_PATHS_BASELINE_QWEN[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC[@]}" "${OSWORLD_SCORED_PATHS_BASELINE[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_DEEPSEEK[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_DEEPSEEK[@]}")
            ;;
        qwen:baseline)
            owned=("${OSWORLD_SCORED_PATHS_BASELINE_QWEN[@]}")
            foreign=("${OSWORLD_SCORED_PATHS_GRAPHPC_QWEN[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC[@]}" "${OSWORLD_SCORED_PATHS_BASELINE[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_DEEPSEEK[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_DEEPSEEK[@]}")
            ;;
        deepseek:graphpc)
            owned=("${OSWORLD_SCORED_PATHS_GRAPHPC_DEEPSEEK[@]}")
            foreign=("${OSWORLD_SCORED_PATHS_BASELINE_DEEPSEEK[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC[@]}" "${OSWORLD_SCORED_PATHS_BASELINE[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_QWEN[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_QWEN[@]}")
            ;;
        deepseek:baseline)
            owned=("${OSWORLD_SCORED_PATHS_BASELINE_DEEPSEEK[@]}")
            foreign=("${OSWORLD_SCORED_PATHS_GRAPHPC_DEEPSEEK[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC[@]}" "${OSWORLD_SCORED_PATHS_BASELINE[@]}"
                     "${OSWORLD_SCORED_PATHS_GRAPHPC_QWEN[@]}" "${OSWORLD_SCORED_PATHS_BASELINE_QWEN[@]}")
            ;;
        *)
            echo "ERROR: osworld_cell_guard_launch_args: unknown arm '${arm}'." >&2
            exit 1
            ;;
    esac
    foreign+=("${OSWORLD_LOCKED_PATHS[@]}")

    if ! command -v realpath >/dev/null 2>&1; then
        echo "ERROR: realpath is not on PATH, so this launch's write paths cannot be compared with the" >&2
        echo "       scored ones and this guard cannot show that it stays out of them." >&2
        exit 1
    fi

    # Every scored path below is refused by CONTAINMENT, not by equality: a store kept inside a result
    # tree (support_set/), like any write anywhere under a scored tree, lands in that tree. A sibling
    # that only shares the name's prefix (graphpc_qwen37plus_smoke beside graphpc_qwen37plus) is outside it.

    # EVERY path this launch could write to. Both the value the script passes AND any appended
    # override count: argparse is last-wins, so the runner obeys the appended one, while the
    # baseline's resume pre-clean reads the shell variable — so a launch can be destructive through
    # either, and one cannot launder the other.
    local -a candidates=()
    local option value
    for option in "${!script_paths[@]}"; do
        candidates+=("${script_paths[${option}]}")
    done
    local index token name
    for ((index = 0; index < ${#appended[@]}; index++)); do
        token="${appended[index]}"
        name="${token%%=*}"
        if [[ "${name}" != --* || "${name}" == "--" ]]; then
            continue
        fi
        for option in "${OSWORLD_WRITE_PATH_OPTIONS[@]}"; do
            if [[ "${option}" != "${name}"* ]]; then
                continue
            fi
            if [[ "${token}" == *=* ]]; then
                value="${token#*=}"
            elif (( index + 1 < ${#appended[@]} )); then
                value="${appended[index + 1]}"
            else
                continue
            fi
            candidates+=("${value}")
            script_paths["${option}"]="${value}"   # last-wins, as argparse resolves it
        done
    done

    local candidate resolved scored scored_resolved
    for candidate in "${candidates[@]}"; do
        resolved="$(realpath -m -- "${candidate}")"
        for scored in "${foreign[@]}"; do
            scored_resolved="$(realpath -m -- "${scored}")"
            if [[ "${resolved}" == "${scored_resolved}" || "${resolved}" == "${scored_resolved}/"* ]]; then
                echo "ERROR: this ${OSWORLD_CELL} ${arm} launch would write to ${candidate}, which belongs to the OTHER arm" >&2
                echo "       of its cell, to another cell, or to a locked tree (${scored}). Both arms of a cell default" >&2
                echo "       to the same model, action space and observation" >&2
                echo "       type, so the trees collide exactly: the baseline's resume pre-clean walks the" >&2
                echo "       directory and upstream's get_unfinished EMPTIES every unfinished task dir under" >&2
                echo "       it. Point RESULT_DIR (and SUPPORT_SET_ROOT on the GraphPC arm) somewhere else." >&2
                exit 1
            fi
        done
    done

    # The support set must not sit inside a result tree: the runner's get_unfinished walks every
    # directory under <result_dir>/<action_space>/<observation_type>/<model>/ as a domain, so a
    # store underneath one is wiped on the next resume. Stated in the GraphPC launcher; enforced here.
    local store="${script_paths[--support_set_root]:-}"
    local tree="${script_paths[--result_dir]:-}"
    if [[ -n "${store}" && -n "${tree}" ]]; then
        local store_resolved tree_resolved
        store_resolved="$(realpath -m -- "${store}")"
        tree_resolved="$(realpath -m -- "${tree}")"
        if [[ "${store_resolved}" == "${tree_resolved}" || "${store_resolved}" == "${tree_resolved}/"* ]]; then
            echo "ERROR: the support set (${store}) resolves inside the result tree (${tree}). The runner's" >&2
            echo "       get_unfinished walks every directory under the result tree as a domain and empties" >&2
            echo "       unfinished ones, so the store would be destroyed on the next resume. Keep it outside." >&2
            exit 1
        fi
    fi

    if [[ ${#appended[@]} -eq 0 ]]; then
        return 0
    fi

    if [[ "${OSWORLD_UNSCORED_LAUNCH:-0}" != "1" ]]; then
        echo "ERROR: this launch appends argument(s) the launch preflight never saw: ${appended[*]}" >&2
        echo "       The preflight resolves this script with NO arguments, so its verdict — harness-flag" >&2
        echo "       identity between the arms, the method under test, the task list — says nothing about" >&2
        echo "       anything appended here, and argparse is last-wins." >&2
        echo "       A SCORED CELL LAUNCH APPENDS NOTHING. To run a smoke or a deliberate ablation, declare" >&2
        echo "       it: OSWORLD_UNSCORED_LAUNCH=1 with a throwaway RESULT_DIR (and SUPPORT_SET_ROOT on the" >&2
        echo "       GraphPC arm), and record it in that run's README. To change a harness setting for a" >&2
        echo "       scored run, change osworld_cell_config.sh." >&2
        exit 1
    fi

    for candidate in "${candidates[@]}"; do
        resolved="$(realpath -m -- "${candidate}")"
        for scored in "${owned[@]}" "${foreign[@]}"; do
            scored_resolved="$(realpath -m -- "${scored}")"
            if [[ "${resolved}" == "${scored_resolved}" || "${resolved}" == "${scored_resolved}/"* ]]; then
                echo "ERROR: OSWORLD_UNSCORED_LAUNCH=1 would write to ${candidate}, which is a SCORED path" >&2
                echo "       (${scored}). The result tree's unfinished episodes are emptied by the resume" >&2
                echo "       pre-clean and by upstream's get_unfinished, which ignores --specific_task_id /" >&2
                echo "       --domain on the control arm; the support-set store takes an entry from any" >&2
                echo "       episode that scores 1.0 whenever retrieval is on, so one written by a smoke" >&2
                echo "       becomes retrieval evidence. Point RESULT_DIR and" >&2
                echo "       SUPPORT_SET_ROOT at throwaway paths." >&2
                exit 1
            fi
        done
    done
    echo "WARNING: DECLARED-UNSCORED LAUNCH (OSWORLD_UNSCORED_LAUNCH=1): appending ${appended[*]}" >&2
    echo "         into ${tree:-<no result dir>}. This is NOT a scored-cell launch; record it in the run README." >&2
}

# CWD DEPENDENCY, WRITTEN DOWN BECAUSE IT IS LOAD-BEARING BY ACCIDENT. This check tests a relative
# path against the CALLER's cwd, and on the baseline arm it sits between the launch guard and the
# resume pre-clean — which uses ${RESULT_DIR} directly and is therefore cwd-independent. So a
# destructive launch from the wrong cwd dies HERE, before the pre-clean, for a reason that has
# nothing to do with containment. Resolving TEST_ALL_META_PATH against BASH_SOURCE would be an
# obvious improvement and would remove that accidental stop; the guard above is what must be
# trusted instead, which is why its scored constants are anchored and this one is not. If you make
# this path absolute, or anchor it, check that the guard alone refuses a destructive launch from the
# wrong cwd.
osworld_cell_require_task_list() {
    TEST_ALL_META_PATH="${TEST_ALL_META_PATH:-${OSWORLD_TASK_LIST_SUBSET}}"
    if [[ ! -f "${TEST_ALL_META_PATH}" ]]; then
        echo "ERROR: task list ${TEST_ALL_META_PATH} not found (copy it from data/osworld_v2/ to the host first)." >&2
        exit 1
    fi
}

# Fills the array OSWORLD_CELL_HARNESS_ARGS; identical for both arms by construction, except MAX_STEPS, which the
# Qwen GraphPC launcher sets first (the defaults below apply otherwise).
osworld_cell_harness_args() {
    TEST_ALL_META_PATH="${TEST_ALL_META_PATH:-${OSWORLD_TASK_LIST_SUBSET}}"
    MAX_STEPS="${MAX_STEPS:-${OSWORLD_MAX_STEPS_DEFAULT}}"
    OSWORLD_CELL_HARNESS_ARGS=(
        --headless
        --observation_type "${OBSERVATION_TYPE}"
        --action_space "${ACTION_SPACE}"
        --screen_width "${SCREEN_WIDTH}"
        --screen_height "${SCREEN_HEIGHT}"
        --test_all_meta_path "${TEST_ALL_META_PATH}"
        --eval_version v2
        --max_steps "${MAX_STEPS}"
        --checkpoint_eval_mode inline
        --checkpoint_steps "${CHECKPOINT_STEPS}"
        --save_model_eval_raw_info
        --provider_name "${PROVIDER_NAME}"
        --region "${REGION}"
        --client_password "${CLIENT_PASSWORD}"
        --volume_size "${VOLUME_SIZE}"
    )
    if [[ "${USE_PUBLIC_IP:-0}" == "1" ]]; then
        OSWORLD_CELL_HARNESS_ARGS+=(--use_public_ip)
    fi
}

# Under DRY_RUN=1 print the fully resolved command (marker line, then one argv element per
# line, which launch_preflight.py parses) and exit 0 without running anything; otherwise run it.
# Both arms launch with `uv run --no-sync`: the checkout is pinned at a tag, and a sync at launch
# would silently revert any out-of-lock package installed into the checkout's environment to the
# lock's versions.
osworld_cell_launch() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN argv (one element per line):"
        printf '%s\n' "$@"
        exit 0
    fi
    mkdir -p logs   # the runners open logs/normal-*.log relative to cwd at startup
    "$@"
}
