# scripts/

What running an experiment needs: runners, launch and sync scripts, the paired scorer, and the checks a launch
depends on (their answer depends on the run's environment — the host, the provider, the synced snapshot). A check
whose answer depends only on the code is a test; write it under `tests/`.

**The benchmark checkouts sit beside the repository**, and the scripts find them there: `../WebArena` (the
`--webarena_root` default of `webarena/run_graphpc.py`, `run_baseline.py`, `run_graphpc_parallel.py`, `run_lanes.py`
and `verify_auth_logins.py`) and `../OSWorld-V2`, the local OSWorld-V2 staging clone. The sync script writes into the
staging clone by default (`--osworld_root` overrides it). `osworld/verify_runner_deltas.py` reads upstream's files
from it: its `--cell qwen` and `--cell deepseek` modes take the clone as `--checkout_root` (default: the staging clone
beside the repository), while the default mode's check that the baseline runner equals upstream's
`run_multienv_m3.py` reads upstream's runner only from `../OSWorld-V2` (or, on the host, from the vendored runner's
sibling) — so run it from a tree that sits beside the clone.

## `osworld/` — OSWorld 2.0

- **Top level: everything synced to the run host** (`src/benchmarks/osworld/sync.py`), plus the sync script
  `sync_osworld_graphpc.py`. These files cannot move into a subfolder: on the host they sit flat in
  `scripts/python/` and `scripts/bash/`, and each finds the checkout root two directories up, which holds in both
  places. Runners and launch scripts (`run_multienv_*`, `osworld_cell_config.sh`), the launch preflight
  (`launch_preflight.py`) with the run-order solver it imports (`solve_subset_run_order.py`), the runner-delta check
  (`verify_runner_deltas.py`) and the per-model vision gates (`verify_{minimax,qwen,deepseek}_vision.py`).
- **`qwen_control/`** — the Qwen cell's control: upstream's `osworld-v2.1` Qwen agent with its declared deltas,
  synced to the checkout's `mm_agents/qwen/`; its runner is the top-level `run_multienv_qwen_baseline.py` and its
  launcher `run_multienv_qwen_baseline.sh`. `verify_runner_deltas.py --cell qwen` holds every file to its upstream
  blob.
- **`deepseek_control/`** — the DeepSeek cell's control: upstream's `v2026.08.08` M3 agent (`mm_agents/m3`) with the
  marked deltas its files declare (the computer tool declared natively, the action read from the tool_use block, the
  tool protocol in the history), synced to the checkout's `mm_agents/deepseek/`; its runner is the top-level
  `run_multienv_deepseek_baseline.py` (upstream's `run_multienv_m3.py` plus the `--volume_size` port and the
  control-agent import) and its launcher `run_multienv_deepseek_baseline.sh`. `verify_runner_deltas.py --cell
  deepseek` holds every file to its upstream blob and the two DeepSeek-cell runners to parity.
- **`tools/score_paired.py`** — run locally on the two pulled result trees: the paired per-task score. Not synced.

## `webarena/` — WebArena

- `run_graphpc.py` (the GraphPC arm), `run_baseline.py` (the control: WebArena's own `run.py` loop behind a thin
  wrapper), `run_graphpc_parallel.py` (the site-partitioned Lite-165 launcher), `run_lanes.py` (the lane driver for a
  full-812 arm: a control, a cold arm, or a warm round against a frozen store — site lanes in two phases from a pinned
  code root, recorded PIDs, the phase-start `.auth` refresh, a heartbeat with data-defined tripwires, freeze / resume,
  casualty listing and re-run lanes), `launch_preflight.py` (the launch preflight of a GraphPC arm) and
  `verify_auth_logins.py` (every `.auth` state a task list uses still logs in; the lane driver runs it after
  re-rendering `.auth` at each phase start).
