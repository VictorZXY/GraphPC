# Benchmarks

Per-benchmark deployment, runner usage, subset semantics, baseline, and adapter architecture. Pair with `experiments.md` for the benchmark-agnostic experiment protocol.

Benchmark adapters live under `src/benchmarks/<name>/`; per-benchmark runners under `scripts/<name>/`. Each benchmark exposes the same inner boundary to the orchestrator (`decide_next_action` / `update_runtime` on `Observation` / `ActionBatch` / `GraphPCDecision` types — see `design.md` Workflow and "Benchmark integration boundary") and its own outer boundary to its harness. Each benchmark's baseline is its own bundled control (run once, lock, never re-run).

**WebArena is the primary exploration benchmark; OSWorld 2.0 is the second benchmark.** Both run against remote hosts that must be set up and configured before any run; each runbook starts with that setup.

- `webarena.md` — WebArena: site-host and driver setup, running (single tasks, Lite-165, full-812 lanes, the warm round), the smoke, subsets, the baseline, the per-backbone routes, outputs, adapter architecture.
- `osworld_v2.md` — OSWorld 2.0: one-time setup (AWS account, run host, staging clone), per-run steps (sync → ship → preflight → run → score), the smoke test, what differs per cell (Qwen, DeepSeek), task set, baseline, standing facts and rules, adapter architecture and how to add a model.
