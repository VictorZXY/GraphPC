# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Cursor, OpenCode, etc.) when working with code in this repository. It is the canonical agent-context doc; `CLAUDE.md` is a symlink to this file.

---

## Project overview

**GraphPC — From Actions to Effects: Procedure-Causal Graphs for Tracking Progress in Computer-Use Agents**

GraphPC is a training-free harness that wraps an unmodified foundation model to carry out multi-step computer-use tasks — on WebArena from the accessibility tree, on OSWorld 2.0 from screenshots. Its core is the **procedure-causal graph** ("PC"), which records progress as the effects of actions: nodes are stages, typed actions and outcomes; edges order the stages and the actions within them, group each action under its stage, and tie each action to its outcome by an effect from a closed vocabulary (`field_filled`, `saved_persisted`, `no_change`, `error`, …). The model predicts each action's effect before acting and labels it afterwards; a model-free test of whether the interface changed (a diff of the observation text, or a pointer-masked screenshot diff) turns a progress label on an unchanged step into `no_change`.

The same graph serves three roles:
- **Progress verification** — a completion gate admits a completion claim only on the record's evidence (a confirmed effect beyond navigation, or an answer grounded in recent observations); no model call.
- **Context management** — the model rewrites a compressed summary of the graph at every step; with the plan's stages and the recent labelled actions it carries the episode from step to step.
- **Reusable experience** — records of successful episodes, their labels re-checked at storage, are retrieved into later tasks as effect-labelled procedures (deterministic text retrieval; no model call).

Each step makes two model calls: a deciding call that reads the observation, the graph's summary and any retrieved episodes, and writes the previous step's effect label, the plan and summary, and the next action with its predicted effect (the graph update is fused into this call); and a grounding call that turns the action into an executable command. The change test, the gate and retrieval call no model. In code the graph is kept as that record — plan, labelled per-step actions, summary — not as node/edge JSON (`.claude/docs/design.md` §3).

Externally GraphPC is a single-agent interface; internally it is a fixed pipeline coordinated by an explicit orchestrator — **NOT** a free-form multi-agent conversation framework. Internal roles: graph-reasoning (the deciding call, with the fused graph update), retrieval, action grounding, and the monitor (the model-free change test).

This is a research-oriented system. Debuggability and explicit structure are important.

---

## Read first

Before any non-trivial task, read `.claude/docs/design.md` (the method as it ships, with its key-decisions log) and `.claude/docs/experiments.md` (the experiment protocol); per-benchmark host setup, launch and scoring are under `.claude/docs/benchmarks/`. These state docs — this file included — describe only the current state, with no change history.

`.claude/docs/project_state.md` is the progress doc: where your work stands and how it evolved (runs launched and scored, decisions that change what is run or measured, result-affecting defects, open and parked items). Update it only at task or run completion, not continuously mid-run. Per-run *live* working state belongs in that run's own location (e.g. `runs/<id>/`, gitignored) or branch / worktree, so parallel runs never contend on it. Keep it concise.

---

## Architecture

### Big picture

GraphPC is externally a single-agent interface, internally a **structured multi-agent workflow coordinated by an explicit orchestrator** — *not* a free-form multi-agent conversation framework. Understanding this distinction matters: do not introduce agent-to-agent chat patterns or hide intermediate state inside reasoning text.

The orchestrator (`src/agents/graphpc/orchestrator.py:GraphPCOrchestrator`) drives the benchmark-agnostic control flow. At the call level there are **two LLM calls per step** (three in two-pass); retrieval and the change test are deterministic.

**Single-pass (default, `design.md` Workflow / Measurement discipline):**

```
retrieval (deterministic) → graph-reasoning (model call; reasoning + graph update fused) → action (model call; grounding)
```

**Two-pass (`--two_pass`; a backup design, not the default):**

```
graph-reasoning(initial) → retrieval (deterministic) → graph-reasoning(refine) → action
```

The graph stage is **fused** into the graph-reasoning call — one model call produces both the reasoning and the structured graph update; there is no separate graph LLM call. Retrieval ranks deterministically (local sentence-embedding text arm + relevance gate, top-2; zero LLM calls). Stages repeat iteratively as the runtime advances through the task.

### Internal agents and schema

The internal agent modules (at the call level, graph is fused into reasoning — see Big picture above; retrieval is deterministic with no LLM call):

- `src/agents/reasoning/` — **graph-reasoning**: one model call producing reasoning + the fused graph update: the prior step's effect label, the plan, the rewritten `graph_summary`, and each intended action with its predicted effect class — the record from which the procedure-causal graph (stages, actions, outcomes; *precedes* / *groups* / *causes*; `design.md` §3) is read. The call emits no node/edge JSON. Modes: `decide` (single-pass), `initial`/`refine` (two-pass). One implementation shared by every backbone: `graph_reasoning.py` (`GraphReasoningAgent`).
- `src/agents/graph/` — holds `a11y_parser.py` only: the accessibility-tree relevance-scoring helpers (`score_node`, the focused-window heuristic, the stable node id) that the WebArena parser reuses for its node selection (the OSWorld 2.0 path is screenshot-primary and reads no accessibility tree). The procedure-causal graph (stages, actions and outcomes; *precedes* / *groups* / *causes* — `.claude/docs/design.md` §3) has no code object of its own: it is kept as a record — the typed effect-class fields of the reasoning output contract (`src/agents/common/schemas_json.py`; predicted vs actual, which enforce the predict/verify discipline), the plan, and the model-maintained `graph_summary`, the only graph artifact on `RuntimeState.content`. The graph stage is **compulsory** and **fused into the graph-reasoning call** (no separate model call).
- `src/agents/retrieval/` — **deterministic text-only retrieval**: local sentence-embedding ranking (top-2 hard cap), relevance gate, zero LLM calls. Ranking is text-only; the text arm is a hard requirement (an unavailable embedder raises `TextArmUnavailableError`, never degrades). `support_set.py` holds `LocalSupportSetStore` (persistent JSON store under `artifacts/support_set/<backend>/`; entry shape intentionally fluid).
- `src/agents/action/` — thin grounding layer: turns the upstream intended-action (`action_class` + semantic target + params) into concrete operations in `ActionBatch.raw["input"]`. Holds only the base contract (`base.py`); the concrete grounding agents are benchmark-side because grounding is backbone- and benchmark-specific — `src/benchmarks/webarena/action_agent.py` (`WebArenaActionAgent`: id-based browser actions, shared by every backbone) and, on OSWorld 2.0, one per model profile: `src/benchmarks/osworld/action_minimax.py` (M3's `<tool_call>` format), `action_qwen.py` (the Qwen control's `computer_use` call) and `action_deepseek.py` (a native function call).
- `src/agents/monitor/` — currently `HeuristicMonitorAgent`, a minimal non-LLM monitor for orchestrator feedback.

Shared structured dataclasses live in `src/agents/common/schema.py` (`Observation`, `PlanState`, `RuntimeState`, `Action`, `ActionBatch`, `RetrievalResult`, `GraphPCDecision`, `GraphPCRuntimeUpdate`, `AgentMessage`). **This schema is considered stable** — extend additively, do not repurpose fields, do not refactor it for cleanup. Non-additive edits are allowed only when keeping the current schema would force materially worse code; see the schema-modification policy below. Graph artifacts piggy-back on existing `content` / `metadata` fields rather than adding new top-level types (`graph_summary` alone).

### Observation handling

The orchestrator passes the **RAW environment observation** (the same accessibility-tree text the bare baseline reads) to the deciding reasoning call; the **action** stage reads the relevance-selected projection for id-grounding, and retrieval the compact projection (`design.md` §2). The rule: the reasoner never sees less than the baseline sees. Where a benchmark's raw observation is not naturally bounded, its instantiation is decided per benchmark (`design.md` §2): OSWorld 2.0 is screenshot-primary with no accessibility channel, at the resolution V2's M3 agent sends; each profile reads the frame as its own control sends it (`design.md` §2). Compaction helpers live in `src/agents/common/compaction.py`; backbone-specific observation processing lives in the relevant backbone module.

### Backbones

Each backbone lives under `src/agents/backbone/<name>/` (currently `claude/`, `gpt/`, `deepseek/`, `minimax/` and `qwen/`; every wrapper is driven by the shared reasoning agent, and by the shared action agent on WebArena; on OSWorld 2.0 the reasoning agent is the benchmark-side subclass `OSWorldReasoningAgent`, and each model's backbone class (over the timeline route) and grounding agent are benchmark-side, selected by its model profile in `src/benchmarks/osworld/`) and provides a thin wrapper (`main.py`) that the per-backbone agents share for model calls, screenshot prep, and tool-call parsing. What the shared agents need from a backbone is written down in `src/agents/backbone/base.py`: the `Backbone` Protocol, one method (`complete_json`), with its result keys, error contract and the documented per-backbone divergences; it is documentation, not enforced (nothing subclasses it). Claude uses the Anthropic first-party API (native Messages API).

---

## Benchmarks

Benchmark adapters live under `src/benchmarks/<name>/`. Each benchmark's section below covers its adapter, deployment workflow (if any), and end-to-end execution guide. New benchmarks should be added as sibling subsections following the same pattern.

**WebArena is the primary exploration benchmark; OSWorld 2.0 is the second benchmark.** Each benchmark's baseline is its own bundled control (WebArena: WebArena's bundled `PromptAgent` (`run.py --agent_type prompt`), run through our thin wrapper `scripts/webarena/run_baseline.py`; OSWorld 2.0: OSWorld-V2's own runner + agent for the model — for MiniMax-M3 that is `mm_agents/m3` unmodified, driven by a verbatim vendored copy of `run_multienv_m3.py` whose one delta is upstream's own `--volume_size` flag, carried identically by the GraphPC arm, launched by our `scripts/osworld/run_multienv_m3_baseline.sh`).

### OSWorld 2.0

The second benchmark is OSWorld 2.0 (`xlang-ai/OSWorld-V2`, 108 long-horizon tasks). Three cells — MiniMax-M3, Qwen `qwen3.7-plus`, DeepSeek `deepseek-flash` — each a same-model pair on the pre-registered 28-task subset (`data/osworld_v2/subset_prereg.json`).

**Adapter.** `src/benchmarks/osworld/graphpc_agent.py:OSWorldGraphPCAgent` is the boundary layer. Outward it is exactly the agent contract V2's `lib_run_single.run_single_example` drives — `reset(runtime_logger)`, `predict(instruction, obs) -> (response_text, [pyautogui code | "DONE" | "FAIL" | "WAIT"])`, optional `set_api_log_dir` — so the V2 driver runs unmodified; inward it calls only `decide_next_action` / `update_runtime`. It builds a screenshot-primary `Observation` (no accessibility channel; the M3 system prompt's environment facts ride on the optional `environment_brief` content key), grounds through the benchmark-side `src/benchmarks/osworld/action_minimax.py` + `m3_action_translator.py` (V2's native M3 `<tool_call>` format and pyautogui translation, copied from `mm_agents/m3` with documented edits; coordinates the deciding call wrote into `target` / `intent` are stripped from the grounding line), degrades every step-level failure to the env `WAIT` token, and persists full-score trajectories to the support set through the runner's post-episode `finalize_task` hook. The give-up is the `fail` action: the action agent hands it over as a `done` carrying `terminal_intent: "fail"`, so the completion gate evaluates it like a success claim, and the adapter emits the profile's give-up token once admitted (`FAIL`; `DONE` on Qwen, whose control never emits `FAIL`); converting a plain `done` that declares infeasibility to the give-up token (audited) is a fallback. The deciding request is rendered benchmark-side by `src/benchmarks/osworld/reasoning.py` (`OSWorldReasoningAgent` over the backbone's model-neutral timeline route `TimelineRoute`, which every OSWorld backbone class inherits (`OSWorldMiniMaxAgent`, `OSWorldQwenAgent`, `OSWorldDeepSeekAgent`); constructed only by `build_default`) as an executed-step timeline (`src/benchmarks/osworld/timeline.py:EpisodeTimeline`, written by the adapter, read by the reasoning agent) with the current screen last. Everything that varies with the model is one profile, `graphpc_agent.py:OSWorldModelProfile` (backbone class, action agent class, translator, action vocabulary, deciding suffix, environment brief, call attempts, grounding temperature, and the per-model regime fields: typing note, image retention, give-up token, output cap, repair shape, request timeout, give-up block line, ask channel, lost-episode classifier, failed-capture handling, and two method fields of the DeepSeek profile alone: the requirement ledger with its completion-gate rule, and the step-budget line), which `build_default(backbone_type=...)` looks up in `OSWORLD_MODEL_PROFILES`; two per-model settings sit beside it as backbone-keyed tables in `graphpc_agent.py`, the step budget (`OSWORLD_MAX_STEPS_BY_BACKBONE`, which runner delta 15 imports) and the Qwen arm's stuck-stop (`OSWORLD_STUCK_STOP_BY_BACKBONE`); what a backbone must provide is `reasoning.py:OSWorldBackbone`. Three profiles are registered: MiniMax-M3, Qwen `qwen3.7-plus` (`src/benchmarks/osworld/profile_qwen.py`, whose control is upstream's `osworld-v2.1` Qwen agent) and DeepSeek `deepseek-flash` (`src/benchmarks/osworld/profile_deepseek.py`, whose control is upstream's `mm_agents/m3` carrying the native-tool deltas `scripts/osworld/deepseek_control/` declares — the model ignores M3's text tool grammar); adding a model is described in `.claude/docs/benchmarks/osworld_v2.md` ("Adding a model to the OSWorld adapter").

**Sync workflow** (important, easy to get wrong). GraphPC is developed in this repo, but **runs inside the OSWorld-V2 checkout on the run host**. A sync script copies a vendored snapshot into `<OSWorld-V2>/mm_agents/graphpc/` plus the vendored V2 runners (`scripts/python/run_multienv_graphpc.py`, `scripts/python/run_multienv_m3_baseline.py`, and the Qwen cell's control runner `scripts/python/run_multienv_qwen_baseline.py` with its agent under `mm_agents/qwen/`), the DeepSeek cell's control (`mm_agents/deepseek/`, upstream's `mm_agents/m3` with the native-tool deltas its files declare, and its runner `scripts/python/run_multienv_deepseek_baseline.py`), the launch scripts (`scripts/bash/run_multienv_graphpc.sh`, `scripts/bash/run_multienv_m3_baseline.sh`, `scripts/bash/run_multienv_qwen_baseline.sh`, `scripts/bash/run_multienv_deepseek_baseline.sh`) with their shared cell config, the host gates (`scripts/python/verify_minimax_vision.py`, `scripts/python/verify_qwen_vision.py`, `scripts/python/verify_deepseek_vision.py`), the runner-delta check (`scripts/python/verify_runner_deltas.py`), the launch preflight (`scripts/python/launch_preflight.py`) with the run-order solver it imports (`scripts/python/solve_subset_run_order.py`) — `SYNCED_TARGETS` in `src/benchmarks/osworld/sync.py` is the exact list; every sync also writes `mm_agents/graphpc/vendored_manifest.json`, which the host preflight checks the snapshot against:

```bash
python scripts/osworld/sync_osworld_graphpc.py                  # copy current snapshot
python scripts/osworld/sync_osworld_graphpc.py --dry_run        # preview
python scripts/osworld/sync_osworld_graphpc.py --clean          # remove old snapshot first
python scripts/osworld/sync_osworld_graphpc.py --clean_only     # remove and stop
```

The file list lives in `src/benchmarks/osworld/sync.py:SYNC_SOURCE_TARGETS`. **When adding a new GraphPC source file that needs to run inside the benchmark checkout, add it to that tuple** — otherwise it won't be copied and the runner will fail with import errors at benchmark time. The default sync target is the local OSWorld-V2 staging clone at `<repo>/../OSWorld-V2` (`--osworld_root` overrides it), from which the snapshot is rsynced to the host; sync from a git clone of this repository at a clean commit, since the snapshot records that commit and the host preflight checks it (`.claude/docs/benchmarks/osworld_v2.md`).

The OSWorld-V2 checkout is **read-only**: read it for reference, never edit it. If a change appears needed in its code, wrap or adapt it locally instead. Our runner entry-point inside the checkout is `scripts/python/run_multienv_graphpc.py` — a copy of OSWorld-V2's `run_multienv_m3.py` with the deltas listed in its header (the header is the authority; its numbering has gaps) (source: `scripts/osworld/run_multienv_graphpc.py`); it accepts `--backbone {minimax,qwen,deepseek}` plus the upstream flags, refuses a launch without `--retrieval_off` (every GraphPC run on the subset runs retrieval off), and drives upstream's `lib_run_single.py` unmodified. The baseline arm has its own vendored entry-point `scripts/python/run_multienv_m3_baseline.py` (source: `scripts/osworld/run_multienv_m3_baseline.py`) — upstream's `run_multienv_m3.py` verbatim plus the one delta listed in its header, carried byte-identically by the GraphPC runner: `--volume_size` for the AMI disk defect (ported from upstream's own `run_multienv_gpt_response_api.py`; it costs an in-guest `apt-get` install per episode); `scripts/osworld/verify_runner_deltas.py` holds the vendored copy against upstream's file — see `.claude/docs/benchmarks/osworld_v2.md`. The paired partial-score comparison is computed by `scripts/osworld/tools/score_paired.py`.

**End-to-end execution.** For AWS setup (one-time human steps), per-run agent commands and scoring, see `.claude/docs/benchmarks/osworld_v2.md`. The data files scripts read by path live in the tracked `data/`: under `data/osworld_v2/` the subset manifest (`prereg_task_list.json`), the pre-registration (`subset_prereg.json`) and the task-class extract, all three copied to the host by the runbook; under `data/webarena/` the Lite-165 site plan `scripts/webarena/run_graphpc_parallel.py` reads by default (`lite165_site_partitions.json`), the full-812 lane partitions `scripts/webarena/run_lanes.py` takes (`full812_site_partitions/<lane>/task_ids.json`) and example lane-driver inputs (`run_lanes_argv.example.json`, `run_lanes_tripwires.example.json`).

### WebArena

**Adapter.** `src/benchmarks/webarena/graphpc_agent.py:WebArenaGraphPCAgent` is the translation layer — **inward** it mirrors `OSWorldGraphPCAgent` and calls only the stable inner boundary `decide_next_action`/`update_runtime` (the decoupling invariant: check it with a strict-stub orchestrator whose `__getattribute__` throws on any non-boundary access); **outward** it conforms to WebArena's `Agent` interface (`next_action` / `reset`) and only uses the released factories (`create_id_based_action()` / `create_stop_action()` / `create_none_action()`). All five backbones supported (`--backbone {claude,gpt,deepseek,minimax,qwen}`), via the shared browser-action agent `src/benchmarks/webarena/action_agent.py`. WebArena imports are **lazy**, centralized in `src/benchmarks/webarena/_webarena_imports.py`, so the package loads cleanly without a WebArena checkout.

**No sync workflow.** WebArena is self-hosted; the runner constructs our agent directly and runs against a local/cloned WebArena checkout — no vendored snapshot, no external-repo edits.

WebArena itself is **read-only** (external-repo policy). The runner does not touch `construct_agent()` or any other WebArena file.

**End-to-end execution.** `scripts/webarena/run_graphpc.py` is the vendored runner; it reuses WebArena's `run.py` loop pattern (reset → loop → done-check → evaluator → finalize) without copying their file. For deployment prerequisites (clone WebArena, install Playwright, provision the local embedding model for the retrieval text arm), CLI flags, dry-run smoke, subset semantics, and baseline invocation, see `.claude/docs/benchmarks/webarena.md`. The baseline run is WebArena's bundled control, launched through `scripts/webarena/run_baseline.py`, which calls `run.py`'s own `prepare` / `construct_agent` / `test` unchanged and adds, around them, the openai-2.x bridge, the non-contiguous task list, the reasoning-model compatibility its docstring lists and the environment hardening (`src/benchmarks/webarena/_env_hardening.py`) both WebArena runners install.

### Conda environment

The repo runs in a dedicated Conda environment named `GraphPC`, built from `requirements.txt` (prerequisites outside pip: README "Setup"). Activate it first — the system `python3` may be missing required dependencies and may fail at import time:

```bash
conda create -n GraphPC python=3.10 && conda activate GraphPC && pip install -r requirements.txt   # once
conda activate GraphPC
```

`GraphPC` carries the retrieval stack (`sentence-transformers`), which retrieval-on runs need. WebArena launches and their preflights run in the `WebArena` env, which holds the WebArena checkout; its recipe installs this `requirements.txt` over WebArena's own (`.claude/docs/benchmarks/webarena.md`).

### Tests

There is no build system, no `pyproject.toml`, no test runner config; tests are plain `unittest` under `tests/`. The repository ships none: write the ones your work needs.

```bash
python -m unittest discover -s tests                                          # all tests/
python -m unittest tests.<module>                                             # a single test module
python -m unittest tests.<module>.ClassName.test_method_name                  # a single test method
```

**All agent-written tests MUST go under `tests/`.** This includes scratch tests, smoke tests, repro scripts, ad-hoc verification scripts, and any other test-like Python file. Do not create tests at the repo root, inside `src/`, alongside the code under test, or in new top-level directories (no `local_tests/`, `scratch/`, `tmp_tests/`, etc.).

- `tests/` is gitignored: tests stay local and are never committed.
- A `unittest.TestCase` module is named `test_*.py` so `unittest discover` picks it up.
- **`scripts/` holds only what running an experiment needs**: runners, launch and sync scripts, the scorer, and the checks whose answer depends on the run's environment — the host, the provider, the synced snapshot — or that a launch sequence requires (the preflights, the runner-delta check, the vision gates). A check whose answer depends only on the repository's code is a test and belongs in `tests/`. Script names carry no internal code names (no version or bundle labels). The layout is in `scripts/README.md`.
- **A file under `tests/` that is NOT a `unittest.TestCase` must not be named `test_*.py`.** `discover` imports every match, so a standalone script named that way runs its module body at collection time and, if it asserts something the current interpreter lacks (the sentence embedder, a benchmark checkout), reports as a `unittest.loader._FailedTest` ERROR — indistinguishable from a broken test. Name those `conf_check_*.py` (or anything outside `test*`) and run them directly.

---

## Conventions and principles

### Working principles

- Think before coding: surface assumptions, tradeoffs, and confusion early
- Prefer minimal diffs over large refactors
- Preserve existing architecture unless explicitly required
- **Copy first, adapt minimally (benchmark-side code).** Runners, episode drivers, baseline agents and scoring glue are copied from the benchmark repository's own code (or from our existing benchmark code) and changed as little as possible — keep upstream names, CLI flags, result layout and control flow; put GraphPC's delta behind the inner boundary. The test of compliance: a vendored file dropped into the pinned checkout on the run host runs as-is. In force for OSWorld 2.0 (`.claude/docs/benchmarks/osworld_v2.md`, Migration rule).
- Prefer simple and concise implementations
- Do not over-simplify if it harms clarity, debuggability, or extensibility
- Do not add speculative flexibility, configurability, or abstractions
- Prefer designs that are easy to inspect and debug during experimentation

### Research iteration discipline

When refining or running experiments on the system, keep work at the **method level** and **root-fix** rather than surface-patch. These invariants govern every iteration / experiment session.

- **Refine at the method level — not backbone / benchmark micro-tuning.** Threshold knobs, prompt tweaks, model-routing tricks, and format-specific patches are the recurring failure mode for this kind of system. If a refine cycle keeps landing on niche micro-tuning, the root cause has not been found yet — step back and diagnose, do not tune.
- **Root-fix, not surface-patch.** A finding (test failure, anomalous score, gate result) is the *symptom*. Locate the underlying scoring artifact / contract gap / data-shape issue and fix it there; do not reweight, re-threshold, or otherwise paper over a real defect.
- **Delete, do not disable.** Removed mechanisms must be removed at the source — no flags, no dormant code, no vestigial fields. Dormant scaffolding is the dominant masking failure mode for staged pipelines; the design's removal mandate (`.claude/docs/design.md` Implementation invariants) keeps it gone. Re-enabling a removed mechanism requires a validated-benefit case under `design.md` decomposition tax.
- **Measurement integrity.** Telemetry must distinguish primary from degraded paths. Any gate / measurement / Δ-claim is invalid if produced by a degraded mode (e.g. an ablation flag left on, a mock environment, or a retrieval measurement whose telemetry does not show the real embedder). Verify the active configuration in resolution-level telemetry before trusting the number.
- **Escalate spec ambiguity — do not silently improvise.** If `design.md` or the integration docs are unclear or seem wrong, raise it; do not invent a default and ship.
- **Every added stage / call must earn its keep** (`design.md` decomposition tax). The current design is deliberately minimal — `graph-reasoning + action` — for exactly this reason; new stages are gated by measurement, not faith.

A new team picking up an iteration / experiment work-stream should read `.claude/docs/experiments.md` first (the benchmark-agnostic protocol) — it points to the rest of the docs in priority order and includes the anti-patterns table.

### Surgical changes

- Touch only files and lines needed for the requested task
- Do not refactor, reformat, or "improve" adjacent code while making a change
- Match existing style, even if a different style would be preferable
- If unrelated dead code or issues are noticed, mention them instead of changing them
- Remove only unused code introduced by the current change
- A commit labelled `docs` must not touch code files; split it, or say in the subject that code files are touched

### Naming conventions

- CLI arguments MUST use `snake_case` (not `kebab-case`) — applies to all `scripts/` entry points
- Prefer clear and explicit naming
- Avoid abbreviations unless they are widely accepted (e.g. API, URL)

### Schema modification policy

The schema is considered stable. "Stable" means the default is not to change it, and cosmetic or cleanup-only edits are off the table — it does not mean strictly frozen. Edits are allowed when keeping the current schema would force materially worse code (e.g. clotted adaptation layers written purely to fit the existing shape).

Default:
- Avoid modifying existing schema
- New backbones must adapt to the existing schema, not the reverse

Preferred:
- Reuse existing schema where possible
- Extend schema in an additive way when needed

Avoid:
- Breaking changes
- Repurposing existing fields

Allowed:
- Introduce new fields or structures if needed for clarity or integration (e.g. OSWorld 2.0)
- Non-additive edits (renames, structural reshapes, breaking changes) ONLY when keeping the current schema would force materially worse code

If a change is needed:
1. Explain why current schema is insufficient
2. Propose minimal change
3. Analyze impact (runtime / interfaces / downstream)
4. Avoid unnecessary redesign

### External repository policy

External repositories (e.g. OSWorld-V2, WebArena) may be used as references, but are treated as **read-only** outside the GraphPC deployment subtree (see exception below).

Rules:
- DO NOT modify files in external repositories
- DO NOT create patches targeting external repositories
- DO NOT refactor external codebases

Allowed:
- Read external code
- Reuse patterns
- Copy/adapt logic into this repository — and for benchmark-side code this is the DEFAULT, not an option: copy the harness's own runner / driver / agent file and make the minimal edit (see "Copy first, adapt minimally" above)

All implementation work must remain within this repository. If a change seems required in external code, adapt or wrap it locally instead.

**Exception — GraphPC deployment subtree inside the benchmark checkout.** The OSWorld-V2 checkout on the run host carries a vendored snapshot of GraphPC under `mm_agents/graphpc/`, the OSWorld Qwen cell's control under `mm_agents/qwen/` with its runner `scripts/python/run_multienv_qwen_baseline.py` (upstream's `osworld-v2.1` agent and runner, which the pinned tag lacks, copied from `scripts/osworld/qwen_control/` and `scripts/osworld/run_multienv_qwen_baseline.py` with the deltas their headers declare), the OSWorld DeepSeek cell's control under `mm_agents/deepseek/` with its runner `scripts/python/run_multienv_deepseek_baseline.py` (upstream's `mm_agents/m3` and `run_multienv_m3.py` at the pinned tag, copied from `scripts/osworld/deepseek_control/` and `scripts/osworld/run_multienv_deepseek_baseline.py` with the deltas their headers declare), plus the runner, launch and check scripts under the checkout's `scripts/python/` and `scripts/bash/` (the entries listed in `src/benchmarks/osworld/sync.py:SYNCED_TARGETS`). These paths are written by `scripts/osworld/sync_osworld_graphpc.py` into a local staging clone and shipped to the host over SSH (`rsync`); they are untracked additions kept out of git via the host checkout's `.git/info/exclude` and are never committed or pushed anywhere — there is no fork. They are NOT subject to the read-only rule. The rest of the checkout (everything not listed in `SYNCED_TARGETS`) remains strictly read-only; the host checkout stays pinned at the release tag.

### Task workflow

For each task:

1. Understand the task and relevant modules
2. Provide a short plan
3. Define success criteria and verification steps for non-trivial work
4. Implement with minimal changes
5. Validate if applicable, and loop until verification passes when feasible

### Handling ambiguity

If something is ambiguous:

1. Make the smallest reasonable assumption consistent with the current repo and docs
2. Explicitly state the assumption and any meaningful tradeoffs
3. If multiple interpretations exist, present them instead of silently choosing
4. Push back when a simpler or safer approach exists

Prefer asking for clarification over making large or irreversible decisions — especially when architecture, schema, or interfaces could be affected.

### Task scoping

When project docs mention multiple current or future tasks, follow the task explicitly requested in the current user prompt. Do not proactively continue to later roadmap items unless explicitly asked.

### Documentation updates

After completing a task, update relevant project docs when needed so they remain consistent with the current implementation and task status.

Keep documentation updates scoped:
- Update only the docs affected by the completed task
- Do not rewrite unrelated sections
- Keep project docs concise and current

### What shipped material may cite

Code and the state docs (`README.md`, this file, `design.md`, `experiments.md` and the benchmark runbooks) never cite run outputs (`runs/`) or other untracked material; where a line's reason sits there, the line states the reason itself (a task id or a run's configuration is a fact and stays). Data that code reads at run time lives in the tracked `data/`. A run's output location written as a placeholder (`runs/<run id>/`) is not a citation. `project_state.md` may point at your own runs.

---

## Operational notes

### Secret handling

Never commit secrets.

- Local secrets live in `.env` (gitignored). Loaded automatically via `python-dotenv` by the runners and checks under `scripts/` (e.g. `scripts/osworld/run_multienv_graphpc.py`, `scripts/webarena/run_graphpc.py`).
- AWS credentials live in `.env` alongside other secrets (gitignored). `aws configure` (writes to `~/.aws/credentials`, outside the repo) is also supported if you prefer to keep them out of a file consumed by `load_dotenv()`.
- Run-host credentials (the OSWorld host's `.env`, Google OAuth files) live only on that host and never enter a tracked path.

---

## Output expectations

### Code explanation requirement

All non-trivial code changes must be accompanied by concise explanations.

Explain:
- What the code does
- Why this design is used
- How it fits into the overall agent system
- How it interacts with other components in the system, and integration with other repos when relevant

Focus on system role and design rationale rather than line-by-line narration.

Avoid:
- Pedantic walkthroughs
- Restating obvious syntax
- Overly long explanations without architectural insight

Explanations should connect to the internal multi-agent workflow when relevant.

### Output format

When completing a task, include:

- Assumptions
- Changes made
- Affected modules
- Concise code explanation
- Risks / follow-ups

Be concise and technical.
