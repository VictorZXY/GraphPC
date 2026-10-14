# Experiment Protocol

The protocol for full-benchmark × model comparisons of GraphPC against the same-model vanilla agent.

Read this before running anything. Pair with `.claude/docs/benchmarks/README.md` (index of the per-benchmark folders) for deployment/runner mechanics, and record your runs' status in `project_state.md`.

---

## Mission

Produce paper-grade paired comparisons: **GraphPC vs the same-model vanilla agent (WebArena's `PromptAgent`: obs → one LLM call → action; on OSWorld 2.0 the benchmark's own agent for the model), identical harness**, across benchmarks and foundation models.

**Model selection.**

- **One natively multimodal model per foundation-model family; both arms of a cell always run the same model.** Fairness lives inside each pair: numbers are not compared across models or tiers.
- **The models:** DeepSeek `deepseek-flash` (V4.1-Flash), MiniMax-M3, Qwen `qwen3.7-plus` (reached over OpenRouter as `qwen/qwen3.7-plus`), and, as the closed-source pair, GPT `gpt-6-sol` and Claude Sonnet 5.
- **Benchmarks.** The open-source models run full WebArena-812 (a cold run plus a pre-registered warm round); the closed-source models run WebArena-Lite-165, cold, with `--retrieval_off`, no warm round. **OSWorld 2.0 is the second benchmark** (`xlang-ai/OSWorld-V2`, 108 long-horizon tasks, screenshot-primary, 500-step budget, binary-completion + partial-score metrics). It runs natively multimodal models only, because its action path grounds from screenshots and a text-only family would need a bespoke text-grounding arm plus a non-standard text-only control.
- **Opening a cell is a paid decision; a human operator makes it.**

**The comparison matrix** (the cells this protocol defines):

| | DeepSeek `deepseek-flash` | MiniMax-M3 | Qwen `qwen3.7-plus` | GPT `gpt-6-sol` | Claude Sonnet 5 |
|---|---|---|---|---|---|
| **WebArena-812: cold run + pre-registered warm round** | cell | cell | cell | — | — |
| **WebArena-Lite-165: cold, `--retrieval_off`** | — | — | — | cell | cell |
| **OSWorld 2.0: the pre-registered 28-task subset** | cell | cell | cell | — | — |

The warm cells are not one comparable column: each cell's warm subset is derived from its own cold run, and the pre-registered endpoint is the loser-flip stratum, not the subset total.

**OSWorld 2.0 row — what the cells are.** The harness is `xlang-ai/OSWorld-V2` at a pinned release (the tag pins the harness code; the task content ships from a gated dataset), screenshot-primary; deployment: `.claude/docs/benchmarks/osworld_v2.md`. Each cell is a same-model pair on the pre-registered 28-task subset, run by both arms in the same pre-registered order. A cost projection that assumes every episode runs to the step cap is an upper bound.

**Pre-register what a cell can and cannot measure, before it runs** — falsifiable predictions, indicators, thresholds — on the principle that an unfalsifiable disclosure is worth little. The short form: retrieval is off on every GraphPC run on the subset, so the cells measure whether the reasoning scaffolding ports from an accessibility-tree web benchmark to a screenshot-primary desktop one (`design.md` §3, "What the structure does for reasoning"). A null retrieval result is acceptable, up to and including reporting the cell as an ablation of the scaffolding. State this up front rather than explaining it afterwards.

**Standing engineering.** Each is a rule that binds every cell:

1. **N/A-answer adapter parity.** The stock baseline prompt teaches WebArena's literal-"N/A" infeasibility dialect; ours does not. Infeasibility conclusions are normalized to "N/A" in the ADAPTER/translator with full audit telemetry (`_na_normalized_events`), and NEVER in contract text. The design is precision-first: a false-positive normalization would destroy a win. Acceptance smoke on the GraphPC arm: tasks 225 and 313 (reference answer `fuzzy_match: N/A`) must emit the literal N/A, whether the model wrote it or the adapter normalized it (`_na_normalized_events`), and the feasible task 190 must record no normalization event.
2. **The candidate-encode cache is the STANDING encode path** — required at full-store sizes, since the store would otherwise re-encode every candidate each step. Its ~1e-3 batch-composition shift against a batched path can never be neutral, so it is the encode path of every scored run.
3. **Support-store behaviour holds at scale** — tolerant read, the launcher's pre-spawn sweep and add-time dedup must hold at full-store sizes, and the encode cache keeps per-step retrieval latency low there.

**Open research workstream:** retrieval selectivity — whether injection harms tasks the agent would win alone is not separable without a no-injection control (`design.md` §5, "Open: selectivity"). Changes here follow the standard design→review→smoke cycle; scored comparisons use fixed configurations per arm.

**Run protocol per benchmark×backbone cell:**

1. Bare-baseline arm: full-budget semantics, run once CLEAN, LOCK (policy below), never re-run.
2. GraphPC arm: cold run (fresh store) is the primary comparison; the warm round is optional and pre-registered.
3. Both arms: identical harness (evaluator, judge model, reset discipline, step cap — one declared exception: on the OSWorld Qwen cell the GraphPC arm runs 800 single-action steps against the control's 500, which can batch several actions each; `osworld_v2.md`, the Qwen cell), pinned worktrees, active-configuration verification in telemetry before trusting numbers.
4. **Full-WebArena task ordering: Lite-first within one continuous run — never two stitched passes.** WebArena-Lite's 165 ids are a subset of the full 812 (same config files). Running Lite and the 647-complement as separate passes and merging them is not a scored number: 143/190 intent-template families straddle the Lite boundary (675 tasks), so a split severs the experience-store trajectory mid-family — a first-order effect, since same-family evidence is what retrieval transfers and a fresh-store restart pays the cold-store penalty again — on top of two site-reset epochs (411/812 tasks are site-state-scored) and cross-week provider drift. Instead: order each site-disjoint lane's configs Lite-members-first (by id) then the rest (by id); multi-site Lite members (10) lead the cross-site phase. The Lite subtotal reads out as an early checkpoint ~1/5 into the run at zero extra cost. Ordering is a fixed harness choice: BOTH arms, same order, recorded in the run README (the run's `runs/<run id>/README`, gitignored, where its configuration and launch record go). The shipped lane partitions (`data/webarena/full812_site_partitions/`) follow this rule.

**Verdict methodology**:
paired per-task comparison (win/loss flips, per-site and per-evaluator-class tables, `design.md` §7) is PRIMARY; aggregate deltas are secondary (single-run churn is several points on a 165-task set, proportionally tighter but nonzero on 812). Multi-round averaging / pass@k only when declared in advance, naming the arms it applies to; a best-of-k arm's tokens are priced at k attempts per task, with the actual spend beside. **No task leaves the comparison.** An unscored task (an `infra_error` / `eval_error` row, or no row) is re-run rather than counted as a loss; a task still unscored after its allowed re-runs counts 0.0 for that arm, and the paired comparison is taken over all of the cell's tasks. Report leads / closable-lags / structural-lags per the three-segment framing (tasks GraphPC wins; control wins with a known or plausible closing path; control wins that look structurally hard to close, or are variance).

**Deliverables per cell:** locked baseline summary; GraphPC run verdict; paired-comparison table + per-evaluator-class analysis; updated three-segment table.

---

## Reading order

1. `design.md` — the method as it ships.
2. This file — the experiment protocol.
3. `.claude/docs/benchmarks/<name>.md` (index: `.claude/docs/benchmarks/README.md`) — the benchmark's host setup, deployment and runner mechanics.
4. `project_state.md` — your own progress record.

---

## Hard constraints — do NOT break

- **`common/schema.py`** is stable: extend additively (`content` / `metadata`); a non-additive edit only under AGENTS.md's schema-modification policy.
- **Stable inner boundary** — `decide_next_action` / `update_runtime` signatures + the `Observation` / `ActionBatch` / `GraphPCDecision` types. The benchmark adapters depend on these; check it with a strict-stub orchestrator whose `__getattribute__` throws on any non-boundary access.
- **External benchmark repositories are read-only.** No edits to WebArena's `run.py` / `construct_agent()` / anything under `../WebArena/`; same for the OSWorld-V2 checkout (the GraphPC deployment subtree is the only exception, written by the sync script and rsynced to the host — see `.claude/docs/benchmarks/osworld_v2.md`).
- **Every benchmark-motivated change is argued into a class and scoped accordingly** (`design.md` Implementation invariants). Class one is neutral *with respect to the comparison* and is validated on the reference benchmark before it is trusted; class two cannot execute anywhere else. A change that alters what our arm gets relative to what the control gets is not neutral whatever its motivation — and a within-arm gate cannot detect it, because the control never appears in that comparison.
- **What a launch or a run needs lives under `scripts/`** — preflights, host and provider gates — as loud-fail runnables; a check that depends only on the code is a test (`tests/`).
- **Delete, do not disable** (`design.md` Implementation invariants). Removed mechanisms stay removed — no flags, no dormant code, no vestigial fields. Adding a new mechanism that resembles a previously-removed one requires a validated-benefit case under `design.md` decomposition tax.
- **Locked-baseline policy: lock only on a clean run.** A baseline locks only when its run is **clean** — no runtime-error contamination. A run is contaminated if any of: ≥3 task `retry-exhausted`, ≥10% steps killed by timeout / env crash, any task required docker / browser restart mid-run. The triggers are diagnose-first, not stop-first: reaching one starts a diagnosis of whether a defect corrupts the results, and only such a defect makes the run contaminated. A contaminated baseline run yields a **provisional** lock and is re-run with the same backbone + config until clean. Once a clean run locks, the locked numbers stand for the project's lifetime — do **not** re-run the baseline to chase shifts in API behavior or pricing, as that silently changes the control. Per-benchmark baseline specifics (e.g. OSWorld's lock rule for fault-lost episodes) in `.claude/docs/benchmarks/<name>.md`.

## Pre-flight checklist (before reporting any gate verdict)

Run through this before declaring a gate pass / fail:

- [ ] **Telemetry says the primary path was active** — e.g. any retrieval-related measurement requires the real `Qwen/Qwen3-Embedding-0.6B` embedder in the run's telemetry (an unavailable embedder raises and the episode is re-run; there is no fallback ranker); any A/B requires both arms on identical gate/config settings. Verify in the run's telemetry, not by assumption.
- [ ] **Mechanism is verified alive at the verdict level**, not just the firing-count level — confirmed / refuted / inapplicable counts, retrieved-and-used vs retrieved-and-ignored counts, etc. (`design.md` Design principles #6 + #9).
- [ ] **Locked baseline is unchanged** — no accidental re-running of the per-benchmark baseline; the locked numbers stand.
- [ ] **No degraded-mode runs are being reported as primary** — ablation runs go in the ablation column, not the primary verdict.
- [ ] **Schema contract intact** — no unargued change to `common/schema.py`; the tests you wrote under `tests/` still pass.

---

## Anti-patterns — concrete traps

Each row is a known failure mode of this kind of system; the second column is the discipline that catches it.

| Trap | Do this instead |
|---|---|
| Report a retrieval measurement without checking which embedder ranked | Confirm the embedder (`Qwen/Qwen3-Embedding-0.6B`) and the retrieval mode in telemetry; a run with retrieval off is an ablation, never the retrieval measurement. |
| Tweak `--completion_gate_window` to "fix" a negative Δ | Threshold knobs do **not** fix mechanism defects. If a gate fires wrongly, isolate the artifact (a score that equals the threshold by construction, e.g. on an empty query, is an artifact, not a threshold problem). Tune thresholds only on effect-class-reliability-validated reliability data. |
| Reweight a similarity to make the test pass | Find the underlying artifact and fix it (a vacuous comparison scoring 1.0 is a defect: make vacuous comparisons neutral). |
| Re-enable a removed mechanism because "the new run misses safety" | A removed mechanism was removed for cause. If a real safety gap is found, design the addition against the 9 principles + propose with a validated-benefit case; do not resurrect the old code. |
| Add a new stage / agent / mode "just in case" | Decomposition tax. Every added call must clear validated-benefit > failure-surface cost (`design.md` decomposition tax). |
| Trust a Δ because "the gate fired" without checking the verdict-level telemetry | Mechanism-firing ≠ mechanism-alive (`design.md` verdict-level liveness). Verify the resolved verdict (confirmed / refuted / inapplicable) before believing the number. |
| Silently improvise when the spec is unclear | Escalate ("Escalation" below). |
| Refine by drifting into backbone-specific prompt tuning or benchmark-format patches | Stay at the method level (AGENTS.md "Research iteration discipline"). If refine rounds keep landing on niche micro-tuning, the root cause has not been found yet. |
| Half-fixing a mechanism and disabling the broken parts with flags | Delete, do not disable. Dormant flagged code is what masks the dead-core failure mode. |
| Editing external benchmark core (WebArena / OSWorld-V2 files) | Read-only. Use the vendored / shim patterns documented in `.claude/docs/benchmarks/<name>.md`. |

---

## Escalation

If `design.md` or the integration docs are ambiguous, or a finding touches the schema / decoupling invariant / 9 design principles, **escalate — do not silently improvise**.
