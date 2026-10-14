# GraphPC Design

> This file states the method as it ships. Per-benchmark deployment and runs: `.claude/docs/benchmarks/`.

---

## 0. Design intent

GraphPC is a training-free harness for computer-use agents built on the **procedure-causal graph**: a record of
the episode that captures progress as the effects of actions. It wraps an unmodified model. Externally it is
one agent; internally a fixed two-call pipeline per step, coordinated by an explicit orchestrator (§1).

**What the graph records** (§3). Three node types — stages, actions (one per step) and outcomes (one per labelled
step) — and three edge types: *precedes* orders consecutive stages and consecutive actions within a stage; *groups*
ties each action to its stage; *causes* ties each action to its outcome and carries the pair (predicted effect,
recorded effect) from a closed ten-class vocabulary (§4). The task anchor is the instruction, not a node. Recovery
is a pattern readable on the graph — an action whose recorded outcome is `error` or `no_change`, followed by the
corrective action — not an edge type.

**Predict, act, test, label, record.** The deciding call predicts the effect of the action it decides; after
execution a model-free test reports whether the observation changed (a raw-text diff on WebArena, a
pointer-masked screenshot diff on OSWorld 2.0; §6); the next deciding call labels the effect; the recorded label is
the model's, except that a progress label — or a missing or off-vocabulary one — on an unchanged step becomes
`no_change`. A confirmed progress effect is therefore only ever recorded on a step the change test saw change, and
a prediction of progress on a step it saw unchanged always fails.

**Three roles of one record.**
1. **Progress verification.** The completion gate (§6) admits an answer-free claim only if the record holds a
   confirmed non-navigation effect, and an answer-bearing claim only if the answer is grounded in recent
   observations: a lookup, no model call.
2. **Context management.** The model rewrites `graph_summary` at every step; with the plan and the recent labelled
   actions as fields of their own, it carries the episode (§3; on OSWorld 2.0 an executed-step timeline, §2).
3. **Reusable experience.** A successful episode is stored as a recipe — its labelled action log, the labels
   re-checked at storage (§6) — and retrieved by text similarity into the deciding call, never into grounding (§5).

**How it ships.** The graph is kept as a record — the plan, the per-step labelled actions and the summary — and is
not materialized as node/edge JSON on every step (§3 "How the code carries the graph"); no mechanism reads
its connectivity, and retrieval ranks on text alone (§5, §11).

---

## Design principles (review checklist)

Portable, backbone- and benchmark-agnostic invariants, derived from observed failure modes of multi-stage
agent designs. Every change is reviewed against them.

1. **Every stage and every added call must earn its keep.** Decomposing one competent call into stages has
   a certain cost (new failure surfaces, lost holistic context, lost one-shot adaptivity); a stage is
   justified only when its *validated* benefit exceeds it. Gate additions behind measurement
   ("decomposition tax"; Cemri et al., MAST, NeurIPS'25).
2. **Every mechanism's output must be consumed by behavior.** A stage whose output changes no downstream
   decision is pure tax.
3. **A field a mechanism depends on must be required in every mode that should emit it.** A mode-specific
   contract that drops the field disables the dependent mechanism invisibly.
4. **Isolate the core; do not stack unvalidated mechanisms alongside it.** Validate one mechanism in
   isolation before composing.
5. **Recovery and give-up gate on positive-progress evidence, never on weak proxies.**
6. **Telemetry is resolution-level, not firing-counts.** Record the resolved verdict (confirmed / refuted /
   inapplicable; retrieved-and-injected vs gated-out), not that a mechanism ran.
7. **Verification rests on environment signal, not self-critique; costly actions gate on accumulated
   evidence.** Intrinsic self-correction without an external check degrades performance (Huang et al.,
   ICLR'24).
8. **Gate retrieved experience on a relevance threshold before injection.** Weakly matched context can drop
   a model below no-retrieval (Yoran et al., ICLR'24; Fang et al., ACL'24).
9. **Never infer a mechanism is alive from a metric moving or the mechanism firing.** Confirm liveness at
   the verdict level. A design whose value core is silently inert looks like one whose core works.

---

## 1. Workflow — graph-reasoning + action

**Single-pass (default) — 2 model calls per step:**

```
retrieval [deterministic] → graph-reasoning "decide" [model call] → action [model call]
```

- **retrieval** ranks the support set deterministically (§5) and returns at most 2 exemplars.
- **graph-reasoning** is one model call whose output contract carries, together: the prior action's
  `actual_effect_class`, an optional plan update, the rewritten `graph_summary`, and the decided
  `intended_actions`, each with a `predicted_effect_class`. Fusing reasoning and the graph update into one
  call is what keeps per-step calls at two; there is no separate graph call.
- **action** grounds the intended action onto the concrete interface (§1, "Action boundary").

**Two-pass (`--two_pass`) — 3 model calls per step; a backup design, not the default:**

```
graph-reasoning "initial" → retrieval [request or deterministic] → graph-reasoning "refine" → action
```

`initial` closes the prior effect, updates the summary and emits a `retrieval_request = {scope, query, k}`;
`refine` consumes retrieval and decides. This is the only shape in which the model controls retrieval
scope.

**Retrieval scope** in single-pass is chosen deterministically (`task` at step 0, `stage` afterwards,
`recovery` after a monitor-reported stall) and is recorded as telemetry. Ranking does not read it (§5).

### Action boundary

Action is a **thin grounding layer**, not a second reasoner. It receives the observation plus the
structured intended action(s) and emits concrete operations (WebArena id-based action strings;
OSWorld 2.0 pyautogui code from V2's M3 tool-call format with `[0,1000]` relative coordinates). It does
not see the reasoning trace or the retrieved experience, does not re-plan, and reports a target it
cannot ground back upstream as a step-level failure rather than improvising. Grounding is backbone- and benchmark-specific, which is why it stays its own stage: the
method layer never bakes in one backbone's grounding mechanism.

### Reasoning → action handoff

A small structured payload per intended action: `action_class`, a **semantic target by label/role, not
coordinates**, parameters (`text` / `keys` / `value`), an optional `intent`, and the
`predicted_effect_class`. Reasoning never emits coordinates; action retains final say over the concrete
operation. Additive on the open `content` dicts; no schema change.

**OSWorld 2.0: the invariant is enforced at the boundary.** The deciding call has no coordinate field but
writes coordinates into `target` / `intent` prose. Those coordinates are in no consistent unit and may be
copied from history, and grounding copies them.
So the grounding line is built from the intended action with every coordinate span stripped from `target` and
`intent` (`src/benchmarks/osworld/action_minimax.py:strip_grounding_coordinates`); payload fields are never
touched, and a field the strip would empty keeps its text. The timeline's step lines show the same stripped
target. The DeepSeek profile runs three
patterns of its own before this strip, and strips `value` when it holds only a bare coordinate pair or a start-and-end pair of points
(`DEEPSEEK_VALUE_POINT`; `src/benchmarks/osworld/action_deepseek.py:strip_deepseek_grounding_coordinates`);
MiniMax and Qwen keep the shared strip.

### Degrade contract

A step-level failure costs a step, never the episode: an unparseable **or wrong-shaped** reasoning
response (`require_decision_shape`, `reasoning/base.py` — a non-dict `plan` or a non-list
`intended_actions` is rejected at the boundary rather than escaping as a TypeError that would cost the
episode), an unparseable grounding response, or an action string the environment's parser rejects each
becomes a counted no-op step with the failure surfaced to the next reasoning call through the monitor
feedback channel. An
unparseable model output is re-sampled up to a benchmark-scoped budget (identical request bytes, no step
cost) before
the step degrades — the retry budget of OSWorld-V2's M3 agent: re-sampled on any grounding reply that
yields no executable action for our pipeline (no tool call, malformed call, terminal/ask name,
`[INFEASIBLE]`) — the grounding route has no termination authority; termination is decided on the
reasoning route and verified by the completion gate; a well-formed call with invalid arguments is a
consumed no-op step, as in V2's M3 agent. Residual difference from the baseline, to be named beside the
actions-per-logical-step asymmetry in any verdict: on a terminal-name reply the baseline emits DONE/FAIL
and ends its episode, while our grounding route re-samples it (up to two extra model calls, no step
cost). Transport errors are retried separately by the backbone (`src/agents/common/resample.py`). Telemetry counts three things per task, always present:
re-samples (extra samples beyond the first call), exhausted steps (all attempts failed, the degrade
ran), and parse failures counted per attempt (every unparseable sample, rescued or not) — the last is
the sum of the first two, not a duplicate of the exhausted count. On OSWorld 2.0 a deciding reply that
parses but carries no decision (no list-typed `intended_actions`, or the JSON Schema itself) is
re-sampled the same way — the control re-samples a zero-action parse — and counted apart
(`reasoning_decision_missing`, by kind), so there parse failures + decision-missing = re-samples +
exhausted steps. Every
other exception propagates loudly and is classified `agent_error` (counted 0.0) or `infra_error`
(never scored; re-run) in the run ledger.

**The OSWorld 2.0 lost-episode ledger (Qwen and DeepSeek profiles).** On the Qwen and DeepSeek profiles the
adapter classifies what would escape `predict`, each cell with one cause table for both arms:
- **Infrastructure is re-raised**, so the task has no result and is re-run, never scored. That covers
  transport after the retries; HTTP 5xx, 429, 402, 408 and 401; HTTP 404 (on this route, the routing pin or
  the slug going away); a transient in-band provider error; a failed screenshot capture; `MemoryError`; and a
  dead retrieval text arm.
- **Everything else ends the episode as `DONE`**: the environment is evaluated as it stands, and an
  `agent_error` record is written.

On Qwen the control carries the same table as a declared delta (D7), so the two arms classify the listed causes
alike; only the GraphPC arm retries an HTTP 400 or in-band error in no known form before calling it infra
(`.claude/docs/benchmarks/osworld_v2.md`, the Qwen cell). Keep the two tables in step by hand when either side is
edited. On DeepSeek the control keeps upstream's handling and the table discloses the
asymmetries. A step-level failure still costs only a step; the table applies only to
what escapes that. On the MiniMax profile every such exception leaves `predict`.

---

## 2. Observation granularity

Three consumers, three views of the same observation envelope:

- **The deciding reasoning call reads the RAW environment observation** — on WebArena the full
  accessibility tree the bare baseline reads, untruncated. Rule: **the reasoner must never see less than the
  baseline sees.** Reason: a compact projection drops decision-relevant rows and hierarchy, and multi-field
  forms become illegible.
  **WebArena:** the tree is followed by the control's own `URL: <alias form of the page URL>` line, with the
  control's joiner, at every step. Without it the deciding call sees less than the control, and episodes leave
  the benchmark's hosts. The line lives in `a11y_tree` only, so grounding, retrieval, completion-gate rule 1
  and stored entries do not carry it. The monitor's text diff does see a URL-only change.
- **The action call reads the relevance-selected projection** (`parse_axtree`, 200 nodes, element ids
  kept) — it needs the id space, not the whole page.
- **Retrieval embeds the compact projection text** (cut at 6000 chars) together with the
  instruction as the query (§5).
- **The monitor diffs the full raw observation** between steps to decide whether the page changed (raw
  text on WebArena; the screenshot on OSWorld 2.0 — §6).

**Portability caveat.** The raw WebArena tree is viewport-bounded and shorter than the projection text.
On benchmarks whose raw observation is not naturally bounded, the instantiation must be re-decided
(viewport-limiting, screenshot-primary, or a bounded projection) before any run; the principle that
carries is the parity floor, not "feed the raw dump".

**OSWorld 2.0 instantiation: screenshot-primary at the resolution V2's M3 agent sends, resolution parity with each model's control.** The deciding
call and the grounding call both read the raw 1920×1080 screenshot (JPEG q90, no resize — exactly what
V2's M3 agent sends); there is no accessibility channel (V2's M3 path never produces one). `observation_text` is empty on OSWorld, so retrieval
ranks on the instruction alone. The environment facts the baseline's system prompt carries (platform,
home directory, sudo password, task date) ride on the optional `environment_brief` content key, which
is kept out of the retrieval text so a constant prefix cannot shift the relevance gate, and are
rendered into the deciding call's system prompt, where the control keeps them.

**OSWorld 2.0: the instantiation is per model.** On the Qwen profile both GraphPC calls read the frame as
the Qwen control sends it: JPEG at quality 90 at 1920×1088, after the control's own `smart_resize` (the
control's delta D9; PNG frames can exceed OpenRouter's 30 MiB image cap). The deciding request's image count follows that control's folding: at most
20 frames, folded 10 at a time, no opening frame kept, and folded frames replaced by the control's
placeholder text.
The environment brief is the date only, the one fact that control's prompt states. The rule is the same
parity floor: the reasoner never sees less than the model's own control sees.

**OSWorld 2.0: the history view.** The CURRENT observation's view is as above. What the
deciding call sees of the history is an executed-step timeline, rendered benchmark-side
(`src/benchmarks/osworld/timeline.py`, `reasoning.py`) in one user message:
- step 0: the opening screen and the instruction;
- every executed step, as the environment received it: the action (a WAIT says why), the target as
  grounding received it, the monitor's verdict and the reconciled effect label;
- inside the control's image window: the step's reasoning (≤ 1,500 chars) and the frame after it,
  labelled with its step and action;
- outside the window: one line of ≤ 200 chars;
- then the state block (plan, graph_summary, facts ledger, retrieved exemplars) and the schema block;
- last, the CURRENT SCREEN, with nothing after it.

The image count is the control's own arithmetic. There is no item cap, and the 240-char
compaction of `recent_history` is not on this path: the timeline is built from the adapter's record
of what was executed, not from `recent_history`.

Reason: when retained frames are unlabelled and the current frame is named only by a caption at the top,
the model describes past screens as current.

The schema block sits at the end of the text, not in the system prompt; a reply that returns the schema instead of
a decision is re-sampled by the reply check.

**The facts ledger** (`facts` in the OSWorld deciding schema) is the benchmark's instance of evidence
attributes. It holds verbatim values the model read from a source (names, numbers, paths, rules) with
their source, accumulated for the episode outside the orchestrator and shown back at every step.
Entries are ≤ 300 chars, and the rendered ledger is ≤ 32,000 chars: past that, the earliest 16,000 and the
newest 16,000 of the source entries around a counted elision marker (`src/benchmarks/osworld/timeline.py`).
Reason: `graph_summary` is rewritten every step and drops such values, which the model then retypes as
invented ones.

**DeepSeek profile only:** the STATE block carries a requirement ledger (§6) and one
step-budget line; the control's agent never states the budget, a disclosed difference between the arms.

---

## 3. The procedure-causal graph — schema and its role in the reasoning call

The graph of an episode records progress as the effects of actions. It has three node types and three edge
types.

### Node types

| node | content | consumers |
|---|---|---|
| `stage` | one step of the plan: its label and its place in the stage order | reasoning (plan), summary |
| `action` | one per executed step: `action_class` (on WebArena a free-form string in the schema, in the agent's action names: click / type / press / scroll / goto / hover / tab ops / wait / done / fail; on OSWorld 2.0 the schema enumerates each profile's own normalised names — M3's on MiniMax and DeepSeek (`M3_ACTION_CLASS_VOCABULARY`, `graphpc_agent.py:_minimax_m3_profile`, `profile_deepseek.py:deepseek_model_profile`), the Qwen control's on Qwen (`QWEN_ACTION_CLASS_VOCABULARY`, `profile_qwen.py:qwen_model_profile`), each with `done` / `fail`), the semantic target and parameters, `intent`, and the `predicted_effect_class` | verification, completion gate, storage-time relabel, stored recipe, summary |
| `outcome` | one per labelled step: the recorded `effect_class` (§4) | verification, completion gate, storage-time relabel, stored recipe, summary |

### Edge types

| edge | from → to | role |
|---|---|---|
| `precedes` | stage → stage; action → action within a stage | order |
| `groups` | action → its stage | the stage an action serves |
| `causes` | action → outcome | **the causal core**: carries the pair (predicted effect, recorded effect) that verification closes |

- **The task anchor is the instruction** (with `plan.goal`), not a node.
- **Recovery is a pattern readable on the graph, not an edge type**: an action whose recorded outcome is `error`
  or `no_change` (a progress claim the change test overrode, §6), followed by the corrective action the next step
  decides. A failure is not a node either: it is an outcome recorded as `error` or `no_change`.
- `intent` is an attribute of `action`, not a node.
- **Interface-element and raw-observation detail are attributes, never nodes.** They attach to the relevant
  action or outcome (target label, parameters, observation refs) and never enter structural matching: they are
  high-cardinality and application-specific, and admitting them would degrade structure matching toward
  bag-of-tokens behaviour. Any structural matching (§9) keeps this rule and reads the stage / action / outcome
  skeleton only.
- Node identity within an episode: actions and outcomes are keyed on the step index, stages on their label.
  Cross-episode canonicalization is out of scope.

### How the code carries the graph

The deciding call does not emit node/edge JSON: no consumer reads topology, and emitting it would dominate the
call's output tokens. The graph is kept as a record that the call writes and the runtime already holds:

| graph element | where it lives at runtime / in the store |
|---|---|
| `stage` nodes, stage-level `precedes` | `plan.stages` (ordered), `plan.current_stage` (`PlanState`); the deciding call re-emits `plan` only when it changes |
| `action` nodes, action-level `precedes` | `recent_history[k].actions[]` in step order (`action_type`, parameters, `predicted_effect_class` in metadata), plus the step's `intended_actions` (`action_class`, semantic target, intent) |
| `groups` | the stage tag every history step carries: `current_stage` / `stage_idx` |
| `outcome` nodes and `causes` with (predicted, recorded) | the `actual_effect_class` the **next** step's deciding call labels, reconciled against the monitor before it is recorded (§6); `_verification_verdicts[k]` holds the predicted and recorded labels, the model's own label, the override tag and the change bit |
| the recovery pattern | read off the record: an `error` or reconciled `no_change` outcome followed by the next step's action; the failure reason reaches the next call on the monitor feedback channel |
| attributes | the action's target, parameters and intent; the observation snapshot on the stored entry (not injected) |
| the whole graph, compressed | `graph_summary` — the model's natural-language rendering, rewritten each step |

The stored support entry (§5) is the graph in recipe form: the plan, the ordered action log whose every
`causes` edge carries a predicted and an externally reconciled recorded effect class, and the summary. Explicit
topology can be re-materialized from it if a structural consumer earns its place.

### What the reasoning call sees and emits

**Inputs the deciding call sees** (WebArena; `reasoning/graph_reasoning.py`, shared by every backbone):
instruction, step index and current stage, the raw observation text, the plan, the last 3 `recent_history` steps
(each compacted to its actions, a short reasoning excerpt, the intended actions with their predicted effects, the
labelled effect, the retrieved entry ids and the monitor status), `existing_graph = {graph_summary}`, and the
retrieved exemplars. On OSWorld 2.0 the benchmark-side `OSWorldReasoningAgent` renders the executed-step timeline
instead of the 3-step window (§2).

**Outputs it must emit** (`common/schemas_json.py`, strict schema on backbones that support it, post-parse
normalization otherwise):

| field | role |
|---|---|
| `actual_effect_class` | labels the prior step's outcome; null at step 0. Reconciled against the monitor before it is recorded (§6). |
| `plan` (nullable) | goal / current stage / stages; re-emitted only when it changes. |
| `graph_summary` | a few sentences, rewritten each step: what is done, the current sub-goal, what remains. The only graph artifact on `RuntimeState.content`. |
| `intended_actions[]` | each with `predicted_effect_class` — required in every deciding mode (principle 3), so verification can never go silently inert. |

**What the structure does for reasoning.** The predicted/recorded pair makes the model commit to an expected
effect before acting and judge the observed effect after, against a closed vocabulary rather than free prose; the
monitor's change test then corrects that judgment where the observation did not change. `graph_summary` is the
model's compressed working memory across the episode, in place of a growing per-step log. The deciding prompt
instructs the model not to emit structure; the record carries the graph.

**What the labels feed downstream.** Per-step verification and the completion gate (§6), the storage-time relabel
(§6), and the stored recipe that later retrieval injects (§5). The graph stage is compulsory: skipping it breaks
the summary the next call reads and the labels the store keeps.

---

## 4. Controlled effect-class vocabulary

```
window_opened · window_closed · navigation · field_filled · selection_changed
· content_modified · state_toggled · saved_persisted · error · no_change
```

- **Agent-emitted against a closed enum**, enforced at emission natively (strict JSON schema) or by
  post-parse normalization; anything off-vocabulary becomes the out-of-band `unknown` sentinel, which
  never matches a prediction and never counts as progress.
- **Predicted vs recorded.** The expected class is recorded before acting; the model labels the actual
  class after, against the resulting observation, and the recorded class is that label reconciled against
  the change test (§6). Their equality is the verification primitive.
- **Families** (`interaction_state`, `persistence`, `navigation`, `null`) partition the vocabulary for the
  episode-close relabel check, where exact-class identity across two model labels drawn at different
  times is too brittle. The `null` family (`no_change`, `error`, `unknown`) never confirms progress.
- **The full vocabulary is kept.** Some classes are hard to observe (§8); they stay in the enum and are
  simply not trusted as primary evidence.
- **Terminal decisions are the model's own, under a plain contract** (`TERMINAL_CLAIM_CONTRACT`,
  `reasoning/base.py`): decide `done` when the objective is achieved, carry the answer for
  answer-seeking tasks, no answer text for state-change tasks, state impossibility briefly rather than
  fabricate. **Do not add obligations to this shared contract.** Evidence requirements, give-up channels
  and format-only rules each perturb termination more than they fix; the mere availability of a give-up
  path acts as a give-up cue. Benchmark answer dialects are normalized at the adapter, never in the contract: WebArena's
  literal `"N/A"` and OSWorld 2.0's `FAIL` token are emitted from one shared whole-answer matcher
  (`common/infeasible_declaration.py`), each adapter owning the dialect it emits and the audit record.
- **The trailing block of the deciding instruction is a defaulted parameter**
  (`deciding_instruction_suffix`) whose shared default is exactly the contract above. One benchmark
  overrides it: OSWorld 2.0 passes a benchmark-side variant
  (`src/benchmarks/osworld/prompts.py`) whose infeasibility clause carries the guardrails the bare control's
  own prompt already carries, under the rule **match, don't exceed** — an instruction both arms carry
  cancels out of the delta, one only we carry does not, so every clause names the upstream line it
  matches. Class two under §10, scoped so it cannot execute on WebArena. Reason: how often an unguarded
  infeasibility clause fires depends strongly on the benchmark, and on OSWorld 2.0 the control's own prompt
  carries guardrails against a premature give-up, which ours must match. A cell on a different base model
  takes that model's own upstream prompt. **WebArena passes its own block:** the contract followed by the control's
  "Homepage" paragraph, verbatim (`src/benchmarks/webarena/prompts.py:WEBARENA_DECIDING_INSTRUCTION_SUFFIX`),
  under the same rule. The paragraph's first sentence names the site list; its second, `password.html`, is
  a dead page in both arms and is copied as a match.
- **OSWorld 2.0, the Qwen profile.** The Qwen profile's suffix carries only that
  control's own clauses: no infeasibility section, and `fail` stays in the vocabulary. A give-up is
  emitted as `DONE`, never `FAIL`, because that control never emits `FAIL`: its LLM judge turns every
  terminal candidate into `DONE` or a continue. The executed-action record keeps `terminal_intent:
  "fail"`, so give-ups stay countable. When one of the gate's two give-up rules
  (`fail_rests_on_step_budget`, `unverified_barrier`) blocks a give-up, the line carries the judge's BLOCKED
  text verbatim; a block by any other rule keeps the gate's own rule text. The judge itself stays a
  control-only mechanism.
- **Benchmark-side action mapping (WebArena).** The grounded action string passes through upstream's
  `map_url_to_local` (alias host → local site) in `build_action_batch`, as the control's
  `extract_action` does, including upstream's substring quirks (`www.` hosts are not mapped).

---

## 5. Retrieval — deterministic, text-only

Retrieval adds **no model call**. One class serves every backbone (`retrieval/deterministic.py` over
`retrieval/support_set.py`).

**Store.** One JSON file per successful episode under `artifacts/support_set/<backend>/`: `task_id`,
`instruction`, `plan`, `recent_history` (the recipe: every step's action with its predicted and
reconciled actual effect class), the final-step observation text, `metadata`, and
`graph_artifacts = {graph_summary}`. Stores are per backend and disjoint. Hygiene: atomic writes,
tolerant reads under concurrent partitions, unique entry ids, dedup that keeps the best entry per task
(win first, then newest). The entry shape is deliberately fluid.

**Ranking.** Cosine similarity from a local sentence-embedding model (`Qwen/Qwen3-Embedding-0.6B`, 32K
context, so the encoder never truncates the (6000-char) query text). Query text = instruction + compact observation
text; candidate text = stored instruction + stored observation text. Then, in order: an **absolute
relevance gate** (weakly matched experience is never injected — principle 8), **at most one
entry per task id**, and a **hard cap of 2 injected entries**. The cap is a cap, not a default: a two-pass
model asking for more cannot reopen slots. **The gate is a fixed constant by default and a calibrated percentile where a benchmark opts in.** A
constant carries no meaning across a change of query composition: the constant chosen for WebArena sits
near the bottom of the distribution it was chosen against, so once the scale moves under it — as it does
when the query loses its observation text — it admits nearly everything, and a store holding one entry
injects that entry into almost every later task. The percentile of the offline pairwise
instruction-similarity distribution is the pre-registered parameter; it is fixed on the reference
benchmark by a criterion stated entirely about that benchmark and then applied unchanged, so it is not
tuned to the set it is applied to. The resolved value is bound to the corpus it was derived over, and a
disagreement raises rather than degrading to the constant while telemetry reports the rule as active.

Candidate encodings are cached across steps and invalidated on
every store mutation (the standing encode path).

**Injection.** What is ranked and what is injected differ. Every injected copy is stripped to the
recipe: the exemplar's `graph_artifacts`, its captured observation snapshot and any retrieval keys are
dropped, and the action string recorded twice per step is deduplicated. The exemplar's summary is
therefore *not* injected; its labelled action log is. That entry-level strip is unconditional.

A second compression applies the same principle one level down, **within each step** — the verbatim
model reply, the per-step narration and the per-step bookkeeping keys are diary rather than recipe.
It is **benchmark-scoped and off by default**: a defaulted parameter the adapter passes, so a benchmark
that asks for nothing gets the entry-level strip alone. Where it is on, every step
survives — it shortens what each step says, it does not drop steps. Truncating **across** steps is not
used, for the opposite reason: on a long budget it discards real procedure from a verified success, and a
head-and-tail extract of a long flailing attempt reads as a short crisp one unless the elision is marked.

**The text arm is a hard requirement.** An unavailable embedder raises `TextArmUnavailableError`; the
runner classifies the episode `infra_error` and re-runs it. There is no lexical fallback, because a
degraded ranker would silently mint a scored row on a retriever that never ran (principle 9).

**Retrieval off is an operational mode, not a switch.** `--retrieval_off` swaps in a bypass agent that
opens no store and suppresses store writes. It is the protocol's setting for task sets with unique intent
templates, where no stored entry shares a procedure with the query (pass it explicitly; the runners default to
retrieval on).

### Where retrieval is meant to pay off

- **Same-family recipe transfer on long-horizon, state-changing tasks.** Full WebArena groups its 812
  tasks into 190 intent-template families, so a later task can retrieve a sibling's verified recipe. The
  pre-registered warm round measures exactly this: tasks the cold run lost, and whose family later produced a
  win, are re-run with the finished store as read-only input, in a stratum where reverse flips are impossible
  by construction; re-runs of cold winners are the churn reference.
- **The sibling recipe is what transfers, not surface tokens of the observation.** Text similarity picks the
  sibling and the labelled action log carries the procedure; exemplar topology in the injection and
  structure-based ranking beside the text arm are not used.
- **On unique-template sets retrieval contributes nothing by construction**: nothing in the store shares a
  procedure with the query. This is why those runs pass `--retrieval_off`.
- **Cost lives in injected tokens.** The decomposition-tax lever is injection volume, not call count.

**Open: selectivity.** Injection can perturb a task the agent would have won alone; separating that from
run-to-run churn needs a no-injection control. This is the one open retrieval workstream; the candidate
directions are a tighter admit rule, or an abstain path when the top candidate is not a family sibling.

### Demand-driven retrieval (two-pass only)

In the two-pass shape the `initial` call emits `retrieval_request = {scope ∈ task|stage|recovery|none,
query, k}`; `scope = none` skips retrieval, a missing request falls back to the deterministic policy, and
`k` is capped at 2. Single-pass retrieves before any reasoning has run and therefore always uses the
deterministic policy.

---

## 6. Verification — one external signal, three projections

The signal is the heuristic monitor's **diff of the full observation between steps** plus the
model's closed-vocabulary effect labels. No LLM verifier, no self-critique. On WebArena the diff is
over the raw text. On screenshot-primary benchmarks (OSWorld 2.0) it is a **screenshot diff**:
grayscale, full resolution, a pixel counts as changed at |Δ| > 32 and the observation counts as changed
at ≥ 64 changed pixels — deliberately the safe side of the asymmetry below (a false "changed" leaves
the model's label authoritative; a false "unchanged" would override real effects to `no_change`); a
caret blink (~32 px) does not trip, a clock tick (~150 px) does. The floor is an instantiation constant,
carried on the per-step `diff_pixel_count` telemetry every episode records. **It is not calibrated from
a distribution, and does not need to be**: real transitions change far more pixels than the floor, and the
`unchanged` verdicts that occur are decided by the pointer mask below rather than by the floor's value.
**The mouse pointer is composited into the frame**, so a small pointer move alone clears any floor a
real change would also clear, and an unmasked detector reports `changed` on essentially every step — which
silently disables the three mechanisms that act only on *no change*: per-step reconciliation, the
completion gate's second rule, and the episode-close relabel. The instantiation therefore masks the
pointer at **both** bracketing coordinates — both, because the frames bracket the action, so the cursor
is removed at one and added at the other, and masking only one leaves a sprite's worth of changed
pixels. The earlier coordinate is the nearest *preceding record that carries a pointer position*, not
the immediately preceding step: a step that types into the field it clicked two steps ago must still
mask where the cursor is sitting, and reading only the last record would miss it. No mask is applied on
a step whose own action moves no pointer, or a step that types into the field it clicked on the
previous step would have its typed characters masked. **Each episode's first closed step is the one
exception**: the guest's boot cursor position is never recorded, so that step masks one coordinate
rather than two — inventing the other would blank a region the cursor may never have occupied, which is
the over-masking failure this mechanism must not have. The consequence is that a pointer-only no-op can
still read as `changed` on that step.

The audit trail makes the mask judgeable after the fact. The masked count, the unmasked
count and **the coordinates actually masked** are persisted to the episode record — the coordinates
live rather than re-derived, because the locator is the function under suspicion and re-running it
offline would reproduce a coordinate error identically and agree with itself. A recorded empty list
("the locator ran and masked nothing") and an absent record are kept distinct; collapsing them would
lose the difference between a fact about the step and a gap in the record. The first-closed-step
exemption is identifiable from the record rather than merely declared: those are the steps carrying a
single masked coordinate. The mask is benchmark-scoped and off by default.
Completion-gate rule 1 (answer grounding) is off on OSWorld by construction (it grounds answers against
observation text, which is empty on a screenshot-only benchmark); rule 2 applies.

**Per-step reconciliation (M1).** After each action the monitor reports `changed`. The deciding call's
`actual_effect_class` is reconciled against it before being recorded: on a **stall** (no change) a
positive-progress claim, an `unknown`, or a missing label all become `no_change`; on a **changed** page
the model's label stands (we never manufacture `no_change` out of a page that changed). The verdict
`predicted == reconciled actual` is recorded per step with both labels and the override tag, so the
telemetry is verdict-level and doubles as label-reliability data.

**Completion gate (M2).** A `done` claim is admitted only on external evidence:

1. an answer-bearing `done` must be **grounded** — its informative tokens overlap the last 3 observations
   plus the current one at ≥ 0.5, or every numeric token in the answer appears there;
2. an empty / `N/A` `done` needs **one monitor-confirmed, non-navigation positive effect** anywhere in the
   trajectory (navigation alone never admits a completion).

A blocked `done` becomes a corrective no-op with the failed rule fed back to the next reasoning call;
after two blocks in one episode the next `done` is submitted. The benchmark evaluator is the final judge;
the gate only forces evidence-gathering. Toggle: `--completion_gate_off` (ablation only).

**OSWorld 2.0 (benchmark-scoped, `src/benchmarks/osworld/orchestrator.py`).** Rule 1 is off (it grounds answers
against observation text, which is empty on a screenshot-only benchmark). Five more rules block a claim on facts the episode recorded, never on the model's own verdict. Two
are give-up rules: a give-up resting on the step budget, or on a barrier its own statement concedes is
unverified, before 80% of the steps are used (`fail_rests_on_step_budget`, `unverified_barrier`). Three are
success-claim rules: a media CLI bypassing an instruction-named application; a material file named as the thing
completed; a media deliverable synthesised from nothing (`named_app_bypassed_by_command_line`,
`use_file_claimed_done`, `synthetic_media_deliverable`). They share the two strikes. Rules marked post-hoc in the
code had their lexicon written from inspected episodes of the failure they target. **DeepSeek profile only:** the
deciding reply keeps a requirement ledger (at most 20 items rendered, each open / verified / unmet with an
evidence step), and `requirements_unverified` blocks a success claim that its own ledger contradicts, with its
own two strikes.

**Episode-close relabel write-back.** Before a winning episode is persisted, every step claiming a
positive-progress effect whose *family* has zero monitor-confirmed instances anywhere in the episode is
relabelled `no_change` with provenance. If the episode has no confirmed family at all, nothing is
touched (prefer missed corrections over wrong ones). Both benchmark adapters call it.

**What verification does not do.** There is no progress-gated terminate: a run of `no_change` is not
terminal evidence at this model class, and the max-steps cap is the sole budget bound. One OSWorld 2.0 arm
is the exception: the Qwen profile carries a **stuck-stop**. Past 500
predict calls, an episode spinning in place — one executed command at least 50 times among the last 55
executed steps with at most 11 of them changing the screen (clause repeat), or 50 consecutive transparent
WAITs with at most 10 changing it (clause W) — ends with the profile's give-up token, without a model call
and without passing the completion gate. It is keyed by backbone in the OSWorld adapter
(`graphpc_agent.py:OSWORLD_STUCK_STOP_BY_BACKBONE`), so it cannot run on WebArena. Completion gates on answer-type terminals beyond rule 1 are structurally unsound (the monitor
cannot confirm an answer was submitted) and are not attempted.

---

## 7. Measurement discipline

- **Paired per-task comparison is primary** (win/loss flips, exact McNemar, per-site and per-evaluator
  tables); aggregate deltas are context. Single-run churn on a few hundred tasks is several points, so an
  aggregate delta of that size is weak evidence on its own.
- **Both arms run the same model on the same harness.** Baselines are locked once clean and never
  re-run. Cross-model reference lines are not evidence.
- **Additions are gated by measured gain; subtractions by a pre-registered non-inferiority test** on
  paired rows, with one decision instrument named in advance.
- **The active configuration is verified in resolution-level telemetry before any number is trusted**:
  the embedder that ranked, the retrieval mode, the injection diet, the gate settings. A degraded path
  (fallback ranker, mock env, ablation flag left on) invalidates the row.

---

## 8. Known limitations

- **Hard-to-observe effects.** `saved_persisted`, and to a lesser extent `selection_changed` /
  `state_toggled`, can leave no trace in the observation text and degrade to `no_change`. "Did the
  persist succeed" is exactly what matters on write tasks and exactly what is hardest to see; persist
  verification may need a signal outside the structured observation.
- **One action per observation.** Effect labels assume one action is attributed per resulting
  observation (the WebArena configuration runs one action per step); batches would break the attribution.
- **Retrieval selectivity is unsolved** (§5). Injection can perturb tasks the agent would have won alone.
- **URL-graded and short-navigation tasks give the pipeline little to work with**; on combobox / date-entry
  grinds the predict/verify overhead can cost more than it adds.
- **OSWorld 2.0.** The harness is OSWorld-V2's own driver with vendored copies of its runners,
  screenshot-primary, and the cells run a 28-task subset. The screenshot-diff floor (§6) is an instantiation
  constant, not a calibrated one. The vision gate's grounding median carries residual noise, handled by the
  launch rule: one re-run allowed, two consecutive failures block.

---

## 9. Parked (deferred, with rationale)

- **Structural matching / topology revival.** Not used on WebArena, where same-family tasks share surface
  wording and text similarity already finds the sibling. Its remaining test bed is a regime that is lexically
  divergent but procedurally similar (OSWorld 2.0 sub-procedures across applications). Revival requires a
  validated-benefit case under principle 1 (no structural-matching code ships).
- **Global cross-task experience graph + propagation retrieval** (HippoRAG / Graph-of-Skills style):
  data-gated; build only if telemetry shows recurring cross-trajectory structure the per-entry store does
  not already surface.
- **Graph embeddings; graph state steering re-planning; an LLM-based monitor; further stage
  decomposition.** Each is an added call or mechanism and pays the decomposition tax up front.
- **Elastic retrieval-k (default 2, model-selectable ≤ 3).** Only meaningful in two-pass. Revisit only on a
  retrieval-on run with recurring sub-procedures, with a pre-registered non-inferiority check.
- **Write-landing confirmation** as a *positive* mechanism (never as terminal pressure) and a
  **judgment-level answer-validity check**: two lag classes a terminal-contract change does not close (§4:
  obligations added to the contract perturb termination more than they fix).

---

## 10. Implementation invariants

- **A benchmark-motivated change is argued into one of two classes, and scoped accordingly.** *Class one* is
  neutral **with respect to the comparison** — not merely benchmark-neutral in motivation; a change that
  alters what our arm gets relative to what the control gets is not neutral whatever its motivation, and a
  within-arm gate cannot settle it because the control never appears in that comparison. Class one is built
  as an isolated plugin and validated on the reference benchmark before it is trusted anywhere, unless its
  reference-benchmark instantiation reproduces that benchmark's existing behaviour by construction, in which
  case the validation is vacuous and is waived (the percentile relevance gate). *Class two*
  is specific to one benchmark's characteristics and is scoped so it **cannot execute** elsewhere —
  "provably inert elsewhere" is not the guarantee, because inert-by-accident comes alive the day a data
  format changes. Scoping is a defaulted parameter the benchmark's adapter passes, never a conditional
  inside `src/agents/`, and the shared default is the behaviour the other benchmark already runs, so its
  requests stay byte-identical on every path (check it by hand: diff a WebArena request before and after any change
  to a shared default). Heuristic: a change that would be a no-op on the other
  benchmark is almost certainly class two, and its being a no-op is exactly why class one's validation route
  is closed to it.
- **Every property a launch or a run depends on gets a loud-fail check under `scripts/`.** A property
  that depends only on the code is a test (`tests/`). Each check carries a negative control that runs on
  every invocation — a check that
  has gone vacuous must fail on *that* ground rather than passing — and each reports the coverage it
  actually demonstrated rather than the size of what it inspected.
- **Delete, do not disable.** Removed mechanisms are removed at the source — no flags, no dormant code,
  no vestigial fields. Disabled-but-present code masks a live mechanism and confounds attribution.
  Re-enabling anything removed requires a validated-benefit case under principle 1.
- **Schema additivity.** `src/agents/common/schema.py` is stable. New signal rides on the open `content`
  / `metadata` dicts or on the non-frozen retrieval input envelope. `graph_summary` lives on
  `RuntimeState.content`; the support-set entry shape is fluid.
- **Action grounding stays a separate stage** (§1).
- **Benchmark-side code is copied from the harness and adapted minimally** (AGENTS.md working principle): runners,
  episode drivers and baseline agents keep upstream's names, flags and result layout; the method layer's delta sits
  behind the inner boundary (§12), so a vendored file runs as-is in the pinned checkout on the run host.
- **Sync boundary (OSWorld 2.0).** Every module that runs inside the OSWorld-V2 checkout is registered in
  `src/benchmarks/osworld/sync.py:SYNC_SOURCE_TARGETS`; new files use `src.`-absolute imports.
- **Benchmark adapters call only the inner boundary** (§12); external benchmark repositories are
  read-only outside the vendored deployment subtree.
- **Backbones adapt to the schema, never the reverse.** Every backbone reuses the shared
  reasoning agent (on OSWorld 2.0 through its benchmark-side subclass `OSWorldReasoningAgent`) and, on
  WebArena, the shared action agent; OSWorld 2.0 grounding is benchmark-side per model (`action_minimax.py`,
  `action_qwen.py` and `action_deepseek.py`); the backbone wrapper routes grounding vs reasoning calls. On
  OSWorld 2.0 what varies with the model is one profile per model
  (`graphpc_agent.py:OSWorldModelProfile`, selected by `backbone_type`), so a model is added as a profile
  and its classes, never as a branch in the model-neutral adapter, timeline or gate rules. Two per-model
  settings sit beside the profile as backbone-keyed tables in the same file: the step budget
  (`OSWORLD_MAX_STEPS_BY_BACKBONE`, which the vendored runner and the launch checks read where no profile is
  built) and the Qwen arm's stuck-stop (`OSWORLD_STUCK_STOP_BY_BACKBONE`, §6).

---

## 11. Key decisions (rationale, so they are not re-litigated)

- **Central failure mode this design avoids: decomposition tax with an inert value core.** A pipeline
  whose value stage never reaches behavior pays the cost of stages without collecting the benefit.
  Everything else here follows from that.
- **The graph schema serves the discipline and the memory, not the ranking.** Typed effect fields
  enforce predict/verify; the running summary is the working memory; retrieval ranks on text. Structure is
  not the matching key: retrieval ranks on text and injects the labelled recipe, not topology, and the
  summary is the graph artifact the model reads.
- **Closed effect vocabulary is the keystone primitive**: one closed-vocab construct serving
  verification, the completion gate, the relabel write-back and the persisted recipe.
- **Perception before gating.** The reasoner seeing the full observation comes before any gate; failure
  classes that look like termination defects are often perception defects.
- **Single-pass is the configuration; two-pass is a backup design, not the default.**
- **One external signal, projected three ways** (per-step verification, completion gate, relabel
  write-back) rather than a separate verifier mechanism.
- **Terminal contract stays plain**; benchmark dialects are adapter concerns.
- **Retrieval off where families are absent**; retrieval on where they exist.
- **Exploration benchmark = WebArena** (190 intent templates × ~4.3 instantiations give natural procedure
  families); **OSWorld 2.0 is the second benchmark**, restricted to natively multimodal models, because its
  action path grounds from screenshots.
- **Novelty:** (i) the procedure-causal graph — a record of what each action did (a
  closed effect class, predicted before acting and checked after by a model-free change test), against UI
  transition graphs that record where an action leads (action spaces are standard; effect taxonomies on
  interface agents are not — cf. EAI, NeurIPS'24; "Web Agents with World Models", ICLR'25, predicts free-form
  outcomes and stores nothing); (ii) one record serving three roles — progress verification, context
  management and reusable experience — in a training-free harness around an unmodified model. Structural-signature
  matching and graph search are not claimed.

---

## 12. Benchmark integration boundary

**Two boundaries.**

- **Outer (benchmark runner ↔ our agent):** benchmark-shaped and replaced per benchmark. OSWorld 2.0: V2's
  `lib_run_single.run_single_example` contract — `reset(runtime_logger)` + `predict(instruction, obs) ->
  (response_text, [pyautogui code | "DONE" | "FAIL" | "WAIT"])` (+ optional `set_api_log_dir`), the score
  reaching the agent through the vendored runner's post-episode `finalize_task` hook. WebArena: `next_action(trajectory,
  intent, meta_data) -> Action` + `reset(config)` via WebArena's `Agent` class and released action
  factories only.
- **Inner (our agent ↔ orchestrator) — the stable, schema-typed one:** exactly two methods,
  `decide_next_action(*, instruction, observation_input, runtime_state) -> GraphPCDecision` and
  `update_runtime(*, decision, current_observation_input) -> GraphPCRuntimeUpdate`, over the types in
  `common/schema.py`. Obs-dict → `Observation` happens inside the orchestrator (injected perception
  adapter); `ActionBatch` → benchmark action happens in the adapter.

**Decoupling invariant.** Adapters call only the two public methods and the schema types, never
orchestrator internals. Design changes live inside those methods and ride on `content` / `metadata`. Check
the invariant with a strict-stub orchestrator whose `__getattribute__` throws on any non-boundary access.

**Risk: stable at the type level, not the payload level.** `ActionBatch.actions[].raw["input"]` is
action-space-specific and produced by the action agent. A new benchmark needs both a new adapter and a
new action agent; an adapter alone is necessary, not sufficient. Within OSWorld 2.0 the same holds per
model: its action agent and translator are the model's (its model profile), the adapter around them is not.

**Per-benchmark specifics** (deployment, runners, baselines, adapters): `.claude/docs/benchmarks/<name>.md` (index: `.claude/docs/benchmarks/README.md`).

---

## 13. References

**Principle anchors**
1. Huang et al. — *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024.
   https://arxiv.org/abs/2310.01798
2. Cemri et al. — *Why Do Multi-Agent LLM Systems Fail? (MAST).* NeurIPS 2025 D&B.
   https://arxiv.org/abs/2503.13657
3. Yoran et al. — *Making Retrieval-Augmented Language Models Robust to Irrelevant Context.* ICLR 2024.
   https://arxiv.org/abs/2310.01558
4. Fang et al. — *Enhancing Noise Robustness of Retrieval-Augmented Language Models with Adaptive
   Adversarial Training.* ACL 2024. https://aclanthology.org/2024.acl-long.540/

**Closest prior work on experience reuse for GUI/web agents** (differentiate: flat exemplar or workflow
memory, no effect-side vocabulary, no external reconciliation of stored labels)
5. Zheng et al. — *Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control.* ICLR
   2024. https://arxiv.org/abs/2306.07863
6. Wang et al. — *Agent Workflow Memory.* ICML 2025. https://arxiv.org/abs/2409.07429
7. Agashe et al. — *Agent S: An Open Agentic Framework that Uses Computers Like a Human.* ICLR 2025.
   https://arxiv.org/abs/2410.08164

**Effect-side comparators**
8. Choi et al. — *Embodied Agent Interface (EAI).* NeurIPS 2024 D&B. https://arxiv.org/abs/2410.07166
9. Chae et al. — *Web Agents with World Models.* ICLR 2025. https://arxiv.org/abs/2410.13232

**Graph-memory comparators (parked direction, §9)**
10. Gutiérrez et al. — *HippoRAG.* NeurIPS 2024. https://arxiv.org/abs/2405.14831
11. Li et al. — *Simple is Effective: The Roles of Graphs and LLMs in KG-Based RAG (SubgraphRAG).*
    ICLR 2025. https://arxiv.org/abs/2410.20724
12. Liu et al. — *Graph of Skills.* arXiv preprint. https://arxiv.org/abs/2604.05333

**Benchmarks**
13. Zhou et al. — *WebArena.* ICLR 2024. https://arxiv.org/abs/2307.13854
14. Xie et al. — *OSWorld.* NeurIPS 2024. https://arxiv.org/abs/2404.07972
15. XLANG Lab — *OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks.* 2026.
    https://arxiv.org/abs/2606.29537 · https://github.com/xlang-ai/OSWorld-V2
