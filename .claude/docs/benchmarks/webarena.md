# WebArena

**Status: primary.** Runner at `scripts/webarena/run_graphpc.py`; adapter under `src/benchmarks/webarena/`.
This file is the deployment doc: how to stand the sites up, how to launch each arm, what regime a scored run uses,
and the per-backbone route facts. The site host and the driver machine must both be set up and configured (the
one-time setup below) before any run.

**Deployment topology.** WebArena is also AWS-hosted, but the topology is different from OSWorld. OSWorld 2.0 provisions a **per-task** EC2 VM; WebArena instead runs a **single long-lived EC2** that hosts all six sites (Shopping / Shopping-Admin / Reddit / GitLab / Wikipedia / the Map frontend) as Docker containers, plus optionally a separate Map-backend EC2. Our runner (`scripts/webarena/run_graphpc.py`) runs on a separate driver machine (laptop / dev box / CI) and drives the AWS-hosted sites over the network using Playwright. There is no per-task VM, no sync workflow, no external-repo deployment subtree.

## One-time setup — AWS-hosted WebArena sites

WebArena publishes a pre-built AMI in `us-east-2` with all sites pre-installed. **Use the AMI** — building from individual Docker images is in WebArena's `environment_docker/README.md` if needed, but the AMI path is what's supported. Most steps here require human action (AWS console, key pairs, security groups); they are done once and only redone if the host is rebuilt.

1. **Launch the WebArena AMI.** EC2 console → AMIs → switch region to **us-east-2 (Ohio)** → search public AMI:
   - **AMI ID:** `ami-08a862bf98e3bd7aa` — name: `webarena-with-configurable-map-backend`
   - **Recommended instance:** `t3a.xlarge`, **1000GB EBS root volume**
   - **Security group inbound:** SSH (22) from your IP + TCP `3000, 7770, 7780, 8023, 8888, 9999` from `0.0.0.0/0` (or your driver-machine CIDR)
   - **Elastic IP:** allocate and bind it, so the sites keep one public hostname across reboots
   - **(Optional) Map backend:** if not using shared infra, set `MAP_BACKEND_IP=<your-backend-ip>` in user data; see WebArena's environment_docker README for standing up a separate map-backend instance (~60–90 min auto-setup, ~180 GB)

2. **Boot the dockerized sites** (SSH to the instance):
   ```bash
   docker start gitlab shopping shopping_admin forum kiwix33
   cd /home/ubuntu/openstreetmap-website/ && docker compose start
   # wait ~1 min for services to start
   ```

   If services aren't externally reachable, apply iptables redirects (full block in WebArena's environment_docker README).

3. **Configure base URLs** to point at the Elastic IP / hostname (replace `<HOSTNAME>` in each):
   ```bash
   # shopping + shopping_admin: magento setup:store-config + DB update + cache flush
   # gitlab: external_url + gitlab-ctl reconfigure
   # full commands: see WebArena environment_docker/README.md "Pre-installed Amazon Machine Image" section
   ```

4. **Verify all sites are HTTP 200:**
   ```bash
   HOSTNAME="<your-elastic-ip-hostname>"
   for p in 7770 7780 9999 8888 3000 8023; do
     curl -s -o /dev/null -w "Port $p: %{http_code}\n" http://$HOSTNAME:$p
   done
   ```

## One-time setup — driver machine (where `scripts/webarena/run_graphpc.py` runs)

5. **WebArena code checkout + install.** Our runner reuses WebArena's `browser_env` / `evaluation_harness` / `run.py` loop pattern; clone alongside this repo and install:

   ```bash
   git clone https://github.com/web-arena-x/webarena.git ../WebArena
   cd ../WebArena
   git checkout dce0468   # the commit the upstream code copied into url_mappings.py is taken from
   conda create -n WebArena python=3.10 && conda activate WebArena  # dedicated env — do NOT share the GraphPC dev/test env (it lacks the WebArena stack)
   pip install -r requirements.txt
   pip install -e .
   playwright install
   # WebArena pins openai==0.27, but this repo's backbones and control shims need openai 2.x. Install this repo's pinned set
   # (openai 2.x, anthropic, the retrieval text arm) after the requirements install; WebArena's 0.27-style call sites then run
   # under openai 2.x through src/benchmarks/webarena/_openai_compat.py (no downgrade, no WebArena edit):
   pip install -r <repo>/requirements.txt
   ```

   The runner defaults `--webarena_root ../WebArena`; override if you cloned elsewhere. The runner validates the directory contains `run.py` and fails fast otherwise. Both runners (`run_graphpc.py`, `run_baseline.py`) install the same two runtime patches on WebArena's `browser_env/` from `src/benchmarks/webarena/_env_hardening.py` (no WebArena file is edited; "Adapter architecture" below).

6. **Export per-domain URLs** pointing at the AWS-hosted sites (these are read by WebArena's evaluator + config generator). Pin them in `.env` at the repo root so `load_dotenv()` picks them up:

   ```bash
   export SHOPPING="http://<HOSTNAME>:7770"
   export SHOPPING_ADMIN="http://<HOSTNAME>:7780/admin"
   export REDDIT="http://<HOSTNAME>:9999"
   export GITLAB="http://<HOSTNAME>:8023"
   export MAP="http://<HOSTNAME>:3000"
   export WIKIPEDIA="http://<HOSTNAME>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
   export HOMEPAGE="http://127.0.0.1:4399"  # required: the homepage app runs on the driver machine (see "Homepage app and launch directory")
   ```

7. **Generate per-task config files** (one `*.json` per test example, with the URLs substituted in):
   ```bash
   cd ../WebArena
   set -a; source <repo>/.env; set +a   # browser_env/env_config.py asserts the site variables are set
   python scripts/generate_test_data.py
   # produces config_files/*.json
   ```

8. **Auto-login cookies** (sites require logged-in sessions):
   ```bash
   cd ../WebArena
   mkdir -p ./.auth
   set -a; source <repo>/.env; set +a   # auto_login.py does not load .env; a bare run fails silently on the empty-URL assert
   python browser_env/auto_login.py
   # writes session cookies under ./.auth/
   ```

   Re-run this after any environment reset (see "Per-run operational note" below) — cookies are tied to specific
   session state — and check the logins with `python <repo>/scripts/webarena/verify_auth_logins.py` (its defaults
   resolve against the repository, so it runs from any directory; every `.auth` state the task configs use must
   answer its logged-in page; exit 1 otherwise).

9. **Local embedding for the retrieval text arm** (see `design.md` Retrieval + Implementation invariants). **Required for every retrieval-on run**: the text arm is the only ranker, with no fallback, so a missing model makes each retrieval raise (`TextArmUnavailableError` -> `infra_error`) rather than degrade. Only `--retrieval_off` runs are unaffected; the launch preflight fails loudly on this:

   ```bash
   # sentence-transformers comes with <repo>/requirements.txt (step 5)
   # First import caches the model under ~/.cache/huggingface/hub/ (on the driver machine; uses GPU if available, else CPU)
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
   ```

10. **API credentials.** The Anthropic first-party API (`ANTHROPIC_API_KEY`) for the Claude backbone, `OPENAI_API_KEY` for the WebArena judge (every cell) and the GPT backbone, `DEEPSEEK_API_KEY` / `MINIMAX_API_KEY` for DeepSeek and MiniMax, and `OPENROUTER_API_KEY` for Qwen. The `.env` loader at the repo root picks them up. Qwen reaches `qwen3.7-plus` through **OpenRouter** (`https://openrouter.ai/api/v1`, slug `qwen/qwen3.7-plus`), the only Qwen route the code carries: one credential, no base-URL override variable, no second provider and nothing to fall back to.

## Homepage app and launch directory

- **The homepage app** (the `HOMEPAGE` site, `127.0.0.1:4399`) is upstream's `environment_docker/webarena-homepage`,
  set up by the "Homepage" section of WebArena's `environment_docker/README.md`: a copy of that directory whose
  `templates/index.html` has the `<your-server-hostname>` placeholder replaced by `http://<HOSTNAME>`, its only
  difference from upstream. It is served on the driver machine from the WebArena env, bound to
  localhost: `python -m flask --app app run --host=127.0.0.1 --port=4399` from inside the copy. It does not start
  on its own: after a reboot, relaunch it (rotating its log) before any run.
- **Launch from the WebArena checkout** (`cd ../WebArena`, script paths absolute): the task configs' `storage_state`
  paths (`./.auth/<site>_state.json`) are relative to the working directory, and `run_graphpc.py` does not change
  directory. `run_graphpc_parallel.py` and `run_lanes.py` set the subprocess working directory to the checkout
  themselves.

## Per-run operational note — environment reset (critical for Δ measurement)

Four of the six sites are **stateful**: Shopping, Shopping-Admin, Reddit, GitLab accumulate every agent edit (orders placed, posts created, repo state mutated). Wikipedia and Map are read-only.

**Reset between any two runs whose Δ you intend to compare** — otherwise the second run starts on a state polluted by the first, and the comparison is not clean (counts as contamination under the locked-baseline policy). Concretely:

- **Required reset:** before every SCORED full run (a baseline arm, a cold run, a warm run).
- **Smoke runs:** reset optional — smoke accepts noise, and the ~5 min GitLab warm-up per reset adds up. Reset if smoke results drift suspiciously between iterations of the same change.

```bash
# On the WebArena AWS host
docker stop shopping_admin forum gitlab shopping
docker rm shopping_admin forum gitlab shopping
docker run --name shopping -p 7770:80 -d shopping_final_0712
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
# wait ~1 min; then re-run the base-URL config commands from step 3
# then on the driver: re-run auto_login.py (step 8)
```

GitLab takes ~5 min to come fully online; verify with the curl HTTP 200 loop from step 4 before the next run. Document the reset cadence in `runs/<id>/README` so a later reader knows which runs share state.

## Running

### Live

One task at a time, launched from the WebArena checkout (see "Homepage app and launch directory"):

```bash
conda activate WebArena   # the dedicated env with the WebArena checkout + text-arm + openai-shim; NOT the GraphPC dev/test env
cd ../WebArena
python <repo>/scripts/webarena/run_graphpc.py \
  --backbone claude --model claude-sonnet-5 \
  --config_file config_files/0.json \
  --result_dir <repo>/webarena_results \
  --support_set_root <repo>/artifacts/support_set   # the default ./artifacts/support_set would land inside ../WebArena
```

A directory of tasks:

```bash
python <repo>/scripts/webarena/run_graphpc.py \
  --backbone gpt --model gpt-6-sol \
  --config_dir config_files/ \
  --result_dir <repo>/webarena_results \
  --support_set_root <repo>/artifacts/support_set
```

### Smoke both arms before a scored run

Before any scored arm on a route, run each arm on one task into throwaway result directories — the control with
`run_baseline.py --task_ids <id>` and its regime flags (the "Harness regime" table below), the GraphPC arm with
`run_graphpc.py --config_file config_files/<id>.json` and the arm's flags — and read the run logs:

- every model call leaves one `<route> usage:` line on both arms (`gpt`, `claude`, `deepseek`, `minimax`, `qwen`):
  read the model id the line echoes where it carries one, the reasoning count on every call whose route enables
  thinking (each control call, the GraphPC deciding call), and that no call ended at its output cap;
- the control's actions parse (no early stop on parse failures), and the GraphPC arm wrote `run_config.json` (the
  resolved switches and retrieval mode) and `telemetry_<id>.json`.

A failed smoke stops the launch until it is understood.

### Lite-165 and full-812: the lane driver

Both arms of a full-812 cell — the control, the GraphPC cold arm and the GraphPC warm round — run through the lane
driver `scripts/webarena/run_lanes.py` (`prepare` / `launch` / `status` / `freeze` / `resume` / `casualties` /
`add_lane`; its module docstring is the reference). One lane per site partition: the five single-site lanes run in
parallel as phase 1, `cross_site` (every multi-site task) runs after them as phase 2. The control lanes call
`run_baseline.py`, the GraphPC lanes `run_graphpc.py`, each with the WebArena checkout as the working directory. At
every phase start (and every relaunch) the driver re-renders `.auth` with WebArena's `auto_login.py` and runs
`scripts/webarena/verify_auth_logins.py`; a red result freezes the run. Run the driver in the GraphPC env (it needs only
the standard library); the runners run under `--webarena_python`, the WebArena env's interpreter.

**The code root.** `prepare` refuses a code root that is not a git checkout, is on a branch, or has uncommitted
changes to tracked files, and it requires the code root's own `.env` to set the seven site variables (`SHOPPING`,
`SHOPPING_ADMIN`, `REDDIT`, `GITLAB`, `MAP`, `WIKIPEDIA`, `HOMEPAGE`) — the runners load their keys from it too. Run
each arm from a detached worktree of the commit you launch, in a git clone of this repository (an unpacked archive
is refused): `git worktree add --detach <dir> <commit>`, then copy `.env` into it; the plan records the commit. Pass
`--webarena_root <WebArena checkout>` unless the worktree sits beside it (the default is `<code root>/../WebArena`).

**The inputs.**

- `--partitions_dir data/webarena/full812_site_partitions` — one directory per lane (`cross_site`, `gitlab`, `map`,
  `reddit`, `shopping`, `shopping_admin`), each holding `task_ids.json` = `{"task_ids": [...]}`: a single-site task
  sits in its site's lane, a multi-site task in `cross_site`, and each lane is ordered Lite-165 members first (by id),
  then the rest (by id) (`experiments.md` run protocol, item 4). The lanes' union is the 812 ids.
- `--argv_json` + `--argv_key` — a JSON object of named runner argv lists; `--argv_key` picks this arm's. The argv
  carries the regime flags only: the driver appends `--webarena_root`, `--result_dir` and, per arm, `--task_ids_json`
  (control) or `--config_dir` and `--support_set_root` (GraphPC and warm), so none of those may appear in it.
  `data/webarena/run_lanes_argv.example.json` holds the DeepSeek `deepseek-flash` cell's two, under the keys
  `control` (the "Harness regime" table below) and `graphpc`; for another model, adapt both lists (its `--model`,
  `--backbone`, `--max_tokens` and effort flags, per its route block below).
- `--tripwires_json` — a list of rules `{"name", "action": "freeze" | "alert" | "change", "regex"[, "baseline"]}`
  matched against the lanes' run logs: `freeze` stops every runner at the first match and keeps the state (the HTTP
  402 rule), `alert` alerts at the 1st, 2nd, 4th, 8th, … match, `change` alerts once per new value of the regex's
  group 1 other than `baseline` (or the first value seen). `data/webarena/run_lanes_tripwires.example.json` is the
  `deepseek-flash` cell's set: the 402 freeze, and alert / change rules on its `deepseek usage:` lines (a `length`
  finish, a thinking call with zero reasoning, a model echo or effort other than the pinned one, a new
  `system_fingerprint`, a failed or unrecorded call, 429s, dropped connections). Its `model` rule pins the id
  `deepseek-flash`, so for another model adapt the file: keep the 402 freeze and write the route's rules against its
  own usage lines (e.g. on Qwen a `change` rule on the `model=` value of the `qwen usage:` lines, "Alias-move policy"
  below). Alerts land in the run directory's `ALERTS.log`; nothing stops a run on an
  alert (diagnose first).
- GraphPC arms: `--support_set_root <absolute path>` — the one store every lane shares (in-run accumulation is the
  method). A relative path is refused.
- Optional: `--env_set NAME=VALUE` (repeatable) sets a variable for every runner. On a retrieval-on GPU run,
  `--env_set GRAPHPC_EMBED_BATCH_SIZE=8` caps the embedder's encode batch (default 32), a GPU-memory knob only, for
  when several lanes share one card.

```bash
# the deepseek-flash control, from the detached code root, GraphPC env active; the run directory must be new
python scripts/webarena/run_lanes.py prepare --arm control --run_dir runs/<run id> \
    --code_root . --webarena_root <WebArena checkout> --webarena_python <WebArena env python> \
    --partitions_dir data/webarena/full812_site_partitions \
    --argv_json data/webarena/run_lanes_argv.example.json --argv_key control \
    --tripwires_json data/webarena/run_lanes_tripwires.example.json
python scripts/webarena/run_lanes.py launch --run_dir runs/<run id>    # the supervisor, in its own session
python scripts/webarena/run_lanes.py status --run_dir runs/<run id>
python scripts/webarena/run_lanes.py casualties --run_dir runs/<run id>  # unscored tasks, to re-run with add_lane
```

The GraphPC cold arm is the same with `--arm graphpc --argv_key graphpc --support_set_root <absolute empty store>`.

**Lite-165 through the same partitions.** `prepare --task_ids <ids>` keeps only those tasks, in each lane's order,
so a Lite-165 arm of either kind is the same command with the 165 ids (a retrieval-off arm carries `--retrieval_off`
in its argv). The ids are the Lite plan's:

```bash
LITE=$(python -c "import json; p = json.load(open('data/webarena/lite165_site_partitions.json')); print(','.join(str(t) for ids in [*p['phase1_parallel_partitions'].values(), p['phase2_sequential_cross_site']] for t in ids))")
```

`scripts/webarena/run_graphpc_parallel.py` is an alternative for a retrieval-on Lite-165 GraphPC cold run (outside the comparison matrix): it reads
that plan (`--partitions_json`, default `data/webarena/lite165_site_partitions.json`), runs the five site partitions
in parallel and then the cross-site tasks, all on one shared store (`--support_set_root`, default
`<run_root>/support_set`), and merges every partition's `summary.json` into `<run_root>/summary.json`. Its runner argv
carries no `--retrieval_off`, and its `--backbone` choices exclude `minimax` (use the lane driver for a MiniMax
Lite-165 arm).

### The warm round

The warm round re-runs a pre-registered subset of the cold run's tasks (its derivation: "How transfer is measured"
below) against the cold run's finished store, frozen as a read-only master. `prepare --arm warm --argv_key graphpc`
takes the subset as `--task_ids`, the master as `--frozen_master <absolute path>` — a copy of the cold arm's
`--support_set_root`, made once after the cold arm ends and never written again — and its hash as
`--master_sha256`; each lane runs one
`run_graphpc.py` per task, restoring its working store byte-identical from the master before every episode and
checking the master's hash before and after. The hash is the driver's own (SHA-256 over the master's
`webarena/*.json` entries, each file name then its bytes, in name order); print it, with the entry count, by

```bash
python -c "import sys, pathlib; sys.path.insert(0, 'scripts/webarena'); from run_lanes import _sha256_tree; print(_sha256_tree(pathlib.Path('<frozen master>')))"
```

Run the launch preflight in warm mode first (below), with the cell's own strata file, master and entry count.

### Site table and launch preflight

- **Site variables are required.** The GraphPC factory resolves the control's own site table once, at construction:
  `SHOPPING`, `SHOPPING_ADMIN`, `REDDIT`, `GITLAB`, `WIKIPEDIA`, `MAP` and `HOMEPAGE`, read exactly as
  `browser_env/env_config.py` reads them (`src/benchmarks/webarena/url_mappings.py`). An empty value aborts before
  the first episode, so launch with the repo `.env` sourced.
- **Launch preflight.** `scripts/webarena/launch_preflight.py` gates every scored GraphPC launch; it prints one line
  per check and exits 1 on any failure. Its retrieval mode is passed with the runner's own spelling:
  - `--retrieval_off --result_dir <dir> --working_store <the run's store>` — the bypass is active on both sides (no
    store read, no store write);
  - `--cold_store --result_dir <dir> --working_store <the empty store>` — retrieval on from an empty store: the
    embedder is live, the store is empty and writable;
  - warm (no mode flag): `--result_dir <dir> --working_store <a restored copy of the master> --frozen_store_root
    <the frozen master> --expected_entry_count <its entry count> --strata_file <the cell's strata file>` — the
    embedder is live, the injection diet is enforced on a probe entry, the working store's hash equals the master's,
    and the strata file is consistent and its B1 tasks' families are covered by the master;
  - pass the arm's `--backbone` and `--model` in every mode;
  - `--site_mapping_live --result_dir <dir> --live_config_file <WebArena>/config_files/<id>.json
    --homepage_app_root <the served homepage app's directory> -- <the cell's run_graphpc.py argv>` (WebArena
    interpreter, live sites, site variables set, no model call) drives the cell's own build against the live sites
    and checks:
    - the alias `URL:` line and the Homepage paragraph are rendered;
    - a mapped `goto` lands on the local host;
    - our table, our mappers and the paragraph equal upstream's;
    - the homepage templates are unchanged.

### Workflow shapes + gates

Default is single-pass with the completion-gate **on**. There is no progress-gated terminate: the max-steps cap is the sole budget bound (`design.md` §6). To switch shapes / toggle the gate for ablation:

```bash
# Two-pass (a backup design, not the default)
… --two_pass

# Ablation: completion-gate off
… --completion_gate_off

# Completion-gate answer-grounding obs-recency window
… --completion_gate_window 5
```

### Dry-run (no WebArena checkout needed)

The runner is exerciseable without a WebArena checkout. The agent is constructed against a stub orchestrator + stub action factories; the loop walks a synthetic trajectory against a stub env that mirrors the real WebArena call shape (`reset(options=)` → 5-tuple `step` → evaluator with `page`/`client`) and the stub agent emits a terminal stop within the step budget, so the stop-break path is exercised. It verifies the boundary contract:

```bash
python scripts/webarena/run_graphpc.py --model fake-model --dry_run \
  --result_dir ./wa_dryrun_results
```

This is the pre-deployment smoke. Dropping `--dry_run` against a real WebArena checkout is the live end-to-end form of the same check (the end-to-end check of the boundary in `design.md` §12).

## Subsets

- **WebArena-Lite (165 tasks, human-verified; introduced with VisualAgentBench, Liu et al., ICLR 2025, and used by WebRL as its evaluation split)** — the set for the retrieval-off cells (`experiments.md` comparison matrix); its site plan is `data/webarena/lite165_site_partitions.json`.
- **WebArena full (812 tasks, 190 intent templates, ~4.3 instantiations each, across 6 sites)** — full benchmark. Use only for final / publication runs; its lane partitions are `data/webarena/full812_site_partitions/`.
- **WebArena-Verified (ServiceNow)** — a separate human-verified re-annotation of the full set; cleaner eval at full size.

### How transfer is measured

Every full-812 cell runs the same two-round shape: a **cold** arm over all 812 tasks with an empty store, then a **warm** re-run of a pre-registered subset against the finished store, read-only. The subset is stratified by intent-template family — family-cold losers are the primary stratum (reverse flips impossible there by construction), openers are the churn reference.

**Deriving the subset** (pre-register it before the warm launch; no tool ships for it). Over the cold run's scored rows, with families keyed on `intent_template_id`:

- `A` = tasks whose scored episode started with fewer than 20 entries in the store;
- `fam_won` = the task's family has at least one cold GraphPC win;
- `sib_win_before` = a same-family sibling won cold in an episode that started earlier (start to start);
- `B1` = `fam_won` and not `sib_win_before`; the subset is `A ∪ B1`.

The three strata partition the subset: `b1_pre_opener_losers` = `B1` tasks lost cold (the primary endpoint: they failed with no sibling recipe in the store), `b1_openers` = `B1` tasks won cold (the churn reference), `a_only` = `A` minus `B1`. The lane logs carry no timestamps: an episode's end is its `telemetry_<id>.json` write time, its start is estimated as the previous episode's end in the same lane (a large gap marks a restart), and the store size at an episode's start counts the entries whose `created_at` precedes it, so an episode never counts its own entry.

**The strata file** the launch preflight reads (`--strata_file`) is one JSON object:

- `task_ids` (the subset) and `n_subset` (its size);
- `strata`: `{"b1_pre_opener_losers": [...], "b1_openers": [...], "a_only": [...]}` — disjoint, their union exactly `task_ids`;
- `stratum_sizes` (optional): the declared size of each stratum;
- `rows`: one `{"task_id", "stratum", "family"}` per subset task;
- `master_table`: one `{"task_id", "intent_template_id"}` per cold-run task, which the families are checked against;
- `recipe_eligibility` (optional): `{"eligible": [...], "premise_failed": [...], "eligible_count": n}`, disjoint and inside `task_ids`.

The preflight checks it for internal consistency and that every `B1` task's family has an entry in the frozen master.

A template-disjoint re-split (holding whole `intent_template_id`s out) is not part of the protocol: the family-stratified warm round answers the transfer question inside a single cell.

## Baseline

**One baseline per model: the same-model bare control.** WebArena's bundled `PromptAgent` — `obs → single LLM call → action`, no scaffolding — running on the **same model GraphPC uses**, evaluating whether GraphPC lifts that model above its no-scaffolding ceiling. This is the core ablation. The control is WebArena's own `run.py` loop (`prepare` / `construct_agent` / `test`, called verbatim) behind our wrapper `scripts/webarena/run_baseline.py`, which installs the bridge that runs WebArena's openai-0.27 call sites under openai 2.x and the environment hardening the GraphPC runner installs too (`_env_hardening.py`, below) and builds the task list; WebArena stays read-only. Run it with the WebArena env interpreter, from the WebArena checkout, with the regime flags of the "Harness regime" table below:

```bash
cd ../WebArena
python <repo>/scripts/webarena/run_baseline.py --model <same-as-GraphPC-run> \
  --task_ids_json <ids.json> \
  --result_dir <dir> <regime flags>
```

`--task_ids_json` is a file holding `{"task_ids": [<int>, ...]}`.

Locked **per model** when the model is introduced, and never re-run; record the lock in `project_state.md`.

**Locked-baseline policy.** **Lock prerequisite: the run must be `clean`** — no contamination from runtime errors. A baseline run is **contaminated** if any of:

- ≥3 task `retry-exhausted` (API rate-limit / repeated 5xx),
- ≥10% of steps killed by timeout / env crash,
- any task required docker/browser restart mid-run.

**Contamination triggers are diagnose-first, not stop-first (the same rule holds on OSWorld 2.0; policy: `experiments.md` Hard constraints).** Reaching one, including ≥3 retry-exhausted, does not by itself stop an arm or condemn a run: it starts a diagnosis of whether a defect corrupts the result, and only then is the handling decided. A transient provider fault (an upstream 429 storm exhausting a retry chain) or a harness-side scoring failure (the evaluator asserting on the judge's wording) costs individual tasks, which are unscored and re-runnable; neither touches what the model saw or what the site state became. Do not stop a run to begin a diagnosis that can be made while it continues.

A contaminated run — one where the diagnosis finds a defect that corrupts the result — yields a **provisional** lock — re-run with the same backbone + config until a clean run is captured; the clean run is what locks. Once locked, every later GraphPC run computes Δ against the locked clean numbers (never re-run, even if backbone pricing or API behavior shifts — re-running silently changes the control).

### Per-backbone route and launch facts

What follows is what a run on each route needs.

**DeepSeek and MiniMax.** Each thinking shim (`_inject_deepseek_thinking` / `_inject_minimax_thinking`, in `src/benchmarks/webarena/_openai_compat.py`) enables thinking but does NOT strip WebArena's `temperature=1.0` / `top_p=0.9`, and `run.py`'s `--max_tokens` defaults to 384. Under the regime table below (thinking on, temperature/top_p sent unchanged) neither DeepSeek nor MiniMax rejects the sampling parameters, so no param-drop mirror exists for them; `gpt-5*` is the family that 400s under thinking, which `_adapt_reasoning_model_params` handles. **384 is unsafe on any thinking model**: reasoning tokens fold into `completion_tokens`, so the budget is shared with the reply; on MiniMax, 384 returns `finish_reason=length` with empty content.

On the MiniMax control route, reasoning must arrive outside `message.content`: an inline `<think>` would let WebArena's action regex pick a backticked action out of the reasoning — a silently wrong action, not a no-op.

### DeepSeek `deepseek-flash`

- **Control shim** (`_openai_compat.py`): exact id `deepseek-flash` (other DeepSeek ids take the default route, unmodified), effort `high`, and a provider-interrupted finish set (`insufficient_system_resource`, `aborted`) that marks a failed call. The control launches with `--max_tokens 65536`, the GraphPC deciding call's effective cap.
- **GraphPC arm**: deciding with thinking + `json_object`, no cap; grounding non-thinking with a forced tool, `strict` sent but not enforced.
- **Pass `--model deepseek-flash` explicitly on every launch, both arms.** The GraphPC backbone's own default id is `deepseek-v4-flash`, and the response validation and the effort pin apply to `deepseek-flash` only, so a launch that leaves the model to a default silently runs a different setup.
- **Metering**: one `deepseek usage:` line per call on both arms (route, thinking and effort sent, reasoning tokens, finish reason, model echo, `system_fingerprint`).

### MiniMax `MiniMax-M3`

- **Route, both arms**: MiniMax's OpenAI-compatible endpoint `https://api.minimax.io/v1`, key `MINIMAX_API_KEY` (read at call time and passed explicitly), model id `MiniMax-M3`; the control shim routes every `minimax`-prefixed id there (case-insensitive). Thinking is binary — `{"type": "adaptive"}` on, `{"type": "disabled"}` off, no effort scale — and is pinned explicitly in `extra_body` together with `reasoning_split: true`, which delivers the reasoning in `reasoning_content` / `reasoning_details` and keeps `message.content` clean.
- **Control shim** (`_inject_minimax_thinking`): when `--reasoning_effort` is set (it defaults to `high`), every control call carries `{"thinking": {"type": "adaptive"}, "reasoning_split": true}`; the effort value itself is ignored. WebArena's `temperature=1.0` / `top_p=0.9` are sent unchanged. Launch with `--max_tokens 16384`. Each call logs a `minimax usage: prompt=… completion=… reasoning=… cached=…` line; MiniMax has no balance endpoint for pay-as-you-go keys, so run cost is metered from these lines.
- **GraphPC arm** (`src/agents/backbone/minimax/main.py`): deciding = thinking + `response_format: json_object` (MiniMax ignores a `json_schema` response format, so the schema rides in the prompt and closed enums are normalised after parsing); grounding = non-thinking + a forced function call with `strict: true`, whose `pattern` MiniMax does not enforce server-side (the grammar is held by the action prompt and the translator's degrade path). No sampling parameter is sent on the thinking route, and no output cap (`max_tokens` None). Each call logs a `minimax usage:` line.

### Qwen `qwen3.7-plus` over OpenRouter

Model `qwen3.7-plus`, the plus tier (not the family's top tier).

**The route is OpenRouter, and only OpenRouter.** Both arms reach `qwen3.7-plus` through OpenRouter's OpenAI-compatible API; the code carries no other Qwen route — no fallback host, no base-URL override variable, no unused code path. Each arm writes the route literals out for itself (the control arm keeps a standalone import graph on purpose: it is the control, so it must not import our agent code to share a constant), so the two copies must equal the table below: keep them in step by hand when either side is edited.

**Pass `--model qwen/qwen3.7-plus` on both arms.** The control shim refuses any other id on this route, and it is the
GraphPC backbone's own default.

| literal | value |
|---|---|
| host | `https://openrouter.ai/api/v1` |
| credential | `OPENROUTER_API_KEY`, read at call time and passed explicitly (under a custom base URL the openai SDK would otherwise fall back to `OPENAI_API_KEY`) |
| model slug sent | `qwen/qwen3.7-plus` |
| route pin, every request, both arms | `{"only": ["alibaba"], "allow_fallbacks": false, "require_parameters": true}` |
| reasoning ON — deciding call and control arm | `{"enabled": true}` |
| reasoning OFF — grounding call | `{"effort": "none"}` |
| endpoint cap ceiling | 131,072 |
| control-arm launch cap | 16,384 |

**Provider facts**, as read from OpenRouter's model catalog, endpoint listing and documentation (re-read them before a run: a catalog can change).

- **Exactly one endpoint**: provider tag `alibaba` (display name `Alibaba`), context 1,000,000, `max_prompt_tokens` 983,616, `supports_tool_choice.function: true`.
- **The cap, unambiguously**: the CATALOG field is named `max_completion_tokens` and its value **131,072** is the ceiling; the REQUEST parameter this endpoint advertises is `max_tokens`. Assert against the former, send the latter, never send the former as a parameter. The shim's `max_tokens` → `max_completion_tokens` rewrite is gated on a `gpt-5` prefix and never fires for a qwen slug.
- **Routing defaults to fallbacks ON plus price-based load balancing** — which is why the route pin is explicit and is injected unconditionally, merged into whatever `extra_body` a caller supplied. An unpinned request still returns 200; it is simply served by an endpoint the cell never measured, so a skipped pin is a silent measurement failure. `require_parameters` is deliberate: it turns "this endpoint no longer supports `tools` / `response_format`" into a 503 rather than a silent parameter drop. The pin is falsifiable, never silent — an unavailable pinned endpoint returns **503**, a conflict with an account-wide allowed-providers ceiling **404**, and both classify as infra in `run_graphpc.py`.
- **Reasoning control**: a per-call `reasoning` field; this model reports `mandatory: false`, `default_enabled: true`, no effort scale and no token budget. `{"effort": "none"}` disables reasoning (`reasoning_tokens` present and 0, `message.reasoning` and `reasoning_details` empty, and a forced named function still honoured, which cannot happen under thinking), and `{"enabled": true}` enables it. `exclude: true` and `include_reasoning: false` are **NOT** a disable: the model still reasons, the tokens are still billed and still count against `max_tokens`, and the only evidence it happened is removed. A bare `{}` defers to `default_enabled`, a provider default this cell does not accept. All three are forbidden in both arms.
- **Thinking lands off `content`**: on `choices[].message.reasoning` (string|null) and `choices[].message.reasoning_details` (a typed array), so there is nothing to strip. If a `<think>` block ever appears inside `content`, report it — never add a stripper.
- **Reasoning is billed as output AND bounded by `max_tokens`**: at a tiny `max_tokens` the reply comes back `finish_reason=length` with the whole budget spent on reasoning and empty content. `reasoning_tokens` sits INSIDE `completion_tokens`, and `prompt + completion == total`.
- **Usage is always returned** (`usage: {include: true}` is a deprecated no-op) and carries `cost`, `cost_details`, `completion_tokens_details.reasoning_tokens` and `prompt_tokens_details.cached_tokens`. There is **no `system_fingerprint`** field on this route, so the metering lines carry `cost=` in its place.
- **Route identity is NOT on a successful response**: which endpoint actually served a call is visible only when the request sends `X-OpenRouter-Metadata: enabled`, which production requests do not send. Per call, the response's `model` echo is the route record.
- **Model echo**: the response echoes **the slug that was sent** (`qwen/qwen3.7-plus`), not a canonical dated form. The echo therefore detects a RENAME rather than a silent re-point of the alias, and it rides on every metering line as `model=`.
- **Forced tool calling**: `tool_choice: {"type": "function", …}` is supported, and `strict: true` is accepted at the API edge with no 400 — but its provider semantics are unspecified, so it is a hint: returned arguments are validated on our side, as on every route. A 400 naming `tools` / `tool_choice` STOPS the cell and is reported, never worked around in the moment.
- **A 200 can carry a failure**, in two documented shapes: a body with an `error` object and NO `choices` key at all, and a `choices[0]` holding partial content plus its own `error` and `finish_reason: "error"`. Every response is validated before use on both arms (see the refusal boundary below); without the guard, an empty `length` reply would reach PromptAgent's parser on the control arm and become a scored 0.0.
- **Rate limits**: no platform RPM cap on paid models; upstream 429s still occur. Nothing lowers the openai SDK's 600 s default timeout, which is the ≥180 s client-timeout property the cell relies on.
- **Price**: the per-token price triples in BOTH directions once the prompt exceeds **256k tokens** (the cost tripwire below watches for it); check OpenRouter's current list prices and its credit-purchase fee before a run.
- **Vision**: modality `text+image->text`, a base64 `image_url` content part. Per-image token accounting is not published for this endpoint; images bill at the input price.

**Cap regime — a control arm launches with `--max_tokens 16384` (65536 on `deepseek-flash`, above).** The cap bounds chain-of-thought and reply together, and 16384 ≤ the Qwen endpoint ceiling 131,072. The cap is fixed for the whole run and for both arms of a cell, because a control whose regime changes mid-run is inhomogeneous against the arm it is compared with; results across base models are not comparable. The GraphPC arm sends NO cap and therefore inherits the endpoint's own default; a GraphPC deciding call that ends on `length` is a truncated reply. `run.py`'s 384 default remains unsafe: a reply cap that small truncates the answer itself.

**Harness regime — assemble a control arm's launch command from this table** (flags of `scripts/webarena/run_baseline.py`). It is the tracked statement of the control regime; no launch-time gate checks it.

| flag | value | why it matters |
|---|---|---|
| `--parsing_failure_th` | **999** | **NOT a default — the default is 3.** WebArena's bundled agent terminates a task after 3 consecutive unparseable actions. |
| `--repeating_action_failure_th` | **999** | **NOT a default — the default is 3.** Same mechanism for 3 consecutive repeated actions. |
| `--max_tokens` | 16384; 65536 on `deepseek-flash` | the cap regime above (`deepseek-flash`: the control's cap equals the GraphPC deciding call's effective cap) |
| `--max_steps` | 30 | full budget (`run.py`'s own default) |
| `--env_chunk_size` | 15 | fresh-env lifecycle boundary, the driver-heap-OOM mitigation (already the default) |
| `--log_level` | INFO | the `<route> usage:` metering lines are emitted at INFO; anything coarser silently degrades the cost record to an estimate (already the default) |
| `--reasoning_effort` | high | gates the shim's reasoning injection; already the default, so forgetting it does not unpin reasoning |
| runner env | the WebArena conda env | the checkout plus the openai 2.x stack the shim needs |
| lane layout | site-pure lanes first, then the cross-site phase, **Lite members first within each lane** | `experiments.md` run protocol, item 4; the shipped partitions follow it |

**The two 999s are the item to get right, and the reason is within-cell fairness, not comparability.** The GraphPC arm has **no early-stop mechanism at all** — `run_graphpc.py` drives its own loop with `--max_steps` as the only exit — so leaving the control arm at the default 3 gives the two arms of the SAME cell different budgets, and a task killed that way is indistinguishable in the scores from one the model genuinely failed. So every control arm runs 999/999.

**Account balance.** A 402 (balance exhausted) returns on every later call, so it halts the run rather than degrading it; the lane driver's 402 tripwire freezes the arm at the first one. Fund the account before launching.

**Smoke, before any scored Qwen run** ("Smoke both arms before a scored run" above): every `qwen usage:` line on both arms carries `model=qwen/qwen3.7-plus` and `cost=`; reasoning tokens are non-zero on every control and deciding call and zero on grounding calls.

**Alias-move policy.** Every `qwen usage:` line carries `model=`. A model move mid-arm contaminates that arm under the locked-baseline rules; a move between arms breaks the pairing, and the cell re-runs from the arm that moved. The run watches for it with the lane driver's tripwires (`scripts/webarena/run_lanes.py`, the rules `prepare --tripwires_json` writes into the run's plan): a `change` rule on the `model=` value, which alerts on each new value other than its `baseline` (or, without one, the first seen), and beside it the cost rules (`output=` approaching the launch cap; `input=` approaching 256,000 tokens, where the price triples in both directions).

**Three route-specific facts a Qwen run record carries.**

1. **Per-attempt metering.** Both arms emit a `qwen usage:` line for every billed ATTEMPT, including a retried one and a billed partial, so `call_count` and per-episode call counts count attempts.
2. **The request dicts differ from the other routes' by the provider pin as well as by model and reasoning dialect.** The pin is a ROUTING DIRECTIVE, not a sampling change: it selects which endpoint serves the request and forbids a silent fallback. It does not touch the prompt, the decoding parameters or the schema.
3. **The refusal boundary, identical on both arms.** A bare `finish_reason: "content_filter"` is a COUNTED outcome: on this route a model's own refusal comes back as a COMPLETED response carrying `message.refusal`, and this endpoint reports `is_moderated: false`, so a provider-side moderation block is not the mechanism — it is the model's behaviour on a scored step. It flows down the normal path (the GraphPC arm degrades ONE counted step when the refusal will not parse) and is logged so it is never silent. A `content_filter` that ALSO carries a choice-level `error` envelope is a FAILED call and raises. The same line runs through `length`: with NON-empty content it is a truncated reply and a counted degradation; with empty or whitespace-only content it is a failed call. Widening either half would silently take real degradations out of the denominator.

**Method surface of the GraphPC arms on WebArena** (`design.md` is the authority): orchestrator `max_actions` 1, two-pass off, completion gate on with window 3, reasoning trajectory 3 and `model_call_attempts` 1, retrieval cap 2 with `injected_history_diet` None and the store's relevance gate at the FIXED 0.15 (`rule="fixed"`), monitor 32/64 with no pointer locator, one shared backbone instance, `max_tokens` None. These are identical across gpt / deepseek / minimax / qwen (claude: caps 32000 / 4096, below); the request dicts differ only in `model`, in the provider's reasoning dialect inside `extra_body`, and on Qwen in the `provider` route pin. The deciding request carries the control's `URL:` line and Homepage paragraph, and the grounded action is alias-mapped (`url_mappings.py`). The arms' thinking regimes differ by design: the control thinks on its single call, the GraphPC arm on the deciding call and not for grounding.

### `gpt-6-sol` (OpenAI)

What a run on this route needs:

- **Model id is an alias only** — no dated snapshot and no fingerprint on the response, so a silent re-point cannot be detected; record the provider's model entry (`GET /v1/models/gpt-6-sol`) at each arm boundary and disclose the limit.
- **Control arm** (Chat Completions through `_openai_compat.py`): the gpt-5 parameter adaptation also matches the EXACT id `gpt-6-sol` (not a `gpt-6` prefix) — `max_tokens` → `max_completion_tokens`, `temperature`/`top_p` dropped under effort, `reasoning_effort` injected. `run_baseline.py --reasoning_effort` defaults to `high` and the value IS forwarded on this route, so pass it explicitly. `max` effort is rejected on Chat Completions (`xhigh` is the ceiling).
- **GraphPC arm** (Responses API): `--reasoning_effort` sets the deciding call; `--grounding_reasoning_effort` (gpt backbone only; default: the `--reasoning_effort` value) sets grounding; the JSON candidate is parsed from the answer text only, never from reasoning summaries.
- **The cell's regime**: `--reasoning_effort medium` on both arms (the control's `run_baseline.py` and the GraphPC deciding call) and `--grounding_reasoning_effort none` on the GraphPC arm. Both runners default to `high`, so pass all three explicitly.
- **Metering.** Both arms emit a per-call `gpt usage:` line carrying the effort sent (and, on GraphPC, the effort the server echoes); read it after the first episode of every invocation and at arm end.
- **Cookies go idle-stale.** The sites expire idle sessions server-side. The control logs in per task into temp dirs and never reads `.auth`; the GraphPC runner reads `.auth` every episode and never re-logs in — so re-render `.auth` before every launch after the first phase, and only when no runner is alive (a refresh while lanes read the files races them and can lose tasks). The logged-in check for Magento admin is `/admin/dashboard/` (auto_login's own URL 404s for a logged-in admin).
- **Known arm asymmetry:** WebArena's keyboard map cannot type non-ASCII, so a control that types `—` / `’` crashes the episode unscored, while GraphPC's strict grounding schema keeps typed text ASCII.

### Claude Sonnet 5 (Anthropic first-party)

What a run on this route needs:

- **First-party API only, native Messages API on both arms** (never an OpenAI-compatible endpoint). The model id is
  the alias `claude-sonnet-5` — no dated snapshot, no fingerprint; record the models-API entry
  (`GET /v1/models/claude-sonnet-5`) at each arm boundary. The key (`ANTHROPIC_API_KEY`) is read at call time; both
  launchers check it before any task.
- **The cell's regime**: `--reasoning_effort high` (the runners' default) on both arms: adaptive thinking on the
  control and the deciding call, grounding with thinking disabled.
- **Control arm** (`_openai_compat.py`, exact id `claude-sonnet-5`): PromptAgent's intro + few-shot
  system messages are joined into ONE system string (the `name` labels are lost — a disclosure);
  `temperature` 1.0 kept (the only value accepted), `top_p` dropped (any value is a 400), `max_tokens`
  from `--max_tokens` (16384); adaptive thinking + `effort` from `--reasoning_effort`; the answer text
  only is handed back, never thinking. Every failure raises an `openai.APIError` subclass — a `ValueError`
  would be SCORED by `run.py`'s `except ValueError`.
- **GraphPC arm** (`src/agents/backbone/claude/main.py`, shared reasoning/action agents): deciding =
  adaptive thinking + effort, JSON requested with MiniMax's suffix verbatim (no server-side schema);
  grounding = thinking disabled + a forced tool on the UNCHANGED shared schema with `strict` off, the
  MiniMax/Qwen regime — so, like those arms, there is **no decode-time ASCII guard** (a server pattern
  would rewrite stop answers, e.g. "Norén" → "Noren"). Caps 32000 deciding / 4096 grounding, `timeout=600`
  on both arms (the SDK refuses non-streaming caps above 21,333 without one). The runner refuses
  `--temperature` / `--top_p` / `--reasoning_effort none` for claude.
- **Effort is not echoed** by the API: it is verified as sent, on the per-call `claude usage:` lines.
- **Billing errors** (`429 enforced_spend_limit_reached` — an organization spend cap, which OSWorld's evaluator also
  draws on when it uses the same key — the user-limit 400, 402) are ONE infra class on both arms: unscored and
  re-run (still unscored after the allowed re-runs → 0.0), and a tripwire freezes the arm.
- At `high` the GraphPC deciding call produces long outputs; plan more wall-clock than for the control.
- Without `strict`, grounding format is not enforced: empty-text and multi-line `type`, replies without `action_strings` and non-ASCII keystrokes each cost a counted no-op step; a stricter guard is worth having once a pattern that spares stop answers is expressible.

## Outputs

The runner writes under `--result_dir`: per live invocation an immutable `summary_<UTC timestamp>.json`, every invocation's rows merged by `task_id` into the canonical `summary.json` (a dry run writes `dry_run_summary.json`), one row per task `{config_file, success, score, steps, errored, error_type, task_id}` — `errored` rows (`error_type` `eval_error` or `infra_error`) are unscored and re-run (a task still unscored after its allowed re-runs counts 0.0, `experiments.md` Verdict methodology), while an `agent_error` is a counted 0.0; `run_config.json` (the resolved launch config: argv, the switch state read off the built agent, the retrieval text arm); and per task `telemetry_<task_id>.json`. Per-episode telemetry: `reasoning_parse_failures` / `grounding_parse_failures` count unparseable ATTEMPTS (= `*_resamples` + `resample_exhausted_*`). Successful trajectories + graph artifacts are persisted by the agent's `finalize_task` to the support set under `--support_set_root/webarena/` (a separate backend from the OSWorld store).

**Scoring a cell.** `run_baseline.py` assembles the same `summary.json` row shape from `run.py`'s `[Result]` lines, so
both arms are read the same way. In a lane-driver run each lane's rows are under `<run dir>/lanes/<lane>/results/`
(GraphPC: `summary.json` merged with every `summary_<UTC timestamp>.json`, later rows winning); for the control the
per-task verdicts are also the `[Result] (PASS|FAIL)` lines of the lane's `run.log`, which is what
`run_lanes.py status` counts, with per-lane and total scored / passed counts. Pair the two arms' rows by `task_id`
over the cell's task set; a task passes when `score > 0`. A row with `errored: true` is unscored: re-run it
(`casualties`, then `add_lane`), and count a task still unscored after its allowed re-runs as 0.0 for that arm
(`experiments.md`, Verdict methodology). The primary statistic is the discordant-pair table with an exact McNemar test
(`design.md` §7); no paired scorer ships for WebArena.

## Adapter architecture

(Reference for anyone modifying the adapter or porting to a new benchmark. For the high-level boundary principle, see `design.md` "Benchmark integration boundary".)

**`WebArenaGraphPCAgent` is the translation layer — mirrors `OSWorldGraphPCAgent` inside, WebArena-shaped outside.**

### WebArena code reused as-is (external, read-only)

- `browser_env/` — Playwright environment (`ScriptBrowserEnv`), observation processors, and action execution + the id-based action grammar. Canonical action parser: `create_id_based_action()` (`browser_env/actions.py`). Accepted actions: `click[id]`, `hover[id]`, `type[id][text][flag]` (flag 0/1 = without/with Enter; defaults to `[1]`), `press[key_comb]`, `scroll[direction]`, `goto[url]`, `go_back`, `go_forward`, `new_tab`, `tab_focus[index]`, `close_tab`, `stop[answer]`.
- The run loop in `run.py` (reset → loop[agent `next_action` → env.step] → evaluator → finalize). Reused via the vendored runner. The runner (`run_episode_live`) mirrors WebArena's own contract, which a deployer should confirm against the pinned checkout: `env.reset(options={"config_file": <path>}) -> (obs, info)`; `env.step(action) -> (obs, reward, terminated, truncated, info)` (**5-tuple**); on an agent `ActionTypes.STOP` action the loop **breaks WITHOUT stepping the env**, leaving the stop action as the trajectory's last element; on env `terminated` it appends a `create_stop_action("")` as the final element; the evaluator is called as `evaluator(trajectory=, config_file=, page=env.page, client=env.get_page_client(env.page)) -> float`. Our `--max_steps` cap is an extra guard on top of these exits.
- `evaluation_harness/` — evaluators (string-match / url-match / program-html) + per-task config JSONs (intent + eval criteria). The LLM judge (`llm_fuzzy_match` / `llm_ua_match`) hardcodes the now-retired `gpt-4-1106-preview` model; since WebArena is read-only we remap that exact id to `gpt-4-turbo-2024-04-09` (the closest reachable successor) in our `_openai_compat.py` shim, overridable via `WEBARENA_JUDGE_MODEL`. MEASUREMENT CAVEAT: any judge ≠ `gpt-4-1106-preview` shifts scores vs published WebArena numbers; unavoidable now that 1106-preview is retired, and the default minimizes drift.
- **Two runtime patches on `browser_env/`**, from our `_env_hardening.py` (below), installed identically by both runners: stock WebArena loses the episodes they rescue as `[Unhandled Error]`. MEASUREMENT CAVEAT: absolute numbers on flaky-page and tab-opening tasks can sit above published upstream runs; the paired Δ is de-confounded but not guaranteed untouched, because the page-client patch fires at a rate that depends on how often the agent opens tabs.

### Our thin shim (`src/benchmarks/webarena/`)

- **`graphpc_agent.py:WebArenaGraphPCAgent`** — subclasses WebArena's `Agent` (`agent/agent.py`), exact signature: `next_action(self, trajectory, intent, meta_data) -> Action` + `reset(self, test_config_file) -> None`. Mirrors `OSWorldGraphPCAgent`'s structure (obs → `Observation` → call inner boundary → `ActionBatch` → benchmark action → persist on success).
- **`observation_adapter.py`** — WebArena observation → our `Observation`. Reads the current state from `trajectory[-1]` (a `StateInfo` TypedDict `{observation, info}`): `observation["text"]` is the accessibility-tree string (default) or `observation["image"]` the screenshot ndarray; choices are `accessibility_tree` / `html` / `image` (no `som` in base WebArena — that is VisualWebArena). The element-id ↔ a11y-node mapping for resolving action target ids lives in `trajectory[-1]["info"]`.
- **`action_translator.py`** — our grounded `ActionBatch.actions` → a WebArena `Action`. **Never hand-build the `Action` TypedDict**: emit the id-based action string (e.g. `"click [5]"`, `"type [3] [hello] [1]"`) → call `create_id_based_action(action_str)`. Stop → `create_stop_action(answer)`; no-op → `create_none_action()`.
- **`a11y_parser.py`** — WebArena's AXTree parser with relevance selection: it parses WebArena's own tree format and, past the node cap, keeps the highest-relevance nodes using the scoring helpers in `src/agents/graph/a11y_parser.py` (`score_node`, the focused-window heuristic, the stable node id).
- **`action_agent.py`** — the WebArena action emitter (`WebArenaActionAgent`), shared by every backbone; its building blocks (the strict output schema of id-based action strings with the grammar pattern, the action system prompt, `build_action_batch`) are in **`_action_common.py`**. The action stage's biggest porting cost — the action-space coupling flagged in the "Benchmark integration boundary" payload risk.
- **`url_mappings.py`** — verbatim copies of upstream's site table (`browser_env/env_config.py`) and of `map_url_to_real` / `map_url_to_local` (`agent/prompts/prompt_constructor.py`), at WebArena dce0468. One table object feeds both directions:
  - the observation adapter appends the control's `URL: <alias URL>` line after the tree, in `a11y_tree`; a missing page URL raises `MissingPageUrlError`, which the runner classifies as `infra_error`;
  - the action agent maps alias hosts back to local over the whole action string in `build_action_batch`.
- **`prompts.py`** — `WEBARENA_DECIDING_INSTRUCTION_SUFFIX`: the shared terminal contract followed by the control's "Homepage" paragraph, verbatim.
- **`na_normalizer.py`** — the adapter-layer `N/A` dialect for infeasible-task answers, over the shared matcher in `src/agents/common/infeasible_declaration.py`; precision-first, because a false-positive normalization would destroy a win.
- **`_openai_compat.py`** — the openai-2.x bridge for WebArena's retired 0.27 call sites, plus the per-provider thinking/route shims the control arms run through (`_adapt_reasoning_model_params`, `_inject_deepseek_thinking`, `_inject_minimax_thinking`, the OpenRouter route, the native Anthropic branch for `claude-sonnet-5`) and the judge remap above.
- **`_env_hardening.py`** — the two environment patches both runners install (`run_graphpc.py`, `run_baseline.py`), runtime only, with no WebArena file edited: a bounded re-fetch of the observation (`TextObervationProcessor.process`, up to 3 attempts on the `ZeroDivisionError` / Playwright `TimeoutError` of a page still settling; the last failure re-raises unchanged) and the on-demand recreation of a missing `page.client` CDP session (a tab or popup that becomes the active page without one). Both count their firings for telemetry (`OBS_RETRY_STATS`, `PAGE_CLIENT_STATS`).
- **`_webarena_imports.py`** — centralizes lazy imports of WebArena types. The package loads cleanly without a WebArena checkout.

### Decoupling invariant

The shim/runner calls our agent **only** through the predict/reset surface and the `schema.py` types; it **never** reaches into orchestrator internals. Check it with a strict-stub orchestrator whose `__getattribute__` throws `AttributeError` on any non-boundary access: driving a short canned trajectory through the shim against the strict stub passes if and only if the shim has zero dependency on agent internals.
