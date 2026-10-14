# OSWorld 2.0

**The second benchmark is OSWorld 2.0** — `xlang-ai/OSWorld-V2`, "Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks": 108 tasks across 7 domains / 21 sub-categories, 31 self-hosted websites plus desktop apps, 500-step budget, binary-completion and partial-score metrics; task classes are the gated HuggingFace dataset `xlangai/osworld_v2_tasks`; the harness needs Python ≥ 3.12 + `uv`; Docker and AWS images are provided.

**The cells.** Three cells — MiniMax-M3, Qwen `qwen3.7-plus` and DeepSeek `deepseek-flash` — each a same-model pair
on the pre-registered 28-task subset. Each launch writes to its cell's default result trees (MiniMax:
`graphpc_minimax_m3` with its store `graphpc_minimax_m3_support_set`, and `baseline_minimax_m3`; the other cells':
"What differs per cell" below), and the launch guard refuses a launch into another cell's trees or the other arm's.
To protect a finished tree from every later launch, rename it on the host and add it to `OSWORLD_LOCKED_PATHS` in
`osworld_cell_config.sh` (empty as shipped); the guard refuses a launch into a locked tree as well.

**Deployment shape.** A separate long-lived AWS host instance runs the runner and provisions per-task EC2 VMs via `--provider_name aws`. Its benchmark checkout is pinned to one release (`osworld-v2-2026.08.08` — the release name; git tag `v2026.08.08`, commit d578d2d). **No fork.** The vendored snapshot travels from the dev machine to the host over SSH (`rsync`); nothing is committed to any OSWorld repository, and the sync target is a local OSWorld-V2 clone used only as a staging area. The adapter lives under `src/benchmarks/osworld/`; runners and launch scripts are copies of the `v2026.08.08` files; the paired scorer (`scripts/osworld/tools/score_paired.py`) is our own and reads the per-task files the V2 driver writes. The run host and the staging clone must be set up (the one-time setup below) before any run.

The flow is: **sync vendored snapshot into the local staging clone → rsync it to the host checkout over SSH → invoke the runner there → score results**.

**Migration rule — copy first, adapt minimally (AGENTS.md working principle; in force here).** Every file we vendor into the checkout is a copy of OSWorld-V2's own code at the pinned tag with the smallest possible edit, so that it runs as-is once rsynced: the GraphPC runner (one for all three cells) is a copy of `scripts/python/run_multienv_m3.py` whose deltas are exactly those listed in its header — the header is the authority, not this document, and its numbering has gaps (numbers 6, 7, 13 and 14 are unused). Delta 15 is the startup refusals: (a) any `--max_steps` other than the adapter's budget for the `--backbone` (`OSWORLD_MAX_STEPS_BY_BACKBONE`: 500; 800 on the Qwen profile) — the budget the completion-gate rules are written against, so a short smoke cannot lower `--max_steps` on this runner; (b) any launch without `--retrieval_off` (every GraphPC run on the subset runs retrieval off). Each cell's control has its own vendored runner (MiniMax: below; Qwen and DeepSeek: "What differs per cell"); the **baseline** runner `scripts/python/run_multienv_m3_baseline.py` is a verbatim copy of `scripts/python/run_multienv_m3.py` with the one delta listed in its header, upstream's own `--volume_size`, carried identically by both arms (source `scripts/osworld/run_multienv_m3_baseline.py`; `scripts/osworld/verify_runner_deltas.py` holds the vendored copy against upstream's file on the host) — the agent stays upstream's `mm_agents/m3`; the episode driver is upstream's `lib_run_single.py:run_single_example`, used unmodified (do not re-implement or fork the loop); the launch scripts are copies of `scripts/bash/run_multienv_m3.sh` with our variables. Upstream names, CLI flags, result-directory layout and checkpoint evaluation are kept verbatim — that is what makes our results score with upstream's tools and stay comparable.

## One-time setup (lifecycle, not per-run)

Most steps require human action (AWS console, OAuth screens, credential management); a human operator completes them before the benchmark is invoked. The *Host env provisioning* step is scriptable. Configure the run host and the staging clone before any run; redo a step only when the host is rebuilt, a credential rotates, or a new variable is added.

### AWS account + credentials
1. AWS account with EC2 access.
2. IAM Access Key ID + Secret Access Key created.
3. Record the access key + secret in this repo's `.env` (gitignored). `aws configure` also works; see *Host env provisioning* below.
4. EC2 key pair created in the target region — for SSH access to the long-lived host instance.
5. Client-VM security group with exactly the 10 inbound rules of the V2 public-evaluation guideline (tasks fail without them): TCP 22 and 5910 from `0.0.0.0/0`; TCP 80, 3000, 5000, 8000, 8006, 8080, 8081, 9222 from your VPC's CIDR (3000/8000 = V2 task services, 5000 = OSWorld backend, 8006 = VNC, 9222 = Chrome control). Outbound: all. The host may share this group; then the monitor dashboard (host port 8080) is reachable only from inside the VPC or over an SSH tunnel.
   - Outbound: allow all to 0.0.0.0/0
6. VPC + Subnet selected; record the Subnet ID and Security Group ID.

### Optional — Google account (only for tasks that touch Google Drive — check the OSWorld 2.0 task list)
- Register a dedicated blank Google account (disable 2-Step Verification).
- Create a Google Cloud Project, enable Drive API, configure OAuth consent screen (External, add Drive scope, register the test user).
- Generate OAuth2.0 Desktop credentials.
- Place `credentials.json` (and `client_secrets.json` if generated) in `<OSWorld-V2>/evaluation_examples/settings/googledrive/`, and `settings.json` in `<OSWorld-V2>/evaluation_examples/settings/google/`. They live only on the host; never copy them into any git-tracked location.

Skipping it leaves every other task runnable.

### Optional — Proxy / residential IPs
**Proxy stays off**, and the three proxy-flagged offline tasks (050, 056, 098) are excluded from every run, so nothing here is required. For reference: the harness routes a task's VM through a DataImpulse residential proxy when the task class sets `proxy = True` AND the runner enables proxy at system level; credentials would live at `<OSWorld-V2>/evaluation_examples/settings/proxy/dataimpulse.json`.

### Host instance

One long-lived EC2 instance runs the runners and provisions the per-task client VMs. `<checkout>` below is the absolute path of its OSWorld-V2 checkout.

1. **Launch it** in the region of the V2 client AMI — `us-east-1`, where the release manifest lists the client AMI and the cell config's region defaults — in the subnet and security group above (the client VMs share the host's subnet), with the key pair above. An Ubuntu 24.04 server image with 4 vCPU / 16 GB fits four workers per arm, the cells' run shape (the launch preflight prints the worker count that fits).
2. **Put `uv` on the non-interactive `PATH`**: install `uv` (upstream's installer, e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`), then symlink it into `/usr/local/bin`. Both launch scripts invoke `uv` bare, and a daemonised launch runs in a non-interactive shell, which reads no `PATH` line in `~/.profile` and returns from `~/.bashrc` at its interactivity guard; without the symlink the launch dies immediately into a log nobody is watching. Do the same for the `aws` CLI if an ops script of yours shells out to it (the preflight uses `boto3` and is unaffected). Check from a fresh `ssh <user>@<host> 'command -v uv'`.
3. **Install the system package OpenCV needs** (`libgl1`; "Host system packages" below).
4. **Clone OSWorld-V2 and fetch the gated task data**, as upstream's README describes for release `osworld-v2-2026.08.08`:
   - `git clone https://github.com/xlang-ai/OSWorld-V2.git <checkout>`;
   - request access to the gated Hugging Face datasets `xlangai/osworld_v2_tasks` and `xlangai/osworld_v2_assets_gated`, and log in (`uvx --from huggingface_hub hf auth login`);
   - download the task classes into `evaluation_examples/task_class/` (`uv run scripts/tools/download_osworld_v2_tasks.py --benchmark-release osworld-v2-2026.08.08`) and the assets into `cache/osworld_v2_assets` (`uv run scripts/tools/download_osworld_v2_assets.py --benchmark-release osworld-v2-2026.08.08 --target-dir cache/osworld_v2_assets --clean`);
   - `git switch --detach v2026.08.08`, `uv sync --frozen` (upstream's lock), and confirm `git describe --tags --exact-match` prints `v2026.08.08`.

   The host preflight checks the number of task classes on disk against the release manifest's `task_count`, and every subset task's asset files under `OSWORLD_FILE_BASE_URL`.
5. **No package beyond upstream's lock is needed under `--retrieval_off`**, which every GraphPC launch on the subset carries, as far as the snapshot's imports show: everything it imports on that path (`openai`, `anthropic`, `httpx`, `requests`, `Pillow`, `python-dotenv`, `boto3`) is in the lock, and it runs under the lock's versions, not this repository's `requirements.txt` pins. The host preflight confirms it on the host before any VM is provisioned: it imports the arms' dependencies under the launch interpreter and constructs the arm's agent through the vendored `build_default`. The retrieval text arm (`sentence-transformers`, `torch`) is imported only by a retrieval-on build, which the shipped scripts do not run ("Operations" below). Launch with `uv run --no-sync` all the same (the launch scripts do): a sync at launch re-syncs the environment to the lock.
6. **Write the host `.env`** ("Host env provisioning" below) and the checkout's `.git/info/exclude` lines (step 2 of the per-run steps).

### Host system packages (not in the official guideline)

OpenCV (`cv2`, pulled in by the GIMP-style evaluators in `desktop_env/evaluators/metrics`) needs `libGL.so.1`, which the
Ubuntu 24.04 server image lacks — the task classes that use those evaluators fail to import without it. Install once on
the host (re-do on a rebuilt host):

```bash
sudo apt-get install -y libgl1
```

Verify by importing every task class through the harness loader (`task_loader.load_task_from_file`); the only classes that
may still fail are the GitLab-backed ones (026, 041), which raise until `GITLAB_URL` / `GITLAB_PRIVATE_TOKEN` are set.

### Host env provisioning

The V2 runners call `load_dotenv()`, which searches upward from `scripts/python/`, so `<checkout>/.env` on the host is the file they read. Ship ONLY the OSWorld-relevant variables — exactly the keys of the host `.env` block under *Per-run agent steps* below, named in an allow-list, so no other line of the local `.env` can reach the host — and keep it mode 600; it is gitignored by the checkout. Redo when a credential rotates:

```bash
# From this repo's root. Not in the list, on purpose: AWS_INSTANCE_TYPE (the harness reads that key as a
# client-VM-type fallback, so a value meant for the host would reach the clients).
HOST_KEYS='AWS_REGION|AWS_DEFAULT_REGION|AWS_SUBNET_ID|AWS_SECURITY_GROUP_ID|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|OSWORLD_CLIENT_PASSWORD|WEBSITE_HOST_SUFFIX|OSWORLD_FILE_BASE_URL|OSWORLD_EVAL_MODEL_PROVIDER|OSWORLD_EVAL_MODEL_API_KEY_ENV|OSWORLD_EVAL_MODEL_NAME|OSWORLD_USER_SIM_PROVIDER|OSWORLD_USER_SIM_API_KEY_ENV|OSWORLD_USER_SIM_MODEL|DEFAULT_TTL_MINUTES|ANTHROPIC_API_KEY|OPENAI_API_KEY|DEEPSEEK_API_KEY|MINIMAX_API_KEY|OPENROUTER_API_KEY'
HOST_ENV="$(mktemp)"
command grep -E "^($HOST_KEYS)=" .env > "$HOST_ENV"
scp -i <key file> "$HOST_ENV" <user>@<host>:<checkout>/.env && shred -u "$HOST_ENV"
ssh -i <key file> <user>@<host> 'chmod 600 <checkout>/.env'
```

`DEFAULT_TTL_MINUTES` must be present in THIS repo's `.env` for the allow-list above to carry it — it is not a
credential, so it is the line most easily lost to a rotation or a host rebuild, and losing it silently
restores the provider's short auto-termination (see *Per-run agent steps* below). The launch preflight
requires the key and holds its value against `MIN_CLIENT_VM_TTL_MINUTES`, so a host `.env` missing or
under-setting it fails before any VM is provisioned rather than mid-run.

Re-run the scp only when a credential rotates, a new variable is added, or the host is rebuilt.

### Local staging clone

The sync writes into a local OSWorld-V2 clone, and the runner-delta checks read upstream's files from it. It sits beside
this repo at exactly `../OSWorld-V2`: the sync's default target is that path, and `verify_runner_deltas.py`'s check that
the baseline runner equals upstream's reads upstream's runner only from there (the Qwen and DeepSeek modes also take
`--checkout_root`), so run the checks from a tree that sits beside the clone. A plain full clone, no `--depth` and no
`--single-branch`, so that tag `osworld-v2.1`, which the Qwen runner-delta check reads, comes with it:

```bash
# from this repo's root
git clone https://github.com/xlang-ai/OSWorld-V2.git ../OSWorld-V2
git -C ../OSWorld-V2 fetch --tags
git -C ../OSWorld-V2 checkout v2026.08.08                  # detached at the pinned tag
git -C ../OSWorld-V2 rev-parse 'v2026.08.08^{commit}' 'osworld-v2.1^{commit}'   # optional: d578d2d4e0dc..., 3d778a3c9a34...
```

## Per-run agent steps

Required host `.env` entries (the V2 runners read them via `load_dotenv()`; the evaluator and the user simulator read the `OSWORLD_EVAL_MODEL_*` / `OSWORLD_USER_SIM_*` names — see `desktop_env/evaluators/model_client.py` and `desktop_env/user_simulator.py`):

```dotenv
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1
AWS_SUBNET_ID=<subnet id>                       # the host's subnet (clients must share it)
AWS_SECURITY_GROUP_ID=<security group id>       # the client-VM security group above
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
OSWORLD_CLIENT_PASSWORD=osworld-public-evaluation
WEBSITE_HOST_SUFFIX=site.hku.icu                 # the team-hosted mocked-website suffix OSWorld-V2's README names for release 2026.08.08 (web.hku.icu: 06.24)
OSWORLD_FILE_BASE_URL=<checkout>/cache/osworld_v2_assets   # ABSOLUTE path; dotenv does not expand $(pwd)
OSWORLD_EVAL_MODEL_PROVIDER=anthropic            # model-based checkpoints — official default claude-sonnet-4-6
OSWORLD_EVAL_MODEL_API_KEY_ENV=ANTHROPIC_API_KEY
OSWORLD_EVAL_MODEL_NAME=claude-sonnet-4-6
OSWORLD_USER_SIM_PROVIDER=anthropic              # human-in-the-loop tasks' simulated user
OSWORLD_USER_SIM_API_KEY_ENV=ANTHROPIC_API_KEY
OSWORLD_USER_SIM_MODEL=claude-sonnet-4-6
DEFAULT_TTL_MINUTES=1440                        # see below — the default is shorter than a full-length episode
ANTHROPIC_API_KEY=...                            # required for the evaluator + user simulator whatever the agent backbone
OPENAI_API_KEY=...                               # the agent backbone(s): the ones your cells use
DEEPSEEK_API_KEY=...
MINIMAX_API_KEY=...
OPENROUTER_API_KEY=sk-or-...                    # the Qwen cell: both arms reach qwen/qwen3.7-plus over OpenRouter
```

**`DEFAULT_TTL_MINUTES` is not optional.** Every client VM is created with a cloud-side auto-termination
schedule whose default is shorter than a full-length episode at the benchmark's pace. Left alone it kills the
machine mid-episode; no `result.txt` is written, the task silently goes unscored and has to be re-run, and
because the episodes it kills are the ones still grinding at the step cap, it removes an arm's **worst**
results and biases the comparison in that arm's favour. The variable reaches the provider because the AWS
provider is imported lazily, well after the runner loads the environment file. **Acceptance check, free:**
the provider logs `TTL config: … minutes=<value>` as its first line on the next launch or probe — the edit
is not done until that line reads the value you set.

**The checkout's `.git/info/exclude` must list every vendored path**, or the preflight's unmodified-checkout
assertion fails on the snapshot itself. The sync prints the current expected list; a fresh clone or a
rebuilt host comes up without it, so the report firing then is expected rather than a new problem.

The evaluator model is part of the harness: both arms of a cell must use the same one, and it is fixed at `claude-sonnet-4-6`, the official default; the host preflight checks the evaluator configuration in the host `.env`. `GITLAB_URL` / `GITLAB_PRIVATE_TOKEN` are deliberately absent (GitLab not run).

### 1. Sync the vendored snapshot into the local staging clone

```bash
python scripts/osworld/sync_osworld_graphpc.py --clean   # remove old snapshot, then copy fresh
```

**After any change that removes or renames a synced path, sync with `--clean` and ship with the `rsync --delete` of step 2.** A plain sync never removes a module that has left `SYNC_SOURCE_TARGETS` (`--clean` does, by removing the snapshot wholesale), and no sync, `--clean` included, removes a path that has left `SYNCED_TARGETS`; such a path would stay in the staging clone and reach the host with nothing to flag it: the host preflight checks only that the manifest's paths exist, and `mm_agents/graphpc/` is excluded from git, so no untracked-file check sees extras. Delete a path that has left `SYNCED_TARGETS` from the staging clone and the host by hand, together with its exclude line.

Use `--dry_run` to preview. The default target is the staging clone at `../OSWorld-V2`; `--osworld_root` points the sync elsewhere. **Sync from a git clone of this repository, at a clean commit**, and note that commit's hash in the run README (the run's `runs/<run id>/README`, gitignored, where a run's configuration and launch record go) — it is the snapshot's provenance (there is no snapshot commit). A tree without `.git` (an unpacked archive, say) records no source commit, and the host preflight then fails the snapshot; so does a snapshot synced from a tree with uncommitted changes under `src/` or `scripts/osworld/`.

The sync copies our GraphPC modules into `<OSWorld-V2>/mm_agents/graphpc/` and writes every path in `SYNCED_TARGETS` (`src/benchmarks/osworld/sync.py`): the Qwen and DeepSeek cells' control packages, every cell's runners and launchers, the shared cell config `scripts/bash/osworld_cell_config.sh`, the vision gates, the launch preflight, the runner-delta check and the run-order solver. The sync refuses a checkout that tracks any of these paths: `--clean` removes them wholesale, and `osworld-v2.1` itself tracks `mm_agents/qwen/`. If a new GraphPC source file needs to ride along, add it to `SYNC_SOURCE_TARGETS` in `src/benchmarks/osworld/sync.py` before re-running the sync. Every sync also writes `mm_agents/graphpc/vendored_manifest.json` (the list of every path it wrote, the GraphPC commit and whether the synced sources were dirty); the host preflight checks the snapshot against that manifest: a partial or pre-manifest snapshot fails loudly, a snapshot synced from a dirty tree fails (no reproducible commit), and the source commit is printed for the run README and asserted when `--expect_source_commit` is passed — pass the commit you synced from. The subset manifest is not part of the sync: copy the tracked `data/osworld_v2/prereg_task_list.json`, together with `data/osworld_v2/subset_prereg.json` and `data/osworld_v2/task_class_static_extract_v2026.08.08.json` (the preflight checks the manifest, the task-class overrides and the per-task asset files against them on the host), to `<STAGE>/evaluation_examples/` in the local staging clone; step 2 ships them to the host:

```bash
# from this repo's root
STAGE=../OSWorld-V2
cp data/osworld_v2/{prereg_task_list.json,subset_prereg.json,task_class_static_extract_v2026.08.08.json} "$STAGE/evaluation_examples/"
```

Re-copy and re-ship `subset_prereg.json` whenever it changes: the host preflight reads the host copy.

### 2. Ship the snapshot to the host over SSH

```bash
# from the repo root, GraphPC env active; STAGE=../OSWorld-V2, the local staging clone (step 1)
RSYNC=(rsync -avz --delete -e "ssh -i <key file>")   # an array: -e's command stays one word
for d in $(python -c "from src.benchmarks.osworld.sync import SYNCED_TARGETS as T; print(' '.join(str(p) for p in T if str(p).startswith('mm_agents/')))"); do
  "${RSYNC[@]}" "$STAGE/$d/" "<user>@<host>:<checkout>/$d/"      # mm_agents/graphpc, mm_agents/qwen, mm_agents/deepseek
done
for f in $(python -c "from src.benchmarks.osworld.sync import SYNC_SOURCE_TARGETS as T; print(' '.join(t for _, t in T if not t.startswith('mm_agents/')))"); do
  "${RSYNC[@]}" "$STAGE/$f" "<user>@<host>:<checkout>/$f"
done
for f in prereg_task_list.json subset_prereg.json task_class_static_extract_v2026.08.08.json; do
  "${RSYNC[@]}" "$STAGE/evaluation_examples/$f" "<user>@<host>:<checkout>/evaluation_examples/$f"
done
```

**On the host: `<checkout>/.git/info/exclude` must cover every path the sync writes.** The host preflight FAILs on untracked files it does not recognise, so a sync that writes a path missing from this list breaks the next preflight. **Whenever `SYNCED_TARGETS` grows, the exclude list must grow with it** — that is the failure mode — and a path that leaves the list is deleted from the host checkout by hand (step 1). Derive the list instead of retyping it, from the GraphPC repo root:

```bash
python -c "from src.benchmarks.osworld.sync import SYNCED_TARGETS; print('\n'.join(str(p) + ('/' if not p.suffix else '') for p in SYNCED_TARGETS))"
```

which prints exactly the paths the sync writes, with `mm_agents/graphpc/` covering the whole vendored package and `mm_agents/qwen/` / `mm_agents/deepseek/` the vendored controls. Its output is the list; do not retype it here.

Append those, plus the hand-copied files the sync does not write (`evaluation_examples/prereg_task_list.json`, `evaluation_examples/subset_prereg.json`, `evaluation_examples/task_class_static_extract_v2026.08.08.json`), to `<checkout>/.git/info/exclude`, de-duplicating against what is already there. **Neither of the two signals below is evidence about task CONTENT**: `evaluation_examples/task_class/task_*.py` is in the checkout's own `.gitignore`, because the task classes ship from a gated HuggingFace dataset rather than from git. A re-download inside the same tag can change an instruction, a setup body or an evaluator with no git signal at all, and nothing on a retrieval-off launch watches the instruction text: re-download the task classes only deliberately, and record in the run README when you did. What the tag pins is the harness code. Then `git status` stays clean and `git describe --tags --exact-match` keeps reporting the pinned release. Note what the checkout's own `.gitignore` does and does not cover: `*.log` is ignored, so upstream's `logs/normal-*.log` and `logs/debug-*.log` never show up — but the `logs/` **directory** is not ignored, so any non-`.log` file written there IS untracked and does fail the check. Nothing of ours writes into the checkout; keep any scratch records you write outside it. Nothing is committed or pushed; the host checkout is never edited outside these paths (read-only external-repo policy).

### 3. Run the benchmark on the host

**Both arms share one harness configuration.** Every harness flag (task list, `--max_steps 500` (800 on the Qwen cell's GraphPC arm, its launcher's default), `--eval_version v2`, inline checkpoints `150,300`, `--save_model_eval_raw_info`, provider/region, screenshot + pyautogui, 1920×1080, client password, `--headless`, `--volume_size 60`) comes from `scripts/bash/osworld_cell_config.sh`, sourced by both launch scripts; each script adds only its agent flags (GraphPC: `--backbone` from `OSWORLD_CELL`, `--model`, `--support_set_root`, `--retrieval_off`; baseline: `--base_url --api_key --thinking_mode --max_tokens`) and its own concurrency default: `NUM_ENVS=4` on both arms, sized for a 16 GB / 4 vCPU run host, and the GraphPC script starts its workers 30 s apart (`--env_start_delay 30`). The preflight's memory budget for the GraphPC arm's workers (`num_envs × 3025 MB + 1 GB` must fit in available memory; the startup peak `num_envs × 2760 MB` must leave 1 GB; 2 GB/worker as a floor) is sized with the embedder loaded in every worker, so it is an upper bound under retrieval off. More than 4 is refused unless `FORCE_NUM_ENVS=1`; re-run the preflight after any host resize — it prints the fitting count. `DRY_RUN=1 bash scripts/bash/<script>` prints the fully resolved command (one argv element per line, the MiniMax key shown as `<MINIMAX_API_KEY>`) and exits 0 without running anything. The baseline script also removes `api_logs/` and `model_eval_file_log*/` from task dirs that have no `result.txt` before launching (logged), because upstream's `get_unfinished` crashes on those directories on resume.

**`--volume_size 60` (both arms).** The V2 client AMI ships a ~29.5 GB root partition on a 40 GB volume with neither `growpart` nor `cloud-init`, so the root filesystem starts near 100 % full; snapd's overdue auto-refresh then downloads several GB and fills it, after which the guest server's `/screenshot` handler dies writing its temp PNG (errno 28) and Flask answers 500 — a few minutes after `env.reset`, on a fully idle VM, in both arms alike. A non-`None` `volume_size` is what makes `DesktopEnv` run its own `_prepare_volume` / `_finalize_volume` pair, whose `expand_guest_volume` grows the guest root partition to the volume the AMI never allocates. The flag is upstream's, ported from `run_multienv_gpt_response_api.py` (argparse `:206-211`, per-task lookup `:429-436`); it lives in the shared cell config because it must be identical in both arms, and `VOLUME_SIZE=<n>` overrides it for a probe.

**What `--volume_size` costs: about a minute and one in-guest package install per episode.** It runs for **every task**, not once per worker: `reset` reverts then restarts the guest, the volume is finalised at the end of that path, and on AWS the revert terminates the old instance and launches a fresh one — so every task after the first repeats it. Setup retries and env-spec changes repeat it again within one task, so the per-cell total is a floor rather than an expectation. Because the image ships no partition tool, each run does a package-index update and an install **inside the guest**, and those are the only two commands in that path without failure suppression — an unreachable mirror therefore fails environment construction for that task on both arms. Expect env-construction failures, not screenshot failures, if the mirrors are flaky.

**The requested size survives the revert**, so it is not an artefact of the first construction: `revert_to_snapshot` passes `resolve_aws_root_volume_size(..., self.volume_size, ...)` as the new instance's `BlockDeviceMappings` `VolumeSize` (`desktop_env/providers/aws/provider.py:185`, `:217`), so every episode's fresh VM is created at 60 GB and then expanded in the guest. With it the guest root filesystem comes up at the volume's size with room to spare, and the snap refresh, which still runs, is no longer fatal. A task's own `volume_size` still wins over the flag, above it or below it, so `--volume_size` is a **default, not a floor**: five of the 28 subset classes declare their own and keep it — **048 (64), 057 (80), 064 (50), 107 (80), 108 (80)**, the pre-registration's `volume_size_overrides` — so 064 runs below the cell's 60 and the other four above it, and the remaining **23** take the flag's value. The static extract (`task_class_static_extract_v2026.08.08.json`) supplies only `assets` and `uses_website` per task; every other environment spec is read from the task classes themselves, which the host preflight loads through the harness's own loader.

**Offline checks (free; locally, on the commit you sync from).** Each is a standalone loud-fail runnable with its own
negative controls, and each reports the coverage it demonstrated rather than the number of things it
looked at. Capture their exit status directly — not through a pipe, which reports the last command's
status rather than the check's:

```bash
python scripts/osworld/verify_runner_deltas.py                 # both runners carry the same delta; the baseline is upstream + that port, byte-identical
python scripts/osworld/solve_subset_run_order.py --check       # the installed order is the pre-registered one
python scripts/osworld/launch_preflight.py --mode local        # add --cell qwen / --cell deepseek for those cells
```

`verify_runner_deltas.py`'s default mode reads upstream's `run_multienv_m3.py` only from the clone at exactly
`../OSWorld-V2` beside this repository (on the host, from the vendored runner's sibling); no flag changes that, so run
it from a tree that sits beside the staging clone.

**Launch preflight (free; run before any VM is provisioned).** `scripts/osworld/launch_preflight.py` (vendored as `scripts/python/launch_preflight.py`) is a standalone loud-fail gate that prints a PASS/FAIL table and exits 1 on any failure. Locally, from the repo root: `python scripts/osworld/launch_preflight.py --mode local` (the runners are vendored here, so no V2 clone is needed) checks that the manifest ids equal `data/osworld_v2/subset_prereg.json` exactly, that both scripts' `DRY_RUN` argv have identical harness flags (only agent / run-shape flags may differ, and `--model` must match), that the GraphPC argv carries `--retrieval_off` (required on every scored launch; the launcher writes it into its command), that the GraphPC launch's `--max_steps`, the adapter's budget and the cell's pre-registered `max_steps` agree, that each runner's argparse accepts its script's flags, and that every sync source exists. On the host, from the checkout root: `uv run --no-sync python scripts/python/launch_preflight.py --mode host --num_envs_graphpc N --num_envs_baseline N --expect_source_commit <the commit you synced from>` (give it the exact `NUM_ENVS` each arm will be launched with — each launch script is resolved under `NUM_ENVS=<that count>`, so a `NUM_ENVS=1` smoke is preflighted at 1/1 and a scored run at the cell's configured worker counts; a count the GraphPC script refuses fails there) repeats those against the vendored files and adds: pinned tag, no modified tracked harness file, the subset task classes load (instance/volume/image overrides, checkpoint safety, user simulator, multi-phase reported per task), the assets directory, the release manifest and the default AMI, `.env` keys with the official evaluator config, STS identity, subnet + security group incl. the V2 inbound ports, orphan task VMs, vCPU quota vs the planned concurrency, the mocked websites the 28 tasks use (resolved and probed over HTTPS or HTTP from the host), a cold support set and no prior `result.txt` (or `--acknowledge_resume <arm>`, per arm: a GraphPC launch against a finished control passes `baseline` alone, so its own tree must still be cold), no runner alive (a live GraphPC runner fails the preflight and skips the build check; a live baseline runner warns), the python deps in-process, both launchers carrying `--no-sync`, AMI availability, every per-task asset file, the MiniMax key shape (`sk-`, distinct from the Anthropic key), `build_default` constructing the arm's agent with retrieval off, the step budget against the cell's method surface, and the GraphPC workers' memory budget (`num_envs × measured footprint + 1 GB` must fit in available memory, 2 GB/worker as a floor; the fitting worker count is printed). `--allow_api_calls` adds one tiny call on each arm's MiniMax route and one evaluator call (cents). It ends with the resolved regime of both arms for the run README; record the output there.

Before provisioning any VM, run the two loud-fail gates on the host, from the checkout root. First the runner-delta gate, `uv run --no-sync python scripts/python/verify_runner_deltas.py` (synced with the snapshot; on the host upstream's `scripts/python/run_multienv_m3.py` is a sibling of the vendored baseline runner, so its baseline-equals-upstream check runs against the file that will actually launch, and it exits 1 if that file is missing rather than skipping). The vision gate `uv run --no-sync python scripts/python/verify_minimax_vision.py` sends a synthetic 1920×1080 frame through the GraphPC arm's two model routes (grounding: median of 3 samples ≤ 60 px per square; reasoning: colour + quadrant 3/3, sampled on the deciding call's own route, `OSWorldMiniMaxAgent.complete_timeline_json`, with the schema in the request text and the image as the last part) and exits 1 on failure — launch rule: one re-run allowed, two consecutive failures block, because the grounding median carries residual noise (read the failing kind from the printed samples). **No embedder gate.** Every GraphPC launch on the subset runs `--retrieval_off`, so no worker loads the text arm and no launch check covers it. Launch everything with `uv run --no-sync` all the same: the checkout is pinned at a tag, and a sync at launch re-syncs the environment to the lock.

From the checkout root on the host, the GraphPC arm is launched by the vendored script (all env config comes from `<checkout>/.env` via the runner's `load_dotenv()`; the script exports nothing, because `load_dotenv` never overrides a variable that is already set). The script serves all three cells and has **no default cell**: every launch names it in `OSWORLD_CELL` (`minimax`, `qwen` or `deepseek`), and a launch without it is refused. Each control launcher sets its own cell.

```bash
OSWORLD_CELL=minimax bash scripts/bash/run_multienv_graphpc.sh     # the GraphPC arm; a scored launch appends NOTHING (see the guard below)
bash scripts/bash/run_multienv_m3_baseline.sh                       # the control
```

The GraphPC launch expands to (always through `uv run --no-sync`: the checkout is pinned at a tag, and a sync at launch would re-sync the environment to the lock):

```bash
uv run --no-sync python scripts/python/run_multienv_graphpc.py \
    --headless --observation_type screenshot --action_space pyautogui \
    --screen_width 1920 --screen_height 1080 \
    --test_all_meta_path evaluation_examples/prereg_task_list.json \
    --eval_version v2 --max_steps 500 \
    --checkpoint_eval_mode inline --checkpoint_steps 150,300 --save_model_eval_raw_info \
    --provider_name aws --region us-east-1 --client_password osworld-public-evaluation \
    --volume_size 60 \
    --backbone minimax --model MiniMax-M3 \
    --result_dir ./results/osworld_v2/graphpc_minimax_m3 \
    --support_set_root ./results/osworld_v2/graphpc_minimax_m3_support_set \
    --retrieval_off \
    --num_envs 4 --env_start_delay 30
```

Key arguments:
- `--backbone {minimax,qwen,deepseek}` — the GraphPC backbone, one model profile each (the launch script picks it from `OSWORLD_CELL`); `minimax` reads `MINIMAX_API_KEY`, `qwen` reads `OPENROUTER_API_KEY` and `deepseek` reads `DEEPSEEK_API_KEY` from the host `.env`, and the runner refuses to start without the backbone's key. On `qwen` the profile's output cap wins over `--max_tokens`, and the launch preflight refuses a non-None `--max_tokens` with `--backbone qwen`. GraphPC-only flags: `--support_set_root`, `--retrieval_off`, `--two_pass`, `--completion_gate_off`, `--completion_gate_window`. `--retrieval_off` is written into the launcher's command for every cell (every GraphPC run on the subset runs retrieval off) and the preflight requires it; under it nothing reads or writes the support set. `--max_tokens` is kept and forwarded to the backbone; its default `None` sends no cap (the regime of the WebArena GraphPC-M3 arm; an explicit cap goes in the run README). The other M3Agent-only flags (`--base_url`, `--api_key`, `--thinking_mode`, `--temperature`, …) do not exist on this runner — our backbone pins its own regime. The runner also forwards `client_password` (the sudo password) to the agent for its environment brief.
- `--support_set_root` — must stay OUTSIDE the `--result_dir` task tree: the runner's `get_unfinished` wipes unfinished task dirs on resume and walks every directory under `<result_dir>/<action_space>/<observation_type>/<model>/` as a domain.
- `--test_all_meta_path` — a `{"tasks": ["001", ...]}` file. Official full set `evaluation_examples/test_v2.json` (108); official offline list `test_v2_offline_no_internet.json` (82, `main` only); **ours = the 28 ids of `data/osworld_v2/subset_prereg.json`**, the tracked `prereg_task_list.json` copied to the host.
- `--eval_version v2` — the 2.0 evaluators (default on the V2 runner; state it anyway).
- `--max_steps 500` — the official budget; the leaderboard also reports 150/300.
- `--checkpoint_eval_mode inline --checkpoint_steps 150,300` — partial-score checkpoints during the run (the official sample); `--save_model_eval_raw_info` keeps the evaluator's raw outputs.
- `--num_envs N` — one EC2 client VM per concurrent task (t3.xlarge by default; per-task overrides are the pre-registration's `instance_type_overrides` — in the subset, 048 requests t3.large); a 4 vCPU / 16 GB host is sized for four workers per arm (the cells' run shape); the preflight must be run with the exact counts that will be launched — its DRY_RUN check fails when a launcher's resolved `--num_envs` differs from `--num_envs_graphpc` / `--num_envs_baseline`.
- `--specific_task_id <id>` — single task (smoke tests); it can only be **appended**, and an appended argument requires the declaration below. `--observation_type` accepts `screenshot | a11y_tree | screenshot_a11y_tree | som`; the official runs are screenshot-only.

**Appended arguments: a scored launch carries none.** Both launch scripts end in `"$@"` and the preflight resolves them **bare**, so nothing appended at launch is covered by any preflight verdict — and `argparse` is last-wins, so an appended `--max_steps` / `--observation_type` / `--test_all_meta_path` silently breaks the harness-flag identity the preflight asserts. `osworld_cell_guard_launch_args` (in `scripts/bash/osworld_cell_config.sh`, called by both scripts) therefore refuses **any** appended argument unless the launch declares itself unscored with `OSWORLD_UNSCORED_LAUNCH=1`, and refuses a declared-unscored launch whose result tree — `RESULT_DIR`, or an appended `--result_dir`, in any path spelling — is any cell's scored tree or a locked tree (`OSWORLD_SCORED_PATHS_*` and `OSWORLD_LOCKED_PATHS` in `scripts/bash/osworld_cell_config.sh`). That is what keeps the baseline's deliberately unscoped resume pre-clean and upstream's `get_unfinished` (which ignores `--specific_task_id` / `--domain` on the control arm) from emptying a live scored arm's in-flight episodes when someone runs a one-task smoke. To change a harness setting for a scored run, change the cell config — not the command line. The cross-arm half of the rule runs on **every** launch including a bare one, because `RESULT_DIR=<the other arm's tree>` with no flags at all is the shortest route to emptying the other arm's episodes; each arm's own defaults pass it, so the scored argv is unchanged. Path options are matched by **prefix**, because both runners leave argparse's `allow_abbrev` at its default and `--resu` resolves to `--result_dir`. The GraphPC arm's **support-set store is guarded like its result tree** — with retrieval on, `finalize_task` writes an entry at score 1.0, so a smoke that happens to score would seed the store — and it may not sit inside a result tree, which the runner's `get_unfinished` would walk as domains.

**One accepted residual, written here because it is the only place an operator meets it.** The guard keys on appended arguments and on the result/support paths, not on `MODEL`. So `MODEL=Other bash scripts/bash/run_multienv_m3_baseline.sh` with the default `RESULT_DIR` appends nothing, passes the guard, and its resume pre-clean then reaches the scored `MiniMax-M3` subtree — removing `api_logs/` and `model_eval_file_log*/` from unfinished task dirs there. Bounded: no `result.txt` is touched and an unfinished dir holds no scored data, so no score can be lost. It is not closed because the pre-clean's glob cannot be scoped to the model without re-implementing the runner's `_model_for_paths` sanitiser in bash. **If you change `MODEL`, change `RESULT_DIR` with it.**

**Launch through the scripts, from the checkout root.** The guard is a launcher-level control, so invoking `scripts/python/run_multienv_*.py` directly bypasses it entirely — and on the baseline arm that runner is upstream's, with no scoping delta, so its `get_unfinished` empties every unfinished task dir under whatever `--result_dir` it is given. There is no second line of defence there by design: the control's value is being upstream's code. Run the bash scripts. Run them from the checkout root, too: the scored paths the guard compares against are anchored to the checkout, but `RESULT_DIR` and the task-list path are resolved against your **current directory**, so a launch from elsewhere silently means a different result tree and a different manifest.

**Paid steps are the operator's.** Opening a cell, every launch, the `--allow_api_calls` preflight and the vision gates spend money; a human operator decides each.

### During the run: the mid-run stop rule

The GraphPC arm runs the 28 in one launch and is never halted by design, with one exception: the score rule. After
every task both arms have completed (in run order), take k = the number of completed pairs, B = the control's
cumulative score over those k tasks and D = the cumulative paired delta (GraphPC − control) over the same k tasks.
**Stop the GraphPC arm when k ≥ 8, B ≥ 0.5 and D/B ≤ −0.7**, then diagnose. A ratio rather than a delta sum because
the sum is order-dependent; D/B is order-invariant and bounded below at −1, which reads as "zero everywhere the
control scored". There is no give-up rule: the two arms' give-up detectors count different things. Compute D and B
from the per-task rows `score_paired.py --allow_missing` (step 4) prints for pulled `result.txt` files, over the tasks
both arms have finished; a reading that could not be made is "could not tell", never "continue". Every other finding during a run — a lost
episode, a stalled task, a cost line — is diagnose-first: report it, and stop only when the diagnosis finds a defect
that corrupts the result. Stopping a run leaves its client VMs running ("Operations" below).

### 4. Score results

OSWorld-V2's `show_result.py` cannot read the V2 layout (its `get_result` hard-codes the 1.0 domain names and its `__main__` hard-codes `("pyautogui", "gpt-4o", "a11y_tree", "./results")`). The paired per-task partial-score comparison (our verdict instrument, `experiments.md`) is computed by our own scorer over the files the V2 driver writes per task — `result.txt` (authoritative scalar), `result.json` (evaluator breakdown, when returned) and `checkpoint_results.json` (inline checkpoints):

```bash
# LOCAL only, GraphPC env, after `scp -r` of both result trees: this scorer is NOT in SYNC_SOURCE_TARGETS and
# does not exist on the host. Each --*_root is the runner's --result_dir, copied back.
python scripts/osworld/tools/score_paired.py \
    --baseline_root <pulled baseline tree> \
    --graphpc_root <pulled GraphPC tree> \
    --task_list data/osworld_v2/prereg_task_list.json \
    --output_json runs/<run id>/scoring/paired_scores.json
```

It prints the paired table (per task: baseline, GraphPC, delta, checkpoint scores), mean partial score and binary-completion count per arm, discordant counts and an exact two-sided sign test on the per-task deltas, and dumps the same as JSON. A task listed in the manifest but without `result.txt` in an arm is **UNSCORED** (reported with the error row from `summary/results.json` when there is one) and never counted as 0; the script exits 1 in that case unless `--allow_missing`. Checkpoint columns are **NA** (never 0) for a multi-phase task (final-only, `phase_results.json`; 069 is the only one among the offline tasks and is **not** in the subset, so that path is unexercised) and for tasks whose checkpoints were skipped (`intermediate_eval_safe=False`, **7 of the 28**: 034, 035, 053, 067, 079, 092, 104 — the pre-registration's `checkpoint_unsafe`); a checkpoint mean covers only tasks with a successful value at that step in **both** arms, so the per-checkpoint `n` is the same for both arms and says how many tasks the column covers. Copy results back with `scp -r`.

**Before quoting a delta**:
- every manifest task has a `result.txt` in both arms; a task lost to infrastructure is re-run through its launcher's resume (a second fault loss of the same task is escalated, not retried again), and a task still unscored after its allowed re-runs counts 0.0 for that arm (`experiments.md`, Verdict methodology);
- no task VM is left running (terminate orphans; the host preflight lists them);
- a finalize failure on the GraphPC arm leaves `graphpc_finalize_error.txt` next to that task's `result.txt`: count those files;
- a 0.00 the EVALUATOR could not substantiate (a mocked service down at evaluation time) is indistinguishable in `result.txt` from a 0.00 the agent earned, and `result.json` records nothing of the former: read each 0.00 task's final-evaluation window of `eval.log` (the records that end at its closing `Result:` line) and its `model_eval_file_log/` on both arms for ERROR / WARNING records of the task class or an evaluator, and adjudicate those tasks by hand;
- count the baseline's `ASK_USER` rows per task in `traj.jsonl` (an exhausted model-call retry yields one, "Baseline" below); on the pre-registration's `hitl` tasks (007, 034), where a simulated user answers, record any such row in the verdict and flag the task there (it is not re-run);
- the GraphPC arm ran its retrieval regime: every task's `graphpc_telemetry.json` `active_config` records `retrieval_off: true` and `retrieval_agent: "BypassRetrievalAgent"`, and its retrieval events are bypassed with nothing injected.

## Smoke test before full runs

A single-task run to verify wiring (after steps 1–2 above and a passing preflight run with `--num_envs_graphpc 1 --num_envs_baseline 1`, the counts the smoke launches with). On the AWS host, pick one task id from the subset list and run it alone through the launch script. A smoke appends arguments, so it must declare itself unscored and point its result tree somewhere throwaway **through the environment** — `RESULT_DIR`, not an appended `--result_dir`, because the baseline arm's resume pre-clean reads the variable while the runner reads the flag, and an appended one would leave the pre-clean scrubbing the scored tree. Prefix `DRY_RUN=1` to see the resolved command first:

```bash
OSWORLD_CELL=minimax NUM_ENVS=1 OSWORLD_UNSCORED_LAUNCH=1 \
RESULT_DIR=./results/smoke_tests/graphpc \
SUPPORT_SET_ROOT=./results/smoke_tests/graphpc_support_set \
  bash scripts/bash/run_multienv_graphpc.sh \
    --specific_task_id <task_id> --checkpoint_steps 5,10
```

The GraphPC arm takes no `--max_steps`: its runner (delta 15) refuses any budget but the adapter's for its backbone (500; 800 on the Qwen profile), so the smoke is kept short by stopping it once it has passed step 10, not by lowering the budget. `--checkpoint_steps 5,10` makes it touch checkpoint evaluation early. The baseline arm has no such check: a baseline smoke may append `--max_steps 15`, and then needs the same checkpoint override, because the driver drops every checkpoint step `>= max_steps`.

Run the same single task through `scripts/bash/run_multienv_m3_baseline.sh` to smoke the control arm (with `OSWORLD_UNSCORED_LAUNCH=1 RESULT_DIR=./results/smoke_tests/m3_baseline`, for the same reason); its acceptance criterion is that step 1 in `traj.jsonl` is a pyautogui action string, not `ASK_USER` (a misrouted MiniMax key does not error there: every call 401s and the task scores 0.0), and that `api_logs/001_*.json` carries a `response_body`. A smoke exercises setup → steps → checkpoint evaluation → the post-episode `finalize_task` hook (telemetry file written; support-set write only at score 1.0) only when its episode ends on its own. A 15-step baseline smoke always does. A GraphPC smoke stopped by hand has no final evaluation and no `finalize_task`. It is a wiring test, not a score. A smoke's telemetry carries `diff_pixel_count` per step; the screenshot-diff floor is an instantiation constant (`design.md` §6).

## What differs per cell

The per-run steps above apply to every cell; each cell names itself with `--cell <cell>` on every preflight run (`qwen`, `deepseek`; the default is `minimax`) and with `OSWORLD_CELL=<cell>` on the GraphPC launch. The control launchers set their own cell. The rest of this section is only what differs from the MiniMax cell.

### The Qwen `qwen3.7-plus` cell

- **Route, both arms:** `qwen/qwen3.7-plus` over OpenRouter, with the provider pin on every request (the route facts: `webarena.md`, "Qwen `qwen3.7-plus` over OpenRouter"); the key is `OPENROUTER_API_KEY` in the host `.env` (`ANTHROPIC_API_KEY` stays, for the evaluator and the user simulator). Each arm writes the route literals out for itself: keep the two copies in step by hand when either side is edited.
- **The control** is upstream's modular Qwen agent and runner from tag `osworld-v2.1` (the pinned tag has none), vendored with the deltas D1–D9 its headers declare: `scripts/osworld/qwen_control/mm_agents/qwen/` (synced to `mm_agents/qwen/`), `scripts/osworld/run_multienv_qwen_baseline.py` and its launcher `run_multienv_qwen_baseline.sh`, which sets `OSWORLD_CELL=qwen` itself and passes `--generation_timeout 300` (upstream's default is 200). Before a resume, the control's launcher moves every unfinished selected task dir to `<RESULT_DIR>_quarantine/<UTC timestamp>/`, because upstream's runner deletes such dirs before re-running them.
- **Frames, both arms:** JPEG at quality 90 at the control's `smart_resize` size (1920×1088), the control's delta D9, which the GraphPC arm's copy of `images.py` carries block for block. Upstream's PNG frames can exceed OpenRouter's image cap (about 30 MiB of base64).
- **Provider faults, both arms:** a provider failure carrying "JSON decode error: unexpected EOF",
  "Download multimodal file timed out" or "Failed to download multimodal content",
  in-band or as an HTTP 400, is retried in the arm's own loop and is infra once the retries run out (control
  delta D2; GraphPC `backbone_qwen.transient_rule`). So is an HTTP 200 the openai SDK
  cannot decode at the provider call (a JSON body that does not decode, a non-JSON content type, a stream
  chunk that is not JSON), typed `UndecodableResponseError` by each arm; a `JSONDecodeError` raised
  anywhere else stays agent_error. The host preflight checks the SDK's behaviour offline on the Qwen cell.
  On the GraphPC arm only, an HTTP 400 or an in-band error in no known form is retried for 3 attempts and is then
  infra (`UnrecognisedProviderError`); a context overflow on a 400 (the control's own patterns) and a 200 with no
  choices stay the agent's at once; 403/413/422 stay the agent's and 401/402/404/408 are infra, none retried, and an
  in-band error's code follows the same map. Each call record counts its unrecognised attempts
  (`transport_unrecognised`). Both classes' texts start with their name ("UndecodableResponseError: ",
  "UnrecognisedProviderError: "). The control has no such fallback (its context-overflow rescue needs the 400 at
  once). Each arm carries its own copy of the provider-fault fragments (control delta D2, GraphPC
  `backbone_qwen.transient_rule`); keep the two copies in step by hand when either side is edited.
- **Result trees**, under `results/osworld_v2/`: the GraphPC arm at `graphpc_qwen37plus`, its store `graphpc_qwen37plus_support_set` beside it (outside the tree; written to by nothing under retrieval off), and `baseline_qwen37plus`; the model directory in both is `qwen_qwen3.7-plus`. The launch guard holds a Qwen launch to these paths and refuses every other cell's tree and every locked tree.
- **The GraphPC arm's grounding, Qwen profile** (`action_qwen.py`). The scroll composite: a decided scroll grounded as [`mouse_move`, `scroll`] on the decided axis runs as one command. The grounding class-family check: a grounding reply whose first call is outside the decided class's family, or is a `screenshot` / `wait`, is repaired ("{decided} was decided but the reply is {got}; reply with a {decided}"); a substitution within the click family (left/right/double/triple/middle click, left_click_drag) executes; exhausted, it is a WAIT with reason `grounding_class_mismatch`. The drag composite: a decided drag grounded as [`mouse_move`, `left_click_drag`] runs as one command, as the scroll composite does. The rule's surface is the `active_config` key `grounding_class_check`.
- **The GraphPC arm runs 800 steps.** The GraphPC launcher's Qwen branch passes `--max_steps 800` (`OSWORLD_MAX_STEPS_QWEN_GRAPHPC` in the cell config); the control's launcher keeps 500, because the Qwen control can batch several actions in one step while the GraphPC arm runs one per step. The runner refuses any other value for `--backbone qwen` (delta 15, `OSWORLD_MAX_STEPS_BY_BACKBONE`); the preflight holds the argv, the adapter and `method_surface_qwen.max_steps` to one value, and each arm to its own. Checkpoints stay at 150 and 300; the final score is read at the episode's end. A smoke runs at 800 too.
- **The stuck-stop, the GraphPC arm only** (`OSWORLD_STUCK_STOP_BY_BACKBONE` in `graphpc_agent.py`, `design.md` §6). Past 500 predict calls, an episode spinning in place ends with `DONE`, without a model call and without passing the completion gate: clause repeat, one executed command at least 50 times among the last 55 executed steps with at most 11 of them changing the screen; or clause W, 50 consecutive transparent WAITs with at most 10 changing it. A decided wait and a `grounding_class_mismatch` exhaustion WAIT are transparent; "changing" is the monitor's masked diff at its own floor. The rule is the `active_config` key `stuck_stop`.
- **The pre-registered method surface** is `method_surface_qwen` in `subset_prereg.json`: the host preflight holds the built agent's `active_config` to it, including `completion_gate_rules`, `grounding_class_check`, `max_steps` and `stuck_stop`.
- **Host exclude lines** new for this cell: `mm_agents/qwen/`, `scripts/python/run_multienv_qwen_baseline.py`, `scripts/bash/run_multienv_qwen_baseline.sh` and `scripts/python/verify_qwen_vision.py` (re-derive the list, step 2).
- **Offline checks** add `python scripts/osworld/verify_runner_deltas.py --cell qwen --checkout_root <staging clone>`: Q1 holds every vendored control file, and the GraphPC arm's copies of upstream Qwen code, to their `osworld-v2.1` blobs (the clone must have that tag); Q2 holds the two Qwen-cell runners to feeding the unmodified driver and `DesktopEnv` the same values. The local preflight runs with `--cell qwen`.
- **On the host:** `uv run --no-sync python scripts/python/launch_preflight.py --mode host --cell qwen --num_envs_graphpc N --num_envs_baseline N --expect_source_commit <the commit you synced from>` (add `--acknowledge_resume baseline` before the GraphPC launch once the control has finished; the GraphPC tree must be cold). With `--allow_api_calls` it makes one tiny call on the pinned route, which must be served by alibaba. The vision gate is `uv run --no-sync python scripts/python/verify_qwen_vision.py`: a synthetic frame through both GraphPC routes (reasoning: colour and quadrant 3 of 3; grounding: median error ≤ 60 px per square over 3 samples); its launch rule is MiniMax's.
- **Launch** (a scored launch appends nothing):

  ```bash
  OSWORLD_CELL=qwen bash scripts/bash/run_multienv_graphpc.sh      # the GraphPC arm
  bash scripts/bash/run_multienv_qwen_baseline.sh                   # the control
  ```

- **Smoke** (throwaway paths, declared unscored, one task):

  ```bash
  OSWORLD_CELL=qwen NUM_ENVS=1 OSWORLD_UNSCORED_LAUNCH=1 \
  RESULT_DIR=./results/smoke_tests/graphpc_qwen SUPPORT_SET_ROOT=./results/smoke_tests/graphpc_qwen_support_set \
    bash scripts/bash/run_multienv_graphpc.sh --specific_task_id <task_id> --checkpoint_steps 5,10
  NUM_ENVS=1 OSWORLD_UNSCORED_LAUNCH=1 RESULT_DIR=./results/smoke_tests/qwen_baseline \
    bash scripts/bash/run_multienv_qwen_baseline.sh --specific_task_id <task_id> --max_steps 15 --checkpoint_steps 5,10
  ```

- **Score** with `score_paired.py` on the two pulled Qwen trees. The control's `eval.log` has no record header (upstream's per-task handler has no formatter), so read each control task's final-evaluation window from the control runner's own log (`logs/qwen-<timestamp>.log` in the checkout, one per launch, re-runs included).

### The DeepSeek `deepseek-flash` cell

- **Two endpoints, one model.** The control reaches `deepseek-flash` through DeepSeek's Anthropic-compatible endpoint (`https://api.deepseek.com/anthropic`), the GraphPC arm through the OpenAI-compatible one (`https://api.deepseek.com`); the key is `DEEPSEEK_API_KEY` in the host `.env` (`ANTHROPIC_API_KEY` stays, for the evaluator and the user simulator). The control launcher passes the key through `--api_key` only. The host environment must NOT carry `M3_THINKING_MODE` or `M3_THINKING_BUDGET`: upstream's agent and runner read them, and a value there silently changes the control's thinking form; the preflight checks this.
- **The control** is upstream's `mm_agents/m3` at the pinned tag, vendored as `scripts/osworld/deepseek_control/mm_agents/deepseek/` (synced to `mm_agents/deepseek/`) with the marked deltas D2–D4 and D6 its README lists (the `computer` tool declared natively, the reply read from the `tool_use` block, the tool-result protocol, a call-time request stamp in each `api_logs` record), because the model does not use M3's text tool grammar. Its runner `scripts/osworld/run_multienv_deepseek_baseline.py` (synced to `scripts/python/`) is upstream's `run_multienv_m3.py` plus the `--volume_size` port and the agent import; its launcher `run_multienv_deepseek_baseline.sh` sets `OSWORLD_CELL=deepseek` itself and passes `--base_url https://api.deepseek.com/anthropic --model deepseek-flash --max_tokens 65536 --thinking_budget 4096 --num_envs 4` and no `--thinking_mode` (thinking on at the model's default effort `high`; `THINKING_BUDGET` and `MAX_TOKENS` are overridable for unscored probes only). The GraphPC arm's `computer` tool is this control's tool with `done` removed (`action_deepseek.py` holds copies of the control's description and schema): keep the two in step by hand when either side is edited.
- **The GraphPC arm** is the same runner and launcher as the other cells', launched with `OSWORLD_CELL=deepseek`, which passes `--model deepseek-flash` (leave `MODEL` unset: the response validation and the effort pin apply to that id only); it reads the 28 (`prereg_task_list.json`) in ONE launch. Its deciding call runs thinking at effort `high` with `json_object`; its grounding call is a forced native `computer` function call without thinking at temperature 0.6, [0,1000] coordinates; retrieval off; 500 steps on both arms. Its deciding call keeps a requirement ledger that the completion gate's `requirements_unverified` rule judges a success claim by, and its STATE block states the step budget (this profile only; the control's agent never states the budget, a disclosed difference between the arms). The pre-registered method surface is `method_surface_deepseek`; the host preflight holds the built agent's `active_config` to it.
- **Result trees**, under `results/osworld_v2/`: `graphpc_deepseek_flash` with its store `graphpc_deepseek_flash_support_set` beside it (written to by nothing under retrieval off) and `baseline_deepseek_flash`; the model directory is `deepseek-flash` on both. The launch guard holds a DeepSeek launch to these paths and refuses every other cell's tree and the locked trees.
- **The thinking history** is upstream's: the control re-sends every prior reply whole, so its input grows with the episode against the model's 1M-token window. Watch the per-call input on the control's first calls.
- **Host exclude lines** new for this cell: `mm_agents/deepseek/`, `scripts/python/run_multienv_deepseek_baseline.py`, `scripts/bash/run_multienv_deepseek_baseline.sh`, `scripts/python/verify_deepseek_vision.py` (re-derive the list, step 2). The cell's synced files are the control package, its runner and launcher, the vision gate, and the GraphPC snapshot's `backbone_deepseek.py`, `action_deepseek.py`, `profile_deepseek.py` and `src/agents/backbone/deepseek/`.
- **Offline checks** add `python scripts/osworld/verify_runner_deltas.py --cell deepseek --checkout_root <staging clone>`: DS1 holds every control file to its `v2026.08.08` blob minus the marked hunks, DS2 the runner to upstream's `run_multienv_m3.py` minus its two hunks, DS3 the two DeepSeek-cell runners to feeding the driver and `DesktopEnv` the same values, each launch reading the 28. The local preflight runs with `--cell deepseek`.
- **On the host**, before each launch: `uv run --no-sync python scripts/python/launch_preflight.py --mode host --cell deepseek --num_envs_graphpc 4 --num_envs_baseline 4 --expect_source_commit <the commit you synced from>` (before the control's launch nothing is acknowledged; before the GraphPC launch, `--acknowledge_resume baseline`). With `--allow_api_calls` it makes one tiny call on the OpenAI-compatible endpoint (the model echo checked) and one `tool_use` / `tool_result` round trip on the Anthropic-compatible endpoint, through the control package's own agent. The vision gate is `uv run --no-sync python scripts/python/verify_deepseek_vision.py`: a synthetic frame through both GraphPC routes (reasoning: colours and quadrants 3 of 3; grounding: a click per square with median error ≤ 60 px over 3 samples, a scroll inside a square with the decided direction, a drag from one square to the other); its launch rule is MiniMax's.
- **Smokes** (throwaway paths, declared unscored; task 038 as the example):

  ```bash
  OSWORLD_CELL=deepseek NUM_ENVS=1 OSWORLD_UNSCORED_LAUNCH=1 \
  RESULT_DIR=./results/smoke_tests/graphpc_deepseek SUPPORT_SET_ROOT=./results/smoke_tests/graphpc_deepseek_support_set \
    bash scripts/bash/run_multienv_graphpc.sh --specific_task_id 038 --checkpoint_steps 5,10
    # no --max_steps: the GraphPC runner refuses any budget but the adapter's (500); stop it once it has passed step 10
  NUM_ENVS=1 OSWORLD_UNSCORED_LAUNCH=1 RESULT_DIR=./results/smoke_tests/deepseek_baseline \
    bash scripts/bash/run_multienv_deepseek_baseline.sh --specific_task_id 038 --max_steps 15 --checkpoint_steps 5,10
  ```

  The GraphPC smoke's acceptance is the MiniMax smoke's (step 1 of `traj.jsonl` a pyautogui action, both checkpoints evaluated, `active_config` present in the telemetry). The control smoke's `api_logs` records must show `response.model` = `deepseek-flash`, a thinking block and one `tool_use` on every call, no `max_tokens` stop, and the tool-result turn accepted at turn 2.
- **Launch** (a scored launch appends nothing), the control first:

  ```bash
  bash scripts/bash/run_multienv_deepseek_baseline.sh                           # the control, all 28, first
  OSWORLD_CELL=deepseek bash scripts/bash/run_multienv_graphpc.sh                 # GraphPC, all 28, one launch
  ```

- **During the control run**, read its `api_logs` records on the host on the first calls and periodically: a wrong model echo, a missing thinking block, a format regression, or an input-token growth that would exhaust the window stops the control for diagnosis.
- **Score** with `score_paired.py` on the two pulled trees.

## Task set

OSWorld 2.0 ships 108 tasks (gated HuggingFace task classes + assets, pinned to one release together with the code and the mocked-website deployment); the official OFFLINE list (`evaluation_examples/test_v2_offline_no_internet.json`, on `main` only — 82 tasks that touch no third-party public service) is the only official subset. The cells run offline only: GitLab task 026 and the proxy-flagged 050/056/098 are excluded, leaving 78, and the scored set is the pre-registered 28-task subset `data/osworld_v2/subset_prereg.json` (`subset_ids`) — a minimum cover of the 78 across phenomenon, application family, application count and evaluation-mechanics strata, with twin pairs (`twin_pairs`) as corroborating transfer evidence. **Both arms run the same ids in the same pre-registered order** (`run_order`). The order separates every twin pair as widely as the set allows; it is derived by `scripts/osworld/solve_subset_run_order.py` and is provably optimal for the set rather than searched, so a reader can check it without re-running a solver. The manifest is the only channel that carries it — the runner preserves manifest order into the worker queue, and a resume rebuilds the manifest in place, so the order survives a restart. Order matters because a pair whose members run concurrently cannot demonstrate transfer at all; the order is part of the pre-registration and stays fixed with retrieval off, where no transfer can occur. Give the runner a `{"tasks": [...]}` file with those ids via `--test_all_meta_path`. A different subset needs its own mocked-website sweep before it runs: the preflight's resolver (`collect_website_hosts`) derives each task's hosts from its class source, which a text search of the sources can miss.

## Baseline

The control is a **bare single-call agent on the same multimodal model** as the GraphPC arm (per the `experiments.md` model policy): screenshot in → one model call → action out, no GraphPC scaffolding. By the copy-first rule the control is **OSWorld-V2's own runner + agent for that model, run unmodified** wherever the pinned tag ships one: MiniMax M3 → `mm_agents/m3` unmodified, driven by `scripts/python/run_multienv_m3_baseline.py` (a verbatim copy of upstream's `run_multienv_m3.py` with ONE delta, a harness setting applied identically in both arms: `--volume_size`, ported from upstream's own `run_multienv_gpt_response_api.py` and forced by the AMI disk defect above — see that file's header), launched by our `scripts/bash/run_multienv_m3_baseline.sh` (a copy of upstream's `run_multienv_m3.sh` with the same task list / steps / checkpoints as the GraphPC arm and MiniMax pinned explicitly — `--base_url https://api.minimax.io/anthropic --model MiniMax-M3 --thinking_mode adaptive --api_key $MINIMAX_API_KEY` — because the pinned tag has no MiniMax defaults); Claude → `run_multienv_claude.py`; GPT → `run_multienv_gpt_response_api.py`; Qwen `qwen3.7-plus` → upstream's modular Qwen agent and its runner from the later tag `osworld-v2.1` (the pinned tag has none), vendored with the nine declared deltas D1–D9 and launched by `scripts/bash/run_multienv_qwen_baseline.sh` ("What differs per cell" above); DeepSeek `deepseek-flash` → upstream's `mm_agents/m3` vendored as
`mm_agents/deepseek/` with the declared deltas D2–D4 and D6 (the model's native tool-calling format), driven by
`run_multienv_deepseek_baseline.py` (upstream's `run_multienv_m3.py` + the volume port and the import), launched by `run_multienv_deepseek_baseline.sh` ("What differs per cell" above). For any other model without an upstream agent at the tag, the control is the closest upstream code copied with the minimal endpoint/model edit, recorded, never a hand-written agent.

Run once clean and lock per the locked-baseline policy (same contamination definition as the WebArena section: ≥3 retry-exhausted, ≥10% steps killed, or any mid-run docker restart → provisional lock; re-run until clean; the triggers are diagnose-first). **The baseline's contamination definition counts two retry mechanisms** — a **model-call retry** exhausted (upstream M3 re-calls the model on a raise or a zero-action parse, 3 attempts, `mm_agents/m3/agent.py:647`; exhaustion yields an `ASK_USER` row and a simulated human answer) or a **guest-command retry** exhausted (`desktop_env/controllers/python.py:execute_python_command` re-POSTs a command to the guest up to 3 times, logging `Retrying to execute command.` between attempts). They are unrelated; GraphPC's **re-sample** (`src/agents/common/resample.py`) is a third thing again, the arm's counterpart of the first. **The literal line `Failed to execute command.` is not that signal on its own**: the controller logs the same line when an attempt hits its 90 s read timeout, because it breaks out of the loop without retrying. Exhaustion is that line after a `Retrying to execute command.` from the same worker. Say which one any count refers to. **The ≥3 retry-exhausted investigation trigger** (diagnose-first) is deliberately inclusive and also counts a third mechanism: an **env-setup retry** exhausted (`desktop_env/desktop_env.py`, `Environment setup failed with exception, retrying (n/5)`, `MAX_RETRIES = 5`) that ends in the runner's per-task exception (`Exception in EnvProcess-N <domain>/<id>`), a lost task. Record each control's lock in `project_state.md`. The lock rule: **a fault-lost episode does not degrade the lock if it was re-run to completion on an arm whose agent carries no state across episodes** — true of upstream's `mm_agents/m3`, deliberately NOT true of the GraphPC arm, whose support set makes a re-run non-equivalent.

## Standing facts and rules the run depends on

### Operations

- **The benchmark checkout is read-only** outside the deployment subtree (see AGENTS.md "External repository policy") — fixes go in this repo and propagate via the sync script.
- **Two-tier AWS footprint**: the long-lived AWS host instance (where the runner executes) and the per-task EC2 VMs it provisions are separate. The host stays up across runs; task VMs are short-lived.
- **VM cleanup**: the AWS provider terminates task VMs at end-of-run. On `VcpuLimitExceeded`, identify orphans (AMI/tag match AND launch-time > 10 min before current run, OR stuck `stopping` > 10 min), terminate, then optionally request a quota increase (`aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-1216C47A`).
- **Stopping a run**: upstream's parent signal handler SIGTERMs the workers, sleeps 1 s, then SIGKILLs any survivor (`scripts/python/run_multienv_m3.py` `signal_handler`: `p.terminate()`, `time.sleep(1)`, then `SIGKILL`; the vendored runners carry the same handler), so the workers' `finally: env.close()` never completes and the client VMs keep running after the runner is gone — terminate them explicitly and confirm none is left before relaunching (the host preflight lists orphan task VMs).
- **Cost**: parallelism trades wall-clock for $$. OSWorld 2.0 tasks run to hundreds of steps each (500-step budget), so size `--num_envs` against per-task VM hours plus model tokens; the leaderboard publishes full-run token costs per model — check them before opening a cell.
- **Retrieval on is unsupported on OSWorld by the shipped scripts.** The runner refuses a launch without `--retrieval_off` (delta 15) and the preflight requires the flag. The adapter's retrieval-on path (`build_default` without `--retrieval_off`: the percentile relevance gate over the pre-registration's `relevance_threshold_*` keys, and the store) stays in code, but no shipped launcher runs it and no calibration tool ships for its threshold. Were it run, successful trajectories would accumulate in `--support_set_root` and feed later retrieval, so repeat runs would NOT be independent unless each points at a fresh store.

### The step budget and actions per step

Count actions per LOGICAL step from `traj.jsonl` rows grouped by `step_num`: a key chord in one row (`keyDown ctrl; keyDown s; keyUp s; keyUp ctrl`) is one action, not several verb-runs. Our arm is at one action per step by construction (`max_actions=1`), so `--max_steps 500` buys each arm the same number of attempts exactly when the control also emits one action per step — although `mm_agents/m3/prompts.py:33` tells it to "try to chain multiple of these calls all into one function calls request". Check that on the control's `traj.jsonl` before quoting the budget as symmetric. A task ending at the cap is scored at its budget like the control's; no cap-truncated task is re-run.

Not on the Qwen cell: its control can batch several actions per step at 500, and the GraphPC arm runs one action per step at 800.

**A cost of `max_actions=1`, and NOT a model defect.** The deciding call may propose more than one action; all but the first are discarded, so a coherent multi-step plan (an install followed by a clean-up, say) is derived, truncated to one action, thrown away and re-derived the next step. This belongs beside the one-action-per-step invariant as a cost of it; changing the invariant is a method change and needs its own argument.

### Scrolling: one divergence to expect

The guardrail suffix carries upstream `:31` ("scroll through it
to see everything before deciding that something is not available"), which the control's prompt carries too, and the
two arms may respond to the same instruction differently — for instance, scrolling at different rates. Behaviour
downstream of a MATCHED clause is not a prompt asymmetry; there is nothing to correct.

### The coordinate-free bypass

The bypass routes the coordinate-free classes — `type`, `key`, `screenshot`, `wait` — straight to the translator
instead of through the grounding model (`hold_key` is excluded, see below). For an action whose tool call carries no
coordinate that call is a total function of one field of the intent, so the grounding model cannot acquire
information it did not already have — it can only corrupt it, for instance by substituting a `left_click` for a
`type` whose content is then never entered — and after the bypass the `type` class does not reach grounding at all.
A "click-to-focus-then-type rescue" reading of such substitutions does not hold: a focus-acquisition mechanism would
obey the re-issued `type`, and a substitution that recurs on the identical retry is corruption.

**A substitution count must say WHICH KIND it counts.** For a `key` payload that is not a keystroke at all — a shell
command labelled `key` — the substitution is PROTECTIVE: the model declining to press nonsense. Genuine corruption is
substitution of a payload that was fine; a `type` intent that carries its text cannot be refused as malformed input.
Any count that mixes the two measures two things at once.

**The residual risk the bypass creates is invisibility, not a lost step**: text typed with nothing focused goes
nowhere and the trajectory records no loss at all. The mitigation is telemetry and not a gate, so no screen-state
judgement re-enters grounding — the monitor's screen-change verdict is recorded on every bypassed action
(`SCREENSHOT_CHANGED_MIN_PIXELS = 64`, `screenshot_diff` in `monitor/heuristic.py`), the `grounding_bypassed` flag
records which route produced the action, and because the verdict already reaches the next deciding call through
`_env_feedback`, a no-change-after-type becomes something the reasoning stage can act on, which is where a judgement
about screen state has to live.

**Two intent-level shapes that survive the bypass**, neither worth acting on. A `scroll` intent may carry a keysym in
`keys`; `scroll` takes a direction and an amount, so the field is simply ignored, and the direction is in the intent
prose the grounding model reads. And X11-style key names — `Page_Up`, `Page_Down`, `super` — are **not** defects at all:
`m3_action_translator.py:tool_action_to_pyautogui` converts them (its key-name table) before they reach pyautogui, and that table is upstream's own,
byte-identical to `mm_agents/m3/parser.py:153-159`. An audit that measures the raw `keys` string instead of what the
translator emits will re-derive that false positive; measure post-conversion. The one genuine gap is `grave`, which
upstream's table also lacks — see the repair rule below.

### `hold_key` leaves the key held — a latent defect that stays unrepaired

`tool_action_to_pyautogui`'s `hold_key` branch emits `pyautogui.keyDown(...)` for each
`+`-separated part and **no `keyUp` at all**, and never reads the `duration` its own argument
contract accepts. One `hold_key` would leave the modifier held down for the remainder of the
episode, corrupting every action after it. It is latent: a `hold_key` reaches the translator only if the grounding
model emits one.

**The repair rule, which governs `grave` too: a fix present on our side and absent from the control
is a one-sided advantage.** This is the verbatim copy of upstream's parser, so the defect is present
in both arms identically and the symmetric failure cancels out of the paired comparison. That is why
it stays unrepaired, and why `hold_key` is excluded from the coordinate-free bypass despite being
coordinate-free — a bypass would remove the grounding model's opportunity to decline it.

### A bypassed step carries no grounding text — recorded, NOT fixed

`ActionBatch.content["reasoning_text"]` is `"\n".join(instructions)` and a bypassed step appends no
instruction, so the field is `""` on every fully-bypassed step.
`graphpc_agent.py:_extract_response_text` returns the reasoning call's own text when it is non-empty
and otherwise falls back to the batch's, so on a grounded step a reasoning call that produced no text
still puts the grounding model's one-line imperative into `traj.jsonl`'s `response`; on a bypassed
step there is nothing to fall back to and `response` is empty. Exposure is small: it needs a bypassed step whose
reasoning call returned no text.

**The reason it is not fixed is the rule, not the size.** Writing `render_intended_action(...)` into
that field would put a string WE composed into a field whose meaning is "what the model said", on
every bypassed step, with nothing telling a later reader it is ours. The honest empty is better, and the
`model_call: false` telemetry field already carries what a reader needs. It cannot turn into a false
give-up count either: our adapter degrades every step-level failure to the `WAIT` token rather than to an empty
action list, so it never reaches the driver's empty-action branch at all.

### The terminal token reaches scoring on this subset only through two tasks' own `evaluate()` (010, 034)

**On the 28-task subset the terminal token reaches scoring only through two tasks' own `evaluate()`:**
**034**'s calls `generated_task_utils._last_action_failed(env)` and returns 0.0 when the last action is
`FAIL`, and **010**'s calls `evaluate_metric`, which makes the same check first. On those two a final
`FAIL` scores 0.0 where a final `DONE` would be scored on its work; on the other 26 the token changes no
score. The control emits `FAIL` for its own `[INFEASIBLE]` declarations, so the two tasks treat both
arms' give-ups alike.

**Standing fact, scoped to the 28-task pre-registered subset: every one of those tasks supplies its
own `evaluate()`, so the terminal token reaches scoring only where that `evaluate()` reads it itself
(010, 034, above).** Converting a give-up `DONE` into `FAIL` changes no score on the other 26.

**Why.** `desktop_env/desktop_env.py:584` dispatches before anything reads a terminal:

```python
def evaluate(self):
    if self.task_config is not None and self._has_custom_evaluate(self.task_config):
        return self._call_task_evaluate(self.task_config)
    return self._evaluate_with_evaluator()
```

`_has_custom_evaluate` is true when the task class overrides `BaseTask.evaluate`, and
`_call_task_evaluate` runs the task's own `evaluate(env)`, which touches `action_history` only through
`_last_action_failed` on 010 and 034. The
`FAIL` handling — `if evaluator['func'] == "infeasible": FAIL -> 1 else 0`, and in the `else` branch
`if last_action == "FAIL": return 0`, which short-circuits the metric — lives entirely in
`_evaluate_with_evaluator`, the branch these 28 tasks never take. **Over the subset ids, 28 of
28 define their own `evaluate()`**; none calls `evaluate_infeasible_task`, and only 010 and 034 read
`action_history` (through `_last_action_failed`). Check a task by evaluating the dispatch predicate itself — load the
class on the host and evaluate `cls.evaluate is not BaseTask.evaluate` — rather than by asking whether the text
`def evaluate` appears in the file. Note what does NOT establish this: the `evaluator` attribute is `None` for every
one of the 28, so that field separates nothing and is not the dispatch condition.

**"How many infeasible tasks are in the subset" has the answer ZERO, for a different reason than the
question assumes.** It is not that none of the 28 happens to be infeasible — it is that no task in
this subset uses the JSON-evaluator path at all, so the `infeasible` evaluator func is not reachable
here regardless. Two decoys will turn up in any grep and mean nothing: 010's import of
`evaluate_infeasible_task` is dead (no call site), and task 102's `"FAIL"` is a string inside a
PASS/FAIL logger line.

**SCOPE THIS CAREFULLY. It is a property of these 28 tasks, not of OSWorld.** A future V2 task
carrying a JSON evaluator instead of its own `evaluate()` reaches that branch, and there the terminal
token decides the score outright. Anyone extending the subset must re-check the dispatch before
assuming the conversion is free.

### The deciding call does not carry the action vocabulary

The tool list sits with the call that emits the tool call — grounding (`M3_GROUNDING_SYSTEM_PROMPT`,
`action_minimax.py`) — as it does in the control, whose one model both decides and grounds; the deciding schema
and the translator's alias normalisation close `action_class` to its controlled vocabulary whatever the prompt says.
Giving the deciding call upstream's `<TOOLS>` list (`mm_agents/m3/prompts.py:60-71`) would add nothing the pipeline
lacks:

- **`wait` sleeps 0.5 s in both arms**: `m3_action_translator.py:tool_action_to_pyautogui`'s `wait` branch emits a fixed `pyautogui.sleep(0.5)` and
  discards the validated `duration`, verbatim from upstream's own parser (`mm_agents/m3/parser.py:231-232`);
  `screenshot` is `sleep(0.1)`. A `wait` with `"duration": 45` sleeps half a second in the control too.
  (Upstream's prompt `:68` advertises an argument its own parser throws away.)
- **Grounding does not need the list to get a scroll direction**: a direction missing from a scroll intent's named
  fields reaches grounding through the `intent` / `target` prose that `render_intended_action` puts in front of the
  grounding model.

> **Checking which FIELDS are populated is not checking what the model was TOLD.** The intended action reaches
> grounding as rendered prose, not as a field dump, so a field-level audit of it measures the wrong artefact.

The guarded deciding-instruction suffix itself — the terminal-claim contract, the operating rules from upstream
`:31-32`, and the feasibility guardrails from `:34-45` — carries no action list.

### Rules for any measurement computed from run artefacts

Measurement defects return a **plausible number rather than an error**, so analysis code that gates a decision needs
the same broken-shut discipline as the gates themselves.

- **Frame N is the state AFTER action N; pair `frame(N-1) -> frame(N)`.**
  `lib_run_single.py:944-948` writes `step_N_<timestamp>.png` from the obs returned by
  `env.step(action)`. Pairing `frame(N) -> frame(N+1)` measures the NEXT action and still returns a
  plausible number: a `type` followed by a `screenshot` action shows no change, and `screenshot` is a
  common action, so a forward pairing can report a no-change rate several times the true one. The first pair's
  "before" must be `step_0_initial.png`; a tree with no opening frame is skipped with a named reason,
  never started at row 1.
- **An invariant that only holds in aggregate cannot guard a per-run computation.** The
  `screenshot`-versus-click unchanged-rate contrast — a no-op should leave the screen alone far
  more often than a click does — separates the two frame pairings over many episodes, but on a single run it has no
  discriminating power and can point the wrong way, because even a `sleep(0.1)` step can change more pixels than the
  64-pixel floor.
- **Name a quantity whose expected value is known a priori and is FAR from what a broken version
  produces, and assert it beside the result** — e.g. `target` is a non-empty `str` in every intent, so
  `assert all(r["target"] for r in rows)` fails at once on an extraction that returns empty `target` and `text`.
- **A reader is only validated by pointing it at a REAL artefact.** Unit tests and gate cases that
  CONSTRUCT their inputs cannot catch a reader that counts nothing on real files (a counter of `FAIL`
  terminals reads zero on a tree whose give-ups end in `DONE`). Budget one real-tree run before
  trusting any checker, however green its suite.
- **Name a fixture for what it CONTAINS, never for what it resembles**, so that its result is never
  reported as a measurement of the thing it resembles.
- **Every number must name its population.** Ask WHICH POPULATION before doubting the arithmetic.
  The cheap defence is a labelling convention: write the population into the figure itself, as in
  "obeyed-intent `type` actions (n = <obeyed> of <executed> `type` intents)". Two populations can both
  be correct for their own purpose, and the label is what says so — the frame-pairing discriminator
  takes every `type` intent because it is a physics check, while an attribution baseline takes only
  the obeyed subset.
- **Never order frames lexicographically**: `step_10_…` sorts BEFORE `step_2_…`, so any enumeration
  that sorts by filename is wrong from step 10 onward. Join frames to steps through `traj.jsonl`'s
  `screenshot_file` instead — which also survives the `phase_{i}_step_{n}_*.png` pattern that a
  `step_*.png` glob misses on multi-phase tasks. Trees open with `step_0_initial.png`
  (`graphpc_agent.py:_write_initial_screenshot`); do not drop it.
- **An extraction that can fail per-record without raising reports the failures as absence** — the
  same shape as a free-retry count that reads zero off a stripped `api_logs` mirror, and as the "none" a check
  reports when it globs an empty directory.

## Adapter architecture

`src/benchmarks/osworld/graphpc_agent.py:OSWorldGraphPCAgent` is the boundary layer. Outward it is exactly the agent contract V2's `lib_run_single.run_single_example` drives (`reset(runtime_logger)`; `predict(instruction, obs) -> (response_text, actions)` where `actions` is a one-element list of pyautogui code or the literal `"DONE"` / `"FAIL"` / `"WAIT"`; optional `set_api_log_dir`; `task_current_date` set by the driver) — so the V2 driver runs unmodified. Inward it calls only `decide_next_action` / `update_runtime`; decision k is closed at the start of predict k+1 with the new observation. Per step: `obs["screenshot"]` → a screenshot-primary `Observation` (`observation_adapter.py`; no accessibility channel; the optional `environment_brief` content key carries the M3 system prompt's environment facts — platform, home directory, sudo password from `client_password`, task date — which the OSWorld reasoning agent renders into the deciding call's system prompt); grounding through `action_minimax.py` (V2's native M3 `<tool_call>` format: the M3 coordinate-rule line + `<TOOLS>` block, minus the `done` entry and the two termination rule lines, as the system prompt; one imperative line from the intended action, non-thinking, grounding temperature 0.6, `[0,1000]` relative coordinates) and `m3_action_translator.py` (`tool_action_to_pyautogui` copied verbatim from V2, `parse_m3_response` copied with seven documented edits; every argument defect raises `ValueError`); every step-level failure (reasoning/grounding parse, translation, completion-gate block, empty batch) becomes the env `WAIT` token with the failure fed to the next reasoning call through the monitor feedback channel; the deciding contract's give-up is the `fail` action, which the action agent hands over as a `done` carrying `terminal_intent: "fail"` so the completion gate evaluates it exactly as it evaluates a success claim (same admit rule, same two-strike budget), and which the adapter emits as the profile's give-up token once admitted (`FAIL`; `DONE` on Qwen, whose control never emits `FAIL`); a plain `done` whose statement declares infeasibility is still emitted as `FAIL` as a fallback (audit event in telemetry). Post-episode (`finalize_task`, called by the runner hook with `result_payload = {score, result_json, checkpoint_results, phase_results}`): relabel write-back, then a support-set write only at score 1.0 (never for multi-phase tasks, never under `--retrieval_off`), and `graphpc_telemetry.json` (retrieval events, verification verdicts, per-step monitor signals, degrade counters, usage, active config) into the task result dir; **`monitor_signals` is where the pointer mask's audit pair lands** — on every screenshot step it carries `diff_pixel_count` (masked, the operative one), `diff_pixel_count_unmasked`, `pointer_mask_applied` and `pointer_mask_points`, the ≤2 coordinate pairs the mask actually covered, recorded live rather than re-derived so a coordinate or scaling bug cannot reproduce itself in the audit; a mask audit reads exactly these fields (the steps whose masked and unmasked counts sit on opposite sides of the 64-pixel floor are the frames to inspect), and because the file is written at episode close an episode killed mid-flight leaves no mask record at all (unscored, so it is re-run — but an absent record is never a clean one); an episode that dies before the hook is written as an orphan record (`persist: orphaned_<reason>`, no score) by the runner's per-task except branch and again at worker exit through `agent.close_episode(reason=...)` (runner delta 8; idempotent, so the last task of a worker or a one-task smoke keeps its record too), and by the next task's `set_api_log_dir` as a fallback. Per-call diagnostics (the GraphPC counterpart of the baseline's `api_logs/`): every model call of a task is written to `<task dir>/graphpc_calls/NNN_<stage>.json` — stage (reasoning | grounding), attempt within the step's stage (2+ = a re-sample), prompt sizes (system/user chars, image bytes), the full response text, usage and latency; never the screenshot bytes. Volume: ~40 KB per call worst case (a 10k-token reasoning reply), two calls per step, so a 500-step task can leave up to ~1,000 files / ~40 MB under its task dir — keep them out of scp -r pulls you do not need, and never under the support-set root. `build_default(backbone_type=..., ...)` looks up that model's profile (`OSWORLD_MODEL_PROFILES`; MiniMax-M3, Qwen `qwen3.7-plus` and DeepSeek `deepseek-flash` are registered, see "Adding a model" below and "What differs per cell") and wires one shared backbone (MiniMax: `OSWorldMiniMaxAgent`, grounding_temperature 0.6) for `OSWorldReasoningAgent` + the profile's action agent (MiniMax: `MiniMaxActionAgent`), one `EpisodeTimeline` shared by the adapter (writer) and the reasoning agent (reader), the deterministic retrieval agent over `LocalSupportSetStore(backend="osworld")` (or the bypass agent with no store/embedder under `--retrieval_off`), the heuristic monitor (screenshot-diff signal, `design.md` §6) and the observation adapter.

**The deciding request.** `OSWorldReasoningAgent` (`reasoning.py`) renders it over the
backbone's `complete_timeline_json` (`reasoning.py:TimelineRoute`, which every OSWorld backbone class inherits:
`OSWorldMiniMaxAgent`, `OSWorldQwenAgent`, `OSWorldDeepSeekAgent`);
the WebArena reasoning agent and backbone are the shared classes, untouched.
- **System** (static for the task): the deciding instruction, the OSWorld suffix (`prompts.py`: `done`
  declares success in the control's :72/:77 words, infeasibility is `fail`), the `facts` instruction,
  and the environment facts. The control's coordinate line (:49) is not here; grounding carries it.
- **User**: one message, laid out by `timeline.py:EpisodeTimeline`. It holds step 0 (the opening
  screen and the instruction), then every executed step. A step's line gives the action as the
  environment received it, with a WAIT saying why, the target as the grounding line carried it, the
  monitor's verdict, and the reconciled label. Inside the image window it also carries the step's
  reasoning (≤ 1,500 chars) and the labelled frame after it. Outside the window it is one line
  (≤ 200 chars, outcome first) ending "[screenshot omitted]". Then come the state block (plan,
  graph_summary, facts ledger, retrieved exemplars) and the schema block ("Return a single JSON
  object conforming to this schema …", byte-identical every step). Last is "CURRENT SCREEN, after
  step N (…):" and the current frame, with nothing after it.
- **Image count**: the control's arithmetic, computed on steps, so a failed capture is a hole and
  never a shift. No item cap.
- **Reply checks**: a decide/refine reply without a list-typed `intended_actions` is re-sampled, and
  so is a reply whose top level is a JSON Schema. The control re-samples a zero-action parse the same
  way.
- **Grounding line**: coordinates the deciding call wrote into `target` / `intent` are stripped from
  it (`action_minimax.py:strip_grounding_coordinates`); `text` / `keys` / `value` are never touched.
  The patterns are the coordinate forms the deciding call writes, plus "~" / "≈" before a number and a range
  on a lone x= / y=, with integer numbers only, so decimal document measurements such as "(11, 0.5)" stay. The DeepSeek profile runs three patterns of its
  own first, and strips the `value` field when it holds only a bare coordinate pair or a start-and-end pair of
  points (`DEEPSEEK_VALUE_POINT`; `action_deepseek.py:strip_deepseek_grounding_coordinates`); MiniMax and Qwen keep
  the shared strip.

**Records.**
- **Per-call record** (`graphpc_calls/NNN_<stage>.json`): `image_count`, `history_image_count`, and
  `image_slots` (per image: slot, step label, predict call, sha1 of the source PNG). It also records
  `cached_tokens`, `thinking` ({chars, text} when the provider returned any), and `request_file`, the
  gzipped sibling `NNN_<stage>.request.json.gz` holding the request's text parts as sent. On deciding
  calls it adds `feedback_text` (the line about the previous step) and `rejection` (null, or {kind,
  detail} with kind unparseable / wrong_shape / schema_echo / native_tool_call / other).
- **Telemetry** (`graphpc_telemetry.json`):
  - `facts_ledger` (the whole ledger) and `facts_elision` (renders that elided, most entries elided);
  - `reasoning_decision_missing` and `reasoning_decision_missing_kinds` (deciding samples that parsed
    but carried no decision, counted apart from `reasoning_parse_failures`, so that parse failures +
    decision-missing = re-samples + exhausted steps);
  - `grounding_inputs_stripped` and `grounding_strip_fallbacks`;
  - on `_executed_actions` rows: `terminal_intent`, and `grounding_input_strip` ({stripped, removed,
    fallback}) on grounded steps.
- **`active_config`**: `deciding_request: "timeline"`, `max_trajectory_length: null`, and
  `timeline_text_limits` (reasoning 1,500; out-of-window line 200; executed text 300; fact entry
  300 and fact source 100, both applied when the fact is recorded; facts render 32,000 / 16,000 /
  16,000). The pre-registration's `method_surface` declares the same values.

### Adding a model to the OSWorld adapter

Everything that varies with the model is one profile, `graphpc_agent.py:OSWorldModelProfile`, except two
backbone-keyed tables beside it in the same file (below, "What else moves"), and a
model is added by registering a zero-argument builder for it in `OSWORLD_MODEL_PROFILES` under its
`backbone_type`. `build_default` builds the profile at call time and constructs through it; any other
`backbone_type` raises `NotImplementedError`. Each model's control is its own upstream agent, so every
per-model piece follows that agent.

- **What a profile holds.** As classes: `backbone_class` (the model's client with `reasoning.TimelineRoute`
  before it in the bases) and `action_agent_class` (grounds an intended action into the model's own tool
  call; owns the grounding prompt, the reply parser and the terminal mapping, and emits a give-up as a
  `done` carrying `terminal_intent: "fail"`), plus `translate_action` (the tool call → pyautogui source or
  `DONE` / `FAIL`, `ValueError` on any defect). The grounding call's wire format is the action agent's as
  well: MiniMax and Qwen ground in their controls' text tool grammars (MiniMax through the client class's
  `complete_text(grounding=True)` with a stop sequence; Qwen through `backbone_qwen.py:complete_grounding`, no
  stop, because the copied parser needs the closing tag); DeepSeek grounds with a forced native function call
  (`backbone_deepseek.py:complete_grounding` with `tools` + `tool_choice`, no stop), because the model does not
  follow a text tool grammar. A model's
  action agent therefore owns the choice between a text grammar and a native call. Where the arm and the control
  each hold a copy of the same text — DeepSeek's tool description and schema (the arm's is the control's with `done`
  removed), Qwen's route literals and provider-fault lists — keep the copies in step by hand when either side is
  edited. As
  values: `action_classes` (the deciding schema's
  vocabulary), `deciding_suffix` (matched line by line to the control's prompt, as `prompts.py` is to
  `mm_agents/m3/prompts.py`), `environment_brief` (the environment facts of the control's system prompt),
  `call_attempts` (the control's retry budget) and `grounding_temperature`. The regime fields, each at
  MiniMax's existing value for MiniMax: `typing_note`, `image_retention` (the
  control's image-folding function, `timeline.py:ImageRetention`), `give_up_token` (`FAIL`, or `DONE` for a
  control that never emits `FAIL`), `max_tokens` (None = the runner's), `deciding_repair` (None = identical
  re-sample; else the control's repair turn), `request_timeout`, `give_up_block_text` (None = the gate's
  rule text), `ask_user_channel`, `infra_error` (None = every exception leaves `predict`; else the
  lost-episode classifier, with everything else ending the episode as an evaluated `agent_error`) and
  `capture_failure_raises` (False = a failed capture is a hole in the timeline; True on Qwen and DeepSeek,
  whose controls lose the episode on a failed capture). Two method fields are set on the DeepSeek profile only:
  `requirement_ledger` (False by default; True: the deciding reply keeps a requirement ledger and the
  completion gate adds the rule `requirements_unverified`, with its own two strikes) and `step_budget_line` (None by
  default; the STATE block's step-budget line). The Qwen and DeepSeek profiles (`profile_qwen.py`,
  `profile_deepseek.py`) set each from their controls.
- **What the backbone must provide** is written down in `reasoning.py:OSWorldBackbone`, a Protocol
  extending the shared `Backbone`: from the client class, the grounding route the profile's action agent
  calls (MiniMax: `complete_text(grounding=True, stop=...)`; Qwen and DeepSeek: their OSWorld subclass's
  `complete_grounding`), `_base_request`, `_extra_body`, `_encode_screenshot`, `_create_with_backoff` returning an OpenAI
  chat-completions response, `_extract_json_object`, `_notify_observer`, `reset_session`; three per-model
  hooks, `IMAGE_URL_PREFIX`, `_record_response_usage` and `_thinking_text`; and the counters the telemetry
  reads (`model_name`, `call_count`, `total_input_tokens`, `total_output_tokens`), without which it
  records zeros. It is documentation, not enforced.
- **Each profile declares its action vocabulary** (`action_classes`, the deciding schema's closed enum):
  M3's normalised names on MiniMax and DeepSeek (`M3_ACTION_CLASS_VOCABULARY`), the Qwen control's names
  on Qwen (`QWEN_ACTION_CLASS_VOCABULARY`), each with the two terminal classes `done` / `fail`. A model
  whose tool calls use other names maps them into its profile's vocabulary in its action agent.
- **What stays model-neutral**: the adapter's `predict` flow and telemetry, the timeline's text
  rendering (its image retention is the profile's function, matched to its own control's image arithmetic), the
  facts ledger, the five shared completion-gate rules in
  `orchestrator.py` (a profile with the requirement ledger adds `requirements_unverified`), the per-call record, the frame memo, the coordinate strip and the give-up contract
  (its emitted token and block line are profile values).
- **What else moves with a new model**: the vendored runner's `--backbone` choices and key check (with
  `verify_runner_deltas.py`'s delta list), the launch script and cell config, the launch preflight,
  `SYNC_SOURCE_TARGETS` for any new module, and the two backbone-keyed tables in `graphpc_agent.py`:
  `OSWORLD_MAX_STEPS_BY_BACKBONE` (the step budget; indexed directly by `build_default` and by runner delta 15,
  so a backbone without an entry raises `KeyError`, and read from the adapter's source by the launch preflight
  and `verify_runner_deltas.py`) and `OSWORLD_STUCK_STOP_BY_BACKBONE` (read with `.get`: no entry, no
  stuck-stop). A control that needs declared deltas (DeepSeek) ships
  as a vendored package under `scripts/osworld/<model>_control/mm_agents/<model>/` with marked hunks, its own
  runner and launcher, and a `--cell <model>` mode in `verify_runner_deltas.py` that holds every file to its
  upstream blob minus the marked hunks (the Qwen cell's Q1 shape).

## References

- OSWorld 2.0 site: https://osworld-v2.xlang.ai/
- OSWorld-V2 repository (AWS / Docker images): https://github.com/xlang-ai/OSWorld-V2
- OSWorld 2.0 paper: https://arxiv.org/abs/2606.29537
- Task catalog: https://osworld-v2-monitor.xlang.ai/
- Gated task classes: https://huggingface.co/datasets/xlangai/osworld_v2_tasks
