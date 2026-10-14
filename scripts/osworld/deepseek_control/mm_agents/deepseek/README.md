<!--
GraphPC vendored copy of OSWorld-V2 mm_agents/m3/README.md
at tag v2026.08.08 (commit d578d2d4e0dc82b43e270fdaa7fa89d9708cd154), blob 9eaf9c66f5a641fe18ca961fc7e91f1c5efbe5d6.
The OSWorld DeepSeek cell's CONTROL is upstream's own M3 agent, run against DeepSeek-V4.1-Flash on its
Anthropic-format endpoint; this file is byte-identical to that blob except this header. The deltas the
package carries are marked in place in the other files (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`, upstream
lines kept as `#|`), and scripts/osworld/verify_runner_deltas.py --cell deepseek holds every file to its blob
minus exactly those blocks. Why each delta exists is in the header of
scripts/osworld/run_multienv_deepseek_baseline.py; what each does, by file:
  prompts.py D2 (lines 104-173) — NATIVE_TOOLS_BLOCK, COMPUTER_TOOL (description + input_schema: the 16
    prompt-listed actions; the two field descriptions are the GraphPC arm's tool bytes) and M3_SYSTEM_PROMPT_TEMPLATE
    rebound to its <SYSTEM_CAPABILITY> block + NATIVE_TOOLS_BLOCK.
  agent.py D2 (lines 67-69, 646-649) — the import; tools=[COMPUTER_TOOL], tool_choice auto on every request
    (_build_request_body).
  agent.py D3 (lines 355-364, 394-400, 527-599, 686-721, 821-827, 833-845, 919-922) — _call_llm keeps the text blocks' text and
    sets aside the tool_use / thinking blocks; predict reads the action from the first `computer` tool_use through
    _parse_native_reply (upstream's parser on the one synthesized <tool_call> block), [INFEASIBLE] from the text
    blocks only (it wins over a tool call, as upstream); every other tool_use block goes to the api_log
    (_second_tool_use), never executed or stored; a failed call clears the blocks. The blocks labelled D3/D4
    hold the shared state, reset and helpers.
  agent.py D4 (lines 162-165, 355-364, 394-400, 495-522, 527-599, 874-879) — the history is stored per turn as blocks
    (_stored_turn) and re-sent in the tool protocol (_build_messages, _user_turn_after): one
    tool_result per stored tool_use, the screenshot inside it when retained and M3's placeholder text when folded,
    the driver's "User response:" beside it; _content_to_anthropic passes the protocol's block types through.
  agent.py D6 (lines 800-804, 868-872, 891-897, 913-916) — the api_log entry's request_time is stamped once in predict before
    the first attempt (upstream stamped it at save time, so the record carried no latency); response_time stays
    the save-time stamp, so the two bracket the call, retries included. Diagnostics only; no request byte moves.
  The runner, scripts/osworld/run_multienv_deepseek_baseline.py, labels its own hunks R1, R2: R1 (lines 277-285, 542-554)
    the --volume_size port, R2 (lines 48-51) the control-agent import.
  The line ranges above are as of this copy; the markers are the authority.
Deltas carried: none.
end of GraphPC header -->
# M3 Agent

Standalone agent module for evaluating an M3-trained model on
[OSWorld](https://github.com/xlang-ai/OSWorld). Provides the system
prompt, response parser and screenshot handling needed to drive M3
through an Anthropic-Messages-compatible endpoint.

LLM transport: the [`anthropic` Python SDK](https://github.com/anthropics/anthropic-sdk-python).
Endpoint URL / API key / model name come from the standard Anthropic
SDK env vars (`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` /
`ANTHROPIC_MODEL`) or from the equivalent constructor kwargs.

## Files

| File | Purpose |
| --- | --- |
| `prompts.py`  | `M3_SYSTEM_PROMPT_TEMPLATE` — formatted with `{date_str, client_password}` and passed as the system message |
| `parser.py`   | `parse_m3_response()` — extracts `<tool_call>` blocks and converts them to pyautogui code |
| `agent.py`    | `M3Agent` class — OSWorld `predict` / `reset` / `set_api_log_dir` lifecycle + Anthropic transport |
| `__init__.py` | Re-exports `M3Agent`, prompt template, parser functions |

## Behaviour

- 1920×1080 screenshots, sent verbatim (no resize)
- `[0, 1000]` normalised coordinates (`coordinate_type="relative"`)
- System prompt = `M3_SYSTEM_PROMPT_TEMPLATE.format(date_str=..., client_password=...)`,
  passed as the first `{"role": "system", ...}` message
- 16-action computer-use tool enum (`left_click`, `key`, `type`,
  `scroll`, `wait`, `done`, `screenshot`, …)
- `<tool_call>{...}</tool_call>` XML-wrapped JSON tool call around
  `{"name": "computer", "arguments": {...}}`
- Image truncation: when the recent-screenshot count exceeds a lower
  bound `keep_min`, older ones are replaced with a short text placeholder
  in chunks of `chunk` steps. The actual count oscillates in
  `[keep_min, keep_min+chunk-1]` — chunking keeps the message prefix
  byte-stable between drops so the Anthropic prompt cache can be reused.
  See [Image truncation](#image-truncation) below for the exact policy.
- Stop sequences: `["</tool_call>", "Perform the next action. Perform"]`
- `[INFEASIBLE]` token → returns the `FAIL` special action
- BPE-artifact recovery for malformed action names
  (e.g. `_lick` → `_click`, `"key_ "` → `"key_"`, truncated tail tokens)

## Image truncation

OSWorld trajectories can run 50–100 steps. Sending every screenshot back
to the model each turn would blow past context limits and is wasteful —
old screenshots are rarely needed once the page has moved on. M3 keeps
only the most recent screenshots and replaces older ones with a short
`Tool result: Success` text placeholder.

**Two knobs** (constructor kwargs on `M3Agent`):

| Kwarg | Default | Meaning |
| --- | --- | --- |
| `only_n_most_recent_images` (alias: `max_trajectory_length`) | `10` | **Lower bound** — call this `keep_min`. After a drop, this many recent screenshots are guaranteed to remain. **Not an upper bound** — see range below. |
| `image_truncation_threshold` | `20` | **Chunk size** — call this `chunk`. Drops happen in batches of this size; partial chunks are never dropped. |

> ⚠️ **Counter-intuitive but important:** despite the name
> `only_n_most_recent_images=10`, the agent does **not** cap images at
> 10. The actual number of recent screenshots sent each turn is
> `keep_min + (k mod chunk)` where `k` is the current step count — it
> oscillates between `keep_min` and `keep_min + chunk − 1`. With the
> defaults that's **10 to 29** recent screenshots, plus the always-kept
> initial screenshot = **11 to 30** images per request.

**The algorithm** (see `agent.py:_build_messages`):

```python
k = len(self.responses)               # how many turns have happened so far
remove = max(0, k - keep_min)         # candidate count over the lower bound
remove -= remove % chunk              # round DOWN to a multiple of chunk
                                      # (don't drop a partial chunk)
```

The first `remove` tool-result screenshots get replaced with the text
placeholder; the rest stay as images. The **initial screenshot (step 0)
is always kept** — it's the entry-point observation that the instruction
references, so dropping it would orphan the task.

**Worked example** with defaults `keep_min=10`, `chunk=20`:

| Step `k` | `k − keep_min` | `remove` after rounding | Images sent |
| ---: | ---: | ---: | ---: |
| 9  |  −1 |  0 | 1 (initial) + 9 = **10** |
| 10 |   0 |  0 | 1 + 10 = **11** |
| 19 |   9 |  0 | 1 + 19 = **20** |
| 29 |  19 |  0 | 1 + 29 = **30** ← peak before first drop |
| 30 |  20 | 20 | 1 + 10 = **11** ← first drop fires |
| 49 |  39 | 20 | 1 + 29 = **30** ← peak again |
| 50 |  40 | 40 | 1 + 10 = **11** ← second drop fires |

The image count sawtooths up to `keep_min + chunk` then resets to
`keep_min + 1` — it does **not** stay flat at `keep_min`. This is
deliberate.

**Why chunked drops?** Anthropic's prompt cache hashes the message
prefix and gives you a discount when consecutive requests share a long
prefix (5-minute TTL). If we dropped one image per step, the message
prefix would change every single turn and the cache would never hit.
By dropping in chunks of `chunk=20`, the prefix is byte-stable for 20
consecutive requests — the cache gets ~20× reuse, latency drops, and
input-token spend on long trajectories drops by a similar factor.

**Tuning.** Increase `keep_min` if the model loses track of state that
scrolled off screen. Increase `chunk` if your endpoint has a longer
cache TTL or you want larger cache windows; decrease it (e.g. to `5`)
if you're context-budget-constrained and willing to pay more in cache
misses. Setting `chunk=1` makes this degenerate to plain "keep last
`keep_min`" with no chunking and no cache stability. The runner flag
`--max_trajectory_length` is a shortcut for `keep_min`; `chunk` is
constructor-only.

## Alignment with OSWorld runners

`M3Agent.__init__` accepts the same kwargs `PromptAgent` does, so
`run.py` and `scripts/python/run_multienv_*.py` work without
modification:

- `platform`, `model`, `max_tokens`, `top_p`, `temperature`,
  `action_space`, `observation_type`, `client_password` — direct
  semantic parity
- `max_trajectory_length` — mapped to `only_n_most_recent_images`
  (both bound how many past observations the model still sees)
- `a11y_tree_max_tokens` — accepted but discarded (M3 is screenshot-only)
- `**_unused` — any other future-runner kwarg silently absorbed
- `platform="macos"` — appends `MACOS_KEYBOARD_MAPPING` after the
  formatted system prompt
- `reset(_logger, vm_ip=...)` — `vm_ip` silently absorbed
  (M3 routes through the LLM transport, not the VM)

`M3Agent.predict(instruction, obs)` returns
`[response_text, pyautogui_code]` — same shape as `PromptAgent.predict`.
`pyautogui_code` items that are the special tokens `WAIT` / `DONE` /
`FAIL` / `CALL_USER` are kept as standalone strings so
`desktop_env.env.step` recognises them.

`set_api_log_dir(path)` writes `NNN_<ts>.json` per LLM call — useful for
offline analysis of per-step request/response bodies and retry metadata.
The saved `request_body` is the exact Anthropic Messages API payload that
went over the wire (`system` at top level, `image.source.base64` blocks,
`stop_sequences` key) — it can be replayed verbatim via
`anthropic.Anthropic().messages.create(**request_body)`.

## Minimal usage

```python
from mm_agents.m3 import M3Agent

agent = M3Agent(
    base_url="https://your-anthropic-compatible-endpoint",
    api_key="sk-...",           # or a JWT — header style auto-detected by prefix
    model="<model-name>",
    max_tokens=8192,
    temperature=0.6,
)
agent.reset(runtime_logger)
if hasattr(agent, "set_api_log_dir"):
    agent.set_api_log_dir(os.path.join(example_result_dir, "api_logs"))

response, actions = agent.predict(instruction, obs)
# actions is a list of pyautogui-code strings or special tokens
# ("WAIT" / "DONE" / "FAIL" / "CALL_USER") that desktop_env.env.step accepts
```

`base_url` / `api_key` / `model` also accept defaults from
`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` env
vars. `ANTHROPIC_AUTH_TOKEN` is accepted as a legacy alias for
`ANTHROPIC_API_KEY`.

If your endpoint requires extra HTTP headers (e.g. enterprise-gateway
routing hints, `anthropic-beta` flags, observability tracing), pass
them as a JSON dict via the `extra_headers` constructor kwarg,
`--extra_headers` runner flag, or `ANTHROPIC_EXTRA_HEADERS` env var.
They are forwarded verbatim to the Anthropic SDK's `default_headers`.

```bash
# CLI flag
--extra_headers '{"X-Custom-Routing": "my-upstream"}'

# env var
export ANTHROPIC_EXTRA_HEADERS='{"X-Custom-Routing": "my-upstream"}'
```

## Running with the parallel runner

```bash
# Option A — env vars (matches the Anthropic SDK convention)
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export ANTHROPIC_API_KEY=sk-...
export ANTHROPIC_MODEL=<model-name>

uv run python scripts/python/run_multienv_m3.py \
    --headless --observation_type screenshot \
    --provider_name aws --num_envs 5 --max_steps 100 \
    --result_dir ./results \
    --test_all_meta_path evaluation_examples/test_v2.json

# Option B — CLI flags
uv run python scripts/python/run_multienv_m3.py \
    --base_url https://api.anthropic.com \
    --api_key sk-... \
    --model <model-name> \
    --headless --observation_type screenshot \
    --provider_name aws --num_envs 5 --max_steps 100 \
    --result_dir ./results \
    --test_all_meta_path evaluation_examples/test_v2.json
```

## Auth header auto-detection

Anthropic-compatible endpoints accept tokens via one of two headers:

| Token prefix | Header sent | SDK kwarg |
| --- | --- | --- |
| `sk-...` | `x-api-key: ...` | `api_key=` |
| anything else (e.g. a JWT) | `Authorization: Bearer ...` | `auth_token=` |

`M3Agent` picks by prefix automatically — no extra config needed for
either an API-key endpoint or a JWT-bearer proxy.

## Response-shape retry

`predict()` wraps the LLM call in a small retry loop (up to
`M3_MAX_LLM_RETRIES + 1` attempts, default 3) that resamples when:

1. `_call_llm` raises an exception (network / 5xx after the Anthropic
   SDK has already done its own retries)
2. The response parses to zero pyautogui actions despite a non-empty
   text body

Retries do not consume environment steps. Per-step retry metadata is
stored in the saved `api_logs/NNN_*.json` under `_retry_attempts` so
offline analysis can spot tasks that needed multiple resamples.

## Runner knobs

Optional CLI flags worth knowing about:

| Flag | Default | Effect |
| --- | --- | --- |
| `--enable_proxy` | off | Allow tasks whose config sets `"proxy": true` to run their proxy-aware setup. Off by default — tasks that request proxy will still run but their proxy step is skipped. |

Two env vars tune the multi-env runner's resilience to transient
cloud-provider errors (capacity / API throttling / network blips):

| Env var | Default | Effect |
| --- | --- | --- |
| `M3_TASK_MAX_REQUEUE` | `3` | Max times a single task may be put back on the queue after a transient infra error before giving up. |
| `M3_TASK_REQUEUE_BACKOFF_S` | `5.0` | Seconds to sleep before re-queueing, to give the cloud API a moment to recover. |

Two env vars control the on-wire screenshot encoding:

| Env var | Default | Effect |
| --- | --- | --- |
| `M3_IMAGE_FORMAT` | `JPEG` | `JPEG` or `PNG`. JPEG shrinks wire bytes ~6× with negligible pixel difference. |
| `M3_IMAGE_QUALITY` | `90` | JPEG quality (1–100). Ignored when format is PNG. |
