"""Launch preflight for the OSWorld 2.0 cells (MiniMax-M3, Qwen, DeepSeek): free checks, exit 1 on any FAIL.

Proves, before any money is spent, that BOTH arms (GraphPC = scripts/bash/
run_multienv_graphpc.sh, baseline = scripts/bash/run_multienv_m3_baseline.sh)
run exactly the pre-registered 28-task subset under an identical harness, and
that nothing in either launch path can raise at runtime or bias the comparison.
Prints a PASS/FAIL/WARN table (WARN never fails the run; it is for the README);
``--output_json`` dumps it.

Local mode (from the GraphPC repo root, conda env; no host access needed):

    python scripts/osworld/launch_preflight.py --mode local \\
        [--manifest ...] [--prereg ...]

  * the subset manifest ids == subset_prereg.json run_order == the ids
    shipped in this file, exactly, IN ORDER (the manifest's order is the run order);
  * both launch scripts under DRY_RUN=1 resolve to argv whose harness flags are
    identical (only the per-arm agent flags and the run-shape flags may differ);
  * each arm's runner argparse accepts every flag its script passes -- both
    are vendored here, so this needs no OSWorld-V2 clone (each runner
    imported with the harness modules stubbed);
  * every source in the sync list exists.

Host mode (on the run host, from the OSWorld-V2 checkout root, under uv):

    uv run --no-sync python scripts/python/launch_preflight.py --mode host \\
        [--num_envs_graphpc 4] [--num_envs_baseline 4] [--allow_api_calls] [--acknowledge_resume ARM ...]

  Repeats the local checks against the vendored files, then: pinned tag, the
  snapshot complete per the sync's own manifest and from a clean (and, with
  ``--expect_source_commit``, the expected) GraphPC commit, no
  modified tracked harness file, the python deps the arms need under THIS
  interpreter (cv2, desktop_env) and, on the Qwen cell, what the openai SDK under it does with
  an undecodable HTTP 200 (offline), the 28 task classes
  load with their overrides asserted against the prereg (no multi-phase task,
  user simulator only 007/034, no dated task, no proxy), assets +
  release manifest + default AMI + every per-task asset file, .env keys with
  the pinned evaluator config and the MiniMax key shape, STS identity, AMI
  availability, subnet + security group (+ the V2 inbound ports), orphan task
  VMs, vCPU quota vs the planned concurrency, the mocked websites the 28
  tasks use (DNS + HTTPS/HTTP probe from the host), cold store / no prior
  results (or ``--acknowledge_resume`` naming the arm whose resume is meant), no
  runner alive (a live GraphPC runner fails and skips the build check below),
  the vendored package constructing ``build_default`` with retrieval off (the
  arm's regime), the memory bound (the per-worker cost x num_envs + 1 GB must fit
  in MemAvailable; 2 GB/worker floor), and, only with ``--allow_api_calls``
  (cents): one tiny MiniMax call on each arm's route and one tiny evaluator
  call. Ends with the resolved regime for the run README.

``--cell qwen`` runs the same checks for the Qwen qwen3.7-plus cell (see CELLS): the control is
scripts/bash/run_multienv_qwen_baseline.sh + its vendored runner (parsed through its own config(),
its agent flags and default-on switches read from its argparse), the GraphPC launcher is resolved
under OSWORLD_CELL=qwen, the model key is OPENROUTER_API_KEY (an sk-or- key), and the one route
probe is the pinned OpenRouter endpoint, which must answer served by alibaba.

``--cell deepseek`` runs them for the DeepSeek deepseek-flash cell: the control is
scripts/bash/run_multienv_deepseek_baseline.sh + its vendored runner (upstream's M3 runner and agent with the
native-tool deltas), whose scored launch must carry exactly the pinned agent flags (DEEPSEEK_CONTROL_PINNED: the
Anthropic-format endpoint, the model, --max_tokens 65536, --thinking_budget 4096) and none of
the forbidden ones (DEEPSEEK_CONTROL_FORBIDDEN, --thinking_mode first); the GraphPC launcher is resolved under
OSWORLD_CELL=deepseek and reads the 28 into a cold tree, as the MiniMax cell's does; the model key
is DEEPSEEK_API_KEY (an sk- key); M3_THINKING_MODE / M3_THINKING_BUDGET must be unset in the launch environment;
and the two route probes are one tiny call on the GraphPC arm's route (the model echo checked) and, on the
control's route, the control package's own agent sending its six turn shapes in one tool_use/tool_result round
trip of three calls (a 400 on any blocks the control launch).

RETRIEVAL IS OFF on every scored GraphPC launch, all cells: every GraphPC run on the 28-task subset
runs --retrieval_off. The GraphPC launcher writes the flag into its command, this preflight REQUIRES it
on the resolved argv (REQUIRED_SCORED_FLAGS), and the runner refuses a launch without it (delta 15). No
check here makes a retrieval-ON launch safe (embedder, text-arm deps, relevance-threshold calibration):
no scored launch runs one.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import json
import logging
import os
import re
import resource
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import types

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from unittest import mock
from urllib import error as urllib_error, request as urllib_request

# scripts/osworld/<this> -> GraphPC repo root; scripts/python/<this> -> OSWorld-V2 checkout root.
_ROOT = Path(__file__).resolve().parents[2]

PINNED_TAG = "v2026.08.08"
RELEASE_MANIFEST = "benchmark_releases/osworld-v2-2026.08.08.json"
TASK_CLASS_META = "evaluation_examples/task_class_static_extract_v2026.08.08.json"  # shipped by hand next to the manifests
DEFAULT_INSTANCE_TYPE = "t3.xlarge"  # desktop_env/providers/aws/manager.py INSTANCE_TYPE
VCPU_QUOTA_CODE = "L-1216C47A"  # Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances
# The V2 public-evaluation guideline's inbound TCP ports (.claude/docs/benchmarks/osworld_v2.md, setup).
REQUIRED_INBOUND_PORTS = (22, 5910, 80, 3000, 5000, 8000, 8006, 8080, 8081, 9222)
DRY_RUN_MARKER = "DRY_RUN argv (one element per line):"
# The pre-registered subset (data/osworld_v2/subset_prereg.json),
# shipped here so the manifest is checked against a source that travels with the code. The
# name carries no count on purpose.
#
# Shipped in the PRE-REGISTERED RUN ORDER, not sorted, because the order is part of the identity:
# the V2 runner flattens the manifest into the shared worker queue in list order, so this sequence
# is what separates the twin pairs. Pinning only the id SET would leave the order checkable against
# the record alone — and replacing the record's `run_order` with the sorted ids, which puts two twin
# pairs one queue slot apart, would pass while still printing the stated gap from the untouched
# field beside it.
PREREG_RUN_ORDER = (
    "035", "042", "053", "048", "066", "090", "103", "107", "001", "007", "008", "010", "034",
    "043", "046", "057", "064", "067", "084", "092", "038", "102", "044", "068", "079", "087",
    "104", "108",
)
# The id SET, derived so the two can never disagree.
PREREG_SUBSET_IDS = tuple(sorted(PREREG_RUN_ORDER))
# The twin pairs the order exists to separate, and the gap it attains — both shipped so the gap is
# COMPUTED from the declared order rather than read back out of the record that declares it. The
# pair table is held against the record's own `twin_pairs` in check_manifest_vs_prereg: shipping it
# without comparing it would leave the gap computable over stale pairs, silently.
PREREG_TWIN_PAIRS = (
    ("035", "038"), ("042", "044"), ("044", "053"), ("048", "068"), ("066", "079"),
    ("066", "087"), ("087", "090"), ("103", "104"), ("107", "108"),
)
PREREG_RUN_ORDER_MIN_GAP = 20  # the solver's optimum for this subset (solve_subset_run_order.py)
# The DeepSeek control (upstream's M3 agent with the native-tool deltas, scripts/python/run_multienv_deepseek_baseline.py)
# launches at exactly these agent flags — its launcher's values (scripts/bash/run_multienv_deepseek_baseline.sh): the
# Anthropic-format endpoint, the cell's model, the cell's pre-registered 65536 output cap and thinking ON as an explicit
# budget (the launcher passes no --thinking_mode) — and with none of the agent flags below, each of which would move the
# control's regime (its thinking form, sampling, history depth, coordinate space, stop tokens or headers) away from the
# pre-registered one.
DEEPSEEK_CONTROL_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_CONTROL_PINNED = {"--base_url": DEEPSEEK_CONTROL_BASE_URL, "--model": "deepseek-flash", "--max_tokens": "65536",
                           "--thinking_budget": "4096"}
DEEPSEEK_CONTROL_FORBIDDEN = frozenset({"--thinking_mode", "--temperature", "--top_p", "--max_trajectory_length",
                                        "--coordinate_type", "--stop_token", "--extra_headers"})
# Environment variables upstream's M3 agent and runner READ for the control's thinking form and budget (the runner's
# --thinking_mode default and the agent's own fallback, resolved after load_dotenv()): the DeepSeek launcher passes the
# budget explicitly and no mode, so a value in the host's .env or shell would silently change the control's regime.
# Refused in the launch environment, both sources (check_env_keys).
DEEPSEEK_FORBIDDEN_ENV = ("M3_THINKING_MODE", "M3_THINKING_BUDGET")
# Subset-scoped expectations: what the subset's classes must declare. Both are empty for the
# 28-task subset, and that is the assertion — the benchmark's task 069 (its only MultiPhaseTask)
# and 002/018/024/065 (task_current_date) are outside it, so the multi-phase and task-date paths
# are unexercised. The multi-phase comparison is a set equality; the date check skips ids absent
# from the subset, so it is made symmetric: an UNEXPECTED date is a problem too. A check that
# cannot notice its own subject has left is worth less than one that fails loudly.
# Tasks EXPECTED to announce a website skip, pinned like MULTI_PHASE_TASKS below. An expected
# skip prints as information; anything outside this set is a WARN. A standing WARN that never
# changes is how a real one gets ignored, so the known exemption is pinned and only a NEW one
# raises. 043 calls prepare_stateful_website_urls(app=None) and takes its URLs from `state`.
EXPECTED_WEBSITE_SKIP_TASKS: tuple[str, ...] = ("043",)
MULTI_PHASE_TASKS: tuple[str, ...] = ()
TASK_CURRENT_DATE_TASKS: tuple[str, ...] = ()
GRAPHPC_MEMORY_FLOOR_PER_WORKER_MB = 2048  # the floor: MemAvailable must at least cover num_envs x 2 GB
# The memory-bound rule: FAIL when num_envs x the operating per-worker cost + this exceeds
# MemAvailable.
GRAPHPC_MEMORY_HEADROOM_MB = 1024
# Per-worker costs are HOST-level (MemAvailable), measured on a 16 GB / 4 vCPU host with each worker holding the
# embedder (Qwen3-Embedding-0.6B, CPU torch build):
#   * startup: four workers loading the embedder, MemAvailable sampled every 0.5 s from the pre-load baseline — about
#     2760 MB each, at once or 30 s apart (the launcher's --env_start_delay); the largest drop DURING the loads equals
#     the resident footprint (no transient).
#   * operating: one working worker after a few smoke steps, ~265 MB above its load footprint (screenshots,
#     history) — a LOWER BOUND of the growth over a 500-step task.
# The rule charges the OPERATING cost; the startup peak is a separate guard. Both are below the in-process rss delta
# (~3.4 GB) because the weights file is shared page cache. Under --retrieval_off a worker holds less (no embedder),
# so the bound is conservative.
GRAPHPC_OPERATING_PER_WORKER_MB = 3025    # working worker, lower bound: 4 x 3025 + 1024 = 13124 MB fits that host
GRAPHPC_MEASURED_PEAK_PER_WORKER_MB = 2760  # largest MemAvailable drop during four loads / 4
GRAPHPC_PEAK_HEADROOM_MIN_MB = 1024  # FAIL when MemAvailable - num_envs x the startup peak leaves less than this
GRAPHPC_MAX_NUM_ENVS = 4  # the launch script refuses more without FORCE_NUM_ENVS=1 — basis: a 4 vCPU run host
# shared by 4 GraphPC + 4 baseline workers, and the operating-cost memory rule (below) decides within that
# cap
# The cell's concurrency policy for a SCORED run, which launches the bare scripts (no NUM_ENVS exported): when the
# plan equals this policy, a launcher default that differs is a FAIL; any other plan (a smoke) gets a WARN naming
# the export the operator owes.
SCORED_RUN_NUM_ENVS = {"graphpc": 4, "baseline": 4}
RUNNER_PROCESS_PATTERN = r"python[^ ]* .*run_multienv_(graphpc|m3|m3_baseline|qwen|qwen_baseline|deepseek_baseline)\.py"

HARNESS_FLAGS = {
    "--headless", "--observation_type", "--action_space", "--screen_width", "--screen_height",
    "--test_all_meta_path", "--eval_version", "--max_steps", "--checkpoint_eval_mode",
    "--checkpoint_steps", "--save_model_eval_raw_info", "--provider_name", "--region",
    "--client_password", "--use_public_ip", "--volume_size",
    "--model",  # same-model paired arms: not a harness flag of the runner, but must be identical
}
EXPECTED_HARNESS_VALUES: dict[str, Any] = {
    "--headless": True,
    "--observation_type": "screenshot",
    "--action_space": "pyautogui",
    "--screen_width": "1920",
    "--screen_height": "1080",
    "--eval_version": "v2",
    "--max_steps": "500",
    "--checkpoint_eval_mode": "inline",
    "--checkpoint_steps": "150,300",
    "--save_model_eval_raw_info": True,
    "--provider_name": "aws",
    "--region": "us-east-1",
    "--client_password": "osworld-public-evaluation",
    # Non-None is what makes DesktopEnv expand the guest root partition; the AMI ships a ~29.5 GB
    # root on a 40 GB volume with no growpart/cloud-init and fills it ~6 min after boot. It re-runs on every task and
    # installs cloud-guest-utils in the guest each time, so both arms must carry the same value.
    "--volume_size": "60",
    "--model": "MiniMax-M3",
}
# May differ between arms (concurrency, output tree, worker start stagger: the GraphPC arm starts its
# workers 30 s apart; the baseline keeps upstream's 1 s default).
RUN_SHAPE_FLAGS = {"--num_envs", "--result_dir", "--env_start_delay"}
AGENT_FLAGS = {
    "graphpc": {
        "--backbone", "--support_set_root", "--retrieval_off", "--two_pass",
        "--completion_gate_off", "--completion_gate_window", "--max_tokens",
    },
    "baseline": {
        "--base_url", "--api_key", "--thinking_mode", "--thinking_budget", "--max_tokens",
        "--temperature", "--top_p", "--coordinate_type", "--max_trajectory_length",
        "--extra_headers", "--stop_token",
    },
}
# Agent flags a SCORED launch must not carry. Permitted in AGENT_FLAGS above (a smoke may use
# them), but a cell resolving with --two_pass or --completion_gate_off would produce a verdict
# for a method that is not the pre-registered surface. Same shape as
# SCORED_RUN_NUM_ENVS: the policy is stated here and the scored plan is held against it.
#
# ARM-SCOPED on purpose. --max_tokens is an ablation on the GraphPC arm (the run sends no cap)
# and the LEGITIMATE regime on the baseline arm, whose launcher passes 8192; a shared set would
# either fail the control or excuse the treatment.
FORBIDDEN_SCORED_FLAGS: dict[str, set[str]] = {
    "graphpc": {"--two_pass", "--completion_gate_off",
                 "--completion_gate_window", "--max_tokens"},
    "baseline": set(),
}
# Agent switches a SCORED launch must carry. --retrieval_off:
# every GraphPC run on the 28-task subset runs retrieval off, in every cell. Absent on a scored
# argv is a FAIL, on a smoke plan a WARN, like the pinned flags below.
REQUIRED_SCORED_FLAGS: dict[str, set[str]] = {"graphpc": {"--retrieval_off"}, "baseline": set()}
# Agent flags a SCORED launch must carry at exactly this value (argv string): absent, or any other
# value, is refused like a forbidden flag. Empty on the MiniMax cell; select_cell installs the cell's.
PINNED_SCORED_FLAGS: dict[str, dict[str, str]] = {"graphpc": {}, "baseline": {}}
REQUIRED_ENV_KEYS = (
    "MINIMAX_API_KEY", "ANTHROPIC_API_KEY",
    "OSWORLD_EVAL_MODEL_PROVIDER", "OSWORLD_EVAL_MODEL_API_KEY_ENV", "OSWORLD_EVAL_MODEL_NAME",
    "OSWORLD_USER_SIM_PROVIDER", "OSWORLD_USER_SIM_API_KEY_ENV", "OSWORLD_USER_SIM_MODEL",
    "WEBSITE_HOST_SUFFIX", "OSWORLD_FILE_BASE_URL",
    "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_SUBNET_ID", "AWS_SECURITY_GROUP_ID",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    # The client-VM auto-termination override. Not a credential and easy to lose for exactly that
    # reason: the host .env is rebuilt by filtering the repo's, so a rotation or a host rebuild
    # drops any line the repo copy never carried. Left at the provider default it kills machines
    # mid-episode, writes no result.txt, and silently removes the tasks still grinding at the step
    # cap — an arm's WORST episodes — from the paired comparison, biasing it in that arm's favour.
    # Presence alone is not enough (a wrong value is as fatal as a missing one), so check_env_keys
    # also holds it against MIN_CLIENT_VM_TTL_MINUTES by name.
    "DEFAULT_TTL_MINUTES",
)
EVALUATOR_MODEL = "claude-sonnet-4-6"  # both arms, same evaluator

# THE CELL (--cell). Every model-specific value above is the MiniMax-M3 cell's; CELLS holds what differs
# per cell and select_cell() installs it on every main() call (EXPECTED_HARNESS_VALUES["--model"],
# AGENT_FLAGS / FORBIDDEN_SCORED_FLAGS / PINNED_SCORED_FLAGS["baseline"], REQUIRED_ENV_KEYS, CELL), so the
# module-level values are MiniMax's. The Qwen control's agent flags are READ from its vendored runner's argparse
# (control_agent_flags) rather than listed here, so this file never restates the control's flag set, its
# thinking flag included. The DeepSeek control's flags are
# MiniMax's set, pinned and forbidden as DEEPSEEK_CONTROL_* state.
# Per cell: `key_prefix` (the model key's shape), `baseline_pinned` (the control's pinned agent flags; None = the
# Qwen table), `method_surface_key` (the record's method surface for the cell's GraphPC arm, its step budget among
# them), `forbidden_env` (environment variables the launch must not carry).
CELLS: dict[str, dict[str, Any]] = {
    "minimax": {
        "baseline_script": "run_multienv_m3_baseline.sh", "baseline_runner": "run_multienv_m3_baseline.py",
        "model": "MiniMax-M3", "model_key": "MINIMAX_API_KEY", "key_label": "MiniMax", "key_prefix": "sk-",
        "backbone_type": "minimax",
        "graphpc_store": "./results/osworld_v2/graphpc_minimax_m3_support_set",
        "vision_gate": "scripts/python/verify_minimax_vision.py", "launch_env": {"OSWORLD_CELL": "minimax"},
        "method_surface_key": "method_surface", "forbidden_env": (),
        "baseline_agent_flags": frozenset(AGENT_FLAGS["baseline"]),
        "baseline_forbidden": frozenset(FORBIDDEN_SCORED_FLAGS["baseline"]), "baseline_pinned": {},
        "route_probes": (("minimax_openai_route", "MiniMax reachable on the GraphPC arm's route (OpenAI-compatible)"),
                         ("minimax_anthropic_route", "MiniMax reachable on the baseline arm's route (/anthropic, x-api-key)")),
    },
    "qwen": {
        "baseline_script": "run_multienv_qwen_baseline.sh", "baseline_runner": "run_multienv_qwen_baseline.py",
        "model": "qwen/qwen3.7-plus", "model_key": "OPENROUTER_API_KEY", "key_label": "OpenRouter", "key_prefix": "sk-or-",
        "backbone_type": "qwen",
        "graphpc_store": "./results/osworld_v2/graphpc_qwen37plus_support_set",
        "vision_gate": "scripts/python/verify_qwen_vision.py", "launch_env": {"OSWORLD_CELL": "qwen"},
        "method_surface_key": "method_surface_qwen", "forbidden_env": (),
        "baseline_agent_flags": None, "baseline_forbidden": None, "baseline_pinned": None,   # read from the vendored runner
        "route_probes": (("qwen_openrouter_route", "Qwen reachable on the pinned OpenRouter route, served by alibaba "
                                                   "(both arms: same base, key and provider pin)"),),
    },
    "deepseek": {
        "baseline_script": "run_multienv_deepseek_baseline.sh", "baseline_runner": "run_multienv_deepseek_baseline.py",
        "model": "deepseek-flash", "model_key": "DEEPSEEK_API_KEY", "key_label": "DeepSeek", "key_prefix": "sk-",
        "backbone_type": "deepseek",
        "graphpc_store": "./results/osworld_v2/graphpc_deepseek_flash_support_set",
        "vision_gate": "scripts/python/verify_deepseek_vision.py", "launch_env": {"OSWORLD_CELL": "deepseek"},
        "method_surface_key": "method_surface_deepseek", "forbidden_env": DEEPSEEK_FORBIDDEN_ENV,
        "baseline_agent_flags": frozenset(AGENT_FLAGS["baseline"]),
        "baseline_forbidden": DEEPSEEK_CONTROL_FORBIDDEN, "baseline_pinned": dict(DEEPSEEK_CONTROL_PINNED),
        "route_probes": (("deepseek_openai_route", "DeepSeek reachable on the GraphPC arm's route (api.deepseek.com, "
                                                   "OpenAI-compatible; the reply echoes deepseek-flash)"),
                         ("deepseek_anthropic_route", "DeepSeek accepts the control's wire shapes on its route "
                                                      "(api.deepseek.com/anthropic): the control package's own history "
                                                      "code sends the six turn shapes in one tool_use/tool_result round "
                                                      "trip (three calls); a 400 on any blocks the control launch")),
    },
}
CELL: dict[str, Any] = CELLS["minimax"]
_REQUIRED_ENV_KEYS_SHARED = REQUIRED_ENV_KEYS[1:]
# A scored Qwen control launch leaves every agent default upstream's except the pinned ones below: these
# overrides are refused, as is switching off any of the runner's default-on mechanisms (their --no- forms,
# read from its argparse).
QWEN_CONTROL_FORBIDDEN = frozenset({"--base_url", "--api_key", "--smoke_only", "--temperature", "--top_p",
                                    "--top_k", "--max_tokens", "--request_timeout"})
# ...and these it must carry at exactly this value. --generation_timeout 300 (upstream's default is
# 200 s): the reason is at the control launcher's flag.
QWEN_CONTROL_PINNED = {"--generation_timeout": "300"}


def control_agent_flags(runner_source: str) -> tuple[frozenset[str], frozenset[str]]:
    """(every option the vendored Qwen control runner defines, incl. the --no- forms, minus the harness
    and run-shape flags; the --no- forms of its default-on switches). Read from its argparse by AST."""
    options: set[str] = set()
    switched_off: set[str] = set()
    for node in ast.walk(ast.parse(runner_source)):
        if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "parser.add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith("--")):
            continue
        name = node.args[0].value
        options.add(name)
        keywords = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
        if keywords.get("action") == "argparse.BooleanOptionalAction":
            options.add("--no-" + name[2:])
            if keywords.get("default") == "True":
                switched_off.add("--no-" + name[2:])
    return frozenset(options - HARNESS_FLAGS - RUN_SHAPE_FLAGS), frozenset(switched_off)


def select_cell(name: str, control_runner_source: Optional[str] = None) -> None:
    global CELL, REQUIRED_ENV_KEYS  # noqa: PLW0603
    CELL = CELLS[name]
    EXPECTED_HARNESS_VALUES["--model"] = CELL["model"]
    if CELL["baseline_agent_flags"] is not None:
        AGENT_FLAGS["baseline"] = set(CELL["baseline_agent_flags"])
        FORBIDDEN_SCORED_FLAGS["baseline"] = set(CELL["baseline_forbidden"])
        PINNED_SCORED_FLAGS["baseline"] = dict(CELL["baseline_pinned"] or {})
    else:
        flags, switched_off = control_agent_flags(control_runner_source or "")
        AGENT_FLAGS["baseline"] = set(flags)
        FORBIDDEN_SCORED_FLAGS["baseline"] = set(QWEN_CONTROL_FORBIDDEN | switched_off)
        PINNED_SCORED_FLAGS["baseline"] = dict(QWEN_CONTROL_PINNED)
    REQUIRED_ENV_KEYS = (CELL["model_key"],) + _REQUIRED_ENV_KEYS_SHARED
# The floor the runbook pins (.claude/docs/benchmarks/osworld_v2.md, Per-run agent steps:
# DEFAULT_TTL_MINUTES=1440). A full-length episode must fit inside it with room for the
# provisioning and evaluation either side; 1440 (24 h) is the recorded value and this check
# refuses anything below it rather than re-deriving a number the runbook already fixed.
MIN_CLIENT_VM_TTL_MINUTES = 1440
# Tracked files that may legitimately be modified on the host: per-account credential/settings
# files the V2 guideline tells the operator to fill in (proxy, Google). Never harness code.
MODIFIED_TRACKED_ALLOWLIST = ("evaluation_examples/settings/",)
# Written by src/benchmarks/osworld/sync.py on every sync: the list of every path it wrote (+ source
# provenance). The snapshot check reads it, so a new sync entry is covered automatically; a snapshot
# without it predates the manifest and must be re-synced.
VENDORED_MANIFEST = "mm_agents/graphpc/vendored_manifest.json"
# Copied by hand next to the official manifests (not part of the sync).
HAND_COPIED_HOST_FILES = (
    "evaluation_examples/prereg_task_list.json",
    "evaluation_examples/subset_prereg.json",
    TASK_CLASS_META,
)
HOST_PYTHON_DEPS = ("cv2", "desktop_env")
# How the task classes name the mocked websites: literal app names handed to
# desktop_env/controllers/website.py, which builds "<app>.<WEBSITE_HOST_SUFFIX>".
# The website-controller entry points a task class calls to name a mocked site. Resolving their
# arguments is what tells us which hosts to probe.
WEBSITE_CONTROLLER_FUNCTIONS = ("build_website_url", "resolve_website_url", "prepare_stateful_website_urls")
WEBSITE_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def run(self, name: str, fn: Callable[[], CheckResult]) -> CheckResult:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 — every check must land in the table
            result = CheckResult(name, "FAIL", f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=3))
        result.name = name
        self.results.append(result)
        print(f"[{result.status}] {name}")
        for line in result.detail.rstrip().splitlines():
            print(f"        {line}")
        return result

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "FAIL"]

    def render(self) -> str:
        width = max(len(r.name) for r in self.results) if self.results else 10
        lines = [f"{'check':<{width}}  status", f"{'-' * width}  ------"]
        lines += [f"{r.name:<{width}}  {r.status}" for r in self.results]
        counts = {s: sum(1 for r in self.results if r.status == s) for s in ("PASS", "FAIL", "WARN", "SKIP")}
        lines.append(f"PASS {counts['PASS']}  FAIL {counts['FAIL']}  WARN {counts['WARN']}  SKIP {counts['SKIP']}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {"results": [r.__dict__ for r in self.results], "failed": len(self.failed)}


def _status(problems: list[str], warnings: list[str]) -> str:
    return "FAIL" if problems else ("WARN" if warnings else "PASS")


# --------------------------------------------------------------------------- shared checks
def load_manifest(path: Path) -> dict[str, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"{path}: expected a non-empty {{domain: [ids]}} object")
    return {str(k): [str(i) for i in v] for k, v in manifest.items()}


def compare_prereg_class_tables(prereg: dict[str, Any], rows: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """The two tables the pre-registration records about the installed task classes,
    held against what the classes actually declare: ``volume_size_overrides`` and
    ``checkpoint_unsafe``.

    Like ``instance_type_overrides`` beside them, both are compared with what the
    classes declare: read from the record alone, a class whose disk size or
    checkpoint safety moved under an unchanged id would be invisible, and the
    record would describe a subset that no longer exists. Same shape as the
    instance_type pair: a declared value that disagrees is a problem, an
    undeclared one is a warning.

    A separate function because ``check_task_classes`` can only run where the
    benchmark checkout is, and a comparison nothing can exercise off the host is a
    comparison nobody has seen fail."""
    problems: list[str] = []
    warnings: list[str] = []
    expected_volumes = {str(k): v for k, v in (prereg.get("volume_size_overrides") or {}).items()}
    for task_id, expected in expected_volumes.items():
        if task_id in rows and rows[task_id]["volume_size"] != expected:
            problems.append(f"{task_id}: volume_size {rows[task_id]['volume_size']!r} != prereg {expected!r} — the "
                            f"AMI disk defect is worked around per task, so a changed size is a task that may run "
                            f"out of disk mid-episode")
    extra_volumes = {t: r["volume_size"] for t, r in rows.items() if r["volume_size"] and t not in expected_volumes}
    if extra_volumes:
        warnings.append(f"volume_size set by the class but absent from the prereg table: {extra_volumes}")
    declared_unsafe = sorted(str(t) for t in (prereg.get("checkpoint_unsafe") or []))
    actual_unsafe = sorted(t for t, r in rows.items() if not r["intermediate_eval_safe"])
    loaded_declared = sorted(t for t in declared_unsafe if t in rows)
    if actual_unsafe != loaded_declared:
        problems.append(f"checkpoint-unsafe tasks {actual_unsafe} != prereg checkpoint_unsafe {loaded_declared}: "
                        f"+{sorted(set(actual_unsafe) - set(loaded_declared))} "
                        f"-{sorted(set(loaded_declared) - set(actual_unsafe))}. The scorer reports those tasks' "
                        f"checkpoint columns as NA on the strength of this list, so a disagreement means it is "
                        f"NA-ing the wrong tasks")
    missing_declared = sorted(set(declared_unsafe) - set(rows))
    if missing_declared:
        warnings.append(f"prereg checkpoint_unsafe names task(s) not loaded here: {missing_declared} — unchecked, "
                        f"which is not the same as verified")
    return problems, warnings


def _run_order_min_gap(order: list[str]) -> tuple[Optional[int], Optional[str]]:
    """``(gap, reason_it_could_not_be_computed)``: the closest twin pair's
    queue-slot separation in ``order``, through the solver's own measure so a
    checker and the solver cannot define "gap" differently. The gap is ``None``
    only alongside a reason, and the caller turns that into a FAIL, never a skip.

    ``sys.path.append``, never ``insert``: never shadow the checkout's own modules;
    appended once, so sys.path does not grow per call.

    The catch is ``Exception``, not ``ValueError``. Anything else out of
    ``run_order_gaps`` — a malformed pair table, a duplicated id — would land in
    the report through the generic per-check handler as a traceback rather than as
    this function's own diagnosis, so the two failure routes would read differently
    for the same kind of defect. Loud either way; this way also legible."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.append(here)
    try:
        from solve_subset_run_order import run_order_gaps  # noqa: PLC0415
    except ImportError as exc:
        return None, f"solve_subset_run_order is not importable ({exc})"
    try:
        return int(run_order_gaps(list(order), [tuple(p) for p in PREREG_TWIN_PAIRS])["run_order_min_gap"]), None
    except Exception as exc:  # noqa: BLE001 — the caller reports it as a FAIL with this reason
        return None, f"{type(exc).__name__}: {exc}"


def check_manifest_vs_prereg(manifest_path: Path, prereg_path: Path) -> CheckResult:
    """Three-way: manifest ids == prereg subset_ids == the ids shipped in this file, and the
    manifest's ORDER == the prereg's run_order.

    The order is part of the identity, not presentation: the V2 runner flattens the manifest into
    the shared worker queue in list order, so the manifest's order is the run order for BOTH arms
    and is the only channel that carries it. A sorted manifest would run 042/044 and 103/104 one
    queue slot apart and destroy the twin pairs' cold-vs-seeded contrast, silently."""
    manifest = load_manifest(manifest_path)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    expected = [str(i) for i in prereg["subset_ids"]]
    run_order = [str(i) for i in prereg.get("run_order") or []]
    problems = []
    # THE PRE-REGISTRATION'S OWN DRIFT SPEAKS FIRST, before anything compared AGAINST it.
    #
    # Every other guard here holds the manifest against the prereg, so if the PREREG is the thing
    # that drifted they all fire with true messages whose invited fix is "make the manifest match
    # the prereg" — re-running the manifest writer, or editing ids — which PROPAGATES the drift.
    # With a prereg that loses one id and gains another against a correct manifest, the leading
    # message would otherwise be "manifest order != prereg run_order ... re-run the manifest writer",
    # and doing that would rewrite a correct manifest from a corrupt record.
    #
    # The shipped tuple is the only leg of the three-way that travels with the CODE rather than with
    # the record, so it is the one that can say the record is wrong. Most specific first — and the
    # test is not that the first message is true but that the fix it invites is the right one.
    if sorted(expected) != list(PREREG_SUBSET_IDS):
        problems.append(f"prereg subset_ids differ from the {len(PREREG_SUBSET_IDS)} ids shipped in this preflight: "
                        f"+{sorted(set(expected) - set(PREREG_SUBSET_IDS))} -{sorted(set(PREREG_SUBSET_IDS) - set(expected))}. "
                        f"FIX THE PRE-REGISTRATION, not the manifest: every check below compares the manifest against "
                        f"this record, so any of them reporting a difference is downstream of this one")
    if list(manifest) != ["tasks"]:
        problems.append(f"manifest domains {list(manifest)} != ['tasks']")
    ids = manifest.get("tasks", [])
    if len(set(ids)) != len(ids):
        problems.append("manifest has duplicate ids")
    # THE ORDER, IN THE SAME DISCIPLINE AS THE IDS ABOVE: the shipped sequence speaks before
    # anything compared against the record's. Checking only the record's own consistency here would
    # pass a prereg whose run_order had been replaced by the sorted ids — and the message that fired
    # downstream would tell the operator to re-run the manifest writer, which rewrites a CORRECT
    # manifest from the CORRUPT record and turns this check green. That is a true message whose
    # invited fix is wrong, which is worse than a false one.
    run_order_corrupt = False
    if not run_order:
        problems.append("prereg declares no run_order: the manifest order is the run order, so it "
                        "cannot be checked (the record is never edited on the host: copy the GraphPC repo's "
                        "subset_prereg.json to the host's evaluation_examples/ and re-sync)")
        run_order_corrupt = True
    elif run_order != list(PREREG_RUN_ORDER):
        run_order_corrupt = True
        if sorted(run_order) != sorted(PREREG_RUN_ORDER):
            problems.append(f"prereg run_order is not a permutation of the {len(PREREG_RUN_ORDER)} ids shipped in "
                            f"this preflight: +{sorted(set(run_order) - set(PREREG_RUN_ORDER))} "
                            f"-{sorted(set(PREREG_RUN_ORDER) - set(run_order))}")
        else:
            first = next(i for i, (a, b) in enumerate(zip(run_order, PREREG_RUN_ORDER)) if a != b)
            problems.append(
                f"prereg run_order is a REORDERING of the pre-registered one (first difference at position "
                f"{first + 1}: prereg {run_order[first:first + 3]} vs pre-registered "
                f"{list(PREREG_RUN_ORDER)[first:first + 3]}). FIX THE PRE-REGISTRATION, not the manifest: the "
                f"order is what separates the twin pairs, and rewriting the manifest FROM this record "
                f"would make every check below agree with it")
    # THE PAIR TABLE, BOUND THE SAME WAY THE IDS AND THE ORDER ARE. The gap below is computed over
    # PREREG_TWIN_PAIRS, which is pinned in this file. Compared against nothing, a pair table that
    # moved under an unchanged id set (a re-solve at a different twin rung, a hand edit) would leave
    # the gap computed over STALE pairs and reported as a fact about the run order, silently. The
    # shipped table speaks, as the shipped ids and the shipped order do: it travels with the CODE,
    # so it is the leg that can say the record is wrong.
    declared_pairs = [tuple(str(task_id) for task_id in pair)
                      for pair in (prereg.get("twin_pairs") or [])
                      if isinstance(pair, (list, tuple))]
    shipped_pairs = [tuple(pair) for pair in PREREG_TWIN_PAIRS]
    if declared_pairs != shipped_pairs:
        if sorted(declared_pairs) != sorted(shipped_pairs):
            problems.append(f"prereg twin_pairs differ from the {len(shipped_pairs)} pairs shipped in this "
                            f"preflight: +{sorted(set(declared_pairs) - set(shipped_pairs))} "
                            f"-{sorted(set(shipped_pairs) - set(declared_pairs))}. FIX THE PRE-REGISTRATION: the "
                            f"twin gap below is computed over the SHIPPED pairs, so a record whose pairs moved "
                            f"leaves that number true about a different table than the one the record declares")
        else:
            problems.append(f"prereg twin_pairs is a REORDERING of the pre-registered pairs "
                            f"({declared_pairs[:2]} vs {shipped_pairs[:2]}). FIX THE PRE-REGISTRATION; the pair "
                            f"table is part of the subset's identity, like its run order")

    # The gap the declared order actually achieves, computed — never the number beside it.
    if run_order and sorted(run_order) == sorted(PREREG_RUN_ORDER):
        measured_gap, gap_error = _run_order_min_gap(run_order)
        if measured_gap is None:
            problems.append(f"could not compute the twin-pair gaps of the declared run_order ({gap_error}), so "
                            f"the order's whole purpose was NOT CHECKED — solve_subset_run_order.py ships next "
                            f"to this file; re-sync the snapshot")
        else:
            if measured_gap < PREREG_RUN_ORDER_MIN_GAP:
                problems.append(f"the declared run_order separates its closest twin pair by only {measured_gap} "
                                f"queue slot(s); the pre-registered order attains {PREREG_RUN_ORDER_MIN_GAP}. A "
                                f"short gap destroys the twin pairs' cold-vs-seeded contrast, which is the reason "
                                f"the order exists")
            recorded_gap = prereg.get("run_order_min_gap")
            if recorded_gap != measured_gap:
                problems.append(f"prereg run_order_min_gap={recorded_gap!r} but the declared run_order measures "
                                f"{measured_gap}: the record's stated gap is not its order's gap")
    if not run_order_corrupt and ids != run_order:
        first = next((i for i, (a, b) in enumerate(zip(ids, run_order)) if a != b), min(len(ids), len(run_order)))
        problems.append(f"manifest order != prereg run_order (first difference at position {first + 1}: "
                        f"manifest {ids[first:first + 3]} vs run_order {run_order[first:first + 3]}); "
                        f"the manifest order IS the run order; the task list is frozen, so the manifest must "
                        f"equal it")
    extra = sorted(set(ids) - set(expected))
    missing = sorted(set(expected) - set(ids))
    if extra:
        problems.append(f"ids not in the prereg subset: {extra}")
    if missing:
        problems.append(f"prereg ids missing from the manifest: {missing}")
    if prereg.get("subset_n") is not None and len(expected) != prereg["subset_n"]:
        problems.append(f"prereg subset_n={prereg['subset_n']} but {len(expected)} subset_ids")
    detail = f"{len(ids)} ids in {manifest_path.name}; prereg subset_n={prereg.get('subset_n')} ({prereg_path.name}); {len(PREREG_SUBSET_IDS)} ids shipped in the preflight"
    if problems:
        return CheckResult("", "FAIL", detail + "\n" + "\n".join(problems))
    # COMPUTED, not `prereg.get('run_order_min_gap')`: the record's own claim about itself could
    # report a healthy gap over an order whose closest pair is adjacent.
    return CheckResult("", "PASS", detail + f"; all three identical, in the pre-registered run_order over the "
                                            f"{len(PREREG_TWIN_PAIRS)} pre-registered twin pairs "
                                            f"(min twin gap {_run_order_min_gap(run_order)[0]}, computed)")


def check_task_list_defaults(graphpc_script: Path, baseline_script: Path, manifest: Path) -> CheckResult:
    """Each launcher's OWN default task list, resolved with nothing exported, from a scratch directory laid out as
    the checkout's evaluation_examples/ (the scored launch runs the bare scripts, so a default is what it reads):
    both launchers read the 28."""
    with tempfile.TemporaryDirectory(prefix="preflight_task_lists_") as tmp:
        (Path(tmp) / "evaluation_examples").mkdir()
        (Path(tmp) / "evaluation_examples" / "prereg_task_list.json").write_bytes(manifest.read_bytes())
        resolved = {arm: parse_runner_argv(dry_run_argv(script, cwd=Path(tmp), extra_env=CELL["launch_env"]))[1]
                    .get("--test_all_meta_path") for arm, script in (("graphpc", graphpc_script), ("baseline", baseline_script))}
    expected = "evaluation_examples/prereg_task_list.json"
    wrong = {arm: (value, expected) for arm, value in resolved.items() if value != expected}
    if wrong:
        return CheckResult("", "FAIL", f"launcher default task list (resolved, expected): {wrong}")
    return CheckResult("", "PASS", f"the bare launchers read {resolved}")


def dry_run_argv(script: Path, *, cwd: Path, extra_env: Optional[dict[str, str]] = None) -> list[str]:
    """Run a launch script under DRY_RUN=1 in a minimal environment; return its argv."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "DRY_RUN": "1"}
    env.update(extra_env or {})
    proc = subprocess.run(["bash", str(script)], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"{script.name} DRY_RUN exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}")
    lines = proc.stdout.splitlines()
    if DRY_RUN_MARKER not in lines:
        raise RuntimeError(f"{script.name} DRY_RUN printed no argv marker; stdout was: {proc.stdout[:300]!r}")
    return lines[lines.index(DRY_RUN_MARKER) + 1:]


def parse_runner_argv(argv: list[str]) -> tuple[str, dict[str, Any]]:
    """['uv','run','python','<runner.py>', flags...] -> (runner path, {flag: value | True})."""
    script_index = next((i for i, tok in enumerate(argv) if tok.endswith(".py")), None)
    if script_index is None:
        raise ValueError(f"no runner .py in argv: {argv}")
    pairs: dict[str, Any] = {}
    tokens = argv[script_index + 1:]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"unexpected positional token {tok!r} in argv")
        if tok in pairs:
            raise ValueError(f"flag {tok} passed twice")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            pairs[tok] = tokens[i + 1]
            i += 2
        else:
            pairs[tok] = True
            i += 1
    return argv[script_index], pairs


def served_endpoint_matches_pin(served: Any, pinned_tags: Any) -> bool:
    """The served endpoint satisfies the route pin: case-insensitive, and a suffixed regional endpoint
    (``alibaba/<region>``) matches its base slug; a near-miss (``alibabacloud``) and an absent string do
    not."""
    pinned = [str(tag).lower() for tag in (pinned_tags or [])]
    observed = str(served or "").lower()
    return bool(observed) and any(observed == tag or observed.startswith(f"{tag}/") for tag in pinned)


def served_providers(response: Any) -> list[str]:
    """The providers the router metadata names as SELECTED, verbatim (empty when the metadata is absent)."""
    def mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        dump = getattr(value, "model_dump", None)
        return dump() if callable(dump) else {}
    metadata = mapping(getattr(response, "openrouter_metadata", None)
                       or mapping(response).get("openrouter_metadata"))
    available = mapping(metadata.get("endpoints")).get("available") or []
    return [str(entry.get("provider", "")) for entry in (mapping(e) for e in available) if entry.get("selected")]


def qwen_max_tokens_problem(graphpc: dict[str, Any]) -> Optional[str]:
    """On the Qwen profile the output cap is a profile value and wins over the runner's
    --max_tokens, so the flag would be a silent second source. Refused on every launch, scored or not."""
    if graphpc.get("--backbone") == "qwen" and graphpc.get("--max_tokens") is not None:
        return (f"graphpc: --max_tokens {graphpc['--max_tokens']!r} with --backbone qwen — the Qwen profile's cap "
                f"is the profile's own value and wins over the flag, so the flag would be a silent second source")
    return None


def check_generation_pin_bites(graphpc: dict[str, Any], baseline: dict[str, Any]) -> CheckResult:
    """The pinned --generation_timeout refuses the resolved control argv with the flag dropped and with
    upstream's default 200 in its place, as a scored launch."""
    value = PINNED_SCORED_FLAGS["baseline"].get("--generation_timeout")
    if value is None:
        return CheckResult("", "FAIL", "no --generation_timeout pin is installed for this cell's control")
    refusal = f"must carry --generation_timeout {value} exactly"
    doctored = {"without the flag": {k: v for k, v in baseline.items() if k != "--generation_timeout"},
                "with upstream's 200": {**baseline, "--generation_timeout": "200"}}
    missed = [label for label, argv in doctored.items()
              if refusal not in check_arm_argv_identity(graphpc, argv, scored=True).detail]
    if missed or refusal in check_arm_argv_identity(graphpc, baseline, scored=True).detail:
        return CheckResult("", "FAIL", f"the --generation_timeout {value} pin does not bite (missed: {missed}) or "
                                       f"refuses the resolved argv itself")
    return CheckResult("", "PASS", f"the resolved control argv carries --generation_timeout {value}; the same argv "
                                   f"{' and '.join(doctored)} is refused")


def check_deepseek_control_pins_bite(graphpc: dict[str, Any], baseline: dict[str, Any]) -> CheckResult:
    """The DeepSeek control's pins refuse the resolved control argv with each pinned flag dropped and with another
    value in its place, and its forbidden thinking flag (--thinking_mode) added is refused too, as a scored launch;
    the resolved argv itself is not refused."""
    pinned, forbidden = PINNED_SCORED_FLAGS["baseline"], FORBIDDEN_SCORED_FLAGS["baseline"]
    if pinned != DEEPSEEK_CONTROL_PINNED or "--thinking_mode" not in forbidden:
        return CheckResult("", "FAIL", f"the DeepSeek control's pins are not installed: pinned {pinned}, forbidden {sorted(forbidden)}")
    missed: list[str] = []
    for flag, value in pinned.items():
        refusal = f"must carry {flag} {value} exactly"
        doctored = {f"without {flag}": {k: v for k, v in baseline.items() if k != flag},
                    f"with {flag} changed": {**baseline, flag: value + "x"}}
        missed += [label for label, argv in doctored.items()
                   if refusal not in check_arm_argv_identity(graphpc, argv, scored=True).detail]
    if "must not carry ['--thinking_mode']" not in check_arm_argv_identity(
            graphpc, {**baseline, "--thinking_mode": "disabled"}, scored=True).detail:
        missed.append("with --thinking_mode disabled added")
    own = check_arm_argv_identity(graphpc, baseline, scored=True).detail
    if "must carry" in own or "must not carry" in own:
        missed.append("the resolved argv itself is refused")
    if missed:
        return CheckResult("", "FAIL", f"the control's pins do not bite: {missed}")
    return CheckResult("", "PASS", f"the resolved control argv carries {pinned} and no forbidden flag; each pin dropped or "
                                   f"changed is refused, --thinking_mode added is refused")


def check_forbidden_env_bites(ctx: HostContext) -> CheckResult:
    """Each variable in the cell's ``forbidden_env`` planted in a copy of the host's .env makes check_env_keys FAIL
    naming it; the real .env (checked beside this) is left alone."""
    missed = []
    for variable in CELL["forbidden_env"]:
        planted = HostContext(root=ctx.root, env={**ctx.env, variable: "disabled"})
        result = check_env_keys(planted)
        if result.status != "FAIL" or f"{variable} is set in the host .env" not in result.detail:
            missed.append(variable)
    if missed:
        return CheckResult("", "FAIL", f"planting {missed} in the .env is not refused")
    return CheckResult("", "PASS", f"each of {list(CELL['forbidden_env'])} planted in the .env is refused by name")


def check_retrieval_off_required_bites(graphpc: dict[str, Any], baseline: dict[str, Any]) -> CheckResult:
    """The required --retrieval_off refuses the resolved GraphPC argv with the switch dropped, as a
    scored launch, and does not refuse the resolved argv itself."""
    refusal = "a scored launch must carry --retrieval_off"
    dropped = {k: v for k, v in graphpc.items() if k != "--retrieval_off"}
    if refusal not in check_arm_argv_identity(dropped, baseline, scored=True).detail:
        return CheckResult("", "FAIL", "the GraphPC argv without --retrieval_off is not refused")
    if refusal in check_arm_argv_identity(graphpc, baseline, scored=True).detail:
        return CheckResult("", "FAIL", f"the resolved GraphPC argv is refused: it does not carry --retrieval_off "
                                       f"({graphpc.get('--retrieval_off')!r})")
    return CheckResult("", "PASS", "the resolved GraphPC argv carries --retrieval_off; the same argv without it is refused")


def check_arm_argv_identity(graphpc: dict[str, Any], baseline: dict[str, Any], scored: bool = True) -> CheckResult:
    """``scored``: this argv is the one a SCORED cell launches, so FORBIDDEN_SCORED_FLAGS is a FAIL
    rather than a WARN. Local mode and the bare-script resolution are scored by construction (a
    scored run launches the bare script); a smoke plan gets the WARN."""
    problems: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    # The task list: the pre-registered 28 on both arms, compared by file name (the harness flags below are compared
    # across the arms as well).
    for arm, flags in (("graphpc", graphpc), ("baseline", baseline)):
        if flags.get("--test_all_meta_path") is not None and not str(flags["--test_all_meta_path"]).endswith("prereg_task_list.json"):
            problems.append(f"{arm}: --test_all_meta_path is {flags['--test_all_meta_path']!r}, not prereg_task_list.json")
    # The step budget: the cell config's 500 on both arms, except on the Qwen cell, whose GraphPC arm runs the
    # adapter's budget for its backbone (800) while the control runs 500. Each arm against its own.
    budgets, where = _adapter_step_budgets(Path(__file__))
    step_budgets = {"graphpc": str(budgets[CELL["backbone_type"]]) if budgets and CELL["backbone_type"] in budgets else None,
                    "baseline": EXPECTED_HARNESS_VALUES["--max_steps"]}
    same_steps = step_budgets["graphpc"] == step_budgets["baseline"]
    if step_budgets["graphpc"] is None:
        problems.append(f"the adapter's step budget for {CELL['backbone_type']!r} cannot be read: {where}")
    elif not same_steps:
        for arm, flags in (("graphpc", graphpc), ("baseline", baseline)):
            if flags.get("--max_steps") != step_budgets[arm]:
                problems.append(f"{arm}: --max_steps is {flags.get('--max_steps')!r}, not this arm's {step_budgets[arm]!r}")
    for flag in sorted(HARNESS_FLAGS):
        if flag == "--max_steps" and not same_steps:
            continue
        g, b = graphpc.get(flag), baseline.get(flag)
        if g is None and b is None:
            if flag in EXPECTED_HARNESS_VALUES:
                problems.append(f"{flag} missing from BOTH arms (expected {EXPECTED_HARNESS_VALUES[flag]!r})")
            continue
        if g != b:
            problems.append(f"{flag} differs: graphpc={g!r} baseline={b!r}")
            continue
        expected = EXPECTED_HARNESS_VALUES.get(flag)
        if expected is not None and g != expected:
            problems.append(f"{flag}={g!r} but the cell config expects {expected!r} (env override left set?)")
    qwen_cap = qwen_max_tokens_problem(graphpc)
    if qwen_cap:
        problems.append(qwen_cap)
    for arm, pairs in (("graphpc", graphpc), ("baseline", baseline)):
        unclassified = sorted(set(pairs) - HARNESS_FLAGS - RUN_SHAPE_FLAGS - AGENT_FLAGS[arm])
        if unclassified:
            problems.append(f"{arm}: flags outside the harness / run-shape / {arm} agent sets: {unclassified}")
        forbidden = sorted(set(pairs) & FORBIDDEN_SCORED_FLAGS[arm])
        if forbidden:
            message = (f"{arm}: a scored launch must not carry {forbidden} — these change the METHOD "
                       f"under test (ablation / cap), so the cell would produce a verdict for something "
                       f"other than the pre-registered surface and nothing downstream would say so")
            (problems if scored else warnings).append(
                message if scored else message.replace("a scored launch must not carry", "this plan is not the scored one, but it carries"))
        for flag in sorted(REQUIRED_SCORED_FLAGS[arm]):
            if pairs.get(flag) is not True:
                message = (f"{arm}: a scored launch must carry {flag} (every GraphPC run on the 28-task "
                           f"subset runs retrieval off), and this one carries {pairs.get(flag)!r}")
                (problems if scored else warnings).append(
                    message if scored else message.replace("a scored launch must carry", "this plan is not the scored one; the scored launch carries"))
        for flag, value in sorted(PINNED_SCORED_FLAGS[arm].items()):
            if pairs.get(flag) != value:
                message = (f"{arm}: a scored launch must carry {flag} {value} exactly (the cell's pinned value), "
                           f"and this one carries {pairs.get(flag)!r}")
                (problems if scored else warnings).append(
                    message if scored else message.replace("a scored launch must carry", "this plan is not the scored one; the scored launch carries"))
        agent_only = {k: ("<redacted>" if k == "--api_key" else v) for k, v in pairs.items() if k in AGENT_FLAGS[arm]}
        info.append(f"{arm} agent flags: {agent_only}")
        info.append(f"{arm} run-shape: {{{', '.join(f'{k}: {pairs.get(k)!r}' for k in sorted(RUN_SHAPE_FLAGS))}}}")
    shared = {k: graphpc[k] for k in sorted(HARNESS_FLAGS) if k in graphpc and (same_steps or k != "--max_steps")}
    info.insert(0, f"identical harness flags ({len(shared)}): {shared}" + ("" if same_steps else
                   f"; --max_steps per arm: graphpc {graphpc.get('--max_steps')!r} (the adapter's budget for the "
                   f"backbone), baseline {baseline.get('--max_steps')!r} (the control's)"))
    info.insert(1, f"scored-run flag policy applied as {'FAIL' if scored else 'WARN'}; forbidden on the "
                   f"graphpc arm: {sorted(FORBIDDEN_SCORED_FLAGS['graphpc'])}; required on it: "
                   f"{sorted(REQUIRED_SCORED_FLAGS['graphpc'])}"
                   + (f"; pinned on the baseline arm: {PINNED_SCORED_FLAGS['baseline']}"
                      if PINNED_SCORED_FLAGS["baseline"] else ""))
    # SCOPE, stated because this verdict is easily read as covering more than it does: the launch scripts
    # end in "$@" and this check sees only the argv they resolve with no arguments. Anything
    # appended at launch is refused by osworld_cell_guard_launch_args in the shared cell config,
    # on BOTH arms — a scored launch appends nothing at all — which is where that gap is closed.
    info.insert(2, "SCOPE: resolved from the bare launch scripts; arguments appended at launch are not visible "
                   "here, and both launchers refuse any appended argument unless the launch declares itself "
                   "unscored (OSWORLD_UNSCORED_LAUNCH=1) with a result tree that is not either arm's scored one")
    return CheckResult("", _status(problems, warnings), "\n".join(problems + warnings + info))


def _harness_stubs() -> dict[str, types.ModuleType]:
    stubs: dict[str, types.ModuleType] = {}
    for name in ("_repo_path", "lib_run_single", "lib_results_logger", "desktop_env", "desktop_env.desktop_env",
                 "task_loader", "mm_agents", "mm_agents.graphpc", "mm_agents.graphpc.benchmarks", "mm_agents.graphpc.benchmarks.osworld",
                 "mm_agents.graphpc.benchmarks.osworld.graphpc_agent", "mm_agents.m3", "dotenv",
                 "mm_agents.qwen", "mm_agents.qwen.client", "mm_agents.deepseek"):
        stubs[name] = types.ModuleType(name)
    stubs["lib_run_single"].run_single_example = mock.MagicMock()
    stubs["lib_results_logger"].log_task_error = mock.MagicMock()
    stubs["desktop_env.desktop_env"].DesktopEnv = mock.MagicMock()
    stubs["task_loader"].load_task_config = mock.MagicMock()
    stubs["task_loader"].resolve_task_json_path = mock.MagicMock()
    stubs["mm_agents.graphpc"].OSWorldGraphPCAgent = mock.MagicMock()
    stubs["mm_agents.m3"].M3Agent = mock.MagicMock()
    stubs["mm_agents.deepseek"].M3Agent = mock.MagicMock()   # the DeepSeek control runner (its R2 import)
    stubs["mm_agents.qwen"].QwenAgent = mock.MagicMock()   # the Qwen control runner (its D1/D8 imports below)
    stubs["mm_agents.qwen.client"].QwenClient = mock.MagicMock()
    stubs["mm_agents.qwen.client"].OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
    stubs["mm_agents.qwen.client"].REASONING_ENABLED = {"enabled": True}
    stubs["mm_agents.qwen.client"].REASONING_DISABLED = {"effort": "none"}
    stubs["dotenv"].load_dotenv = lambda *a, **k: None
    return stubs


def _adapter_step_budgets(anchor: Path) -> tuple[Optional[dict[str, int]], str]:
    """The step budget per backbone the adapter defines, ``OSWORLD_MAX_STEPS_BY_BACKBONE`` (its values integer
    literals or ``OSWORLD_MAX_STEPS``), read from the source of the adapter next to ``anchor`` (this file or a
    runner: src/ in the repo, the vendored package in the checkout). Runner delta 15 refuses any other
    --max_steps for the backbone."""
    root = anchor.resolve().parents[2]
    for adapter in (root / "src" / "benchmarks" / "osworld" / "graphpc_agent.py",
                    root / "mm_agents" / "graphpc" / "benchmarks" / "osworld" / "graphpc_agent.py"):
        if adapter.is_file():
            names: dict[str, Any] = {}
            for node in ast.parse(adapter.read_text(encoding="utf-8")).body:
                target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                          else node.target if isinstance(node, ast.AnnAssign) else None)
                if isinstance(target, ast.Name) and node.value is not None:
                    names[target.id] = node.value
            default = names.get("OSWORLD_MAX_STEPS")
            table = names.get("OSWORLD_MAX_STEPS_BY_BACKBONE")
            if not (isinstance(default, ast.Constant) and type(default.value) is int and isinstance(table, ast.Dict)):
                return None, f"{adapter} defines no integer OSWORLD_MAX_STEPS and dict OSWORLD_MAX_STEPS_BY_BACKBONE"
            budgets: dict[str, int] = {}
            for key, value in zip(table.keys, table.values):
                if isinstance(value, ast.Name) and value.id == "OSWORLD_MAX_STEPS":
                    value = default
                if not (isinstance(key, ast.Constant) and isinstance(value, ast.Constant) and type(value.value) is int):
                    return None, f"{adapter}: OSWORLD_MAX_STEPS_BY_BACKBONE holds an entry that is not a literal budget"
                budgets[key.value] = value.value
            return budgets, str(adapter)
    return None, f"no GraphPC adapter under {root} (src/ or mm_agents/graphpc/)"


def check_runner_accepts_argv(runner: Path, argv: list[str]) -> CheckResult:
    """Import the runner file with the harness stubbed and the launch argv; argparse or a
    startup check that rejects it exits, which is the FAIL we want to see here, not on the host.
    The GraphPC runner's startup refusals (delta 15) run as they are: the step budget against the
    adapter's real OSWORLD_MAX_STEPS_BY_BACKBONE, read from the adapter file, and the retrieval-off refusal."""
    if not runner.exists():
        return CheckResult("", "FAIL", f"runner not found: {runner}")
    stubs = _harness_stubs()
    if runner.name == "run_multienv_graphpc.py":
        budgets, where = _adapter_step_budgets(runner)
        if budgets is None:
            return CheckResult("", "FAIL", f"{runner.name}'s step budget (delta 15) cannot be read: {where}")
        stubs["mm_agents.graphpc.benchmarks.osworld.graphpc_agent"].OSWORLD_MAX_STEPS_BY_BACKBONE = budgets
    script_index = next(i for i, tok in enumerate(argv) if tok.endswith(".py"))
    # Each launcher prints its key as <NAME_API_KEY> under DRY_RUN; the runners take the token as the key's value.
    sys_argv = [runner.name] + [("placeholder-key" if re.fullmatch(r"<[A-Z_]+_API_KEY>", tok) else tok)
                                for tok in argv[script_index + 1:]]
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    stderr = io.StringIO()
    stdout = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "logs").mkdir()
        previous_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with mock.patch.dict(sys.modules, stubs), \
                    mock.patch.object(sys, "argv", sys_argv), \
                    mock.patch.dict(os.environ, {"MINIMAX_API_KEY": "placeholder-key",
                                                 "OPENROUTER_API_KEY": "placeholder-key",
                                                 "DEEPSEEK_API_KEY": "placeholder-key"}), \
                    contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                spec = importlib.util.spec_from_file_location(f"preflight_runner_{runner.stem}_{time.time_ns()}", runner)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    # The Qwen control runner parses inside main(), so it has no module-level args:
                    # its own config() parses the same argv here.
                    parsed = getattr(module, "args", None)
                    if parsed is None and callable(getattr(module, "config", None)):
                        parsed = module.config(sys_argv[1:])
                except SystemExit as exc:
                    # argparse writes to stderr, the runner's startup checks print to stdout
                    said = (stdout.getvalue() + stderr.getvalue()).strip()[-800:]
                    return CheckResult("", "FAIL", f"{runner.name} exited {exc.code} on the launch argv:\n{said}")
        finally:
            os.chdir(previous_cwd)
            for handler in list(root_logger.handlers):
                if handler not in handlers_before:
                    root_logger.removeHandler(handler)
                    handler.close()
    return CheckResult("", "PASS", f"{runner.name} parsed {len(sys_argv) - 1} argv tokens; model={parsed.model!r} max_steps={parsed.max_steps} test_all_meta_path={parsed.test_all_meta_path!r}")


def check_launch_scripts(graphpc_script: Path, baseline_script: Path, config_script: Path, *, cwd: Path,
                         extra_env: Optional[dict[str, str]], report: Report,
                         graphpc_runner: Path, baseline_runner: Path,
                         expected_num_envs: Optional[dict[str, int]] = None) -> dict[str, list[str]]:
    """``expected_num_envs`` (host mode): each launcher is resolved under ``NUM_ENVS=<planned>`` — the same
    environment the operator launches with (``OSWORLD_CELL=<cell> NUM_ENVS=1 bash scripts/bash/run_multienv_graphpc.sh
    ...`` for a smoke, the script with nothing appended for a scored run) — and its resolved ``--num_envs`` must
    equal that count, so the memory
    and vCPU plans bind the launch and not just this run. A plan the launcher refuses (above its cap without
    FORCE_NUM_ENVS=1) fails here."""
    for script in (graphpc_script, baseline_script):
        report.run(f"launch script sources the shared cell config: {script.name}", lambda s=script: (
            CheckResult("", "PASS", f"sources {config_script.name}")
            if config_script.exists() and "osworld_cell_config.sh" in s.read_text(encoding="utf-8")
            else CheckResult("", "FAIL", f"{s} does not source {config_script.name} (or the config is missing)")))
    argvs: dict[str, list[str]] = {}
    for arm, script in (("graphpc", graphpc_script), ("baseline", baseline_script)):
        def _dry(s=script, a=arm):
            env = {**CELL["launch_env"], **(extra_env or {})}
            if expected_num_envs is not None:
                env["NUM_ENVS"] = str(expected_num_envs[a])
            argv = dry_run_argv(s, cwd=cwd, extra_env=env)
            argvs[a] = argv
            runner_path, _ = parse_runner_argv(argv)
            prefix = argv[:argv.index(runner_path)]
            if argv[:3] != ["uv", "run", "--no-sync"]:
                return CheckResult("", "FAIL", f"launcher does not run `uv run --no-sync` (a plain uv run re-syncs the venv to the lock at launch): {' '.join(prefix)}")
            detail = f"{len(argv)} argv elements; {' '.join(prefix)} {runner_path} (--no-sync present"
            if expected_num_envs is not None:
                launched = parse_runner_argv(argv)[1].get("--num_envs")
                planned = expected_num_envs[a]
                if str(launched) != str(planned):
                    return CheckResult("", "FAIL", f"launcher resolves --num_envs {launched!r} under NUM_ENVS={planned} but this preflight planned num_envs_{a}={planned}: "
                                                   f"the launch script does not honour NUM_ENVS (the memory and vCPU rules evaluated {planned})")
                detail += f"; --num_envs {planned} == planned, resolved under NUM_ENVS={planned})"
                # The bare script (nothing exported) is what a scored run launches: resolve it once more and hold
                # its default against the plan — a FAIL when the plan is the cell policy (a scored run must launch
                # the plan from the bare script), a WARN otherwise (a smoke: the operator owes the export).
                bare = parse_runner_argv(dry_run_argv(s, cwd=cwd, extra_env={**CELL["launch_env"], **(extra_env or {})}))[1].get("--num_envs")
                if str(bare) != str(planned):
                    scored = planned == SCORED_RUN_NUM_ENVS[a]
                    return CheckResult("", "FAIL" if scored else "WARN",
                                       f"launch with NUM_ENVS={planned} explicitly; the bare script launches {bare}"
                                       + (" — the plan is the cell policy for a scored run, which launches the bare script: fix the launcher default" if scored else "")
                                       + "\n" + detail)
                return CheckResult("", "PASS", detail + f"; the bare script also launches {bare}")
            return CheckResult("", "PASS", detail + ")")
        report.run(f"DRY_RUN resolves: {script.name}", _dry)
    if len(argvs) == 2:
        parsed = {arm: parse_runner_argv(argv)[1] for arm, argv in argvs.items()}
        # A scored cell launches the bare scripts under the cell's own concurrency policy; local
        # mode resolves exactly that argv, so "no plan given" is the scored shape, not an unknown.
        scored = expected_num_envs is None or expected_num_envs == SCORED_RUN_NUM_ENVS
        report.run("harness flags identical across arms; no ablation flag and --retrieval_off on a scored launch",
                   lambda: check_arm_argv_identity(parsed["graphpc"], parsed["baseline"], scored=scored))
        report.run("the required --retrieval_off refusal bites (negative control)",
                   lambda: check_retrieval_off_required_bites(parsed["graphpc"], parsed["baseline"]))
        if CELL["backbone_type"] == "qwen":
            report.run("the --max_tokens + --backbone qwen refusal bites (negative control)", lambda: (
                CheckResult("", "PASS", "a doctored GraphPC argv with --max_tokens 8192 --backbone qwen is refused; "
                                        "the same argv without --max_tokens is not")
                if qwen_max_tokens_problem({"--backbone": "qwen", "--max_tokens": "8192"})
                and not qwen_max_tokens_problem({"--backbone": "qwen"})
                else CheckResult("", "FAIL", "the --max_tokens + --backbone qwen refusal does not bite")))
            report.run("the control's pinned --generation_timeout refusal bites (negative control)",
                       lambda: check_generation_pin_bites(parsed["graphpc"], parsed["baseline"]))
        if CELL["backbone_type"] == "deepseek":
            report.run("the control's pinned agent flags and forbidden thinking flags bite (negative control)",
                       lambda: check_deepseek_control_pins_bite(parsed["graphpc"], parsed["baseline"]))
        report.run("GraphPC runner argparse accepts its launch argv",
                   lambda: check_runner_accepts_argv(graphpc_runner, argvs["graphpc"]))
        report.run(f"baseline runner (vendored {baseline_runner.name}) argparse accepts its launch argv",
                   lambda: check_runner_accepts_argv(baseline_runner, argvs["baseline"]))
    return argvs


# --------------------------------------------------------------------------- local mode
def local_checks(args: argparse.Namespace, report: Report) -> None:
    repo = Path(args.repo_root).resolve()
    manifest = Path(args.manifest).resolve()
    prereg = Path(args.prereg).resolve()
    report.run("subset manifest == prereg subset_ids == shipped ids", lambda: check_manifest_vs_prereg(manifest, prereg))
    if CELL["method_surface_key"] != "method_surface":
        report.run(f"the pre-registration declares {CELL['method_surface_key']} (this cell's method surface)",
                   lambda: check_method_surface_declared(json.loads(prereg.read_text(encoding="utf-8"))))
    scripts_dir = repo / "scripts" / "osworld"
    argvs = check_launch_scripts(
        scripts_dir / "run_multienv_graphpc.sh", scripts_dir / CELL["baseline_script"],
        scripts_dir / "osworld_cell_config.sh", cwd=repo, extra_env={"TEST_ALL_META_PATH": str(manifest)},
        report=report, graphpc_runner=scripts_dir / "run_multienv_graphpc.py",
        baseline_runner=scripts_dir / CELL["baseline_runner"],
    )
    if "graphpc" in argvs:
        report.run("the GraphPC arm's step budget: launch argv == adapter == record", lambda: check_graphpc_step_budget(
            json.loads(prereg.read_text(encoding="utf-8")), parse_runner_argv(argvs["graphpc"])[1]))
    report.run("each bare launcher's default task list is the 28", lambda: check_task_list_defaults(
        scripts_dir / "run_multienv_graphpc.sh", scripts_dir / CELL["baseline_script"], manifest))

    def _sync_sources() -> CheckResult:
        # Loaded by path, not imported: sync.py is not part of the vendored snapshot.
        sync_path = repo / "src" / "benchmarks" / "osworld" / "sync.py"
        spec = importlib.util.spec_from_file_location("osworld_sync_for_preflight", sync_path)
        sync_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync_module)
        SYNC_SOURCE_TARGETS, SYNCED_TARGETS = sync_module.SYNC_SOURCE_TARGETS, sync_module.SYNCED_TARGETS
        missing = [s for s, _ in SYNC_SOURCE_TARGETS if not (repo / s).exists()]
        targets = set(sync_module.vendored_target_paths())
        expected_targets = {"scripts/bash/osworld_cell_config.sh", "scripts/python/launch_preflight.py",
                            CELL["vision_gate"], VENDORED_MANIFEST}
        not_listed = sorted(expected_targets - targets)
        detail = f"{len(SYNC_SOURCE_TARGETS)} sources, {len(SYNCED_TARGETS)} clean targets, {len(targets)} paths in the vendored manifest"
        if missing or not_listed:
            return CheckResult("", "FAIL", detail + f"; missing sources {missing}; not in sync list {not_listed}")
        return CheckResult("", "PASS", detail + "; all sources exist; cell config, preflight, vision gate and the manifest are synced")
    report.run("sync list sources exist; the derived vendored list covers cell config, preflight, vision gate and the manifest", _sync_sources)


# --------------------------------------------------------------------------- host mode
@dataclass
class HostContext:
    """Host-side dependencies, overridable for tests (every network / harness touch goes through here)."""
    root: Path
    env: dict[str, str]
    num_envs_graphpc: int = 1
    num_envs_baseline: int = 1
    allow_api_calls: bool = False
    acknowledge_resume: frozenset[str] = frozenset()  # the arms whose prior results / store a resume accepts
    git_describe: Callable[[], str] = None  # type: ignore[assignment]
    git_status: Callable[[], str] = None  # type: ignore[assignment]
    python_import: Callable[[str], Any] = None  # type: ignore[assignment]
    load_task: Callable[[str], Any] = None  # type: ignore[assignment]
    boto3_client: Callable[[str], Any] = None  # type: ignore[assignment]
    ami_maps: Callable[[], dict[str, dict]] = None  # type: ignore[assignment]
    default_ami: Callable[[], str] = None  # type: ignore[assignment]
    build_agent: Callable[[bool, str], Any] = None  # type: ignore[assignment]
    grounding_temperature: Callable[[], Optional[float]] = None  # type: ignore[assignment]
    other_runners: Callable[[], list[str]] = None  # type: ignore[assignment]
    resolve_host: Callable[[str], list[str]] = None  # type: ignore[assignment]
    http_probe: Callable[[str], tuple[Optional[int], str]] = None  # type: ignore[assignment]
    tiny_call: Callable[[str], str] = None  # type: ignore[assignment]
    mem_available_mb: Callable[[], Optional[float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        defaults = {
            "git_describe": self._git_describe, "git_status": self._git_status, "python_import": self._python_import,
            "load_task": self._load_task,
            "boto3_client": self._boto3_client, "ami_maps": self._ami_maps,
            "default_ami": self._default_ami, "build_agent": self._build_agent,
            "grounding_temperature": self._grounding_temperature, "other_runners": self._other_runners,
            "resolve_host": self._resolve_host, "http_probe": self._http_probe,
            "tiny_call": self._tiny_call, "mem_available_mb": self._mem_available_mb,
        }
        for name, default in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, default)

    # ---- default implementations (real harness / network) ----
    def _git_describe(self) -> str:
        return subprocess.run(["git", "describe", "--tags", "--exact-match"], cwd=str(self.root),
                              capture_output=True, text=True, check=True).stdout.strip()

    def _git_status(self) -> str:
        return subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=str(self.root),
                              capture_output=True, text=True, check=True).stdout

    @staticmethod
    def _python_import(name: str) -> Any:
        return importlib.import_module(name)

    def _load_task(self, task_id: str) -> Any:
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from task_loader import load_task_config, resolve_task_json_path  # noqa: PLC0415
        kwargs = {"task_id": task_id, "base_dir": "evaluation_examples", "domain": "tasks", "eval_version": "v2"}
        return load_task_config(resolve_task_json_path(**kwargs), **kwargs)

    def _boto3_client(self, service: str) -> Any:
        import boto3  # noqa: PLC0415
        return boto3.client(service, region_name=self.env.get("AWS_REGION", "us-east-1"))

    def _ami_maps(self) -> dict[str, dict]:
        from desktop_env.providers.aws import manager  # noqa: PLC0415
        return {"Ubuntu": manager.UBUNTU_IMAGE_ID_MAP, "Windows": manager.WINDOWS_IMAGE_ID_MAP}

    def _default_ami(self) -> str:
        from desktop_env.providers.aws.manager import get_default_ami_id  # noqa: PLC0415
        return get_default_ami_id(region=self.env.get("AWS_REGION", "us-east-1"), screen_size=(1920, 1080), os_type="Ubuntu")

    def _build_agent(self, retrieval_off: bool, support_set_root: str) -> Any:
        from mm_agents.graphpc import OSWorldGraphPCAgent  # noqa: PLC0415
        return OSWorldGraphPCAgent.build_default(
            backbone_type=CELL["backbone_type"], model=CELL["model"], support_set_root=support_set_root,
            retrieval_off=retrieval_off, two_pass=False, enable_completion_gate=True, completion_gate_window=3,
            screen_size=(1920, 1080), max_tokens=None, client_password="osworld-public-evaluation",
        )

    @staticmethod
    def _grounding_temperature() -> Optional[float]:
        from mm_agents.graphpc.benchmarks.osworld.graphpc_agent import GROUNDING_TEMPERATURE  # noqa: PLC0415
        return GROUNDING_TEMPERATURE

    @staticmethod
    def _other_runners() -> list[str]:
        proc = subprocess.run(["pgrep", "-af", "run_multienv_"], capture_output=True, text=True)
        own = str(os.getpid())
        return [line for line in proc.stdout.splitlines()
                if re.search(RUNNER_PROCESS_PATTERN, line) and not line.startswith(own + " ")]

    @staticmethod
    def _resolve_host(host: str) -> list[str]:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)})

    @staticmethod
    def _http_probe(url: str) -> tuple[Optional[int], str]:
        """HEAD, then GET on 405; (status, note). A raised HTTPError still carries a status."""
        for method in ("HEAD", "GET"):
            req = urllib_request.Request(url, method=method, headers={"User-Agent": "graphpc-launch-preflight"})
            try:
                with urllib_request.urlopen(req, timeout=WEBSITE_PROBE_TIMEOUT_SECONDS) as resp:
                    return resp.status, method
            except urllib_error.HTTPError as exc:
                if exc.code == 405 and method == "HEAD":
                    continue
                return exc.code, method
            except Exception as exc:  # noqa: BLE001 — URLError, timeout, SSL: reported by the caller
                return None, f"{method}: {type(exc).__name__}: {exc}"
        return None, "no response"

    def _tiny_call(self, route: str) -> str:
        prompt = "Reply with the single word OK."
        if route == "minimax_openai_route":
            import openai  # noqa: PLC0415
            from mm_agents.graphpc.agents.backbone.minimax.main import _BASE_URL  # noqa: PLC0415
            client = openai.OpenAI(api_key=self.env["MINIMAX_API_KEY"], base_url=_BASE_URL)
            response = client.chat.completions.create(model="MiniMax-M3", messages=[{"role": "user", "content": prompt}], max_tokens=8)
            return f"{_BASE_URL} model={response.model} finish={response.choices[0].finish_reason}"
        if route == "minimax_anthropic_route":
            import anthropic  # noqa: PLC0415
            client = anthropic.Anthropic(api_key=self.env["MINIMAX_API_KEY"], base_url="https://api.minimax.io/anthropic")
            response = client.messages.create(model="MiniMax-M3", max_tokens=8, messages=[{"role": "user", "content": prompt}])
            return f"https://api.minimax.io/anthropic (x-api-key) model={response.model} stop={response.stop_reason}"
        if route == "qwen_openrouter_route":
            # The control's own route values (delta D1); the GraphPC arm carries the same values (keep the
            # two in step when editing either). One tiny call with the disable form and the router-metadata
            # header, which is the only way the served endpoint is observable (src/agents/backbone/qwen/main.py),
            # judged by served_endpoint_matches_pin.
            import openai  # noqa: PLC0415
            from mm_agents.qwen.client import (  # noqa: PLC0415
                OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL, PROVIDER_PIN, REASONING_DISABLED,
            )
            client = openai.OpenAI(api_key=self.env[OPENROUTER_API_KEY_ENV], base_url=OPENROUTER_BASE_URL, max_retries=0)
            response = client.chat.completions.create(
                model=CELL["model"], messages=[{"role": "user", "content": prompt}], max_tokens=16,
                extra_headers={"X-OpenRouter-Metadata": "enabled"},
                extra_body={"provider": PROVIDER_PIN, "reasoning": REASONING_DISABLED})
            served = served_providers(response)
            if not served:
                raise RuntimeError("endpoint identity not observable: the router metadata is absent from the reply — "
                                   "not a provider mismatch, and nothing about the route was proven; re-run the probe")
            if not all(served_endpoint_matches_pin(one, PROVIDER_PIN["only"]) for one in served):
                raise RuntimeError(f"served provider {served!r} is not the pinned {PROVIDER_PIN['only']}")
            return (f"{OPENROUTER_BASE_URL} model={response.model} served provider={served!r} (verbatim) "
                    f"finish={response.choices[0].finish_reason}")
        if route == "deepseek_openai_route":
            # The GraphPC arm's route: the vendored backbone's own base URL, the cell's model, no SDK retries; the
            # reply must echo the cell's model (an echo of another model id is not the cell's model).
            import openai  # noqa: PLC0415
            from mm_agents.graphpc.agents.backbone.deepseek.main import _BASE_URL  # noqa: PLC0415
            client = openai.OpenAI(api_key=self.env[CELL["model_key"]], base_url=_BASE_URL, max_retries=0)
            response = client.chat.completions.create(model=CELL["model"], messages=[{"role": "user", "content": prompt}],
                                                      max_tokens=8)
            if getattr(response, "model", None) != CELL["model"]:
                raise RuntimeError(f"the reply echoes model {getattr(response, 'model', None)!r}, not {CELL['model']!r}")
            return f"{_BASE_URL} model={response.model} finish={response.choices[0].finish_reason}"
        if route == "deepseek_anthropic_route":
            # The control's route, driven through the control package's OWN agent (mm_agents.deepseek, the synced
            # copy): its request builder, its history code, its client — so what is sent is what the control sends.
            from mm_agents.deepseek import M3Agent  # noqa: PLC0415
            agent = M3Agent(model=CELL["model"], base_url=DEEPSEEK_CONTROL_BASE_URL, api_key=self.env[CELL["model_key"]],
                            thinking_budget=DEEPSEEK_PROBE_THINKING_BUDGET, max_tokens=DEEPSEEK_PROBE_MAX_TOKENS,
                            only_n_most_recent_images=1, image_truncation_threshold=1)
            return "\n".join(deepseek_control_round_trip(agent))
        if route == "evaluator":
            from desktop_env.evaluators.model_client import generate_text  # noqa: PLC0415
            text = generate_text(prompt, options={"max_tokens": 8})
            return f"provider={self.env.get('OSWORLD_EVAL_MODEL_PROVIDER')} model={self.env.get('OSWORLD_EVAL_MODEL_NAME')} reply={text.strip()[:40]!r}"
        raise ValueError(route)

    @staticmethod
    def _mem_available_mb() -> Optional[float]:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
        except OSError:
            return None
        return None


# The control-route probe's regime: thinking on at a small budget, a small cap (the control runs 4096 / 65536; the
# shapes do not depend on the numbers), the control's own fold at K = T = 1 so that every stored turn but the last
# is FOLDED (the arithmetic is upstream's; only the numbers are the probe's).
DEEPSEEK_PROBE_THINKING_BUDGET = 1024
DEEPSEEK_PROBE_MAX_TOKENS = 2048
DEEPSEEK_PROBE_FRAME_SIZE = (128, 128)
DEEPSEEK_PROBE_INSTRUCTION = "Click the red square."
DEEPSEEK_PROBE_USER_RESPONSE = "User response: go on."
DEEPSEEK_PROBE_TEXT_TURN = "I will look at the screen first."
# The six turn shapes the probe sends (the control's deltas D3/D4), each the control
# package's own ``_stored_turn`` / ``_user_turn_after`` output: a 400 on the request that carries them all blocks
# the control launch, since the control would send them.
DEEPSEEK_PROBE_SHAPES = (
    "turn 1: a signed thinking block re-sent with the FIRST tool_use of the reply (a second one dropped), FOLDED: "
    "one tool_result (the placeholder text) beside the User response text",
    "turn 2: a reply with text and no tool_use, folded: upstream's placeholder user text, no tool_result",
    "turn 3: an empty reply stored as the '(no action)' placeholder, folded: the placeholder user text, no tool_result",
    "turn 4: a thinking-only reply stored as [signed thinking, '(no action)'], retained: a plain image user turn (the "
    "current screen), no tool_result",
)


def _probe_frame_png(size: tuple[int, int] = DEEPSEEK_PROBE_FRAME_SIZE) -> bytes:
    """A small grey frame with a red square in the middle: a referent for the probe's instruction."""
    from PIL import Image, ImageDraw  # noqa: PLC0415
    image = Image.new("RGB", size, (235, 235, 235))
    quarter = (size[0] // 4, size[1] // 4)
    ImageDraw.Draw(image).rectangle([quarter[0], quarter[1], size[0] - quarter[0], size[1] - quarter[1]], fill=(220, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _reply_summary(response: Any) -> str:
    blocks = [str(getattr(block, "type", "?")) for block in (getattr(response, "content", None) or [])]
    return f"model={getattr(response, 'model', None)!r} stop={getattr(response, 'stop_reason', None)!r} blocks={blocks}"


def _require_model_echo(response: Any, model: str, label: str) -> None:
    echo = getattr(response, "model", None)
    if echo != model:
        raise RuntimeError(f"{label}: the reply echoes model {echo!r}, not {model!r} ({_reply_summary(response)})")


def round_trip_shape_problems(body: dict[str, Any], tool_use_id: Any) -> list[str]:
    """What is wrong with the probe's third request, as the control's own builder composed it: every one of the
    DEEPSEEK_PROBE_SHAPES must be there, in the Anthropic-format wire shape (the message list after the system
    prompt). Empty when the request carries them all."""
    problems: list[str] = []
    messages = body.get("messages") or []
    if not (body.get("tools") and body.get("thinking", {}).get("type") == "enabled"):
        problems.append(f"the request declares tools={bool(body.get('tools'))} thinking={body.get('thinking')!r}: not the "
                        f"control's request (the computer tool, thinking enabled)")
    if len(messages) != 9:
        return problems + [f"{len(messages)} messages, not 9 (the opening user turn, then four stored turns each with "
                           f"its user turn, the current screen in the last)"]

    def types(message: dict[str, Any]) -> list[str]:
        content = message.get("content")
        return [str(block.get("type")) for block in content] if isinstance(content, list) else ["<str>"]

    def blocks(message: dict[str, Any], kind: str) -> list[dict[str, Any]]:
        content = message.get("content")
        return [block for block in content if block.get("type") == kind] if isinstance(content, list) else []

    opening, turn1, user1, turn2, user2, turn3, user3, turn4, user4 = messages
    if opening.get("role") != "user" or "image" not in types(opening) or "text" not in types(opening):
        problems.append(f"the opening user turn is {types(opening)}, not image + instruction")
    # turn 1: signed thinking + exactly one tool_use; folded: one tool_result (text placeholder) + the User response
    thinking1 = blocks(turn1, "thinking")
    if not thinking1 or not thinking1[0].get("signature"):
        problems.append(f"turn 1 carries no signed thinking block ({types(turn1)})")
    tool_uses1 = blocks(turn1, "tool_use")
    if len(tool_uses1) != 1 or tool_uses1[0].get("id") != tool_use_id:
        problems.append(f"turn 1 carries {len(tool_uses1)} tool_use block(s) (expected exactly the executed one, id {tool_use_id!r})")
    results1 = blocks(user1, "tool_result")
    if len(results1) != 1 or results1[0].get("tool_use_id") != tool_use_id \
            or [c.get("type") for c in results1[0].get("content") or []] != ["text"]:
        problems.append(f"the user turn after turn 1 is {types(user1)} with tool_result ids "
                        f"{[r.get('tool_use_id') for r in results1]}: expected exactly one tool_result for {tool_use_id!r} "
                        f"holding the folded placeholder text")
    if [b.get("text") for b in blocks(user1, "text")] != [DEEPSEEK_PROBE_USER_RESPONSE]:
        problems.append(f"the user turn after turn 1 does not carry the User response text beside the tool_result ({types(user1)})")
    # turn 2: text, no tool_use; folded: placeholder text, no tool_result
    if types(turn2) != ["text"] or blocks(turn2, "text")[0].get("text") != DEEPSEEK_PROBE_TEXT_TURN:
        problems.append(f"turn 2 is {types(turn2)}, not the one text block without a tool_use")
    if types(user2) != ["text"] or blocks(user2, "tool_result"):
        problems.append(f"the user turn after turn 2 is {types(user2)}, not the placeholder text alone")
    # turn 3: the '(no action)' placeholder; folded: placeholder text, no tool_result
    if types(turn3) != ["text"] or blocks(turn3, "text")[0].get("text") != "(no action)":
        problems.append(f"turn 3 is {types(turn3)} {[b.get('text') for b in blocks(turn3, 'text')]}, not the '(no action)' placeholder")
    if types(user3) != ["text"] or blocks(user3, "tool_result"):
        problems.append(f"the user turn after turn 3 is {types(user3)}, not the placeholder text alone")
    # turn 4: signed thinking + '(no action)'; retained: a plain image user turn, no tool_result
    thinking4 = blocks(turn4, "thinking")
    if not thinking4 or not thinking4[0].get("signature") or blocks(turn4, "tool_use") \
            or [b.get("text") for b in blocks(turn4, "text")] != ["(no action)"]:
        problems.append(f"turn 4 is {types(turn4)}, not [signed thinking, '(no action)']")
    if "image" not in types(user4) or blocks(user4, "tool_result"):
        problems.append(f"the user turn after turn 4 is {types(user4)}, not a plain image user turn")
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            problems.append(f"message {index} has role {message.get('role')!r}, not {expected_role!r}")
    return problems


def deepseek_control_round_trip(agent: Any) -> list[str]:
    """Three live calls through the control agent's own request builder, LLM call and history code — every
    message the control would send, composed by the control: (1) the opening request (the system prompt with the
    computer tool, the frame, the instruction); the reply's thinking and FIRST tool_use become turn 1 through
    ``_parse_native_reply`` + ``_stored_turn`` (a second tool_use is set aside, as the control does); turns 2 and 3
    are stored through the same storer from a text-only reply and an empty reply; (2) the request carrying turns
    1-3 folded (K = T = 1) behind the frame; the reply's thinking becomes turn 4, stored thinking-only; (3) the
    request carrying all four turns — the six shapes (DEEPSEEK_PROBE_SHAPES), checked by
    ``round_trip_shape_problems`` BEFORE it is sent — with the current frame last. Every reply must be accepted
    and echo the cell's model; the SDK's status error on any call (a 400 on a shape the control sends) propagates
    as the FAIL. Returns the detail lines."""
    from mm_agents.deepseek.agent import _encode_screenshot  # noqa: PLC0415 — the control's own encoder
    frame, _mime = _encode_screenshot(_probe_frame_png())
    width, height = DEEPSEEK_PROBE_FRAME_SIZE
    lines: list[str] = []
    agent.reset()
    agent.screenshots = [frame]
    text1, reply1 = agent._call_llm(agent._build_messages(DEEPSEEK_PROBE_INSTRUCTION, frame))
    _require_model_echo(reply1, agent.model, "call 1")
    agent._parse_native_reply(text1, width, height, width, height)
    if agent._executed_tool_use is None:
        raise RuntimeError(f"call 1 answered with no `computer` tool_use ({_reply_summary(reply1)}; text {text1[:80]!r}): the "
                           f"control's first step would be a no-op, and the tool_use/tool_result shapes cannot be sent")
    if not any(block.get("type") == "thinking" and block.get("signature") for block in agent._last_thinking_blocks):
        raise RuntimeError(f"call 1 answered with no signed thinking block ({_reply_summary(reply1)}): the control's "
                           f"regime (thinking on, re-sent) does not hold on this route")
    set_aside = len(agent._second_tool_use)
    turn1 = agent._stored_turn(text1)
    tool_use_id = turn1["tool_use_id"]
    lines.append(f"call 1 ({_reply_summary(reply1)}): {len(agent._last_tool_uses)} tool_use block(s), {set_aside} set aside "
                 + ("(the second-tool_use drop exercised)" if set_aside else
                    "(the second-tool_use drop NOT exercised: the model emitted one tool_use, which the probe cannot force)")
                 + f"; turn 1 stored as {[b.get('type') for b in turn1['blocks']]}")
    # Stored as predict stores a turn: the text in upstream's `responses` (the builder's turn count), the blocks
    # beside it, the driver's User response and the next frame.
    agent.responses, agent.turn_blocks, agent.user_responses = [text1], [turn1], [DEEPSEEK_PROBE_USER_RESPONSE]
    agent.screenshots.append(frame)
    agent._last_thinking_blocks, agent._executed_tool_use, agent._second_tool_use = [], None, []
    turn2 = agent._stored_turn(DEEPSEEK_PROBE_TEXT_TURN)
    turn3 = agent._stored_turn("")
    agent.responses += [DEEPSEEK_PROBE_TEXT_TURN, ""]
    agent.turn_blocks += [turn2, turn3]
    agent.user_responses += [None, None]
    agent.screenshots += [frame, frame]
    text2, reply2 = agent._call_llm(agent._build_messages(DEEPSEEK_PROBE_INSTRUCTION, frame))
    _require_model_echo(reply2, agent.model, "call 2")
    if not any(block.get("type") == "thinking" and block.get("signature") for block in agent._last_thinking_blocks):
        raise RuntimeError(f"call 2 answered with no signed thinking block ({_reply_summary(reply2)}): the thinking-only "
                           f"shape cannot be sent")
    lines.append(f"call 2 ({_reply_summary(reply2)}): turns 1-3 folded were accepted; turn 4 stored thinking-only")
    agent._executed_tool_use, agent._second_tool_use = None, []
    turn4 = agent._stored_turn("")
    agent.responses.append("")
    agent.turn_blocks.append(turn4)
    agent.user_responses.append(None)
    agent.screenshots.append(frame)
    messages = agent._build_messages(DEEPSEEK_PROBE_INSTRUCTION, frame)
    problems = round_trip_shape_problems(agent._build_request_body(messages), tool_use_id)
    if problems:
        raise RuntimeError("the probe's third request, as the control's builder composed it, does not carry the six "
                           "shapes — a probe defect, not a route finding: " + "; ".join(problems))
    text3, reply3 = agent._call_llm(messages)
    _require_model_echo(reply3, agent.model, "call 3")
    lines.append(f"call 3 ({_reply_summary(reply3)}): all six shapes accepted — " + "; ".join(DEEPSEEK_PROBE_SHAPES))
    return lines


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def check_vendored_snapshot(root: Path, expect_source_commit: Optional[str] = None) -> CheckResult:
    """Every path the last sync wrote (its own manifest) plus the hand-copied metadata files must exist;
    the snapshot must come from a clean tree (a dirty-tree snapshot has no reproducible commit, so a scored
    cell on it would carry a false provenance) and, when ``expect_source_commit`` is given, from that commit."""
    manifest_path = root / VENDORED_MANIFEST
    if not manifest_path.exists():
        return CheckResult("", "FAIL", f"{VENDORED_MANIFEST} missing: the snapshot predates the manifest or was never synced — re-sync from the GraphPC checkout")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        targets = [str(t) for t in manifest["targets"]]
    except (ValueError, KeyError, TypeError) as exc:
        return CheckResult("", "FAIL", f"{VENDORED_MANIFEST} unreadable ({type(exc).__name__}: {exc}); re-sync")
    expected = targets + list(HAND_COPIED_HOST_FILES)
    missing = [f for f in expected if not (root / f).exists()]
    source = manifest.get("source") or {}
    commit = str(source.get("commit") or "")
    provenance = f"synced {manifest.get('synced_at')} from GraphPC {commit[:12] or '?'}"
    problems = []
    if missing:
        problems.append(f"{len(missing)} vendored/metadata file(s) missing on the host (re-sync): {missing}")
    if source.get("synced_sources_dirty"):
        problems.append("the snapshot was synced from a DIRTY GraphPC tree (uncommitted edits under src/ or scripts/osworld/): "
                        "no reproducible commit — commit, re-sync, re-run")
    if source.get("commit") is None:
        problems.append("the manifest carries no source commit (sync ran outside a git checkout): re-sync from the repository")
    if expect_source_commit:
        wanted = expect_source_commit.strip().lower()
        if len(wanted) < 7:
            problems.append(f"--expect_source_commit {expect_source_commit!r} is shorter than 7 characters; pass a full or 7+ character sha")
        elif not commit.lower().startswith(wanted):
            problems.append(f"snapshot source commit {commit[:12] or '?'} != expected {expect_source_commit} (re-sync the host from that commit)")
        else:
            provenance += f" == expected {expect_source_commit}"
    if problems:
        return CheckResult("", "FAIL", "\n".join(problems + [provenance]))
    return CheckResult("", "PASS", f"{len(targets)} synced paths + {len(HAND_COPIED_HOST_FILES)} hand-copied files present; {provenance}")


def check_checkout_unmodified(ctx: HostContext) -> CheckResult:
    """Fairness/provenance: the TRACKED harness code both arms run must be the pinned tag's.

    Its scope is narrower than "the benchmark is pinned", and the difference matters: the 108 task
    classes under `evaluation_examples/task_class/task_*.py` are in the checkout's OWN `.gitignore`
    and ship from a gated HuggingFace dataset, so this check never sees them and neither does
    `git describe`. Task content — instructions, setup, evaluators — can change inside one tag with
    no git signal at all. What watches that is `check_task_classes` (nine per-task metadata fields
    held against the pre-registration), not this; the instruction text itself is not checked on a
    retrieval-off launch. Untracked
    files are fine only if excluded (the vendored snapshot); modified tracked files only under
    the settings allow-list."""
    modified, untracked, allowed = [], [], []
    for line in ctx.git_status().splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip()
        if code == "??":
            untracked.append(path)
        elif path.startswith(MODIFIED_TRACKED_ALLOWLIST):
            allowed.append(f"{code.strip()} {path}")
        else:
            modified.append(f"{code.strip()} {path}")
    problems = []
    if modified:
        problems.append(f"tracked harness files modified on the host (both arms would run a non-pinned harness): {modified}")
    if untracked:
        problems.append(f"untracked files not covered by .git/info/exclude: {untracked}")
    info = [f"allowed settings edits: {allowed}" if allowed else "no tracked file modified"]
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + info))


def check_python_deps(ctx: HostContext) -> CheckResult:
    """The arms' python deps under THIS interpreter (the one `uv run` gives the runners)."""
    problems, info = [], []
    for name in HOST_PYTHON_DEPS:
        try:
            module = ctx.python_import(name)
            info.append(f"{name} {getattr(module, '__version__', '')}".rstrip())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"import {name} failed under {sys.executable}: {type(exc).__name__}: {exc}")
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + info))


# What both Qwen arms' decode rule is written against: a blank JSON body of this shape (the SDK's decode message on
# it reads "Expecting value: line 25 column 1 (char 132)") and a non-JSON page.
UNDECODABLE_BLANK_BODY = b" " * 108 + b"\n" * 24
UNDECODABLE_PAGE = b"<html><body>Service busy</body></html>"


def check_sdk_undecodable_200(ctx: HostContext) -> CheckResult:
    """The openai SDK under THIS interpreter, offline (an httpx.MockTransport answers; no socket opens), does what
    both Qwen arms' decode rule catches at the provider call: a non-streamed create() on an HTTP 200 whose JSON
    body is blank raises json.JSONDecodeError, unwrapped; on a non-JSON content type it returns the body as a str;
    a streamed chunk that is not JSON raises json.JSONDecodeError while the stream is read. An SDK that wraps the
    error or parses such a body otherwise would leave the rule catching nothing."""
    try:
        openai, httpx = ctx.python_import("openai"), ctx.python_import("httpx")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("", "FAIL", f"import openai / httpx failed under {sys.executable}: {type(exc).__name__}: {exc}")

    def create(content: bytes, content_type: str, stream: bool = False) -> Any:
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, headers={"content-type": content_type}, content=content, request=request))
        client = openai.OpenAI(api_key="sk-or-offline-preflight", base_url="https://openrouter.invalid/api/v1",
                               max_retries=0, http_client=httpx.Client(transport=transport))
        response = client.chat.completions.create(model="probe", messages=[{"role": "user", "content": "probe"}],
                                                  stream=stream)
        return list(response) if stream else response

    chunk = b'data: {"id": "gen-offline", "choi\n\n'
    problems, info = [], [f"openai {getattr(openai, '__version__', '?')}, httpx {getattr(httpx, '__version__', '?')}"]
    for label, content, content_type, stream, want in (
            ("a blank JSON 200, non-streamed", UNDECODABLE_BLANK_BODY, "application/json", False, "decode error"),
            ("a non-JSON 200 (text/html), non-streamed", UNDECODABLE_PAGE, "text/html; charset=utf-8", False, "str"),
            ("a stream chunk that is not JSON", chunk, "text/event-stream", True, "decode error")):
        try:
            got = create(content, content_type, stream)
        except json.JSONDecodeError as exc:
            if want == "decode error":
                info.append(f"{label}: json.JSONDecodeError ({exc})")
            else:
                problems.append(f"{label}: raised json.JSONDecodeError ({exc}), not a str returned")
            continue
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{label}: raised {type(exc).__name__}: {str(exc)[:200]}, not "
                            f"{'json.JSONDecodeError' if want == 'decode error' else 'a str returned'}")
            continue
        if want == "str" and isinstance(got, str):
            info.append(f"{label}: returned a str of {len(got)} characters")
        else:
            problems.append(f"{label}: returned {type(got).__name__}, expected "
                            f"{'json.JSONDecodeError raised' if want == 'decode error' else 'a str'}")
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + info))


class _WebsiteAppResolver(ast.NodeVisitor):
    """Resolve the app names a task class hands to the website controller.

    Three outcomes per task, and the distinction is the point of this class:

      * ``apps``        — names resolved from website-controller calls; these get probed.
      * no calls at all — the task uses no mocked site; nothing to probe (a genuine pass).
      * ``unresolved``  — a call was found whose argument could not be resolved to a literal.

    The third state must never collapse into the second. A resolver that returns an empty set
    when it does not understand a call makes the reachability check pass VACUOUSLY for that
    task, which is precisely the defect removed from the task-date check (a gate that cannot
    fail). So every unresolvable argument is recorded and surfaced as a FAIL.

    Scope is deliberately narrow: module-level string constants and simple local assignments
    (057 assigns a module constant and passes it; 072 passes a local). The scope is NOT a claim
    that the subset contains no other form — anything else lands in ``unresolved`` and FAILs
    loudly, which is what makes narrowing it safe as the subset changes.
    """

    def __init__(self) -> None:
        self.module_constants: dict[str, str] = {}
        self.local_constants: dict[str, str] = {}
        self.imported_names: set[str] = set()
        self.apps: set[str] = set()
        self.unresolved: list[str] = []
        self.skipped: list[str] = []
        self.calls_seen = 0
        self.names_from_website_calls: set[str] = set()
        self.scope: tuple[str, list[str]] = ("", [])
        # (helper function name, parameter name, where) for one level of substitution
        self.pending_parameters: list[tuple[str, str, str]] = []
        self.tree: Optional[ast.AST] = None

    # -- symbol tables -------------------------------------------------------------------
    def _names_assigned_from_website_calls(self, node: ast.AST) -> set[str]:
        """Names assigned from an expression containing a website call, WITHIN one scope.

        Scoped per function on purpose: keyed module-wide on bare names, a common name such as
        ``url`` assigned from a website call in one function would silence unresolved-flagging
        of that same name everywhere else in the file — a false-negative route straight through
        the three-state contract.
        """
        found: set[str] = set()
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(inner, ast.Call) and self._called_name(inner.func) in WEBSITE_CONTROLLER_FUNCTIONS
                       for inner in ast.walk(stmt.value)):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
        return found

    @staticmethod
    def _direct_body(node: ast.AST) -> list:
        """Statements in this scope's own body, not those of nested definitions."""
        return [s for s in getattr(node, "body", [])
                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    def _names_rebound_from_non_website_expressions(self, node: ast.AST) -> set[str]:
        rebound: set[str] = set()
        for stmt in self._direct_body(node):
            for inner in ast.walk(stmt):
                if not isinstance(inner, ast.Assign):
                    continue
                if any(isinstance(call, ast.Call) and self._called_name(call.func) in WEBSITE_CONTROLLER_FUNCTIONS
                       for call in ast.walk(inner.value)):
                    continue  # reassigned FROM a website call: still legitimately suppressed
                for target in inner.targets:
                    if isinstance(target, ast.Name):
                        rebound.add(target.id)
        return rebound

    def visit_Module(self, node: ast.Module) -> None:
        # module scope only: statements not inside a function body
        module_level = ast.Module(body=[s for s in node.body
                                        if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
                                  type_ignores=[])
        self.names_from_website_calls = self._names_assigned_from_website_calls(module_level)
        for stmt in node.body:  # module-level NAME = "literal" (the 057 form)
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.module_constants[target.id] = stmt.value.value
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.endswith("controllers.website"):
            for alias in node.names:
                if alias.name in WEBSITE_CONTROLLER_FUNCTIONS:
                    self.imported_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def _visit_scope(self, node) -> None:
        outer_constants, outer_scope = self.local_constants, self.scope
        outer_suppressed = self.names_from_website_calls
        self.local_constants = dict(outer_constants)
        # Per-function, never module-wide on bare names.
        # Nested-rebind leak (same class, one scope deeper): a name inherited as suppressed must
        # LOSE that suppression as soon as this scope reassigns it from anything that is not a
        # website call, or an inner `url = something_else` would keep riding the outer scope's
        # exemption. Rebound names are subtracted before the inheritance is applied.
        rebound = self._names_rebound_from_non_website_expressions(node)
        self.names_from_website_calls = ((outer_suppressed - rebound)
                                         | self._names_assigned_from_website_calls(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            self.scope = (node.name, params)
        for stmt in ast.walk(node):  # local NAME = "literal" (the 072 form)
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.local_constants[target.id] = stmt.value.value
        self.generic_visit(node)
        self.local_constants, self.scope = outer_constants, outer_scope
        self.names_from_website_calls = outer_suppressed

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    # -- calls ---------------------------------------------------------------------------
    def _called_name(self, func: ast.expr) -> Optional[str]:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _resolve(self, node: ast.expr, where: str) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.apps.add(node.value)
            return
        if isinstance(node, ast.Constant) and node.value is None:
            # An explicit app=None is a deliberate absence, not a failure to resolve: the call
            # takes its URLs from `state` instead (task 043). Contributes no host — but the
            # exemption ANNOUNCES itself, because whether the state-derived URLs are covered by
            # the task's other calls is a judgment a reader must make from the source. An
            # unverifiable exemption that declares itself is acceptable; a silent one is not.
            self.skipped.append(f"{where}: app=None, a deliberate absence — this call contributes "
                                f"no host, and any URL it derives from `state` is NOT verified here")
            return
        if isinstance(node, ast.Name):
            value = self.local_constants.get(node.id, self.module_constants.get(node.id))
            if value is not None:
                self.apps.add(value)
                return
            if node.id in self.names_from_website_calls:
                # e.g. teamchat_url = f"{build_website_url(TEAMCHAT_APP)}/channel/..." (task 072).
                # The inner call is visited on its own, so the host is already accounted for and
                # re-flagging the outer reference would be a false positive.
                return
            helper, params = self.scope
            if helper and node.id in params:
                # ONE level of substitution: the app is a parameter of a local helper, so it is
                # resolvable from that helper's own call sites (task 072's _resolve_service_base).
                # Deferred to _resolve_pending_parameters, which fails closed.
                self.pending_parameters.append((helper, node.id, where))
                return
            self.unresolved.append(f"{where}: name {node.id!r} is not a resolvable string constant")
            return
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                self._resolve(element, where)
            return
        self.unresolved.append(f"{where}: {type(node).__name__} argument cannot be resolved to a literal")

    def visit_Call(self, node: ast.Call) -> None:
        name = self._called_name(node.func)
        if name in WEBSITE_CONTROLLER_FUNCTIONS:
            self.calls_seen += 1
            where = f"{name}() line {node.lineno}"
            targets = list(node.args[:1])
            targets += [kw.value for kw in node.keywords if kw.arg in ("app", "apps", "path", "target")]
            if not targets:
                self.unresolved.append(f"{where}: no app argument found")
            for target in targets:
                self._resolve(target, where)
        self.generic_visit(node)


    def resolve_pending_parameters(self, tree: ast.AST) -> None:
        """One level of helper substitution, and it FAILS CLOSED in three ways.

        When a website call's app is a parameter of a local helper, the value is resolvable
        from that helper's own call sites. Three constraints keep this from degrading into a
        guess, and each of them yields UNRESOLVED rather than a value:

          * more than one call site -> the UNION of all of them, never the first and never one
            of them; if any single site fails to resolve, the whole substitution is unresolved;
          * a call site whose argument is itself a parameter -> that is the second level, and
            this pass stops there rather than recursing;
          * no call sites at all, or any other unresolvable argument -> unresolved.
        """
        for helper, param, where in self.pending_parameters:
            sites = [c for c in ast.walk(tree)
                     if isinstance(c, ast.Call) and self._called_name(c.func) == helper]
            if not sites:
                self.unresolved.append(f"{where}: parameter {param!r} of {helper}() has no call site to resolve it from")
                continue
            values: set[str] = set()
            failed: list[str] = []
            for site in sites:
                argument = self._argument_for_parameter(site, helper, param, tree)
                if argument is None:
                    failed.append(f"call at line {site.lineno} passes no value for {param!r}")
                    continue
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    values.add(argument.value)
                elif isinstance(argument, ast.Name) and argument.id in self.module_constants:
                    values.add(self.module_constants[argument.id])
                elif isinstance(argument, ast.Name) and self._is_a_parameter_somewhere(argument.id, tree):
                    failed.append(f"call at line {site.lineno} passes parameter {argument.id!r} (second level; not followed)")
                else:
                    failed.append(f"call at line {site.lineno} passes an unresolvable {type(argument).__name__}")
            if failed:
                self.unresolved.append(
                    f"{where}: parameter {param!r} of {helper}() not resolvable from all "
                    f"{len(sites)} call site(s) — " + "; ".join(failed))
            else:
                self.apps.update(values)

    def _argument_for_parameter(self, site: ast.Call, helper: str, param: str, tree: ast.AST) -> Optional[ast.expr]:
        for keyword in site.keywords:
            if keyword.arg == param:
                return keyword.value
        for definition in ast.walk(tree):
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)) and definition.name == helper:
                names = [a.arg for a in definition.args.args]
                if param in names:
                    index = names.index(param)
                    if index < len(site.args):
                        return site.args[index]
                return None
        return None

    @staticmethod
    def _is_a_parameter_somewhere(name: str, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if name in [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]:
                    return True
        return False


def resolve_website_apps(source: str) -> tuple[list[str], list[str], bool, list[str]]:
    """(apps, unresolved reasons, imports-the-controller, announced skips).

    The announced skips are RETURNED rather than left in module state: a side effect invisible
    in the signature detaches the skip data from the result it describes.

    A syntax error is itself unresolvable, never an empty result.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], [f"source does not parse: {exc}"], False, []
    resolver = _WebsiteAppResolver()
    resolver.visit(tree)
    resolver.resolve_pending_parameters(tree)
    imports_controller = bool(resolver.imported_names)
    unresolved = list(resolver.unresolved)
    # Guard against a SILENT miss: the module imports the website controller but no call was
    # recognised (an alias, an indirection, a construct this pass does not model). Without this
    # the task would look like "uses no website" and pass vacuously.
    if imports_controller and resolver.calls_seen == 0:
        unresolved.append("imports the website controller but no resolvable call was found")
    return sorted(resolver.apps), unresolved, imports_controller, list(resolver.skipped)


def collect_website_hosts(root: Path, task_ids: list[str], suffix: str) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Mocked-website hosts per task, resolved from the task class sources with an AST pass.

    Returns ({host: [task ids]}, {task id: [apps]}, {task id: [unresolved reasons]},
    {task id: [announced skips]}). The third element is what keeps this check able to fail: a task
    whose website call cannot be resolved is reported, never silently treated as having no
    website. The fourth carries exemptions that declare themselves.
    """
    hosts: dict[str, list[str]] = {}
    per_task: dict[str, list[str]] = {}
    unresolved: dict[str, list[str]] = {}
    skipped: dict[str, list[str]] = {}
    for task_id in task_ids:
        source_path = root / "evaluation_examples" / "task_class" / f"task_{task_id}.py"
        if not source_path.exists():
            unresolved[task_id] = ["task class source not found"]
            continue
        source = source_path.read_text(encoding="utf-8", errors="replace")
        apps, reasons, _imports, announced = resolve_website_apps(source)
        per_task[task_id] = apps
        if announced:
            skipped[task_id] = announced
        if reasons:
            unresolved[task_id] = reasons
        for app in apps:
            hosts.setdefault(f"{app}.{suffix}", []).append(task_id)
    return {h: sorted(t) for h, t in sorted(hosts.items())}, per_task, unresolved, skipped


def check_website_reachability(ctx: HostContext, task_ids: list[str]) -> CheckResult:
    suffix = ctx.env.get("WEBSITE_HOST_SUFFIX", "")
    if not suffix:
        return CheckResult("", "FAIL", "WEBSITE_HOST_SUFFIX is not set")
    hosts, per_task, unresolved, skipped = collect_website_hosts(ctx.root, task_ids, suffix)
    meta_path = ctx.root / TASK_CLASS_META
    if not meta_path.exists():
        # A check that cannot run has NOT passed. This cross-check is the sole backstop for every
        # construct the resolver does not model, so letting its absence pass silently would
        # reintroduce the exact defect this resolver was written to remove.
        return CheckResult("", "FAIL", f"task class metadata missing: {meta_path} — the "
                                       f"uses_website cross-check cannot run, and a check that "
                                       f"cannot run has not passed "
                                       f"(copy data/osworld_v2/{TASK_CLASS_META.split('/')[-1]} next to the manifests)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    unusable = [t for t in task_ids if not meta.get(t, {}).get("id")]
    website_tasks = [t for t in task_ids if meta.get(t, {}).get("uses_website")]
    problems, info = [], []
    if unusable:
        # Same defect as the asset check: an unresolved class yields uses_website=False
        # for a task that may well use one, so the cross-check silently covers nothing.
        problems.append(f"no usable metadata row (class did not resolve): {unusable}")
    # State 3: a website-controller call whose argument could not be resolved. This must FAIL,
    # never pass as "no website" — an extractor that returns empty when it does not understand
    # makes this check vacuous for that task.
    if unresolved:
        for task_id, reasons in sorted(unresolved.items()):
            problems.append(f"{task_id}: website app name could not be resolved — " + "; ".join(reasons))
    # Independent cross-check on the other direction: metadata says the task uses a website but
    # the resolver found none. Catches a silent miss the resolver itself did not notice.
    without_hosts = [t for t in website_tasks if not per_task.get(t) and t not in unresolved]
    if without_hosts:
        problems.append(f"metadata says these are website tasks but the resolver found no app: {without_hosts}")
    for host, tasks in hosts.items():
        try:
            addresses = ctx.resolve_host(host)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{host}: DNS failed ({type(exc).__name__}: {exc}); tasks {tasks}")
            continue
        status, note = None, ""
        scheme_used = None
        for scheme in ("https", "http"):  # the website controller picks https when it answers, else http
            status, note = ctx.http_probe(f"{scheme}://{host}")
            if status is not None:
                scheme_used = scheme
                break
        if status is None:
            problems.append(f"{host}: DNS {addresses} but no HTTP(S) response ({note}); tasks {tasks}")
        else:
            info.append(f"{host} -> {addresses} {scheme_used} {status} ({note}); tasks {tasks}")
    unexpected_skips = sorted(set(skipped) - set(EXPECTED_WEBSITE_SKIP_TASKS))
    for task_id, notes in sorted(skipped.items()):
        marker = "ANNOUNCED SKIP" if task_id in EXPECTED_WEBSITE_SKIP_TASKS else "UNEXPECTED SKIP"
        for note in notes:
            info.append(f"{task_id}: {marker} — {note}")
    no_website = sorted(t for t in task_ids if not per_task.get(t) and t not in unresolved)
    info.insert(0, f"{len(hosts)} mocked-website host(s) for {len([t for t in per_task if per_task[t]])} task(s) "
                   f"(metadata says {len(website_tasks)} website tasks: {website_tasks}); "
                   f"{len(no_website)} task(s) use no mocked site; {len(unresolved)} unresolved")
    # THE OTHER DIRECTION, which the date check above already has and this one lacked: a pinned task
    # that STOPS announcing a skip leaves the exemption standing over nothing, and the next real skip
    # on that id would then print as expected. An exemption that cannot notice its subject has left
    # is worth less than no exemption.
    stale_pins = sorted(t for t in EXPECTED_WEBSITE_SKIP_TASKS if t in task_ids and t not in skipped)
    if problems:
        return CheckResult("", "FAIL", "\n".join(problems + info))
    notes = []
    if unexpected_skips:
        notes.append(f"website skips announced by tasks outside the pinned set {list(EXPECTED_WEBSITE_SKIP_TASKS)}: "
                     f"{unexpected_skips} — each is a call whose URLs this check does NOT verify")
    if stale_pins:
        notes.append(f"pinned website-skip task(s) {stale_pins} are installed but announced NO skip: the exemption in "
                     f"EXPECTED_WEBSITE_SKIP_TASKS is stale and now silently excuses a future real skip on that id. "
                     f"Re-check the class and shrink the pin.")
    if notes:
        return CheckResult("", "WARN", "\n".join(notes + info))
    return CheckResult("", "PASS", "\n".join(info))


def _task_field(task: Any, name: str) -> Any:
    value = None
    if hasattr(task, "get") and callable(getattr(task, "get")):
        value = task.get(name)
    if value is None:
        value = getattr(task, name, None)
    return value


def check_task_classes(ctx: HostContext, task_ids: list[str], prereg: dict[str, Any]) -> tuple[CheckResult, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    problems, warnings = [], []
    for task_id in task_ids:
        try:
            task = ctx.load_task(task_id)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{task_id}: {type(exc).__name__}: {exc}")
            continue
        get_phases = getattr(task, "get_phases", None)
        phases = 0
        if callable(get_phases):
            try:
                phases = len(get_phases() or [])
            except NotImplementedError:
                phases = 0
        rows[task_id] = {
            "instance_type": _task_field(task, "instance_type"),
            "volume_size": _task_field(task, "volume_size"),
            "image": _task_field(task, "image"),
            "intermediate_eval_safe": bool(_task_field(task, "intermediate_eval_safe") if _task_field(task, "intermediate_eval_safe") is not None else True),
            "user_simulator": bool(_task_field(task, "user_simulator")),
            "task_current_date": _task_field(task, "task_current_date"),
            "multi_phase": phases,
            "proxy": bool(_task_field(task, "proxy")),
            "platform": _task_field(task, "platform"),
        }
    # Expectations: prereg tables + the two subset facts pinned above (multi-phase, task_current_date).
    expected_hitl = {str(i) for i in prereg.get("hitl", [])}
    actual_hitl = {t for t, r in rows.items() if r["user_simulator"]}
    if actual_hitl != expected_hitl:
        problems.append(f"user_simulator tasks {sorted(actual_hitl)} != prereg hitl {sorted(expected_hitl)}")
    actual_multi = {t for t, r in rows.items() if r["multi_phase"]}
    if actual_multi != set(MULTI_PHASE_TASKS):
        problems.append(f"multi-phase tasks {sorted(actual_multi)} != expected {list(MULTI_PHASE_TASKS)}")
    for task_id in TASK_CURRENT_DATE_TASKS:
        if task_id in rows and not rows[task_id]["task_current_date"]:
            problems.append(f"{task_id} has no task_current_date (the driver hands it to the agent; the class should set it)")
    dated = {t: r["task_current_date"] for t, r in rows.items() if r["task_current_date"]}
    unexpected_dated = sorted(set(dated) - set(TASK_CURRENT_DATE_TASKS))
    if unexpected_dated:
        problems.append(f"tasks declare task_current_date but are not in the expected list: {unexpected_dated}")
    proxied = [t for t, r in rows.items() if r["proxy"]]
    if proxied:
        problems.append(f"proxy-flagged tasks in the subset (proxy is OFF on this subset): {proxied}")
    expected_overrides = {str(k): v for k, v in prereg.get("instance_type_overrides", {}).items()}
    for task_id, expected in expected_overrides.items():
        if task_id in rows and rows[task_id]["instance_type"] != expected:
            problems.append(f"{task_id}: instance_type {rows[task_id]['instance_type']!r} != prereg {expected!r}")
    extra_overrides = {t: r["instance_type"] for t, r in rows.items() if r["instance_type"] and t not in expected_overrides}
    if extra_overrides:
        warnings.append(f"instance_type set by the class but absent from the prereg table (harness default is {DEFAULT_INSTANCE_TYPE}): {extra_overrides}")
    table_problems, table_warnings = compare_prereg_class_tables(prereg, rows)
    problems.extend(table_problems)
    warnings.extend(table_warnings)
    lines = [f"{'task':>4} {'instance_type':>13} {'vol':>4} {'image':>22} {'ckpt_safe':>9} {'usersim':>7} {'date':>10} {'phases':>6} {'proxy':>5} platform"]
    for task_id, row in rows.items():
        lines.append(f"{task_id:>4} {str(row['instance_type'] or '-'):>13} {str(row['volume_size'] or '-'):>4} "
                     f"{str(row['image'] or '-'):>22} {str(row['intermediate_eval_safe']):>9} {str(row['user_simulator']):>7} "
                     f"{str(row['task_current_date'] or '-'):>10} {row['multi_phase']:>6} {str(row['proxy']):>5} {row['platform']}")
    overrides = ", ".join(f"{t}: {r['instance_type']}" for t, r in rows.items() if r["instance_type"])
    summary = (f"{len(rows)}/{len(task_ids)} loaded; checkpoint-unsafe {[t for t, r in rows.items() if not r['intermediate_eval_safe']]}; "
               f"user-simulator {sorted(actual_hitl)}; multi-phase {sorted(actual_multi)}; task_current_date {dated}; "
               f"instance overrides {overrides}")
    return CheckResult("", _status(problems, warnings), "\n".join([summary] + problems + warnings + lines)), rows


def check_assets_and_release(ctx: HostContext) -> CheckResult:
    problems, info = [], []
    assets = ctx.env.get("OSWORLD_FILE_BASE_URL", "")
    assets_path = Path(assets)
    if not assets or not assets_path.is_absolute():
        problems.append(f"OSWORLD_FILE_BASE_URL must be an absolute path (dotenv does not expand $(pwd)); got {assets!r}")
    elif not assets_path.is_dir():
        problems.append(f"assets dir does not exist: {assets}")
    else:
        entries = sum(1 for _ in assets_path.iterdir())
        info.append(f"assets dir {assets}: {entries} entries")
        if entries == 0:
            problems.append("assets dir is empty")
    manifest_path = ctx.root / RELEASE_MANIFEST
    if not manifest_path.exists():
        problems.append(f"release manifest missing: {manifest_path}")
    else:
        release = json.loads(manifest_path.read_text(encoding="utf-8"))
        info.append(f"release {release.get('release')} code tag {release.get('osworld_code', {}).get('tag')} "
                    f"tasks {release.get('tasks', {}).get('tag')} assets {release.get('assets', {}).get('tag')}")
        if release.get("osworld_code", {}).get("tag") != PINNED_TAG:
            problems.append(f"release manifest code tag {release.get('osworld_code', {}).get('tag')} != {PINNED_TAG}")
        task_count = release.get("task_hash_manifest", {}).get("task_count")
        present = sorted(p.name for p in (ctx.root / "evaluation_examples" / "task_class").glob("task_*.py"))
        info.append(f"task classes on disk: {len(present)} (release task_count {task_count})")
        if task_count != len(present):
            problems.append(f"task classes on disk ({len(present)}) != release task_count ({task_count})")
        manifest_ami = (release.get("provider_images", {}).get("aws", {}).get("ubuntu", {})
                        .get("us-east-1", {}).get("1920x1080", {}).get("ami_id"))
        try:
            default_ami = ctx.default_ami()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"get_default_ami_id raised: {exc}")
            default_ami = None
        info.append(f"AMI: release manifest {manifest_ami}, get_default_ami_id {default_ami}")
        if manifest_ami and default_ami and manifest_ami != default_ami:
            problems.append("release manifest AMI != harness default AMI")
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + info))


def check_task_assets(ctx: HostContext, task_ids: list[str]) -> CheckResult:
    meta_path = ctx.root / TASK_CLASS_META
    if not meta_path.exists():
        return CheckResult("", "FAIL", f"task class metadata missing: {meta_path} (copy data/osworld_v2/task_class_static_extract_v2026.08.08.json next to the manifests)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base = Path(ctx.env.get("OSWORLD_FILE_BASE_URL", ""))
    missing, listed, unknown = [], 0, []
    for task_id in task_ids:
        entry = meta.get(task_id)
        # A row whose class did not resolve carries no id and an EMPTY asset list. Reading that
        # as "this task needs no assets" makes the check unfalsifiable for exactly the tasks
        # whose metadata is broken, so it is an unusable row, not an empty one.
        if entry is None or not entry.get("id"):
            unknown.append(task_id)
            continue
        for rel in entry.get("assets", []):
            listed += 1
            if not (base / rel).exists():
                missing.append(rel)
    problems = []
    if unknown:
        problems.append(f"tasks with no usable metadata row (absent, or the class did not "
                        f"resolve so the row carries no id and an empty asset list): {unknown}")
    if missing:
        problems.append(f"{len(missing)} asset file(s) missing under {base}: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + [f"{listed} asset files listed for {len(task_ids)} tasks; all present under {base}" if not missing else f"{listed - len(missing)}/{listed} present"]))


def check_env_keys(ctx: HostContext) -> CheckResult:
    missing = [k for k in REQUIRED_ENV_KEYS if not ctx.env.get(k)]
    problems = [f"missing/empty in .env: {missing}"] if missing else []
    e = ctx.env
    if e.get("OSWORLD_EVAL_MODEL_NAME") != EVALUATOR_MODEL or e.get("OSWORLD_USER_SIM_MODEL") != EVALUATOR_MODEL:
        problems.append(f"evaluator/user-sim model must be {EVALUATOR_MODEL} (both arms, same evaluator; the harness silently defaults to gpt-4o): eval={e.get('OSWORLD_EVAL_MODEL_NAME')!r} sim={e.get('OSWORLD_USER_SIM_MODEL')!r}")
    if e.get("OSWORLD_EVAL_MODEL_PROVIDER") != "anthropic" or e.get("OSWORLD_USER_SIM_PROVIDER") != "anthropic":
        problems.append("evaluator/user-sim provider must be anthropic (the harness silently defaults to openai)")
    if e.get("OSWORLD_EVAL_MODEL_API_KEY_ENV") != "ANTHROPIC_API_KEY" or e.get("OSWORLD_USER_SIM_API_KEY_ENV") != "ANTHROPIC_API_KEY":
        problems.append("evaluator/user-sim key env must be ANTHROPIC_API_KEY")
    if CELL["model_key"] != "MINIMAX_API_KEY":
        # The Qwen cell: an OpenRouter key (sk-or-); the DeepSeek cell: a DeepSeek key (sk-), which the control's
        # M3 agent sends as x-api-key only with that prefix. Never the evaluator's.
        model_key = e.get(CELL["model_key"], "")
        if model_key and model_key == e.get("ANTHROPIC_API_KEY"):
            problems.append(f"{CELL['model_key']} equals ANTHROPIC_API_KEY: the model key was re-exported under the evaluator's name")
        if model_key and not model_key.startswith(CELL["key_prefix"]):
            problems.append(f"{CELL['model_key']} does not start with {CELL['key_prefix']!r}: it is not a {CELL['key_label']} key")
    # The variables this cell's launch must not carry (DEEPSEEK_FORBIDDEN_ENV): in the host's .env, which the
    # runner loads before parsing its flags, or in the shell this preflight runs in, which the launch inherits.
    for variable in CELL["forbidden_env"]:
        sources = [where for where, values in (("the host .env", e), ("the shell environment", os.environ))
                   if values.get(variable) not in (None, "")]
        if sources:
            problems.append(f"{variable} is set in {' and '.join(sources)}: upstream's M3 agent and runner read it, so "
                            f"the control's thinking regime would silently differ from the launcher's explicit "
                            f"--thinking_budget (and no --thinking_mode); unset it before launching")
    minimax_key = e.get("MINIMAX_API_KEY", "") if CELL["model_key"] == "MINIMAX_API_KEY" else ""
    if minimax_key and minimax_key == e.get("ANTHROPIC_API_KEY"):
        problems.append("MINIMAX_API_KEY equals ANTHROPIC_API_KEY: the MiniMax key was re-exported under the evaluator's name")
    if minimax_key and not minimax_key.startswith("sk-"):
        problems.append("MINIMAX_API_KEY does not start with 'sk-': upstream's M3Agent would send it as a Bearer token instead of the x-api-key header the /anthropic route takes")
    # CHECKED, not merely present: the failure this variable prevents is silent, and a value that
    # is set-but-too-short fails exactly the same way as one that is absent.
    ttl_raw = e.get("DEFAULT_TTL_MINUTES", "")
    if ttl_raw:
        try:
            ttl = int(str(ttl_raw).strip())
        except ValueError:
            problems.append(f"DEFAULT_TTL_MINUTES={ttl_raw!r} is not an integer number of minutes; the provider would "
                            f"fall back to its short default and kill client VMs mid-episode")
        else:
            if ttl < MIN_CLIENT_VM_TTL_MINUTES:
                problems.append(f"DEFAULT_TTL_MINUTES={ttl} is below the pinned floor of {MIN_CLIENT_VM_TTL_MINUTES} "
                                f"(.claude/docs/benchmarks/osworld_v2.md): a full-length episode does not fit inside it, and the VMs it "
                                f"kills are the ones still at the step cap — an arm's worst episodes, dropped from the "
                                f"paired comparison with no result.txt and no error row")
    for k in ("GITLAB_URL", "GITLAB_PRIVATE_TOKEN"):
        if e.get(k):
            problems.append(f"{k} is set but GitLab is not run on this subset")
    present = [k for k in REQUIRED_ENV_KEYS if ctx.env.get(k)]
    if CELL["model_key"] != "MINIMAX_API_KEY":
        shape_ok = e.get(CELL["model_key"], "").startswith(CELL["key_prefix"])
        absent = f"; {', '.join(CELL['forbidden_env'])} unset" if CELL["forbidden_env"] else ""
        return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + [f"present: {present}", f"{CELL['model_key']} prefix ok ({CELL['key_prefix']}), distinct from ANTHROPIC_API_KEY{absent}" if shape_ok else ""]).rstrip())
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + [f"present: {present}", f"MINIMAX_API_KEY prefix ok (sk-), distinct from ANTHROPIC_API_KEY" if minimax_key.startswith("sk-") else ""]).rstrip())


def check_sts(ctx: HostContext) -> CheckResult:
    identity = ctx.boto3_client("sts").get_caller_identity()
    return CheckResult("", "PASS", f"account {identity.get('Account')} arn {identity.get('Arn')}")


def check_ami_available(ctx: HostContext) -> CheckResult:
    ami = ctx.default_ami()
    images = ctx.boto3_client("ec2").describe_images(ImageIds=[ami]).get("Images", [])
    if not images:
        return CheckResult("", "FAIL", f"{ami} not visible to this account in {ctx.env.get('AWS_REGION')}")
    image = images[0]
    state = image.get("State")
    detail = f"{ami} state={state} name={image.get('Name')!r} created={image.get('CreationDate')}"
    return CheckResult("", "PASS" if state == "available" else "FAIL", detail)


def check_network(ctx: HostContext) -> CheckResult:
    ec2 = ctx.boto3_client("ec2")
    subnet_id, sg_id = ctx.env.get("AWS_SUBNET_ID"), ctx.env.get("AWS_SECURITY_GROUP_ID")
    problems, info = [], []
    subnet = ec2.describe_subnets(SubnetIds=[subnet_id])["Subnets"][0]
    info.append(f"subnet {subnet_id}: vpc {subnet.get('VpcId')} az {subnet.get('AvailabilityZone')} cidr {subnet.get('CidrBlock')} free ips {subnet.get('AvailableIpAddressCount')}")
    group = ec2.describe_security_groups(GroupIds=[sg_id])["SecurityGroups"][0]
    if group.get("VpcId") != subnet.get("VpcId"):
        problems.append(f"security group vpc {group.get('VpcId')} != subnet vpc {subnet.get('VpcId')}")
    covered: dict[int, list[str]] = {p: [] for p in REQUIRED_INBOUND_PORTS}
    for rule in group.get("IpPermissions", []):
        proto = rule.get("IpProtocol")
        low, high = rule.get("FromPort"), rule.get("ToPort")
        sources = [r.get("CidrIp") for r in rule.get("IpRanges", [])] + [g.get("GroupId") for g in rule.get("UserIdGroupPairs", [])]
        for port in REQUIRED_INBOUND_PORTS:
            if proto == "-1" or (proto == "tcp" and low is not None and low <= port <= high):
                covered[port].extend(sources)
    missing = [p for p, s in covered.items() if not s]
    if missing:
        problems.append(f"inbound TCP ports with no rule: {missing} (V2 public-evaluation guideline needs {list(REQUIRED_INBOUND_PORTS)})")
    info.append(f"security group {sg_id} ({group.get('GroupName')}): " + ", ".join(f"{p}<-{sorted(set(s))}" for p, s in covered.items()))
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + info))


def check_orphan_vms(ctx: HostContext) -> CheckResult:
    ec2 = ctx.boto3_client("ec2")
    amis = {ami for os_map in ctx.ami_maps().values() for region_map in os_map.values() for ami in region_map.values()}
    pages = ec2.describe_instances(Filters=[{"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
    orphans, others = [], []
    for reservation in pages.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            line = f"{inst.get('InstanceId')} {inst.get('InstanceType')} {inst.get('State', {}).get('Name')} image {inst.get('ImageId')} launched {inst.get('LaunchTime')}"
            (orphans if inst.get("ImageId") in amis else others).append(line)
    info = [f"non-terminated instances not from an OSWorld AMI (the host itself, etc.): {len(others)}"] + [f"  {l}" for l in others]
    if orphans:
        return CheckResult("", "FAIL", "\n".join([f"orphan task VMs from an OSWorld AMI: {len(orphans)} (terminate before launching)"] + [f"  {l}" for l in orphans] + info))
    return CheckResult("", "PASS", "\n".join(["no non-terminated instance from an OSWorld AMI"] + info))


def check_vcpu_quota(ctx: HostContext, task_rows: dict[str, dict[str, Any]]) -> CheckResult:
    quotas = ctx.boto3_client("service-quotas")
    quota = float(quotas.get_service_quota(ServiceCode="ec2", QuotaCode=VCPU_QUOTA_CODE)["Quota"]["Value"])
    ec2 = ctx.boto3_client("ec2")
    types_in_play = sorted({(r["instance_type"] or DEFAULT_INSTANCE_TYPE) for r in task_rows.values()} | {DEFAULT_INSTANCE_TYPE})
    vcpus = {t["InstanceType"]: int(t["VCpuInfo"]["DefaultVCpus"])
             for t in ec2.describe_instance_types(InstanceTypes=types_in_play)["InstanceTypes"]}
    per_task = sorted((vcpus[r["instance_type"] or DEFAULT_INSTANCE_TYPE] for r in task_rows.values()), reverse=True)
    running = ec2.describe_instances(Filters=[{"Name": "instance-state-name", "Values": ["pending", "running"]}])
    in_use = sum(int(i.get("CpuOptions", {}).get("CoreCount", 0)) * int(i.get("CpuOptions", {}).get("ThreadsPerCore", 1))
                 for res in running.get("Reservations", []) for i in res.get("Instances", []))
    headroom = quota - in_use

    def worst(n: int) -> int:
        return sum(per_task[:n]) if per_task else n * vcpus[DEFAULT_INSTANCE_TYPE]
    plans = {"graphpc alone": worst(ctx.num_envs_graphpc), "baseline alone": worst(ctx.num_envs_baseline),
             "both arms concurrently": worst(ctx.num_envs_graphpc) + worst(ctx.num_envs_baseline)}
    problems = [f"{name}: worst-case {need} vCPU > headroom {headroom:.0f}" for name, need in plans.items()
                if name != "both arms concurrently" and need > headroom]
    warnings = [] if plans["both arms concurrently"] <= headroom else ["both arms concurrently would EXCEED the vCPU headroom; run them sequentially"]
    info = [f"quota {VCPU_QUOTA_CODE} = {quota:.0f} vCPU; in use now {in_use} (incl. this host); headroom {headroom:.0f}",
            f"vCPU per type {vcpus}; per-task requirement sorted desc {per_task}",
            f"plan: num_envs graphpc={ctx.num_envs_graphpc} baseline={ctx.num_envs_baseline}; worst-case vCPU {plans}"]
    return CheckResult("", _status(problems, warnings), "\n".join(problems + warnings + info))


def check_cold_start(ctx: HostContext, argvs: dict[str, list[str]]) -> CheckResult:
    """A scored launch is cold by definition: an empty support set and no prior result.txt in either arm's
    result tree (or the operator acknowledges a resume FOR THAT ARM: a GraphPC arm launched against a
    finished control acknowledges the control's tree alone, so its own tree must still be cold — a
    prior scored tree left at its path is a FAIL, not a WARN). Also lists the baseline's unfinished task
    dirs still holding api_logs/ (the baseline launch script removes them; the baseline
    runner keeps upstream's get_unfinished, which would crash on them)."""
    problems, warnings, info = [], [], []
    pairs = {arm: parse_runner_argv(argv)[1] for arm, argv in argvs.items()}
    store_root = Path(pairs.get("graphpc", {}).get("--support_set_root", CELL["graphpc_store"]))
    store = (ctx.root / store_root / "osworld") if not store_root.is_absolute() else store_root / "osworld"
    entries = [p for p in store.rglob("*") if p.is_file()] if store.exists() else []
    if entries:
        (warnings if "graphpc" in ctx.acknowledge_resume else problems).append(f"support set is NOT cold: {len(entries)} file(s) under {store} (warm store at launch changes the arm; --acknowledge_resume graphpc to accept)")
    else:
        info.append(f"support set cold: {store} {'absent' if not store.exists() else 'empty'}")
    for arm in ("graphpc", "baseline"):
        result_dir = Path(pairs.get(arm, {}).get("--result_dir", ""))
        result_dir = (ctx.root / result_dir) if not result_dir.is_absolute() else result_dir
        finished = list(result_dir.rglob("result.txt")) if result_dir.exists() else []
        if finished:
            (warnings if arm in ctx.acknowledge_resume else problems).append(
                f"{arm}: {len(finished)} prior result.txt under {result_dir}; the runner would SKIP those tasks (resume) — --acknowledge_resume {arm} to accept")
        else:
            info.append(f"{arm}: no prior result.txt under {result_dir}")
        if arm == "baseline" and result_dir.exists():
            # <result_dir>/<action_space>/<observation_type>/<model>/<domain>/<task>/
            stale = [d for d in result_dir.glob("*/*/*/*/*") if d.is_dir() and not (d / "result.txt").exists()
                     and ((d / "api_logs").is_dir() or any(p.is_dir() for p in d.glob("model_eval_file_log*")))]
            if stale and CELL["baseline_script"] not in ("run_multienv_m3_baseline.sh", "run_multienv_deepseek_baseline.sh"):
                warnings.append(f"baseline: {len(stale)} unfinished task dir(s) hold api_logs/ or model_eval_file_log*/; the control's launch script moves every selected unfinished task dir to <RESULT_DIR>_quarantine/ before starting: {[str(d.relative_to(result_dir)) for d in stale[:5]]}")
            elif stale:
                warnings.append(f"baseline: {len(stale)} unfinished task dir(s) still hold api_logs/ or model_eval_file_log*/ (the launch script removes them before starting): {[str(d.relative_to(result_dir)) for d in stale[:5]]}")
    return CheckResult("", _status(problems, warnings), "\n".join(problems + warnings + info))


def check_other_runner_active(ctx: HostContext) -> CheckResult:
    """A live GraphPC runner is a FAIL: a scored launch never shares the host with a running GraphPC
    arm, and this preflight's own in-process build is skipped after this FAIL so it cannot compete with
    that runner's workers for memory (silent task loss + leaked EC2 instance). A live baseline runner is
    a WARN: the arms would share the host's vCPUs (4 on a 4 vCPU run host, planned for 4 GraphPC + 4
    baseline workers) and one uplink."""
    procs = ctx.other_runners()
    graphpc = [p for p in procs if "run_multienv_graphpc.py" in p]
    if graphpc:
        return CheckResult("", "FAIL", "\n".join(["a GraphPC runner is LIVE on this host; the in-process build_default check is skipped to protect its workers:"] + [f"  {p}" for p in procs]))
    if procs:
        return CheckResult("", "WARN", "\n".join(["a baseline runner is alive on this host (both arms would share the host's vCPUs, one uplink and the memory bound):"] + [f"  {p}" for p in procs]))
    return CheckResult("", "PASS", f"no run_multienv_graphpc.py / {CELL['baseline_runner']} process alive")



def check_build_default(ctx: HostContext, measured: dict[str, Any]) -> CheckResult:
    """The vendored package constructs the arm's agent: ``build_default`` with retrieval OFF, the
    regime every scored GraphPC launch carries. Its config is
    the arm's regime (``agent_config``, printed for the run README) and, in the shape the telemetry
    writes as ``active_config``, what a cell's method surface is compared against
    (``agent_active_config``; the Qwen and DeepSeek cells', at launch, by check_method_surface)."""
    info, problems = [], []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            agent = ctx.build_agent(True, tmp)
            config = dict(getattr(agent, "config", {}) or {})
            measured["agent_config"] = config
            if CELL["method_surface_key"] != "method_surface":
                measured["agent_active_config"] = {**config,
                                                   "retrieval_off": bool(getattr(agent, "retrieval_off", False)),
                                                   "screen_size": list(getattr(agent, "screen_size", ()) or ()),
                                                   "model": getattr(agent, "_model_name", lambda: None)()}
            if config.get("retrieval_off") is not True or config.get("retrieval_agent") != "BypassRetrievalAgent":
                problems.append(f"build_default(retrieval_off=True) built retrieval_off={config.get('retrieval_off')!r} "
                                f"with {config.get('retrieval_agent')!r}: not the bypass the arm runs")
            info.append(f"retrieval_off: {type(agent).__name__} constructed with {config.get('retrieval_agent')} "
                        f"(rss {_rss_mb():.0f} MB)")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"build_default(retrieval_off=True) raised {type(exc).__name__}: {exc}")
    return CheckResult("", "FAIL" if problems else "PASS", "\n".join(problems + info))


def check_graphpc_step_budget(prereg: dict[str, Any], graphpc: dict[str, Any]) -> CheckResult:
    """The GraphPC arm's step budget, three ways: the launch argv's --max_steps, the adapter's budget for this
    cell's backbone (OSWORLD_MAX_STEPS_BY_BACKBONE: what runner delta 15 holds the launch to and the orchestrator
    places its give-up budget rule on) and the record, the cell's method surface's max_steps. The Qwen GraphPC arm
    runs 800; every other GraphPC arm runs the cell config's 500, as the controls do."""
    budgets, where = _adapter_step_budgets(Path(__file__))
    if budgets is None:
        return CheckResult("", "FAIL", f"the adapter's step budgets cannot be read: {where}")
    adapter = budgets.get(CELL["backbone_type"])
    surface = prereg.get(CELL["method_surface_key"])
    declared = surface.get("max_steps") if isinstance(surface, dict) else None
    source = f"{CELL['method_surface_key']}.max_steps"
    launched = graphpc.get("--max_steps")
    if type(declared) is not int or adapter != declared or launched != str(declared):
        return CheckResult("", "FAIL", f"the GraphPC arm's step budget disagrees: launch argv --max_steps {launched!r}, "
                                       f"adapter OSWORLD_MAX_STEPS_BY_BACKBONE[{CELL['backbone_type']!r}] = {adapter!r} "
                                       f"({where}), {source} = {declared!r}")
    return CheckResult("", "PASS", f"--max_steps {launched} == adapter OSWORLD_MAX_STEPS_BY_BACKBONE"
                                   f"[{CELL['backbone_type']!r}] {adapter} == {source} {declared}")


def check_method_surface_declared(prereg: dict[str, Any]) -> CheckResult:
    key = CELL["method_surface_key"]
    surface = prereg.get(key)
    if not isinstance(surface, dict) or not surface:
        return CheckResult("", "FAIL", f"{key} absent from the pre-registration: this cell's method surface is not "
                                       f"declared, so this preflight has nothing to hold the arm to")
    return CheckResult("", "PASS", f"{key} declares {sorted(surface)}")


def check_method_surface(prereg: dict[str, Any], measured: dict[str, Any]) -> CheckResult:
    """The built GraphPC agent's active_config against this cell's pre-registered method surface: every
    declared key must be recorded and equal. Absent from the pre-registration is a FAIL (nothing to
    hold the arm to), as is a build that did not complete (nothing measured)."""
    declared = check_method_surface_declared(prereg)
    if declared.status != "PASS":
        return declared
    surface = dict(prereg[CELL["method_surface_key"]])
    if "agent_active_config" not in measured:
        return CheckResult("", "FAIL", "the build check did not complete, so the built agent's active_config "
                                       "was not measured and the method surface was compared against nothing")
    config = measured["agent_active_config"]
    missing = sorted(k for k in surface if k not in config)
    wrong = {k: {"built": config[k], "declared": v} for k, v in surface.items() if k in config and config[k] != v}
    problems = ([f"declared but not in the built agent's active_config: {missing}"] if missing else []) + \
               ([f"the built agent differs from the pre-registered surface: {wrong}"] if wrong else [])
    if problems:
        return CheckResult("", "FAIL", "\n".join(problems))
    return CheckResult("", "PASS", f"{len(surface)} declared key(s) equal in the built agent's active_config: {sorted(surface)}")


def check_memory_bound(ctx: HostContext, measured: dict[str, Any]) -> CheckResult:
    """The memory-bound rule: the MEASURED per-worker cost is the failing rule —
    num_envs_graphpc x GRAPHPC_OPERATING_PER_WORKER_MB + 1024 MB must fit in MemAvailable (sampled
    before this preflight's own build); the 2 GB/worker constant stays only as a
    floor. The startup peak is recorded separately and guarded on its own: MemAvailable minus
    num_envs x GRAPHPC_MEASURED_PEAK_PER_WORKER_MB must leave GRAPHPC_PEAK_HEADROOM_MIN_MB. The
    detail says how many GraphPC workers the host fits, which is what to re-run after a resize.
    Both costs were measured with each worker holding the embedder; under retrieval off (every scored
    launch) no worker loads it, so they bound the arm from above."""
    mem = measured.get("mem_available_before_build_mb")
    if mem is None:
        mem = ctx.mem_available_mb()
    problems, warnings, info = [], [], []
    if mem is None:
        return CheckResult("", "WARN", "MemAvailable unreadable (/proc/meminfo); memory bound not checked")
    n = ctx.num_envs_graphpc
    floor = n * GRAPHPC_MEMORY_FLOOR_PER_WORKER_MB
    info.append(f"MemAvailable {mem:.0f} MB (sampled before this preflight's build; now {ctx.mem_available_mb() or 0:.0f} MB); "
                f"GraphPC arm num_envs={n}")
    if mem < floor:
        problems.append(f"floor: MemAvailable {mem:.0f} MB < {n} x {GRAPHPC_MEMORY_FLOOR_PER_WORKER_MB} MB = {floor} MB")
    per_worker = GRAPHPC_OPERATING_PER_WORKER_MB
    required = n * per_worker + GRAPHPC_MEMORY_HEADROOM_MB
    fits = max(0, int((mem - GRAPHPC_MEMORY_HEADROOM_MB) // per_worker))
    info.append(f"measured operating cost per worker {per_worker} MB (host-level, a working worker with the embedder loaded — a lower bound "
                f"of its growth; retrieval off loads no embedder) -> "
                f"{n} x {per_worker} + {GRAPHPC_MEMORY_HEADROOM_MB} = {required} MB required; "
                f"this host fits {fits} GraphPC worker(s) ({fits + 1} would need {(fits + 1) * per_worker + GRAPHPC_MEMORY_HEADROOM_MB} MB)")
    if required > mem:
        problems.append(f"measured: {required} MB required > MemAvailable {mem:.0f} MB for {n} GraphPC worker(s) — "
                        f"set NUM_ENVS={fits} (a worker killed by OOM loses its task silently and leaks its EC2 instance)")
    peak = GRAPHPC_MEASURED_PEAK_PER_WORKER_MB
    peak_headroom = mem - n * peak
    info.append(f"startup peak per worker {peak} MB (largest MemAvailable drop during four staggered loads / 4) -> "
                f"peak headroom {mem:.0f} - {n} x {peak} = {peak_headroom:.0f} MB (minimum {GRAPHPC_PEAK_HEADROOM_MIN_MB})")
    if peak_headroom < GRAPHPC_PEAK_HEADROOM_MIN_MB:
        problems.append(f"startup peak: {n} worker(s) starting would leave {peak_headroom:.0f} MB < "
                        f"{GRAPHPC_PEAK_HEADROOM_MIN_MB} MB of MemAvailable at the peak")
    if n > GRAPHPC_MAX_NUM_ENVS:
        warnings.append(f"num_envs_graphpc={n} > {GRAPHPC_MAX_NUM_ENVS}: the launch script refuses this without FORCE_NUM_ENVS=1")
    return CheckResult("", _status(problems, warnings), "\n".join(problems + warnings + info))


def regime_summary(ctx: HostContext, argvs: dict[str, list[str]], measured: dict[str, Any]) -> str:
    def redacted(argv: list[str]) -> str:
        out = list(argv)
        for i, tok in enumerate(out):
            if tok == "--api_key" and i + 1 < len(out):
                out[i + 1] = f"<{CELL['model_key']}>"
        return " ".join(out)
    try:
        grounding_temperature = ctx.grounding_temperature()
    except Exception as exc:  # noqa: BLE001
        grounding_temperature = f"unavailable ({exc})"
    lines = ["== resolved regime for the run README ==",
             f"GraphPC arm argv: {redacted(argvs.get('graphpc', []))}",
             f"baseline arm argv: {redacted(argvs.get('baseline', []))}",
             f"GraphPC agent: GROUNDING_TEMPERATURE={grounding_temperature}; max_tokens None (no cap sent); "
             # NOT `.get(..., {})`: an empty dict reads as "this arm has no configuration", which is
             # a statement, and a false one. The build check is this key's only writer, so its
             # absence means that build did not complete — say so, because this block is printed on a
             # failed preflight too and is what gets pasted into the README.
             f"config={measured['agent_config'] if 'agent_config' in measured else 'NOT MEASURED (the build check did not complete; this preflight did not establish the arm regime)'}; "
             f"workers={ctx.num_envs_graphpc}",
             f"GraphPC worker footprint rule: {GRAPHPC_OPERATING_PER_WORKER_MB} MB host-level operating cost per worker, startup peak "
             f"{GRAPHPC_MEASURED_PEAK_PER_WORKER_MB} MB (measured with the embedder loaded; retrieval off loads none)",
             ("baseline (upstream M3Agent @ v2026.08.08): thinking_mode adaptive -> temperature forced to 1 and top_p dropped "
              "(mm_agents/m3/agent.py _build_request_body); max_tokens 8192; coordinate_type relative (default); "
              "max_trajectory_length 10 (default: 11-30 images per request); stop_sequences ['</tool_call>', 'Perform the next action. Perform']; "
              if CELL["model_key"] == "MINIMAX_API_KEY" else
              "baseline (upstream's osworld-v2.1 Qwen agent + deltas D1-D9, scripts/python/run_multienv_qwen_baseline.py): "
              "OpenRouter qwen/qwen3.7-plus pinned to alibaba, reasoning enabled on the thinking calls and off on the judge and "
              "summarizer; upstream's agent defaults otherwise (temperature 0.0, top_p 0.9, its output cap, streaming, batching, "
              "the terminal judge, context rescue, image_max 20 / fold 10, the smart_resize size), except frames as JPEG "
              "quality 90 (delta D9; upstream's are PNG) and "
              "generation_timeout 300 s (the launcher's pinned value; upstream's default is 200 s); "
              "the key from .env, never on the command line; "
              if CELL["model_key"] == "OPENROUTER_API_KEY" else
              f"baseline (upstream M3Agent @ v2026.08.08 with the native-tool deltas, mm_agents/deepseek, "
              f"scripts/python/run_multienv_deepseek_baseline.py): {DEEPSEEK_CONTROL_BASE_URL} {CELL['model']}; the pinned "
              f"agent flags {DEEPSEEK_CONTROL_PINNED} (thinking ON as an explicit budget -> temperature forced to 1 and "
              f"top_p dropped by the agent; no --thinking_mode); coordinate_type "
              f"relative (default); max_trajectory_length 10 (default: 11-30 images per request); upstream's stop "
              f"sequences; the computer tool declared natively, one tool_result per stored tool_use; ")
             + f"workers={ctx.num_envs_baseline}",
             f"evaluator / user simulator: provider {ctx.env.get('OSWORLD_EVAL_MODEL_PROVIDER')} model {ctx.env.get('OSWORLD_EVAL_MODEL_NAME')} "
             f"(user sim {ctx.env.get('OSWORLD_USER_SIM_MODEL')}); website suffix {ctx.env.get('WEBSITE_HOST_SUFFIX')}; assets {ctx.env.get('OSWORLD_FILE_BASE_URL')}"]
    return "\n".join(lines)


def host_checks(args: argparse.Namespace, report: Report, ctx: Optional[HostContext] = None) -> Optional[str]:
    root = Path(args.checkout_root).resolve()
    if ctx is None:
        from dotenv import dotenv_values, load_dotenv  # noqa: PLC0415
        load_dotenv(root / ".env")  # the V2 modules (aws manager) and boto3 read os.environ
        env = {k: v for k, v in dotenv_values(root / ".env").items() if v is not None}
        ctx = HostContext(root=root, env=env, num_envs_graphpc=args.num_envs_graphpc,
                          num_envs_baseline=args.num_envs_baseline, allow_api_calls=args.allow_api_calls,
                          acknowledge_resume=frozenset(args.acknowledge_resume))
    report.run(f"checkout tag == {PINNED_TAG}", lambda: (
        CheckResult("", "PASS", ctx.git_describe()) if ctx.git_describe() == PINNED_TAG
        else CheckResult("", "FAIL", f"git describe --tags --exact-match -> {ctx.git_describe()!r}")))

    report.run("vendored snapshot complete per the sync's manifest, from a clean (and, if given, the expected) commit; hand-copied metadata present",
               lambda: check_vendored_snapshot(root, getattr(args, "expect_source_commit", None)))
    report.run("checkout unmodified apart from excluded snapshot + settings files", lambda: check_checkout_unmodified(ctx))
    runner_check = report.run("no runner alive on this host (a live GraphPC runner = FAIL, baseline = WARN)", lambda: check_other_runner_active(ctx))
    heavy_checks_allowed = runner_check.status != "FAIL"
    measured: dict[str, Any] = {}
    report.run("python deps importable under this interpreter (cv2, desktop_env)", lambda: check_python_deps(ctx))
    if CELL["backbone_type"] == "qwen":
        report.run("the openai SDK under this interpreter types an undecodable HTTP 200 as the decode rule expects "
                   "(offline)", lambda: check_sdk_undecodable_200(ctx))
    manifest = root / "evaluation_examples" / "prereg_task_list.json"
    prereg_path = root / "evaluation_examples" / "subset_prereg.json"
    report.run("subset manifest == prereg subset_ids == shipped ids", lambda: check_manifest_vs_prereg(manifest, prereg_path))
    bash_dir, py_dir = root / "scripts" / "bash", root / "scripts" / "python"
    argvs = check_launch_scripts(bash_dir / "run_multienv_graphpc.sh", bash_dir / CELL["baseline_script"],
                                 bash_dir / "osworld_cell_config.sh", cwd=root, extra_env=None, report=report,
                                 graphpc_runner=py_dir / "run_multienv_graphpc.py", baseline_runner=py_dir / CELL["baseline_runner"],
                                 expected_num_envs={"graphpc": ctx.num_envs_graphpc, "baseline": ctx.num_envs_baseline})
    task_ids = load_manifest(manifest).get("tasks", []) if manifest.exists() else []
    prereg = json.loads(prereg_path.read_text(encoding="utf-8")) if prereg_path.exists() else {}
    if "graphpc" in argvs:
        report.run("the GraphPC arm's step budget: launch argv == adapter == record",
                   lambda: check_graphpc_step_budget(prereg, parse_runner_argv(argvs["graphpc"])[1]))
    if CELL["method_surface_key"] != "method_surface":
        report.run(f"the pre-registration declares {CELL['method_surface_key']} (this cell's method surface)",
                   lambda: check_method_surface_declared(prereg))
    rows_holder: dict[str, dict[str, Any]] = {}

    def _tasks() -> CheckResult:
        result, rows = check_task_classes(ctx, task_ids, prereg)
        rows_holder.update(rows)
        return result
    report.run(f"{len(task_ids)} task classes load via task_loader; overrides asserted against the prereg", _tasks)
    report.run("assets dir + release manifest + default AMI", lambda: check_assets_and_release(ctx))
    report.run("default AMI available in the region", lambda: check_ami_available(ctx))
    report.run("every per-task asset file present under OSWORLD_FILE_BASE_URL", lambda: check_task_assets(ctx, task_ids))
    report.run(f".env carries the required keys with the pinned evaluator config and {CELL['key_label']} key shape"
               + (f"; {', '.join(CELL['forbidden_env'])} unset" if CELL["forbidden_env"] else ""), lambda: check_env_keys(ctx))
    if CELL["forbidden_env"]:
        report.run("the forbidden thinking variables are refused when set (negative control)", lambda: check_forbidden_env_bites(ctx))
    report.run("aws sts get_caller_identity", lambda: check_sts(ctx))
    report.run("subnet + security group exist; V2 inbound ports covered", lambda: check_network(ctx))
    report.run("no orphan task VMs from an OSWorld AMI", lambda: check_orphan_vms(ctx))
    report.run("vCPU quota vs planned concurrency (incl. per-task instance overrides)", lambda: check_vcpu_quota(ctx, rows_holder))
    report.run("mocked websites of the subset reachable from the host (DNS + HTTPS/HTTP)", lambda: check_website_reachability(ctx, task_ids))
    if len(argvs) == 2:
        report.run("cold start: empty support set, no prior result.txt (or --acknowledge_resume <arm>)", lambda: check_cold_start(ctx, argvs))
    skipped_live = "skipped: a GraphPC runner is live on this host (see the runner check)"
    if heavy_checks_allowed:
        measured["mem_available_before_build_mb"] = ctx.mem_available_mb()
        report.run("vendored package: build_default constructs the arm's agent (retrieval off)", lambda: check_build_default(ctx, measured))
        report.run("memory bound for the GraphPC arm (per-worker cost x num_envs + 1 GB must fit; 2 GB/worker floor)", lambda: check_memory_bound(ctx, measured))
        if CELL["method_surface_key"] != "method_surface":
            report.run(f"built agent's active_config == the pre-registered {CELL['method_surface_key']}",
                       lambda: check_method_surface(prereg, measured))
    else:
        for name in ("vendored package: build_default constructs the arm's agent (retrieval off)",
                     "memory bound for the GraphPC arm (per-worker cost x num_envs + 1 GB must fit; 2 GB/worker floor)",
                     *([f"built agent's active_config == the pre-registered {CELL['method_surface_key']}"]
                       if CELL["method_surface_key"] != "method_surface" else [])):
            report.run(name, lambda: CheckResult("", "SKIP", skipped_live))
    for route, label in (*CELL["route_probes"],
                         ("evaluator", "evaluator key + model answer (desktop_env model_client)")):
        if ctx.allow_api_calls:
            report.run(label, lambda r=route: CheckResult("", "PASS", ctx.tiny_call(r)))
        else:
            report.run(label, lambda: CheckResult("", "SKIP", "pass --allow_api_calls to spend a few cents on one tiny call"))
    return regime_summary(ctx, argvs, measured) if len(argvs) == 2 else None


# --------------------------------------------------------------------------- CLI
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["local", "host"], required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--cell", choices=sorted(CELLS), default="minimax",
                        help="the cell whose arms are launched: MiniMax-M3 (default), Qwen qwen3.7-plus or DeepSeek deepseek-flash")
    local = parser.add_argument_group("local mode")
    local.add_argument("--repo_root", type=Path, default=_ROOT)
    local.add_argument("--manifest", type=Path, default=_ROOT / "data" / "osworld_v2" / "prereg_task_list.json")
    local.add_argument("--prereg", type=Path, default=_ROOT / "data" / "osworld_v2" / "subset_prereg.json")
    host = parser.add_argument_group("host mode")
    host.add_argument("--checkout_root", type=Path, default=Path.cwd())
    host.add_argument("--num_envs_graphpc", type=int, default=4)
    host.add_argument("--num_envs_baseline", type=int, default=4)
    host.add_argument("--allow_api_calls", action="store_true", help="spend a few cents: one tiny call per route")
    host.add_argument("--acknowledge_resume", nargs="+", choices=["graphpc", "baseline"], default=[],
                      help="the arm(s) whose prior result.txt (and, for graphpc, warm support set) a resume accepts "
                           "instead of failing the cold-start check; e.g. a GraphPC launch against a finished control "
                           "passes `baseline` alone")
    host.add_argument("--expect_source_commit", type=str, default=None,
                      help="GraphPC commit the snapshot must come from (7+ character prefix): the commit it was synced from")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None, ctx: Optional[HostContext] = None) -> int:
    args = parse_args(argv)
    # The cell's values, installed on every call (the Qwen control's flags are read from its vendored runner).
    runner_dir = (Path(args.repo_root) / "scripts" / "osworld" if args.mode == "local"
                  else Path(args.checkout_root) / "scripts" / "python")
    control_runner = runner_dir / CELLS[args.cell]["baseline_runner"]
    select_cell(args.cell, control_runner.read_text(encoding="utf-8") if control_runner.is_file() else None)
    report = Report()
    print(f"launch preflight ({args.mode} mode) — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    regime = None
    if args.mode == "local":
        local_checks(args, report)
    else:
        regime = host_checks(args, report, ctx)
    print()
    print(report.render())
    if regime:
        print()
        print(regime)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_json()
        payload["regime"] = regime
        args.output_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    if report.failed:
        print(f"\nPREFLIGHT FAILED: {len(report.failed)} check(s) — do not launch.", file=sys.stderr)
        return 1
    print("\nPREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
