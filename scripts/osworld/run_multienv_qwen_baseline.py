# GraphPC vendored copy of OSWorld-V2 scripts/python/run_multienv_qwen.py
# at tag osworld-v2.1 (commit 3d778a3c9a34a079316f70df023b166700445792), blob 0720027b9002ab02582cc4ca1013d168cef81a24.
# The OSWorld Qwen cell's CONTROL runner: upstream's own file, byte-identical except this header
# and the deltas below. Each delta is marked in place (`# >>> GraphPC Dn` ... `# <<< GraphPC
# Dn`) and carries the upstream lines it replaces as `#|` lines, so the file reverses
# mechanically to upstream's; the Qwen mode of scripts/osworld/verify_runner_deltas.py holds this
# file and the agent files under scripts/osworld/qwen_control/mm_agents/qwen/ to their blobs.
# Synced to <OSWorld-V2>/scripts/python/run_multienv_qwen_baseline.py and the agent to
# <OSWorld-V2>/mm_agents/qwen/, where `import _repo_path` resolves the checkout root and
# `mm_agents.qwen` imports unchanged; the driver is the checkout's unmodified lib_run_single.
#
# THE DELTAS, the complete list. Everything else is upstream's: batching, the terminal judge,
# context rescue, tool repair, image folding, the prompts, the output cap, temperature 0.0,
# top_p 0.9, --generation_timeout 200, resume and the worker pool.
#  D1 route (client.py, main.py, this file's _smoke): every chat request (main, judge,
#     summarizer, repair, endless, timeout retry, the smoke call) carries the provider pin
#     {"only": ["alibaba"], "allow_fallbacks": false, "require_parameters": true} and an explicit
#     reasoning form ({"enabled": true} on the thinking calls, {"effort": "none"} on the judge and
#     the summarizer) in place of upstream's vLLM thinking kwarg, which is never sent. The base is
#     https://openrouter.ai/api/v1 and the key is read from OPENROUTER_API_KEY, with no OPENAI_* /
#     localhost / "dummy" fallback (a missing key raises). The tokenize POST uses the same base
#     and key. The thinking CLI flag keeps upstream's name.
#  D2 in-band failures (client.py): an HTTP 200 carrying `error` (a body, or a stream chunk,
#     which the SDK raises as a bare APIError), a choice with finish_reason "error", or a
#     non-streamed body without choices raises into the client's existing retry loop: a
#     transient error_type is retried like an incomplete stream, anything else goes to D7. Three
#     provider faults are transient whatever their envelope: an in-band failure whose `message`,
#     or an HTTP 400 whose error `message`
#     or `metadata.raw`, carries "JSON decode error: unexpected EOF", "Download multimodal file
#     timed out" or "Failed to download multimodal content" is retried the same way (the 400 typed
#     TransientStatusError, its body kept on the D4 record). An HTTP 200 the SDK cannot decode at
#     the provider call — a body or stream chunk that is not JSON, or a non-JSON content type that
#     makes a non-streamed create() return a str — is typed UndecodableResponseError and retried
#     the same way.
#  D3 endless-output retry (recovery.py): resends the identical request without
#     repetition_penalty, which the pinned endpoint does not support; at temperature 0 that
#     disables the escape (a retry re-samples the same request).
#  D4 per-call records (client.py, main.py, this file): the runner calls
#     agent.set_api_log_dir(<task>/api_logs) and every HTTP attempt writes api_logs/NNN.json
#     (fields listed in client.py). Observer-only: no request byte changes.
#  D5 result path (this file): the model directory is sanitized like the M3 runner's
#     _model_for_paths, so --model qwen/qwen3.7-plus writes under qwen_qwen3.7-plus/.
#  D6 checkpoints (this file): upstream's --checkpoint_eval_mode / --checkpoint_steps, verbatim
#     from run_multienv_m3.py at v2026.08.08; the driver reads them by getattr.
#  D7 the lost-episode cause table (main.py): predict wraps upstream's body; infra (transport after
#     the retries, 5xx / 429 / 402 / 408 / 401 / 404, an incomplete stream, a transient in-band
#     failure, a provider-fault 400 or an UndecodableResponseError after the retries, a failed
#     screenshot capture, MemoryError)
#     re-raises; everything else returns DONE and writes api_logs/agent_error.json.
#  D8 startup key check (this file): without OPENROUTER_API_KEY the runner exits 1 before
#     logging is configured and before any VM is provisioned (the shape of the GraphPC
#     runner's delta 5).
#  D9 frames (images.py): every frame goes out as JPEG, quality 90, PIL defaults otherwise, at the
#     same smart_resize size, as data:image/jpeg;base64, in place of upstream's lossless PNG:
#     20 retained PNG frames at 1920x1088 reach about 42 MB as base64, past
#     OpenRouter's 30 MB image cap (HTTP 413). The GraphPC arm's copy of images.py carries the same
#     blocks under the same label, so both arms send the same frame.
# Deltas carried: D1, D4, D5, D6, D8 (this file); the agent files under qwen_control carry the rest.
# end of GraphPC header
"""Run OSWorld-V2 tasks with the modular Qwen computer-use agent."""

from __future__ import annotations

try:
    import _repo_path  # noqa: F401
except ModuleNotFoundError:
    from scripts.python import _repo_path  # type: ignore[no-redef]  # noqa: F401

import argparse
import datetime
import json
import logging
import os
import queue
# >>> GraphPC D5
import re
# <<< GraphPC D5
import shutil
import signal
import sys
from multiprocessing import Manager, Process, current_process
from typing import Any

from dotenv import load_dotenv

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from lib_results_logger import log_task_error
from mm_agents.qwen import QwenAgent
# >>> GraphPC D1/D8: the route's key variable and reasoning forms
#|from mm_agents.qwen.client import QwenClient
from mm_agents.qwen.client import (
    OPENROUTER_API_KEY_ENV,
    REASONING_DISABLED,
    REASONING_ENABLED,
    QwenClient,
)
# <<< GraphPC D1/D8
from task_loader import load_task_config, resolve_task_json_path


load_dotenv()

logger = logging.getLogger("desktopenv.experiment")
processes: list[Process] = []


def _volume_size(value: str) -> int:
    text = value.strip().lower()
    for suffix in ("gb", "g"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    try:
        result = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "volume size must be a positive GB value"
        ) from exc
    if result < 1:
        raise argparse.ArgumentTypeError("volume size must be positive")
    return result


def config(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSWorld-V2 with an OpenAI-compatible Qwen computer-use model"
    )
    parser.add_argument("--path_to_vm", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_space", choices=["pyautogui"], default="pyautogui")
    parser.add_argument(
        "--observation_type", choices=["screenshot"], default="screenshot"
    )
    parser.add_argument(
        "--provider_name",
        choices=["aws", "virtualbox", "vmware", "docker", "azure"],
        default="aws",
    )
    parser.add_argument("--client_password", default="")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=500)
    # >>> GraphPC D6: upstream's checkpoint flags, verbatim from run_multienv_m3.py (v2026.08.08)
    parser.add_argument(
        "--checkpoint_eval_mode",
        choices=["off", "inline"],
        default="off",
        help=(
            "Optional intermediate evaluation mode. 'off' preserves the historical "
            "single final evaluation behavior. 'inline' evaluates safe tasks at "
            "the steps listed in --checkpoint_steps without stopping the agent."
        ),
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=str,
        default="",
        help=(
            "Comma-separated logical step numbers for optional checkpoint evals, "
            "for example '150'. The final max_steps evaluation is still written "
            "to result.txt as before.50,100,150"
        ),
    )
    # <<< GraphPC D6
    parser.add_argument(
        "--enable_proxy", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--use_public_ip", action="store_true")
    parser.add_argument(
        "--enable_vnc", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--enable_recording", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--save_model_eval_raw_info", action="store_true")

    parser.add_argument("--test_config_base_dir", default="evaluation_examples")
    parser.add_argument(
        "--test_all_meta_path", default="evaluation_examples/test_v2.json"
    )
    parser.add_argument("--eval_version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--specific_task_id", default=None)

    parser.add_argument("--model", default="qwen-vl")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enable_thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--batch_actions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enable_zoom", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--coord", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--model_screen_width", type=int, default=None)
    parser.add_argument("--model_screen_height", type=int, default=None)
    parser.add_argument("--history_n", type=int, default=500)
    parser.add_argument("--image_max", type=int, default=20)
    parser.add_argument("--fold_size", type=int, default=10)
    parser.add_argument(
        "--summary_folding", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--summary_max_tokens", type=int, default=4096)
    parser.add_argument("--summary_retries", type=int, default=10)
    parser.add_argument(
        "--context_rescue", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--context_max_payload_chars", type=int, default=10_000_000)
    parser.add_argument("--context_max_text_chars", type=int, default=120_000)
    parser.add_argument("--context_rough_text_token_budget", type=int, default=100_000)
    parser.add_argument("--context_summary_chunk_steps", type=int, default=20)
    parser.add_argument(
        "--terminal_status_judge", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--terminal_status_recovery_retries", type=int, default=3)
    parser.add_argument("--terminal_status_judge_max_tokens", type=int, default=2048)
    parser.add_argument(
        "--endless_retry", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--endless_min_chars", type=int, default=8000)
    parser.add_argument("--endless_retry_limit", type=int, default=10)
    parser.add_argument("--tool_repair_retries", type=int, default=2)
    parser.add_argument("--max_api_retries", type=int, default=50)
    parser.add_argument("--retry_sleep_base", type=float, default=3.0)
    parser.add_argument("--retry_sleep_cap", type=float, default=15.0)
    parser.add_argument("--request_timeout", type=float, default=500.0)
    parser.add_argument("--generation_timeout", type=float, default=200.0)
    parser.add_argument("--smoke_only", action="store_true")

    parser.add_argument("--result_dir", default="./results/osworld_v2/qwen")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--snapshot_name", default="init_state")
    parser.add_argument("--volume_size", type=_volume_size, default=None)
    parser.add_argument("--os_type", choices=["Ubuntu", "Windows"], default="Ubuntu")

    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num_envs must be at least 1")
    if args.max_steps < 1 or args.max_tokens < 1:
        parser.error("--max_steps and --max_tokens must be positive")
    if (args.model_screen_width is None) != (args.model_screen_height is None):
        parser.error("model screen width and height must be provided together")
    if args.model_screen_width is not None and args.coord != "absolute":
        parser.error("fixed model dimensions require --coord absolute")
    if args.model_screen_width is not None and (
        args.model_screen_width < 1 or args.model_screen_height < 1
    ):
        parser.error("model screen dimensions must be positive")
    return args


def configure_logging(args: argparse.Namespace) -> None:
    os.makedirs("logs", exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, args.log_level))
    timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s] %(message)s"
    )
    for handler in (
        logging.FileHandler(f"logs/qwen-{timestamp}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        handler.setLevel(getattr(logging, args.log_level))
        root.addHandler(handler)


def _safe_args(args: argparse.Namespace) -> dict[str, Any]:
    values = dict(vars(args))
    for key in ("api_key", "client_password"):
        if values.get(key):
            values[key] = "<redacted>"
    return values


def _example_value(example: Any, key: str, default: Any = None) -> Any:
    getter = getattr(example, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            result = getter(key)
            return default if result is None else result
    return getattr(example, key, default)


def _load_example(domain: str, task_id: str, args: argparse.Namespace) -> Any:
    path = resolve_task_json_path(
        task_id=task_id,
        base_dir=args.test_config_base_dir,
        domain=domain,
        eval_version=args.eval_version,
    )
    return load_task_config(
        path,
        task_id=task_id,
        base_dir=args.test_config_base_dir,
        domain=domain,
        eval_version=args.eval_version,
    )


def _task_os_type(example: Any, args: argparse.Namespace) -> str:
    platform = str(_example_value(example, "platform", "")).strip().lower()
    if platform:
        return "Windows" if platform.startswith("win") else "Ubuntu"
    return args.os_type


def _env_spec(example: Any, args: argparse.Namespace) -> dict[str, Any]:
    os_type = _task_os_type(example, args)
    snapshot = args.snapshot_name
    region = args.region if args.provider_name == "aws" else None
    if args.provider_name == "aws":
        from desktop_env.providers.aws.manager import get_default_ami_id

        snapshot = _example_value(example, "image") or get_default_ami_id(
            region=args.region,
            screen_size=(args.screen_width, args.screen_height),
            os_type=os_type,
        )
    task_volume_size = _example_value(example, "volume_size")
    if task_volume_size is None:
        task_volume_size = args.volume_size
    return {
        "os_type": os_type,
        "snapshot_name": snapshot,
        "region": region,
        "instance_type": _example_value(example, "instance_type"),
        "volume_size": task_volume_size,
    }


def _new_agent(args: argparse.Namespace, platform: str) -> QwenAgent:
    return QwenAgent(
        platform=platform,
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        history_n=args.history_n,
        coordinate_type=args.coord,
        image_max=args.image_max,
        fold_size=args.fold_size,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_thinking=args.enable_thinking,
        stream=args.stream,
        enable_batch_actions=args.batch_actions,
        enable_zoom=args.enable_zoom,
        model_screen_width=args.model_screen_width,
        model_screen_height=args.model_screen_height,
        summary_folding=args.summary_folding,
        summary_max_tokens=args.summary_max_tokens,
        summary_retries=args.summary_retries,
        enable_context_rescue=args.context_rescue,
        context_max_payload_chars=args.context_max_payload_chars,
        context_max_text_chars=args.context_max_text_chars,
        context_rough_text_token_budget=args.context_rough_text_token_budget,
        context_summary_chunk_steps=args.context_summary_chunk_steps,
        terminal_status_judge=args.terminal_status_judge,
        terminal_status_recovery_retries=args.terminal_status_recovery_retries,
        terminal_status_judge_max_tokens=args.terminal_status_judge_max_tokens,
        enable_endless_retry=args.endless_retry,
        endless_min_chars=args.endless_min_chars,
        endless_retry_limit=args.endless_retry_limit,
        tool_repair_retries=args.tool_repair_retries,
        max_api_retries=args.max_api_retries,
        retry_sleep_base=args.retry_sleep_base,
        retry_sleep_cap=args.retry_sleep_cap,
        request_timeout=args.request_timeout,
        generation_timeout=args.generation_timeout,
        top_k=args.top_k,
    )


# >>> GraphPC D5: the model directory, sanitized like the M3 runner's _model_for_paths
_PATH_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def _model_for_paths(args: argparse.Namespace) -> str:
    return _PATH_SANITIZE_RE.sub("_", args.model).strip("_") or "qwen"


# <<< GraphPC D5
def run_env_tasks(task_queue: Any, args: argparse.Namespace, scores: Any) -> None:
    env: DesktopEnv | None = None
    agent: QwenAgent | None = None
    current_spec: dict[str, Any] | None = None

    def stop_child(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop_child)
    signal.signal(signal.SIGTERM, stop_child)
    try:
        while True:
            try:
                item = task_queue.get(timeout=5)
            except queue.Empty:
                break
            if item is None:
                break
            domain, task_id = item
            task_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                # >>> GraphPC D5
#|                args.model,
                _model_for_paths(args),
                # <<< GraphPC D5
                domain,
                task_id,
            )
            handler: logging.Handler | None = None
            try:
                example = _load_example(domain, task_id, args)
                spec = _env_spec(example, args)
                if env is None or spec != current_spec:
                    if env is not None:
                        env.close()
                    env = DesktopEnv(
                        path_to_vm=args.path_to_vm,
                        action_space=args.action_space,
                        provider_name=args.provider_name,
                        region=spec["region"],
                        snapshot_name=spec["snapshot_name"],
                        screen_size=(args.screen_width, args.screen_height),
                        headless=args.headless,
                        os_type=spec["os_type"],
                        require_a11y_tree=False,
                        enable_proxy=args.enable_proxy,
                        client_password=args.client_password,
                        instance_type=spec["instance_type"],
                        volume_size=spec["volume_size"],
                        use_public_ip=args.use_public_ip,
                        force_disable_vnc=not args.enable_vnc,
                        force_disable_recording=not args.enable_recording,
                        cache_dir=args.cache_dir,
                    )
                    agent = _new_agent(args, spec["os_type"])
                    current_spec = spec
                os.makedirs(task_dir, exist_ok=True)
                handler = logging.FileHandler(
                    os.path.join(task_dir, "eval.log"), encoding="utf-8"
                )
                handler.setLevel(logging.DEBUG)
                logging.getLogger().addHandler(handler)
                if args.save_model_eval_raw_info:
                    os.environ["OSWORLD_EVAL_SAVE_RAW_DIR"] = os.path.join(
                        task_dir, "model_eval_file_log"
                    )
                assert env is not None and agent is not None
                # >>> GraphPC D4: this task's per-call records
                agent.set_api_log_dir(os.path.join(task_dir, "api_logs"))
                # <<< GraphPC D4
                logger.info(
                    "[%s] task=%s/%s instruction=%s",
                    current_process().name,
                    domain,
                    task_id,
                    _example_value(example, "instruction"),
                )
                lib_run_single.run_single_example(
                    agent,
                    env,
                    example,
                    args.max_steps,
                    _example_value(example, "instruction"),
                    args,
                    task_dir,
                    scores,
                )
            except Exception as exc:
                logger.exception("Task failed: %s/%s", domain, task_id)
                os.makedirs(task_dir, exist_ok=True)
                try:
                    log_task_error({"id": task_id}, str(exc), task_dir, args)
                except Exception:
                    logger.exception("Failed to write structured task error")
                with open(
                    os.path.join(task_dir, "traj.jsonl"), "a", encoding="utf-8"
                ) as file:
                    file.write(
                        json.dumps({"Error": f"{domain}/{task_id} - {exc}"}) + "\n"
                    )
            finally:
                os.environ.pop("OSWORLD_EVAL_SAVE_RAW_DIR", None)
                if handler is not None:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
    finally:
        if env is not None:
            env.close()


def _filter_unfinished(
    metadata: dict[str, list[str]], args: argparse.Namespace
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    root = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        # >>> GraphPC D5
#|        args.model,
        _model_for_paths(args),
        # <<< GraphPC D5
    )
    for domain, task_ids in metadata.items():
        remaining: list[str] = []
        for task_id in task_ids:
            task_dir = os.path.join(root, domain, str(task_id))
            if os.path.isfile(os.path.join(task_dir, "result.txt")):
                continue
            if os.path.lexists(task_dir):
                shutil.rmtree(task_dir)
            remaining.append(str(task_id))
        if remaining:
            result[domain] = remaining
    return result


def _select_tasks(
    metadata: dict[str, list[str]], args: argparse.Namespace
) -> dict[str, list[str]]:
    selected = metadata
    if args.domain != "all":
        selected = {args.domain: list(metadata.get(args.domain, []))}
    if args.specific_task_id:
        task_id = str(args.specific_task_id)
        selected = {
            domain: [str(item) for item in items if str(item) == task_id]
            for domain, items in selected.items()
        }
        selected = {domain: items for domain, items in selected.items() if items}
        if not selected:
            raise ValueError(
                f"Task {task_id} is not present in {args.test_all_meta_path}"
            )
    return _filter_unfinished(selected, args)


def _smoke(args: argparse.Namespace) -> None:
    client = QwenClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        stream=args.stream,
        max_retries=min(args.max_api_retries, 3),
        retry_sleep_base=args.retry_sleep_base,
        retry_sleep_cap=args.retry_sleep_cap,
        request_timeout=args.request_timeout,
        generation_timeout=args.generation_timeout,
        top_k=args.top_k,
    )
    completion = client.complete(
        {
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "max_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            # >>> GraphPC D1: the smoke call states its reasoning form (the client adds the pin)
#|            "extra_body": {"enable_thinking": args.enable_thinking},
            "extra_body": {
                "reasoning": dict(
                    REASONING_ENABLED if args.enable_thinking else REASONING_DISABLED
                )
            },
            # <<< GraphPC D1
        }
    )
    print(f"Qwen smoke response: {(completion.content or completion.reasoning)!r}")


def _shutdown(signum: int, _frame: Any) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> None:
    global processes
    args = config(argv)
    # >>> GraphPC D8: no key, no start (before logging and before any VM)
    if not os.environ.get(OPENROUTER_API_KEY_ENV):
        print(
            f"ERROR: the Qwen control requires {OPENROUTER_API_KEY_ENV} in the environment "
            "(host .env)."
        )
        sys.exit(1)
    # <<< GraphPC D8
    configure_logging(args)
    if args.smoke_only:
        _smoke(args)
        return

    with open(args.test_all_meta_path, encoding="utf-8") as file:
        metadata = json.load(file)
    selected = _select_tasks(metadata, args)
    tasks = [
        (domain, str(task_id))
        for domain, task_ids in selected.items()
        for task_id in task_ids
    ]
    target_root = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        # >>> GraphPC D5
#|        args.model,
        _model_for_paths(args),
        # <<< GraphPC D5
    )
    os.makedirs(target_root, exist_ok=True)
    with open(os.path.join(target_root, "args.json"), "w", encoding="utf-8") as file:
        json.dump(_safe_args(args), file, indent=2, ensure_ascii=False)
        file.write("\n")
    if not tasks:
        logger.info("No unfinished tasks")
        return

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    worker_count = min(args.num_envs, len(tasks))
    with Manager() as manager:
        task_queue = manager.Queue()
        scores = manager.list()
        for task in tasks:
            task_queue.put(task)
        for _ in range(worker_count):
            task_queue.put(None)
        processes = [
            Process(
                target=run_env_tasks,
                args=(task_queue, args, scores),
                name=f"EnvProcess-{index + 1}",
            )
            for index in range(worker_count)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed = [process for process in processes if process.exitcode not in (0, None)]
        if failed:
            raise SystemExit(
                "Qwen workers failed: "
                + ", ".join(f"{process.name}={process.exitcode}" for process in failed)
            )
        values = list(scores)
        logger.info("Average score: %s", sum(values) / len(values) if values else 0.0)


if __name__ == "__main__":
    main()
