# GraphPC vendored copy of OSWorld-V2 mm_agents/m3/agent.py
# at tag v2026.08.08 (commit d578d2d4e0dc82b43e270fdaa7fa89d9708cd154), blob 8943c6df512c47c237f8ac80c798da553b8552ab.
# The OSWorld DeepSeek cell's CONTROL is upstream's own M3 agent: this file is byte-identical to
# that blob except this header and the deltas listed below. Each delta is marked in place
# (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`) and carries the upstream lines it
# replaces as `#|` lines, so the file reverses mechanically to upstream's; the DeepSeek mode
# of scripts/osworld/verify_runner_deltas.py holds it to that. The delta list with its
# reasons is in the header of scripts/osworld/run_multienv_deepseek_baseline.py.
# Deltas carried: D2 (the computer tool declared natively), D3 (the action read from the tool_use block; the thinking never parsed), D4 (the tool protocol in the stored history), D6 (the api_log's request_time stamped before the call).
# end of GraphPC header
"""M3 agent — OSWorld-official agent interface (``predict`` / ``reset`` /
``set_api_log_dir``) with an Anthropic-Messages-API transport.

Mirrors how `lib_run_single.py` and `scripts/python/run_multienv_*.py` drive
all other OSWorld agents:

    agent = M3Agent(model="<model-name>", max_tokens=8192, temperature=0.6)
    agent.reset(runtime_logger)
    if hasattr(agent, "set_api_log_dir"):
        agent.set_api_log_dir(os.path.join(example_result_dir, "api_logs"))
    while True:
        response, actions = agent.predict(instruction, obs)
        ...

Behaviour summary:

  * 1920x1080 ORIGINAL screenshots — no LANCZOS resize.
  * ``[0, 1000]`` normalized coordinates (``coordinate_type="relative"``).
  * System prompt = ``M3_SYSTEM_PROMPT_TEMPLATE.format(date_str=...)`` passed
    as the first ``{"role": "system", ...}`` message (mapped onto Anthropic's
    top-level ``system=`` parameter inside ``_call_llm``).
  * First user turn = image + instruction text; subsequent user turns =
    image only (tool result is implicit).
  * Image truncation: keep most recent ``only_n_most_recent_images`` images,
    drop older ones in chunks of ``image_truncation_threshold``.
  * Assistant turns stored as raw text with ``<tool_call>...</tool_call>``
    wrapper restored (see ``parser.wrap_for_history``).
  * Special tokens ``WAIT`` / ``DONE`` / ``FAIL`` returned as standalone
    strings. ``CALL_USER`` is converted to an empty action list so the local
    runner records an ``ASK_USER`` turn and feeds the user response back.

Transport: the model is reached through the Anthropic Messages API. Endpoint
is configured via standard Anthropic SDK env vars (``ANTHROPIC_BASE_URL`` /
``ANTHROPIC_API_KEY``), or passed directly as constructor kwargs. M3 emits
tool calls as ``<tool_call>{json}</tool_call>`` text inside the assistant's
text block (not as native Anthropic ``tool_use`` blocks); the parser handles
the text shape directly.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from PIL import Image

from .parser import parse_m3_response, wrap_for_history
from .prompts import M3_SYSTEM_PROMPT_TEMPLATE, MACOS_KEYBOARD_MAPPING
# >>> GraphPC D2: the computer tool declared natively (its definition lives beside the prompt)
from .prompts import COMPUTER_TOOL
# <<< GraphPC D2

logger = logging.getLogger("desktopenv.agent.m3")


_IMAGE_FORMAT = os.environ.get("M3_IMAGE_FORMAT", "JPEG").upper()
_IMAGE_QUALITY = int(os.environ.get("M3_IMAGE_QUALITY", "90"))
_IMAGE_MIME = "image/jpeg" if _IMAGE_FORMAT == "JPEG" else f"image/{_IMAGE_FORMAT.lower()}"


def _normalize_platform(platform: str) -> str:
    value = (platform or "ubuntu").strip().lower()
    if value in ("ubuntu", "linux"):
        return "ubuntu"
    if value.startswith("win"):
        return "windows"
    if value in ("mac", "macos", "darwin"):
        return "macos"
    return value


def _format_system_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%A, %B %d, %Y")
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10]).strftime("%A, %B %d, %Y")
    except ValueError:
        return raw


def _encode_screenshot(image_bytes: bytes) -> Tuple[str, str]:
    """Return ``(base64_str, mime_type)`` for the on-wire image content.

    M3 expects 1920x1080 screenshots. The default JPEG encoding
    (env-overridable via ``M3_IMAGE_FORMAT`` / ``M3_IMAGE_QUALITY``)
    shrinks wire bytes ~6× vs PNG with negligible pixel difference, which
    keeps requests well under any gateway body-size limit even with
    cumulative image history. Set ``M3_IMAGE_FORMAT=PNG`` to skip the
    transcoding.
    """
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = BytesIO()
    save_kwargs: Dict[str, Any] = {"format": _IMAGE_FORMAT}
    if _IMAGE_FORMAT == "JPEG":
        save_kwargs["quality"] = _IMAGE_QUALITY
        save_kwargs["optimize"] = True
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), _IMAGE_MIME


def _split_data_url(url: str) -> Tuple[str, str]:
    """``data:image/png;base64,XXXX`` → ``("image/png", "XXXX")``."""
    if not url.startswith("data:"):
        raise ValueError(f"expected data: URL, got {url[:32]!r}...")
    head, _, payload = url.partition(",")
    media_type = head[len("data:") :].split(";")[0] or "image/png"
    return media_type, payload


def _content_to_anthropic(content: Any) -> Any:
    """Translate one OpenAI-style message ``content`` field to Anthropic shape.

    ``str`` passes through. A list of OpenAI content blocks is converted:
    ``image_url`` blocks become Anthropic ``image`` blocks with a base64
    ``source``; ``text`` blocks are preserved verbatim; unknown block types
    are passed through with a warning so a server-side change to the block
    vocabulary doesn't silently drop turns.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    out: List[Dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            media_type, data = _split_data_url(url)
            out.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        elif btype == "image":
            out.append(block)
        # >>> GraphPC D4: the stored history is already in the tool protocol's block shapes
        elif btype in ("tool_result", "tool_use", "thinking", "redacted_thinking"):
            out.append(block)
        # <<< GraphPC D4
        else:
            logger.warning("Unknown content block type %r — passing through", btype)
            out.append(block)
    return out


class M3Agent:
    """M3 agent — OSWorld-official interface, Anthropic-Messages transport.

    Inputs / outputs match ``lib_run_single.py`` expectations exactly:
      * ``predict(instruction, obs) -> (response_text, pyautogui_code)``
      * ``reset(_logger=None, **kwargs)`` — clears trajectory state
      * ``set_api_log_dir(path)`` — opt-in; writes one JSON per LLM call
        with the full request body, response body and retry metadata
        (useful for offline analysis).

    Endpoint configuration follows the standard Anthropic SDK convention.
    Either pass ``base_url`` / ``api_key`` to the constructor, or set the
    ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` env vars. Tokens whose
    value starts with ``sk-`` are sent as ``x-api-key`` (the SDK's
    ``api_key=`` kwarg); other strings are sent as ``Authorization: Bearer``
    (the SDK's ``auth_token=`` kwarg) — picks the right header for both
    Anthropic-style API keys and JWT-bearer proxies without an extra knob.
    """

    def __init__(
        self,
        platform: str = "ubuntu",
        model: Optional[str] = None,
        max_tokens: int = 8192,
        top_p: Optional[float] = None,
        temperature: Optional[float] = 0.6,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        coordinate_type: str = "relative",   # [0, 1000] normalised
        only_n_most_recent_images: int = 10,
        image_truncation_threshold: int = 20,
        system_date: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        client_password: str = "osworld-public-evaluation",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        thinking_mode: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        # ───────── runner-compat kwargs (accepted but mapped/discarded) ─────────
        # ``run.py`` and ``scripts/python/run_multienv*.py`` always pass these
        # because every OSWorld baseline agent accepts them. We keep the names
        # so the canonical entrypoints work without modification.
        max_trajectory_length: Optional[int] = None,  # → mapped to only_n_most_recent_images
        a11y_tree_max_tokens: Optional[int] = None,   # → discarded (M3 is screenshot-only)
        **_unused,
    ):
        if action_space != "pyautogui":
            raise ValueError(
                f"M3Agent only supports action_space='pyautogui', got {action_space!r}"
            )
        if observation_type != "screenshot":
            raise ValueError(
                "M3Agent only supports observation_type='screenshot' (a11y_tree / som "
                f"are not supported by M3). Got {observation_type!r}"
            )
        if coordinate_type not in ("relative", "absolute"):
            raise ValueError(
                f"coordinate_type must be 'relative' or 'absolute', got {coordinate_type!r}"
            )
        platform = _normalize_platform(platform)
        if platform not in ("ubuntu", "macos", "windows"):
            logger.warning(
                "Unknown platform=%r; treating as 'ubuntu'. macOS keyboard mapping "
                "is only appended when platform='macos'.", platform,
            )
            platform = "ubuntu"

        # Resolve transport config: explicit args > env vars. ``ANTHROPIC_AUTH_TOKEN``
        # is accepted as a legacy alias for ``ANTHROPIC_API_KEY`` so old configs
        # that pre-date the rename keep working.
        resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        resolved_api_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        resolved_model = model or os.environ.get("ANTHROPIC_MODEL") or "m3"
        if not resolved_base_url or not resolved_api_key:
            raise RuntimeError(
                "M3Agent requires base_url + api_key — pass them as kwargs or set "
                "ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) "
                "in the environment."
            )

        # Extra HTTP headers to send on every request (e.g. enterprise-gateway
        # routing hints, ``anthropic-beta`` flags, custom auth). Resolution:
        # kwarg > ``ANTHROPIC_EXTRA_HEADERS`` env var (JSON-encoded dict).
        resolved_extra_headers: Dict[str, str] = {}
        if extra_headers:
            resolved_extra_headers = dict(extra_headers)
        else:
            env_val = os.environ.get("ANTHROPIC_EXTRA_HEADERS", "").strip()
            if env_val:
                try:
                    parsed = json.loads(env_val)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "ANTHROPIC_EXTRA_HEADERS is not valid JSON (%s); ignored.", e
                    )
                else:
                    if isinstance(parsed, dict):
                        resolved_extra_headers = {str(k): str(v) for k, v in parsed.items()}
                    else:
                        logger.warning(
                            "ANTHROPIC_EXTRA_HEADERS must decode to a dict, got %s; ignored.",
                            type(parsed).__name__,
                        )

        self.platform = platform
        self.model = resolved_model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.coordinate_type = coordinate_type
        self.thinking_mode = (thinking_mode or os.environ.get("M3_THINKING_MODE") or "").strip().lower()
        if thinking_budget is None:
            raw_budget = os.environ.get("M3_THINKING_BUDGET", "").strip()
            thinking_budget = int(raw_budget) if raw_budget.isdigit() and int(raw_budget) > 0 else None
        self.thinking_budget = thinking_budget
        # Response-retry policy. Each predict() call wraps ``_call_llm`` in
        # a loop: if ``_call_llm`` raises (network / 5xx) or the response
        # parses to zero actions despite a non-empty body, resample. Costs
        # one extra LLM call per retry, capped at ``max_llm_retries``
        # (default 2, so up to 3 total attempts). Override via the
        # ``M3_MAX_LLM_RETRIES`` env var.
        self.max_llm_retries = int(os.environ.get("M3_MAX_LLM_RETRIES", "2"))
        # If the runner passed max_trajectory_length, take it as the image budget
        # (closest semantic equivalent — both bound how many past observations
        # the model still sees).
        if max_trajectory_length is not None:
            only_n_most_recent_images = max_trajectory_length
        self.only_n_most_recent_images = only_n_most_recent_images
        self.image_truncation_threshold = image_truncation_threshold
        self.default_system_date = system_date or datetime.today().strftime("%A, %B %d, %Y")
        self.system_date = self.default_system_date
        self.task_current_date = None
        self.client_password = client_password
        self.a11y_tree_max_tokens = a11y_tree_max_tokens  # kept for introspection; unused

        # Stops protect against M3 degenerate loops:
        #   - "</tool_call>" — natural turn-end token
        #   - "Perform the next action. Perform" — repeat detected, stop at 2nd
        # Without these a single bad emit can grow context by 8K and OOM by
        # step 50.
        self.stop_sequences = stop_sequences if stop_sequences is not None else [
            "</tool_call>",
            "Perform the next action. Perform",
        ]

        # Auth-style auto-detection: tokens starting with ``sk-`` (common API-key
        # convention) go via the ``x-api-key`` header (``api_key=`` kwarg).
        # Everything else (JWT bearer tokens) goes via ``Authorization: Bearer``
        # (``auth_token=`` kwarg). Each Anthropic-compatible deployment accepts
        # exactly one of these, and the prefix tells us which.
        client_kwargs: Dict[str, Any] = dict(
            base_url=resolved_base_url,
            max_retries=4,
            timeout=180.0,
        )
        if resolved_extra_headers:
            client_kwargs["default_headers"] = resolved_extra_headers
        if resolved_api_key.startswith("sk-"):
            client_kwargs["api_key"] = resolved_api_key
            auth_mode = "api_key (x-api-key)"
        else:
            client_kwargs["auth_token"] = resolved_api_key
            auth_mode = "auth_token (Authorization: Bearer)"
        self._client = Anthropic(**client_kwargs)
        logger.info(
            "M3Agent ready — base_url=%s model=%s max_tokens=%d temperature=%s auth=%s",
            resolved_base_url, self.model, self.max_tokens, self.temperature, auth_mode,
        )

        # Trajectory state — wiped by reset()
        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []     # raw assistant text per step (history-wrapped)
        self.screenshots: List[str] = []   # base64 image strings (format from M3_IMAGE_FORMAT)
        self.user_responses: List[Optional[str]] = []
        # >>> GraphPC D3/D4: the stored turns (tool protocol), the last reply's blocks
        # D4: one entry per predict, {"blocks": [...], "tool_use_id": str | None} — the assistant turn as the
        # protocol re-sends it (_stored_turn). D3: the last reply's tool_use / thinking blocks (_call_llm) and
        # what predict executed / set aside.
        self.turn_blocks: List[Dict[str, Any]] = []
        self._last_tool_uses: List[Dict[str, Any]] = []
        self._last_thinking_blocks: List[Dict[str, Any]] = []
        self._executed_tool_use: Optional[Dict[str, Any]] = None
        self._second_tool_use: List[Dict[str, Any]] = []
        # <<< GraphPC D3/D4

        # API log capture (multi_turn.json prerequisite). lib_run_single
        # calls set_api_log_dir() if it exists.
        self._api_log_dir: Optional[str] = None
        self._api_call_count: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # OSWorld lifecycle hooks (called by lib_run_single / run_multienv_*)
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, _logger=None, **kwargs):
        """Clear trajectory state between tasks.

        ``lib_run_single.run_single_example`` calls
        ``agent.reset(runtime_logger, vm_ip=env.vm_ip)`` so the agent can pin
        snapshot-aware behaviour to the freshly-booted VM. M3 doesn't need
        ``vm_ip`` (the LLM transport handles routing on its own), so we
        silently absorb the kwarg — anything else passed in is also ignored
        for forward-compat with future runners.
        """
        global logger
        if _logger is not None:
            logger = _logger
        self.thoughts.clear()
        self.actions.clear()
        self.observations.clear()
        self.responses.clear()
        self.screenshots.clear()
        self.user_responses.clear()
        # >>> GraphPC D3/D4: the stored turns and the last reply go with the trajectory
        self.turn_blocks.clear()
        self._last_tool_uses = []
        self._last_thinking_blocks = []
        self._executed_tool_use = None
        self._second_tool_use = []
        # <<< GraphPC D3/D4
        self._api_call_count = 0
        task_date = _format_system_date(getattr(self, "task_current_date", None))
        self.system_date = task_date or self.default_system_date

    def set_api_log_dir(self, path: str):
        self._api_log_dir = path
        os.makedirs(path, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Prompt assembly
    # ─────────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        prompt = M3_SYSTEM_PROMPT_TEMPLATE.format(
            date_str=self.system_date,
            client_password=self.client_password,
        )
        if self.platform == "windows":
            prompt = prompt.replace(
                "You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.",
                "You are utilising a Windows virtual machine using x86_64 architecture with internet access.",
            ).replace(
                "Home directory of this Ubuntu system is '/home/user'.",
                "Home directory of this Windows system is 'C:\\\\Users\\\\user'.",
            )
        elif self.platform == "macos":
            prompt = prompt.replace(
                "You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.",
                "You are utilising a macOS virtual machine using x86_64 architecture with internet access.",
            ).replace(
                "Home directory of this Ubuntu system is '/home/user'.",
                "Home directory of this macOS system is '/Users/user'.",
            )
        if self.platform == "macos":
            prompt += MACOS_KEYBOARD_MAPPING
        return prompt

    def _user_message_with_image(self, b64_img: str, text: Optional[str] = None) -> Dict:
        """Build an OpenAI-style user message with image + optional text.

        The image is delivered as a ``data:<mime>;base64,...`` URL inside
        an ``image_url`` content block — the de-facto standard accepted by
        OpenAI-compatible and most VLM endpoints. Mime type matches
        whatever ``_encode_screenshot`` produced (JPEG by default; PNG
        when configured via ``M3_IMAGE_FORMAT``). The shape is translated
        to Anthropic's ``image.source.base64`` block inside
        ``_translate_messages`` before being sent over the wire.
        """
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:{_IMAGE_MIME};base64,{b64_img}"}}
        ]
        if text:
            content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    @staticmethod
    def _user_message_tool_result_placeholder() -> Dict:
        """Placeholder user turn for steps whose screenshot was dropped by the
        image-truncation policy. Text matches what the chat template
        rendered when older retained images were still visible, so the
        truncated turn slots into the conversation seamlessly."""
        return {"role": "user", "content": [{"type": "text", "text": "Tool result: Success"}]}

    def _build_messages(self, instruction: str, current_b64: str) -> List[Dict]:
        """Assemble the full messages list for one ``predict`` call.

        Layout:
          1. system: M3_SYSTEM_PROMPT_TEMPLATE.format(date_str=...)
          2. user: image_0 (initial obs) + instruction text
          3. for i in 0..k-1:
               assistant: self.responses[i]                       (history-wrapped)
               user:      screenshot[i+1]   (or placeholder if dropped)
          4. user: current screenshot

        Image truncation: keep most recent ``K = only_n_most_recent_images``
        screenshots, drop older ones in chunks of
        ``T = image_truncation_threshold``.
        """
        K = self.only_n_most_recent_images
        T = self.image_truncation_threshold
        k = len(self.responses)
        remove = max(0, k - K)
        remove -= remove % T
        if remove > 0:
            logger.info(
                f"[ImageTruncation] step={k+1} k={k} K={K} T={T} → "
                f"drop oldest {remove} of {k} tool_result images, "
                f"keep {k - remove} + initial = {k - remove + 1}"
            )

        messages: List[Dict] = [
            {"role": "system", "content": self._build_system_prompt()},
            self._user_message_with_image(self.screenshots[0], instruction),
        ]
        # >>> GraphPC D4: the tool protocol. The Anthropic-format endpoint requires exactly one tool_result for
        # every tool_use of the previous assistant turn, and none otherwise (a mismatch is an HTTP 400:
        # "tool_use ids were found without tool_result blocks immediately after"). Each stored turn goes back as
        # its blocks; the user turn after it carries one tool_result per stored tool_use — the screenshot inside
        # it when the turn is retained, M3's placeholder text when the image fold dropped it — with the driver's
        # "User response:" text, when there is one, as a text block BESIDE the tool_result (never instead of it,
        # which is what upstream's fold did with the placeholder). A turn without a tool_use gets upstream's own
        # user message (image + text, or the text / placeholder when folded).
#|        for i in range(k):
#|            messages.append({"role": "assistant", "content": self.responses[i]})
#|            tool_result_idx = i + 1
#|            if tool_result_idx <= remove:
#|                user_text = self.user_responses[i] if i < len(self.user_responses) else None
#|                if user_text:
#|                    messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
#|                else:
#|                    messages.append(self._user_message_tool_result_placeholder())
#|            else:
#|                user_text = self.user_responses[i] if i < len(self.user_responses) else None
#|                messages.append(self._user_message_with_image(self.screenshots[tool_result_idx], user_text))
        for i in range(k):
            turn = self.turn_blocks[i]
            tool_result_idx = i + 1
            retained = tool_result_idx > remove
            user_text = self.user_responses[i] if i < len(self.user_responses) else None
            messages.append({"role": "assistant", "content": list(turn["blocks"])})
            messages.append(self._user_turn_after(turn, tool_result_idx, retained=retained, user_text=user_text))
        # <<< GraphPC D4
        # Current observation appended by predict() before this method runs,
        # so self.screenshots[-1] == current_b64 — no extra append needed.
        return messages

    # >>> GraphPC D3/D4: the stored turn and the user turn after it
    @staticmethod
    def _anthropic_image_block(b64_img: str) -> Dict[str, Any]:
        return {"type": "image", "source": {"type": "base64", "media_type": _IMAGE_MIME, "data": b64_img}}

    def _stored_turn(self, text: str) -> Dict[str, Any]:
        """The assistant turn as the history keeps it, for every reply shape (D3): the thinking blocks as
        returned, the text block when there is text, the EXECUTED tool_use only. A reply with no tool_use
        and no text (a thinking-only reply, or a retry exhaustion that left nothing) stores a one-token
        placeholder text, because an assistant turn with no blocks is invalid on the wire."""
        blocks: List[Dict[str, Any]] = [dict(block) for block in self._last_thinking_blocks]
        tool_use = self._executed_tool_use
        if text:
            blocks.append({"type": "text", "text": text})
        if tool_use is not None:
            blocks.append({"type": "tool_use", "id": tool_use["id"], "name": tool_use["name"],
                           "input": tool_use.get("input") or {}})
        elif not text:
            blocks.append({"type": "text", "text": "(no action)"})
        return {"blocks": blocks, "tool_use_id": tool_use["id"] if tool_use is not None else None}

    def _user_turn_after(self, turn: Dict[str, Any], tool_result_idx: int, *, retained: bool,
                         user_text: Optional[str]) -> Dict[str, Any]:
        """The user turn that follows a stored assistant turn (D4)."""
        tool_use_id = turn.get("tool_use_id")
        if tool_use_id is None:
            if retained:
                return self._user_message_with_image(self.screenshots[tool_result_idx], user_text)
            if user_text:
                return {"role": "user", "content": [{"type": "text", "text": user_text}]}
            return self._user_message_tool_result_placeholder()
        if retained:
            result_content: List[Dict[str, Any]] = [self._anthropic_image_block(self.screenshots[tool_result_idx])]
        else:
            result_content = [{"type": "text", "text": "Tool result: Success"}]
        content: List[Dict[str, Any]] = [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result_content}]
        if user_text:
            content.append({"type": "text", "text": user_text})
        return {"role": "user", "content": content}

    def _parse_native_reply(self, text: str, original_width: int, original_height: int,
                            processed_width: int, processed_height: int) -> Tuple[str, List[str]]:
        """D3: the action comes from the FIRST tool_use block named `computer`; the thinking is never read.
        `[INFEASIBLE]` is read from the TEXT blocks only and, as upstream, wins over any tool call. The
        executed call is handed to upstream's own parser as the one <tool_call> block it expects, so the
        coordinate scaling, the argument validation (a defect is a `sleep(0.1)` no-op) and the token handling
        are upstream's unchanged. Every other tool_use block is set aside in _second_tool_use (recorded in the
        api_log by _save_api_log, never executed, never stored)."""
        tool_uses = list(self._last_tool_uses)
        if "[INFEASIBLE]" in (text or ""):
            # upstream's precedence: the token wins over any tool call; every tool_use goes to the record
            self._executed_tool_use = None
            self._second_tool_use = tool_uses
            return "[INFEASIBLE]", ["FAIL"]
        first = next((block for block in tool_uses if block.get("name") == "computer"), None)
        self._executed_tool_use = first
        self._second_tool_use = [block for block in tool_uses if block is not first]
        if first is None:
            return "", []
        payload = json.dumps({"name": "computer", "arguments": first.get("input") or {}}, ensure_ascii=False)
        low_level_instruction, pyautogui_code = parse_m3_response(
            f"<tool_call>\n{payload}\n</tool_call>",
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            coordinate_type=self.coordinate_type,
        )
        if not low_level_instruction:
            low_level_instruction = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
        return low_level_instruction, pyautogui_code

    # <<< GraphPC D3/D4
    # ─────────────────────────────────────────────────────────────────────
    # LLM transport — Anthropic Messages API
    # ─────────────────────────────────────────────────────────────────────

    def _translate_messages(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """OpenAI-style messages → ``(system_text, anthropic_messages)``.

        ``_build_messages`` produces messages in OpenAI shape (system role +
        image_url content blocks). Anthropic Messages API takes ``system`` as
        a top-level parameter and expects ``image.source.base64`` blocks, so
        we split system out and convert image blocks via
        ``_content_to_anthropic``.
        """
        system_parts: List[str] = []
        anth_messages: List[Dict] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
                else:
                    system_parts.append(str(content))
                continue
            anth_messages.append({
                "role": role,
                "content": _content_to_anthropic(msg.get("content", "")),
            })
        return "\n".join(s for s in system_parts if s), anth_messages

    def _build_request_body(self, messages: List[Dict]) -> Dict[str, Any]:
        """Build the actual Anthropic Messages API payload from OpenAI-shape
        messages. Used by both ``_call_llm`` (to issue the request) and
        ``predict`` (to log the payload that was sent on the wire), so the
        saved ``api_logs/NNN_*.json`` is a faithful replay of the request.
        """
        system_text, anth_messages = self._translate_messages(messages)
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": anth_messages,
            "max_tokens": self.max_tokens,
        }
        if system_text:
            body["system"] = system_text
        # >>> GraphPC D2: the computer tool declared natively; the model answers with a tool_use block
        body["tools"] = [COMPUTER_TOOL]
        body["tool_choice"] = {"type": "auto"}
        # <<< GraphPC D2
        thinking_enabled = self.thinking_mode == "adaptive" or bool(self.thinking_budget)
        if thinking_enabled:
            body["temperature"] = 1
        elif self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None and not thinking_enabled:
            body["top_p"] = self.top_p
        if self.stop_sequences:
            body["stop_sequences"] = list(self.stop_sequences)
        if self.thinking_mode == "adaptive":
            body["thinking"] = {"type": "adaptive"}
            if self.thinking_budget:
                body["thinking"]["budget_tokens"] = self.thinking_budget
        elif self.thinking_budget:
            body["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        elif self.thinking_mode == "disabled":
            body["thinking"] = {"type": "disabled"}
        return body

    def _call_llm(self, messages: List[Dict]) -> Tuple[str, Any]:
        """Send ``messages`` via the Anthropic SDK and return
        ``(response_text, raw_response)``.

        Exceptions are NOT caught here — they propagate to ``predict()``'s
        retry loop, which treats them as a retryable failure (so transient
        500s from the upstream proxy don't silently no-op a whole step).

        The response body is the concatenation of all ``TextBlock.text`` in
        ``response.content``. If the model emits a ``thinking`` block (e.g.
        when extended thinking is enabled on the upstream), it is wrapped as
        ``<mm:think>...</mm:think>`` and prepended so the chat-template
        renders the thinking block consistently across turns.
        """
        body = self._build_request_body(messages)
        response = self._client.messages.create(**body)

        # >>> GraphPC D3: the reply's text is its text blocks alone; the tool_use blocks and the thinking blocks are
        # kept beside it (predict reads the action from the former; the history re-sends the latter). Upstream
        # prepended the thinking to the parsed text, which is what let a token or a drafted call inside the
        # thinking act; under native tool use nothing in the thinking is ever read.
#|        text_parts: List[str] = []
#|        for block in getattr(response, "content", []) or []:
#|            btype = getattr(block, "type", None)
#|            if btype == "text":
#|                text_parts.append(getattr(block, "text", ""))
#|            elif btype == "thinking":
#|                thinking = getattr(block, "thinking", "")
#|                if thinking and "<mm:think>" not in "".join(text_parts):
#|                    text_parts.insert(0, f"<mm:think>{thinking}</mm:think>\n")
#|        return "".join(text_parts), response
        text_parts: List[str] = []
        tool_uses: List[Dict[str, Any]] = []
        thinking_blocks: List[Dict[str, Any]] = []
        for block in getattr(response, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_uses.append({"id": getattr(block, "id", None), "name": getattr(block, "name", None),
                                  "input": copy.deepcopy(getattr(block, "input", None) or {})})
            elif btype == "thinking":
                stored = {"type": "thinking", "thinking": getattr(block, "thinking", "")}
                signature = getattr(block, "signature", None)
                if signature:
                    stored["signature"] = signature
                thinking_blocks.append(stored)
            elif btype == "redacted_thinking":
                thinking_blocks.append({"type": "redacted_thinking", "data": getattr(block, "data", "")})
        self._last_tool_uses = tool_uses
        self._last_thinking_blocks = thinking_blocks
        return "".join(text_parts), response
        # <<< GraphPC D3

    # ─────────────────────────────────────────────────────────────────────
    # predict (canonical OSWorld agent entrypoint)
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, instruction: str, obs: Dict) -> List:
        """Predict the next action.

        Returns ``[response_text, pyautogui_code]`` — same shape as every
        other OSWorld agent. ``pyautogui_code`` is a list of executable
        strings; special tokens ``WAIT`` / ``DONE`` / ``FAIL`` are kept as
        standalone elements. ``CALL_USER`` is accepted from the parser and
        converted to an empty action list for ``lib_run_single``.
        """
        screenshot_bytes = obs["screenshot"]
        image = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = image.size
        logger.info(f"Original screen resolution: {original_width}x{original_height}")

        # Save the very first observation under ``api_logs/..``'s sibling
        # directory as ``step_0_initial.png`` — used by trajectory viewers
        # that expect both initial obs and step screenshots present.
        if self._api_call_count == 0 and self._api_log_dir:
            try:
                screenshot_dir = os.path.dirname(self._api_log_dir)
                initial_path = os.path.join(screenshot_dir, "step_0_initial.png")
                if not os.path.exists(initial_path):
                    with open(initial_path, "wb") as f:
                        f.write(screenshot_bytes)
                    logger.info(f"Saved initial screenshot: {initial_path}")
            except Exception:
                logger.exception("Failed to save step_0_initial.png")

        processed_b64, _ = _encode_screenshot(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_b64)))
        processed_width, processed_height = processed_img.size
        logger.info(
            f"Processed image resolution: {processed_width}x{processed_height} "
            f"(M3 native, no resize)"
        )

        self.screenshots.append(processed_b64)
        if len(self.user_responses) < len(self.responses):
            user_response = (obs.get("user_response") or "").strip()
            self.user_responses.append(
                f"User response: {user_response}" if user_response else None
            )
        messages = self._build_messages(instruction, processed_b64)

        # Build the faithful wire payload up-front so the saved api_log is a
        # true replay of what was sent (same shape Anthropic SDK receives:
        # ``system`` at top level, ``image.source.base64`` blocks, key
        # ``stop_sequences`` not ``stop``). Wrapped in try/except so a
        # logging-only failure can never crash the step — the actual API
        # request is built independently inside ``_call_llm``.
        try:
            request_body = self._build_request_body(messages)
        except Exception:
            logger.exception(
                "Failed to build request_body for logging; api_log entry "
                "will have a stub body but the live API call is unaffected."
            )
            request_body = {"_log_build_error": True}

        # ── LLM call with retry ─────────────────────────────────────────────
        # Resample up to self.max_llm_retries times if:
        #   1. _call_llm raises (network / API error)
        #   2. response parses to 0 actions despite being non-empty
        # On final failure, fall through with the last response_text so history
        # stays well-formed and the env step receives an empty action list
        # (existing no-op behaviour). Each retry's brief metadata is recorded
        # in the saved api_log so post-hoc analysis can spot retry patterns.
        retry_attempts: List[Dict[str, Any]] = []
        response_text = ""
        raw_response: Any = {"error": "no_call_made"}
        low_level_instruction = ""
        pyautogui_code: List[str] = []

        # >>> GraphPC D6: the request time, stamped once before the first attempt (upstream stamped it at save
        # time, so the record carried no latency); response_time stays the save-time stamp, so the two bracket the
        # call, retries included. Diagnostics only: no request byte moves.
        request_started_ms = int(time.time() * 1000)
        # <<< GraphPC D6
        for attempt in range(self.max_llm_retries + 1):
            try:
                response_text, raw_response = self._call_llm(messages)
                logger.info(
                    f"M3 Output (attempt {attempt + 1}/{self.max_llm_retries + 1}): "
                    f"{(response_text or '')[:600]}"
                )
            except Exception as e:
                logger.exception(f"LLM call failed (attempt {attempt + 1}): {e}")
                retry_attempts.append({
                    "attempt": attempt + 1,
                    "outcome": "exception",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}",
                })
                response_text = ""
                raw_response = {"error": str(e)}
                # >>> GraphPC D3: a failed call leaves no tool_use / thinking block behind, and nothing of an
                # earlier attempt or turn is stored in its place
                self._last_tool_uses = []
                self._last_thinking_blocks = []
                self._executed_tool_use = None
                self._second_tool_use = []
                # <<< GraphPC D3
                if attempt < self.max_llm_retries:
                    continue
                break

            # Parse to actions; if 0 actions yet response was non-empty → retry
            # >>> GraphPC D3: the action from the tool_use block (text-grammar parsing has no referent here)
#|            low_level_instruction, pyautogui_code = parse_m3_response(
#|                response_text,
#|                original_width=original_width,
#|                original_height=original_height,
#|                processed_width=processed_width,
#|                processed_height=processed_height,
#|                coordinate_type=self.coordinate_type,
#|            )
            low_level_instruction, pyautogui_code = self._parse_native_reply(
                response_text, original_width, original_height, processed_width, processed_height,
            )
            # <<< GraphPC D3
            if not pyautogui_code and response_text.strip():
                logger.warning(
                    f"[parse-empty] attempt {attempt + 1} parsed 0 actions from "
                    f"{len(response_text)}-char response; retrying"
                    if attempt < self.max_llm_retries
                    else f"[parse-empty] attempt {attempt + 1} parsed 0 actions from "
                         f"{len(response_text)}-char response; out of retries, no-op"
                )
                retry_attempts.append({
                    "attempt": attempt + 1,
                    "outcome": "parse_empty",
                    "reason": f"response_len={len(response_text)} but 0 actions parsed",
                })
                if attempt < self.max_llm_retries:
                    continue

            # accepted
            break

        if pyautogui_code == ["CALL_USER"]:
            pyautogui_code = []

        # >>> GraphPC D6: the stamp goes with the record
#|        self._save_api_log(request_body, raw_response, retry_attempts=retry_attempts or None)
        self._save_api_log(request_body, raw_response, retry_attempts=retry_attempts or None,
                           request_time=request_started_ms)
        # <<< GraphPC D6

        # >>> GraphPC D4: the turn is stored as its blocks (the protocol re-sends blocks, not wrapped text)
#|        wrapped = wrap_for_history(response_text)
#|        self.responses.append(wrapped)
        self.responses.append(response_text)
        self.turn_blocks.append(self._stored_turn(response_text))
        # <<< GraphPC D4

        self.actions.append(low_level_instruction or "(no-op)")
        if low_level_instruction:
            self.thoughts.append(low_level_instruction)

        return [response_text, pyautogui_code]

    # ─────────────────────────────────────────────────────────────────────
    # api_log persistence (multi_turn.json prerequisite)
    # ─────────────────────────────────────────────────────────────────────

    # >>> GraphPC D6: request_time, when given, is the stamp predict took before the call
#|    def _save_api_log(self, request_body: dict, response: Any,
#|                      retry_attempts: Optional[List[Dict[str, Any]]] = None):
    def _save_api_log(self, request_body: dict, response: Any,
                      retry_attempts: Optional[List[Dict[str, Any]]] = None,
                      request_time: Optional[int] = None):
    # <<< GraphPC D6
        if not self._api_log_dir:
            return
        ts = int(time.time() * 1000)
        if hasattr(response, "model_dump"):
            response_body = response.model_dump()
        elif isinstance(response, str):
            response_body = {"choices": [{"message": {"role": "assistant", "content": response}}]}
        elif isinstance(response, dict):
            response_body = response
        else:
            response_body = {"raw": str(response)}
        entry = {
            "model": self.model,
            "request_body": copy.deepcopy(request_body),
            "response_body": response_body,
            # >>> GraphPC D6: the pre-call stamp (the save-time stamp when predict did not pass one)
#|            "request_time": ts,
            "request_time": request_time if request_time is not None else ts,
            # <<< GraphPC D6
            "response_time": int(time.time() * 1000),
        }
        # >>> GraphPC D3: every tool_use block beyond the executed one is recorded here, never executed or stored
        if self._second_tool_use:
            entry["_second_tool_use"] = copy.deepcopy(self._second_tool_use)
        # <<< GraphPC D3
        if retry_attempts:
            # Compact metadata about parse-fail / network retries that
            # happened on this step. Useful for offline analysis to spot
            # tasks where M3 needed multiple resamples to produce clean output.
            entry["_retry_attempts"] = retry_attempts
        self._api_call_count += 1
        path = os.path.join(self._api_log_dir, f"{self._api_call_count:03d}_{ts}.json")
        try:
            with open(path, "w") as f:
                json.dump(entry, f, ensure_ascii=False)
            logger.info(f"Saved API log: {path}")
        except Exception:
            logger.exception(f"Failed to save API log: {path}")
