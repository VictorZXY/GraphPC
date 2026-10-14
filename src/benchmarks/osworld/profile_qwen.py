"""The Qwen (qwen3.7-plus over OpenRouter) model profile, against its control,
``osworld-v2.1:mm_agents/qwen`` (its code copied under ``qwen_upstream/``), and the
profile's pieces that are functions of that control: the timeline's image retention and
the environment brief. A module of its own, imported only when the Qwen profile is
built, so the MiniMax build imports nothing of the Qwen control.
"""

from __future__ import annotations

from typing import Any

from .action_qwen import QWEN_ACTION_CLASS_VOCABULARY, QwenActionAgent, translate_qwen_action
from .backbone_qwen import OSWorldQwenAgent, is_infra_error
from .graphpc_agent import DONE_TOKEN, OSWorldModelProfile
from .prompts import QWEN_DECIDING_INSTRUCTION_SUFFIX, qwen_deciding_repair
from .qwen_upstream.history import COLLAPSED_SCREENSHOT_TEXT, folded_prefix
from .qwen_upstream.prompts import TERMINAL_BLOCKED_TEXT, _prompt_date
from .timeline import ImageRetention

# The Qwen control's folding (osworld-v2.1:mm_agents/qwen/history.py, runner defaults
# --image_max 20 --fold_size 10): of the T frames an episode has produced (the current
# one included), the first f = folded_prefix(T) are replaced by COLLAPSED_SCREENSHOT_TEXT
# and the rest are sent, so the live count sawtooths 1 -> 20, then 11 -> 20; the opening
# frame folds like any other. The control advances f one frame at a time; computed from
# 0 at every call it is the same number, the smallest multiple of the fold with T - f <=
# image_max. In the timeline, step s's after-frame is frame s + 1.
QWEN_IMAGE_MAX = 20
QWEN_FOLD_SIZE = 10


def _qwen_omitted(steps_executed: int) -> tuple[int, bool]:
    folded = folded_prefix(steps_executed + 1, 0, QWEN_IMAGE_MAX, QWEN_FOLD_SIZE)
    return max(0, folded - 1), folded == 0


QWEN_IMAGE_RETENTION = ImageRetention(
    omitted=_qwen_omitted,
    marker=f" [{COLLAPSED_SCREENSHOT_TEXT}]",
    config=(("image_max", QWEN_IMAGE_MAX), ("fold_size", QWEN_FOLD_SIZE), ("opening_frame_kept", False)),
)


def qwen_environment_brief(*, client_password: str, task_current_date: Any = None) -> str:
    """The Qwen control's one environment fact — its rule "Current date: {date}."
    (qwen_upstream/prompts.py ``build_system_prompt``), the date rendered by its own
    ``_prompt_date`` — that model profile's brief. The control never states the sudo
    password, so it is not used."""
    del client_password
    return f"Current date: {_prompt_date(task_current_date)}."


# The profile's own numbers, each the Qwen control's (its agent and its runner
# run_multienv_qwen.py, defaults). Attempts: its tool_repair_retries 2 + 1. The output
# cap: its --max_tokens, on both calls. The request timeout: the rule max(500, 1.5 x the
# cap / r), with r = 57.01 tokens per second, gives 862 s, rounded up to 900 s for
# margin against slower replies. Either way the output cap binds before the timeout
# does (backbone_qwen.py). r is the slowest output / client wall time measured over
# non-streamed json_object calls in the deciding call's regime: 20 retained frames, a
# prompt of about 41k tokens, reasoning on.
QWEN_CALL_ATTEMPTS = 3
QWEN_MAX_TOKENS = 32768
QWEN_REQUEST_TIMEOUT_S = 900.0


def qwen_model_profile() -> OSWorldModelProfile:
    """The Qwen profile, built at call time like MiniMax's."""
    return OSWorldModelProfile(
        backbone_class=OSWorldQwenAgent,
        action_agent_class=QwenActionAgent,
        translate_action=translate_qwen_action,
        action_classes=QWEN_ACTION_CLASS_VOCABULARY,
        deciding_suffix=QWEN_DECIDING_INSTRUCTION_SUFFIX,
        environment_brief=qwen_environment_brief,
        call_attempts=QWEN_CALL_ATTEMPTS,
        grounding_temperature=OSWorldQwenAgent.TEMPERATURE,
        # The control's `type` is pyautogui.typewrite(chunk), which presses each character
        # with no pause (pyautogui applies its 0.1 s PAUSE once per call), so the
        # long-typing hazard MiniMax's note describes does not arise: no note.
        typing_note="",
        image_retention=QWEN_IMAGE_RETENTION,
        # The control never emits FAIL: its judge turns every terminal candidate into DONE
        # or a continue.
        give_up_token=DONE_TOKEN,
        max_tokens=QWEN_MAX_TOKENS,
        deciding_repair=qwen_deciding_repair,
        request_timeout=QWEN_REQUEST_TIMEOUT_S,
        # The control judge's BLOCKED text, verbatim.
        give_up_block_text=TERMINAL_BLOCKED_TEXT,
        ask_user_channel=True,
        infra_error=is_infra_error,
        # The control raises on a failed capture and its cause table re-runs the task:
        # the same here, never a hole.
        capture_failure_raises=True,
    )
