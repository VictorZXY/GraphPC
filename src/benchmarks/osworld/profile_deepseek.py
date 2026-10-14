"""The DeepSeek (``deepseek-flash``, V4.1-Flash) model profile, against its control —
upstream M3 with the native-tool deltas (``scripts/osworld/deepseek_control/``)
— and the profile's values that are functions of that control.
A module of its own, imported only when the DeepSeek profile is built, so the MiniMax and
Qwen builds import nothing of it.

Every control-matched piece is the MiniMax profile's, because the control is M3's agent
with its tool declaration made native: the deciding suffix and the environment brief
(matched line by line to M3's system prompt), the M3 action vocabulary and translator (the
profile's ``translate_deepseek_action`` is M3's translator with the guest's program checked
before it is handed over: action_deepseek.py, "THE GUEST'S PROGRAM IS CHECKED"), the
typing note (the same translator types the same way), M3's image folding, ``FAIL`` as the
give-up token (M3 emits FAIL on ``[INFEASIBLE]``), 3 attempts per call (M3's
``M3_MAX_LLM_RETRIES`` 2 + 1), the grounding temperature 0.6 (the M3 runner's non-thinking
default, honoured by DeepSeek in non-thinking mode), no output cap of the profile's own
(the runner's value; none = DeepSeek's 64K default, parity with the control's 65536), an
unusable deciding reply re-sampled identically (thinking ignores temperature, so a
re-sample is a fresh draw; M3 re-samples identically), no ask channel (the control's tool
schema excludes ``call_user``, as M3's prompt does).

The profile's own numbers: the request timeout, 500 s — the Qwen profile's rule
``max(500, 1.5 x output cap / r)`` with r ≈ 200 tokens per second measured over
20-frame deciding calls, where the cap term is 492 s; the transport's 15 HTTP
attempts per call (``backbone_deepseek._DeepSeekOSWorldTransport.MAX_ATTEMPTS``: the control's 4 SDK
retries inside each of M3's 3 attempts); and the lost-episode
classifier and the capture rule of the Qwen profile (a provider fault after this backbone's
retries re-runs the task; a failed capture raises, as the M3 control raises on a None
screenshot). The class-family rule is the action agent's (``action_deepseek.py``), recorded
through its ``method_surface``.

Two GraphPC method fields are on the DeepSeek profile only: ``requirement_ledger`` (the deciding
reply keeps a ledger of the task's requirements, and the completion gate's requirements_unverified rule refuses a
success claim the ledger contradicts) and ``step_budget_line`` (the STATE block states the true budget at every step).
Neither is a control clause; both are off on the MiniMax and Qwen profiles.
"""

from __future__ import annotations

from .action_deepseek import DeepSeekActionAgent, translate_deepseek_action
from .action_minimax import M3_ACTION_CLASS_VOCABULARY
from .backbone_deepseek import OSWorldDeepSeekAgent, is_infra_error
from .graphpc_agent import FAIL_TOKEN, GROUNDING_TEMPERATURE, MODEL_CALL_ATTEMPTS, OSWorldModelProfile
from .observation_adapter import environment_brief
from .prompts import OSWORLD_DECIDING_INSTRUCTION_SUFFIX, OSWORLD_STEP_BUDGET_LINE, OSWORLD_TYPING_NOTE
from .timeline import M3_IMAGE_RETENTION

DEEPSEEK_REQUEST_TIMEOUT_S = 500.0


def deepseek_model_profile() -> OSWorldModelProfile:
    """The DeepSeek profile, built at call time like MiniMax's."""
    return OSWorldModelProfile(
        backbone_class=OSWorldDeepSeekAgent,
        action_agent_class=DeepSeekActionAgent,
        translate_action=translate_deepseek_action,
        action_classes=M3_ACTION_CLASS_VOCABULARY,
        deciding_suffix=OSWORLD_DECIDING_INSTRUCTION_SUFFIX,
        environment_brief=environment_brief,
        call_attempts=MODEL_CALL_ATTEMPTS,
        grounding_temperature=GROUNDING_TEMPERATURE,
        typing_note=OSWORLD_TYPING_NOTE,
        image_retention=M3_IMAGE_RETENTION,
        give_up_token=FAIL_TOKEN,
        max_tokens=None,
        deciding_repair=None,
        request_timeout=DEEPSEEK_REQUEST_TIMEOUT_S,
        give_up_block_text=None,
        ask_user_channel=False,
        infra_error=is_infra_error,
        # The two GraphPC method fields of this profile alone: the requirement ledger and its
        # completion-gate rule, and the step budget stated at every step (prompts.py, the DeepSeek section).
        requirement_ledger=True,
        step_budget_line=OSWORLD_STEP_BUDGET_LINE,
        capture_failure_raises=True
    )
