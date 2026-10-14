# GraphPC vendored copy of OSWorld-V2 mm_agents/qwen/history.py
# at tag osworld-v2.1 (commit 3d778a3c9a34a079316f70df023b166700445792), blob 046515d960581633328596f5ce3b5d6734abacc6.
# The OSWorld Qwen cell's CONTROL is upstream's own code: this file is byte-identical to
# that blob except this header and the deltas listed below. Each delta is marked in place
# (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`) and carries the upstream lines it
# replaces as `#|` lines, so the file reverses mechanically to upstream's; the Qwen mode
# of scripts/osworld/verify_runner_deltas.py holds it to that. The delta list with its
# reasons is in the header of scripts/osworld/run_multienv_qwen_baseline.py.
# Deltas carried: none.
# end of GraphPC header
"""Conversation history construction and screenshot folding."""

from __future__ import annotations

from typing import Any

from .prompts import build_instruction_prompt


COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed to save context."


def folded_prefix(
    total_steps: int, current: int, image_max: int, fold_size: int
) -> int:
    result = current
    while total_steps - result > image_max:
        result += fold_size
    return min(total_steps, result)


def previous_actions_text(actions: list[str], start_step: int) -> str:
    values = [
        f"Step {index + 1}: {actions[index]}"
        for index in range(min(start_step - 1, len(actions)))
    ]
    return "\n".join(values) if values else "None"


def _tool_response(parts: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for _ in range(max(1, count) - 1):
        content.extend(
            [
                {"type": "text", "text": "<tool_response>\n"},
                {"type": "text", "text": "success"},
                {"type": "text", "text": "\n</tool_response>\n"},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": "<tool_response>\n"},
            *parts,
            {"type": "text", "text": "\n</tool_response>"},
        ]
    )
    return content


def build_messages(
    *,
    system_prompt: str,
    instruction: str,
    screenshots: list[str],
    responses: list[str],
    actions: list[str],
    tool_call_counts: list[int],
    user_responses: dict[int, str],
    terminal_recoveries: dict[int, str],
    folded_prefix_count: int,
    history_n: int,
    folded_summary: str = "",
    hard_start_step: int | None = None,
) -> list[dict[str, Any]]:
    total_steps = len(screenshots)
    start_step = hard_start_step or max(1, total_steps - max(1, history_n) + 1)
    previous = previous_actions_text(actions, start_step)
    if folded_summary:
        previous = f"Summary of older progress:\n{folded_summary}\n\nPrevious actions:\n{previous}"
    instruction_prompt = build_instruction_prompt(instruction, previous)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for step in range(start_step, total_steps + 1):
        screenshot = screenshots[step - 1]
        if step <= folded_prefix_count or not screenshot:
            observation_parts = [{"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT}]
        else:
            observation_parts = [
                {"type": "image_url", "image_url": {"url": screenshot}}
            ]

        user_response = user_responses.get(step)
        terminal_recovery = terminal_recoveries.get(step)
        if step == start_step:
            parts = [*observation_parts, {"type": "text", "text": instruction_prompt}]
        elif user_response:
            parts = [
                {
                    "type": "text",
                    "text": f"User response to your previous request:\n{user_response}",
                },
                *observation_parts,
            ]
        elif terminal_recovery:
            parts = [{"type": "text", "text": terminal_recovery}, *observation_parts]
        else:
            prior_count = (
                tool_call_counts[step - 2] if step - 2 < len(tool_call_counts) else 1
            )
            parts = _tool_response(observation_parts, prior_count)
        messages.append({"role": "user", "content": parts})

        if step <= total_steps - 1 and step - 1 < len(responses):
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": responses[step - 1]}],
                }
            )
    return messages
