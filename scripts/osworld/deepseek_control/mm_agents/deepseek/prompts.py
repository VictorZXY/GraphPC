# GraphPC vendored copy of OSWorld-V2 mm_agents/m3/prompts.py
# at tag v2026.08.08 (commit d578d2d4e0dc82b43e270fdaa7fa89d9708cd154), blob 3fc13d41c30b87bdf8b879d52345c8a160f3aba5.
# The OSWorld DeepSeek cell's CONTROL is upstream's own M3 agent: this file is byte-identical to
# that blob except this header and the deltas listed below. Each delta is marked in place
# (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`) and carries the upstream lines it
# replaces as `#|` lines, so the file reverses mechanically to upstream's; the DeepSeek mode
# of scripts/osworld/verify_runner_deltas.py holds it to that. The delta list with its
# reasons is in the header of scripts/osworld/run_multienv_deepseek_baseline.py.
# Deltas carried: D2 (the computer tool declared natively: NATIVE_TOOLS_BLOCK, COMPUTER_TOOL, the template rebound).
# end of GraphPC header
"""M3 system prompt.

The sudo password is templated as ``{client_password}`` so callers can
override it via the runner's ``--client_password`` flag without having
to edit the prompt body.

The chat template auto-injects ``<thinking_instructions>`` so the
developer block here only needs to contain SYSTEM_CAPABILITY + TOOLS.

Coordinate space: ``[0, 1000]`` normalised integers — matches the
convention the model was trained on.

Format ``M3_SYSTEM_PROMPT_TEMPLATE`` with both ``date_str`` and
``client_password`` before passing to the model:

    prompt = M3_SYSTEM_PROMPT_TEMPLATE.format(
        date_str="Friday, May 29, 2026",
        client_password="osworld-public-evaluation",
    )

For macOS targets, append ``MACOS_KEYBOARD_MAPPING`` after the formatted
template.
"""

from __future__ import annotations


M3_SYSTEM_PROMPT_TEMPLATE = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using available tools.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* TASK FEASIBILITY — this is a frequent failure point, read carefully. Some tasks are intentionally impossible. You may declare a task infeasible at any point: immediately after the first screenshot, or later after attempting actions and hitting a hard barrier. A task is infeasible when it cannot be completed due to:
  - Missing required applications or dependencies that cannot be installed
  - Insufficient permissions or system limitations
  - Contradictory, fictional, or impossible requirements
  - The task requires a specific application, but you have verified that application does not provide the required feature or capability
  - Any other fundamental barrier that makes completion impossible
  When you conclude a task is infeasible, you MUST output the literal token "[INFEASIBLE]" (with the square brackets) in that same turn. This exact token is the ONLY signal the system recognizes for an impossible task.
  CRITICAL — when a task is infeasible, do NOT do any of the following instead:
  - Do NOT emit the `done` action. `done` means "I have successfully completed the task" and will be graded as WRONG for an impossible task.
  - Do NOT only explain in prose that it is "not possible" / "does not exist" / "cannot be done". Prose refusals are NOT detected — only the literal "[INFEASIBLE]" token counts.
  - Do NOT propose workarounds or alternatives, and do NOT ask the user what they prefer. Decide yourself, output "[INFEASIBLE]", and stop.
  Only declare a task infeasible when you are genuinely confident it is impossible. Do NOT give up on a task that is merely difficult, slow, or unfamiliar — try reasonable approaches first.
* The current date is {date_str}.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is '{client_password}'.
* All `coordinate` values in `computer` tool calls are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen, regardless of the underlying screen resolution.
</SYSTEM_CAPABILITY>

<TOOLS>
You have access to the `computer` tool for interacting with the desktop GUI. Output one tool call per turn.

For each tool call, return a json object with name and arguments inside <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "computer", "arguments": {{"action": "<action>", ...}}}}
</tool_call>

Supported `action` values include:
* `key`        — args: {{"text": "<key combo, e.g. ctrl+s>"}}
* `type`       — args: {{"text": "<text to type>"}}
* `mouse_move` — args: {{"coordinate": [x, y]}}
* `left_click` / `right_click` / `middle_click` / `double_click` / `triple_click`
                 — args: {{"coordinate": [x, y]}}, optional {{"text": "<modifier keys>"}}
* `left_click_drag` — args: {{"coordinate": [x, y]}}, optional {{"start_coordinate": [x, y]}}
* `scroll`     — args: {{"coordinate": [x, y], "scroll_direction": "up|down|left|right", "scroll_amount": <int>}}
* `wait`       — args: {{"duration": <seconds>}}
* `screenshot` — args: {{}}
* `hold_key`   — args: {{"text": "<keys>", "duration": <seconds>}}
* `left_mouse_down` / `left_mouse_up` — args: {{"coordinate": [x, y]}}
* `done`       — args: {{}}: declare the task has been completed successfully. Output this as the final tool call after you are confident the task is fully done.

Rules:
- Output exactly one short imperative `Action: <text>` line followed by exactly one <tool_call>...</tool_call> block.
- Coordinates are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen.
- When the task has been completed and there is nothing more to do, your FINAL turn must end with a <tool_call> using action `done` to signal completion.
- For impossible/infeasible tasks, your response MUST contain the literal `[INFEASIBLE]` token — not the `done` action, and not a prose-only explanation.
</TOOLS>"""


# macOS-suffix block. M3Agent appends this to the formatted prompt when
# ``platform="macos"``.
MACOS_KEYBOARD_MAPPING = """
IMPORTANT: You are operating on macOS. Key differences from Linux:
- Use 'command' key instead of 'ctrl' for shortcuts (e.g., command+c to copy, command+v to paste)
- Use 'option' key instead of 'alt'
- Close window: command+w; Quit app: command+q
- Spotlight search: command+space
- The menu bar is always at the top of the screen
- The Dock is at the bottom of the screen
- Use pyautogui.hotkey('command', 'c') not pyautogui.hotkey('ctrl', 'c')
"""
# >>> GraphPC D2: the computer tool declared natively. The model (DeepSeek-V4.1-Flash) does not write M3's
# <tool_call> text grammar (its replies carry its own native tool markup instead), so the request
# declares the `computer` tool as a native tool and the TOOLS text block of the
# template above is replaced by NATIVE_TOOLS_BLOCK: the same words minus the text-format rules, which have
# no referent under native tool use. The <SYSTEM_CAPABILITY> block, its coordinate bullet included, is
# untouched. The action list and the argument shapes move, in the same words, into the tool definition
# (COMPUTER_TOOL_DESCRIPTION / COMPUTER_TOOL_SCHEMA), the native place for them. The GraphPC arm's
# grounding call carries the identical tool definition and NATIVE_TOOLS_BLOCK minus its two termination
# rules.

NATIVE_TOOLS_BLOCK = """<TOOLS>
You have access to the `computer` tool for interacting with the desktop GUI, provided as a function. Call it exactly once per turn.

Rules:
- When the task has been completed and there is nothing more to do, your FINAL turn must call the `computer` tool with action `done` to signal completion.
- For impossible/infeasible tasks, your response MUST contain the literal `[INFEASIBLE]` token — not the `done` action, and not a prose-only explanation.
</TOOLS>"""

# The 16 actions the M3 prompt lists, `done` included; NOT the parser's wider set (fail / call_user /
# terminate ...), so infeasibility reaches the parser only through the [INFEASIBLE] text token.
COMPUTER_TOOL_ACTIONS = (
    "key", "type", "mouse_move", "left_click", "right_click", "middle_click", "double_click", "triple_click",
    "left_click_drag", "scroll", "wait", "screenshot", "hold_key", "left_mouse_down", "left_mouse_up", "done",
)

COMPUTER_TOOL_DESCRIPTION = (
    "Interact with the desktop GUI. Supported `action` values: `key` (text: key combo, e.g. ctrl+s); `type` (text: "
    "text to type); `mouse_move` (coordinate); `left_click` / `right_click` / `middle_click` / `double_click` / "
    "`triple_click` (coordinate, optional text: modifier keys); `left_click_drag` (coordinate, optional "
    "start_coordinate); `scroll` (coordinate, scroll_direction, scroll_amount); `wait` (duration); `screenshot`; "
    "`hold_key` (text, duration); `left_mouse_down` / `left_mouse_up` (coordinate); `done`: declare the task has "
    "been completed successfully."
)

# The field description of both coordinate arguments: the coordinate bullet condensed, byte-identical to the
# GraphPC arm's grounding tool's coordinate field, so both arms describe the coordinates in the same words.
COMPUTER_TOOL_COORDINATE_DESCRIPTION = (
    "[x, y], normalized integers in [0, 1000]; (0, 0) top-left, (1000, 1000) bottom-right of the screen."
)

_COORDINATE_FIELD = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
    "description": COMPUTER_TOOL_COORDINATE_DESCRIPTION,
}

COMPUTER_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(COMPUTER_TOOL_ACTIONS)},
        "coordinate": dict(_COORDINATE_FIELD),
        "start_coordinate": dict(_COORDINATE_FIELD),
        "text": {"type": "string", "description": "Key combo (key/hold_key), text to type (type), or modifier keys (clicks)."},
        "scroll_direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
        "scroll_amount": {"type": "integer"},
        "duration": {"type": "number"},
    },
    "required": ["action"],
}

# The tool as the Anthropic-format request carries it (agent.py:_build_request_body).
COMPUTER_TOOL = {"name": "computer", "description": COMPUTER_TOOL_DESCRIPTION, "input_schema": COMPUTER_TOOL_SCHEMA}

# The template the agent formats: upstream's <SYSTEM_CAPABILITY> block (byte-identical, the coordinate bullet
# included) followed by the native TOOLS block. The upstream literal above is kept whole so the file reverses
# to its blob; this rebinding is the delta.
M3_SYSTEM_PROMPT_TEMPLATE = M3_SYSTEM_PROMPT_TEMPLATE.split("<TOOLS>")[0] + NATIVE_TOOLS_BLOCK
# <<< GraphPC D2
