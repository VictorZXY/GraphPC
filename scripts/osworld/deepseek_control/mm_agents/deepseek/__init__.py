# GraphPC vendored copy of OSWorld-V2 mm_agents/m3/__init__.py
# at tag v2026.08.08 (commit d578d2d4e0dc82b43e270fdaa7fa89d9708cd154), blob f0b9665ec8253044b0131ccb75448fce224278c4.
# The OSWorld DeepSeek cell's CONTROL is upstream's own M3 agent: this file is byte-identical to
# that blob except this header and the deltas listed below. Each delta is marked in place
# (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`) and carries the upstream lines it
# replaces as `#|` lines, so the file reverses mechanically to upstream's; the DeepSeek mode
# of scripts/osworld/verify_runner_deltas.py holds it to that. The delta list with its
# reasons is in the header of scripts/osworld/run_multienv_deepseek_baseline.py.
# Deltas carried: none.
# end of GraphPC header
"""Standalone M3 agent for OSWorld evaluation.

Provides the system prompt, ``<tool_call>`` response parser, screenshot
handling (1920x1080 native, [0, 1000] normalised coordinates), and an
Anthropic-Messages-API transport.
"""

from .agent import M3Agent
from .prompts import M3_SYSTEM_PROMPT_TEMPLATE, MACOS_KEYBOARD_MAPPING
from .parser import parse_m3_response, tool_action_to_pyautogui

__all__ = [
    "M3Agent",
    "M3_SYSTEM_PROMPT_TEMPLATE",
    "MACOS_KEYBOARD_MAPPING",
    "parse_m3_response",
    "tool_action_to_pyautogui",
]
