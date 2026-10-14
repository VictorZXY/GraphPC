"""Single source of truth for extracting a done action's answer from its WebArena
``stop [answer]`` command.

Both the completion gate (``GraphPCOrchestrator._done_answer``, which routes an
answer-bearing done through M2 rule-1 answer-grounding) AND the WebArena action
translator (which builds ``create_stop_action(answer)`` for submission) MUST use
this ONE parser, so they can never disagree on what the answer is — a
translator that read a never-set ``parameters['answer']`` would submit ``""``
while the gate read the command.

The ``stop [answer]`` grammar is WebArena's; a done with no such command (dry-run
stub / OSWorld-style state done) yields ``""`` here — callers handle their own
no-command fallback (e.g. the translator falls back to ``parameters['answer']``).
"""

from __future__ import annotations

import re
from typing import Any

# ``.*`` + DOTALL so an answer with brackets/newlines/punctuation is captured
# verbatim between the first ``stop [`` and the final ``]`` (greedy to the last
# bracket). The completion gate and the translator share this exact regex.
STOP_ANSWER_RE = re.compile(r"^\s*stop\s*\[(?P<answer>.*)\]\s*$", re.IGNORECASE | re.DOTALL)


def stop_command(action: Any) -> str:
    """The WebArena ``stop [...]`` command string carried on a done action, if any
    — from ``parameters['command']`` (where ``build_action_batch`` puts it), else
    ``raw['input']['command']``. ``""`` when neither is present."""
    command = ""
    parameters = getattr(action, "parameters", None)
    if isinstance(parameters, dict):
        command = str(parameters.get("command") or "")
    if not command:
        raw = getattr(action, "raw", None)
        if isinstance(raw, dict):
            command = str((raw.get("input") or {}).get("command") or "")
    return command


def done_answer(action: Any) -> str:
    """Extract the answer from a done action's ``stop [answer]`` command. ``""``
    when there is no such command or it does not match (an empty ``stop []`` also
    yields ``""``)."""
    match = STOP_ANSWER_RE.match(stop_command(action))
    return match.group("answer").strip() if match else ""
