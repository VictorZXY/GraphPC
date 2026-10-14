"""Action-stage base contract. The concrete grounding agents are
benchmark-side (``src/benchmarks/webarena/action_agent.py``; on OSWorld 2.0 one
per model profile: ``src/benchmarks/osworld/action_minimax.py``,
``action_qwen.py``, ``action_deepseek.py``): grounding is backbone- and
benchmark-specific, so the method layer holds only the interface."""

from .base import ActionAgent, ActionInput

__all__ = [
    "ActionAgent",
    "ActionInput",
]
