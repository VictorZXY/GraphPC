from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..common.schema import Observation, PlanState, RetrievalResult, RuntimeState


@dataclass
class RetrievalInput:
    """
    retrieval_scope examples:
      - "task"
      - "stage"
      - "recovery"
      - "custom"
    """

    retrieval_scope: str
    instruction: str
    observation: Observation
    runtime_state: RuntimeState

    plan: Optional[PlanState] = None
    # Two injected entries at most. The retrieval agent's ``candidate_limit`` is
    # the authority and caps whatever arrives here, so a two-pass model asking
    # for more cannot widen the injection; this default keeps the single-pass path
    # (which passes no top_k) honest about what it actually gets.
    top_k: int = 2
    # Optional rewritten retrieval query distinct from the raw instruction
    # (demand-driven retrieval). When None, retrieval falls back to the
    # instruction. Additive on the (non-frozen) retrieval input envelope.
    query: Optional[str] = None

    content: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalAgent(ABC):
    @abstractmethod
    def retrieve(self, retrieval_input: RetrievalInput) -> RetrievalResult:
        """
        Return RetrievalResult.
        """
        raise NotImplementedError
