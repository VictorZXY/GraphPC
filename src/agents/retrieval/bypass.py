"""Retrieval-bypass agent — the retrieval-off operational mode.

``GraphPCOrchestrator.__init__`` requires a retrieval agent, and the workflow
calls it every step, so "retrieval off" is expressed as a retrieval agent that
performs no store read rather than as a branch inside the orchestrator. The
store-WRITE half lives at the benchmark adapter (it owns persistence), so the
two halves together give the run a store it never touches in either direction.

The bypass state rides on ``RetrievalResult.metadata`` — never on ``content``,
which is serialized into the deciding prompt — so the model sees only the
neutral "no stored experience" surface, while telemetry can name the mode.

**An operational mode, not a diagnostic switch.** On a task set of unique
intent templates (e.g. WebArena-Lite's 165 tasks) no stored entry shares a
procedure with the query, so retrieval has nothing to transfer while its
injected tokens still cost; there the scaffolding, not the retrieved
experience, does the work. Family-rich sets (full WebArena, whose 812 tasks
fall into intent-template families) keep retrieval ON. Selected via
``--retrieval_off`` on the WebArena runner and on the OSWorld runner, which
refuses a launch without it.
"""

from __future__ import annotations

import logging

from .base import RetrievalAgent, RetrievalInput
from ..common.schema import RetrievalResult

logger = logging.getLogger("desktopenv.agent")


class BypassRetrievalAgent(RetrievalAgent):
    """Returns an empty result without opening the support-set store."""

    def __init__(self) -> None:
        logger.info("retrieval OFF (--retrieval_off): no store reads this run")

    def retrieve(self, retrieval_input: RetrievalInput) -> RetrievalResult:
        return RetrievalResult(
            scope=retrieval_input.retrieval_scope,
            items=[],
            content={
                "retrieval_text": "No stored experience is available.",
                "candidate_count": 0,
                "selected_entry_ids": [],
            },
            metadata={"retrieval_bypassed": True},
        )
