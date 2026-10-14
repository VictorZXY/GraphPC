"""Deterministic retrieval agent (design.md Retrieval).

Retrieval adds **no LLM call**: ranking is deterministic. It is
backbone-agnostic (one class serves every backbone): it hands the query text to
the support store's deterministic text-arm ranker (local sentence-embedding
similarity + relevance gate) and wraps the gated, ranked entries in a
``RetrievalResult``.

The retrieved experience is consumed by the graph-reasoning call (experience
into reasoning, Workflow) — it reaches the action stage only indirectly. Telemetry is
resolution-level (design.md §7): the result records which entries passed the gate and
their scores, not a firing count.

**Injection diet.** What is RANKED and what is INJECTED are deliberately
different: the store persists a full trajectory record, but only the parts a
model can act on are put in front of it. Everything an exemplar carries that is
diary rather than recipe — the observation snapshot it was captured under, its
retrieval keys, its graph artifacts, and the action string it records twice — is
dropped from the injected copy here. Store persistence is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .base import RetrievalAgent, RetrievalInput
from .embedding import Embedder, TextArmUnavailableError
from .support_set import LocalSupportSetStore
from ..common.schema import RetrievalResult


# Entry-level fields dropped before injection (the injection diet):
#   graph_artifacts — the exemplar's graph record; retrieval injects the
#                     labelled recipe, not the graph (design.md §5)
#   observation     — the a11y snapshot the win was captured under; stale
#                     against the live page by construction
#   signatures      — a structural retrieval key some store entries carry;
#                     ranking is text-only, so nothing reads it
_INJECTION_DROPPED_FIELDS = ("graph_artifacts", "observation", "signatures")

# Recipe-log dedup: each step's action string is recorded twice — under
# ``parameters.command`` and again under ``metadata.webarena_action_string``.
# ``parameters`` is the schema's canonical slot for an
# action's executable payload and is backbone-neutral, so the WebArena-only
# mirror in ``metadata`` is the copy that goes.
_DUPLICATED_ACTION_FIELD = "webarena_action_string"

# Per-step bookkeeping keys on a history item: the orchestrator's own loop
# accounting, not part of the procedure a reader would follow. Dropped only when
# a benchmark asks (see InjectedHistoryDiet).
_BOOKKEEPING_HISTORY_KEYS = ("step_idx", "status", "monitor", "retrieval")


@dataclass(frozen=True)
class InjectedHistoryDiet:
    """WITHIN-step compression of an injected exemplar, per benchmark.

    The entry-level diet above is unconditional; this is the same
    principle — recipe in, diary out — applied one level down, inside the
    history items, where the volume actually is: ``recent_history`` is nearly
    all of an injected exemplar. Three components of it are not recipe:

      * ``drop_action_raw_response`` — each action's
        ``metadata.raw_response``, the verbatim model reply, when the PARSED
        action it produced already sits beside it in ``parameters``;
      * ``drop_reasoning_text`` — the step's narration;
      * ``drop_bookkeeping_keys`` — ``step_idx`` / ``status`` /
        ``monitor`` / ``retrieval`` on each item.

    Together they are most of the exemplar, and EVERY STEP is preserved — this
    shortens what each step says, it does not drop steps.

    All three default to False, the shared default, so a benchmark that passes
    no diet gets the entry-level strip alone. They are separate
    switches rather than one because ``reasoning_text`` is the row that
    sometimes carries a transferable observation about the environment (as
    opposed to per-step narration), so it must be droppable — or keepable —
    independently of the other two.
    """

    drop_action_raw_response: bool = False
    drop_reasoning_text: bool = False
    drop_bookkeeping_keys: bool = False


def _diet_injected_entry(
    entry: dict[str, Any], *, history_diet: Optional[InjectedHistoryDiet] = None,
) -> None:
    """Strip an injected exemplar down to its recipe, in place.

    ``entry`` is already a deepcopy owned by this call (the store returns copies),
    so this never touches persisted state — the store keeps the full record for
    later analysis, and only the prompt-bound copy is trimmed.

    ``history_diet`` is the benchmark-scoped within-step compression; ``None``
    (the default, and what WebArena passes) leaves every history item as stored,
    apart from the recipe-log dedup above.
    """
    for field in _INJECTION_DROPPED_FIELDS:
        entry.pop(field, None)

    history = entry.get("recent_history")
    if not isinstance(history, list):
        return
    for item in history:
        if not isinstance(item, dict):
            continue
        for action in item.get("actions") or []:
            if not isinstance(action, dict):
                continue
            metadata = action.get("metadata")
            if isinstance(metadata, dict):
                metadata.pop(_DUPLICATED_ACTION_FIELD, None)
                if history_diet is not None and history_diet.drop_action_raw_response:
                    metadata.pop("raw_response", None)
        if history_diet is None:
            continue
        if history_diet.drop_reasoning_text:
            reasoning = item.get("reasoning")
            if isinstance(reasoning, dict):
                reasoning.pop("reasoning_text", None)
        if history_diet.drop_bookkeeping_keys:
            for key in _BOOKKEEPING_HISTORY_KEYS:
                item.pop(key, None)


class DeterministicRetrievalAgent(RetrievalAgent):
    def __init__(
        self,
        *,
        support_set_store: LocalSupportSetStore,
        embedder: Optional[Embedder] = None,
        candidate_limit: int = 2,
        injected_history_diet: Optional[InjectedHistoryDiet] = None,
    ) -> None:
        self.support_set_store = support_set_store
        # Required, not optional: ranking is text-only, so a store searched
        # without an embedder raises rather than silently ranking nothing. The
        # benchmark adapter passes the shared local embedding model.
        #
        # Asserted HERE, at construction, not only at search time: on an EMPTY
        # store ``search`` returns [] before it ever touches the embedder, so a
        # retrieval-on run with a dead text arm looks healthy until the first
        # entry persists and then aborts every later task. Raising here fails
        # the run before the first model call — on WebArena before the browser;
        # on OSWorld the first VM is already provisioned (the V2 runner
        # constructs DesktopEnv before build_default) and the error repeats per
        # task. The OSWorld runner refuses any launch without
        # ``--retrieval_off``, which never constructs this agent — it uses
        # BypassRetrievalAgent with no embedder.
        if embedder is None or not bool(getattr(embedder, "available", False)):
            raise TextArmUnavailableError(
                "retrieval-on run cannot start: the text arm is unavailable "
                + ("(no embedder was configured)" if embedder is None
                   else "(the embedding model is not loaded)")
                + " — text is the only ranking, so every episode would "
                "abort once the store holds an entry. Fix the embedder "
                "(sentence-transformers + the model weights) or run --retrieval_off."
            )
        self.embedder = embedder
        # A HARD CAP on injected entries, not a default. When a same-family
        # sibling exists the text arm ranks it first, so further slots mostly
        # inject other-family distractors at full context cost. A cap rather
        # than a default because the two-pass path lets the MODEL request k,
        # and a model asking for more must not be able to widen the injection.
        self.candidate_limit = candidate_limit
        # Benchmark-scoped within-step compression (design.md §10): None, the
        # shared default, leaves an injected entry to the entry-level strip
        # alone, so a benchmark that does not ask for it (WebArena) is not
        # merely unaffected — it cannot enter the code path. Runs strictly
        # downstream of ranking (see retrieve below), so it cannot change WHICH
        # entries are selected, only what a selected entry says.
        self.injected_history_diet = injected_history_diet

    def retrieve(self, retrieval_input: RetrievalInput) -> RetrievalResult:
        observation_text = self._observation_text(retrieval_input.observation)
        top_k = min(retrieval_input.top_k or self.candidate_limit, self.candidate_limit)
        results = self.support_set_store.search(
            instruction=retrieval_input.instruction,
            observation_text=observation_text,
            query=retrieval_input.query,
            top_k=top_k,
            embedder=self.embedder,
        )
        for result in results:
            _diet_injected_entry(result, history_diet=self.injected_history_diet)

        selected_ids = [str(r.get("entry_id")) for r in results]
        return RetrievalResult(
            scope=retrieval_input.retrieval_scope,
            items=results,
            content={
                "retrieval_text": self._summary(results, retrieval_input.retrieval_scope),
                "candidate_count": self.support_set_store.entry_count(),
                "selected_entry_ids": selected_ids,
                "text_arm_enabled": bool(
                    self.embedder is not None and getattr(self.embedder, "available", False)
                ),
                "query_used": retrieval_input.query or retrieval_input.instruction,
                # resolution-level telemetry: per-entry scores (design.md §7)
                "scores": {
                    str(r.get("entry_id")): r.get("retrieval", {}) for r in results
                },
            },
            # Rides on metadata, NOT content: content is serialized into the
            # deciding prompt (and into recent_history), so a new key there would
            # change the default prompt bytes.
            metadata={
                "deterministic": True,
                "candidate_limit": self.candidate_limit,
                # Which admit rule actually ran (measurement integrity: the
                # calibrated path must be distinguishable from the fixed default
                # in the record, not inferred from the number). On metadata, NOT
                # content — content is serialized into the deciding prompt, so a
                # key there would change prompt bytes on every benchmark.
                **self._relevance_telemetry(),
            },
        )

    def _relevance_telemetry(self) -> dict[str, Any]:
        """The store's resolved admit rule, or nothing when the store does not
        expose one (e.g. a test double). Read defensively because this is
        telemetry: it must never be the reason a retrieval raises."""
        resolved = getattr(self.support_set_store, "resolved_relevance", None)
        if resolved is None:
            return {}
        return resolved.as_telemetry()

    @staticmethod
    def _summary(results: list[dict[str, Any]], scope: str) -> str:
        if not results:
            return f"No support-set entries passed the relevance gate (scope={scope})."
        ids = ", ".join(str(r.get("entry_id")) for r in results)
        return f"Retrieved {len(results)} support entr{'y' if len(results) == 1 else 'ies'} (scope={scope}): {ids}"

    @staticmethod
    def _observation_text(observation) -> str:
        if hasattr(observation, "content") and isinstance(observation.content, dict):
            return str(observation.content.get("observation_text", ""))
        if isinstance(observation, dict):
            return str(observation.get("observation_text", ""))
        return ""
