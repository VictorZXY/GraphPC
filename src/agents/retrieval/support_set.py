from __future__ import annotations

import json
import logging
import os
import re
import uuid

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .embedding import Embedder, TextArmUnavailableError
from .relevance import (
    PercentileRelevanceGate,
    ResolvedRelevanceThreshold,
    resolve_relevance_threshold,
)


logger = logging.getLogger("desktopenv.agent")


# Relevance gate (design.md §5): never inject weakly-matched experience. A candidate
# must clear this on the text arm to be eligible for injection. Conservative
# default.
#
# This FIXED value is the shared default, and WebArena's: a store constructed
# without a `relevance_gate` applies it. Its portability defect (it sits near the
# bottom of the WebArena pairwise distribution, so it approves nearly everything
# once the scale shifts under it) is what `relevance.py`'s percentile rule fixes,
# for the benchmark that opts in.
DEFAULT_RELEVANCE_THRESHOLD = 0.15


def _text_arm_failure_reason(embedder: Optional[Embedder]) -> str:
    """Name the ACTUAL branch that left the text arm unavailable, for the
    ``TextArmUnavailableError`` message. The mid-run case is read off the
    embedder's ``last_failure_kind`` rather than assumed to be OOM — a non-OOM
    encode error reaches the same dead-arm state by a different route, and
    mislabelling it would send the next reader chasing a GPU problem."""
    if embedder is None:
        return "no embedder was configured"
    if not getattr(embedder, "available", False):
        return "the embedding model is not loaded"
    kind = getattr(embedder, "last_failure_kind", None)
    if kind == "cuda_oom_retries_exhausted":
        return "the embedding model failed mid-run (CUDA-OOM retries exhausted)"
    if kind == "encode_error":
        return "the embedding model failed mid-run (encode error, not OOM)"
    return "the embedding model returned no scores mid-run (cause not recorded)"


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, bytes):
        return "<bytes omitted>"
    return value


def _entry_observation_text(entry: dict[str, Any]) -> str:
    observation = entry.get("observation") or {}
    if not isinstance(observation, dict):
        return ""
    for key in ("observation_text", "a11y_tree", "accessibility_tree", "screen_summary"):
        value = observation.get(key)
        if value:
            return str(value)
    return ""


def _keep_best_key(entry: dict[str, Any]) -> tuple:
    """Keep-best sort key; the MAX entry wins its task_id group (design.md §5):
    (1) genuine win; (2) newer created_at.

    There is no structural tiebreaker: the reasoning call emits no node/edge
    topology (``graph_artifacts`` carries only ``graph_summary``), so a key
    read from an entry's task graph would be a constant for every entry and
    decide nothing."""
    return (
        int(bool(entry.get("success"))),
        str(entry.get("created_at") or ""),
    )


class LocalSupportSetStore:
    """
    Local persistent store for successful support-set entries.

    Entries are stored as JSON files under ``<root_dir>/<backend>/``,
    one file per entry. The persisted shape is not a stable contract —
    it reflects what the two benchmark adapters' ``finalize_task`` writes
    (``WebArenaGraphPCAgent``, ``OSWorldGraphPCAgent``).

    Caller-supplied fields (see either adapter's ``finalize_task``):
      - ``task_id``, ``instruction``, ``success``, ``result``
      - ``current_stage``, ``plan`` (serialized ``PlanState``)
      - ``recent_history``, ``observation``, ``metadata``
      - ``graph_artifacts`` (when present): ``{graph_summary}``

    Auto-injected by ``add_success_entry``:
      - ``backend``, ``created_at``, ``entry_id``

    Retrieval (``search``) is deterministic — no LLM call (Retrieval). It ranks by
    the local-embedding text arm alone and applies a relevance gate; there is no
    structural ranking arm (design.md §5).

    Values are JSON-sanitized; raw ``bytes`` are replaced with a placeholder string.
    """

    def __init__(
        self,
        root_dir: str | Path = "artifacts/support_set",
        backend: str = "osworld",
        *,
        relevance_gate: Optional[PercentileRelevanceGate] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.backend = backend
        self.backend_dir = self.root_dir / backend
        # Benchmark-scoped admit rule. `relevance_gate=None` — the shared
        # default, and what WebArena constructs — keeps the fixed threshold;
        # the percentile rule cannot execute without a
        # benchmark handing over its task set. Resolved HERE, once at
        # construction, rather than per search: the reference distribution is an
        # offline property of the corpus, and re-deriving it per query would
        # both cost n*(n-1) encodes a step and let the threshold drift within a
        # run. A gate that cannot be resolved RAISES (see relevance.py) — it
        # never degrades to the fixed default behind a telemetry field claiming
        # otherwise.
        self.resolved_relevance: ResolvedRelevanceThreshold = resolve_relevance_threshold(
            relevance_gate, fixed_default=DEFAULT_RELEVANCE_THRESHOLD, embedder=embedder,
        )
        self.relevance_threshold = self.resolved_relevance.value
        # Held ONLY to invalidate its candidate-encode cache when this
        # store mutates — search still receives its embedder per-call. Optional
        # and guarded, so a store constructed without one (tests, write-only
        # callers) is unaffected.
        self._embedder = embedder

    def list_entries(self) -> list[dict[str, Any]]:
        if not self.backend_dir.exists():
            return []

        entries: list[dict[str, Any]] = []
        for path in sorted(self.backend_dir.glob("*.json")):
            # Tolerant read: the site partitions of a parallel run share one store,
            # so a sibling's dedup unlink or in-flight rewrite can race a scan.
            # A transient FileNotFoundError / JSONDecodeError on ONE file must
            # skip that file, not crash retrieval (which would end the episode
            # as an agent_error).
            try:
                with path.open("r", encoding="utf-8") as file:
                    entries.append(json.load(file))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return entries

    def entry_count(self) -> int:
        """Number of stored entries — a cheap glob (no JSON load), so callers
        wanting just the count don't trigger a second full scan after search."""
        if not self.backend_dir.exists():
            return 0
        return sum(1 for _ in self.backend_dir.glob("*.json"))

    def _invalidate_embedder_cache(self) -> None:
        """Signal the (optional) embedder to drop its candidate-encode cache
        after this store mutates. Guarded, so an embedder without the optional
        invalidate() capability — or no embedder at all — is a no-op."""
        invalidate = getattr(self._embedder, "invalidate", None)
        if callable(invalidate):
            invalidate()

    # ------------------------------------------------------------------
    # Dedup / keep-best (hygiene). Grouped by the entry ``task_id`` FIELD
    # (never filename — OSWorld ids contain hyphens). No scoring/threshold
    # change; the store only removes same-task duplicate FILES.
    # ------------------------------------------------------------------

    def _delete_entry(self, entry: dict[str, Any]) -> None:
        entry_id = entry.get("entry_id")
        if not entry_id:
            return
        try:
            (self.backend_dir / f"{entry_id}.json").unlink()
        except FileNotFoundError:
            pass  # already gone — idempotent / concurrent sibling
        # A delete mutates the entry set. Invalidating at this lowest-level
        # op covers EVERY delete path (add-time dedup + keep_best_sweep) at once.
        self._invalidate_embedder_cache()

    def _dedup_task_id_group(self, group: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Keep the single best entry in a same-task_id group; delete the losers.
        Returns a ``_dedup_events`` record, or None when there is nothing to do."""
        if len(group) <= 1:
            return None
        best = max(group, key=_keep_best_key)
        losers = [entry for entry in group if entry is not best]
        for loser in losers:
            self._delete_entry(loser)
        return {
            "task_id": str(best.get("task_id")),
            "kept": best.get("entry_id"),
            "dropped": [loser.get("entry_id") for loser in losers],
            "kept_criterion": {
                "win": bool(best.get("success")),
                "created_at": best.get("created_at"),
            },
        }

    def keep_best_sweep(self) -> list[dict[str, Any]]:
        """One-shot dedup sweep — among entries sharing a ``task_id`` field keep
        the single best (keep-best ordering) and delete the losers. Idempotent
        and single-process: the launcher runs it PRE-SPAWN (after seed copy,
        before any partition starts, store quiescent). Returns ``_dedup_events``."""
        by_task: dict[str, list[dict[str, Any]]] = {}
        for entry in self.list_entries():
            by_task.setdefault(str(entry.get("task_id")), []).append(entry)
        events: list[dict[str, Any]] = []
        for group in by_task.values():
            event = self._dedup_task_id_group(group)
            if event is not None:
                events.append(event)
        if events:
            logger.info(
                "support-set keep-best sweep (%s): deduped %d task_id group(s), dropped %d entry(ies)",
                self.backend, len(events), sum(len(e["dropped"]) for e in events),
            )
        return events

    def add_success_entry(self, entry: dict[str, Any]) -> Path:
        payload = deepcopy(entry)
        payload["backend"] = self.backend
        payload["created_at"] = payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        payload["success"] = bool(payload.get("success", True))
        payload["entry_id"] = payload.get("entry_id") or self._default_entry_id(payload)

        sanitized_payload = _sanitize_for_json(payload)

        self.backend_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.backend_dir / f"{sanitized_payload['entry_id']}.json"
        # Atomic write: dump to a temp file in the SAME dir, then os.replace
        # (atomic rename on POSIX). The plain open("w") truncates immediately and
        # streams the dump, leaving a window where a concurrent list_entries in
        # another partition of a parallel run (all partitions share one support_set_root)
        # could json.load a half-written file and raise. The temp-then-rename
        # removes that window. The temp name carries the pid so concurrent writer
        # processes never collide, and its ".tmp" suffix keeps it out of the
        # list_entries "*.json" glob.
        tmp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(sanitized_payload, file, indent=2, sort_keys=True)
        os.replace(tmp_path, target_path)
        # A new file (or a re-key of an existing win) mutated the entry set —
        # drop the candidate-encode cache. Together with the delete-path invalidation
        # in _delete_entry, this covers every mutation path (add / add-time dedup /
        # keep_best_sweep). The relabel write-back needs NO invalidation: it mutates the
        # steps' ``recent_history[*].reasoning.actual_effect_class``, never an entry's
        # instruction/observation_text, which are the only encode inputs (see search()
        # text_inputs).
        self._invalidate_embedder_cache()

        # Add-time dedup — write-new-THEN-delete-loser, so a crash in this
        # window leaves BOTH entries, never neither. Among same-task_id entries
        # keep the best (keep-best ordering), delete the losers. Handles the
        # mid-run seed-vs-new-win collision (warm config); a no-op in the
        # disjoint-partition cold run (task_ids are unique per partition).
        task_id = str(sanitized_payload.get("task_id"))
        group = [e for e in self.list_entries() if str(e.get("task_id")) == task_id]
        event = self._dedup_task_id_group(group)
        if event is not None:
            logger.info(
                "support-set add-time dedup (%s): task_id=%s kept=%s dropped=%s",
                self.backend, task_id, event["kept"], event["dropped"],
            )
        return target_path

    def search(
        self,
        *,
        instruction: str,
        observation_text: str = "",
        query: Optional[str] = None,
        top_k: int = 2,
        embedder: Optional[Embedder] = None,
    ) -> list[dict[str, Any]]:
        """Deterministic text-arm ranking (Retrieval): local-embedding similarity
        -> relevance gate -> top-k. No LLM call.

        ``query`` is an optional rewritten query string (Demand-driven retrieval) —
        defaults to ``instruction``. Returns deepcopied entries (gated, top-k),
        each annotated with a ``retrieval`` telemetry dict {text_score}.

        One arm, no fusion: text similarity is the whole ranking (design.md §5),
        and rank fusion over a single ranked list would be an
        order-preserving no-op. The relevance gate is correspondingly the text
        condition alone, still keyed on the raw per-entry score (an absolute
        precision threshold, which a rank-based fused score could never carry).

        The text arm is a HARD REQUIREMENT, not a graceful upgrade: with no
        other arm there is no condition under which an entry could pass
        the gate without it, so a dead arm would return empty for every step and
        let the episode run on to a SCORED outcome — an infra failure silently
        minting an invalid measurement row. It raises instead
        (``TextArmUnavailableError``; the runner classifies it ``infra_error`` and
        the lane script retries the task).
        """
        threshold = self.relevance_threshold
        entries = self.list_entries()
        if not entries:
            return []

        text_inputs: dict[str, str] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for entry in entries:
            entry_id = str(entry.get("entry_id") or "")
            if not entry_id:
                continue
            by_id[entry_id] = entry
            text_inputs[entry_id] = " ".join(
                p for p in (str(entry.get("instruction") or ""), _entry_observation_text(entry)) if p
            )

        # --- text arm (the only arm) ---
        text_scores: dict[str, float] = {}
        text_enabled = embedder is not None and getattr(embedder, "available", False)
        if text_enabled:
            ids = list(by_id.keys())
            query_text = " ".join(p for p in ((query or instruction or ""), observation_text) if p)
            # FIXED-COMPOSITION INVARIANT (candidate-encode cache): pass the FULL
            # candidate list, never a query-dependent subset. Together with the
            # store's invalidate-on-every-mutation this keeps every cached row
            # from one standing full-set encode; a subset call would mint rows
            # under a DIFFERENT batch composition (~1e-3 shift) and silently void
            # the cache's bit-identity guarantee (see the cache note in
            # embedding.py).
            sims = embedder.similarity(query_text, [text_inputs[i] for i in ids])
            if sims is not None and len(sims) == len(ids):
                text_scores = {i: float(s) for i, s in zip(ids, sims)}
            else:
                text_enabled = False  # arm failed mid-run

        if not text_enabled:
            raise TextArmUnavailableError(
                f"retrieval ran without its text arm ({_text_arm_failure_reason(embedder)}; "
                f"backend={self.backend}). Ranking is text-only, so no entry could pass the "
                "relevance gate — refusing to return an empty result, which would score this "
                "episode on a degraded retriever."
            )

        # --- relevance gate (design.md §5) ---
        # An absolute per-entry precision threshold on the text score: weakly
        # matched experience is never injected.
        eligible = [i for i in by_id if text_scores.get(i, 0.0) >= threshold]
        if not eligible:
            return []

        # Descending text score, ties broken by entry id for determinism.
        ranked = sorted(eligible, key=lambda i: (-text_scores.get(i, 0.0), i))
        # Retrieval-time belt: at most ONE entry per task_id in the
        # injected top-k — walk the ranking so the highest-ranked entry per task
        # wins. Makes correctness independent of dedup timing. The query's OWN
        # task is NOT excluded (warm's legitimate same-task reuse): self-task
        # entries rank normally and one is kept. No scoring / threshold change.
        final_order: list[str] = []
        seen_task_ids: set[str] = set()
        for entry_id in ranked:
            task_id = str(by_id[entry_id].get("task_id"))
            if task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            final_order.append(entry_id)
            if len(final_order) >= top_k:
                break

        results: list[dict[str, Any]] = []
        for entry_id in final_order:
            result = deepcopy(by_id[entry_id])
            result["retrieval"] = {"text_score": round(text_scores.get(entry_id, 0.0), 6)}
            results.append(result)
        return results

    def _default_entry_id(self, entry: dict[str, Any]) -> str:
        # entry_id = <task_id>-<second-timestamp>-<short random suffix>. The
        # second-resolution timestamp alone collides when the same
        # task_id is persisted twice within one second (warm re-run / --no_resume),
        # and the atomic os.replace then makes the second write CLOBBER the first —
        # "last writer wins" silently discards a win before keep-best can rank them.
        # The random suffix makes such same-second, same-task ids distinct so both
        # files persist and dedup picks the winner deterministically. It is process-
        # and clock-independent, so it also covers concurrent same-second writes
        # from different partitions. Dedup groups by the task_id FIELD, so the
        # longer filename does not affect grouping.
        task_id = entry.get("task_id") or "support"
        task_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(task_id)).strip("-") or "support"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{task_id}-{timestamp}-{uuid.uuid4().hex[:8]}"
