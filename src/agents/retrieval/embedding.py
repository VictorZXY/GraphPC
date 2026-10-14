"""Local-embedding text arm for retrieval (design.md Retrieval).

The text arm scores stored experience by instruction + observation-summary
similarity using a **local, offline** sentence-embedding model — never a
backbone SDK embeddings call (not every backbone exposes one, and a local model
keeps the arm backbone-agnostic and offline).

This module never raises: a missing library / missing weights / a mid-run encode
failure leaves the embedder simply ``available == False`` (logged once) and
``similarity`` returns None. The CALLER does not degrade around that — the
text arm is the whole ranking, so the store turns an unavailable arm into
a loud ``TextArmUnavailableError`` rather than serving an empty result that would
score the episode on a retriever it never ran (see the store's ``search``).

The default model is ``Qwen/Qwen3-Embedding-0.6B`` (Apache-2.0, ~600M params,
32K-token native context, a strong MTEB-multilingual model). 32K context
is chosen so the ~1500-token compact-observation passed to retrieval is
embedded without truncation (a 256-token cap would silently drop ~85% of the
observation from the text the arm ranks on). The embedder is
injectable so tests can supply a deterministic fake and the real model can be
swapped without touching call sites.
"""

from __future__ import annotations

import logging
import os
import time

from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

logger = logging.getLogger("desktopenv.agent")

# Default encode batch size — the sentence-transformers default.
_DEFAULT_EMBED_BATCH_SIZE = 32

# Backoff between CUDA-OOM retries of one encode. On a GPU shared with other
# processes an encode can OOM while a co-tenant's usage fluctuates; such OOMs
# are transient and a short wait is the right first line. One initial attempt
# plus these three retries (max ~65s added latency on a fully failing call).
_OOM_RETRY_DELAYS_SECONDS = (5.0, 15.0, 45.0)


class TextArmUnavailableError(RuntimeError):
    """The retrieval text arm is unavailable in a configuration that has no
    defined behavior without it.

    Raised by the store whenever a retrieval-ON run is served without its text
    arm. The text arm is the whole ranking, so a dead arm has no
    condition under which any entry can pass the relevance gate: retrieval would
    return empty for every step and the episode would run on to a SCORED outcome
    — an infrastructure failure silently minting an invalid measurement row.
    Failing the episode loudly is the semantics, not an option; the runner
    classifies this as ``infra_error`` so the lane script can retry the task.

    ``--retrieval_off`` runs never reach here: they open no store at all.
    """


def _is_cuda_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure. Detected by class name / message
    rather than by importing torch — this module deliberately keeps the heavy
    deps out of module scope so it imports in a torch-less env.

    Breadth is deliberate: the ``"out of memory"`` substring also matches HOST
    allocation failures (and any other allocator whose message says so), not just
    CUDA. Treating those as retryable is the intended trade — the worst case is
    ~65s of backoff before the same degradation verdict the caller would have
    reached immediately, whereas a missed OOM is a degraded measurement row."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def _release_cuda_cache() -> None:
    """Best-effort release of OUR cached CUDA blocks between OOM retries — the
    co-tenant holds its own, but returning our unused cache to the allocator is
    what most often makes the next attempt fit. Fully guarded: a torch-less or
    CPU-only install is a no-op."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — never let cleanup raise.
        pass


def _embed_batch_size() -> int:
    """Encode batch size, read once from ``GRAPHPC_EMBED_BATCH_SIZE`` (default
    32 = the sentence-transformers default). A GPU-memory knob ONLY: it trades
    peak VRAM vs throughput. Through the production ``similarity()`` path the
    same pair scores bit-identically with its partner encoded alone and inside a
    32-item batch, and both orderings agree to every digit. The one composition
    effect is narrower: a text batched with a much LONGER companion differs
    ~1e-3 from the same text encoded alone (the candidate-cache note in
    ``SentenceTransformerEmbedder.__init__``), which is why the cache relies on its
    fixed-composition invariant rather than on this knob. Long raw-tree texts can
    push the default-32 encode peak past a shared card's free memory when several
    partitions run concurrently; a smaller batch size is the remedy there.
    Invalid / non-positive values degrade to the default with a warning rather
    than crashing a run (this module never crashes a run)."""
    raw = os.environ.get("GRAPHPC_EMBED_BATCH_SIZE")
    if raw is None:
        return _DEFAULT_EMBED_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        logger.warning("GRAPHPC_EMBED_BATCH_SIZE=%r is not an int; using %d", raw, _DEFAULT_EMBED_BATCH_SIZE)
        return _DEFAULT_EMBED_BATCH_SIZE
    if value <= 0:
        logger.warning("GRAPHPC_EMBED_BATCH_SIZE=%r must be positive; using %d", raw, _DEFAULT_EMBED_BATCH_SIZE)
        return _DEFAULT_EMBED_BATCH_SIZE
    return value


@runtime_checkable
class Embedder(Protocol):
    """Minimal text-embedding contract the retrieval ranker depends on."""

    @property
    def available(self) -> bool:
        ...

    def similarity(self, query: str, candidates: Sequence[str]) -> Optional[List[float]]:
        """Cosine similarities of ``query`` vs each candidate in [0, 1], or
        ``None`` when the arm is unavailable — which the caller must treat as
        fatal (``TextArmUnavailableError``), not as a cue to rank some other
        way: there is no second arm to fall back to."""
        ...

    def invalidate(self) -> None:
        """Drop any cached candidate embeddings (call on support-set mutation).
        Optional capability — callers guard for implementations that omit it."""
        ...


class SentenceTransformerEmbedder:
    """Local sentence-embedding text arm backed by ``sentence-transformers``.

    Construction never raises: if the library or weights are missing the
    instance is simply ``available == False`` and ``similarity`` returns None.
    Model load is attempted once at construction so a per-call import/cost is
    avoided; share one instance across the store.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        self.model_name = model_name
        # Read the memory knob once at construction (the embedder is a per-process
        # singleton). Semantics-neutral — see _embed_batch_size.
        self._batch_size = _embed_batch_size()
        # Candidate-encode cache: text -> normalized embedding row. The
        # candidate texts (stored experience) are re-scored on EVERY step, so
        # re-encoding them each call is the dominant text-arm cost; caching keyed
        # on the exact text collapses it to one encode per distinct candidate.
        # Exactness caveat: a row is BATCH-COMPOSITION sensitive (the same text
        # encoded alone vs batched with a long companion differs ~1e-3), so a hit
        # is bit-exact ONLY under the FIXED-COMPOSITION INVARIANT the real caller
        # maintains — search() always passes the FULL candidate list (miss set
        # all-or-none, see the note at its call site) and the store invalidates
        # on EVERY mutation — which keeps every cached row from one standing
        # full-set encode. Only wrong-TEXT staleness is impossible by keying
        # alone (the key is the exact text).
        self._candidate_cache: dict[str, Any] = {}
        # Count of similarity() calls that gave up and degraded to no text arm
        # (OOM retries exhausted, or any other mid-run failure). Process-cumulative
        # diagnostic counter: no production code reads it — the store names a failure
        # through ``last_failure_kind`` below.
        self.failed_call_count: int = 0
        # Which branch produced the most recent failure, so an error message can
        # name the ACTUAL cause instead of assuming OOM:
        # "cuda_oom_retries_exhausted" | "encode_error" | None (the arm is
        # currently healthy — never failed, or failed and has since recovered).
        self.last_failure_kind: Optional[str] = None
        self._model = None
        self._np = None
        try:  # optional heavy deps — degrade if absent (offline / not installed)
            import numpy as np  # noqa: F401
            from sentence_transformers import SentenceTransformer

            self._np = np
            self._model = SentenceTransformer(model_name)
            logger.info("Retrieval text arm enabled (sentence-transformers/%s)", model_name)
        except Exception as exc:  # noqa: BLE001 — any failure => text arm off.
            logger.warning(
                "Retrieval text arm UNAVAILABLE: could not load "
                "sentence-transformers/%s (%s). Retrieval-on runs will ABORT — "
                "text is the only ranking; there is no structure arm to fall back "
                "to. Only --retrieval_off runs are unaffected.",
                model_name,
                exc,
            )

    @property
    def available(self) -> bool:
        return self._model is not None

    def similarity(self, query: str, candidates: Sequence[str]) -> Optional[List[float]]:
        if not self.available or not candidates:
            return None
        # CUDA OOM is transient (co-tenant fluctuation), so retry the whole encode
        # with backoff before giving up. Any OTHER failure warns and degrades,
        # with no retry.
        for attempt in range(len(_OOM_RETRY_DELAYS_SECONDS) + 1):
            try:
                scores = self._similarity_once(query, candidates)
                # A success clears the failure kind: it describes the arm's
                # CURRENT state, and a stale label from an earlier transient OOM
                # would make the next error message name the wrong cause.
                self.last_failure_kind = None
                return scores
            except Exception as exc:  # noqa: BLE001 — never crash a run on the text arm.
                if not _is_cuda_oom(exc):
                    logger.warning("Retrieval text arm failed mid-run, falling back: %s", exc)
                    self.failed_call_count += 1
                    self.last_failure_kind = "encode_error"
                    return None
                if attempt >= len(_OOM_RETRY_DELAYS_SECONDS):
                    logger.error(
                        "Retrieval text arm exhausted %d CUDA-OOM retries; text arm is DOWN "
                        "for this call: %s",
                        len(_OOM_RETRY_DELAYS_SECONDS), exc,
                    )
                    self.failed_call_count += 1
                    self.last_failure_kind = "cuda_oom_retries_exhausted"
                    return None
                delay = _OOM_RETRY_DELAYS_SECONDS[attempt]
                logger.warning(
                    "Retrieval text arm CUDA OOM (attempt %d/%d); retrying in %.0fs: %s",
                    attempt + 1, len(_OOM_RETRY_DELAYS_SECONDS) + 1, delay, exc,
                )
                _release_cuda_cache()
                time.sleep(delay)
        return None

    def _similarity_once(self, query: str, candidates: Sequence[str]) -> Optional[List[float]]:
        """One encode+score attempt. Raises on failure; the retry policy lives in
        ``similarity``. Split out so a retry re-runs the encode from a clean
        state — the candidate cache only ever holds fully-encoded rows, so a
        partial failure leaves nothing half-written behind."""
        # Encode only the candidate texts not already cached (dedup-preserving
        # order), then reuse cached rows for the rest. The query is ALWAYS
        # re-encoded (it changes every step, so caching it would only bloat the
        # cache). Under the fixed-composition invariant (see the cache note in
        # __init__) the miss set here is all-or-none in practice, so reused
        # rows always come from one standing full-set encode and hits are
        # bit-exact. A partial
        # miss would mint rows from a DIFFERENT batch composition (~1e-3 off
        # a full-set row) — supported mechanically, but no real caller does it.
        missing = [t for t in dict.fromkeys(candidates) if t not in self._candidate_cache]
        if missing:
            encoded = self._model.encode(
                missing, normalize_embeddings=True, batch_size=self._batch_size
            )
            for text, vec in zip(missing, encoded):
                self._candidate_cache[text] = vec
        q = self._model.encode(
            [query], normalize_embeddings=True, batch_size=self._batch_size
        )[0]
        sims = []
        for text in candidates:
            # rows are L2-normalized => dot product is cosine in [-1, 1];
            # clamp to [0, 1] (negative similarity is "unrelated" for ranking).
            sims.append(max(0.0, float(self._np.dot(q, self._candidate_cache[text]))))
        return sims

    def invalidate(self) -> None:
        """Clear the candidate-encode cache. The store calls this on EVERY mutation
        (add / dedup delete) so the cache never outlives the entry set it was built
        for. Cheap: the next similarity() re-encodes the now-current candidates on
        its first call, then reuses them across the episode's remaining steps.
        Numeric fidelity DOES depend on it: rows are batch-composition sensitive,
        and invalidate-on-every-mutation is one half of the fixed-composition
        invariant (the other: search() passes the FULL candidate list) that keeps
        every cached row from one standing full-set encode. Only wrong-TEXT
        staleness is invalidation-independent (the key is the exact text). It also
        bounds cache growth to the live store."""
        self._candidate_cache.clear()


# Module-level lazy singleton so the (heavy) model loads at most once per
# process. Tests inject their own embedder and never touch this.
_DEFAULT_EMBEDDER: Optional[Embedder] = None


def get_default_embedder() -> Embedder:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = SentenceTransformerEmbedder()
    return _DEFAULT_EMBEDDER
