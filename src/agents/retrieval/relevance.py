"""The relevance-gate admit rule: a percentile of the offline pairwise
instruction-similarity distribution.

WHY A PERCENTILE AND NOT A CONSTANT. The gate decides whether a retrieved
exemplar is similar enough to inject at all. The shared default is a fixed
``0.15`` (``support_set.DEFAULT_RELEVANCE_THRESHOLD``). A fixed cosine number is
not portable across benchmarks, because the *scale* of the similarity
distribution moves with the corpus: 0.15 sits near the bottom of the WebArena
pairwise distribution, i.e. it approves essentially everything, and it
approves everything on OSWorld 2.0 too — where a store may hold ONE entry, so
"approve everything" means that entry is injected into nearly every later task
whether or not it has anything to do with them. Replacing the constant with a
statistic OF THE BENCHMARK'S OWN TASK SET is what makes the rule portable: it
self-calibrates to whatever corpus it is pointed at, and reads only task
instructions — no outcome data, no answer key, nothing benchmark-specific.

THE PRE-REGISTERED PARAMETER IS THE PERCENTILE, NOT THE NUMBER. It is
``p74.5``, fixed from the **WebArena** distribution alone by a criterion stated
entirely about that benchmark — the largest percentile that refuses at most 2%
of a WebArena reference run's actual injections — and then applied UNCHANGED to
any benchmark. A hard-coded cosine would reintroduce the portability defect
above, so the percentile is what is pinned here and the threshold is derived
per corpus.

THE POPULATION is the ordered ``(query task, store-entry task)`` pairs with
query != entry: one score per judgement the retriever can be asked to make. The
ordering matters in principle even where it does not in practice — query text
and entry text are built by different code paths and encoded by different calls,
so the two forms genuinely differ on WebArena, while on OSWorld 2.0 both reduce
to the bare instruction. With the real embedder the two conventions differ by
about 0.0002 at p74.5 on the OSWorld 2.0 subset; the choice is made on
principle, not on the number.

BEHAVIOUR THIS BUYS, on the 28-task OSWorld 2.0 subset at the resolved value:
every pre-registered twin pair is admitted and most non-twin pairs are refused,
so a lone stored entry reaches only the later tasks it resembles, not nearly
every one. With a near-empty store the gate ABSTAINS rather than injecting the
one or two entries that happen to exist in order to fill its two slots. Note
the margin at the bottom is thin — the weakest genuine twin pair sits at 0.3730
against a threshold of 0.35, about 0.023 — so an implementation detail that
shifted the effective threshold upward would start costing real transfers.

WHERE THE INSTRUCTIONS COME FROM, and where the arithmetic happens. The corpus
is read from the benchmark harness's OWN task loader inside its checkout — the
place the instructions actually live — never from a derived export, which is
neither tracked nor synced and would simply be absent on the run host. The
percentile arithmetic (`recompute=True`) runs outside the worker: calibrate mode
is run offline, before any worker (no shipped script runs it: call
``resolve_relevance_threshold`` with a ``PercentileRelevanceGate(..., recompute=True)``
on the installed corpus to re-derive the pinned value); workers run verify mode.
Every OSWorld GraphPC launch runs ``--retrieval_off``; the OSWorld runner refuses a
retrieval-on launch. The worker verifies that the installed corpus is the
declared one and applies the pre-registered value (`recompute=False`), which
costs no encodes. See `PercentileRelevanceGate` for why that split is the right
one and for what it does not catch.

NOT EVERY INSTRUCTION IS A FIXED STRING. Two of OSWorld 2.0's installed tasks
(057, 072) mint an Overleaf account at every module exec and interpolate it into
the instruction, so they have no installed text: every read disagrees with every
other, including two reads a second apart in one process, and a digest over them
is a nonce no reader can reproduce. The rule this gate holds survives that
untouched: on the 28-task subset p74.5 resolves to the SAME value on every read,
because those two tasks' pairs do not sit at the 74.5th order statistic. So the
threshold is pinned and the DIGEST'S SCOPE is narrowed instead — see
``digest_excluded_task_ids`` for what that stops covering.

AND THE RELEASE TAG DOES NOT HOLD THE TEXT CONSTANT. It is tempting to reason
that a pinned benchmark release makes text drift a low-probability event; on
OSWorld 2.0 that is exactly backwards. `evaluation_examples/task_class/task_*.py`
is in the checkout's OWN `.gitignore` — the task classes ship separately, from a
gated HuggingFace dataset — so `git describe` and a clean `git status` are not
evidence about instruction text at all, and a re-download inside one tag changes
it with no git signal whatsoever. Drift INSIDE a release is the blind case; a
version bump at least moves something visible. This digest is what watches it.

SCOPING. This module is inert unless a benchmark asks for it. The shared default
stays the fixed ``0.15``: ``LocalSupportSetStore`` takes ``relevance_gate=None``
and then applies the fixed threshold, so a WebArena run cannot execute this rule
(benchmark-scoped, design.md §10). The percentile is a METHOD-LEVEL parameter —
the rule is portable and a benchmark only instantiates the number. On WebArena
the instantiation reproduces the fixed gate by construction (the 2%-refusal
criterion above), so WebArena keeps the ``None`` default, the fixed 0.15. It is
class one by nature (design.md §10), and a Lite-165 validation of it would be
vacuous: on a unique-template task set retrieval has nothing to transfer, and
``--retrieval_off`` is the default there.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

from .embedding import Embedder, TextArmUnavailableError

# The pre-registered parameter. Fixed from the WebArena distribution alone;
# applied unchanged to every benchmark. Do not re-derive it per benchmark —
# a per-benchmark percentile would be a tuned constant wearing a rule's clothes.
PREREGISTERED_RELEVANCE_PERCENTILE = 74.5

# The pre-registration records the resolved value to this tolerance, so a run
# that recomputes the percentile at load and a run applying the recorded rounded
# value are both conformant. Wider than the 0.0002 ordering effect and narrower
# than the 0.023 margin to the weakest genuine twin.
PREREGISTERED_THRESHOLD_TOLERANCE = 0.005


class RelevanceCalibrationError(RuntimeError):
    """The percentile rule was requested but could not be resolved.

    Raised rather than falling back to the fixed default: a run that silently
    reverted to 0.15 would inject on the fixed rule while its telemetry claimed
    the percentile one, which is the degraded-path-masquerading-as-primary
    failure the measurement-integrity rule exists to prevent.
    """


@dataclass(frozen=True)
class PercentileRelevanceGate:
    """A benchmark's opt-in to the percentile rule.

    ``corpus`` maps task id -> instruction for the benchmark's INSTALLED task
    set, resolved from the benchmark harness's OWN task loader — the place the
    instructions actually live. Not from a derived export: a snapshot of the
    corpus is one more artefact that has to be kept in step with the
    pre-registration, and a stale one is precisely the silent-wrong-corpus
    failure this rule exists to prevent. ``expected_task_ids`` is the subset the
    pre-registration declares, and the two are checked against each other in
    BOTH modes.

    ``pinned_population_size`` is the ordered-pair count the pinned threshold
    was DERIVED over (n*(n-1)). Optional, and checked in both modes when given:
    it is the one declared quantity that travels with the threshold rather than
    with the task list, so it is what detects a pre-registration whose task set
    and whose threshold were written at different times. Left None, that class
    of staleness is only caught if it happens to shift the percentile past the
    tolerance — see the population-size note in ``resolve_relevance_threshold``.

    ``recompute`` chooses where the percentile arithmetic happens, and it is a
    placement decision, not a behaviour switch — both modes gate at the same
    number and both fail loudly:

      * ``True`` — CALIBRATE. Encode the corpus, take the percentile, and if a
        ``pinned_threshold`` is declared require it to agree within
        ``tolerance``. A calibration step's mode: calibrate mode is run
        offline, before any worker (no shipped script runs it: call
        ``resolve_relevance_threshold`` with a ``PercentileRelevanceGate(..., recompute=True)``
        on the installed corpus to re-derive the pinned value); workers run
        verify mode. It costs n*(n-1) encodes, and a failure there costs nothing
        because no VM has been paid for yet.
      * ``False`` (default) — VERIFY. No encoding at all: confirm the installed
        corpus is the one the pre-registration was written against, then apply
        the declared ``pinned_threshold``. This is the WORKER's mode.

    WHY THE WORKER DOES NOT RECOMPUTE. The disagreement the recomputation
    catches — installed task set or encoder not the one the pre-registration was
    written against — is a RUN-LEVEL, STATIC property: it cannot change between
    episodes. Checking it inside every worker's construction pays 756 encodes
    per worker (and again on every environment recycle) to re-derive a constant,
    and discovers a launch-blocking problem in the middle of a paid run, once
    per worker, after VM time has been spent. That is the wrong place. What the
    worker still owes is that it must not APPLY the rule while unable to
    substantiate it, which is what the corpus-identity check preserves at the
    cost of reading files it can already read and zero encodes.

    WHAT VERIFY MODE DOES NOT CATCH, stated rather than implied: a changed
    ENCODER. Same corpus, different embedding model, and the pinned threshold is
    calibrated against a geometry that no longer applies. That is not detectable
    without encoding, so only a calibrate-mode run catches it, and no launch
    path re-checks the encoder: every OSWorld GraphPC launch runs
    ``--retrieval_off``, which never applies this rule.
    """

    corpus: Mapping[str, str]
    expected_task_ids: Sequence[str]
    percentile: float = PREREGISTERED_RELEVANCE_PERCENTILE
    pinned_threshold: Optional[float] = None
    pinned_population_size: Optional[int] = None
    pinned_corpus_digest: Optional[str] = None
    tolerance: float = PREREGISTERED_THRESHOLD_TOLERANCE
    recompute: bool = False
    # Task ids the digest DOES NOT COVER, declared by the pre-registration. Empty
    # for any corpus whose instructions are all fixed strings, which is the normal
    # case and WebArena's (it has no gate at all). It is non-empty when a task has
    # no installed instruction to hash: a class that mints a value per construction
    # and interpolates it into the instruction has a different instruction on every
    # read, so a digest over it is a nonce that no reader can reproduce — including
    # the reader that recorded it. On OSWorld 2.0, tasks 057 and 072 mint an
    # Overleaf account at module exec, so no digest over them is reproducible.
    #
    # WHAT EXCLUDING AN ID COSTS, so that it is a decision and not a silence: the
    # digest stops seeing any edit to that task's instruction — including the parts
    # that ARE stable, such as a rewording or a changed file list. The id and
    # population checks still cover it, and the excluded ids stay in the corpus and
    # therefore in the percentile's population, because the retriever will be asked
    # about them at run time and the reference distribution must be the decision
    # distribution.
    digest_excluded_task_ids: Sequence[str] = ()


@dataclass(frozen=True)
class ResolvedRelevanceThreshold:
    """What the rule resolved to, and how — resolution-level telemetry, so a
    reader can tell the calibrated path from the fixed default without inferring
    it from the number."""

    value: float
    rule: str                      # "fixed" | "percentile"
    # How the number was arrived at THIS run: "calibrated" (recomputed from the
    # corpus here) or "verified" (corpus identity confirmed, pre-registered
    # value applied). Recorded because a reader must never have to assume which
    # — a record claiming more work than was done is the degraded-path-as-primary
    # failure in its purest form.
    resolution: Optional[str] = None
    percentile: Optional[float] = None
    # Ordered pairs scored. None — NOT 0 — in verify mode, where no population
    # was computed at all. 0 is a count and a reader is entitled to treat it as
    # a measured zero ("empty corpus", "gate inert"); None is "not applicable
    # here". This mirrors `computed_value` two fields down, which is None for
    # exactly the same reason: both are calibrate-mode quantities, and encoding
    # the same absence two different ways is how a downstream check ends up
    # asserting on a sentinel it has misread.
    population_size: Optional[int] = None
    corpus_size: int = 0           # instructions the identity check saw
    computed_value: Optional[float] = None   # before pinning; None in verify mode
    pinned: bool = False

    def as_telemetry(self) -> dict:
        """What the retrieval record should carry about the admit rule.

        EMPTY under the fixed rule, deliberately. A benchmark that did not opt
        in records no relevance keys — the same discipline as every other
        benchmark-scoped parameter — which keeps always-None percentile fields
        off a WebArena record, where they would be vestigial. There is nothing
        to attest there anyway: the fixed threshold is a compile-time constant.
        """
        if self.rule == "fixed":
            return {}
        return {
            "relevance_threshold": self.value,
            "relevance_rule": self.rule,
            "relevance_resolution": self.resolution,
            "relevance_percentile": self.percentile,
            "relevance_population_size": self.population_size,
            "relevance_corpus_size": self.corpus_size,
            "relevance_computed_value": self.computed_value,
            "relevance_pinned": self.pinned,
        }


def corpus_digest(corpus: Mapping[str, str]) -> str:
    """A stable fingerprint of a task corpus: ``"sha256:<hex>"``.

    ONE implementation, imported by everything that needs the value — the
    pre-registration writer that records it and the run-time check that compares
    it — so the recorded digest and the checked digest come off the same code by
    construction rather than by two parties agreeing to stay in step.

    Definition, and the reasoning for each choice:

      * ASCENDING task-id order. A dict's insertion order must never move the
        digest.
      * The task ID IS hashed, not only the text — but NOT for the reason it is
        tempting to give. A swap of two instructions between ids is already
        caught without it, because the text sub-digests are fed in ascending-id
        order and a swap reorders them (verified, not assumed). What hashing the
        id actually buys is distinguishing the same texts under a SORT-ORDER-
        PRESERVING RELABEL of the ids, which a texts-only digest maps to the
        identical value. Be precise about that: "renaming" is too loose, because
        a rename that MOVES an id's sort position (007 -> 999) reorders the text
        sub-digests and so changes a texts-only digest too. Both of the
        mutations that prove nothing — an instruction swap, and a
        position-moving rename — look like they test the id and do not; only
        007 -> 006, or relabelling every id keeping relative order, binds it.
        Inside this module the id check covers this case too; the id belongs in
        the digest so the function is self-sufficient for its other caller, the
        pre-registration writer, which has no id check.
      * The instruction is the RAW string the harness's loader returns: no
        strip, no Unicode normalisation, no whitespace collapsing. Every
        normalisation is a place two implementations can silently differ; raw
        means byte-identical or it fails. It matters here — real instructions in
        the installed corpus carry trailing spaces and trailing newlines.
      * Each field is hashed SEPARATELY and the fixed-width sub-digests are fed
        to the outer hash. This is the one point where a delimiter-framed
        canonical string ("id\\tinstruction\\n" and similar) cannot be used:
        11 of the 28 installed instructions CONTAIN NEWLINES, and several
        contain other whitespace, so a newline-terminated record framing is
        ambiguous on the real corpus — two different corpora can frame to
        identical bytes and collide. Fixed-width sub-digests have no delimiter
        to collide with, and leave no length-encoding convention for two
        implementations to disagree about.
      * Scope is whatever corpus is passed. The caller passes the INSTALLED
        SUBSET, because the threshold is a function of exactly those texts.
    """
    outer = hashlib.sha256()
    for task_id in sorted(corpus):
        outer.update(hashlib.sha256(str(task_id).encode("utf-8")).digest())
        outer.update(hashlib.sha256(str(corpus[task_id]).encode("utf-8")).digest())
    return "sha256:" + outer.hexdigest()


def percentile_of(values: Sequence[float], percentile: float) -> float:
    """The ``percentile``-th value of ``values`` by LINEAR INTERPOLATION between
    closest ranks — numpy's default convention, written out rather than imported
    because the convention is part of the specification.

    This matters: two implementations that differ only in the percentile
    convention (or in how they encode) resolve the same population to
    different thresholds, by more than the tolerance. Within one population a
    LOWER percentile can never yield a HIGHER threshold, which is how such a
    discrepancy is told apart from a population difference. Pinning the
    convention here keeps every reader on one number.
    """
    if not values:
        raise RelevanceCalibrationError("cannot take a percentile of an empty population")
    if not 0.0 <= percentile <= 100.0:
        raise RelevanceCalibrationError(f"percentile must be in [0, 100], got {percentile}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def ordered_pairwise_similarities(
    instructions: Sequence[str], *, embedder: Optional[Embedder],
) -> List[float]:
    """Cosine similarity for every ORDERED ``(query, entry)`` pair with
    query != entry — one score per judgement the retriever can be asked to make.

    Scored through the same ``embedder.similarity`` call the run-time gate uses, so
    the reference distribution and the run-time scores come off the same code
    path. A corpus of n instructions yields n*(n-1) scores.
    """
    texts = [str(text) for text in instructions if str(text).strip()]
    if len(texts) < 2:
        raise RelevanceCalibrationError(
            f"the percentile rule needs at least 2 task instructions to have a pairwise "
            f"distribution at all, got {len(texts)}"
        )
    if embedder is None or not bool(getattr(embedder, "available", False)):
        raise TextArmUnavailableError(
            "the percentile relevance rule cannot be calibrated without the embedding model — "
            "the reference distribution is computed with the same encoder the gate scores with, "
            "so a fallback encoder would silently calibrate against a different geometry."
        )

    scores: List[float] = []
    for index, query in enumerate(texts):
        candidates = texts[:index] + texts[index + 1:]
        similarities = embedder.similarity(query, candidates)
        if similarities is None or len(similarities) != len(candidates):
            raise TextArmUnavailableError(
                "the embedding model failed while calibrating the relevance gate "
                f"(query {index} of {len(texts)}); refusing to calibrate on a partial population."
            )
        scores.extend(float(s) for s in similarities)
    return scores


def verified_corpus_instructions(gate: PercentileRelevanceGate) -> List[str]:
    """The corpus's instructions, after confirming it IS the declared task set.

    Runs in both modes, and is the whole of what verify mode does. It answers
    "is the thing installed here the thing the pre-registration was written
    against?" from files the caller can already read, with no encoding — which
    is why the worker can afford it and why a missing or stale corpus fails at
    construction instead of being papered over.
    """
    corpus = dict(gate.corpus or {})
    expected = [str(task_id) for task_id in (gate.expected_task_ids or [])]
    if not expected:
        raise RelevanceCalibrationError(
            "the percentile rule needs the pre-registered task-id set to check the installed "
            "corpus against; without it the rule cannot tell a correct corpus from a stale one."
        )

    present = {str(task_id) for task_id, text in corpus.items() if str(text).strip()}
    missing = sorted(set(expected) - present)
    unexpected = sorted(present - set(expected))
    if missing or unexpected:
        raise RelevanceCalibrationError(
            "the installed task corpus is not the one the pre-registration declares: "
            f"{len(missing)} declared task(s) missing or without an instruction "
            f"({missing[:6]}{'...' if len(missing) > 6 else ''}), "
            f"{len(unexpected)} unexpected ({unexpected[:6]}{'...' if len(unexpected) > 6 else ''}). "
            "The pre-registered threshold is calibrated against the declared set, so applying it "
            "here would gate on a number derived from a different corpus. Resolve the corpus — "
            "do not relax the rule."
        )
    return [str(corpus[task_id]) for task_id in expected]


def resolve_relevance_threshold(
    gate: Optional[PercentileRelevanceGate],
    *,
    fixed_default: float,
    embedder: Optional[Embedder] = None,
) -> ResolvedRelevanceThreshold:
    """Resolve the gate a store should apply.

    ``gate is None`` — the shared default, and what WebArena gets — returns the
    fixed threshold unchanged, touching neither the embedder nor the corpus.

    There is NO path from here back to ``fixed_default`` once a gate is given:
    every failure raises. A run that could not substantiate the rule must not
    quietly gate at 0.15 while its telemetry reports the percentile rule.
    """
    if gate is None:
        return ResolvedRelevanceThreshold(value=float(fixed_default), rule="fixed")

    instructions = verified_corpus_instructions(gate)

    # Bind the pinned value to the corpus it was CALIBRATED over, not merely to
    # the corpus declared beside it. The failure this catches: a pre-registration
    # whose threshold fields were refreshed from a larger corpus than its id list
    # names. The installed tasks match the ids, so the identity check above
    # passes, and a threshold derived over the larger corpus's pairs is applied
    # to the smaller one. Nothing read from that file can detect it, because
    # everything else read from it is equally stale — EXCEPT the population
    # size, which travels WITH the threshold fields and therefore disagrees
    # with the stale id set.
    #
    # Without this the only backstop is a calibrate-mode run noticing the resulting
    # drift, and that is luck rather than construction: a small corpus change
    # can move the percentile by less than the tolerance and then passes both.
    # Three checks, deliberately not one, because they catch three different
    # failures and each names its own:
    #   expected_task_ids       — the INSTALLED corpus is not the DECLARED one
    #   pinned_population_size  — the declared THRESHOLD and the declared TASK
    #                             LIST were written at different times
    #   pinned_corpus_digest    — the ids match but a TEXT changed underneath
    # The digest is the strongest and subsumes the other two on paper; the other
    # two survive because their failure messages name the size or the missing
    # ids, which is what an operator can act on. A digest mismatch alone says
    # only "different corpus".
    if gate.pinned_population_size is not None:
        actual_population = len(instructions) * (len(instructions) - 1)
        if actual_population != gate.pinned_population_size:
            raise RelevanceCalibrationError(
                f"the pre-registered threshold was calibrated over "
                f"{gate.pinned_population_size} ordered pairs, but the installed corpus of "
                f"{len(instructions)} tasks yields {actual_population}. The declared threshold "
                "does not belong to this corpus — the pre-registration's task set and its "
                "threshold were written at different times. Re-derive the threshold at the "
                "installed corpus size; do not apply it across."
            )


    # LAST of the three, deliberately. Ordering decides which diagnosis an
    # operator acts on, and the digest is the least specific: it can only say
    # "different corpus". Running it after the id and size checks means that by
    # the time it fires, those two have already agreed — so text drift is the
    # remaining explanation rather than an assumption. Reversed, a corpus that
    # is simply the wrong SIZE reports as edited text, which sends the reader
    # looking for a diff that does not exist.
    if gate.pinned_corpus_digest is not None:
        covered = ({str(i) for i in gate.expected_task_ids}
                   - {str(i) for i in gate.digest_excluded_task_ids})
        actual_digest = corpus_digest(
            {task_id: text for task_id, text in gate.corpus.items() if str(task_id) in covered}
        )
        if actual_digest != gate.pinned_corpus_digest:
            scope = ("" if not gate.digest_excluded_task_ids else
                     f" The digest covers {len(covered)} of {len(gate.expected_task_ids)} declared "
                     f"tasks; {sorted(str(i) for i in gate.digest_excluded_task_ids)} are excluded by "
                     f"the pre-registration and a change to them is NOT what fired this.")
            settled = ("the declared task ids and pair count both match, so the difference is in "
                       "instruction TEXT" if gate.pinned_population_size is not None
                       else "the declared task ids match, so the difference is in instruction "
                            "TEXT or in the corpus size (no pair count was declared to check)")
            raise RelevanceCalibrationError(
                f"the installed corpus digests to {actual_digest}, but the pre-registration "
                f"declares {gate.pinned_corpus_digest}: {settled}.{scope} THREE CAUSES ARE "
                "POSSIBLE AND THEY NEED DIFFERENT FIXES. (1) An instruction really was edited. "
                "(2) The recorded digest and the corpus read here came from DIFFERENT SOURCES for "
                "what must be one string — the pre-registration's digest is written from one view "
                "of the task set and the corpus here is read from the benchmark harness's loader. "
                "(3) The instruction is not a fixed string at all: a task class may build it from "
                "a per-construction random value, in which case no two reads agree. LOAD THE "
                "CORPUS TWICE AND COMPARE before hunting a diff — a nonce shows up as every read "
                "disagreeing with every other, including two reads one second apart in the same "
                "process, whereas a corpus edit shows up as a changed instruction and a provenance "
                "mismatch shows up as every instruction agreeing when compared by hand. Then "
                "either re-derive the threshold against what is installed, or install what the "
                "pre-registration was written against."
            )

    if not gate.recompute:
        if gate.pinned_threshold is None:
            raise RelevanceCalibrationError(
                "verify mode applies the pre-registered threshold, so one must be declared "
                "(pinned_threshold). To DERIVE the threshold here instead, pass recompute=True — "
                "but that belongs in a calibration step before the run, not in a worker."
            )
        return ResolvedRelevanceThreshold(
            value=float(gate.pinned_threshold), rule="percentile", resolution="verified",
            percentile=gate.percentile, corpus_size=len(instructions), pinned=True,
        )

    population = ordered_pairwise_similarities(instructions, embedder=embedder)
    computed = percentile_of(population, gate.percentile)

    if gate.pinned_threshold is None:
        return ResolvedRelevanceThreshold(
            value=computed, rule="percentile", resolution="calibrated",
            percentile=gate.percentile, population_size=len(population),
            corpus_size=len(instructions), computed_value=computed, pinned=False,
        )

    drift = abs(computed - gate.pinned_threshold)
    if drift > gate.tolerance:
        raise RelevanceCalibrationError(
            f"the pre-registered threshold {gate.pinned_threshold:.4f} and the p{gate.percentile} "
            f"of this corpus ({computed:.4f}) differ by {drift:.4f}, over the recorded tolerance "
            f"of {gate.tolerance}. The corpus is the declared task set, so the likely cause is a "
            "changed ENCODER — which invalidates the pre-registered value. Resolve it before "
            "launching; do not widen the tolerance."
        )
    return ResolvedRelevanceThreshold(
        value=float(gate.pinned_threshold), rule="percentile", resolution="calibrated",
        percentile=gate.percentile, population_size=len(population),
        corpus_size=len(instructions), computed_value=computed, pinned=True,
    )
