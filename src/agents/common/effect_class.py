"""Closed effect-class vocabulary + normalizer (design.md Effect-class vocabulary).

The effect-class is the agent-emitted, closed-vocabulary label for *what an
action did to the environment*. It is the matchable, transferable primitive the
design is built on:

- the effect side of the ``(action_class -> effect_class)`` edge is a closed
  label, so it is comparable across trajectories (the stored recipe a retrieved
  exemplar carries, Retrieval);
- the predicted (pre-action) vs actual (post-action) effect-class is the basis of
  verification (Verification).

The vocabulary is **closed** and enforced at emission — natively where the
backbone supports schema-constrained output (the GPT json_schema path), and via
``normalize_effect_class`` post-parse otherwise (the Claude path). An off-vocab
or missing emission normalizes to ``UNKNOWN`` (an out-of-band sentinel, *not* a
vocabulary member), which downstream treats as non-comparable / non-progress —
the conservative choice while per-class extraction reliability is unmeasured.

This module is backbone- and benchmark-agnostic (no LLM call, no I/O); it lives
in ``common/`` alongside the other shared primitives.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# The closed vocabulary — the full ~10 classes; not trimmed (see Known limitations in design.md).
# Order is stable for deterministic iteration / display.
# ---------------------------------------------------------------------------

EFFECT_CLASSES: tuple[str, ...] = (
    "window_opened",
    "window_closed",
    "navigation",
    "field_filled",
    "selection_changed",
    "content_modified",
    "state_toggled",
    "saved_persisted",
    "error",
    "no_change",
)

EFFECT_CLASS_SET: frozenset[str] = frozenset(EFFECT_CLASSES)

# Out-of-band sentinel for an unrecognized / missing effect label. NOT a
# vocabulary member; verification treats it as never-equal, and progress logic
# treats it as non-progress.
UNKNOWN: str = "unknown"

# The non-progress classes (``error`` / ``no_change``). They DEFINE the
# positive-progress set below (their complement); nothing counts runs of them —
# there is no progress-gated stop (design.md §6).
NON_PROGRESS_EFFECTS: frozenset[str] = frozenset({"error", "no_change"})

# Positive-progress effects: the closed classes that witness a task-relevant
# environment change. Used by the completion-gate (Verification) — a "done" claim is
# accepted only when backed by at least one of these.
POSITIVE_PROGRESS_EFFECTS: frozenset[str] = EFFECT_CLASS_SET - NON_PROGRESS_EFFECTS


# ---------------------------------------------------------------------------
# Synonym map for post-parse normalization (Claude path). Kept deliberately
# small — only obvious, unambiguous surface variants. Anything not resolved
# here and not an exact vocabulary member falls through to UNKNOWN.
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, str] = {
    "opened": "window_opened",
    "window_open": "window_opened",
    "dialog_opened": "window_opened",
    "closed": "window_closed",
    "window_close": "window_closed",
    "dialog_closed": "window_closed",
    "navigated": "navigation",
    "navigate": "navigation",
    "page_changed": "navigation",
    "filled": "field_filled",
    "field_filled_in": "field_filled",
    "text_entered": "field_filled",
    "selection": "selection_changed",
    "selected": "selection_changed",
    "content_changed": "content_modified",
    "modified": "content_modified",
    "edited": "content_modified",
    "toggled": "state_toggled",
    "toggle": "state_toggled",
    "state_changed": "state_toggled",
    "saved": "saved_persisted",
    "persisted": "saved_persisted",
    "save": "saved_persisted",
    "failed": "error",
    "failure": "error",
    "errored": "error",
    "no_op": "no_change",
    "noop": "no_change",
    "unchanged": "no_change",
    "none": "no_change",
}


def normalize_effect_class(raw: Optional[str], *, default: str = UNKNOWN) -> str:
    """Coerce a raw effect-class string to a closed-vocabulary member.

    Recognizes exact vocabulary members (case/whitespace-insensitive) and a
    small set of obvious synonyms. Anything else -> ``default`` (``UNKNOWN``).

    This is the post-parse enforcement used where the backbone cannot
    schema-constrain the output; it is a no-op on already-valid labels.
    """
    if raw is None:
        return default
    token = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return default
    if token in EFFECT_CLASS_SET:
        return token
    if token in _SYNONYMS:
        return _SYNONYMS[token]
    return default


def is_valid_effect_class(value: Optional[str]) -> bool:
    """True iff ``value`` is an exact closed-vocabulary member (not UNKNOWN)."""
    return isinstance(value, str) and value in EFFECT_CLASS_SET


def is_progress_effect(value: Optional[str]) -> bool:
    """True iff the effect witnesses task-relevant progress (Verification completion-gate).

    ``error`` / ``no_change`` / ``UNKNOWN`` / off-vocab -> False.
    """
    return isinstance(value, str) and value in POSITIVE_PROGRESS_EFFECTS


def effects_match(predicted: Optional[str], actual: Optional[str]) -> bool:
    """Verification primitive (Verification): clean equality of predicted vs actual.

    Both sides are normalized first. ``UNKNOWN`` never matches (an
    uninterpretable actual effect cannot confirm a prediction), so a missing or
    off-vocab actual conservatively reads as a mismatch.
    """
    p = normalize_effect_class(predicted)
    a = normalize_effect_class(actual)
    if p == UNKNOWN or a == UNKNOWN:
        return False
    return p == a


# ---------------------------------------------------------------------------
# Effect FAMILIES — a fixed a-priori partition of the closed vocabulary, used by
# the episode-close relabel write-back (orchestrator ``reconcile_effect_labels``,
# design.md §6): a step's positive-progress claim stands when a
# monitor-confirmed effect of the SAME FAMILY occurs anywhere in the episode.
# Two labels drawn at different steps rarely agree on the exact class (e.g.
# ``selection_changed`` vs ``state_toggled`` on a genuine win), so the match is at
# FAMILY granularity — an ontology-level contract, NOT a tunable threshold. Every
# closed class (+ the UNKNOWN sentinel) is assigned to exactly one family; the
# ``null`` family (no-progress / unknown) is never a confirmation.
# ---------------------------------------------------------------------------

EFFECT_FAMILIES: dict[str, frozenset[str]] = {
    "interaction_state": frozenset({"field_filled", "selection_changed", "state_toggled"}),
    "persistence": frozenset({"saved_persisted", "content_modified"}),
    "navigation": frozenset({"navigation", "window_opened", "window_closed"}),
    "null": frozenset({"no_change", "error", UNKNOWN}),
}
_EFFECT_TO_FAMILY: dict[str, str] = {
    effect: family for family, members in EFFECT_FAMILIES.items() for effect in members
}


def effect_family(value: Optional[str]) -> str:
    """The family of an effect class (the relabel write-back's granularity).
    Normalizes first; anything off-vocabulary falls to ``null`` (UNKNOWN's family)."""
    return _EFFECT_TO_FAMILY.get(normalize_effect_class(value), "null")
