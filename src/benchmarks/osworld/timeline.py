"""The OSWorld 2.0 deciding call's episode history, rendered as one timeline.

WHY A TIMELINE. A deciding request laid out as unlabelled retained frames, the
current frame, and then a long JSON history (instruction, environment brief, plan,
compacted history items, graph_summary, retrieved memory, schema) identifies the
current frame only by a caption and ties no retained frame to the step that
produced it — and the model then decides on the wrong frame ("the alert is still
visible" while the current screen shows no alert).

WHAT THE CONTROL DOES (mm_agents/m3/agent.py ``_build_messages``): system, then
user(opening frame + instruction), then per past step an assistant turn (its own
reply) followed by a user turn carrying the screen that reply produced — or a
"Tool result: Success" placeholder once the image budget drops it — and the
current screen is the LAST content of the request.

WHAT THIS RENDERS, as a single user message (the control's multi-turn form is not
reproduced):

    Step 0 — task start + the opening screen + the task instruction
    Step k: executed <what the environment executed, incl. WAIT + why>;
            <screen changed | no screen change (N px)>; your label: <reconciled>
            [+ your reasoning at step k, <= 1,500 chars]   inside the image window
            [+ Screen after step k (<action>): <frame>]    inside the image window
            or, outside it, ONE line <= 200 chars ending "[screenshot omitted]"
    STATE BEFORE STEP P: plan, graph_summary, facts, retrieved exemplars, then the
            reply instruction and JSON schema (the caller's text)
    CURRENT SCREEN, after step P-1 (<action>): <current frame>   <- nothing after it

The most recent step (P-1) shows its executed string raw and whole, in its step line
and in the CURRENT SCREEN header; every earlier step is clipped (EXECUTED_TEXT_LIMIT,
_LABEL_ACTION_LIMIT) — except the most recent `type` step while every step after it is
a `wait` or a `screenshot` (IDLE_ACTIONS): its step line stays raw too, because the
typing it started can still be running (prompts.OSWORLD_TYPING_NOTE). That extension is
not separately replay-tested: it carries the raw-text mechanism, replay-tested at the
decision right after the type, to the idle steps that follow it.

Every step is rendered (no item cap). The image count is the control's own
arithmetic, unchanged (``retained_frame_count``), computed on STEPS: a failed
capture leaves a hole in its slot, it never shifts a later frame into it.

WHO WRITES, WHO READS. ``OSWorldGraphPCAgent`` (the adapter) is the only writer:
it is called once per driver step and is the boundary where the executed string,
the frame and the monitor verdict are in hand. ``OSWorldReasoningAgent`` is the
only reader. Both hold the ONE instance ``build_default`` constructs — the
``_RecordingPointerLocator`` pattern — so the adapter still calls nothing but the
orchestrator's inner boundary. The frames live here rather than on
``RuntimeState`` because that is dumped into telemetry and persisted to the
support set.

STEP NUMBERING. Step k is the action decided at the adapter's k-th ``predict``
call; the frame observed at predict call p is the screen after step p-1 (frame 1
is the opening screen, "step 0").
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.agents.monitor.heuristic import SCREENSHOT_CHANGED_MIN_PIXELS


class ScreenshotCaptureFailed(RuntimeError):
    """V2's controller returned no screenshot after its own retries (``screenshot:
    None`` on the obs). Raised by the adapter's ``predict`` only on a profile whose
    ``capture_failure_raises`` is set, for its ``infra_error`` to re-raise; on any
    other profile the frame is a hole in this timeline (``record_frame``). Defined
    here, below both the adapter and the backbones, so a classifier can name it
    without importing the adapter."""


# ----------------------------------------------------------------------
# Visual history — the control's own image-retention MECHANISM, not a number.
#
# V2's M3 agent re-sends screenshots on every call (mm_agents/m3/agent.py
# `_build_messages`): the episode's opening frame is always kept, the newest
# RETAINED_IMAGE_BUDGET are kept, and the number dropped is rounded DOWN to a
# multiple of RETAINED_IMAGE_TRUNCATION_THRESHOLD, so the count sawtooths
# 1 -> 30, falls to 11, ramps again (mean ~20.5).
#
# Reproducing the MECHANISM rather than pinning a number is the whole point: a
# fixed 10 would send strictly FEWER images than the control at every step past
# warm-up while being described as parity.
RETAINED_IMAGE_BUDGET = 10          # the control's only_n_most_recent_images
RETAINED_IMAGE_TRUNCATION_THRESHOLD = 20  # the control's image_truncation_threshold

# The oldest step whose frame the arithmetic can still ask for sits this many
# steps behind the current one (`retained_frame_count` peaks at budget + threshold
# - 1 earlier frames, the opening one included). Frames older than that are
# dropped from memory; the opening frame is kept for the whole episode. Derived,
# because a change to either constant must move this with it.
_FRAME_WINDOW_DEPTH = RETAINED_IMAGE_BUDGET + RETAINED_IMAGE_TRUNCATION_THRESHOLD - 1

# Per-step text inside the image window: the reasoning the deciding call wrote at
# that step. The compacted history item cuts it at 240 characters, which cuts a
# typical reasoning text and with it the values a later step needs. The control
# resends every reply whole, thinking included, so this is the direction of parity;
# the bound keeps the worst-case request inside the model's served context at the
# image peak, up to step 500.
REASONING_TEXT_LIMIT = 1500
# Outside the image window a step is ONE line, this long at most, the omitted-
# screenshot marker included — the analogue of the control's 21-character
# "Tool result: Success" placeholder, carrying what was executed and what came of it.
OUT_OF_WINDOW_LINE_LIMIT = 200
# The executed-action text inside the window. A typed paragraph is the long tail;
# the values worth keeping from it belong in the facts ledger. The MOST RECENT step is
# the exception: its step line and the CURRENT SCREEN header carry its executed string
# raw and whole (_raw), so the model can see what it just typed and where that text
# ends — clipped, a long typing read as truncated or finished, and the model
# interrupted it (prompts.OSWORLD_TYPING_NOTE says what the replays showed).
EXECUTED_TEXT_LIMIT = 300
# The action named in an earlier frame's label.
_LABEL_ACTION_LIMIT = 120
# Actions that send the machine nothing: a `type` followed only by these can still be
# typing, so its step line stays raw (EpisodeTimeline._held_type_step). A degraded WAIT
# is recorded as a `wait` and counts, and so does a question to the user (`call_user`,
# a profile's ask channel).
IDLE_ACTIONS = frozenset({"wait", "screenshot", "call_user"})

_OMITTED_MARKER = " [screenshot omitted]"

# How a question's answer is introduced (a profile's ask channel): the Qwen control's
# words (history.py: "User response to your previous request:\n{answer}").
_USER_RESPONSE_LEAD = "User response to your previous request:"

# The facts ledger (the `facts` field of the OSWorld deciding schema): verbatim
# values the model read from a source and will need later — names, numbers, paths,
# rules — accumulated for the whole episode OUTSIDE the orchestrator and shown back
# at every step. It exists because the running graph_summary is rewritten each step
# and loses them: a name or an email address read early leaves the summary while the
# model works on something else, and is later typed as an invented value.
#
# Bounds, applied WHEN AN ENTRY IS RECORDED: its value <= FACT_ENTRY_LIMIT chars and its
# source <= FACT_SOURCE_LIMIT, each on its own — the value never yields room to the
# "(source: …; step N)" suffix: a rule message's deciding clauses can sit past its first
# 150 characters, and a long source would cut them. Long entries are common (a pasted
# record runs far past the bound); clipping them when recorded bounds the ledger, while
# the duplicate checks still compare the unclipped value (record_facts). The rendered
# ledger is bounded by FACTS_RENDER_LIMIT; past that the render selects by CLASS before
# position (see facts_for_prompt), every elision is marked with its count and counted
# for the telemetry. Nothing is ever deleted from the ledger itself.
#
# An entry whose `source` names where it was read (class "source") is kept ahead of one
# that is the model's view of the screen, an action's outcome, its progress or a
# conclusion it derived, and of an unsourced one (all class "other"). A position rule
# alone (keep the earliest and the newest of everything) fills early on a long episode,
# and from then on each new entry is shown for a few calls before the tail drops it —
# whatever it was, a chat-channel rule or a screen state.
#
# The render budget is 32K chars (about 8K tokens of a deciding request near 100K),
# split 16K earliest + 16K newest when the source entries alone exceed it. A smaller
# budget is passed by the source entries alone early in a task that reads many
# records, and from then on a value read after the head has filled stays shown only
# until a short tail moves past it.
FACT_ENTRY_LIMIT = 300
FACT_SOURCE_LIMIT = 100
FACTS_RENDER_LIMIT = 32_000
FACTS_RENDER_HEAD = 16_000
FACTS_RENDER_TAIL = 16_000

# What recording normalises away — render artefacts the model copies back into what it
# records. The render shows an entry as "<value> (source: <source>; step N)", and the
# model echoes that suffix back into values (a string-only entry carries its source
# inside it), which defeats the exact-repeat check: the same fact, re-recorded with its
# rendered suffix, is a new string every step. A source often comes back with
# "; step N" appended too. Only that form is stripped — "; step N", singular,
# one number, as the render writes it (repeated when echoed more than once); a source's
# own ", steps 1-13" or "; steps 11–12" is the model's provenance and is kept.
_ECHOED_SUFFIX = re.compile(r"\s*\(source:\s*(?P<source>[^()]*(?:\([^()]*\)[^()]*)*)\)\s*$", re.I)
_STEP_TAIL = re.compile(r"(?:\s*;\s*step \d+)+\s*$")
_ELISION_MARKER = re.compile(r"^\[\.\.\. .*elided.* \.\.\.\]$")
# What a duplicate check compares, on the casefolded value (a Unicode minus read as "-").
# Key tokens are words and numbers plus what flips a value's meaning without being a word:
# comparison signs, "%", "$", and a minus sign directly before a digit ("employees >= 100"
# vs "< 100", "-500" vs "500", "10%" vs "10"). The SIGNATURE is the ordered sequence of
# those signs and the numbers, with their minus; a near repeat must match it exactly. The
# NEGATIONS are the negating words a value carries; a near repeat must carry the same set.
_TOKEN = re.compile(r"<=|>=|!=|[<>≤≥≠=%$]|(?<![a-z0-9])-\d[a-z0-9]*|[a-z0-9]+")
_SIGNATURE = re.compile(r"<=|>=|!=|[<>≤≥≠=%$]|(?:(?<![a-z0-9])-)?\d+(?:[.,]\d+)*")
_NEGATION = re.compile(r"\b(?:not|no|never|without|none|cannot)\b|n't\b")


def normalise_fact(value: Any, source: Any) -> tuple[str, str]:
    """``(value, source)`` with whitespace collapsed and the render's own artefacts
    removed: every trailing "(source: …)" echoed into the value (its text becomes the
    source when the source field is empty), a trailing "; step N" on either, and an
    echoed elision marker (which normalises to an empty value)."""
    value = " ".join(str(value or "").split())
    source = " ".join(str(source or "").split())
    while (match := _ECHOED_SUFFIX.search(value)):
        if not source:
            source = _STEP_TAIL.sub("", match.group("source")).strip()
        value = value[: match.start()].rstrip()
    if _ELISION_MARKER.match(value):
        value = ""
    return value, _STEP_TAIL.sub("", source).strip()


def _folded(text: str) -> str:
    return text.casefold().replace("\u2212", "-").replace("\u2019", "'")


def _fact_key(value: str) -> str:
    return " ".join(_TOKEN.findall(_folded(value)))


def _is_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """Every token of ``short`` occurs in ``long`` in the same order."""
    remaining = iter(long)
    return all(token in remaining for token in short)


def _source_key(source: str) -> str:
    """The source's part of a duplicate key. The FACTS instruction asks `source` to name
    the record itself, so a record's identity can sit in the source alone:
    "Employees: 120" read from lead-3's record and from lead-9's are two facts."""
    return _fact_key(source)


# How a `source` says the entry is NOT something read from a document, message, email,
# record, file or page, in two strengths. Generic words, no application names.
#
# STRONG: the screen itself or a look at it — "screenshot", "screen" (also "on-screen"), a
# visual observation or inspection, "observation(s)" unless it names a log, and "after"
# an action whose result is UI state (a click, a key press, Ctrl+<key>, Escape, a save, an
# undo). These beat a filename: sources such as "deck.pptx; step 23 screen observation"
# and "deck.pptx … visual observation after first Ctrl+Z" name the file the model is
# LOOKING AT, and are screen readings. "after scroll", "after reload" and "after typing"
# are not on the list: they precede record data a scroll or a search revealed.
#
# WEAK: the model's own derivation, progress or plan, or a UI widget. A FILENAME in the
# source beats these ("Project tracker.xlsx", "Progress report.pdf", "Observation log.csv"
# name files). "my" counts only as "my (own) observation / reasoning / summary / conclusion
# / estimate", so "My Documents/…" is not caught; "decision" is not on the list ("decision
# table in policy.pdf" is a document).
#
# Checked against an independent per-entry classification of a long episode's ledger, it
# classes nearly every entry whose source names a chat channel "source", and most
# screen-state, action-outcome and progress entries "other". Its weak spots: record data
# whose source says "screenshot" (the FACTS instruction asks `source` to name the record
# instead), and a bare host name at the end of a source ("… shows example.com"), which
# reads as a filename.
# THE REQUIREMENT LEDGER (a profile's ``requirement_ledger``: on the DeepSeek profile only). The deciding reply's
# `requirements` field: what the task's result must satisfy, rewritten whole at every step as graph_summary is,
# kept on RuntimeState.content by the orchestrator (the one writer:
# orchestrator.OSWorldOrchestrator._apply_graph_update_from_reasoning) and judged there by the completion gate's
# requirements_unverified rule. The reply's FULL list is stored and judged; only the STATE render is clipped, to
# REQUIREMENTS_RENDER_ITEMS items with a marker that says how many are not shown. Each text field is clipped to
# REQUIREMENT_TEXT_LIMIT when it is stored (the reason field too, so every rendered item is bounded). An item is
# well formed when it has an id and a status of REQUIREMENT_STATUSES; anything else counts as open.
REQUIREMENT_STATUSES = ("open", "verified", "unmet")
REQUIREMENT_FIELDS = ("id", "requirement", "status", "evidence_step", "evidence", "reason")
REQUIREMENT_TEXT_LIMIT = 200
REQUIREMENTS_RENDER_ITEMS = 20
# The block's header says whose ledger it is: a reply without a ledger keeps the previous one (orchestrator.py), so it is
# the most recent reply that carried one, not always the last reply.
_REQUIREMENTS_HEADER = ("REQUIREMENTS (your requirement ledger as your most recent reply carrying one wrote it, one item "
                        "per line):")


def _ledger_text(value: Any) -> str:
    """A ledger text field as stored: a string clipped to REQUIREMENT_TEXT_LIMIT; None is empty; any other value is
    its JSON."""
    if value is None:
        return ""
    return _clip(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), REQUIREMENT_TEXT_LIMIT)


def normalise_requirements(value: Any) -> Optional[list[dict[str, Any]]]:
    """The ledger a deciding reply carried, as it is stored and judged: one dict per item with exactly
    REQUIREMENT_FIELDS, in that order; None when the reply carried no list. An id is a string or an integer (as
    text), else ""; a status is matched stripped and lower-cased; an evidence_step is an integer, else None (a
    numeric string or a float is not one); an item that is not an object is kept as its text, with no id."""
    if not isinstance(value, list):
        return None
    items = []
    for item in value:
        entry = item if isinstance(item, dict) else {"requirement": item}
        raw_id, status, step = entry.get("id"), entry.get("status"), entry.get("evidence_step")
        items.append({
            "id": str(raw_id).strip() if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) else "",
            "requirement": _ledger_text(entry.get("requirement")),
            "status": status.strip().lower() if isinstance(status, str) else "",
            "evidence_step": step if isinstance(step, int) and not isinstance(step, bool) else None,
            "evidence": _ledger_text(entry.get("evidence")),
            "reason": _ledger_text(entry.get("reason")),
        })
    return items


def requirement_well_formed(item: dict[str, Any]) -> bool:
    """An item with an id and a status of REQUIREMENT_STATUSES; anything else counts as open."""
    return bool(item.get("id")) and item.get("status") in REQUIREMENT_STATUSES


def render_requirements(items: Optional[list[dict[str, Any]]]) -> tuple[str, int]:
    """``(text, items not shown)``: the ledger's block in the STATE text, one compact JSON object per item, the first
    REQUIREMENTS_RENDER_ITEMS of them, then "… N more items not shown" when the list is longer."""
    if not items:
        return f"{_REQUIREMENTS_HEADER}\nnone listed yet", 0
    lines = [_REQUIREMENTS_HEADER] + [json.dumps(item, ensure_ascii=False) for item in items[:REQUIREMENTS_RENDER_ITEMS]]
    clipped = max(0, len(items) - REQUIREMENTS_RENDER_ITEMS)
    if clipped:
        lines.append(f"… {clipped} more items not shown")
    return "\n".join(lines), clipped


_SCREEN_OBSERVATION = re.compile(
    r"screenshot|\bscreen\b|\bon-screen\b|\bvisual(?:ly)? (?:observ|inspect)|\bobservations?\b(?!\s+logs?\b)"
    r"|\bafter (?:the |a |an |first |second |another )?(?:click(?:ing|ed)?\b|left_click|press(?:ing|ed)?\b"
    r"|ctrl\+|escape\b|sav(?:e|ed|ing)\b|undo\b)",
    re.I,
)
_NOT_READ_FROM_A_SOURCE = re.compile(
    r"observ|\bderived\b|\binferred\b|cumulative|progress|tracker|reasoning"
    r"|strategy|\bplanning\b|\bmy (?:own )?(?:observation|reasoning|summary|conclusion|estimate)"
    r"|\beffect\b|\bmodal\b|\btoast\b|dropdown|overlay|tooltip|url bar",
    re.I,
)
# A filename: "name.ext" whose name carries no dot and at least one letter, whose extension
# is 2-4 chars led by a letter, and is not followed by ".", "/" or ":". The bare
# "\.\w{2,4}\b" also matches every label of a host name ("cloudcrm.site.hku.icu/leads/…"),
# which would re-class a screen-state entry ("Screenshot at step 123; URL
# cloudcrm.site.hku.icu/…") as "source"; without the letters a decimal ("total 48000.00",
# "$30.0M", "rates 2.5 and 3.14") would read as a filename too.
_FILENAME = re.compile(r"(?<![\w.-])[\w-]*[A-Za-z][\w-]*\.[A-Za-z]\w{1,3}(?![\w./:-])")


def fact_class(source: str) -> str:
    """"source" when ``source`` names where the value was read, else "other". In order: an
    empty source or a STRONG screen marker is "other"; a filename is "source"; a WEAK
    marker is "other"; anything else is "source"."""
    if not source or _SCREEN_OBSERVATION.search(source):
        return "other"
    return "source" if _FILENAME.search(source) or not _NOT_READ_FROM_A_SOURCE.search(source) else "other"


def retained_frame_count(
    frames_seen: int,
    *,
    budget: int = RETAINED_IMAGE_BUDGET,
    threshold: int = RETAINED_IMAGE_TRUNCATION_THRESHOLD,
) -> int:
    """How many EARLIER frames the deciding call re-sends, given how many frames
    the episode has already produced.

    The control's own arithmetic (mm_agents/m3/agent.py `_build_messages`), with
    ``frames_seen`` its ``k``::

        remove = max(0, k - K); remove -= remove % T; kept = k - remove

    The call then carries ``kept`` earlier frames plus the current one, which is the
    control's ``1 + (k - remove)`` — its always-kept opening frame is the first of
    ours. ``remove`` only becomes non-zero at k = K + T, which is what makes the
    profile a sawtooth (1 -> 30, 11, -> 30, 11) rather than a ceiling at K."""
    k = max(0, int(frames_seen))
    remove = max(0, k - budget)
    remove -= remove % threshold
    return k - remove


@dataclass(frozen=True)
class ImageRetention:
    """Which earlier frames a deciding request re-sends: a MODEL PROFILE's policy
    (graphpc_agent.OSWorldModelProfile.image_retention), because each control folds its
    image history with its own function, not with different numbers.

    ``omitted(k)``, at a call with ``k`` steps executed, is ``(n, opening_kept)``: steps
    1..n are rendered in the one-line form with their after-frame not sent, and the
    opening frame is sent iff ``opening_kept``; ``marker`` ends each such line (and the
    step-0 line of an opening frame not sent); ``config`` is what the telemetry records
    as the mechanism."""

    omitted: Callable[[int], tuple[int, bool]]
    marker: str
    config: tuple[tuple[str, Any], ...]


def _m3_omitted(steps_executed: int) -> tuple[int, bool]:
    """M3's: the opening frame always kept and ``retained_frame_count`` earlier frames,
    read at CALL time, so a lever on the module global reaches it."""
    return steps_executed - retained_frame_count(steps_executed), True


M3_IMAGE_RETENTION = ImageRetention(
    omitted=_m3_omitted,
    marker=_OMITTED_MARKER,
    config=(("budget", RETAINED_IMAGE_BUDGET), ("truncation_threshold", RETAINED_IMAGE_TRUNCATION_THRESHOLD)),
)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _raw(text: str) -> str:
    """The most recent step's executed string as recorded: no clip, no whitespace collapsing."""
    return str(text)


@dataclass
class TimelineStep:
    """What the adapter recorded about one executed step. ``executed`` is the
    action as the environment received it (a WAIT says why); the rest arrives on
    later calls: ``screen_change`` when the next ``update_runtime`` closes the step,
    ``label`` when the next deciding call has labelled it."""

    step: int
    executed: str
    reasoning_text: str = ""
    screen_change: Optional[str] = None
    label: Optional[str] = None
    # The text an executed `type` action typed ("" for any other action) — what the
    # orchestrator's named_app_bypassed_by_command_line and synthetic_media_deliverable
    # rules read (orchestrator.py).
    typed_text: str = ""
    # The executed action's type ("type", "wait", "screenshot", …; a degraded WAIT is
    # "wait") — what the held-type rule reads.
    action_type: str = ""
    # A question to the user (a profile's ask channel): the step executed nothing, so
    # its line carries no screen change and no label, and ``user_response`` is the
    # answer the driver brought back, shown in the control's words.
    ask: bool = False
    user_response: Optional[str] = None


class EpisodeTimeline:
    """Per-episode record shared by the adapter (writer) and the OSWorld
    reasoning agent (reader). ``reset`` at every episode (and phase) boundary.
    ``retention`` is the model profile's image retention (M3's by default)."""

    def __init__(self, retention: ImageRetention = M3_IMAGE_RETENTION) -> None:
        self.retention = retention
        self.reset()

    def reset(self) -> None:
        # predict call -> the frame observed at it, or None when the capture failed
        # (a HOLE: rendered as unavailable, never filled by a neighbour). Frames that
        # have left the window are deleted; the opening frame (1) never is.
        self.frames: dict[int, Optional[bytes]] = {}
        self.steps: dict[int, TimelineStep] = {}
        # The facts ledger, append-only: {"step", "value", "source", "class"} per entry
        # (class: fact_class of the source), the value at most FACT_ENTRY_LIMIT chars and
        # the source at most FACT_SOURCE_LIMIT.
        self.facts: list[dict[str, Any]] = []
        # The duplicate checks' indexes: the (value, source) keys held, and per (source key,
        # signature, negations) the token sequences held — the near-repeat rule compares
        # only within one such group.
        self._fact_keys: set[tuple[str, str]] = set()
        self._near_repeat_index: dict[tuple[str, tuple[str, ...], frozenset[str]], list[tuple[str, ...]]] = {}
        # Entries NOT appended, by why: an exact repeat of a held value (after
        # normalisation), a near repeat (see record_facts), or nothing left after it.
        self.facts_dropped = {"repeat": 0, "near_repeat": 0, "empty": 0}
        # How many renders had to elide part of the ledger, the predict call of the first,
        # the most entries one render left out, and the last eliding render's counts kept
        # and elided by class — the telemetry for the overflow rule.
        self.facts_elided_renders = 0
        self.facts_first_elision_step: Optional[int] = None
        self.facts_max_elided = 0
        self.facts_last_render: Optional[dict[str, dict[str, int]]] = None
        # The requirement ledger's renders that left items out (requirements_for_prompt; a profile with the ledger
        # only), and the most items one render left out — telemetry for the render clip.
        self.requirements_clipped_renders = 0
        self.requirements_max_clipped = 0

    # ---- writer API (the adapter) -------------------------------------

    def record_frame(self, predict_call: int, screenshot: Any) -> None:
        """The frame observed at ``predict_call`` — the screen after step
        ``predict_call - 1``. Called once per predict, before the deciding call."""
        frame = bytes(screenshot) if isinstance(screenshot, (bytes, bytearray)) and screenshot else None
        self.frames[int(predict_call)] = frame
        oldest_needed = int(predict_call) - _FRAME_WINDOW_DEPTH
        for stale in [p for p in self.frames if 1 < p < oldest_needed]:
            del self.frames[stale]

    def record_step(self, step: int, *, executed: str, reasoning_text: str = "", typed_text: str = "",
                    action_type: str = "", ask: bool = False) -> None:
        self.steps[int(step)] = TimelineStep(step=int(step), executed=executed, reasoning_text=reasoning_text or "",
                                             typed_text=typed_text, action_type=action_type, ask=ask)

    def record_user_response(self, step: int, answer: str) -> None:
        """The driver's answer to the question asked at ``step``."""
        record = self.steps.get(int(step))
        if record is not None and record.ask:
            record.user_response = answer

    def record_screen_change(self, step: int, monitor_content: Optional[dict[str, Any]]) -> None:
        """The monitor's verdict on step ``step``, as the phrase the step line shows."""
        record = self.steps.get(int(step))
        if record is None or not isinstance(monitor_content, dict):
            return
        pixels = monitor_content.get("diff_pixel_count")
        if monitor_content.get("signal") != "screenshot":
            record.screen_change = "screen change unknown (no screenshot to compare)"
        elif monitor_content.get("changed"):
            record.screen_change = f"screen changed ({pixels} px)"
        elif not pixels:
            record.screen_change = "no screen change"
        else:
            # Pixels did move, under the floor the monitor counts as a change (a
            # short string typed into a small field can): say so rather than "no
            # change", which the model would read as "nothing happened".
            record.screen_change = (f"change below the {SCREENSHOT_CHANGED_MIN_PIXELS}-px detection threshold "
                                    f"({pixels} px)")

    def record_label(self, step: int, verdict: Optional[dict[str, Any]]) -> None:
        """The reconciled effect label of step ``step`` (the orchestrator's
        verification verdict, recorded by the deciding call AFTER that step). When
        the monitor overrode the model's label, the line says so: a claimed effect
        on a screen that did not change is exactly the feedback the reasoner must
        receive."""
        record = self.steps.get(int(step))
        if record is None or not isinstance(verdict, dict):
            return
        recorded = verdict.get("actual_recorded")
        claimed = verdict.get("actual_model")
        if recorded is None:
            return
        label = str(recorded)
        if verdict.get("override") and claimed is not None and claimed != recorded:
            label += f" (you labelled it {claimed}; no change was detected)"
        record.label = label

    def record_facts(self, step: int, facts: Any) -> None:
        """Append the ``facts`` a deciding call emitted at ``step``. Tolerant of the
        shapes an advisory schema gets back (MiniMax does not enforce it): a dict
        entry gives ``value``/``source``, a bare string is a value with no source,
        anything else is skipped.

        Each entry is normalised (``normalise_fact``), then checked, then clipped:
        - an EXACT REPEAT — the same normalised value from the same source, case and
          punctuation aside — is not appended;
        - a NEAR REPEAT is not appended either: it comes from the same source as a held
          entry, carries the same signature (numbers and signs, in order) and the same
          negating words, AND its tokens occur in that entry in the same order — so it
          says nothing the held entry does not. The source is part of both checks because
          it can carry the record's identity (``_source_key``). Each condition keeps a
          kind of distinct fact apart: two records ("Employees: 120" from lead-3 and
          lead-9), a changed number or sign ("< 100" vs ">= 100", "-500" vs "500", "10%"
          vs "10"), a dropped negation ("not paid" vs "paid"), a swapped relation ("Alice
          reports to Bob" vs "Bob reports to Alice"). (A token-overlap threshold would also
          collapse near-identical lines about different records, which is why none is
          used.)
        - the value is clipped to FACT_ENTRY_LIMIT and the source to FACT_SOURCE_LIMIT.
          The duplicate checks see the UNCLIPPED value, so two long records that differ
          only past the clip stay apart.
        A CORRECTION is a new value and is appended. Every drop is counted."""
        if not isinstance(facts, list):
            return
        for item in facts:
            if isinstance(item, dict):
                value, source = normalise_fact(item.get("value"), item.get("source"))
            elif isinstance(item, str):
                value, source = normalise_fact(item, "")
            else:
                continue
            if not value:
                self.facts_dropped["empty"] += 1
                continue
            key = (_fact_key(value), _source_key(source))
            if key in self._fact_keys:
                self.facts_dropped["repeat"] += 1
                continue
            folded = _folded(value)
            tokens = tuple(key[0].split())
            group = self._near_repeat_index.setdefault(
                (key[1], tuple(_SIGNATURE.findall(folded)), frozenset(_NEGATION.findall(folded))), [])
            if any(_is_subsequence(tokens, held) for held in group):
                self.facts_dropped["near_repeat"] += 1
                continue
            self._fact_keys.add(key)
            group.append(tokens)
            source = _clip(source, FACT_SOURCE_LIMIT)
            self.facts.append({"step": int(step), "value": _clip(value, FACT_ENTRY_LIMIT), "source": source,
                               "class": fact_class(source)})

    # ---- reader API (the reasoning agent) -----------------------------

    def facts_for_prompt(self) -> list[str]:
        """The ledger as the state block shows it, one string per entry.

        Within FACTS_RENDER_LIMIT: every entry, earliest first. Over it, the entries read
        from a source come first:
        - if they fit, all of them, then the newest "other" entries that fit in the room
          left, after one marker counting the other entries left out;
        - if they alone exceed the limit, their earliest FACTS_RENDER_HEAD chars and newest
          FACTS_RENDER_TAIL, with one marker counting the source entries elided between,
          and no "other" entry.
        So an overflowing render carries at most two markers, and both say how many."""
        rendered = [
            f"{entry['value']} (source: {entry['source'] or 'unstated'}; step {entry['step']})"
            for entry in self.facts
        ]
        if sum(len(text) for text in rendered) <= FACTS_RENDER_LIMIT:
            return rendered
        source = [i for i, entry in enumerate(self.facts) if entry["class"] == "source"]
        other = [i for i, entry in enumerate(self.facts) if entry["class"] != "source"]
        kept_other: list[int] = []
        if sum(len(rendered[i]) for i in source) <= FACTS_RENDER_LIMIT:
            kept_source, gap = source, None
            room = FACTS_RENDER_LIMIT - sum(len(rendered[i]) for i in source)
            for i in reversed(other):
                if len(rendered[i]) > room:
                    break
                kept_other.insert(0, i)
                room -= len(rendered[i])
        else:
            head, used = [], 0
            for i in source:
                if used + len(rendered[i]) > FACTS_RENDER_HEAD:
                    break
                head.append(i)
                used += len(rendered[i])
            tail, used = [], 0
            for i in reversed(source[len(head):]):
                if used + len(rendered[i]) > FACTS_RENDER_TAIL:
                    break
                tail.insert(0, i)
                used += len(rendered[i])
            kept_source, gap = head + tail, len(head)
        elided_source, elided_other = len(source) - len(kept_source), len(other) - len(kept_other)
        lines = [rendered[i] for i in kept_source]
        if elided_source:
            lines.insert(gap, f"[... {elided_source} facts read from sources elided here: shown are the "
                              f"earliest and the newest ...]")
        if elided_other:
            lines.append(f"[... {elided_other} other facts (not read from a source) elided"
                         + (f"; the newest {len(kept_other)} follow" if kept_other else "") + " ...]")
        lines += [rendered[i] for i in kept_other]
        self.facts_elided_renders += 1
        if self.facts_first_elision_step is None:
            self.facts_first_elision_step = self.current_predict_call
        self.facts_max_elided = max(self.facts_max_elided, elided_source + elided_other)
        self.facts_last_render = {"kept": {"source": len(kept_source), "other": len(kept_other)},
                                  "elided": {"source": elided_source, "other": elided_other}}
        return lines

    def requirements_for_prompt(self, items: Optional[list[dict[str, Any]]]) -> str:
        """The requirement ledger's block in the STATE text (render_requirements), the clip counted."""
        text, clipped = render_requirements(items)
        if clipped:
            self.requirements_clipped_renders += 1
            self.requirements_max_clipped = max(self.requirements_max_clipped, clipped)
        return text

    @property
    def current_predict_call(self) -> int:
        return max(self.frames) if self.frames else 0

    def render(self, *, instruction: str, state_text: str) -> list[dict[str, Any]]:
        """The deciding call's user content: text parts and image parts, the
        current frame last. Image parts are ``{"type": "image", "image": bytes,
        "label": str, "predict_call": int}``; the backbone encodes them."""
        current = self.current_predict_call
        if current < 1:
            raise RuntimeError("the timeline has no frame recorded for this step; the adapter records one per predict")
        k = current - 1  # steps executed so far
        remove, opening_kept = self.retention.omitted(k)  # steps 1..remove have their screenshot omitted
        parts: list[dict[str, Any]] = []

        if k == 0:
            _text(parts, "Step 0 — task start. No action has been taken yet; the CURRENT SCREEN at the end "
                         "of this message is the opening screen.")
        elif opening_kept:
            _text(parts, "Step 0 — task start. The screen before any action:")
            self._frame(parts, 1, "step 0 (task start)")
        else:
            _text(parts, "Step 0 — task start. The screen before any action:" + self.retention.marker)
        _text(parts, f"TASK INSTRUCTION: {instruction}")

        held = self._held_type_step(k)
        for step in range(1, k + 1):
            record = self.steps.get(step)
            if step <= remove:
                _text(parts, self._one_line(step, record))
                continue
            _text(parts, self._full_line(step, record, last=(step == k), whole=step in (k, held)))
            if step < k:
                _text(parts, f"Screen after step {step} ({self._label_action(record)}):")
                self._frame(parts, step + 1, f"after step {step}")

        _text(parts, state_text)
        if k == 0:
            _text(parts, "CURRENT SCREEN, at task start:")
        else:
            latest = self.steps.get(k)
            action = _raw(latest.executed) if latest is not None else "no record"
            _text(parts, f"CURRENT SCREEN, after step {k} ({action}):")
        self._frame(parts, current, "current")
        return parts

    def feedback_line(self) -> Optional[str]:
        """The line the current request carries about the most recent step — what
        the model is told about the action it took last (the per-call record's
        ``feedback_text``). None at task start."""
        k = self.current_predict_call - 1
        return self._full_line(k, self.steps.get(k), last=True, whole=True).split("\n", 1)[0] if k >= 1 else None

    # ---- rendering helpers ---------------------------------------------

    def _frame(self, parts: list[dict[str, Any]], predict_call: int, label: str) -> None:
        if predict_call not in self.frames:
            raise RuntimeError(
                f"frame {predict_call} ({label}) is inside the image window but not held; the window "
                f"arithmetic and the frame pruning disagree"
            )
        frame = self.frames[predict_call]
        if frame is None:
            _text(parts, "[screenshot unavailable: the capture failed at this step]")
            return
        parts.append({"type": "image", "image": frame, "label": label, "predict_call": predict_call})

    @staticmethod
    def _label_action(record: Optional[TimelineStep]) -> str:
        return _clip(record.executed, _LABEL_ACTION_LIMIT) if record is not None else "no record"

    @staticmethod
    def _outcome(record: TimelineStep, *, with_label: bool) -> str:
        pieces = []
        if record.screen_change:
            pieces.append(record.screen_change)
        if with_label and record.label:
            pieces.append(f"your label: {record.label}")
        return "; ".join(pieces)

    def _held_type_step(self, k: int) -> Optional[int]:
        """The most recent `type` step when every step after it, up to step k, is idle
        (IDLE_ACTIONS); else None. Inside the image window its step line is raw; one
        that has left the window keeps its one-line form."""
        for step in range(k, 0, -1):
            record = self.steps.get(step)
            if record is None:
                return None
            if record.action_type == "type":
                return step
            if record.action_type not in IDLE_ACTIONS:
                return None
        return None

    def _full_line(self, step: int, record: Optional[TimelineStep], *, last: bool, whole: bool) -> str:
        """``whole``: the executed string raw (the most recent step, or the held type
        step); else clipped at EXECUTED_TEXT_LIMIT."""
        if record is None:
            return f"Step {step}: no record of what was executed."
        executed = _raw(record.executed) if whole else _clip(record.executed, EXECUTED_TEXT_LIMIT)
        if record.ask:
            # A question executed nothing: no screen change and no label to show.
            line = f"Step {step}: {executed}"
            if record.user_response is not None:
                line += f"\n{_USER_RESPONSE_LEAD}\n{record.user_response}"
        else:
            # The last step's label is being decided in THIS call (actual_effect_class).
            outcome = self._outcome(record, with_label=not last)
            line = f"Step {step}: executed {executed}"
            if outcome:
                line += f"; {outcome}"
        if record.reasoning_text:
            line += f"\nYour reasoning at step {step}: {_clip(record.reasoning_text, REASONING_TEXT_LIMIT)}"
        if last:
            line += f"\nThe screen after step {step} is the CURRENT SCREEN at the end of this message."
        return line

    def _one_line(self, step: int, record: Optional[TimelineStep]) -> str:
        """One line <= OUT_OF_WINDOW_LINE_LIMIT. The outcome — the monitor's verdict
        and the label — is built first and never cut; the executed action gets the
        room that is left, because a typed payload is the long field and the outcome
        is what the step came to."""
        if record is None:
            marker = self.retention.marker
            return _clip(f"Step {step}: no record", OUT_OF_WINDOW_LINE_LIMIT - len(marker)) + marker
        if record.ask:
            text = record.executed
            if record.user_response is not None:
                text += f"; {_USER_RESPONSE_LEAD} {record.user_response}"
            prefix = f"Step {step}: "
            return prefix + _clip(text, max(1, OUT_OF_WINDOW_LINE_LIMIT - len(prefix) - len(self.retention.marker))) \
                + self.retention.marker
        prefix = f"Step {step}: executed "
        outcome = self._outcome(record, with_label=True)
        suffix = (f"; {outcome}" if outcome else "") + self.retention.marker
        room = max(1, OUT_OF_WINDOW_LINE_LIMIT - len(prefix) - len(suffix))
        return prefix + _clip(record.executed, room) + suffix


def _text(parts: list[dict[str, Any]], text: str) -> None:
    """Append a text part, merged into the previous one when that is text too, so
    the request alternates text and images and nothing else."""
    if parts and parts[-1]["type"] == "text":
        parts[-1]["text"] += "\n" + text
    else:
        parts.append({"type": "text", "text": text})


def render_state_text(step: int, state: dict[str, Any], *, budget_line: Optional[str] = None,
                      requirements: Optional[str] = None) -> str:
    """The dynamic state block, placed after the step history and before the
    current screen (the caller appends the schema block after it). A profile's
    step-budget line sits between the header and the JSON, and its requirement
    ledger's block (render_requirements) after the JSON; without them the block is
    the header and the JSON alone (the MiniMax and Qwen profiles)."""
    text = f"STATE BEFORE STEP {step}:\n"
    if budget_line is not None:
        text += budget_line + "\n"
    text += json.dumps(state, indent=2, ensure_ascii=False)
    if requirements is not None:
        text += "\n" + requirements
    return text
