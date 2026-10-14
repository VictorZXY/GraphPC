"""OSWorld 2.0's orchestrator: ``GraphPCOrchestrator`` with five more completion-gate
rules, each read off what the episode RECORDED (the terminal's statement, the commands it
typed, the instruction's own words) rather than the model's own verdict.

WHERE IT SITS. It overrides ``_completion_admits`` and calls the shared rules
first, so every shared block stands and the rules below can only add blocks. A
block is the shared gate's: a corrective noop, and the one two-strike budget both
terminal forms share (``_MAX_BLOCKED_DONE_CORRECTIONS``) — after two blocks the next
terminal stands. ``build_default`` is the only place it is constructed; WebArena
never builds it, so nothing here can reach a WebArena request.

THE RULES. Each records its fact on ``state.content["_completion_gate_facts"]`` (the
decision's runtime state), which the adapter renders as the blocked step's line —
the fact, never the step cap or the steps remaining. (On the MiniMax and Qwen profiles
no request shows the budget; on the DeepSeek profile the STATE block states it in one
line, its ``step_budget_line``, and a blocked step's line still never does.)

ONE MORE RULE, ON A PROFILE WITH THE REQUIREMENT LEDGER (``requirement_ledger``: on the
DeepSeek profile only), with its OWN two-strike count:

  requirements_unverified (REQUIREMENTS_UNVERIFIED) — examined after every rule above,
    on a success `done` only (never a give-up, by either route), through an override of
    ``_apply_completion_gate``: blocks the claim when the claim's own reply carried no
    ledger, when the ledger is empty, or when any item is not admitted. An item is
    admitted when it is well formed (an id, a status of open / verified / unmet) and
    either `verified` with an ``evidence_step`` k, or `unmet` with a non-empty ``reason``
    and such a k, where k is an executed step of the timeline at or before P-1 (P the
    claim's predict call; never ``state.step_idx``) and k >= j, j the last predict call
    whose reply listed the item open (absent from a reply counts as open). An id listed in
    any earlier reply and absent from the claim's ledger is not admitted either. After two
    blocks by THIS rule the next claim stands; its count is apart from the shared one, so
    neither switches the other off (the give-up rules keep theirs). The rule is
    structural: whether an evidence is real is the model's; the gate refuses only a claim
    the model's own ledger contradicts. Each terminal claim's ledger counts and outcome are
    recorded (``_requirements_claims``).

  fail_rests_on_step_budget (FAIL_RESTS_ON_STEP_BUDGET) — a give-up whose statement
    rests on the step budget, or on how many steps the remaining work would take, or on
    how many operations, edits or clicks it requires, while fewer than
    FAIL_BUDGET_RULE_BEFORE of the episode's steps are used. A give-up is a `fail`, or a
    `done` whose statement the infeasibility matcher accepts (the fallback route the
    adapter converts to FAIL), so blocking one route does not leave the other open; a
    success `done` the matcher does not accept is never examined by this rule. The
    contract's infeasibility conditions are barriers (a missing application, permissions,
    contradictory requirements, a verified missing feature); a task being long is not
    one. Typical statements: "infeasible … within the available step budget", "the GUI
    approach would require 150-300+ steps", "within a practical step budget", "exceeds the
    remaining action budget"; in work units, "the task requires authoring ~40-50 distinct
    OOXML operations" — the same claim, that the task is long. Post-hoc: the work-unit
    form was written from observed failure transcripts.
  unverified_barrier (UNVERIFIED_BARRIER) — a give-up (the same two routes, the same
    FAIL_BUDGET_RULE_BEFORE line) whose statement itself concedes that its barrier is
    unverified, or a matter of reliability or risk: the contract's conditions are
    barriers "you have verified", after trying reasonable approaches first. Examined
    after fail_rests_on_step_budget, so a statement carrying both records
    fail_rests_on_step_budget. Typical statements: "prevents reliable delivery", "lxml
    direct OOXML effectLst XML injection is fragile and WPS Presentation rendering of
    injected effect XML is unverified". Post-hoc: the lexicon (_UNVERIFIED_BARRIER) was
    written from observed failure transcripts.
  named_app_bypassed_by_command_line (NAMED_APP_BYPASSED_BY_COMMAND_LINE) — blocks a
    success claim when a media CLI (the listed ones, MEDIA_CLIS) transformed an
    instruction-named input directly into an instruction-named deliverable, while the
    instruction directs the use of an app from the list (MEDIA_APPLICATIONS): "use /
    using / in / with <App>", so the line's "the instruction says to use <App>" is a
    detected fact. One hop, in a command the episode typed: e.g. "Please use Shotcut to
    edit the video…" answered by an `ffmpeg` command writing the named output from the
    named input. Post-hoc: the rule and its input condition were written from observed
    failure transcripts.
  use_file_claimed_done (USE_FILE_CLAIMED_DONE) — examined after
    named_app_bypassed_by_command_line — blocks a success
    claim whose statement names, as the thing completed, a file the instruction gives as
    material to use ("use Guidelines.docx and proposal.docx to complete my application"),
    anchored on a completion verb ("proposal.docx has been successfully populated"), while
    the instruction says what to complete ("to complete my application"); an incidental
    mention ("filled the portal using proposal.docx values") is not a claim. Post-hoc:
    written from a single observed failure transcript, its blocked-step line chosen by replay.
  synthetic_media_deliverable (SYNTHETIC_MEDIA_DELIVERABLE) — examined last — blocks a
    success claim when the instruction directs a listed application and the LAST typed
    command that wrote a media file outside /tmp was a media CLI (SYNTHESIS_TOOLS) whose
    every input is synthetic: a PIL image drawn on Image.new in a typed text that reads no
    real image or video, an ImageMagick canvas operand, or an ffmpeg lavfi source (e.g. a
    logo the instruction says to create in Blender, drawn with PIL into /tmp and looped by
    ffmpeg into the deliverable video). Post-hoc: written from a single observed failure
    transcript. Its limit: a GUI or bpy render written straight to the final file, or an
    ffmpeg a Python script runs, is invisible to command parsing, so a corrected done can still be blocked with a
    stale fact — bounded by the two strikes. And synthesis does not propagate: a synthetic
    file re-encoded once through /tmp before the final write is not caught (a false
    negative, which never blocks a sound claim).
"""

from __future__ import annotations

import posixpath
import re
import shlex

from typing import Any, Callable, NamedTuple, Optional

from src.agents.common.schema import Action, ActionBatch, AgentMessage, Observation, RuntimeState
from src.agents.graphpc.orchestrator import _MAX_BLOCKED_DONE_CORRECTIONS, GraphPCOrchestrator

from .timeline import EpisodeTimeline, normalise_requirements, requirement_well_formed

# The rules' names: in the gate's failed-rule field and the recorded facts.
FAIL_RESTS_ON_STEP_BUDGET = "fail_rests_on_step_budget"
UNVERIFIED_BARRIER = "unverified_barrier"
NAMED_APP_BYPASSED_BY_COMMAND_LINE = "named_app_bypassed_by_command_line"
USE_FILE_CLAIMED_DONE = "use_file_claimed_done"
SYNTHETIC_MEDIA_DELIVERABLE = "synthetic_media_deliverable"
# Every rule this orchestrator can block with, as the episode telemetry states them
# (active_config "completion_gate_rules").
OSWORLD_COMPLETION_GATE_RULES = (FAIL_RESTS_ON_STEP_BUDGET, UNVERIFIED_BARRIER,
                                 NAMED_APP_BYPASSED_BY_COMMAND_LINE, USE_FILE_CLAIMED_DONE,
                                 SYNTHETIC_MEDIA_DELIVERABLE)
# The requirement ledger's rule: one more rule on a profile with the ledger only, listed in that profile's
# active_config beside the five above and never in the others'.
REQUIREMENTS_UNVERIFIED = "requirements_unverified"
# Its own two-strike count: after this many blocks by it in one episode the next success claim stands (the shared
# gate's number, counted apart from the shared count).
REQUIREMENTS_STRIKES = _MAX_BLOCKED_DONE_CORRECTIONS

# The share of the episode's steps below which the give-up rules
# (fail_rests_on_step_budget, unverified_barrier) block.
FAIL_BUDGET_RULE_BEFORE = 0.8

# A give-up statement resting on the number of steps: the budget itself, the steps (or
# actions) left, or how many steps the work would take. One claim, in its synonyms: "action"
# stands for "step" ("exceeds the remaining action budget"). A
# history report ("After 225 actions", "after 180+ steps") is a different claim and is not here.
# A count as both patterns read it: an optional qualifier, a number or range, an optional "+".
_QUALIFIED_COUNT = (r"(?:about |roughly |approximately |at least |over |more than |~)?"
                    r"\d[\d,]*(?: ?(?:-|–|to) ?\d[\d,]*)?\+?")
_STEP_BUDGET = re.compile(
    r"\b(?:step|action) budget\b"
    r"|\b(?:remaining|available|allotted|allowed|limited) (?:number of )?(?:steps|actions)\b"
    r"|\bsteps? (?:remaining|left|available)\b"
    r"|\bactions? (?:remaining|left)\b"
    r"|\b(?:would|will|could|might) (?:require|take|need) " + _QUALIFIED_COUNT
    + r" (?:more |additional |further )?steps\b"
    r"|\b(?:not enough|insufficient|too few) steps\b"
    r"|\b(?:run|running|ran) out of steps\b"
    r"|\bstep (?:limit|cap)\b",
    re.I,
)
# The same claim counted in work units: how many operations, edits or clicks the work
# requires ("requires authoring ~40-50 distinct OOXML operations"). Searched after
# _STEP_BUDGET, so a statement carrying both records its step phrase.
_WORK_UNIT_COUNT = re.compile(
    r"\b(?:(?:would|will) )?(?:requires?|required|needs?|take)(?: \w+ing)? " + _QUALIFIED_COUNT
    + r"(?: [\w/-]+){0,3} (?:operations|edits|clicks)\b",
    re.I,
)
# A give-up statement conceding that its barrier is unverified, or a matter of reliability
# or risk. "reliable / reliably" counts only negated (a negation up to three words before
# it): a server that "reliably rejects every upload" is a verified, tried-and-failed
# barrier, the opposite claim. "would / may / might" are deliberately absent: a hedged
# prediction is not a concession that the barrier is unverified.
_UNVERIFIED_BARRIER = re.compile(
    r"\b(?:fragile|unverified|unreliabl(?:e|y)"
    r"|(?:not|cannot|can't|couldn't|could not|no|never|without|unable to|prevent(?:s|ed|ing)?)\W+(?:\w+\W+){0,3}?"
    r"reliabl(?:e|y)"
    r"|risk(?:y|s)?|imprecise)\b",
    re.I,
)

# Applications whose work a deliverable is meant to come from.
MEDIA_APPLICATIONS = ("Shotcut", "REAPER", "Blender", "FreeCAD", "KiCad", "GIMP", "Impress")
# The instruction directing an application's use: "use / using / in / with <App>"
# ("LibreOffice Impress" is Impress; "impress" in lower case is the English word).
_DIRECTED_APPLICATION = re.compile(
    r"\b(?:use|using|in|with)\s+(?:LibreOffice\s+)?(" + "|".join(MEDIA_APPLICATIONS) + r")\b", re.I)
# Command-line tools that write media files without the named application.
MEDIA_CLIS = frozenset({"ffmpeg", "avconv", "sox", "convert", "magick", "gst-launch-1.0", "gst-launch",
                        "handbrakecli", "handbrake-cli", "mencoder"})
# A file path or name as an instruction writes it: an absolute or ~ path, or a bare
# "name.ext" with a letter in the name and a letter-led extension.
_NAMED_PATH = re.compile(r"(?:~|/)[\w./-]+\.[A-Za-z]\w{1,4}\b|\b[\w-]*[A-Za-z][\w-]*\.[A-Za-z]\w{1,4}\b")
# One piece of shell text: a quoted string (kept whole, so a `;` or `|` inside a quoted
# `-filter_complex` is not a separator), an escaped character, a command separator (group
# 1: a newline, `&&`, `||`, `;`, `|`), a run of plain characters, or any single character
# (an unmatched quote stays a plain character).
_SHELL_PIECE = re.compile(r"""'[^']*'|"(?:\\.|[^"\\])*"|\\.|(\n|&&|\|\||;|\|)|[^'"\\\n;&|]+|.""", re.S)
# A heredoc: its operator's delimiter (quoted or not), the rest of the operator's line
# (group 3) and the body (group 4), up to the delimiter alone on a line.
_HEREDOC = re.compile(r"""(?<!<)<<(?!<)-?[ \t]*(['"]?)(\w+)\1([^\n]*)\n(.*?)\n[ \t]*\2[ \t]*(?=\n|$)""", re.S)
_COMMAND_PREFIXES = frozenset({"sudo", "nohup", "time", "exec", "env", "nice"})

# A document or data file as an instruction or a statement names it.
_DOCUMENT_FILE = r"[\w.~/-]*\w\.(?:docx|xlsx|pptx|pdf|txt|csv|md|json|html)\b"
# The files an instruction gives as material: named right after "use / using / based on /
# refer(ring) to / according to", alone or as a list ("use Guidelines.docx and proposal.docx").
_USE_FILES = re.compile(
    r"\b(?:use|using|based on|refer(?:ring)? to|according to)\s+(?:(?:the|my)\s+)?("
    + _DOCUMENT_FILE + r"(?:\s*(?:,|and|or|&)\s*(?:(?:the|my)\s+)?" + _DOCUMENT_FILE + r")*)",
    re.I,
)
# What the instruction says to complete: "to complete (my|the|a|an) <words>", whose head noun
# is the last word before a preposition, a conjunction or the end of the phrase.
_COMPLETION_OBJECT = re.compile(r"\bto complete (?:(?:my|the|a|an) )?([A-Za-z][\w-]*(?: [A-Za-z][\w-]*){0,2})", re.I)
_PHRASE_STOP = frozenset({"for", "of", "by", "with", "using", "in", "on", "at", "to", "from", "and", "or",
                          "that", "which", "so", "before", "after", "as", "per", "via"})
# Words that end the object phrase without being its noun: pronouns, determiners and
# adverbs ("to complete it", "… the table below"); so do words ending in -ly.
_NOT_A_NOUN = frozenset({"it", "them", "this", "that", "these", "those", "him", "her", "us", "me", "you",
                         "everything", "something", "anything", "all", "each", "both", "one", "ones",
                         "the", "a", "an", "my", "your", "our", "their", "his", "its", "some", "any", "every",
                         "no", "below", "above", "here", "there", "now", "then", "today", "soon", "later",
                         "first", "again", "too", "also", "next", "afterwards", "online", "manually"})
_COMPLETION_VERB = r"(?:populated|updated|completed|filled|edited|modified|written|saved)"

# A media file, by its extension.
_MEDIA_FILE = re.compile(r"\.(?:png|jpe?g|gif|bmp|tiff?|webp|mp4|webm|mkv|mov|avi|wav|mp3|flac|ogg|m4a)$", re.I)
# The media CLIs whose write synthetic_media_deliverable examines.
SYNTHESIS_TOOLS = frozenset({"ffmpeg", "avconv", "convert", "magick", "sox"})
# Synthetic inputs a command draws itself: an ImageMagick canvas pseudo-format operand.
_CANVAS = re.compile(r"^(?:xc|canvas|label|caption|gradient|pattern|plasma):", re.I)
# A PIL image saved to a literal path; a new PIL canvas; a real image or video read.
_PIL_SAVE = re.compile(r"""\.save\(\s*[rRfF]?(['"])([^'"\n]+)\1""")
_PIL_NEW = re.compile(r"\bImage\.new\(")
_REAL_READ = re.compile(r"\bImage\.open\(|\bcv2\.imread\(|\bVideoCapture\(|"
                        r"\b(?:imageio(?:\.v[23])?|iio)\.(?:imread|mimread|get_reader|imiter|imopen)\(")


def step_budget_phrase(statement: str) -> Optional[str]:
    """The phrase by which a give-up statement rests on the step budget or on a count of
    work units, or None. The step pattern is searched first, not the leftmost match of
    either: a statement with "step budget" after "required ~100+ precise GUI operations"
    records "step budget"."""
    for pattern in (_STEP_BUDGET, _WORK_UNIT_COUNT):
        match = pattern.search(statement or "")
        if match:
            return match.group(0)
    return None


def unverified_barrier_word(statement: str) -> Optional[str]:
    """The first word by which a give-up statement concedes its barrier is unverified,
    unreliable or risky, or None."""
    match = _UNVERIFIED_BARRIER.search(statement or "")
    return match.group(0) if match else None


def use_files(instruction: str) -> list[str]:
    """The document or data files the instruction gives as material to use, as written."""
    return [f.group(0) for use in _USE_FILES.finditer(instruction or "")
            for f in re.finditer(_DOCUMENT_FILE, use.group(1), re.I)]


def _noun_head(word: str) -> bool:
    """Whether the object phrase's last word can be its noun (not a pronoun, a determiner or
    an adverb)."""
    return word.lower() not in _NOT_A_NOUN and not word.lower().endswith("ly")


def completion_object(instruction: str) -> Optional[str]:
    """The head noun of what the instruction says to complete, or None — also when the
    phrase ends on a word that cannot be its noun ("to complete it"): the rule then fails
    closed rather than render "not the it"."""
    match = _COMPLETION_OBJECT.search(instruction or "")
    if match is None:
        return None
    words = []
    for word in match.group(1).split():
        if word.lower() in _PHRASE_STOP:
            break
        words.append(word)
    return words[-1] if words and _noun_head(words[-1]) else None


def use_file_claimed_done(instruction: str, statement: str) -> Optional[dict[str, Any]]:
    """The use-file a success statement claims as the thing completed, with the object the
    instruction says to complete; else None. The claim is anchored on a completion verb:
    "<file> (has been|was) (successfully) <verb>", or "<verb> (in|out) (the|my) <file>"."""
    obj = completion_object(instruction)
    if obj is None:
        return None
    claims = []
    for name in use_files(instruction):
        file = r"(?<![\w.])(?:[~\w.-]*/)*" + re.escape(posixpath.basename(name)) + r"(?!\w|\.\w)"
        claim = re.search(file + r"\s+(?:(?:has been|was)\s+)?(?:successfully\s+)?" + _COMPLETION_VERB + r"\b"
                          + r"|\b" + _COMPLETION_VERB + r"(?:\s+(?:in|out))?\s+(?:(?:the|my)\s+)?" + file,
                          statement or "", re.I)
        if claim is not None:
            claims.append((claim.start(), posixpath.basename(name)))
    if not claims:
        return None
    return {"file": min(claims)[1], "object": obj}


def _directed_in_order(instruction: str) -> list[str]:
    """The listed applications the instruction directs the use of, by their listed names, in
    the order it first directs them."""
    canonical = {name.lower(): name for name in MEDIA_APPLICATIONS}
    ordered = []
    for match in _DIRECTED_APPLICATION.finditer(instruction or ""):
        name = canonical[match.group(1).lower()]
        if match.group(1) != "impress" and name not in ordered:
            ordered.append(name)
    return ordered


def directed_applications(instruction: str) -> list[str]:
    """The listed applications the instruction directs the use of, by their listed names."""
    return sorted(_directed_in_order(instruction))


def named_paths(instruction: str) -> list[str]:
    """The file paths and names the instruction names, ``~`` expanded."""
    return sorted({_normalise(m.group(0)) for m in _NAMED_PATH.finditer(instruction or "")})


def _normalise(path: str) -> str:
    path = path.strip().strip("'\"")
    return "/home/user" + path[1:] if path.startswith("~") else path


def _files(tokens: list[str]) -> tuple[list[str], list[str]]:
    """(inputs, outputs) of a media-CLI command, by the tool's own argument rules."""
    tool = posixpath.basename(tokens[0]).lower()
    args = tokens[1:]
    if tool in ("ffmpeg", "avconv"):
        inputs = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "-i"]
        return inputs, [a for i, a in enumerate(args) if not a.startswith("-") and "=" not in a
                        and (i == 0 or args[i - 1] != "-i")]
    if tool in ("sox", "convert", "magick"):
        files = [a for a in args if not a.startswith("-") and _NAMED_PATH.fullmatch(a.strip("'\""))]
        return files[:-1], files[-1:]
    if tool in ("mencoder", "handbrakecli", "handbrake-cli"):
        outputs = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-o", "--output")]
        inputs = [a for i, a in enumerate(args) if not a.startswith("-") and a not in outputs
                  and _NAMED_PATH.fullmatch(a.strip("'\""))]
        return inputs, outputs
    if tool.startswith("gst-launch"):
        locations = [(i, a.split("=", 1)[1]) for i, a in enumerate(args) if a.startswith("location=")]
        sink = next((i for i, a in enumerate(args) if a.endswith("filesink")), len(args))
        return [path for i, path in locations if i < sink], [path for i, path in locations if i > sink][:1]
    return [], []


def _shell_segments(text: str) -> list[tuple[int, str]]:
    """(offset, command) for the commands of a shell text, split at its separators outside
    quotes."""
    segments, current, start = [], [], 0
    for piece in _SHELL_PIECE.finditer(text):
        if piece.group(1) is not None:
            segments.append((start, "".join(current)))
            current, start = [], piece.end()
        else:
            current.append(piece.group(0))
    return segments + [(start, "".join(current))]


def shell_commands(typed_text: str) -> list[tuple[int, list[str]]]:
    """(offset, tokens) for every command of a typed text, in text order, its leading
    prefixes (sudo, env, VAR=…) dropped. A heredoc's body is data to its command (a script,
    a file's content): it is blanked out of the shell text, so its quotes cannot unbalance
    the commands after it, and read as commands on its own at its own offset (a script's
    ffmpeg line writes when the script runs)."""
    commands = []

    def blank(heredoc: re.Match) -> str:
        commands.extend((heredoc.start(4) + offset, tokens) for offset, tokens in shell_commands(heredoc.group(4)))
        return (" " * (heredoc.start(3) - heredoc.start()) + heredoc.group(3)
                + " " * (heredoc.end() - heredoc.end(3)))

    for offset, segment in _shell_segments(_HEREDOC.sub(blank, typed_text or "")):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        while tokens and (tokens[0] in _COMMAND_PREFIXES or re.fullmatch(r"\w+=\S*", tokens[0])):
            tokens = tokens[1:]
        if tokens:
            commands.append((offset, tokens))
    return sorted(commands, key=lambda command: command[0])


def media_cli_writes(typed_text: str) -> list[tuple[str, list[str], list[str]]]:
    """(tool, inputs, outputs) for every media-CLI command in a typed text."""
    writes = []
    for _offset, tokens in shell_commands(typed_text):
        if posixpath.basename(tokens[0]).lower() in MEDIA_CLIS:
            inputs, outputs = _files(tokens)
            writes.append((posixpath.basename(tokens[0]), [_normalise(p) for p in inputs],
                           [_normalise(p) for p in outputs]))
    return writes


def _same_file(path: str, named: str) -> bool:
    """Whether a command's path is a path the instruction names: equal when both are
    absolute, else the same file name (a relative path's directory is unknown)."""
    if path.startswith("/") and named.startswith("/"):
        return path == named
    return posixpath.basename(path) == posixpath.basename(named)


# The definition is deliberately narrow — one hop, media CLIs only, between paths the
# instruction names: a deliverable that `ffmpeg` converts from the named application's own
# render (a file the instruction does not name), or a report hand-written with `cat`
# heredocs, lies outside it. Any widening to non-media writers or to multi-hop lineage
# would block such sound workflows too.
def named_app_bypassed_by_command_line(instruction: str,
                                       typed: list[tuple[int, str]]) -> Optional[dict[str, Any]]:
    """The first (step, tool, path, source) at which a typed media-CLI command wrote a
    path the instruction names from another file the instruction names, on an
    instruction that directs the use of a media application; else None. ``typed`` is
    (step, typed text) in step order."""
    applications = directed_applications(instruction)
    named = named_paths(instruction)
    if not applications or len(named) < 2:
        return None
    for step, text in typed:
        for tool, inputs, outputs in media_cli_writes(text):
            for deliverable in named:
                if not any(_same_file(out, deliverable) for out in outputs):
                    continue
                source = next((name for name in named if not _same_file(deliverable, name)
                               and any(_same_file(src, name) for src in inputs)), None)
                if source is not None:
                    return {"step": step, "tool": tool, "path": deliverable, "source": source,
                            "applications": applications}
    return None


class MediaWrite(NamedTuple):
    """One write of a media file in a typed text."""
    offset: int                 # where it sits in the typed text
    writer: str                 # the command's tool as typed, "blender", or "Python PIL"
    path: str                   # the file written
    inputs: tuple[str, ...]     # what a media CLI read: files, canvas operands, lavfi sources
    drawn: tuple[str, ...]      # the inputs it synthesised itself (canvas operands, lavfi sources)
    synthetic: bool             # the written file is synthetic


def _lavfi_inputs(args: list[str]) -> list[str]:
    """The ffmpeg inputs read through `-f lavfi` (a generated source, not a file)."""
    inputs, lavfi = [], False
    for i, arg in enumerate(args[:-1]):
        if arg == "-f":
            lavfi = args[i + 1] == "lavfi"
        elif arg == "-i":
            if lavfi:
                inputs.append(args[i + 1])
            lavfi = False
    return inputs


def media_writes(typed_text: str) -> list[MediaWrite]:
    """Every write of a media file a typed text makes, in text order: a media CLI's output
    (its canvas operands and lavfi sources are inputs it draws itself), a Blender render's
    -o / --render-output, and a PIL image saved to a literal path (synthetic when the text
    draws on Image.new and reads no real image or video)."""
    writes = []
    for offset, tokens in shell_commands(typed_text):
        tool, args = posixpath.basename(tokens[0]), tokens[1:]
        if tool.lower() in MEDIA_CLIS:
            files, outputs = _files(tokens)
            drawn = [a for a in args if _CANVAS.match(a)] if tool.lower() in ("convert", "magick") \
                else _lavfi_inputs(args) if tool.lower() in ("ffmpeg", "avconv") else []
            inputs = [_normalise(f) for f in files if f not in drawn] + drawn
            for output in outputs:
                if _MEDIA_FILE.search(_normalise(output)):
                    writes.append(MediaWrite(offset, tool, _normalise(output), tuple(inputs), tuple(drawn),
                                             bool(drawn) and len(drawn) == len(inputs)))
        elif tool.lower() == "blender":
            writes += [MediaWrite(offset, "blender", _normalise(args[i + 1]), (), (), False)
                       for i, arg in enumerate(args[:-1]) if arg in ("-o", "--render-output")]
    drawn_text = bool(_PIL_NEW.search(typed_text or "")) and not _REAL_READ.search(typed_text or "")
    for save in _PIL_SAVE.finditer(typed_text or ""):
        if _MEDIA_FILE.search(_normalise(save.group(2))):
            writes.append(MediaWrite(save.start(), "Python PIL", _normalise(save.group(2)), (), (), drawn_text))
    return sorted(writes, key=lambda write: write.offset)


def _producer(write: MediaWrite, source: Optional[str] = None) -> str:
    """Who drew a synthetic file or pseudo-input: PIL, an ImageMagick canvas, a lavfi source."""
    if write.writer == "Python PIL":
        return "Python PIL"
    drawn = source if source is not None else write.drawn[0]
    return "ImageMagick" if _CANVAS.match(drawn) else f"{write.writer} lavfi"


def _every_input_drawn(sources: list) -> bool:
    """Whether a write read something, and only synthetic things (a None source is real)."""
    return bool(sources) and None not in sources


def _examined(writes: list) -> list:
    """The writes outside /tmp the rule examines: the last one."""
    return writes[-1:]


def synthetic_media_deliverable(instruction: str, typed: list[tuple[int, str]]) -> Optional[dict[str, Any]]:
    """The last media write outside /tmp, when it was a SYNTHESIS_TOOLS command whose every
    input was synthetic at that point, on an instruction that directs a listed
    application: (step, tool, path, inputs, and for each input who drew it at which step),
    with the first application the instruction directs; else None. ``typed`` is (step,
    typed text) in step order."""
    applications = _directed_in_order(instruction)
    if not applications:
        return None
    drawn_by: dict[str, tuple[str, int]] = {}
    outside_tmp = []
    for step, text in typed:
        for write in media_writes(text):
            sources = []
            for source in write.inputs:
                if source in write.drawn:
                    sources.append((source, _producer(write, source), step))
                else:
                    known = next((drawn_by[path] for path in drawn_by if _same_file(source, path)), None)
                    sources.append((source, *known) if known else None)
            if write.synthetic:
                drawn_by[write.path] = (_producer(write), step)
            else:
                drawn_by.pop(write.path, None)
            if not write.path.startswith("/tmp/"):
                outside_tmp.append((step, write, sources))
    for step, write, sources in _examined(outside_tmp):
        if write.writer.lower() in SYNTHESIS_TOOLS and _every_input_drawn(sources):
            return {"step": step, "tool": write.writer, "path": write.path, "inputs": list(write.inputs),
                    "drawn": [list(source) for source in sources if source is not None],
                    "application": applications[0]}
    return None


class OSWorldOrchestrator(GraphPCOrchestrator):
    """``GraphPCOrchestrator`` with the five OSWorld completion-gate rules (module
    docstring). ``timeline`` is the episode's record of what was executed (the one
    ``build_default`` hands the adapter and the reasoning agent); ``max_steps`` is the
    episode's step budget, used here only to place the budget rule (the gate never shows
    it; on the DeepSeek profile the reasoning agent's budget line does);
    ``infeasible_declaration_rule`` is the adapter's own test for a `done` it converts to
    FAIL (graphpc_agent.infeasible_declaration_rule), handed over so both read one setting.
    ``requirement_ledger`` (the profile's, off by default; on for DeepSeek only) stores
    each reply's requirement ledger on the runtime state and adds the
    requirements_unverified rule (module docstring); off, nothing here differs."""

    def __init__(self, *args: Any, timeline: EpisodeTimeline, max_steps: int,
                 infeasible_declaration_rule: Callable[[str], Optional[str]], requirement_ledger: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.timeline = timeline
        self.max_steps = int(max_steps)
        self.infeasible_declaration_rule = infeasible_declaration_rule
        self.requirement_ledger = bool(requirement_ledger)

    def _gives_up(self, done: Action, statement: str) -> bool:
        """A give-up by either route: the `fail`, or a `done` the adapter converts to FAIL (its own matcher, handed
        over by build_default)."""
        return (done.metadata or {}).get("terminal_intent") == "fail" \
            or self.infeasible_declaration_rule(statement) is not None

    def _completion_admits(
        self,
        *,
        action_batch: ActionBatch,
        state: RuntimeState,
        current_observation: Optional[Observation],
    ) -> tuple[bool, str]:
        admit, failed_rule = super()._completion_admits(
            action_batch=action_batch, state=state, current_observation=current_observation,
        )
        if not admit:
            return admit, failed_rule
        done = next((a for a in action_batch.actions if a.action_type == "done"), None)
        if done is None:
            return True, ""
        statement = str((done.parameters or {}).get("text") or "")
        if self._gives_up(done, statement):
            if state.step_idx < FAIL_BUDGET_RULE_BEFORE * self.max_steps:
                phrase = step_budget_phrase(statement)
                if phrase is not None:
                    return self._block(state, FAIL_RESTS_ON_STEP_BUDGET, {"phrase": phrase})
                word = unverified_barrier_word(statement)
                if word is not None:
                    return self._block(state, UNVERIFIED_BARRIER, {"word": word})
            return True, ""
        typed = [(step, record.typed_text) for step, record in sorted(self.timeline.steps.items())
                 if record.typed_text]
        fact = named_app_bypassed_by_command_line(state.instruction or "", typed)
        if fact is not None:
            return self._block(state, NAMED_APP_BYPASSED_BY_COMMAND_LINE, fact)
        fact = use_file_claimed_done(state.instruction or "", statement)
        if fact is not None:
            return self._block(state, USE_FILE_CLAIMED_DONE, fact)
        fact = synthetic_media_deliverable(state.instruction or "", typed)
        if fact is not None:
            return self._block(state, SYNTHETIC_MEDIA_DELIVERABLE, fact)
        return True, ""

    @staticmethod
    def _block(state: RuntimeState, rule: str, fact: dict[str, Any]) -> tuple[bool, str]:
        state.content.setdefault("_completion_gate_facts", []).append(
            {"step_idx": state.step_idx, "rule": rule, **fact})
        return False, rule

    # ---- the requirement ledger (a profile's requirement_ledger) ----------------------------------------------------

    def _apply_graph_update_from_reasoning(self, *, state: RuntimeState, reasoning_message: AgentMessage) -> None:
        """The shared write (graph_summary), then, on a profile with the ledger, the reply's ledger onto the runtime
        state BEFORE the completion gate runs, so the gate judges the claim's own ledger (the reasoning agent sets a
        graph_update on every reply there, so this runs on each). Kept on ``state.content``:
          * ``requirements`` — the reply's full list, normalised (timeline.normalise_requirements); a reply with no
            list keeps the previous one, as graph_summary is kept, and is counted
            (``_requirements_replies_without_ledger``); never a reason to re-sample; a degraded step (no usable
            reply: reasoning_parse_error) changes nothing here and is not counted;
          * ``_requirements_call`` — the predict call whose reply wrote it (the gate's test that the claim carried one);
          * ``_requirements_seen`` — every id ever listed, with its latest requirement text;
          * ``_requirements_last_open`` — per id, the last predict call whose reply listed it open, or left it out,
            or gave it a status outside the three (each counts as open);
          * ``_requirements_malformed_items`` — items with no id or a status outside the three, summed over replies."""
        super()._apply_graph_update_from_reasoning(state=state, reasoning_message=reasoning_message)
        if not self.requirement_ledger or reasoning_message.content.get("reasoning_parse_error"):
            return
        ledger = normalise_requirements(reasoning_message.content.get("requirements"))
        if ledger is None:
            state.content["_requirements_replies_without_ledger"] = \
                int(state.content.get("_requirements_replies_without_ledger") or 0) + 1
            return
        call = self.timeline.current_predict_call
        seen = state.content.setdefault("_requirements_seen", {})
        last_open = state.content.setdefault("_requirements_last_open", {})
        listed = set()
        malformed = 0
        for item in ledger:
            if not requirement_well_formed(item):
                malformed += 1
            if not item["id"]:
                continue
            listed.add(item["id"])
            seen[item["id"]] = item["requirement"]
            if item["status"] not in ("verified", "unmet"):
                last_open[item["id"]] = call
        for dropped in set(seen) - listed:
            last_open[dropped] = call
        if malformed:
            state.content["_requirements_malformed_items"] = \
                int(state.content.get("_requirements_malformed_items") or 0) + malformed
        state.content["requirements"] = ledger
        state.content["_requirements_call"] = call

    def requirements_verdict(self, state: RuntimeState) -> dict[str, Any]:
        """The requirements_unverified rule's reading of the ledger on ``state`` at the current predict call P: the
        ledger's counts and, under ``problems``, every item not admitted (module docstring) with why (the kinds the
        adapter renders, graphpc_agent.REQUIREMENT_PROBLEM_TEXTS) — ``ledger`` "missing" when P's reply carried none,
        "empty" when it carried [], else "listed". The claim is admitted when ``ledger`` is "listed" and ``problems``
        is empty."""
        call = self.timeline.current_predict_call
        if state.content.get("_requirements_call") != call:
            return {"ledger": "missing", "items": 0, "verified": 0, "unmet": 0, "open": 0, "problems": []}
        ledger = state.content.get("requirements") or []
        last_open = state.content.get("_requirements_last_open") or {}
        problems: list[dict[str, Any]] = []
        for index, item in enumerate(ledger, 1):
            step, opened = item["evidence_step"], last_open.get(item["id"])
            if not requirement_well_formed(item):
                why = "malformed"
            elif item["status"] == "open":
                why = "open"
            elif item["status"] == "unmet" and not item["reason"]:
                why = "no_reason"
            elif step is None:
                why = "no_evidence_step"
            elif step not in self.timeline.steps or step > call - 1:
                why = "evidence_step"
            elif opened is not None and step < opened:
                why = "stale_evidence"
            else:
                continue
            problems.append({"id": item["id"] or f"#{index}", "requirement": item["requirement"], "why": why,
                             "evidence_step": step, "last_open": opened})
        listed = {item["id"] for item in ledger if item["id"]}
        seen = state.content.get("_requirements_seen") or {}
        problems += [{"id": identifier, "requirement": text, "why": "dropped", "evidence_step": None,
                      "last_open": last_open.get(identifier)}
                     for identifier, text in seen.items() if identifier not in listed]
        return {"ledger": "listed" if ledger else "empty", "items": len(ledger),
                "verified": sum(item["status"] == "verified" for item in ledger),
                "unmet": sum(item["status"] == "unmet" for item in ledger),
                "open": sum(item["status"] == "open" for item in ledger), "problems": problems}

    def _apply_completion_gate(self, *, action_batch: ActionBatch, state: RuntimeState,
                               current_observation: Optional[Observation] = None) -> bool:
        """The shared gate first (every rule above, with the shared two-strike budget); then, on a profile with the
        ledger, the requirements_unverified rule on a success `done` the shared gate did not block, with its own
        two-strike count (``_requirements_blocked_count``). A block is the shared gate's corrective noop and markers,
        built here as the base builds them. Every terminal claim on such a profile is recorded with its ledger's
        counts and its outcome (``_requirements_claims``): blocked by a rule (named), admitted, or stood after this
        rule's two strikes."""
        done = next((a for a in action_batch.actions if a.action_type == "done"), None)
        blocked = super()._apply_completion_gate(action_batch=action_batch, state=state,
                                                 current_observation=current_observation)
        if not self.requirement_ledger or done is None or not self.enable_completion_gate:
            return blocked
        verdict = self.requirements_verdict(state)
        gives_up = self._gives_up(done, str((done.parameters or {}).get("text") or ""))
        claim = {"predict_call": self.timeline.current_predict_call, "step_idx": state.step_idx,
                 "terminal": "fail" if gives_up else "done",
                 **{key: verdict[key] for key in ("ledger", "items", "verified", "unmet", "open")},
                 "not_admitted": len(verdict["problems"]),
                 "dropped": sum(problem["why"] == "dropped" for problem in verdict["problems"])}
        strikes = int(state.content.get("_requirements_blocked_count") or 0)
        admitted = verdict["ledger"] == "listed" and not verdict["problems"]
        if blocked:
            claim["outcome"] = f"blocked:{action_batch.content.get('completion_gate_failed_rule')}"
        elif gives_up or admitted:
            claim["outcome"] = "admitted"
        elif strikes >= REQUIREMENTS_STRIKES:
            claim["outcome"] = "stood_after_two_strikes"
        else:
            claim["outcome"] = f"blocked:{REQUIREMENTS_UNVERIFIED}"
        state.content.setdefault("_requirements_claims", []).append(claim)
        if claim["outcome"] != f"blocked:{REQUIREMENTS_UNVERIFIED}":
            return blocked
        state.content["_requirements_blocked_count"] = strikes + 1
        self._block(state, REQUIREMENTS_UNVERIFIED, {"ledger": verdict["ledger"], "problems": verdict["problems"]})
        action_batch.actions = [
            Action(
                action_type="wait",
                parameters={"webarena_verb": "noop", "command": "noop"},
                raw={"name": "computer", "input": {"command": "noop", "verb": "noop"}},
                metadata={"reason": "completion_gate_corrective_noop", "tool_name": "computer"},
            )
        ]
        action_batch.content = {
            **action_batch.content,
            "completion_gate_blocked": True,
            "completion_gate_failed_rule": REQUIREMENTS_UNVERIFIED,
        }
        return True
