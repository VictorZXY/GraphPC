"""OSWorld 2.0 reasoning prompt text.

Mirrors upstream's own file layout (``mm_agents/m3/prompts.py``): the prompt
text a benchmark's runs are made of lives beside that benchmark's adapter, not
in shared agent code. ``action_minimax.py`` already holds the grounding half
(``M3_GROUNDING_SYSTEM_PROMPT``) with the same provenance discipline; this is
the reasoning half.

Holds each model profile's deciding-call text (MiniMax-M3's first, then the
DeepSeek profile's two method texts, then Qwen's).
``OSWorldGraphPCAgent.build_default`` passes the profile's suffix to the
OSWorld reasoning agent (``reasoning.py:OSWorldReasoningAgent``, a
``GraphReasoningAgent``) as ``deciding_instruction_suffix`` AT CONSTRUCTION (the
text is invariant for a whole run, unlike ``environment_brief``, which is a
per-step observation key), and nothing else passes it anywhere. The shared
default in ``src/agents/reasoning/base.py`` is untouched, so this text cannot
execute on WebArena: it is class two (design.md §10), scoped to OSWorld.

NAMING. The parameter and this constant are named for WHAT THEY HOLD: the block
appended to every DECIDING-mode instruction, which here carries general operating
rules (upstream :31, :32) beside the terminal content — a name that says "terminal
claim" over a block containing "do not ask the user for clarification" would be a
name that lies. The shared default keeps its narrower name: on WebArena the block
holds the terminal-claim contract and nothing else, so ``TERMINAL_CLAIM_CONTRACT``
is accurate there.
"""

from __future__ import annotations


# WHY A BENCHMARK-SCOPED VARIANT EXISTS AT ALL.
# The shared contract offers the infeasibility escape in one clause — "If the
# objective is genuinely impossible/unachievable, decide `done` with a brief
# statement to that effect" — with nothing around it. On WebArena that clause
# rarely fires; on OSWorld 2.0's long tasks, with nothing around it, it invites a
# give-up. The control arm is upstream's `mm_agents/m3`, UNMODIFIED, and its
# system prompt carries guardrails the shared contract lacks
# (mm_agents/m3/prompts.py:31-45). So the escape rate is not a property of the
# benchmark: it is the symmetric difference between the two prompts.
#
# THE RULE THIS CONSTANT IS BUILT UNDER: MATCH, DON'T EXCEED. A delta is a
# difference between two arms, so any instruction BOTH arms carry cancels out of
# it. Adding a clause whose FUNCTION the control already has shrinks the
# symmetric difference — confound removal, legitimate. Adding a clause the
# control lacks GROWS it in our favour — tuning past the control, not allowed.
# Deleting one of ours with no counterpart in the control shrinks it too —
# legitimate. Every clause below therefore names the upstream line it matches,
# so the rule is auditable after the fact rather than asserted once in a review.
#
# PROVENANCE — OSWorld-V2/mm_agents/m3/prompts.py, `M3_SYSTEM_PROMPT_TEMPLATE`.
#   CARRIED, <SYSTEM_CAPABILITY>: :31 (see-it-all before concluding absence), :32
#             (never ask the user, always act), :34-39 (the salience header, the
#             "intentionally impossible" claim, the when-to-declare permission, the
#             five infeasibility criteria), :40 (the one recognised signal), :42
#             (`done` is never the answer to an impossible task), :43 (a prose
#             refusal is not detected), :44 and :45.
#   CARRIED, <TOOLS>: :72 (`done` declares success, after you are confident the
#             task is fully done) and :77 (the final step is `done` once there is
#             nothing more to do). Our deciding call never saw the control's TOOLS
#             block — the grounding prompt strips its termination content
#             (action_minimax.py) because termination is the reasoning call's — so
#             the control's only `done` cue reaches the deciding call here or not at
#             all.
#   EXCLUDED: :29-30, :46-48, environment facts — they reach the deciding call
#             through the per-step environment brief (observation_adapter.py), so
#             carrying them here would duplicate, not add.
#   EXCLUDED: :33, "try to chain multiple of these calls all into one function
#             calls request". Not a method difference: the control is one call per
#             turn too (`mm_agents/m3/agent.py` stops at the first `</tool_call>`),
#             so carrying the line would be a match. The exclusion stands on a
#             request-level replay instead: carried verbatim, it drew no script
#             routes, and the replies that echoed it read it as chaining GUI actions
#             inside `intended_actions`.
#   EXCLUDED: :78, the TOOLS block's restatement of :40 in the same words — carried
#             once, not twice.
# The carried clauses are FUNCTIONAL matches in our own wording, with the control's
# signal translated into ours: its `[INFEASIBLE]` token is our `fail` action (the
# reasoning contract's give-up channel, emitted to the environment as V2's `FAIL`
# token), and its `done` tool call is our `done` action.
#
# THE TERMINAL CONTRACT, both directions against the shared one
# (src/agents/reasoning/base.py:TERMINAL_CLAIM_CONTRACT), under the match-don't-exceed
# rule above:
#   DROPPED "decide `done` with a brief statement" for an impossible objective. Two
#           routes to one outcome, one of them the control's opposite: the control
#           grades `done` as success and forbids it for an impossible task (:42), so
#           a `done` that means "I give up" is a signal the control never sends and
#           the adapter would have to decode after the fact. Infeasibility is `fail` alone.
#           `match_infeasible_declaration` stays on `done` in the adapter as a
#           FALLBACK and an audit, not as a route this prompt offers.
#   DROPPED the answer-bearing `done` ("MUST carry the final answer itself", "no
#           answer text" for a state change, "never fabricate an answer"). V2's
#           `DONE` token carries nothing to the evaluator, the control's `done` takes
#           no arguments (:72), and an answer typed into a `done` is discarded — so
#           the clause told the model an answer channel exists where none does. It
#           is ours alone; removing it shrinks the symmetric difference.
#   DROPPED "termination is YOUR explicit decision" and "— do not run out the step
#           budget". Upstream says nothing about the step budget, and the function
#           of the first — cue the terminal action explicitly — is :77's, carried in
#           the control's words.
#   KEPT    the "TERMINAL-CLAIM CONTRACT:" label, a section heading with no
#           instruction in it.
#
# THE CLAUSE WORTH RECORDING. "Some tasks are intentionally impossible" is a
# claim about the benchmark rather than about agent behaviour, and it points the
# model at exactly the behaviour this variant exists to suppress. It is NOT
# accurate for this subset: none of the 28 subset tasks is scored through the
# evaluator's infeasible path (task 010's class imports `evaluate_infeasible_task` as
# generated boilerplate and never calls it — it scores a `compare_docx_files` metric —
# and no other subset class mentions the infeasible path at all). It is carried because
# the control carries it, and symmetry is the whole rule: dropping a control clause
# that happens to work against us would be exceeding the control in the other
# direction. The same holds for :34's "at any point" permission, which is carried
# whole rather than trimmed.
OSWORLD_DECIDING_INSTRUCTION_SUFFIX = (
    # upstream :72 and :77 — `done` is a success declaration, made last, once
    # confident, when nothing is left to do.
    "TERMINAL-CLAIM CONTRACT: `done` declares that the task has been completed "
    "successfully; decide it as the final action after you are confident the task is "
    "fully done. When the task has been completed and there is nothing more to do, "
    "your final step must be the `done` action, to signal completion. "
    # OPERATING RULE 1 — upstream :31. Look at all of it before concluding that
    # something is not there: a give-up that concludes absence from a partial view
    # (the same menu opened again and again, a different item clicked each time; "no
    # typed text ever appeared") is the failure it addresses.
    "OPERATING RULES: when viewing a page or window it can help to zoom out so that you "
    "can see all of it; either that, or scroll through it to see everything before "
    "deciding that something is not available. "
    # OPERATING RULE 2 — upstream :32, a STANDING instruction, not scoped to the
    # infeasibility case: a declaration such as "If you'd like, I can instead draft
    # the filled-in answers as a plain-text or .docx file" is what this rule forbids.
    "Do NOT ask the user for clarification during task execution, and do NOT stop to "
    "request more information from the user — always take action with the tools you "
    "have. "
    # upstream :34, first sentence. The salience header: the control tells the
    # model this is a frequent failure point BEFORE it reads the criteria.
    "TASK FEASIBILITY — this is a frequent failure point, read carefully. "
    # upstream :34 (remainder) and :35-39. The "intentionally impossible" claim, the
    # when-to-declare permission, and the five criteria, which turn a one-word test
    # ("impossible") into five things the model has to be able to point at. See the
    # clause note above for why this is carried although it is untrue of the subset.
    "Some tasks are intentionally impossible. You may conclude that a task is "
    "infeasible at any point: immediately after the first screenshot, or later after "
    "attempting actions and hitting a hard barrier. A task is infeasible when it cannot "
    "be completed due to: missing required applications or dependencies that cannot be "
    "installed; insufficient permissions or system limitations; contradictory, "
    "fictional, or impossible requirements; the task requiring a specific application "
    "that you have verified does not provide the required feature or capability; or any "
    "other fundamental barrier that makes completion impossible. "
    # upstream :40 — the one signal, with `fail` for `[INFEASIBLE]`. The brief
    # statement rides in the action's `text` (MiniMaxActionAgent._terminal_action
    # reads it there) so the record says why.
    "When you conclude a task is infeasible, you MUST decide the `fail` action, with a "
    "brief statement of the barrier in its `text`, in that same step — `fail` is the ONLY "
    "signal the system recognizes for an impossible task. "
    # upstream :42 — the route this contract dropped, closed in the control's words.
    "When a task is infeasible, do NOT decide the `done` action: `done` means \"I have "
    "successfully completed the task\" and will be graded as WRONG for an impossible "
    "task. "
    # upstream :43 — a declaration in mixed prose that offers alternatives, a refusal
    # that lives only in the reasoning, reaches nothing.
    "Do NOT only explain in your reasoning that it is not possible — a prose refusal is "
    "NOT detected; only the `fail` action counts. "
    # upstream :44, with `fail` for the literal token.
    "Do NOT propose workarounds or alternatives, and do NOT ask the user what they "
    "prefer — decide yourself, decide `fail`, and stop. "
    # upstream :45. The core anti-give-up clause.
    "Only declare a task infeasible when you are genuinely confident it is impossible. "
    "Do NOT give up on a task that is merely difficult, slow, or unfamiliar — try "
    "reasonable approaches first."
)

# How a long `type` behaves on this benchmark: the last sentence of the environment brief
# (observation_adapter.py, from build_default), so it ends the ENVIRONMENT paragraph of the
# deciding system prompt, after "The current date is …." The driver types at
# about 0.1 s per character and a step returns after about 90 s, so a long text is still
# being typed when the next screen is captured; with the text shown whole (timeline.py,
# the most recent step), the model can tell unfinished typing from a finished command.
# In replays at states where a long text was still being typed, the raw full text
# together with this sentence made the model wait or look at the screen rather than
# interrupt the typing (with Ctrl+C); either one alone rarely did.
OSWORLD_TYPING_NOTE = (
    "Environment note: a `type` action is carried out as simulated keystrokes at about 0.1 s per "
    "character; the step's time limit of about 90 s does not cut it short — when a step returns "
    "before its typing has finished, the remaining keystrokes continue on their own, so a long text "
    "can still be being typed when the next screen is captured."
)


# ----------------------------------------------------------------------
# The DeepSeek profile's own deciding-call text. Neither is a control clause: they are GraphPC method surface, read
# only where the profile sets its fields (graphpc_agent.OSWorldModelProfile requirement_ledger / step_budget_line, on
# the DeepSeek profile only), so the MiniMax and Qwen requests carry neither.
#
# THE REQUIREMENT LEDGER. The deciding reply's `requirements` field (timeline.py: the ledger's bounds and render;
# orchestrator.py: the requirements_unverified rule that judges it). A success claim can leave one requirement
# unchecked as its consumer reads it, and the completion gate's other rules cannot see that: they test a recent screen
# change and the give-up wording, never a requirement. This lifts predict/verify from the step to the task. Its
# own block of the system prompt, after the facts paragraph, NOT in the deciding suffix (which is matched to the
# control's prompt). The sentence defining `evidence_step` is the one the gate applies: an executed step k at or before
# P-1, and not before the step at which the reply last listed the item open.
OSWORLD_REQUIREMENTS_INSTRUCTION = (
    "REQUIREMENTS: keep a requirement ledger in `requirements`, rewritten in full at every step as `graph_summary` "
    "is, and shown back to you under STATE. At the first step, list every requirement the task instruction states: "
    "each deliverable, its path or name, its format, its values, the application it must be made with, and each "
    "constraint. Add the requirements you find later in briefs, reference or example files and on screen when you "
    "read them, including the formats an example shows. One item per requirement, at most 20 items, each with `id` "
    "(a short name kept for the whole task), `requirement` (at most 200 characters), `status` (`open`, `verified` or "
    "`unmet`), `evidence_step`, `evidence` (at most 200 characters) and `reason`. A requirement is `verified` only by "
    "an observation, made after the last change that could affect it, of the deliverable as its consumer reads it: "
    "reopen it in the application it is meant for, or read the saved file back and compare it with the requirement; "
    "record that step as `evidence_step` and what it showed as `evidence`. That an action had its predicted effect is "
    "not verification. `evidence_step` is the number k of an executed step after which the screen showed the evidence, "
    "the \"Screen after step k\" you read it on; under STATE BEFORE STEP P, the CURRENT SCREEN is the screen after step "
    "P-1. When you change something, set the requirements it can affect back to `open`. A requirement you have found "
    "cannot be met is `unmet`, with `reason` saying why and `evidence_step` the step whose screen showed it. Keep every "
    "id you have listed: an id missing from the ledger counts as open, and so does an item whose `evidence_step` is "
    "before the step at which you last listed it as open. `done` is accepted only when every requirement is `verified` "
    "or `unmet`. `facts` keeps the values you read, verbatim; the ledger says what the result must satisfy."
)

# THE STEP BUDGET LINE. Without a stated budget the model reasons about step cost against a budget it imagines, and
# skips a read or a check to save steps it does not need to save. The true budget, stated in one line of the STATE
# block at every step: upstream's own wording, copied from the parametrised form of
# OSWorld-V2 mm_agents/gta1/gta15_agent.py:321 (used at :571; prompts.py:1330 hard-codes "100", so it is not the one
# copied), without its tool-result lead "Action (tool call) is executed." `{step}` is the P of "STATE BEFORE STEP P"
# and `{max_steps}` the arm's budget (OSWORLD_MAX_STEPS_BY_BACKBONE; runner delta 15 refuses any other).
OSWORLD_STEP_BUDGET_LINE = (
    "For your reference, you have maximum of {max_steps} steps, and current step is {step} out of {max_steps}."
)


# ----------------------------------------------------------------------
# The Qwen profile's deciding-call text (the Qwen control is
# osworld-v2.1:mm_agents/qwen; its prompt is qwen_upstream/prompts.py).
#
# THE SAME RULE: MATCH, DON'T EXCEED — and here it cuts the other way from M3. The
# Qwen control's prompt has NO infeasibility section, no "do not ask the user" clause
# and no environment facts but the date, so the suffix carries none of them; `fail`
# stays in the vocabulary. Each clause is the
# control's, and each of the control's clauses is in exactly ONE of the two GraphPC
# calls, by function (action_qwen.py has the grounding half):
#   `terminate` — "End the task with success or failure." (prompts.py
#       ACTION_DESCRIPTIONS), split over our two terminal actions;
#   Rules :136 "Do not use macros, scripts, terminals, or application APIs." and :137
#       "Verify the visible or functional result before terminating.", verbatim;
#   `call_user` — "Ask the user for required information or confirmation.", with the
#       schema's `text` parameter ("Text for type or call_user.") as where the
#       question goes;
#   Rules :139 "Old screenshots may be replaced by: This screenshot has been collapsed
#       to save context.", verbatim — the timeline's retention for this profile uses
#       that text for a folded frame (timeline.QWEN_IMAGE_RETENTION).
# The two section labels are MiniMax's, headings with no instruction in them.
QWEN_DECIDING_INSTRUCTION_SUFFIX = (
    "TERMINAL-CLAIM CONTRACT: `done` ends the task with success and `fail` ends it with failure. "
    "Verify the visible or functional result before terminating. "
    "OPERATING RULES: Do not use macros, scripts, terminals, or application APIs. "
    "`call_user` asks the user for required information or confirmation, with the question in its `text`. "
    "Old screenshots may be replaced by: This screenshot has been collapsed to save context."
)

# The repair of a deciding reply that could not be used: the
# Qwen control's repair turn (recovery.py:repair_payload, "Your
# previous computer_use response
# could not execute: {error}. Return a corrected complete XML tool call with all required
# parameters.") with the deciding call's object in place of the tool call. {error} is
# DETERMINISTIC by construction, since it is request bytes: a fixed sentence for an
# unparseable reply (the parser's own message carries the raw reply and an object
# address); for a reply of the wrong shape, the shape check's sentence up to the repr of
# the offending value (a set parsed by the literal fallback reprs in hash order); for a
# reply with no decision, its own message, which names only the kind.
QWEN_DECIDING_REPAIR = (
    "Your previous response could not be used: {error}. Return a corrected complete JSON object with all "
    "required fields."
)
UNPARSEABLE_REPLY_ERROR = "the reply did not contain a parseable JSON object"
# The shape check's sentence (src/agents/reasoning/base.py:_shape_message) ends with the
# value's repr in parentheses after this phrase.
_SHAPE_SENTENCE_END = "not the contracted container"


def qwen_deciding_repair(exc: Exception) -> str:
    """The repair turn for a rejected deciding reply (``reasoning.TimelineRoute``
    attaches the rejection it recorded as ``exc.rejection``)."""
    kind = (getattr(exc, "rejection", None) or {}).get("kind")
    message = str(exc)
    if kind == "wrong_shape" and _SHAPE_SENTENCE_END in message:
        error = message[: message.index(_SHAPE_SENTENCE_END) + len(_SHAPE_SENTENCE_END)]
    elif kind in ("schema_echo", "native_tool_call", "other"):
        error = message
    else:
        error = UNPARSEABLE_REPLY_ERROR
    return QWEN_DECIDING_REPAIR.format(error=error)
