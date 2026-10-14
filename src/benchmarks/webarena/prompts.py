"""WebArena reasoning prompt text.

Mirrors ``src/benchmarks/osworld/prompts.py``: the prompt text a benchmark's runs
are made of lives beside that benchmark's adapter, not in shared agent code.

Holds two constants. ``WebArenaGraphPCAgent.build_default`` passes
``WEBARENA_DECIDING_INSTRUCTION_SUFFIX`` to the shared ``GraphReasoningAgent`` as
``deciding_instruction_suffix`` at construction, at all five backbone call sites,
and nothing else passes it anywhere. The shared default in
``src/agents/reasoning/base.py`` is the contract alone, so this text cannot
execute on OSWorld. That makes it class two (design.md §10).
"""

from __future__ import annotations

from src.agents.reasoning.base import TERMINAL_CLAIM_CONTRACT


# WHY THE CONTROL'S "Homepage" PARAGRAPH IS HERE: MATCH, DON'T EXCEED.
# An instruction both arms carry cancels out of the delta, and one only the control carries
# does not. The control's intro carries this paragraph on every call; without it, nothing
# in ours tells the model where the sites are. It has the same function as the URL
# line: saying where the model is and how to reach the other sites.
#
# PROVENANCE: WebArena agent/prompts/raw/p_cot_id_actree_2s.py:33-35 at
# dce04686a56253aefba7b18a4fa0937cf1dc987b. This is the "Homepage:" header line and its two
# lines, joined by "\n" as upstream writes them, with no trailing newline. The live launch
# preflight asserts that it is still byte-equal to the checkout's text.
#
# WHAT EACH SENTENCE DOES IN THE TWO ARMS:
#   * Sentence 1 is live. The homepage lists the sites, and its links point at the real
#     hosts. `http://homepage.com` maps to the served homepage (url_mappings.py).
#   * Sentence 2 is a DEAD lure in both arms. `/password.html` returns HTTP 500, because its
#     template is missing both from the served app and from upstream's
#     environment_docker/webarena-homepage/templates/. Copying it is a match, not a working
#     credentials channel. The live launch preflight asserts the template is still absent,
#     so nobody turns it into a channel the control never had.
#
# NOT COPIED:
#   * The two few-shot examples. They are a method difference, and they show the aliases
#     `onestopmarket.com` and `openstreetmap.org`, which the control can reach without a
#     homepage visit. This gap is accepted: the URL line carries the current site's alias
#     and the homepage lists the rest.
#   * The intro's input-description list, because the `URL:` label is self-describing.
WEBARENA_HOMEPAGE_PARAGRAPH = (
    "Homepage:\n"
    "If you want to visit other websites, check out the homepage at http://homepage.com. "
    "It has a list of websites you can visit.\n"
    "http://homepage.com/password.html lists all the account name and password for the websites. "
    "You can use them to log in to the websites."
)

# The shared terminal-claim contract, unchanged, then the paragraph. It sits between the
# DECIDE/REFINE instruction and the graph instruction (graph_reasoning.py
# `_decide_instruction` / `_refine_instruction` / `_system_prompt`), and the blank line sets
# it off as its own block, as it is in the control's intro.
WEBARENA_DECIDING_INSTRUCTION_SUFFIX = TERMINAL_CLAIM_CONTRACT + "\n\n" + WEBARENA_HOMEPAGE_PARAGRAPH
