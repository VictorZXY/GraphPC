"""WebArena accessibility-tree parser.

Reuses the **selection helpers** of ``graph/a11y_parser.py`` (instruction-token
relevance + focused-window heuristics + interactivity score); **the format
parser is WebArena's own** — its AXTree text is bracketed ``[<int>] role 'name'``
lines.

Returns ``{"nodes": [...], "edges": [...]}`` with each node ``{id, role, name,
type:"ui", parent_id, webarena_id}``; the observation adapter reads ``nodes``.
``webarena_id`` is the integer id from the ``[N]`` prefix (or ``None`` if a node lacks one, e.g. the root), retained so
the compact observation can surface the bracketed id back to the reasoning /
action stages (the id space the action agent emits into). It is NOT used to
validate emitted ids: our parse is relevance-capped, so a legitimately-valid id
can be absent from it, and a hard validator would wrongly drop good actions.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from src.agents.common.compaction import instruction_tokens
from src.agents.graph.a11y_parser import (
    _focused_window_descendants,
    _stable_id,
    score_node,
)


# Regex for one AXTree line. The id prefix is optional (the root typically
# lacks one); the role is mandatory; the name is optional. The "rest" group
# catches the trailing attrs (``focused: True``, ``required: True``, etc.)
# which we keep as a single string for inspection but don't parse further.
_AXTREE_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<id>\d+)\]\s+)?(?P<role>[A-Za-z][A-Za-z0-9_]*)"
    r"(?:\s+'(?P<name>(?:[^'\\]|\\.)*)')?"
    r"(?P<rest>.*)$"
)


def _leading_indent(line: str) -> int:
    """Indent width — tabs and spaces each count as 1 (WebArena's serializer is
    consistent within a tree, so absolute units don't matter; relative depth
    does)."""
    width = 0
    for ch in line:
        if ch == "\t":
            width += 1
        elif ch == " ":
            width += 1
        else:
            break
    return width


def parse_axtree(
    text: str,
    *,
    max_nodes: int = 200,
    instruction: Optional[str] = None,
) -> dict[str, Any]:
    """Parse a WebArena AXTree string into ``{"nodes": [...], "edges": [...]}``.

    Each node: ``{id, role, name, type:"ui", parent_id, webarena_id}``.
    Each edge: ``{source, target, type:"parent_of"}``.

    When the parsed tree exceeds ``max_nodes`` the relevance-scored selection
    (instruction-keyword match + focused-window membership + role
    interactivity) keeps the top-``max_nodes``, with original parse order as
    tie-break — the selection helpers of ``graph/a11y_parser.py``, applied to
    the WebArena-shaped node dicts.
    """
    if not text:
        return {"nodes": [], "edges": []}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []  # (depth, node_id)

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _AXTREE_LINE_RE.match(stripped)
        if not match:
            continue

        wa_id_raw = match.group("id")
        role = match.group("role")
        name = match.group("name") or ""

        depth = _leading_indent(line)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent_id = stack[-1][1] if stack else None

        node_id = _stable_id(role, name, parent_id)
        nodes.append({
            "id": node_id,
            "role": role,
            "name": name,
            "type": "ui",
            "parent_id": parent_id,
            "webarena_id": int(wa_id_raw) if wa_id_raw is not None else None,
        })
        if parent_id is not None:
            edges.append({
                "source": parent_id,
                "target": node_id,
                "type": "parent_of",
            })
        stack.append((depth, node_id))

    # Relevance-based selection when over cap (graph/a11y_parser.py's scoring
    # helpers; the format parser above is WebArena's own).
    if len(nodes) > max_nodes:
        instr_tokens = frozenset(instruction_tokens(instruction or ""))
        focused_ids = _focused_window_descendants(nodes)
        scored = [
            (score_node(n, instr_tokens, focused_ids), idx, n)
            for idx, n in enumerate(nodes)
        ]
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        kept_ids = {scored[i][2]["id"] for i in range(max_nodes)}
        nodes = [n for n in nodes if n["id"] in kept_ids]
        edges = [e for e in edges if e["source"] in kept_ids and e["target"] in kept_ids]

    return {"nodes": nodes, "edges": edges}
