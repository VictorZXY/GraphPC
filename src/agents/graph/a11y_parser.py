"""Accessibility-tree relevance-scoring helpers, reused by WebArena's parser.

WebArena's relevance selection (``benchmarks/webarena/a11y_parser.py``) parses
its own AXTree format and, when a tree exceeds the node cap, keeps the
highest-relevance nodes with these helpers: ``score_node`` (instruction-keyword
match + focused-window membership + role interactivity), the focused-window
heuristic ``_focused_window_descendants`` and the deterministic node id
``_stable_id``. The OSWorld 2.0 path reads no accessibility tree.
"""

from __future__ import annotations

import hashlib
import re

from typing import Any, Optional


# Relevance-based node-selection role buckets.
# Parse-all-then-select (rather than truncating the tree's tail at the node cap)
# keeps the highest-relevance nodes so the LLM sees instruction-bearing UI
# elements instead of an arbitrary tree prefix.
_INTERACTABLE_ROLES = frozenset({
    "push button", "button", "toggle button",
    "text", "entry", "text box", "edit", "password text",
    "menu item", "menu", "link", "check box", "radio button",
    "combo box", "list box", "list item", "tree item",
    "tab", "tab list", "slider", "spin button",
})
_STRUCTURAL_ROLES = frozenset({
    "heading", "list", "tree", "table", "table cell",
    "panel", "frame", "document frame", "dialog", "alert",
    "tool bar", "status bar", "menu bar", "scroll bar",
})
_NOISE_ROLES = frozenset({
    "filler", "section", "separator", "image",
    "static", "label",
})


def score_node(
    node: dict[str, Any],
    instruction_token_set: frozenset[str],
    focused_window_descendant_ids: Optional[frozenset[str]],
) -> int:
    """Relevance score for an a11y node.

    Components:
      * Instruction-keyword match on node name — up to 3 hits × 5 (cap prevents
        monopolization by a single dominant token).
      * Focused-window membership — +3 when the node descends from the most
        substantial frame/dialog/application on screen.
      * Role interactivity — +2 interactable, +1 structural, -1 noise.
      * Mild bias toward named nodes with reasonable label length.
    """
    name = (node.get("name") or "").lower()
    role = (node.get("role") or "").lower()
    score = 0
    if name and instruction_token_set:
        hits = sum(1 for tok in instruction_token_set if tok and tok in name)
        score += min(hits, 3) * 5
    if focused_window_descendant_ids and node.get("id") in focused_window_descendant_ids:
        score += 3
    if role in _INTERACTABLE_ROLES:
        score += 2
    elif role in _STRUCTURAL_ROLES:
        score += 1
    elif role in _NOISE_ROLES:
        score -= 1
    if name and 0 < len(name) <= 80:
        score += 1
    return score


def _focused_window_descendants(nodes: list[dict[str, Any]]) -> frozenset[str]:
    """Identify descendants of the most substantial named window/frame.

    Heuristic: among frame/dialog/application nodes that have a non-empty
    name, the one with the most descendants is the "focused" surface. Returns
    the descendant ID set for that surface. Empty frozenset when no named
    candidate exists (e.g. minimal observation or pre-parse failure).
    """
    candidates = [
        n for n in nodes
        if (n.get("role") or "").lower() in {"frame", "dialog", "application"}
        and (n.get("name") or "").strip()
    ]
    if not candidates:
        return frozenset()
    parent_to_children: dict[Optional[str], list[str]] = {}
    for n in nodes:
        parent_to_children.setdefault(n.get("parent_id"), []).append(n["id"])

    def descendants_of(node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(parent_to_children.get(nid, []))
        return seen

    biggest = max(candidates, key=lambda n: len(descendants_of(n["id"])))
    return frozenset(descendants_of(biggest["id"]))


def _stable_id(role: str, name: str, parent_id: Optional[str]) -> str:
    """Deterministic short ID for an a11y node.

    Format: ``a11y:<role>:<name8>:<chain6>`` where ``chain6`` is a 6-hex
    digest of the parent chain. Two re-parses of the same tree produce
    identical IDs; small name perturbations (whitespace, case) collapse
    via the normalize step. Long names are truncated to 24 chars in the
    label-visible portion but the chain hash incorporates the full name
    so collisions are unlikely.
    """
    normalized_name = (name or "").strip().lower()
    chain_input = f"{parent_id or '/'}|{role}|{normalized_name}"
    digest = hashlib.blake2b(chain_input.encode("utf-8"), digest_size=4).hexdigest()
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", normalized_name[:24]) or "_"
    return f"a11y:{role}:{safe_name}:{digest}"
