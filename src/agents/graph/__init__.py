"""Accessibility-tree relevance scoring reused by WebArena's observation path.

The procedure-causal graph has no code object of its own: the reasoning call
emits no node/edge topology and retrieval ranks on text alone, so the graph is
carried as a record (design.md §3). This package holds only ``a11y_parser``,
whose scoring helpers WebArena's relevance selection reuses
(``benchmarks/webarena/a11y_parser.py`` imports them; the OSWorld 2.0 path reads
no accessibility tree).
"""
