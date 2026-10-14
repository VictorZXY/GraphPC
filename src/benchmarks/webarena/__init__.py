"""WebArena-facing benchmark adapter for GraphPC.

Layout mirrors ``src/benchmarks/osworld/``:
WebArena-shaped on the outside (subclasses WebArena's ``Agent``), OSWorld-shaped
on the inside (calls the stable inner boundary ``decide_next_action`` /
``update_runtime`` on the orchestrator). External WebArena code is read-only;
this package is the only thing we own under it.
"""

__all__ = ["WebArenaGraphPCAgent"]


def __getattr__(name):
    # Lazy attribute access — WebArena types are imported only when the agent
    # class is actually constructed, so this package imports cleanly when the
    # WebArena checkout isn't installed in the local env.
    if name == "WebArenaGraphPCAgent":
        from .graphpc_agent import WebArenaGraphPCAgent as _Agent
        return _Agent
    raise AttributeError(name)
