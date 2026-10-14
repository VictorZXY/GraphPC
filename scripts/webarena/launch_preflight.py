"""Loud-fail launch preflight for a WebArena GraphPC run.

A standalone runnable that ships with the code, so it cannot silent-skip. Any
failed check prints a loud line and the script exits non-zero; launch is
expected to be gated on exit status.

Scope. The run's RETRIEVAL MODE selects which checks apply, and it is passed
with the same ``--retrieval_off`` spelling the runner uses so the two cannot
drift apart.

  always              (i)   the run's retrieval mode reaches the constructed
                            objects AND is echoed into resolution-level telemetry
                      (ii)  a full config dump is written into the run record —
                            including, for a backbone that pins a provider
                            route, the route facts it will send on (host, key
                            variable, slug, provider pin, both reasoning pins),
                            so at least one run-record artefact names the route

  retrieval ON        (iii) the text arm's embedding model is genuinely live —
    (warm; per-cell)        it is the whole ranking, so an unavailable
                            model is a dead retriever, not a degraded one
                      (iv)  the injection diet is ENFORCED, proven through
                            the real retrieval path on an AUTHORED probe entry
                            carrying every dieted field (non-vacuous by
                            construction; a temp store, never the run's)
                      (v)   the working store's content hash matches THE CELL'S
                            frozen master (``--frozen_store_root`` +
                            ``--expected_entry_count``), and the run is
                            read-only against that master (it writes a
                            per-episode working copy)
                      (vi)  the cell's pre-registration file (``--strata_file``)
                            is internally consistent AND its B1 strata are
                            covered by the frozen store's families

                      Warm mode takes the CELL'S facts as REQUIRED inputs,
                      never constants: a script holding one cell's store size
                      and strata would show a GREEN line while validating
                      another cell's data. A misleading PASS is worse than an
                      honest FAIL — a check that cannot run for
                      the given cell must fail loudly or not exist, never pass
                      on another cell's numbers.

  retrieval OFF       bypass active on BOTH sides (no store read, no adapter
                      store write). The embedder / store / stratum checks are
                      vacuous with retrieval off and are skipped.

  cold store          (``--cold_store``) retrieval ON starting from an EMPTY
                      store. Check (iii) applies unchanged — the embedder is
                      the part that matters. The warm store checks (v)/(vi) are
                      replaced: the working store root must exist and hold ZERO
                      entries (a cold run on a non-empty store is a config
                      error — the failure lists what was found), and the store
                      write path is exercised (write → read back → delete a
                      probe entry, store left empty). The injection-diet check
                      (iv) needs stored entries and is vacuous cold.

  live site URLs      (``--site_mapping_live``) the site-URL mapping, with no
                      model call. Run it in the WebArena interpreter,
                      against the live sites, with the site variables set. It
                      builds the agent from THE CELL'S OWN RUNNER ARGV (given
                      after ``--``): the runner's own ``main`` runs up to its
                      ``build_default`` call, and ``build_default`` is then
                      called with exactly those kwargs. Both backbone slots
                      are swapped for a recording stub. Then it:
                        1. resets a real env on ``--live_config_file`` (a
                           read-only task config);
                        2. renders the deciding request for that page;
                        3. asserts the ``URL:`` line in its alias form (read
                           off the live ``DetachedPage``) and the control's
                           "Homepage" paragraph;
                        4. pushes ``goto [http://gitlab.com/explore]`` through
                           the action agent's mapping into ``env.step``,
                           asserts that the page lands on the local GitLab,
                           and asserts the next step's URL line and the
                           final_page_url telemetry;
                        5. compares the code with upstream, in the checkout:
                           the table equals ``browser_env.env_config.URL_MAPPINGS``
                           in values and order; both mappers equal
                           ``PromptConstructor``'s on a fixture set; the
                           paragraph equals the raw prompt's text and the
                           generated JSON's (the file the control loads);
                        6. records the homepage app's template hashes, and
                           asserts that ``password.html`` is still absent, on
                           disk and as served.
                      It prints a summary (table digest, one rendered sample)
                      for the launch record, and writes every check into
                      ``--result_dir``.

Usage:

  python scripts/webarena/launch_preflight.py \\
      --result_dir <run dir>/lanes/gitlab/results \\
      --working_store <run dir>/lanes/gitlab/working_store \\
      --frozen_store_root <the cell's frozen master store> \\
      --expected_entry_count <cell store size> \\
      --strata_file <the cell's pre-registration JSON>
  python scripts/webarena/launch_preflight.py --retrieval_off ...
  python scripts/webarena/launch_preflight.py --cold_store ...
  <WebArena env python> scripts/webarena/launch_preflight.py --site_mapping_live \\
      --result_dir <dir> --live_config_file <WebArena>/config_files/7.json \\
      --homepage_app_root <the served homepage app's directory> \\
      -- <the cell's run_graphpc.py argv>

Exit codes: 0 = every check passed; 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NO per-cell constants live here (see the module docstring): every cell fact
# arrives via --frozen_store_root / --expected_entry_count / --strata_file.

# The text arm's embedding model. A different model is a different retriever,
# so no arm may launch against a re-pointed default.
EXPECTED_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# The retrieval agent each mode must end up with. Read back off a constructed
# agent, so a flag that fails to reach the objects is caught here.
MODE_RETRIEVAL_AGENT = {
    False: "DeterministicRetrievalAgent",
    True: "BypassRetrievalAgent",
}
# Entry fields the injection diet must remove before an exemplar reaches the
# prompt, and the duplicated recipe-log field it must collapse. Kept as the
# gate's OWN copy (not imported from deterministic.py) so a silent drift in the
# module's tuple is a gate failure, not a moved goalpost.
DIETED_ENTRY_FIELDS = ("graph_artifacts", "observation", "signatures")
DEDUPED_ACTION_FIELD = "webarena_action_string"
# Slot cap: the maximum number of exemplars any single retrieval may inject.
MAX_INJECTED_ENTRIES = 2


class Preflight:
    """Collects check results; every failure is recorded, none short-circuits,
    so one run reports the full picture instead of the first problem."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, fn: Callable[[], Any]) -> None:
        try:
            detail = fn()
            self.checks.append({"check": name, "passed": True, "detail": detail})
            print(f"  PASS  {name}: {detail}")
        except Exception as exc:  # noqa: BLE001 — a raising check is a failed check.
            self.checks.append({"check": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    @property
    def failed(self) -> list[str]:
        return [c["check"] for c in self.checks if not c["passed"]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _store_content_hash(store_root: Path) -> tuple[str, int]:
    """Content hash of a support-set store: sha256 over the sorted
    (filename, file-sha256) pairs of ``<root>/webarena/*.json``. Path-prefix
    independent, so a working copy and the frozen master compare equal."""
    backend_dir = store_root / "webarena"
    _require(backend_dir.is_dir(), f"no webarena/ backend dir under {store_root}")
    rows = []
    for path in sorted(backend_dir.glob("*.json")):
        rows.append(f"{path.name}  {hashlib.sha256(path.read_bytes()).hexdigest()}")
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return digest, len(rows)


def _build_agent(args: argparse.Namespace):
    """Construct the real agent in this run's retrieval mode. Uses the same
    ``build_default`` the runner calls, so the echo under test is the shipped
    path. Backbone clients are constructed but never called."""
    from src.benchmarks.webarena.graphpc_agent import WebArenaGraphPCAgent

    return WebArenaGraphPCAgent.build_default(
        model=args.model,
        backbone_type=args.backbone,
        support_set_root=args.working_store,
        retrieval_off=args.retrieval_off,
    )


def _check_embedder_live() -> str:
    """The text arm must be the real sentence-embedding model. It is the ONLY
    ranking arm, so an unavailable model does not degrade retrieval — it
    kills it (the store raises rather than serve an ungated empty result)."""
    from src.agents.retrieval.embedding import SentenceTransformerEmbedder, get_default_embedder

    embedder = get_default_embedder()
    _require(
        isinstance(embedder, SentenceTransformerEmbedder),
        f"default embedder is {type(embedder).__name__}, not the sentence-embedding model",
    )
    _require(
        bool(getattr(embedder, "available", False)),
        "embedding model did NOT load — there is no structure-only fallback to "
        "degrade into, so every retrieval-on episode would abort: check "
        "sentence-transformers + the Qwen3-Embedding-0.6B weights",
    )
    _require(
        embedder.model_name == EXPECTED_EMBEDDING_MODEL,
        f"text arm is running {embedder.model_name!r}, not the run regime's "
        f"{EXPECTED_EMBEDDING_MODEL!r} — a different model is a different retriever",
    )
    sims = embedder.similarity("open the issues page", ["open the issues page", "delete a product"])
    _require(sims is not None and len(sims) == 2, "embedder.similarity() returned no scores")
    _require(sims[0] > sims[1], f"embedder scores are not discriminative: {sims}")
    return f"{type(embedder).__name__}/{embedder.model_name} live (probe {sims[0]:.3f} > {sims[1]:.3f})"


def _check_cold_store_empty(args: argparse.Namespace) -> str:
    """A cold run must start from an EMPTY store: any pre-existing entry means
    a mis-pointed or contaminated store root (a config error — the cold
    trajectory would be unattributable), so the failure LISTS what was found.
    The ``webarena/`` backend dir may not exist yet (the store creates it on
    first write) — only its entries are a failure, not its absence."""
    root = Path(args.working_store)
    _require(root.is_dir(), f"cold store root does not exist: {root} — create it (empty) before launch")
    backend_dir = root / "webarena"
    entries = sorted(p.name for p in backend_dir.glob("*.json")) if backend_dir.is_dir() else []
    _require(
        not entries,
        f"cold store is NOT empty — {len(entries)} entr(y|ies) under {backend_dir}: "
        + ", ".join(entries[:20])
        + (f" … (+{len(entries) - 20} more)" if len(entries) > 20 else ""),
    )
    return f"store root {root} exists and holds zero entries"


def _check_store_write_path(args: argparse.Namespace) -> str:
    """The cold run's first win must be able to persist: exercise the store's
    REAL write path (``add_success_entry`` → read back via ``list_entries`` →
    delete), and leave the store empty — the probe must never survive into the
    cold trajectory."""
    from src.agents.retrieval.support_set import LocalSupportSetStore

    store = LocalSupportSetStore(root_dir=args.working_store, backend="webarena")
    entries_before = store.entry_count()
    probe_task_id = "preflight_write_probe"
    written = store.add_success_entry({
        "task_id": probe_task_id,
        "instruction": "preflight write-path probe (must never persist)",
        "success": True,
        "result": 1.0,
    })
    try:
        readback = [e for e in store.list_entries() if e.get("task_id") == probe_task_id]
        _require(len(readback) == 1, f"probe entry did not read back (found {len(readback)})")
    finally:
        # The probe must never be left behind — even when the read-back failed.
        if written.exists():
            written.unlink()
    _require(
        store.entry_count() == entries_before,
        "store entry count changed after the probe delete — the probe leaked",
    )
    return f"write → read → delete probe OK under {store.backend_dir} (no residue)"


def _check_retrieval_mode_echoed(agent, retrieval_off: bool) -> str:
    """The run's retrieval mode must reach the constructed objects AND be echoed
    into the resolution-level telemetry record the runner writes."""
    import scripts.webarena.run_graphpc as runner

    want_agent = MODE_RETRIEVAL_AGENT[retrieval_off]
    state = agent.ablation_switch_state()
    _require(
        state.get("retrieval_off") == retrieval_off,
        f"retrieval_off: agent reports {state.get('retrieval_off')!r}, the run needs {retrieval_off!r}",
    )
    _require(
        state.get("retrieval_agent") == want_agent,
        f"retrieval agent is {state.get('retrieval_agent')!r}, this mode needs {want_agent!r}",
    )

    telemetry = runner._collect_episode_telemetry(
        agent=agent, task_id="preflight", config_file="preflight",
        score=None, success=False, steps=0,
    )
    echoed = telemetry.get("ablation_switches")
    _require(echoed is not None, "episode telemetry carries no switch block")
    _require(
        echoed.get("retrieval_off") == retrieval_off,
        f"telemetry echo retrieval_off: {echoed.get('retrieval_off')!r} != {retrieval_off!r}",
    )
    return f"agent + episode telemetry both report retrieval_off={retrieval_off}, {want_agent}"


def _check_injection_diet_enforced() -> str:
    """The injection diet, proven through the REAL retrieval path on an
    AUTHORED probe entry that carries every dieted field — non-vacuous by
    construction. The run's own store entries cannot prove it: they do not
    carry every dieted field (``signatures`` among them). A live probe (not a
    contract read) is proportionate here: warm mode already pays the embedder
    load, the probe adds one encode, and it attests the deployed injection path
    rather than a mirror of it. The probe lives in a TEMP store and never
    touches the run's working store."""
    import shutil
    import tempfile

    from src.agents.common.schema import Observation, RuntimeState
    from src.agents.retrieval.base import RetrievalInput
    from src.agents.retrieval.deterministic import DeterministicRetrievalAgent
    from src.agents.retrieval.embedding import get_default_embedder
    from src.agents.retrieval.support_set import LocalSupportSetStore

    instruction = "preflight injection-diet probe instruction"
    probe = {
        "task_id": "preflight_diet_probe",
        "instruction": instruction,
        "success": True,
        "result": 1.0,
        "graph_artifacts": {"task_graph": {"nodes": []}},
        "observation": {"observation_text": "stale a11y snapshot"},
        "signatures": {"legacy": "structure-arm retrieval key"},
        "recent_history": [{
            "actions": [{
                "parameters": {"command": "click [1]"},
                "metadata": {"webarena_action_string": "click [1]", "other": "kept"},
            }],
        }],
    }
    tmp_root = tempfile.mkdtemp(prefix="preflight_diet_probe_")
    try:
        store = LocalSupportSetStore(root_dir=tmp_root, backend="webarena")
        store.add_success_entry(dict(probe))
        retrieval_agent = DeterministicRetrievalAgent(
            support_set_store=store, embedder=get_default_embedder(),
        )
        result = retrieval_agent.retrieve(
            RetrievalInput(
                retrieval_scope="task", instruction=instruction,
                observation=Observation(backend="webarena", content={"observation_text": ""}),
                runtime_state=RuntimeState(instruction=instruction),
            )
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    items = result.items
    _require(bool(items), "the probe entry did not pass the relevance gate — the diet cannot be attested")
    _require(
        len(items) <= MAX_INJECTED_ENTRIES,
        f"retrieval injected {len(items)} entries, the slot cap is {MAX_INJECTED_ENTRIES}",
    )
    item = items[0]
    for field in DIETED_ENTRY_FIELDS:
        _require(field not in item, f"injected exemplar still carries {field!r}")
    for history_item in item.get("recent_history") or []:
        for action in history_item.get("actions") or []:
            metadata = action.get("metadata") or {}
            _require(
                DEDUPED_ACTION_FIELD not in metadata,
                f"injected recipe log still carries the duplicated {DEDUPED_ACTION_FIELD!r}",
            )
            _require(
                bool((action.get("parameters") or {}).get("command")),
                "the recipe log lost its action: parameters.command is the copy that must SURVIVE",
            )
    return (
        f"probe authored WITH {', '.join(DIETED_ENTRY_FIELDS)} + the duplicated action "
        "string; the injected copy carries none of them and keeps parameters.command"
    )


def _check_store_frozen(args: argparse.Namespace) -> str:
    """The working store the run reads must be a byte-identical copy of THE
    CELL'S frozen master — declared per cell via --frozen_store_root /
    --expected_entry_count, never a constant — and must not BE the master: the warm
    protocol's read-only guarantee is "run against a per-episode working copy,
    never the master"."""
    frozen = Path(args.frozen_store_root).resolve()
    working = Path(args.working_store).resolve()
    _require(frozen.is_dir(), f"frozen store not found: {frozen}")
    _require(working.is_dir(), f"working store not found: {working}")
    _require(
        working != frozen,
        f"the run would write to the FROZEN master ({frozen}) — point --support_set_root "
        "at a per-episode working copy",
    )
    frozen_hash, frozen_n = _store_content_hash(frozen)
    working_hash, working_n = _store_content_hash(working)
    _require(
        frozen_n == args.expected_entry_count,
        f"frozen store has {frozen_n} entries, the cell declares {args.expected_entry_count}",
    )
    _require(
        working_hash == frozen_hash,
        f"working store ({working_n} entries, {working_hash[:12]}) differs from the frozen "
        f"master ({frozen_n} entries, {frozen_hash[:12]}) — restore it before launch",
    )
    return f"{working_n} entries, content hash {working_hash[:12]} == frozen master"


def _check_retrieval_bypass(agent, args: argparse.Namespace) -> str:
    """Retrieval-off must hold on BOTH sides: the retrieval call opens no store,
    and the adapter performs no support-set write on a successful episode."""
    from src.agents.common.schema import Observation, RuntimeState
    from src.agents.retrieval.base import RetrievalInput
    from src.agents.retrieval.bypass import BypassRetrievalAgent

    _require(
        isinstance(agent.retrieval_agent, BypassRetrievalAgent),
        f"retrieval agent is {type(agent.retrieval_agent).__name__}, not the bypass agent",
    )
    result = agent.retrieval_agent.retrieve(
        RetrievalInput(
            retrieval_scope="task", instruction="preflight",
            observation=Observation(backend="webarena", content={"observation_text": ""}),
            runtime_state=RuntimeState(instruction="preflight"),
        )
    )
    _require(not result.items, f"bypass retrieval returned {len(result.items)} item(s)")
    _require(
        bool((result.metadata or {}).get("retrieval_bypassed")),
        "bypass result does not carry retrieval_bypassed on metadata (telemetry would be silent)",
    )

    # Write side: a successful finalize must not create a store file.
    store_dir = Path(args.working_store) / "webarena"
    before = sorted(p.name for p in store_dir.glob("*.json")) if store_dir.is_dir() else []
    agent.runtime_state = RuntimeState(instruction="preflight")
    agent.task_instruction = "preflight"
    written = agent.finalize_task(task_id="preflight", success=True, result=1.0)
    after = sorted(p.name for p in store_dir.glob("*.json")) if store_dir.is_dir() else []
    _require(written is None, f"finalize_task wrote a store entry: {written}")
    _require(before == after, f"store gained {len(after) - len(before)} file(s) during finalize")
    _require(
        agent.runtime_state.content.get("_store_write_skipped") == "retrieval_off",
        "the skipped store write is not recorded on the episode record",
    )
    agent.runtime_state = None
    return "no store read (empty result, flagged) and no store write on a successful finalize"


def _check_strata_file(args: argparse.Namespace) -> str:
    """The cell's pre-registration file must be INTERNALLY consistent, and its
    B1 strata must be covered by the frozen store's families — both verified
    from the file's own data against the cell's store, never against numbers
    baked into this script. A check holding one cell's quantities would show
    GREEN in another cell's launch context — validating the WRONG cell. A
    misleading PASS is worse than an honest FAIL, so anything this check needs
    but cannot find in the file (rows, master_table) is a loud FAILURE, never a
    silent skip.

    Legs: (1) the three protocol strata exist, are disjoint, and their union is
    exactly ``task_ids`` (= ``n_subset``); (2) declared ``stratum_sizes`` match
    the lists; (3) per-row ``stratum`` labels match the lists, rows cover
    exactly ``task_ids``, and row families agree with the master table;
    (4) the ``recipe_eligibility`` split (when declared) is disjoint, inside
    ``task_ids``, and matches its declared count; (5) every B1-stratum task's
    family has at least one entry in the frozen store (B1 means "family has a
    cold win", and the frozen store holds the cold wins)."""
    strata_path = Path(args.strata_file)
    _require(strata_path.is_file(), f"pre-registration file not found: {strata_path}")
    payload = json.loads(strata_path.read_text())

    strata = payload.get("strata")
    _require(isinstance(strata, dict) and strata, "file has no strata block")
    required_strata = ("b1_pre_opener_losers", "b1_openers", "a_only")
    missing = [n for n in required_strata if not isinstance(strata.get(n), list)]
    _require(not missing, f"strata block lacks the protocol strata: {missing}")

    ids_by_stratum = {n: [str(t) for t in strata[n]] for n in required_strata}
    union: list[str] = [t for ids in ids_by_stratum.values() for t in ids]
    _require(len(union) == len(set(union)), "strata are not disjoint (or repeat ids)")

    task_ids_raw = payload.get("task_ids")
    _require(isinstance(task_ids_raw, list) and task_ids_raw,
             "file has no task_ids list — the strata cover cannot be cross-checked")
    task_ids = {str(t) for t in task_ids_raw}
    _require(len(task_ids) == len(task_ids_raw), "task_ids contains duplicates")
    _require(
        set(union) == task_ids,
        f"strata union ({len(set(union))}) != task_ids ({len(task_ids)}): "
        f"only-in-strata {sorted(set(union) - task_ids)[:5]}, "
        f"only-in-task_ids {sorted(task_ids - set(union))[:5]}",
    )
    n_subset = payload.get("n_subset")
    _require(n_subset == len(task_ids),
             f"n_subset declares {n_subset}, task_ids holds {len(task_ids)}")

    declared_sizes = payload.get("stratum_sizes") or {}
    for name in required_strata:
        if name in declared_sizes:
            _require(
                declared_sizes[name] == len(ids_by_stratum[name]),
                f"stratum_sizes[{name}] declares {declared_sizes[name]}, "
                f"the list holds {len(ids_by_stratum[name])}",
            )

    rows = payload.get("rows")
    _require(isinstance(rows, list) and rows,
             "file has no per-row data (rows) — stratum labels and family "
             "coverage cannot be verified")
    stratum_by_task = {t: n for n, ids in ids_by_stratum.items() for t in ids}
    row_ids = {str(r.get("task_id")) for r in rows}
    _require(
        row_ids == task_ids,
        f"rows cover {len(row_ids)} ids, task_ids holds {len(task_ids)} — they must coincide",
    )
    family_by_subset_task: dict[str, Any] = {}
    for row in rows:
        task = str(row.get("task_id"))
        _require(
            row.get("stratum") == stratum_by_task.get(task),
            f"row {task} labelled stratum {row.get('stratum')!r}, the lists put it in "
            f"{stratum_by_task.get(task)!r}",
        )
        family_by_subset_task[task] = row.get("family")

    master = payload.get("master_table")
    _require(isinstance(master, list) and master,
             "file has no master_table — the frozen store's family coverage "
             "cannot be established")
    family_by_task = {str(r.get("task_id")): r.get("intent_template_id") for r in master}
    for task, family in family_by_subset_task.items():
        _require(
            family_by_task.get(task) == family,
            f"row {task} declares family {family!r}, the master table says "
            f"{family_by_task.get(task)!r}",
        )

    recipe = payload.get("recipe_eligibility")
    recipe_note = "no recipe_eligibility block declared"
    if isinstance(recipe, dict):
        eligible = {str(t) for t in recipe.get("eligible") or []}
        premise_failed = {str(t) for t in recipe.get("premise_failed") or []}
        _require(not (eligible & premise_failed),
                 f"recipe eligible/premise_failed overlap: {sorted(eligible & premise_failed)[:5]}")
        _require(eligible <= task_ids and premise_failed <= task_ids,
                 "recipe ids outside task_ids")
        declared = recipe.get("eligible_count")
        _require(declared == len(eligible),
                 f"eligible_count declares {declared}, eligible holds {len(eligible)}")
        recipe_note = f"recipe split {len(eligible)}/{len(premise_failed)} consistent"

    # --- family coverage of the B1 strata against the frozen store ---
    frozen = Path(args.frozen_store_root)
    backend_dir = frozen / "webarena"
    _require(backend_dir.is_dir(), f"no webarena/ backend dir under {frozen}")
    store_families: set = set()
    unmapped: list[str] = []
    for path in sorted(backend_dir.glob("*.json")):
        entry_task = str(json.loads(path.read_text()).get("task_id"))
        if entry_task in family_by_task:
            store_families.add(family_by_task[entry_task])
        else:
            unmapped.append(entry_task)
    _require(
        not unmapped,
        f"{len(unmapped)} store entr(y|ies) with task ids absent from the master "
        f"table (family unknowable): {sorted(set(unmapped))[:5]}",
    )
    family_covered_tasks = ids_by_stratum["b1_pre_opener_losers"] + ids_by_stratum["b1_openers"]
    uncovered = sorted(
        t for t in family_covered_tasks if family_by_subset_task.get(t) not in store_families
    )
    _require(
        not uncovered,
        f"{len(uncovered)} B1 task(s) whose family has NO entry in the frozen "
        f"store: {uncovered[:10]}",
    )

    sizes = "/".join(str(len(ids_by_stratum[n])) for n in required_strata)
    return (
        f"{len(task_ids)} ids in 3 disjoint strata ({sizes}); rows + families "
        f"consistent; {recipe_note}; all {len(family_covered_tasks)} B1 tasks family-covered "
        f"by the frozen store ({len(store_families)} store families)"
    )


# ---------------------------------------------------------------------------
# Live site-URL mode: the site-URL mapping against the live sites
# ---------------------------------------------------------------------------

# The alias the step-1 grounding reply writes. `http://gitlab.com` is an exact alias, so the action
# agent's mapping must map it onto the GITLAB site. /explore is a public GitLab page, and visiting it writes nothing.
LIVE_ALIAS_GOTO = "goto [http://gitlab.com/explore]"
_LIVE_DECIDING_REPLY = {
    "reasoning_text": "launch preflight: open the code host's explore page.",
    "plan": None,
    "actual_effect_class": None,
    "next_intent": "open the explore page",
    "graph_summary": "launch preflight",
    "intended_actions": [{
        "action_class": "goto", "target": "http://gitlab.com/explore", "text": None, "keys": None,
        "value": None, "intent": "open the explore page", "predicted_effect_class": "navigation",
    }],
}


class _RecordingBackbone:
    """Answers every model call itself and keeps its keyword arguments, so no request leaves the
    host. Grounding replies are queued: the alias goto, then a noop."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._grounding_replies = [LIVE_ALIAS_GOTO, "noop"]

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("grounding"):
            reply = {"reasoning_text": "launch preflight", "action_strings": [self._grounding_replies.pop(0)]}
        else:
            reply = json.loads(json.dumps(_LIVE_DECIDING_REPLY))
        return {"json": reply, "raw_response": json.dumps(reply)}

    def deciding(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if not call.get("grounding")]


class _BuildDefaultReached(Exception):
    pass


def _cell_runner_build(runner_argv: list[str]):
    """``(build_default kwargs, the runner's parsed args)`` for the CELL'S OWN runner argv.

    The runner's own ``main`` runs on that argv up to its ``build_default`` call. That is the
    launcher layer, not a restatement of it, so the kwargs are the ones a launch passes. Everything
    main does before that call is real: the WebArena checkout goes on the path, the openai-0.27 shim
    and both env-hardening patches are installed, and the evaluator is imported. That is also
    exactly the environment the env below then runs in. Only ``--result_dir`` is redirected, to a
    temp dir, so the cell's own record is not touched.

    The runner's own first steps (checkout on the path, then the shim) are taken once more BEFORE
    this module imports the agent, which is what the ``build_default`` patch needs. Importing the
    agent first would resolve its WebArena ``Agent`` base before the shim exists. In the WebArena
    interpreter, where the checkout is importable, that half-imports ``llms`` and pins the stub
    base (the hazard ``run_episode_live``'s import comment names). Both steps are idempotent."""
    import tempfile
    from unittest import mock

    import scripts.webarena.run_graphpc as runner
    from src.benchmarks.webarena._openai_compat import install_openai_027_compat

    runner._ensure_webarena_on_path(Path(runner.build_parser().parse_args([*runner_argv, "--result_dir", "."]).webarena_root))
    install_openai_027_compat()
    from src.benchmarks.webarena.graphpc_agent import WebArenaGraphPCAgent

    captured: dict[str, Any] = {}

    def _build_default(**kwargs):
        captured.update(kwargs)
        raise _BuildDefaultReached

    with tempfile.TemporaryDirectory(prefix="preflight_runner_") as scratch:
        argv = [*runner_argv, "--result_dir", scratch]
        with mock.patch.object(WebArenaGraphPCAgent, "build_default", _build_default):
            try:
                runner.main(argv)
            except _BuildDefaultReached:
                return captured, runner.build_parser().parse_args(argv)
            except SystemExit as exc:
                raise AssertionError(f"the runner refused the cell argv before build_default (exit {exc.code})") from None
    raise AssertionError(f"the runner returned without reaching build_default on {runner_argv}")


def _live_fixtures(table: dict[str, str]) -> list[str]:
    """Strings both mappers are compared on: every site in both directions, and each case the live preflight
    compares (one commented group each below). Built from the live
    table, so they are the hosts the run uses."""
    local_of = {alias: local for local, alias in table.items()}
    admin_host = local_of["http://luma.com/admin"].rsplit("/admin", 1)[0]
    wiki_root = local_of["http://wikipedia.org"].split("/A/", 1)[0]
    return [
        *[local + "/x?y=1" for local in table], *[f"goto [{alias}/x?y=1]" for alias in table.values()],
        "goto [https://gitlab.com/explore]", "goto [https://reddit.com/f/news]",                  # https
        "goto [http://www.reddit.com/]", "goto [https://www.openstreetmap.org/#map=7/42/-75]",    # www.
        "goto [http://luma.com/]", admin_host + "/customer/account/",                               # luma w/o /admin
        "chrome-error://chromewebdata/", "goto [chrome-error://chromewebdata/]",                     # chrome-error
        "about:blank", "goto [about:blank]",                                                         # about:blank
        wiki_root + "/A/Carnegie_Mellon_University", "goto [http://wikipedia.org/wiki/Pittsburgh]",  # non-landing wiki
        local_of["http://homepage.com"] + "/password.html", "goto [http://homepage.com/calculator.html]",  # homepage
        "type [3] [http://onestopmarket.com/x] [1]", "stop [http://gitlab.com/byteblaze]",           # whole string
    ]


def _run_site_mapping_live(preflight: Preflight, args: argparse.Namespace, runner_argv: list[str]) -> dict:
    """Checks 1-6 of the module docstring's live mode, in order, on one real env. Each check
    records its own verdict; a check whose input an earlier failure left missing fails naming it."""
    import importlib.util
    import urllib.error
    import urllib.request

    ctx: dict[str, Any] = {}

    def need(key: str):
        _require(key in ctx, f"not reached: the check that produces {key!r} failed above")
        return ctx[key]

    def build() -> str:
        kwargs, runner_args = _cell_runner_build(runner_argv)
        from src.benchmarks.webarena.graphpc_agent import WebArenaGraphPCAgent

        agent = WebArenaGraphPCAgent.build_default(**kwargs)
        backbone = _RecordingBackbone()
        agent.reasoning_agent.backbone = backbone
        agent.action_agent.backbone = backbone
        _require(agent.perception_processor.url_mappings is agent.action_agent.url_mappings,
                 "the adapter and the action agent hold different site-URL tables")
        ctx.update(kwargs=kwargs, runner_args=runner_args, agent=agent, backbone=backbone,
                   table=agent.perception_processor.url_mappings)
        return (f"build_default({', '.join(f'{k}={v!r}' for k, v in kwargs.items() if k != 'support_set_root')}) "
                f"from the runner's own main; both backbone slots are a recording stub")

    def reset_and_render() -> str:
        import scripts.webarena.run_graphpc as runner
        from src.benchmarks.webarena.prompts import WEBARENA_HOMEPAGE_PARAGRAPH
        from src.benchmarks.webarena.url_mappings import map_url_to_real

        agent, backbone, table = need("agent"), need("backbone"), need("table")
        runner._set_node_heap_headroom()
        env = ctx["env"] = runner._make_script_browser_env(need("runner_args"))
        config_file = str(Path(args.live_config_file).resolve())
        intent = runner._intent_from_config(json.loads(Path(config_file).read_text()))
        obs, info = env.reset(options={"config_file": config_file})
        first = {"observation": obs, "info": info}
        agent.reset(config_file)
        ctx["action"] = agent.next_action([first], intent, meta_data={})
        ctx.update(first=first, intent=intent)
        page_url = info["page"].url
        displayed = map_url_to_real(page_url, table)
        _require(type(info["page"]).__name__ == "DetachedPage",
                 f"the live state's page is a {type(info['page']).__name__}, not WebArena's DetachedPage")
        _require(displayed != page_url and any(displayed.startswith(alias) for alias in table.values()),
                 f"the start page {page_url!r} did not map to an alias ({displayed!r}): not an on-host start page")
        deciding = backbone.deciding()[0]
        observation_text = json.loads(deciding["user_prompt"])["observation_text"]
        _require(observation_text.endswith("\nURL: " + displayed),
                 f"the deciding request's observation does not end with the alias URL line: ...{observation_text[-160:]!r}")
        _require(WEBARENA_HOMEPAGE_PARAGRAPH in deciding["system_prompt"],
                 "the deciding system prompt does not carry the control's Homepage paragraph")
        history = agent.runtime_state.content.get("_page_url_history")
        tree_sha256 = hashlib.sha256((obs.get("text") or "").encode("utf-8")).hexdigest()
        _require(history == [{"step_idx": 0, "page_url": page_url, "displayed_url": displayed,
                              "tree_sha256": tree_sha256}],
                 f"_page_url_history after step 1 is {history!r}")
        ctx["sample"] = {"step1_observation_tail": observation_text.splitlines()[-3:], "step1_page_url": page_url}
        return f"{page_url} renders as 'URL: {displayed}'; the deciding system prompt carries the Homepage paragraph"

    def goto_lands_local() -> str:
        from src.benchmarks.webarena.url_mappings import map_url_to_real

        agent, backbone, env, table = need("agent"), need("backbone"), need("env"), need("table")
        action = need("action")
        gitlab = next(local for local, alias in table.items() if alias == "http://gitlab.com")
        events = agent.runtime_state.content.get("_url_mapping_events")
        _require(events == [{"step_idx": 0, "action_index": 0, "issued": LIVE_ALIAS_GOTO,
                             "mapped": f"goto [{gitlab}/explore]"}], f"_url_mapping_events is {events!r}")
        _require(action.get("url") == f"{gitlab}/explore", f"the env receives url={action.get('url')!r}")
        obs, _reward, _terminated, _truncated, info = env.step(action)
        landed = info["page"].url
        _require(landed.startswith(gitlab), f"the goto landed on {landed!r}, not on the local GitLab {gitlab!r}")
        _require(env.page.url == landed, f"env.page.url (the runner's final page URL) is {env.page.url!r}")
        agent.next_action([need("first"), action, {"observation": obs, "info": info}], need("intent"), meta_data={})
        second = json.loads(backbone.deciding()[1]["user_prompt"])["observation_text"]
        displayed = map_url_to_real(landed, table)
        _require(displayed.startswith("http://gitlab.com") and second.endswith("\nURL: " + displayed),
                 f"step 2's deciding request does not end with 'URL: {displayed}': ...{second[-160:]!r}")
        history = agent.runtime_state.content.get("_page_url_history") or []
        _require(len(history) == 2 and history[1]["displayed_url"] == displayed, f"_page_url_history is {history!r}")
        ctx.setdefault("sample", {}).update(step2_url_line=second.splitlines()[-1], step2_page_url=landed)
        return f"{LIVE_ALIAS_GOTO} -> goto [{gitlab}/explore]; landed on {landed}; step 2 renders 'URL: {displayed}'"

    def upstream_table() -> str:
        from browser_env.env_config import URL_MAPPINGS

        table = need("table")
        _require(list(table.items()) == list(URL_MAPPINGS.items()),
                 "our table differs from browser_env.env_config.URL_MAPPINGS in values or order")
        return f"{len(table)} entries equal browser_env.env_config.URL_MAPPINGS, in order"

    def upstream_mappers() -> str:
        from agent.prompts.prompt_constructor import PromptConstructor
        from src.benchmarks.webarena.url_mappings import map_url_to_local, map_url_to_real

        table = need("table")
        fixtures = _live_fixtures(table)
        differ = [f for f in fixtures
                  if map_url_to_real(f, table) != PromptConstructor.map_url_to_real(None, f)
                  or map_url_to_local(f, table) != PromptConstructor.map_url_to_local(None, f)]
        _require(not differ, f"our mappers differ from PromptConstructor's on {differ}")
        to_real = sum(map_url_to_real(f, table) != f for f in fixtures)
        to_local = sum(map_url_to_local(f, table) != f for f in fixtures)
        _require(to_real >= len(table) and to_local >= len(table),
                 f"only {to_real} / {to_local} fixtures change under the mappers: the comparison is vacuous")
        return (f"map_url_to_real / map_url_to_local equal PromptConstructor's on {len(fixtures)} fixtures "
                f"({to_real} / {to_local} of them rewritten)")

    def upstream_paragraph() -> str:
        from src.benchmarks.webarena.prompts import WEBARENA_HOMEPAGE_PARAGRAPH

        prompts_dir = Path(need("runner_args").webarena_root) / "agent" / "prompts"
        raw = prompts_dir / "raw" / "p_cot_id_actree_2s.py"
        spec = importlib.util.spec_from_file_location("preflight_p_cot_id_actree_2s", raw)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _require("\n\n" + WEBARENA_HOMEPAGE_PARAGRAPH + "\n\n" in module.prompt["intro"],
                 f"WEBARENA_HOMEPAGE_PARAGRAPH is not a whole paragraph of {raw}'s intro")
        # The control's run.py loads the JSON generated from the raw prompt (run_baseline.py's
        # --instruction_path default), so the artefact it actually sends is held too.
        loaded = prompts_dir / "jsons" / "p_cot_id_actree_2s.json"
        _require("\n\n" + WEBARENA_HOMEPAGE_PARAGRAPH + "\n\n" in json.loads(loaded.read_text())["intro"],
                 f"WEBARENA_HOMEPAGE_PARAGRAPH is not a whole paragraph of {loaded}'s intro, the prompt the control loads")
        return f"the paragraph is byte-equal to a whole paragraph of {raw} and of {loaded}"

    def homepage() -> str:
        table = need("table")
        templates = Path(args.homepage_app_root) / "templates"
        _require(templates.is_dir(), f"no templates/ under --homepage_app_root {args.homepage_app_root}")
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(templates.iterdir()) if path.is_file()}
        _require("password.html" not in hashes,
                 "templates/password.html exists: the control's credentials sentence is a DEAD lure in both arms "
                 "(HTTP 500); serving it gives our arm a channel the control never had")
        homepage_url = next(local for local, alias in table.items() if alias == "http://homepage.com")
        with urllib.request.urlopen(homepage_url + "/", timeout=10) as response:
            _require(response.status == 200, f"{homepage_url}/ answered {response.status}")
        try:
            with urllib.request.urlopen(homepage_url + "/password.html", timeout=10) as response:
                password_status = response.status
        except urllib.error.HTTPError as exc:
            password_status = exc.code
        _require(password_status != 200, f"{homepage_url}/password.html answered 200: it is being served")
        ctx["homepage"] = {"templates_sha256": hashes, "password_html_status": password_status}
        return f"{len(hashes)} template(s) hashed, no password.html; {homepage_url}/ is up, /password.html answers {password_status}"

    try:
        for name, check in (("cell_agent_built", build), ("start_page_url_line_and_homepage_paragraph", reset_and_render),
                            ("alias_goto_lands_on_the_local_site", goto_lands_local),
                            ("table_equals_upstream", upstream_table), ("mappers_equal_upstream", upstream_mappers),
                            ("paragraph_equals_upstream", upstream_paragraph), ("homepage_templates", homepage)):
            preflight.check(name, check)
    finally:
        if "env" in ctx:
            ctx["env"].close()

    table = ctx.get("table") or {}
    return {
        "runner_argv": runner_argv,
        "build_default_kwargs": ctx.get("kwargs"),
        "table_sha256": hashlib.sha256(json.dumps(list(table.items())).encode("utf-8")).hexdigest() if table else None,
        "table": table,
        "rendered_sample": ctx.get("sample"),
        "homepage": ctx.get("homepage"),
    }


def _write_site_mapping_live_dump(preflight: Preflight, args: argparse.Namespace, summary: dict) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / "launch_preflight_site_mapping_live.json"
    payload = {"mode": preflight.mode, "argv": list(sys.argv), "args": vars(args), "summary": summary,
               "checks": preflight.checks, "passed": not preflight.failed}
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def _backbone_route_facts(agent) -> dict | None:
    """The route a route-PINNED backbone will send on, for the run record.

    Some backbones reach their model through a shared aggregator rather than a
    provider-specific host, and then the model slug alone does not say which
    endpoint served: the Qwen cell runs ``qwen/qwen3.7-plus`` over OpenRouter
    behind an explicit provider pin, and without this block NO run-record
    artefact would name the route the arm actually ran on. So the pinned facts —
    resolved base URL, key variable, model slug, provider pin and BOTH reasoning
    pins — are dumped alongside the rest of the resolved config.

    Read off the BUILT backbone (and the module that defines it), never off a
    CLI flag, so what lands in the record is what the constructed object will
    send. Every read is defensive: a backbone that pins no provider returns
    None and its dump carries no route block, and no attribute access here
    can raise."""
    backbone = getattr(getattr(agent, "reasoning_agent", None), "backbone", None)
    if backbone is None:
        return None
    module = sys.modules.get(getattr(type(backbone), "__module__", "") or "")
    # The provider pin is the marker of a route-pinned backbone; the rest of the
    # facts only mean something alongside it.
    if module is None or getattr(module, "_PROVIDER_PIN", None) is None:
        return None
    return {
        "base_url": getattr(module, "_BASE_URL", None),
        "api_key_env": getattr(module, "_API_KEY_ENV", None),
        "model": getattr(backbone, "model_name", None),
        "provider_pin": getattr(module, "_PROVIDER_PIN", None),
        "reasoning_deciding": getattr(module, "_REASONING_ENABLED", None),
        "reasoning_grounding": getattr(module, "_REASONING_DISABLED", None),
        "note": "config echo — what this arm sends, not evidence of what served",
    }


def _write_config_dump(preflight: Preflight, args: argparse.Namespace, agent) -> Path:
    """The full resolved config + every check result, into the run record."""
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / "launch_preflight.json"
    embedder = getattr(getattr(agent, "retrieval_agent", None), "embedder", None)
    payload = {
        "mode": preflight.mode,
        "argv": list(sys.argv),
        "args": vars(args),
        "required_retrieval_agent": MODE_RETRIEVAL_AGENT[args.retrieval_off],
        "agent_switch_state": agent.ablation_switch_state() if agent is not None else None,
        "retrieval_text_arm": {
            "embedder": type(embedder).__name__ if embedder is not None else None,
            "available": bool(getattr(embedder, "available", False)),
            "model_name": getattr(embedder, "model_name", None),
        },
        "checks": preflight.checks,
        "passed": not preflight.failed,
    }
    # Only for a backbone that pins a route (see above) — every other backbone's
    # dump carries no route block.
    route = _backbone_route_facts(agent)
    if route is not None:
        payload["backbone_route"] = route
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retrieval_off", action="store_true",
                   help="The retrieval mode the run will launch in. Spelled exactly "
                        "as on run_graphpc.py, so the gate and the run cannot drift.")
    p.add_argument("--result_dir", required=True,
                   help="The run record directory the config dump is written into.")
    p.add_argument("--working_store", default=None,
                   help="The --support_set_root the run will use (retrieval on: a "
                        "restored copy of the frozen store; retrieval off: the store "
                        "the bypass must leave untouched). Required in every mode "
                        "but --site_mapping_live, which takes the runner argv's.")
    p.add_argument("--frozen_store_root", default=None,
                   help="The CELL'S frozen master store root (warm mode; REQUIRED "
                        "there — no default, per-cell facts are inputs).")
    p.add_argument("--expected_entry_count", type=int, default=None,
                   help="The CELL'S frozen-store entry count (warm mode; REQUIRED "
                        "there).")
    p.add_argument("--strata_file", default=None,
                   help="The CELL'S pre-registration JSON (warm mode; REQUIRED "
                        "there). Verified for internal consistency + store family "
                        "coverage, never against baked-in numbers.")
    p.add_argument("--cold_store", action="store_true",
                   help="Cold-run store mode: retrieval ON starting from an EMPTY "
                        "store. Replaces the frozen-master / stratum / diet checks "
                        "with empty-store + write-path checks; the embedder check "
                        "is unchanged. Incompatible with --retrieval_off.")
    p.add_argument("--backbone", default="claude", choices=["claude", "gpt", "deepseek", "minimax", "qwen"])
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--site_mapping_live", action="store_true",
                   help="Live site-URL mode: URL line, mapping and homepage against the live "
                        "sites, with the agent built from the cell's runner argv given after `--` (see the "
                        "module docstring). WebArena interpreter; no model call.")
    p.add_argument("--live_config_file", default=None,
                   help="--site_mapping_live: the READ-ONLY WebArena task config the env is reset on "
                        "(an on-host start page; e.g. a map string_match task).")
    p.add_argument("--homepage_app_root", default=None,
                   help="--site_mapping_live: the directory of the served homepage app (holding templates/).")
    return p


def _main_site_mapping_live(parser: argparse.ArgumentParser, args: argparse.Namespace, runner_argv: list[str]) -> int:
    set_elsewhere = [flag for flag, value in (
        ("--retrieval_off", args.retrieval_off), ("--cold_store", args.cold_store),
        ("--working_store", args.working_store), ("--frozen_store_root", args.frozen_store_root),
        ("--expected_entry_count", args.expected_entry_count), ("--strata_file", args.strata_file),
    ) if value]
    if set_elsewhere:
        parser.error(f"--site_mapping_live takes the run's retrieval mode and store from the runner argv; "
                     f"drop {', '.join(set_elsewhere)}")
    missing = [flag for flag, value in (("--live_config_file", args.live_config_file),
                                        ("--homepage_app_root", args.homepage_app_root)) if not value]
    if missing or not runner_argv:
        parser.error(f"--site_mapping_live needs {', '.join(missing) or ''}"
                     f"{' and ' if missing and not runner_argv else ''}"
                     f"{'the cell runner argv after `--`' if not runner_argv else ''}")
    preflight = Preflight("site_mapping_live")
    print("launch preflight — site_mapping_live")
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001 — an exported environment works too.
        pass
    summary = _run_site_mapping_live(preflight, args, runner_argv)
    preflight.check("config_dump_written", lambda: str(_write_site_mapping_live_dump(preflight, args, summary)))
    print("\nsummary for the launch record:")
    print(f"  table sha256: {summary['table_sha256']}")
    print(f"  rendered sample: {json.dumps(summary['rendered_sample'], indent=2)}")
    if preflight.failed:
        print(f"\nPREFLIGHT FAILED (site_mapping_live): {len(preflight.failed)} check(s) failed — "
              f"{', '.join(preflight.failed)}. DO NOT LAUNCH.")
        return 1
    print(f"\npreflight passed (site_mapping_live): {len(preflight.checks)} check(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    runner_argv: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, runner_argv = argv[:split], argv[split + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.site_mapping_live:
        return _main_site_mapping_live(parser, args, runner_argv)
    if runner_argv:
        parser.error("a runner argv after `--` is taken only by --site_mapping_live")
    if args.working_store is None:
        parser.error("--working_store is required (every mode but --site_mapping_live)")
    if args.cold_store and args.retrieval_off:
        parser.error("--cold_store is a retrieval-ON mode (the run grows the store); "
                     "it cannot combine with --retrieval_off")
    if not args.cold_store and not args.retrieval_off:
        missing = [flag for flag, value in (
            ("--frozen_store_root", args.frozen_store_root),
            ("--expected_entry_count", args.expected_entry_count),
            ("--strata_file", args.strata_file),
        ) if value is None]
        if missing:
            parser.error(
                "warm mode takes the CELL'S facts as inputs (no baked-in cell "
                f"constants): missing {', '.join(missing)}"
            )
    mode = (
        "retrieval_off" if args.retrieval_off
        else "cold_store" if args.cold_store
        else "retrieval_on"
    )
    preflight = Preflight(mode)
    print(f"launch preflight — {mode}")

    # Same .env load the runner does, so backbone construction here sees the same
    # credentials the run will.
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001 — an exported environment works too.
        pass

    agent = None
    try:
        agent = _build_agent(args)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  agent_construction: {type(exc).__name__}: {exc}")
        preflight.checks.append(
            {"check": "agent_construction", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}
        )

    if agent is not None:
        preflight.check("retrieval_mode_echoed", lambda: _check_retrieval_mode_echoed(agent, args.retrieval_off))

    if args.retrieval_off:
        if agent is not None:
            preflight.check("retrieval_bypass_active", lambda: _check_retrieval_bypass(agent, args))
    elif args.cold_store:
        preflight.check("embedder_live", _check_embedder_live)
        preflight.check("cold_store_empty", lambda: _check_cold_store_empty(args))
        preflight.check("store_write_path", lambda: _check_store_write_path(args))
    else:
        preflight.check("embedder_live", _check_embedder_live)
        preflight.check("store_matches_frozen_master", lambda: _check_store_frozen(args))
        preflight.check("strata_file_consistent", lambda: _check_strata_file(args))
        # Independent of agent construction: the probe builds its own temp
        # store + retrieval agent, so a failed backbone build cannot hide it.
        preflight.check("injection_diet_enforced", _check_injection_diet_enforced)

    if agent is not None:
        preflight.check("config_dump_written", lambda: str(_write_config_dump(preflight, args, agent)))

    if preflight.failed:
        print(f"\nPREFLIGHT FAILED ({mode}): {len(preflight.failed)} check(s) failed — "
              f"{', '.join(preflight.failed)}. DO NOT LAUNCH.")
        return 1
    print(f"\npreflight passed ({mode}): {len(preflight.checks)} check(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
