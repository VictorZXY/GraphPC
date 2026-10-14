"""Loud-fail gate: are the two arms' vendored runners still the same machine?

Offline. No AWS, no network, no model. Run it before a scored launch, and after
any edit to either runner, from the GraphPC repo root:

    python scripts/osworld/verify_runner_deltas.py

or, synced, from the OSWorld-V2 checkout root on the host (so it checks the
files that will actually run):

    uv run --no-sync python scripts/python/verify_runner_deltas.py

WHY THIS FILE EXISTS AT ALL. The properties below are what make the paired
comparison a comparison: if the two arms' guests differ, the delta measures the
harness. A check that gates a run has to be a runnable that ships with the code,
because an operator must be able to run it on the machine that is about to spend
money.

Every check either passes, or FAILS — none of them returns quietly. A check that
could not run has not passed, and that distinction is the whole point of a gate.

  1. Both runners carry the identical ``--volume_size`` plumbing (upstream's own
     flag, ported verbatim). A flag on one arm only would make the arms run on
     differently-configured machines.
  2. The baseline runner IS upstream's ``run_multienv_m3.py`` plus exactly that
     port: with its header and the port reversed, the file is byte-identical to
     the pinned upstream file (a sibling on the host, the staging clone locally).
     The control's whole evidentiary value is being upstream's code, so this is
     held against the file itself, not against a claim in a header. An absent
     upstream file FAILS naming both paths tried; it never skips.
  3. Each arm's ``get_unfinished`` has the wipe shape its runner claims, and the
     baseline's containment holds. The GRAPHPC wipe is scoped to the tasks the
     invocation asked for and survives a subdirectory (deltas 10 + 4). The
     BASELINE wipe is upstream's, verbatim: unscoped, and it raises on a task dir
     holding a subdirectory. That stays upstream's — the control's value is being
     upstream's code — so what is checked instead is the containment it depends
     on: the arms default to distinct result trees, both launchers CALL the
     launch-argument guard (which is what keeps a declared-unscored launch out of
     either scored tree at the moment an operator types it), and the baseline
     launcher still CALLS its pre-clean. The two call-site checks each carry a
     negative control that runs every invocation. See the check for the argument.
  4. Unused (the number stays, as the runner's delta numbers do).
  5. The GraphPC runner's delta 15: its own module-level ``--max_steps`` check, executed per
     ``--backbone`` against the adapter's ``OSWORLD_MAX_STEPS_BY_BACKBONE`` as the adapter file defines
     it (``src/`` in the repo, the vendored package on the host; the Qwen GraphPC arm runs an 800-step budget
     with the stuck-stop; every other arm, and every control, runs 500), passes
     each backbone's budget only and exits 1 one step either side, at upstream's default 100 and at
     every other backbone's budget, and it runs before logging is set up (so before any log file or
     VM). An adapter without the table, or a MiniMax budget other than OSWORLD_MAX_STEPS (500), FAILS;
     it never skips. Its second statement, the refusal of a launch without
     ``--retrieval_off`` (every GraphPC run on the 28-task subset runs retrieval off), executed: exit 1
     without the flag, nothing
     with it, placed right after the budget check and so before any gate, log file or VM.

Checks 3 and 5 execute the shipped code; they are not text matches.
Checks 2, 3 and 5 each carry negative controls that run on every invocation.

QWEN MODE (``--cell qwen``) replaces checks 1-5 with the Qwen cell's two (Q1, Q2; see the
section above ``main``): the vendored control is upstream's osworld-v2.1 files minus exactly
their declared hunks, and the two Qwen-cell runners feed upstream's driver and DesktopEnv the
same values. Q1 needs a clone that has the osworld-v2.1 tag (the staging clone:
``--checkout_root``); Q2 resolves both Qwen-cell launchers under DRY_RUN.

DEEPSEEK MODE (``--cell deepseek``) replaces them with the DeepSeek cell's three (DS1-DS3; see
the section above ``main``): the vendored control package is upstream's v2026.08.08 mm_agents/m3
minus exactly its declared hunks, its runner is upstream's run_multienv_m3.py minus exactly its
three (with what each hunk declares held directly), and the two DeepSeek-cell runners feed
upstream's driver and DesktopEnv the same values, each launch reading the 28.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

# scripts/osworld/<this> in the repo, scripts/python/<this> in the checkout:
# either way both runners are siblings.
HERE = Path(__file__).resolve().parent
GRAPHPC_RUNNER = HERE / "run_multienv_graphpc.py"
BASELINE_RUNNER = HERE / "run_multienv_m3_baseline.py"


def _launcher(name: str) -> Path:
    """The bash launchers are siblings in the repo (scripts/osworld/) but live in
    scripts/bash/ once synced into the checkout, where this file is scripts/python/."""
    sibling = HERE / name
    return sibling if sibling.is_file() else HERE.parent / "bash" / name


GRAPHPC_LAUNCHER = _launcher("run_multienv_graphpc.sh")
BASELINE_LAUNCHER = _launcher("run_multienv_m3_baseline.sh")
CELL_CONFIG = _launcher("osworld_cell_config.sh")

# A CALL SITE, NOT A NAME. The bare string "osworld_baseline_resume_preclean" is satisfied by the
# function's own DEFINITION and its header comment both — so with that needle, deleting the call and
# keeping the function would leave this check green while a resumed control arm dies on upstream's
# IsADirectoryError. A needle that matches a definition can never tell you there is a caller. These
# match a line that is the invocation and nothing else; a definition line ends in `() {`.
PRECLEAN_NAME = "osworld_baseline_resume_preclean"
PRECLEAN_CALL = re.compile(rf"^[ \t]*{PRECLEAN_NAME}[ \t]*$", re.MULTILINE)
GUARD_NAME = "osworld_cell_guard_launch_args"
# The guard is invoked with the arm, its write paths as <option> <value> pairs, then `-- "$@"`.
# Matched across line continuations (the GraphPC arm's call wraps), and the `-- "$@"` tail is
# part of the needle: without it the guard would not see what was appended at all.
GUARD_CALL = re.compile(
    rf'^[ \t]*{GUARD_NAME}[ \t]+(graphpc|baseline)\b(?:[^\n]|\\\n)*?--[ \t]+"\$@"[ \t]*$',
    re.MULTILINE)

# The two fragments of upstream's --volume_size flag, which both arms must carry
# identically (argparse entry, then the per-task lookup in _build_env_spec).
VOLUME_FRAGMENTS = (
    '        "--volume_size",\n',
    '            task_volume_size = example.get("volume_size")\n'
    '            if task_volume_size is None:\n'
    '                task_volume_size = args.volume_size\n',
)

# The pinned upstream file the baseline runner is a copy of. On the host it is a sibling of the
# vendored copy (scripts/python/); in the repo it is the staging clone next to the repository.
# Neither existing is a FAIL that names both, never a skip.
UPSTREAM_RUNNER_CANDIDATES = (
    HERE / "run_multienv_m3.py",
    HERE.parents[2] / "OSWorld-V2" / "scripts" / "python" / "run_multienv_m3.py",
)
# The vendored header sits between upstream's second docstring line and the closing quotes;
# upstream's docstring closes right after that line.
HEADER_ANCHOR = "Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.\n"
# The --volume_size port IN FULL, as it has to be reversed to get upstream back: the argparse
# entry (removed) and the per-task lookup in _build_env_spec (replaced by upstream's one-line
# read). VOLUME_FRAGMENTS above are the parts both arms must share; these are the whole port.
VOLUME_ARGPARSE_ENTRY = '''    parser.add_argument(
        "--volume_size",
        type=int,
        default=None,
        help="Default root volume size in GB for tasks without their own volume_size",
    )
'''
VOLUME_LOOKUP_PORTED = '''            task_volume_size = example.get("volume_size")
            if task_volume_size is None:
                task_volume_size = args.volume_size
            spec = {
                "os_type": task_os_type,
                "instance_type": example.get("instance_type"),
                "volume_size": task_volume_size,
'''
VOLUME_LOOKUP_UPSTREAM = '''            spec = {
                "os_type": task_os_type,
                "instance_type": example.get("instance_type"),
                "volume_size": example.get("volume_size"),
'''

failures: list[str] = []
notes: list[str] = []


def fail(message: str) -> None:
    failures.append(message)


def _read(path: Path) -> str:
    """Reading is itself a check: a missing OR EMPTY file must FAIL here, because
    every caller below degrades to "nothing to compare", which would otherwise
    read as a pass."""
    if not path.is_file():
        fail(f"missing file: {path}")
        return ""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        fail(f"empty file: {path}")
    return text


# --------------------------------------------------------------------------
# 1: the two arms carry the same delta, byte for byte.
# --------------------------------------------------------------------------
def check_arms_carry_identical_deltas() -> None:
    graphpc, baseline = _read(GRAPHPC_RUNNER), _read(BASELINE_RUNNER)
    if not graphpc or not baseline:
        return

    volume_ok = True
    for index, fragment in enumerate(VOLUME_FRAGMENTS, start=1):
        for name, text in (("graphpc", graphpc), ("baseline", baseline)):
            if fragment not in text:
                fail(f"{name} runner is missing --volume_size fragment {index}")
                volume_ok = False
    # Local, not `if not failures`: an unrelated earlier failure must not decide
    # whether THIS property gets reported.
    if volume_ok:
        notes.append("--volume_size plumbing present and identical in both arms")


# --------------------------------------------------------------------------
# 2: the baseline runner is upstream's file plus exactly the port.
# --------------------------------------------------------------------------
def _baseline_beyond_upstream(baseline: str, upstream: str) -> list[str]:
    """Everything the baseline runner differs from upstream by ONCE its header and the
    --volume_size port are reversed; empty means "upstream plus exactly the port". A pure
    function of the two texts, so the negative controls can run it on doctored copies."""
    start = baseline.find(HEADER_ANCHOR)
    if start < 0:
        return ["the baseline runner no longer opens with upstream's docstring, so its header "
                "cannot be located and nothing below it was compared"]
    start += len(HEADER_ANCHOR)
    end = baseline.find('"""\n', start)
    if end < 0:
        return ["the baseline runner's docstring never closes, so its header cannot be located"]
    restored = baseline[:start] + baseline[end:]
    problems = []
    for fragment, replacement, what in ((VOLUME_ARGPARSE_ENTRY, "", "argparse entry"),
                                        (VOLUME_LOOKUP_PORTED, VOLUME_LOOKUP_UPSTREAM, "_build_env_spec lookup")):
        count = restored.count(fragment)
        if count != 1:
            problems.append(f"the --volume_size {what} appears {count} time(s) in the baseline runner, "
                            f"expected exactly 1: the port is missing, doubled or no longer verbatim")
            continue
        restored = restored.replace(fragment, replacement)
    if problems:
        return problems
    if restored != upstream:
        diff = list(difflib.unified_diff(upstream.splitlines(), restored.splitlines(),
                                         "upstream run_multienv_m3.py",
                                         "baseline runner with its header and the port reversed",
                                         lineterm="", n=0))
        problems.append("the baseline runner differs from upstream BEYOND its header and the --volume_size "
                        "port — an undeclared delta on the control arm:\n  "
                        + "\n  ".join(diff[:12]) + ("\n  ..." if len(diff) > 12 else ""))
    return problems


def check_the_baseline_is_upstream_plus_the_port() -> None:
    upstream_path = next((path for path in UPSTREAM_RUNNER_CANDIDATES if path.is_file()), None)
    if upstream_path is None:
        fail("upstream run_multienv_m3.py not found at any of: "
             + ", ".join(str(path) for path in UPSTREAM_RUNNER_CANDIDATES)
             + " — so the baseline runner was NOT CHECKED against upstream (on the host it is a sibling "
               "of the vendored copy; locally it is the OSWorld-V2 staging clone next to the repository)")
        return
    baseline, upstream = _read(BASELINE_RUNNER), _read(upstream_path)
    if not baseline or not upstream:
        return
    problems = _baseline_beyond_upstream(baseline, upstream)
    if problems:
        for problem in problems:
            fail(f"baseline runner vs upstream ({upstream_path}): {problem}")
        return
    # NEGATIVE CONTROLS, every invocation: the comparison has to go red on ONE extra line and on
    # a removed port, or the green above says nothing about what it compared.
    if not _baseline_beyond_upstream(baseline + "# one extra line\n", upstream):
        fail("negative control: the baseline-vs-upstream comparison still passes with ONE extra line "
             "added to the baseline runner, so it is not comparing the files")
        return
    without_port = (baseline.replace(VOLUME_ARGPARSE_ENTRY, "", 1)
                    .replace(VOLUME_LOOKUP_PORTED, VOLUME_LOOKUP_UPSTREAM, 1))
    if without_port == baseline or not _baseline_beyond_upstream(without_port, upstream):
        fail("negative control: the comparison still passes with the --volume_size port REMOVED from "
             "the baseline runner, so it does not demand the port")
        return
    notes.append(f"baseline runner == upstream run_multienv_m3.py ({upstream_path}) + exactly the "
                 f"--volume_size port (argparse entry + _build_env_spec lookup), header excluded; "
                 f"negative controls: one extra line and a removed port both rejected")


# --------------------------------------------------------------------------
# 3: the resume wipe cannot leave its own filter.
# --------------------------------------------------------------------------
def _extract_function(source_path: Path, name: str):
    """Compile ONE function out of a runner without importing it (the runners
    parse argv, open log files and gate on an API key at import)."""
    text = _read(source_path)
    if not text:
        return None
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace: dict = {"os": __import__("os"), "shutil": shutil}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
            return namespace[name]
    fail(f"{source_path.name} has no function {name}()")
    return None


MARKER = "graphpc_telemetry.json"


def _populate_result_tree(tmp: str, *, with_subdir: bool) -> tuple:
    """Four in-flight (unfinished) task dirs: the one a --specific_task_id smoke
    asks for, another task in the same domain, a whole other domain, and one under
    a DIFFERENT MODEL directory.

    ``with_subdir`` matters because upstream's wipe RAISES on the first task dir
    holding one — so its unscoped REACH is only observable on a tree without.

    The model dimension is here because the baseline launcher's pre-clean comment
    rests on it. ``get_unfinished`` builds ``result_dir/action_space/
    observation_type/use_model`` and walks only that, so it is UNSCOPED IN THE TASK
    DIMENSION AND SCOPED IN THE MODEL ONE — while the pre-clean's
    ``${model_root}/*/*/*/`` glob iterates model/domain/task and therefore reaches
    one level further than the wipe it exists to protect. That asymmetry is stated
    in the launcher, and stating it without checking it would just move the
    unchecked claim."""
    root = Path(tmp) / "pyautogui" / "screenshot"

    def in_flight(model: str, domain: str, task_id: str) -> Path:
        path = root / model / domain / task_id
        path.mkdir(parents=True)
        if with_subdir:
            (path / "api_logs").mkdir()
            (path / "api_logs" / "001.json").write_text("{}", encoding="utf-8")
        (path / MARKER).write_text("{}", encoding="utf-8")
        return path

    return (in_flight("MiniMax-M3", "tasks", "004"), in_flight("MiniMax-M3", "tasks", "017"),
            in_flight("MiniMax-M3", "other", "099"), in_flight("Some-Other-Model", "tasks", "004"))


def _drive_wipe(get_unfinished, arm: str, *, with_subdir: bool):
    """Run one arm's wipe against a fresh tree. Returns None if the fixture itself
    failed — the negative control, so a fixture that stopped creating files fails
    on THAT ground instead of making every survival assertion pass vacuously."""
    with tempfile.TemporaryDirectory() as tmp:
        mine, other_task, other_domain, other_model = _populate_result_tree(tmp, with_subdir=with_subdir)
        if not all((p / MARKER).is_file() for p in (mine, other_task, other_domain, other_model)):
            fail(f"{arm}: the wipe-scope fixture did not create its four in-flight task dirs, so "
                 f"nothing was actually tested. Fix the fixture; this gate is NOT green.")
            return None
        raised, remaining = None, None
        try:
            remaining = get_unfinished("pyautogui", "MiniMax-M3", "screenshot", tmp, {"tasks": ["004"]})
        except Exception as exc:  # noqa: BLE001 — the behaviour under test
            raised = type(exc).__name__
        return {
            "raised": raised,
            "remaining": remaining,
            "in_scope_emptied": not list(mine.iterdir()),
            "out_of_scope_survived": (other_task / MARKER).is_file() and (other_domain / MARKER).is_file(),
            "other_model_survived": (other_model / MARKER).is_file(),
        }


def check_the_resume_wipe_is_scoped() -> None:
    """PER ARM, because the two arms differ here and the property is not shared.

    GraphPC's runner carries deltas 4 + 10 and its wipe is scoped and
    directory-safe. The BASELINE runner's ``get_unfinished`` is upstream's,
    verbatim: it empties every unfinished task dir under --result_dir regardless
    of the invocation's filter, and it raises IsADirectoryError on a task dir
    containing a subdirectory. That is asserted here, not assumed: given
    {"tasks": ["004"]} it empties task 017 and the whole "other" domain, and with
    api_logs/ present it raises partway through.

    WHY THAT IS AN OPERATIONAL RULE HERE AND NOT A THIRD DELTA. The baseline
    runner's entire evidentiary value is that it is upstream's own code — it is
    the control. Its one existing delta is an ops property that must hold
    IDENTICALLY IN BOTH ARMS or the comparison measures the harness; the wipe
    scope is not such a property:
      * the arms write to distinct result trees, so neither wipe can reach the
        other's data — this is per-arm data safety, not a comparison property;
      * the scored run is UNFILTERED (the full manifest), so every task in the
        baseline tree is in scope and upstream's wipe does exactly what ours
        does on the production path;
      * the one production-path hazard, IsADirectoryError on a plain resume, is
        already handled OUTSIDE the vendored file by run_multienv_m3_baseline.sh's
        pre-clean, which removes those subdirectories before the runner starts.
    What is left is a mis-aimed smoke, which the launch guard's throwaway-RESULT_DIR
    rule already covers. So the containment is checked here instead: the shapes
    of both wipes are asserted, and so are the two facts the rule rests on.

    The baseline assertions therefore RECORD upstream's behaviour rather than
    endorse it. If upstream's shape changes, or someone quietly adds a scoping
    delta, this fails — which is correct: the header claims exactly one delta."""
    scripts_ok = _check_the_wipe_containment_holds()

    graphpc = _extract_function(GRAPHPC_RUNNER, "get_unfinished")
    baseline = _extract_function(BASELINE_RUNNER, "get_unfinished")
    for arm, fn in (("GraphPC", graphpc), ("baseline", baseline)):
        if fn is None:
            # A check that could not RUN has not passed. Returning quietly here would
            # make a rename or a refactor of get_unfinished look like a green gate on
            # the one property whose failure destroys a scored run's data.
            fail(f"could not extract get_unfinished() from the {arm} runner, so its wipe shape was "
                 f"NOT CHECKED. Fix the extraction (a rename or a refactor of that function will do "
                 f"it) before reading this gate as green.")

    # ---- GraphPC: scoped AND directory-safe, both observable on one tree.
    if graphpc is not None:
        out = _drive_wipe(graphpc, "GraphPC", with_subdir=True)
        if out is not None:
            if out["raised"] is not None:
                fail(f"GraphPC: get_unfinished raised {out['raised']} on a task dir containing a "
                     f"subdirectory; delta 4 (the isdir guard) is gone and a resume dies partway "
                     f"through, having already deleted files.")
            elif not out["out_of_scope_survived"]:
                fail("GraphPC: get_unfinished DELETED task directories outside its "
                     "--specific_task_id / --domain filter — delta 10 is gone. A smoke against a "
                     "scored --result_dir would destroy that run's in-flight tasks. Do NOT launch; "
                     "and until this passes, every smoke must use a throwaway RESULT_DIR.")
            elif not out["in_scope_emptied"]:
                fail("GraphPC: get_unfinished did not clean the in-scope unfinished task; a resume "
                     "would re-run it on top of a partial directory")
            elif not out["other_model_survived"]:
                fail("GraphPC: get_unfinished reached into ANOTHER MODEL's subtree of the same "
                     "result tree. Upstream's builds result_dir/action_space/observation_type/"
                     "use_model and walks only that; a wipe that crosses models invalidates the "
                     "launcher's pre-clean comment and widens the blast radius of every smoke.")
            elif out["remaining"] != {"tasks": ["004"]}:
                fail(f"GraphPC: get_unfinished returned {out['remaining']!r}, expected "
                     f"{{'tasks': ['004']}}")
            else:
                notes.append("GraphPC arm: the resume wipe stays inside --specific_task_id / "
                             "--domain and inside its own model directory, and survives a "
                             "subdirectory (deltas 10 + 4)")

    # ---- baseline: upstream's shape, asserted on TWO trees because the raise
    #      hides the reach — on a tree with subdirectories it dies before it ever
    #      gets to the out-of-scope tasks. Both directions fail: an undeclared
    #      delta is as much a finding as a regression, the header claiming one.
    if baseline is not None:
        reach = _drive_wipe(baseline, "baseline", with_subdir=False)
        crash = _drive_wipe(baseline, "baseline", with_subdir=True)
        if reach is not None and crash is not None:
            if reach["raised"] is not None:
                fail(f"baseline: get_unfinished raised {reach['raised']} on a plain tree; that is not "
                     f"upstream's shape and this gate can no longer characterise the control arm.")
            elif reach["out_of_scope_survived"]:
                fail("baseline: get_unfinished did NOT reach outside the filter, but upstream's does. "
                     "An undeclared scoping delta has been added to the control arm's runner, whose "
                     "header claims exactly one delta. Declare it in the header or revert it.")
            elif not reach["other_model_survived"]:
                fail("baseline: get_unfinished reached into ANOTHER MODEL's subtree. Upstream's is "
                     "scoped to use_model, so this is either an undeclared delta or upstream moved "
                     "— and the launcher's pre-clean comment, which states that the pre-clean "
                     "reaches one level FURTHER than this wipe, is no longer true.")
            elif crash["raised"] != "IsADirectoryError":
                fail(f"baseline: get_unfinished did NOT raise IsADirectoryError on a task dir holding "
                     f"a subdirectory (got {crash['raised']!r}) — upstream's does. Either upstream "
                     f"moved or an undeclared second delta was added; re-check the header, and if the "
                     f"wipe is now directory-safe the launcher's pre-clean may no longer be needed.")
            else:
                notes.append("baseline arm: the resume wipe is upstream's — unscoped in the TASK "
                             "dimension, scoped to its own MODEL directory, and not directory-safe, "
                             "all verified — and its containment holds"
                             + ("" if scripts_ok else ", EXCEPT as reported above"))


# The launcher's DRY_RUN short-circuit, verbatim. A call placed after this line and before the
# matching `else` runs only under DRY_RUN=1, i.e. never on a real launch.
DRY_RUN_BRANCH = 'if [[ "${DRY_RUN}" == "1" ]]; then'
# The launchers keep their `RESULT_DIR:-./results/...` defaults relative (that is what an operator
# reads, and what the runner resolves against its own cwd); the guard's scored constants are
# anchored to the checkout. Comparing the two means dropping whichever prefix each carries.
CELL_ROOT_VAR = "OSWORLD_CELL_ROOT"


def _unanchor(path: str) -> str:
    for prefix in (f"${{{CELL_ROOT_VAR}}}/", "./"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _is_on_the_live_launch_path(text: str, needle: re.Pattern) -> bool:
    """Does ``needle`` match a line OUTSIDE the launcher's DRY_RUN short-circuit?

    A call-site needle proves a caller exists, not that a real launch reaches it —
    moving the invocation into the DRY_RUN branch would keep one green while no
    resume was ever cleaned. The branch is found by its literal text and closed at
    the first column-0 ``else`` / ``fi`` after it, which is this file's shape; a
    reshape makes the branch unfindable and that is reported rather than passed."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == DRY_RUN_BRANCH)
    except StopIteration:
        return True  # no DRY_RUN branch at all: every call site is on the live path
    end = next((i for i in range(start + 1, len(lines)) if lines[i] in ("else", "fi")), len(lines))
    inside = set(range(start + 1, end))
    offset, live = 0, False
    for index, line in enumerate(lines):
        if needle.match(line) and index not in inside:
            live = True
        offset += len(line) + 1
    return live


def _check_the_wipe_containment_holds() -> bool:
    """The facts the baseline's unscoped wipe rests on, checked rather than assumed:
    the arms cannot share a result tree by default, the baseline launcher still
    CALLS the pre-clean, and both launchers still call the launch-argument guard
    that keeps a declared-unscored launch out of either scored tree.

    Each of the two call-site checks carries a NEGATIVE CONTROL that runs on every
    invocation: the call is removed from the text and the check must go red on
    that text WHILE THE NAME IS STILL IN IT, which is exactly the case a bare
    substring needle accepts."""
    ok = True
    result_dirs = {}
    texts = {}
    for arm, script in (("GraphPC", GRAPHPC_LAUNCHER), ("baseline", BASELINE_LAUNCHER)):
        text = _read(script)
        if not text:
            return False
        texts[arm] = text
        match = re.search(r'^RESULT_DIR="\$\{RESULT_DIR:-([^}"]+)\}"', text, re.MULTILINE)
        if match is None:
            fail(f"{arm}: could not read the default RESULT_DIR out of {script.name}, so the claim that "
                 f"the arms cannot share a result tree was NOT CHECKED")
            ok = False
            continue
        result_dirs[arm] = match.group(1)
    if len(result_dirs) == 2 and result_dirs["GraphPC"] == result_dirs["baseline"]:
        fail(f"both arms default to RESULT_DIR={result_dirs['GraphPC']}: the baseline's unscoped wipe "
             f"would reach the GraphPC arm's trajectories, which is the containment that lets that "
             f"wipe stay upstream's. Separate them or scope the wipe.")
        ok = False

    # ---- the launch-argument guard, on BOTH arms. It is the only thing that enforces "an
    #      unscored launch does not write into a scored tree" at the moment an operator types it;
    #      this file runs before launch and cannot see the arguments.
    for arm, text in texts.items():
        if GUARD_CALL.search(text) is None:
            fail(f"{arm}: run_multienv_*.sh does not call {GUARD_NAME} <arm> <write paths> -- \"$@\". "
                 f"Without the call, any flag appended at launch reaches the runner unexamined (the "
                 f"preflight resolves these scripts BARE) and a launch can be pointed at the other arm's "
                 f"scored tree, where the resume pre-clean and upstream's get_unfinished empty the "
                 f"in-flight episodes. Without the `-- \"$@\"` tail the guard never sees what was "
                 f"appended; without the write paths it cannot tell where the launch would write.")
            ok = False
            continue
        stripped = GUARD_CALL.sub("", text)
        if GUARD_CALL.search(stripped) is not None:
            fail(f"{arm}: negative control — the guard call-site check still passes after the call was "
                 f"removed, so it is not testing for a caller")
            ok = False

    # ---- the pre-clean must be CALLED, not merely defined (see PRECLEAN_CALL).
    baseline_text = texts.get("baseline", "")
    if baseline_text:
        if PRECLEAN_CALL.search(baseline_text) is None:
            fail(f"run_multienv_m3_baseline.sh never CALLS {PRECLEAN_NAME} (a definition alone is not a "
                 f"caller): upstream's get_unfinished raises IsADirectoryError on a resumed task dir "
                 f"holding api_logs/, and that pre-clean is the only thing preventing it. A resume of the "
                 f"control arm would die.")
            ok = False
        else:
            # NEGATIVE CONTROL, every invocation: remove ONLY the call. The check must go red on a
            # text that STILL CONTAINS THE NAME — the definition-only case a bare substring needle
            # passes.
            without_call = PRECLEAN_CALL.sub("", baseline_text)
            if PRECLEAN_CALL.search(without_call) is not None:
                fail("negative control: the pre-clean call-site check still passes on a launcher whose "
                     "CALL was removed — it cannot tell a caller from a definition, which is the defect "
                     "a bare substring check has")
                ok = False
            elif PRECLEAN_NAME not in without_call:
                fail("negative control is vacuous: removing the pre-clean CALL also removed its "
                     "DEFINITION from the text, so the control never exercises the definition-only case "
                     "a bare substring needle accepts")
                ok = False
            # A CALLER IS NOT ENOUGH: it has to be on the path a REAL launch takes. The script
            # skips the pre-clean under DRY_RUN=1, so a call moved into that branch keeps a
            # call-site needle green while no resume is ever cleaned. Check the branch it sits in.
            elif not _is_on_the_live_launch_path(baseline_text, PRECLEAN_CALL):
                fail("run_multienv_m3_baseline.sh calls osworld_baseline_resume_preclean only on the "
                     "DRY_RUN path. DRY_RUN prints the argv and exits, so a real resume would never be "
                     "cleaned and upstream's get_unfinished would raise IsADirectoryError. The call "
                     "belongs in the else branch, where the launch actually happens.")
                ok = False
            else:
                # NEGATIVE CONTROL for that branch test, every invocation: move the call into the
                # DRY_RUN branch and require the check to go red there.
                moved = PRECLEAN_CALL.sub("", baseline_text).replace(
                    DRY_RUN_BRANCH, DRY_RUN_BRANCH + f"\n    {PRECLEAN_NAME}", 1)
                if _is_on_the_live_launch_path(moved, PRECLEAN_CALL):
                    fail("negative control: the pre-clean branch test still passes with the call moved "
                         "into the DRY_RUN branch, so it is not testing which branch the call is in")
                    ok = False

    # ---- the guard's scored-tree list against each launcher's own default, so the two literals
    #      cannot drift apart unnoticed (the launchers keep their inline defaults because that is
    #      what an operator reads and what the arms-identity check parses).
    config_text = _read(CELL_CONFIG)
    if not config_text:
        ok = False
    else:
        declared = {}
        for arm, variable in (("GraphPC", "OSWORLD_SCORED_PATHS_GRAPHPC"),
                              ("baseline", "OSWORLD_SCORED_PATHS_BASELINE")):
            block = re.search(rf"^{variable}=\((.*?)^\)", config_text, re.MULTILINE | re.DOTALL)
            if block is None:
                fail(f"{CELL_CONFIG.name} declares no {variable}, so the launch guard has nothing to "
                     f"hold a launch's write paths against for the {arm} arm")
                ok = False
                continue
            declared[arm] = re.findall(r'"([^"]+)"', block.group(1))
        # THE SCORED CONSTANTS MUST BE ANCHORED TO THE CHECKOUT, not relative. `realpath -m`
        # resolves a relative path against the CALLER's cwd, so with relative constants the guard
        # would recognise a scored tree only when the launcher is invoked from the checkout
        # root: from $HOME the same absolute scored path would PASS, with the guard printing its
        # declared-unscored warning about the arm's own scored tree.
        unanchored = {arm: [d for d in paths if not d.startswith(f"${{{CELL_ROOT_VAR}}}/")]
                      for arm, paths in declared.items()}
        unanchored = {arm: paths for arm, paths in unanchored.items() if paths}
        if unanchored:
            fail(f"scored path(s) {unanchored} are not anchored to ${{{CELL_ROOT_VAR}}}: a relative "
                 f"constant resolves against the CALLER's cwd, so the guard stops recognising either "
                 f"arm's scored tree the moment a launcher is invoked from anywhere but the checkout root")
            ok = False
        if f'{CELL_ROOT_VAR}="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/../.." && pwd)"' not in config_text:
            fail(f"{CELL_CONFIG.name} no longer derives {CELL_ROOT_VAR} from BASH_SOURCE two levels up "
                 f"(the checkout root in both layouts, scripts/osworld/ here and scripts/bash/ on the "
                 f"host), so the scored constants are anchored to something else or to nothing")
            ok = False
        if len(declared) == 2:
            declared = {arm: [_unanchor(path) for path in paths] for arm, paths in declared.items()}
            everything = declared["GraphPC"] + declared["baseline"]
            if len(set(everything)) != len(everything):
                fail(f"the two arms' scored path lists overlap ({everything!r}): the cross-arm check "
                     f"would refuse an arm's own default and the scored launch would not run")
                ok = False
            missing = {arm: d for arm, d in result_dirs.items() if _unanchor(d) not in declared[arm]}
            if missing:
                fail(f"launcher default RESULT_DIR(s) {missing} are not in that arm's scored path list "
                     f"{ {a: declared[a] for a in missing} !r}: the guard would let another arm write "
                     f"into a scored tree it does not know about. Keep the two together.")
                ok = False
            # The GraphPC arm's store is a scored artefact too — whenever retrieval is on, an entry
            # written by a smoke becomes retrieval evidence — so its default must be listed.
            store = re.search(r'^SUPPORT_SET_ROOT="\$\{SUPPORT_SET_ROOT:-([^}"]+)\}"',
                              _read(GRAPHPC_LAUNCHER) or "", re.MULTILINE)
            if store is None:
                fail("could not read the default SUPPORT_SET_ROOT out of run_multienv_graphpc.sh, so "
                     "the claim that the scored support set is guarded was NOT CHECKED")
                ok = False
            elif _unanchor(store.group(1)) not in declared["GraphPC"]:
                fail(f"the GraphPC arm's default support set {store.group(1)} is not in "
                     f"OSWORLD_SCORED_PATHS_GRAPHPC {declared['GraphPC']!r}: a declared-unscored smoke "
                     f"could write an entry into the scored store, which becomes retrieval evidence "
                     f"whenever retrieval is on.")
                ok = False

    if ok and len(result_dirs) == 2:
        notes.append(f"wipe containment: arms default to distinct result trees "
                     f"({result_dirs['GraphPC']} vs {result_dirs['baseline']}), both of them named in "
                     f"their arms' scored path lists (result trees and the GraphPC store); both "
                     f"launchers call {GUARD_NAME} with their write paths and `-- \"$@\"`; and the "
                     f"baseline launcher CALLS the pre-clean on the LIVE launch path (call site and "
                     f"branch, each with its negative control red as it must be)")
    return ok



# --------------------------------------------------------------------------
# 4: unused (the number stays, as the runner's delta numbers do).
# --------------------------------------------------------------------------
# The OSWorld-V2 checkout the Qwen and DeepSeek modes compare against: on the host this file is
# <checkout>/scripts/python/, in the repo the staging clone sits next to the repository.
CHECKOUT_CANDIDATES = (HERE.parents[1], HERE.parents[2] / "OSWorld-V2")


# --------------------------------------------------------------------------
# 5: delta 15 — the runner refuses any --max_steps but the adapter's step budget.
# --------------------------------------------------------------------------
# The adapter the budget comes from: src/ in the repo, the vendored package in the checkout.
ADAPTER_CANDIDATES = (
    HERE.parents[1] / "src" / "benchmarks" / "osworld" / "graphpc_agent.py",
    HERE.parents[1] / "mm_agents" / "graphpc" / "benchmarks" / "osworld" / "graphpc_agent.py",
)
STEP_BUDGET_IMPORT = "from mm_agents.graphpc.benchmarks.osworld.graphpc_agent import OSWORLD_MAX_STEPS_BY_BACKBONE"


def _adapter_step_budgets(source: str) -> dict[str, int] | None:
    """``OSWORLD_MAX_STEPS_BY_BACKBONE`` as the adapter source defines it, its values integer literals or
    ``OSWORLD_MAX_STEPS``; None when either is missing or an entry is not a literal budget. The MiniMax entry
    is returned as written: the caller holds it to OSWORLD_MAX_STEPS (``"default"`` in the result)."""
    names: dict[str, ast.expr] = {}
    for node in ast.parse(source).body:
        target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                  else node.target if isinstance(node, ast.AnnAssign) else None)
        if isinstance(target, ast.Name) and node.value is not None:
            names[target.id] = node.value
    default, table = names.get("OSWORLD_MAX_STEPS"), names.get("OSWORLD_MAX_STEPS_BY_BACKBONE")
    if not (isinstance(default, ast.Constant) and type(default.value) is int and isinstance(table, ast.Dict)):
        return None
    budgets = {"default": default.value}
    for key, value in zip(table.keys, table.values):
        if isinstance(value, ast.Name) and value.id == "OSWORLD_MAX_STEPS":
            value = default
        if not (isinstance(key, ast.Constant) and isinstance(value, ast.Constant) and type(value.value) is int):
            return None
        budgets[key.value] = value.value
    return budgets


def _step_budget_check_nodes(body: list[ast.stmt]) -> list[int]:
    return [i for i, node in enumerate(body) if isinstance(node, ast.If)
            and "args.max_steps" in ast.unparse(node.test) and "OSWORLD_MAX_STEPS" in ast.unparse(node.test)]


def _problems_step_budget(runner: str, budgets: dict[str, int]) -> list[str]:
    """Execute the runner's own module-level delta-15 statement for each backbone at its budget, one step either
    side, upstream's default 100 and every other backbone's budget: only the backbone's own budget may pass,
    every other value must exit 1. A pure function of the runner text, so the negative controls can run it on
    doctored copies."""
    body = ast.parse(runner).body
    if not any(isinstance(node, ast.ImportFrom) and ast.unparse(node) == STEP_BUDGET_IMPORT for node in body):
        return [f"the runner does not import the budget at module level ({STEP_BUDGET_IMPORT})"]
    checks = _step_budget_check_nodes(body)
    if len(checks) != 1:
        return [f"the runner has {len(checks)} module-level --max_steps / OSWORLD_MAX_STEPS_BY_BACKBONE check(s), "
                f"expected exactly 1"]
    logging_setup = [i for i, node in enumerate(body) if ast.unparse(node).startswith("logger = logging.getLogger()")]
    if not logging_setup or checks[0] > logging_setup[0]:
        return ["the --max_steps check does not run before logging is set up (so before any log file or VM)"]
    code = compile(ast.Module(body=[body[checks[0]]], type_ignores=[]), "<delta 15>", "exec")
    table = {backbone: budget for backbone, budget in budgets.items() if backbone != "default"}
    problems = []
    for backbone, budget in sorted(table.items()):
        for value in sorted({budget, budget - 1, budget + 1, 100, *table.values()}):
            try:
                exec(code, {"args": argparse.Namespace(max_steps=value, backbone=backbone),
                            "OSWORLD_MAX_STEPS_BY_BACKBONE": dict(table), "sys": sys,
                            "print": lambda *args, **kwargs: None})
                exit_code = None
            except SystemExit as exc:
                exit_code = exc.code
            expected = None if value == budget else 1
            if exit_code != expected:
                problems.append(f"--backbone {backbone} --max_steps {value} against its budget {budget} gave exit "
                                f"{exit_code}, expected {expected}")
    return problems


RETRIEVAL_OFF_TEST = "not args.retrieval_off"


def _retrieval_off_refusal_nodes(body: list[ast.stmt]) -> list[int]:
    """Module-level ``if not args.retrieval_off:`` statements that only refuse (no subprocess call: a
    subprocess-calling gate would share the test)."""
    return [i for i, node in enumerate(body) if isinstance(node, ast.If) and ast.unparse(node.test) == RETRIEVAL_OFF_TEST
            and "sys.exit(1)" in ast.unparse(node) and "subprocess" not in ast.unparse(node)]


def _problems_retrieval_off_refusal(runner: str) -> list[str]:
    """Execute the runner's own module-level refusal of a launch without --retrieval_off: it must exit 1
    without the flag and do nothing with it, and sit right after the --max_steps check (so before any
    gate, log file or VM). A pure function of the runner text, for the negative controls."""
    body = ast.parse(runner).body
    refusals = _retrieval_off_refusal_nodes(body)
    if len(refusals) != 1:
        return [f"the runner has {len(refusals)} module-level refusal(s) of a launch without --retrieval_off, "
                f"expected exactly 1"]
    checks = _step_budget_check_nodes(body)
    if len(checks) != 1 or refusals[0] != checks[0] + 1:
        return ["the refusal of a launch without --retrieval_off does not follow the --max_steps check directly"]
    code = compile(ast.Module(body=[body[refusals[0]]], type_ignores=[]), "<delta 15 retrieval>", "exec")
    problems = []
    for retrieval_off, expected in ((False, 1), (True, None)):
        try:
            exec(code, {"args": argparse.Namespace(retrieval_off=retrieval_off), "sys": sys,
                        "print": lambda *args, **kwargs: None})
            exit_code = None
        except SystemExit as exc:
            exit_code = exc.code
        if exit_code != expected:
            problems.append(f"retrieval_off={retrieval_off} gave exit {exit_code}, expected {expected}")
    return problems


def check_the_step_budget_is_enforced() -> None:
    """Delta 15, executed: the runner's own statement against the budget the adapter file defines.
    Three negative controls (the check removed, its comparison weakened to `>`, the import removed)
    must each fail."""
    adapter = next((path for path in ADAPTER_CANDIDATES if path.is_file()), None)
    if adapter is None:
        fail("delta 15 NOT CHECKED: no adapter at any of " + ", ".join(str(path) for path in ADAPTER_CANDIDATES))
        return
    budgets = _adapter_step_budgets(adapter.read_text(encoding="utf-8"))
    if budgets is None:
        fail(f"delta 15 NOT CHECKED: {adapter} defines no integer OSWORLD_MAX_STEPS and literal "
             f"OSWORLD_MAX_STEPS_BY_BACKBONE")
        return
    if budgets.get("minimax") != budgets["default"] or budgets["default"] != 500:
        fail(f"delta 15: the MiniMax GraphPC arm's budget is {budgets.get('minimax')!r} and OSWORLD_MAX_STEPS is "
             f"{budgets['default']!r}; both are 500 (the Qwen GraphPC arm runs an 800-step budget with the "
             f"stuck-stop; every other arm, and every control, runs 500)")
    runner = _read(GRAPHPC_RUNNER)
    if not runner:
        return
    problems = _problems_step_budget(runner, budgets)
    for problem in problems:
        fail(f"delta 15: {problem}")

    body = ast.parse(runner).body
    checks = _step_budget_check_nodes(body)
    statement = ast.get_source_segment(runner, body[checks[0]]) if len(checks) == 1 else None
    controls = {}
    if statement is not None and runner.count(statement) == 1:
        controls["check_removed"] = runner.replace(statement, "")
        lookup = "args.max_steps != OSWORLD_MAX_STEPS_BY_BACKBONE[args.backbone]"
        weakened = statement.replace(lookup, "args.max_steps > OSWORLD_MAX_STEPS_BY_BACKBONE[args.backbone]", 1)
        if weakened != statement:
            controls["check_weakened"] = runner.replace(statement, weakened)
        one_budget = statement.replace(lookup, "args.max_steps != OSWORLD_MAX_STEPS_BY_BACKBONE['minimax']", 1)
        if one_budget != statement:
            controls["backbone_ignored"] = runner.replace(statement, one_budget)
    if runner.count(STEP_BUDGET_IMPORT) == 1:
        controls["import_removed"] = runner.replace(
            STEP_BUDGET_IMPORT, "OSWORLD_MAX_STEPS_BY_BACKBONE = {'minimax': 100, 'qwen': 100}", 1)
    controls_ok = True
    for name in ("check_removed", "check_weakened", "backbone_ignored", "import_removed"):
        if name not in controls:
            fail(f"delta 15 negative control {name!r} could not be built from the runner, so the property it "
                 f"guards is NOT CHECKED")
            controls_ok = False
        elif not _problems_step_budget(controls[name], budgets):
            fail(f"delta 15 negative control {name!r} passed the check it must fail: the check cannot see that defect")
            controls_ok = False
    refusal_problems = _problems_retrieval_off_refusal(runner)
    for problem in refusal_problems:
        fail(f"delta 15: {problem}")
    refusals = _retrieval_off_refusal_nodes(body)
    refusal = ast.get_source_segment(runner, body[refusals[0]]) if len(refusals) == 1 else None
    refusal_controls = {}
    if refusal is not None and runner.count(refusal) == 1:
        refusal_controls["refusal_removed"] = runner.replace(refusal, "")
        refusal_controls["refusal_inverted"] = runner.replace(
            refusal, refusal.replace(f"if {RETRIEVAL_OFF_TEST}:", "if args.retrieval_off:", 1))
    for name in ("refusal_removed", "refusal_inverted"):
        if name not in refusal_controls:
            fail(f"delta 15 negative control {name!r} could not be built from the runner, so the property it "
                 f"guards is NOT CHECKED")
            controls_ok = False
        elif not _problems_retrieval_off_refusal(refusal_controls[name]):
            fail(f"delta 15 negative control {name!r} passed the check it must fail: the check cannot see that defect")
            controls_ok = False
    if not problems and not refusal_problems and controls_ok:
        table = {backbone: budget for backbone, budget in budgets.items() if backbone != "default"}
        notes.append(f"delta 15: the runner's own --max_steps check, against OSWORLD_MAX_STEPS_BY_BACKBONE = {table} "
                     f"from {adapter.relative_to(HERE.parents[1])}, passes each backbone's budget only and exits 1 one "
                     f"step either side, at 100 and at the other backbone's budget, before logging is set up; its "
                     f"refusal of a launch without --retrieval_off exits 1 without the flag and nothing with it, right "
                     f"after; six negative controls each red")


# ==========================================================================
# QWEN MODE (--cell qwen). The Qwen cell's control is upstream's osworld-v2.1 Qwen agent and
# runner, vendored with declared deltas; its GraphPC arm runs run_multienv_graphpc.py. The
# M3-pair checks above are the default mode and do not run here.
#   Q1. Every vendored control file IS its osworld-v2.1 blob minus exactly the declared hunks:
#       the header stripped and every marked block replaced by the upstream lines it records,
#       the file is byte-identical to `git show osworld-v2.1:<path>` in a clone that has the
#       tag; every block's delta is declared in the header and every declared delta has a block.
#       Across the arms: a blob both arms copy is the same text in both once each copy is reversed
#       of its own deltas, and a delta both arms must carry (QWEN_DELTAS_IN_BOTH_ARMS) is carried
#       by both, block for block.
#   Q2. Parity: resolved from the two Qwen-cell launchers under DRY_RUN and parsed through each
#       runner's own argparse, the two runners feed the checkout's unmodified run_single_example
#       and DesktopEnv the same values — every args field the checkout's lib_run_single.py and
#       lib_results_logger.py read (derived from their AST), --model, max_steps, the manifest and
#       its dispatch order, task loading, the raw-eval dump, the evaluator / user-simulator
#       environment and every DesktopEnv kwarg (evaluated on fixture tasks).
# Each carries negative controls that run on every invocation.
# ==========================================================================
QWEN_UPSTREAM_TAG = "osworld-v2.1"
QWEN_UPSTREAM_RUNNER = "scripts/python/run_multienv_qwen.py"
QWEN_CONTROL_RUNNER = HERE / "run_multienv_qwen_baseline.py"
QWEN_CONTROL_LAUNCHER = _launcher("run_multienv_qwen_baseline.sh")
# The vendored agent: scripts/osworld/qwen_control/mm_agents/qwen/ in the repo, mm_agents/qwen/ in the checkout.
QWEN_AGENT_DIR_CANDIDATES = (HERE / "qwen_control" / "mm_agents" / "qwen", HERE.parents[1] / "mm_agents" / "qwen")
QWEN_HEADER_END = "end of GraphPC header"
QWEN_BLOCK_OPEN = "# >>> GraphPC "
QWEN_BLOCK_CLOSE = "# <<< GraphPC "
QWEN_UPSTREAM_LINE = "#|"
# A delta label is a capital letter and a number: the control's are D1-D9; the GraphPC arm's upstream
# copies label their own (G1). Declared in the header's "Deltas carried:" line, marked on each block.
QWEN_DELTA = re.compile(r"\b([A-Z]\d+)\b")
# The GraphPC arm's own copies of upstream Qwen files, each headed like the control's: the repo
# layout, and the synced one.
QWEN_GRAPHPC_COPY_DIRS = (HERE.parents[1] / "src" / "benchmarks" / "osworld" / "qwen_upstream",
                           HERE.parents[1] / "mm_agents" / "graphpc" / "benchmarks" / "osworld" / "qwen_upstream")
# The control's deltas the GraphPC arm's copy of the same blob carries too, under the control's label
# and block for block, because the GraphPC arm runs the control's own code for that piece: D9, the
# frame both arms send (JPEG). This gate's own literal, never read off the files.
QWEN_DELTAS_IN_BOTH_ARMS = {"mm_agents/qwen/images.py": ("D9",)}
QWEN_HEADER_PATH = re.compile(r"OSWorld-V2 (\S+\.py)\b")
# The manifest a scored launch reads: evaluation_examples/ in the checkout, the tracked copy in the repo.
QWEN_MANIFEST_CANDIDATES = (HERE.parents[1] / "evaluation_examples" / "prereg_task_list.json",
                            HERE.parents[1] / "data" / "osworld_v2" / "prereg_task_list.json")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)


def _qwen_upstream_repo(checkout_root: Path | None, tag: str = QWEN_UPSTREAM_TAG) -> Path | None:
    candidates = (checkout_root,) if checkout_root is not None else CHECKOUT_CANDIDATES
    for candidate in candidates:
        if candidate.is_dir() and _git(candidate, "rev-parse", "--verify", "--quiet",
                                       f"{tag}^{{commit}}").returncode == 0:
            return candidate
    return None


def _qwen_reverse(text: str, keep: frozenset[str] = frozenset()) -> tuple[str, str, list[str]]:
    """(upstream text, header, block labels). A block whose deltas are all in ``keep`` stays as it
    is, markers included. Raises ValueError on a malformed file."""
    lines = text.splitlines(keepends=True)
    end = next((i for i, line in enumerate(lines) if QWEN_HEADER_END in line), None)
    if end is None:
        raise ValueError(f"no header ending in {QWEN_HEADER_END!r}")
    out: list[str] = []
    labels: list[str] = []
    inside = kept = False
    for number, line in enumerate(lines[end + 1:], start=end + 2):
        stripped = line.lstrip()
        if stripped.startswith(QWEN_BLOCK_OPEN):
            if inside:
                raise ValueError(f"line {number}: a block opens inside a block")
            inside = True
            labels.append(stripped[len(QWEN_BLOCK_OPEN):].strip())
            deltas = set(QWEN_DELTA.findall(labels[-1].split(":")[0]))
            kept = bool(deltas) and deltas <= keep
            if kept:
                out.append(line)
        elif stripped.startswith(QWEN_BLOCK_CLOSE):
            if not inside:
                raise ValueError(f"line {number}: a block closes outside a block")
            inside = False
            if kept:
                out.append(line)
        elif inside:
            if kept:
                out.append(line)
            elif line.startswith(QWEN_UPSTREAM_LINE):
                out.append(line[len(QWEN_UPSTREAM_LINE):])
        elif line.startswith(QWEN_UPSTREAM_LINE):
            raise ValueError(f"line {number}: an upstream line outside any block")
        else:
            out.append(line)
    if inside:
        raise ValueError("a block never closes")
    return "".join(out), "".join(lines[:end + 1]), labels


def _qwen_file_problems(text: str, upstream: str, upstream_path: str, blob: str, tag: str = QWEN_UPSTREAM_TAG) -> list[str]:
    """Everything the vendored file differs from its blob by beyond the declared hunks. A pure
    function of the texts, so the negative controls can run it on doctored copies. ``tag`` only
    labels the diff (the DeepSeek mode reads a different tag)."""
    try:
        restored, header, labels = _qwen_reverse(text)
    except ValueError as exc:
        return [f"cannot be reversed: {exc}"]
    problems = []
    if f"OSWorld-V2 {upstream_path}" not in header or f"blob {blob}" not in header:
        problems.append(f"the header does not name {upstream_path} and blob {blob}")
    carried = re.search(r"Deltas carried:(.*?)(?:\n# end|\nend| -->|$)", header, re.S)
    declared = set(QWEN_DELTA.findall(carried.group(1))) if carried else None
    marked = {number for label in labels for number in QWEN_DELTA.findall(label.split(":")[0])}
    if declared is None:
        problems.append("the header has no 'Deltas carried:' line")
    elif declared != marked:
        problems.append(f"the header declares deltas {sorted(declared)} but the file marks {sorted(marked)}")
    if restored != upstream:
        diff = list(difflib.unified_diff(upstream.splitlines(), restored.splitlines(), f"{tag}:{upstream_path}",
                                         "vendored copy with its header and marked blocks reversed", lineterm="", n=0))
        problems.append("differs from upstream BEYOND its declared hunks:\n  " + "\n  ".join(diff[:12])
                        + ("\n  ..." if len(diff) > 12 else ""))
    return problems


def _qwen_cross_arm_problems(control: str, graphpc: str, shared: tuple[str, ...]) -> list[str]:
    """What one blob's two copies (the control's, the GraphPC arm's) differ by beyond each arm's own
    deltas. Each is reversed of every delta but ``shared``, whose blocks stay as they are, markers
    included; the two must then be the same text, so a shared delta is the same code in both arms. A
    pure function of the texts, so the negative controls can run it on doctored copies."""
    try:
        control_kept, _, control_labels = _qwen_reverse(control, keep=frozenset(shared))
        graphpc_kept, _, graphpc_labels = _qwen_reverse(graphpc, keep=frozenset(shared))
    except ValueError as exc:
        return [f"cannot be reversed: {exc}"]
    problems = []
    for arm, labels in (("control", control_labels), ("GraphPC", graphpc_labels)):
        marked = {number for label in labels for number in QWEN_DELTA.findall(label.split(":")[0])}
        if set(shared) - marked:
            problems.append(f"the {arm} copy does not carry {sorted(set(shared) - marked)}, which both arms' "
                            f"copies must carry")
    if control_kept != graphpc_kept:
        diff = list(difflib.unified_diff(control_kept.splitlines(), graphpc_kept.splitlines(), "the control's copy",
                                         "the GraphPC arm's copy", lineterm="", n=0))
        problems.append(f"the two arms' copies differ beyond their own deltas (kept as they are: {list(shared)}):\n  "
                        + "\n  ".join(diff[:12]) + ("\n  ..." if len(diff) > 12 else ""))
    return problems


def check_qwen_control_is_upstream_plus_declared_deltas(checkout_root: Path | None = None) -> None:
    failures_before = len(failures)
    repo = _qwen_upstream_repo(checkout_root)
    if repo is None:
        fail(f"Q1 NOT CHECKED: no clone with tag {QWEN_UPSTREAM_TAG} at "
             f"{[str(c) for c in ((checkout_root,) if checkout_root else CHECKOUT_CANDIDATES)]}")
        return
    agent_dir = next((path for path in QWEN_AGENT_DIR_CANDIDATES if path.is_dir()), None)
    if agent_dir is None:
        fail(f"Q1 NOT CHECKED: no vendored Qwen agent at {[str(p) for p in QWEN_AGENT_DIR_CANDIDATES]}")
        return
    listed = _git(repo, "ls-tree", "-r", "--name-only", QWEN_UPSTREAM_TAG, "mm_agents/qwen")
    upstream_agent = sorted(listed.stdout.decode().split())
    vendored_names = sorted(path.name for path in agent_dir.iterdir() if path.is_file())
    if [Path(p).name for p in upstream_agent] != vendored_names:
        fail(f"Q1: the vendored agent files {vendored_names} are not upstream's {upstream_agent}")
    pairs = [(agent_dir / Path(p).name, p) for p in upstream_agent] + [(QWEN_CONTROL_RUNNER, QWEN_UPSTREAM_RUNNER)]
    control_copies = {upstream_path: local for local, upstream_path in pairs}
    # The GraphPC arm's copies: each names its upstream path in its header.
    copy_dir = next((d for d in QWEN_GRAPHPC_COPY_DIRS if d.is_dir()), None)
    graphpc_copies = []
    for copy in sorted(copy_dir.glob("*.py")) if copy_dir else []:
        named = QWEN_HEADER_PATH.search(_read(copy).split(QWEN_HEADER_END)[0])
        if copy.name == "__init__.py" and not named:
            continue   # the package's own __init__, not a copy
        if not named:
            fail(f"Q1 {copy}: no GraphPC header naming the upstream file it copies")
            continue
        graphpc_copies.append((copy, named.group(1)))
    pairs += graphpc_copies
    blocks = 0
    first_with_blocks = None
    for local, upstream_path in pairs:
        text = _read(local)
        if not text:
            continue
        upstream = _git(repo, "show", f"{QWEN_UPSTREAM_TAG}:{upstream_path}").stdout.decode("utf-8")
        blob = _git(repo, "rev-parse", f"{QWEN_UPSTREAM_TAG}:{upstream_path}").stdout.decode().strip()
        for problem in _qwen_file_problems(text, upstream, upstream_path, blob):
            fail(f"Q1 {local.name}: {problem}")
        try:
            count = len(_qwen_reverse(text)[2])
        except ValueError:
            count = 0
        blocks += count
        if count and first_with_blocks is None:
            first_with_blocks = (text, upstream, upstream_path, blob)
    if first_with_blocks is None:
        fail("Q1 negative controls NOT RUN: no vendored file carries a marked block")
        return
    text, upstream, upstream_path, blob = first_with_blocks
    mark_line = next(line for line in text.splitlines(keepends=True) if line.startswith(QWEN_UPSTREAM_LINE))
    close_line = next(line for line in text.splitlines(keepends=True) if line.lstrip().startswith(QWEN_BLOCK_CLOSE))
    open_line = next(line for line in text.splitlines(keepends=True) if line.lstrip().startswith(QWEN_BLOCK_OPEN))
    controls = {
        "one extra line outside the blocks": text + "# one extra line\n",
        "a recorded upstream line altered": text.replace(mark_line, mark_line.replace(QWEN_UPSTREAM_LINE, QWEN_UPSTREAM_LINE + " ", 1), 1),
        "a block's closing marker removed": text.replace(close_line, "", 1),
        "a block labelled with an undeclared delta": text.replace(open_line, open_line.replace(QWEN_BLOCK_OPEN, QWEN_BLOCK_OPEN + "D99 ", 1), 1),
    }
    for name, doctored in controls.items():
        if doctored == text or not _qwen_file_problems(doctored, upstream, upstream_path, blob):
            fail(f"Q1 negative control {name!r} passed the check it must fail")
            return
    # Two copies of the same blob, one per arm, must agree beyond each arm's own deltas, and a delta
    # both arms must carry is the same block in both.
    agreeing = 0
    both_arms = {}
    for local, upstream_path in graphpc_copies:
        control_copy = control_copies.get(upstream_path)
        if control_copy is None:
            continue
        shared = QWEN_DELTAS_IN_BOTH_ARMS.get(upstream_path, ())
        problems = _qwen_cross_arm_problems(_read(control_copy), _read(local), shared)
        for problem in problems:
            fail(f"Q1 {upstream_path} across the arms ({control_copy} | {local}): {problem}")
        agreeing += not problems
        if shared:
            both_arms[upstream_path] = (_read(control_copy), _read(local), shared)
    unchecked = sorted(set(QWEN_DELTAS_IN_BOTH_ARMS) - set(both_arms))
    if unchecked:
        fail(f"Q1 NOT CHECKED across the arms: no control copy and GraphPC copy of {unchecked}")
    if len(failures) > failures_before:
        return   # the FAILs above are Q1's verdict; the cross-arm controls need a green pair to doctor
    cross_controls = 0
    for upstream_path, (control_text, graphpc_text, shared) in both_arms.items():
        graphpc_lines = graphpc_text.splitlines(keepends=True)
        opening = next(i for i, line in enumerate(graphpc_lines) if line.lstrip().startswith(QWEN_BLOCK_OPEN)
                       and set(QWEN_DELTA.findall(line.lstrip()[len(QWEN_BLOCK_OPEN):].split(":")[0])) <= set(shared))
        own = frozenset(number for label in _qwen_reverse(graphpc_text)[2]
                        for number in QWEN_DELTA.findall(label.split(":")[0]) if number not in shared)
        without_shared, header, _ = _qwen_reverse(graphpc_text, keep=own)
        cross = {
            "a line added inside the GraphPC copy's shared block":
                "".join(graphpc_lines[:opening + 1] + ["# one extra line\n"] + graphpc_lines[opening + 1:]),
            "the GraphPC copy without the shared delta (its blocks reversed)": header + without_shared,
        }
        for name, doctored in cross.items():
            cross_controls += 1
            if doctored == graphpc_text or not _qwen_cross_arm_problems(control_text, doctored, shared):
                fail(f"Q1 negative control {name!r} ({upstream_path}) passed the check it must fail")
                return
    if len(failures) > failures_before:
        return   # the FAILs above are Q1's verdict; no OK line beside them
    shared_text = "; ".join(f"{path}: {', '.join(shared)}" for path, shared in sorted(QWEN_DELTAS_IN_BOTH_ARMS.items()))
    notes.append(f"Q1: {len(pairs) - len(graphpc_copies)} vendored control files and {len(graphpc_copies)} GraphPC-arm "
                 f"copies ({copy_dir or 'none present'}) == their {QWEN_UPSTREAM_TAG} blobs ({repo}) minus exactly "
                 f"their {blocks} marked blocks, header excluded; every block's delta declared and every declared "
                 f"delta marked; {agreeing} blob(s) copied in both arms agree beyond each arm's own deltas, the "
                 f"deltas both arms carry block for block ({shared_text}) included; {len(controls)} + "
                 f"{cross_controls} negative controls each red")


def _qwen_manifest() -> Path | None:
    return next((path for path in QWEN_MANIFEST_CANDIDATES if path.is_file()), None)


def _dry_run_tokens(script: Path, extra_env: dict[str, str], cwd: Path | None = None) -> list[str]:
    env = {"PATH": __import__("os").environ.get("PATH", ""), "HOME": __import__("os").environ.get("HOME", ""),
           "DRY_RUN": "1", **extra_env}
    proc = subprocess.run(["bash", str(script)], cwd=str(cwd or HERE.parents[1]), env=env, capture_output=True,
                          text=True, timeout=60)
    marker = "DRY_RUN argv (one element per line):"
    lines = proc.stdout.splitlines()
    if proc.returncode != 0 or marker not in lines:
        raise RuntimeError(f"{script.name} under DRY_RUN exited {proc.returncode}: "
                           f"{(proc.stderr.strip() or proc.stdout.strip())[-400:]}")
    argv = lines[lines.index(marker) + 1:]
    script_index = next(i for i, token in enumerate(argv) if token.endswith(".py"))
    return argv[script_index + 1:]


def _module_functions(text: str, names: tuple[str, ...]) -> dict:
    namespace: dict = {"argparse": argparse, "os": __import__("os"), "re": re, "json": json}
    body = [node for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef) and node.name in names
            or isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets)]
    exec(compile(ast.Module(body=body, type_ignores=[]), "<runner>", "exec"), namespace)
    missing = [name for name in names if name not in namespace]
    if missing:
        raise RuntimeError(f"the runner defines no {missing}")
    return namespace


def _parse_runner_args(text: str, tokens: list[str], *, argv_param: bool) -> argparse.Namespace:
    """The runner's own argparse on the launcher's argv. The GraphPC runner's config() reads
    sys.argv; the control's config(argv) takes it."""
    names = ("config", "_volume_size") if argv_param else ("config",)
    config_fn = _module_functions(text, names)["config"]
    if argv_param:
        return config_fn(tokens)
    saved = sys.argv
    sys.argv = ["runner"] + tokens
    try:
        return config_fn()
    finally:
        sys.argv = saved


def _driver_fields(texts: list[str]) -> dict[str, tuple[str, object]]:
    """{field: ("bare", None) | ("getattr", default)} for every args.X / getattr(args, "X"[, default])
    in the checkout's driver and results logger. A field read bare anywhere must exist on args."""
    bare: set[str] = set()
    via_getattr: dict[str, object] = {}
    for text in texts:
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
                bare.add(node.attr)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr")
                  and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "args"
                  and isinstance(node.args[1], ast.Constant)):
                via_getattr.setdefault(node.args[1].value, ast.literal_eval(node.args[2]) if len(node.args) > 2 else None)
    return {name: ("bare", None) if name in bare else ("getattr", via_getattr[name])
            for name in bare | set(via_getattr)}


def _single_call(tree: ast.AST, dotted: str) -> ast.Call | None:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted]
    return calls[0] if len(calls) == 1 else None


def _desktop_env_defaults(desktop_env_text: str) -> dict[str, object]:
    for node in ast.walk(ast.parse(desktop_env_text)):
        if isinstance(node, ast.ClassDef) and node.name == "DesktopEnv":
            init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            params = init.args.args[1:]
            defaults = dict(zip([p.arg for p in params[len(params) - len(init.args.defaults):]], init.args.defaults))
            out = {}
            for name, expr in defaults.items():
                try:
                    out[name] = ast.literal_eval(expr)
                except ValueError:
                    out[name] = ("<unevaluated default>", ast.unparse(expr))
            return out
    raise RuntimeError("the checkout's desktop_env.py defines no DesktopEnv.__init__")


QWEN_FIXTURE_TASKS = (
    {},                                                                  # the flag's volume, the default AMI
    {"image": "ami-fixture", "volume_size": 80, "instance_type": "t3.2xlarge"},  # the task's own spec
    {"platform": "windows"},                                             # os_type from the task
)


def _desktop_env_kwargs(text: str, args: argparse.Namespace, example: dict, *, graphpc: bool,
                        defaults: dict[str, object]) -> dict[str, object]:
    """Every DesktopEnv kwarg the runner passes for ``example``, evaluated with the runner's own
    spec code (the AWS default-AMI lookup stubbed), and DesktopEnv's own default for the rest."""
    import types  # noqa: PLC0415
    manager = types.ModuleType("desktop_env.providers.aws.manager")
    manager.get_default_ami_id = lambda region, screen_size, os_type: f"default-ami:{region}:{tuple(screen_size)}:{os_type}"
    stub_names = ("desktop_env", "desktop_env.providers", "desktop_env.providers.aws", "desktop_env.providers.aws.manager")
    saved = {name: sys.modules.get(name) for name in stub_names}
    for name in stub_names[:-1]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["desktop_env.providers.aws.manager"] = manager
    try:
        tree = ast.parse(text)
        call = _single_call(tree, "DesktopEnv")
        if call is None:
            raise RuntimeError("the runner does not construct DesktopEnv exactly once")
        namespace: dict = {"args": args, "getattr": getattr}
        if graphpc:
            tasks = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_env_tasks")
            nested = [n for n in ast.walk(tasks) if isinstance(n, ast.FunctionDef)
                      and n.name in ("_platform_to_os_type", "_get_task_os_type", "_build_env_spec")]
            size = [n for n in ast.walk(tasks) if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["screen_size"]]
            exec(compile(ast.Module(body=size[:1] + nested, type_ignores=[]), "<graphpc spec>", "exec"), namespace)
            os_type, _ = namespace["_get_task_os_type"](example)
            namespace["env_spec"] = namespace["_build_env_spec"](example, os_type)
        else:
            namespace.update(_module_functions(text, ("_example_value", "_task_os_type", "_env_spec")))
            namespace["spec"] = namespace["_env_spec"](example, args)
        kwargs = {kw.arg: eval(compile(ast.Expression(kw.value), "<kwarg>", "eval"), namespace)  # noqa: S307
                  for kw in call.keywords}
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return {name: kwargs.get(name, defaults.get(name)) for name in sorted(set(defaults) | set(kwargs))}


QWEN_EVAL_ENV = re.compile(r"OSWORLD_(EVAL|USER_SIM)_[A-Z_]+")


def _env_writes(text: str) -> list[str]:
    """Evaluator / user-simulator environment keys the runner itself writes (the raw-eval dump dir
    aside, which is compared separately)."""
    written = []
    for node in ast.walk(ast.parse(text)):
        targets = node.targets if isinstance(node, ast.Assign) else []
        for target in targets:
            if isinstance(target, ast.Subscript) and ast.unparse(target.value) == "os.environ":
                written.append(ast.unparse(target.slice).strip("'\""))
        if isinstance(node, ast.Call) and ast.unparse(node.func) in ("os.environ.setdefault", "os.environ.update", "os.putenv"):
            written.append(ast.unparse(node))
    return [key for key in written if QWEN_EVAL_ENV.search(key) and "OSWORLD_EVAL_SAVE_RAW_DIR" not in key]


def _raw_eval_dump(text: str) -> list[str]:
    tree = ast.parse(text)
    return [ast.unparse(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Subscript) and ast.unparse(t.value) == "os.environ"
                    and ast.unparse(t.slice).strip("'\"") == "OSWORLD_EVAL_SAVE_RAW_DIR" for t in node.targets)]


def _dispatch_order(text: str, args: argparse.Namespace, manifest: dict, *, graphpc: bool) -> list[tuple[str, str]]:
    if graphpc:
        tasks = _module_functions(text, ("distribute_tasks",))["distribute_tasks"](manifest)
        return [(domain, str(task_id)) for domain, task_id in tasks]
    namespace = _module_functions(text, ("_select_tasks", "_filter_unfinished"))
    namespace["_filter_unfinished"] = lambda selected, _args: selected   # upstream's deletion step, not run
    selected = namespace["_select_tasks"](manifest, args)
    return [(domain, str(task_id)) for domain, task_ids in selected.items() for task_id in task_ids]


def _qwen_parity_problems(graphpc_tokens: list[str], control_tokens: list[str], graphpc_text: str, control_text: str,
                          driver_texts: list[str], desktop_env_text: str, manifest: dict,
                          step_budgets: dict[str, int]) -> tuple[list[str], list[str]]:
    """(problems, facts). A pure function of its inputs, so the negative controls run it on doctored ones."""
    problems: list[str] = []
    facts: list[str] = []
    try:
        g = _parse_runner_args(graphpc_text, graphpc_tokens, argv_param=False)
        c = _parse_runner_args(control_text, control_tokens, argv_param=True)
    except SystemExit as exc:
        return [f"a runner's argparse refused its launcher's argv (exit {exc.code})"], facts
    fields = _driver_fields(driver_texts)
    compared = []
    for field, (how, default) in sorted(fields.items()):
        values = {arm: getattr(ns, field) if hasattr(ns, field) else
                  ("<absent, but the driver reads it bare>" if how == "bare" else default)
                  for arm, ns in (("graphpc", g), ("control", c))}
        if field == "result_dir":
            if values["graphpc"] == values["control"]:
                problems.append(f"both arms write result_dir={values['graphpc']!r}: the arms must keep distinct trees")
            continue
        compared.append(field)
        if values["graphpc"] != values["control"]:
            problems.append(f"driver field {field!r} differs: graphpc={values['graphpc']!r} control={values['control']!r}")
    facts.append(f"driver fields (lib_run_single + lib_results_logger AST): {sorted(fields)}; equal: {compared}; "
                 f"result_dir distinct per arm by design")
    for field in ("model", "eval_version", "test_config_base_dir", "save_model_eval_raw_info"):
        if getattr(g, field, None) != getattr(c, field, None):
            problems.append(f"{field} differs: graphpc={getattr(g, field, None)!r} control={getattr(c, field, None)!r}")
    for arm, ns in (("graphpc", g), ("control", c)):
        if not str(getattr(ns, "test_all_meta_path", "")).endswith("prereg_task_list.json"):
            problems.append(f"{arm} reads the manifest {getattr(ns, 'test_all_meta_path', None)!r}, not the 28")
    # The step budget per arm: the GraphPC arm the adapter's Qwen budget (800), the control the cell's 500
    # (OSWORLD_MAX_STEPS).
    for arm, ns, budget in (("graphpc", g, step_budgets.get("qwen")), ("control", c, step_budgets.get("default"))):
        if getattr(ns, "max_steps", None) != budget:
            problems.append(f"{arm} runs --max_steps {getattr(ns, 'max_steps', None)!r}, not its budget {budget!r}")
    facts.append(f"--model {c.model!r}, eval_version {c.eval_version}, test_config_base_dir "
                 f"{c.test_config_base_dir}, save_model_eval_raw_info {c.save_model_eval_raw_info}: equal; max_steps "
                 f"per arm: graphpc {g.max_steps} (the adapter's Qwen budget), control {c.max_steps}; "
                 f"manifests: graphpc {g.test_all_meta_path} (the 28), control {c.test_all_meta_path} (the 28)")
    for arm, text in (("graphpc", graphpc_text), ("control", control_text)):
        tree = ast.parse(text)
        call = _single_call(tree, "lib_run_single.run_single_example")
        if call is None or len(call.args) < 6 or ast.unparse(call.args[3]) != "args.max_steps" or ast.unparse(call.args[5]) != "args":
            problems.append(f"{arm}: not exactly one run_single_example(agent, env, example, args.max_steps, instruction, "
                            f"args, ...) call")
        dumps = _raw_eval_dump(text)
        if len(dumps) != 1 or not dumps[0].endswith("'model_eval_file_log')"):
            problems.append(f"{arm}: OSWORLD_EVAL_SAVE_RAW_DIR is not set exactly once to <task dir>/model_eval_file_log: {dumps}")
        writes = _env_writes(text)
        if writes:
            problems.append(f"{arm}: the runner writes evaluator / user-simulator environment itself: {writes}")
        if not any(isinstance(n, ast.Expr) and ast.unparse(n) == "load_dotenv()" for n in tree.body):
            problems.append(f"{arm}: no module-level load_dotenv(), so its evaluator / user-simulator env is not the .env's")
    facts.append("both call the checkout's run_single_example(..., args.max_steps, ..., args, ...) once, dump raw evals "
                 "to <task>/model_eval_file_log, load the same .env and write no OSWORLD_EVAL_* / OSWORLD_USER_SIM_* key")
    try:
        # The two runners' dispatch code on the SAME manifest (the 28) must agree, in manifest order.
        orders = {"graphpc": _dispatch_order(graphpc_text, g, manifest, graphpc=True),
                  "control": _dispatch_order(control_text, c, manifest, graphpc=False)}
        listed = [(domain, str(t)) for domain, ids in manifest.items() for t in ids]
        if orders["graphpc"] != orders["control"]:
            problems.append(f"dispatch order differs: graphpc {orders['graphpc'][:3]}... control {orders['control'][:3]}...")
        elif orders["control"] != listed:
            problems.append(f"the dispatch order is not the 28's manifest order: {orders['control'][:3]}...")
        else:
            facts.append(f"dispatch order equal on the 28 ({len(listed)} tasks, manifest order)")
        defaults = _desktop_env_defaults(desktop_env_text)
        for index, example in enumerate(QWEN_FIXTURE_TASKS):
            kw = {"graphpc": _desktop_env_kwargs(graphpc_text, g, dict(example), graphpc=True, defaults=defaults),
                  "control": _desktop_env_kwargs(control_text, c, dict(example), graphpc=False, defaults=defaults)}
            differ = {k: (kw["graphpc"].get(k), kw["control"].get(k)) for k in kw["graphpc"] if kw["graphpc"].get(k) != kw["control"].get(k)}
            if differ:
                problems.append(f"DesktopEnv kwargs differ on fixture task {index} {example}: {differ}")
        facts.append(f"DesktopEnv kwargs equal on {len(QWEN_FIXTURE_TASKS)} fixture tasks: {sorted(kw['control'])}")
    except Exception as exc:  # noqa: BLE001 — a check that could not run has not passed
        problems.append(f"could not evaluate dispatch order / DesktopEnv kwargs: {type(exc).__name__}: {exc}")
    return problems, facts


def check_qwen_runner_parity(checkout_root: Path | None = None) -> None:
    root = next((c for c in ((checkout_root,) if checkout_root else CHECKOUT_CANDIDATES)
                 if (c / "lib_run_single.py").is_file()), None)
    manifest_path = _qwen_manifest()
    if root is None or manifest_path is None:
        fail(f"Q2 NOT CHECKED: checkout {root} / manifest {manifest_path} not found")
        return
    try:
        graphpc_tokens = _dry_run_tokens(GRAPHPC_LAUNCHER, {"OSWORLD_CELL": "qwen", "TEST_ALL_META_PATH": str(manifest_path)})
        control_tokens = _dry_run_tokens(QWEN_CONTROL_LAUNCHER, {"TEST_ALL_META_PATH": str(manifest_path)})
    except (RuntimeError, StopIteration) as exc:
        fail(f"Q2 NOT CHECKED: a Qwen-cell launcher did not resolve: {exc}")
        return
    inputs = dict(graphpc_text=_read(GRAPHPC_RUNNER), control_text=_read(QWEN_CONTROL_RUNNER),
                  driver_texts=[_read(root / "lib_run_single.py"), _read(root / "lib_results_logger.py")],
                  desktop_env_text=_read(root / "desktop_env" / "desktop_env.py"),
                  manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
                  step_budgets=_adapter_step_budgets(_read(next((path for path in ADAPTER_CANDIDATES if path.is_file()),
                                                                 ADAPTER_CANDIDATES[0]))) or {})
    problems, facts = _qwen_parity_problems(graphpc_tokens, control_tokens, **inputs)
    for problem in problems:
        fail(f"Q2: {problem}")
    if problems:
        return
    # NEGATIVE CONTROLS, every invocation: each doctored input must produce a problem.
    def swap(tokens: list[str], flag: str, value: str) -> list[str]:
        index = tokens.index(flag)
        return tokens[:index + 1] + [value] + tokens[index + 2:]
    controls = {
        "control --max_steps changed": (graphpc_tokens, swap(control_tokens, "--max_steps", "499"), inputs),
        "control --max_steps at the GraphPC arm's 800": (graphpc_tokens, swap(control_tokens, "--max_steps", "800"), inputs),
        "graphpc --max_steps at the control's 500": (swap(graphpc_tokens, "--max_steps", "500"), control_tokens, inputs),
        "graphpc --max_steps 801": (swap(graphpc_tokens, "--max_steps", "801"), control_tokens, inputs),
        "control --checkpoint_steps changed": (graphpc_tokens, swap(control_tokens, "--checkpoint_steps", "150"), inputs),
        "graphpc --model changed": (swap(graphpc_tokens, "--model", "qwen/qwen3-max"), control_tokens, inputs),
        "graphpc reads another manifest": (swap(graphpc_tokens, "--test_all_meta_path",
                                                "evaluation_examples/test_v2.json"), control_tokens, inputs),
        "graphpc dispatches its manifest sorted": (graphpc_tokens, control_tokens, {**inputs, "graphpc_text": inputs[
            "graphpc_text"].replace("    return all_tasks\n", "    return sorted(all_tasks)\n", 1)}),
        "control DesktopEnv proxy off": (graphpc_tokens, control_tokens,
                                         {**inputs, "control_text": inputs["control_text"].replace(
                                             "enable_proxy=args.enable_proxy", "enable_proxy=False", 1)}),
        "control writes an evaluator variable": (graphpc_tokens, control_tokens,
                                                 {**inputs, "control_text": inputs["control_text"] +
                                                  "\nos.environ[\"OSWORLD_EVAL_MODEL\"] = \"other\"\n"}),
    }
    for name, (g_tokens, c_tokens, doctored) in controls.items():
        if not _qwen_parity_problems(g_tokens, c_tokens, **doctored)[0]:
            fail(f"Q2 negative control {name!r} passed the check it must fail")
            return
    notes.append("Q2: " + "; ".join(facts) + f"; {len(controls)} negative controls each red. NOT compared, because "
                 "they do not reach a score: resume (the control's runner deletes a selected unfinished task dir, "
                 "which its launcher quarantines first; the GraphPC runner's wipe is scoped) and worker restart "
                 "(the GraphPC runner restarts a dead worker, the control's does not)")


# ==========================================================================
# DEEPSEEK MODE (--cell deepseek). The DeepSeek cell's control is upstream's v2026.08.08 M3 agent
# vendored as mm_agents/deepseek with the marked deltas its files declare (the computer tool
# declared natively, the action read from the tool_use block, the tool protocol in the history),
# driven by its own vendored runner (upstream's run_multienv_m3.py plus the --volume_size port and
# the control-agent import), against the same GraphPC runner and launcher as the other cells. The MiniMax control's files are not involved.
#   DS1. The control package == upstream's mm_agents/m3 blobs at v2026.08.08 minus exactly the marked
#        hunks (the Qwen mode's reversal, every file, README included: the same file names as upstream,
#        every block's delta declared and every declared delta marked), and each file declares exactly
#        the deltas this gate's own literal lists (DEEPSEEK_CONTROL_DELTAS), so a new delta declared and
#        marked consistently still goes red here until the literal names it.
#   DS2. The control runner == upstream's run_multienv_m3.py minus exactly its two marked hunks (R1 the
#        --volume_size port, R2 the control-agent import; the agent files label theirs D2, D3, D4, D6),
#        and what the hunks declare is held DIRECTLY, because a reversal cannot tell a hunk's content:
#        R1's two fragments are the bytes run_multienv_m3_baseline.py and run_multienv_graphpc.py carry,
#        and R2's live import names mm_agents.deepseek and no live line imports mm_agents.m3.
#   DS3. Parity: resolved from the two DeepSeek-cell launchers under DRY_RUN and parsed through each
#        runner's own argparse (both M3-shaped), the two runners feed the checkout's unmodified
#        run_single_example and DesktopEnv the same values — every args field the driver and the results
#        logger read, --model, the step budget (equal on both arms, the adapter's deepseek budget), the
#        28's dispatch order, the raw-eval dump, the evaluator environment and every DesktopEnv kwarg
#        (evaluated on fixture tasks) — and both launchers read the 28: the GraphPC arm runs them in one
#        launch, as the control does.
# Each carries negative controls that run on every invocation.
# ==========================================================================
DEEPSEEK_UPSTREAM_TAG = "v2026.08.08"
DEEPSEEK_UPSTREAM_AGENT_DIR = "mm_agents/m3"
DEEPSEEK_UPSTREAM_RUNNER = "scripts/python/run_multienv_m3.py"
DEEPSEEK_CONTROL_RUNNER = HERE / "run_multienv_deepseek_baseline.py"
DEEPSEEK_CONTROL_LAUNCHER = _launcher("run_multienv_deepseek_baseline.sh")
# The vendored agent: scripts/osworld/deepseek_control/mm_agents/deepseek/ in the repo, mm_agents/deepseek/ in the checkout.
DEEPSEEK_AGENT_DIR_CANDIDATES = (HERE / "deepseek_control" / "mm_agents" / "deepseek",
                                 HERE.parents[1] / "mm_agents" / "deepseek")
# The deltas each control file carries, by upstream file name: this gate's own literal, never read off the
# files. D2 the computer tool declared natively; D3 the action read from the tool_use block; D4 the tool
# protocol in the stored history; D6 the api_log's request_time stamped before the call. The runner
# labels its own R1, R2 (DS2).
DEEPSEEK_CONTROL_DELTAS = {
    "README.md": frozenset(),
    "__init__.py": frozenset(),
    "agent.py": frozenset({"D2", "D3", "D4", "D6"}),
    "parser.py": frozenset(),
    "prompts.py": frozenset({"D2"}),
}
# What the runner's hunk R2 must contain, held directly (DS2); R1 is VOLUME_FRAGMENTS.
DEEPSEEK_IMPORT = "from mm_agents.deepseek import M3Agent\n"
DEEPSEEK_UPSTREAM_IMPORT = "from mm_agents.m3 import M3Agent\n"
DEEPSEEK_FIXTURE_TASKS = QWEN_FIXTURE_TASKS


def _live_text(text: str) -> str:
    """The lines the interpreter sees: the header dropped, the markers and the `#|` upstream lines removed."""
    lines = text.splitlines(keepends=True)
    end = next(i for i, line in enumerate(lines) if QWEN_HEADER_END in line)
    return "".join(line for line in lines[end + 1:]
                   if not line.startswith(QWEN_UPSTREAM_LINE)
                   and not line.lstrip().startswith((QWEN_BLOCK_OPEN, QWEN_BLOCK_CLOSE)))


def _deepseek_runner_problems(text: str, upstream: str, blob: str, graphpc_text: str, m3_text: str) -> list[str]:
    """DS2 as a pure function of the texts: the reversal, then what each hunk (R1, R2) must contain."""
    problems = _qwen_file_problems(text, upstream, DEEPSEEK_UPSTREAM_RUNNER, blob, tag=DEEPSEEK_UPSTREAM_TAG)
    if problems:
        return problems
    live = _live_text(text)
    for index, fragment in enumerate(VOLUME_FRAGMENTS, start=1):
        for name, other in (("the GraphPC runner", graphpc_text), ("the MiniMax control runner", m3_text)):
            if fragment not in other:
                problems.append(f"R1: --volume_size fragment {index} is missing from {name}, so the port cannot be held identical")
        if live.count(fragment) != 1:
            problems.append(f"R1: --volume_size fragment {index} appears {live.count(fragment)} time(s) in the live text, expected 1: "
                            f"the port is not carried as the other runners carry it")
    if live.count(DEEPSEEK_IMPORT) != 1:
        problems.append(f"R2: the live text does not import the DeepSeek control agent exactly once ({DEEPSEEK_IMPORT.strip()!r})")
    if DEEPSEEK_UPSTREAM_IMPORT in live:
        problems.append(f"R2: a live line imports upstream's MiniMax control agent ({DEEPSEEK_UPSTREAM_IMPORT.strip()!r}): "
                        f"the DeepSeek control would run the M3 agent")
    return problems


def _deepseek_control_file_problems(text: str, upstream: str, upstream_path: str, blob: str) -> list[str]:
    """DS1 for one file: the reversal, then the declared deltas held to this gate's own literal. A pure function
    of the texts, so the negative controls can run it on doctored copies."""
    problems = _qwen_file_problems(text, upstream, upstream_path, blob, tag=DEEPSEEK_UPSTREAM_TAG)
    expected = DEEPSEEK_CONTROL_DELTAS.get(Path(upstream_path).name)
    if expected is None:
        problems.append(f"{Path(upstream_path).name} is not a file DEEPSEEK_CONTROL_DELTAS knows")
        return problems
    try:
        labels = _qwen_reverse(text)[2]
    except ValueError:
        return problems
    marked = {number for label in labels for number in QWEN_DELTA.findall(label.split(":")[0])}
    if marked != set(expected):
        problems.append(f"marks the deltas {sorted(marked)}; this gate's literal for the file is {sorted(expected)}")
    return problems


def check_deepseek_control_is_upstream_plus_declared_deltas(checkout_root: Path | None = None) -> None:
    """DS1 (the package) and DS2 (the runner)."""
    failures_before = len(failures)
    repo = _qwen_upstream_repo(checkout_root, tag=DEEPSEEK_UPSTREAM_TAG)
    if repo is None:
        fail(f"DS1/DS2 NOT CHECKED: no clone with tag {DEEPSEEK_UPSTREAM_TAG} at "
             f"{[str(c) for c in ((checkout_root,) if checkout_root else CHECKOUT_CANDIDATES)]}")
        return
    agent_dir = next((path for path in DEEPSEEK_AGENT_DIR_CANDIDATES if path.is_dir()), None)
    if agent_dir is None:
        fail(f"DS1 NOT CHECKED: no vendored DeepSeek control at {[str(p) for p in DEEPSEEK_AGENT_DIR_CANDIDATES]}")
        return
    listed = _git(repo, "ls-tree", "-r", "--name-only", DEEPSEEK_UPSTREAM_TAG, DEEPSEEK_UPSTREAM_AGENT_DIR)
    upstream_agent = sorted(listed.stdout.decode().split())
    vendored_names = sorted(path.name for path in agent_dir.iterdir() if path.is_file())
    if [Path(p).name for p in upstream_agent] != vendored_names:
        fail(f"DS1: the vendored control files {vendored_names} are not upstream's {upstream_agent}")
    blocks = 0
    first_with_blocks = None
    for upstream_path in upstream_agent:
        local = agent_dir / Path(upstream_path).name
        text = _read(local)
        if not text:
            continue
        upstream = _git(repo, "show", f"{DEEPSEEK_UPSTREAM_TAG}:{upstream_path}").stdout.decode("utf-8")
        blob = _git(repo, "rev-parse", f"{DEEPSEEK_UPSTREAM_TAG}:{upstream_path}").stdout.decode().strip()
        for problem in _deepseek_control_file_problems(text, upstream, upstream_path, blob):
            fail(f"DS1 {local.name}: {problem}")
        try:
            count = len(_qwen_reverse(text)[2])
        except ValueError:
            count = 0
        blocks += count
        if count and first_with_blocks is None:
            first_with_blocks = (text, upstream, upstream_path, blob)
    if first_with_blocks is None:
        fail("DS1 negative controls NOT RUN: no vendored control file carries a marked block")
    else:
        text, upstream, upstream_path, blob = first_with_blocks
        lines = text.splitlines(keepends=True)
        mark_line = next(line for line in lines if line.startswith(QWEN_UPSTREAM_LINE))
        close_line = next(line for line in lines if line.lstrip().startswith(QWEN_BLOCK_CLOSE))
        open_line = next(line for line in lines if line.lstrip().startswith(QWEN_BLOCK_OPEN))
        header_end = text.index(QWEN_HEADER_END)
        consistent = (text[:header_end].replace("Deltas carried:", "Deltas carried: D99 (a new delta),", 1)
                      + text[header_end:].replace(open_line, open_line.replace(QWEN_BLOCK_OPEN, QWEN_BLOCK_OPEN + "D99 ", 1), 1))
        controls = {
            "one extra line outside the blocks": text + "# one extra line\n",
            "a recorded upstream line altered": text.replace(mark_line, mark_line.replace(QWEN_UPSTREAM_LINE, QWEN_UPSTREAM_LINE + " ", 1), 1),
            "a block's closing marker removed": text.replace(close_line, "", 1),
            "a block labelled with an undeclared delta": text.replace(open_line, open_line.replace(QWEN_BLOCK_OPEN, QWEN_BLOCK_OPEN + "D99 ", 1), 1),
            "a new delta declared and marked consistently, unknown to this gate": consistent,
        }
        for name, doctored in controls.items():
            if doctored == text or not _deepseek_control_file_problems(doctored, upstream, upstream_path, blob):
                fail(f"DS1 negative control {name!r} passed the check it must fail")
        if len(failures) == failures_before:
            declared = "; ".join(f"{name}: {', '.join(sorted(labels)) or 'none'}" for name, labels in sorted(DEEPSEEK_CONTROL_DELTAS.items()) if labels)
            notes.append(f"DS1: {len(upstream_agent)} vendored control files ({agent_dir}) == their {DEEPSEEK_UPSTREAM_TAG} "
                         f"{DEEPSEEK_UPSTREAM_AGENT_DIR} blobs ({repo}) minus exactly their {blocks} marked blocks, header "
                         f"excluded; every block's delta declared and every declared delta marked, and each file declares "
                         f"exactly this gate's literal ({declared}); {len(controls)} negative controls each red")
    # DS2: the runner.
    failures_before_runner = len(failures)
    runner, graphpc_text, m3_text = _read(DEEPSEEK_CONTROL_RUNNER), _read(GRAPHPC_RUNNER), _read(BASELINE_RUNNER)
    if not runner or not graphpc_text or not m3_text:
        return
    upstream = _git(repo, "show", f"{DEEPSEEK_UPSTREAM_TAG}:{DEEPSEEK_UPSTREAM_RUNNER}").stdout.decode("utf-8")
    blob = _git(repo, "rev-parse", f"{DEEPSEEK_UPSTREAM_TAG}:{DEEPSEEK_UPSTREAM_RUNNER}").stdout.decode().strip()
    for problem in _deepseek_runner_problems(runner, upstream, blob, graphpc_text, m3_text):
        fail(f"DS2 {DEEPSEEK_CONTROL_RUNNER.name}: {problem}")
    if len(failures) > failures_before_runner:
        return
    live_import_line = DEEPSEEK_IMPORT
    controls = {
        "one extra line outside the blocks": runner + "# one extra line\n",
        "the --volume_size argparse entry removed": runner.replace(VOLUME_ARGPARSE_ENTRY, "", 1),
        "the --volume_size lookup removed": runner.replace(VOLUME_LOOKUP_PORTED, "", 1),
        "the import reverted to upstream's MiniMax agent": runner.replace(live_import_line, DEEPSEEK_UPSTREAM_IMPORT, 1),
    }
    for name, doctored in controls.items():
        if doctored == runner or not _deepseek_runner_problems(doctored, upstream, blob, graphpc_text, m3_text):
            fail(f"DS2 negative control {name!r} passed the check it must fail")
    if len(failures) == failures_before_runner:
        labels = _qwen_reverse(runner)[2]
        notes.append(f"DS2: {DEEPSEEK_CONTROL_RUNNER.name} == {DEEPSEEK_UPSTREAM_TAG} {DEEPSEEK_UPSTREAM_RUNNER} minus exactly its "
                     f"{len(labels)} marked blocks ({', '.join(sorted({n for l in labels for n in QWEN_DELTA.findall(l.split(':')[0])}))}), "
                     f"header excluded; R1: the --volume_size port's two fragments are the bytes {GRAPHPC_RUNNER.name} and "
                     f"{BASELINE_RUNNER.name} carry; R2: the live import is {DEEPSEEK_IMPORT.strip()!r} and no live line imports "
                     f"mm_agents.m3; {len(controls)} negative controls each red")


def _deepseek_parity_problems(graphpc_tokens: list[str], control_tokens: list[str], graphpc_text: str, control_text: str,
                              driver_texts: list[str], desktop_env_text: str, manifest: dict,
                              step_budgets: dict[str, int]) -> tuple[list[str], list[str]]:
    """DS3 as a pure function of its inputs, so the negative controls run it on doctored ones."""
    problems: list[str] = []
    facts: list[str] = []
    try:
        g = _parse_runner_args(graphpc_text, graphpc_tokens, argv_param=False)
        c = _parse_runner_args(control_text, control_tokens, argv_param=False)
    except SystemExit as exc:
        return [f"a runner's argparse refused its launcher's argv (exit {exc.code})"], facts
    fields = _driver_fields(driver_texts)
    compared = []
    for field, (how, default) in sorted(fields.items()):
        values = {arm: getattr(ns, field) if hasattr(ns, field) else
                  ("<absent, but the driver reads it bare>" if how == "bare" else default)
                  for arm, ns in (("graphpc", g), ("control", c))}
        if field == "result_dir":
            if values["graphpc"] == values["control"]:
                problems.append(f"both arms write result_dir={values['graphpc']!r}: the arms must keep distinct trees")
            continue
        compared.append(field)
        if values["graphpc"] != values["control"]:
            problems.append(f"driver field {field!r} differs: graphpc={values['graphpc']!r} control={values['control']!r}")
    facts.append(f"driver fields (lib_run_single + lib_results_logger AST): {sorted(fields)}; equal: {compared}; "
                 f"result_dir distinct per arm by design")
    for field in ("model", "eval_version", "test_config_base_dir", "save_model_eval_raw_info", "max_steps"):
        if getattr(g, field, None) != getattr(c, field, None):
            problems.append(f"{field} differs: graphpc={getattr(g, field, None)!r} control={getattr(c, field, None)!r}")
    # The step budget: one value on both arms, the adapter's deepseek budget (runner delta 15 refuses any other on
    # the GraphPC arm; the control has no check, so the launcher's value is held here).
    budget = step_budgets.get("deepseek")
    if budget is None:
        problems.append("the adapter declares no deepseek step budget (OSWORLD_MAX_STEPS_BY_BACKBONE['deepseek']): the "
                        "GraphPC runner's delta 15 refuses every deepseek launch until it does, and the control's "
                        "--max_steps cannot be held to it")
    else:
        for arm, ns in (("graphpc", g), ("control", c)):
            if getattr(ns, "max_steps", None) != budget:
                problems.append(f"{arm} runs --max_steps {getattr(ns, 'max_steps', None)!r}, not the adapter's deepseek budget {budget!r}")
    if not str(getattr(c, "test_all_meta_path", "")).endswith("prereg_task_list.json"):
        problems.append(f"the control reads the manifest {getattr(c, 'test_all_meta_path', None)!r}, not the 28")
    if not str(getattr(g, "test_all_meta_path", "")).endswith("prereg_task_list.json"):
        problems.append(f"graphpc reads the manifest {getattr(g, 'test_all_meta_path', None)!r}, not the 28")
    facts.append(f"--model {c.model!r}, eval_version {c.eval_version}, test_config_base_dir {c.test_config_base_dir}, "
                 f"save_model_eval_raw_info {c.save_model_eval_raw_info}, max_steps {c.max_steps} (the adapter's deepseek "
                 f"budget {budget}): equal; manifests: control {c.test_all_meta_path} (the 28), graphpc "
                 f"{g.test_all_meta_path} (the 28)")
    for arm, text in (("graphpc", graphpc_text), ("control", control_text)):
        tree = ast.parse(text)
        call = _single_call(tree, "lib_run_single.run_single_example")
        if call is None or len(call.args) < 6 or ast.unparse(call.args[3]) != "args.max_steps" or ast.unparse(call.args[5]) != "args":
            problems.append(f"{arm}: not exactly one run_single_example(agent, env, example, args.max_steps, instruction, "
                            f"args, ...) call")
        dumps = _raw_eval_dump(text)
        if len(dumps) != 1 or not dumps[0].endswith("'model_eval_file_log')"):
            problems.append(f"{arm}: OSWORLD_EVAL_SAVE_RAW_DIR is not set exactly once to <task dir>/model_eval_file_log: {dumps}")
        writes = _env_writes(text)
        if writes:
            problems.append(f"{arm}: the runner writes evaluator / user-simulator environment itself: {writes}")
        if not any(isinstance(n, ast.Expr) and ast.unparse(n) == "load_dotenv()" for n in tree.body):
            problems.append(f"{arm}: no module-level load_dotenv(), so its evaluator / user-simulator env is not the .env's")
    facts.append("both call the checkout's run_single_example(..., args.max_steps, ..., args, ...) once, dump raw evals "
                 "to <task>/model_eval_file_log, load the same .env and write no OSWORLD_EVAL_* / OSWORLD_USER_SIM_* key")
    try:
        orders = {"graphpc": _dispatch_order(graphpc_text, g, manifest, graphpc=True),
                  "control": _dispatch_order(control_text, c, manifest, graphpc=True)}
        listed = [(domain, str(t)) for domain, ids in manifest.items() for t in ids]
        if orders["graphpc"] != orders["control"]:
            problems.append(f"dispatch order differs: graphpc {orders['graphpc'][:3]}... control {orders['control'][:3]}...")
        elif orders["control"] != listed:
            problems.append(f"the dispatch order is not the 28's manifest order: {orders['control'][:3]}...")
        else:
            facts.append(f"dispatch order equal on the 28 ({len(listed)} tasks, manifest order)")
        defaults = _desktop_env_defaults(desktop_env_text)
        for index, example in enumerate(DEEPSEEK_FIXTURE_TASKS):
            kw = {"graphpc": _desktop_env_kwargs(graphpc_text, g, dict(example), graphpc=True, defaults=defaults),
                  "control": _desktop_env_kwargs(control_text, c, dict(example), graphpc=True, defaults=defaults)}
            differ = {k: (kw["graphpc"].get(k), kw["control"].get(k)) for k in kw["graphpc"] if kw["graphpc"].get(k) != kw["control"].get(k)}
            if differ:
                problems.append(f"DesktopEnv kwargs differ on fixture task {index} {example}: {differ}")
        facts.append(f"DesktopEnv kwargs equal on {len(DEEPSEEK_FIXTURE_TASKS)} fixture tasks: {sorted(kw['control'])}")
    except Exception as exc:  # noqa: BLE001 — a check that could not run has not passed
        problems.append(f"could not evaluate dispatch order / DesktopEnv kwargs: {type(exc).__name__}: {exc}")
    return problems, facts


def check_deepseek_runner_parity(checkout_root: Path | None = None) -> None:
    root = next((c for c in ((checkout_root,) if checkout_root else CHECKOUT_CANDIDATES)
                 if (c / "lib_run_single.py").is_file()), None)
    manifest_path = _qwen_manifest()
    if root is None or manifest_path is None:
        fail(f"DS3 NOT CHECKED: checkout {root} / manifest {manifest_path} not found")
        return
    # The launchers resolve the task list from the cwd (osworld_cell_require_task_list), so their defaults are
    # exercised from a scratch checkout root that holds the 28 under evaluation_examples/.
    with tempfile.TemporaryDirectory() as tmp:
        examples = Path(tmp) / "evaluation_examples"
        examples.mkdir()
        shutil.copy(manifest_path, examples / "prereg_task_list.json")
        try:
            control_tokens = _dry_run_tokens(DEEPSEEK_CONTROL_LAUNCHER, {}, cwd=Path(tmp))
            graphpc_tokens = _dry_run_tokens(GRAPHPC_LAUNCHER, {"OSWORLD_CELL": "deepseek"}, cwd=Path(tmp))
        except (RuntimeError, StopIteration) as exc:
            fail(f"DS3 NOT CHECKED: a DeepSeek-cell launcher did not resolve: {exc}")
            return
    inputs = dict(graphpc_text=_read(GRAPHPC_RUNNER), control_text=_read(DEEPSEEK_CONTROL_RUNNER),
                  driver_texts=[_read(root / "lib_run_single.py"), _read(root / "lib_results_logger.py")],
                  desktop_env_text=_read(root / "desktop_env" / "desktop_env.py"),
                  manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
                  step_budgets=_adapter_step_budgets(_read(next((path for path in ADAPTER_CANDIDATES if path.is_file()),
                                                                 ADAPTER_CANDIDATES[0]))) or {})
    problems, facts = _deepseek_parity_problems(graphpc_tokens, control_tokens, **inputs)
    for problem in problems:
        fail(f"DS3: {problem}")
    if problems:
        return
    # NEGATIVE CONTROLS, every invocation: each doctored input must produce a problem.
    def swap(tokens: list[str], flag: str, value: str) -> list[str]:
        index = tokens.index(flag)
        return tokens[:index + 1] + [value] + tokens[index + 2:]
    budget = inputs["step_budgets"]["deepseek"]
    controls = {
        "control --max_steps changed": (graphpc_tokens, swap(control_tokens, "--max_steps", str(budget - 1)), inputs),
        "graphpc --max_steps changed": (swap(graphpc_tokens, "--max_steps", str(budget + 1)), control_tokens, inputs),
        "control --checkpoint_steps changed": (graphpc_tokens, swap(control_tokens, "--checkpoint_steps", "150"), inputs),
        "graphpc --model changed": (swap(graphpc_tokens, "--model", "deepseek-chat"), control_tokens, inputs),
        "the adapter declares no deepseek budget": (graphpc_tokens, control_tokens,
                                                    {**inputs, "step_budgets": {k: v for k, v in inputs["step_budgets"].items() if k != "deepseek"}}),
        "graphpc dispatches its manifest sorted": (graphpc_tokens, control_tokens, {**inputs, "graphpc_text": inputs[
            "graphpc_text"].replace("    return all_tasks\n", "    return sorted(all_tasks)\n", 1)}),
        "control DesktopEnv proxy off": (graphpc_tokens, control_tokens,
                                         {**inputs, "control_text": inputs["control_text"].replace(
                                             "enable_proxy=True", "enable_proxy=False", 1)}),
        "control writes an evaluator variable": (graphpc_tokens, control_tokens,
                                                 {**inputs, "control_text": inputs["control_text"] +
                                                  "\nos.environ[\"OSWORLD_EVAL_MODEL\"] = \"other\"\n"}),
    }
    for name, (g_tokens, c_tokens, doctored) in controls.items():
        if not _deepseek_parity_problems(g_tokens, c_tokens, **doctored)[0]:
            fail(f"DS3 negative control {name!r} passed the check it must fail")
            return
    # The one GraphPC launch reads the 28: a launch reading another manifest is red for that reason (the driver reads no
    # task-list field, so nothing else in DS3 can catch it).
    other = swap(graphpc_tokens, "--test_all_meta_path", "evaluation_examples/test_v2.json")
    if not any(p.startswith("graphpc reads the manifest") and p.endswith("not the 28")
               for p in _deepseek_parity_problems(other, control_tokens, **inputs)[0]):
        fail("DS3 negative control 'the GraphPC launch reads a manifest other than the 28' passed the check it must fail")
        return
    notes.append("DS3: " + "; ".join(facts) + f"; {len(controls) + 1} negative controls each red, the GraphPC launch reading "
                 f"a manifest other than the 28 among them. NOT compared, because they do not reach "
                 "a score: resume (the control's runner wipes unfinished task dirs unscoped, its launcher pre-cleans "
                 "first; the GraphPC runner's wipe is scoped) and worker restart (bounded on the GraphPC arm only)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkout_root", type=Path, default=None,
                        help="OSWorld-V2 checkout the qwen and deepseek modes compare against, a full clone carrying "
                             "the v2026.08.08 and osworld-v2.1 tags (default: this file's checkout on the host, the "
                             "staging clone next to the repository locally)")
    parser.add_argument("--cell", choices=["minimax", "qwen", "deepseek"], default="minimax",
                        help="minimax: the M3 pair (checks 1-3 and 5); qwen: the Qwen cell (checks Q1-Q2); deepseek: the "
                             "DeepSeek cell (checks DS1-DS3)")
    args = parser.parse_args(argv)
    if args.cell == "qwen":
        check_qwen_control_is_upstream_plus_declared_deltas(args.checkout_root)
        check_qwen_runner_parity(args.checkout_root)
    elif args.cell == "deepseek":
        check_deepseek_control_is_upstream_plus_declared_deltas(args.checkout_root)
        check_deepseek_runner_parity(args.checkout_root)
    else:
        check_arms_carry_identical_deltas()
        check_the_baseline_is_upstream_plus_the_port()
        check_the_resume_wipe_is_scoped()
        check_the_step_budget_is_enforced()

    for note in notes:
        print(f"OK: {note}")
    if failures:
        print(file=sys.stderr)
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"\n{len(failures)} check(s) failed. Do not launch a scored run.", file=sys.stderr)
        return 1
    print(f"\nAll {len(notes)} runner-delta checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
