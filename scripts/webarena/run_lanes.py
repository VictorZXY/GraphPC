"""Lane driver for one arm of a full WebArena-812 cell: the control (run_baseline.py), the GraphPC cold arm
(run_graphpc.py), or the GraphPC warm round (run_graphpc.py, one task per process against a frozen store).

    # write the arm's plan: lane inputs, regime argv, tripwires (checks the code root; starts nothing)
    python scripts/webarena/run_lanes.py prepare --arm control|graphpc|warm --run_dir DIR --code_root PINNED \\
        --webarena_python WEBARENA_ENV_PYTHON --partitions_dir PARTITIONS --argv_json ARGV.json --argv_key KEY \\
        --tripwires_json TRIPWIRES.json [--support_set_root STORE] [--frozen_master MASTER --master_sha256 HASH] \\
        [--lanes l1,l2] [--task_ids 7] [--env_set NAME=VALUE ...] [--webarena_root W]
    python scripts/webarena/run_lanes.py launch --run_dir DIR        # start the supervisor (its own session)
    python scripts/webarena/run_lanes.py status --run_dir DIR
    python scripts/webarena/run_lanes.py freeze --run_dir DIR --reason TEXT   # what the 402 tripwire does
    python scripts/webarena/run_lanes.py resume --run_dir DIR [--long]
    python scripts/webarena/run_lanes.py casualties --run_dir DIR
    python scripts/webarena/run_lanes.py add_lane --run_dir DIR --lane rerun_pass1 --phase 3 --task_ids 1,2 \\
        [--extra_argv "--max_tokens 65536"]

Interpreter: the GraphPC env (stdlib only here); the runners run under --webarena_python with CWD = the WebArena
checkout, as every WebArena lane has, from the code root (a pinned detached worktree with its own .env).

LANES. One lane per site partition (``PARTITIONS/<lane>/task_ids.json``, Lite-first order); ``cross_site`` is phase
2 and runs after every phase-1 lane is done, the others are phase 1 and run in parallel. Each lane's runner output is
APPENDED to ``DIR/lanes/<lane>/run.log``. Every runner gets the operator's environment plus ``--env_set`` and minus
``ENV_UNSET`` (SDK base URLs, organisation / project, the judge model, the effort switch).
  control   one run_baseline.py over the lane's ``task_ids.json`` (its resume skips tasks with a render).
  graphpc  one run_graphpc.py over ``DIR/lanes/<lane>/configs/NNNN_<id>.json`` (ordinal symlinks: the filename
            sort IS the Lite-first order) with the one shared ``--support_set_root`` (in-run accumulation is the
            method); its resume skips scored tasks.
  warm      one run_graphpc.py PER TASK: before each episode the lane's working store is restored byte-identical
            from the frozen master (``--frozen_master``, its content hash checked before and after every episode),
            the single task runs, the store delta is archived under ``DIR/warm_wins_archive/<id>/``. A task with
            genuine telemetry is skipped; an infra / eval casualty's telemetry is moved aside and the task re-runs in
            a later pass, up to 3 attempts. The lane waits between episodes while ``DIR/FREEZE`` exists.

PROCESSES AND PIDS. ``launch`` starts the supervisor in its own session; the supervisor starts the heartbeat (and
respawns it, alerted, whenever it finds it gone while the run is unfinished) and,
phase by phase, one lane worker per lane; each worker starts its runner(s) in their own sessions. Every PID is
appended to ``DIR/pids.tsv`` with its role; a PID counts as ours only while its command line still names DIR. Nothing
here matches processes by name or pattern, and nothing kills anything except ``resume --long`` (the run's own
recorded, frozen runners) and a timed-out auto_login (its own recorded process group).

SITE LOGINS (a site session can expire between phases or during a long freeze). At every phase start and every
relaunch, once
``DIR/FREEZE`` is absent and none of this run's runners is alive, WebArena's ``browser_env/auto_login.py`` re-renders
``.auth`` in place (CWD = the WebArena root, the code root's .env sourced) and ``scripts/webarena/verify_auth_logins.py``
checks every state file the phase's tasks use; a red writes ``DIR/FREEZE`` and alerts, and the supervisor waits.

TRIPWIRES (``--tripwires_json``, pre-registered data). The heartbeat reads each lane's run.log incrementally (offsets
kept in ``DIR/tripwire_state.json``, so a relaunch never re-fires on old lines). A rule is
``{"name", "action": "freeze"|"alert"|"change", "regex", ["baseline"]}``:
  freeze  the first match writes ``DIR/FREEZE`` and SIGSTOPs every alive recorded runner's process group (the
          402 tripwire: stop the burn, keep the state). Alerted.
  alert   a match is alerted at its 1st, 2nd, 4th, 8th, ... occurrence (counted in heartbeat.json).
  change  the regex's group 1 is a value (a fingerprint); a value other than ``baseline`` (or the first seen) is
          alerted once per new value.
Also alerted: a lane log idle past --stall_minutes while its runner runs unfrozen, a worker gone while its lane
runs, the homepage app down, and the run-wide casualty count reaching 3 (the critical threshold). Alerts
are lines of ``DIR/ALERTS.log``; nothing here stops a run on an alert (diagnose first).

FREEZE AND RESUME. ``freeze`` (or the freeze tripwire) STOPs the runners mid-episode. ``resume``:
  held < 30 min   SIGCONT the stopped runners and remove the FREEZE (sessions survive 45 min idle).
  held >= 30 min (or --long)
                  sessions may have died while the episode in flight held them: SIGTERM + SIGCONT the stopped
                  runners' groups (SIGKILL after 60 s), then remove the FREEZE. Each worker sees its runner end by a
                  signal, sets the in-flight task aside (control: its render html, which would otherwise mark it
                  finished; GraphPC: nothing, an unfinished episode writes no summary row; warm: the attempt's
                  telemetry if any) and exits as INTERRUPTED; the supervisor re-renders and checks .auth, then
                  relaunches those lanes, whose resume runs the set-aside task first.
A runner that ends by a signal for any other reason is handled the same way; a lane interrupted 5 times without a
newly scored task stops the run instead of looping. A runner that exits non-zero by itself stops the run before the
next phase (RUN_STOPPED), alerted.

CASUALTIES. ``casualties`` lists, per lane, the planned tasks that have no score in ANY lane (control: no
``[Result]`` line in the lane log; GraphPC: no scored row in the merged summaries) — the re-run class of the
casualty policy. ``status`` totals count each task once, whichever lane scored it.
``add_lane`` appends a re-run lane (a later phase, its own task list and optional extra flags) for them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = ("control", "graphpc", "warm")
RUNNER = {"control": "scripts/webarena/run_baseline.py", "graphpc": "scripts/webarena/run_graphpc.py",
          "warm": "scripts/webarena/run_graphpc.py"}
PHASE2_LANE = "cross_site"
CASUALTY = ("infra_error", "eval_error")
ATTEMPT_CAP = 3                  # warm: attempts per task (a pass each)
INTERRUPTED_EXIT = 3             # a worker whose runner ended by a signal
LONG_FREEZE_S = 30 * 60          # site sessions survive 45 min idle and are dead after 2 h 15 min
KILL_GRACE_S = 60
MAX_RELAUNCHES_WITHOUT_PROGRESS = 5
CASUALTY_ALERT = 3               # the critical casualty threshold
# Removed from every runner's environment: an operator's shell must not
# re-point an SDK, an organisation or project, the judge, or the control's effort switch. The names are recorded in
# plan.json; the code root's .env (which the runners load) still supplies any that it sets.
ENV_UNSET = ("WEBARENA_JUDGE_MODEL", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_ORGANIZATION",
             "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "WEBARENA_AGENT_REASONING_EFFORT")
SUPERVISOR_POLL_S = 15
TRIPWIRE_POLL_S = 15             # the 402 tripwire must stop the burn within seconds, not a heartbeat period
RELAUNCH_AUTH = "RELAUNCH_AUTH"  # written by resume --long: relaunch only after a new login refresh
_RESULT_RE = re.compile(r"\[Result\] \((PASS|FAIL)\) (\S+)")
SITE_VARIABLES = ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "MAP", "WIKIPEDIA", "HOMEPAGE")
# WebArena's own login script, run from the WebArena root, and the check run after it (SITE LOGINS above)
AUTO_LOGIN = "browser_env/auto_login.py"
AUTH_CHECK = Path(__file__).resolve().parent / "verify_auth_logins.py"
AUTO_LOGIN_TIMEOUT_S = 600
AUTO_LOGIN_KILL_GRACE_S = 30   # between the group's SIGTERM (Playwright closes its browsers on it) and its SIGKILL


# --------------------------------------------------------------------------- process helpers

def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload) -> None:
    """Atomic: a reader (the heartbeat, a status call) never sees a torn file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def append_line(path: Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def pid_alive(pid: int, run_dir: Path) -> bool:
    """A recorded PID is ours only while its command line still names this run dir (a PID can be reused)."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return str(run_dir) in cmdline


def _group_alive(pgid: int) -> bool:
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                state, _ppid, pgrp = (d / "stat").read_text().rsplit(")", 1)[1].split()[:3]
            except (OSError, ValueError):
                continue
            if int(pgrp) == pgid and state != "Z":
                return True
    return False


def _auto_login(run: Run, rec: dict, out: Path, env_file: str, log: Path) -> int | None:
    """auto_login in its own session, so its recorded PID leads a process group holding it and every process it
    started in that group; None when it outlived AUTO_LOGIN_TIMEOUT_S and that group was ended."""
    # bash's $0 is the log dir, so the recorded PID names this run while auto_login runs (pid_alive)
    with open(log, "wb") as f:
        proc = subprocess.Popen(["bash", "-c", f'set -a && . "$1" && set +a && "$2" {AUTO_LOGIN}', str(out), env_file,
                                 rec["webarena_python"]], cwd=rec["webarena_root"], stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    run.record_pid("auto_login", proc.pid)
    try:
        return proc.wait(timeout=AUTO_LOGIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pass
    for sig, grace in ((signal.SIGTERM, AUTO_LOGIN_KILL_GRACE_S), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + grace
        while time.time() < deadline:
            proc.poll()
            if not _group_alive(proc.pid):
                break
            time.sleep(1)
        if not _group_alive(proc.pid):
            break
    proc.wait()
    return None


def homepage_status(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return {"url": url, "status": resp.status, "ok": resp.status == 200}
    except Exception as exc:  # noqa: BLE001 — down is a result, not a crash
        return {"url": url, "status": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


def git(code_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(code_root), *args], capture_output=True, text=True)


def check_code_root(code_root: Path) -> str:
    if not (code_root / "scripts/webarena/run_graphpc.py").is_file():
        refuse(f"{code_root} has no scripts/webarena/run_graphpc.py")
    head = git(code_root, "rev-parse", "HEAD")
    if head.returncode != 0:
        refuse(f"{code_root} is not a git checkout: {head.stderr.strip()}")
    if git(code_root, "symbolic-ref", "-q", "HEAD").returncode == 0:
        refuse(f"{code_root} is on a branch; the code root must be a pinned DETACHED worktree")
    dirty = git(code_root, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        refuse(f"{code_root} has uncommitted changes to tracked files:\n{dirty}")
    env = code_root / ".env"
    if not env.is_file():
        refuse(f"{code_root}/.env is missing (the runner loads the site URLs and keys from its own checkout)")
    names = {line.split("=", 1)[0].strip() for line in env.read_text().splitlines()
             if "=" in line and not line.lstrip().startswith("#") and line.split("=", 1)[1].strip()}
    missing = [v for v in SITE_VARIABLES if v not in names]
    if missing:
        refuse(f"{code_root}/.env sets no value for {missing}")
    return head.stdout.strip()


class Run:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir).resolve()
        self.plan_path = self.dir / "plan.json"
        self.pids = self.dir / "pids.tsv"
        self.alerts = self.dir / "ALERTS.log"
        self.log = self.dir / "driver.log"
        self.freeze = self.dir / "FREEZE"

    def plan(self) -> dict:
        return read_json(self.plan_path)

    def say(self, msg: str, log: Path | None = None) -> None:
        append_line(log or self.log, f"{utc()} {msg}")

    def alert(self, msg: str) -> None:
        append_line(self.alerts, f"{utc()} ALERT {msg}")
        self.say(f"ALERT {msg}")

    def record_pid(self, role: str, pid: int, lane: str = "-", task: str = "-", attempt: str = "-") -> None:
        append_line(self.pids, f"{utc()}\t{role}\t{lane}\t{task}\t{attempt}\t{pid}")

    def recorded(self, role: str | None = None) -> list[dict]:
        if not self.pids.exists():
            return []
        rows = []
        for line in self.pids.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) != 6 or not parts[5].isdigit():   # a line still being appended
                continue
            ts, r, lane, task, attempt, pid = parts
            if role is None or r == role:
                rows.append({"utc": ts, "role": r, "lane": lane, "task": task, "attempt": attempt, "pid": int(pid)})
        return rows

    def alive(self, role: str | None = None) -> list[dict]:
        return [p for p in self.recorded(role) if pid_alive(p["pid"], self.dir)]

    def lane_dir(self, lane: str) -> Path:
        return self.dir / "lanes" / lane


# --------------------------------------------------------------------------- progress

def merged_summary_rows(result_dir: Path) -> list[dict]:
    """run_graphpc's merged view: canonical summary.json, then every summary_<ts>.json, later rows winning."""
    rows: dict[str, dict] = {}
    for path in [result_dir / "summary.json", *sorted(result_dir.glob("summary_*.json"))]:
        try:
            for row in json.loads(path.read_text()):
                if isinstance(row, dict):
                    rows[str(row.get("task_id") or Path(str(row.get("config_file"))).stem.split("_")[-1])] = row
        except (OSError, json.JSONDecodeError):
            continue
    return list(rows.values())


def control_verdicts(log: Path) -> dict[int, bool]:
    verdicts: dict[int, bool] = {}
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            match = _RESULT_RE.search(line)
            if match:
                verdicts[int(Path(match.group(2)).stem)] = match.group(1) == "PASS"
    return verdicts


def lane_progress(run: Run, plan: dict, lane: dict) -> dict:
    """Planned / scored / passed / casualty task ids of one lane, read from what its runner wrote."""
    ldir = run.lane_dir(lane["lane"])
    planned = list(lane["tasks"])
    if plan["arm"] == "control":
        verdicts = control_verdicts(ldir / "run.log")
        scored = {t: v for t, v in verdicts.items() if t in planned}
        rendered = sorted((ldir / "results").glob("render_*.html"), key=lambda p: p.stat().st_mtime)
        state = _lane_state(ldir)
        in_flight = (int(rendered[-1].stem.split("_")[1]) if rendered and state.get("runner_pid")
                     and pid_alive(state["runner_pid"], run.dir) else None)
        errored = sorted(t for t in planned if t not in scored and t != in_flight
                         and (ldir / "results" / f"render_{t}.html").exists())
        return {"planned": len(planned), "scored": len(scored), "passed": sum(scored.values()), "casualties": errored,
                "unscored": sorted(t for t in planned if t not in scored), "passed_ids": sorted(t for t, v in scored.items() if v),
                "scored_ids": sorted(scored)}
    rows = {int(r["task_id"]): r for r in merged_summary_rows(ldir / "results") if str(r.get("task_id", "")).isdigit()}
    if plan["arm"] == "warm":
        for t in planned:
            tele = ldir / "results" / f"telemetry_{t}.json"
            if tele.exists() and t not in rows:
                try:      # the runner writes telemetry in place, not atomically: a beat may read it mid-write
                    data = read_json(tele)
                except (OSError, json.JSONDecodeError):
                    continue
                rows[t] = {"score": data.get("score"), "errored": data.get("error_type") in CASUALTY}
    scored = {t: r for t, r in rows.items() if t in planned and not r.get("errored") and r.get("score") is not None}
    return {"planned": len(planned), "scored": len(scored),
            "passed": sum(1 for r in scored.values() if (r.get("score") or 0) > 0),
            "casualties": sorted(t for t, r in rows.items() if t in planned and r.get("errored")),
            "unscored": sorted(t for t in planned if t not in scored),
            "passed_ids": sorted(t for t, r in scored.items() if (r.get("score") or 0) > 0), "scored_ids": sorted(scored)}


# --------------------------------------------------------------------------- prepare

def _sha256_tree(root: Path) -> tuple[str, int]:
    files = sorted((root / "webarena").glob("*.json"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest(), len(files)


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSED: {msg}")


def prepare(args: argparse.Namespace) -> int:
    run = Run(Path(args.run_dir))
    code_root = Path(args.code_root).resolve()
    commit = check_code_root(code_root)
    webarena_root = Path(args.webarena_root).resolve()
    if not (webarena_root / AUTO_LOGIN).is_file():
        refuse(f"{webarena_root} has no {AUTO_LOGIN}, which re-renders .auth at every phase start")
    argv = json.loads(Path(args.argv_json).read_text())[args.argv_key]
    tripwires = json.loads(Path(args.tripwires_json).read_text())
    for rule in tripwires:
        if rule.get("action") not in ("freeze", "alert", "change"):
            refuse(f"tripwire {rule} has no valid action")
        re.compile(rule["regex"])
    partitions = Path(args.partitions_dir)
    names = args.lanes.split(",") if args.lanes else sorted(p.name for p in partitions.iterdir()
                                                            if (p / "task_ids.json").is_file())
    only = [int(t) for t in args.task_ids.split(",")] if args.task_ids else None
    lanes = []
    for name in names:
        ids = [int(t) for t in read_json(partitions / name / "task_ids.json")["task_ids"]]
        if only is not None:
            ids = [t for t in ids if t in only]
        if ids:
            lanes.append({"lane": name, "phase": 2 if name == PHASE2_LANE else 1, "tasks": ids, "extra_argv": []})
    if only is not None and sorted(t for lane in lanes for t in lane["tasks"]) != sorted(only):
        refuse(f"--task_ids {only} are not all in the partitions' lanes")
    plan = {"arm": args.arm, "code_root": str(code_root), "code_commit": commit,
            "webarena_root": str(webarena_root), "webarena_python": args.webarena_python,
            "argv": argv, "env_set": dict(kv.split("=", 1) for kv in args.env_set), "env_unset": list(ENV_UNSET),
            "support_set_root": str(Path(args.support_set_root).resolve()) if args.support_set_root else None,
            "frozen_master": str(Path(args.frozen_master).resolve()) if args.frozen_master else None,
            "master_sha256": args.master_sha256, "lanes": lanes, "tripwires": tripwires,
            "homepage_url": args.homepage_url, "heartbeat_seconds": args.heartbeat_seconds,
            "stall_minutes": args.stall_minutes, "lane_stagger_seconds": args.lane_stagger_seconds,
            "partitions_dir": str(partitions.resolve())}
    for flag, value in (("--support_set_root", args.support_set_root), ("--frozen_master", args.frozen_master)):
        if value and not Path(value).is_absolute():
            refuse(f"{flag} {value!r} is relative; pass an absolute path (the runners run with CWD = the WebArena checkout)")
    if args.arm == "graphpc" and not plan["support_set_root"]:
        refuse("--arm graphpc needs an absolute --support_set_root (one shared store; the runners run with "
               "CWD = the WebArena checkout)")
    if args.arm == "warm":
        if not (plan["frozen_master"] and plan["master_sha256"]):
            refuse("--arm warm needs --frozen_master and its --master_sha256")
        digest, count = _sha256_tree(Path(plan["frozen_master"]))
        if digest != plan["master_sha256"]:
            refuse(f"the frozen master {plan['frozen_master']} hashes {digest} ({count} entries), not "
                   f"{plan['master_sha256']}")
    if run.plan_path.exists():
        old = run.plan()
        for key in ("arm", "code_commit", "argv", "env_set", "env_unset", "support_set_root", "frozen_master",
                    "master_sha256", "lanes", "tripwires"):
            if old[key] != plan[key]:
                refuse(f"{run.dir} was prepared with a different {key}: {old[key]!r} (now {plan[key]!r})")
        print(f"{run.dir}: plan unchanged")
        return 0
    if run.dir.exists() and any(run.dir.iterdir()):
        refuse(f"{run.dir} exists, is not empty and holds no plan.json of this driver")
    for lane in lanes:
        ldir = run.lane_dir(lane["lane"])
        (ldir / "results").mkdir(parents=True)
        if args.arm == "control":
            write_json(ldir / "task_ids.json", {"task_ids": lane["tasks"],
                                                "_source": str(partitions / lane["lane"] / "task_ids.json")})
        elif args.arm == "graphpc":
            (ldir / "configs").mkdir()
            for ordinal, tid in enumerate(lane["tasks"], 1):
                (ldir / "configs" / f"{ordinal:04d}_{tid}.json").symlink_to(webarena_root / "config_files" / f"{tid}.json")
    if args.arm == "graphpc":
        Path(plan["support_set_root"]).mkdir(parents=True, exist_ok=True)
    plan["prepared_utc"] = utc()
    write_json(run.plan_path, plan)
    print(f"{args.arm} plan at {run.dir}: code {commit[:12]}; lanes "
          + ", ".join(f"{l['lane']}(phase {l['phase']}): {len(l['tasks'])}" for l in lanes))
    return 0


def _settle_finished_run(run: Run, plan: dict) -> None:
    """After RUN_DONE / RUN_STOPPED the heartbeat exits at its next tripwire poll; wait for that, never kill it."""
    if not ((run.dir / "RUN_DONE").exists() or (run.dir / "RUN_STOPPED").exists()):
        return
    deadline = time.time() + 3 * min(TRIPWIRE_POLL_S, plan["heartbeat_seconds"]) + 5
    while time.time() < deadline and run.alive() and all(p["role"] == "heartbeat" for p in run.alive()):
        time.sleep(1)


def add_lane(args: argparse.Namespace) -> int:
    run = Run(Path(args.run_dir))
    plan = run.plan()
    if any(l["lane"] == args.lane for l in plan["lanes"]):
        refuse(f"lane {args.lane} exists")
    _settle_finished_run(run, plan)
    if run.alive():
        refuse(f"processes of this run are alive: {run.alive()}")
    ids = [int(t) for t in args.task_ids.split(",")]
    lane = {"lane": args.lane, "phase": args.phase, "tasks": ids, "extra_argv": args.extra_argv.split()}
    ldir = run.lane_dir(args.lane)
    (ldir / "results").mkdir(parents=True)
    if plan["arm"] == "control":
        write_json(ldir / "task_ids.json", {"task_ids": ids})
    elif plan["arm"] == "graphpc":
        (ldir / "configs").mkdir()
        for ordinal, tid in enumerate(ids, 1):
            (ldir / "configs" / f"{ordinal:04d}_{tid}.json").symlink_to(
                Path(plan["webarena_root"]) / "config_files" / f"{tid}.json")
    plan["lanes"].append(lane)
    write_json(run.plan_path, plan)
    for marker in ("RUN_DONE", "RUN_STOPPED"):
        if (run.dir / marker).exists():
            (run.dir / marker).rename(run.dir / f"{marker}.{utc()}")
    print(f"added lane {lane}")
    return 0


# --------------------------------------------------------------------------- runners and workers

def runner_argv(plan: dict, lane: dict, ldir: Path, *, config_dir: Path | None = None,
                store: Path | None = None) -> list[str]:
    base = [plan["webarena_python"], str(Path(plan["code_root"]) / RUNNER[plan["arm"]]), *plan["argv"],
            *lane.get("extra_argv", []), "--webarena_root", plan["webarena_root"], "--result_dir", str(ldir / "results")]
    if plan["arm"] == "control":
        return [*base, "--task_ids_json", str(ldir / "task_ids.json")]
    return [*base, "--config_dir", str(config_dir or ldir / "configs"),
            "--support_set_root", str(store or plan["support_set_root"])]


def runner_env(plan: dict) -> dict:
    env = dict(os.environ)
    env.update(plan["env_set"])
    for name in plan.get("env_unset", []):
        env.pop(name, None)
    return env


def start_runner(run: Run, plan: dict, lane: dict, argv: list[str], task: str = "-") -> subprocess.Popen:
    ldir = run.lane_dir(lane["lane"])
    with open(ldir / "run.log", "ab") as out:
        proc = subprocess.Popen(argv, cwd=plan["webarena_root"], env=runner_env(plan), stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    run.record_pid("runner", proc.pid, lane["lane"], task)
    write_json(ldir / "lane_state.json", {**_lane_state(ldir), "runner_pid": proc.pid, "current_task": task,
                                          "runner_started_utc": utc()})
    run.say(f"[{lane['lane']}] runner pid {proc.pid} started (task {task})", ldir / "lane.log")
    return proc


def _lane_state(ldir: Path) -> dict:
    path = ldir / "lane_state.json"
    return read_json(path) if path.exists() else {}


def quarantine_control_render(run: Run, lane: dict) -> None:
    """The control's resume treats a task with a render html as finished, so the task in flight when its runner
    was killed would never run again. Set its render aside: the newest render of the lane, if no [Result] names it."""
    ldir = run.lane_dir(lane["lane"])
    renders = sorted((p for p in (ldir / "results").glob("render_*.html")), key=lambda p: p.stat().st_mtime)
    if not renders:
        return
    newest = renders[-1]
    tid = int(newest.stem.split("_")[1])
    if tid in control_verdicts(ldir / "run.log") or tid not in lane["tasks"]:
        return
    target_dir = ldir / "quarantine"
    target_dir.mkdir(exist_ok=True)
    k = 1
    while (target := target_dir / f"{newest.name}.interrupted{k}").exists():
        k += 1
    newest.rename(target)
    run.say(f"[{lane['lane']}] task {tid}: interrupted mid-task; its render set aside as {target.name}, the "
            "relaunch runs it again", ldir / "lane.log")


def run_continuous_lane(run: Run, plan: dict, lane: dict) -> int:
    ldir = run.lane_dir(lane["lane"])
    proc = start_runner(run, plan, lane, runner_argv(plan, lane, ldir))
    rc = proc.wait()
    write_json(ldir / "lane_state.json", {**_lane_state(ldir), "runner_pid": None, "runner_rc": rc})
    if rc < 0:
        if plan["arm"] == "control":
            quarantine_control_render(run, lane)
        run.say(f"[{lane['lane']}] runner ended by signal {-rc}: INTERRUPTED", ldir / "lane.log")
        return INTERRUPTED_EXIT
    if rc != 0:
        run.alert(f"[{lane['lane']}] the runner exited rc={rc} by itself; see {ldir / 'run.log'}")
        return 2
    return 0


def run_warm_lane(run: Run, plan: dict, lane: dict) -> int:
    ldir = run.lane_dir(lane["lane"])
    master = Path(plan["frozen_master"])
    store = ldir / "working_store"
    archive = run.dir / "warm_wins_archive"
    for pass_no in range(1, ATTEMPT_CAP + 1):
        for tid in lane["tasks"]:
            tele = ldir / "results" / f"telemetry_{tid}.json"
            moved = sorted((ldir / "results" / "casualty_attempts").glob(f"telemetry_{tid}.attempt*.json"))
            if tele.exists():
                if read_json(tele).get("error_type") not in CASUALTY:
                    continue
                (ldir / "results" / "casualty_attempts").mkdir(exist_ok=True)
                tele.rename(ldir / "results" / "casualty_attempts" / f"telemetry_{tid}.attempt{len(moved) + 1}.json")
                moved.append(None)
            if len(moved) >= pass_no:     # this pass already ran (or set aside) an attempt of this task
                continue
            if len(moved) >= ATTEMPT_CAP:
                continue
            while run.freeze.exists():
                time.sleep(15)
            digest, count = _sha256_tree(master)
            if digest != plan["master_sha256"]:
                run.alert(f"[{lane['lane']}] PROTOCOL BREACH: the frozen master hashes {digest} ({count} entries); "
                          "lane aborted")
                return 2
            shutil.rmtree(store, ignore_errors=True)
            shutil.copytree(master, store)
            subprocess.run(["chmod", "-R", "u+w", str(store)], check=True)
            cfg = ldir / "cfg" / str(tid)
            shutil.rmtree(cfg, ignore_errors=True)
            cfg.mkdir(parents=True)
            (cfg / f"{tid}.json").symlink_to(Path(plan["webarena_root"]) / "config_files" / f"{tid}.json")
            run.say(f"[{lane['lane']}] EPISODE {tid} pass {pass_no}: store restored ({count} entries)", ldir / "lane.log")
            proc = start_runner(run, plan, lane, runner_argv(plan, lane, ldir, config_dir=cfg, store=store), str(tid))
            rc = proc.wait()
            write_json(ldir / "lane_state.json", {**_lane_state(ldir), "runner_pid": None, "runner_rc": rc})
            delta = 0
            for path in (store / "webarena").glob("*.json"):
                twin = master / "webarena" / path.name
                if not twin.exists() or twin.read_bytes() != path.read_bytes():
                    (archive / str(tid)).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, archive / str(tid) / path.name)
                    delta += 1
            run.say(f"[{lane['lane']}] EPISODE {tid} rc={rc} store_delta={delta}", ldir / "lane.log")
            if _sha256_tree(master)[0] != plan["master_sha256"]:
                run.alert(f"[{lane['lane']}] PROTOCOL BREACH: the frozen master changed during task {tid}; lane aborted")
                return 2
            if rc < 0:
                if tele.exists():
                    (ldir / "results" / "casualty_attempts").mkdir(exist_ok=True)
                    tele.rename(ldir / "results" / "casualty_attempts" / f"telemetry_{tid}.interrupted.json")
                run.say(f"[{lane['lane']}] task {tid}: runner ended by signal {-rc}: INTERRUPTED", ldir / "lane.log")
                return INTERRUPTED_EXIT
            if rc != 0 or not tele.exists():
                run.alert(f"[{lane['lane']}] task {tid}: the runner exited rc={rc} without telemetry; lane aborted")
                return 2
    shutil.rmtree(store, ignore_errors=True)
    return 0


def run_lane(run: Run, name: str) -> int:
    plan = run.plan()
    lane = next(l for l in plan["lanes"] if l["lane"] == name)
    ldir = run.lane_dir(name)
    write_json(ldir / "lane_state.json", {"lane": name, "pid": os.getpid(), "status": "running",
                                          "started_utc": utc(), "runner_pid": None})
    run.say(f"[{name}] LANE START pid {os.getpid()}: {len(lane['tasks'])} tasks", ldir / "lane.log")
    code = (run_warm_lane if plan["arm"] == "warm" else run_continuous_lane)(run, plan, lane)
    status = {0: "done", INTERRUPTED_EXIT: "interrupted"}.get(code, "failed")
    write_json(ldir / "lane_state.json", {**_lane_state(ldir), "status": status, "ended_utc": utc()})
    run.say(f"[{name}] LANE {status.upper()}", ldir / "lane.log")
    return code


# --------------------------------------------------------------------------- supervisor

def _spawn(run: Run, args: list[str], role: str, lane: str = "-") -> subprocess.Popen:
    with open(run.log, "ab") as out:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *args], stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    run.record_pid(role, proc.pid, lane)
    return proc


def ensure_heartbeat(run: Run) -> None:
    """The heartbeat alone scans the tripwires (the 402 freeze among them): a gone heartbeat is alerted and
    respawned while the run is unfinished."""
    if run.alive("heartbeat") or (run.dir / "RUN_DONE").exists() or (run.dir / "RUN_STOPPED").exists():
        return
    proc = _spawn(run, ["heartbeat", "--run_dir", str(run.dir)], "heartbeat")
    run.alert(f"HEARTBEAT GONE: respawned as pid {proc.pid} (the tripwires and the 402 freeze depend on it)")


def refresh_auth(run: Run, plan: dict, phase: int, tasks: list[int]) -> None:
    """Re-render .auth in place and check it before lanes start (SITE LOGINS in the module docstring)."""
    out = run.dir / "auth_refresh"
    out.mkdir(exist_ok=True)
    env_file = str(Path(plan["code_root"]) / ".env")
    while True:
        while run.freeze.exists():
            ensure_heartbeat(run)
            time.sleep(15)
        if run.alive("runner"):   # no episode may read a half-written state file
            run.alert(f"PHASE {phase}: runner(s) {[p['pid'] for p in run.alive('runner')]} alive; the auth refresh waits")
            while run.alive("runner"):
                ensure_heartbeat(run)
                time.sleep(15)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        login_log, check_log = out / f"phase{phase}_{stamp}_auto_login.log", out / f"phase{phase}_{stamp}_check.log"
        login_rc = _auto_login(run, plan, out, env_file, login_log)
        if login_rc is None:
            why = f"auto_login timed out and its process group was killed (see {login_log.name})"
        else:
            with open(check_log, "wb") as f:
                check = subprocess.Popen([plan["webarena_python"], str(AUTH_CHECK), "--env", env_file, "--webarena_root",
                                          plan["webarena_root"], "--task_ids", ",".join(map(str, tasks))],
                                         cwd=plan["webarena_root"], stdin=subprocess.DEVNULL, stdout=f,
                                         stderr=subprocess.STDOUT)
            run.record_pid("auth_check", check.pid)
            check_rc = check.wait()
            last = (check_log.read_text(errors="replace").strip().splitlines() or ["(no output)"])[-1]
            append_line(run.dir / "auth_refresh.log", f"{utc()} phase {phase}: auto_login rc={login_rc}; check "
                        f"{'PASS' if check_rc == 0 else 'FAIL'} rc={check_rc}: {last} [{check_log.name}]")
            if check_rc == 0:
                run.say(f"PHASE {phase}: .auth re-rendered and checked: {last}")
                return
            why = f"a required .auth state does not log in after auto_login ({last}; see {check_log.name})"
        write_json(run.freeze, {"utc": utc(), "reason": f"phase {phase} auth refresh: {why}", "stopped": []})
        run.alert(f"PHASE {phase}: {why}; run FROZEN. Diagnose, then remove {run.freeze}: the refresh runs again")


def supervise(run: Run) -> int:
    """Phase by phase: refresh the logins, start one worker per lane, poll them. A lane INTERRUPTED by
    ``resume --long`` (the RELAUNCH_AUTH marker) is relaunched once every worker of the phase has ended, after a new
    login refresh; one interrupted by anything else is relaunched at once (its site sessions were in use moments
    ago). A failed lane stops the run once the phase's other lanes end."""
    plan = run.plan()
    run.say(f"SUPERVISOR START pid {os.getpid()}")
    live: dict[str, subprocess.Popen] = {}

    def stop(signum, _frame):
        for proc in live.values():
            if proc.poll() is None:
                proc.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    _spawn(run, ["heartbeat", "--run_dir", str(run.dir)], "heartbeat")
    marker = run.dir / RELAUNCH_AUTH
    for phase in sorted({l["phase"] for l in run.plan()["lanes"]}):
        plan = run.plan()
        lanes = {l["lane"]: l for l in plan["lanes"] if l["phase"] == phase}
        todo = [n for n in lanes if _lane_state(run.lane_dir(n)).get("status") != "done"]
        if not todo:
            continue
        no_progress = {name: 0 for name in lanes}
        last_scored = {name: lane_progress(run, plan, lanes[name])["scored"] for name in lanes}
        failed: dict[str, int] = {}
        waiting: list[str] = list(todo)      # lanes to (re)start
        needs_auth = True
        while waiting or live:
            ensure_heartbeat(run)
            if waiting and not failed:
                if needs_auth or marker.exists():
                    if live:                  # a login refresh waits for every worker of the phase to end
                        pass
                    else:
                        run.say(f"PHASE {phase}: (re)starting {waiting} after a login refresh")
                        refresh_auth(run, plan, phase, [t for n in waiting for t in lanes[n]["tasks"]])
                        marker.unlink(missing_ok=True)
                        needs_auth = False
                        for index, name in enumerate(waiting):
                            if index:
                                time.sleep(plan["lane_stagger_seconds"])
                            live[name] = _spawn(run, ["lane", "--run_dir", str(run.dir), "--lane", name], "worker", name)
                        waiting = []
                elif not run.freeze.exists():
                    for name in waiting:
                        run.say(f"PHASE {phase}: [{name}] relaunched at once (interrupted, no long freeze)")
                        live[name] = _spawn(run, ["lane", "--run_dir", str(run.dir), "--lane", name], "worker", name)
                    waiting = []
            for name, proc in list(live.items()):
                code = proc.poll()
                if code is None:
                    continue
                del live[name]
                if code == 0:
                    continue
                if code != INTERRUPTED_EXIT:
                    failed[name] = code
                    run.alert(f"PHASE {phase}: [{name}] failed (exit {code}); no lane starts again in this run")
                    continue
                scored = lane_progress(run, run.plan(), lanes[name])["scored"]
                no_progress[name] = 0 if scored > last_scored[name] else no_progress[name] + 1
                last_scored[name] = scored
                if no_progress[name] >= MAX_RELAUNCHES_WITHOUT_PROGRESS:
                    failed[name] = code
                    run.alert(f"[{name}] interrupted {no_progress[name]} times without a newly scored task; it is not "
                              "relaunched and the run stops")
                    continue
                waiting.append(name)
                run.say(f"PHASE {phase}: [{name}] INTERRUPTED; relaunch pending")
            time.sleep(min(SUPERVISOR_POLL_S, plan["heartbeat_seconds"]))
        if failed:
            (run.dir / "RUN_STOPPED").write_text(f"{utc()} phase {phase} lanes {failed}\n")
            run.alert(f"PHASE {phase}: lanes {failed} did not finish; the run STOPS before any later phase")
            return 1
    (run.dir / "RUN_DONE").write_text(f"{utc()}\n")
    run.say("RUN DONE")
    return 0


# --------------------------------------------------------------------------- freeze / resume

def freeze(run: Run, reason: str) -> list[int]:
    """Write FREEZE and SIGSTOP every alive recorded runner's process group. Idempotent: a second call while frozen
    stops nothing new and keeps the first record."""
    if run.freeze.exists():
        return []
    stopped = []
    for row in run.alive("runner"):
        try:
            os.killpg(row["pid"], signal.SIGSTOP)
            stopped.append(row["pid"])
        except ProcessLookupError:
            continue
    write_json(run.freeze, {"utc": utc(), "epoch": time.time(), "reason": reason, "stopped": stopped})
    run.alert(f"FROZEN: {reason}; runners {stopped} stopped (SIGSTOP). `resume` after the cause is cleared")
    return stopped


def resume(run: Run, force_long: bool = False) -> int:
    if not run.freeze.exists():
        print("not frozen")
        return 1
    record = read_json(run.freeze)
    held = time.time() - float(record.get("epoch") or time.time())
    stopped = [pid for pid in record.get("stopped", []) if pid_alive(pid, run.dir)]
    if held < LONG_FREEZE_S and not force_long:
        for pid in stopped:
            os.killpg(pid, signal.SIGCONT)
        run.freeze.unlink()
        run.alert(f"RESUMED after {held / 60:.1f} min (< {LONG_FREEZE_S // 60}): runners {stopped} continued (SIGCONT)")
        return 0
    (run.dir / RELAUNCH_AUTH).write_text(f"{utc()} resume after {held / 60:.1f} min\n")
    for pid in stopped:
        for sig in (signal.SIGTERM, signal.SIGCONT):
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                pass
    deadline = time.time() + KILL_GRACE_S
    while time.time() < deadline and any(_group_alive(pid) for pid in stopped):
        time.sleep(1)
    for pid in stopped:
        if _group_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    run.freeze.unlink()
    run.alert(f"RESUMED after {held / 60:.1f} min: runners {stopped} ended (their sessions may have died while "
              "frozen); the supervisor re-renders .auth and relaunches the interrupted lanes")
    return 0


# --------------------------------------------------------------------------- heartbeat and tripwires

def scan_tripwires(run: Run, plan: dict, state: dict) -> None:
    rules = plan["tripwires"]
    counts = state.setdefault("counts", {})
    seen = state.setdefault("values", {})
    offsets = state.setdefault("offsets", {})
    for lane in plan["lanes"]:
        log = run.lane_dir(lane["lane"]) / "run.log"
        if not log.exists():
            continue
        start = offsets.get(str(log), 0)
        size = log.stat().st_size
        if size < start:
            start = 0
        with open(log, "rb") as f:
            f.seek(start)
            chunk = f.read(size - start)
        complete = chunk.rfind(b"\n") + 1      # a live log's last line without a newline is read next time
        offsets[str(log)] = start + complete
        for line in chunk[:complete].decode("utf-8", errors="replace").splitlines():
            for rule in rules:
                match = re.search(rule["regex"], line)
                if not match:
                    continue
                name = rule["name"]
                if rule["action"] == "freeze":
                    counts[name] = counts.get(name, 0) + 1
                    freeze(run, f"tripwire {name} in [{lane['lane']}]: {line.strip()[:200]}")
                elif rule["action"] == "alert":
                    counts[name] = counts.get(name, 0) + 1
                    n = counts[name]
                    if n & (n - 1) == 0:
                        run.alert(f"TRIPWIRE {name} (occurrence {n}) in [{lane['lane']}]: {line.strip()[:240]}")
                else:
                    value = match.group(1)
                    baseline = rule.get("baseline") or seen.setdefault(f"{name}:baseline", value)
                    known = seen.setdefault(name, [])
                    if value != baseline and value not in known:
                        known.append(value)
                        run.alert(f"TRIPWIRE {name}: NEW value {value} (baseline {baseline}) in [{lane['lane']}]: "
                                  f"{line.strip()[:200]}")


def _beat(run: Run, plan: dict, state: dict, memory: dict) -> None:
    """One heartbeat: the lanes' liveness and progress, the stall / worker / homepage / casualty alerts,
    heartbeat.json and heartbeat.log."""
    stall_s = plan["stall_minutes"] * 60
    beat = {"utc": utc(), "frozen": read_json(run.freeze) if run.freeze.exists() else None, "lanes": {},
            "tripwire_counts": state.get("counts", {})}
    if plan.get("homepage_url"):
        beat["homepage"] = homepage_status(plan["homepage_url"])
        if beat["homepage"]["ok"] != memory["homepage_ok"]:
            run.alert(f"HOMEPAGE {'UP' if beat['homepage']['ok'] else 'DOWN'}: {beat['homepage']}")
            memory["homepage_ok"] = beat["homepage"]["ok"]
    casualties = 0
    for lane in plan["lanes"]:
        ldir = run.lane_dir(lane["lane"])
        st = _lane_state(ldir)
        progress = lane_progress(run, plan, lane)
        casualties += len(progress["casualties"])
        log = ldir / "run.log"
        idle = int(time.time() - log.stat().st_mtime) if log.exists() else None
        runner_alive = bool(st.get("runner_pid")) and pid_alive(st["runner_pid"], run.dir)
        row = {"status": st.get("status", "not started"), "runner_pid": st.get("runner_pid"),
               "runner_alive": runner_alive, "task": st.get("current_task"), "idle_s": idle,
               **{k: progress[k] for k in ("planned", "scored", "passed")}, "casualties": progress["casualties"]}
        if row["status"] == "running" and not pid_alive(st.get("pid", 0), run.dir):
            row["alert"] = "WORKER GONE while its lane runs"
        elif runner_alive and not run.freeze.exists() and idle is not None and idle > stall_s:
            row["alert"] = f"STALL: the lane log has not grown for {idle // 60} min"
        if "alert" in row and not st.get("alerted") == row["alert"]:
            run.alert(f"[{lane['lane']}] {row['alert']} (runner {row['runner_pid']})")
            write_json(ldir / "lane_state.json", {**st, "alerted": row["alert"]})
        beat["lanes"][lane["lane"]] = row
        append_line(run.dir / "heartbeat.log",
                    f"{beat['utc']} {lane['lane']:15} {row['status']:11} scored {row['scored']}/{row['planned']} "
                    f"pass {row['passed']} casualties {len(row['casualties'])} runner {row['runner_pid']}"
                    f"({'up' if runner_alive else '-'}) idle {idle}s {row.get('alert', '')}")
    beat["casualties"] = casualties
    marker = run.dir / "CASUALTY_THRESHOLD_ALERTED"
    if casualties >= CASUALTY_ALERT and not marker.exists():
        marker.write_text(f"{utc()} {casualties}\n")
        run.alert(f"THRESHOLD: {casualties} casualties (unscored tasks) in this run; diagnose now, nothing stops")
    write_json(run.dir / "heartbeat.json", beat)


def heartbeat(run: Run) -> int:
    """Tripwires every poll, a beat every --heartbeat_seconds. Neither ever ends the heartbeat: an exception in
    either is logged and alerted (once per distinct message) and the loop goes on — the runner writes
    its telemetry in place, so a read can land mid-write."""
    plan = run.plan()
    every = plan["heartbeat_seconds"]
    state_path = run.dir / "tripwire_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    if not state:   # a fresh state over existing logs starts at their ends: old lines never fire
        state["offsets"] = {str(run.lane_dir(l["lane"]) / "run.log"): (run.lane_dir(l["lane"]) / "run.log").stat().st_size
                            for l in plan["lanes"] if (run.lane_dir(l["lane"]) / "run.log").exists()}
    run.say(f"HEARTBEAT START pid {os.getpid()} (every {every}s, stall {plan['stall_minutes']} min)",
            run.dir / "heartbeat.log")
    memory = {"homepage_ok": True, "last_error": None}

    def guarded(what, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — a beat never ends the heartbeat
            message = f"{what}: {type(exc).__name__}: {exc}"
            append_line(run.dir / "heartbeat.log", f"{utc()} HEARTBEAT ERROR {message}")
            if message != memory["last_error"]:
                run.alert(f"HEARTBEAT ERROR {message} (the heartbeat goes on)")
                memory["last_error"] = message

    def scan() -> None:
        scan_tripwires(run, plan, state)
        write_json(state_path, state)

    nothing_alive = 0
    while True:
        try:
            plan = run.plan()
        except (OSError, json.JSONDecodeError):
            pass                       # add_lane rewrites plan.json atomically; keep the last good one
        poll = min(TRIPWIRE_POLL_S, every)
        for _ in range(max(1, every // poll)):
            guarded("tripwires", scan)
            if (run.dir / "RUN_DONE").exists() or (run.dir / "RUN_STOPPED").exists():
                break
            time.sleep(poll)
        guarded("beat", lambda: _beat(run, plan, state, memory))
        if (run.dir / "RUN_DONE").exists() or (run.dir / "RUN_STOPPED").exists():
            run.say("HEARTBEAT DONE (run finished)", run.dir / "heartbeat.log")
            return 0
        nothing_alive = 0 if run.alive("supervisor") or run.alive("worker") else nothing_alive + 1
        if nothing_alive >= 2:
            run.alert("HEARTBEAT: no supervisor or worker of this run is alive, and the run is not finished")
            return 1


# --------------------------------------------------------------------------- launch / status / casualties

def launch(args: argparse.Namespace) -> int:
    run = Run(Path(args.run_dir))
    plan = run.plan()
    check_code_root(Path(plan["code_root"]))
    _settle_finished_run(run, plan)
    alive = run.alive()
    if alive:
        refuse(f"processes this run recorded are still alive: {alive}")
    if run.freeze.exists():
        refuse(f"{run.freeze} exists: {read_json(run.freeze)}; resolve it first")
    if (run.dir / "RUN_DONE").exists():
        refuse("RUN_DONE: every lane is done (add_lane appends a re-run lane)")
    stale = run.dir / "RUN_STOPPED"
    if stale.exists():
        stale.rename(run.dir / f"RUN_STOPPED.{utc()}")
    for lane in plan["lanes"]:
        st = _lane_state(run.lane_dir(lane["lane"]))
        if st.get("status") in ("running", "failed", "interrupted"):
            write_json(run.lane_dir(lane["lane"]) / "lane_state.json", {**st, "status": f"relaunch after {st['status']}"})
    sup = _spawn(run, ["supervise", "--run_dir", str(run.dir)], "supervisor")
    print(f"supervisor pid {sup.pid} (own session); watch {run.dir}/heartbeat.log and ALERTS.log")
    return 0


def status(run: Run) -> int:
    plan = run.plan()
    print(f"{plan['arm']} @ {plan['code_commit'][:12]}  frozen: {read_json(run.freeze) if run.freeze.exists() else 'no'}")
    planned, scored, passed = set(), set(), set()
    for lane in plan["lanes"]:
        st = _lane_state(run.lane_dir(lane["lane"]))
        progress = lane_progress(run, plan, lane)
        planned |= set(lane["tasks"])
        scored |= set(progress["scored_ids"])
        passed |= set(progress["passed_ids"])
        print(f"  {lane['lane']:16} phase {lane['phase']} {st.get('status', 'not started'):24} scored "
              f"{progress['scored']}/{progress['planned']} pass {progress['passed']} casualties {progress['casualties']}")
    # unique tasks: a re-run lane's score is the task's first and only one (the casualty policy re-runs never-scored
    # tasks only), so a task counts once whichever lane scored it
    print(f"  total scored {len(scored)}/{len(planned)} pass {len(passed)}")
    return 0


def casualties(run: Run) -> int:
    plan = run.plan()
    progress = {lane["lane"]: lane_progress(run, plan, lane) for lane in plan["lanes"]}
    scored = {t for p in progress.values() for t in p["scored_ids"]}
    out = {name: [t for t in p["unscored"] if t not in scored] for name, p in progress.items()}
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--arm", required=True, choices=ARMS)
    pr.add_argument("--run_dir", required=True)
    pr.add_argument("--code_root", required=True)
    pr.add_argument("--webarena_python", required=True)
    pr.add_argument("--webarena_root", default=str(_REPO_ROOT.parent / "WebArena"))
    pr.add_argument("--partitions_dir", required=True)
    pr.add_argument("--argv_json", required=True, help="a JSON object of named runner argvs")
    pr.add_argument("--argv_key", required=True, help="which argv of --argv_json this arm runs")
    pr.add_argument("--tripwires_json", required=True)
    pr.add_argument("--support_set_root", default=None)
    pr.add_argument("--frozen_master", default=None)
    pr.add_argument("--master_sha256", default=None)
    pr.add_argument("--lanes", default=None, help="comma-separated lanes (default: every partition)")
    pr.add_argument("--task_ids", default=None, help="only these tasks (a smoke)")
    pr.add_argument("--env_set", action="append", default=[], help="NAME=VALUE for the runners; repeat")
    pr.add_argument("--homepage_url", default="http://127.0.0.1:4399/")
    pr.add_argument("--heartbeat_seconds", type=int, default=300)
    pr.add_argument("--stall_minutes", type=int, default=45)
    pr.add_argument("--lane_stagger_seconds", type=int, default=10,
                    help="between lane starts: prepare() -> to_json races on WebArena's shared prompt json dir")
    al = sub.add_parser("add_lane")
    al.add_argument("--run_dir", required=True)
    al.add_argument("--lane", required=True)
    al.add_argument("--phase", type=int, required=True)
    al.add_argument("--task_ids", required=True)
    al.add_argument("--extra_argv", default="")
    fr = sub.add_parser("freeze")
    fr.add_argument("--run_dir", required=True)
    fr.add_argument("--reason", required=True)
    rs = sub.add_parser("resume")
    rs.add_argument("--run_dir", required=True)
    rs.add_argument("--long", action="store_true", help="treat the freeze as long whatever its age")
    for name in ("launch", "status", "casualties", "supervise", "heartbeat"):
        sub.add_parser(name).add_argument("--run_dir", required=True)
    ln = sub.add_parser("lane")
    ln.add_argument("--run_dir", required=True)
    ln.add_argument("--lane", required=True)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    if args.command == "add_lane":
        return add_lane(args)
    if args.command == "launch":
        return launch(args)
    run = Run(Path(args.run_dir))
    if args.command == "freeze":
        return 0 if freeze(run, args.reason) or run.freeze.exists() else 1
    if args.command == "resume":
        return resume(run, args.long)
    return {"status": status, "casualties": casualties, "supervise": supervise, "heartbeat": heartbeat}.get(
        args.command, lambda r: run_lane(r, args.lane))(run)


if __name__ == "__main__":
    sys.exit(main())
