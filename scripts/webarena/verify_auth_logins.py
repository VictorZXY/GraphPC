"""Cookie validity: every WebArena `.auth` state the given task configs use still logs in.

    python scripts/webarena/verify_auth_logins.py [--env <repo>/.env] [--webarena_root ../WebArena]
        [--task_ids 1,2,3] [--auth_dir DIR]

Exit 0 = every state file x every site it covers answers 200 on that site's logged-in page with
no redirect; 1 = any missing file, redirect (a login redirect is the stale-cookie signature, e.g.
shopping_admin 302 -> admin/auth/login), non-200 or error.

Why: the GraphPC runner reads the `.auth` files present at an episode's start and never re-logs
in, and the sites expire idle sessions server-side (45 min to 2 h). The lane
driver (run_lanes.py) runs this at every phase start, right after re-rendering `.auth` in
place with WebArena's own auto_login.py, and does not start the phase on a red. The task list
is an argument.

What it asks (read-only GETs; no redirect is followed): the page WebArena's own auto_login.py
uses to decide a cookie has expired (its URLS / KEYWORDS), with the cookies the state file holds
for that host, sent as a browser would (cookies are host-scoped, not port-scoped). Where
auto_login.py checks a keyword (shopping_admin "Dashboard", reddit "Delete"), the body must
carry it too, and no 200 may carry a login form. One deviation: shopping_admin is asked at
Magento's canonical dashboard route, /admin/dashboard/. auto_login's URL, /admin/dashboard, answers
a LOGGED-IN admin with a 404 whose menu still says "Dashboard" and a logged-out one with the login
form at 200, so neither its status nor its keyword separates the two (checked with a fresh cookie
and with none). Required files = every `storage_state` named by the configs of --task_ids
(default: every numbered task config under <webarena_root>/config_files).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

_REPO_ROOT = Path(__file__).resolve().parents[2]
REDDIT_USER = "MarvelsGrantMan136"   # browser_env/env_config.py ACCOUNTS["reddit"]


def load_env(path: Path) -> dict[str, str]:
    # python-dotenv, as the runners' load_dotenv: it expands the ${WEBARENA_PUBLIC_DNS} references.
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(path).items() if v is not None}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def probe(url: str, cookies: list[dict], keyword: str) -> tuple[bool, str]:
    parts = urlsplit(url)
    host, now = parts.hostname or "", time.time()
    jar = []
    for c in cookies:
        domain = c.get("domain", "").lstrip(".")
        if not (host == domain or host.endswith("." + domain)):
            continue
        if not parts.path.startswith(c.get("path") or "/"):
            continue
        if c.get("expires", -1) not in (-1, None) and 0 < c["expires"] < now:
            continue
        jar.append(f"{c['name']}={c['value']}")
    request = urllib.request.Request(url, headers={"Cookie": "; ".join(jar), "User-Agent": "Mozilla/5.0"})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            # Magento answers a stale admin cookie with its login form at 200, on the same URL.
            if "login[username]" in body or 'name="_username"' in body:
                return False, f"200 but a login form was served ({len(jar)} cookies sent)"
            if keyword and keyword not in body:
                return False, f"200 but keyword {keyword!r} absent ({len(jar)} cookies sent)"
            return True, f"200 ({len(jar)} cookies sent)"
    except urllib.error.HTTPError as exc:
        location = exc.headers.get("Location", "")
        return False, f"{exc.code}{' -> ' + location if location else ''} ({len(jar)} cookies sent)"
    except Exception as exc:  # noqa: BLE001 — any failure is a red, named
        return False, f"error {type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", type=Path, default=_REPO_ROOT / ".env")
    p.add_argument("--webarena_root", type=Path, default=_REPO_ROOT.parent / "WebArena")
    p.add_argument("--task_ids", default=None,
                   help="Comma-separated task ids whose configs name the required state files (default: every "
                        "numbered config under <webarena_root>/config_files).")
    p.add_argument("--auth_dir", type=Path, default=None,
                   help="Probe this folder of state files instead of <webarena_root>/.auth (diagnosis).")
    args = p.parse_args()
    auth_dir = args.auth_dir or args.webarena_root / ".auth"

    env = {**load_env(args.env), **{k: v for k, v in os.environ.items() if k in
                                     ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB")}}
    # auto_login.py's URLS / KEYWORDS (browser_env/auto_login.py:24-33), shopping_admin excepted
    pages = {
        "gitlab": (f"{env['GITLAB']}/-/profile", ""),
        "shopping": (f"{env['SHOPPING']}/wishlist/", ""),
        "shopping_admin": (f"{env['SHOPPING_ADMIN']}/admin/dashboard/", "Dashboard"),   # see docstring
        "reddit": (f"{env['REDDIT']}/user/{REDDIT_USER}/account", "Delete"),
    }
    configs = args.webarena_root / "config_files"
    if args.task_ids:
        task_ids = [int(t) for t in args.task_ids.split(",")]
    else:
        task_ids = sorted(int(f.stem) for f in configs.glob("*.json") if f.stem.isdigit())
    required = set()
    for tid in task_ids:
        state = json.loads((configs / f"{tid}.json").read_text()).get("storage_state")
        if state:
            required.add(Path(state).name)

    failures = 0
    print(f"cookie check {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: "
          f"{len(required)} state file(s) required by {len(task_ids)} task config(s)")
    for name in sorted(required):
        path = auth_dir / name
        if not path.is_file():
            print(f"  FAIL  {name}: missing")
            failures += 1
            continue
        rendered = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        cookies = json.loads(path.read_text()).get("cookies", [])
        for site in name.rsplit("_", 1)[0].split("."):
            url, keyword = pages[site]
            ok, detail = probe(url, cookies, keyword)
            failures += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  {name:<36} {site:<15} {detail}  [rendered {rendered}]")
    print("PASS: every required .auth state logs in" if not failures else
          f"FAIL: {failures} state/site pair(s) do not log in — re-run auto_login (after any reset) and re-check")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
