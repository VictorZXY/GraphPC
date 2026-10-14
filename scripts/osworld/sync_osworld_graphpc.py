from __future__ import annotations

import argparse
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.osworld.sync import (
    DEFAULT_OSWORLD_ROOT,
    missing_exclude_entries,
    sync_to_osworld,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the minimal GraphPC OSWorld package and runner into an OSWorld checkout.",
    )
    parser.add_argument(
        "--osworld_root",
        type=Path,
        default=DEFAULT_OSWORLD_ROOT,
        help="Path to the target OSWorld repository checkout.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously synced GraphPC files before copying the new ones.",
    )
    parser.add_argument(
        "--clean_only",
        action="store_true",
        help="Remove previously synced GraphPC files and stop without copying.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the files that would be removed or copied without modifying OSWorld.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    touched_paths = sync_to_osworld(
        args.osworld_root,
        clean=args.clean,
        clean_only=args.clean_only,
        dry_run=args.dry_run,
    )

    action = "Would touch" if args.dry_run else "Touched"
    for path in touched_paths:
        print(f"{action}: {path}")
    print(f"Total {action.lower()} paths: {len(touched_paths)}")

    if not args.clean_only:
        # The snapshot is untracked by design; .git/info/exclude is what keeps it
        # out of `git status`, and the launch preflight FAILs on any untracked
        # path it does not cover. That list is hand-maintained, so it drifts every
        # time this sync grows a file — report it here instead of at the gate.
        missing = missing_exclude_entries(args.osworld_root)
        if missing:
            print(
                f"\nWARNING: {len(missing)} synced path(s) are not covered by "
                f"{args.osworld_root}/.git/info/exclude.\n"
                "The launch preflight will FAIL on them as untracked files. Append:\n"
            )
            for entry in missing:
                print(f"    {entry}")
            print("\n    (this script never writes into the checkout; do it by hand, "
                  "on the staging clone AND on the run host)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
