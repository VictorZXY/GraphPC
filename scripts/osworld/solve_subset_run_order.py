"""Derive the pre-registered run order for the OSWorld 2.0 subset: separate every twin pair.

Both arms consume the SAME task-list manifest, and the V2 runner preserves its list order
(``distribute_tasks`` flattens the manifest into a FIFO the workers pull from), so the manifest's
order IS the run order for both arms. This script derives that order from the pre-registration and
is the only place it is authored -- the frozen task-list manifest carries it and
``launch_preflight.py`` asserts the three agree, so a hand-edited order cannot reach a launch.

WHY an order at all. A twin pair is the cell's corroborating transfer evidence: the SECOND member is
only informative if the FIRST could plausibly have seeded the store before it started. Adjacent
members break that both ways -- with N workers pulling a shared queue, two tasks a few slots apart
run concurrently, so the later one is neither cleanly cold nor cleanly warm. Identity (id-sorted)
order is the worst case: 042-044 and 103-104 sit one slot apart.

THE RULE (it reads nothing but the pair list):

    maximise the MINIMUM queue-slot gap over all pre-registered twin pairs;
    among the arrangements attaining it, take the lexicographically first one under a fixed
    enumeration (paired tasks in ascending id to ascending positions, then the unpaired tasks
    in ascending id into the remaining slots).

The tie-break exists only to make the answer unique and reproducible -- it is not a second
objective. No outcome data, no similarity score and no task content enters this; only the pair
list and the task count, both pre-registered.

    python scripts/osworld/solve_subset_run_order.py                 # derive and print
    python scripts/osworld/solve_subset_run_order.py --check         # exit 1 unless the prereg matches
"""
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREREG_PATH = REPO_ROOT / "data" / "osworld_v2" / "subset_prereg.json"


def _feasible(paired: list[str], pairs: list[tuple[str, str]], n: int, gap: int) -> Optional[dict[str, int]]:
    """Lexicographically first placement of the paired tasks into 0..n-1 with every pair >= gap
    apart, or None (exact -- an exhaustive search, not a heuristic).

    Variables are taken in ascending task id and values in ascending position, so the first
    solution found is the lexicographically first one; forward checking (prune each unplaced
    partner's domain after every assignment, fail on an empty one) only removes branches that
    provably hold no solution, so it changes the cost and not the answer. Without it, proving
    the high gaps infeasible costs minutes rather than milliseconds.
    """
    adjacency: dict[str, list[str]] = {task: [] for task in paired}
    for a, b in pairs:
        adjacency[a].append(b)
        adjacency[b].append(a)
    domains = {task: set(range(n)) for task in paired}
    placed: dict[str, int] = {}

    def place(index: int) -> bool:
        if index == len(paired):
            return True
        task = paired[index]
        for position in sorted(domains[task]):
            pruned: list[tuple[str, set[int]]] = []
            dead = False
            for other in adjacency[task] + [t for t in paired[index + 1:] if t not in adjacency[task]]:
                if other in placed:
                    continue
                keep = {p for p in domains[other] if p != position
                        and (other not in adjacency[task] or abs(p - position) >= gap)}
                if keep == domains[other]:
                    continue
                pruned.append((other, domains[other]))
                domains[other] = keep
                if not keep:
                    dead = True
                    break
            if not dead:
                placed[task] = position
                if place(index + 1):
                    return True
                del placed[task]
            for other, previous in pruned:
                domains[other] = previous
        return False

    return dict(placed) if place(0) else None


def _two_colour(paired: list[str], pairs: list[tuple[str, str]]) -> Optional[list[tuple[int, int]]]:
    """Per connected component, the sizes of its two colour classes -- or None if the pair graph
    has an odd cycle (no 2-colouring). Twin pairs form paths and isolated edges in practice, so
    this normally succeeds."""
    adjacency: dict[str, list[str]] = {task: [] for task in paired}
    for a, b in pairs:
        adjacency[a].append(b)
        adjacency[b].append(a)
    colour: dict[str, int] = {}
    sizes: list[tuple[int, int]] = []
    for start in paired:
        if start in colour:
            continue
        colour[start], stack, counts = 0, [start], [1, 0]
        while stack:
            task = stack.pop()
            for other in adjacency[task]:
                if other not in colour:
                    colour[other] = 1 - colour[task]
                    counts[colour[other]] += 1
                    stack.append(other)
                elif colour[other] == colour[task]:
                    return None
        sizes.append((counts[0], counts[1]))
    return sizes


def _gap_upper_bound(paired: list[str], pairs: list[tuple[str, str]], n: int) -> int:
    """Largest gap not refuted by counting alone -- where the search starts, so the answer it
    returns is a PROVEN optimum rather than the best a search happened to find.

    For gap >= ceil(n/2) the low band [0, n-1-gap] and the high band [gap, n-1] cover every
    position, neither band can hold both ends of a pair, and a position in neither band can hold
    no paired task at all. So every pair crosses the two bands: the bands are a 2-colouring of the
    pair graph, and each holds at most n-gap tasks. Minimising the larger colour class over the
    free per-component choice of which side is which (a subset sum over the components) therefore
    caps the gap at ``n - that minimum``. Below ceil(n/2) the argument lapses, so the bound is
    never taken lower than that.
    """
    sizes = _two_colour(paired, pairs)
    floor = (n + 1) // 2
    if sizes is None:
        return n - 1
    reachable = {0}
    for first, second in sizes:
        reachable = {total + side for total in reachable for side in (first, second)}
    total_paired = len(paired)
    smallest_max = min(max(low, total_paired - low) for low in reachable)
    return max(min(n - 1, n - smallest_max), floor)


def run_order_gaps(order: list[str], twin_pairs: list[tuple[str, str]]) -> dict:
    """The queue-slot gaps a GIVEN order achieves: every pair's, and the minimum.

    Separate from the solver above, which derives the OPTIMAL order at real cost.
    This measures an order someone hands us, which is what a checker needs — the
    launch preflight computes the pre-registration's declared order through this
    rather than echoing the ``run_order_min_gap`` the record states about itself.
    One implementation, so a checker and the solver cannot disagree about what a
    gap is."""
    index = {str(task): position for position, task in enumerate(order)}
    pairs = [(str(a), str(b)) for a, b in twin_pairs]
    unknown = sorted({t for pair in pairs for t in pair} - set(index))
    if unknown:
        raise ValueError(f"twin_pairs name tasks outside the order: {unknown}")
    gaps = {f"{a}-{b}": abs(index[a] - index[b]) for a, b in pairs}
    return {"run_order_pair_gaps": gaps, "run_order_min_gap": min(gaps.values()) if gaps else 0}


def solve_run_order(subset_ids: list[str], twin_pairs: list[tuple[str, str]]) -> dict:
    """The pre-registered order, its minimum gap, and the gap of every pair."""
    ids = sorted(str(i) for i in subset_ids)
    pairs = [(str(a), str(b)) for a, b in twin_pairs]
    unknown = sorted({t for pair in pairs for t in pair} - set(ids))
    if unknown:
        raise ValueError(f"twin_pairs name tasks outside subset_ids: {unknown}")
    n = len(ids)
    paired = sorted({t for pair in pairs for t in pair})
    best: Optional[dict[str, int]] = None
    best_gap = 0
    for gap in range(_gap_upper_bound(paired, pairs, n), 0, -1):  # descending: first feasible = maximum
        placement = _feasible(paired, pairs, n, gap)
        if placement is not None:
            best, best_gap = placement, gap
            break
    if best is None:
        raise RuntimeError("no arrangement separates the twin pairs at any gap (unreachable for gap 1)")
    order: list[Optional[str]] = [None] * n
    for task, position in best.items():
        order[position] = task
    free = iter(t for t in ids if t not in best)
    order = [task if task is not None else next(free) for task in order]
    measured = run_order_gaps(order, pairs)
    if measured["run_order_min_gap"] != best_gap:
        raise RuntimeError(f"the placement claims gap {best_gap} but the order measures "
                           f"{measured['run_order_min_gap']}; the search and the measure disagree")
    return {"run_order": order, "run_order_min_gap": best_gap,
            "run_order_pair_gaps": measured["run_order_pair_gaps"]}


def render(result: dict, subset_ids: list[str], twin_pairs: list[tuple[str, str]]) -> str:
    order, index = result["run_order"], {t: i for i, t in enumerate(result["run_order"])}
    identity = {t: i for i, t in enumerate(sorted(str(i) for i in subset_ids))}
    identity_gap = min(abs(identity[str(a)] - identity[str(b)]) for a, b in twin_pairs)
    lines = [f"{len(order)} tasks, {len(twin_pairs)} twin pairs",
             f"id-sorted (identity) order: minimum gap {identity_gap}",
             f"pre-registered order:       minimum gap {result['run_order_min_gap']}", "",
             "run order (queue position 1..n):"]
    for start in range(0, len(order), 7):
        lines.append("  " + " ".join(f"{t}" for t in order[start:start + 7]))
    lines.append("")
    lines.append("per-pair gap (queue slots between the members; the earlier one runs first):")
    for a, b in twin_pairs:
        first, second = (a, b) if index[str(a)] < index[str(b)] else (b, a)
        lines.append(f"  {first} (pos {index[str(first)] + 1:>2}) -> {second} (pos {index[str(second)] + 1:>2})   "
                     f"gap {result['run_order_pair_gaps'][f'{a}-{b}']:>2}")
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prereg_path", type=Path, default=DEFAULT_PREREG_PATH)
    parser.add_argument("--check", action="store_true", help="exit 1 unless the prereg's recorded order is the derived one")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    prereg = json.loads(args.prereg_path.read_text(encoding="utf-8"))
    ids, pairs, target = prereg["subset_ids"], [tuple(p) for p in prereg["twin_pairs"]], prereg
    result = solve_run_order(ids, pairs)
    print(render(result, ids, pairs))
    if args.check:
        problems = [f"{key}: prereg {target.get(key)!r} != derived {value!r}"
                    for key, value in result.items() if target.get(key) != value]
        if problems:
            print("\nRUN ORDER CHECK FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        print("\nrun order check: the pre-registration records the derived order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
