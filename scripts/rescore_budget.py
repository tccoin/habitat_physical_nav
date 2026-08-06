"""What a geometric episode budget would have done to a run already on disk.

Every run written with the trajectory log on (the default) carries every
control tick's planar pose, so the budget it spent -- and the point at which any
cap would have cut it -- is arithmetic, not a re-run. That is the whole reason
the budget is a *currency* rather than a switch: a finished run can be re-scored
under a cap it never ran with, and different runs can be compared under a cap
none of them ran with.

    # what a 500-unit cap would have truncated in one run
    python scripts/rescore_budget.py dump/eval_ckpt5800

    # sweep several caps, and print every episode
    python scripts/rescore_budget.py dump/eval_ckpt5800 \
        --budget-units 250,500,750 --per-episode

    # several runs side by side under the same cap
    python scripts/rescore_budget.py dump/eval_a dump/eval_b --budget-units 500

Reads a run directory, a ``*_results.json``, or a ``shards/`` directory.

What it can and cannot answer is set out in
``objectnav_eval.budget.rescore_record``: the budget and the truncation point
are exact; ``oracle_success`` survives truncation with a one-chunk ambiguity;
``success``/``spl``/``soft_spl`` do not survive it at all, because they are read
at a final pose the truncated episode never reaches. Nothing here estimates
them. See docs/GEOMETRIC_BUDGET.md.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from objectnav_eval.budget import (  # noqa: E402
    DEFAULT_BUDGET_UNITS,
    DEFAULT_TIME_BACKSTOP_S,
    rescore_results,
)
from objectnav_eval.config import DEFAULT_GAP  # noqa: E402


def load_run(path: Path) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """``(results, config)`` from a run dir, a results JSON, or a shards dir."""
    path = Path(path)
    if path.is_file():
        payload = json.loads(path.read_text())
        return payload.get("results", []), payload.get("config")
    candidates = sorted(path.glob("*_results.json"))
    if not candidates and (path / "shards").is_dir():
        candidates = sorted((path / "shards").glob("*_results.json"))
    if not candidates:
        raise SystemExit(f"no *_results.json under {path}")
    results: List[Dict[str, Any]] = []
    config: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        payload = json.loads(candidate.read_text())
        results.extend(payload.get("results", []))
        config = config or payload.get("config")
    return results, config


def print_run(
    name: str, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]],
    caps: List[float], gap: int, dt: float, per_episode: bool,
) -> Dict[str, Any]:
    print(f"\n=== {name}")
    print(
        f"    {len(results)} evaluated episode(s), gap {gap}, dt {dt}"
        + (f", max_steps {config.get('max_steps')}" if config else "")
        + (f", ran under --episode-budget {config.get('episode_budget', 'steps')}"
           if config else "")
    )
    out: Dict[str, Any] = {"run": name, "gap": gap, "dt": dt, "caps": {}}
    first = True
    for cap in caps:
        scored = rescore_results(results, budget_units=cap, gap=gap)
        aggregate = scored["aggregate"]
        out["caps"][str(cap)] = aggregate
        if first:
            missing = aggregate["episodes_without_trajectory"]
            if missing:
                print(
                    f"    {missing} episode(s) have no trajectory log and cannot be "
                    "re-scored (run made with --no-trajectory-log, or before it "
                    "existed). They are excluded, not counted as zero."
                )
            if not aggregate["episodes"]:
                print("    nothing to re-score")
                return out
            print(
                f"    spent  mean {aggregate['units_mean']:7.1f}  "
                f"p50 {aggregate['units_p50']:7.1f}  "
                f"p95 {aggregate['units_p95']:7.1f}  "
                f"max {aggregate['units_max']:7.1f} units   "
                f"({aggregate['rotation_fraction_mean'] * 100:.0f}% rotation, "
                f"{aggregate['units_per_sim_second_mean']:.2f} units/s)"
            )
            print(
                f"\n    {'cap':>7}  {'truncated':>17}  {'oracle now':>10}  "
                f"{'oracle capped':>13}  {'ambiguous':>9}"
            )
            first = False
        print(
            f"    {cap:7.0f}  {aggregate['truncated']:5d}/{aggregate['episodes']:<5d}"
            f" {aggregate['truncated_fraction'] * 100:5.1f}%  "
            f"{aggregate['oracle_success']:10.3f}  "
            f"{aggregate['oracle_success_under_cap']:13.3f}  "
            f"{aggregate['oracle_ambiguous']:9d}"
        )
        if per_episode:
            for row in sorted(scored["episodes"], key=lambda r: -r["units"]):
                mark = "CUT" if row["truncated"] else "   "
                where = (
                    f" at step {row['truncated_at_step']:>4} / "
                    f"{row.get('truncated_at_sim_seconds', float('nan')):6.1f} s"
                    if row["truncated"] else ""
                )
                print(
                    f"      {mark} {str(row['uid']):<28} {row['units']:7.1f} u "
                    f"({row['rotation_fraction'] * 100:3.0f}% rot, "
                    f"{row.get('sim_seconds', float('nan')):5.1f} s, "
                    f"{row['steps']} steps){where}"
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+",
                        help="run directories, *_results.json files, or shards/ dirs")
    parser.add_argument("--budget-units", default=str(int(DEFAULT_BUDGET_UNITS)),
                        help="cap(s) to score against, comma-separated. 500 = the "
                             "classic ObjectNav 500-step cap")
    parser.add_argument("--gap", type=int, default=None,
                        help="control ticks per policy step. Read from the run's own "
                             "config when present; needed only to convert a truncation "
                             "tick into a policy step")
    parser.add_argument("--dt", type=float, default=None,
                        help="control tick in seconds, for the same reason")
    parser.add_argument("--per-episode", action="store_true",
                        help="one line per episode, biggest spender first")
    parser.add_argument("--json", default=None, help="write the aggregates here")
    args = parser.parse_args()

    caps = [float(c) for c in str(args.budget_units).split(",") if c.strip()]
    payloads = []
    for run in args.runs:
        results, config = load_run(Path(run))
        gap = args.gap or (config or {}).get("gap") or DEFAULT_GAP
        dt = args.dt or (config or {}).get("dt") or 0.05
        payloads.append(
            print_run(Path(run).name, results, config, caps, int(gap), float(dt),
                      args.per_episode)
        )

    print(
        f"\nA geometric cap always runs with a sim-time backstop "
        f"(default {DEFAULT_TIME_BACKSTOP_S:.0f} s): the budget is blind to a frozen "
        "agent and to one pushing into geometry, both of which spend nothing. "
        "This re-scoring applies the budget only, so an episode it does NOT truncate "
        "may still have been ended by that backstop in a live run."
    )
    print(
        "success / spl / soft_spl are NOT re-scored: they are read at the final pose, "
        "which truncation changes. Only oracle_success survives, and only because it "
        "latches."
    )
    if args.json:
        Path(args.json).write_text(json.dumps(payloads, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
