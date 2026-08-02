"""Compare evaluation runs on the episodes they share, with what is resolvable.

Reads the per-episode JSONL that :mod:`objectnav_eval.reporting` writes, so it
never re-runs a rollout and works equally on a finished run, a recovered partial
one, or a run someone else produced last week.

    # every run in a directory, against the first one
    python scripts/compare_policies.py --baseline dump/baseline/forward \
        --runs dump/baseline/ckpt1500 dump/baseline/ckpt2500 dump/baseline/ckpt6500

    # machine-readable, for a report
    python scripts/compare_policies.py --baseline ... --runs ... --json out.json

The comparison is *paired*: two runs over the same episode set are compared per
episode, because episode difficulty dominates the variance and pairing removes
it. Every difference carries a bootstrap CI, binary ones an exact McNemar p, and
every metric reports the minimum effect this episode count could have detected —
so a null result can be read as "no difference" or "not enough episodes", which
are different claims.
"""

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from objectnav_eval.compare import (  # noqa: E402
    BEHAVIOUR_METRICS,
    BINARY_METRICS,
    CONTINUOUS_METRICS,
    common_uids,
    compare_runs,
    load_jsonl,
    mde_paired_rate,
    mde_unpaired_rate,
    rate_table,
)


def _fmt(value, digits=3):
    if value is None:
        return "  --  "
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", required=True,
                        help="run directory or JSONL every other run is compared against")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run directories or JSONLs to compare against --baseline")
    parser.add_argument("--names", nargs="*", default=None,
                        help="display names, in the order of --baseline then --runs")
    parser.add_argument("--json", default=None, help="also write the comparison as JSON")
    parser.add_argument("--metrics", nargs="*",
                        default=list(BINARY_METRICS) + list(CONTINUOUS_METRICS))
    parser.add_argument("--behaviour", action="store_true",
                        help="also compare the behavioural diagnostics (frozen time, "
                             "obstruction, rotation sign-flip rate). They are what "
                             "separate two heads that score the same on the outcome "
                             "metrics, and they need the pairing and the MDE just as "
                             "much")
    args = parser.parse_args()
    if args.behaviour:
        args.metrics = list(args.metrics) + list(BEHAVIOUR_METRICS)

    paths = [args.baseline] + list(args.runs)
    names = args.names or [Path(p).name for p in paths]
    if len(names) != len(paths):
        raise SystemExit("--names must give one name per run")
    runs = [load_jsonl(p) for p in paths]

    shared = common_uids(runs)
    print(f"{len(shared)} episodes evaluated in all {len(runs)} runs")
    for name, run in zip(names, runs):
        extra = len(run) - len(shared)
        print(f"    {name:<22} {len(run):>4} evaluated" + (f"  ({extra} not shared)" if extra else ""))

    print("\n=== per-run, on the shared episodes ===")
    header = f"{'run':<22}" + "".join(f"{m:>16}" for m in args.metrics)
    print(header)
    tables = []
    for name, run in zip(names, runs):
        table = rate_table(run, shared)
        tables.append(table)
        row = f"{name:<22}" + "".join(f"{_fmt(table.get(m)):>16}" for m in args.metrics)
        print(row)
    for name, table in zip(names, tables):
        for metric in BINARY_METRICS:
            if f"{metric}_ci" in table:
                low, high = table[f"{metric}_ci"]
                print(f"    {name:<20} {metric:<16} {table[metric]:.3f}  95% CI [{low:.3f}, {high:.3f}]")

    payload = {
        "episodes_shared": len(shared),
        "runs": {name: table for name, table in zip(names, tables)},
        "comparisons": {},
    }

    for name, run in zip(names[1:], runs[1:]):
        print(f"\n=== {name}  vs  {names[0]}  (paired, n={len(shared)}) ===")
        print(f"{'metric':<16}{'baseline':>10}{'run':>10}{'delta':>10}"
              f"{'95% CI':>20}{'p':>10}{'MDE':>10}")
        rows = []
        for comparison in compare_runs(runs[0], run, args.metrics):
            marker = " *" if comparison.significant() else ""
            print(
                f"{comparison.metric:<16}{comparison.mean_a:>10.3f}{comparison.mean_b:>10.3f}"
                f"{comparison.delta:>+10.3f}"
                f"{f'[{comparison.ci_low:+.3f}, {comparison.ci_high:+.3f}]':>20}"
                f"{_fmt(comparison.p_value):>10}"
                f"{_fmt(comparison.extra.get('mde_paired')):>10}{marker}"
            )
            rows.append({
                "metric": comparison.metric, "kind": comparison.kind, "n": comparison.n,
                "mean_baseline": comparison.mean_a, "mean_run": comparison.mean_b,
                "delta": comparison.delta,
                "ci": [comparison.ci_low, comparison.ci_high],
                "p_value": comparison.p_value,
                "significant": comparison.significant(),
                **{k: v for k, v in comparison.extra.items()},
            })
        payload["comparisons"][name] = rows

    n = len(shared)
    print(f"\n=== what this episode set can resolve (n={n}, alpha=0.05, power=0.80) ===")
    for baseline_rate in (0.10, 0.20, 0.30):
        print(f"    unpaired, baseline success {baseline_rate:.2f}: "
              f"MDE {mde_unpaired_rate(n, baseline_rate):.3f}")
    for discordance in (0.10, 0.20, 0.30):
        print(f"    paired,   discordance {discordance:.2f}:      "
              f"MDE {mde_paired_rate(n, discordance):.3f}")
    payload["resolution"] = {
        "n": n,
        "mde_unpaired": {f"{p:.2f}": mde_unpaired_rate(n, p) for p in (0.1, 0.2, 0.3)},
        "mde_paired": {f"{d:.2f}": mde_paired_rate(n, d) for d in (0.1, 0.2, 0.3)},
    }

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nJSON -> {args.json}")


if __name__ == "__main__":
    main()
