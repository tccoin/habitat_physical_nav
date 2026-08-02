"""Comparing two evaluation runs, and saying what a comparison can resolve.

A summary line like ``success 0.36 vs 0.21`` is not a result until you know how
many episodes it rests on. On a 14-episode split an image-blind ``forward``
control outscores a trained checkpoint on SPL, and the difference is noise; the
same gap over 200 episodes is not. This module is the part of the harness that
answers *"is this difference bigger than what this episode set can resolve?"*.

Three things it does, and why each is the version it is:

**Pairing.** Two runs over the same episode set are compared per episode, not as
two independent samples. Episode difficulty dominates the variance -- a split
where half the goals are three rooms away scores every policy badly -- and
pairing removes it entirely. The unpaired comparison is also reported, because
it is the one that applies when a future run uses a different episode set.

**Exact tests, not asymptotic ones.** Paired binary outcomes are tested with
McNemar's *exact* binomial on the discordant pairs. With a success rate near
0.2 and a few hundred episodes, the discordant count is small enough that the
chi-square approximation is visibly wrong, and it errs towards significance.

**A stated minimum detectable effect.** ``mde_*`` inverts the power calculation:
given the episode count actually run, what is the smallest difference this
harness would detect 80% of the time? Reporting it alongside a null result is
the difference between "the checkpoints are equivalent" and "this split cannot
tell them apart" -- which are entirely different claims, and only the second one
is usually supported.

Distances are handled the way :mod:`objectnav_eval.reporting` handles them: an
episode that ends off the navmesh has an infinite final distance, and averaging
it in destroys the mean. Such values are dropped per metric, and the count of
episodes each metric actually covers travels with it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Per-episode metrics compared by default. ``progress_m`` is derived rather than
# stored: it is the one continuous measure that separates a real policy from an
# image-blind control on a small split (see docs/OBJECTNAV_EVAL.md), because
# success and SPL both saturate at zero for most episodes.
BINARY_METRICS = ("success", "oracle_success")
CONTINUOUS_METRICS = ("spl", "oracle_spl", "progress_m", "min_m", "final_m", "path_m")
# The behavioural diagnostics, compared the same way. They are the ones most
# likely to separate two heads that score indistinguishably on the outcome
# metrics -- and they need the pairing and the MDE just as much, because "this
# policy freezes more" is exactly the kind of claim that gets made from two
# means and no interval.
BEHAVIOUR_METRICS = (
    "frozen_frac", "turn_frac", "drive_frac",
    "obstructed_frac", "collisions_per_m", "flip_rate",
)

Z_ALPHA_TWO_SIDED_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Evaluated episodes of one run, keyed by uid.

    Accepts the run directory, the ``*.jsonl`` itself, or the ``*.json``. Skipped
    episodes are dropped here on purpose: they are part of the *coverage* record,
    not of any comparison, and counting them as failures is exactly the mistake
    the screening layer exists to prevent.
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("objectnav_eval_results*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"no objectnav_eval_results*.jsonl under {path}")
        path = candidates[0]
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "evaluated":
                continue
            rows[row["uid"]] = _derive(row)
    if not rows:
        raise ValueError(f"{path} contains no evaluated episodes")
    return rows


def _derive(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add the derived per-episode measures the comparison needs."""
    start, minimum = row.get("start_m"), row.get("min_m")
    row = dict(row)
    row["progress_m"] = (
        start - minimum
        if isinstance(start, (int, float)) and isinstance(minimum, (int, float))
        else None
    )
    return row


def common_uids(runs: Sequence[Dict[str, Dict[str, Any]]]) -> List[str]:
    """Episodes evaluated in *every* run, in canonical order.

    Screening is deterministic given the episode set and the robot footprint, so
    this is normally every episode -- but a crashed shard silently shrinks one
    run, and comparing its 180 episodes against another's 214 would attribute the
    easier subset's score to the policy.
    """
    if not runs:
        return []
    shared = set(runs[0])
    for run in runs[1:]:
        shared &= set(run)
    return sorted(shared)


# ---------------------------------------------------------------------------
# Intervals and tests
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = Z_ALPHA_TWO_SIDED_95) -> Tuple[float, float]:
    """Wilson score interval for a rate.

    Not the normal approximation: at n = 14 with 3 successes the normal interval
    runs below zero, and these splits produce rates near the boundary routinely.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for paired binary outcomes with ``b``/``c`` discordant.

    Under the null the discordant pairs split 50/50, so this is a two-sided
    binomial test on ``min(b, c)`` out of ``b + c``. Concordant pairs carry no
    information about the difference and do not appear.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    diffs: np.ndarray, iterations: int = 20000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a mean paired difference.

    Resamples *episodes*, which is the unit that is exchangeable here. A t-based
    interval would assume a symmetric difference distribution; per-episode SPL
    differences are a spike at zero with a long tail, which is neither.
    """
    diffs = np.asarray([d for d in diffs if d is not None and np.isfinite(d)], dtype=float)
    if diffs.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, diffs.size, size=(iterations, diffs.size))
    means = diffs[draws].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


# ---------------------------------------------------------------------------
# Minimum detectable effect
# ---------------------------------------------------------------------------


def mde_unpaired_rate(
    n_per_arm: int, baseline: float, z_alpha: float = Z_ALPHA_TWO_SIDED_95,
    z_power: float = Z_POWER_80,
) -> float:
    """Smallest success-rate difference two independent arms of ``n`` resolve.

    Solved by fixed point rather than in closed form because the variance term
    depends on the effect being detected: raising the treatment rate raises its
    own variance, so the naive ``2p(1-p)`` version understates the requirement.
    """
    if n_per_arm <= 0:
        return float("nan")
    baseline = min(max(baseline, 1e-6), 1 - 1e-6)
    delta = 0.0
    for _ in range(200):
        p2 = min(baseline + delta, 1.0 - 1e-6)
        variance = baseline * (1 - baseline) + p2 * (1 - p2)
        updated = (z_alpha + z_power) * math.sqrt(variance / n_per_arm)
        if abs(updated - delta) < 1e-9:
            return updated
        delta = updated
    return delta


def mde_paired_rate(
    n_pairs: int, discordance: float, z_alpha: float = Z_ALPHA_TWO_SIDED_95,
    z_power: float = Z_POWER_80,
) -> float:
    """Smallest rate difference a *paired* comparison over ``n`` episodes resolves.

    ``discordance`` is the fraction of episodes on which the two policies
    disagree, which is what carries the signal in McNemar's test. It has to be
    measured, not assumed: two checkpoints of the same run agree on most
    episodes (low discordance, high power), while a checkpoint and an
    image-blind control disagree often (high discordance, less power per
    episode than the raw n suggests).
    """
    if n_pairs <= 0 or discordance <= 0:
        return float("nan")
    return (z_alpha + z_power) * math.sqrt(discordance / n_pairs)


def mde_paired_mean(n_pairs: int, sd_of_differences: float,
                    z_alpha: float = Z_ALPHA_TWO_SIDED_95,
                    z_power: float = Z_POWER_80) -> float:
    """Smallest mean difference in a continuous measure a paired run resolves."""
    if n_pairs <= 0 or not np.isfinite(sd_of_differences):
        return float("nan")
    return (z_alpha + z_power) * sd_of_differences / math.sqrt(n_pairs)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    metric: str
    kind: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        if self.p_value is not None:
            return self.p_value < alpha
        return np.isfinite(self.ci_low) and (self.ci_low > 0 or self.ci_high < 0)


def compare_runs(
    run_a: Dict[str, Dict[str, Any]],
    run_b: Dict[str, Dict[str, Any]],
    metrics: Sequence[str] = BINARY_METRICS + CONTINUOUS_METRICS + BEHAVIOUR_METRICS,
    seed: int = 0,
) -> List[MetricComparison]:
    """Per-metric paired comparison of ``b`` against ``a`` on their shared episodes."""
    uids = common_uids([run_a, run_b])
    out: List[MetricComparison] = []
    for metric in metrics:
        pairs = [
            (run_a[u].get(metric), run_b[u].get(metric))
            for u in uids
            if _finite(run_a[u].get(metric)) and _finite(run_b[u].get(metric))
        ]
        if not pairs:
            continue
        a_values = np.array([p[0] for p in pairs], dtype=float)
        b_values = np.array([p[1] for p in pairs], dtype=float)
        diffs = b_values - a_values
        if metric in BINARY_METRICS:
            b_only = int(np.sum((a_values == 0) & (b_values == 1)))
            a_only = int(np.sum((a_values == 1) & (b_values == 0)))
            low, high = paired_bootstrap_ci(diffs, seed=seed)
            comparison = MetricComparison(
                metric=metric, kind="binary", n=len(pairs),
                mean_a=float(a_values.mean()), mean_b=float(b_values.mean()),
                delta=float(diffs.mean()), ci_low=low, ci_high=high,
                p_value=mcnemar_exact(b_only, a_only),
                extra={
                    "b_only": b_only, "a_only": a_only,
                    "discordant": b_only + a_only,
                    "discordance": (b_only + a_only) / len(pairs),
                    "wilson_a": wilson_interval(int(a_values.sum()), len(pairs)),
                    "wilson_b": wilson_interval(int(b_values.sum()), len(pairs)),
                    "mde_paired": mde_paired_rate(
                        len(pairs), max((b_only + a_only) / len(pairs), 1e-9)
                    ),
                    "mde_unpaired": mde_unpaired_rate(len(pairs), float(a_values.mean())),
                },
            )
        else:
            low, high = paired_bootstrap_ci(diffs, seed=seed)
            comparison = MetricComparison(
                metric=metric, kind="continuous", n=len(pairs),
                mean_a=float(a_values.mean()), mean_b=float(b_values.mean()),
                delta=float(diffs.mean()), ci_low=low, ci_high=high,
                extra={
                    "sd_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan"),
                    "mde_paired": mde_paired_mean(
                        len(pairs),
                        float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan"),
                    ),
                },
            )
        out.append(comparison)
    return out


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return value is not None
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def rate_table(run: Dict[str, Dict[str, Any]], uids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Headline numbers for one run, restricted to ``uids`` if given.

    Restricting matters when runs cover slightly different episode sets: the
    published number for a comparison must come from the same episodes as the
    comparison, or the table and the test disagree.
    """
    keys = list(uids) if uids is not None else sorted(run)
    rows = [run[u] for u in keys if u in run]
    out: Dict[str, Any] = {"n": len(rows)}
    for metric in BINARY_METRICS:
        values = [r[metric] for r in rows if _finite(r.get(metric))]
        if values:
            successes = int(sum(values))
            out[metric] = successes / len(values)
            out[f"{metric}_ci"] = wilson_interval(successes, len(values))
            out[f"{metric}_n"] = len(values)
    for metric in CONTINUOUS_METRICS + BEHAVIOUR_METRICS:
        values = [float(r[metric]) for r in rows if _finite(r.get(metric))]
        if values:
            out[metric] = float(np.mean(values))
            out[f"{metric}_n"] = len(values)
    return out
