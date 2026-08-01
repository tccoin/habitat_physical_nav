"""Aggregation, the per-episode JSONL, and the printed summary.

Two rules shape everything here.

**Non-finite distances poison averages.** A run can end with the robot off its
navmesh -- driven into a nook the mesh does not cover, or onto a disconnected
island -- and then the geodesic to the goal is ``inf``. That is a legitimate
(failed) outcome and stays a failure in ``success``/``spl``, but averaging it
into a mean distance makes the whole mean ``inf`` and destroys the one number
that says how close the policy got. Distances are therefore averaged over the
finite episodes only, and the excluded count is reported explicitly as
``episodes_off_navmesh``.

**Skipped episodes are part of the record.** They appear in the JSONL with
``status: "skipped"`` and their reason, and are counted (by reason) in the
summary. Reading only the evaluated episodes would silently overstate coverage.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

RATE_KEYS = ["success", "oracle_success", "spl", "oracle_spl", "soft_spl"]
DISTANCE_KEYS = [
    "distance_to_goal_m",
    "min_distance_to_goal_m",
    "path_length_m",
    "geodesic_start_m",
    "distance_to_goal_euclidean_m",
    "steps",
    "sim_seconds",
]
# Tracking-diagnostic fields worth an aggregate. Ratios are averaged over
# episodes (each episode is one observation of "can the PID keep up"), not
# pooled over ticks, so one long episode cannot dominate.
TRACKING_KEYS = [
    "commanded_disp_mean_m",
    "achieved_disp_mean_m",
    "undershoot_ratio",
    "commanded_disp_mean_moving_m",
    "achieved_disp_mean_moving_m",
    "undershoot_ratio_moving",
    "undershoot_ratio_median",
    "tracking_lag_mean_m",
    "tracking_lag_p95_m",
    "chunk_end_lag_mean_m",
    "chunk_commanded_net_mean_m",
    "chunk_achieved_net_mean_m",
    "chunk_undershoot_ratio",
    "yaw_commanded_mean_deg",
    "yaw_achieved_mean_deg",
    "yaw_undershoot_ratio",
    "yaw_lag_mean_deg",
    "saturated_v_frac",
    "saturated_w_frac",
    "saturated_a_frac",
    "saturated_aw_frac",
]

# The behavioural diagnostics, aggregated the same way: mean over *episodes*,
# each episode one observation, so a 200-step episode cannot outvote a 20-step
# one. ``PROBE_KEYS`` maps a probe's record key to the fields worth a summary.
PROBE_KEYS: Dict[str, List[str]] = {
    "motion": [
        "frac_drive", "frac_turn", "frac_settle", "frac_frozen", "frac_stationary",
        "commanded_frac_drive", "commanded_frac_turn", "commanded_frac_settle",
        "commanded_frac_frozen", "commanded_frac_stationary",
        "blocked_drive_frac",
        "stationary_run_max_s", "stationary_run_median_ticks",
        "stationary_share_in_runs_ge_10",
        "median_speed_mps", "p90_speed_mps", "median_turnrate_degps",
    ],
    "contact": [
        "mean_ground_contacts", "mean_obstruction_contacts", "obstructed_frac",
        "obstructed_while_driving_frac", "collisions", "collisions_per_100_ticks",
        "collisions_per_m", "longest_obstructed_run_ticks",
        "peak_obstruction_force_max",
    ],
    "coherence": [
        "flip_rate_within_chunk", "flip_rate_all_adjacent",
        "flip_rate_within_chunk_turning", "achieved_flip_rate_within_chunk",
        "rotation_directness", "rotation_directness_median",
        "forward_rotation_corr", "mean_abs_turnrate_degps",
    ],
}


def is_finite(value: Any) -> bool:
    """Finiteness that survives a JSON round trip.

    ``inf``/``nan`` are written as ``null`` (strict JSON has no literal for
    them), so a merged shard reports a genuinely infinite distance as ``None``.
    Treating ``None`` as non-finite keeps the off-navmesh count and the distance
    means identical whether they are computed in-process or re-aggregated from
    files.
    """
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if is_finite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def summarize(
    results: Sequence[Dict[str, Any]],
    skipped: Sequence[Dict[str, Any]],
    total: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if results:
        for key in RATE_KEYS:
            present = [r[key] for r in results if key in r]
            if present:
                summary[key] = float(np.mean(present))
        # Each distance is averaged over the episodes where *it* is finite, and
        # those subsets differ: an episode that ends off the navmesh has an
        # infinite final distance but a perfectly good minimum. Without the
        # per-key counts the summary can show a mean minimum larger than the
        # mean final distance, which looks like a bug and is not one.
        counts = {}
        for key in DISTANCE_KEYS:
            values = [r.get(key) for r in results]
            summary[key] = _finite_mean(values)
            counts[key] = sum(1 for v in values if is_finite(v))
        summary["distance_means_n"] = counts

    summary.update(
        episodes_evaluated=len(results),
        episodes_skipped=len(skipped),
        episodes_total=total,
        episodes_off_navmesh=sum(
            1 for r in results if not is_finite(r.get("distance_to_goal_m"))
        ),
        skipped_by_reason=_counts(skipped, "reason_code"),
        terminated_by_reason=_counts(results, "terminated"),
    )
    tracking = summarize_tracking(results)
    if tracking:
        summary["tracking"] = tracking
    for probe_name, keys in PROBE_KEYS.items():
        aggregated = summarize_probe(results, probe_name, keys)
        if aggregated:
            summary[probe_name] = aggregated
    by_category = defaultdict(list)
    for record in results:
        by_category[record.get("category")].append(record)
    summary["by_category"] = {
        category: {
            "n": len(rows),
            **{
                key: float(np.mean([r[key] for r in rows]))
                for key in RATE_KEYS
                if key in rows[0]
            },
        }
        for category, rows in sorted(by_category.items(), key=lambda kv: str(kv[0]))
    }
    return summary


def summarize_tracking(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r["tracking"] for r in results if isinstance(r.get("tracking"), dict)]
    rows = [row for row in rows if row.get("ticks")]
    if not rows:
        return {}
    out = {"episodes": len(rows), "ticks": int(sum(row["ticks"] for row in rows))}
    for key in TRACKING_KEYS:
        values = [row[key] for row in rows if is_finite(row.get(key))]
        if values:
            out[key] = _finite_mean(values)
    return out


def summarize_probe(
    results: Sequence[Dict[str, Any]], probe_name: str, keys: Sequence[str]
) -> Dict[str, Any]:
    """Mean over episodes of one probe's fields.

    An episode that produced no ticks (or a probe that was switched off, and so
    reported ``ticks: 0``) contributes nothing rather than a zero -- a run with
    the collision probe disabled must not report a collision rate of 0.0, which
    would read as "no collisions" instead of "not measured".
    """
    rows = [r[probe_name] for r in results if isinstance(r.get(probe_name), dict)]
    rows = [row for row in rows if row.get("ticks")]
    if not rows:
        return {}
    out: Dict[str, Any] = {
        "episodes": len(rows),
        "ticks": int(sum(row["ticks"] for row in rows)),
    }
    for key in keys:
        values = [row[key] for row in rows if is_finite(row.get(key))]
        if values:
            out[key] = _finite_mean(values)
            if len(values) != len(rows):
                out[f"{key}_n"] = len(values)
    return out


def _counts(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get(key))] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def jsonable(value: Any, ndigits: int = 3) -> Any:
    """Round floats and turn inf/nan into null, so a line is strict JSON.

    ``json.dumps`` emits bare ``Infinity`` for a non-finite float, which most
    parsers reject; an off-navmesh episode would otherwise produce a line that
    will not load.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, ndigits) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v, ndigits) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: jsonable(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, ndigits) for v in value]
    return value


def episode_line(record: Dict[str, Any]) -> Dict[str, Any]:
    """One compact object per episode: the at-a-glance view of a run."""
    if record.get("status") == "skipped":
        return {
            "uid": record["uid"], "scene": record["scene"],
            "episode_id": record["episode_id"], "category": record["category"],
            "status": "skipped", "reason_code": record.get("reason_code"),
            "reason": record.get("reason"),
            "start_m": jsonable(record.get("geodesic_start_m"), 2),
            "start_dataset_m": jsonable(record.get("geodesic_start_dataset_m"), 2),
        }
    tracking = record.get("tracking") or {}
    motion = record.get("motion") or {}
    contact = record.get("contact") or {}
    coherence = record.get("coherence") or {}
    line = {
        "uid": record["uid"], "scene": record["scene"],
        "episode_id": record["episode_id"], "category": record["category"],
        "status": "evaluated", "policy": record.get("policy"),
        "success": int(record.get("success", 0)),
        "oracle_success": int(record.get("oracle_success", 0)),
        "spl": jsonable(record.get("spl")), "oracle_spl": jsonable(record.get("oracle_spl")),
        "soft_spl": jsonable(record.get("soft_spl")),
        "start_m": jsonable(record.get("geodesic_start_m"), 2),
        "final_m": jsonable(record.get("distance_to_goal_m"), 2),
        "min_m": jsonable(record.get("min_distance_to_goal_m"), 2),
        "path_m": jsonable(record.get("path_length_m"), 2),
        "steps": record.get("steps"),
        "terminated": record.get("terminated"),
        "undershoot": jsonable(tracking.get("undershoot_ratio_moving"), 3),
        "lag_m": jsonable(tracking.get("tracking_lag_mean_m"), 3),
        # Behaviour, not outcome: how much of the episode went nowhere, how much
        # of it was spent against something, and how incoherent the commanded
        # motion was. One field each -- the full probe output is in the .json.
        "frozen_frac": jsonable(motion.get("frac_stationary"), 3),
        "turn_frac": jsonable(motion.get("frac_turn"), 3),
        "drive_frac": jsonable(motion.get("frac_drive"), 3),
        "obstructed_frac": jsonable(contact.get("obstructed_frac"), 3),
        "collisions_per_m": jsonable(contact.get("collisions_per_m"), 3),
        "flip_rate": jsonable(coherence.get("flip_rate_within_chunk"), 3),
        "video": Path(record["video"]).name if record.get("video") else None,
    }
    return line


def write_episode_lines(
    path: Path, results: Sequence[Dict[str, Any]], skipped: Sequence[Dict[str, Any]]
) -> Path:
    """Write the JSONL: every episode the run considered, one line each."""
    rows = [episode_line(r) for r in results] + [episode_line(s) for s in skipped]
    rows.sort(key=lambda r: (r["scene"], r["episode_id"], r["uid"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"
    )
    return path


def write_results(
    path: Path,
    summary: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    skipped: Sequence[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": jsonable(summary, 6),
        "results": jsonable(sorted(results, key=lambda r: r["uid"]), 6),
        "skipped": jsonable(sorted(skipped, key=lambda r: r["uid"]), 6),
    }
    if config is not None:
        payload["config"] = config
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_summary(
    summary: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    skipped: Sequence[Dict[str, Any]],
) -> None:
    print(
        f"\n=== ObjectNav evaluation: {summary['episodes_evaluated']}/"
        f"{summary['episodes_total']} episodes evaluated, "
        f"{summary['episodes_skipped']} skipped ==="
    )
    if skipped:
        print("  skipped (unwinnable for this robot, NOT counted as failures):")
        by_reason: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in skipped:
            by_reason[record.get("reason", "?")].append(record)
        for reason, items in sorted(by_reason.items()):
            print(f"    {len(items)}x  {reason}")
            for item in items:
                dataset = item.get("geodesic_start_dataset_m", float("nan"))
                dataset_text = f"{dataset:.2f} m" if np.isfinite(dataset) else "n/a"
                print(
                    f"        {item['uid']:<28} {str(item['category']):<14} "
                    f"dataset-navmesh geodesic {dataset_text}"
                )
    if not results:
        print("  no episodes were evaluated")
        return

    for key in RATE_KEYS:
        if key in summary:
            print(f"  {key:<16} {summary[key]:.3f}")
    counts = summary.get("distance_means_n", {})
    evaluated = summary["episodes_evaluated"]

    def _n(key: str) -> str:
        count = counts.get(key, evaluated)
        return "" if count == evaluated else f" [n={count}]"

    print(
        f"  final distance   {summary['distance_to_goal_m']:.2f} m"
        f"{_n('distance_to_goal_m')}  "
        f"(min reached {summary['min_distance_to_goal_m']:.2f} m"
        f"{_n('min_distance_to_goal_m')}, "
        f"start {summary['geodesic_start_m']:.2f} m)"
    )
    if summary.get("episodes_off_navmesh"):
        print(
            f"                   {summary['episodes_off_navmesh']} episode(s) ended off "
            "the navmesh (geodesic inf); scored as failures, and excluded from the "
            "means marked [n=...] above -- so those means are over different "
            "episode subsets and are not directly comparable to each other"
        )
    print(
        f"  path length      {summary['path_length_m']:.2f} m over "
        f"{summary['steps']:.0f} policy steps"
    )
    if summary.get("terminated_by_reason"):
        reasons = ", ".join(
            f"{count}x {reason}"
            for reason, count in summary["terminated_by_reason"].items()
        )
        print(f"  terminated       {reasons}")

    print_tracking(summary.get("tracking"))
    print_behaviour(summary)

    if summary.get("by_category"):
        print("  by category:")
        for category, stats in summary["by_category"].items():
            print(
                f"    {str(category):<16} n={stats['n']:<3} "
                f"success {stats.get('success', float('nan')):.2f}  "
                f"oracle {stats.get('oracle_success', float('nan')):.2f}  "
                f"spl {stats.get('spl', float('nan')):.3f}"
            )


def print_tracking(tracking: Optional[Dict[str, Any]]) -> None:
    """The PID tracking diagnostic. Reported, never gating."""
    if not tracking:
        return
    print(f"\n  --- PID tracking diagnostic ({tracking['episodes']} episodes, "
          f"{tracking['ticks']} control ticks) ---")
    print(
        f"  per tick         commanded {tracking.get('commanded_disp_mean_m', float('nan')):.4f} m"
        f"  ->  achieved {tracking.get('achieved_disp_mean_m', float('nan')):.4f} m"
        f"   (undershoot {tracking.get('undershoot_ratio', float('nan')):.3f})"
    )
    if "undershoot_ratio_moving" in tracking:
        print(
            f"  moving ticks     commanded "
            f"{tracking.get('commanded_disp_mean_moving_m', float('nan')):.4f} m"
            f"  ->  achieved {tracking.get('achieved_disp_mean_moving_m', float('nan')):.4f} m"
            f"   (undershoot {tracking['undershoot_ratio_moving']:.3f}, "
            f"median {tracking.get('undershoot_ratio_median', float('nan')):.3f})"
        )
    if "chunk_undershoot_ratio" in tracking:
        print(
            f"  per chunk        commanded net "
            f"{tracking.get('chunk_commanded_net_mean_m', float('nan')):.4f} m"
            f"  ->  achieved {tracking.get('chunk_achieved_net_mean_m', float('nan')):.4f} m"
            f"   (undershoot {tracking['chunk_undershoot_ratio']:.3f})"
        )
    print(
        f"  tracking lag     mean {tracking.get('tracking_lag_mean_m', float('nan')):.4f} m"
        f"   p95 {tracking.get('tracking_lag_p95_m', float('nan')):.4f} m"
        f"   end-of-chunk {tracking.get('chunk_end_lag_mean_m', float('nan')):.4f} m"
    )
    print(
        f"  heading          commanded "
        f"{tracking.get('yaw_commanded_mean_deg', float('nan')):.2f} deg/tick"
        f"  ->  achieved {tracking.get('yaw_achieved_mean_deg', float('nan')):.2f} deg"
        f"   (undershoot {tracking.get('yaw_undershoot_ratio', float('nan')):.3f})"
    )
    saturation = {
        key.replace("saturated_", "").replace("_frac", ""): value
        for key, value in tracking.items()
        if key.startswith("saturated_")
    }
    if saturation:
        print(
            "  saturated        "
            + "  ".join(f"{name} {value * 100:.1f}%" for name, value in saturation.items())
            + "   (v/w = velocity limits, a/aw = acceleration limits)"
        )


def print_behaviour(summary: Dict[str, Any]) -> None:
    """The behavioural diagnostics: frozen time, collisions, motion coherence.

    Like the tracking diagnostic, reported and never gating. These are what
    separate two policies that score the same SPL but do not look remotely alike
    in the video.
    """
    motion = summary.get("motion") or {}
    contact = summary.get("contact") or {}
    coherence = summary.get("coherence") or {}
    if not (motion or contact or coherence):
        return

    def value(source: Dict[str, Any], key: str) -> float:
        return float(source.get(key, float("nan")))

    print(f"\n  --- behaviour ({motion.get('episodes', 0)} episodes, "
          f"{motion.get('ticks', 0)} control ticks) ---")
    if motion:
        print(
            f"  tick classes     drive {value(motion, 'frac_drive') * 100:.1f}%  "
            f"turn-in-place {value(motion, 'frac_turn') * 100:.1f}%  "
            f"settle {value(motion, 'frac_settle') * 100:.1f}%  "
            f"frozen {value(motion, 'frac_frozen') * 100:.1f}%"
        )
        print(
            f"  not moving       {value(motion, 'frac_stationary') * 100:.1f}% of ticks "
            f"(commanded {value(motion, 'commanded_frac_stationary') * 100:.1f}%)   "
            f"longest stall {value(motion, 'stationary_run_max_s'):.1f} s   "
            f"blocked-drive {value(motion, 'blocked_drive_frac') * 100:.1f}%"
        )
        print(
            f"  speed            median {value(motion, 'median_speed_mps'):.3f} m/s  "
            f"p90 {value(motion, 'p90_speed_mps'):.3f} m/s  "
            f"turn {value(motion, 'median_turnrate_degps'):.1f} deg/s"
        )
    if contact:
        print(
            f"  contact          obstructed {value(contact, 'obstructed_frac') * 100:.1f}% "
            f"of ticks, {value(contact, 'obstructed_while_driving_frac') * 100:.1f}% "
            f"of driving ticks   "
            f"({value(contact, 'mean_ground_contacts'):.1f} ground contact pts, "
            f"which are not collisions)"
        )
        print(
            f"  collisions       {value(contact, 'collisions'):.1f} per episode  "
            f"{value(contact, 'collisions_per_m'):.2f} per metre travelled  "
            f"longest {value(contact, 'longest_obstructed_run_ticks'):.0f} ticks"
        )
    if coherence and not is_finite(coherence.get("flip_rate_within_chunk")):
        # A policy that never commands a rotation above the deadband -- the
        # `forward` control, say -- has no direction to reverse. Saying so beats
        # printing "nan%", which reads as a bug.
        print("  coherence        no rotating ticks: the sign-flip rate is undefined "
              "for this policy")
    elif coherence:
        print(
            f"  coherence        rotation sign-flip {value(coherence, 'flip_rate_within_chunk') * 100:.1f}% "
            f"within chunk ({value(coherence, 'flip_rate_all_adjacent') * 100:.1f}% all adjacent)   "
            f"directness {value(coherence, 'rotation_directness'):.3f}"
        )
        print(
            f"                   corr(|forward|,|rotation|) "
            f"{value(coherence, 'forward_rotation_corr'):+.3f}   "
            f"(negative in real driving; ~0 means the two are sampled independently)"
        )


def load_and_merge(path: Path) -> Dict[str, Any]:
    """Re-aggregate results already on disk (a file, or a shards/ directory).

    Recovers the summary after a partial or crashed run without repeating any
    rollout.
    """
    path = Path(path)
    if path.is_dir():
        payloads = [
            json.loads(p.read_text()) for p in sorted(path.glob("*_results.json"))
        ]
        results = [r for p in payloads for r in p.get("results", [])]
        skipped = [s for p in payloads for s in p.get("skipped", [])]
        return {"results": results, "skipped": skipped, "path": path / "merged_results.json"}
    payload = json.loads(path.read_text())
    return {
        "results": payload.get("results", []),
        "skipped": payload.get("skipped", []),
        "path": path,
        "config": payload.get("config"),
    }
