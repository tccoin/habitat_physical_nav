"""Shared trajectory-tracking evaluation helpers: reference-trajectory
loading/selection, ADE/FDE/heading-RMSE/cross-track metrics, and the BEV plot.

Used by both ``eval_pid_tracking.py`` (tracks the raw, un-chunked ``pose2d``
trajectories from ``generate_pose2d_dataset.py``) and
``replay_chunked_actions.py`` (tracks trajectories reconstructed from an
action-chunked arrow dataset) -- pulled out here per this repo's convention
that scripts never import each other.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from continuous_demos.pid_pose_controller import wrap_angle


def load_metadata_records(metadata_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_episodes(records: List[Dict[str, Any]], num_episodes: int) -> List[Dict[str, Any]]:
    """Pick ``num_episodes`` records spanning short/median/long trajectory
    lengths (deterministic, sorted by length -- not random) for variety.
    """
    by_len = sorted(records, key=lambda r: len(r["pose2d"]))
    n = len(by_len)
    if num_episodes >= n:
        return by_len
    fractions = np.linspace(0.05, 0.95, num_episodes)
    idxs = sorted(set(int(f * (n - 1)) for f in fractions))
    # De-duplicate collisions by nudging forward, then take the first
    # `num_episodes` unique indices.
    seen = []
    for i in idxs:
        while i in seen and i + 1 < n:
            i += 1
        seen.append(i)
    return [by_len[i] for i in seen[:num_episodes]]


def cross_track_error(actual_xy: np.ndarray, ref_polyline: np.ndarray) -> float:
    """Perpendicular distance from ``actual_xy`` to the nearest point on the
    reference polyline (not just the same-time-index reference point) -- the
    true tracking deviation, robust to small timing/lag effects.
    """
    if len(ref_polyline) < 2:
        return float(np.linalg.norm(actual_xy - ref_polyline[0]))
    p = ref_polyline[:-1]
    q = ref_polyline[1:]
    seg = q - p
    seg_len2 = np.sum(seg * seg, axis=1)
    seg_len2 = np.where(seg_len2 < 1e-12, 1e-12, seg_len2)
    t = np.sum((actual_xy - p) * seg, axis=1) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    closest = p + t[:, None] * seg
    dists = np.linalg.norm(actual_xy - closest, axis=1)
    return float(np.min(dists))


def compute_metrics(actual: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pos_err = np.linalg.norm(actual[:, :2] - ref[:, :2], axis=1)
    th_err = np.array([wrap_angle(a - r) for a, r in zip(actual[:, 2], ref[:, 2])])

    ct_errs = np.array([cross_track_error(actual[i, :2], ref[:, :2]) for i in range(len(actual))])

    return {
        "ade": float(np.mean(pos_err)),
        "fde": float(pos_err[-1]),
        "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(th_err ** 2)))),
        "max_position_error": float(np.max(pos_err)),
        "max_cross_track_error": float(np.max(ct_errs)),
        "mean_cross_track_error": float(np.mean(ct_errs)),
    }


def plot_bev(title: str, ref: np.ndarray, actual: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.plot(ref[:, 0], ref[:, 1], "-", color="tab:purple", label="reference", linewidth=2)
    ax.plot(actual[:, 0], actual[:, 1], "-", color="tab:cyan", label="actual", linewidth=1.5)
    ax.scatter([ref[0, 0]], [ref[0, 1]], color="green", marker="o", s=40, label="start", zorder=5)
    ax.scatter([ref[-1, 0]], [ref[-1, 1]], color="red", marker="x", s=50, label="goal", zorder=5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, wrap=True)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
