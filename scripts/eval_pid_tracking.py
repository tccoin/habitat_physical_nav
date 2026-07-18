"""Validate PID global-pose/trajectory tracking against already-generated,
physically-realistic ``pose2d`` trajectories.

This does NOT re-run the pure-pursuit replay. It reads trajectories already
produced by ``generate_pose2d_dataset.py`` (``data/metadata/<scene>.jsonl``),
resets the robot to the exact original spawn for each selected episode, then
drives it with ``pid_pose_controller.PIDPoseController`` against the recorded
``pose2d`` sequence (one setpoint per recorded frame, at the recording fps),
and reports standard trajectory-tracking error metrics: ADE, FDE, heading
RMSE, max position error, max cross-track error, and controller-limit
saturation rate.

No frames are rendered and no video/images are written -- only ``step()`` +
``get_2d_pose()`` are called -- to keep this fast and disk-free (outputs are a
handful of small PNGs + JSON under ``data/pid_eval/``, which is already
``.gitignore``d).

Usage (run from this folder, inside the ``habitat`` conda env):

    python eval_pid_tracking.py 17DRP5sb8fy.json.gz \\
        --metadata data/metadata/17DRP5sb8fy.jsonl --num-episodes 4
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from continuous_demos.pid_pose_controller import PIDPoseController, PIDPoseControllerConfig, track_trajectory, wrap_angle
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    extract_demo,
    find_scene_dataset_config,
    load_demo_objectnav,
    resolve_scene_path,
)


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


def plot_bev(episode_id: str, ref: np.ndarray, actual: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.plot(ref[:, 0], ref[:, 1], "-", color="tab:purple", label="reference", linewidth=2)
    ax.plot(actual[:, 0], actual[:, 1], "-", color="tab:cyan", label="actual", linewidth=1.5)
    ax.scatter([ref[0, 0]], [ref[0, 1]], color="green", marker="o", s=40, label="start", zorder=5)
    ax.scatter([ref[-1, 0]], [ref[-1, 1]], color="red", marker="x", s=50, label="goal", zorder=5)
    ax.set_aspect("equal")
    ax.set_title(f"PID tracking -- {episode_id}", fontsize=9, wrap=True)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_episode(
    robot_sim,
    controller: PIDPoseController,
    episode: Dict[str, Any],
    record: Dict[str, Any],
    fps: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Reset the robot to the exact pose the stored ``pose2d`` trajectory
    starts from, then track it.

    NOTE: we deliberately do *not* reset via ``path_3d[0]`` (the raw human-demo
    start position) with navmesh snapping re-applied. ``data/metadata/*.jsonl``
    was generated on an earlier run, and empirically the navmesh snap for at
    least one episode here now resolves to a different point than it did at
    generation time (~1m shift) -- a drift in environment/navmesh state,
    unrelated to trajectory tracking. Spawning from the recorded ``pose2d[0]``
    instead (holding height from ``path_3d[0]``, which comes straight from the
    human demo and isn't subject to snapping) sidesteps that drift entirely and
    lets the tracking error measure what it's supposed to: PID performance, not
    stale-snap reproducibility.
    """
    path_3d, _headings, _actions = extract_demo(episode)
    ref_traj = np.asarray(record["pose2d"], dtype=float)

    height = float(path_3d[0][1])
    x0, y0, _theta0 = ref_traj[0]
    spawn_translation = np.array([x0, height, -y0], dtype=float)

    robot_sim.reset(
        robot_translation=spawn_translation,
        robot_rotation=episode["start_rotation"].copy(),
        snap_to_navmesh=False,
    )
    controller.reset()

    result = track_trajectory(robot_sim, controller, ref_traj, dt=1.0 / fps)
    return result["actual_poses"], ref_traj, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PID global pose/trajectory tracking against recorded pose2d trajectories."
    )
    parser.add_argument("dataset", help="Path to the .json.gz Habitat-web ObjectNav dataset.")
    parser.add_argument("--metadata", required=True, help="Path to data/metadata/<scene>.jsonl.")
    parser.add_argument("--scene-root", default=None)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30, help="Must match the fps the pose2d data was recorded at.")
    parser.add_argument("--output-dir", default="data/pid_eval")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = load_metadata_records(Path(args.metadata))
    selected = select_episodes(metadata_records, args.num_episodes)
    print(f"Selected {len(selected)} episodes (of {len(metadata_records)}) by trajectory length:")
    for r in selected:
        print(f"  {r['episode_id']}  len={len(r['pose2d'])}")

    data = load_demo_objectnav(args.dataset)
    episodes_by_id = {ep["episode_id"]: ep for ep in data["episodes"]}

    scene_id = selected[0]["scene_id"]
    scene_path = resolve_scene_path(scene_id, args.scene_root)
    scene_dataset_config = find_scene_dataset_config(scene_path, args.scene_root)
    robot_sim = build_robot_sim(
        scene_path, scene_dataset_config, args.width, args.height, str(DEFAULT_URDF_PATH)
    )

    controller = PIDPoseController(robot_sim, PIDPoseControllerConfig())

    all_metrics = []
    for record in selected:
        ep_id = record["episode_id"]
        episode = episodes_by_id[ep_id]
        if episode["scene_id"] != scene_id:
            print(f"  [skip] {ep_id}: scene mismatch ({episode['scene_id']} != {scene_id})")
            continue

        actual, ref, result = run_episode(robot_sim, controller, episode, record, args.fps)
        metrics = compute_metrics(actual, ref)
        metrics["episode_id"] = ep_id
        metrics["num_steps"] = result["num_steps"]
        sat = result["saturation_counts"]
        n = max(result["num_steps"], 1)
        metrics["saturation_rate"] = {k: v / n for k, v in sat.items()}
        all_metrics.append(metrics)

        plot_bev(ep_id, ref, actual, plots_dir / f"{ep_id}.png")
        print(
            f"  [{ep_id}] ADE={metrics['ade']:.3f}m FDE={metrics['fde']:.3f}m "
            f"heading_rmse={metrics['heading_rmse_deg']:.1f}deg "
            f"max_pos_err={metrics['max_position_error']:.3f}m "
            f"max_xtrack={metrics['max_cross_track_error']:.3f}m"
        )

    summary = {
        "episodes": all_metrics,
        "aggregate": {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in ["ade", "fde", "heading_rmse_deg", "max_position_error", "max_cross_track_error"]
        }
        if all_metrics
        else {},
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {output_dir / 'summary.json'} and {len(all_metrics)} BEV plots under {plots_dir}/")


if __name__ == "__main__":
    main()
