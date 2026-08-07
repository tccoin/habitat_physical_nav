"""Validate PID global-pose/trajectory tracking against already-generated,
physically-realistic ``pose2d`` trajectories.

This does NOT re-run the pure-pursuit replay. It reads trajectories already
produced by ``generate_pose2d_dataset.py`` (``data/metadata/<scene>.jsonl``),
resets the robot to the exact original spawn for each selected episode, then
drives it with ``pid_pose_controller.PIDPoseController`` against the recorded
``pose2d`` sequence (one setpoint per recorded frame, at the fps the shard's
own ``recording`` block says it was recorded at -- see
``continuous_demos.recording_metadata``),
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
from typing import Any, Dict, Tuple

import numpy as np

from continuous_demos.pid_pose_controller import PIDPoseController, PIDPoseControllerConfig, track_trajectory
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    extract_demo,
    find_scene_dataset_config,
    load_demo_objectnav,
    resolve_scene_path,
)
from continuous_demos.recording_metadata import resolve_corpus_params
from continuous_demos.tracking_eval import (
    compute_metrics,
    load_metadata_records,
    plot_bev,
    select_episodes,
)


def run_episode(
    robot_sim,
    controller: PIDPoseController,
    episode: Dict[str, Any],
    record: Dict[str, Any],
    fps: float,
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
    parser.add_argument(
        "--fps", type=float, default=None,
        help="NOT normally needed: the rate the pose2d data was recorded at is read "
             "from the shard's own 'recording' block. This is the explicit fallback "
             "for shards generated before that schema existed, all of which were "
             "recorded at 30 Hz -- pass --fps 30 for those. A value contradicting "
             "what the shard records is an error, not an override.",
    )
    parser.add_argument("--output-dir", default="data/pid_eval")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # The control tick has to be the tick the data was recorded at, or every
    # setpoint is fed to the PID at the wrong rate. Read it off the shard.
    metadata_path = Path(args.metadata)
    resolved = resolve_corpus_params(
        metadata_path.parent,
        assumed_native_fps=args.fps,
        shard_paths=[metadata_path],
        flag_name="--fps",
        # allow_multiview: this replays recorded pose2d trajectories through the
        # PID controller and never opens an image, so the size of the camera rig
        # cannot affect any number it reports.
        allow_multiview=True,
    )
    print(resolved.banner())
    fps = resolved.params.native_fps

    metadata_records = load_metadata_records(metadata_path)
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

        actual, ref, result = run_episode(robot_sim, controller, episode, record, fps)
        metrics = compute_metrics(actual, ref)
        metrics["episode_id"] = ep_id
        metrics["num_steps"] = result["num_steps"]
        sat = result["saturation_counts"]
        n = max(result["num_steps"], 1)
        metrics["saturation_rate"] = {k: v / n for k, v in sat.items()}
        all_metrics.append(metrics)

        plot_bev(f"PID tracking -- {ep_id}", ref, actual, plots_dir / f"{ep_id}.png")
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
