"""Score human demonstrations as ObjectNav episodes through the continuous sim.

This is the end-to-end check on the ObjectNav task layer
(``continuous_demos.objectnav``): it replays each episode's human demo with the
pure-pursuit / heading-rotation stack, feeds every physics tick to the task's
path tracker, calls stop at the end of the demo, and reports success, SPL,
soft-SPL, oracle success and oracle SPL.

Because these demos are *successful human trajectories*, success should come out
at (or very near) 1.0 and SPL comfortably above zero. The script also prints, per
episode, the discrepancy between the task's ``distance_to_goal`` at spawn and the
episode's recorded ``info["geodesic_distance"]`` -- the sharpest single check
that the planar<->habitat frame conversion and the goal set are both right.

Expect that discrepancy to be small but non-zero (~0.2 m typically, occasionally
metres in narrow spaces): distances here are computed on the navmesh
``build_robot_sim`` recomputes for the *robot* (radius 0.28 m, height 1.0 m,
max climb 0.2 m), not the point agent the dataset was authored with, so some
passages the human demo used are simply not traversable for this base.

Usage (run from the repo root, inside the ``habitat`` conda env):

    python scripts/eval_objectnav.py examples/17DRP5sb8fy.json.gz \\
        --num-episodes 8 --output-dir data/objectnav_eval
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    extract_demo,
    find_looking_slices,
    find_scene_dataset_config,
    resolve_scene_path,
)
from continuous_demos.objectnav import ObjectNavConfig, ObjectNavTask
from continuous_demos.objectnav_dataset import episodes_by_scene, load_objectnav_episodes
from continuous_demos.replay import record_segmented_demo

_HEADLINE_METRICS = [
    "success",
    "spl",
    "soft_spl",
    "oracle_success",
    "oracle_spl",
    "distance_to_goal",
    "path_length",
]


def run_episode(
    task: ObjectNavTask,
    episode: Any,
    fps: int,
    timeout: int,
    evaluate_every: int,
) -> Dict[str, Any]:
    """Replay one demo and score it.

    ``record_segmented_demo`` owns the stepping loop and hands back the poses it
    visited, so the task is fed after the fact -- the task never wraps stepping.
    Metrics are updated every ``evaluate_every`` poses (the expensive part);
    path length always sees every pose.
    """
    start_metrics = task.reset(episode)

    path_3d, path_headings, actions = extract_demo(episode)
    looking_slices = find_looking_slices(actions)
    _frames, poses = record_segmented_demo(
        task.robot_sim,
        path_3d,
        path_headings,
        looking_slices,
        timeout=timeout,
        fps=fps,
        debug_draw=False,
    )

    for index, pose in enumerate(poses, start=1):
        task.observe_pose(pose)
        if index % evaluate_every == 0:
            task.evaluate()

    # The demo ends with the human's STOP: the agent declares termination here.
    task.call_stop()
    metrics = task.evaluate()

    recorded = (episode.info or {}).get("geodesic_distance")
    metrics["episode_id"] = episode.episode_id
    metrics["object_category"] = episode.object_category
    metrics["num_poses"] = len(poses)
    # record_segmented_demo stops once it has produced timeout*fps frames; a
    # truncated replay never reaches the goal, so flag it rather than silently
    # scoring it as a navigation failure.
    metrics["replay_truncated"] = len(poses) >= timeout * fps
    metrics["start_distance_to_goal"] = start_metrics["distance_to_goal"]
    metrics["recorded_geodesic_distance"] = recorded
    metrics["start_distance_error"] = (
        abs(start_metrics["distance_to_goal"] - recorded) if recorded is not None else None
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ObjectNav metrics for human-demo replays in the continuous sim."
    )
    parser.add_argument("dataset", help="Path to the .json.gz Habitat-web ObjectNav dataset.")
    parser.add_argument("--scene-root", default=None)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Max seconds of sim time per episode. Long human demos (900+ actions) "
        "need several hundred seconds; a truncated replay is flagged as "
        "replay_truncated in the summary.",
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=30,
        help="Update measures every N poses. Success/SPL are unaffected; oracle "
        "metrics are lower bounds at this cadence.",
    )
    parser.add_argument("--success-distance", type=float, default=0.1)
    parser.add_argument("--distance-to", default="VIEW_POINTS", choices=["VIEW_POINTS", "POINT"])
    parser.add_argument("--output-dir", default="data/objectnav_eval")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_objectnav_episodes(args.dataset)
    grouped = episodes_by_scene(episodes)
    print(f"Loaded {len(episodes)} episodes across {len(grouped)} scene(s).")

    config = ObjectNavConfig(
        success_distance=args.success_distance, distance_to=args.distance_to
    )

    all_metrics: List[Dict[str, Any]] = []
    remaining = args.num_episodes
    for scene_id, scene_episodes in grouped.items():
        if remaining <= 0:
            break
        selected = scene_episodes[:remaining]
        remaining -= len(selected)

        scene_path = resolve_scene_path(scene_id, args.scene_root)
        scene_dataset_config = find_scene_dataset_config(scene_path, args.scene_root)
        robot_sim = build_robot_sim(
            scene_path, scene_dataset_config, args.width, args.height, str(DEFAULT_URDF_PATH)
        )
        task = ObjectNavTask(robot_sim, config, scene_id=scene_id)
        print(f"\nScene {scene_id}: evaluating {len(selected)} episode(s)")

        for episode in selected:
            metrics = run_episode(
                task, episode, args.fps, args.timeout, args.evaluate_every
            )
            all_metrics.append(metrics)
            d0_error = metrics["start_distance_error"]
            d0_error_str = "n/a" if d0_error is None else f"{d0_error:.3f}m"
            print(
                f"  [{metrics['episode_id']}] goal={metrics['object_category']} "
                f"success={metrics['success']:.0f} spl={metrics['spl']:.3f} "
                f"oracle_success={metrics['oracle_success']:.0f} "
                f"oracle_spl={metrics['oracle_spl']:.3f} "
                f"d_goal={metrics['distance_to_goal']:.3f}m "
                f"path={metrics['path_length']:.2f}m "
                f"d0_err={d0_error_str}"
                + (" [replay truncated]" if metrics["replay_truncated"] else "")
            )

        robot_sim.sim.close()

    summary = {
        "config": {
            "success_distance": args.success_distance,
            "distance_to": args.distance_to,
            "fps": args.fps,
            "evaluate_every": args.evaluate_every,
        },
        "episodes": all_metrics,
        "num_replay_truncated": sum(1 for m in all_metrics if m["replay_truncated"]),
        "aggregate": {
            key: float(np.mean([m[key] for m in all_metrics]))
            for key in _HEADLINE_METRICS
        }
        if all_metrics
        else {},
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if all_metrics:
        agg = summary["aggregate"]
        print(
            f"\nAggregate over {len(all_metrics)} episodes: "
            f"success={agg['success']:.3f} spl={agg['spl']:.3f} "
            f"soft_spl={agg['soft_spl']:.3f} "
            f"oracle_success={agg['oracle_success']:.3f} "
            f"oracle_spl={agg['oracle_spl']:.3f}"
        )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
