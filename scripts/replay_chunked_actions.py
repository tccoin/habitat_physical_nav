"""Replay an action-chunked SFT arrow dataset (``build_action_chunk_dataset.py``
output) through the simulator, with no trained model involved: every action
chunk in the dataset is *ground truth* (built directly from a recorded
``pose2d`` trajectory), so decoding the chunks back into an absolute
trajectory and PID-tracking it is a correctness check on the chunk encoding
itself, not on any policy.

------------------------------------------------------------------------
Reconstruction -- "synchronous at the observation rate"
------------------------------------------------------------------------
Per ``continuous_demos.action_chunking``, chunk ``i`` (issued at observation
``i``, anchored on that observation's own ``obs_pose``) and chunk ``i + 1``
(issued however many native frames later that observation actually landed --
a fixed stride, or a per-observation sampled gap simulating variable
inference latency, see ``action_chunking.SpacingConfig``) overlap: the
actions of chunk ``i`` up to that realized gap are the ones that stay in
force until chunk ``i + 1`` takes over; anything chunk ``i`` predicts past
that point is superseded before it would ever be used. Replaying
"synchronously at the observation rate" means exactly that: consume only
each chunk's actions up to the realized gap to the next observation (the
last observation's chunk, which nothing supersedes, is consumed in full).
``action_chunking.reconstruct_reference_trajectory`` does this generically
for both fixed and variable spacing -- it reads the realized gap straight
off consecutive rows' ``frame_index`` values rather than taking a stride
parameter, so the alignment math is identical either way; see that
function's docstring. De-relativizing those actions against their own
chunk's ``obs_pose`` and concatenating across observations reconstructs the
*exact* original native-fps ``pose2d`` trajectory (frames ``frame_index[0]
.. frame_index[-1] + action_chunk_len``), since ground truth chunks are
lossless relative encodings of it -- see ``tests/test_action_chunking.py``
for the round-trip proof this relies on, under both spacing modes.

That reconstructed trajectory (prefixed with the first observation's own
``obs_pose`` -- see ``reconstruct_reference_trajectory`` for why that leading
pose is required, not optional) is then tracked with
``pid_pose_controller.track_trajectory``, exactly as ``eval_pid_tracking.py``
tracks the raw ``pose2d`` trajectory, with an identical tick-for-tick setpoint
stream over the reconstructed span. A correct chunk encoding + reconstruction
should therefore reproduce ``eval_pid_tracking.py``'s actual trajectory (not
just similar-sized tracking-error metrics) to within the simulator's own
run-to-run noise floor -- i.e. two non-chunked reruns of the same episode
should disagree from each other by about as much as a chunked run disagrees
from a non-chunked one. That, not merely "close-looking ADE/FDE", is the
correctness bar this script is checked against; see the empirical validation
in ``docs/ACTION_CHUNKING_REPORT.md``.

Usage (run from this folder, inside the ``habitat`` conda env):

    python scripts/replay_chunked_actions.py examples/17DRP5sb8fy.json.gz \\
        --dataset-dir data/arrowtable --num-episodes 4
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import load_from_disk

from continuous_demos.action_chunking import reconstruct_reference_trajectory
from continuous_demos.pid_pose_controller import PIDPoseController, PIDPoseControllerConfig, track_trajectory
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    extract_demo,
    find_scene_dataset_config,
    load_demo_objectnav,
    resolve_scene_path,
)
from continuous_demos.tracking_eval import compute_metrics, plot_bev


def group_rows_by_episode(split_ds) -> Dict[str, List[Dict[str, Any]]]:
    """``{episode_id: [row, ...]}``, each episode's rows sorted by ``obs_index``."""
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in split_ds:
        by_episode[row["episode_id"]].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: r["obs_index"])
    return by_episode


def select_episode_ids(by_episode: Dict[str, List[Dict[str, Any]]], num_episodes: int) -> List[str]:
    """Pick ``num_episodes`` episode ids spanning short/median/long observation
    counts (deterministic, sorted by length), mirroring
    ``tracking_eval.select_episodes``.
    """
    ids_by_len = sorted(by_episode.keys(), key=lambda eid: len(by_episode[eid]))
    n = len(ids_by_len)
    if num_episodes >= n:
        return ids_by_len
    fractions = np.linspace(0.05, 0.95, num_episodes)
    idxs = sorted(set(int(f * (n - 1)) for f in fractions))
    seen = []
    for i in idxs:
        while i in seen and i + 1 < n:
            i += 1
        seen.append(i)
    return [ids_by_len[i] for i in seen[:num_episodes]]


def run_chunked_episode(
    robot_sim,
    controller: PIDPoseController,
    episode: Dict[str, Any],
    rows: List[Dict[str, Any]],
    dt_native: float,
):
    """Spawn at the first observation's own recorded pose (same rationale as
    ``eval_pid_tracking.run_episode``: avoids re-triggering navmesh-snap
    drift), then track the chunk-reconstructed reference trajectory. See
    ``action_chunking.reconstruct_reference_trajectory`` for why its leading
    pose (the robot's own spawn pose) matters and isn't optional -- dropping
    it produces a real, non-decaying tracking discrepancy, not a negligible
    one (see ``docs/ACTION_CHUNKING_REPORT.md``).
    """
    path_3d, _headings, _actions = extract_demo(episode)
    ref_traj = reconstruct_reference_trajectory(rows)

    obs_pose0 = np.asarray(rows[0]["obs_pose"], dtype=float)
    height = float(path_3d[0][1])
    x0, y0, _theta0 = obs_pose0
    spawn_translation = np.array([x0, height, -y0], dtype=float)

    robot_sim.reset(
        robot_translation=spawn_translation,
        robot_rotation=episode["start_rotation"].copy(),
        snap_to_navmesh=False,
    )
    controller.reset()

    # No explicit initial_pose: ref_traj[0] is already obs_pose0 (see
    # reconstruct_reference_trajectory), so track_trajectory's default
    # (use ref_traj[0] as the point before ref_traj[0], i.e. zero feedforward
    # on the first tick) matches eval_pid_tracking.run_episode exactly.
    result = track_trajectory(robot_sim, controller, ref_traj, dt=dt_native)
    return result["actual_poses"], ref_traj, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a build_action_chunk_dataset.py arrow dataset through the "
        "PID controller with no trained model, to validate the chunk encoding round-trips."
    )
    parser.add_argument("dataset", help="Path to the .json.gz Habitat-web ObjectNav dataset.")
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to the arrow DatasetDict written by build_action_chunk_dataset.py.",
    )
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--scene-root", default=None)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--output-dir", default="data/chunked_replay_eval")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_from_disk(args.dataset_dir)
    split_ds = dataset_dict[args.split]
    if len(split_ds) == 0:
        raise ValueError(f"Split '{args.split}' of {args.dataset_dir} is empty.")

    dt_native = float(split_ds[0]["dt_native"])
    obs_spacing_mode = split_ds[0].get("obs_spacing_mode", "fixed")
    print(
        f"Loaded {len(split_ds)} row(s) from split '{args.split}'. "
        f"dt_native={dt_native:.4f}s obs_spacing_mode={obs_spacing_mode!r} "
        f"(mean obs_stride_frames={split_ds[0]['obs_stride_frames']}; realized per-observation "
        "gaps are read from consecutive rows' frame_index, not this mean, so this script needs "
        "no special-casing between fixed and variable spacing)"
    )

    by_episode = group_rows_by_episode(split_ds)
    selected_ids = select_episode_ids(by_episode, args.num_episodes)
    print(f"Selected {len(selected_ids)} episodes (of {len(by_episode)}) by chunk-count:")
    for eid in selected_ids:
        print(f"  {eid}  num_obs={len(by_episode[eid])}")

    data = load_demo_objectnav(args.dataset)
    episodes_by_id = {ep["episode_id"]: ep for ep in data["episodes"]}

    scene_id = by_episode[selected_ids[0]][0]["scene_id"]
    scene_path = resolve_scene_path(scene_id, args.scene_root)
    scene_dataset_config = find_scene_dataset_config(scene_path, args.scene_root)
    robot_sim = build_robot_sim(
        scene_path, scene_dataset_config, args.width, args.height, str(DEFAULT_URDF_PATH)
    )

    controller = PIDPoseController(robot_sim, PIDPoseControllerConfig())

    all_metrics = []
    for ep_id in selected_ids:
        rows = by_episode[ep_id]
        episode = episodes_by_id[ep_id]
        if rows[0]["scene_id"] != scene_id:
            print(f"  [skip] {ep_id}: scene mismatch ({rows[0]['scene_id']} != {scene_id})")
            continue

        actual, ref, result = run_chunked_episode(robot_sim, controller, episode, rows, dt_native)
        metrics = compute_metrics(actual, ref)
        metrics["episode_id"] = ep_id
        metrics["num_steps"] = result["num_steps"]
        sat = result["saturation_counts"]
        n = max(result["num_steps"], 1)
        metrics["saturation_rate"] = {k: v / n for k, v in sat.items()}
        all_metrics.append(metrics)

        plot_bev(f"Chunked replay -- {ep_id}", ref, actual, plots_dir / f"{ep_id}.png")
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
