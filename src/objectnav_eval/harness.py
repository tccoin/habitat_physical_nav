"""Running a list of episodes in one process.

This is the scene loop: build one simulator per scene (scene load dominates a
short episode), screen that scene's episodes against the robot's own recomputed
navmesh, and run what survives. It is what both a single-process run and a
parallel worker execute; the only difference is which episodes they are given.

Building one simulator per scene rather than switching scenes in place is
deliberate here: episodes are grouped by scene anyway, and a task configured
with ``auto_switch_scene=False`` turns a sharding bug into an exception instead
of a silent, expensive scene reload. (Two *simultaneously live* simulators do
abort habitat-sim at teardown -- that is why the sim is closed before the next
scene is built.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from objectnav_eval.episodes import EvalEpisode, group_by_scene
from objectnav_eval.executor import ChunkExecutor
from objectnav_eval.navmesh import (
    ROBOT_NAVMESH_HEIGHT,
    ROBOT_NAVMESH_RADIUS,
    dataset_pathfinder,
    navmesh_stats,
    resolve_scene,
)
from objectnav_eval.probes import default_probes
from objectnav_eval.runner import EpisodeRunner, RunnerContext, format_episode_line
from objectnav_eval.screening import EpisodeScreener, ScreeningLog
from objectnav_eval.task import build_task


def build_scene_simulator(episode: EvalEpisode, config: Any):
    """One ``HabitatRobotSim`` for the episode's scene, plus its two navmeshes."""
    from continuous_demos.episode_io import DEFAULT_URDF_PATH, build_robot_sim

    scene_path, scene_dataset_config = resolve_scene(episode, config.scene_root)
    robot_sim = build_robot_sim(
        scene_path, scene_dataset_config, config.width, config.height,
        str(DEFAULT_URDF_PATH),
    )
    return robot_sim, scene_path


def run_episodes(
    episodes: Sequence[EvalEpisode], config: Any, server_script: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate every episode, reusing one simulator per scene."""
    from continuous_demos.pid_pose_controller import (
        PIDPoseController,
        PIDPoseControllerConfig,
    )

    from objectnav_eval.policies import build_policy

    log_path = (
        Path(config.output_dir) / "logs" / f"policy_shard{config.shard_index}.log"
    )
    policy = build_policy(config, server_script, str(log_path))

    results: List[Dict[str, Any]] = []
    screening = ScreeningLog()
    try:
        for scene_id, scene_episodes in group_by_scene(episodes).items():
            robot_sim, scene_path = build_scene_simulator(scene_episodes[0], config)
            print(
                f"[shard {config.shard_index}] scene {Path(scene_path).stem} "
                f"({len(scene_episodes)} episodes)",
                flush=True,
            )
            try:
                screener = EpisodeScreener(
                    robot_pathfinder=robot_sim.sim.pathfinder,
                    dataset_pathfinder=dataset_pathfinder(scene_path),
                    success_distance=config.success_distance,
                    screen_trivial=config.screen_trivial,
                )
                task = build_task(
                    robot_sim,
                    success_distance=config.success_distance,
                    distance_to=config.distance_to,
                    scene_id=scene_id,
                    scene_root=config.scene_root,
                    live_height=config.live_height,
                )
                runner = EpisodeRunner(
                    RunnerContext(
                        robot_sim=robot_sim,
                        task=task,
                        executor=ChunkExecutor(
                            robot_sim,
                            PIDPoseController(robot_sim, PIDPoseControllerConfig()),
                            dt=config.dt,
                            gap=config.gap,
                        ),
                        policy=policy,
                        config=config,
                        output_dir=Path(config.output_dir),
                        probes=default_probes(config.keep_tracking_series),
                    )
                )
                for episode in scene_episodes:
                    verdict = screener.screen(episode)
                    if not verdict.ok:
                        screening.add(episode, verdict)
                        print(
                            f"[shard {config.shard_index}] SKIP {episode.uid} "
                            f"({episode.object_category}): {verdict.reason}",
                            flush=True,
                        )
                        continue
                    record = runner.run(episode, verdict)
                    results.append(record)
                    print(format_episode_line(record, config.shard_index), flush=True)
            finally:
                robot_sim.sim.close()
    finally:
        policy.close()
    return results, screening.records


def dry_run(episodes: Sequence[EvalEpisode], config: Any) -> Dict[str, Any]:
    """Screen every episode without running a policy, and print the verdicts.

    Costs one scene load per scene and no model at all, which makes it the right
    way to find out what a split will actually evaluate before committing GPUs
    to it.
    """
    runnable: List[EvalEpisode] = []
    screening = ScreeningLog()
    for _scene_id, scene_episodes in group_by_scene(episodes).items():
        # A small render target: nothing is rendered, but the sensor is built.
        small = _shrunk(config)
        robot_sim, scene_path = build_scene_simulator(scene_episodes[0], small)
        try:
            robot_pathfinder = robot_sim.sim.pathfinder
            data_pathfinder = dataset_pathfinder(scene_path)
            screener = EpisodeScreener(
                robot_pathfinder, data_pathfinder, config.success_distance,
                config.screen_trivial,
            )
            print(f"\n=== {Path(scene_path).stem}: {len(scene_episodes)} episodes")
            robot = navmesh_stats(robot_pathfinder)
            print(
                f"    robot navmesh   {robot.get('navigable_area_m2', float('nan')):7.1f} m2 / "
                f"{robot.get('num_islands', -1)} islands "
                f"(r={ROBOT_NAVMESH_RADIUS} m, h={ROBOT_NAVMESH_HEIGHT} m)"
            )
            dataset = navmesh_stats(data_pathfinder)
            if dataset:
                print(
                    f"    dataset navmesh {dataset['navigable_area_m2']:7.1f} m2 / "
                    f"{dataset['num_islands']} islands"
                )
            for episode in scene_episodes:
                verdict = screener.screen(episode)
                print(
                    f"    {episode.uid:<26} {str(episode.object_category):<14} "
                    f"dataset {verdict.geodesic_start_dataset_m:7.2f} m | "
                    f"robot {verdict.geodesic_start_m:7.2f} m"
                    + ("" if verdict.ok else "   SKIP: " + verdict.reason)
                )
                if verdict.ok:
                    runnable.append(episode)
                else:
                    screening.add(episode, verdict)
        finally:
            robot_sim.sim.close()
    print(
        f"\n{len(runnable)}/{len(episodes)} episodes runnable, {len(screening)} skipped"
    )
    for code, count in sorted(screening.counts_by_reason().items()):
        print(f"    {count:>4}x {code}")
    return {"runnable": runnable, "skipped": screening.records}


def _shrunk(config: Any):
    """A copy of the config with a tiny render target (dry runs never render)."""
    from dataclasses import replace

    return replace(config, width=160, height=120)
