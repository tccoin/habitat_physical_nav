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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

    from objectnav_eval.navmesh import apply_navmesh_choice

    scene_path, scene_dataset_config = resolve_scene(episode, config.scene_root)
    robot_sim = build_robot_sim(
        scene_path, scene_dataset_config, config.width, config.height,
        str(DEFAULT_URDF_PATH),
        depth_sensor_uuid=getattr(config, "depth_sensor_uuid", None),
        topdown_sensor_uuid=(
            "video_bev_topdown"
            if config.record_video and getattr(config, "video_bev", True)
            else None
        ),
    )
    # Which mesh every distance in this run is measured on. Printed rather than
    # assumed -- it changes what the numbers mean, not how the robot moves.
    navmesh_description = apply_navmesh_choice(robot_sim, scene_path, config.navmesh)
    robot_sim.navmesh_step_filter = bool(
        getattr(config, "navmesh_step_filter", False)
    )
    print(f"    navmesh         {navmesh_description}", flush=True)
    if robot_sim.navmesh_step_filter:
        print("    execution fence active (per-substep navmesh clipping)", flush=True)
    return robot_sim, scene_path


def _build_runner(robot_sim, scene_id: str, config: Any, policy) -> EpisodeRunner:
    from continuous_demos.pid_pose_controller import PIDPoseController

    from .control import build_pid_config

    pid_config = build_pid_config(
        getattr(config, "pid_preset", "baseline"),
        getattr(config, "pid_overrides", None),
    )

    # Printed for the same reason the navmesh choice is: it changes what the
    # robot can execute, and a run that silently used different limits than the
    # one it is being compared against is unreadable after the fact.
    from .control import describe_pid, resolve_pid_settings

    print(
        "    pid             "
        f"{getattr(config, 'pid_preset', 'baseline')} "
        f"({describe_pid(resolve_pid_settings(getattr(config, 'pid_preset', 'baseline'), getattr(config, 'pid_overrides', None)))})",
        flush=True,
    )

    return EpisodeRunner(
        RunnerContext(
            robot_sim=robot_sim,
            task=build_task(
                robot_sim,
                success_distance=config.success_distance,
                distance_to=config.distance_to,
                scene_id=scene_id,
                scene_root=config.scene_root,
                live_height=config.live_height,
            ),
            executor=ChunkExecutor(
                robot_sim,
                PIDPoseController(robot_sim, pid_config),
                dt=config.dt,
                gap=config.gap,
                collect_contacts=config.collision_probe,
                log_camera=getattr(config, "log_trajectory", True),
                sensor_uuid=config.sensor_uuid,
            ),
            policy=policy,
            config=config,
            output_dir=Path(config.output_dir),
            probes=default_probes(
                config.keep_tracking_series,
                dt=config.dt,
                collect_contacts=config.collision_probe,
                log_trajectory=getattr(config, "log_trajectory", True),
            ),
        )
    )


def _evaluate_episode(
    episode: EvalEpisode,
    config: Any,
    screener: EpisodeScreener,
    runner: EpisodeRunner,
    results: List[Dict[str, Any]],
    screening: ScreeningLog,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
) -> None:
    verdict = screener.screen(episode)
    if not verdict.ok:
        screening.add(episode, verdict)
        record = screening.records[-1]
        if progress_callback is not None:
            progress_callback("skipped", record)
        print(
            f"[shard {config.shard_index}] SKIP {episode.uid} "
            f"({episode.object_category}): {verdict.reason}",
            flush=True,
        )
        return
    record = runner.run(episode, verdict)
    results.append(record)
    if progress_callback is not None:
        progress_callback("result", record)
    print(format_episode_line(record, config.shard_index), flush=True)


def run_episodes(
    episodes: Sequence[EvalEpisode],
    config: Any,
    server_script: str,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate every episode, reusing one simulator per scene.

    ``HabitatRobotSim.reset()`` does not fully restore the physics state between
    episodes -- measured with a *stateless* image-blind policy, the same episode
    executes a ~1.4% different path depending on whether another episode ran
    before it in the same simulator (see docs/OBJECTNAV_EVAL.md, "Episode order
    matters"). ``--fresh-sim-per-episode`` trades a scene load per episode for a
    clean slate. It is off by default because the aggregate metrics are stable
    without it and the cost is large; turn it on when per-episode numbers have
    to be reproducible independently of shard composition.
    """
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
                f"({len(scene_episodes)} episodes)"
                + ("  [fresh sim per episode]" if config.fresh_sim_per_episode else ""),
                flush=True,
            )
            try:
                screener = EpisodeScreener(
                    robot_pathfinder=robot_sim.sim.pathfinder,
                    dataset_pathfinder=dataset_pathfinder(scene_path),
                    success_distance=config.success_distance,
                    screen_trivial=config.screen_trivial,
                )
                runner = _build_runner(robot_sim, scene_id, config, policy)
                for index, episode in enumerate(scene_episodes):
                    if config.fresh_sim_per_episode and index > 0:
                        # Close before opening: two live habitat-sim Simulators
                        # abort the process at teardown.
                        robot_sim.sim.close()
                        robot_sim, _ = build_scene_simulator(episode, config)
                        screener.robot_pathfinder = robot_sim.sim.pathfinder
                        runner = _build_runner(robot_sim, scene_id, config, policy)
                    _evaluate_episode(
                        episode,
                        config,
                        screener,
                        runner,
                        results,
                        screening,
                        progress_callback,
                    )
            finally:
                robot_sim.sim.close()
    finally:
        policy.close()
    return results, screening.records


def run_episode_queue(
    next_episode: Callable[[Optional[str]], Optional[EvalEpisode]],
    config: Any,
    server_script: str,
    progress_callback: Callable[[str, Dict[str, Any]], None],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep one policy alive while dynamically claiming episodes."""
    from objectnav_eval.policies import build_policy

    log_path = (
        Path(config.output_dir) / "logs" / f"policy_shard{config.shard_index}.log"
    )
    policy = build_policy(config, server_script, str(log_path))
    results: List[Dict[str, Any]] = []
    screening = ScreeningLog()
    robot_sim = None
    current_scene_id: Optional[str] = None
    screener = None
    runner = None
    try:
        while True:
            episode = next_episode(current_scene_id)
            if episode is None:
                break
            rebuild = (
                robot_sim is None
                or config.fresh_sim_per_episode
                or episode.scene_id != current_scene_id
            )
            if rebuild:
                if robot_sim is not None:
                    robot_sim.sim.close()
                robot_sim, scene_path = build_scene_simulator(episode, config)
                current_scene_id = episode.scene_id
                print(
                    f"[shard {config.shard_index}] scene {Path(scene_path).stem} "
                    f"(queue episode {episode.uid})"
                    + (
                        "  [fresh sim per episode]"
                        if config.fresh_sim_per_episode
                        else ""
                    ),
                    flush=True,
                )
                screener = EpisodeScreener(
                    robot_pathfinder=robot_sim.sim.pathfinder,
                    dataset_pathfinder=dataset_pathfinder(scene_path),
                    success_distance=config.success_distance,
                    screen_trivial=config.screen_trivial,
                )
                runner = _build_runner(
                    robot_sim, episode.scene_id, config, policy
                )
            _evaluate_episode(
                episode,
                config,
                screener,
                runner,
                results,
                screening,
                progress_callback,
            )
    finally:
        if robot_sim is not None:
            robot_sim.sim.close()
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
