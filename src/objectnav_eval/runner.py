"""One episode, from spawn to a result record.

The runner owns the rollout loop and nothing else: episodes come from a source,
screening decides what is runnable, the policy decides what to do, the executor
turns that into motion, the task computes the metrics and the probes watch.
Swapping any one of those does not touch this file.

The loop is:

    render -> policy.act() -> execute the chunk's first `gap` setpoints,
    one control tick at a time -> feed every tick to the task's path tracker
    and (every `evaluate_every` ticks) to the measures -> repeat.

Success semantics
-----------------
The policies evaluated here emit pose chunks and mostly have no STOP action, so
"did it stop at the goal" is read off the trajectory: the runner calls
``task.call_stop()`` when the episode ends, which makes habitat's ``Success``
mean *distance to goal is under the threshold at the final step* -- the policy
has to still be at the goal when the episode ends, not merely have brushed past
it. ``oracle_success`` is the "did it ever find the object" number. A policy
that *does* have a termination head sets ``ActionChunk.stop`` and gets the
strict interpretation for free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from objectnav_eval.episodes import EvalEpisode
from objectnav_eval.executor import ChunkExecutor, TickRecord
from objectnav_eval.navmesh import robot_position_3d
from objectnav_eval.policies import ActionChunk, Observation, Policy
from objectnav_eval.probes import GoalDistanceProbe, RolloutProbe, TrackingDiagnostic
from objectnav_eval.screening import ScreenResult
from objectnav_eval.video import OverlayState, Recorder, annotate

# Metrics lifted out of the task and reported per episode. Adding a measure to
# the task makes it appear here automatically -- this list only fixes the names
# the summary knows how to aggregate as rates.
RATE_METRICS = ("success", "oracle_success", "spl", "oracle_spl", "soft_spl")


@dataclass
class RunnerContext:
    """Everything a runner needs that is scene-scoped rather than per-episode."""

    robot_sim: Any
    task: Any
    executor: ChunkExecutor
    policy: Policy
    config: Any
    output_dir: Path
    probes: Sequence[RolloutProbe] = field(default_factory=tuple)


class EpisodeRunner:
    def __init__(self, context: RunnerContext):
        self.ctx = context
        self.tracking = next(
            (p for p in context.probes if isinstance(p, TrackingDiagnostic)), None
        )

    # ------------------------------------------------------------------
    def video_path(self, episode: EvalEpisode) -> Optional[Path]:
        config = self.ctx.config
        if not config.record_video:
            return None
        name = f"{config.tag.lstrip('_') + '_' if config.tag else ''}{episode.uid}"
        return self.ctx.output_dir / "videos" / f"{name.replace(':', '_')}.mp4"

    # ------------------------------------------------------------------
    def run(self, episode: EvalEpisode, screen: ScreenResult) -> Dict[str, Any]:
        ctx = self.ctx
        config = ctx.config
        task = ctx.task

        start_metrics = task.reset(episode.episode, snap_to_navmesh=config.snap_start)
        ctx.executor.reset()
        ctx.policy.reset(episode.object_category)
        for probe in ctx.probes:
            probe.reset(episode)
        goal_probe = GoalDistanceProbe()
        goal_probe.reset(episode)

        start_distance = float(start_metrics["distance_to_goal"])
        distance = start_distance
        min_distance = distance
        goal_probe.observe(distance)

        recorder = Recorder(
            self.video_path(episode), fps=config.fps, scale=config.video_scale,
            enabled=config.record_video,
        )

        first_reach_step: Optional[int] = None
        latencies: List[float] = []
        terminated: Optional[str] = None
        chunk: Optional[ActionChunk] = None
        steps = 0
        started = time.perf_counter()

        def on_tick(record: TickRecord) -> None:
            nonlocal distance, min_distance, first_reach_step
            task.observe_pose(record.pose)
            for probe in ctx.probes:
                probe.on_tick(record)
            if record.tick % config.evaluate_every == 0:
                metrics = task.evaluate()
                distance = float(metrics["distance_to_goal"])
                if np.isfinite(distance):
                    min_distance = min(min_distance, distance)
                if distance < config.success_distance and first_reach_step is None:
                    first_reach_step = record.chunk_index
            # The next step's [obs] frame covers the final tick of this chunk,
            # so it is not re-rendered here.
            if config.record_video and record.tick_in_chunk < ctx.executor.gap - 1:
                recorder.add(
                    annotate(
                        self._render(),
                        OverlayState(
                            step=record.chunk_index, sim_time=record.sim_time,
                            goal=episode.object_category, distance=distance,
                            start_distance=start_distance,
                            reached=first_reach_step is not None,
                        ),
                    )
                )

        try:
            for step in range(config.max_steps):
                rgb = self._render()
                if config.record_video:
                    recorder.add(
                        annotate(
                            rgb,
                            OverlayState(
                                step=step, sim_time=ctx.executor.sim_time,
                                goal=episode.object_category, distance=distance,
                                start_distance=start_distance,
                                reached=first_reach_step is not None,
                                is_observation=True,
                            ),
                        )
                    )

                observation = Observation(
                    rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
                    goal=episode.object_category,
                    step=step,
                    pose=ctx.robot_sim.get_2d_pose(),
                    sim_time=ctx.executor.sim_time,
                )
                chunk = ctx.policy.act(observation)
                latencies.append(float(chunk.stats.get("latency_s", float("nan"))))
                steps = step + 1

                execution = ctx.executor.execute(chunk, step, on_tick=on_tick)
                for probe in ctx.probes:
                    probe.on_chunk(execution)
                goal_probe.observe(distance)

                if chunk.stop:
                    terminated = "policy called stop"
                    break
                if ctx.robot_sim.is_tipped_over():
                    terminated = "robot tipped over"
                    break
                if config.auto_stop and first_reach_step is not None:
                    terminated = "auto-stop on arrival"
                    break
        finally:
            recorder.close()

        # The episode is over: the agent is where it declared it was done.
        task.call_stop()
        metrics = task.evaluate()
        distance = float(metrics["distance_to_goal"])
        if np.isfinite(distance):
            min_distance = min(min_distance, distance)

        return self._record(
            episode=episode, screen=screen, metrics=metrics, start_metrics=start_metrics,
            distance=distance, min_distance=min_distance, steps=steps,
            first_reach_step=first_reach_step, terminated=terminated,
            latencies=latencies, goal_probe=goal_probe, recorder=recorder,
            wall_seconds=time.perf_counter() - started, chunk=chunk,
        )

    # ------------------------------------------------------------------
    def _render(self) -> np.ndarray:
        obs = self.ctx.robot_sim.get_obs()[self.ctx.config.sensor_uuid]
        return np.asarray(obs)[..., :3]

    def _euclidean_to_goal(self, episode: EvalEpisode) -> float:
        """Straight-line distance to the nearest view point.

        Carried alongside the geodesic because the robot is a physics body that
        can end up slightly off the navmesh, where the geodesic snaps (or goes
        infinite); a large gap between the two says that is what happened rather
        than the robot genuinely being far away.
        """
        if not len(episode.view_points):
            return float("nan")
        position = robot_position_3d(self.ctx.robot_sim)
        return float(
            np.linalg.norm(episode.view_points - position[None, :], axis=1).min()
        )

    def _record(
        self, *, episode: EvalEpisode, screen: ScreenResult, metrics: Dict[str, Any],
        start_metrics: Dict[str, Any], distance: float, min_distance: float, steps: int,
        first_reach_step: Optional[int], terminated: Optional[str],
        latencies: List[float], goal_probe: GoalDistanceProbe, recorder: Recorder,
        wall_seconds: float, chunk: Optional[ActionChunk],
    ) -> Dict[str, Any]:
        ctx = self.ctx
        record: Dict[str, Any] = {
            **episode.identity(),
            "status": "evaluated",
            "policy": ctx.policy.name,
            "uses_observation": ctx.policy.uses_observation,
        }
        for key in RATE_METRICS:
            if key in metrics:
                record[key] = float(metrics[key])
        record.update(
            distance_to_goal_m=float(distance),
            distance_to_goal_euclidean_m=self._euclidean_to_goal(episode),
            min_distance_to_goal_m=float(min_distance),
            path_length_m=float(metrics.get("path_length", float("nan"))),
            # Two start distances, deliberately: the task's own measure (the SPL
            # denominator) and screening's (explicitly snapped goal set). They
            # should agree closely; where they do not, one of them is answering
            # a different question and the run should say so.
            geodesic_start_m=float(start_metrics["distance_to_goal"]),
            geodesic_start_screen_m=float(screen.geodesic_start_m),
            geodesic_start_dataset_m=float(screen.geodesic_start_dataset_m),
            dataset_geodesic_m=episode.dataset_geodesic,
            steps=int(steps),
            ticks=int(ctx.executor.tick),
            sim_seconds=float(ctx.executor.sim_time),
            wall_seconds=float(wall_seconds),
            mean_policy_latency_s=(
                float(np.nanmean(latencies)) if latencies else float("nan")
            ),
            first_reach_step=first_reach_step,
            terminated=terminated or "max steps",
            video=str(recorder.path) if recorder.enabled else None,
            video_frames=int(recorder.frames),
            goal_distance=goal_probe.summary(),
        )
        for probe in ctx.probes:
            record[probe.name] = probe.summary()
        if chunk is not None and chunk.extra:
            record["policy_extra"] = chunk.extra
        return record


def format_episode_line(record: Dict[str, Any], shard_index: int = 0) -> str:
    """The one-line progress report a worker prints per episode."""
    tracking = record.get("tracking") or {}
    undershoot = tracking.get("undershoot_ratio_moving", tracking.get("undershoot_ratio"))
    undershoot_text = f" track {undershoot:.2f}" if undershoot is not None and np.isfinite(
        undershoot
    ) else ""
    return (
        f"[shard {shard_index}] {record['uid']} ({record['category']}): "
        f"dist {record['geodesic_start_m']:.2f} -> {record['distance_to_goal_m']:.2f} m "
        f"(min {record['min_distance_to_goal_m']:.2f})  "
        f"success {int(record.get('success', 0))} "
        f"oracle {int(record.get('oracle_success', 0))}  "
        f"spl {record.get('spl', float('nan')):.3f} "
        f"oracle_spl {record.get('oracle_spl', float('nan')):.3f}  "
        f"path {record['path_length_m']:.1f} m in {record['steps']} steps "
        f"({record['wall_seconds']:.0f}s){undershoot_text}"
    )
