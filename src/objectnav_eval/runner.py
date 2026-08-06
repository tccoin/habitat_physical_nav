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

from objectnav_eval.budget import (
    TERMINATED_BUDGET,
    TERMINATED_TIME,
    GeometricBudget,
    terminated_code,
)
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


def _npwpl(start: float, closest: float, path: float) -> float:
    """Normalised progress to the closest approach, weighted by path efficiency.

    ``max(0, 1 - closest/start) * start/max(start, path)``. Both terms are
    evaluated at the same instant -- the closest approach -- which is what keeps
    it bounded below by nothing in particular but comparable across episodes.
    Zero when the start distance is degenerate or anything is non-finite, so an
    off-navmesh episode contributes 0 rather than a NaN that poisons the mean.
    """
    if not (np.isfinite(start) and np.isfinite(closest) and np.isfinite(path)):
        return 0.0
    if start <= 0:
        return 0.0
    progress = max(0.0, 1.0 - closest / start)
    return progress * (start / max(start, path))


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
        # The geometric budget is accumulated on EVERY run, whether or not it is
        # the termination rule -- it costs one subtraction per tick and it is
        # what lets a finished run be re-scored under a different cap instead of
        # re-run (scripts/rescore_budget.py). `rule` is None unless the cap is
        # active, and only then does anything terminate on it.
        rule = (
            config.build_budget_rule()
            if hasattr(config, "build_budget_rule") else None
        )
        budget = rule.budget if rule is not None else GeometricBudget(
            budget=float(getattr(config, "budget_units", float("inf")))
        )
        # Seeded with the spawn pose: without it the first chunk's motion is
        # charged from wherever tick 1 landed and the episode starts one tick
        # of travel in credit.
        budget.reset(ctx.robot_sim.get_2d_pose())
        ctx.policy.reset(episode.object_category)
        for probe in ctx.probes:
            probe.reset(episode)
        goal_probe = GoalDistanceProbe()
        goal_probe.reset(episode)

        start_distance = float(start_metrics["distance_to_goal"])
        # Path length at the moment of closest approach, frozen the same way
        # `OracleSPL` freezes it at first success. Both halves of `npwpl` have to
        # refer to the same instant or the metric charges the agent for travel
        # that happened *after* the progress it is being credited for -- which is
        # exactly the "reach the goal then wander off" pattern these stop-head-less
        # checkpoints show, and would penalise it twice.
        path_at_min = 0.0
        # Sampled here, before a single tick, so it is the *start* euclidean and
        # not wherever the robot ended up. Its ratio to the geodesic is the
        # signal for a navmesh that routes the long way round something the
        # physics body can simply drive through -- see `detour_ratio` below.
        start_euclidean = self._euclidean_to_goal(episode)
        distance = start_distance
        min_distance = distance
        goal_probe.observe(distance)

        recorder = Recorder(
            self.video_path(episode), fps=config.fps, scale=config.video_scale,
            enabled=config.record_video,
        )

        first_reach_step: Optional[int] = None
        # Separate from first_reach_step: arrival is judged at success_distance
        # (what the oracle measures latch on and what the overlay reports),
        # termination at config.stop_distance, which may be tighter.
        stop_step: Optional[int] = None
        latencies: List[float] = []
        # Per-step stop-classifier scores, kept whether or not the head is acting.
        # A threshold and a temperature are one scalar each, fitted after the
        # fact on saved scores (`stop_head.py`: they do not change AP or AUC), so
        # recording them turns "the head is miscalibrated" into an offline sweep
        # rather than a re-run of the whole split. `stop_trace` stays None on a
        # checkpoint without a stop head, so nothing else grows a column.
        stop_trace: Optional[List[List[float]]] = None
        # Proof that pose injection actually happened, per step: the value the
        # model was *shown*, straight out of the policy. The failure this catches
        # is the silent one -- a checkpoint that declares a pose spec while the
        # harness feeds it nothing would still produce a full set of plausible
        # metrics, and this column would be empty.
        pose_trace: Optional[List[List[float]]] = None
        # Times the classifier said stop somewhere the oracle did not agree, under
        # --stop-requires-oracle. Counted rather than merely ignored: this IS the
        # head's false-positive rate at the configured operating point, measured
        # on the trajectory the policy actually drives, and it is the number the
        # gate exists to expose.
        stop_suppressed = 0
        terminated: Optional[str] = None
        chunk: Optional[ActionChunk] = None
        steps = 0
        started = time.perf_counter()

        def on_tick(record: TickRecord) -> None:
            nonlocal distance, min_distance, first_reach_step, stop_step, path_at_min
            task.observe_pose(record.pose)
            budget.observe(record.pose)
            for probe in ctx.probes:
                probe.on_tick(record)
            if record.tick % config.evaluate_every == 0:
                metrics = task.evaluate()
                distance = float(metrics["distance_to_goal"])
                if np.isfinite(distance) and distance < min_distance:
                    min_distance = distance
                    path_at_min = float(getattr(task.path_tracker, "length", 0.0))
                if distance < config.success_distance and first_reach_step is None:
                    first_reach_step = record.chunk_index
                if distance < config.stop_distance and stop_step is None:
                    stop_step = record.chunk_index
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

        # Which cap the episode budget rule fired on, set from inside the tick
        # predicate below. None while the episode is still running.
        cap_fired: Optional[str] = None
        # The oracle's own mid-chunk cut, unchanged: it applies only when
        # arrival is meant to END the episode. With a delay the window has just
        # opened and the robot should keep driving through it.
        oracle_cuts = bool(config.auto_stop) and not config.auto_stop_delay

        def should_stop_tick(record: TickRecord) -> bool:
            """Terminations the EPISODE owns, evaluated at the tick they happen.

            Both callers cut mid-chunk deliberately -- an oracle that only fired
            at chunk boundaries would let the robot arrive on tick 1 and drift
            back out by tick 5, and a budget that only fired at chunk boundaries
            would hand out up to ``gap`` ticks of free motion.

            The oracle is checked first: if the agent arrived on the same tick
            the budget ran out, the episode was won, and reporting it as a cap
            would be wrong. Otherwise the cap answers, and the loop below reads
            ``cap_fired`` to say which one.
            """
            nonlocal cap_fired
            if oracle_cuts and stop_step is not None:
                return True
            if rule is not None:
                cap_fired = rule.exceeded(record.sim_time)
                if cap_fired is not None:
                    return True
            return False

        # None when nothing wants a mid-chunk cut, which keeps the default
        # configuration on exactly the code path it has always run.
        stop_when = should_stop_tick if (oracle_cuts or rule is not None) else None

        try:
            for step in range(config.step_limit if hasattr(config, "step_limit")
                              else config.max_steps):
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
                # Sampled at *observation* time, because that is the frame the
                # classifier looked at. Checking the post-chunk distance instead
                # would credit or blame it for `gap` ticks of motion it had not
                # seen when it decided.
                distance_when_asked = distance
                chunk = ctx.policy.act(observation)
                latencies.append(float(chunk.stats.get("latency_s", float("nan"))))
                if chunk.stats.get("stop_prob") is not None:
                    if stop_trace is None:
                        stop_trace = []
                    # Path length alongside the score, because an offline
                    # threshold sweep needs BOTH to be exact. Stopping does not
                    # change the trajectory -- the policy never sees the stop
                    # decision -- so "what would threshold X have scored" is a
                    # lookup into these traces, not an estimate. Without the path
                    # length only `success` is recoverable; SPL would have to be
                    # computed from the final path, which is wrong in the
                    # flattering direction. Recorded only when a stop head is
                    # present, so no other run grows a column.
                    stop_trace.append([
                        float(chunk.stats["stop_prob"]),
                        float(chunk.stats.get("stop_logit", float("nan"))),
                        float(getattr(task.path_tracker, "length", float("nan"))),
                    ])
                if chunk.stats.get("pose_value") is not None:
                    if pose_trace is None:
                        pose_trace = []
                    pose_trace.append([float(v) for v in chunk.stats["pose_value"]])
                steps = step + 1

                # The oracle stop, for a policy with no termination head of its
                # own: end the episode on the tick the agent first gets close
                # enough, not at the end of whatever chunk that tick fell in.
                # `on_tick` runs first and sets `first_reach_step`, so the
                # arriving tick is the one that stops it.
                execution = ctx.executor.execute(
                    chunk, step, on_tick=on_tick, stop_when=stop_when,
                )
                for probe in ctx.probes:
                    probe.on_chunk(execution)
                goal_probe.observe(distance)

                if cap_fired is not None:
                    # Ahead of the policy's own stop on purpose: the cap
                    # physically truncated this chunk, so the episode ended at
                    # that tick and the policy's declaration is about a chunk
                    # that did not finish running. Reporting it as a clean
                    # policy stop would credit a termination that never happened.
                    terminated = {
                        TERMINATED_BUDGET: "budget exhausted",
                        TERMINATED_TIME: "sim time limit",
                    }[cap_fired]
                    break
                if chunk.stop:
                    # Gated: the classifier only gets to end the episode where the
                    # oracle already agrees. Elsewhere its call is recorded as a
                    # false positive and the rollout carries on.
                    if config.stop_requires_oracle and not (
                        distance_when_asked < config.success_distance
                    ):
                        stop_suppressed += 1
                    else:
                        terminated = "policy called stop"
                        break
                if ctx.robot_sim.is_tipped_over():
                    terminated = "robot tipped over"
                    break
                if config.auto_stop and stop_step is not None and (
                    step >= stop_step + config.auto_stop_delay
                ):
                    terminated = (
                        "auto-stop on arrival" if not config.auto_stop_delay
                        else f"oracle stop +{config.auto_stop_delay} steps"
                    )
                    break
        finally:
            recorder.close()

        # The episode is over: the agent is where it declared it was done.
        task.call_stop()
        metrics = task.evaluate()
        distance = float(metrics["distance_to_goal"])
        if np.isfinite(distance) and distance < min_distance:
            min_distance = distance
            path_at_min = float(getattr(task.path_tracker, "length", 0.0))

        return self._record(
            episode=episode, screen=screen, metrics=metrics, start_metrics=start_metrics,
            distance=distance, min_distance=min_distance, steps=steps,
            first_reach_step=first_reach_step, terminated=terminated,
            start_euclidean=start_euclidean,
            latencies=latencies, goal_probe=goal_probe, recorder=recorder,
            wall_seconds=time.perf_counter() - started, chunk=chunk,
            stop_trace=stop_trace, pose_trace=pose_trace,
            stop_suppressed=stop_suppressed, path_at_min=path_at_min,
            budget=budget,
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
        start_euclidean: float,
        first_reach_step: Optional[int], terminated: Optional[str],
        latencies: List[float], goal_probe: GoalDistanceProbe, recorder: Recorder,
        wall_seconds: float, chunk: Optional[ActionChunk],
        stop_trace: Optional[List[List[float]]] = None,
        pose_trace: Optional[List[List[float]]] = None,
        stop_suppressed: int = 0,
        path_at_min: float = 0.0,
        budget: Optional[GeometricBudget] = None,
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
            # Path travelled up to the closest approach -- the denominator `npwpl`
            # needs, and the counterpart of the freeze `OracleSPL` performs.
            path_at_min_m=float(path_at_min),
            # NORMALISED PROGRESS WEIGHTED BY PATH LENGTH. `soft_spl` with two
            # changes, both because these checkpoints have no stop head:
            #
            #   * progress is measured to the CLOSEST APPROACH, not the final
            #     pose. `soft_spl` charges an agent that reaches the goal and then
            #     drifts, which here is a stopping deficit rather than a
            #     navigation one -- 22 of 38 successes at one checkpoint reached
            #     the goal and left during the delay window.
            #   * efficiency uses the path AT that moment, so numerator and
            #     denominator refer to the same instant. Using the total path
            #     with a frozen numerator penalises post-arrival wandering twice
            #     and breaks the bound against `oracle_spl`.
            #
            # Continuous rather than binary, which is the point: measured on this
            # episode set its MDE is 0.075 against `oracle_success`'s 0.180, so it
            # resolves differences ~2.4x smaller for the same number of episodes.
            npwpl=_npwpl(
                float(start_metrics["distance_to_goal"]), float(min_distance),
                float(path_at_min),
            ),
            path_length_m=float(metrics.get("path_length", float("nan"))),
            # Two start distances, deliberately: the task's own measure (the SPL
            # denominator) and screening's (explicitly snapped goal set). They
            # should agree closely; where they do not, one of them is answering
            # a different question and the run should say so.
            geodesic_start_m=float(start_metrics["distance_to_goal"]),
            geodesic_start_screen_m=float(screen.geodesic_start_m),
            geodesic_start_dataset_m=float(screen.geodesic_start_dataset_m),
            geodesic_start_euclidean_m=float(start_euclidean),
            # How far out of its way the navmesh says the robot must go. The
            # robot is a *physics body*, not a navmesh agent -- it can drive
            # routes the conservative robot navmesh (r=0.28, h=1.0) forbids. On
            # MP3D `8194nk5LbLH:39` the geodesic start was 9.15 m and the goal
            # was 2.20 m away in a straight line; the robot reached it in 1.9 m
            # of travel, the distance trace fell 9.99 -> 0.02 in three policy
            # steps, and SPL pinned at 1.000 because its reference path was
            # longer than the route actually taken. A high ratio is where that
            # happens. Diagnostic only -- nothing filters on it.
            detour_ratio=(
                float(start_metrics["distance_to_goal"]) / float(start_euclidean)
                if np.isfinite(start_euclidean) and start_euclidean > 1e-6
                else float("nan")
            ),
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
            # The same fact as `terminated`, as a stable code. `terminated` is
            # left phrased exactly as it always was -- finished runs and the
            # scripts that read them (sweep_stop_threshold.py) match on those
            # strings -- so the canonical vocabulary lives beside it rather than
            # replacing it. See budget.terminated_code.
            terminated_by=terminated_code(terminated),
            video=str(recorder.path) if recorder.enabled else None,
            video_frames=int(recorder.frames),
            goal_distance=goal_probe.summary(),
        )
        if budget is not None:
            # Always present, like path_length_m: an episode's consumption in
            # discrete-step equivalents, and the translation/rotation split
            # (which is a finding, not bookkeeping -- policies spend ~50% of
            # their budget turning against ~36% for human demonstrations).
            record["budget"] = {
                **budget.summary(),
                "rule": getattr(ctx.config, "episode_budget", "steps"),
                "units_per_sim_second": (
                    budget.spent / ctx.executor.sim_time
                    if ctx.executor.sim_time > 0 else float("nan")
                ),
            }
        for probe in ctx.probes:
            record[probe.name] = probe.summary()
        if chunk is not None and chunk.extra:
            record["policy_extra"] = chunk.extra
        if stop_trace is not None:
            probs = [row[0] for row in stop_trace]
            record["stop"] = {
                # `acted`: did the policy's own head end this episode, as opposed
                # to the oracle or the step cap. Distinguishable from `terminated`
                # only because both can be present in one run's records.
                "acted": record["terminated"] == "policy called stop",
                # Fired where the oracle disagreed and was discarded. Only ever
                # nonzero under --stop-requires-oracle; it is the head's false
                # positive count on this episode's own trajectory.
                "suppressed": stop_suppressed,
                "steps": len(probs),
                "max_prob": max(probs),
                "max_prob_step": int(np.argmax(probs)),
                "final_prob": probs[-1],
                # (prob, logit, path_length_m) per step. The logit is what a
                # post-hoc temperature refit needs, the probability is what a
                # threshold sweep needs, and the path length is what makes the
                # swept SPL exact rather than an estimate.
                "trace": stop_trace,
            }
        if pose_trace is not None:
            record["pose_injected"] = {
                "steps": len(pose_trace),
                # Relative to this episode's first observation, so row 0 is
                # exactly (0, 0, 0) -- a cheap end-to-end assertion that the
                # frame convention survived the trip.
                "first": pose_trace[0],
                "last": pose_trace[-1],
                "trace": pose_trace,
            }
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
