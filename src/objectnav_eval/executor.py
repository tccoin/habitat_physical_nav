"""How an action chunk becomes motion.

The policy emits a chunk of poses relative to where the robot is *now*; the
executor anchors them on the robot's live pose, converts them to global
setpoints and PID-tracks the first ``gap`` of them, one control tick each. The
rest of the chunk is discarded -- it would be superseded by the next
observation, exactly as in training.

Two details are load-bearing:

* **Anchoring on the current pose is what closes the loop.** Anchoring on a
  recorded or previously-commanded pose hides exactly the drift that
  closed-loop evaluation exists to measure.
* **Setpoints are fed one at a time**, with ``initial_pose`` set to the previous
  *setpoint*. That reproduces ``track_trajectory``'s own finite-difference
  feedforward exactly, and the controller carries its integral state across the
  calls, so the control the robot sees is identical to handing it the whole
  block -- while giving the caller a hook at every tick (for video frames,
  metric evaluation and the tracking diagnostic).

Everything the executor observes about tracking quality is handed out as
:class:`TickRecord`; it computes no statistics itself (see
:mod:`objectnav_eval.probes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class TickRecord:
    """One control tick: what was asked for, and what happened."""

    tick: int  # index within the episode
    chunk_index: int  # which policy step emitted it
    tick_in_chunk: int
    setpoint: np.ndarray  # global [X, Y, theta] commanded this tick
    previous_setpoint: np.ndarray  # the tick before (the anchor, for tick 0)
    pose: np.ndarray  # global pose reached
    previous_pose: np.ndarray
    sim_time: float
    saturation: Dict[str, int] = field(default_factory=dict)

    @property
    def commanded_displacement(self) -> float:
        return float(np.linalg.norm(self.setpoint[:2] - self.previous_setpoint[:2]))

    @property
    def achieved_displacement(self) -> float:
        return float(np.linalg.norm(self.pose[:2] - self.previous_pose[:2]))

    @property
    def lag(self) -> float:
        """Distance from where the robot is to where it was told to be."""
        return float(np.linalg.norm(self.pose[:2] - self.setpoint[:2]))


@dataclass
class ChunkExecution:
    """The result of executing one chunk."""

    chunk_index: int
    anchor: np.ndarray
    setpoints: np.ndarray
    poses: np.ndarray
    ticks: List[TickRecord]

    @property
    def end_lag(self) -> float:
        """How far the robot is from the chunk's last setpoint when the next
        observation arrives -- the per-chunk error the closed loop accumulates."""
        if not len(self.setpoints):
            return float("nan")
        return float(np.linalg.norm(self.poses[-1][:2] - self.setpoints[-1][:2]))


class ChunkExecutor:
    """Executes action chunks through the PID pose controller.

    ``gap`` is the training observation stride: at 20 Hz control with an
    observation every 5 ticks, ``gap=5`` gives 4 Hz observations. It must match
    what the checkpoint was trained on, or the robot executes each chunk for the
    wrong length of time and the policy is evaluated at a speed it never saw.
    """

    def __init__(self, robot_sim, controller, dt: float, gap: int):
        self.robot_sim = robot_sim
        self.controller = controller
        self.dt = float(dt)
        self.gap = int(gap)
        self.tick = 0
        self.sim_time = 0.0

    def reset(self) -> None:
        self.controller.reset()
        self.tick = 0
        self.sim_time = 0.0

    def execute(
        self,
        chunk: Any,
        chunk_index: int,
        on_tick: Optional[Callable[[TickRecord], None]] = None,
    ) -> ChunkExecution:
        from continuous_demos.action_chunking import relative_to_pose
        from continuous_demos.pid_pose_controller import track_trajectory

        poses = np.asarray(getattr(chunk, "poses", chunk), dtype=np.float64).reshape(-1, 3)
        gap = min(self.gap, len(poses))
        if gap == 0:
            raise ValueError("policy returned an empty action chunk")

        anchor = np.asarray(self.robot_sim.get_2d_pose(), dtype=np.float64)
        setpoints = np.stack([relative_to_pose(anchor, poses[j]) for j in range(gap)])

        reached = np.zeros((gap, 3), dtype=np.float64)
        ticks: List[TickRecord] = []
        previous_setpoint = anchor
        previous_pose = anchor
        for j, setpoint in enumerate(setpoints):
            segment = track_trajectory(
                self.robot_sim,
                self.controller,
                setpoint[None, :],
                dt=self.dt,
                initial_pose=previous_setpoint,
            )
            pose = np.asarray(segment["actual_poses"][-1], dtype=np.float64)
            reached[j] = pose
            self.tick += 1
            self.sim_time += self.dt
            record = TickRecord(
                tick=self.tick,
                chunk_index=chunk_index,
                tick_in_chunk=j,
                setpoint=setpoint,
                previous_setpoint=previous_setpoint,
                pose=pose,
                previous_pose=previous_pose,
                sim_time=self.sim_time,
                saturation=dict(segment["saturation_counts"]),
            )
            ticks.append(record)
            if on_tick is not None:
                on_tick(record)
            previous_setpoint = setpoint
            previous_pose = pose

        return ChunkExecution(
            chunk_index=chunk_index,
            anchor=anchor,
            setpoints=setpoints,
            poses=reached,
            ticks=ticks,
        )
