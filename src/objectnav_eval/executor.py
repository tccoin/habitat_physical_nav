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

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from objectnav_eval.navmesh import camera_pose


@dataclass
class ContactSnapshot:
    """What the robot was touching at the end of one control tick.

    ``ground`` and ``obstruction`` are separated because **the robot is always
    in contact with the floor** -- a healthy episode sits at roughly 6 broadphase
    pairs and 2 contact points with nothing wrong (measured in
    ``dump/drive_failure_detection/FINDINGS.md``), so a raw contact count is not
    a collision signal at all. The discriminator is the contact normal: standing
    on a floor produces a near-vertical normal, pressing into a wall or a chair
    leg produces a near-horizontal one.
    """

    total: int = 0
    ground: int = 0
    obstruction: int = 0
    max_obstruction_force: float = 0.0


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
    #: Only populated when the executor was built with ``collect_contacts``;
    #: probes never query the simulator themselves.
    contacts: Optional[ContactSnapshot] = None
    #: Vertical travel (the ``joint_z`` prismatic joint), metres, sampled by the
    #: executor for the same reason ``contacts`` is: probes do not query the
    #: simulator. Not in ``pose``, which is the planar ``[X, Y, theta]``. This
    #: is what :class:`~objectnav_eval.probes.EscapeProbe` watches -- the robot
    #: falling out of the world moves only in this coordinate.
    height: Optional[float] = None
    #: Habitat **world** pose of the camera, ``[x, y, z, qw, qx, qy, qz]``, or
    #: None when not being sampled. Not derivable from `pose` (planar control
    #: frame, and the base rather than the camera) nor from `height` (a joint
    #: value). This is the pose the rendered frame was taken from, mounting
    #: offset and camera pitch included. Sampled by the executor, like
    #: `contacts` and `height`, because probes must not touch the simulator.
    camera: Optional[List[float]] = None

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

    def commanded_body_delta(self) -> np.ndarray:
        """Commanded ``[forward, left, dtheta]`` in the robot's own frame.

        Not ``setpoint - previous_setpoint``: that is a displacement expressed in
        a *shared* frame, and on turn-heavy motion the two diverge badly because
        the heading rotates within the step. Composing in the previous frame is
        what ``dump/data_diagnostics/analyze_chunk_motion.py`` does, so the
        numbers derived from this are directly comparable to the corpus.
        """
        return _body_delta(self.previous_setpoint, self.setpoint)

    def achieved_body_delta(self) -> np.ndarray:
        """Achieved ``[forward, left, dtheta]`` in the robot's own frame."""
        return _body_delta(self.previous_pose, self.pose)


def _body_delta(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    theta = float(before[2])
    cos, sin = math.cos(theta), math.sin(theta)
    dx = float(after[0]) - float(before[0])
    dy = float(after[1]) - float(before[1])
    dtheta = float(after[2]) - float(before[2])
    dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
    return np.array([cos * dx + sin * dy, -sin * dx + cos * dy, dtheta])


#: A contact normal within this much of vertical is the robot resting on
#: something it is standing on rather than pressing into something in its way.
#: 45 degrees: the floor and a wall are the two cases that matter and they are
#: 90 degrees apart, so the split is nowhere near either.
_GROUND_NORMAL_COS = 0.7071067811865476


def robot_contacts(robot_sim) -> ContactSnapshot:
    """Classify this instant's contact points into ground vs obstruction.

    Reads ``sim.get_physics_contact_points()``, keeps the pairs with the robot on
    exactly one side (self-collision between the gantry's own links is not
    information about the scene), and splits them on the world-frame verticality
    of the contact normal.

    Honest limitations, because this is descriptive and not a gate:

    * A ramp or a stair tread steeper than 45 degrees reads as an obstruction,
      and a low sill the robot rides over reads as ground.
    * It is an *instantaneous* sample at the end of a tick. A glancing contact
      entirely inside a tick is invisible.
    * It says the robot is touching something, not that the something stopped
      it; pair it with the motion taxonomy's ``commanded_drive`` mask to get
      "was in contact while trying to move".
    """
    snapshot = ContactSnapshot()
    try:
        robot_id = int(robot_sim.robot.object_id)
        points = robot_sim.sim.get_physics_contact_points()
    except Exception:  # no physics, or a simulator that does not expose it
        return snapshot
    for point in points:
        a, b = int(point.object_id_a), int(point.object_id_b)
        if (a == robot_id) == (b == robot_id):
            continue  # neither side is the robot, or both are (self-collision)
        normal = point.contact_normal_on_b_in_ws
        vertical = abs(float(normal[1]))
        snapshot.total += 1
        if vertical >= _GROUND_NORMAL_COS:
            snapshot.ground += 1
        else:
            snapshot.obstruction += 1
            force = abs(float(point.normal_force))
            if force > snapshot.max_obstruction_force:
                snapshot.max_obstruction_force = force
    return snapshot


@dataclass
class ChunkExecution:
    """The result of executing one chunk."""

    chunk_index: int
    anchor: np.ndarray
    setpoints: np.ndarray
    poses: np.ndarray
    ticks: List[TickRecord]
    #: True when ``stop_when`` cut the chunk short. ``setpoints``/``poses`` are
    #: truncated to the ticks that actually ran, so every probe and diagnostic
    #: sees the executed trajectory and nothing else.
    stopped_early: bool = False
    #: The chunk exactly as the policy emitted it -- (T, 3) relative offsets,
    #: BEFORE truncation to `gap` and before anchoring. `setpoints` is what was
    #: executed; this is what was asked for, which is the thing you need to tell
    #: "the policy commanded something odd" from "the controller failed to
    #: track it". Carried by reference, so it costs nothing.
    chunk: Optional[Any] = None
    action_target_reached: Optional[bool] = None
    action_sim_time: float = 0.0

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

    def __init__(
        self,
        robot_sim,
        controller,
        dt: float,
        gap: int,
        collect_contacts: bool = False,
        log_camera: bool = False,
        sensor_uuid: str = "color_sensor",
    ):
        self.robot_sim = robot_sim
        self.controller = controller
        self.dt = float(dt)
        self.gap = int(gap)
        # Sampling the contact manifold is the one per-tick observation that
        # costs simulator time rather than arithmetic, so it is opt-in and the
        # executor -- not a probe -- is what reads it.
        self.collect_contacts = bool(collect_contacts)
        # Sampling the camera's world transform walks the scene-node chain; cheap,
        # but it IS a simulator read, so it is opt-in and driven by whether the
        # trajectory log wants it.
        self.log_camera = bool(log_camera)
        self.sensor_uuid = sensor_uuid
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
        stop_when: Optional[Callable[[TickRecord], bool]] = None,
    ) -> ChunkExecution:
        """Track ``chunk``'s first ``gap`` setpoints, one control tick each.

        ``stop_when`` is consulted after ``on_tick`` on every tick, and a true
        answer ends the chunk there. It exists for terminations the *episode*
        owns rather than the policy -- the oracle stop, where an imaginary
        observer ends the episode the moment the agent is close enough. Cutting
        at the tick rather than at the chunk boundary is what makes that a real
        oracle: stopping only at the boundary lets the robot arrive on tick 1
        and drift back out by tick 5, so ``success`` would sit below
        ``oracle_success`` even though the oracle had already fired.
        """
        from continuous_demos.action_chunking import relative_to_pose
        from continuous_demos.pid_pose_controller import track_trajectory

        poses = np.asarray(getattr(chunk, "poses", chunk), dtype=np.float64).reshape(
            -1, 3
        )
        if len(poses) == 0:
            raise ValueError("policy returned an empty action chunk")

        anchor = np.asarray(self.robot_sim.get_2d_pose(), dtype=np.float64)
        extra = getattr(chunk, "extra", {}) or {}
        completion = extra.get("action_target_completion")
        if completion is None:
            nominal_ticks = min(self.gap, len(poses))
            nominal_setpoints = np.stack(
                [relative_to_pose(anchor, poses[j]) for j in range(nominal_ticks)]
            )
            target_setpoint = nominal_setpoints[-1]
            max_ticks = nominal_ticks
            position_tolerance = yaw_tolerance = None
        else:
            nominal_ticks = len(poses)
            nominal_setpoints = np.stack(
                [relative_to_pose(anchor, pose) for pose in poses]
            )
            target_setpoint = relative_to_pose(anchor, poses[-1])
            max_time = float(completion["max_sim_time_s"])
            position_tolerance = float(completion["position_tolerance_m"])
            yaw_tolerance = math.radians(float(completion["yaw_tolerance_deg"]))
            if max_time <= 0 or position_tolerance < 0 or yaw_tolerance < 0:
                raise ValueError("invalid action target completion limits")
            max_ticks = int(math.floor(max_time / self.dt + 1e-9))
            if max_ticks <= 0:
                raise ValueError("action time limit is shorter than one control tick")

        camera_action = extra.get("camera_action")
        if camera_action is not None:
            self.robot_sim.apply_camera_action(
                camera_action,
                float(extra.get("camera_angle_deg", 30.0)),
            )

        reached = []
        executed_setpoints = []
        ticks: List[TickRecord] = []
        stopped_early = False
        target_reached = None if completion is None else False
        previous_setpoint = anchor
        previous_pose = anchor
        for j in range(max_ticks):
            setpoint = (
                nominal_setpoints[j] if j < len(nominal_setpoints) else target_setpoint
            )
            segment = track_trajectory(
                self.robot_sim,
                self.controller,
                setpoint[None, :],
                dt=self.dt,
                initial_pose=previous_setpoint,
            )
            pose = np.asarray(segment["actual_poses"][-1], dtype=np.float64)
            reached.append(pose)
            executed_setpoints.append(setpoint)
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
                contacts=(
                    robot_contacts(self.robot_sim) if self.collect_contacts else None
                ),
                # Unconditional, unlike contacts: one joint read per tick, and
                # the failure it detects is one that every other metric scores
                # as healthy.
                height=float(self.robot_sim.get_joint_state("joint_z")),
                camera=(
                    camera_pose(self.robot_sim, self.sensor_uuid)
                    if self.log_camera
                    else None
                ),
            )
            ticks.append(record)
            if on_tick is not None:
                on_tick(record)
            previous_setpoint = setpoint
            previous_pose = pose
            if stop_when is not None and stop_when(record):
                stopped_early = True
                break
            if completion is not None:
                position_error = float(np.linalg.norm(pose[:2] - target_setpoint[:2]))
                yaw_error = abs(
                    (float(pose[2] - target_setpoint[2]) + math.pi)
                    % (2 * math.pi)
                    - math.pi
                )
                target_reached = (
                    position_error <= position_tolerance
                    and yaw_error <= yaw_tolerance
                )
                if target_reached:
                    break

        # Truncate to what ran. `reached` is preallocated, so a short chunk
        # would otherwise hand every probe trailing rows of zeros -- read as a
        # jump to the origin by anything that looks at poses[-1].
        n = len(ticks)
        return ChunkExecution(
            chunk_index=chunk_index,
            anchor=anchor,
            setpoints=np.asarray(executed_setpoints, dtype=np.float64).reshape(-1, 3),
            poses=np.asarray(reached, dtype=np.float64).reshape(-1, 3),
            ticks=ticks,
            stopped_early=stopped_early,
            chunk=chunk,
            action_target_reached=target_reached,
            action_sim_time=n * self.dt,
        )
