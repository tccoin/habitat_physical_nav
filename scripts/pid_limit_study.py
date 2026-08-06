"""Can the inference PID execute the chunks the corpus is made of?

The training demonstrations were not produced by the controller that executes
policy output at evaluation time. ``generate_pose2d_dataset.py`` drives the
base through ``HolonomicRobotController.step``, which writes
``clip(heading_error * gain, +/-max_vw)`` straight into the chassis twist --
no angular acceleration limit on the body command at all. ``PIDPoseController``
ramps at ``aw_max * dt``. This script measures what that costs, by replaying
**real corpus action chunks** through the **real controller**.

The plant is stubbed, and that is sound rather than a shortcut: the controller
emits ``twist_frame="world"`` and ``HabitatRobotSim`` integrates it
kinematically, so ``pose += twist * dt`` is exactly the simulator's own update
for collision-free motion. What the stub drops is geometry -- a stubbed robot
never hits a wall. That makes this an *upper bound* on tracking quality: the
real robot can only do worse. A limit that fails to track even here fails a
fortiori in a scene.

    python scripts/pid_limit_study.py --episodes 40
    python scripts/pid_limit_study.py --presets baseline,demo_half,demo_matched

Reports, per preset, the fraction of commanded heading actually delivered and
how often each limit binds. Compare presets; the absolute numbers inherit the
no-geometry caveat above.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_DATASET = "/Projects/spatial_training/data/v2_25hz/formatted_pose"


class KinematicBase:
    """The subset of ``HabitatRobotSim`` that ``PIDPoseController`` touches.

    World-frame twist in, world-frame pose out. Joint *velocities* read back as
    the last commanded twist, which is what the real prismatic/revolute world
    joints carry under kinematic integration.
    """

    def __init__(self, pose):
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        self.twist = np.zeros(3, dtype=np.float64)

    def get_2d_pose(self):
        return self.pose.copy()

    def get_joint_state(self, name, is_vel=False):
        index = {"joint_x": 0, "joint_y": 1, "joint_th": 2}.get(name)
        if index is None:
            return 0.0
        return float(self.twist[index] if is_vel else self.pose[index])

    def step(self, chassis_twist, twist_frame="world", dt=0.04, **_):
        twist = np.asarray(chassis_twist, dtype=np.float64)
        if twist_frame != "world":
            raise AssertionError(
                f"the study assumes the world-frame twist the PID emits, got {twist_frame!r}"
            )
        self.twist = twist
        self.pose = self.pose + twist * dt
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def replay_chunk(chunk, preset, dt, gap):
    """One observation's chunk, executed exactly as ``ChunkExecutor`` does it.

    Returns (commanded heading sweep, achieved heading sweep, saturation counts).
    """
    from continuous_demos.action_chunking import relative_to_pose
    from continuous_demos.pid_pose_controller import PIDPoseController, track_trajectory

    from objectnav_eval.control import build_pid_config

    anchor = np.zeros(3, dtype=np.float64)
    base = KinematicBase(anchor)
    controller = PIDPoseController(base, build_pid_config(preset))

    n = min(gap, len(chunk))
    setpoints = np.stack([relative_to_pose(anchor, chunk[j]) for j in range(n)])

    counts = {"v": 0, "w": 0, "a": 0, "aw": 0}
    previous_setpoint = anchor
    for setpoint in setpoints:
        segment = track_trajectory(
            base, controller, setpoint[None, :], dt=dt, initial_pose=previous_setpoint
        )
        for key in counts:
            counts[key] += int(segment["saturation_counts"].get(key, 0))
        previous_setpoint = setpoint

    commanded = float(abs(wrap(setpoints[-1][2] - anchor[2])))
    achieved = float(abs(wrap(base.pose[2] - anchor[2])))
    lag = float(np.linalg.norm(base.pose[:2] - setpoints[-1][:2]))
    return commanded, achieved, lag, counts, n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="validation")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--gap", type=int, default=5)
    p.add_argument("--presets", default="baseline,demo_half,demo_matched")
    p.add_argument("--turn-threshold", type=float, default=0.05,
                   help="rad of commanded sweep below which a chunk is not a turn; "
                        "the ratio is meaningless on chunks that barely rotate")
    args = p.parse_args()

    from datasets import load_from_disk

    data = load_from_disk(args.dataset)
    rows = data[args.split] if hasattr(data, "keys") else data
    chunks = []
    for i in range(min(args.episodes, len(rows))):
        chunks.extend(np.asarray(rows[i]["action_chunks"], dtype=np.float64))
    print(f"{len(chunks)} chunks from {min(args.episodes, len(rows))} episodes, "
          f"dt={args.dt} gap={args.gap}\n")

    header = (f"{'preset':<14}{'turn kept':>11}{'p10':>8}{'lag m':>9}"
              f"{'aw sat':>9}{'w sat':>8}{'a sat':>8}")
    print(header)
    print("-" * len(header))
    for preset in args.presets.split(","):
        preset = preset.strip()
        ratios, lags, sat, ticks = [], [], {"v": 0, "w": 0, "a": 0, "aw": 0}, 0
        for chunk in chunks:
            commanded, achieved, lag, counts, n = replay_chunk(
                chunk, preset, args.dt, args.gap
            )
            for key in sat:
                sat[key] += counts[key]
            ticks += n
            lags.append(lag)
            if commanded >= args.turn_threshold:
                ratios.append(achieved / commanded)
        ratios = np.asarray(ratios)
        print(f"{preset:<14}{ratios.mean():>10.3f} {np.percentile(ratios, 10):>7.3f} "
              f"{np.mean(lags):>8.4f} {sat['aw'] / ticks:>8.1%} "
              f"{sat['w'] / ticks:>7.1%} {sat['a'] / ticks:>7.1%}")

    print(f"\nturning chunks: {len(ratios)}/{len(chunks)} "
          f"(commanded sweep >= {args.turn_threshold} rad)")
    print("'turn kept' = achieved / commanded heading sweep over the chunk, "
          "1.0 = perfect.\nNo geometry in this plant, so these are upper bounds.")


if __name__ == "__main__":
    main()
