"""Empirically verify the action/state ordering and axis convention of the
Habitat-web ObjectNav human demonstration `reference_replay` data.

Run:  python verify_replay_format.py <dataset.json.gz>

See REPLAY_DATA_FORMAT.md for the documented conclusions. This script is the
executable proof behind them; rerun it on a new dataset shard to re-confirm.
"""
import gzip
import json
import sys
from collections import Counter

import numpy as np


def yaw_from_quat(q):
    """Heading (rotation about the world up-axis y) from an xyzw quaternion."""
    x, y, z, w = q
    return np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))


def angdiff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def forward_vec(q):
    """World-frame forward direction (local -z) for xyzw quaternion q."""
    x, y, z, w = q
    return np.array([-(2 * (x * z + w * y)),
                     -(2 * (y * z - w * x)),
                     -(1 - 2 * (x * x + y * y))])


def main(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)

    episodes = data["episodes"]
    total_ordering_violations = 0
    fwd_steps, turnsL, turnsR, planar_dy = [], [], [], []

    for ep in episodes:
        replay = ep["reference_replay"]
        acts = [e["action"] for e in replay]
        pos = np.array([e["agent_state"]["position"] for e in replay], float)
        rot = np.array([e["agent_state"]["rotation"] for e in replay], float)
        yaw = np.array([yaw_from_quat(r) for r in rot])

        assert acts[0] == "STOP", f"entry 0 action was {acts[0]!r}, expected STOP"

        # Ordering hypothesis: the action in entry i is the action that PRODUCED
        # state s_i, i.e. it explains the transition s_{i-1} -> s_i.
        for i in range(1, len(replay)):
            dpos = float(np.linalg.norm(pos[i] - pos[i - 1]))
            dyaw = float(np.degrees(angdiff(yaw[i], yaw[i - 1])))
            moved, turned = dpos > 1e-4, abs(dyaw) > 1e-2
            a = acts[i]
            if a == "MOVE_FORWARD":
                # a blocked forward (collision) yields no displacement -> allowed
                if turned:
                    total_ordering_violations += 1
                elif moved:
                    fwd_steps.append(dpos)
                    planar_dy.append(abs(pos[i][1] - pos[i - 1][1]))
                    # forward = local -z of the PRECEDING state
                    f = forward_vec(rot[i - 1])
                    d = (pos[i] - pos[i - 1]) / dpos
                    assert np.dot(d, f) > 0.999, "forward is not local -z"
            elif a in ("TURN_LEFT", "TURN_RIGHT"):
                if moved:
                    total_ordering_violations += 1
                elif turned:
                    (turnsL if a == "TURN_LEFT" else turnsR).append(dyaw)
            elif a == "STOP":
                if moved or turned:
                    total_ordering_violations += 1

    print(f"episodes checked           : {len(episodes)}")
    print(f"ordering violations (total): {total_ordering_violations}")
    print(f"MOVE_FORWARD step size (m)  : mean={np.mean(fwd_steps):.4f} "
          f"max={np.max(fwd_steps):.4f}")
    print(f"|dy| during forward (m)     : max={np.max(planar_dy):.6f}  "
          f"(~0 confirms y is the up-axis)")
    print(f"TURN_LEFT  yaw (deg)        : mean={np.mean(turnsL):+.2f}")
    print(f"TURN_RIGHT yaw (deg)        : mean={np.mean(turnsR):+.2f}")
    print()
    if total_ordering_violations == 0:
        print("CONFIRMED: entry i holds (state s_i, action that produced s_i).")
        print("           entry 0 is the initial state (action=STOP placeholder).")
        print("           axes: y up, forward = local -z, ground plane = x-z.")
    else:
        print("Ordering hypothesis FAILED -- investigate before trusting the data.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "17DRP5sb8fy.json.gz")
