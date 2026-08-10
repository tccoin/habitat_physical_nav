"""Separate "does it move well" from "does it move toward the goal".

The hypothesis this exists to test: a model co-trained on ObjectNav + PointNav has picked
up better low-level movement from PointNav -- weaving between obstacles, carrying speed
through corridors -- but applies ObjectNav's high-level behaviour to it, wandering the room
before eventually heading for the goal instead of making a beeline.

That is testable precisely because **a point goal is known**. For ObjectNav, exploration is
the correct policy: the target's location is not given, so covering ground is progress. For
PointNav the goal pose is handed to the model every step, so any motion that does not
reduce the distance to it is pure loss. The two halves therefore separate cleanly:

  MOVEMENT QUALITY -- already measured by the existing probes and reported per episode:
    collisions_per_m, obstructed_frac, obstructed_while_driving_frac,
    saturated_v_frac, frozen_frac, tracking undershoot.
    These say nothing about *where* the robot went.

  GOAL-DIRECTEDNESS -- what this script adds, computed offline from the recorded trace:

    progress_ratio   net geodesic distance closed, over distance travelled.
                     1.0 is a beeline; 0.0 is motion that gets no nearer.
                     Defined for FAILED episodes too, which is the point -- SPL is 0 for
                     every failure and so cannot distinguish "wandered and gave up" from
                     "drove straight at it and stopped a metre short".

    descent_frac     fraction of ticks whose step reduced the geodesic distance to the
                     current goal. A beeline is ~1.0; aimless motion is ~0.5.

    alignment        mean cos(angle) between the step taken and the step that would have
                     reduced geodesic distance fastest. Per-tick, so it is unaffected by
                     whether the episode ultimately succeeded.

    revisit_frac     fraction of ticks spent in a grid cell the robot had already
                     occupied. The signature of exploration: a beeline never revisits.

    detour_ratio     distance travelled over the initial geodesic. The coarse version of
                     progress_ratio, kept because it is the number people quote.

Geodesic distances come from the scene's own pathfinder, queried at the ROBOT navmesh
height (`robot_pose.navmesh_height`), so they mean the same thing as the goals the
generator sampled. Queries are subsampled (`--stride`) because a 1500-tick episode would
otherwise be 1500 pathfinder calls.

    python scripts/analyse_goal_directedness.py --traces 'dump/*.json' \
        --scene-root /Projects/data/scene_datasets
"""

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    find_scene_dataset_config,
    resolve_scene_path,
)
from continuous_demos.pointnav_episode import plan_path
from continuous_demos.robot_pose import navmesh_height


def geodesic(pathfinder, xy, goal_xy, height):
    """Geodesic distance from a planar point to a planar goal, or inf."""
    start = np.array([float(xy[0]), height, -float(xy[1])], dtype=np.float32)
    end = np.array([float(goal_xy[0]), height, -float(goal_xy[1])], dtype=np.float32)
    _points, d = plan_path(pathfinder, start, end)
    return d


def analyse_episode(pathfinder, height, trace, goals, stride=5, cell_m=0.5):
    """Goal-directedness metrics for one rollout's trace.

    ``trace`` is ``[[X, Y, theta, goal_index], ...]`` per control tick; ``goals`` maps a
    goal index to its planar ``[X, Y, theta]``.
    """
    trace = np.asarray(trace, dtype=float)
    if trace.shape[0] < 2:
        return None

    xy = trace[:, :2]
    idx = trace[:, 3].astype(int)
    travelled = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

    # --- revisit fraction: exploration leaves the same cell twice ------------------
    cells = np.floor(xy / cell_m).astype(int)
    seen = set()
    revisits = 0
    for c in map(tuple, cells):
        if c in seen:
            revisits += 1
        seen.add(c)
    revisit_frac = revisits / len(cells)

    # --- geodesic trace, subsampled ------------------------------------------------
    samples = list(range(0, len(trace), stride))
    if samples[-1] != len(trace) - 1:
        samples.append(len(trace) - 1)
    dists, kept = [], []
    for i in samples:
        g = goals.get(int(idx[i]))
        if g is None:
            continue
        d = geodesic(pathfinder, xy[i], g, height)
        if math.isfinite(d):
            dists.append(d)
            kept.append(i)
    if len(dists) < 2:
        return None
    dists = np.asarray(dists)

    # Only compare consecutive samples that pursue the SAME goal: a goal change makes the
    # distance jump for reasons that have nothing to do with the policy's motion.
    same = np.array([idx[kept[k]] == idx[kept[k - 1]] for k in range(1, len(kept))])
    deltas = np.diff(dists)
    descent_frac = float((deltas[same] < 0).mean()) if same.any() else float("nan")

    # Net progress per metre travelled, summed over goals so a chained episode is not
    # penalised for the discontinuity at each handover.
    net = 0.0
    for g in sorted(set(idx.tolist())):
        rows = [k for k, i in enumerate(kept) if idx[i] == g]
        if len(rows) >= 2:
            net += float(dists[rows[0]] - dists[rows].min())
    progress_ratio = net / travelled if travelled > 1e-6 else float("nan")

    # --- per-tick alignment with the descent direction ----------------------------
    aligns = []
    for k in range(1, len(kept)):
        if not same[k - 1]:
            continue
        i0, i1 = kept[k - 1], kept[k]
        step = xy[i1] - xy[i0]
        n = np.linalg.norm(step)
        if n < 1e-6:
            continue
        g = goals.get(int(idx[i0]))
        pts, d = plan_path(
            pathfinder,
            np.array([xy[i0][0], height, -xy[i0][1]], dtype=np.float32),
            np.array([g[0], height, -g[1]], dtype=np.float32),
        )
        if pts is None or len(pts) < 2:
            continue
        want = np.array([pts[1][0] - pts[0][0], -(pts[1][2] - pts[0][2])])
        wn = np.linalg.norm(want)
        if wn < 1e-6:
            continue
        aligns.append(float(step @ want / (n * wn)))

    first = goals.get(int(idx[0]))
    d0 = geodesic(pathfinder, xy[0], first, height) if first is not None else float("nan")
    return {
        "ticks": int(len(trace)),
        "travelled_m": round(travelled, 3),
        "initial_geodesic_m": round(float(d0), 3) if math.isfinite(d0) else None,
        "detour_ratio": round(travelled / d0, 3) if math.isfinite(d0) and d0 > 1e-6 else None,
        "progress_ratio": round(progress_ratio, 4),
        "descent_frac": round(descent_frac, 4) if descent_frac == descent_frac else None,
        "alignment": round(float(np.mean(aligns)), 4) if aligns else None,
        "revisit_frac": round(revisit_frac, 4),
    }


def load_summary(path):
    blob = json.loads(Path(path).read_text())
    trace = blob.get("trace")
    goals = {int(g["index"]): g["goal2d"] for g in blob.get("goals", [])
             if g.get("goal2d") is not None}
    return blob, trace, goals


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--traces", required=True,
                   help="glob of --metrics-json files written by pointnav_rollout.py")
    p.add_argument("--scene-root", required=True)
    p.add_argument("--stride", type=int, default=5,
                   help="sample every Nth tick for the pathfinder queries")
    p.add_argument("--cell-m", type=float, default=0.5,
                   help="grid size for the revisit fraction")
    p.add_argument("--label-from", default="run",
                   help="summary key to group rows by (e.g. 'run' or 'ckpt')")
    p.add_argument("--out", default=None, help="write the per-episode rows as JSON")
    args = p.parse_args(argv)

    files = sorted(glob.glob(args.traces))
    if not files:
        raise SystemExit(f"no files matched {args.traces!r}")

    by_scene = defaultdict(list)
    for f in files:
        blob, trace, goals = load_summary(f)
        if not trace or not goals:
            print(f"  skipping {Path(f).name}: no trace or no goals recorded")
            continue
        by_scene[blob.get("scene", "?")].append((f, blob, trace, goals))

    rows = []
    for scene_id, entries in by_scene.items():
        scene_path = resolve_scene_path(scene_id, args.scene_root)
        sim = build_robot_sim(
            scene_path=scene_path,
            scene_dataset_config=find_scene_dataset_config(scene_path, args.scene_root),
            width=64, height=48, urdf_path=str(DEFAULT_URDF_PATH),
        )
        try:
            height = navmesh_height(sim)
            for f, blob, trace, goals in entries:
                m = analyse_episode(sim.sim.pathfinder, height, trace, goals,
                                    stride=args.stride, cell_m=args.cell_m)
                if m is None:
                    continue
                m["file"] = Path(f).name
                m["scene"] = Path(scene_id).stem
                m["label"] = str(blob.get(args.label_from, "?"))
                rows.append(m)
        finally:
            sim.sim.close(destroy=True)

    if not rows:
        raise SystemExit("nothing analysable")

    keys = ("progress_ratio", "descent_frac", "alignment", "revisit_frac",
            "detour_ratio", "travelled_m")
    groups = defaultdict(list)
    for r in rows:
        groups[r["label"]].append(r)

    width = max(len(k) for k in groups) + 2
    print(f"\n{'label':<{width}}{'n':>4}" + "".join(f"{k:>16}" for k in keys))
    for label, rs in sorted(groups.items()):
        line = f"{label:<{width}}{len(rs):>4}"
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            line += f"{np.mean(vals):>16.3f}" if vals else f"{'-':>16}"
        print(line)
    print("\nprogress_ratio 1.0 = beeline, 0.0 = no ground gained per metre driven")
    print("descent_frac   1.0 = every step got nearer, 0.5 = aimless")
    print("alignment      1.0 = always moving along the geodesic, 0.0 = orthogonal")
    print("revisit_frac   0.0 = never re-enters a cell, high = exploring/circling")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
