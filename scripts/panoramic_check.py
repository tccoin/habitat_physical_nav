"""Empirical proof, in pixels, that the panoramic rig points where it claims.

``continuous_demos.panoramic`` asserts that camera ``k`` is the forward camera
yawed by ``psi_k = -2*pi*k/N``, i.e. that its heading is ``joint_th + psi_k``
and that the index therefore runs CLOCKWISE seen from above. ``tests/
test_panoramic.py`` checks that claim as quaternion algebra. This script checks
the half the algebra cannot reach -- that habitat actually renders it -- by
exploiting the claim's own consequence:

    if camera k's heading is theta + psi_k, then rotating the BASE to
    theta + psi_k must make camera 0 render exactly what camera k rendered at
    theta.

So the test is a pixel comparison of two renders of the same scene, and it is
falsifiable in the right way: a sign error swaps which camera matches, and a
wrong rotation axis matches none of them. The mismatched pairs are printed
alongside as the control, so "everything matches" cannot pass by rendering a
blank wall.

It also reports what the rig costs (ms per observation at N=1 vs N=5) and that
the rig survives both a plain ``reset()`` and a genuine ``reset(scene_id=...)``
scene switch -- which it must, because both destroy and rebuild the sensor
nodes the yaw offsets live on.

    python scripts/panoramic_check.py examples/17DRP5sb8fy.json.gz \
        --scene-root /path/to/scene_datasets --switch-scene mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb
"""

import argparse
import math
import time
from typing import Dict, List

import numpy as np

from continuous_demos import panoramic
from continuous_demos.panoramic import camera_mount_offset_xy as camera_mount_offset
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    extract_demo,
    find_scene_dataset_config,
    load_demo_objectnav,
    resolve_scene_path,
)


def spawn(robot_sim, position, heading: float, keep_camera_at=None) -> None:
    """Put the base at ``position`` facing ``heading`` (planar frame, rad).

    ``keep_camera_at`` is a planar ``(X, Y)`` the CAMERA is to be held at
    instead, the base being shifted to whatever position puts it there at this
    heading. That is what makes two renders at different headings comparable:
    the rig is mounted 0.1 m behind the yaw axis (see
    ``panoramic.camera_mount_offset_xy``), so turning the base otherwise moves
    the viewpoint as well as aiming it, and the resulting parallax swamps the
    thing being measured.
    """
    position = np.asarray(position, dtype=np.float32).copy()
    if keep_camera_at is not None:
        forward, left = camera_mount_offset(robot_sim)
        cos_t, sin_t = math.cos(heading), math.sin(heading)
        base_x = keep_camera_at[0] - (forward * cos_t - left * sin_t)
        base_y = keep_camera_at[1] - (forward * sin_t + left * cos_t)
        # planar (X, Y) -> habitat (hx, hz) = (X, -Y); height is left alone.
        position[0], position[2] = base_x, -base_y
    robot_sim.reset(
        robot_translation=position,
        snap_to_navmesh=False,
        vertical_offset=None,
        home_joint_positions={"joint_th": float(heading)},
    )
    # One zero-velocity tick so the physics link transforms the sensors hang off
    # are synced to the joint positions just written.
    robot_sim.step(chassis_twist=np.zeros(3), dt=0.04)


def camera_position(robot_sim, position, heading: float):
    """Planar ``(X, Y)`` the rig ends up at when spawned like this."""
    spawn(robot_sim, position, heading)
    forward, left = camera_mount_offset(robot_sim)
    base_x, base_y, _ = robot_sim.get_2d_pose()
    cos_t, sin_t = math.cos(heading), math.sin(heading)
    return (base_x + forward * cos_t - left * sin_t,
            base_y + forward * sin_t + left * cos_t)


def views(robot_sim, num_cameras: int) -> List[np.ndarray]:
    """Camera 0..N-1's RGB images, as float arrays, in camera-index order."""
    obs = robot_sim.get_obs()
    uuids = panoramic.observation_cameras(obs)
    assert len(uuids) == num_cameras, f"expected {num_cameras} cameras, got {uuids}"
    return [np.asarray(obs[u])[..., :3].astype(np.float64) for u in uuids]


def difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference, in 0-255 levels."""
    return float(np.abs(a - b).mean())


def check_directions(robot_sim, position, heading: float, num_cameras: int,
                     tolerance: float) -> bool:
    """The core proof. Returns True if every camera points where it claims."""
    print("\n=== Rotation direction: is camera k's heading joint_th + psi_k? ===")
    print(f"    {panoramic.describe_rig(num_cameras)}")
    eye = camera_position(robot_sim, position, heading)
    forward, left = camera_mount_offset(robot_sim)
    print(f"    rig mounted {forward:+.3f} m forward, {left:+.3f} m left of the yaw "
          f"axis; holding the\n    camera fixed at planar ({eye[0]:.3f}, {eye[1]:.3f}) "
          "so only the AIM changes")

    spawn(robot_sim, position, heading, keep_camera_at=eye)
    rig_views = views(robot_sim, num_cameras)

    # Camera 0 alone, re-rendered with the base turned to each psi_k.
    forward_at: Dict[int, np.ndarray] = {}
    for k, psi in enumerate(panoramic.camera_yaw_offsets(num_cameras)):
        spawn(robot_sim, position, heading + psi, keep_camera_at=eye)
        forward_at[k] = views(robot_sim, num_cameras)[0]

    print("\n    mean |pixel difference| between camera k at theta and camera 0 at "
          "theta + psi_j\n    (the diagonal is the claim; every other cell is the "
          "control that makes it falsifiable)\n")
    print("        cam k \\ psi_j" + "".join(
        f"{math.degrees(p):9.0f}d" for p in panoramic.signed_yaw_offsets(num_cameras)
    ))
    ok = True
    for k in range(num_cameras):
        row = [difference(rig_views[k], forward_at[j]) for j in range(num_cameras)]
        print(f"        camera {k}      " + "".join(f"{d:9.2f} " for d in row))
        on_claim = row[k]
        off_claim = min(d for j, d in enumerate(row) if j != k) if num_cameras > 1 else float("inf")
        if on_claim > tolerance:
            ok = False
            print(f"          ! camera {k} does NOT match the base turned to psi_{k} "
                  f"({on_claim:.2f} > {tolerance})")
        elif off_claim <= on_claim:
            ok = False
            print(f"          ! camera {k} matches some OTHER offset just as well "
                  f"({off_claim:.2f} <= {on_claim:.2f}) -- the scene is too "
                  "featureless for this to prove anything")
    if ok:
        print("\n    PASS: every camera k renders exactly what the forward camera "
              "renders with the\n          base turned to psi_k = -2*pi*k/N, and "
              "matches no other offset. Index 0\n          is forward and the index "
              "runs CLOCKWISE seen from above (camera 1 looks\n          to the "
              "robot's right).")
    else:
        print("\n    FAIL")
    return ok


def check_survives_reset(robot_sim, position, heading, num_cameras, tolerance,
                         switch_scene, scene_root) -> bool:
    """The rig must be re-formed after anything that rebuilds the sensor nodes."""
    print("\n=== Survival: is the rig still a rig after a reset / scene switch? ===")
    spawn(robot_sim, position, heading)
    before = views(robot_sim, num_cameras)
    ok = True

    spawn(robot_sim, position, heading)  # a second plain reset()
    after = views(robot_sim, num_cameras)
    worst = max(difference(a, b) for a, b in zip(before, after))
    print(f"    plain reset():                 {num_cameras} cameras, worst view "
          f"differs by {worst:.3f}")
    ok &= worst <= tolerance

    if switch_scene is not None:
        other = resolve_scene_path(switch_scene, scene_root)
        robot_sim.reset(
            scene_id=str(other),
            scene_dataset_config=find_scene_dataset_config(other, scene_root),
            robot_translation=np.zeros(3, dtype=np.float32),
            snap_to_navmesh=True,
        )
        elsewhere = views(robot_sim, num_cameras)
        print(f"    reset(scene_id={other.name}): {len(elsewhere)} cameras still "
              f"present and rendering")
        # Same falsifiable test as above, in the new scene: the yaw offsets have
        # to have been re-applied, not merely the sensors re-created.
        centre = robot_sim.sim.pathfinder.get_random_navigable_point()
        ok &= check_directions(robot_sim, centre, 0.4, num_cameras, tolerance)
    return bool(ok)


def measure_render_cost(scene_path, scene_dataset_config, width, height, urdf_path,
                        position, num_cameras: int, iterations: int) -> float:
    """Mean ms per ``get_obs()`` for a rig of ``num_cameras``.

    Builds its own simulator and closes it before returning: habitat-sim aborts
    the process at teardown if two ``Simulator``s are ever alive at once.
    """
    robot_sim = build_robot_sim(
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        width=width,
        height=height,
        urdf_path=urdf_path,
        num_cameras=num_cameras,
    )
    try:
        spawn(robot_sim, position, 0.0)
        for _ in range(5):  # warm up: first renders pay for GPU-side setup
            robot_sim.get_obs()
        start = time.perf_counter()
        for _ in range(iterations):
            robot_sim.get_obs()
        return 1000.0 * (time.perf_counter() - start) / iterations
    finally:
        robot_sim.sim.close(destroy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", help="Path to the .json.gz dataset file.")
    parser.add_argument("--episode-index", type=int, default=0,
                        help="Episode whose scene and start pose are used.")
    parser.add_argument("--scene-root", default=None)
    parser.add_argument("--panoramic", type=int, default=5, metavar="N",
                        help="Rig size to verify.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Mean absolute pixel difference (0-255 levels) that "
                             "still counts as the same render.")
    parser.add_argument("--switch-scene", default=None,
                        help="A second scene_id to switch to, to prove the rig "
                             "survives reset(scene_id=...).")
    parser.add_argument("--cost-iterations", type=int, default=100,
                        help="Renders to time per rig size. 0 skips the timing.")
    args = parser.parse_args()

    data = load_demo_objectnav(args.dataset_path)
    episode = data["episodes"][args.episode_index]
    scene_path = resolve_scene_path(episode["scene_id"], args.scene_root)
    scene_dataset_config = find_scene_dataset_config(scene_path, args.scene_root)
    path_3d, _headings, _actions = extract_demo(episode)
    position = path_3d[0]

    robot_sim = build_robot_sim(
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        width=args.width,
        height=args.height,
        urdf_path=args.urdf_path,
        num_cameras=args.panoramic,
    )
    try:
        ok = check_directions(robot_sim, position, 0.0, args.panoramic, args.tolerance)
        ok &= check_survives_reset(robot_sim, position, 0.0, args.panoramic,
                                   args.tolerance, args.switch_scene, args.scene_root)
    finally:
        robot_sim.sim.close(destroy=True)

    if args.cost_iterations:
        print(f"\n=== Render cost ({args.width}x{args.height}, "
              f"{args.cost_iterations} observations each) ===")
        costs = {}
        for n in (1, args.panoramic):
            costs[n] = measure_render_cost(
                scene_path, scene_dataset_config, args.width, args.height,
                args.urdf_path, position, n, args.cost_iterations,
            )
            print(f"    N={n}: {costs[n]:6.2f} ms/observation "
                  f"({costs[n] / n:5.2f} ms/camera)")
        if args.panoramic != 1:
            print(f"    N={args.panoramic} costs {costs[args.panoramic] / costs[1]:.2f}x "
                  f"a single camera (a perfectly linear rig would be "
                  f"{args.panoramic:.2f}x)")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
