"""Drawing the path to follow into the RGB frame itself, by projecting it.

The cheap alternative to compositing a top-down panel: the planned path and the goal are
projected through the camera that rendered the frame and stroked on with cv2, so the frame
stays the render's own size and nothing is rasterised per tick.

------------------------------------------------------------------------
Why not ``HabitatRobotSim.draw_path`` / ``draw_axes``
------------------------------------------------------------------------
Those are the obvious answer and they do not work for this. Habitat's
``DebugLineRender`` draws into the interactive viewer's render pass, **not** into
offscreen sensor observations. Measured on ``17DRP5sb8fy``: a ``draw_path`` and a
``draw_axes`` issued immediately before ``get_obs()`` change **0** pixels of the returned
observation, while ``sim.navmesh_visualization = True`` (which is a scene-graph drawable,
not a debug line) changes 1771 in the same frame.

That is also why ``replay.record_path``'s ``debug_draw`` branch has never actually put
its axes or its purple path into a recorded frame -- the BEV panel it hstacks is what
carries the information. Nothing downstream depended on the lines, so nothing broke; the
calls are simply inert for a sensor render.

Projecting is cheaper than the alternative anyway: no scene-graph mutation, no physics
objects to add and remove per segment, and it works identically for any sensor in a rig.

------------------------------------------------------------------------
What is and is not modelled
------------------------------------------------------------------------
Perspective and clipping are; **occlusion is not**. The path is stroked over whatever is
in front of it, so a leg heading through a wall into the next room is drawn on the wall.
For a debug overlay that is the more useful behaviour -- an occluded path is invisible
exactly when you most want to see where it went -- but it means the overlay cannot be
read as "the robot can see this".
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

#: The overlay's palette, matching `topdown.py` so magenta means the same thing in both.
PATH_COLOR = (206, 62, 196)
GOAL_COLOR = (218, 44, 44)
HORIZON_COLOR = (120, 120, 128)

#: Points closer than this to the image plane are clipped: at the apex the projection
#: diverges, and a single point behind the camera drags a line across the whole frame.
NEAR_CLIP_M = 0.05


def camera_matrix(robot_sim, sensor_uuid: str = "color_sensor"):
    """The sensor's world transform (4x4) and its ``(width, height, hfov_deg)``.

    Read live off the scene graph rather than reconstructed from the base pose: the
    sensor is re-parented onto a robot link at setup and carries its own fixed offset, so
    its world transform is the link chain times that offset. Returns ``None`` if the
    sensor is not present -- against a rig, ``color_sensor`` does not exist and the caller
    must name one (``color_sensor_0``).
    """
    agent = robot_sim.sim.get_agent(robot_sim.agent_id)
    node = agent.scene_node.subtree_sensors.get(sensor_uuid)
    if node is None:
        return None
    spec = node.specification()
    resolution = spec.resolution  # (height, width)
    return (
        np.asarray(node.object.absolute_transformation(), dtype=float),
        (int(resolution[1]), int(resolution[0]), float(spec.hfov)),
    )


def project_points(
    points_world: Any, cam_to_world: Any, width: int, height: int, hfov_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Habitat world ``(N, 3)`` -> pixel ``(N, 2)`` plus an in-front-of-camera mask.

    Habitat cameras look down their own ``-z`` with ``+x`` right and ``+y`` up (the
    ``habitat_sim.geo`` convention), and ``hfov`` is horizontal. Pixels are square, so one
    focal length serves both axes; the vertical field of view follows from the aspect
    ratio rather than being specified.

    The returned mask is ``True`` for points genuinely in front of the camera. Points
    behind it project to a mirrored position that looks completely plausible, so a caller
    that ignores the mask draws a path doubling back through the viewer.
    """
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    world_to_cam = np.linalg.inv(np.asarray(cam_to_world, dtype=float))
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    cam = (world_to_cam @ homogeneous.T).T[:, :3]

    depth = -cam[:, 2]  # camera looks down -z, so a point ahead has negative z
    in_front = depth > NEAR_CLIP_M
    safe = np.where(in_front, depth, 1.0)

    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u = width / 2.0 + focal * cam[:, 0] / safe
    v = height / 2.0 - focal * cam[:, 1] / safe
    return np.stack([u, v], axis=1), in_front


def _visible_runs(pixels: np.ndarray, in_front: np.ndarray, width: int, height: int):
    """Split a projected polyline into runs of consecutive in-front points.

    A run is broken wherever the path passes behind the camera, so the two halves are not
    joined by a line across the frame. Runs entirely off-screen are dropped; a run partly
    off-screen is kept, because cv2 clips it correctly and the on-screen part is the point.
    """
    margin = 4 * max(width, height)
    runs, current = [], []
    for i, ok in enumerate(in_front):
        if ok:
            current.append(pixels[i])
        elif current:
            runs.append(np.asarray(current))
            current = []
    if current:
        runs.append(np.asarray(current))
    out = []
    for run in runs:
        if run.shape[0] < 2:
            continue
        if (run[:, 0].max() < -margin or run[:, 0].min() > width + margin
                or run[:, 1].max() < -margin or run[:, 1].min() > height + margin):
            continue
        out.append(np.round(run).astype(np.int32))
    return out


def draw_path_in_view(
    rgb: np.ndarray,
    path_world: Any,
    cam_to_world: Any,
    hfov_deg: float,
    color: Sequence[int] = PATH_COLOR,
    thickness: int = 3,
    lift_m: float = 0.05,
) -> np.ndarray:
    """Stroke a world-space polyline onto a copy of ``rgb``.

    ``lift_m`` raises the line off the floor so it is not z-fought into the geometry it
    lies on -- there is no depth test here, but a path drawn exactly at floor level reads
    as a texture rather than as a path.
    """
    import cv2

    out = rgb.copy()
    pts = np.asarray(path_world, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return out
    lifted = pts + np.array([0.0, lift_m, 0.0])
    height, width = out.shape[:2]
    pixels, in_front = project_points(lifted, cam_to_world, width, height, hfov_deg)
    for run in _visible_runs(pixels, in_front, width, height):
        cv2.polylines(out, [run], False, tuple(int(c) for c in color), thickness,
                      lineType=cv2.LINE_AA)
    return out


def draw_goal_in_view(
    rgb: np.ndarray,
    goal_world: Any,
    cam_to_world: Any,
    hfov_deg: float,
    color: Sequence[int] = GOAL_COLOR,
    post_height_m: float = 1.0,
    radius_m: float = 0.25,
) -> np.ndarray:
    """Stroke a vertical post and a ground ring at the goal onto a copy of ``rgb``.

    A post rather than a dot: a dot at floor level is a few pixels once the goal is more
    than a couple of metres away, and disappears entirely when it falls below the bottom
    of the frame. The ring is the goal-reached radius, so "is the robot inside it" is
    readable rather than inferred from the distance readout.
    """
    import cv2

    out = rgb.copy()
    goal = np.asarray(goal_world, dtype=float).reshape(3)
    height, width = out.shape[:2]
    rgb_color = tuple(int(c) for c in color)

    post = np.stack([goal, goal + np.array([0.0, post_height_m, 0.0])])
    pixels, in_front = project_points(post, cam_to_world, width, height, hfov_deg)
    if in_front.all():
        cv2.line(out, tuple(np.round(pixels[0]).astype(int)),
                 tuple(np.round(pixels[1]).astype(int)), rgb_color, 3, cv2.LINE_AA)

    angles = np.linspace(0, 2 * math.pi, 25)
    ring = goal + np.stack(
        [radius_m * np.cos(angles), np.zeros_like(angles), radius_m * np.sin(angles)],
        axis=1,
    )
    ring_px, ring_front = project_points(ring, cam_to_world, width, height, hfov_deg)
    for run in _visible_runs(ring_px, ring_front, width, height):
        cv2.polylines(out, [run], False, rgb_color, 2, lineType=cv2.LINE_AA)
    return out


def annotate_view(
    rgb: np.ndarray,
    robot_sim,
    *,
    path_world: Optional[Any] = None,
    goal_world: Optional[Any] = None,
    sensor_uuid: str = "color_sensor",
    goal_radius_m: float = 0.5,
) -> np.ndarray:
    """Path and goal drawn into one frame, reading the camera off the live simulator.

    The whole overlay for one tick. Returns ``rgb`` unchanged if the sensor cannot be
    found, so a rig without a plain ``color_sensor`` degrades to no overlay rather than
    taking down the run.
    """
    camera = camera_matrix(robot_sim, sensor_uuid)
    if camera is None:
        return rgb
    cam_to_world, (width, height, hfov) = camera
    out = rgb
    if path_world is not None:
        out = draw_path_in_view(out, path_world, cam_to_world, hfov)
    if goal_world is not None:
        out = draw_goal_in_view(out, goal_world, cam_to_world, hfov,
                                radius_m=goal_radius_m)
    return out
