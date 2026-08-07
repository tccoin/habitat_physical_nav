"""The panoramic camera rig: ``N`` RGB cameras bolted rigidly to the base,
evenly spaced in yaw, all sharing the single camera's intrinsics.

``N = 1`` is the shipped rig and the default everywhere: one forward camera
with uuid ``"color_sensor"``, exactly as it has always been. Everything below
is what happens when a caller asks for more than one.

------------------------------------------------------------------------
Why a rig at all
------------------------------------------------------------------------
Measured over 4,000 training episodes, 44.4% of control ticks are turn-in-place
and 72% of all heading change happens at zero translation. That is not
navigation, it is *looking around*: a behaviour the human demonstrators had to
perform because they were driving through a single 79-degree window. Widen the
window and the need goes away. This module is the widening; ``--no-phases``
(see ``episode_io.demo_looking_slices``) is the removal of the behaviour. They
are independent, and either is useful without the other.

------------------------------------------------------------------------
Geometry, and which way the indices go
------------------------------------------------------------------------
Camera ``k`` is the forward camera yawed by

    psi_k = -2*pi*k / N        (radians, about the robot's up axis)

so **index 0 looks forward and the index increases CLOCKWISE seen from above**
-- each successive camera looks further to the robot's RIGHT. The rig is rigid
to the base: every camera yaws with ``joint_th``, none of them ever pitches,
and none of them moves relative to the others.

The sign convention follows the planar control frame (frame 3 in ``sim.py``:
theta CCW from +X), in which camera ``k``'s heading is ``theta + psi_k``. It is
proved rather than asserted: with the base yawed by ``psi_k``, camera 0 renders
the same image camera ``k`` rendered at ``theta`` -- see
``tests/test_panoramic.py`` for the pure-math half and ``scripts/panoramic_check.py``
for the in-simulator, pixel-level half.

Neighbour spacing is ``360/N`` degrees against an ``hfov`` of
:data:`CAMERA_HFOV_DEG` (79, read off ``episode_io.build_robot_sim`` -- it is
the rig's real intrinsic, not an assumption). Full 360-degree coverage
therefore needs ``ceil(360/79) = 5`` cameras, which at 72-degree spacing leaves
7 degrees of overlap between neighbours. :func:`cameras_for_full_coverage` and
:func:`neighbour_overlap_deg` compute both, so a change of ``hfov`` cannot
silently leave a blind sector behind.

------------------------------------------------------------------------
How it survives a scene switch
------------------------------------------------------------------------
The yaw offset rides on the *sensor spec* (``spec.yaw_offset_rad``), not on the
sensor node. ``HabitatRobotSim._setup_robot_and_sensors`` reads it every time it
re-parents the sensors to the robot's links, and a scene switch destroys and
rebuilds exactly those nodes -- so the rig is re-formed from its declaration
rather than needing to be re-applied by anybody. A spec without the attribute
yaws by zero, which is the single-camera default and is bit-identical to the
code that predates this module.

------------------------------------------------------------------------
Reading the observation
------------------------------------------------------------------------
:func:`rgb_view` turns whatever ``get_obs()`` returned into one RGB frame,
without its caller having to know how many cameras exist. With one camera that
is the forward image unchanged; with ``N`` it is the ``N`` views concatenated
horizontally in :func:`strip_order` -- **left to right as a panorama reads**,
i.e. leftmost is the most anticlockwise view and the forward view sits in the
middle (for ``N = 5``: cameras ``3, 4, 0, 1, 2`` = ``+144, +72, 0, -72, -144``
degrees). That is a cyclic roll of the index order, not a re-labelling: the
uuids and the yaw offsets are what they say they are.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: uuid of the single forward camera. This is the shipped name and the one
#: every existing consumer indexes ``get_obs()`` with; a one-camera rig keeps
#: it so that turning the option off is not merely equivalent to the old
#: behaviour but literally is it.
FORWARD_UUID = "color_sensor"

#: uuid of camera ``k`` of a multi-camera rig is ``f"{UUID_PREFIX}{k}"``.
#: Deliberately NOT the same name as the single-camera case: a consumer that
#: asks for ``"color_sensor"`` on a panoramic rig should get a ``KeyError``
#: rather than one arbitrary fifth of the field of view.
UUID_PREFIX = "color_sensor_"

#: Horizontal field of view of every camera in the rig, degrees. Confirmed by
#: reading ``episode_io.build_robot_sim``, which is the one place the shipped
#: camera has ever been configured, rather than assumed from the brief.
CAMERA_HFOV_DEG = 79.0

#: URDF link the rig is mounted on -- the same link the single camera has
#: always been mounted on, so the rig sits at the same height and the forward
#: camera is unmoved.
SENSOR_LINK = "base_camera_link"


def camera_uuids(num_cameras: int) -> Tuple[str, ...]:
    """Ordered sensor uuids for a rig of ``num_cameras``, index 0 = forward."""
    _check(num_cameras)
    if num_cameras == 1:
        return (FORWARD_UUID,)
    return tuple(f"{UUID_PREFIX}{k}" for k in range(num_cameras))


def camera_yaw_offsets(num_cameras: int) -> Tuple[float, ...]:
    """Yaw of each camera relative to the base's forward, radians.

    ``psi_k = -2*pi*k/num_cameras``: index 0 forward, index increasing
    clockwise from above (successively further to the robot's right). Values
    are returned in ``(-2*pi, 0]`` as computed, i.e. not wrapped -- use
    :func:`signed_yaw_offsets` for the ``(-pi, pi]`` view.
    """
    _check(num_cameras)
    return tuple(-2.0 * math.pi * k / num_cameras for k in range(num_cameras))


def signed_yaw_offsets(num_cameras: int) -> Tuple[float, ...]:
    """:func:`camera_yaw_offsets` wrapped to ``(-pi, pi]`` -- the human view.

    For ``N = 5`` this is ``0, -72, -144, +144, +72`` degrees. The half-open
    interval is closed at ``+pi`` so that the one camera that can land exactly
    opposite the forward one (``N`` even, ``k = N/2``) reports ``+180`` rather
    than ``-180``; they name the same direction.
    """
    wrapped = []
    for psi in camera_yaw_offsets(num_cameras):
        w = (psi + math.pi) % (2.0 * math.pi) - math.pi
        wrapped.append(math.pi if w == -math.pi else w)
    return tuple(wrapped)


def strip_order(num_cameras: int) -> Tuple[int, ...]:
    """Camera indices left-to-right for a panorama strip.

    A panorama reads left-to-right as the view sweeps rightwards, so the strip
    runs from the most anticlockwise camera to the most clockwise one with the
    forward camera in the middle. Because the index runs clockwise, that is
    simply the cyclic roll ``k = -(N//2) .. N-1-(N//2)`` taken mod ``N``:
    ``(3, 4, 0, 1, 2)`` for ``N = 5``, ``(0,)`` for ``N = 1``.
    """
    _check(num_cameras)
    half = num_cameras // 2
    return tuple(k % num_cameras for k in range(-half, num_cameras - half))


def cameras_for_full_coverage(hfov_deg: float = CAMERA_HFOV_DEG) -> int:
    """Fewest cameras whose fields of view cover the full 360 degrees."""
    if hfov_deg <= 0.0:
        raise ValueError(f"hfov_deg must be positive, got {hfov_deg}")
    return int(math.ceil(360.0 / float(hfov_deg)))


def neighbour_overlap_deg(
    num_cameras: int, hfov_deg: float = CAMERA_HFOV_DEG
) -> float:
    """Angular overlap between adjacent cameras, degrees. Negative = blind gap."""
    _check(num_cameras)
    if num_cameras == 1:
        return float(hfov_deg) - 360.0
    return float(hfov_deg) - 360.0 / num_cameras


def _check(num_cameras: int) -> None:
    if int(num_cameras) != num_cameras or num_cameras < 1:
        raise ValueError(f"num_cameras must be a positive integer, got {num_cameras!r}")


# ---------------------------------------------------------------------------
# Reading an observation
# ---------------------------------------------------------------------------
def observation_cameras(obs: Dict[str, Any]) -> List[str]:
    """The rig's uuids present in ``obs``, in camera-index order.

    ``[FORWARD_UUID]`` for a single-camera rig; ``["color_sensor_0", ...]`` for
    a panoramic one. Anything else in the observation (depth, semantics) is
    ignored.
    """
    if FORWARD_UUID in obs:
        return [FORWARD_UUID]
    indexed = []
    for key in obs:
        if key.startswith(UUID_PREFIX):
            suffix = key[len(UUID_PREFIX) :]
            if suffix.isdigit():
                indexed.append((int(suffix), key))
    return [key for _, key in sorted(indexed)]


def rgb_view(obs: Dict[str, Any]) -> np.ndarray:
    """One RGB frame from an observation, whatever rig produced it.

    Single camera -> the forward image, RGB, unchanged. Panoramic rig -> the
    views concatenated horizontally in :func:`strip_order`, so the result is
    ``(H, N*W, 3)`` and reads as a panorama with forward in the middle.

    This is the only thing the replay recorders need to know about the rig,
    which is why they call it instead of indexing ``obs["color_sensor"]``: the
    rig is a property of the simulator, not a mode of the replay.
    """
    uuids = observation_cameras(obs)
    if not uuids:
        raise KeyError(
            f"observation carries no RGB camera; expected {FORWARD_UUID!r} or "
            f"{UUID_PREFIX}<k>, got {sorted(obs)}"
        )
    if len(uuids) == 1:
        return np.asarray(obs[uuids[0]])[..., :3]
    return np.hstack(
        [np.asarray(obs[uuids[k]])[..., :3] for k in strip_order(len(uuids))]
    )


# ---------------------------------------------------------------------------
# Building the rig
# ---------------------------------------------------------------------------
def rgb_sensor_setup(
    width: int,
    height: int,
    num_cameras: int = 1,
    hfov_deg: float = CAMERA_HFOV_DEG,
    link_name: str = SENSOR_LINK,
) -> Dict[str, list]:
    """A ``sensor_setup`` for :class:`~continuous_demos.sim.HabitatRobotSim`.

    ``num_cameras=1`` returns exactly the single forward ``"color_sensor"`` the
    pipeline has always used. Larger values return that same camera plus its
    yawed copies, every one carrying the same resolution and ``hfov`` and
    differing only in ``spec.yaw_offset_rad`` (see the module docstring for how
    that attribute reaches the sensor node, and why it has to live on the spec).
    """
    # Imported here, not at module scope, so the rig's geometry can be reasoned
    # about (and unit-tested) in an environment with no habitat_sim -- the same
    # property the controllers and metrics modules have.
    import habitat_sim

    _check(num_cameras)
    uuids = camera_uuids(num_cameras)
    yaws = camera_yaw_offsets(num_cameras)

    specs = []
    for uuid, yaw in zip(uuids, yaws):
        spec = habitat_sim.sensor.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = habitat_sim.sensor.SensorType.COLOR
        spec.resolution = [height, width]
        spec.hfov = hfov_deg
        if yaw:
            # Left unset for the forward camera so a one-camera rig's spec is
            # indistinguishable from the pre-rig one.
            spec.yaw_offset_rad = float(yaw)
        specs.append(spec)
    return {link_name: specs}


def camera_mount_offset_xy(robot_sim, link_name: str = SENSOR_LINK) -> Tuple[float, float]:
    """Where the rig sits relative to the base's yaw axis, in the body frame.

    Returns ``(forward, left)`` metres. This is **not** zero: the TidyBot URDF
    mounts ``base_camera_link`` at ``xyz = (-0.1, 0, 1.1)``, so the rig is 10 cm
    BEHIND the axis ``joint_th`` turns about, and yawing the base therefore
    swings the cameras around a 0.1 m circle rather than spinning them in place.

    That matters to anyone comparing two renders taken at different base
    headings -- they differ by a real 0.12 m of parallax at 72 degrees, which in
    a cluttered indoor scene is tens of grey levels and looks exactly like a rig
    misalignment. ``scripts/panoramic_check.py`` compensates with this function
    and gets bit-identical images; without it, nothing.

    It does not matter to the rig itself: all ``N`` cameras share this one link
    origin, so they are exactly co-located and the panorama has no parallax
    between its own views.

    Read off the live simulator rather than restated from the URDF, so a change
    of robot cannot leave a stale constant behind.
    """
    robot = robot_sim.robot
    link_id = next(
        (i for i in range(robot.num_links) if robot.get_link_name(i) == link_name),
        None,
    )
    if link_id is None:
        raise ValueError(f"Link {link_name!r} not found on the robot.")
    camera = robot.get_link_scene_node(link_id).absolute_translation
    base_x, base_y, theta = robot_sim.get_2d_pose()
    # Habitat (hx, hy, hz) -> planar (X, Y) = (hx, -hz); then into the body frame.
    dx = float(camera[0]) - float(base_x)
    dy = -float(camera[2]) - float(base_y)
    cos_t, sin_t = math.cos(-theta), math.sin(-theta)
    return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)


def describe_rig(num_cameras: int, hfov_deg: float = CAMERA_HFOV_DEG) -> str:
    """One line naming the rig, its coverage, and whether it has a blind gap."""
    if num_cameras == 1:
        return (
            f"1 camera, hfov {hfov_deg:g} deg (forward only; "
            f"{cameras_for_full_coverage(hfov_deg)} would cover 360 deg)"
        )
    overlap = neighbour_overlap_deg(num_cameras, hfov_deg)
    coverage = (
        f"{overlap:.1f} deg neighbour overlap"
        if overlap >= 0.0
        else f"BLIND GAP of {-overlap:.1f} deg between neighbours"
    )
    return (
        f"{num_cameras} cameras, hfov {hfov_deg:g} deg, "
        f"{360.0 / num_cameras:.1f} deg apart -- {coverage}; "
        f"strip order {strip_order(num_cameras)}"
    )
