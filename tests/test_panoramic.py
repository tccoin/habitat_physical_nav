"""The panoramic rig's pure geometry, and the phase-free path construction.

Neither needs a simulator: the rig's uuids, yaw offsets, coverage arithmetic
and strip order are arithmetic, the sensor-node orientation is quaternion
algebra done here in numpy against the same convention ``sim.py`` applies with
magnum, and the phase-free option is a choice of looking slices. What is left
over -- that habitat actually points the camera where the algebra says -- is
pixel-level and lives in ``scripts/panoramic_check.py``.
"""

import math

import numpy as np
import pytest

from continuous_demos import panoramic
from continuous_demos.drive_failure import near_stop_runs
from continuous_demos.episode_io import demo_looking_slices, find_looking_slices
from continuous_demos.replay import (
    MOVE_STALL_TICKS,
    PHASE_FREE_STALL_TICKS,
    _build_phases,
    _is_stalled,
)


# ---------------------------------------------------------------------------
# uuids and ordering
# ---------------------------------------------------------------------------
def test_single_camera_rig_keeps_the_shipped_uuid():
    # Turning the option off must not merely be equivalent to the old rig, it
    # must be the old rig: every existing consumer indexes this exact key.
    assert panoramic.camera_uuids(1) == ("color_sensor",)


def test_multi_camera_uuids_are_ordered_and_indexed_from_zero():
    assert panoramic.camera_uuids(5) == tuple(f"color_sensor_{k}" for k in range(5))


def test_multi_camera_rig_does_not_answer_to_the_single_camera_uuid():
    # A consumer asking for "color_sensor" on a 5-camera rig would otherwise get
    # one arbitrary fifth of the field of view and never know.
    assert panoramic.FORWARD_UUID not in panoramic.camera_uuids(5)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "3"])
def test_rig_size_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError):
        panoramic.camera_uuids(bad)


# ---------------------------------------------------------------------------
# yaw geometry
# ---------------------------------------------------------------------------
def test_camera_zero_is_forward():
    for n in (1, 2, 3, 5, 8):
        assert panoramic.camera_yaw_offsets(n)[0] == 0.0


def test_cameras_are_evenly_spaced_over_the_full_circle():
    for n in (2, 3, 5, 8):
        yaws = panoramic.camera_yaw_offsets(n)
        gaps = np.diff(np.asarray(yaws))
        assert np.allclose(gaps, -2.0 * math.pi / n)


def test_index_increases_clockwise_seen_from_above():
    # The documented direction: camera 1 looks to the robot's RIGHT, which in
    # the planar control frame (theta CCW from +X) is a NEGATIVE yaw offset.
    assert panoramic.camera_yaw_offsets(5)[1] < 0.0
    signed = panoramic.signed_yaw_offsets(5)
    assert np.allclose(np.degrees(signed), [0.0, -72.0, -144.0, 144.0, 72.0])


def test_signed_offsets_are_in_the_half_open_interval():
    for n in (1, 2, 3, 4, 5, 6, 8):
        for psi in panoramic.signed_yaw_offsets(n):
            assert -math.pi < psi <= math.pi


def test_strip_order_puts_forward_in_the_middle_sweeping_left_to_right():
    assert panoramic.strip_order(1) == (0,)
    assert panoramic.strip_order(5) == (3, 4, 0, 1, 2)
    # Read left to right, the strip's yaw offsets must DECREASE monotonically:
    # that is what makes it a panorama rather than a shuffle.
    signed = panoramic.signed_yaw_offsets(5)
    strip = [signed[k] for k in panoramic.strip_order(5)]
    assert strip == sorted(strip, reverse=True)


def test_strip_order_is_a_permutation():
    for n in (1, 2, 3, 4, 5, 8):
        assert sorted(panoramic.strip_order(n)) == list(range(n))


# ---------------------------------------------------------------------------
# coverage arithmetic -- so a change of hfov cannot leave a blind sector
# ---------------------------------------------------------------------------
def test_five_cameras_cover_360_at_this_repo_s_configured_hfov():
    # 79 degrees is what episode_io.build_robot_sim configures; it is read off
    # the code, not assumed, and this pins the consequence.
    assert panoramic.CAMERA_HFOV_DEG == 79.0
    assert panoramic.cameras_for_full_coverage() == 5
    assert panoramic.neighbour_overlap_deg(5) == pytest.approx(7.0)


def test_four_cameras_would_leave_a_blind_gap_at_79_degrees():
    assert panoramic.neighbour_overlap_deg(4) == pytest.approx(-11.0)
    assert "BLIND GAP" in panoramic.describe_rig(4)


def test_required_camera_count_tracks_hfov():
    assert panoramic.cameras_for_full_coverage(90.0) == 4
    assert panoramic.cameras_for_full_coverage(60.0) == 6
    assert panoramic.cameras_for_full_coverage(120.0) == 3


# ---------------------------------------------------------------------------
# sensor pose math
#
# The same algebra sim.py::_setup_robot_and_sensors applies with magnum, done
# here in numpy so it is checked without a simulator. The URDF link frame is
# x forward, y left, z up; the shipped sensor rotation is
# Ry(-90) * Rz(-90) and a rig camera left-multiplies Rz(psi) onto it.
# ---------------------------------------------------------------------------
def _axis_rotation(angle, axis):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _camera_axes(yaw_offset):
    """(forward, up) of a rig camera, in the URDF link frame."""
    shipped = _axis_rotation(-math.pi / 2, (0, 1, 0)) @ _axis_rotation(-math.pi / 2, (0, 0, 1))
    rotation = _axis_rotation(yaw_offset, (0, 0, 1)) @ shipped
    # A habitat camera looks along its own local -z and its up is local +y.
    return rotation @ np.array([0.0, 0.0, -1.0]), rotation @ np.array([0.0, 1.0, 0.0])


def test_shipped_camera_looks_along_the_robot_s_forward():
    forward, up = _camera_axes(0.0)
    assert np.allclose(forward, [1.0, 0.0, 0.0], atol=1e-12)  # link +x = forward
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-12)       # link +z = up


def test_a_yaw_offset_rotates_the_camera_about_the_robot_s_up_axis():
    for psi in (0.3, -0.3, 2.0, -2.0):
        forward, up = _camera_axes(psi)
        assert np.allclose(forward, [math.cos(psi), math.sin(psi), 0.0], atol=1e-12)
        # Yaw only: the camera never pitches or rolls, whatever the offset.
        assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-12)


def test_rig_camera_headings_are_base_heading_plus_offset():
    # The claim scripts/panoramic_check.py verifies in pixels: camera k points
    # at joint_th + psi_k. Here: its forward's bearing in the link frame (which
    # IS its offset from the base heading) equals psi_k.
    for k, psi in enumerate(panoramic.camera_yaw_offsets(5)):
        forward, _ = _camera_axes(psi)
        bearing = math.atan2(forward[1], forward[0])
        assert math.remainder(bearing - psi, 2 * math.pi) == pytest.approx(0.0, abs=1e-12)
        # ...and camera 1 is to the right, i.e. clockwise from above.
        if k == 1:
            assert forward[1] < 0.0


# ---------------------------------------------------------------------------
# reading an observation
# ---------------------------------------------------------------------------
def _fake_obs(uuids, height=2, width=3):
    return {
        uuid: np.full((height, width, 4), i + 1, dtype=np.uint8)
        for i, uuid in enumerate(uuids)
    }


def test_rgb_view_of_a_single_camera_is_the_forward_image_with_alpha_dropped():
    obs = _fake_obs(panoramic.camera_uuids(1))
    view = panoramic.rgb_view(obs)
    assert view.shape == (2, 3, 3)
    assert np.array_equal(view, np.asarray(obs["color_sensor"])[..., :3])


def test_rgb_view_of_a_rig_concatenates_horizontally_in_strip_order():
    uuids = panoramic.camera_uuids(5)
    obs = _fake_obs(uuids)
    view = panoramic.rgb_view(obs)
    assert view.shape == (2, 15, 3)
    # Each camera contributes a constant-valued block; read them back and check
    # the blocks appear in the documented order 3, 4, 0, 1, 2.
    blocks = [int(view[0, 3 * i, 0]) - 1 for i in range(5)]
    assert tuple(blocks) == panoramic.strip_order(5)


def test_rgb_view_ignores_non_rgb_sensors():
    obs = _fake_obs(panoramic.camera_uuids(1))
    obs["depth_sensor"] = np.zeros((2, 3), dtype=np.float32)
    assert panoramic.rgb_view(obs).shape == (2, 3, 3)


def test_rgb_view_reports_an_observation_with_no_camera():
    with pytest.raises(KeyError):
        panoramic.rgb_view({"depth_sensor": np.zeros((2, 3))})


def test_observation_camera_order_does_not_depend_on_dict_order():
    uuids = panoramic.camera_uuids(5)
    shuffled = {u: np.zeros((1, 1, 4), np.uint8) for u in reversed(uuids)}
    assert panoramic.observation_cameras(shuffled) == list(uuids)


# ---------------------------------------------------------------------------
# phase-free path construction
# ---------------------------------------------------------------------------
_DEMO_ACTIONS = (
    ["MOVE_FORWARD"] * 4
    + ["TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "LOOK_UP", "LOOK_DOWN"]  # a scan
    + ["MOVE_FORWARD"] * 3
    + ["TURN_RIGHT", "TURN_LEFT"]                                       # a peek
    + ["MOVE_FORWARD"] * 2
)


def test_phase_free_off_is_the_shipped_segmentation():
    assert demo_looking_slices(_DEMO_ACTIONS) == find_looking_slices(_DEMO_ACTIONS)
    assert len(demo_looking_slices(_DEMO_ACTIONS)) == 2  # the scan and the peek


def test_phase_free_on_discards_every_looking_slice():
    assert demo_looking_slices(_DEMO_ACTIONS, phase_free=True) == []


def test_phase_free_kwargs_still_reach_the_segmenter_when_off():
    # min_run=2 promotes nothing extra here but must be honoured, not swallowed.
    assert demo_looking_slices(_DEMO_ACTIONS, min_run=2) == find_looking_slices(
        _DEMO_ACTIONS, min_run=2
    )


def test_no_looking_slices_yields_exactly_one_move_phase_over_the_whole_path():
    n = len(_DEMO_ACTIONS)
    phases = _build_phases(n, demo_looking_slices(_DEMO_ACTIONS, phase_free=True))
    assert phases == [("move", 0, n)]


def test_that_single_move_phase_spans_every_demonstrated_position():
    # A move phase [p, q) follows path_3d[p : q + 1], so the one phase above
    # must cover all M + 1 path points of an M-action demo -- no point dropped
    # at either end.
    n = len(_DEMO_ACTIONS)
    path = np.arange(3 * (n + 1), dtype=np.float32).reshape(n + 1, 3)
    (_kind, p, q), = _build_phases(n, [])
    assert np.array_equal(path[p : q + 1], path)


def test_segmented_replay_still_breaks_the_same_demo_into_five_phases():
    # The default reading of the code is unchanged by the option existing.
    phases = _build_phases(len(_DEMO_ACTIONS), demo_looking_slices(_DEMO_ACTIONS))
    assert [kind for kind, _, _ in phases] == ["move", "look", "move", "look", "move"]


def test_phase_tiling_is_gapless_either_way():
    for slices in (demo_looking_slices(_DEMO_ACTIONS),
                   demo_looking_slices(_DEMO_ACTIONS, phase_free=True)):
        phases = _build_phases(len(_DEMO_ACTIONS), slices)
        assert phases[0][1] == 0
        assert phases[-1][2] == len(_DEMO_ACTIONS)
        for (_, _, end), (_, start, _) in zip(phases, phases[1:]):
            assert end == start


# ---------------------------------------------------------------------------
# the move-stall rule, which a single-phase replay must not be subject to
# ---------------------------------------------------------------------------
def test_phase_free_disables_the_move_stall_rule():
    # With one phase there is no phase to hand over to, so bailing out of a
    # stalled one abandons the rest of the demonstration rather than its tail.
    assert PHASE_FREE_STALL_TICKS == 0
    assert MOVE_STALL_TICKS != 0  # ...and the segmented default is untouched


def test_a_dead_phase_stalls_under_the_segmented_default():
    frozen_progress = [1.0] * (MOVE_STALL_TICKS + 1)
    no_speed = [0.0] * (MOVE_STALL_TICKS + 1)
    assert _is_stalled(frozen_progress, no_speed, MOVE_STALL_TICKS, 0.05, 0.02)


def test_the_same_dead_phase_never_stalls_with_the_window_at_zero():
    frozen_progress = [1.0] * 10_000
    no_speed = [0.0] * 10_000
    assert not _is_stalled(frozen_progress, no_speed, PHASE_FREE_STALL_TICKS, 0.05, 0.02)


# ---------------------------------------------------------------------------
# organic near-stops
# ---------------------------------------------------------------------------
def _poses(speeds, fps=25.0):
    """A straight-line trajectory whose per-tick speeds are ``speeds``."""
    xs = np.concatenate([[0.0], np.cumsum(np.asarray(speeds, float) / fps)])
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def test_no_near_stops_while_driving():
    assert near_stop_runs(_poses([1.0] * 20), 25.0) == []


def test_a_run_of_slow_ticks_is_one_stop_of_the_right_duration():
    runs = near_stop_runs(_poses([1.0] * 5 + [0.0] * 10 + [1.0] * 5), 25.0)
    assert runs == pytest.approx([10 / 25.0])


def test_separate_stops_are_counted_separately():
    runs = near_stop_runs(_poses([1.0] * 3 + [0.0] * 4 + [1.0] * 3 + [0.0] * 6), 25.0)
    assert runs == pytest.approx([4 / 25.0, 6 / 25.0])


def test_min_duration_drops_single_tick_sags():
    speeds = [1.0] * 3 + [0.0] + [1.0] * 3 + [0.0] * 10
    runs = near_stop_runs(_poses(speeds), 25.0, min_duration_s=2 / 25.0)
    assert runs == pytest.approx([10 / 25.0])


def test_rotation_alone_is_not_translation_and_does_not_end_a_stop():
    # The base spinning on the spot is still stopped, by this measure.
    poses = np.zeros((11, 3))
    poses[:, 2] = np.linspace(0.0, 3.0, 11)
    assert near_stop_runs(poses, 25.0) == pytest.approx([10 / 25.0])


def test_degenerate_inputs_are_empty_not_an_error():
    assert near_stop_runs([], 25.0) == []
    assert near_stop_runs([[0.0, 0.0, 0.0]], 25.0) == []
    assert near_stop_runs(_poses([0.0] * 5), 0.0) == []
