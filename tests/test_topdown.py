"""The top-down map's world <-> pixel mapping, and the axis order in particular.

A mirrored map renders perfectly plausibly: the trajectory is still smooth, still inside
the walls, still ends at the goal star. The only symptom is that every left turn is a
right turn, which is not something anyone catches by watching a video. So the index order
is pinned here against a hand-built array rather than trusted to the visual gate.

The habitat fact these encode -- ``get_topdown_view`` returns ``bool[z_index, x_index]``
over ``get_bounds()`` -- was measured, not read off documentation: over 400 random
navigable points on ``17DRP5sb8fy``, ``view[z_idx, x_idx]`` hit navigable 360 times and
the transpose 66.

No simulator: the extent is constructed directly.
"""

import numpy as np
import pytest

from continuous_demos.topdown import (
    COLOR_BLOCKED,
    COLOR_FREE,
    TopDownExtent,
    TopDownMap,
    TopDownState,
    pixel_to_world,
    polyline_pixels,
    world_to_pixel,
)


def make_extent(rows=4, cols=6, mpp=0.5, origin_x=-1.0, origin_y=2.0):
    return TopDownExtent(
        origin_x=origin_x, origin_y=origin_y, meters_per_pixel=mpp,
        rows=rows, cols=cols, height_m=0.0,
    )


# ---------------------------------------------------------------------------
# The axis order. This is the test that catches a mirrored map.
# ---------------------------------------------------------------------------
def test_row_zero_is_the_largest_planar_y_and_column_zero_the_smallest_x():
    """+X right, +Y up -- the ordinary orientation of a map, with no flip."""
    ext = make_extent()
    top_left_x, top_left_y = pixel_to_world(ext, 0, 0)
    bot_right_x, bot_right_y = pixel_to_world(ext, ext.rows - 1, ext.cols - 1)
    assert bot_right_x > top_left_x, "+X must be to the right"
    assert bot_right_y < top_left_y, "+Y must be up"


def test_moving_north_decreases_the_row_and_moving_east_increases_the_column():
    ext = make_extent()
    row0, col0 = world_to_pixel(ext, 0.0, 0.0)
    row_north, col_north = world_to_pixel(ext, 0.0, 1.0)  # +Y
    row_east, col_east = world_to_pixel(ext, 1.0, 0.0)  # +X
    assert row_north < row0 and col_north == col0
    assert col_east > col0 and row_east == row0


def test_index_order_pinned_against_a_hand_built_array():
    """A 3x4 raster with one free cell, placed by hand at a known world point.

    Written out in full so a transposed implementation cannot pass: the array is
    deliberately non-square, so a transpose is a shape error, *and* the free cell is
    off-centre, so a flip is a value error.
    """
    #                       col 0    1      2      3
    occupancy = np.array([[False, False, False, False],   # row 0
                          [False, False, True,  False],   # row 1  <- the free cell
                          [False, False, False, False]],  # row 2
                         dtype=bool)
    ext = TopDownExtent(origin_x=10.0, origin_y=20.0, meters_per_pixel=1.0,
                        rows=3, cols=4, height_m=0.0)
    # Cell (row 1, col 2) spans X in [12, 13) and Y in (18, 19].
    assert world_to_pixel(ext, 12.5, 18.5) == (1, 2)
    assert occupancy[world_to_pixel(ext, 12.5, 18.5)]
    # ...and its transpose is not free, which is what a swapped implementation gives.
    assert not occupancy[2, 1]

    tdm = TopDownMap(occupancy, ext)
    base = tdm.base_rgb()
    assert base.shape == (3, 4, 3)
    assert tuple(base[1, 2]) == COLOR_FREE
    assert tuple(base[0, 0]) == COLOR_BLOCKED


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mpp", [0.05, 0.1, 0.5])
def test_pixel_to_world_round_trips_to_the_same_pixel(mpp):
    ext = make_extent(rows=40, cols=60, mpp=mpp)
    for row in range(0, ext.rows, 7):
        for col in range(0, ext.cols, 11):
            x, y = pixel_to_world(ext, row, col)
            assert world_to_pixel(ext, x, y) == (row, col)


def test_pixel_to_world_lands_within_half_a_pixel_of_its_input():
    ext = make_extent(rows=40, cols=60, mpp=0.05)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = ext.origin_x + rng.uniform(0, ext.cols * ext.meters_per_pixel)
        y = ext.origin_y - rng.uniform(0, ext.rows * ext.meters_per_pixel)
        row, col = world_to_pixel(ext, x, y)
        rx, ry = pixel_to_world(ext, row, col)
        assert abs(rx - x) <= ext.meters_per_pixel / 2 + 1e-9
        assert abs(ry - y) <= ext.meters_per_pixel / 2 + 1e-9


# ---------------------------------------------------------------------------
# Bounds behaviour
# ---------------------------------------------------------------------------
def test_off_map_points_clamp_rather_than_raise():
    """A goal a few cm outside the navmesh bounds must still draw, not crash the render."""
    ext = make_extent()
    assert world_to_pixel(ext, -1e6, 1e6) == (0, 0)
    assert world_to_pixel(ext, 1e6, -1e6) == (ext.rows - 1, ext.cols - 1)


def test_clamp_can_be_turned_off_for_a_caller_that_wants_to_know():
    ext = make_extent()
    row, col = world_to_pixel(ext, -100.0, 100.0, clamp=False)
    assert row < 0 and col < 0


def test_bounds_corners_map_to_array_corners():
    ext = make_extent(rows=4, cols=6, mpp=0.5, origin_x=-1.0, origin_y=2.0)
    # The extent covers X in [-1, 2) and Y in (0, 2].
    assert world_to_pixel(ext, -1.0, 2.0) == (0, 0)
    assert world_to_pixel(ext, 2.0 - 1e-9, 0.0 + 1e-9) == (ext.rows - 1, ext.cols - 1)


# ---------------------------------------------------------------------------
# polyline_pixels: cv2's (x, y) is (col, row), the transpose of everything else
# ---------------------------------------------------------------------------
def test_polyline_pixels_emits_col_row_for_cv2():
    ext = make_extent()
    pts = polyline_pixels(ext, [[0.0, 0.0], [1.0, 1.0]])
    assert pts.shape == (2, 2) and pts.dtype == np.int32
    for (col, row), (x, y) in zip(pts, [(0.0, 0.0), (1.0, 1.0)]):
        assert (row, col) == world_to_pixel(ext, x, y)


def test_polyline_pixels_accepts_an_empty_path():
    assert polyline_pixels(make_extent(), np.zeros((0, 2))).shape == (0, 2)


# ---------------------------------------------------------------------------
# Construction guards and the state object
# ---------------------------------------------------------------------------
def test_a_map_refuses_an_occupancy_that_does_not_match_its_extent():
    with pytest.raises(ValueError, match="does not match extent"):
        TopDownMap(np.zeros((3, 3), dtype=bool), make_extent(rows=4, cols=6))


def test_state_distance_to_goal_is_planar_and_ignores_heading():
    state = TopDownState(pose2d=[0.0, 0.0, 1.23], goal2d=[3.0, 4.0, -2.0])
    assert state.distance_to_goal() == pytest.approx(5.0)
    assert TopDownState(pose2d=[0.0, 0.0, 0.0]).distance_to_goal() is None


# ---------------------------------------------------------------------------
# Rendering: shape and squareness, which the recorders' hstack depends on
# ---------------------------------------------------------------------------
def test_render_returns_the_raster_plus_a_header_strip():
    ext = make_extent(rows=20, cols=30)
    tdm = TopDownMap(np.ones((20, 30), dtype=bool), ext)
    img = tdm.render(TopDownState(pose2d=[0.0, 0.0, 0.0], goal2d=[1.0, 1.0, 0.0]))
    assert img.shape[1] == 30 and img.shape[0] > 20  # header adds rows, not columns
    assert img.dtype == np.uint8


def test_panel_is_square_and_preserves_aspect_ratio():
    """A long thin floor plan squashed into a square is the same class of error as
    mirroring it, so the panel letterboxes instead of stretching."""
    ext = make_extent(rows=10, cols=60)
    tdm = TopDownMap(np.ones((10, 60), dtype=bool), ext)
    panel = tdm.panel(TopDownState(pose2d=[0.0, 0.0, 0.0]), side_px=128)
    assert panel.shape == (128, 128, 3)
    # Letterboxed: the top and bottom rows are background, not map.
    assert (panel[0] == np.array(np.uint8(250))).all()


def test_render_survives_an_episode_with_no_goal_and_no_paths():
    ext = make_extent(rows=8, cols=8)
    tdm = TopDownMap(np.ones((8, 8), dtype=bool), ext)
    img = tdm.render(TopDownState(pose2d=[0.0, 0.0, 0.5]))
    assert img.shape[1] == 8
