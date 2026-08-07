"""A navmesh top-down map, and the episode overlay drawn on it.

What this is for: watching a generated episode and being able to say whether it is a
*good* episode. The RGB view alone cannot answer that -- a robot pushing through a wall
and a robot walking down a corridor look similar from inside, and the existing
`controllers.visualize_bev` panel plots the path against nothing, so a trajectory that
leaves the floor plan plots as a perfectly clean line. This draws the floor plan.

Built on ``pathfinder.get_topdown_view(meters_per_pixel, height)``, which is the same
raster the navmesh planner answers from -- so "the planned path lies on free space" is
checkable by eye, against the *robot* navmesh rather than against the rendered geometry.

------------------------------------------------------------------------
The axis convention, which is the one thing here that fails silently
------------------------------------------------------------------------
``get_topdown_view`` returns ``bool[z_index, x_index]`` spanning ``pathfinder.
get_bounds()``. Verified empirically rather than assumed: over 400 random navigable
points on ``17DRP5sb8fy``, ``view[z_idx, x_idx]`` hits navigable 360 times and the
transpose hits 66 (the shortfall in both being points on other storeys, since a view is
one height slice).

Our planar control frame is ``X = habitat_x``, ``Y = -habitat_z``. So:

    col ascends with  +X   ->  +X is right
    row ascends with  +z   =   -Y   ->  +Y is up

which is the ordinary orientation of a map, with **no flip**. That is worth stating
because getting it backwards produces a mirrored map that renders perfectly plausibly:
the trajectory is still smooth, still inside the walls, still ends at the goal star. The
only symptom is that every left turn is a right turn. ``tests/test_topdown.py`` pins the
index order against a hand-built array for exactly this reason.

------------------------------------------------------------------------
Layering
------------------------------------------------------------------------
This module must not import ``objectnav_eval``: that package is strictly additive on top
of ``continuous_demos`` and the reverse edge would be a cycle. It therefore renders
frames and never writes a video; the caller owns the writer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import numpy as np

# Palette. Deliberately low-contrast for the map and saturated for the overlay, so the
# floor plan reads as background and the episode reads as figure.
COLOR_FREE = (236, 236, 236)
COLOR_BLOCKED = (58, 58, 62)
COLOR_EXECUTED = (32, 112, 220)  # where the robot has actually been
COLOR_PLANNED = (206, 62, 196)  # the shortest path for the current segment
COLOR_GOAL = (218, 44, 44)  # the current goal
COLOR_PAST_GOAL = (140, 140, 146)  # goals already reached
COLOR_AGENT = (16, 16, 16)
COLOR_TEXT = (24, 24, 28)
COLOR_HEADER = (250, 250, 250)

#: Default raster resolution. 5 cm is finer than the robot's 0.28 m navmesh radius, so a
#: gap the robot cannot fit through is still visibly a gap.
DEFAULT_METERS_PER_PIXEL = 0.05

#: Passed to ``get_topdown_view``: how far above/below the slice height a navmesh polygon
#: may sit and still be drawn. Habitat's own default.
DEFAULT_SLICE_EPS = 0.5

#: Length of the agent's heading arrow, metres.
AGENT_ARROW_M = 0.5


@dataclass(frozen=True)
class TopDownExtent:
    """The world <-> pixel mapping of one rendered navmesh slice, in the planar frame.

    ``origin_x`` is the planar X of column 0's left edge and ``origin_y`` the planar Y of
    row 0's *top* edge -- so Y decreases as the row index grows, which is what makes the
    raster an ordinary map rather than a mirrored one. See the module docstring.
    """

    origin_x: float
    origin_y: float
    meters_per_pixel: float
    rows: int
    cols: int
    height_m: float

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)


def world_to_pixel(
    extent: TopDownExtent, x: float, y: float, clamp: bool = True
) -> Tuple[int, int]:
    """Planar ``(X, Y)`` -> ``(row, col)``.

    Clamps to the raster by default rather than raising: a goal or a trajectory point can
    legitimately sit a few centimetres outside the navmesh bounds (the robot's own radius
    puts it there), and a debug renderer that raises on that is useless exactly when it is
    most wanted. Pass ``clamp=False`` to get the unclamped indices.
    """
    col = int(math.floor((x - extent.origin_x) / extent.meters_per_pixel))
    row = int(math.floor((extent.origin_y - y) / extent.meters_per_pixel))
    if clamp:
        row = min(max(row, 0), extent.rows - 1)
        col = min(max(col, 0), extent.cols - 1)
    return row, col


def pixel_to_world(extent: TopDownExtent, row: int, col: int) -> Tuple[float, float]:
    """``(row, col)`` -> the planar ``(X, Y)`` of that pixel's **centre**."""
    x = extent.origin_x + (col + 0.5) * extent.meters_per_pixel
    y = extent.origin_y - (row + 0.5) * extent.meters_per_pixel
    return x, y


def polyline_pixels(extent: TopDownExtent, xy: Any) -> np.ndarray:
    """Planar ``(M, 2)`` -> ``(M, 2)`` int array of ``(col, row)``, i.e. cv2's order.

    cv2 takes points as ``(x, y)`` == ``(col, row)``, the transpose of everything else
    here, so the conversion happens once, in this function, rather than at each draw.
    """
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    out = np.empty((pts.shape[0], 2), dtype=np.int32)
    for i, (x, y) in enumerate(pts):
        row, col = world_to_pixel(extent, float(x), float(y))
        out[i] = (col, row)
    return out


@dataclass
class TopDownState:
    """Everything the overlay draws, for one tick.

    Deliberately a plain value object with no simulator in it: the debug script fills it
    from the live sim, and a test or an offline re-render fills it from a recorded
    metadata row. Both get the identical picture.
    """

    #: Agent pose ``[X, Y, theta]``, planar frame.
    pose2d: Sequence[float]
    #: Current goal ``[X, Y, theta]``, or None outside a segment.
    goal2d: Optional[Sequence[float]] = None
    #: The current segment's planned shortest path, planar ``(M, 2)``.
    planned_path: Optional[Any] = None
    #: Where the robot has actually been this episode, planar ``(T, 2)``.
    executed: Optional[Any] = None
    #: Goals already reached, so a chained episode shows its history.
    past_goals: Sequence[Sequence[float]] = ()
    segment_index: int = 0
    num_segments: int = 0
    frame_index: int = 0
    frame_target: Optional[int] = None
    #: Free text for the second header line -- an outcome, a rejection reason.
    note: str = ""

    def distance_to_goal(self) -> Optional[float]:
        if self.goal2d is None:
            return None
        p = np.asarray(self.pose2d, dtype=float)[:2]
        g = np.asarray(self.goal2d, dtype=float)[:2]
        return float(np.linalg.norm(p - g))


class TopDownMap:
    """Occupancy raster of one navmesh height slice, plus the episode overlay.

    The raster is built once and cached; :meth:`render` copies it and draws. At 0.05 m/px
    a whole MP3D floor is a few hundred pixels square, so a render is a handful of cv2
    calls -- cheap enough to do every tick in a debug run and still far too expensive to
    leave on in a generation run, which is why the generator defaults it off.
    """

    def __init__(self, occupancy: Any, extent: TopDownExtent) -> None:
        self.occupancy = np.asarray(occupancy, dtype=bool)
        if self.occupancy.shape != extent.shape:
            raise ValueError(
                f"occupancy {self.occupancy.shape} does not match extent {extent.shape}"
            )
        self.extent = extent
        self._base: Optional[np.ndarray] = None

    # -- construction ------------------------------------------------------
    @classmethod
    def for_pathfinder(
        cls,
        pathfinder,
        height_m: float,
        meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
        eps: float = DEFAULT_SLICE_EPS,
    ) -> "TopDownMap":
        """Render the navmesh slice at ``height_m`` (a **navmesh-level** habitat y).

        Pass ``robot_pose.navmesh_height(robot_sim)``, not ``robot_height_m``: the URDF
        lift would put the slice ~0.25 m above the floor, which on a scene with a low
        ceiling or a mezzanine cuts a different storey.
        """
        if not pathfinder.is_loaded:
            raise RuntimeError("pathfinder has no navmesh loaded")
        occupancy = np.asarray(pathfinder.get_topdown_view(meters_per_pixel, float(height_m)))
        lo, hi = pathfinder.get_bounds()
        extent = TopDownExtent(
            origin_x=float(lo[0]),
            # Planar Y of row 0's top edge. Row index ascends with habitat +z, and
            # Y = -z, so the largest Y is at row 0.
            origin_y=-float(lo[2]),
            meters_per_pixel=float(meters_per_pixel),
            rows=int(occupancy.shape[0]),
            cols=int(occupancy.shape[1]),
            height_m=float(height_m),
        )
        return cls(occupancy, extent)

    # -- drawing -----------------------------------------------------------
    def base_rgb(self) -> np.ndarray:
        """The floor plan alone, cached. Never mutate the returned array."""
        if self._base is None:
            base = np.empty((self.extent.rows, self.extent.cols, 3), dtype=np.uint8)
            base[...] = np.asarray(COLOR_BLOCKED, dtype=np.uint8)
            base[self.occupancy] = np.asarray(COLOR_FREE, dtype=np.uint8)
            self._base = base
        return self._base

    def render(self, state: TopDownState, scale: int = 1, header: bool = True) -> np.ndarray:
        """The floor plan with this tick's overlay on it, as ``(H, W, 3)`` uint8.

        ``header=False`` omits the caption, for a caller that will resize the result and
        wants to caption it afterwards at its own resolution -- see :meth:`panel`.
        """
        import cv2

        img = self.base_rgb().copy()
        ext = self.extent
        if scale > 1:
            img = cv2.resize(
                img,
                (ext.cols * scale, ext.rows * scale),
                interpolation=cv2.INTER_NEAREST,
            )
        ext = (
            ext
            if scale == 1
            else TopDownExtent(
                origin_x=ext.origin_x,
                origin_y=ext.origin_y,
                meters_per_pixel=ext.meters_per_pixel / scale,
                rows=ext.rows * scale,
                cols=ext.cols * scale,
                height_m=ext.height_m,
            )
        )

        # Every stroke width is in scaled pixels, so upsampling makes the map crisper
        # rather than making the overlay hairline-thin against it.
        lw = max(1, int(round(1.5 * scale)))

        # Executed first, so the plan for the current segment draws over it.
        if state.executed is not None and len(state.executed) >= 2:
            cv2.polylines(
                img, [polyline_pixels(ext, state.executed)], False, COLOR_EXECUTED, lw,
                lineType=cv2.LINE_AA,
            )
        if state.planned_path is not None and len(state.planned_path) >= 2:
            cv2.polylines(
                img, [polyline_pixels(ext, state.planned_path)], False, COLOR_PLANNED, lw,
                lineType=cv2.LINE_AA,
            )
        for goal in state.past_goals:
            row, col = world_to_pixel(ext, float(goal[0]), float(goal[1]))
            cv2.circle(img, (col, row), 2 * lw, COLOR_PAST_GOAL, -1, lineType=cv2.LINE_AA)
        if state.goal2d is not None:
            self._draw_goal(img, ext, state.goal2d, lw)
        self._draw_agent(img, ext, state.pose2d, lw)
        return self._draw_header(img, state) if header else img

    def panel(self, state: TopDownState, side_px: int, scale: int = 1) -> np.ndarray:
        """:meth:`render`, letterboxed into a ``side_px`` square, then captioned.

        Square because the recorders hstack a square panel next to the RGB frame (the
        ``Image.fromarray(panel).resize((h, h))`` idiom in ``replay.py``), and resizing a
        long thin floor plan into a square would silently change its aspect ratio -- which
        for a *map* is the same class of error as mirroring it.

        The caption is drawn **after** the resize, at panel resolution. Drawing it before
        makes its size depend on the scene's aspect ratio: a tall narrow floor plan gets
        downscaled hard to fit the square, and takes its caption down with it to
        illegibility.
        """
        import cv2

        rendered = self.render(state, scale=scale, header=False)
        h, w = rendered.shape[:2]
        header_h = int(0.055 * side_px) * (2 if state.note else 1) + 6
        avail_h = side_px - header_h
        factor = min(side_px / float(w), avail_h / float(h))
        new_w, new_h = max(1, int(w * factor)), max(1, int(h * factor))
        resized = cv2.resize(rendered, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out = np.full((side_px, side_px, 3), COLOR_HEADER, dtype=np.uint8)
        top = header_h + (avail_h - new_h) // 2
        left = (side_px - new_w) // 2
        out[top:top + new_h, left:left + new_w] = resized

        font = side_px / 900.0
        line_h = int(0.055 * side_px)
        for i, text in enumerate([_headline(state)] + ([state.note] if state.note else [])):
            cv2.putText(out, text, (6, int(line_h * (i + 0.9))), cv2.FONT_HERSHEY_SIMPLEX,
                        font, COLOR_TEXT, 1, cv2.LINE_AA)
        return out

    # -- overlay pieces ----------------------------------------------------
    @staticmethod
    def _draw_goal(img, ext: TopDownExtent, goal2d: Sequence[float], lw: int = 2) -> None:
        import cv2

        row, col = world_to_pixel(ext, float(goal2d[0]), float(goal2d[1]))
        arm = 4 * lw
        cv2.line(img, (col - arm, row - arm), (col + arm, row + arm), COLOR_GOAL, lw,
                 lineType=cv2.LINE_AA)
        cv2.line(img, (col - arm, row + arm), (col + arm, row - arm), COLOR_GOAL, lw,
                 lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_agent(img, ext: TopDownExtent, pose2d: Sequence[float], lw: int = 2) -> None:
        import cv2

        x, y, theta = float(pose2d[0]), float(pose2d[1]), float(pose2d[2])
        row, col = world_to_pixel(ext, x, y)
        tip_row, tip_col = world_to_pixel(
            ext, x + AGENT_ARROW_M * math.cos(theta), y + AGENT_ARROW_M * math.sin(theta)
        )
        cv2.arrowedLine(img, (col, row), (tip_col, tip_row), COLOR_AGENT, lw,
                        tipLength=0.35, line_type=cv2.LINE_AA)
        cv2.circle(img, (col, row), 2 * lw, COLOR_AGENT, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_header(img: np.ndarray, state: TopDownState) -> np.ndarray:
        """Two text lines above the map, on their own strip.

        Above rather than on top of it: a floor plan has no reliably empty corner, and
        text over the geometry is exactly where a wall would be hiding.
        """
        import cv2

        lines = [_headline(state)]
        if state.note:
            lines.append(state.note)
        # Scaled to the raster's width, not fixed: the panel resizes the whole thing to
        # fit a square, so a fixed 14 px caption is illegible on a wide floor plan and
        # covers a third of a narrow one.
        font = max(0.4, img.shape[1] / 1000.0)
        line_h = int(20 * font)
        thickness = 1 if font < 0.8 else 2
        strip_h = line_h * len(lines) + int(8 * font)
        strip = np.full((strip_h, img.shape[1], 3), COLOR_HEADER, dtype=np.uint8)
        for i, text in enumerate(lines):
            cv2.putText(strip, text, (int(6 * font), int(line_h * (i + 0.85))),
                        cv2.FONT_HERSHEY_SIMPLEX, font, COLOR_TEXT, thickness, cv2.LINE_AA)
        return np.vstack([strip, img])


def _headline(state: TopDownState) -> str:
    parts = []
    if state.num_segments:
        parts.append(f"seg {state.segment_index + 1}/{state.num_segments}")
    else:
        parts.append(f"seg {state.segment_index}")
    if state.frame_target:
        parts.append(f"frame {state.frame_index}/{state.frame_target}")
    else:
        parts.append(f"frame {state.frame_index}")
    dist = state.distance_to_goal()
    if dist is not None:
        parts.append(f"d_goal={dist:.2f}m")
    return "  ".join(parts)
