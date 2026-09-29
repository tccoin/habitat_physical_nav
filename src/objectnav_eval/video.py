"""Real-time rollout video with goal and metric overlays.

One frame per control tick, so playing back at ``1/dt`` fps is real time and the
video is directly comparable to a wall-clock recording of the same robot. The
frames the policy actually acted on are marked, because the rest are
intermediate control ticks the policy never saw.

Frames stream straight to the encoder: a 200-step episode is 1000 frames, and at
640x480 that is ~0.9 GiB held in a list, per episode, per worker. Appending as
we go keeps a worker's memory footprint flat.

The overlay is always drawn on a **copy** -- the policy is handed the raw
render, never these pixels.
"""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class OverlayState:
    """What the overlay draws. Everything is optional-ish so a control run with
    no meaningful distance still produces a readable video."""

    step: int
    sim_time: float
    goal: Optional[str]
    distance: float
    start_distance: float
    reached: bool = False
    is_observation: bool = False
    extra: Optional[str] = None


def annotate(
    rgb: np.ndarray, state: OverlayState, *, header_outside: bool = False
) -> np.ndarray:
    """Draw the rollout metadata without ever modifying ``rgb``.

    ``header_outside`` is for the two-panel RGB+BEV recording.  The status bar
    then occupies newly allocated pixels above the panels instead of covering
    the first-person observation.  The legacy single-panel video keeps its
    historical in-image layout.
    """
    import cv2

    image = np.ascontiguousarray(rgb).copy()
    height, width = image.shape[:2]
    goal = state.goal or "?"
    status = []
    if state.is_observation:
        status.append("OBS")
    if state.reached:
        status.append("REACHED")
    suffix = f"  {' · '.join(status)}" if status else ""
    left = f"step {state.step:>3}  t={state.sim_time:6.2f}s  goal: {goal}{suffix}"
    closer = np.isfinite(state.distance) and state.distance <= state.start_distance
    colour = (120, 255, 160) if closer else (140, 160, 255)
    distance = f"{state.distance:5.2f}" if np.isfinite(state.distance) else "  inf"
    start = f"{state.start_distance:5.2f}" if np.isfinite(state.start_distance) else "inf"
    right = f"dist {distance} m  (start {start})"

    # Measured rather than the fixed `width - 285` this used to assume. That constant is
    # right at 640 px and wrong at anything narrower: at 480 the two strings overlap into
    # unreadable mush, which is exactly the width a quick debug render uses. When they
    # still will not fit side by side the distance drops to its own row instead.
    (left_w, _), _ = cv2.getTextSize(left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    (right_w, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    stacked = left_w + right_w + 24 > width
    bar_h = 60 if stacked else 34
    if header_outside:
        content = image
        image = np.zeros((height + bar_h, width, 3), dtype=np.uint8)
        image[bar_h:] = content
        height = image.shape[0]
    else:
        cv2.rectangle(image, (0, 0), (width, bar_h), (0, 0, 0), -1)

    cv2.putText(image, left, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    right_pos = (8, 49) if stacked else (max(8, width - right_w - 8), 23)
    cv2.putText(image, right, right_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1,
                cv2.LINE_AA)
    if state.is_observation and not header_outside:
        # Marks the frames the policy actually acts on; the rest are
        # intermediate control ticks, present so playback is real time. Its own
        # row, clear of the right-aligned distance readout.
        cv2.circle(image, (14, bar_h + 16), 5, (80, 200, 255), -1)
        cv2.putText(image, "obs", (26, bar_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 200, 255), 1, cv2.LINE_AA)
    if state.reached and not header_outside:
        (reached_w, _), _ = cv2.getTextSize("REACHED", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(image, "REACHED", (max(8, width - reached_w - 8), bar_h + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 160), 2, cv2.LINE_AA)
    if state.extra:
        cv2.putText(image, state.extra, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return image


def planar_xy(points: Any) -> np.ndarray:
    """Habitat world points -> the evaluator's planar ``(X, Y)`` frame."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return values[:, [0, 2]] * np.array([1.0, -1.0])


class EpisodeBev:
    """One static scene-RGB overhead image plus a live executed-route overlay.

    The static image is captured once from a second, orthographic Habitat
    camera.  This object deliberately has no simulator dependency: rendering a
    tick only copies pixels and draws a continuous route, which keeps the video
    path cheap and makes its geometry testable without Habitat.
    """

    _BACKGROUND = (30, 30, 34)
    _VIEW = (255, 232, 128)
    _ROUTE = (178, 178, 178)
    _TARGET = (255, 220, 0)
    _TARGET_OUTLINE = (35, 35, 35)
    _AGENT_OUTLINE = (20, 20, 20)
    _AGENT_FILL = (245, 245, 245)

    def __init__(
        self,
        background_rgb: np.ndarray,
        *,
        world_center_xy: Sequence[float],
        world_span_m: float,
        panel_size: Sequence[int],
        target_xy: Any,
        view_points_xy: Any,
        start_xy: Sequence[float],
        framing_xy: Any = None,
    ) -> None:
        import cv2

        background = np.asarray(background_rgb, dtype=np.uint8)
        if background.ndim != 3 or background.shape[2] != 3:
            raise ValueError("background_rgb must have shape (H, W, 3)")
        if not np.isfinite(world_span_m) or world_span_m <= 0:
            raise ValueError("world_span_m must be positive")
        if len(panel_size) != 2:
            raise ValueError("panel_size must be (width, height)")

        self.background = np.ascontiguousarray(background)
        self.center = np.asarray(world_center_xy, dtype=np.float64).reshape(2)
        self.world_span_m = float(world_span_m)
        self.panel_width, self.panel_height = (int(panel_size[0]), int(panel_size[1]))
        if self.panel_width <= 0 or self.panel_height <= 0:
            raise ValueError("panel dimensions must be positive")
        self.targets = self._points(target_xy)
        self.views = self._points(view_points_xy)
        self.start = np.asarray(start_xy, dtype=np.float64).reshape(2)
        framing = self._points(framing_xy)
        focus = [self.start[None, :], self.targets, self.views, framing]
        focus = np.concatenate([points for points in focus if len(points)], axis=0)
        self._source_box = self._focus_crop(focus)
        focus_pixels = np.asarray([self._source_pixel(point) for point in focus])

        x0, y0, x1, y1 = self._source_box
        crop = self.background[y0:y1, x0:x1]
        # The 18 m minimum framing can make a narrow episode crop appear
        # landscape even though the route and target layout are vertical.  Use
        # the episode geometry, not its padded rectangle, to decide rotation.
        focus_width = float(np.ptp(focus_pixels[:, 0]))
        focus_height = float(np.ptp(focus_pixels[:, 1]))
        self.rotate_clockwise = focus_height > focus_width
        if self.rotate_clockwise:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        crop_h, crop_w = crop.shape[:2]
        self.scale = min(self.panel_width / float(crop_w), self.panel_height / float(crop_h))
        self.render_width = max(1, int(round(crop_w * self.scale)))
        self.render_height = max(1, int(round(crop_h * self.scale)))
        self.left = (self.panel_width - self.render_width) // 2
        self.top = (self.panel_height - self.render_height) // 2

        base = np.full(
            (self.panel_height, self.panel_width, 3), self._BACKGROUND, dtype=np.uint8
        )
        base[self.top:self.top + self.render_height, self.left:self.left + self.render_width] = (
            cv2.resize(crop, (self.render_width, self.render_height), interpolation=cv2.INTER_AREA)
        )
        self._base = self._draw_static_markers(base)
        self._history = self._base.copy()
        self._previous: Optional[tuple[int, int]] = None

    @staticmethod
    def _points(points: Any) -> np.ndarray:
        if points is None:
            return np.zeros((0, 2), dtype=np.float64)
        array = np.asarray(points, dtype=np.float64)
        if array.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return array.reshape(-1, 2)

    def _source_pixel(self, xy: Sequence[float]) -> tuple[float, float]:
        height, width = self.background.shape[:2]
        x, y = float(xy[0]), float(xy[1])
        # The orthographic camera looks down habitat -Y.  Its image has +X to
        # the right and planar +Y (habitat -Z) upwards, matching topdown.py.
        col = ((x - self.center[0]) / self.world_span_m + 0.5) * width - 0.5
        row = ((self.center[1] - y) / self.world_span_m + 0.5) * height - 0.5
        return col, row

    def _focus_crop(self, focus: np.ndarray) -> tuple[int, int, int, int]:
        points = np.asarray([self._source_pixel(point) for point in focus])
        height, width = self.background.shape[:2]
        pixels_per_m = min(width, height) / self.world_span_m
        # A 2 m gutter makes the target and views readable; the 18 m minimum
        # avoids magnifying a tiny source crop into a fuzzy full-height panel.
        half_width = max(0.5 * (points[:, 0].max() - points[:, 0].min()) + 2.0 * pixels_per_m,
                         9.0 * pixels_per_m)
        half_height = max(0.5 * (points[:, 1].max() - points[:, 1].min()) + 2.0 * pixels_per_m,
                          9.0 * pixels_per_m)
        center_x = 0.5 * (points[:, 0].max() + points[:, 0].min())
        center_y = 0.5 * (points[:, 1].max() + points[:, 1].min())
        x0 = max(0, int(math.floor(center_x - half_width)))
        x1 = min(width, int(math.ceil(center_x + half_width + 1)))
        y0 = max(0, int(math.floor(center_y - half_height)))
        y1 = min(height, int(math.ceil(center_y + half_height + 1)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("BEV focus does not intersect the overhead image")
        return x0, y0, x1, y1

    def _to_panel(self, xy: Sequence[float]) -> tuple[int, int]:
        col, row = self._source_pixel(xy)
        x0, y0, x1, y1 = self._source_box
        local_x, local_y = col - x0, row - y0
        if self.rotate_clockwise:
            local_x, local_y = (y1 - y0 - 1) - local_y, local_x
        return (
            int(round(self.left + local_x * self.scale)),
            int(round(self.top + local_y * self.scale)),
        )

    def _draw_static_markers(self, base: np.ndarray) -> np.ndarray:
        import cv2

        overlay = base.copy()
        # A pixel-deduplicated set prevents many nearby ObjectNav view points
        # from becoming opaque blobs while still showing every distinct target
        # location that counts as success.
        for point in {self._to_panel(view) for view in self.views}:
            cv2.circle(overlay, point, 1, self._VIEW, -1, cv2.LINE_AA)
        panel = cv2.addWeighted(overlay, 0.35, base, 0.65, 0.0)
        cv2.circle(panel, self._to_panel(self.start), 4, (110, 110, 110), 1, cv2.LINE_AA)
        for target in self.targets:
            self._draw_star(panel, self._to_panel(target))
        return panel

    def _draw_star(self, panel: np.ndarray, center: tuple[int, int]) -> None:
        import cv2

        points = []
        for index in range(10):
            radius = 9 if index % 2 == 0 else 4
            angle = -math.pi / 2.0 + index * math.pi / 5.0
            points.append(
                [
                    int(round(center[0] + radius * math.cos(angle))),
                    int(round(center[1] + radius * math.sin(angle))),
                ]
            )
        star = np.asarray(points, dtype=np.int32)
        cv2.fillConvexPoly(panel, star, self._TARGET, lineType=cv2.LINE_AA)
        cv2.polylines(panel, [star], True, self._TARGET_OUTLINE, 2, cv2.LINE_AA)

    def render(self, pose2d: Sequence[float]) -> np.ndarray:
        """Draw the travelled route below target markers and the live agent above it."""
        import cv2

        pose = np.asarray(pose2d, dtype=np.float64).reshape(-1)
        if len(pose) < 3:
            raise ValueError("pose2d must contain X, Y, theta")
        current = self._to_panel(pose[:2])
        if self._previous is not None and self._previous != current:
            cv2.line(self._history, self._previous, current, self._ROUTE, 3, cv2.LINE_AA)
        self._previous = current

        panel = self._history.copy()
        theta = float(pose[2])
        tip = self._to_panel(
            (pose[0] + 0.85 * math.cos(theta), pose[1] + 0.85 * math.sin(theta))
        )
        cv2.arrowedLine(panel, current, tip, self._AGENT_OUTLINE, 5, cv2.LINE_AA,
                        tipLength=0.35)
        cv2.arrowedLine(panel, current, tip, self._AGENT_FILL, 2, cv2.LINE_AA,
                        tipLength=0.35)
        cv2.circle(panel, current, 4, self._AGENT_OUTLINE, -1, cv2.LINE_AA)
        return panel


def compose_with_bev(
    rgb: np.ndarray, state: OverlayState, bev: EpisodeBev, pose2d: Sequence[float]
) -> np.ndarray:
    """Put first-person RGB and BEV side by side under a non-obscuring status bar."""
    panel = bev.render(pose2d)
    return annotate(np.hstack([rgb, panel]), state, header_outside=True)


class Recorder:
    """Streaming MP4 writer. A disabled recorder is a cheap no-op."""

    def __init__(self, path: Optional[str], fps: int, scale: float = 1.0,
                 enabled: bool = True):
        self.enabled = bool(enabled and path)
        self.path = Path(path) if path else None
        self.scale = scale
        self.frames = 0
        if not self.enabled:
            return
        import imageio.v2 as imageio

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # macro_block_size=1: 640x480 is already divisible, but --video-scale can
        # make it odd, and a silent resize would desync the overlay from the frame.
        self._writer = imageio.get_writer(str(self.path), fps=max(1, int(fps)),
                                          macro_block_size=1)

    def add(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        if self.scale != 1.0:
            import cv2

            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        self._writer.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        if self.enabled:
            self._writer.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class BackgroundRecorder:
    """Compose and encode full-rate video off the rollout thread."""

    def __init__(
        self,
        path: Optional[str],
        fps: int,
        *,
        scale: float = 1.0,
        enabled: bool = True,
        bev: Optional[EpisodeBev] = None,
        queue_size: int = 16,
    ) -> None:
        self._recorder = Recorder(path, fps=fps, scale=scale, enabled=enabled)
        self.enabled = self._recorder.enabled
        self.path = self._recorder.path
        self._bev = bev
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._write_frames,
                name="objectnav-video-writer",
                daemon=True,
            )
            self._thread.start()

    @property
    def frames(self) -> int:
        return self._recorder.frames

    def add(self, rgb: np.ndarray, state: OverlayState, pose2d: Sequence[float]) -> None:
        if not self.enabled:
            return
        self._raise_error()
        # Habitat can reuse its render buffer on the next tick. Copy it before
        # handing the pixels to the encoder thread.
        payload = (
            np.ascontiguousarray(rgb, dtype=np.uint8).copy(),
            state,
            np.asarray(pose2d, dtype=np.float64).copy(),
        )
        self._queue.put(payload)
        self._raise_error()

    def _write_frames(self) -> None:
        try:
            while True:
                payload = self._queue.get()
                try:
                    if payload is None:
                        return
                    rgb, state, pose2d = payload
                    if self._bev is None:
                        frame = annotate(rgb, state)
                    else:
                        frame = compose_with_bev(rgb, state, self._bev, pose2d)
                    self._recorder.add(frame)
                finally:
                    self._queue.task_done()
        except BaseException as error:
            self._error = error
            # Keep draining so `close` cannot deadlock behind a failed writer.
            while True:
                payload = self._queue.get()
                self._queue.task_done()
                if payload is None:
                    return

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("background video writer failed") from self._error

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        assert self._thread is not None
        self._thread.join()
        self._recorder.close()
        self._raise_error()
