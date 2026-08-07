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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def annotate(rgb: np.ndarray, state: OverlayState) -> np.ndarray:
    import cv2

    image = np.ascontiguousarray(rgb).copy()
    height, width = image.shape[:2]
    goal = state.goal or "?"
    left = f"step {state.step:>3}  t={state.sim_time:6.2f}s  goal: {goal}"
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
    cv2.rectangle(image, (0, 0), (width, bar_h), (0, 0, 0), -1)

    cv2.putText(image, left, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    right_pos = (8, 49) if stacked else (max(8, width - right_w - 8), 23)
    cv2.putText(image, right, right_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1,
                cv2.LINE_AA)
    if state.is_observation:
        # Marks the frames the policy actually acts on; the rest are
        # intermediate control ticks, present so playback is real time. Its own
        # row, clear of the right-aligned distance readout.
        cv2.circle(image, (14, bar_h + 16), 5, (80, 200, 255), -1)
        cv2.putText(image, "obs", (26, bar_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 200, 255), 1, cv2.LINE_AA)
    if state.reached:
        (reached_w, _), _ = cv2.getTextSize("REACHED", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(image, "REACHED", (max(8, width - reached_w - 8), bar_h + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 160), 2, cv2.LINE_AA)
    if state.extra:
        cv2.putText(image, state.extra, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return image


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
