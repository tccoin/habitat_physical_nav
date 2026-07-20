"""Relative-pose action chunking: turns a per-frame ``pose2d`` trajectory (as
produced by ``generate_pose2d_dataset.py``, one pose per rendered RGB frame at
the replay's recording fps) into a lower-frequency stream of *observations*,
each paired with an *action chunk* -- a short run of future poses expressed
relative to that observation's own pose.

This is the format ``build_action_chunk_dataset.py`` assembles into an arrow
table, and that ``replay_chunked_actions.py`` re-expands back into an absolute
trajectory to sanity-check against a simulator. All of the index/frequency
arithmetic that both scripts need to agree on lives here so there is exactly
one place that can get the off-by-one wrong.

------------------------------------------------------------------------
Vocabulary
------------------------------------------------------------------------
* **native trajectory** -- the input: ``pose2d[0 .. N-1]``, one pose per
  frame, recorded at ``native_fps`` (whatever ``--fps`` was passed to
  ``generate_pose2d_dataset.py``; NOT stored in the metadata JSONL, so it must
  be supplied to every script here as an explicit argument that the caller
  guarantees matches the generation run).
* **observation** -- a single native-trajectory frame chosen to anchor a
  training example: its image, and its pose as the origin actions are made
  relative to. Observations are spaced ``obs_stride_frames`` native frames
  apart, i.e. at ``obs_hz = native_fps / obs_stride_frames``.
* **action chunk** -- ``action_chunk_len`` consecutive native-trajectory poses
  immediately following an observation, each expressed as ``[dx, dy, dtheta]``
  relative to that observation's own pose (not to each other -- every entry in
  a chunk is relative to the same anchor, so consuming the chunk does not
  require accumulating error across it).

------------------------------------------------------------------------
Why "chunk duration >= observation interval" is required
------------------------------------------------------------------------
A new action chunk is produced once per observation, i.e. once every
``obs_interval = obs_stride_frames * dt_native`` seconds. A chunk covers
``chunk_duration = action_chunk_len * dt_native`` seconds of future motion
starting at its observation. If ``chunk_duration < obs_interval``, the chunk
runs out of actions *before* the next observation's chunk becomes available --
an uncovered gap that makes synchronous chunk replay ill-defined. Requiring
``chunk_duration >= obs_interval`` guarantees every native frame is covered by
at least one chunk; when it is strictly greater, consecutive chunks overlap
(the default config -- 4 Hz observations, 10-action chunks, 20 Hz native data
-- covers 0.5s of chunk against a 0.25s observation interval, i.e. 2x /
50% overlap).

------------------------------------------------------------------------
Relative-pose convention
------------------------------------------------------------------------
Matches the planar control frame used throughout this package (``X = habitat_x``,
``Y = -habitat_z``, ``theta`` CCW from ``+X`` -- see ``sim.py``'s module
docstring). A pose ``target`` expressed relative to ``anchor`` is ``target``
rotated into ``anchor``'s local frame (anchor's heading becomes local ``+X``):

    dx, dy   = target.xy - anchor.xy
    local_x  =  cos(anchor.theta) * dx + sin(anchor.theta) * dy
    local_y  = -sin(anchor.theta) * dx + cos(anchor.theta) * dy
    local_th = wrap_angle(target.theta - anchor.theta)

``relative_to_pose`` is the exact inverse (rotate back into the world frame
and add the anchor's translation); ``compose(anchor, relative(anchor, p)) == p``
for every ``p``, verified for random poses (including wrapped angles) in
``tests/test_action_chunking.py``.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from continuous_demos.pid_pose_controller import wrap_angle


@dataclass
class ChunkConfig:
    """Resolved, self-consistent chunking parameters.

    ``obs_stride_frames`` is the *quantized* observation spacing (native
    frames must fall on integer frame boundaries); ``obs_interval`` below is
    computed from that quantized stride, not from ``1 / obs_hz`` directly, so
    ``obs_hz`` may end up realized slightly differently than requested when
    ``native_fps`` isn't an exact multiple of it -- ``effective_obs_hz``
    reports what was actually achieved.
    """

    native_fps: float
    obs_hz: float
    action_chunk_len: int
    obs_stride_frames: int
    dt_native: float
    obs_interval: float
    chunk_duration: float

    @property
    def effective_obs_hz(self) -> float:
        return 1.0 / self.obs_interval

    @property
    def overlap_fraction(self) -> float:
        """Fraction of a chunk's duration re-covered by the *next* chunk.

        0.0 means chunks exactly touch (no gap, no overlap); this is only
        meaningful/well-defined because ``chunk_duration >= obs_interval`` is
        enforced at construction time.
        """
        return 1.0 - (self.obs_interval / self.chunk_duration)


def make_chunk_config(native_fps: float, obs_hz: float, action_chunk_len: int) -> ChunkConfig:
    """Build and validate a :class:`ChunkConfig` from the three user-facing knobs.

    Raises ``ValueError`` if ``chunk_duration < obs_interval`` (see the module
    docstring for why that would leave a coverage gap between chunks) or if
    any input is non-positive.
    """
    if native_fps <= 0:
        raise ValueError(f"native_fps must be positive, got {native_fps}")
    if obs_hz <= 0:
        raise ValueError(f"obs_hz must be positive, got {obs_hz}")
    if action_chunk_len < 1:
        raise ValueError(f"action_chunk_len must be >= 1, got {action_chunk_len}")

    dt_native = 1.0 / native_fps
    obs_stride_frames = max(1, round(native_fps / obs_hz))
    obs_interval = obs_stride_frames * dt_native
    chunk_duration = action_chunk_len * dt_native

    if chunk_duration < obs_interval - 1e-9:
        raise ValueError(
            f"chunk_duration ({chunk_duration:.4f}s = action_chunk_len={action_chunk_len} "
            f"* dt_native={dt_native:.4f}s) is shorter than obs_interval "
            f"({obs_interval:.4f}s = obs_stride_frames={obs_stride_frames} * dt_native); "
            "this leaves a gap between one chunk running out and the next observation's "
            "chunk becoming available. Increase --action-chunk-len or decrease --obs-hz."
        )

    return ChunkConfig(
        native_fps=float(native_fps),
        obs_hz=float(obs_hz),
        action_chunk_len=int(action_chunk_len),
        obs_stride_frames=int(obs_stride_frames),
        dt_native=dt_native,
        obs_interval=obs_interval,
        chunk_duration=chunk_duration,
    )


def pose_to_relative(anchor: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """Express ``target`` (``[x, y, theta]``) relative to ``anchor``'s local frame."""
    x0, y0, th0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
    x1, y1, th1 = float(target[0]), float(target[1]), float(target[2])
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(th0), math.sin(th0)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    local_th = wrap_angle(th1 - th0)
    return np.array([local_x, local_y, local_th], dtype=float)


def relative_to_pose(anchor: Sequence[float], relative: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`pose_to_relative`: recover the absolute pose from
    ``anchor`` plus a ``[dx, dy, dtheta]`` relative to it."""
    x0, y0, th0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
    lx, ly, lth = float(relative[0]), float(relative[1]), float(relative[2])
    c, s = math.cos(th0), math.sin(th0)
    x1 = x0 + c * lx - s * ly
    y1 = y0 + s * lx + c * ly
    th1 = wrap_angle(th0 + lth)
    return np.array([x1, y1, th1], dtype=float)


def build_observation_records(
    pose2d: Sequence[Sequence[float]],
    rgb_paths: Sequence[str],
    chunk_config: ChunkConfig,
) -> List[Dict[str, Any]]:
    """Slide an observation window over one episode's native trajectory.

    ``pose2d`` and ``rgb_paths`` must be the same length ``N`` and mutually
    aligned frame-for-frame (exactly the arrays a ``metadata/<scene>.jsonl``
    record stores). Returns one record per observation, in ascending
    ``obs_index`` order:

        {
          "obs_index":    0, 1, 2, ...                 -- position in this episode's chunk stream
          "frame_index":  idx                            -- native-trajectory index this obs anchors on
          "image":        rgb_paths[idx]
          "obs_pose":     pose2d[idx]                    -- [x, y, theta], the chunk's anchor
          "action_chunk": [pose_to_relative(pose2d[idx], pose2d[idx+1]), ...,
                            pose_to_relative(pose2d[idx], pose2d[idx+action_chunk_len])]
        }

    An observation at native index ``idx`` is only emitted if the full chunk
    fits inside the recorded trajectory, i.e. ``idx + action_chunk_len <= N - 1``
    (0-indexed last valid frame is ``N - 1``). Trailing frames that can't
    anchor a full chunk are silently dropped -- there is no padding -- so an
    episode contributes
    ``floor((N - 1 - action_chunk_len) / obs_stride_frames) + 1`` observations
    (0 if ``N - 1 < action_chunk_len``, i.e. the whole episode is shorter than
    one chunk).
    """
    n = len(pose2d)
    if len(rgb_paths) != n:
        raise ValueError(f"pose2d ({n}) and rgb_paths ({len(rgb_paths)}) length mismatch")

    stride = chunk_config.obs_stride_frames
    length = chunk_config.action_chunk_len

    records: List[Dict[str, Any]] = []
    obs_index = 0
    idx = 0
    while idx + length <= n - 1:
        anchor = pose2d[idx]
        chunk = [pose_to_relative(anchor, pose2d[idx + j]) for j in range(1, length + 1)]
        records.append(
            {
                "obs_index": obs_index,
                "frame_index": idx,
                "image": rgb_paths[idx],
                "obs_pose": [float(v) for v in anchor],
                "action_chunk": [c.tolist() for c in chunk],
            }
        )
        obs_index += 1
        idx += stride
    return records
