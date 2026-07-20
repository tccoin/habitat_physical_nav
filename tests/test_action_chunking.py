"""Invariant checks on continuous_demos.action_chunking -- no habitat_sim needed.

These are the cheapest possible guard against getting the relative-pose sign
convention or the chunk index/stride arithmetic wrong, both of which
replay_chunked_actions.py's "near exact same trajectory tracking" acceptance
criterion depends on.
"""

import math

import numpy as np
import pytest

from continuous_demos.action_chunking import (
    build_observation_records,
    make_chunk_config,
    pose_to_relative,
    relative_to_pose,
)


def random_pose(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-4 * math.pi, 4 * math.pi)]
    )


def test_relative_pose_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        anchor = random_pose(rng)
        target = random_pose(rng)
        relative = pose_to_relative(anchor, target)
        recovered = relative_to_pose(anchor, relative)

        assert recovered[0] == pytest.approx(target[0], abs=1e-9)
        assert recovered[1] == pytest.approx(target[1], abs=1e-9)
        # Compare headings on the unit circle since theta is only defined mod 2*pi.
        assert math.cos(recovered[2]) == pytest.approx(math.cos(target[2]), abs=1e-9)
        assert math.sin(recovered[2]) == pytest.approx(math.sin(target[2]), abs=1e-9)


def test_relative_pose_is_anchor_frame():
    # A target one metre ahead of the anchor along the anchor's own heading
    # must show up as pure local +X, regardless of the anchor's world heading.
    anchor = np.array([5.0, -3.0, math.radians(37.0)])
    dx, dy = math.cos(anchor[2]), math.sin(anchor[2])
    target = anchor + np.array([dx, dy, 0.0])
    relative = pose_to_relative(anchor, target)
    assert relative[0] == pytest.approx(1.0, abs=1e-9)
    assert relative[1] == pytest.approx(0.0, abs=1e-9)
    assert relative[2] == pytest.approx(0.0, abs=1e-9)


def test_make_chunk_config_matches_default_half_overlap():
    # 20 Hz native data, 4 Hz observations, 10-action chunks -> the repo's
    # documented default: obs_stride_frames=5, chunk covers 2x the
    # observation interval, i.e. 50% overlap between consecutive chunks.
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    assert cfg.obs_stride_frames == 5
    assert cfg.dt_native == pytest.approx(0.05)
    assert cfg.obs_interval == pytest.approx(0.25)
    assert cfg.chunk_duration == pytest.approx(0.5)
    assert cfg.overlap_fraction == pytest.approx(0.5)


def test_make_chunk_config_rejects_gap():
    # 4 actions * 1/20s = 0.2s chunk duration < 0.25s observation interval:
    # a real coverage gap, must be rejected.
    with pytest.raises(ValueError):
        make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=4)


def test_make_chunk_config_allows_exact_touch():
    # 5 actions * 1/20s = 0.25s chunk duration == observation interval: chunks
    # exactly touch, zero overlap, zero gap -- the boundary case must be allowed.
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=5)
    assert cfg.overlap_fraction == pytest.approx(0.0, abs=1e-9)


def test_build_observation_records_indexing():
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    n = 27  # native frames 0..26 -> last valid index 26
    pose2d = [[float(i), 0.0, 0.0] for i in range(n)]
    rgb_paths = [f"frame_{i:05d}.png" for i in range(n)]

    records = build_observation_records(pose2d, rgb_paths, cfg)

    # idx + 10 <= 26  =>  idx in {0, 5, 10, 15} (stride 5), idx=20 would need
    # frame 30 which doesn't exist.
    assert [r["frame_index"] for r in records] == [0, 5, 10, 15]
    assert [r["obs_index"] for r in records] == [0, 1, 2, 3]

    first = records[0]
    assert first["obs_pose"] == [0.0, 0.0, 0.0]
    assert first["image"] == "frame_00000.png"
    # Poses advance by +1 in x per native frame and anchor heading is 0, so
    # the relative chunk is just [[1,0,0], [2,0,0], ..., [10,0,0]].
    expected_chunk = [[float(j), 0.0, 0.0] for j in range(1, 11)]
    assert np.asarray(first["action_chunk"]) == pytest.approx(np.asarray(expected_chunk))

    second = records[1]
    assert second["frame_index"] == 5
    assert second["obs_pose"] == [5.0, 0.0, 0.0]
    assert second["action_chunk"][0] == pytest.approx([1.0, 0.0, 0.0])
    assert second["action_chunk"][-1] == pytest.approx([10.0, 0.0, 0.0])


def test_build_observation_records_drops_short_tail():
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    n = 8  # N - 1 = 7 < action_chunk_len=10: no observation can anchor a full chunk
    pose2d = [[float(i), 0.0, 0.0] for i in range(n)]
    rgb_paths = [f"frame_{i:05d}.png" for i in range(n)]

    records = build_observation_records(pose2d, rgb_paths, cfg)
    assert records == []
