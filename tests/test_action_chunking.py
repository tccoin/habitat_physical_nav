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
    SpacingConfig,
    build_observation_records,
    make_chunk_config,
    pose_to_relative,
    reconstruct_reference_trajectory,
    relative_to_pose,
    sample_gap_frames,
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


def random_walk_pose2d(n: int, seed: int) -> list:
    """A non-trivial synthetic trajectory (random per-step translation +
    rotation) -- deliberately not a straight line or constant-velocity path,
    so a round-trip test over it actually exercises the rotation math instead
    of accidentally passing on a degenerate case.
    """
    rng = np.random.default_rng(seed)
    pose = np.zeros(3)
    poses = [pose.copy()]
    for _ in range(n - 1):
        step = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2)])
        pose = pose + step
        poses.append(pose.copy())
    return [p.tolist() for p in poses]


# ---------------------------------------------------------------------------
# Variable observation spacing
# ---------------------------------------------------------------------------


def test_sample_gap_frames_fixed_mode_is_constant():
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    spacing = SpacingConfig(mode="fixed")
    rng = np.random.default_rng(0)
    gaps = {sample_gap_frames(rng, spacing, cfg) for _ in range(50)}
    assert gaps == {cfg.obs_stride_frames}  # always exactly the mean stride, no variance


@pytest.mark.parametrize("mode", ["uniform", "poisson"])
def test_sample_gap_frames_always_within_chunk_bounds(mode):
    # The clip must hold no matter how the distribution is parameterized --
    # this is what keeps "chunk covers the gap to the next observation" true
    # for every draw, not just on average.
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    spacing = SpacingConfig(mode=mode, jitter=0.9, seed=1)
    rng = np.random.default_rng(spacing.seed)
    gaps = [sample_gap_frames(rng, spacing, cfg) for _ in range(2000)]
    assert min(gaps) >= 1
    assert max(gaps) <= cfg.action_chunk_len


def test_sample_gap_frames_uniform_matches_requested_mean_and_spread():
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    spacing = SpacingConfig(mode="uniform", jitter=0.4, seed=2)  # mean=5, window [3, 7]
    rng = np.random.default_rng(spacing.seed)
    gaps = np.array([sample_gap_frames(rng, spacing, cfg) for _ in range(5000)])
    assert set(gaps.tolist()) == {3, 4, 5, 6, 7}
    assert np.mean(gaps) == pytest.approx(5.0, abs=0.15)


def test_sample_gap_frames_poisson_clips_the_mean_stride_edge_case():
    # Mean stride == action_chunk_len: Poisson can draw arbitrarily high, so
    # this is the case most likely to expose a broken/missing clip.
    cfg = make_chunk_config(native_fps=20.0, obs_hz=2.0, action_chunk_len=10)  # mean stride = 10
    spacing = SpacingConfig(mode="poisson", seed=3)
    rng = np.random.default_rng(spacing.seed)
    gaps = [sample_gap_frames(rng, spacing, cfg) for _ in range(2000)]
    assert max(gaps) == cfg.action_chunk_len
    assert min(gaps) >= 1


def test_build_observation_records_variable_spacing_gaps_are_bounded_and_increasing():
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    spacing = SpacingConfig(mode="poisson", seed=4)
    n = 500
    pose2d = random_walk_pose2d(n, seed=40)
    rgb_paths = [f"frame_{i:05d}.png" for i in range(n)]

    records = build_observation_records(pose2d, rgb_paths, cfg, spacing)
    frame_indices = [r["frame_index"] for r in records]

    assert len(records) > 10  # sanity: distribution didn't degenerate to ~0 observations
    assert frame_indices == sorted(frame_indices)
    gaps = np.diff(frame_indices)
    assert np.all(gaps >= 1) and np.all(gaps <= cfg.action_chunk_len)
    assert len(set(gaps.tolist())) > 1  # actually varies -- not secretly fixed spacing
    # Every emitted observation must still be able to anchor a full chunk.
    for r in records:
        assert r["frame_index"] + cfg.action_chunk_len <= n - 1


def test_build_observation_records_same_seed_is_reproducible():
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    n = 300
    pose2d = random_walk_pose2d(n, seed=41)
    rgb_paths = [f"frame_{i:05d}.png" for i in range(n)]

    spacing_a = SpacingConfig(mode="uniform", jitter=0.6, seed=7)
    spacing_b = SpacingConfig(mode="uniform", jitter=0.6, seed=7)
    records_a = build_observation_records(pose2d, rgb_paths, cfg, spacing_a)
    records_b = build_observation_records(pose2d, rgb_paths, cfg, spacing_b)
    assert [r["frame_index"] for r in records_a] == [r["frame_index"] for r in records_b]

    spacing_c = SpacingConfig(mode="uniform", jitter=0.6, seed=8)
    records_c = build_observation_records(pose2d, rgb_paths, cfg, spacing_c)
    assert [r["frame_index"] for r in records_a] != [r["frame_index"] for r in records_c]


# ---------------------------------------------------------------------------
# reconstruct_reference_trajectory: the round trip the "near exact tracking"
# acceptance criterion depends on, for both spacing modes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spacing",
    [
        SpacingConfig(mode="fixed"),
        SpacingConfig(mode="uniform", jitter=0.6, seed=11),
        SpacingConfig(mode="poisson", seed=12),
    ],
    ids=["fixed", "uniform", "poisson"],
)
def test_reconstruct_reference_trajectory_round_trip(spacing):
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    n = 400
    pose2d = random_walk_pose2d(n, seed=42)
    rgb_paths = [f"frame_{i:05d}.png" for i in range(n)]

    records = build_observation_records(pose2d, rgb_paths, cfg, spacing)
    assert len(records) >= 2  # otherwise this test isn't exercising cross-row stitching

    recon = reconstruct_reference_trajectory(records)
    last_covered = records[-1]["frame_index"] + cfg.action_chunk_len

    # reconstruct_reference_trajectory prepends obs_pose (== pose2d[0]) as its
    # leading pose, then decodes frame 1, 2, ... up to last_covered -- so it
    # must equal pose2d[0 : last_covered + 1] exactly, regardless of which
    # spacing mode produced the observation gaps in between.
    expected = np.asarray(pose2d[: last_covered + 1], dtype=float)
    assert recon.shape == expected.shape
    assert np.max(np.abs(recon - expected)) < 1e-9


def test_reconstruct_reference_trajectory_rejects_gap_outside_chunk_coverage():
    cfg = make_chunk_config(native_fps=20.0, obs_hz=4.0, action_chunk_len=10)
    n = 200
    pose2d = random_walk_pose2d(n, seed=43)
    rgb_paths = [f"frame_{i:05d}.png" for i in range(n)]
    records = build_observation_records(pose2d, rgb_paths, cfg, SpacingConfig(mode="fixed"))

    # Corrupt the second row's frame_index so its gap from row 0 exceeds
    # row 0's chunk length -- must be caught, not silently mis-decoded.
    corrupted = [dict(r) for r in records]
    corrupted[1]["frame_index"] = corrupted[0]["frame_index"] + cfg.action_chunk_len + 5
    with pytest.raises(ValueError):
        reconstruct_reference_trajectory(corrupted)


def test_reconstruct_reference_trajectory_empty_input():
    assert reconstruct_reference_trajectory([]).shape == (0, 3)


# ---------------------------------------------------------------------------
# PointNav's extra per-tick arrays: goal2d and segment_index
# ---------------------------------------------------------------------------
# Both optional, and the first test is the one that matters most: omitting them must
# leave an ObjectNav build *identical*, not merely equivalent. The existing corpora are
# not going to be rebuilt to accommodate a second task.
def _chunk_cfg(length=4, stride=2, fps=25.0):
    return make_chunk_config(native_fps=fps, action_chunk_len=length,
                             obs_stride_frames=stride)


def test_omitting_the_pointnav_arrays_yields_todays_exact_key_set():
    pose2d = random_walk_pose2d(40, seed=0)
    paths = [f"f{i}.png" for i in range(40)]
    records = build_observation_records(pose2d, paths, _chunk_cfg())
    assert records
    for rec in records:
        assert set(rec) == {"obs_index", "frame_index", "image", "obs_pose", "action_chunk"}


def test_obs_goal_is_the_goal_at_the_anchor_not_at_the_chunk_tail():
    """The image was rendered at `idx`. Taking the goal from `idx + length` would
    condition the model on which goal it is *about to* be given."""
    n = 40
    pose2d = random_walk_pose2d(n, seed=1)
    paths = [f"f{i}.png" for i in range(n)]
    # A goal that changes every tick, so anchor-vs-tail is unambiguous.
    goal2d = [[float(i), 0.0, 0.0] for i in range(n)]
    records = build_observation_records(pose2d, paths, _chunk_cfg(), goal2d=goal2d)
    for rec in records:
        assert rec["obs_goal"] == goal2d[rec["frame_index"]]


def test_crosses_segment_is_true_exactly_when_the_chunk_spans_a_goal_change():
    n = 40
    length, stride = 4, 2
    pose2d = random_walk_pose2d(n, seed=2)
    paths = [f"f{i}.png" for i in range(n)]
    segment_index = [0] * 20 + [1] * 20
    records = build_observation_records(
        pose2d, paths, _chunk_cfg(length, stride), segment_index=segment_index
    )
    assert records
    for rec in records:
        idx = rec["frame_index"]
        expected = segment_index[idx + length] != segment_index[idx]
        assert rec["crosses_segment"] is expected
        assert rec["segment_index"] == segment_index[idx]
    # ...and it is a minority: only the chunks straddling frame 20.
    crossing = [r for r in records if r["crosses_segment"]]
    assert 0 < len(crossing) <= length // stride


def test_a_single_segment_episode_never_crosses():
    n = 30
    records = build_observation_records(
        random_walk_pose2d(n, seed=3), [f"f{i}.png" for i in range(n)],
        _chunk_cfg(), segment_index=[0] * n,
    )
    assert records and not any(r["crosses_segment"] for r in records)


def test_the_two_arrays_are_independent():
    """A corpus may carry goals without segments or the reverse; neither implies the
    other, and asking for one must not silently require the other."""
    n = 30
    pose2d = random_walk_pose2d(n, seed=4)
    paths = [f"f{i}.png" for i in range(n)]
    goals = [[1.0, 2.0, 0.0]] * n

    only_goal = build_observation_records(pose2d, paths, _chunk_cfg(), goal2d=goals)
    assert "obs_goal" in only_goal[0] and "segment_index" not in only_goal[0]

    only_seg = build_observation_records(pose2d, paths, _chunk_cfg(),
                                         segment_index=[0] * n)
    assert "segment_index" in only_seg[0] and "obs_goal" not in only_seg[0]


@pytest.mark.parametrize("name", ["goal2d", "segment_index"])
def test_a_misaligned_extra_array_raises_naming_itself(name):
    """The same guard `rgb_paths` gets: a length mismatch binds every later chunk to the
    wrong goal, and it is silent all the way to training."""
    n = 30
    kwargs = {name: ([[0.0, 0.0, 0.0]] * 12 if name == "goal2d" else [0] * 12)}
    with pytest.raises(ValueError, match=name):
        build_observation_records(
            random_walk_pose2d(n, seed=5), [f"f{i}.png" for i in range(n)],
            _chunk_cfg(), **kwargs,
        )


def test_the_extra_arrays_do_not_disturb_the_chunks_themselves():
    """Adding a goal column must not move an action chunk by a frame."""
    n = 40
    pose2d = random_walk_pose2d(n, seed=6)
    paths = [f"f{i}.png" for i in range(n)]
    plain = build_observation_records(pose2d, paths, _chunk_cfg())
    withgoal = build_observation_records(pose2d, paths, _chunk_cfg(),
                                         goal2d=[[0.0, 0.0, 0.0]] * n,
                                         segment_index=[0] * n)
    assert len(plain) == len(withgoal)
    for a, b in zip(plain, withgoal):
        for key in ("obs_index", "frame_index", "image", "obs_pose", "action_chunk"):
            assert a[key] == b[key]


# ---------------------------------------------------------------------------
# boundary_policy: the one that decides whether arrivals exist in the corpus
# ---------------------------------------------------------------------------
# `drop` was the original default and removes every observation whose chunk reaches its
# goal. Measured on a real 2000-frame / 20-segment episode at clen=20 / stride=10: 18% of
# observations gone and NOT ONE survivor within 1.31 m of the goal it is pursuing. A model
# trained on that has never seen an arrival. These pin the fix.
def _two_segment_episode(n=60, boundary=30):
    """A robot driving +X, with the goal changing at `boundary`."""
    pose2d = [[0.05 * i, 0.0, 0.0] for i in range(n)]
    paths = [f"f{i}.png" for i in range(n)]
    segment_index = [0] * boundary + [1] * (n - boundary)
    goal2d = [[0.05 * (boundary - 1), 0.0, 0.0]] * boundary + [[0.05 * (n - 1), 0.0, 0.0]] * (n - boundary)
    return pose2d, paths, goal2d, segment_index


def test_keep_is_the_default():
    """The default must be the one whose targets are all real recorded motion.

    Both alternatives were tried against a real policy and both broke it. `drop` removes
    every arrival from the corpus (no observation survives within 1.31 m of its goal) and
    the policy overshoots. `clamp` fabricates dead time -- exactly-zero ticks go from 0.79%
    to 11.70%, and 11.65% of chunks get a fully frozen second half -- and the policy then
    stands still for long stretches, which is visible in rollout video.
    """
    import inspect
    assert inspect.signature(build_observation_records).parameters[
        "boundary_policy"].default == "keep"


def test_clamp_fabricates_dead_time_and_keep_does_not():
    """The measurement that condemned `clamp`: targets that are zero in ALL THREE dims.

    A turn-in-place is zero translation with real rotation and is fine -- the corpus is
    ~21% those. A clamped tail is zero in every component: motion that never happened.
    """
    pose2d, paths, goal2d, seg = _two_segment_episode()
    cfg = _chunk_cfg(length=10, stride=5)

    def dead_fraction(policy):
        recs = build_observation_records(pose2d, paths, cfg, goal2d=goal2d,
                                         segment_index=seg, boundary_policy=policy)
        chunks = np.asarray([r["action_chunk"] for r in recs])
        prev = np.concatenate([np.zeros((chunks.shape[0], 1, 3)), chunks], axis=1)
        step = np.diff(prev, axis=1)
        return float(((np.abs(step) < 1e-9).all(axis=2)).mean())

    assert dead_fraction("clamp") > dead_fraction("keep")
    assert dead_fraction("keep") == pytest.approx(0.0, abs=1e-9)


def test_clamp_holds_the_last_within_segment_pose():
    pose2d, paths, goal2d, seg = _two_segment_episode()
    recs = build_observation_records(pose2d, paths, _chunk_cfg(length=10, stride=5),
                                     goal2d=goal2d, segment_index=seg,
                                     boundary_policy="clamp")
    crossing = [r for r in recs if r["crosses_segment"]]
    assert crossing, "the fixture must contain a boundary-crossing chunk"
    for rec in crossing:
        chunk = np.asarray(rec["action_chunk"])
        steps = np.linalg.norm(np.diff(chunk[:, :2], axis=0), axis=1)
        assert steps[0] > 1e-6, "it should move at the start"
        assert steps[-1] == pytest.approx(0.0, abs=1e-9), "and be held at the end"
        # Never past the last pose of its own segment.
        assert chunk[:, 0].max() <= 0.05 * 29 + 1e-9


def test_clamp_keeps_every_observation_and_a_uniform_stride():
    """`drop` punches a hole in the observation stream exactly at each goal change."""
    pose2d, paths, goal2d, seg = _two_segment_episode()
    cfg = _chunk_cfg(length=10, stride=5)
    clamped = build_observation_records(pose2d, paths, cfg, goal2d=goal2d,
                                        segment_index=seg, boundary_policy="clamp")
    dropped = [r for r in build_observation_records(
        pose2d, paths, cfg, goal2d=goal2d, segment_index=seg, boundary_policy="drop")
        if not r["crosses_segment"]]
    assert len(clamped) > len(dropped)
    assert set(np.diff([r["frame_index"] for r in clamped]).tolist()) == {5}
    assert set(np.diff([r["frame_index"] for r in dropped]).tolist()) != {5}


def test_clamp_keeps_observations_that_reach_their_goal():
    """The property that actually matters: arrivals survive."""
    pose2d, paths, goal2d, seg = _two_segment_episode()
    cfg = _chunk_cfg(length=10, stride=5)
    poses = np.asarray(pose2d)

    def nearest(records):
        return min(
            float(np.linalg.norm(np.asarray(r["obs_goal"][:2]) - poses[r["frame_index"]][:2]))
            for r in records
        )

    clamped = build_observation_records(pose2d, paths, cfg, goal2d=goal2d,
                                        segment_index=seg, boundary_policy="clamp")
    dropped = [r for r in build_observation_records(
        pose2d, paths, cfg, goal2d=goal2d, segment_index=seg, boundary_policy="drop")
        if not r["crosses_segment"]]
    assert nearest(clamped) < nearest(dropped)


def test_keep_leaves_the_chunk_exactly_as_recorded():
    pose2d, paths, goal2d, seg = _two_segment_episode()
    cfg = _chunk_cfg(length=10, stride=5)
    kept = build_observation_records(pose2d, paths, cfg, goal2d=goal2d,
                                     segment_index=seg, boundary_policy="keep")
    plain = build_observation_records(pose2d, paths, cfg)
    for a, b in zip(kept, plain):
        assert a["action_chunk"] == b["action_chunk"]


def test_a_corpus_without_segments_is_untouched_by_the_policy():
    """ObjectNav has no segments; every policy must leave it identical."""
    n = 40
    pose2d = random_walk_pose2d(n, seed=9)
    paths = [f"f{i}.png" for i in range(n)]
    base = build_observation_records(pose2d, paths, _chunk_cfg())
    for policy in ("clamp", "drop", "keep"):
        other = build_observation_records(pose2d, paths, _chunk_cfg(),
                                          boundary_policy=policy)
        assert [r["action_chunk"] for r in other] == [r["action_chunk"] for r in base]
        assert set(other[0]) == set(base[0])
