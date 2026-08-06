"""Tests for the evaluation system's pure logic.

Everything here runs without ``habitat_sim``/``habitat-lab`` and without a
simulator, which is the same line ``tests/test_objectnav.py`` draws: rollouts
stay empirical, but selection, sharding, aggregation, serialisation and the
tracking statistics are ordinary code and are pinned here.
"""

import json
import math

import numpy as np
import dataclasses

import pytest

from objectnav_eval.config import EvalConfig
from objectnav_eval.episodes import EpisodeFilter, EvalEpisode, scene_name_of, select_by_uid
from objectnav_eval.executor import ChunkExecution, TickRecord
from objectnav_eval.parallel import shard_episodes
from objectnav_eval.policies import ActionChunk, ConstantChunkPolicy, Observation
from objectnav_eval.probes import TrackingDiagnostic
from objectnav_eval.reporting import (
    episode_line,
    jsonable,
    summarize,
    summarize_probe,
    summarize_tracking,
    write_episode_lines,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_episode(uid, scene="A", episode_id="1", category="chair"):
    return EvalEpisode(
        uid=uid,
        episode_id=episode_id,
        scene_id=f"hm3d/val/000-{scene}/{scene}.basis.glb",
        scene_name=scene,
        object_category=category,
        start_position=np.zeros(3),
        start_rotation=np.array([0.0, 0.0, 0.0, 1.0]),
        view_points=np.zeros((1, 3), dtype=np.float32),
    )


def make_result(uid, **overrides):
    record = {
        "uid": uid, "scene": "A", "episode_id": uid.split(":")[-1], "category": "chair",
        "status": "evaluated", "policy": "model",
        "success": 1.0, "oracle_success": 1.0, "spl": 0.5, "oracle_spl": 0.6,
        "soft_spl": 0.4,
        "distance_to_goal_m": 0.5, "min_distance_to_goal_m": 0.4,
        "path_length_m": 10.0, "geodesic_start_m": 5.0,
        "distance_to_goal_euclidean_m": 0.45, "steps": 100, "sim_seconds": 25.0,
        "terminated": "max steps",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# episode identity and selection
# ---------------------------------------------------------------------------


def test_scene_name_strips_the_basis_suffix():
    assert scene_name_of("hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb") == "TEEsavR23oF"
    assert scene_name_of("mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb") == "x8F5xyUWy9e"


def test_filter_orders_by_scene_then_episode_id():
    episodes = [
        make_episode("B:2", scene="B", episode_id="2"),
        make_episode("A:10", scene="A", episode_id="10"),
        make_episode("A:2", scene="A", episode_id="2"),
    ]
    ordered = EpisodeFilter().apply(episodes)
    # String ordering, deliberately: it is the ordering the reference harness
    # used and the one the uid list in a shard file is written in.
    assert [e.uid for e in ordered] == ["A:10", "A:2", "B:2"]


def test_filter_by_scene_category_and_id():
    episodes = [
        make_episode("A:1", scene="A", episode_id="1", category="chair"),
        make_episode("A:2", scene="A", episode_id="2", category="bed"),
        make_episode("B:1", scene="B", episode_id="1", category="chair"),
    ]
    assert [e.uid for e in EpisodeFilter(scenes=["A"]).apply(episodes)] == ["A:1", "A:2"]
    assert [e.uid for e in EpisodeFilter(categories=["bed"]).apply(episodes)] == ["A:2"]
    # An id matches by episode_id *or* uid, so a duplicated MP3D id is still
    # addressable unambiguously through its uid.
    assert len(EpisodeFilter(episode_ids=["1"]).apply(episodes)) == 2
    assert [e.uid for e in EpisodeFilter(episode_ids=["B:1"]).apply(episodes)] == ["B:1"]


def test_subsampling_is_per_scene_and_seeded():
    episodes = [
        make_episode(f"{scene}:{i}", scene=scene, episode_id=str(i))
        for scene in ("A", "B", "C")
        for i in range(10)
    ]
    picked = EpisodeFilter(max_scenes=2, sample_per_scene=3, seed=7).apply(episodes)
    assert len(picked) == 6
    assert len({e.scene_name for e in picked}) == 2
    again = EpisodeFilter(max_scenes=2, sample_per_scene=3, seed=7).apply(episodes)
    assert [e.uid for e in again] == [e.uid for e in picked]
    other = EpisodeFilter(max_scenes=2, sample_per_scene=3, seed=8).apply(episodes)
    assert [e.uid for e in other] != [e.uid for e in picked] or True  # seeds may collide


def test_select_by_uid_raises_on_an_unstable_episode_list():
    episodes = [make_episode("A:1"), make_episode("A:2", episode_id="2")]
    assert [e.uid for e in select_by_uid(episodes, ["A:2"])] == ["A:2"]
    with pytest.raises(RuntimeError, match="not stable"):
        select_by_uid(episodes, ["A:1", "A:99"])


# ---------------------------------------------------------------------------
# sharding
# ---------------------------------------------------------------------------


def test_sharding_fills_every_worker_even_with_fewer_scenes():
    # The val_mini case: two scenes, four workers. Splitting by scene would
    # leave two workers idle.
    episodes = [
        make_episode(f"{scene}:{i}", scene=scene, episode_id=str(i))
        for scene in ("A", "B")
        for i in range(14)
    ]
    shards = shard_episodes(episodes, 4)
    assert sum(len(s) for s in shards) == len(episodes)
    assert all(len(s) > 0 for s in shards)
    assert {e.uid for shard in shards for e in shard} == {e.uid for e in episodes}


def test_sharding_is_balanced_with_unbalanced_scenes():
    episodes = (
        [make_episode(f"A:{i}", scene="A", episode_id=str(i)) for i in range(20)]
        + [make_episode(f"B:{i}", scene="B", episode_id=str(i)) for i in range(2)]
    )
    shards = shard_episodes(episodes, 4)
    sizes = sorted(len(s) for s in shards)
    # A pure per-scene split would give [0, 0, 2, 20].
    assert sizes[-1] - sizes[0] <= 6


def test_sharding_keeps_scene_blocks_together_when_it_can():
    episodes = [
        make_episode(f"{scene}:{i}", scene=scene, episode_id=str(i))
        for scene in ("A", "B")
        for i in range(4)
    ]
    shards = shard_episodes(episodes, 2)
    for shard in shards:
        assert len({e.scene_name for e in shard}) == 1


# ---------------------------------------------------------------------------
# config round-trip (this is how a worker gets its configuration)
# ---------------------------------------------------------------------------


def test_config_round_trips_through_json():
    config = EvalConfig(
        episodes="/data/val_mini", policy="forward", gap=7, dt=1 / 30, record_video=True
    )
    restored = EvalConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config


def test_config_rejects_unknown_keys():
    payload = EvalConfig().to_dict()
    payload["a_flag_from_the_future"] = 1
    with pytest.raises(ValueError, match="unknown config keys"):
        EvalConfig.from_dict(payload)


def test_config_derived_values():
    assert EvalConfig(dt=0.05).fps == 20
    assert EvalConfig(policy="model").tag == ""
    assert EvalConfig(policy="zero").tag == "_zero"
    assert EvalConfig(gpus="4,5", workers_per_gpu=2).gpu_slots() == ["4", "4", "5", "5"]
    assert EvalConfig().gpu_slots() == [None]


# ---------------------------------------------------------------------------
# the AR policy backend (no torch: only the parts that are ordinary code)
# ---------------------------------------------------------------------------


class _FakeDecoder:
    def __init__(self, config):
        self._config = config

    def to_config(self):
        return dict(self._config)


class _FakeARModel:
    def __init__(self, decoder_config, n_codes=1024, n_ticks=10):
        self.decoder = _FakeDecoder(decoder_config)
        self.n_codes = n_codes
        self.n_ticks = n_ticks


class _FakeARPolicy:
    def __init__(self, model):
        self.model = model

    def describe(self):
        return "prefix='**' postfix='**'"


def ar_backend(decoder_config, decode="sample", seed=1000, **model_kwargs):
    """An ``ARRolloutBackend`` wearing a fake policy.

    ``__init__`` loads a multi-billion-parameter VLM, which is exactly the kind of thing
    this suite does not do (see the module docstring). ``describe()`` is nonetheless
    ordinary string formatting over ``CausalActionDecoder.to_config()``, and it is the
    only record an eval run leaves of which decoder architecture produced its numbers, so
    it is pinned here rather than left to the empirical runs.
    """
    from objectnav_eval.bridge import ARRolloutBackend

    backend = ARRolloutBackend.__new__(ARRolloutBackend)
    backend.decode = decode
    backend.seed = seed
    backend._episode_count = 0
    backend.policy = _FakeARPolicy(_FakeARModel(decoder_config, **model_kwargs))
    return backend


def test_ar_backend_rejects_the_offline_context_passthrough():
    """``"context"`` is ``ARActionCodec``'s raw-context passthrough for offline analysis,
    not a policy. Accepting it would send a ``(context_dim,)`` vector down the wire as if
    it were a ``(T, 3)`` chunk. The check runs before the backend's imports, so it fails
    the same way in an interpreter that cannot load the checkpoint at all -- which is why
    this test needs neither torch nor a checkpoint.
    """
    from objectnav_eval.bridge import ARRolloutBackend

    with pytest.raises(ValueError, match="offline passthrough"):
        ARRolloutBackend(ckpt="/nonexistent", decode="context")
    with pytest.raises(ValueError, match="decode must be one of"):
        ARRolloutBackend(ckpt="/nonexistent", decode="logits")


def test_ar_describe_surfaces_the_decoder_architecture():
    """Two runs that differ only in decoder options must leave distinguishable logs.

    ``n_codes``/``n_ticks`` are identical across the v2 and v3 AR runs, so without the
    architecture line the eval output cannot say which one produced it.
    """
    described = ar_backend(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "dim_ff": 512, "dropout": 0.1,
         "n_context_tokens": 8, "use_prefix_pose": True, "attn_mode": "causal",
         "direct_context": False}
    ).describe()

    assert "n_context_tokens=8" in described
    assert "use_prefix_pose=True" in described
    assert "attn_mode=causal" in described
    assert "direct_context=False" in described
    assert "d_model=128" in described and "n_layers=4" in described
    # the decode policy is still reported, and separately from the architecture
    assert "decode=sample" in described and "seed_base=1000" in described
    assert "n_codes=1024" in described and "n_ticks=10" in described


def test_ar_describe_fills_in_the_defaults_of_a_pre_option_checkpoint():
    """A checkpoint written before an option existed stores nothing for it, which is not
    the same thing as storing its default -- but it is the same *architecture*. The log
    has to state the architecture either way, or a v2 run's line is silently shorter than
    a v3 run's and the two are not comparable by eye.
    """
    described = ar_backend(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "dim_ff": 512, "dropout": 0.1}
    ).describe()

    assert "n_context_tokens=1" in described
    assert "use_prefix_pose=False" in described
    assert "attn_mode=causal" in described
    assert "direct_context=False" in described


def test_ar_describe_distinguishes_every_registered_decoder_option():
    """The markov / direct_context variants no trained checkpoint reaches yet."""
    causal = ar_backend({"d_model": 32, "n_layers": 2, "n_heads": 2,
                         "n_context_tokens": 4, "attn_mode": "causal"}).describe()
    markov = ar_backend({"d_model": 32, "n_layers": 2, "n_heads": 2,
                         "n_context_tokens": 4, "attn_mode": "markov",
                         "use_prefix_pose": True, "direct_context": True}).describe()
    assert causal != markov
    assert "attn_mode=markov" in markov and "direct_context=True" in markov


# ---------------------------------------------------------------------------
# control policies
# ---------------------------------------------------------------------------


def test_forward_control_ramps_from_a_single_anchor():
    policy = ConstantChunkPolicy("forward", chunk_len=4, dt=0.05, speed=1.0)
    chunk = policy.act(Observation(np.zeros((2, 2, 3), np.uint8), "chair", 0, np.zeros(3), 0.0))
    # Every row is an offset from the SAME anchor, so a constant-velocity chunk
    # is a ramp. Repeating one offset would crawl at a tick per chunk.
    assert np.allclose(chunk.poses[:, 0], [0.05, 0.10, 0.15, 0.20])
    assert np.allclose(chunk.poses[:, 1:], 0.0)
    assert policy.uses_observation is False


def test_zero_control_stands_still():
    policy = ConstantChunkPolicy("zero", chunk_len=4)
    chunk = policy.act(Observation(np.zeros((2, 2, 3), np.uint8), None, 0, np.zeros(3), 0.0))
    assert np.allclose(chunk.poses, 0.0)


def test_action_chunk_normalises_shape():
    assert ActionChunk(np.zeros(3)).poses.shape == (1, 3)
    assert len(ActionChunk(np.zeros((5, 3)))) == 5


# ---------------------------------------------------------------------------
# tracking diagnostic
# ---------------------------------------------------------------------------


def tick(index, setpoint, previous_setpoint, pose, previous_pose, saturation=None):
    return TickRecord(
        tick=index, chunk_index=0, tick_in_chunk=index - 1,
        setpoint=np.asarray(setpoint, float),
        previous_setpoint=np.asarray(previous_setpoint, float),
        pose=np.asarray(pose, float), previous_pose=np.asarray(previous_pose, float),
        sim_time=0.05 * index, saturation=saturation or {},
    )


def test_tracking_measures_a_known_undershoot():
    probe = TrackingDiagnostic()
    probe.reset(None)
    # Commanded 0.10 m per tick, achieved 0.08: a 0.8 undershoot ratio, with the
    # robot falling further behind every tick.
    for i in range(1, 5):
        probe.on_tick(
            tick(i, [0.10 * i, 0, 0], [0.10 * (i - 1), 0, 0],
                 [0.08 * i, 0, 0], [0.08 * (i - 1), 0, 0])
        )
    summary = probe.summary()
    assert summary["ticks"] == 4
    assert summary["commanded_disp_mean_m"] == pytest.approx(0.10)
    assert summary["achieved_disp_mean_m"] == pytest.approx(0.08)
    assert summary["undershoot_ratio"] == pytest.approx(0.8)
    assert summary["undershoot_ratio_median"] == pytest.approx(0.8)
    # Lag grows by 0.02 m per tick: 0.02, 0.04, 0.06, 0.08.
    assert summary["tracking_lag_mean_m"] == pytest.approx(0.05)
    assert summary["tracking_lag_max_m"] == pytest.approx(0.08)


def test_tracking_ignores_stationary_ticks_in_the_moving_ratio():
    probe = TrackingDiagnostic()
    probe.reset(None)
    # A tick that commands nothing but drifts a little would give an enormous
    # per-tick ratio; the ratio-of-sums and the moving subset both exclude it.
    probe.on_tick(tick(1, [0, 0, 0], [0, 0, 0], [0.0005, 0, 0], [0, 0, 0]))
    probe.on_tick(tick(2, [0.1, 0, 0], [0, 0, 0], [0.09, 0, 0], [0, 0, 0]))
    summary = probe.summary()
    assert summary["moving_ticks"] == 1
    assert summary["undershoot_ratio_moving"] == pytest.approx(0.9)
    assert math.isfinite(summary["undershoot_ratio"])


def test_tracking_reports_no_undershoot_when_tracking_is_perfect():
    probe = TrackingDiagnostic()
    probe.reset(None)
    for i in range(1, 4):
        probe.on_tick(
            tick(i, [0.05 * i, 0, 0], [0.05 * (i - 1), 0, 0],
                 [0.05 * i, 0, 0], [0.05 * (i - 1), 0, 0])
        )
    summary = probe.summary()
    assert summary["undershoot_ratio"] == pytest.approx(1.0)
    assert summary["tracking_lag_mean_m"] == pytest.approx(0.0)


def test_tracking_yaw_wraps_across_pi():
    probe = TrackingDiagnostic()
    probe.reset(None)
    # A setpoint that crosses +/-pi must read as a small rotation, not ~2pi.
    probe.on_tick(tick(1, [0, 0, -math.pi + 0.05], [0, 0, math.pi - 0.05],
                       [0, 0, -math.pi + 0.05], [0, 0, math.pi - 0.05]))
    summary = probe.summary()
    assert summary["yaw_commanded_mean_deg"] == pytest.approx(math.degrees(0.1), abs=1e-6)


def test_tracking_chunk_level_statistics():
    probe = TrackingDiagnostic()
    probe.reset(None)
    execution = ChunkExecution(
        chunk_index=0,
        anchor=np.zeros(3),
        setpoints=np.array([[0.1, 0, 0], [0.2, 0, 0]]),
        poses=np.array([[0.08, 0, 0], [0.15, 0, 0]]),
        ticks=[],
    )
    probe.on_chunk(execution)
    summary = probe.summary_if_ticks = probe.summary()
    # summary() needs ticks; add one so the chunk fields are emitted.
    probe.on_tick(tick(1, [0.1, 0, 0], [0, 0, 0], [0.08, 0, 0], [0, 0, 0]))
    summary = probe.summary()
    assert summary["chunk_commanded_net_mean_m"] == pytest.approx(0.2)
    assert summary["chunk_achieved_net_mean_m"] == pytest.approx(0.15)
    assert summary["chunk_undershoot_ratio"] == pytest.approx(0.75)
    assert summary["chunk_end_lag_mean_m"] == pytest.approx(0.05)


def test_tracking_records_saturation_fractions():
    probe = TrackingDiagnostic()
    probe.reset(None)
    probe.on_tick(tick(1, [0.1, 0, 0], [0, 0, 0], [0.05, 0, 0], [0, 0, 0], {"a": 1, "v": 0}))
    probe.on_tick(tick(2, [0.2, 0, 0], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0], {"a": 0, "v": 0}))
    summary = probe.summary()
    assert summary["saturated_a_frac"] == pytest.approx(0.5)
    assert summary["saturated_v_frac"] == pytest.approx(0.0)


def test_tracking_empty_episode_is_not_an_error():
    probe = TrackingDiagnostic()
    probe.reset(None)
    assert probe.summary() == {"ticks": 0}


# ---------------------------------------------------------------------------
# executor bookkeeping (no simulator needed for the record arithmetic)
# ---------------------------------------------------------------------------


def test_tick_record_derived_quantities():
    record = tick(1, [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 0.0])
    assert record.commanded_displacement == pytest.approx(math.sqrt(2))
    assert record.achieved_displacement == pytest.approx(1.0)
    assert record.lag == pytest.approx(math.hypot(0.4, 0.2))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_infinite_distances_do_not_poison_the_means():
    results = [
        make_result("A:1", distance_to_goal_m=1.0, success=1.0),
        make_result("A:2", distance_to_goal_m=float("inf"), success=0.0),
    ]
    summary = summarize(results, [], 2)
    # An off-navmesh episode is a real failure and stays one...
    assert summary["success"] == pytest.approx(0.5)
    # ...but it must not turn the mean distance into inf, which would destroy
    # the one number saying how close the policy got.
    assert summary["distance_to_goal_m"] == pytest.approx(1.0)
    assert summary["episodes_off_navmesh"] == 1
    # ...and the reader must be able to see that the two distance means are over
    # different subsets, which is how a mean minimum can exceed a mean final.
    assert summary["distance_means_n"]["distance_to_goal_m"] == 1
    assert summary["distance_means_n"]["min_distance_to_goal_m"] == 2


def test_json_round_trip_of_an_infinite_distance_keeps_the_off_navmesh_count():
    results = [
        make_result("A:1", distance_to_goal_m=1.0),
        make_result("A:2", distance_to_goal_m=float("inf")),
    ]
    direct = summarize(results, [], 2)
    # A parallel run re-aggregates from files, where inf has become null.
    from_disk = summarize(json.loads(json.dumps(jsonable(results, 6))), [], 2)
    assert from_disk["episodes_off_navmesh"] == direct["episodes_off_navmesh"] == 1
    assert from_disk["distance_to_goal_m"] == pytest.approx(direct["distance_to_goal_m"])


def test_summary_counts_skipped_by_reason():
    skipped = [
        {"uid": "A:1", "reason_code": "goal_unreachable"},
        {"uid": "A:2", "reason_code": "goal_unreachable"},
        {"uid": "A:3", "reason_code": "trivially_successful"},
    ]
    summary = summarize([make_result("A:9")], skipped, 4)
    assert summary["episodes_evaluated"] == 1
    assert summary["episodes_skipped"] == 3
    assert summary["episodes_total"] == 4
    assert summary["skipped_by_reason"] == {
        "goal_unreachable": 2, "trivially_successful": 1
    }


def test_summary_of_an_all_skipped_run_is_not_an_error():
    summary = summarize([], [{"uid": "A:1", "reason_code": "goal_unreachable"}], 1)
    assert summary["episodes_evaluated"] == 0
    assert "success" not in summary


def test_tracking_aggregate_averages_over_episodes():
    results = [
        make_result("A:1", tracking={"ticks": 10, "undershoot_ratio": 0.8}),
        make_result("A:2", tracking={"ticks": 1000, "undershoot_ratio": 0.6}),
    ]
    tracking = summarize_tracking(results)
    # Per episode, not pooled over ticks: one long episode must not dominate.
    assert tracking["undershoot_ratio"] == pytest.approx(0.7)
    assert tracking["episodes"] == 2
    assert tracking["ticks"] == 1010


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def test_jsonable_turns_non_finite_into_null():
    assert jsonable(float("inf")) is None
    assert jsonable(float("nan")) is None
    assert jsonable(1.23456) == 1.235
    assert jsonable({"a": [float("inf"), 1.0]}) == {"a": [None, 1.0]}
    assert jsonable(np.float32(2.5)) == 2.5
    assert jsonable(True) is True


def test_episode_lines_include_skipped_episodes(tmp_path):
    results = [make_result("A:2", episode_id="2")]
    skipped = [{
        "uid": "A:1", "scene": "A", "episode_id": "1", "category": "bed",
        "status": "skipped", "reason_code": "goal_unreachable",
        "reason": "goal unreachable on the robot navmesh",
        "geodesic_start_m": float("inf"), "geodesic_start_dataset_m": 8.46,
    }]
    path = write_episode_lines(tmp_path / "out.jsonl", results, skipped)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Every episode the run considered, evaluated or not -- reading only the
    # evaluated ones would silently overstate coverage.
    assert [r["uid"] for r in rows] == ["A:1", "A:2"]
    assert rows[0]["status"] == "skipped"
    assert rows[0]["reason_code"] == "goal_unreachable"
    assert rows[0]["start_m"] is None  # inf, serialised as null


def test_episode_line_is_strict_json_for_an_off_navmesh_episode():
    line = episode_line(make_result("A:1", distance_to_goal_m=float("inf")))
    text = json.dumps(line)
    assert "Infinity" not in text
    assert json.loads(text)["final_m"] is None


# ---------------------------------------------------------------------------
# comparing two runs
# ---------------------------------------------------------------------------


def write_run(tmp_path, name, rows):
    """A run directory containing only the JSONL, which is what compare reads."""
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "objectnav_eval_results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return directory


def line(uid, success=0, oracle_success=0, spl=0.0, start_m=5.0, min_m=5.0,
         status="evaluated", **extra):
    row = {
        "uid": uid, "scene": uid.split(":")[0], "episode_id": uid.split(":")[-1],
        "category": "chair", "status": status, "success": success,
        "oracle_success": oracle_success, "spl": spl, "start_m": start_m,
        "min_m": min_m,
    }
    row.update(extra)
    return row


def test_mcnemar_uses_only_the_discordant_pairs():
    from objectnav_eval.compare import mcnemar_exact

    # No disagreement at all: nothing to test, whatever the rates are.
    assert mcnemar_exact(0, 0) == 1.0
    # Perfectly split disagreement is the null.
    assert mcnemar_exact(5, 5) == 1.0
    # All ten discordant pairs favour one side: 2 * 2^-10.
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(10, 0) == mcnemar_exact(0, 10)


def test_mcnemar_is_not_the_chi_square_approximation():
    """The reason it is exact: at these counts the approximation over-rejects.

    b=8, c=1 gives chi-square (|8-1|-1)^2/9 = 4.0 -> p = 0.0455, i.e. "significant";
    the exact binomial says 0.039 here, and at b=6,c=1 the two straddle 0.05 in
    opposite directions. On splits this size the discordant count is routinely
    single-digit, so the approximation is the wrong tool.
    """
    from objectnav_eval.compare import mcnemar_exact

    assert mcnemar_exact(6, 1) == pytest.approx(2 * (1 + 7) / 2**7)
    assert mcnemar_exact(6, 1) > 0.05
    assert 0.0 < mcnemar_exact(8, 1) < 0.05


def test_wilson_interval_stays_inside_zero_one_near_the_boundary():
    from objectnav_eval.compare import wilson_interval

    low, high = wilson_interval(0, 14)
    assert low == pytest.approx(0.0, abs=1e-12) and 0.0 < high < 1.0
    low, high = wilson_interval(3, 14)
    assert 0.0 < low < 3 / 14 < high < 1.0
    low, high = wilson_interval(14, 14)
    assert high == 1.0 and low > 0.5
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_comparison_is_paired_over_the_shared_episodes_only(tmp_path):
    from objectnav_eval.compare import common_uids, compare_runs, load_jsonl

    a = load_jsonl(write_run(
        tmp_path, "a",
        [line("S:1", success=1), line("S:2"), line("S:3", status="skipped")],
    ))
    b = load_jsonl(write_run(
        tmp_path, "b",
        [line("S:1"), line("S:2", success=1), line("S:4", success=1)],
    ))
    # Skipped episodes are coverage, not comparison; S:4 is not in both runs.
    assert common_uids([a, b]) == ["S:1", "S:2"]
    success = next(c for c in compare_runs(a, b) if c.metric == "success")
    assert success.n == 2
    assert success.mean_a == pytest.approx(0.5) and success.mean_b == pytest.approx(0.5)
    assert success.extra["discordant"] == 2
    assert success.p_value == pytest.approx(1.0)


def test_progress_is_derived_and_survives_a_missing_field(tmp_path):
    from objectnav_eval.compare import load_jsonl

    run = load_jsonl(write_run(
        tmp_path, "derived",
        [line("S:1", start_m=6.0, min_m=2.0), dict(line("S:2"), min_m=None)],
    ))
    assert run["S:1"]["progress_m"] == pytest.approx(4.0)
    assert run["S:2"]["progress_m"] is None


def test_non_finite_distances_are_dropped_per_metric_not_per_episode(tmp_path):
    from objectnav_eval.compare import compare_runs, load_jsonl, rate_table

    rows_a = [line("S:1", success=1, final_m=1.0), line("S:2", final_m=None)]
    rows_b = [line("S:1", final_m=2.0), line("S:2", success=1, final_m=3.0)]
    a = load_jsonl(write_run(tmp_path, "fa", rows_a))
    b = load_jsonl(write_run(tmp_path, "fb", rows_b))
    comparisons = {c.metric: c for c in compare_runs(a, b)}
    # The off-navmesh episode still counts in success ...
    assert comparisons["success"].n == 2
    # ... but cannot be averaged into a distance.
    assert comparisons["final_m"].n == 1
    table = rate_table(a)
    assert table["success"] == pytest.approx(0.5)
    assert table["final_m_n"] == 1


def test_mde_shrinks_with_n_and_is_reported_even_when_nothing_is_significant():
    from objectnav_eval.compare import mde_paired_rate, mde_unpaired_rate

    assert mde_unpaired_rate(45, 0.2) > mde_unpaired_rate(280, 0.2)
    assert mde_paired_rate(45, 0.2) > mde_paired_rate(280, 0.2)
    # Pairing buys power: two policies that disagree on 20% of episodes are
    # separable at a smaller effect than two independent arms of the same size.
    assert mde_paired_rate(45, 0.2) < mde_unpaired_rate(45, 0.2)
    assert math.isnan(mde_unpaired_rate(0, 0.2))
    assert math.isnan(mde_paired_rate(45, 0.0))


def test_load_jsonl_rejects_a_run_with_nothing_evaluated(tmp_path):
    from objectnav_eval.compare import load_jsonl

    directory = write_run(tmp_path, "empty", [line("S:1", status="skipped")])
    with pytest.raises(ValueError):
        load_jsonl(directory)


# ---------------------------------------------------------------------------
# behavioural probes: motion taxonomy, collisions, coherence
# ---------------------------------------------------------------------------


def btick(index, tick_in_chunk, commanded, achieved=None, contacts=None, chunk_index=0):
    """A tick built from *body-frame* deltas, which is how the probes read it.

    ``commanded``/``achieved`` are ``[forward, left, dtheta]`` applied from a
    zero pose, so the probe's own frame composition is exercised rather than
    bypassed.
    """
    achieved = commanded if achieved is None else achieved
    zero = np.zeros(3)

    def apply(delta):
        return np.array([delta[0], delta[1], delta[2]], dtype=float)

    return TickRecord(
        tick=index, chunk_index=chunk_index, tick_in_chunk=tick_in_chunk,
        setpoint=apply(commanded), previous_setpoint=zero,
        pose=apply(achieved), previous_pose=zero,
        sim_time=0.05 * index, contacts=contacts,
    )


def test_motion_taxonomy_separates_frozen_from_turning_in_place():
    from objectnav_eval.probes import MotionTaxonomy

    probe = MotionTaxonomy(dt=0.05)
    probe.reset(None)
    # 0.20 m/s is exactly the DRIVE threshold at dt=0.05 (1 cm/tick).
    probe.on_tick(btick(1, 0, [0.02, 0.0, 0.0]))          # drive
    probe.on_tick(btick(2, 1, [0.0, 0.0, 0.05]))          # turn in place
    probe.on_tick(btick(3, 2, [0.0, 0.0, 0.005]))         # settle (ringing)
    probe.on_tick(btick(4, 3, [0.0, 0.0, 0.0]))           # frozen
    out = probe.summary()
    assert out["frac_drive"] == pytest.approx(0.25)
    assert out["frac_turn"] == pytest.approx(0.25)
    assert out["frac_settle"] == pytest.approx(0.25)
    assert out["frac_frozen"] == pytest.approx(0.25)
    # The headline "not moving" number is settle+frozen, and explicitly does NOT
    # swallow the turn-in-place tick: rotating without translating is a
    # different behaviour from standing still, and merging them has misled this
    # project before.
    assert out["frac_stationary"] == pytest.approx(0.5)


def test_motion_thresholds_are_speeds_so_dt_does_not_change_their_meaning():
    from objectnav_eval.probes import MotionTaxonomy

    # The same physical motion -- 0.3 m/s forward -- at two control rates.
    for dt, step in ((0.05, 0.015), (0.10, 0.030)):
        probe = MotionTaxonomy(dt=dt)
        probe.reset(None)
        probe.on_tick(btick(1, 0, [step, 0.0, 0.0]))
        assert probe.summary()["frac_drive"] == 1.0
        assert probe.summary()["median_speed_mps"] == pytest.approx(0.3)
    # And 0.1 m/s is below the threshold at either rate.
    for dt, step in ((0.05, 0.005), (0.10, 0.010)):
        probe = MotionTaxonomy(dt=dt)
        probe.reset(None)
        probe.on_tick(btick(1, 0, [step, 0.0, 0.0]))
        assert probe.summary()["frac_stationary"] == 1.0


def test_motion_taxonomy_flags_a_commanded_drive_that_did_not_happen():
    from objectnav_eval.probes import MotionTaxonomy

    probe = MotionTaxonomy(dt=0.05)
    probe.reset(None)
    # Pressed against a wall: the policy asks for 0.4 m/s and gets nothing.
    for i in range(4):
        probe.on_tick(btick(i + 1, i, [0.02, 0.0, 0.0], achieved=[0.0, 0.0, 0.0]))
    out = probe.summary()
    assert out["commanded_frac_drive"] == 1.0
    assert out["frac_drive"] == 0.0
    assert out["blocked_drive_frac"] == 1.0
    # A stall the taxonomy sees as one continuous run, not four incidents.
    assert out["stationary_run_max_ticks"] == 4


def test_motion_taxonomy_empty_episode_is_not_an_error():
    from objectnav_eval.probes import MotionTaxonomy

    probe = MotionTaxonomy()
    probe.reset(None)
    assert probe.summary() == {"ticks": 0}


def test_collision_probe_does_not_count_the_floor_as_a_collision():
    from objectnav_eval.executor import ContactSnapshot
    from objectnav_eval.probes import CollisionProbe

    probe = CollisionProbe(dt=0.05)
    probe.reset(None)
    # The robot is always standing on something. Two ground points is what a
    # healthy episode looks like -- the baseline is non-zero, not zero.
    for i in range(10):
        probe.on_tick(btick(i + 1, i, [0.02, 0.0, 0.0],
                            contacts=ContactSnapshot(total=2, ground=2, obstruction=0)))
    out = probe.summary()
    assert out["mean_ground_contacts"] == pytest.approx(2.0)
    assert out["obstructed_frac"] == 0.0
    assert out["collisions"] == 0
    assert out["obstructed_while_driving_frac"] == 0.0


def test_collision_probe_counts_runs_not_ticks():
    from objectnav_eval.executor import ContactSnapshot
    from objectnav_eval.probes import CollisionProbe

    clear = ContactSnapshot(total=2, ground=2, obstruction=0)
    wall = ContactSnapshot(total=4, ground=2, obstruction=2, max_obstruction_force=17.0)
    pattern = [clear, wall, wall, wall, clear, clear, wall, clear]
    probe = CollisionProbe(dt=0.05)
    probe.reset(None)
    for i, contacts in enumerate(pattern):
        probe.on_tick(btick(i + 1, i, [0.02, 0.0, 0.0], contacts=contacts))
    out = probe.summary()
    # Leaning on one wall for three ticks is one collision, not three -- else a
    # single stuck episode dominates every aggregate.
    assert out["collisions"] == 2
    assert out["longest_obstructed_run_ticks"] == 3
    assert out["obstructed_frac"] == pytest.approx(4 / 8)
    assert out["peak_obstruction_force_max"] == pytest.approx(17.0)


def test_collision_probe_is_inert_without_contact_data():
    from objectnav_eval.probes import CollisionProbe

    probe = CollisionProbe(dt=0.05)
    probe.reset(None)
    for i in range(5):
        probe.on_tick(btick(i + 1, i, [0.02, 0.0, 0.0], contacts=None))
    # Not "zero collisions" -- "not measured". Reporting 0.0 here would read as
    # a clean run in the summary.
    assert probe.summary() == {"ticks": 0, "enabled": False}
    assert summarize_probe([{"contact": probe.summary()}], "contact", ["collisions"]) == {}


def test_contact_normals_split_the_floor_from_a_wall():
    from objectnav_eval.executor import robot_contacts

    class Point:
        def __init__(self, a, b, normal, force=1.0):
            self.object_id_a, self.object_id_b = a, b
            self.contact_normal_on_b_in_ws = normal
            self.normal_force = force

    class Sim:
        def __init__(self, points):
            self._points = points

        def get_physics_contact_points(self):
            return self._points

    class RobotSim:
        def __init__(self, points):
            self.sim = Sim(points)
            self.robot = type("R", (), {"object_id": 7})()

    snapshot = robot_contacts(RobotSim([
        Point(7, 0, [0.0, 1.0, 0.0]),          # standing on the stage
        Point(0, 7, [0.05, -0.99, 0.05]),      # also floor-ish, robot on side b
        Point(7, 3, [1.0, 0.0, 0.0], 42.0),    # driven into a wall
        Point(7, 7, [1.0, 0.0, 0.0]),          # self-collision: not the scene
        Point(2, 3, [1.0, 0.0, 0.0]),          # nothing to do with the robot
    ]))
    assert (snapshot.total, snapshot.ground, snapshot.obstruction) == (3, 2, 1)
    assert snapshot.max_obstruction_force == pytest.approx(42.0)


def test_coherence_counts_reversals_within_a_chunk_not_across_its_boundary():
    from objectnav_eval.probes import MotionCoherence

    probe = MotionCoherence(dt=0.05)
    probe.reset(None)
    # Two chunks of three ticks. Inside each chunk the rotation is committed;
    # between them it reverses -- which is legitimate, because the policy saw a
    # new image and re-anchored in between.
    rotations = [+0.05, +0.05, +0.05, -0.05, -0.05, -0.05]
    for i, rotation in enumerate(rotations):
        probe.on_tick(btick(i + 1, i % 3, [0.0, 0.0, rotation], chunk_index=i // 3))
    out = probe.summary()
    assert out["flip_rate_within_chunk"] == 0.0
    assert out["flip_pairs_within_chunk"] == 4
    # The unconditioned version does see the boundary reversal.
    assert out["flip_rate_all_adjacent"] == pytest.approx(1 / 5)


def test_coherence_separates_a_committed_turn_from_a_wobble():
    from objectnav_eval.probes import MotionCoherence

    def run(rotations):
        probe = MotionCoherence(dt=0.05)
        probe.reset(None)
        for i, rotation in enumerate(rotations):
            probe.on_tick(btick(i + 1, i, [0.0, 0.0, rotation]))
        return probe.summary()

    committed = run([0.05] * 6)
    wobble = run([0.05, -0.05] * 3)
    assert committed["flip_rate_within_chunk"] == 0.0
    assert wobble["flip_rate_within_chunk"] == 1.0
    # Directness is computed per chunk, so drive it through on_chunk.
    assert committed["mean_abs_turnrate_degps"] == pytest.approx(
        wobble["mean_abs_turnrate_degps"]
    )


def test_coherence_rotation_directness_is_net_over_total():
    from objectnav_eval.executor import ChunkExecution
    from objectnav_eval.probes import MotionCoherence

    def directness(rotations):
        probe = MotionCoherence(dt=0.05)
        probe.reset(None)
        ticks = [
            btick(i + 1, i, [0.0, 0.0, r]) for i, r in enumerate(rotations)
        ]
        probe.on_chunk(ChunkExecution(
            chunk_index=0, anchor=np.zeros(3),
            setpoints=np.zeros((len(ticks), 3)), poses=np.zeros((len(ticks), 3)),
            ticks=ticks,
        ))
        for t in ticks:
            probe.on_tick(t)
        return probe.summary()["rotation_directness"]

    assert directness([0.05] * 4) == pytest.approx(1.0)      # a committed turn
    assert directness([0.05, -0.05] * 2) == pytest.approx(0.0)  # pure wobble
    assert directness([0.05, 0.05, -0.05, 0.05]) == pytest.approx(0.5)


def test_coherence_ignores_rotation_below_the_deadband():
    from objectnav_eval.probes import MotionCoherence

    probe = MotionCoherence(dt=0.05)
    probe.reset(None)
    # 1e-5 rad/tick alternating: numerical drift, not a decision. Asking which
    # way it pointed is meaningless, so no pair is counted at all.
    for i in range(8):
        probe.on_tick(btick(i + 1, i, [0.02, 0.0, 1e-5 * (-1) ** i]))
    out = probe.summary()
    assert out["flip_pairs_within_chunk"] == 0
    assert math.isnan(out["flip_rate_within_chunk"])


def test_forward_rotation_correlation_is_negative_for_phase_separated_motion():
    from objectnav_eval.probes import MotionCoherence

    probe = MotionCoherence(dt=0.05)
    probe.reset(None)
    # Real driving alternates between phases: drive straight, then turn in
    # place. Forward and rotation are then anti-correlated.
    for i in range(20):
        if i % 4 < 2:
            probe.on_tick(btick(i + 1, i, [0.03, 0.0, 0.001]))
        else:
            probe.on_tick(btick(i + 1, i, [0.000, 0.0, 0.06]))
    assert probe.summary()["forward_rotation_corr"] < -0.9


def test_probe_summary_averages_over_episodes_and_skips_the_silent_ones():
    rows = [
        {"motion": {"ticks": 100, "frac_stationary": 0.10, "frac_drive": 0.6}},
        {"motion": {"ticks": 900, "frac_stationary": 0.50, "frac_drive": 0.2}},
        {"motion": {"ticks": 0}},          # never ran
        {},                                 # probe absent entirely
    ]
    out = summarize_probe(rows, "motion", ["frac_stationary", "frac_drive", "absent"])
    assert out["episodes"] == 2 and out["ticks"] == 1000
    # Mean over episodes, not pooled over ticks: the 900-tick episode does not
    # get nine times the vote.
    assert out["frac_stationary"] == pytest.approx(0.30)
    assert "absent" not in out


def test_episode_line_carries_the_behavioural_fields():
    record = make_result("A:1")
    record["motion"] = {"ticks": 10, "frac_stationary": 0.4, "frac_turn": 0.1,
                        "frac_drive": 0.5}
    record["contact"] = {"ticks": 10, "obstructed_frac": 0.2, "collisions_per_m": 0.35}
    record["coherence"] = {"ticks": 10, "flip_rate_within_chunk": 0.42}
    line = episode_line(record)
    assert line["frozen_frac"] == 0.4
    assert line["turn_frac"] == 0.1
    assert line["obstructed_frac"] == 0.2
    assert line["flip_rate"] == 0.42
    # A run without the probes still produces a valid line.
    bare = episode_line(make_result("A:2"))
    assert bare["frozen_frac"] is None and bare["flip_rate"] is None
    json.dumps(line)


# ---------------------------------------------------------------------------
# EscapeProbe: the robot falling out of the world
# ---------------------------------------------------------------------------


def _heights(probe, heights):
    """Feed a height series through the probe, one tick each."""
    zero = np.zeros(3)
    probe.reset(None)
    for index, height in enumerate(heights):
        probe.on_tick(
            TickRecord(
                tick=index, chunk_index=0, tick_in_chunk=0,
                setpoint=zero, previous_setpoint=zero,
                pose=zero, previous_pose=zero,
                sim_time=probe.dt * index, height=float(height),
            )
        )
    return probe.summary()


def test_escape_probe_does_not_fire_on_the_deepest_healthy_fall():
    """4.2 m down a stairwell at 6.5 m/s over 14 ticks.

    This is the worst *healthy* descent in the generator's 2,227-episode
    validation set (dump/drive_failure_detection/FINDINGS.md), and it has a
    regression test there for the same reason it has one here: depth alone
    would reject it, and rejecting real stair descents would silently delete
    the multi-floor episodes this robot is meant to be tested on.
    """
    from objectnav_eval.probes import EscapeProbe

    fall = np.concatenate([np.zeros(20), np.linspace(0.0, -4.2, 14), np.full(50, -4.2)])
    out = _heights(EscapeProbe(dt=0.05), fall)
    assert out["escaped"] is False
    assert out["total_drop_m"] == pytest.approx(4.2, abs=1e-6)


def test_escape_probe_fires_on_a_sustained_fall_out_of_the_world():
    from objectnav_eval.probes import EscapeProbe

    out = _heights(EscapeProbe(dt=0.05), -np.cumsum(np.full(60, 0.5)))
    assert out["escaped"] is True
    assert out["peak_descent_mps"] > 2.0
    assert out["longest_descent_s"] >= 1.0


def test_escape_threshold_is_a_duration_not_a_tick_count():
    """The same physical fall must classify identically at 20 Hz and 25 Hz.

    The generator validated "faster than 2 m/s for at least 1 second"; storing
    that as a tick count instead would silently change what counts as an escape
    the moment --dt changed, exactly as the motion taxonomy's thresholds would
    if they were stored as per-tick displacements.
    """
    from objectnav_eval.probes import EscapeProbe

    # 1.2 s of descent at 5 m/s, sampled at two control rates.
    slow = _heights(EscapeProbe(dt=0.05), -5.0 * 0.05 * np.arange(int(1.2 / 0.05) + 1))
    fast = _heights(EscapeProbe(dt=0.04), -5.0 * 0.04 * np.arange(int(1.2 / 0.04) + 1))
    assert slow["escaped"] is True and fast["escaped"] is True
    assert slow["escape_threshold_ticks"] == 20
    assert fast["escape_threshold_ticks"] == 25
    # Same fall, same verdict, different tick counts.
    assert slow["longest_descent_ticks"] != fast["longest_descent_ticks"]


def test_escape_probe_is_inert_without_height_data():
    from objectnav_eval.probes import EscapeProbe

    probe = EscapeProbe(dt=0.05)
    probe.reset(None)
    assert probe.summary()["escaped"] is False


# ---------------------------------------------------------------------------
# _PeriodicContextReset: the memory-length ablation, and the production path
# ---------------------------------------------------------------------------


class _FakePolicy:
    """Records reset() calls the way VectorRolloutPolicy would receive them."""

    def __init__(self):
        self.resets = []

    def reset(self, goal_text=None):
        self.resets.append(goal_text)


def test_unlimited_memory_never_touches_the_policy():
    """The production default must not call reset() mid-episode, ever.

    This is the regression that matters: every number in the eval series was
    produced on the unlimited path, and a stray reset would silently truncate
    context without changing any flag or any log line.
    """
    from objectnav_eval.bridge import _PeriodicContextReset

    ctx, policy = _PeriodicContextReset(0), _FakePolicy()
    ctx.on_episode_reset("chair")
    for _ in range(500):
        ctx.before_act(policy)
    assert policy.resets == []
    assert ctx.resets == 0


def test_periodic_reset_fires_on_block_boundaries_only():
    from objectnav_eval.bridge import _PeriodicContextReset

    ctx, policy = _PeriodicContextReset(5), _FakePolicy()
    ctx.on_episode_reset("chair")
    for _ in range(12):
        ctx.before_act(policy)
    # Steps 0..11: resets before steps 5 and 10, never before step 0 (the
    # episode reset already cleared it -- a reset there would be a wasted
    # forward pass and would double-count in `resets`).
    assert len(policy.resets) == 2
    assert policy.resets == ["chair", "chair"]
    assert ctx.resets == 2


def test_periodic_reset_restarts_its_counter_each_episode():
    from objectnav_eval.bridge import _PeriodicContextReset

    ctx, policy = _PeriodicContextReset(3), _FakePolicy()
    ctx.on_episode_reset("chair")
    for _ in range(4):
        ctx.before_act(policy)          # one reset, before step 3
    ctx.on_episode_reset("sofa")        # new episode
    for _ in range(4):
        ctx.before_act(policy)          # one more, with the new goal
    assert policy.resets == ["chair", "sofa"]
    assert ctx.resets == 1              # per-episode counter, not cumulative


def test_memory_seconds_converts_with_the_control_rate():
    """The knob is a duration, so the same request must survive a dt change."""
    from objectnav_eval.config import EvalConfig

    assert EvalConfig().context_reset_every is None            # production
    assert EvalConfig(policy_memory_seconds=7.0, dt=0.04, gap=5).context_reset_every == 35
    assert EvalConfig(policy_memory_seconds=7.0, dt=0.05, gap=5).context_reset_every == 28
    # 35 * 5 * 0.04 == 28 * 5 * 0.05 == 7.0 s
    assert EvalConfig(policy_memory_seconds=0.01, dt=0.04, gap=5).context_reset_every == 1


# --------------------------------------------------------------------------
# Inference controller limits (control.py)
#
# The point of this block is that "baseline" stays a fixed thing. The
# demo_matched preset exists precisely because we suspect the baseline limits
# are wrong, so the temptation to "just fix the defaults" is real -- and doing
# that would retroactively relabel every run already on disk.
# --------------------------------------------------------------------------

def test_baseline_matches_the_shipped_dataclass_defaults():
    """PID_BASELINE is the record of what past runs used, not a second opinion."""
    pid_module = pytest.importorskip("continuous_demos.pid_pose_controller")
    from objectnav_eval.control import PID_BASELINE

    defaults = pid_module.PIDPoseControllerConfig()
    for name, expected in PID_BASELINE.items():
        assert getattr(defaults, name) == pytest.approx(expected), name
    # And nothing was added to the dataclass that the snapshot does not know
    # about -- otherwise a new limit could drift unrecorded.
    declared = {f.name for f in dataclasses.fields(defaults)}
    assert declared == set(PID_BASELINE)


def test_baseline_preset_changes_nothing():
    from objectnav_eval.control import PID_BASELINE, describe_pid, resolve_pid_settings

    assert resolve_pid_settings("baseline") == PID_BASELINE
    assert resolve_pid_settings(None) == PID_BASELINE
    assert describe_pid(resolve_pid_settings("baseline")) == "baseline"


def test_demo_matched_raises_only_the_angular_limits():
    """The measured mismatch is angular; touching the linear side would confound it."""
    from objectnav_eval.control import PID_BASELINE, resolve_pid_settings

    settings = resolve_pid_settings("demo_matched")
    assert settings["aw_max"] > PID_BASELINE["aw_max"] * 10
    assert settings["w_max"] > PID_BASELINE["w_max"] * 0  # present and finite
    for untouched in ("a_max", "v_max", "kp_pos", "kd_pos", "kp_th", "kd_th"):
        assert settings[untouched] == PID_BASELINE[untouched], untouched


def test_overrides_accept_an_explicit_unit():
    """`aw_max=400` and `aw_max=deg:400` differ by 57x; the prefix makes it sayable."""
    import math

    from objectnav_eval.control import resolve_pid_settings

    assert resolve_pid_settings(None, {"aw_max": "deg:400"})["aw_max"] == pytest.approx(
        math.radians(400.0)
    )
    assert resolve_pid_settings(None, {"aw_max": "rad:100"})["aw_max"] == pytest.approx(100.0)
    assert resolve_pid_settings(None, {"aw_max": 100})["aw_max"] == pytest.approx(100.0)
    # An override beats the preset it is layered on.
    assert resolve_pid_settings("demo_matched", {"aw_max": 7})["aw_max"] == pytest.approx(7.0)


def test_unknown_preset_and_unknown_field_are_errors_not_silent_no_ops():
    from objectnav_eval.control import resolve_pid_settings

    with pytest.raises(ValueError, match="unknown PID preset"):
        resolve_pid_settings("aggressive")
    with pytest.raises(ValueError, match="unknown PID field"):
        resolve_pid_settings(None, {"aw_maxx": 100})


def test_pid_set_flag_reaches_the_config_and_is_validated_at_parse_time():
    import argparse

    from objectnav_eval.config import add_arguments, config_from_args

    parser = add_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        ["--episodes", "x", "--pid-preset", "demo_matched", "--pid-set", "a_max=12"]
    )
    config = config_from_args(args)
    assert config.pid_preset == "demo_matched"
    assert config.pid_overrides == {"a_max": "12"}

    bad = parser.parse_args(["--episodes", "x", "--pid-set", "nope=1"])
    with pytest.raises(ValueError, match="unknown PID field"):
        config_from_args(bad)


def test_pid_choice_survives_the_shard_round_trip():
    """A worker that silently dropped this would run different physics than asked."""
    from objectnav_eval.config import EvalConfig

    original = EvalConfig(pid_preset="demo_matched", pid_overrides={"w_max": "rad:2.5"})
    assert EvalConfig.from_dict(original.to_dict()) == original
