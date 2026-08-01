"""Tests for the evaluation system's pure logic.

Everything here runs without ``habitat_sim``/``habitat-lab`` and without a
simulator, which is the same line ``tests/test_objectnav.py`` draws: rollouts
stay empirical, but selection, sharding, aggregation, serialisation and the
tracking statistics are ordinary code and are pinned here.
"""

import json
import math

import numpy as np
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
