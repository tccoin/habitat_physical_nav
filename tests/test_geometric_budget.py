"""The geometric episode budget: the arithmetic, the rule, and the wiring.

Everything here is pure logic -- no simulator. The one test that exercises the
runner drives it with a fake executor whose ticks are prescribed, which is the
only way to assert the thing that actually matters about the termination rule:
that it cuts *at the tick* the budget runs out and not at the next chunk
boundary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from objectnav_eval.budget import (
    DEFAULT_BUDGET_UNITS,
    DEFAULT_TIME_BACKSTOP_S,
    TERMINATED_BUDGET,
    TERMINATED_ORACLE,
    TERMINATED_POLICY,
    TERMINATED_STEPS,
    TERMINATED_TIME,
    BudgetRule,
    GeometricBudget,
    budget_trace,
    rescore_record,
    rescore_results,
    terminated_code,
    trace_from_record,
    wrap_angle,
)

# ---------------------------------------------------------------------------
# Known values. One unit is 0.25 m or 30 degrees, by definition, so these are
# arithmetic rather than measurements and any drift in them is a bug.
# ---------------------------------------------------------------------------


def _drive(distance: float, steps: int = 20):
    """Poses for a straight drive along +X, starting seeded at the origin."""
    return [[distance * i / steps, 0.0, 0.0] for i in range(steps + 1)]


def test_one_metre_of_driving_is_four_units():
    budget = GeometricBudget()
    budget.reset([0.0, 0.0, 0.0])
    for pose in _drive(1.0)[1:]:
        budget.observe(pose)
    assert budget.spent == pytest.approx(4.0)
    assert budget.translation_units == pytest.approx(4.0)
    assert budget.rotation_units == 0.0
    assert budget.translation_m == pytest.approx(1.0)


def test_a_full_turn_in_place_is_twelve_units():
    budget = GeometricBudget()
    budget.reset([0.0, 0.0, 0.0])
    # Sampled in 24 steps of 15 degrees: sampling coarser than pi would alias
    # through the wrap, which is exactly what `wrap_angle` is there to make
    # visible rather than silent.
    for i in range(1, 25):
        budget.observe([0.0, 0.0, i * math.pi / 12.0])
    assert budget.spent == pytest.approx(12.0)
    assert budget.rotation_units == pytest.approx(12.0)
    assert budget.translation_units == pytest.approx(0.0)


def test_a_one_metre_diameter_circle_costs_both_currencies():
    """pi metres of arc (12.57 units) plus one full revolution (12 units)."""
    budget = GeometricBudget()
    radius, n = 0.5, 2000
    poses = [
        [radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n),
         wrap_angle(2 * math.pi * i / n + math.pi / 2)]
        for i in range(n + 1)
    ]
    budget.reset(poses[0])
    for pose in poses[1:]:
        budget.observe(pose)
    expected = math.pi / 0.25 + 12.0
    assert expected == pytest.approx(24.566, abs=1e-3)
    # A polygon inscribed in the circle is slightly shorter than the arc, so the
    # translation term converges from below -- 2000 samples is within 1e-5.
    assert budget.spent == pytest.approx(expected, rel=1e-4)
    assert budget.rotation_units == pytest.approx(12.0)


def test_heading_wrap_does_not_charge_a_full_turn():
    """Crossing the pi/-pi seam is a small turn, not a 12-unit one."""
    budget = GeometricBudget()
    budget.reset([0.0, 0.0, math.pi - 0.01])
    budget.observe([0.0, 0.0, -math.pi + 0.01])
    assert budget.spent == pytest.approx(0.02 / (math.pi / 6))


def test_the_first_sample_only_seeds():
    budget = GeometricBudget()
    budget.reset([3.0, -2.0, 1.0])
    assert budget.spent == 0.0
    assert budget.num_samples == 1


def test_reset_clears_a_previous_episode():
    budget = GeometricBudget()
    budget.reset([0.0, 0.0, 0.0])
    budget.observe([1.0, 0.0, 0.0])
    budget.reset([0.0, 0.0, 0.0])
    assert budget.spent == 0.0
    assert budget.translation_m == 0.0


def test_exhaustion_and_remaining():
    budget = GeometricBudget(budget=10.0)
    budget.reset([0.0, 0.0, 0.0])
    budget.observe([2.0, 0.0, 0.0])  # 8 units
    assert not budget.exhausted
    assert budget.remaining == pytest.approx(2.0)
    budget.observe([2.5, 0.0, 0.0])  # +2 units -> exactly the budget
    assert budget.exhausted
    assert budget.remaining == pytest.approx(0.0)


def test_the_split_between_translation_and_rotation_is_reported():
    budget = GeometricBudget()
    budget.reset([0.0, 0.0, 0.0])
    budget.observe([1.0, 0.0, 0.0])          # 4 units of translation
    budget.observe([1.0, 0.0, math.pi / 3])  # 2 units of rotation
    summary = budget.summary()
    assert summary["translation_units"] == pytest.approx(4.0)
    assert summary["rotation_units"] == pytest.approx(2.0)
    assert summary["rotation_fraction"] == pytest.approx(2.0 / 6.0)
    assert summary["units"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# The rule, and its mandatory backstop
# ---------------------------------------------------------------------------


def test_a_geometric_budget_cannot_be_built_without_a_time_backstop():
    """The invariant this whole design rests on.

    A frozen agent accrues no budget and an agent pushing into a wall achieves
    no displacement, so a geometric cap alone can never end either episode.
    Every way of expressing "no backstop" has to fail.
    """
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="backstop"):
            BudgetRule(budget=GeometricBudget(), time_limit_s=bad)


def test_the_rule_says_which_cap_fired():
    rule = BudgetRule(budget=GeometricBudget(budget=10.0), time_limit_s=100.0)
    rule.budget.reset([0.0, 0.0, 0.0])
    assert rule.exceeded(sim_time=1.0) is None
    assert rule.exceeded(sim_time=100.0) == TERMINATED_TIME
    rule.budget.observe([3.0, 0.0, 0.0])  # 12 units > 10
    # The budget is checked first: it is the rule, the clock is the backstop.
    assert rule.exceeded(sim_time=1.0) == TERMINATED_BUDGET


def test_the_backstop_implies_a_rate_floor():
    rule = BudgetRule(budget=GeometricBudget(budget=500.0), time_limit_s=300.0)
    assert rule.units_per_second_floor == pytest.approx(500.0 / 300.0)


def test_terminated_codes_cover_the_old_strings_too():
    assert terminated_code("max steps") == TERMINATED_STEPS
    assert terminated_code(None) == TERMINATED_STEPS
    assert terminated_code("policy called stop") == TERMINATED_POLICY
    assert terminated_code("auto-stop on arrival") == TERMINATED_ORACLE
    assert terminated_code("oracle stop +30 steps") == TERMINATED_ORACLE
    assert terminated_code("budget exhausted") == TERMINATED_BUDGET
    assert terminated_code("sim time limit") == TERMINATED_TIME


# ---------------------------------------------------------------------------
# Configuration: default off, and it must survive the shard round trip
# ---------------------------------------------------------------------------


def test_the_step_cap_is_still_the_default():
    """Today's behaviour is what a run gets when it asks for nothing."""
    from objectnav_eval.config import EvalConfig

    config = EvalConfig()
    assert config.episode_budget == "steps"
    assert config.budget_rule_active is False
    assert config.build_budget_rule() is None
    assert config.step_limit == config.max_steps


def test_budget_choice_survives_the_shard_round_trip():
    """A worker that silently dropped this would run a different episode length."""
    from objectnav_eval.config import EvalConfig

    original = EvalConfig(
        episode_budget="geometric", budget_units=350.0, budget_time_limit_s=240.0
    )
    assert EvalConfig.from_dict(original.to_dict()) == original


def test_the_budget_flags_reach_the_config():
    import argparse

    from objectnav_eval.config import add_arguments, config_from_args

    parser = add_arguments(argparse.ArgumentParser())
    config = config_from_args(parser.parse_args([
        "--episodes", "x", "--episode-budget", "both",
        "--budget-units", "400", "--budget-time-limit", "120",
    ]))
    assert config.episode_budget == "both"
    assert config.budget_units == 400.0
    assert config.budget_time_limit_s == 120.0
    assert config.budget_rule_active
    # 'both' keeps the step cap as well, so the loop bound is still --max-steps.
    assert config.step_limit == config.max_steps


def test_a_geometric_config_without_a_backstop_is_rejected():
    from objectnav_eval.config import EvalConfig

    with pytest.raises(ValueError, match="backstop"):
        EvalConfig(episode_budget="geometric", budget_time_limit_s=0.0)
    with pytest.raises(ValueError, match="backstop"):
        EvalConfig(episode_budget="both", budget_time_limit_s=float("inf"))
    # ...and it is not silently accepted under the step rule either: there the
    # field is simply unused, so it may hold anything.
    EvalConfig(episode_budget="steps", budget_time_limit_s=0.0)


def test_an_unknown_budget_mode_is_a_usage_error():
    from objectnav_eval.config import EvalConfig

    with pytest.raises(ValueError, match="episode-budget"):
        EvalConfig(episode_budget="geometrical")


def test_the_geometric_loop_bound_comes_from_the_backstop_not_max_steps():
    """--max-steps must not silently bind under a geometric budget.

    Left in place, a default of 200 policy steps would end the episode long
    before the budget did, and the run would report a geometric cap it never
    applied.
    """
    from objectnav_eval.config import EvalConfig

    config = EvalConfig(
        episode_budget="geometric", max_steps=200, gap=10, dt=0.04,
        budget_time_limit_s=300.0,
    )
    # 300 s / (10 * 0.04 s) = 750 policy steps, plus one so the tick-level
    # backstop always fires before the loop bound.
    assert config.step_limit == 751
    assert config.step_limit > config.max_steps


# ---------------------------------------------------------------------------
# Offline: re-scoring a run already on disk
# ---------------------------------------------------------------------------


def test_budget_trace_matches_the_live_accumulation():
    poses = _drive(2.0, steps=8)
    trace = budget_trace(poses, sim_time=[0.04 * i for i in range(len(poses))])
    live = GeometricBudget()
    live.reset(poses[0])
    for pose in poses[1:]:
        live.observe(pose)
    assert trace.total == pytest.approx(live.spent)
    assert trace.total == pytest.approx(8.0)
    assert trace.cumulative[0] == 0.0
    # 8 units over 8 ticks of 0.04 s.
    assert trace.summary(500.0)["units_per_sim_second"] == pytest.approx(8.0 / 0.32)


def test_truncation_tick_is_the_first_tick_at_or_past_the_cap():
    poses = _drive(1.0, steps=4)  # 1 unit per tick
    trace = budget_trace(poses)
    assert trace.truncation_tick(2.0) == 2
    assert trace.truncation_tick(2.5) == 3
    assert trace.truncation_tick(99.0) is None
    assert trace.summary(2.0)["truncated"] is True
    assert trace.summary(99.0)["truncated"] is False


def _record(poses, dt=0.05, **extra):
    return {
        "uid": "s:1", "scene": "s", "category": "chair", "steps": 2,
        "trajectory": {
            "enabled": True, "ticks": len(poses),
            "tick_pose": [[round(dt * (i + 1), 3), *pose] for i, pose in enumerate(poses)],
        },
        **extra,
    }


def test_a_record_without_a_trajectory_log_is_none_not_zero():
    assert trace_from_record({"uid": "s:1"}) is None
    assert trace_from_record({"uid": "s:1", "trajectory": {"enabled": False, "ticks": 0}}) is None
    scored = rescore_results([{"uid": "s:1"}], budget_units=500.0)
    assert scored["aggregate"]["episodes"] == 0
    assert scored["aggregate"]["episodes_without_trajectory"] == 1


def test_rescoring_reports_where_a_cap_would_have_cut():
    poses = _drive(2.5, steps=10)  # 1 unit per tick, 10 units total
    row = rescore_record(_record(poses, first_reach_step=1, oracle_success=1), 4.0, gap=5)
    assert row["units"] == pytest.approx(10.0)
    assert row["truncated"] is True
    assert row["truncated_at_tick"] == 4
    assert row["truncated_at_step"] == 0
    assert row["path_at_truncation_m"] == pytest.approx(1.0)
    # The agent first reached the goal in chunk 1, i.e. ticks 5..9 -- entirely
    # after the cut, so a capped run could not have latched oracle_success.
    assert row["oracle_success_under_cap"] is False
    assert row["oracle_ambiguous"] is False


def test_rescoring_flags_the_one_chunk_ambiguity_rather_than_guessing():
    poses = _drive(2.5, steps=10)
    row = rescore_record(_record(poses, first_reach_step=1, oracle_success=1), 7.0, gap=5)
    # The cut lands inside the arrival chunk: the log records the chunk, not the
    # tick, so whether the agent had arrived by then is genuinely unknown.
    assert row["truncated_at_tick"] == 7
    assert row["oracle_success_under_cap"] is True
    assert row["oracle_success_under_cap_conservative"] is False
    assert row["oracle_ambiguous"] is True


def test_an_uncut_episode_keeps_its_oracle_success():
    poses = _drive(2.5, steps=10)
    row = rescore_record(_record(poses, first_reach_step=1, oracle_success=1), 500.0, gap=5)
    assert row["truncated"] is False
    assert row["oracle_success_under_cap"] == 1
    assert row["oracle_ambiguous"] is False


def test_rescore_results_aggregates():
    short = _record(_drive(0.5, steps=4), first_reach_step=0, oracle_success=1)
    long = _record(_drive(5.0, steps=20), first_reach_step=None, oracle_success=0)
    scored = rescore_results([short, long], budget_units=10.0, gap=5)
    aggregate = scored["aggregate"]
    assert aggregate["episodes"] == 2
    assert aggregate["truncated"] == 1  # only the 20-unit one
    assert aggregate["truncated_fraction"] == pytest.approx(0.5)
    assert aggregate["units_mean"] == pytest.approx((2.0 + 20.0) / 2)


# ---------------------------------------------------------------------------
# The runner: the rule terminates AT THE TICK, not at the chunk boundary
# ---------------------------------------------------------------------------


class _FakeTask:
    """Just enough of the task for the runner: metrics that never change."""

    def __init__(self):
        self.path_tracker = type("T", (), {"length": 0.0})()
        self.stopped = False
        self.poses = []

    def reset(self, episode, snap_to_navmesh=False):
        return {"distance_to_goal": 5.0}

    def observe_pose(self, pose):
        self.poses.append(np.asarray(pose, dtype=float))

    def evaluate(self):
        return {"distance_to_goal": 5.0, "success": 0.0, "path_length": 0.0}

    def call_stop(self):
        self.stopped = True


class _FakeSim:
    def __init__(self):
        self.pose = np.zeros(3)

    def get_2d_pose(self):
        return self.pose.copy()

    def get_obs(self):
        return {"color_sensor": np.zeros((4, 4, 3), dtype=np.uint8)}

    def is_tipped_over(self):
        return False


class _FakeExecutor:
    """Advances the robot 0.25 m (= exactly one budget unit) per tick."""

    def __init__(self, sim, gap=5, dt=0.05, step_m=0.25):
        self.sim = sim
        self.gap = gap
        self.dt = dt
        self.step_m = step_m
        self.tick = 0
        self.sim_time = 0.0

    def reset(self):
        self.tick = 0
        self.sim_time = 0.0
        self.sim.pose = np.zeros(3)

    def execute(self, chunk, chunk_index, on_tick=None, stop_when=None):
        from objectnav_eval.executor import ChunkExecution, TickRecord

        ticks = []
        for j in range(self.gap):
            previous = self.sim.pose.copy()
            self.sim.pose = previous + np.array([self.step_m, 0.0, 0.0])
            self.tick += 1
            self.sim_time += self.dt
            record = TickRecord(
                tick=self.tick, chunk_index=chunk_index, tick_in_chunk=j,
                setpoint=self.sim.pose.copy(), previous_setpoint=previous,
                pose=self.sim.pose.copy(), previous_pose=previous,
                sim_time=self.sim_time, height=0.0,
            )
            ticks.append(record)
            if on_tick is not None:
                on_tick(record)
            if stop_when is not None and stop_when(record):
                return ChunkExecution(
                    chunk_index=chunk_index, anchor=previous,
                    setpoints=np.zeros((len(ticks), 3)), poses=np.zeros((len(ticks), 3)),
                    ticks=ticks, stopped_early=True, chunk=chunk,
                )
        return ChunkExecution(
            chunk_index=chunk_index, anchor=np.zeros(3),
            setpoints=np.zeros((len(ticks), 3)), poses=np.zeros((len(ticks), 3)),
            ticks=ticks, chunk=chunk,
        )


class _FakePolicy:
    name = "fake"
    uses_observation = False

    def reset(self, goal):
        pass

    def act(self, observation):
        from objectnav_eval.policies import ActionChunk

        return ActionChunk(poses=np.zeros((10, 3)))


def _run(config):
    from pathlib import Path

    from objectnav_eval.episodes import EvalEpisode
    from objectnav_eval.runner import EpisodeRunner, RunnerContext
    from objectnav_eval.screening import ScreenResult

    sim = _FakeSim()
    episode = EvalEpisode(
        uid="s:1", episode_id="1", scene_id="s.glb", scene_name="s",
        object_category="chair", start_position=np.zeros(3),
        start_rotation=np.zeros(4), view_points=np.zeros((0, 3)),
    )
    runner = EpisodeRunner(RunnerContext(
        robot_sim=sim, task=_FakeTask(),
        executor=_FakeExecutor(sim, gap=config.gap, dt=config.dt),
        policy=_FakePolicy(), config=config, output_dir=Path("/tmp"), probes=(),
    ))
    return runner.run(episode, ScreenResult(ok=True))


def test_the_budget_ends_the_episode_at_the_tick_not_the_chunk_boundary():
    """A cap that only fired at chunk boundaries would hand out free motion."""
    from objectnav_eval.config import EvalConfig

    # 0.25 m per tick = 1 unit per tick, gap 5, so a 12-unit budget runs out on
    # tick 12: mid-way through chunk index 2, with 3 ticks of chunk 2 unspent.
    record = _run(EvalConfig(
        episodes="x", episode_budget="geometric", budget_units=12.0,
        budget_time_limit_s=300.0, gap=5, dt=0.05, max_steps=50,
    ))
    assert record["ticks"] == 12
    assert record["terminated"] == "budget exhausted"
    assert record["terminated_by"] == "budget_exhausted"
    assert record["budget"]["units"] == pytest.approx(12.0)
    assert record["budget"]["translation_units"] == pytest.approx(12.0)
    assert record["budget"]["rotation_units"] == 0.0


def test_the_sim_time_backstop_ends_a_cheap_episode():
    """The mode a geometric budget is blind to: motion too slow to spend it."""
    from objectnav_eval.config import EvalConfig

    config = EvalConfig(
        episodes="x", episode_budget="geometric", budget_units=1e6,
        budget_time_limit_s=0.37, gap=5, dt=0.05, max_steps=50,
    )
    record = _run(config)
    assert record["terminated_by"] == "time_limit"
    # Fires on the first tick at or past 0.37 s, i.e. tick 8 of 0.05 s each.
    assert record["ticks"] == 8
    # Still reported, even though it was not the cap that fired.
    assert record["budget"]["units"] > 0


def test_the_budget_is_recorded_even_when_the_step_cap_is_the_rule():
    """Point 4: it costs nothing and it is what makes a run re-scorable."""
    from objectnav_eval.config import EvalConfig

    record = _run(EvalConfig(episodes="x", gap=5, dt=0.05, max_steps=3))
    assert record["terminated"] == "max steps"
    assert record["terminated_by"] == "max_steps"
    assert record["steps"] == 3
    # 15 ticks x 0.25 m.
    assert record["budget"]["units"] == pytest.approx(15.0)
    assert record["budget"]["rule"] == "steps"
    assert record["budget"]["budget_units"] == DEFAULT_BUDGET_UNITS
    assert record["budget"]["units_per_sim_second"] == pytest.approx(15.0 / 0.75)


def test_the_step_cap_still_applies_under_both():
    from objectnav_eval.config import EvalConfig

    record = _run(EvalConfig(
        episodes="x", episode_budget="both", budget_units=1e6,
        budget_time_limit_s=DEFAULT_TIME_BACKSTOP_S, gap=5, dt=0.05, max_steps=4,
    ))
    assert record["terminated_by"] == "max_steps"
    assert record["steps"] == 4


def test_summarize_reports_the_budget_across_a_run():
    from objectnav_eval.reporting import summarize_budget

    rows = [
        {"budget": {"units": 100.0, "translation_units": 60.0, "rotation_units": 40.0,
                    "rotation_fraction": 0.4, "samples": 10, "exhausted": False,
                    "budget_units": 500.0, "units_per_sim_second": 4.0, "rule": "steps"}},
        {"budget": {"units": 600.0, "translation_units": 300.0, "rotation_units": 300.0,
                    "rotation_fraction": 0.5, "samples": 10, "exhausted": True,
                    "budget_units": 500.0, "units_per_sim_second": 3.0, "rule": "steps"}},
    ]
    summary = summarize_budget(rows)
    assert summary["episodes"] == 2
    assert summary["units_mean"] == pytest.approx(350.0)
    assert summary["episodes_over_budget"] == 1
    assert summary["rotation_fraction_mean"] == pytest.approx(0.45)
    assert summarize_budget([{"budget": {"samples": 0}}]) == {}
