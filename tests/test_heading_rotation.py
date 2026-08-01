"""Behavioural tests for the in-place rotation controller and the phase-transition
rules that hand control between it and the path controller.

``HeadingRotationController`` only ever touches ``robot_sim`` through
``get_2d_pose()`` and ``get_joint_state("joint_th")``, so it is exercised here
against a tiny fake base instead of a live simulator. The fake integrates the
commanded yaw rate exactly, which is what the real base does: a probe of the
Bullet yaw joint motor (``dump/data_diagnostics/rotation_tuning/probe_dynamics.py``)
found it reaches a commanded velocity within one substep and stops within one
tick of the command being cut -- no lag, no coast, for every command up to
4 rad/s. That is the property the no-overshoot clamp relies on.

``controllers.py`` imports magnum at module scope, so these skip cleanly where
magnum is unavailable, matching ``test_frame_conversions.py``.
"""

import math

import numpy as np
import pytest

pytest.importorskip("magnum", reason="controllers.py imports magnum at module scope")

from continuous_demos.controllers import (  # noqa: E402
    HeadingRotationConfig,
    HeadingRotationController,
)
from continuous_demos.replay import _is_stalled  # noqa: E402

DT = 1.0 / 25.0   # the on-grid recording tick (see replay.RECORD_FPS)
# The SETTLE/TURN boundary from the croppability analysis, in rad/s so that it
# keeps its meaning independent of the recording rate (1.15 deg/tick at 30 Hz).
SETTLE_W = 0.602


class FakeBase:
    """Yaw-only stand-in for HabitatRobotSim: integrates the commanded rate exactly."""

    def __init__(self, heading=0.0):
        self.heading = float(heading)

    def get_2d_pose(self):
        return np.array([0.0, 0.0, self.heading])

    def get_joint_state(self, name, is_vel=False):
        assert name == "joint_th"
        return self.heading

    def apply(self, step_args, dt=DT):
        self.heading += float(step_args["chassis_twist"][2]) * dt


def run(controller, base, dt=DT, max_ticks=2000):
    """Drive the controller to completion; return the per-tick (w_cmd, heading) log."""
    log = []
    for _ in range(max_ticks):
        if controller.is_trajectory_complete():
            break
        args = controller.step(dt=dt)
        w = float(args["chassis_twist"][2])
        base.apply(args, dt)
        log.append((w, base.heading))
        if w == 0.0 and controller.is_trajectory_complete():
            break
    return log


def make(headings, base=None, dt=DT, handoff=False, **cfg):
    base = base if base is not None else FakeBase(0.0)
    controller = HeadingRotationController(base, HeadingRotationConfig(**cfg))
    controller.set_heading_trajectory(headings, handoff_to_path=handoff)
    return controller, base


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class TestNoOvershoot:
    @pytest.mark.parametrize("target_deg", [5, 12, 30, 45, 90, 120, 175, -30, -90, -175])
    @pytest.mark.parametrize(
        "cfg",
        [
            dict(),                                  # shipped default (gain 30, clamp on)
            dict(gain=10.0),
            dict(gain=100.0),                        # far past deadbeat; clamp must save it
            dict(gain=3.0, w_min=1.2),               # trapezoidal: floor + clamp
            dict(gain=30.0, w_max=4.0),
        ],
    )
    def test_never_crosses_the_target(self, target_deg, cfg):
        target = math.radians(target_deg)
        controller, base = make([target], **cfg)
        log = run(controller, base)
        # the sign of the remaining error must never flip: one tick can never
        # carry the base past the heading it is chasing.
        signs = {np.sign(round(wrap(target - h), 12)) for _, h in log}
        assert signs <= {np.sign(target), 0.0}, f"overshot: {sorted(signs)}"

    def test_high_gain_without_the_clamp_does_overshoot(self):
        """Negative control: the clamp is what makes an aggressive gain safe."""
        target = math.radians(90)
        controller, base = make([target], gain=100.0, no_overshoot_clamp=False,
                                heading_tolerance=0.01)
        log = run(controller, base)
        signs = {np.sign(round(wrap(target - h), 12)) for _, h in log}
        assert -np.sign(target) in signs

    def test_velocity_floor_without_the_clamp_limit_cycles(self):
        """Negative control: a floor alone chatters around the deadband forever.

        A floor of w means every tick moves the base by w*dt; once that step is
        larger than the deadband the base hops back and forth across the target
        and never lands. The clamp is what turns a floor into a usable
        trapezoidal profile.
        """
        stuck = []
        for deg in range(5, 180, 7):
            controller, base = make([math.radians(deg)], gain=3.0, w_min=2.0,
                                    w_max=2.0, heading_tolerance=0.01,
                                    no_overshoot_clamp=False)
            if len(run(controller, base, max_ticks=400)) == 400:
                stuck.append(deg)
        assert len(stuck) > 10, "expected a limit cycle for most targets"
        # ...and the same floor WITH the clamp always converges.
        for deg in range(5, 180, 7):
            controller, base = make([math.radians(deg)], gain=3.0, w_min=2.0,
                                    w_max=2.0, heading_tolerance=0.01)
            run(controller, base, max_ticks=400)
            assert controller.is_trajectory_complete()


class TestCapAndDeadband:
    @pytest.mark.parametrize("w_max", [1.0, 2.0, 3.0])
    def test_commanded_rate_never_exceeds_the_cap(self, w_max):
        controller, base = make([math.radians(170)], w_max=w_max)
        log = run(controller, base)
        assert max(abs(w) for w, _ in log) <= w_max + 1e-12

    def test_residual_inside_the_deadband_is_snapped_not_chased(self):
        tol = 0.05
        # start 0.9*tol away: already acceptable, so not a single tick is spent.
        base = FakeBase(0.0)
        controller, _ = make([0.9 * tol], base=base, heading_tolerance=tol)
        log = run(controller, base)
        assert log == [] or all(w == 0.0 for w, _ in log)
        assert controller.is_trajectory_complete()

    def test_lands_within_the_deadband(self):
        tol = 0.05
        controller, base = make([math.radians(90)], heading_tolerance=tol)
        run(controller, base)
        assert abs(wrap(math.radians(90) - base.heading)) <= tol

    def test_saturates_then_lands_in_the_analytic_minimum(self):
        """Deadbeat + cap == the fastest a w_max-capped turn can possibly be."""
        w_max, tol = 2.0, 0.05
        target = math.radians(90)
        controller, base = make([target], w_max=w_max, heading_tolerance=tol)
        log = run(controller, base)
        floor_ticks = math.floor((abs(target) - tol) / (w_max * DT))
        assert floor_ticks <= len(log) <= floor_ticks + 2
        # all but the last tick or two (the landing) are at the cap
        at_cap = sum(1 for w, _ in log if abs(w) == pytest.approx(w_max))
        assert at_cap >= len(log) - 2

    # The discrete demo turn primitives cluster at these sizes.
    @pytest.mark.parametrize("deg,min_speedup", [(15, 4.0), (30, 3.0), (60, 2.1),
                                                 (90, 1.7), (180, 1.4)])
    def test_default_is_much_faster_than_the_old_proportional_law(self, deg, min_speedup):
        """Regression guard on the settle tail this controller was retuned to remove.

        The old law was gain=3, w_max=2, tol=0.03, no clamp: it saturated only
        while |e| > w_max/gain = 0.67 rad and then decayed exponentially into a
        1.7-degree deadband. The speedup is largest for small turns, which is
        exactly where the demos spend most of their turning.
        """
        target = math.radians(deg)
        new, base_new = make([target])
        old, base_old = make([target], gain=3.0, heading_tolerance=0.03,
                             no_overshoot_clamp=False)
        n_new, n_old = len(run(new, base_new)), len(run(old, base_old))
        assert n_old >= min_speedup * n_new, f"{deg}deg: new={n_new} old={n_old}"

    @pytest.mark.parametrize("deg", [15, 30, 60, 90, 180])
    def test_the_old_laws_tail_is_a_fixed_cost_the_new_law_does_not_pay(self, deg):
        """The tail is a FIXED ~0.63 s of sub-SETTLE-threshold crawl per heading
        target, independent of how far the robot actually had to turn -- which is
        why it dominated short turns and accounted for a third of all recorded
        ticks. 0.63 s is ln(0.2/0.03)/3, the time the old gain-3 exponential took
        to decay from the threshold into the deadband.

        The threshold is held in rad/s, not deg/tick, so this does not silently
        change meaning with the recording rate.
        """
        old, base = make([math.radians(deg)], gain=3.0, heading_tolerance=0.03,
                         no_overshoot_clamp=False)
        old_slow = sum(1 for w, _ in run(old, base) if abs(w) < SETTLE_W)
        assert 0.55 <= old_slow * DT <= 0.75, f"{old_slow} ticks = {old_slow * DT:.2f}s"

        new, base2 = make([math.radians(deg)])
        new_slow = sum(1 for w, _ in run(new, base2) if abs(w) < SETTLE_W)
        assert new_slow <= 1


class TestTrapezoidalFloor:
    def test_floor_holds_the_rate_up_until_the_landing_tick(self):
        controller, base = make([math.radians(90)], gain=3.0, w_min=0.8, w_max=2.0)
        log = run(controller, base)
        rates = [abs(w) for w, _ in log[:-1]]
        assert min(rates) >= 0.8 - 1e-12
        assert max(rates) == pytest.approx(2.0)
        # the profile is monotonically non-increasing: ramp at the cap, then coast
        assert all(a >= b - 1e-9 for a, b in zip(rates, rates[1:]))


class TestHandoffTolerance:
    def test_looser_tolerance_applies_only_to_the_final_heading(self):
        cfg = dict(heading_tolerance=0.03, handoff_tolerance=0.20)
        headings = [math.radians(40), math.radians(80)]
        controller, _ = make(headings, handoff=True, **cfg)
        assert controller.tolerance_for_target(0) == 0.03
        assert controller.tolerance_for_target(1) == 0.20

    def test_no_looser_tolerance_when_the_phase_does_not_hand_over(self):
        """A look phase that ends the episode must still land on its heading."""
        controller, _ = make([math.radians(80)], handoff=False,
                             heading_tolerance=0.03, handoff_tolerance=0.20)
        assert controller.tolerance_for_target(0) == 0.03

    def test_handoff_accepts_early_and_costs_ticks_it_would_otherwise_spend(self):
        target = math.radians(90)
        tight, b1 = make([target], handoff=True, heading_tolerance=0.03,
                         handoff_tolerance=0.03)
        loose, b2 = make([target], handoff=True, heading_tolerance=0.03,
                         handoff_tolerance=0.20)
        assert len(run(loose, b2)) < len(run(tight, b1))
        assert abs(wrap(target - b2.heading)) <= 0.20

    def test_handoff_residual_is_bounded_by_the_tolerance(self):
        for deg in (30, 90, 150):
            controller, base = make([math.radians(deg)], handoff=True,
                                    handoff_tolerance=0.17)
            run(controller, base)
            assert abs(wrap(math.radians(deg) - base.heading)) <= 0.17


class TestMultiWaypointScan:
    def test_a_reversing_scan_reaches_every_waypoint_in_order(self):
        headings = [math.radians(d) for d in (60, 120, 60, 0, -60)]
        controller, base = make(headings, heading_tolerance=0.05)
        seen = []
        for _ in range(2000):
            if controller.is_trajectory_complete():
                break
            idx = controller.target_idx
            args = controller.step(dt=DT)
            base.apply(args)
            seen.append((idx, base.heading))
        assert controller.is_trajectory_complete()
        for k, h in enumerate(headings):
            closest = min(abs(wrap(h - hh)) for _, hh in seen)
            assert closest <= 0.05 + 1e-9, f"waypoint {k} never reached"

    def test_target_index_only_ever_advances(self):
        controller, base = make([math.radians(d) for d in (60, 120, 60, 0)])
        idxs = []
        for _ in range(2000):
            if controller.is_trajectory_complete():
                break
            idxs.append(controller.target_idx)
            base.apply(controller.step(dt=DT))
        assert idxs == sorted(idxs)


class TestBackCompat:
    def test_constructor_overrides_still_work(self):
        controller = HeadingRotationController(FakeBase(), w_max=1.25,
                                               heading_tolerance=0.4)
        assert controller.w_max == 1.25 and controller.heading_tolerance == 0.4

    def test_step_gain_argument_overrides_the_config(self):
        controller, base = make([math.radians(90)], gain=30.0,
                                no_overshoot_clamp=False, w_max=99.0)
        args = controller.step(dt=DT, heading_alignment_gain=1.0)
        assert float(args["chassis_twist"][2]) == pytest.approx(math.radians(90))


TICKS, EPS, GATE = 45, 0.05, 0.02


def ramp(speed, n=200):
    """(path_progress, speeds) for a base advancing steadily at ``speed`` m/s."""
    return [i * speed * DT for i in range(n)], [speed] * n


class TestStallDetection:
    def test_disabled_when_stall_ticks_is_zero(self):
        assert not _is_stalled([0.0] * 100, [0.0] * 100, 0, EPS, GATE)

    def test_not_stalled_before_the_window_is_full(self):
        assert not _is_stalled([0.0] * TICKS, [0.0] * TICKS, TICKS, EPS, GATE)

    def test_a_dead_phase_is_stalled(self):
        """The DWPP short-path failure: v == 0 for the entire phase."""
        assert _is_stalled([0.0] * (TICKS + 1), [0.0] * (TICKS + 1), TICKS, EPS, GATE)

    def test_steady_progress_is_not_stalled(self):
        p, s = ramp(2.0)
        assert not _is_stalled(p, s, TICKS, EPS, GATE)

    def test_slow_but_real_progress_is_not_stalled(self):
        """0.1 m/s -- the path controller's min approach speed -- must survive."""
        p, s = ramp(0.1)
        assert not _is_stalled(p, s, TICKS, EPS, GATE)

    def test_the_speed_gate_protects_a_crawl_that_is_still_moving(self):
        """A base creeping forward is making real progress even when the
        along-path index barely advances; only the speed gate can tell it apart
        from a dead phase, and cutting it is what broke fidelity in the sweep.
        """
        # advancing along the path too slowly to clear EPS over the window
        # (0.03 m < 0.05 m) while genuinely driving at 0.1 m/s -- e.g. sliding
        # along geometry, or the projected index lagging round a tight corner.
        p = [i * 0.001 for i in range(200)]
        s = [0.1] * 200
        assert not _is_stalled(p, s, TICKS, EPS, GATE)
        # ...and without the gate the same history is (wrongly) called a stall
        assert _is_stalled(p, s, TICKS, EPS, None)

    def test_a_single_moving_tick_in_the_window_defeats_the_gate(self):
        p, s = ramp(0.0)
        s = list(s)
        s[-5] = 0.5                # one tick of real motion inside the window
        assert not _is_stalled(p, s, TICKS, EPS, GATE)

    def test_only_the_window_is_considered_not_the_whole_phase(self):
        # moved early, then died: must still be caught
        p = [i * 2.0 * DT for i in range(100)] + [100 * 2.0 * DT] * (TICKS + 5)
        s = [2.0] * 100 + [0.0] * (TICKS + 5)
        assert _is_stalled(p, s, TICKS, EPS, GATE)


class TestPhaseHandoffWiring:
    def test_set_heading_trajectory_records_the_handoff_flag(self):
        controller = HeadingRotationController(FakeBase())
        controller.set_heading_trajectory([0.0, 1.0], handoff_to_path=True)
        assert controller.handoff_to_path is True
        controller.set_heading_trajectory([0.0, 1.0], handoff_to_path=False)
        assert controller.handoff_to_path is False

    def test_redundant_heading_removal_still_applies(self):
        controller = HeadingRotationController(FakeBase())
        # a monotone sweep collapses to its endpoints
        headings = [0.0, 0.3, 0.6, 0.9]
        controller.set_heading_trajectory(headings, remove_redundant=True)
        assert len(controller.headings) < len(headings)
        assert controller.headings[-1] == pytest.approx(0.9)
