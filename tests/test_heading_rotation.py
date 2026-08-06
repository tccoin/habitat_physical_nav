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


# The shipped default bounds angular acceleration (HeadingRotationConfig.alpha_max),
# which changes the shape of the turn. Tests that are specifically about the
# UNBOUNDED law -- the bang-bang/proportional family, its optimality, and the
# knobs (gain, w_min) that only that branch reads -- pin it off with this.
UNBOUNDED = dict(alpha_max=None)


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
        """Negative control: the clamp is what makes an aggressive gain safe.

        Pinned to the unbounded law, because the acceleration bound independently
        prevents this: its arrival rate is derived from the remaining angle, so it
        cannot be told to overshoot by raising a gain it does not read.
        """
        target = math.radians(90)
        controller, base = make([target], gain=100.0, no_overshoot_clamp=False,
                                heading_tolerance=0.01, **UNBOUNDED)
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
                                    no_overshoot_clamp=False, **UNBOUNDED)
            if len(run(controller, base, max_ticks=400)) == 400:
                stuck.append(deg)
        assert len(stuck) > 10, "expected a limit cycle for most targets"
        # ...and the same floor WITH the clamp always converges.
        for deg in range(5, 180, 7):
            controller, base = make([math.radians(deg)], gain=3.0, w_min=2.0,
                                    w_max=2.0, heading_tolerance=0.01, **UNBOUNDED)
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
        """Deadbeat + cap == the fastest a w_max-capped turn can possibly be.

        This is the UNBOUNDED law, which is why it is not the shipped default:
        the only way to be this fast is to step to the cap and back to zero in
        one tick each, which no real base can do.
        """
        w_max, tol = 2.0, 0.05
        target = math.radians(90)
        controller, base = make([target], w_max=w_max, heading_tolerance=tol,
                                **UNBOUNDED)
        log = run(controller, base)
        floor_ticks = math.floor((abs(target) - tol) / (w_max * DT))
        assert floor_ticks <= len(log) <= floor_ticks + 2
        # all but the last tick or two (the landing) are at the cap
        at_cap = sum(1 for w, _ in log if abs(w) == pytest.approx(w_max))
        assert at_cap >= len(log) - 2

    # The discrete demo turn primitives cluster at these sizes.
    @pytest.mark.parametrize("deg,min_speedup", [(15, 1.4), (30, 1.35), (60, 1.25),
                                                 (90, 1.15), (180, 1.1)])
    def test_default_is_much_faster_than_the_old_proportional_law(self, deg, min_speedup):
        """Regression guard on the settle tail this controller was retuned to remove.

        The old law was gain=3, w_max=2, tol=0.03, no clamp: it saturated only
        while |e| > w_max/gain = 0.67 rad and then decayed exponentially into a
        1.7-degree deadband. The speedup is largest for small turns, which is
        exactly where the demos spend most of their turning.

        Measured against the SHIPPED default, which bounds angular acceleration
        at 5 rad/s^2 and so gives about half the raw speed back: 1.50x / 1.41x /
        1.29x / 1.19x / 1.14x at 15 / 30 / 60 / 90 / 180 degrees. The thresholds
        below sit just under those. The margin is deliberately thin -- at the
        shipped alpha_max a single turn is only modestly faster than the old law,
        and the win is at the episode level (0.825 of baseline duration), where
        removing the settle tail compounds across every waypoint of every phase.
        """
        target = math.radians(deg)
        new, base_new = make([target])
        old, base_old = make([target], gain=3.0, heading_tolerance=0.03,
                             no_overshoot_clamp=False, **UNBOUNDED)
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
                         no_overshoot_clamp=False, **UNBOUNDED)
        old_slow = sum(1 for w, _ in run(old, base) if abs(w) < SETTLE_W)
        assert 0.55 <= old_slow * DT <= 0.75, f"{old_slow} ticks = {old_slow * DT:.2f}s"

        # The shipped law also spends time below the SETTLE threshold -- it has
        # to, it ramps -- but that time is the two ramps, bounded by
        # 2 * SETTLE_W / alpha_max, and unlike the old law's exponential it does
        # not grow as the deadband shrinks. At the shipped alpha_max = 5 that is
        # 8 ticks, 0.32 s, measured FLAT across every turn size. Flatness is the
        # property that matters: the old law's tail is a fixed 0.63 s that buys
        # nothing, this one is the ramp that the softening is deliberately paying
        # for, and neither grows with the turn.
        new, base2 = make([math.radians(deg)])
        new_slow = sum(1 for w, _ in run(new, base2) if abs(w) < SETTLE_W)
        assert new_slow * DT <= 0.36, f"{new_slow} ticks = {new_slow * DT:.2f}s"

        # ...and with acceleration unbounded it is a single landing tick.
        bb, base3 = make([math.radians(deg)], **UNBOUNDED)
        assert sum(1 for w, _ in run(bb, base3) if abs(w) < SETTLE_W) <= 1


class TestTrapezoidalFloor:
    def test_floor_holds_the_rate_up_until_the_landing_tick(self):
        controller, base = make([math.radians(90)], gain=3.0, w_min=0.8, w_max=2.0,
                                **UNBOUNDED)
        log = run(controller, base)
        rates = [abs(w) for w, _ in log[:-1]]
        assert min(rates) >= 0.8 - 1e-12
        assert max(rates) == pytest.approx(2.0)
        # the profile is monotonically non-increasing: ramp at the cap, then coast
        assert all(a >= b - 1e-9 for a, b in zip(rates, rates[1:]))


class TestBoundedAcceleration:
    """The shipped profile: |dw/dt| <= alpha_max, both edges.

    The motivation is distributional, not aesthetic. The unbounded law's velocity
    trace is a rectangle -- 97.8% of its turning ticks sit at w_max -- and this
    project's central finding is that a conditional-mean regression head cannot
    represent a bimodal action distribution. Bounding acceleration is what puts
    mass at intermediate turn rates. See dump/rotation_segmentation/FINDINGS.md.
    """

    ALPHAS = [30.0, 20.0, 12.0, 8.0, 4.0]

    @pytest.mark.parametrize("alpha", ALPHAS)
    @pytest.mark.parametrize("target_deg", [5, 15, 30, 60, 90, 120, 175, -30, -90, -175])
    def test_commanded_acceleration_never_exceeds_alpha(self, alpha, target_deg):
        """The bound holds over the WHOLE profile, launch and landing included.

        Both edges are asserted by bracketing the commanded trace with the rest
        state the base is genuinely in before and after a look phase. Getting the
        landing edge inside the bound is what the deadband-targeted, discrete-time
        corrected arrival rate in ``_bounded_accel_command`` is for; the naive
        ``sqrt(2*alpha*|e|)`` fails this test by a factor of two.
        """
        controller, base = make([math.radians(target_deg)], alpha_max=alpha)
        log = [w for w, _ in run(controller, base) if w != 0.0]
        rates = [0.0] + log + [0.0]                  # rest to rest
        worst = max(abs(b - a) / DT for a, b in zip(rates, rates[1:]))
        assert worst <= alpha * (1 + 1e-9), f"|dw/dt| = {worst:.2f} > {alpha}"

    @pytest.mark.parametrize("alpha", ALPHAS)
    @pytest.mark.parametrize("target_deg", [5, 15, 30, 60, 90, 175, -30, -90, -175])
    def test_still_never_overshoots(self, alpha, target_deg):
        target = math.radians(target_deg)
        controller, base = make([target], alpha_max=alpha)
        log = run(controller, base)
        signs = {np.sign(round(wrap(target - h), 12)) for _, h in log}
        assert signs <= {np.sign(target), 0.0}, f"overshot: {sorted(signs)}"
        assert controller.is_trajectory_complete()

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_profile_is_a_trapezoid_not_a_rectangle(self, alpha):
        """Rises, holds, falls -- and actually visits intermediate rates."""
        controller, base = make([math.radians(180)], alpha_max=alpha)
        rates = [abs(w) for w, _ in run(controller, base)]
        peak = rates.index(max(rates))
        assert all(a <= b + 1e-9 for a, b in zip(rates[:peak], rates[1:peak + 1]))
        assert all(a >= b - 1e-9 for a, b in zip(rates[peak:], rates[peak + 1:]))
        mid = [r for r in rates if 0.2 < r < 1.8]
        assert len(mid) >= 2, f"no intermediate rates: {rates}"

    def test_lower_alpha_puts_more_mass_at_intermediate_rates(self):
        """The whole point, stated as a monotone property."""
        fracs = []
        for alpha in (None, 30.0, 20.0, 12.0, 8.0, 5.0):
            rates = []
            for deg in (30, 60, 90, 180):
                controller, base = make([math.radians(deg)], alpha_max=alpha)
                rates += [abs(w) for w, _ in run(controller, base)]
            fracs.append(sum(0.2 < r < 1.8 for r in rates) / len(rates))
        assert all(a < b for a, b in zip(fracs, fracs[1:])), fracs
        assert fracs[0] < 0.1 and fracs[-1] > 0.3

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_peak_rate_still_reaches_the_cap_on_a_long_turn(self, alpha):
        """w_max is NOT lowered: softening the ramp must not slow the cruise."""
        controller, base = make([math.radians(180)], alpha_max=alpha, w_max=2.0)
        assert max(abs(w) for w, _ in run(controller, base)) == pytest.approx(2.0, abs=1e-6)

    def test_a_direction_reversal_ramps_through_zero(self):
        """Signed rate limiting: the command may not jump across zero.

        The unbounded law can go from +w_max to -w_max in a single tick, which is
        2*w_max/dt = 100 rad/s^2 -- the worst acceleration in the whole profile.
        """
        base = FakeBase(0.0)
        controller = HeadingRotationController(base, HeadingRotationConfig(alpha_max=12.0))
        controller.set_heading_trajectory([math.radians(90), math.radians(-90)])
        log = run(controller, base)
        rates = [w for w, _ in log]
        assert max(abs(b - a) / DT for a, b in zip(rates, rates[1:])) <= 12.0 + 1e-9
        assert any(abs(r) < 0.5 for r in rates[1:-1]), "never passed through zero"

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_a_target_at_exactly_180_degrees_still_terminates(self, alpha):
        """The one genuinely ambiguous target, asserted rather than avoided.

        At |e| = pi the shortest-path direction is undefined and one tick of
        motion flips its sign, so "overshoot" is not a meaningful measurement
        there -- a near-180 waypoint is what produced the single 179-degree
        overshoot reading in the held-out sweep, on an episode whose final pose,
        goal check and phase timings were all identical to the other arms. What
        must hold is that the controller still converges.
        """
        controller, base = make([math.pi], alpha_max=alpha)
        run(controller, base)
        assert controller.is_trajectory_complete()
        assert abs(wrap(math.pi - base.heading)) <= 0.03 + 1e-9

    def test_alpha_max_none_restores_the_bang_bang_law(self):
        a, ba = make([math.radians(90)], alpha_max=None)
        b, bb = make([math.radians(90)], alpha_max=1e9)
        assert [w for w, _ in run(a, ba)] == pytest.approx([w for w, _ in run(b, bb)])

    @pytest.mark.parametrize("alpha,ceilings", [
        (12.0, {30: 1.5, 60: 1.35, 90: 1.25, 180: 1.15}),
        (5.0, {30: 2.2, 60: 1.8, 90: 1.55, 180: 1.3}),
    ])
    def test_per_turn_cost_of_the_bound_is_characterised(self, alpha, ceilings):
        """Honest cost guard, pinning what the softening costs per turn.

        The ramp is a fixed overhead, so it is proportionally cheapest on long
        turns: at the shipped alpha_max = 5 a 30-degree turn takes 2.1x the
        bang-bang tick count but a 180-degree turn only 1.25x. Episode duration
        rises less than either (0.672 -> 0.825 of baseline) because turning is a
        minority of ticks and the long turns dominate the turning time.
        """
        for deg, max_ratio in ceilings.items():
            slow, b1 = make([math.radians(deg)], alpha_max=alpha)
            fast, b2 = make([math.radians(deg)], alpha_max=None)
            n_slow, n_fast = len(run(slow, b1)), len(run(fast, b2))
            assert n_slow <= max_ratio * n_fast, f"{deg}deg a={alpha}: {n_slow} vs {n_fast}"

    def test_w_min_is_ignored_when_acceleration_is_bounded(self):
        """A floor and a ramp through that floor are contradictory; documented."""
        controller, base = make([math.radians(90)], alpha_max=12.0, w_min=1.5)
        rates = [abs(w) for w, _ in run(controller, base)]
        assert min(rates) < 1.5


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
        """``heading_alignment_gain`` is an override for the UNBOUNDED law only;
        the bounded profile is defined geometrically and has no gain to override."""
        controller, base = make([math.radians(90)], gain=30.0,
                                no_overshoot_clamp=False, w_max=99.0, **UNBOUNDED)
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
