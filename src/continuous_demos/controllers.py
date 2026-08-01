"""Path-following / heading-tracking control layer for :class:`~continuous_demos.sim.HabitatRobotSim`.

Split out of ``sim.py`` (which owns only the simulator itself): these classes
are the non-simulation control logic that drives the sim -- the pure-pursuit
path follower (wrapping ``pure_pursuit.PurePursuitController``) and the
in-place heading-sequence rotator used for "looking" phases of a replay. See
``sim.py``'s module docstring for the three coordinate frames referenced
throughout (habitat world / robot body / planar control frame).
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import magnum as mn

from continuous_demos.pure_pursuit import PurePursuitController, ControllerConfig
from continuous_demos.sim import HabitatRobotSim

class HolonomicControllerConfig(ControllerConfig):
    pass

class HolonomicRobotController:
    def __init__(self, robot_sim: HabitatRobotSim,pure_pursuit_config: HolonomicControllerConfig):
        self.robot_sim = robot_sim

        if pure_pursuit_config is None:
            pure_pursuit_config = HolonomicControllerConfig(
            min_look_ahead_distance=0.05,
            max_look_ahead_distance=0.4,
            look_ahead_time=0.2,
            v_max=2.0,
            v_min=0.0,
            # NOTE: w_max/w_min = +/-70 are effectively inert and look like a
            # degrees value sitting in a rad/s field (70 rad/s = 4010 deg/s).
            # They bound the dynamic window in
            # pure_pursuit._calc_optimal_velocity_considering_dynamic_window.
            # Measured over 7771 recorded drive ticks the w_ref DWPP actually
            # produces never came within a factor of 6 of 70 rad/s (p99 4.35,
            # max 10.26), so the cap never binds. Deliberately NOT "corrected"
            # to 70 deg/s = 1.222 rad/s: that would not be a cosmetic fix, it
            # would clamp 22% of all driving ticks (probe_wmax.json in
            # dump/data_diagnostics/rotation_tuning/) and is a change to the path
            # controller's behaviour. Left as-is pending a separate decision.
            w_max=70.0,
            w_min=-70.0,
            a_max=3.0,
            aw_max=100.0,
            # Must match the recording tick. Habitat's physics grid is a fixed
            # 0.008 s so the pipeline records at 25 Hz (see sim.py::step and
            # generate_pose2d_dataset.py). This dt feeds pure_pursuit's
            # acceleration clamp (a_max * dt) and its dynamic-window bounds, so a
            # stale 1/30 here silently widened both by 20%.
            dt=1.0/25.0,
            approach_velocity_scaling_dist=0.5,
            min_approach_linear_velocity=0.1,
            goal_tolerance_dist=0.15,  
            regulated_linear_scaling_min_radius=0.01,
            regulated_linear_scaling_min_speed=0.1
            )
            print("Warning: No pure pursuit config provided. Using default values.")
        
        self.pure_pursuit_config = pure_pursuit_config
        self.controller = PurePursuitController(self.pure_pursuit_config, method_name="dwpp")
        self.path_2d = None
        self.goal_pos = None
        self.reset()
    
    # to be called each simulator reset/episode
    def reset(self):
        self.virtual_heading = 0.0
        self.virtual_angular_velocity = 0.0
        self.align_virtual_heading = True
        self.pose_history: List[np.ndarray] = []
        
    def set_hab_path3(self, path_3d):
        """Set a path given in HABITAT world coords (x=right, y=up, z=back).

        The controller works in the planar frame (X=habitat_x, Y=-habitat_z), so:
          * path_2d = (habitat_x, -habitat_z)          -- the 2D tracking path.
          * path_3d_standard = (habitat_x, -habitat_z, habitat_y) -- a z-up copy
            used only to densify in 3D; it is mapped straight back to habitat
            coords (path_3d_dense) afterward for debug drawing.
        """
        path_2d = path_3d[:, [0, 2]]*np.array([[1.0,-1.0]])
        path_3d_standard = path_3d[:, [0, 2, 1]]*np.array([[1.0, -1.0, 1.0]])
        resolution  = self.pure_pursuit_config.min_look_ahead_distance/4.0
        # densify in the z-up frame, then reorder/sign back to habitat (x,y,z).
        self.path_3d_dense = self.controller.densify_path(path_3d_standard,resolution)[:, [0, 2, 1]]*np.array([[1.0, 1.0, -1.0]])
        self.path_3d_base = path_3d
        self.set_rob_path2(path_2d)
        
        print(f"dense 3d path length: {len(self.path_3d_dense)}, 2d length: {len(self.controller.path)}")
        
    def set_rob_path2(self, path_2d):
        self.path_2d = np.asarray(path_2d, dtype=float)
        if self.path_2d.ndim != 2 or self.path_2d.shape[1] != 2:
            raise ValueError("path_2d must have shape (N, 2)")
        if len(self.path_2d) == 0:
            raise ValueError("path_2d must contain at least one waypoint")

        self.goal_pos = self.path_2d[-1].copy()
        self.controller.set_path(path_2d)
        if self.align_virtual_heading:
            # align virtual heading with path direction
            if len(self.path_2d) > 1:
                start_pos = self.path_2d[0]
                next_pos = self.path_2d[1]
                virtual_heading = np.arctan2(next_pos[1]-start_pos[1],next_pos[0]-start_pos[0])
                self.virtual_heading = virtual_heading
            else:
                self.virtual_heading = 0.0
            self.align_virtual_heading = False
            
    def get_distance_to_goal(self) -> float:
        dist_to_goal = float(np.linalg.norm(self.current_pose[:2] - self.goal_pos))
        return dist_to_goal
    
    def is_goal_reached(self) -> bool:
        dist_to_goal = self.get_distance_to_goal()
        goal_reached = dist_to_goal < self.pure_pursuit_config.goal_tolerance_dist
        return goal_reached
    
    def step(self, dt: float = 1.0 / 60.0, heading_alignment_gain: float = 3.0,max_vw:float=1.5,final_heading = None) -> dict:
        if self.path_2d is None or self.goal_pos is None:
            raise RuntimeError("No path set. Call set_hab_path3() or set_rob_path2() first.")

        self.virtual_heading += self.virtual_angular_velocity * dt

        self.current_pose, controller_velocity = self.robot_sim.get_2d_pose_and_twist(
            virtual_heading=self.virtual_heading
        )
        # proj_idx = self.controller._calc_index(self.current_pose)
        # proj_point = self.controller.path[proj_idx]
        # angle = np.arctan2(proj_point[1]-self.current_pose[1], proj_point[0]-self.current_pose[0])
        # # invert the virtual heading if the angle to the projected point is more than 90 degrees away from the virtual heading
        # if angle < -np.pi/2:
        #     self.virtual_heading += np.pi
        #     controller_velocity*=-1.0
        # if angle > np.pi/2:
        #     self.virtual_heading -= np.pi
        #     controller_velocity*=-1.0 
                   
        self.pose_history.append(self.current_pose[:2].copy())
        virtual_pose = np.array(
            [self.current_pose[0], self.current_pose[1], self.virtual_heading], dtype=float
        )
        virtual_velocity = np.array(
            [controller_velocity[0], self.virtual_angular_velocity], dtype=float
        )

        goal_reached = self.is_goal_reached()
        heading_error = 0.0
        w_body = 0.0
        next_velocity_ref, debug_info = self.controller.compute_velocity(
            virtual_pose, virtual_velocity
        )
        v_ref, w_ref = next_velocity_ref
        if not goal_reached:
            #heading_error = self.virtual_heading - self.robot_sim.get_joint_state("joint_th")
            tangent_heading = self.controller.get_look_ahead_heading()
            goal_heading = np.arctan2(self.goal_pos[1]-self.current_pose[1],self.goal_pos[0]-self.current_pose[0]) if final_heading is None else final_heading
            # smoothly blend between tangent heading and goal heading based on distance to goal
            dist_to_goal = self.get_distance_to_goal()
            blend_factor = np.clip(dist_to_goal / 3.0, 0.0, 1.0)
            blended_heading = blend_factor * tangent_heading + (1.0 - blend_factor) * goal_heading
            heading_error = blended_heading - self.robot_sim.get_joint_state("joint_th")
            # Normalize heading error to [-pi, pi]
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
            w_body = max(-max_vw, min(max_vw, heading_error*heading_alignment_gain))


        if goal_reached:
            twist_command = np.array([0.0, 0.0, 0.0], dtype=float)
            self.virtual_angular_velocity = 0.0
        else:
            twist_command = np.array([v_ref, 0.0, w_body], dtype=float)
            self.virtual_angular_velocity = w_ref

        return {
            "twist_frame": (self.virtual_heading, self.virtual_angular_velocity),
            "chassis_twist": twist_command,
        }
    def draw_path(self):
        if self.path_3d_dense is not None:
            last_idx = self.controller.last_idx
            purple = mn.Color4(1.0, 0.0, 1.0, 1.0)
            # draw the completed portion of the path and uncompleted portions separately
            self.robot_sim.draw_path(self.path_3d_base,purple)
            # self.robot_sim.draw_path(self.path_3d_dense[last_idx:], purple)
            # self.robot_sim.draw_path(self.path_3d_dense[:last_idx], mn.Color4(0.0, 1.0, 1.0, 1.0))

    def visualize_bev(self, figsize=(5, 5), dpi=100) -> np.ndarray:
        """Renders a top-down (BEV) debug plot of the path-following state as an RGB image.

        Draws the reference trajectory and the robot's traveled path so far, plus the robot's
        actual pose as a large arrow with the controller's internal virtual heading (used for
        path tracking, which can drift from the true heading) overlaid as a smaller arrow at the
        same position -- this divergence is often exactly what a path-following bug looks like.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if self.controller.path is not None:
            ref_path = self.controller.path
            ax.plot(ref_path[:, 0], ref_path[:, 1], "--", color="0.5", linewidth=1.0,
                     label="reference path", zorder=1)
        if self.goal_pos is not None:
            ax.scatter([self.goal_pos[0]], [self.goal_pos[1]], color="red", marker="*", s=150,
                       zorder=4, label="goal")

        if len(self.pose_history) > 1:
            hist = np.array(self.pose_history)
            ax.plot(hist[:, 0], hist[:, 1], "-", color="tab:blue", linewidth=1.5,
                    label="traveled path", zorder=2)

        if hasattr(self.controller, "look_ahead_idx") and self.controller.path is not None:
            look_ahead_pos = self.controller.path[self.controller.look_ahead_idx][:2]
            ax.scatter([look_ahead_pos[0]], [look_ahead_pos[1]], color="tab:green", marker="o",
                       s=40, zorder=4, label="look-ahead pt")

        x, y, actual_heading = self.robot_sim.get_2d_pose()

        arrow_len = 0.5
        ax.annotate(
            "", xytext=(x, y),
            xy=(x + arrow_len * math.cos(actual_heading), y + arrow_len * math.sin(actual_heading)),
            arrowprops=dict(facecolor="black", edgecolor="black", width=2.5, headwidth=12, headlength=12),
            zorder=6,
        )
        v_arrow_len = 0.3
        ax.annotate(
            "", xytext=(x, y),
            xy=(x + v_arrow_len * math.cos(self.virtual_heading), y + v_arrow_len * math.sin(self.virtual_heading)),
            arrowprops=dict(facecolor="tab:orange", edgecolor="tab:orange", width=1.2, headwidth=7, headlength=7),
            zorder=7,
        )
        ax.scatter([x], [y], color="black", s=15, zorder=8)

        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.15)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"BEV debug (dist to goal: {self.get_distance_to_goal():.2f}m)")
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.tight_layout()

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return img

# The recording tick rate. Habitat advances physics only in whole 0.008 s steps,
# so 25 Hz is the on-grid recording rate (see sim.py::step). The rotation gain
# defaults to 1/dt at this rate, which is what makes the law deadbeat.
DEFAULT_ROTATION_FPS = 25.0


@dataclass
class HeadingRotationConfig:
    """Tuning for :class:`HeadingRotationController`.

    Defaults are the values recommended by the rotation-tuning study (see
    ``dump/data_diagnostics/rotation_tuning/FINDINGS.md`` in the spatial_training
    repo). The previous behaviour was ``gain=3.0, w_min=0.0,
    heading_tolerance=0.03, handoff_tolerance=None, no_overshoot_clamp=False``,
    which spent ~35% of every replay tick decaying exponentially towards a
    1.7-degree deadband.

    Attributes:
        w_max: hard cap on commanded |w| (rad/s). The turn is never allowed to be
            more violent than this regardless of the error.
        w_min: floor on commanded |w| while outside the deadband (rad/s). Non-zero
            gives a trapezoidal profile (ramp, coast, land) instead of a
            proportional decay. Safe only together with ``no_overshoot_clamp``.
            Ignored when ``alpha_max`` is set -- a floor and a ramp through that
            floor are contradictory.
        alpha_max: bound on angular ACCELERATION, |dw/dt| (rad/s^2). ``None``
            leaves it unbounded, which with the deadbeat ``gain`` makes the
            commanded profile a rectangle: step to ``w_max``, hold, step to zero
            -- 50 rad/s^2 (2865 deg/s^2) at each edge, and 100 rad/s^2 across a
            direction reversal. No real base can do that; it would skid, and a
            policy trained to imitate it learns something it cannot execute off
            the simulator. Setting it makes the profile a true trapezoid instead:
            ramp at ``alpha_max``, cruise at ``w_max``, brake along
            ``sqrt(2*alpha_max*|e|)`` and land. This is NOT ``w_min`` under
            another name -- ``w_min`` floors the velocity, this bounds its
            derivative, and only the latter puts mass in the intermediate rates.

            The default of 12 comes from a sweep over 192 held-out episodes
            (``dump/rotation_segmentation/FINDINGS.md``). It is cheap: episode
            duration goes from 0.671 to 0.705 of the pre-tuning baseline, i.e.
            the softening costs 4.8% and keeps 95% of the speed win, because only
            the EDGES of the turn are ramped and ``w_max`` is untouched. What it
            buys is the shape of the angular-velocity distribution, which is the
            actual point: saturation falls 0.978 -> 0.738 and the share of
            turning ticks at an intermediate rate rises 0.021 -> 0.210. That
            matters because this project's central finding is that a
            conditional-mean regression head cannot represent a bimodal
            stop-or-drive action distribution; a bang-bang rotation law makes the
            yaw channel bimodal in exactly the same way, and regenerating the
            corpus with it would transplant the disease into a new channel.
            Delivered scan sweep also improves (7.5 -> 3.4 degrees lost).
        gain: proportional gain (1/s). At ``gain == 1/dt`` (25.0 at the 25 Hz
            recording rate) the law is deadbeat: full ``w_max`` until the target
            is one tick away, then land exactly on it. A larger gain is also
            deadbeat because ``no_overshoot_clamp`` binds first; a smaller one
            degrades gracefully to proportional near the target. Ignored when
            ``alpha_max`` is set, which defines the profile geometrically.
        heading_tolerance: acceptance deadband for a heading target (rad). Any
            residual below this is snapped to zero rather than chased.
        handoff_tolerance: looser acceptance (rad) applied to the FINAL heading of
            a trajectory that hands straight over to the path controller. The
            path controller re-aligns the heading itself while driving, and its
            translation is applied in its own virtual-heading frame, so a residual
            here costs camera pointing for a few ticks and nothing else. ``None``
            means "use ``heading_tolerance``".
        no_overshoot_clamp: additionally clamp ``|w| <= |e| / dt`` so a single tick
            can never carry the robot past its target. This is what makes ``w_min``
            and a high ``gain`` safe: without it they produce a limit cycle around
            the deadband.
    """

    w_max: float = 2.0
    w_min: float = 0.0
    alpha_max: Optional[float] = 12.0
    gain: float = DEFAULT_ROTATION_FPS
    heading_tolerance: float = 0.03
    handoff_tolerance: Optional[float] = 0.09
    no_overshoot_clamp: bool = True


class HeadingRotationController:
    """Rotates the robot in place through a sequence of target headings.

    Unlike HolonomicRobotController (which follows a 2D path), this controller only
    commands angular velocity: the chassis is driven through a 1D trajectory of yaw
    headings, advancing to the next heading once the current one is reached.

    There are two control laws, selected by ``config.alpha_max``.

    **Bounded acceleration (the default, ``alpha_max`` set).** A rest-to-rest
    trapezoidal yaw profile: ramp at ``alpha_max``, cruise at ``w_max``, brake
    along the arrival parabola ``sqrt(2*alpha_max*|e|)`` and land on the target.
    See ``_bounded_accel_command``.

    **Unbounded (``alpha_max = None``)** -- a capped, floored, no-overshoot
    proportional law::

        e = shortest-path error to the current target heading
        |e| <= tolerance      ->  accept the target, command 0
        otherwise             ->  w = sign(e) * min(w_max,
                                                    max(w_min, gain*|e|),
                                                    |e|/dt   [if clamped])

    With ``gain = 1/dt`` that law is bang-bang: the robot turns at exactly
    ``w_max`` until it is one control tick away from the target and then lands on
    it. It is the fastest possible profile and it is retained for A/B work, but
    it is not what should be recorded into a dataset -- its velocity trace is a
    rectangle, which is both unrealisable on hardware and maximally bimodal (see
    ``HeadingRotationConfig.alpha_max``).

    Both laws land exactly, never overshoot, and are safe because the base's yaw
    joint is a velocity motor that reaches its commanded rate within one substep
    (measured: no lag, no coast). There is no actuator inertia to absorb an
    error, which is why the ``|e|/dt`` clamp is sufficient -- and equally why the
    acceleration bound has to be imposed by the controller rather than inherited
    from the plant.

    ``tolerance`` is ``config.heading_tolerance``, except for the final heading of
    a trajectory flagged ``handoff_to_path=True``, which uses the looser
    ``config.handoff_tolerance``.
    """

    def __init__(
        self,
        robot_sim: HabitatRobotSim,
        config: Optional[HeadingRotationConfig] = None,
        w_max: Optional[float] = None,
        heading_tolerance: Optional[float] = None,
    ):
        self.robot_sim = robot_sim
        self.config = config if config is not None else HeadingRotationConfig()
        # Back-compat conveniences for the two knobs that used to be constructor
        # arguments; they override the corresponding config fields.
        if w_max is not None:
            self.config.w_max = w_max
        if heading_tolerance is not None:
            self.config.heading_tolerance = heading_tolerance
        self.headings = None
        self.handoff_to_path = False
        self.reset()

    # ``w_max`` was a public attribute before the config existed; keep it working.
    @property
    def w_max(self) -> float:
        return self.config.w_max

    @property
    def heading_tolerance(self) -> float:
        return self.config.heading_tolerance

    # to be called each simulator reset/episode
    def reset(self):
        self.target_idx = 0
        self.current_pose = None
        self.pose_history: List[np.ndarray] = []
        # Last commanded angular velocity, needed by the alpha_max rate limit.
        # Zero is correct at a reset: a look phase always begins from rest.
        self._w_prev = 0.0

    # set the 1D heading trajectory (sequence of target yaw angles in radians)
    def set_heading_trajectory(
        self,
        headings,
        remove_redundant: bool = False,
        handoff_to_path: bool = False,
    ):
        """Install a heading trajectory.

        ``handoff_to_path`` marks that this trajectory is followed immediately by
        a path-following phase, which lets the final heading be accepted at the
        looser ``config.handoff_tolerance`` (see the class docstring).
        """
        headings = np.asarray(headings, dtype=float).reshape(-1)
        if len(headings) == 0:
            raise ValueError("headings must contain at least one target heading")
        if remove_redundant:
            headings = self._remove_redundant_headings(headings)
        self.headings = headings
        self.handoff_to_path = handoff_to_path
        self.target_idx = 0

    def tolerance_for_target(self, idx: Optional[int] = None) -> float:
        """Acceptance deadband for heading ``idx`` (default: the active target)."""
        idx = self.target_idx if idx is None else idx
        cfg = self.config
        is_last = self.headings is not None and idx == len(self.headings) - 1
        if is_last and self.handoff_to_path and cfg.handoff_tolerance is not None:
            return cfg.handoff_tolerance
        return cfg.heading_tolerance

    @staticmethod
    def _remove_redundant_headings(headings):
        """Drops intermediate waypoints the robot would sweep through anyway.

        The controller always rotates the *shortest way* to each kept target, so we can
        only drop a waypoint if a single shortest-path move from the last kept anchor still
        sweeps monotonically through it, in order, on the way to the next waypoint. That
        means tracking the signed rotation accumulated since the last kept heading: a
        waypoint must be kept whenever continuing to the next one would either reverse the
        turn direction or push the accumulated sweep to +/-180 degrees (past which a
        shortest-path move no longer covers the intended arc). Endpoints are always kept.
        """
        def signed_delta(a, b):
            # shortest-path rotation from a to b, in [-pi, pi]
            return (b - a + math.pi) % (2 * math.pi) - math.pi

        n = len(headings)
        kept_idx = [0]
        acc = 0.0  # signed rotation from the last kept anchor to headings[i-1] (|acc| < pi)
        for i in range(1, n):
            d = signed_delta(headings[i - 1], headings[i])  # continuous rotation to the next waypoint
            new_acc = acc + d
            same_dir = (d == 0.0) or (acc == 0.0) or (acc * d > 0.0)
            # can we bridge over headings[i-1], i.e. reach headings[i] from the anchor with one
            # shortest-path move that still passes through everything dropped so far, in order?
            bridges = same_dir and abs(new_acc) < math.pi - 1e-9
            if i == n - 1:
                # the final waypoint is always kept; anchor headings[i-1] first if we can't bridge
                if not bridges and kept_idx[-1] != i - 1:
                    kept_idx.append(i - 1)
                kept_idx.append(i)
            elif bridges:
                acc = new_acc  # headings[i-1] is redundant; extend the sweep to reach headings[i]
            else:
                if kept_idx[-1] != i - 1:
                    kept_idx.append(i - 1)  # headings[i-1] needed to preserve direction/order
                acc = d
        return headings[kept_idx]

    def is_trajectory_complete(self) -> bool:
        return self.headings is None or self.target_idx >= len(self.headings)

    def get_heading_error(self) -> float:
        current_heading = self.robot_sim.get_joint_state("joint_th")
        target_heading = self.headings[self.target_idx]
        # shortest-path error normalized to [-pi, pi]
        error = (target_heading - current_heading + math.pi) % (2 * math.pi) - math.pi
        return error

    def step(
        self,
        dt: float = 1.0 / 60.0,
        heading_alignment_gain: Optional[float] = None,
    ) -> dict:
        """One control tick. ``heading_alignment_gain`` overrides ``config.gain``."""
        if self.headings is None:
            raise RuntimeError("No heading trajectory set. Call set_heading_trajectory() first.")

        self.current_pose = self.robot_sim.get_2d_pose()
        self.pose_history.append(self.current_pose[:2].copy())

        stop = {
            "twist_frame": "robot",
            "chassis_twist": np.array([0.0, 0.0, 0.0], dtype=float),
        }
        if self.is_trajectory_complete():
            self._w_prev = 0.0
            return stop

        heading_error = self.get_heading_error()

        # advance to the next heading once the current one is inside its deadband
        while abs(heading_error) < self.tolerance_for_target():
            self.target_idx += 1
            if self.is_trajectory_complete():
                self._w_prev = 0.0
                return stop
            heading_error = self.get_heading_error()

        cfg = self.config
        if cfg.alpha_max is not None and cfg.alpha_max > 0.0 and dt > 0.0:
            w_cmd = self._bounded_accel_command(heading_error, dt)
        else:
            gain = cfg.gain if heading_alignment_gain is None else heading_alignment_gain
            magnitude = min(cfg.w_max, max(cfg.w_min, gain * abs(heading_error)))
            if cfg.no_overshoot_clamp:
                # never command more rotation than remains: one tick cannot overshoot.
                magnitude = min(magnitude, abs(heading_error) / dt)
            w_cmd = math.copysign(magnitude, heading_error)

        self._w_prev = float(w_cmd)
        return {
            "twist_frame": "robot",
            "chassis_twist": np.array([0.0, 0.0, float(w_cmd)], dtype=float),
        }

    def _bounded_accel_command(self, heading_error: float, dt: float) -> float:
        """Trapezoidal yaw profile with |dw/dt| <= ``config.alpha_max``.

        Three bounds, all upper bounds on the magnitude, plus one signed rate
        limit:

        * ``w_max`` -- the cruise rate, unchanged.
        * the discrete arrival rate ``sqrt((alpha_max*dt)^2 + 2*alpha_max*d)
          - alpha_max*dt``, where ``d = |e| - tol`` -- the fastest rate that can
          be held for one whole tick and still leave room to brake to rest by the
          ACCEPTANCE DEADBAND. This shapes the TRAILING edge, and it is the whole
          reason the profile is graded rather than a rectangle: the deadbeat law's
          trailing edge is a single step from ``w_max`` to zero, 50 rad/s^2
          (2865 deg/s^2) at 25 Hz.

          Two details, both of which the obvious ``sqrt(2*alpha_max*|e|)`` gets
          wrong. First, braking towards the deadband edge rather than towards zero
          error is what makes the bound hold through the *landing*: aiming at zero
          leaves the controller still commanding ``sqrt(2*alpha_max*tol)`` on the
          tick it accepts -- 0.85 rad/s at the defaults, an unbounded 21 rad/s^2
          step to zero. Second, the ``alpha_max*dt`` terms are the discrete-time
          correction. A tick holds its rate constant, so it overshoots the
          continuous braking parabola and the next tick would need to decelerate
          harder than ``alpha_max`` to catch up -- measured at 14.2 against a
          bound of 12. The corrected form is the root of
          ``w*dt + w^2/(2*alpha_max) = d``, i.e. "coast one tick, then stop".
        * a floor of ``alpha_max * dt`` on that arrival rate, which is what stops
          the previous item from reintroducing a settle tail. Without it the rate
          goes to zero exactly as ``|e|`` reaches ``tol`` and the last fraction of
          a degree takes unboundedly many ticks -- the same crawl the deadbeat law
          was introduced to remove. One tick's worth of acceleration is the
          smallest floor that still guarantees the deadband is crossed, and it
          makes the final step to zero exactly ``alpha_max``. So the whole
          profile, landing included, is bounded by ``alpha_max``.
        * ``|e| / dt`` (``no_overshoot_clamp``) -- one tick can never carry the
          robot past the target. It only ever commands LESS motion.
        * a signed clip to ``w_prev +/- alpha_max * dt``, which shapes the LEADING
          edge and, because it is signed rather than applied to the magnitude,
          forces a direction reversal to ramp down through zero instead of
          jumping across it.

        ``config.w_min`` is deliberately not applied here: a velocity floor and a
        ramp through that floor are contradictory, and the floor was measured and
        rejected on its own terms (it is dominated by simply raising the gain).
        """
        cfg = self.config
        alpha = cfg.alpha_max
        e = heading_error
        tol = self.tolerance_for_target()
        # Fastest rate that survives one whole tick and still brakes to rest by
        # the deadband edge, floored at one tick of acceleration so the deadband
        # is always crossed.
        d = max(abs(e) - tol, 0.0)
        adt = alpha * dt
        w_arrive = max(math.sqrt(adt * adt + 2.0 * alpha * d) - adt, adt)
        target = math.copysign(min(cfg.w_max, w_arrive), e)
        # Leading-edge ramp / reversal, in signed space.
        lo, hi = self._w_prev - alpha * dt, self._w_prev + alpha * dt
        w_cmd = min(max(target, lo), hi)
        if cfg.no_overshoot_clamp:
            w_cmd = math.copysign(min(abs(w_cmd), abs(e) / dt), w_cmd)
        return w_cmd
