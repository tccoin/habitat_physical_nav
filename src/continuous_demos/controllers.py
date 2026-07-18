"""Path-following / heading-tracking control layer for :class:`~continuous_demos.sim.HabitatRobotSim`.

Split out of ``sim.py`` (which owns only the simulator itself): these classes
are the non-simulation control logic that drives the sim -- the pure-pursuit
path follower (wrapping ``pure_pursuit.PurePursuitController``) and the
in-place heading-sequence rotator used for "looking" phases of a replay. See
``sim.py``'s module docstring for the three coordinate frames referenced
throughout (habitat world / robot body / planar control frame).
"""

import math
from typing import List

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
            w_max=70.0,
            w_min=-70.0,
            a_max=3.0,
            aw_max=100.0,
            dt=1.0/30.0,
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

class HeadingRotationController:
    """Rotates the robot in place through a sequence of target headings smoothly.

    Unlike HolonomicRobotController (which follows a 2D path), this controller only
    commands angular velocity: the chassis is driven through a 1D trajectory of yaw
    headings, advancing to the next heading once the current one is reached.
    """

    def __init__(
        self,
        robot_sim: HabitatRobotSim,
        w_max: float = 2.0,
        heading_tolerance: float = 0.03,
    ):
        self.robot_sim = robot_sim
        self.w_max = w_max
        self.heading_tolerance = heading_tolerance
        self.headings = None
        self.reset()

    # to be called each simulator reset/episode
    def reset(self):
        self.target_idx = 0
        self.current_pose = None
        self.pose_history: List[np.ndarray] = []

    # set the 1D heading trajectory (sequence of target yaw angles in radians)
    def set_heading_trajectory(self, headings, remove_redundant: bool = False):
        headings = np.asarray(headings, dtype=float).reshape(-1)
        if len(headings) == 0:
            raise ValueError("headings must contain at least one target heading")
        if remove_redundant:
            headings = self._remove_redundant_headings(headings)
        self.headings = headings
        self.target_idx = 0

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

    def step(self, dt: float = 1.0 / 60.0, heading_alignment_gain: float = 3.0) -> dict:
        if self.headings is None:
            raise RuntimeError("No heading trajectory set. Call set_heading_trajectory() first.")

        self.current_pose = self.robot_sim.get_2d_pose()
        self.pose_history.append(self.current_pose[:2].copy())

        if self.is_trajectory_complete():
            return {
                "twist_frame": "robot",
                "chassis_twist": np.array([0.0, 0.0, 0.0], dtype=float),
            }

        heading_error = self.get_heading_error()

        # advance to the next heading once the current one is reached
        while abs(heading_error) < self.heading_tolerance:
            self.target_idx += 1
            if self.is_trajectory_complete():
                return {
                    "twist_frame": "robot",
                    "chassis_twist": np.array([0.0, 0.0, 0.0], dtype=float),
                }
            heading_error = self.get_heading_error()

        w_cmd = np.clip(heading_error * heading_alignment_gain, -self.w_max, self.w_max)
        return {
            "twist_frame": "robot",
            "chassis_twist": np.array([0.0, 0.0, float(w_cmd)], dtype=float),
        }
