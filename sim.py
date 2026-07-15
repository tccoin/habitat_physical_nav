import math
from pathlib import Path
import numpy as np
import habitat_sim
from typing import Dict, List, Any, Optional
from pure_pursuit import PurePursuitController,ControllerConfig
import magnum as mn
from PIL import Image

# =============================================================================
# COORDINATE FRAME CONVENTIONS
# -----------------------------------------------------------------------------
# Three frames appear in this file. Keeping them straight is essential; the
# habitat one in particular was previously mis-documented (see reset()).
#
# 1. HABITAT WORLD FRAME  (right-handed, y-up)
#    The frame of `robot.translation`, scene geometry, and every quaternion in
#    the demonstration `reference_replay`. Per habitat's own constants
#    (`habitat_sim.geo`):  RIGHT = +x,  UP = +y,  FRONT = -z.
#    So an agent's *forward* is local -z (NOT +x). Axes: +x right, +y up,
#    +z backward.
#
# 2. ROBOT BODY FRAME  (right-handed, ROS/URDF style)
#    +x forward, +y left, +z up. `step()` reads `chassis_twist = [v_x, v_y, w]`
#    in this frame: v_x drives the robot's own forward, v_y its left, w yaws
#    about +z. Commanding [1,0,0] moves the robot visually forward.
#
# 3. PLANAR CONTROL / WORLD FRAME  (2D, right-handed, up = habitat +y)
#    The ground-projected world frame the base actually navigates in, and the
#    frame `joint_th` is measured in. Axes:  X = habitat_x,  Y = -habitat_z,
#    heading theta = joint_th measured CCW from +X.  (X x Y = +habitat_y = up,
#    so it is right-handed.) Used consistently by the world prismatic joints
#    joint_x/joint_y, by `get_2d_pose` ([jx + habitat_x, jy - habitat_z, theta]),
#    and by the controllers (`set_hab_path3` projects paths to (habitat_x,
#    -habitat_z)).
#
#    IMPORTANT: frames 2 and 3 are the SAME convention (x forward, y left, z up).
#    Frame 3 is the fixed WORLD instance; frame 2 is that same frame CARRIED BY
#    THE BASE -- translated by (joint_x, joint_y) and yawed by joint_th. Set
#    joint_x = joint_y = joint_th = 0 and the body frame coincides exactly with
#    the planar world frame (body +x -> +X, body +y -> +Y, body +z -> up). This
#    follows directly from step()'s twist rotation at theta = 0, and was verified
#    in-sim: a [1,0,0] body twist at heading theta moves the base along theta in
#    frame 3 to <0.01 deg. (So frame 3's theta=0 reference direction, +X =
#    habitat_x, is simply where the robot's forward points at zero yaw.)
#
#    Conversions between HABITAT (1) and the PLANAR frame (3):
#      * position:  (hx, hy, hz)  ->  (X, Y) = (hx, -hz)
#      * heading:   rotate FRONT=(0,0,-1) by the habitat quaternion to get the
#        world forward (fx, fy, fz), then  theta = atan2(-fz, fx).
#        This is exactly `hab_quat_to_heading`, and it is what maps a demo
#        agent's orientation onto a robot joint_th target.
# =============================================================================


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_URDF_PATH = (
    PACKAGE_DIR / "tidybot_description" / "urdf" / "tidybot_base_minimal.urdf"
)
DEFAULT_SCENE_PATH = (
    PACKAGE_DIR.parent
    / "data"
    / "scene_datasets"
    / "mp3d_example"
    / "17DRP5sb8fy"
    / "17DRP5sb8fy.glb"
)
class HabitatRobotSim:
    def __init__(
        self, 
        sim_settings: Dict[str, Any], 
        urdf_path: str, 
        sensor_setup: Dict[str, List[habitat_sim.sensor.SensorSpec]], 
        fixed_base: bool = False,
        max_substep_dt: float = 1.0 / 120.0  # high frequency for accurate kinematic integration
    ):
        self.urdf_path = urdf_path
        self.sensor_setup = sensor_setup
        self.fixed_base = fixed_base
        self.max_substep_dt = max_substep_dt
        self.robot = None
        self.joint_name_to_motor_id = {}
        
        # 1. Flatten sensor_setup and assign to a dummy agent configuration
        self.cfg = habitat_sim.utils.settings.make_cfg(sim_settings)
        agent_id = sim_settings.get("default_agent", 0)
        self.cfg.agents[agent_id].radius=0.28
        self.cfg.agents[agent_id].height=1.0
        all_sensor_specs = []
        for specs in self.sensor_setup.values():
            all_sensor_specs.extend(specs)
            
        self.cfg.agents[agent_id].sensor_specifications = all_sensor_specs

        # 2. Initialize simulator
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent_id = agent_id
        # self.sim.reconfigure(self.cfg)

        # 3. Setup the robot and attach sensors
        self._setup_robot_and_sensors()
        self.robot = None
    def _setup_robot_and_sensors(self):
        # Load the URDF
        aom = self.sim.get_articulated_object_manager()
        if self.robot is None:
            self.robot = aom.add_articulated_object_from_urdf(
                filepath=self.urdf_path,
                fixed_base=self.fixed_base,
                global_scale=1.0,
                mass_scale=1.0,
                force_reload=True,
                maintain_link_order=False,
                intertia_from_urdf=False,
            )
        self.robot.auto_clamp_joint_limits = True
        # The URDF is authored z-up (ROS/URDF convention), but habitat world is
        # y-up. This -90 deg rotation about +x tilts the whole model so its z-up
        # sits along habitat +y. It is a fixed model-mounting offset, independent
        # of joint_th (planar heading is still driven entirely by joint_th).
        self.base_rotation_offset = mn.Quaternion.rotation(
            mn.Rad(-math.pi / 2), mn.Vector3(1.0, 0.0, 0.0)
        )
        self.robot.rotation = self.base_rotation_offset
        # Map joint names to motor IDs
        self.joint_name_to_motor_id = {
            self.robot.get_link_joint_name(link_id): motor_id
            for motor_id, link_id in self.robot.existing_joint_motor_ids.items()
        }

        # Remove motors for passive joints (just like the viewer script)
        # for _passive_joint in ("joint_z", "joint_pitch", "joint_roll"):
        #     _motor_id = self.joint_name_to_motor_id.pop(_passive_joint, None)
        #     if _motor_id is not None:
        #         self.robot.remove_joint_motor(_motor_id)
        for _lid in range(self.robot.num_links):
            if self.robot.get_link_joint_name(_lid) == "joint_body_mount":
                self.robot.set_link_friction(_lid, 0.01)
                break
        # Map link names to link IDs to easily find where to attach sensors
        link_name_to_id = {
            self.robot.get_link_name(i): i 
            for i in range(self.robot.num_links)
        }

        # Attach sensors to the corresponding robot links
        agent_node = self.sim.get_agent(self.agent_id).scene_node
        for link_name, specs in self.sensor_setup.items():
            if link_name not in link_name_to_id:
                raise ValueError(f"Link '{link_name}' not found in URDF.")
                
            link_id = link_name_to_id[link_name]
            link_scene_node = self.robot.get_link_scene_node(link_id)
            
            for spec in specs:
                # Get the instantiated sensor node from the agent's subtree
                sensor_node = agent_node.subtree_sensors.get(spec.uuid)
                if sensor_node is not None:
                    # Detach from agent and attach to the robot link
                    sensor_node.object.parent = link_scene_node
                    # The sensor now inherits the link's frame (which carries the
                    # z-up base_rotation_offset). Re-orient it so the camera looks
                    # along the robot's forward (+x body) in habitat's -z-forward
                    # image convention, instead of pointing at the floor.
                    sensor_node.object.rotation = sensor_node.object.rotation = mn.Quaternion.rotation(mn.Rad(-math.pi/2), mn.Vector3(0, 1, 0))*mn.Quaternion.rotation(mn.Rad(-math.pi/2), mn.Vector3(0, 0, 1))
                    # Ensure the sensor is exactly at the URDF link's origin
                    sensor_node.object.translation = mn.Vector3(0, 0, 0)
    
    
    def step(
        self, 
        chassis_twist: np.ndarray, 
        joint_targets: Optional[Dict[str, float]] = None, 
        twist_frame: str = "robot",
        dt: float = 1.0 / 60.0
    ) -> dict:
        """Advance physics under a base twist command.

        chassis_twist = [v_x, v_y, w] is in the ROBOT BODY frame (frame 2):
        v_x = forward (+x), v_y = left (+y), w = yaw rate about +z. With
        twist_frame="robot" the body velocity is rotated into the planar world
        frame (frame 3: X=habitat_x, Y=-habitat_z) by the current heading
        joint_th and applied to the world prismatic joints joint_x/joint_y; w
        goes straight to joint_th. twist_frame="world" applies [v_x, v_y]
        directly in that planar world frame; a (theta0, w) tuple integrates the
        heading across substeps (used by the pure-pursuit virtual heading).
        """
        # Determine substepping
        num_steps = max(1, math.ceil(dt / self.max_substep_dt))
        substep_dt = dt / num_steps
        
        v_x_cmd, v_y_cmd, omega_cmd = chassis_twist
        
        for step_idx in range(num_steps):
            # 1. Update chassis velocities
            if twist_frame == "world":
                world_v_x, world_v_y = v_x_cmd, v_y_cmd
            elif twist_frame == "robot" or isinstance(twist_frame, tuple):
                # Get current heading to transform body twist to world frame
                heading = self.get_joint_state("joint_th") if twist_frame == "robot" else float(twist_frame[0])+float(twist_frame[1])*step_idx*substep_dt
                
                # Standard 2D rotation matrix
                world_v_x = v_x_cmd * math.cos(heading) - v_y_cmd * math.sin(heading)
                world_v_y = v_x_cmd * math.sin(heading) + v_y_cmd * math.cos(heading)
            else:
                raise ValueError("twist_frame must be 'world' or 'robot'")

            # Apply to motors
            for joint_name, vel in [("joint_x", world_v_x), ("joint_y", world_v_y), ("joint_th", omega_cmd)]:
                motor_id = self.joint_name_to_motor_id.get(joint_name)
                if motor_id is not None:
                    settings = self.robot.get_joint_motor_settings(motor_id)
                    settings.velocity_target = vel
                    self.robot.update_joint_motor(motor_id, settings)

            # 2. Update other joint targets (e.g., arm, head) if provided
            if joint_targets:
                for j_name, pos_target in joint_targets.items():
                    m_id = self.joint_name_to_motor_id.get(j_name)
                    if m_id is not None:
                        settings = self.robot.get_joint_motor_settings(m_id)
                        settings.position_target = pos_target
                        self.robot.update_joint_motor(m_id, settings)

            # 3. Advance physics
            self.sim.step_world(substep_dt)
            
    def get_obs(self) -> dict:# Return standard observation dictionary
        return self.sim.get_sensor_observations(self.agent_id)
    
    def reset(
        self, 
        scene_id: Optional[str] = None, 
        robot_translation: Optional[np.ndarray] = None,
        robot_rotation: Optional[Any] = None,  # Accepts mn.Quaternion or [x, y, z, w] array
        snap_to_navmesh: bool = True,
        vertical_offset: Optional[float] = None, # Manual override if AABB is weird
        home_joint_positions: Optional[Dict[str, float]] = None
    ) -> dict:
        
        # 1. Handle Scene Changes and Graph Rebuild
        if scene_id is not None and scene_id != self.cfg.sim_cfg.scene_id:
            self.cfg.sim_cfg.scene_id = scene_id
            self.sim.reconfigure(self.cfg)
        else:
            self.sim.reset()
            self._setup_robot_and_sensors()

        # 2. Seed the base yaw (joint_th) from the requested orientation.
        # Goal: joint_th = the requested facing as a heading in the planar control
        # frame (frame 3 in the header): theta CCW from +X, X=habitat_x, Y=-habitat_z.
        if robot_rotation is not None:
            if isinstance(robot_rotation, (np.ndarray, list, tuple)):
                if len(robot_rotation) != 4:
                    raise ValueError("robot_rotation array must have 4 elements: [x, y, z, w]")
                # Habitat agent quaternion [x, y, z, w]. Habitat is x=right, y=up,
                # z=back, so the agent's FORWARD is -z (habitat_sim.geo.FRONT), NOT
                # +x. Remap the quaternion into the magnum/planar frame ((hx,hy,hz)
                # -> (hx,-hz,hy)); under that same remap habitat FRONT=(0,0,-1) maps
                # to +y, so the agent's forward axis is +y here (not +x).
                x, y, z, w = robot_rotation
                robot_rotation = mn.Quaternion(mn.Vector3(x, -z, y), w)
                forward_vec = np.array([0.0, 1.0, 0.0])  # habitat FRONT (-z) in magnum frame
            else:
                # Already a magnum quaternion in the planar frame; its +x is forward.
                forward_vec = np.array([1.0, 0.0, 0.0])
            if home_joint_positions is None:
                home_joint_positions = {}
            rotated_forward = robot_rotation.transform_vector(forward_vec)
            # Heading in the control frame X=habitat_x, Y=-habitat_z (== hab_quat_to_heading).
            home_joint_positions["joint_th"] = float(np.arctan2(rotated_forward[1], rotated_forward[0]))
        self.robot.rotation = self.base_rotation_offset #TODO: allow non aligned base.

        # 3. Handle NavMesh Snapping & Vertical Offsets
        if robot_translation is not None:
            if snap_to_navmesh and self.sim.pathfinder.is_loaded:
                safe_point = self.sim.pathfinder.snap_point(robot_translation)
                if not np.isnan(safe_point[0]):
                    robot_translation = safe_point
            
            # Compute dynamic vertical offset if not manually provided
            if vertical_offset is None:
                aabb = self.robot.root_scene_node.cumulative_bb
                # If lowest point (min.y) is negative, we need to shift the robot UP
                vertical_offset = -aabb.min.y if aabb.min.y < 0 else 0.0

            # Apply translation with offset (Habitat is Y-up)
            robot_translation[1] += vertical_offset
            self.robot.translation = robot_translation
            
        # 4. Clear physics momentum & set home joints
        # Zero out velocities and forces to prevent flying away after teleport
        num_dofs = len(self.robot.joint_velocities)
        self.robot.joint_velocities = [0.0] * num_dofs
        self.robot.joint_forces = [0.0] * num_dofs

        if home_joint_positions:
            joint_positions = self.robot.joint_positions.copy()
            for j_name, pos_target in home_joint_positions.items():
                m_id = self.joint_name_to_motor_id.get(j_name)
                if m_id is not None:
                    # Instantly update the kinematic state
                    l_id = self.robot.existing_joint_motor_ids[m_id]
                    pos_offset = self.robot.get_link_joint_pos_offset(l_id)
                    # joint_name = self.robot.get_link_joint_name(l_id)
                    print(f"homing joint '{j_name}' to {pos_target} (motor_id={m_id}, link_id={l_id}, pos_offset={pos_offset})")
                    joint_positions[pos_offset] = pos_target
            self.robot.joint_positions = joint_positions
                    
                    # # Update the motor target so it doesn't try to snap back
                    # settings = self.robot.get_joint_motor_settings(m_id)
                    # settings.position_target = pos_target
                    # self.robot.update_joint_motor(m_id, settings)
        # self.robot.awake = True
        
    #TODO: fix this
    def get_joint_state(self, joint_name: str, is_vel: bool = False) -> float:
        """
        Retrieves the position or velocity of a specific joint by its name.
        """
        motor_id = self.joint_name_to_motor_id.get(joint_name)
        if motor_id is None:
            raise ValueError(f"Joint '{joint_name}' not found in motor mappings.")
            
        link_id = self.robot.existing_joint_motor_ids[motor_id]
        if is_vel:
            # joint_velocities are indexed by DOF id
            dof_offset = self.robot.get_link_dof_offset(link_id)
            return self.robot.joint_velocities[dof_offset]
        else:
            # joint_positions are indexed by position offset
            pos_offset = self.robot.get_link_joint_pos_offset(link_id)
            return self.robot.joint_positions[pos_offset]
        
    def get_2d_pose_and_twist(self, virtual_heading = None) -> tuple[np.ndarray, np.ndarray]:
        """Pose [X, Y, theta] and twist [v_forward, w] in the planar control frame.

        The world position is X = habitat_x, Y = -habitat_z, assembled as the
        prismatic joint value plus the base translation on each axis
        (jx + habitat_x, jy - habitat_z). theta = joint_th, CCW from +X. Passing
        virtual_heading substitutes the controller's virtual heading for joint_th
        (used by pure-pursuit path tracking).
        """
        jx = self.get_joint_state("joint_x")
        jy = self.get_joint_state("joint_y")
        jth = self.get_joint_state("joint_th") if virtual_heading is None else virtual_heading  # Adjust for URDF orientation

        # Retrieve current velocity
        vx = self.get_joint_state("joint_x", is_vel=True)
        vy = self.get_joint_state("joint_y", is_vel=True)
        vth = self.get_joint_state("joint_th", is_vel=True)

        # rx = habitat_x, rz = habitat_z of the base; planar Y uses -habitat_z.
        rz,rx = self.robot.translation[2],self.robot.translation[0]
        v_forward = vx * math.cos(jth) + vy * math.sin(jth)

        # limit jth to [-pi, pi]
        jth = (jth + math.pi) % (2 * math.pi) - math.pi
        return np.array([jx+rx,jy-rz, jth]), np.array([v_forward, vth])

    def get_2d_pose(self) -> np.ndarray:
        """Planar pose [X, Y, theta] with X=habitat_x, Y=-habitat_z, theta=joint_th."""
        jx = self.get_joint_state("joint_x")
        jy = self.get_joint_state("joint_y")
        jth = self.get_joint_state("joint_th")

        jth = (jth + math.pi) % (2 * math.pi) - math.pi
        rz,rx = self.robot.translation[2],self.robot.translation[0]
        return np.array([jx+rx,jy-rz, jth])

    def is_stationary(self,lin_vel_threshold, ang_vel_threshold) -> bool:
        """
        Checks if the robot is stationary based on linear and angular velocity thresholds.
        """
        vx = self.get_joint_state("joint_x", is_vel=True)
        vy = self.get_joint_state("joint_y", is_vel=True)
        vth = self.get_joint_state("joint_th", is_vel=True)

        linear_speed = math.sqrt(vx**2 + vy**2)
        angular_speed = abs(vth)

        return linear_speed < lin_vel_threshold and angular_speed < ang_vel_threshold
    
    def is_tipped_over(self, angle_threshold_degrees: float = 45.0) -> bool:
        """
        Checks if the robot is tipped over based on its orientation.
        """
        # Get the robot's up vector in world coordinates
        roll = self.get_joint_state("joint_roll")
        pitch = self.get_joint_state("joint_pitch")
        return abs(roll) > math.radians(angle_threshold_degrees) or abs(pitch) > math.radians(angle_threshold_degrees)
    
    def navmesh_config_and_recompute(self) -> None:
        """
        This method is setup to be overridden in for setting config accessibility
        in inherited classes.
        """
        self.navmesh_settings = habitat_sim.NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_height = self.cfg.agents[self.agent_id].height
        self.navmesh_settings.agent_radius = self.cfg.agents[self.agent_id].radius
        self.navmesh_settings.include_static_objects = True
        self.navmesh_settings.agent_max_climb = 0.2
        # print(self.navmesh_settings)
        self.sim.recompute_navmesh(
            self.sim.pathfinder,
            self.navmesh_settings,
        )
        
    def draw_axes(self, translation, axis_len=1.0):
        lr = self.sim.get_debug_line_render()
        lr.set_line_width(5)

        opacity = 1.0   
        red = mn.Color4(1.0, 0.0, 0.0, opacity)
        green = mn.Color4(0.0, 1.0, 0.0, opacity)
        blue = mn.Color4(0.0, 0.0, 1.0, opacity)
        white = mn.Color4(1.0, 1.0, 1.0, opacity)
        # draw axes with x+ = red, y+ = green, z+ = blue
        lr.draw_transformed_line(translation, mn.Vector3(axis_len, 0, 0)+translation, red)
        lr.draw_transformed_line(translation, mn.Vector3(0, axis_len, 0)+translation, green)
        lr.draw_transformed_line(translation, mn.Vector3(0, 0, axis_len)+translation, blue)
    
    def draw_path(self,path_3d,color):
        lr = self.sim.get_debug_line_render()
        lr.set_line_width(5)
        opacity = 1.0   
        offset = mn.Vector3(0, 0.2, 0)
        for i in range(len(path_3d)-1):
            lr.draw_transformed_line(path_3d[i]+offset, path_3d[i+1]+offset, color)

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

from habitat_sim.utils.settings import default_sim_settings

def record_path(robot_sim: HabitatRobotSim, controller: HolonomicRobotController, path_3d, timeout=10, fps=30, debug_draw=True):
    frames = []
    poses = []
    controller.set_hab_path3(path_3d)
    robot_sim.sim.navmesh_visualization = debug_draw
    for _ in range(timeout * fps):
        if debug_draw:
            robot_sim.draw_axes(path_3d[0], axis_len=1)
            robot_sim.draw_axes(path_3d[-1], axis_len=1)
        obs =robot_sim.get_obs()
        step_args = controller.step(dt=1.0/fps, heading_alignment_gain=1.0,max_vw=1.5)
        if debug_draw:
            controller.draw_path()
        robot_sim.step(
            **step_args,
            dt=1.0 / fps
        )
        rgb_frame = obs["color_sensor"][..., :3]
        pose_2d = robot_sim.get_2d_pose()
        if debug_draw:
            bev_frame = controller.visualize_bev()
            bev_frame = np.array(
                Image.fromarray(bev_frame).resize((rgb_frame.shape[0], rgb_frame.shape[0]))
            )
            frame = np.hstack([rgb_frame, bev_frame])
        else:
            frame = rgb_frame
        frames.append(frame)
        poses.append(pose_2d)
        if controller.is_goal_reached() and robot_sim.is_stationary(lin_vel_threshold=0.01, ang_vel_threshold=0.01):
            print("Goal reached!")
            break
        if robot_sim.is_tipped_over(angle_threshold_degrees=45.0):
            print("Robot tipped over! Stopping simulation.")
            break
    return frames, poses


def hab_quat_to_heading(rotation_xyzw) -> float:
    """Convert a Habitat ``[x, y, z, w]`` agent rotation to the robot's 2D heading.

    The heading is returned in the planar control frame (frame 3 in the module
    header) -- the frame ``joint_th`` drives motion in and that
    ``HolonomicRobotController`` aligns to: axes ``X = habitat_x``,
    ``Y = -habitat_z`` (see ``get_2d_pose`` / ``set_hab_path3``). The agent's
    forward axis is habitat local ``-z`` (``habitat_sim.geo.FRONT``); we rotate it
    by the quaternion and take its heading in the control plane as
    ``atan2(-forward_z, forward_x)``. Verified to match the actual per-step
    displacement of every ``MOVE_FORWARD`` in the demos to 0.000 deg, so
    ``HeadingRotationController`` tracks these headings faithfully and a look phase
    leaves the robot facing the direction it must next walk.

    This is the same convention ``HabitatRobotSim.reset`` now uses to seed
    ``joint_th`` (both take habitat FRONT = -z as forward), so spawn heading and
    tracked headings agree.
    """
    x, y, z, w = (float(c) for c in rotation_xyzw)
    # World-frame forward direction = quaternion applied to habitat local -z.
    forward_x = -(2.0 * (x * z + w * y))
    forward_z = -(1.0 - 2.0 * (x * x + y * y))
    # Control/motion plane is (X = habitat_x, Y = -habitat_z).
    return float(math.atan2(-forward_z, forward_x))


def _build_phases(num_actions, looking_slices):
    """Tile the action timeline ``[0, num_actions)`` with ordered move/look phases.

    ``looking_slices`` are ``[start, end)`` half-open ranges in ACTION-index space
    (ascending, non-overlapping, as returned by
    ``replay_human_demo.find_looking_slices``). Every gap between them -- plus the
    head and tail -- becomes a ``move`` phase. Returns a list of
    ``(kind, start, end)`` tuples, ``kind in {'move', 'look'}``, that exactly
    covers ``[0, num_actions)`` in order.
    """
    phases = []
    cursor = 0
    for start, end in looking_slices:
        if start > cursor:
            phases.append(("move", cursor, start))
        phases.append(("look", start, end))
        cursor = end
    if cursor < num_actions:
        phases.append(("move", cursor, num_actions))
    return phases


def record_segmented_demo(
    robot_sim: HabitatRobotSim,
    path_3d,
    path_headings,
    looking_slices,
    timeout: int = 15,
    fps: int = 30,
    debug_draw: bool = False,
    pure_pursuit_config: Optional[HolonomicControllerConfig] = None,
):
    """Replay a discrete demonstration as a continuous trajectory, switching
    between a path-following controller (while moving) and an in-place rotation
    controller (while "looking": panoramic scans / peeks).

    ------------------------------------------------------------------------
    Index conventions and the action -> path-index conversion (read carefully)
    ------------------------------------------------------------------------
    The arrays are built by ``replay_human_demo.extract_demo`` so that they are
    all mutually aligned:

      * ``path_3d[m]``       -- world position at demo path point ``m``
      * ``path_headings[m]`` -- ``joint_th``-frame yaw at demo path point ``m``
        for ``m = 0 .. M``  (there are ``M + 1`` path points).
      * The action timeline has ``M = len(path_3d) - 1`` entries. Action ``j``
        is exactly the demo action that carries the robot FROM path point ``j``
        TO path point ``j + 1``. ``looking_slices`` are ``[start, end)`` ranges
        in this action space, ``0 <= start < end <= M``.

    Converting an action-index phase ``[p, q)`` to reference data:

      * MOVE phase -> follow the sub-path ``path_3d[p : q + 1]``. Actions
        ``p .. q-1`` span path segments ``(p->p+1), ..., (q-1->q)``, i.e. path
        points ``p`` through ``q`` inclusive; hence the ``q + 1`` slice end.
      * LOOK phase -> rotate in place through ``path_headings[p : q + 1]``. Turns
        and looks do not translate, so path points ``p .. q`` share one position;
        instead of a path we track the heading at the phase's anchor point ``p``
        followed by one heading per look action (points ``p+1 .. q``). The robot
        never moves, only re-orients.

    Because move phases start their sub-path at point ``p`` and the preceding look
    phase ended (position-wise) at that same point ``p``, and vice versa, phases
    chain together with no positional gap -- continuity is carried by the shared
    ``robot_sim`` state, which each controller reads live.
    """
    path_3d = np.asarray(path_3d, dtype=np.float32)
    path_headings = np.asarray(path_headings, dtype=float)
    num_actions = len(path_3d) - 1
    if num_actions < 1:
        return [], []

    phases = _build_phases(num_actions, looking_slices)

    # One controller instance of each type, reconfigured (via reset + set_*) per
    # phase; the robot's live pose provides continuity across the handoffs.
    move_controller = HolonomicRobotController(robot_sim, pure_pursuit_config)
    look_controller = HeadingRotationController(robot_sim)

    robot_sim.sim.navmesh_visualization = debug_draw
    frames = []
    poses = []

    for kind, p, q in phases:
        if kind == "move":
            move_controller.reset()
            move_controller.set_hab_path3(path_3d[p : q + 1])
            local_timeout = int(np.ceil(move_controller.controller._path_distances[-1] / 0.5 + 0.5))
            # print(local_timeout)
            # print(type(local_timeout))
            # print(f"Move phase: {p} -> {q}, path length: {move_controller.controller._path_distances[-1]:.2f}m, timeout: {local_timeout:.1f}s")
        else:
            look_controller.reset()
            look_controller.set_heading_trajectory(
                path_headings[p : q + 1], remove_redundant=True
            )
            local_timeout =  2 * (q - p + 1)  # allow more time for longer look phases
            # print(f"Look phase: {p} -> {q}, num headings: {len(look_controller.headings)}, timeout: {local_timeout:.1f}s")
        for _ in range(local_timeout * fps):
            obs = robot_sim.get_obs()
            if debug_draw:
                robot_sim.draw_axes(path_3d[0], axis_len=1)
                robot_sim.draw_axes(path_3d[-1], axis_len=1)

            if kind == "move":
                step_args = move_controller.step(
                    dt=1.0 / fps, heading_alignment_gain=3.0, max_vw=2.0
                )
                if debug_draw:
                    move_controller.draw_path()
            else:
                step_args = look_controller.step(dt=1.0 / fps)

            robot_sim.step(**step_args, dt=1.0 / fps)

            rgb_frame = obs["color_sensor"][..., :3]
            if debug_draw:
                if kind == "move":
                    panel = move_controller.visualize_bev()
                else:
                    # HeadingRotationController has no BEV; pad with a blank panel
                    # so every frame keeps the same shape for imageio.mimsave.
                    panel = np.zeros(
                        (rgb_frame.shape[0], rgb_frame.shape[0], 3), dtype=rgb_frame.dtype
                    )
                panel = np.array(
                    Image.fromarray(panel).resize((rgb_frame.shape[0], rgb_frame.shape[0]))
                )
                frame = np.hstack([rgb_frame, panel])
            else:
                frame = rgb_frame
            frames.append(frame)
            poses.append(robot_sim.get_2d_pose())

            if kind == "move":
                phase_done = move_controller.is_goal_reached() and robot_sim.is_stationary(
                    lin_vel_threshold=0.01, ang_vel_threshold=0.01
                )
            else:
                phase_done = look_controller.is_trajectory_complete() and robot_sim.is_stationary(
                    lin_vel_threshold=0.01, ang_vel_threshold=0.01
                )
            if phase_done:
                break
            if robot_sim.is_tipped_over(angle_threshold_degrees=45.0):
                # print("Robot tipped over! Stopping simulation.")
                return frames, poses
        if len(frames) >= timeout * fps:
            # print("Timeout reached! Stopping simulation.")
            break
    return frames, poses


def main():
    import imageio
    # 1. Deduce settings from your viewer script
    sim_settings = default_sim_settings.copy()
    sim_settings["scene"] = str(DEFAULT_SCENE_PATH)
    # sim_settings["scene"] = "/home/avery/codes/habitat/data/versioned_data/habitat_test_scenes/skokloster-castle.glb"
    sim_settings["enable_physics"] = True
    sim_settings["window_width"] = 640
    sim_settings["window_height"] = 480
    sim_settings["default_agent"] = 0

    urdf_path = str(DEFAULT_URDF_PATH)

    # 2. Configure the sensor
    color_sensor_spec = habitat_sim.sensor.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.sensor.SensorType.COLOR
    color_sensor_spec.resolution = [sim_settings["window_height"], sim_settings["window_width"]]
    
    # NOTE: You may need to change "base_link" to the specific camera link name in your URDF
    sensor_setup = {"base_camera_link": [color_sensor_spec]}

    # 3. Initialize Simulator
    print("Initializing simulator...")
    robot_sim = HabitatRobotSim(
        sim_settings=sim_settings,
        urdf_path=urdf_path,
        sensor_setup=sensor_setup,
        fixed_base=True
    )
    robot_sim.navmesh_config_and_recompute()
    # robot_sim.sim.navmesh_visualization = True
    import random

    # Seed the simulator's internal RNG with a random integer
    robot_sim.sim.seed(random.randint(1, 999999))
    # Reset and snap to NavMesh
    print("Resetting and snapping to NavMesh...")
    pathfinder = robot_sim.sim.pathfinder

    # 2. Sample a random point if the NavMesh is loaded, otherwise fallback
    if pathfinder.is_loaded:
        initial_position = pathfinder.get_random_navigable_point()
        print(f"Sampled random initial position: {initial_position}")
    else:
        print("Warning: NavMesh not loaded. Falling back to origin.")
        initial_position = np.array([0.0, 0.0, 0.0])
    # initial_position = np.array([1.30228, 0.209919, 15.315])
    
    #goal: [-19.06322098  -2.51216865]
    # 3. Pass it to the reset function
    robot_sim.reset(
        scene_id=sim_settings["scene"],
        robot_translation=initial_position, 
        snap_to_navmesh=True,
        vertical_offset=None,
    )
    # 4. Execute a driving pattern
    print("Driving robot and recording observations...")
    fps = 30
    duration_seconds = 10
    

    import time
    import math
    
    # 1. Sample a random reachable goal and find the shortest path
    path = habitat_sim.ShortestPath()
    
    # We assume the robot was just reset, so its root translation is its exact starting point
    path.requested_start = robot_sim.robot.translation 
    
    path.requested_end = robot_sim.sim.pathfinder.get_random_navigable_point()
    print(f"height difference: {path.requested_start[1]-path.requested_end[1]}")
    # disallow height difference greater than 0.5 meters
    max_height_diff = 0.2
    found = robot_sim.sim.pathfinder.find_path(path) and abs(path.requested_start[1]-path.requested_end[1])<max_height_diff
    for i in range(10):
        if found:
            break
        print("Resampling goal point...")
        path.requested_end = robot_sim.sim.pathfinder.get_random_navigable_point()
        found = robot_sim.sim.pathfinder.find_path(path) and abs(path.requested_start[1]-path.requested_end[1])<max_height_diff
    import time
    t0 = time.perf_counter()     
    if not found:
        print("Path not found! Try another start/end point.")
    else:

        height_diff = path.requested_start[1]-path.requested_end[1]
        print(f"Path found with {len(path.points)} points. Following...")
        
        # 2. Extract the 2D path (Habitat X and Z)
        path_3d = np.array(path.points)
        
        controller = HolonomicRobotController(robot_sim, pure_pursuit_config=None)
        frames,poses = record_path(robot_sim, controller, path_3d, timeout=duration_seconds, fps=fps)
            
    t1 = time.perf_counter()
    print(f"Drove {len(frames)/fps} seconds in simulation. Total frames captured: {len(frames)}")
    print(f"Simulation time: {t1 - t0:.2f} seconds. Speedup: {len(frames)/fps / (t1 - t0):.2f}x real-time.")
    # 5. Save to file
    print(f"Navigation Success: {controller.is_goal_reached()}")
    output_file = "robot_drive_output.mp4"
    print(f"Saving video to {output_file}...")
    imageio.mimsave(output_file, frames, fps=fps)
    print("Done!")


def test_rotation_controller():
    import imageio
    import math
    import random
    import time

    # 1. Deduce settings (mirrors main())
    sim_settings = default_sim_settings.copy()
    sim_settings["scene"] = str(DEFAULT_SCENE_PATH)
    sim_settings["enable_physics"] = True
    sim_settings["window_width"] = 640
    sim_settings["window_height"] = 480
    sim_settings["default_agent"] = 0

    urdf_path = str(DEFAULT_URDF_PATH)

    # 2. Configure the sensor
    color_sensor_spec = habitat_sim.sensor.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.sensor.SensorType.COLOR
    color_sensor_spec.resolution = [sim_settings["window_height"], sim_settings["window_width"]]
    sensor_setup = {"base_camera_link": [color_sensor_spec]}

    # 3. Initialize Simulator
    print("Initializing simulator...")
    robot_sim = HabitatRobotSim(
        sim_settings=sim_settings,
        urdf_path=urdf_path,
        sensor_setup=sensor_setup,
        fixed_base=True,
    )
    robot_sim.navmesh_config_and_recompute()
    robot_sim.sim.seed(random.randint(1, 999999))

    # Reset and snap to a random navigable point
    print("Resetting and snapping to NavMesh...")
    pathfinder = robot_sim.sim.pathfinder
    if pathfinder.is_loaded:
        initial_position = pathfinder.get_random_navigable_point()
        print(f"Sampled random initial position: {initial_position}")
    else:
        print("Warning: NavMesh not loaded. Falling back to origin.")
        initial_position = np.array([0.0, 0.0, 0.0])

    robot_sim.reset(
        scene_id=sim_settings["scene"],
        robot_translation=initial_position,
        snap_to_navmesh=True,
        vertical_offset=None,
    )

    # 4. Build a heading trajectory relative to the robot's current heading and rotate through it
    print("Rotating robot through heading trajectory and recording observations...")
    fps = 30
    duration_seconds = 10

    start_heading = robot_sim.get_joint_state("joint_th")
    # sweep a full turn in both directions, in quarter-turn increments
    heading_trajectory = start_heading + np.array(
        [math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi, math.pi, 0.0]
    )

    controller = HeadingRotationController(robot_sim)
    controller.set_heading_trajectory(heading_trajectory,remove_redundant=True)

    frames = []
    t0 = time.perf_counter()
    for _ in range(duration_seconds * fps):
        obs = robot_sim.get_obs()
        step_args = controller.step(dt=1.0 / fps)
        robot_sim.step(**step_args, dt=1.0 / fps)
        frames.append(obs["color_sensor"][..., :3])
        if controller.is_trajectory_complete() and robot_sim.is_stationary(
            lin_vel_threshold=0.01, ang_vel_threshold=0.01
        ):
            print("Heading trajectory complete!")
            break
        if robot_sim.is_tipped_over(angle_threshold_degrees=45.0):
            print("Robot tipped over! Stopping simulation.")
            break
    t1 = time.perf_counter()

    print(f"Rotated for {len(frames)/fps} seconds in simulation. Total frames captured: {len(frames)}")
    print(f"Simulation time: {t1 - t0:.2f} seconds. Speedup: {len(frames)/fps / (t1 - t0):.2f}x real-time.")
    print(f"Trajectory complete: {controller.is_trajectory_complete()}")

    output_file = "robot_rotation_output.mp4"
    print(f"Saving video to {output_file}...")
    imageio.mimsave(output_file, frames, fps=fps)
    print("Done!")


if __name__ == "__main__":
    test_rotation_controller()