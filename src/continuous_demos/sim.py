import math
import numpy as np
import habitat_sim
from typing import Dict, List, Any, Optional
import magnum as mn

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

