import math
import warnings
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


def clip_navmesh_world_velocity(
    pathfinder: Any,
    start: np.ndarray,
    world_v_x: float,
    world_v_y: float,
    dt: float,
) -> tuple[float, float, bool, float]:
    """Clip one planar world-frame velocity step to the active navmesh."""
    if dt <= 0 or not getattr(pathfinder, "is_loaded", False):
        return world_v_x, world_v_y, False, 0.0
    start = np.asarray(start, dtype=np.float64)
    if start.shape != (3,) or not np.isfinite(start).all():
        return 0.0, 0.0, True, 0.0
    target = start.copy()
    target[0] += world_v_x * dt
    target[2] -= world_v_y * dt
    try:
        allowed = np.asarray(pathfinder.try_step(start, target), dtype=np.float64)
    except Exception:
        allowed = np.full(3, np.nan, dtype=np.float64)
    if allowed.shape != (3,) or not np.isfinite(allowed).all():
        return 0.0, 0.0, True, float(np.linalg.norm(target[[0, 2]] - start[[0, 2]]))
    requested = target[[0, 2]]
    accepted = allowed[[0, 2]]
    rejected_m = float(np.linalg.norm(requested - accepted))
    if rejected_m <= 1e-6:
        return world_v_x, world_v_y, False, 0.0
    return (
        float((allowed[0] - start[0]) / dt),
        float(-(allowed[2] - start[2]) / dt),
        True,
        rejected_m,
    )


class HabitatRobotSim:
    def __init__(
        self,
        sim_settings: Dict[str, Any],
        urdf_path: str,
        sensor_setup: Dict[str, List[habitat_sim.sensor.SensorSpec]],
        fixed_base: bool = False,
        topdown_sensor_uuid: Optional[str] = None,
        topdown_world_span_m: float = 60.0,
        # Must be a whole multiple/divisor of habitat's fixed Bullet timestep
        # (0.008 s) or each step_world() call spans a variable number of physics
        # steps -- see step() for the full aliasing story. 0.008 makes one
        # substep == exactly one physics step.
        max_substep_dt: float = 0.008,
    ):
        self.urdf_path = urdf_path
        self.sensor_setup = sensor_setup
        self.fixed_base = fixed_base
        self.topdown_sensor_uuid = topdown_sensor_uuid
        self.topdown_world_span_m = float(topdown_world_span_m)
        if self.topdown_sensor_uuid and self.topdown_world_span_m <= 0:
            raise ValueError("topdown_world_span_m must be positive")
        self.navmesh_step_filter = False
        self.navmesh_step_filter_hits = 0
        self.navmesh_step_filter_rejected_m = 0.0
        self.max_substep_dt = max_substep_dt
        self.robot = None
        self.joint_name_to_motor_id = {}
        # Kept so a scene switch can rebuild an equivalent Configuration from
        # scratch -- see _build_cfg / _reconfigure_scene.
        self.sim_settings = dict(sim_settings)
        self.agent_id = self.sim_settings.get("default_agent", 0)
        self.topdown_agent_id = (
            int(self.agent_id) + 1 if self.topdown_sensor_uuid is not None else None
        )
        # Set by navmesh_config_and_recompute(); a scene switch reloads the
        # stock navmesh, so we re-run the recompute only if the caller asked
        # for one in the first place.
        self._navmesh_recomputed = False

        # 1. Flatten sensor_setup and assign to a dummy agent configuration
        self.cfg = self._build_cfg()

        # 2. Initialize simulator
        self.sim = habitat_sim.Simulator(self.cfg)

        # 3. Setup the robot and attach sensors.
        # NOTE: do NOT clear self.robot afterwards. _setup_robot_and_sensors()
        # adds a new articulated object from the URDF whenever self.robot is
        # None, and reset() calls it on every episode -- so blanking the handle
        # here made the first reset() instantiate a SECOND robot and leave the
        # first one in the physics world forever as an orphaned dynamic
        # multibody at the origin. That orphan was the dominant source of
        # order-dependence between episodes (see reset()).
        self._setup_robot_and_sensors()

    def _build_cfg(self, scene_id: Optional[str] = None):
        """Build a habitat_sim ``Configuration`` from the stored sim settings.

        Always returns a FRESH object. That matters for scene switching:
        ``habitat_sim.Simulator.reconfigure`` is a no-op unless
        ``self.config != config``, and the Simulator stores the very object we
        hand it -- so mutating ``self.cfg.sim_cfg.scene_id`` in place mutates
        the thing being compared against and can never trigger a scene reload.
        """
        settings = dict(self.sim_settings)
        if scene_id is not None:
            settings["scene"] = scene_id
        cfg = habitat_sim.utils.settings.make_cfg(settings)
        cfg.agents[self.agent_id].radius = 0.28
        cfg.agents[self.agent_id].height = 1.0
        all_sensor_specs = []
        for specs in self.sensor_setup.values():
            all_sensor_specs.extend(specs)
        cfg.agents[self.agent_id].sensor_specifications = all_sensor_specs
        if self.topdown_sensor_uuid is not None:
            topdown_agent = habitat_sim.agent.AgentConfiguration()
            topdown_sensor = habitat_sim.CameraSensorSpec()
            topdown_sensor.uuid = self.topdown_sensor_uuid
            topdown_sensor.sensor_type = habitat_sim.SensorType.COLOR
            topdown_sensor.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
            topdown_sensor.resolution = [1024, 1024]
            topdown_sensor.position = [0.0, 0.0, 0.0]
            # Habitat's orthographic scale is reciprocal to the world span.
            topdown_sensor.ortho_scale = 1.0 / self.topdown_world_span_m
            topdown_agent.sensor_specifications = [topdown_sensor]
            cfg.agents.append(topdown_agent)
        return cfg

    def _reconfigure_scene(
        self, scene_id: str, scene_dataset_config: Optional[str] = None
    ) -> None:
        """Load a different scene into this simulator.

        Everything scene-scoped dies with the old scene graph and physics
        world, so all three of these have to be rebuilt afterwards:

          * the articulated robot and its joint motors (the old handle is
            dangling -- ``self.robot`` is dropped before re-adding);
          * the sensor -> robot-link parenting (``_config_agents`` re-creates
            the sensor nodes under the agent);
          * the navmesh, because ``_config_pathfinder`` reloads the scene's
            stock (point-agent) navmesh over any robot-sized one.
        """
        if scene_dataset_config is not None:
            self.sim_settings["scene_dataset_config_file"] = scene_dataset_config
        cfg = self._build_cfg(scene_id)
        self.sim.reconfigure(cfg)
        self.cfg = cfg
        self.sim_settings["scene"] = scene_id

        self.robot = None
        self.joint_name_to_motor_id = {}
        self._setup_robot_and_sensors()

        if self._navmesh_recomputed:
            self.navmesh_config_and_recompute()

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
            self.robot.get_link_name(i): i for i in range(self.robot.num_links)
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
                    sensor_node.object.rotation = sensor_node.object.rotation = (
                        mn.Quaternion.rotation(
                            mn.Rad(-math.pi / 2), mn.Vector3(0, 1, 0)
                        )
                        * mn.Quaternion.rotation(
                            mn.Rad(-math.pi / 2), mn.Vector3(0, 0, 1)
                        )
                    )
                    # OPTIONAL rig yaw. A spec may declare `yaw_offset_rad` to
                    # sit at a fixed yaw relative to the robot's forward -- that
                    # is how continuous_demos.panoramic builds an N-camera ring
                    # out of N copies of this one camera. The link frame is the
                    # URDF's (x forward, y left, z up: the rotation above maps
                    # camera-local -z onto link +x and camera-local +y onto link
                    # +z), so the robot's up axis is link +z and a yaw is a
                    # rotation about it, applied in the LINK frame -- hence the
                    # left-multiply. Sign follows the planar control frame: a
                    # positive offset turns the camera anticlockwise seen from
                    # above, so camera heading = joint_th + yaw_offset_rad.
                    #
                    # Read here, off the spec, rather than applied once by
                    # whoever built the rig: this method is re-run on every
                    # reset() and after every scene switch (both of which
                    # destroy and rebuild the sensor nodes), so the rig has to
                    # be re-formed from its declaration each time. A spec
                    # without the attribute -- every spec that predates this --
                    # skips the multiply entirely and is untouched.
                    yaw_offset = float(getattr(spec, "yaw_offset_rad", 0.0))
                    if yaw_offset:
                        sensor_node.object.rotation = (
                            mn.Quaternion.rotation(
                                mn.Rad(yaw_offset), mn.Vector3(0, 0, 1)
                            )
                            * sensor_node.object.rotation
                        )
                    # Ensure the sensor is exactly at the URDF link's origin
                    sensor_node.object.translation = mn.Vector3(0, 0, 0)

    def step(
        self,
        chassis_twist: np.ndarray,
        joint_targets: Optional[Dict[str, float]] = None,
        twist_frame: str = "robot",
        dt: float = 1.0 / 60.0,
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

        TIMESTEP QUANTIZATION -- read before changing ``dt``.
        Habitat/Bullet advances world time only in whole multiples of a fixed
        internal physics timestep (``sim.get_physics_time_step()``, 0.008 s
        here), carrying the sub-step remainder in an internal accumulator. So a
        control tick of duration ``dt`` delivers ``floor(dt/0.008)`` physics
        steps on most ticks and one more whenever the accumulator rolls over.
        Unless ``dt`` is an exact integer multiple of 0.008, the physics time
        actually delivered per tick therefore ALTERNATES between two values,
        and a constant commanded velocity produces a rippling per-tick
        displacement even though the mean is correct.

        Measured: ``dt = 1/30`` (0.03333 s = 4.1667 steps) delivered 0.032 s on
        50/60 ticks and 0.040 s on 10/60 -- a 24% peak-to-peak ripple, which at
        the controller's v_max = 2.0 m/s and max_vw = 2.0 rad/s quantized every
        saturated tick to exactly 0.064 or 0.080 (m and rad alike). Subdividing
        more finely does NOT help: the ripple is a property of ``dt`` vs the
        physics grid, not of ``max_substep_dt`` (measured identical 24.01%
        ripple for max_substep_dt of 1/120, 0.008 and 0.004).

        The only fix is an on-grid ``dt``. Integer frame rates whose period is a
        whole multiple of 0.008 s are 1/(n*0.008) for n = 1, 5, 25, 125, i.e.
        125, 25, 5 and 1 Hz -- so 25 Hz (dt = 0.04 s = 5 physics steps) is the
        only usable recording rate that is exactly on the grid. Hence the 25 Hz
        default in ``generate_pose2d_dataset.py``. A ``dt`` off the grid still
        runs, but warns once.
        """
        # Align substepping to the simulator's fixed physics timestep so that
        # one substep == one physics step and the velocity command written
        # before each substep is integrated for exactly the duration intended.
        fixed_dt = self.sim.get_physics_time_step()
        if not getattr(self, "_warned_off_grid_dt", False):
            n_phys = dt / fixed_dt
            if abs(n_phys - round(n_phys)) > 1e-6:
                self._warned_off_grid_dt = True
                warnings.warn(
                    f"HabitatRobotSim.step(dt={dt!r}) is not an integer multiple "
                    f"of the fixed physics timestep ({fixed_dt}); physics time "
                    f"delivered per tick will alternate between "
                    f"{math.floor(n_phys) * fixed_dt:.4f}s and "
                    f"{math.ceil(n_phys) * fixed_dt:.4f}s, so a constant "
                    f"commanded velocity will show a "
                    f"{100.0 * (math.ceil(n_phys) - math.floor(n_phys)) / n_phys:.0f}% "
                    "per-tick ripple. Use an on-grid rate (e.g. 25 Hz -> "
                    "dt=0.04) to record uniform motion.",
                    RuntimeWarning,
                    stacklevel=2,
                )

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
                heading = (
                    self.get_joint_state("joint_th")
                    if twist_frame == "robot"
                    else float(twist_frame[0])
                    + float(twist_frame[1]) * step_idx * substep_dt
                )

                # Standard 2D rotation matrix
                world_v_x = v_x_cmd * math.cos(heading) - v_y_cmd * math.sin(heading)
                world_v_y = v_x_cmd * math.sin(heading) + v_y_cmd * math.cos(heading)
            else:
                raise ValueError("twist_frame must be 'world' or 'robot'")

            if self.navmesh_step_filter:
                from continuous_demos.robot_pose import robot_navmesh_position

                world_v_x, world_v_y, blocked, rejected_m = clip_navmesh_world_velocity(
                    self.sim.pathfinder,
                    robot_navmesh_position(self),
                    world_v_x,
                    world_v_y,
                    substep_dt,
                )
                if blocked:
                    self.navmesh_step_filter_hits += 1
                    self.navmesh_step_filter_rejected_m += rejected_m

            # Apply to motors
            for joint_name, vel in [
                ("joint_x", world_v_x),
                ("joint_y", world_v_y),
                ("joint_th", omega_cmd),
            ]:
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

    def get_obs(self) -> dict:  # Return standard observation dictionary
        return self.sim.get_sensor_observations(self.agent_id)

    def navmesh_step_filter_summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.navmesh_step_filter),
            "hits": int(self.navmesh_step_filter_hits),
            "rejected_m": float(self.navmesh_step_filter_rejected_m),
        }

    def capture_topdown_rgb(
        self, center_xy: np.ndarray, height_m: float
    ) -> np.ndarray:
        """Capture a static orthographic scene view from the optional debug camera.

        The overhead camera belongs to a second Habitat agent, never to the
        articulated robot.  Moving it therefore cannot perturb physics or the
        first-person observation supplied to the policy.
        """
        if self.topdown_sensor_uuid is None or self.topdown_agent_id is None:
            raise RuntimeError("top-down camera was not enabled when this simulator was built")
        center = np.asarray(center_xy, dtype=np.float32).reshape(2)
        state = habitat_sim.AgentState()
        state.position = np.asarray([center[0], float(height_m), -center[1]], dtype=np.float32)
        state.rotation = np.asarray(
            [-math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)],
            dtype=np.float32,
        )
        self.sim.get_agent(self.topdown_agent_id).set_state(state, reset_sensors=True)
        rgba = self.sim.get_sensor_observations(self.topdown_agent_id)[self.topdown_sensor_uuid]
        return np.ascontiguousarray(np.asarray(rgba)[..., :3], dtype=np.uint8)

    def apply_camera_action(self, action: str, angle_degrees: float = 30.0) -> None:
        """Apply one Habitat-style LOOK_UP/LOOK_DOWN action to every camera."""
        action = str(action).upper()
        if action not in {"LOOK_UP", "LOOK_DOWN"}:
            raise ValueError(f"unsupported camera action {action!r}")
        angle = float(angle_degrees) if action == "LOOK_UP" else -float(angle_degrees)
        agent_node = self.sim.get_agent(self.agent_id).scene_node
        for specs in self.sensor_setup.values():
            for spec in specs:
                sensor_node = agent_node.subtree_sensors.get(spec.uuid)
                if sensor_node is not None:
                    sensor_node.object.rotate_x_local(mn.Deg(angle))
                    sensor_node.object.rotation = (
                        sensor_node.object.rotation.normalized()
                    )

    # Height (metres) the robot is parked at while its collision caches are
    # flushed. Habitat's Bullet world uses btDbvtBroadphase (a dynamic AABB
    # tree, no world bounds), so any coordinate is legal; 10 km up is outside
    # every habitat scene's AABB, which is what makes the broadphase drop the
    # robot's overlapping pairs.
    _PARK_HEIGHT_M = 1.0e4

    def _flush_physics_contact_state(self) -> None:
        """Drop Bullet's cached collision state for the robot.

        ``habitat_sim.Simulator.reset()`` restores articulated-object
        *kinematics* -- joint positions, velocities and forces, and the root
        transform -- but it does not touch Bullet's collision caches. The
        broadphase overlapping-pair cache and the ``btPersistentManifold`` for
        each pair (its cached contact points, their slot ordering, and the
        warm-start impulses on them) survive the teleport into the next
        episode. Measured on MP3D 17DRP5sb8fy: 6 overlapping pairs / 2 active
        contact points at the end of an episode, still 6 / 2 immediately after
        ``sim.reset()``.

        That residue does not move the robot at spawn -- the first ticks of the
        next episode are bit-identical either way -- but it changes the order
        and the warm-start seeds of the contact constraints the solver sees,
        so as soon as the robot touches geometry the Gauss-Seidel solve lands
        on a slightly different answer (first observed divergence ~1e-7 rad/s
        in ``joint_th``) and the trajectory then diverges chaotically. That is
        how a stateless, image-blind policy could run the same episode two
        different ways depending on what preceded it.

        Bullet exposes no "clear the caches" call through habitat's bindings,
        but the broadphase drops a pair as soon as the two AABBs stop
        overlapping. So: park the robot far outside the scene, run one
        collision pass (pairs go to 0), and put it back. The pairs are then
        rebuilt from scratch on the next ``step_world``, exactly as they are in
        a freshly constructed simulator. Costs ~1 ms.
        """
        if self.robot is None:
            return
        here = self.robot.translation
        self.robot.translation = mn.Vector3(0.0, self._PARK_HEIGHT_M, 0.0)
        self.sim.perform_discrete_collision_detection()
        self.robot.translation = here
        # Rebuild the pairs at the pose the episode actually starts from, so
        # the collision state the first step_world() sees is a function of the
        # spawn pose alone.
        self.sim.perform_discrete_collision_detection()

    # Tried and rejected: destroying and re-creating every joint motor at reset,
    # to drop the impulse Bullet warm-starts each btMultiBodyJointMotor from.
    # remove_joint_motor()/create_joint_motor() do not round-trip faithfully --
    # with them in place the recorded trajectory of a replay differed from tick
    # ZERO (up to 0.005 m), i.e. the rebuild perturbs the spawn itself, where
    # the leak it targets only shows up later. Net worse, so not done.

    def reset(
        self,
        scene_id: Optional[str] = None,
        robot_translation: Optional[np.ndarray] = None,
        robot_rotation: Optional[
            Any
        ] = None,  # Accepts mn.Quaternion or [x, y, z, w] array
        snap_to_navmesh: bool = True,
        vertical_offset: Optional[float] = None,  # Manual override if AABB is weird
        home_joint_positions: Optional[Dict[str, float]] = None,
        scene_dataset_config: Optional[str] = None,
    ) -> dict:
        """Return the simulator to a start-of-episode state.

        CONTRACT: after reset(), the outcome of an episode must not depend on
        what ran before it in this simulator. That is what makes it legitimate
        to reuse one simulator across every episode of a scene, as
        ``generate_pose2d_dataset.py`` and the evaluation harnesses do. It was
        previously violated, in two independent ways, and the violation was
        measurable: replaying one demonstration after three other episodes
        moved its recorded path by up to 0.040 m relative to replaying it
        alone. Steps 1b and 5 below are what restore the contract; do not drop
        either of them for speed. They cost ~1 ms per episode, against ~2 s to
        rebuild the simulator instead.

        It is not a complete guarantee. What survives is the order in which
        Bullet's broadphase reports the robot's contacts, which no habitat
        binding can reset -- see dump/sim_reset_fix/FINDINGS.md and
        tests/test_sim_reset_isolation.py.
        """
        self.navmesh_step_filter_hits = 0
        self.navmesh_step_filter_rejected_m = 0.0
        # 1. Handle Scene Changes and Graph Rebuild
        if scene_id is not None and scene_id != self.cfg.sim_cfg.scene_id:
            self._reconfigure_scene(scene_id, scene_dataset_config)
        else:
            self.sim.reset()
            self._setup_robot_and_sensors()

        # 1b. Re-home the robot's own dynamic state explicitly. sim.reset()
        # already zeroes joint positions/velocities/forces -- but only via
        # resetStateFromSceneInstanceAttr(), which silently no-ops for an
        # articulated object that has no scene-instance attributes. Doing it
        # here makes reset() self-sufficient, and clear_joint_states() also
        # clears the multibody's constraint forces and link velocities (which
        # nothing else does) and handles spherical joints correctly. This is
        # what re-homes the passive suspension joints joint_z/pitch/roll.
        if self.robot is not None:
            self.robot.clear_joint_states()
            self.robot.root_linear_velocity = mn.Vector3(0.0, 0.0, 0.0)
            self.robot.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
            # Bullet's deactivation ("sleep") timers are the other thing that
            # survives sim.reset(). btMultiBody accumulates m_sleepTimer while
            # the body is nearly still and sleeps once it passes m_sleepTimeout;
            # each link collider likewise accumulates m_deactivationTime. An
            # episode that ends stationary -- which is exactly what a
            # goal-reached replay does -- therefore hands the next episode a
            # part-elapsed timer, and the robot falls asleep sooner than it
            # would in a fresh simulator. Nothing re-arms them: the kinematic
            # setters above only wake the body when it is ALREADY asleep, so a
            # merely drowsy body keeps its timer. Assigning awake unconditionally
            # calls setActive(true), which zeroes both. (Measured as a no-op on
            # the replay workload -- the demos never idle long enough to bank a
            # sleep timer -- but it is genuinely leaked state and it is free.)
            self.robot.awake = True

        # 2. Seed the base yaw (joint_th) from the requested orientation.
        # Goal: joint_th = the requested facing as a heading in the planar control
        # frame (frame 3 in the header): theta CCW from +X, X=habitat_x, Y=-habitat_z.
        if robot_rotation is not None:
            if isinstance(robot_rotation, (np.ndarray, list, tuple)):
                if len(robot_rotation) != 4:
                    raise ValueError(
                        "robot_rotation array must have 4 elements: [x, y, z, w]"
                    )
                # Habitat agent quaternion [x, y, z, w]. Habitat is x=right, y=up,
                # z=back, so the agent's FORWARD is -z (habitat_sim.geo.FRONT), NOT
                # +x. Remap the quaternion into the magnum/planar frame ((hx,hy,hz)
                # -> (hx,-hz,hy)); under that same remap habitat FRONT=(0,0,-1) maps
                # to +y, so the agent's forward axis is +y here (not +x).
                x, y, z, w = robot_rotation
                robot_rotation = mn.Quaternion(mn.Vector3(x, -z, y), w)
                forward_vec = np.array(
                    [0.0, 1.0, 0.0]
                )  # habitat FRONT (-z) in magnum frame
            else:
                # Already a magnum quaternion in the planar frame; its +x is forward.
                forward_vec = np.array([1.0, 0.0, 0.0])
            if home_joint_positions is None:
                home_joint_positions = {}
            rotated_forward = robot_rotation.transform_vector(forward_vec)
            # Heading in the control frame X=habitat_x, Y=-habitat_z (== hab_quat_to_heading).
            home_joint_positions["joint_th"] = float(
                np.arctan2(rotated_forward[1], rotated_forward[0])
            )
        self.robot.rotation = self.base_rotation_offset  # TODO: allow non aligned base.

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
                    joint_positions[pos_offset] = pos_target
            self.robot.joint_positions = joint_positions

        # 5. Purge the physics residue of the previous episode. This runs LAST,
        # once the spawn pose is final, so the collision state handed to the
        # first step_world() is a function of the spawn pose and nothing else.
        # See _flush_physics_contact_state().
        self._flush_physics_contact_state()

        # # Update the motor target so it doesn't try to snap back
        # settings = self.robot.get_joint_motor_settings(m_id)
        # settings.position_target = pos_target
        # self.robot.update_joint_motor(m_id, settings)
        # self.robot.awake = True

    # TODO: fix this
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

    def get_2d_pose_and_twist(
        self, virtual_heading=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pose [X, Y, theta] and twist [v_forward, w] in the planar control frame.

        The world position is X = habitat_x, Y = -habitat_z, assembled as the
        prismatic joint value plus the base translation on each axis
        (jx + habitat_x, jy - habitat_z). theta = joint_th, CCW from +X. Passing
        virtual_heading substitutes the controller's virtual heading for joint_th
        (used by pure-pursuit path tracking).
        """
        jx = self.get_joint_state("joint_x")
        jy = self.get_joint_state("joint_y")
        jth = (
            self.get_joint_state("joint_th")
            if virtual_heading is None
            else virtual_heading
        )  # Adjust for URDF orientation

        # Retrieve current velocity
        vx = self.get_joint_state("joint_x", is_vel=True)
        vy = self.get_joint_state("joint_y", is_vel=True)
        vth = self.get_joint_state("joint_th", is_vel=True)

        # rx = habitat_x, rz = habitat_z of the base; planar Y uses -habitat_z.
        rz, rx = self.robot.translation[2], self.robot.translation[0]
        v_forward = vx * math.cos(jth) + vy * math.sin(jth)

        # limit jth to [-pi, pi]
        jth = (jth + math.pi) % (2 * math.pi) - math.pi
        return np.array([jx + rx, jy - rz, jth]), np.array([v_forward, vth])

    def get_2d_pose(self) -> np.ndarray:
        """Planar pose [X, Y, theta] with X=habitat_x, Y=-habitat_z, theta=joint_th."""
        jx = self.get_joint_state("joint_x")
        jy = self.get_joint_state("joint_y")
        jth = self.get_joint_state("joint_th")

        jth = (jth + math.pi) % (2 * math.pi) - math.pi
        rz, rx = self.robot.translation[2], self.robot.translation[0]
        return np.array([jx + rx, jy - rz, jth])

    def is_stationary(self, lin_vel_threshold, ang_vel_threshold) -> bool:
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
        return abs(roll) > math.radians(angle_threshold_degrees) or abs(
            pitch
        ) > math.radians(angle_threshold_degrees)

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
        # Remembered so a later scene switch re-applies it: reconfigure() loads
        # the scene's stock navmesh, which is cut for a point agent, not for
        # this robot's radius/height.
        self._navmesh_recomputed = True

    def draw_axes(self, translation, axis_len=1.0):
        lr = self.sim.get_debug_line_render()
        lr.set_line_width(5)

        opacity = 1.0
        red = mn.Color4(1.0, 0.0, 0.0, opacity)
        green = mn.Color4(0.0, 1.0, 0.0, opacity)
        blue = mn.Color4(0.0, 0.0, 1.0, opacity)
        white = mn.Color4(1.0, 1.0, 1.0, opacity)
        # draw axes with x+ = red, y+ = green, z+ = blue
        lr.draw_transformed_line(
            translation, mn.Vector3(axis_len, 0, 0) + translation, red
        )
        lr.draw_transformed_line(
            translation, mn.Vector3(0, axis_len, 0) + translation, green
        )
        lr.draw_transformed_line(
            translation, mn.Vector3(0, 0, axis_len) + translation, blue
        )

    def draw_path(self, path_3d, color):
        lr = self.sim.get_debug_line_render()
        lr.set_line_width(5)
        opacity = 1.0
        offset = mn.Vector3(0, 0.2, 0)
        for i in range(len(path_3d) - 1):
            lr.draw_transformed_line(
                path_3d[i] + offset, path_3d[i + 1] + offset, color
            )
