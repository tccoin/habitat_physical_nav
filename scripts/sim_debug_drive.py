"""Manual/debug driving scripts for ``HabitatRobotSim`` -- not part of the
documented pipeline (see ``CLAUDE.md``). ``main()`` drives a random
shortest-path goal with the pure-pursuit path controller;
``test_rotation_controller()`` sweeps the heading-rotation controller through
a fixed trajectory. Both save a debug MP4. Split out of ``sim.py``, which
should only contain the simulator itself, not runnable scripts.

Run directly: ``python scripts/sim_debug_drive.py`` (runs
``test_rotation_controller()``, per the ``if __name__`` guard below).
"""

from pathlib import Path

import numpy as np
import habitat_sim
from habitat_sim.utils.settings import default_sim_settings

from continuous_demos import ASSETS_DIR
from continuous_demos.sim import HabitatRobotSim
from continuous_demos.controllers import HolonomicRobotController, HeadingRotationController
from continuous_demos.replay import record_path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_URDF_PATH = (
    ASSETS_DIR / "tidybot_description" / "urdf" / "tidybot_base_minimal.urdf"
)
# Legacy dev-convenience default (see the equivalent note that used to sit next
# to this constant in sim.py): assumes this repo is checked out standalone one
# level inside a parent habitat checkout, i.e. two levels above this script.
DEFAULT_SCENE_PATH = (
    PACKAGE_DIR.parents[1]
    / "data"
    / "scene_datasets"
    / "mp3d_example"
    / "17DRP5sb8fy"
    / "17DRP5sb8fy.glb"
)

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