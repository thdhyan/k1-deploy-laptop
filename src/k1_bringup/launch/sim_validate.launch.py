#!/usr/bin/env python3
"""
Unified Simulation Validation Launch — Booster K1

Single entry point to validate a trained policy across all three sim backends:
  - Gazebo Harmonic (CPU, ROS2-native)
  - Isaac Sim 6.0.1 (GPU, photorealistic)
  - MuJoCo (CPU, lightweight, fastest iteration)

Usage:
  # Gazebo (default)
  ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo policy:=models/k1_velocity_policy.pt

  # Isaac Sim (requires Isaac Lab venv)
  ros2 launch k1_bringup sim_validate.launch.py backend:=isaac policy:=models/k1_velocity_policy.pt usd_path:=/path/to/k1.usd

  # MuJoCo (standalone script, no ROS launch needed)
  python3 src/k1_sim_gazebo/scripts/mujoco_fleet_node.py --n_robots 1 &
  ros2 run k1_locomotion locomotion_node --ros-args -p robot_ns:=k1_0 -p policy_path:=models/k1_velocity_policy.pt

  # Multi-robot fleet validation (Gazebo)
  ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo n_robots:=3

  # With keyboard teleop
  ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo teleop:=true

Validation checklist (all backends):
  [ ] Robot stands stably (no drift, no collapse)
  [ ] Forward walk: vx=0.5 → robot moves forward ~0.5 m/s
  [ ] Lateral walk: vy=0.3 → robot moves laterally
  [ ] Yaw turn: wz=0.5 → robot rotates in place
  [ ] Combined: vx=0.3, wz=0.3 → curved trajectory
  [ ] Stop command → robot halts smoothly
  [ ] Policy obs published on /k1_0/policy_obs (debug)
  [ ] Joint commands within limits (check /k1_0/joint_commands)
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    ExecuteProcess,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
import os


def launch_setup(context, *args, **kwargs):
    backend = LaunchConfiguration('backend').perform(context)
    robot_ns = LaunchConfiguration('robot_ns').perform(context)
    policy_path = LaunchConfiguration('policy_path').perform(context)
    n_robots = int(LaunchConfiguration('n_robots').perform(context))
    headless = LaunchConfiguration('headless').perform(context) == 'true'
    teleop = LaunchConfiguration('teleop').perform(context) == 'true'
    usd_path = LaunchConfiguration('usd_path').perform(context)
    world = LaunchConfiguration('world').perform(context)

    actions = []

    # Log configuration
    actions.append(LogInfo(msg=f"[sim_validate] Backend: {backend}"))
    actions.append(LogInfo(msg=f"[sim_validate] Robot namespace: {robot_ns}"))
    actions.append(LogInfo(msg=f"[sim_validate] Policy: {policy_path}"))
    actions.append(LogInfo(msg=f"[sim_validate] Robots: {n_robots}"))
    actions.append(LogInfo(msg=f"[sim_validate] Headless: {headless}"))

    if backend == 'gazebo':
        actions.extend(_gazebo_bringup(context, robot_ns, policy_path, n_robots, headless, world, teleop))
    elif backend == 'isaac':
        actions.extend(_isaac_bringup(context, robot_ns, policy_path, n_robots, headless, usd_path, teleop))
    elif backend == 'mujoco':
        actions.extend(_mujoco_bringup(context, robot_ns, policy_path, n_robots, teleop))
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose: gazebo, isaac, mujoco")

    return actions


def _gazebo_bringup(context, robot_ns, policy_path, n_robots, headless, world, teleop):
    actions = []

    if n_robots > 1:
        # Fleet: include fleet launch with per-robot namespaces
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_sim_gazebo'), 'launch', 'sim_gazebo_fleet.launch.py'])
            ]),
            launch_arguments={
                'n_robots': str(n_robots),
                'headless': 'true' if headless else 'false',
                'world': world,
            }.items(),
        ))
        # Add locomotion nodes for each robot
        for i in range(n_robots):
            ns = f"k1_{i}"
            actions.append(_locomotion_node(ns, policy_path, 'k1_interfaces/JointCommand'))
    else:
        # Single robot
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_bringup'), 'launch', 'sim_gazebo.launch.py'])
            ]),
            launch_arguments={
                'robot_ns': robot_ns,
                'headless': 'true' if headless else 'false',
                'world': world,
                'policy_path': policy_path,
            }.items(),
        ))

    if teleop:
        actions.append(_teleop_node(robot_ns if n_robots == 1 else 'k1_0'))

    return actions


def _isaac_bringup(context, robot_ns, policy_path, n_robots, headless, usd_path, teleop):
    actions = []

    # Isaac Sim fleet script (handles N robots internally)
    isaac_python = os.path.expanduser('~/Projects/IsaacLab/.venv-isaac/bin/python3.12')
    fleet_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'src', 'k1_sim_isaac', 'scripts', 'fleet_sim.py'
    )

    if n_robots > 1:
        actions.append(ExecuteProcess(
            cmd=[
                isaac_python, fleet_script,
                '--n_robots', str(n_robots),
                '--headless' if headless else '',
            ],
            output='screen',
            additional_env={'ROS_DOMAIN_ID': '77'},
        ))
    else:
        # Single robot: use isaac_standalone + sim_bridge + locomotion
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_bringup'), 'launch', 'sim_isaac.launch.py'])
            ]),
            launch_arguments={
                'robot_ns': robot_ns,
                'headless': 'true' if headless else 'false',
                'usd_path': usd_path,
            }.items(),
        ))

    # Locomotion nodes (Isaac uses sensor_msgs/JointState for commands)
    if n_robots > 1:
        for i in range(n_robots):
            ns = f"k1_{i}"
            actions.append(_locomotion_node(ns, policy_path, 'sensor_msgs/JointState'))
    else:
        actions.append(_locomotion_node(robot_ns, policy_path, 'sensor_msgs/JointState'))

    if teleop:
        actions.append(_teleop_node(robot_ns if n_robots == 1 else 'k1_0'))

    return actions


def _mujoco_bringup(context, robot_ns, policy_path, n_robots, teleop):
    """MuJoCo fleet runs as a standalone Python script, not ROS2 launch.
    This function returns instructions + the locomotion nodes."""
    actions = []

    # Print instructions (user runs the fleet script manually)
    actions.append(LogInfo(msg=""))
    actions.append(LogInfo(msg="=" * 60))
    actions.append(LogInfo(msg="[MuJoCo Backend] Run the fleet script in a SEPARATE terminal:"))
    actions.append(LogInfo(msg=""))
    actions.append(LogInfo(msg=f"  source /opt/ros/jazzy/setup.bash && source install/setup.bash"))
    actions.append(LogInfo(msg=f"  python3 src/k1_sim_gazebo/scripts/mujoco_fleet_node.py --n_robots {n_robots} --rate 200"))
    actions.append(LogInfo(msg=""))
    actions.append(LogInfo(msg="Then run this launch file (it starts locomotion nodes only):"))
    actions.append(LogInfo(msg="=" * 60))
    actions.append(LogInfo(msg=""))

    # Locomotion nodes
    if n_robots > 1:
        for i in range(n_robots):
            ns = f"k1_{i}"
            actions.append(_locomotion_node(ns, policy_path, 'k1_interfaces/JointCommand'))
    else:
        actions.append(_locomotion_node(robot_ns, policy_path, 'k1_interfaces/JointCommand'))

    if teleop:
        actions.append(_teleop_node(robot_ns if n_robots == 1 else 'k1_0'))

    return actions


def _locomotion_node(ns, policy_path, command_type):
    """Create a locomotion_node for the given namespace."""
    from launch_ros.actions import Node
    return Node(
        package='k1_locomotion',
        executable='locomotion_node',
        namespace=ns,
        name='locomotion_node',
        output='screen',
        parameters=[{
            'robot_ns': ns,
            'policy_path': policy_path,
            'control_freq': 50.0,
            'action_scale': 0.25,
            'command_type': command_type,
            'cmd_timeout': 0.5,
            'js_timeout': 0.5,
            'imu_topic': 'imu',
            'odom_topic': 'odom',
            'history_len': 10,
            'input_mode': 'latest',
            'publish_obs_debug': True,
        }],
    )


def _teleop_node(ns):
    """Keyboard teleop for velocity commands."""
    from launch_ros.actions import Node
    return Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        remappings=[('/cmd_vel', f'/{ns}/cmd_vel')],
        prefix='xterm -e',  # opens in new terminal
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='gazebo',
            description='Sim backend: gazebo | isaac | mujoco',
            choices=['gazebo', 'isaac', 'mujoco']),
        DeclareLaunchArgument('robot_ns', default_value='k1_0',
            description='Robot namespace (single robot mode)'),
        DeclareLaunchArgument('policy_path', default_value='models/k1_velocity_policy.pt',
            description='Path to TorchScript policy (relative to workspace root or absolute)'),
        DeclareLaunchArgument('n_robots', default_value='1',
            description='Number of robots for fleet validation (1-6)'),
        DeclareLaunchArgument('headless', default_value='false',
            description='Run without visualizer'),
        DeclareLaunchArgument('teleop', default_value='false',
            description='Launch keyboard teleop (teleop_twist_keyboard)'),
        DeclareLaunchArgument('usd_path', default_value='',
            description='Path to K1 USD file (Isaac Sim only)'),
        DeclareLaunchArgument('world', default_value='flat',
            description='Gazebo world: flat | rough'),

        OpaqueFunction(function=launch_setup),
    ])