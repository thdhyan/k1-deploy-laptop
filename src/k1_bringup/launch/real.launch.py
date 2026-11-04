#!/usr/bin/env python3
"""
Real robot bringup — Booster K1 via booster_robotics_sdk.

Usage:
  ros2 launch k1_bringup real.launch.py robot_ns:=k1_0 sdk_ip:=192.168.1.100

Prerequisites:
  - Robot powered on, in safe area (≥ 3m × 3m clear)
  - Booster Robotics SDK running on robot (firmware ≥ v1.2.0)
  - Laptop ↔ robot WiFi connectivity (ping < 10ms)
  - CycloneDDS configured for WiFi interface (see guide_real.md)

Launches:
  1. k1_description: robot_state_publisher (URDF)
  2. k1_control: sdk_bridge_node (ROS2 ↔ SDK bridge)
  3. k1_locomotion: locomotion_node (policy inference)

NOTE: sdk_bridge_node.py is currently a STUB — you must implement
the booster_robotics_sdk integration before this works on hardware.
See src/k1_control/k1_control/sdk_bridge_node.py and
sdk/booster_robotics_sdk/example/low_level/ for reference.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    robot_ns = LaunchConfiguration('robot_ns')
    sdk_ip = LaunchConfiguration('sdk_ip')
    policy_path = LaunchConfiguration('policy_path')
    use_wbc = LaunchConfiguration('use_wbc')
    control_freq = LaunchConfiguration('control_freq')

    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0',
            description='Robot namespace (k1_0..k1_5 for fleet)'),
        DeclareLaunchArgument('sdk_ip', default_value='192.168.1.100',
            description='Robot SDK IP address (on WiFi)'),
        DeclareLaunchArgument('policy_path', default_value='models/k1_velocity_policy.pt',
            description='Path to TorchScript locomotion policy'),
        DeclareLaunchArgument('use_wbc', default_value='false',
            description='Enable whole-body controller (k1_wbc)'),
        DeclareLaunchArgument('control_freq', default_value='50.0',
            description='Control loop frequency (Hz)'),

        # Safety warning
        LogInfo(msg=""),
        LogInfo(msg="=" * 70),
        LogInfo(msg="⚠️  REAL ROBOT MODE — SAFETY CHECKLIST"),
        LogInfo(msg="  [ ] Robot in safe, open area (≥ 3m × 3m)"),
        LogInfo(msg="  [ ] E-stop accessible and tested"),
        LogInfo(msg="  [ ] Battery > 30%"),
        LogInfo(msg="  [ ] WiFi latency < 10ms (ping robot IP)"),
        LogInfo(msg="  [ ] Policy validated in sim (Gazebo/Isaac/MuJoCo)"),
        LogInfo(msg="  [ ] First cmd_vel will be ZERO — verify before enabling"),
        LogInfo(msg="=" * 70),
        LogInfo(msg=""),

        # 1. Robot description (URDF + robot_state_publisher)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_description'), 'launch', 'view_robot.launch.py'])
            ]),
            launch_arguments={'robot_ns': robot_ns}.items(),
        ),

        # 2. SDK bridge (ROS2 JointCommand → booster_robotics_sdk → real robot)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_control'), 'launch', 'control.launch.py'])
            ]),
            launch_arguments={
                'robot_ns': robot_ns,
                'sdk_ip': sdk_ip,
                'mode': 'real',  # tells control.launch.py to use sdk_bridge_node
                'control_freq': control_freq,
            }.items(),
        ),

        # 3. Locomotion policy node
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_locomotion'), 'launch', 'locomotion.launch.py'])
            ]),
            launch_arguments={
                'robot_ns': robot_ns,
                'policy_path': policy_path,
                'control_freq': control_freq,
                'command_type': 'k1_interfaces/JointCommand',
                'imu_topic': 'imu',
                'odom_topic': 'odom',
                'input_mode': 'latest',
            }.items(),
        ),

        # 4. Optional: Whole-body controller
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_wbc'), 'launch', 'wbc.launch.py'])
            ]),
            launch_arguments={'robot_ns': robot_ns}.items(),
            condition=IfCondition(use_wbc),
        ),
    ])