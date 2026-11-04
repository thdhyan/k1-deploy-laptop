#!/usr/bin/env python3
"""
Top-level Isaac Sim bringup.
Usage: ros2 launch k1_bringup sim_isaac.launch.py robot_ns:=k1_0

Launches:
  1. k1_description: robot_state_publisher
  2. k1_sim_isaac: Isaac Sim + ROS2 bridge
  3. k1_control: sim_bridge_node
  4. k1_locomotion: locomotion_node (policy inference)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    robot_ns = LaunchConfiguration('robot_ns')
    policy_path = LaunchConfiguration('policy_path')
    usd_path = LaunchConfiguration('usd_path')

    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0',
            description='Robot namespace — change for multi-robot fleet'),
        DeclareLaunchArgument('policy_path',
            default_value='models/k1_velocity_policy.pt',
            description='Path to TorchScript policy'),
        DeclareLaunchArgument('usd_path', default_value='',
            description='Path to K1 USD file (from booster_assets)'),

        # 1. Robot description
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_description'), 'launch', 'view_robot.launch.py'])
            ]),
            launch_arguments={'robot_ns': robot_ns}.items(),
        ),

        # 2. Isaac Sim
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_sim_isaac'), 'launch', 'sim_isaac.launch.py'])
            ]),
            launch_arguments={
                'robot_ns': robot_ns,
                'usd_path': usd_path,
            }.items(),
        ),
    ])
