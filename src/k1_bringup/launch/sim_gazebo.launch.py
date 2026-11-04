#!/usr/bin/env python3
"""
Top-level Gazebo bringup.
Usage: ros2 launch k1_bringup sim_gazebo.launch.py robot_ns:=k1_0

Launches:
  1. k1_description: robot_state_publisher
  2. k1_sim_gazebo: Gazebo + ros_gz_bridge + spawn
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

    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0',
            description='Robot namespace — change for multi-robot fleet'),
        DeclareLaunchArgument('policy_path',
            default_value='models/k1_velocity_policy.pt',
            description='Path to TorchScript policy'),

        # 1. Robot description
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_description'), 'launch', 'view_robot.launch.py'])
            ]),
            launch_arguments={'robot_ns': robot_ns}.items(),
        ),

        # 2. Gazebo sim
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_sim_gazebo'), 'launch', 'sim_gazebo.launch.py'])
            ]),
            launch_arguments={'robot_ns': robot_ns}.items(),
        ),
    ])
