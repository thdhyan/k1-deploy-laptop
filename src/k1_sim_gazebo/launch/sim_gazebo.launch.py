#!/usr/bin/env python3
"""
Full Gazebo sim bringup for K1.
Usage: ros2 launch k1_sim_gazebo sim_gazebo.launch.py robot_ns:=k1_0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    robot_ns = LaunchConfiguration('robot_ns')
    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')

    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0',
            description='Robot namespace (k1_0..k1_5 for fleet)'),
        DeclareLaunchArgument('world', default_value='flat',
            description='World name (flat, rough)'),
        DeclareLaunchArgument('headless', default_value='false'),

        # Launch Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r',
                 PathJoinSubstitution([FindPackageShare('k1_sim_gazebo'), 'worlds', 'flat.sdf'])],
            output='screen',
        ),

        # ros_gz_bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            namespace=robot_ns,
            arguments=['--ros-args', '--params-file',
                       PathJoinSubstitution([FindPackageShare('k1_sim_gazebo'), 'config', 'ros_gz_bridge.yaml'])],
            output='screen',
        ),

        # Include spawn
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([FindPackageShare('k1_sim_gazebo'), 'launch', 'spawn_robot.launch.py'])
            ]),
            launch_arguments={'robot_ns': robot_ns}.items(),
        ),
    ])
