#!/usr/bin/env python3
"""Spawn K1 into running Gazebo world."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node

def generate_launch_description():
    robot_ns = LaunchConfiguration('robot_ns')
    x_pos = LaunchConfiguration('x_pos')
    y_pos = LaunchConfiguration('y_pos')

    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0'),
        DeclareLaunchArgument('x_pos', default_value='0.0'),
        DeclareLaunchArgument('y_pos', default_value='0.0'),
        # Spawn via ros_gz_sim create service
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', robot_ns,
                '-topic', 'robot_description',
                '-x', x_pos, '-y', y_pos, '-z', '1.0',
            ],
            output='screen',
        ),
    ])
