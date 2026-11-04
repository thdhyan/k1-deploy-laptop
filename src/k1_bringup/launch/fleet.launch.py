#!/usr/bin/env python3
"""
Multi-robot fleet bringup (Phase 5).
Usage: ros2 launch k1_bringup fleet.launch.py n_robots:=6 sim:=gazebo

Spawns N robots with namespaces k1_0..k1_{N-1}, spaced 2m apart.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def launch_setup(context, *args, **kwargs):
    n = int(LaunchConfiguration('n_robots').perform(context))
    sim = LaunchConfiguration('sim').perform(context)
    actions = []
    for i in range(n):
        # TODO: include bringup launch per robot with robot_ns=k1_{i}, x_pos=2*i
        pass
    return actions

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('n_robots', default_value='1',
            description='Number of robots (1-6)'),
        DeclareLaunchArgument('sim', default_value='gazebo',
            description='Sim backend: gazebo | isaac'),
        OpaqueFunction(function=launch_setup),
    ])
