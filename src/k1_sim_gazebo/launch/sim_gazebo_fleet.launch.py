#!/usr/bin/env python3
"""Multi-robot K1 fleet bring-up for Gazebo Harmonic.

Spawns N K1 robots, each fully namespaced under /k1_{i}:
  /k1_{i}/joint_states                     (sensor_msgs/JointState)
  /k1_{i}/joint_commands                   (k1_interfaces/JointCommand -> sim_bridge)
  /k1_{i}/forward_position_controller/commands (std_msgs/Float64MultiArray)
  /k1_{i}/controller_manager               (services)

Usage:
  ros2 launch k1_sim_gazebo sim_gazebo_fleet.launch.py n_robots:=6 gui:=false
"""
import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    n_robots = int(LaunchConfiguration("n_robots").perform(context))
    headless = LaunchConfiguration("gui").perform(context).lower() not in ("true", "1")

    gz_launch = os.path.join(
        get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
    )
    world = os.path.join(
        get_package_share_directory("k1_sim_gazebo"), "worlds", "flat.sdf"
    )
    gen_script = os.path.join(
        get_package_share_directory("k1_description"),
        "scripts",
        "generate_k1_urdf.py",
    )

    # Generate per-robot URDFs up front (xacro + absolute mesh paths)
    urdf_paths = {}
    for i in range(n_robots):
        ns = f"k1_{i}"
        out = subprocess.check_output(
            ["python3", gen_script, "--robot_ns", ns], text=True
        ).strip()
        with open(out) as f:
            urdf_paths[ns] = (out, f.read())

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={
                "gz_args": f"{'-r ' if headless else ''}-v1 {world}"
                + (" -s --headless-rendering" if headless else "")
            }.items(),
        ),
    ]

    for i in range(n_robots):
        ns = f"k1_{i}"
        urdf_path, robot_desc = urdf_paths[ns]

        actions.append(
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=["-file", urdf_path, "-name", ns,
                           "-x", str(1.5 * i), "-z", "0.02"],
                output="screen",
            )
        )
        actions.append(
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=ns,
                parameters=[{
                    "robot_description": robot_desc,
                    "frame_prefix": f"{ns}/",
                    "use_sim_time": True,
                }],
            )
        )

        # SDK-style endpoint: JointCommand -> forward controller commands
        actions.append(
            Node(
                package="k1_control",
                executable="sim_bridge",
                namespace=ns,
                name="sim_bridge",
                parameters=[{"robot_ns": ns}],
                output="screen",
            )
        )

        for controller in ["joint_state_broadcaster", "forward_position_controller"]:
            actions.append(
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=[controller, "--controller-manager", f"/{ns}/controller_manager"],
                    output="screen",
                )
            )

    # Global clock bridge
    actions.append(
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="screen",
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("n_robots", default_value="2"),
            DeclareLaunchArgument("gui", default_value="false"),
            OpaqueFunction(function=_setup),
        ]
    )
