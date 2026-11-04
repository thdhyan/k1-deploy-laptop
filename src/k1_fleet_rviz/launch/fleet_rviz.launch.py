#!/usr/bin/env python3
"""
Fleet RViz Launch — 18 camera streams + TF + RobotModel for 6x Booster K1.

Architecture:
  - Everything runs in ROS_DOMAIN_ID=0 (laptop domain)
  - domain_relay nodes each spawn a child process in the robot's domain
  - Child subscribes to camera topics, parent republishes in domain 0
  - robot_state_publisher loads K1 URDF for TF visualization

Usage:
    ros2 launch k1_fleet_rviz fleet_rviz.launch.py
    ros2 launch k1_fleet_rviz fleet_rviz.launch.py robots:="k1_0,k1_1,k1_2"
"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction


# ── Fleet definition ──────────────────────────────────────────────────
# All robots publish in domain 0 (default). Each relay child also runs in
# domain 0 and subscribes to the same topic names. The parent republishes
# under /{robot_ns}/... for namespace isolation in RViz.
ROBOTS = {
    'k1_0': {'alias': 'A1', 'ip': '10.86.36.123'},
    'k1_1': {'alias': 'A2', 'ip': '10.86.36.3'},
    'k1_2': {'alias': 'A3', 'ip': '10.86.36.18'},
    'k1_3': {'alias': 'B1', 'ip': '10.86.36.108'},
    'k1_4': {'alias': 'B2', 'ip': '10.86.36.109'},
    'k1_5': {'alias': 'B3', 'ip': '10.86.36.248'},
}

URDF_PATH = os.path.expanduser(
    '~/Projects/booster_ws/src/k1_description/assets/K1_22dof.urdf'
)

RVIZ_CONFIG = os.path.expanduser(
    '~/Projects/booster_ws/install/k1_fleet_rviz/share/k1_fleet_rviz/config/fleet_rviz.rviz'
)
# Fallback to source if install not yet available
if not os.path.exists(RVIZ_CONFIG):
    RVIZ_CONFIG = os.path.expanduser(
        '~/Projects/booster_ws/src/k1_fleet_rviz/config/fleet_rviz.rviz'
    )


def _read_urdf():
    with open(URDF_PATH, 'r') as f:
        return f.read()


def _make_pelvis_urdf(urdf_content):
    """Add pelvis virtual root link above trunk_link for TF base frame."""
    pelvis_urdf = urdf_content.replace(
        '<link name="trunk_link">',
        '<link name="pelvis"/>\n  <link name="trunk_link">'
    )
    # Insert pelvis_to_trunk fixed joint before first <joint name=
    marker = '<joint name='
    idx = pelvis_urdf.find(marker)
    if idx > 0:
        pelvis_joint = (
            '  <joint name="pelvis_to_trunk" type="fixed">\n'
            '    <parent link="pelvis"/>\n'
            '    <child link="trunk_link"/>\n'
            '    <origin xyz="0 0 0" rpy="0 0 0"/>\n'
            '  </joint>\n\n'
        )
        pelvis_urdf = pelvis_urdf[:idx] + pelvis_joint + pelvis_urdf[idx:]
    return pelvis_urdf


def launch_setup(context, *args, **kwargs):
    robots_str = LaunchConfiguration('robots').perform(context)
    active = [r.strip() for r in robots_str.split(',') if r.strip()]
    if not active:
        active = list(ROBOTS.keys())

    actions = []

    # ── 1. Domain relay nodes (one process per robot) ────────────────
    relay_script = os.path.expanduser(
        '~/Projects/booster_ws/install/k1_fleet_rviz/lib/k1_fleet_rviz/domain_relay'
    )
    if not os.path.exists(relay_script):
        relay_script = os.path.expanduser(
            '~/Projects/booster_ws/src/k1_fleet_rviz/k1_fleet_rviz/domain_relay.py'
        )

    for ns, cfg in ROBOTS.items():
        if ns not in active:
            continue

        relay_proc = ExecuteProcess(
            cmd=[
                'python3', relay_script,
                '--ros-args',
                '-p', f'robot_ns:={ns}',
                '-p', 'source_domain:=0',  # all robots publish in domain 0
            ],
            output='screen',
            name=f'relay_{ns}',
        )
        actions.append(relay_proc)

    # ── 2. Robot state publishers ────────────────────────────────────
    pelvis_urdf = _make_pelvis_urdf(_read_urdf())

    for ns in active:
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=ns,
            output='screen',
            parameters=[{
                'robot_description': pelvis_urdf,
            }],
        )
        actions.append(rsp)

    # ── 3. RViz2 ─────────────────────────────────────────────────────
    rviz_node = ExecuteProcess(
        cmd=['rviz2', '-d', RVIZ_CONFIG],
        output='screen',
        name='rviz2',
    )
    actions.append(rviz_node)

    actions.append(LogInfo(msg=[
        'Fleet RViz: ', robots_str
    ]))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robots',
            default_value=','.join(ROBOTS.keys()),
            description='Comma-separated robot namespaces',
        ),
        OpaqueFunction(function=launch_setup),
    ])
