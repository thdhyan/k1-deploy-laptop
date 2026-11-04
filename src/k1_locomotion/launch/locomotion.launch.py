import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_ns_arg = DeclareLaunchArgument(
        'robot_ns',
        default_value='k1_0',
        description='Robot namespace'
    )
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value='models/k1_velocity_policy.pt',
        description='Path to TorchScript policy file (absolute, or relative '
                    'to the workspace root — resolved by the node)'
    )
    command_type_arg = DeclareLaunchArgument(
        'command_type',
        default_value='k1_interfaces/JointCommand',
        description="Command msg type: 'k1_interfaces/JointCommand' "
                    "(Gazebo/MuJoCo) or 'sensor_msgs/JointState' (Isaac fleet)"
    )
    control_freq_arg = DeclareLaunchArgument(
        'control_freq', default_value='50.0',
        description='Policy inference / command rate in Hz')
    input_mode_arg = DeclareLaunchArgument(
        'input_mode', default_value='latest',
        description="'latest' (50-dim blind policy) or 'stacked' (50x10 "
                    "history, distilled student policy)")

    robot_ns = LaunchConfiguration('robot_ns')

    locomotion_node = Node(
        package='k1_locomotion',
        executable='locomotion_node',
        name='locomotion',
        output='screen',
        parameters=[{
            'robot_ns': robot_ns,
            'policy_path': LaunchConfiguration('policy_path'),
            'command_type': LaunchConfiguration('command_type'),
            'control_freq': LaunchConfiguration('control_freq'),
            'input_mode': LaunchConfiguration('input_mode'),
        }]
    )

    return LaunchDescription([
        robot_ns_arg,
        policy_path_arg,
        command_type_arg,
        control_freq_arg,
        input_mode_arg,
        locomotion_node,
    ])
