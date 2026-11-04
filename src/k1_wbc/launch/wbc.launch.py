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
    wbc_mode_arg = DeclareLaunchArgument(
        'wbc_mode',
        default_value='passthrough',
        description='WBC mode: "passthrough" or "qp" (Phase 5+)'
    )

    robot_ns = LaunchConfiguration('robot_ns')
    wbc_mode = LaunchConfiguration('wbc_mode')

    wbc_node = Node(
        package='k1_wbc',
        executable='wbc_node',
        name='wbc',
        namespace=robot_ns,
        output='screen',
        parameters=[
            {'robot_ns': robot_ns},
            {'wbc_mode': wbc_mode},
        ]
    )

    return LaunchDescription([
        robot_ns_arg,
        wbc_mode_arg,
        wbc_node,
    ])
