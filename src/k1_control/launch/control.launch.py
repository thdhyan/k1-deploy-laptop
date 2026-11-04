from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    robot_ns = LaunchConfiguration('robot_ns').perform(context)
    mode = LaunchConfiguration('mode').perform(context)
    sdk_ip = LaunchConfiguration('sdk_ip').perform(context)
    control_freq = LaunchConfiguration('control_freq').perform(context)

    if mode == 'real':
        bridge_node = Node(
            package='k1_control',
            executable='sdk_bridge',
            name='sdk_bridge_node',
            namespace=robot_ns,
            output='screen',
            parameters=[{
                'robot_ns': robot_ns,
                'sdk_ip': sdk_ip,
                'control_freq': float(control_freq),
            }],
        )
    else:
        # sim mode (Gazebo/Isaac/MuJoCo)
        bridge_node = Node(
            package='k1_control',
            executable='sim_bridge',
            name='sim_bridge_node',
            namespace=robot_ns,
            output='screen',
            parameters=[{'robot_ns': robot_ns}],
        )

    return [bridge_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_ns', default_value='k1_0',
            description='Robot namespace'),
        DeclareLaunchArgument('mode', default_value='sim',
            description='Bridge mode: sim (Gazebo/Isaac/MuJoCo) | real (hardware SDK)'),
        DeclareLaunchArgument('sdk_ip', default_value='192.168.1.100',
            description='Robot SDK IP (real mode only)'),
        DeclareLaunchArgument('control_freq', default_value='50.0',
            description='Control loop frequency (Hz, real mode only)'),
        OpaqueFunction(function=launch_setup),
    ])