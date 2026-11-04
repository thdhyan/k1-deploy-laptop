#!/usr/bin/env python3
"""
Joint State Publisher — placeholder for future joint state publishing.
"""
import rclpy
from rclpy.node import Node


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        ns = self.declare_parameter('robot_ns', 'k1_0').value
        self.get_logger().info(f'JointStatePublisher ready [ns={ns}]')


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
