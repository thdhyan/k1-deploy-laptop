#!/usr/bin/env python3
"""
Whole-Body Controller (WBC) Node — decoupled lower/upper body control.

Architecture (from GR00T Decoupled WBC):
  Lower body: locomotion policy -> 12 leg joint targets
  Upper body: default/telop arm poses -> 10 arm/head joint targets

This node merges both into a single 22-joint JointCommand.
Phase 1-4: passthrough locomotion commands only.
Phase 5+: full QP-based WBC.

Topics:
  Sub: /{robot_ns}/locomotion/joint_commands  (k1_interfaces/JointCommand, 12 leg joints)
  Sub: /{robot_ns}/arm/joint_commands         (k1_interfaces/JointCommand, 10 arm+head joints)
  Pub: /{robot_ns}/joint_commands             (k1_interfaces/JointCommand, 22 joints merged)

Params:
  robot_ns: str = 'k1_0'
  wbc_mode: str = 'passthrough'  # 'passthrough' | 'qp' (Phase 5+)
"""
import rclpy
from rclpy.node import Node

ALL_JOINTS = [
    'Head_yaw', 'Head_pitch',
    'Left_Shoulder_Pitch', 'Left_Shoulder_Roll', 'Left_Elbow_Pitch', 'Left_Elbow_Yaw',
    'Right_Shoulder_Pitch', 'Right_Shoulder_Roll', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw',
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
]


class WbcNode(Node):
    def __init__(self):
        super().__init__('wbc_node')
        ns = self.declare_parameter('robot_ns', 'k1_0').value
        mode = self.declare_parameter('wbc_mode', 'passthrough').value
        self.get_logger().info(f'WbcNode ready [ns={ns}, mode={mode}]')
        if mode == 'qp':
            self.get_logger().warn('QP WBC not yet implemented (Phase 5+), falling back to passthrough')


def main(args=None):
    rclpy.init(args=args)
    node = WbcNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
