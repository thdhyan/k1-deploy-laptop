#!/usr/bin/env python3
"""
Sim Bridge Node — SDK-style joint command endpoint for simulated K1 robots.

Mirrors the booster_robotics_sdk MotorCmd semantics (per-joint q/dq/tau targets)
on ROS2: subscribes k1_interfaces/JointCommand and forwards position targets to
the gz_ros2_control forward_position_controller.

Topics:
  Sub: /{robot_ns}/joint_commands                       (k1_interfaces/JointCommand)
  Pub: /{robot_ns}/forward_position_controller/commands  (std_msgs/Float64MultiArray)

Params:
  robot_ns: str = 'k1_0'
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from k1_interfaces.msg import JointCommand

# Controller joint order (matches k1_controllers.yaml / URDF-exact names)
K1_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


class SimBridgeNode(Node):
    def __init__(self):
        super().__init__("sim_bridge_node")
        ns = self.declare_parameter("robot_ns", "k1_0").value
        self._ns = ns
        self._cmd_pub = self.create_publisher(
            Float64MultiArray, f"/{ns}/forward_position_controller/commands", 10
        )
        self._sub = self.create_subscription(
            JointCommand, f"/{ns}/joint_commands", self._cmd_cb, 10
        )
        self.get_logger().info(f"SimBridgeNode ready [ns={ns}]")

    def _cmd_cb(self, msg: JointCommand):
        if len(msg.joint_names) != len(msg.positions):
            self.get_logger().warn("joint_names/positions length mismatch; dropping")
            return
        # Map named joints onto the controller's fixed ordering
        out = [float("nan")] * len(K1_JOINTS)
        for name, pos in zip(msg.joint_names, msg.positions):
            try:
                out[K1_JOINTS.index(name)] = float(pos)
            except ValueError:
                self.get_logger().warn(f"unknown joint '{name}'; ignored")
        if all(nan != nan for nan in out):
            pass
        m = Float64MultiArray()
        m.data = [0.0 if v != v else v for v in out]
        self._cmd_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = SimBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
