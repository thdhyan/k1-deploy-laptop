#!/usr/bin/env python3
"""
MuJoCo fleet node — N K1 robots, one physics world each, ROS2 namespaced.

SDK-style endpoints per robot (mirrors the Gazebo fleet):
  Sub: /{robot_ns}/joint_commands  (k1_interfaces/JointCommand)  -> PD targets
  Pub: /{robot_ns}/joint_states    (sensor_msgs/JointState)

Each robot gets its own MjModel/MjData compiled from the booster_assets URDF
(fixed links fused). Control: torque = kp*(q_des - q) - kd*dq applied via
qfrc_applied — same semantics as the SDK's MotorCmd(kp,kd,q_des).

Usage (system ROS2 shell):
  python3 mujoco_fleet_node.py --n_robots 2 --rate 200
"""
import argparse
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from k1_interfaces.msg import JointCommand

# Per-joint-group PD gains (from booster actuator specs, Nm/rad & Nms/rad)
def gains_for(name: str):
    if "_Hip_" in name:
        kp = {"Pitch": 30.2, "Roll": 21.4, "Yaw": 17.8}[name.split("_")[-1]]
        return kp, kp * 0.12
    if "_Knee_" in name:
        return 60.4, 4.8
    if "_Ankle_" in name:
        return 35.7, 4.3
    if "Head" in name or "_Elbow" in name or "_Shoulder" in name:
        return 4.0, 0.25
    return 10.0, 0.5


class K1MujocoRobot:
    def __init__(self, ns, urdf_path, parent_node):
        import mujoco

        self.ns = ns
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        # Featherweight ankle-cross linkage bodies need implicit damping for
        # stiff PD stability (explicit Euler diverges).
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.qpos_adr = [self.model.jnt_qposadr[i] for i in range(self.model.njnt)]
        self.dof_adr = [self.model.jnt_dofadr[i] for i in range(self.model.njnt)]
        self.kp = [gains_for(n)[0] for n in self.joint_names]
        self.kd = [gains_for(n)[1] for n in self.joint_names]
        self.q_des = [0.0] * len(self.joint_names)

        self.pub = parent_node.create_publisher(
            JointState, f"/{ns}/joint_states", 10
        )
        parent_node.create_subscription(
            JointCommand, f"/{ns}/joint_commands", self.cmd_cb, 10
        )

    def cmd_cb(self, msg: JointCommand):
        for name, pos in zip(msg.joint_names, msg.positions):
            if name in self.joint_names:
                self.q_des[self.joint_names.index(name)] = float(pos)

    def step(self, mujoco, n_substeps):
        # PD torques on joint DoFs
        for j, (qa, da) in enumerate(zip(self.qpos_adr, self.dof_adr)):
            q = self.data.qpos[qa]
            dq = self.data.qvel[da]
            tau = self.kp[j] * (self.q_des[j] - q) - self.kd[j] * dq
            self.data.qfrc_applied[da] = max(-120.0, min(120.0, tau))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def publish_state(self, node, stamp):
        m = JointState()
        m.header.stamp = stamp
        m.name = self.joint_names
        m.position = [self.data.qpos[a] for a in self.qpos_adr]
        m.velocity = [self.data.qvel[a] for a in self.dof_adr]
        m.effort = [self.data.qfrc_applied[a] for a in self.dof_adr]
        self.pub.publish(m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_robots", type=int, default=2)
    parser.add_argument("--rate", type=int, default=200, help="control loop Hz")
    parser.add_argument("--urdf", default="")
    args = parser.parse_args()

    import mujoco
    from ament_index_python.packages import get_package_share_directory

    urdf = args.urdf or os.path.join(
        get_package_share_directory("k1_description"),
        "assets", "robots", "K1", "K1_22dof.urdf",
    )

    rclpy.init()
    node = rclpy.create_node("mujoco_fleet")

    robots = []
    for i in range(args.n_robots):
        ns = f"k1_{i}"
        robots.append(K1MujocoRobot(ns, urdf, node))
        node.get_logger().info(
            f"[{ns}] mj model: njnt={robots[-1].model.njnt} "
            f"joints={robots[-1].joint_names[:3]}..."
        )

    dt = 1.0 / args.rate
    substeps = max(1, int(round(dt / 0.002)))  # 500 Hz physics

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)
            stamp = node.get_clock().now().to_msg()
            for r in robots:
                r.step(mujoco, substeps)
                r.publish_state(node, stamp)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
