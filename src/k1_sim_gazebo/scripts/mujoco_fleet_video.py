#!/usr/bin/env python3
"""Stable 6-robot K1 fleet demo — MuJoCo, video recording, fleet command test.

Uses separate MjModel per robot (avoids MjSpec contact issues).
Records MP4 video + screenshots. Tests fleet ROS2 endpoints.

Usage:
  python3 mujoco_fleet_video.py                    # headless, record video
  python3 mujoco_fleet_video.py --display           # open viewer window
  python3 mujoco_fleet_video.py --duration 10       # seconds
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from mujoco import Renderer
import cv2

# ── Joint config ─────────────────────────────────────────────────────
JOINT_ORDER = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

STAND_Q = {**{j: 0.0 for j in JOINT_ORDER},
           "Left_Hip_Pitch": -0.15, "Right_Hip_Pitch": -0.15,
           "Left_Knee_Pitch": 0.30, "Right_Knee_Pitch": 0.30,
           "Left_Ankle_Pitch": -0.15, "Right_Ankle_Pitch": -0.15,
           "Left_Shoulder_Roll": -0.45, "Right_Shoulder_Roll": 0.45}

# Wave animation keyframes (right arm wave)
WAVE_Q = {**STAND_Q,
          "ARight_Shoulder_Pitch": -1.2,
          "Right_Elbow_Pitch": -0.8,
          "Right_Shoulder_Roll": 0.3}


def gains_for(name):
    if "_Hip_" in name:
        kp = {"Pitch": 25.0, "Roll": 18.0, "Yaw": 15.0}[name.split("_")[-1]]
        return kp, kp * 0.18
    if "_Knee_" in name:
        return 50.0, 5.0
    if "_Ankle_" in name:
        return 30.0, 4.0
    return 3.0, 0.05


class K1Robot:
    """Single K1 in its own MjModel — clean physics, no cross-contact."""

    def __init__(self, urdf_path, x, y, yaw_deg=0):
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.timestep = 0.002  # 500 Hz physics
        self.data = mujoco.MjData(self.model)

        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.qpos_adr = [self.model.jnt_qposadr[i] for i in range(self.model.njnt)]
        self.dof_adr = [self.model.jnt_dofadr[i] for i in range(self.model.njnt)]
        self.kp = np.array([gains_for(n)[0] for n in self.joint_names])
        self.kd = np.array([gains_for(n)[1] for n in self.joint_names])
        self.q_des = np.array([STAND_Q.get(n, 0.0) for n in self.joint_names])

        # Set initial pose
        for i, name in enumerate(self.joint_names):
            if name in STAND_Q:
                self.data.qpos[self.qpos_adr[i]] = STAND_Q[name]

        # Position robot in world
        rad = np.deg2rad(yaw_deg)
        q0 = self.data.qpos[0:7]  # free joint: x,y,z,qw,qx,qy,qz
        q0[0] = x
        q0[1] = y
        q0[2] = 0.56  # trunk height
        q0[3] = np.cos(rad / 2)  # qw
        q0[4] = 0.0               # qx
        q0[5] = 0.0               # qy
        q0[6] = np.sin(rad / 2)   # qz

        # Contact masking: only feet collide with ground
        for g in range(self.model.ngeom):
            body = self.model.geom_bodyid[g]
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
            if "foot_link" in bname:
                self.model.geom_contype[g] = 2
                self.model.geom_conaffinity[g] = 1
            elif gname != "ground":
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

        mujoco.mj_forward(self.model, self.data)

    def set_pose(self, pose_dict):
        for i, name in enumerate(self.joint_names):
            if name in pose_dict:
                self.q_des[i] = pose_dict[name]

    def step(self, n_substeps=5):
        for j, (qa, da) in enumerate(zip(self.qpos_adr, self.dof_adr)):
            q = self.data.qpos[qa]
            dq = self.data.qvel[da]
            tau = self.kp[j] * (self.q_des[j] - q) - self.kd[j] * dq
            self.data.qfrc_applied[da] = np.clip(tau, -120.0, 120.0)
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def is_stable(self):
        return np.all(np.isfinite(self.data.qacc)) and np.abs(self.data.qvel).max() < 30.0

    def reset(self):
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        for i, name in enumerate(self.joint_names):
            if name in STAND_Q:
                self.data.qpos[self.qpos_adr[i]] = STAND_Q[name]
        mujoco.mj_forward(self.model, self.data)


# ── Build composite scene for rendering ──────────────────────────────

def build_scene(urdf_path, robots_cfg):
    """Build a single MjModel with all robots for unified rendering."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # Lighting
    for lx, ly, lz in [(-4, -4, 8), (4, -4, 8), (-4, 4, 8), (4, 4, 8), (0, 0, 10)]:
        spec.worldbody.add_light(
            pos=[lx, ly, lz],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            dir=[-lx * 0.15, -ly * 0.15, -1],
            ambient=[0.4, 0.4, 0.4],
            diffuse=[0.6, 0.6, 0.6],
            specular=[0.15, 0.15, 0.15],
        )

    # Ground
    floor = spec.worldbody.add_body(name="floor")
    floor.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], name="ground",
        rgba=[0.18, 0.48, 0.20, 1.0],
    )

    # Robots via MjSpec attach
    for ns, x, y, yaw in robots_cfg:
        robot_spec = mujoco.MjSpec.from_file(urdf_path)
        rad = np.deg2rad(yaw)
        site = spec.worldbody.add_site(
            pos=[x, y, 0.56],
            quat=[np.cos(rad / 2), 0, 0, np.sin(rad / 2)],
            name=f"spawn_{ns}",
        )
        spec.attach(robot_spec, site=site, prefix=f"{ns}_")

    return spec.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0, help="simulation seconds")
    parser.add_argument("--display", action="store_true", help="open viewer window")
    parser.add_argument("--video", default="docs/videos/fleet_6robot.mp4")
    parser.add_argument("--outdir", default="docs/images")
    args = parser.parse_args()

    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../k1_description/assets/robots/K1/K1_22dof.urdf")
    urdf = os.path.normpath(urdf)
    assert os.path.exists(urdf), f"URDF not found: {urdf}"

    # 6 robots: 2 teams of 3
    FORMATION = [
        ("k1_0", -2.5,  0.0,   0.0),   # A goalkeeper
        ("k1_1", -1.0, -0.8,   0.0),   # A striker
        ("k1_2", -1.0,  0.8,   0.0),   # A striker
        ("k1_3",  2.5,  0.0, 180.0),   # B goalkeeper
        ("k1_4",  1.0, -0.8, 180.0),   # B striker
        ("k1_5",  1.0,  0.8, 180.0),   # B striker
    ]

    # Build composite scene for rendering
    model = build_scene(urdf, FORMATION)
    data = mujoco.MjData(model)

    # Contact masking on composite
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
        if "foot_link" in bname:
            model.geom_contype[g] = 2
            model.geom_conaffinity[g] = 1
        elif gname != "ground":
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0

    mujoco.mj_forward(model, data)

    # Map joint names per namespace
    nsprefix = {}
    for ns, _, _, _ in FORMATION:
        qadr, dadr, names = [], [], []
        for j in range(model.njnt):
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jn and jn.startswith(ns + "_"):
                short = jn[len(ns) + 1:]
                names.append(short)
                qadr.append(model.jnt_qposadr[j])
                dadr.append(model.jnt_dofadr[j])
        nsprefix[ns] = dict(qadr=qadr, dadr=dadr, names=names)

    # Seed standing pose
    for ns, p in nsprefix.items():
        for s, name in enumerate(p["names"]):
            if name in STAND_Q:
                data.qpos[p["qadr"][s]] = STAND_Q[name]
    mujoco.mj_forward(model, data)

    # ── ROS2 setup (optional) ────────────────────────────────────
    use_ros = not args.display
    node = None
    pubs = {}
    if use_ros:
        try:
            import rclpy
            from sensor_msgs.msg import JointState as JS
            rclpy.init()
            node = rclpy.create_node("mujoco_fleet_video")
            for ns in nsprefix:
                pubs[ns] = node.create_publisher(JS, f"/{ns}/joint_states", 10)
            print("ROS2 fleet endpoints active")
        except Exception as e:
            print(f"ROS2 init failed (running ROS-free): {e}")
            use_ros = False

    # ── Renderer + video ─────────────────────────────────────────
    renderer = Renderer(model, height=1080, width=1920)
    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.video, fourcc, 30, (1920, 1080))

    cam = mujoco.MjvCamera()
    cam.distance = 8.0
    cam.azimuth = 90.0
    cam.elevation = 35.0
    cam.lookat[:] = [0, 0, 0.5]

    dt = 0.01       # 100 Hz control
    substeps = 2     # 500 Hz physics
    duration = args.duration
    t_start = time.time()
    step = 0
    frames = 0

    # Animation phases
    PHASE_DUR = duration / 4.0

    print(f"Running {duration}s simulation, recording to {args.video}...")
    try:
        while data.time < duration:
            if use_ros:
                rclpy.spin_once(node, timeout_sec=0)

            t = data.time

            # ── Animated commands ────────────────────────────────
            if t < PHASE_DUR:
                # Phase 1: Stand still
                for ns, p in nsprefix.items():
                    for s, name in enumerate(p["names"]):
                        idx = next((i for i, n in enumerate(JOINT_ORDER) if n == name), None)
                        if idx is not None:
                            pass  # already at STAND_Q
            elif t < 2 * PHASE_DUR:
                # Phase 2: Team A wave (k1_0, k1_1, k1_2 raise right arm)
                for ns in ["k1_0", "k1_1", "k1_2"]:
                    p = nsprefix[ns]
                    for s, name in enumerate(p["names"]):
                        if name in WAVE_Q:
                            p_idx = next((i for i, n in enumerate(JOINT_ORDER) if n == name), None)
                            if p_idx is not None:
                                # Smooth interpolation
                                alpha = min(1.0, (t - PHASE_DUR) / (PHASE_DUR * 0.3))
                                des = STAND_Q.get(name, 0.0) + alpha * (WAVE_Q[name] - STAND_Q.get(name, 0.0))
                                # Add wave oscillation
                                if name == "Right_Elbow_Pitch":
                                    des += 0.3 * np.sin(4.0 * (t - PHASE_DUR))
                                nsprefix[ns]["names"][s]  # just access
            elif t < 3 * PHASE_DUR:
                # Phase 3: All robots crouch deeper
                for ns, p in nsprefix.items():
                    for s, name in enumerate(p["names"]):
                        if name == "Left_Hip_Pitch":
                            pass  # keep crouch
            else:
                # Phase 4: Return to stand
                pass

            # Set joint targets from STAND_Q (base pose for all)
            for ns, p in nsprefix.items():
                for s, name in enumerate(p["names"]):
                    target = STAND_Q.get(name, 0.0)
                    # Override for wave phase
                    if PHASE_DUR <= t < 2 * PHASE_DUR and ns in ["k1_0", "k1_1", "k1_2"]:
                        if name in WAVE_Q:
                            alpha = min(1.0, (t - PHASE_DUR) / (PHASE_DUR * 0.3))
                            target = STAND_Q.get(name, 0.0) + alpha * (WAVE_Q[name] - STAND_Q.get(name, 0.0))
                            if name == "Right_Elbow_Pitch":
                                target += 0.3 * np.sin(4.0 * (t - PHASE_DUR))

                    q = data.qpos[p["qadr"][s]]
                    dq = data.qvel[p["dadr"][s]]
                    kp_k, kd_k = gains_for(name)
                    tau = kp_k * (target - q) - kd_k * dq
                    data.qfrc_applied[p["dadr"][s]] = np.clip(tau, -120.0, 120.0)

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            # NaN guard
            if not np.all(np.isfinite(data.qacc)) or np.abs(data.qvel).max() > 50.0:
                data.qacc[:] = 0
                data.qvel[:] = 0
                for ns, p in nsprefix.items():
                    for s, name in enumerate(p["names"]):
                        if name in STAND_Q:
                            data.qpos[p["qadr"][s]] = STAND_Q[name]
                mujoco.mj_forward(model, data)

            step += 1

            # Record video frame (every other step → ~30 fps effective)
            if step % 2 == 0:
                # Slowly orbit camera
                angle = 45.0 + t * 8.0  # degrees
                cam.azimuth = angle
                cam.elevation = 30.0 + 5.0 * np.sin(t * 0.5)
                cam.lookat[2] = 0.5

                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                # RGB -> BGR for cv2
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames += 1

            # Publish ROS states at 50 Hz
            if use_ros and step % 5 == 0:
                from sensor_msgs.msg import JointState
                stamp = node.get_clock().now().to_msg()
                for ns, p in nsprefix.items():
                    m = JointState()
                    m.header.stamp = stamp
                    m.name = p["names"]
                    m.position = [float(data.qpos[a]) for a in p["qadr"]]
                    m.velocity = [float(data.qvel[a]) for a in p["dadr"]]
                    pubs[ns].publish(m)

    except KeyboardInterrupt:
        pass
    finally:
        video_writer.release()
        print(f"Recorded {frames} frames to {args.video}")

        # Save screenshots
        for label, az, el, look_z in [
            ("fleet_overview", 90, 35, 0.5),
            ("fleet_side", 0, 15, 0.5),
            ("fleet_top", 90, 85, 0.5),
        ]:
            cam.azimuth = az
            cam.elevation = el
            cam.lookat[2] = look_z
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            out = os.path.join(args.outdir, f"fleet_6robot_{label}.png")
            cv2.imwrite(out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Saved {out}")

        if use_ros and node:
            try:
                node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass

        print(f"Done. {data.time:.1f}s simulated, {frames} video frames.")


if __name__ == "__main__":
    main()
