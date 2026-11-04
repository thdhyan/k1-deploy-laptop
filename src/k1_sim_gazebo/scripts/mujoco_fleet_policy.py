#!/usr/bin/env python3
"""6-robot K1 fleet with trained velocity policy — MuJoCo, video recording.

Loads the trained RSL-RL velocity policy (50-dim obs → 12 leg joint actions)
and has all 6 robots walk forward on a green field with realistic lighting.

Usage:
  python3 mujoco_fleet_policy.py --duration 10
  python3 mujoco_fleet_policy.py --duration 15 --video docs/videos/fleet_policy_walk.mp4
"""
import argparse
import math
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

# 12 leg joints in policy order
LEG_JOINTS = [
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
]

DEFAULT_LEG_POS = np.zeros(12, dtype=np.float32)
ACTION_SCALE = 0.25
# 48 proprioception + 2-dim gait phase clock — same contract as
# k1_velocity/tasks/velocity/gait_clock.py and locomotion_node.OBS_DIM.
OBS_DIM = 50
PHASE_FREQ_HZ = 1.0              # one gait cycle = two steps = 2.0 steps/s
PHASE_DT = PHASE_FREQ_HZ / 50.0  # control loop runs at 50 Hz

STAND_Q = {**{j: 0.0 for j in JOINT_ORDER},
           "Left_Hip_Pitch": -0.15, "Right_Hip_Pitch": -0.15,
           "Left_Knee_Pitch": 0.30, "Right_Knee_Pitch": 0.30,
           "Left_Ankle_Pitch": -0.15, "Right_Ankle_Pitch": -0.15,
           "Left_Shoulder_Roll": -0.45, "Right_Shoulder_Roll": 0.45}


def gains_for(name):
    if "_Hip_" in name:
        kp = {"Pitch": 25.0, "Roll": 18.0, "Yaw": 15.0}[name.split("_")[-1]]
        return kp, kp * 0.18
    if "_Knee_" in name:
        return 50.0, 5.0
    if "_Ankle_" in name:
        return 30.0, 4.0
    return 3.0, 0.05


def load_policy(path):
    """Load TorchScript velocity policy."""
    import torch
    if not os.path.isfile(path):
        ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(ws, path)
    policy = torch.jit.load(path, map_location="cpu")
    policy.eval()
    # Verify
    out = policy(torch.zeros(1, OBS_DIM))
    assert out.numel() == 12, f"Expected 12 outputs, got {out.numel()}"
    print(f"Policy loaded: {path} ({OBS_DIM}-dim obs → 12 actions)")
    return policy


def build_obs(policy_state, joint_pos, joint_vel, cmd_vel, last_action, phase):
    """Build 50-dim observation matching training layout (48 proprio + clock)."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0:3] = policy_state["base_lin_vel"]
    obs[3:6] = policy_state["base_ang_vel"]
    obs[6:9] = policy_state["projected_gravity"]
    obs[9:12] = cmd_vel
    obs[12:24] = joint_pos - DEFAULT_LEG_POS
    obs[24:36] = joint_vel
    obs[36:48] = last_action
    angle = 2.0 * math.pi * phase
    obs[48:50] = (math.sin(angle), math.cos(angle))
    return obs


def build_scene(urdf_path, robots_cfg):
    """Build a MuJoCo world with ground, lighting, and N robots."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # Stadium lighting — warm overhead + fill
    for lx, ly, lz in [(-5, -5, 10), (5, -5, 10), (-5, 5, 10), (5, 5, 10),
                        (0, 0, 12)]:
        spec.worldbody.add_light(
            pos=[lx, ly, lz],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            dir=[-lx * 0.1, -ly * 0.1, -1],
            ambient=[0.3, 0.3, 0.35],
            diffuse=[0.7, 0.7, 0.65],
            specular=[0.2, 0.2, 0.2],
        )

    # Ground plane — green field with grid lines
    floor = spec.worldbody.add_body(name="floor")
    floor.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], name="ground",
        rgba=[0.18, 0.50, 0.22, 1.0], friction=[0.8, 0.005, 0.0001],
    )

    # Field markings
    def line(x, y, sx, sy):
        b = spec.worldbody.add_body(pos=[x, y, 0.005])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, 0.005],
                   rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)

    # Center circle and lines
    line(0, 0, 12.0, 0.05)         # center line
    line(0, 3.0, 12.0, 0.05)       # north touchline
    line(0, -3.0, 12.0, 0.05)      # south touchline
    line(-6.0, 0, 0.05, 6.0)       # west goal line
    line(6.0, 0, 0.05, 6.0)        # east goal line

    # Center circle
    for angle in range(0, 360, 5):
        rad = np.deg2rad(angle)
        r = 1.5
        b = spec.worldbody.add_body(pos=[r * np.cos(rad), r * np.sin(rad), 0.005])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.005],
                   rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)

    # 6 K1 robots
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
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--video", default="docs/videos/fleet_policy_walk.mp4")
    parser.add_argument("--outdir", default="docs/images")
    parser.add_argument("--policy", default="models/k1_velocity_policy.pt")
    parser.add_argument("--cmd-vx", type=float, default=0.5, help="forward velocity cmd (m/s)")
    args = parser.parse_args()

    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../k1_description/assets/robots/K1/K1_22dof.urdf")
    urdf = os.path.normpath(urdf)

    # Formation: 2 teams of 3, facing each other
    FORMATION = [
        ("k1_0", -3.0,  0.0,   0.0),
        ("k1_1", -1.5, -1.0,   0.0),
        ("k1_2", -1.5,  1.0,   0.0),
        ("k1_3",  3.0,  0.0, 180.0),
        ("k1_4",  1.5, -1.0, 180.0),
        ("k1_5",  1.5,  1.0, 180.0),
    ]

    # Load policy
    import torch
    policy = load_policy(args.policy)

    # Build scene
    model = build_scene(urdf, FORMATION)
    data = mujoco.MjData(model)

    # Contact masking
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

    # Map joint indices per namespace
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

    # Per-robot policy state
    robot_state = {}
    for ns in nsprefix:
        robot_state[ns] = {
            "base_lin_vel": np.zeros(3, dtype=np.float32),
            "base_ang_vel": np.zeros(3, dtype=np.float32),
            "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "last_action": np.zeros(12, dtype=np.float32),
            "q_des": np.zeros(len(nsprefix[ns]["names"]), dtype=np.float64),
            "cmd_vel": np.array([args.cmd_vx, 0.0, 0.0], dtype=np.float32),
            # gait phase clock origin (arbitrary: training re-seeds per episode)
            "phase": 0.0,
        }
        # Initialize q_des from stand pose
        for i, name in enumerate(nsprefix[ns]["names"]):
            robot_state[ns]["q_des"][i] = STAND_Q.get(name, 0.0)

    # Setup video
    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.video, fourcc, 30, (1920, 1080))
    renderer = Renderer(model, height=1080, width=1920)

    cam = mujoco.MjvCamera()
    cam.distance = 10.0
    cam.azimuth = 45.0
    cam.elevation = 30.0
    cam.lookat[:] = [0, 0, 0.5]

    dt = 0.02         # 50 Hz control (matches training)
    substeps = 10     # 500 Hz physics
    frames = 0
    t_start = time.time()

    print(f"Running {args.duration}s with velocity policy (vx={args.cmd_vx} m/s)...")

    try:
        while data.time < args.duration:
            t = data.time

            # ── Policy inference per robot ─────────────────────────
            for ns, p in nsprefix.items():
                st = robot_state[ns]

                # Build observation from MuJoCo state
                joint_pos = np.array([data.qpos[qa] for qa in p["qadr"]], dtype=np.float32)
                joint_vel = np.array([data.qvel[da] for da in p["dadr"]], dtype=np.float32)

                # Extract leg joint values (12) in policy order
                leg_pos = np.zeros(12, dtype=np.float32)
                leg_vel = np.zeros(12, dtype=np.float32)
                for lj_idx, lj_name in enumerate(LEG_JOINTS):
                    if lj_name in p["names"]:
                        j_idx = p["names"].index(lj_name)
                        leg_pos[lj_idx] = joint_pos[j_idx]
                        leg_vel[lj_idx] = joint_vel[j_idx]

                # Compute projected gravity from trunk orientation
                trunk_adr = p["qadr"][0] if p["qadr"] else 0
                # Free joint: x,y,z,qw,qx,qy,qz
                # Find trunk body orientation
                trunk_qadr = None
                for j in range(model.njnt):
                    jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if jn and jn.startswith(ns + "_") and "trunk" in jn.lower():
                        trunk_qadr = model.jnt_qposadr[j]
                        break

                if trunk_qadr is not None:
                    qw, qx, qy, qz = [data.qpos[trunk_qadr + i] for i in range(4)]
                    q = np.array([qw, qx, qy, qz], dtype=np.float64)
                    q /= np.linalg.norm(q)
                    w, x, y, z = q
                    R = np.array([
                        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
                    ])
                    st["projected_gravity"] = (R.T @ [0, 0, -1]).astype(np.float32)

                # Simple velocity estimation from qvel (base link linear vel)
                base_v_adr = None
                for j in range(model.njnt):
                    jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if jn and jn.startswith(ns + "_trunk"):
                        base_v_adr = model.jnt_dofadr[j]
                        break
                if base_v_adr is not None:
                    st["base_lin_vel"] = data.qvel[base_v_adr:base_v_adr+3].astype(np.float32)
                    st["base_ang_vel"] = data.qvel[base_v_adr+3:base_v_adr+6].astype(np.float32)

                obs = build_obs(st, leg_pos, leg_vel, st["cmd_vel"],
                                st["last_action"], st["phase"])
                st["phase"] = (st["phase"] + PHASE_DT) % 1.0

                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

                st["last_action"] = action
                targets = DEFAULT_LEG_POS + ACTION_SCALE * action

                # Map policy output back to joint order
                for lj_idx, lj_name in enumerate(LEG_JOINTS):
                    if lj_name in p["names"]:
                        j_idx = p["names"].index(lj_name)
                        st["q_des"][j_idx] = targets[lj_idx]

            # ── PD torques ────────────────────────────────────────
            for ns, p in nsprefix.items():
                st = robot_state[ns]
                for s, (qa, da) in enumerate(zip(p["qadr"], p["dadr"])):
                    q = data.qpos[qa]
                    dq = data.qvel[da]
                    kp_k, kd_k = gains_for(p["names"][s])
                    tau = kp_k * (st["q_des"][s] - q) - kd_k * dq
                    data.qfrc_applied[da] = np.clip(tau, -120.0, 120.0)

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

            # ── Video frame ───────────────────────────────────────
            step_num = int(data.time / 0.002)
            if step_num % 3 == 0:  # ~33 fps effective
                # Orbit camera slowly around the formation
                angle = 30.0 + t * 5.0
                cam.azimuth = angle
                cam.elevation = 25.0 + 8.0 * np.sin(t * 0.3)
                cam.lookat[0] = 0.0
                cam.lookat[1] = 0.0
                cam.lookat[2] = 0.5

                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames += 1

    except KeyboardInterrupt:
        pass
    finally:
        video_writer.release()

        # Save key screenshots
        for label, az, el in [("fleet_policy_overview", 45, 30),
                               ("fleet_policy_side", 0, 15),
                               ("fleet_policy_top", 90, 80)]:
            cam.azimuth = az
            cam.elevation = el
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            out = os.path.join(args.outdir, f"fleet_policy_{label}.png")
            cv2.imwrite(out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Saved {out}")

        print(f"Done. {data.time:.1f}s simulated, {frames} video frames → {args.video}")


if __name__ == "__main__":
    main()
