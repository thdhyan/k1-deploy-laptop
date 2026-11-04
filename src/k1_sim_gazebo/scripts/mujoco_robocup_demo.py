#!/usr/bin/env python3
"""RoboCup 3v3 K1 demo on a soccer field — MuJoCo backend, CPU physics.

Assembles ONE world: field markings, goals, ball, and six K1 humanoids
(teams of 3 facing off) using MjSpec URDF attach with per-robot name prefixes.
Exposes the standard fleet endpoints per robot:

    /{ns}/joint_states    pub @50 Hz
    /{ns}/joint_commands  sub -> PD torques (SDK MotorCmd-style)

Capture mode renders offscreen snapshots to docs/images/ for the README.

Usage:
  python3 src/k1_sim_gazebo/scripts/mujoco_robocup_demo.py --capture
  python3 src/k1_sim_gazebo/scripts/mujoco_robocup_demo.py --n_robots 6
"""
import argparse
import os
import time

# EGL offscreen rendering must be selected before mujoco import
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

FIELD_L, FIELD_W = 8.0, 5.0   # demo-scale pitch (m)
GOAL_W, GOAL_H = 2.0, 0.9
BALL_R = 0.11

FORMATION = [  # ns, x, y, yaw_deg
    ("k1_0", -3.4,  0.0,   0.0),   # team A goalkeeper
    ("k1_1", -1.6, -0.9,   0.0),   # team A striker
    ("k1_2", -1.6,  0.9,   0.0),   # team A striker
    ("k1_3",  3.4,  0.0, 180.0),   # team B goalkeeper
    ("k1_4",  1.6, -0.9, 180.0),   # team B striker
    ("k1_5",  1.6,  0.9, 180.0),   # team B striker
]

JOINT_ORDER = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# slight crouch + arms tucked: much more stable than straight legs
STAND_Q = {**{j: 0.0 for j in JOINT_ORDER},
           "Left_Hip_Pitch": -0.15, "Right_Hip_Pitch": -0.15,
           "Left_Knee_Pitch": 0.30, "Right_Knee_Pitch": 0.30,
           "Left_Ankle_Pitch": -0.15, "Right_Ankle_Pitch": -0.15,
           "Left_Shoulder_Roll": -0.45, "Right_Shoulder_Roll": 0.45}


def gains_for(name):
    if "_Hip_" in name:
        kp = {"Pitch": 30.2, "Roll": 21.4, "Yaw": 17.8}[name.split("_")[-1]]
        return kp, kp * 0.15
    if "_Knee_" in name:
        return 60.4, 6.0
    if "_Ankle_" in name:
        return 35.7, 5.0
    # head/arms: tiny link inertias (~1e-4 kg m^2) -> keep kd*dt/I << 1
    return 3.0, 0.02


def build_field_spec():
    import mujoco

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # stadium lighting
    for lx, ly, lz in [(-3, -3, 6), (3, -3, 6), (-3, 3, 6), (3, 3, 6)]:
        spec.worldbody.add_light(pos=[lx, ly, lz],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 dir=[-lx * 0.2, -ly * 0.2, -1],
                                 ambient=[0.35, 0.35, 0.35],
                                 diffuse=[0.7, 0.7, 0.7],
                                 specular=[0.2, 0.2, 0.2])

    floor = spec.worldbody.add_body(name="floor")
    floor.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], name="ground",
        rgba=[0.15, 0.45, 0.16, 1.0],
    )

    def line(x, y, sx, sy):
        b = spec.worldbody.add_body(pos=[x, y, 0.004])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, 0.004],
                   rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)

    # pitch markings
    line(0, 0, FIELD_L + 0.1, 0.06)                      # halfway
    line(0, FIELD_W / 2, 0.06, FIELD_W)                  # touchline N
    line(0, -FIELD_W / 2, 0.06, FIELD_W)                 # touchline S
    line(-FIELD_L / 2, 0, 0.06, FIELD_W)                 # goal line W
    line(FIELD_L / 2, 0, 0.06, FIELD_W)                  # goal line E

    def goal(x, sign):
        posts = [(x, GOAL_W / 2), (x, -GOAL_W / 2)]
        for i, (px, py) in enumerate(posts):
            b = spec.worldbody.add_body(pos=[px, py, GOAL_H / 2])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.04, GOAL_H / 2, 0], rgba=[0.9, 0.9, 0.9, 1])
        bar = spec.worldbody.add_body(pos=[x, 0, GOAL_H])
        bar.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[0.02, GOAL_W / 2 + 0.04, 0.02], rgba=[0.9, 0.9, 0.9, 1])

    goal(-FIELD_L / 2, -1)
    goal(+FIELD_L / 2, +1)

    # ball
    ball = spec.worldbody.add_body(name="ball", pos=[0, 0, BALL_R + 0.01])
    j = ball.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="ball_free")
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
                  mass=0.27, friction=[0.7, 0.005, 0.0001],
                  rgba=[1.0, 0.85, 0.1, 1])

    # six K1 humanoids
    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../k1_description/assets/robots/K1/K1_22dof.urdf")
    urdf = os.path.normpath(urdf)
    for ns, x, y, yaw in FORMATION:
        robot_spec = mujoco.MjSpec.from_file(urdf)
        rad = np.deg2rad(yaw)
        site = spec.worldbody.add_site(
            pos=[x, y, 0.56],
            quat=[np.cos(rad / 2), 0, 0, np.sin(rad / 2)],
            name=f"spawn_{ns}",
        )
        spec.attach(robot_spec, site=site, prefix=f"{ns}_")

    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true",
                        help="run scripted kick + save README snapshots, then exit")
    parser.add_argument("--outdir", default="docs/images")
    args, _ = parser.parse_known_args()

    import mujoco
    from mujoco import Renderer

    JointState = JointCommand = None
    if not args.capture:  # ROS mode needs the workspace-sourced stack
        import rclpy
        from sensor_msgs.msg import JointState
        JointCommand = JointState

    spec = build_field_spec()
    model = spec.compile()

    # Contact masking: URDF import makes every geom collide with everything,
    # so tucked arms clip the torso and explode the solver.
    # Scheme: floor c=1/a=1 (default), foot geoms c=2/a=1 -> feet-floor only;
    # every other robot geom c=0/a=0 -> no self-/inter-robot contacts.
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
        if bname == "ball" or gname == "ground":
            continue  # keep world/ball contacts at defaults
        if "foot_link" in bname:
            model.geom_contype[g] = 2
            model.geom_conaffinity[g] = 1   # feet <-> floor only
        else:
            model.geom_contype[g] = 0       # no self-/inter-robot contacts
            model.geom_conaffinity[g] = 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    nsprefix = {}
    for ns, _, _, _ in FORMATION:
        qadr, dadr, names = [], [], []
        for j in range(model.njnt):
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jn.startswith(ns + "_"):
                short = jn[len(ns) + 1:]
                names.append(short)
                qadr.append(model.jnt_qposadr[j])
                dadr.append(model.jnt_dofadr[j])
        # seed standing pose
        for s, jj in enumerate(names):
            if jj in STAND_Q:
                data.qpos[qadr[s]] = STAND_Q[jj]
        nsprefix[ns] = dict(qadr=qadr, dadr=dadr, names=names)

    mujoco.mj_forward(model, data)

    use_ros = not args.capture   # capture mode runs ROS-free (endpoints proven in fleet mode)
    node = None
    pubs, subs = {}, {}
    if use_ros:
        rclpy.init()
        node = rclpy.create_node("mujoco_robocup")
    state = {}

    for ns, p in nsprefix.items():
        state[ns] = dict(q_des=np.array([STAND_Q.get(n, 0.0) for n in p["names"]]))
        if use_ros:
            pubs[ns] = node.create_publisher(JointState, f"/{ns}/joint_states", 10)

        kp = np.array([gains_for(n)[0] for n in p["names"]])
        kd = np.array([gains_for(n)[1] for n in p["names"]])

        def make_cb(p=p, kp=kp, kd=kd, ns=ns):
            def cb(msg):
                for name, pos in zip(msg.joint_names, msg.positions):
                    if name in p["names"]:
                        i = p["names"].index(name)
                        state[ns]["q_des"][i] = float(pos)
            return cb

        if use_ros:
            subs[ns] = node.create_subscription(
                JointCommand, f"/{ns}/joint_commands", make_cb(), 10)
            node.get_logger().info(f"[{ns}] robocup endpoint ready ({len(p['names'])} joints)")

    renderer = None
    if args.capture:
        os.makedirs(args.outdir, exist_ok=True)
        renderer = Renderer(model, height=720, width=1280)
    _cam = mujoco.MjvCamera()
    log = (lambda m: node.get_logger().info(m)) if use_ros else (lambda m: print(m, flush=True))

    def snap(label, pos, target):
        """Render one viewpoint with retries (EGL can hiccup under load)."""
        import imageio.v2 as imageio

        out = os.path.join(args.outdir, f"robocup_{label}.png")
        look(_cam, pos, target)
        for attempt in range(5):
            try:
                renderer.update_scene(data, camera=_cam)
                imageio.imwrite(out, renderer.render())
                log(f"saved {out}")
                return True
            except Exception as e:
                log(f"render retry {attempt}: {e}")
                time.sleep(0.5)
        return False

    def look(cam, pos, target):
        cam.lookat[:] = target
        cam.distance = float(np.linalg.norm(np.array(pos) - np.array(target)))
        cam.azimuth = float(np.degrees(np.arctan2(pos[1] - target[1], pos[0] - target[0])))
        cam.elevation = float(np.degrees(np.arctan2(
            pos[2] - target[2],
            np.linalg.norm(np.array(pos[:2]) - np.array(target[:2])))))

    kick_t = 2.5      # sim time of scripted kick by k1_1
    shots = {}        # label -> sim time to snap

    dt = 0.01         # 100 Hz control
    substeps = 10     # 1000 Hz physics
    t_start = time.time()
    step = 0
    try:
        while True:
            if use_ros:
                rclpy.spin_once(node, timeout_sec=0)

            # scripted kick: right-leg swing of striker k1_1 toward the ball
            if args.capture:
                i_rhp = nsprefix["k1_1"]["names"].index("Right_Hip_Pitch")
                if data.time > kick_t and data.time < kick_t + 0.6:
                    state["k1_1"]["q_des"][i_rhp] = -0.9      # wind back
                elif data.time >= kick_t + 0.6 and data.time < kick_t + 1.2:
                    state["k1_1"]["q_des"][i_rhp] = 0.7       # swing through ball
                elif data.time >= kick_t + 1.2:
                    state["k1_1"]["q_des"][i_rhp] = -0.15     # recover

            # PD torques
            for ns, p in nsprefix.items():
                st = state[ns]
                for s, da in enumerate(p["dadr"]):
                    q = data.qpos[p["qadr"][s]]
                    dq = data.qvel[da]
                    kp_l, kd_l = gains_for(p["names"][s])
                    tau = kp_l * (st["q_des"][s] - q) - kd_l * dq
                    data.qfrc_applied[da] = max(-120.0, min(120.0, tau))

            for _ in range(substeps):
                mujoco.mj_step(model, data)
            # NaN guard: reset robot joints if the solver diverges
            unstable = (not np.all(np.isfinite(data.qacc))) or \
                       (np.abs(data.qvel).max() > 50.0)
            if unstable:
                log("qacc diverged — resetting robots")
                data.qacc[:] = 0
                data.qvel[:] = 0
                for ns, p in nsprefix.items():
                    for s, qa in enumerate(p["qadr"]):
                        data.qpos[qa] = STAND_Q.get(p["names"][s], 0.0)
                mujoco.mj_forward(model, data)

            step += 1
            if step % 5 == 0 and use_ros:  # 50 Hz states
                stamp = node.get_clock().now().to_msg()
                if True:
                    for ns, p in nsprefix.items():
                        m = JointState()
                        m.header.stamp = stamp
                        m.name = p["names"]
                        m.position = [float(data.qpos[a]) for a in p["qadr"]]
                        m.velocity = [float(data.qvel[a]) for a in p["dadr"]]
                        pubs[ns].publish(m)

            if args.capture:
                snaps = [
                    ("field_top", [0.0, 0.0, 9.0], [0, 0, 0]),
                    ("field_side", [0.0, -7.5, 2.2], [0, 0, 0.6]),
                    ("kick_close", [-4.2, -3.6, 1.5], [-1.0, -0.4, 0.5]),
                ]
                for label, pos, tgt in snaps:
                    if label not in shots and data.time > (kick_t - 1.0 if label != "field_top" else 0.5):
                        if snap(label, pos, tgt):
                            shots[label] = True
                if len(shots) == len(snaps) and data.time > kick_t + 1.5:
                    break

            if not args.capture and time.time() - t_start > 3600:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if use_ros:
            try:
                node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
