#!/usr/bin/env python3
"""K1 stereo head capture — ZED 2i replica in MuJoCo (CPU).

Mirrors src/k1_sim_isaac/scripts/stereo_cam_test.py: two cameras mounted on
the Head_2 link (+/-60 mm baseline matching the ZED 2i's 120 mm), HD720 with
the rectified FOV of the ZED 2i 2.1 mm lens per Stereolabs' table
(H101 / V68 -> f_px ~= 527 @ 1280 px). Renders left/right RGB, computes SGBM
disparity and colorized depth, saves under docs/images/.

ROS-free capture-only script.

Usage:
  python3 src/k1_sim_gazebo/scripts/mujoco_stereo_capture.py
"""
import argparse
import os

# EGL offscreen rendering must be selected before mujoco import
os.environ.setdefault("MUJOCO_GL", "egl")

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="docs/images")
parser.add_argument("--steps", type=int, default=300,
                    help="PD-hold settle steps before capture")
parser.add_argument("--baseline", type=float, default=0.120,
                    help="ZED 2i baseline m")
args, _ = parser.parse_known_args()

import numpy as np  # noqa: E402

RES_W, RES_H = 1280, 720          # HD720 mode
HFOV_DEG = 101.0                  # ZED 2i 2.1mm @ HD720 (Stereolabs table)
# MuJoCo cameras look down -Z of their frame with image-up along +Y_cam.
# Compose: R_y(-90) maps view -z_cam -> +x (robot forward); then a +90 roll
# about the view axis maps image-up to world +z. Verified: quat (0.5,0.5,-0.5,-0.5)
# maps -z_cam->+x, +y_cam->+z, +x_cam(right)->-y. Horizontal epipolar geometry
# is preserved only with this exact orientation (SGBM requires it).
FWD_QUAT_WXYZ = (0.5, 0.5, -0.5, -0.5)

STAND_Q = {
    "Left_Hip_Pitch": -0.15, "Right_Hip_Pitch": -0.15,
    "Left_Knee_Pitch": 0.30, "Right_Knee_Pitch": 0.30,
    "Left_Ankle_Pitch": -0.15, "Right_Ankle_Pitch": -0.15,
    "Left_Shoulder_Roll": -0.45, "Right_Shoulder_Roll": 0.45,
}


def gains_for(name):
    if "_Hip_" in name:
        kp = {"Pitch": 30.2, "Roll": 21.4, "Yaw": 17.8}[name.split("_")[-1]]
        return kp, kp * 0.15
    if "_Knee_" in name:
        return 60.4, 6.0
    if "_Ankle_" in name:
        return 35.7, 5.0
    return 3.0, 0.02


def build_spec():
    import mujoco

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = RES_W
    spec.visual.global_.offheight = RES_H

    for lx, ly in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
        spec.worldbody.add_light(pos=[lx, ly, 6],
                                 dir=[-lx * 0.2, -ly * 0.2, -1],
                                 ambient=[0.35, 0.35, 0.35],
                                 diffuse=[0.7, 0.7, 0.7],
                                 specular=[0.2, 0.2, 0.2])

    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], rgba=[0.15, 0.45, 0.16, 1])

    # obstacles ahead of the robot so disparity/depth have structure
    # (same layout as the Isaac Sim stereo scene)
    for i, (x, y, z, c) in enumerate([
            (1.5, -0.3, 0.25, (0.8, 0.2, 0.2, 1)),
            (2.8, 0.4, 0.40, (0.2, 0.4, 0.85, 1)),
            (4.2, 0.0, 0.18, (0.9, 0.7, 0.1, 1))]):
        spec.worldbody.add_body(pos=[x, y, z]).add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.25, 0.25, z], rgba=list(c), contype=0, conaffinity=0)

    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../k1_description/assets/robots/K1/K1_22dof.urdf")
    spec.attach(mujoco.MjSpec.from_file(os.path.normpath(urdf)),
                site=spec.worldbody.add_site(pos=[0, 0, 0], name="spawn_k1"),
                prefix="k1_")

    # ZED 2i replica: +/-60 mm on Head_2, x=+0.10 clears the head mesh
    head = [b for b in spec.bodies if b.name.endswith("Head_2")]
    assert head, "Head_2 body not found"
    f_px = RES_W / (2 * np.tan(np.deg2rad(HFOV_DEG) / 2))
    fovy = 2 * np.rad2deg(np.arctan(RES_H / (2 * f_px)))
    for side, sign in (("zed_left", +1), ("zed_right", -1)):
        head[0].add_camera(
            name=f"k1_{side}",
            pos=[0.10, sign * args.baseline / 2, 0.0],
            quat=list(FWD_QUAT_WXYZ),
            fovy=float(fovy),
        )
    return spec, fovy


def main():
    import cv2
    import mujoco
    from mujoco import Renderer
    import imageio.v2 as imageio

    os.makedirs(args.outdir, exist_ok=True)
    spec, fovy = build_spec()
    model = spec.compile()

    # Contact masking (same scheme as mujoco_robocup_demo): feet-floor only
    for g in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[g])) or ""
        if "foot" in bname.lower():
            model.geom_contype[g] = 2
            model.geom_conaffinity[g] = 1
        elif bname:  # robot geoms (world geoms have empty names)
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # seed standing crouch
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in range(model.njnt)]
    kp, kd, q_des = [], [], []
    for j, jn in enumerate(jnames):
        qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[qa + 2] = 0.56  # nominal trunk height
            continue
        target = STAND_Q.get(jn.removeprefix("k1_"), 0.0)
        data.qpos[qa] = target
        g = gains_for(jn.removeprefix("k1_"))
        kp.append(g[0]); kd.append(g[1]); q_des.append(target)
    kp, kd, q_des = map(np.array, (kp, kd, q_des))
    mujoco.mj_forward(model, data)

    hinge_joints = [j for j in range(model.njnt)
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    for _ in range(args.steps):
        dq = np.array([data.qvel[model.jnt_dofadr[j]] for j in hinge_joints])
        q = np.array([data.qpos[model.jnt_qposadr[j]]
                      for j in range(model.njnt)
                      if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE])
        tau = kp * (q_des - q) - kd * dq
        data.qfrc_applied[:len(kp)] = np.clip(tau, -120, 120)
        mujoco.mj_step(model, data)

    cam_ids = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                       f"k1_{side}")
               for side in ("zed_left", "zed_right")}
    renderer = Renderer(model, height=RES_H, width=RES_W)

    def render_rgb(cam_id):
        renderer.update_scene(data, camera=cam_id)
        return renderer.render()

    left = render_rgb(cam_ids["zed_left"])
    right = render_rgb(cam_ids["zed_right"])

    renderer.enable_depth_rendering()   # method call — metric float depth (H, W)
    renderer.update_scene(data, camera=cam_ids["zed_left"])
    depth = renderer.render().astype(np.float32)
    renderer.disable_depth_rendering()

    imageio.imwrite(f"{args.outdir}/k1_stereo_mjc_left.png", left)
    imageio.imwrite(f"{args.outdir}/k1_stereo_mjc_right.png", right)

    # SGBM disparity (same settings as the Isaac capture for comparability)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=96, blockSize=7,
        P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
        uniquenessRatio=8, speckleWindowSize=80, speckleRange=2,
    )
    disp = sgbm.compute(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY),
                        cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
                        ).astype(np.float32) / 16.0
    disp_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(disp, alpha=255.0 / 96.0), cv2.COLORMAP_TURBO)
    disp_vis[disp <= 0] = 0
    imageio.imwrite(f"{args.outdir}/k1_stereo_mjc_disparity.png",
                    cv2.cvtColor(disp_vis, cv2.COLOR_RGB2BGR))

    # metric depth from left camera; sky reads ~0 in reversed-Z, mask it.
    # 10 m visualization range matches the Isaac capture for comparability.
    valid = depth[(depth > 0.1) & (depth < 50.0)]
    dcol = cv2.applyColorMap((np.clip(depth / 10.0, 0, 1) * 255
                              ).astype(np.uint8), cv2.COLORMAP_MAGMA)
    imageio.imwrite(f"{args.outdir}/k1_stereo_mjc_depth.png",
                    cv2.cvtColor(dcol, cv2.COLOR_RGB2BGR))

    imageio.imwrite(f"{args.outdir}/k1_stereo_mjc_pair.png",
                    np.hstack([left, right]))

    med = float(np.median(valid)) if valid.size else float("nan")
    print("[stereo-mjc] saved:")
    for f in ("k1_stereo_mjc_left", "k1_stereo_mjc_right",
              "k1_stereo_mjc_pair", "k1_stereo_mjc_disparity",
              "k1_stereo_mjc_depth"):
        print("  ", os.path.join(args.outdir, f + ".png"))
    print(f"[stereo-mjc] fovy={fovy:.1f} deg  "
          f"baseline={args.baseline:.3f} m  median depth={med:.2f} m")


if __name__ == "__main__":
    main()
