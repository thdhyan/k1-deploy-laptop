"""Sim-to-sim for the K1 walk base: MuJoCo K1_22dof_mjlab.xml, scripted stick or live gamepad (viewer), get-up FSM test.

Split out of mjlab_k1/deploy/k1_gamepad_deploy.py (--sim). Usage:
    python gamepad_sim.py --policy flat_swing10.pt [--video walk.mp4] [--pad] [--terrain rough] [--getup getup.pt]
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time

import numpy as np
import torch

from k1_runtime import gamepad_core as _core
from k1_runtime import wbc
from k1_runtime.gamepad_core import (  # noqa: F401
    ARM_POSE,
    Body,
    DT,
    FALLS,
    GETUP_KD,
    Gamepad,
    HOME,
    KD,
    KP,
    LIM,
    PREPARE,
    PREP_KD,
    PREP_KP,
    RAMP_S,
    SCRIPT,
    Still,
    Walker,
    ZERO,
    rough_heights,
)


def run_sim(args):
  import mujoco
  import k1_description

  xml = k1_description.robot_file(k1_description.MJCF_MJLAB)
  spec = mujoco.MjSpec.from_file(str(xml))
  spec.option.timestep = 0.005
  if getattr(args, "terrain", "flat") == "rough":
    h, size = rough_heights()
    spec.add_hfield(name="terrain", size=(size, size, h.max(), 0.1), nrow=h.shape[0], ncol=h.shape[1],
                    userdata=(h / h.max()).ravel().tolist())
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain", rgba=(0.55, 0.6, 0.65, 1),
                            friction=(0.6, 0.005, 0.0001))
  else:
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(50, 50, 0.1), friction=(0.6, 0.005, 0.0001))
  spec.worldbody.add_light(pos=(0, 0, 3), dir=(0, 0, -1))
  if args.getup:  # as Mjlab-GetUp: every collision geom (group 0) touches the ground, no self-collision
    for g in spec.geoms:
      if g.group == 0 and g.parent.name != "world" and g.conaffinity == 0:
        g.contype, g.conaffinity, g.friction = 0, 1, (0.6, 0.005, 0.0001)
  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  qadr = np.array([m.joint(j).qposadr[0] for j in wbc.JOINTS])
  vadr = np.array([m.joint(j).dofadr[0] for j in wbc.JOINTS])
  act = np.array([m.actuator(j).id for j in wbc.JOINTS])

  def gains(kp, kd):  # the gains the robot gets in LowCmd
    m.actuator_gainprm[act, 0], m.actuator_biasprm[act, 1], m.actuator_biasprm[act, 2] = kp, -kp, -kd

  gyro_adr = m.sensor_adr[m.sensor("imu_ang_vel").id]
  walker = Walker(args.policy)

  body = Body(walker, args.getup)

  def step(cmd, head=(0.0, 0.0), dh=0.0, pad=None, arms=ARM_POSE):
    """One 50 Hz policy step (pad: through the get-up FSM); returns (gravity_b, body velocity (vx, vy, wz))."""
    inv = d.qpos[3:7] * np.array([1.0, -1.0, -1.0, -1.0])
    grav, vb = np.zeros(3), np.zeros(3)
    mujoco.mju_rotVecQuat(grav, np.array([0.0, 0.0, -1.0]), inv)
    gyro = d.sensordata[gyro_adr:gyro_adr + 3]
    if pad is None:
      d.ctrl[act] = walker(gyro, grav, d.qpos[qadr], d.qvel[vadr], np.asarray(cmd), head, dh, arms)
    else:
      q = d.qpos[qadr].copy()
      target, kp, kd = body(gyro, grav, q, d.qvel[vadr].copy(), pad) or (q, ZERO, GETUP_KD)  # gave up: limp
      gains(kp, kd)
      d.ctrl[act] = target
    for _ in range(4):
      mujoco.mj_step(m, d)
    mujoco.mju_rotVecQuat(vb, d.qvel[:3], inv)
    return grav, (vb[0], vb[1], d.qvel[5])

  # start like the robot: 1 s in the stock Prepare pose (straight legs), RAMP_S to HOME, then the policy
  gains(PREP_KP, PREP_KD)
  d.qpos[2] = 0.62
  d.qpos[qadr] = d.ctrl[act] = PREPARE
  for _ in range(int(1.0 / 0.005)):
    mujoco.mj_step(m, d)
  for q in np.linspace(d.qpos[qadr].copy(), HOME, int(RAMP_S / DT)):
    d.ctrl[act] = q
    for _ in range(4):
      mujoco.mj_step(m, d)
  gains(KP, KD)
  print(f"ramp done: trunk z {d.qpos[2]:.3f}")
  if args.pad:
    return run_sim_pad(m, d, walker, step, body if args.getup else None, args.keyboard)
  if args.getup:
    return run_sim_getup(m, d, body, step, qadr)
  renderer = mujoco.Renderer(m, 480, 640) if args.video else None
  frames, t0 = [], 0.0
  for name, dur, cmd in SCRIPT:
    vel = []
    for _ in range(int(dur / DT)):
      grav, v = step(cmd)
      vel.append(v)
      if grav[2] > -0.5 or d.qpos[2] < 0.3:
        print(f"FELL at t={t0:.1f}s during '{name}'")
        return 1
      if renderer and len(vel) % 2 == 0:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = d.qpos[:3], 2.0, 120, -15
        renderer.update_scene(d, cam)
        frames.append(renderer.render())
      t0 += DT
    v = np.array(vel[len(vel) // 2:]).mean(0)  # settled half
    print(f"{name:9s} cmd {np.round(cmd, 2)} -> vx {v[0]:+.2f} vy {v[1]:+.2f} wz {v[2]:+.2f}")
  if renderer:
    import imageio
    imageio.mimsave(args.video, frames, fps=25)
    print("video", args.video)
  print("sim-to-sim OK: no fall")
  return 0


def run_sim_getup(m, d, body, step, qadr):
  """Drop the robot in each FALLS pose with random joints; the FSM must get up and stand 2 s at zero command."""
  import mujoco

  rng, pad, ok = np.random.default_rng(0), Still(), 0
  for name, roll, pitch in FALLS:
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[2] = 0.32
    mujoco.mju_euler2Quat(d.qpos[3:7], np.array([roll, pitch, rng.uniform(-3.14, 3.14)]), "xyz")
    d.qpos[qadr] = np.clip(np.array(wbc.HOME) + rng.uniform(-0.6, 0.6, 22), LIM[:, 0], LIM[:, 1])
    body.state, body.tries, walked = "WALK", 0, None
    for i in range(int(25.0 / DT)):
      grav, _ = step(None, pad=pad)
      if walked is None and body.state == "WALK" and i > 0:
        walked = i * DT
      if walked is not None and i * DT > walked + 2.0:
        break
    good = walked is not None and body.state == "WALK" and grav[2] < -0.9
    ok += good
    print(f"{name:10s}: {'stood + walk base in %.1f s' % walked if good else 'FAILED (%s)' % body.state}, "
          f"trunk z {d.qpos[2]:.2f}")
  print(f"get-up {ok}/{len(FALLS)}")
  return 0 if ok == len(FALLS) else 1


def run_sim_pad(m, d, walker, step, body=None, keyboard=False):
  """Drive the sim live with the gamepad, real time, in the MuJoCo viewer. A: reset robot, B: quit.
  With a get-up FSM (body): no reset on a fall, X/Square pushes the robot over, Y/Triangle forces a get-up."""
  import mujoco
  import mujoco.viewer

  pad = Gamepad(None, keyboard=keyboard)
  with mujoco.viewer.launch_passive(m, d) as view:
    t_next = time.monotonic()
    while view.is_running() and not pad.stop:
      if pad.start or (body is None and d.qpos[2] < 0.3):
        print("reset" if pad.start else "FELL -> reset")
        pad.start = False
        mujoco.mj_resetDataKeyframe(m, d, 0)
        walker.reset()
        if body:
          body.state, body.tries = "WALK", 0
      if pad.knock:
        pad.knock = False
        a = np.random.uniform(-np.pi, np.pi)
        d.qvel[:2] += 2.5 * np.array([np.cos(a), np.sin(a)])
      if body:
        _, v = step(None, pad=pad)
      else:
        _, v = step(pad.walk_cmd(), pad.head_step(DT), pad.height_step(DT), arms=pad.arm_step(DT))
      view.cam.lookat[:] = d.qpos[:3]
      view.sync()
      print(f"cmd {np.round(pad.cmd, 2)} -> vx {v[0]:+.2f} vy {v[1]:+.2f} wz {v[2]:+.2f} | head {np.round(pad.head, 2)} dh {pad.dh:+.2f}   ",
            end="\r")
      t_next += DT
      time.sleep(max(0.0, t_next - time.monotonic()))
  return 0




if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", required=True, help="mjlab walk-base rsl-rl checkpoint (model_*.pt)")
  ap.add_argument("--video", help="write an mp4")
  ap.add_argument("--pad", action="store_true", help="drive live with the gamepad in the viewer")
  ap.add_argument("--terrain", choices=("flat", "rough"), default="flat",
                  help="rough = rough_heights() field (rough +x, stairs -x, slope +y, waves -y)")
  ap.add_argument("--keyboard", action="store_true", help="drive from this terminal's keyboard (with --pad)")
  ap.add_argument("--cap", type=float, nargs=3, default=_core.CAP, metavar=("VX", "VY", "WZ"), help="command limits (m/s, rad/s)")
  ap.add_argument("--getup", help="Mjlab-GetUp checkpoint: fall -> get-up FSM instead of Damping")
  a = ap.parse_args()
  _core.CAP = tuple(a.cap)
  sys.exit(run_sim(a))
