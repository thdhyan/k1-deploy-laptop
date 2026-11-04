"""CPU MuJoCo preview of a PushIK run: real deploy code (PushIKController + walk base) on 2 K1s and a box, from given poses.

Same start as the robots (Prepare pose, ramp, stand), same stops as push_ik_web.py. Mocap is delayed 2 steps, no noise.
Collision groups as tasks_push_ma.robot_cfg. Not the training engine (MuJoCo Warp): a preview, not a guarantee.
"""
from __future__ import annotations

import math
from collections import deque
import k1_description
import mujoco
import numpy as np

from k1_runtime import gamepad_core as D
from k1_runtime.push_ik_core import SUCCESS_R, SUCCESS_YAW, PushIKController, box_errors

XML = k1_description.robot_file(k1_description.MJCF_MJLAB)
PHYS_DT, DECIM = 0.005, 4
DT = PHYS_DT * DECIM
STAND_S = 2.0        # with the 2 s ramp = push_ik_web.SETTLE_S before the policy moves
MOCAP_DELAY = 2      # training: 1-3 steps
TIP_DEG = 30.0       # tasks_push_ik.box_tip
ARENA_R = 4.5        # push_ik_phasespace.ARENA_R
AGENTS = ("a", "b")
FEET = ("left_foot1_collision", "right_foot1_collision")
GHOST = (0.0, -0.8)  # one-robot mode: the policy's partner, this far in robot a's frame (m), moving with it
PARKED = (30.0, 30.0)  # one-robot mode: where the sim keeps robot b, standing, out of the way


def ghost(a, offset=GHOST):
  """One-robot mode: pose of the partner the policy is shown, fixed beside robot a (it has no body)."""
  c, s = math.cos(a[2]), math.sin(a[2])
  return (a[0] + c * offset[0] - s * offset[1], a[1] + s * offset[0] + c * offset[1], a[2])


def build(half=(0.25, 0.6, 0.2), mass=3.0, friction=0.45):
  root = mujoco.MjSpec()
  root.option.timestep, root.option.impratio = PHYS_DT, 10
  root.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(50, 50, 0.1), friction=(0.1, 0.005, 0.0001))
  for k, name in enumerate(AGENTS):
    child = mujoco.MjSpec.from_file(str(XML))
    child.option.timestep, child.option.tolerance = PHYS_DT, root.option.tolerance
    for key in list(child.keys):
      child.delete(key)
    bit = 2 << k
    for g in child.geoms:
      if g.group == 1:
        continue
      is_foot = g.parent.name in ("left_foot_link", "right_foot_link") and g.type == mujoco.mjtGeom.mjGEOM_BOX
      g.contype = 0 if is_foot else bit
      g.conaffinity = 1 if is_foot or k == 0 else 1 | 2
      g.condim, g.priority = 3, 1 if is_foot else 0
      g.friction = (0.6, 0.005, 0.0001)
    root.attach(child, prefix=f"{name}_", frame=root.worldbody.add_frame())
  body = root.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(half), mass=mass, priority=1, condim=3,
                friction=(friction, 0.005, 0.0001), contype=1, conaffinity=1, rgba=(0.8, 0.5, 0.2, 1))
  root.worldbody.add_light(pos=(0, 0, 4), dir=(0, 0, -1))
  return root.compile()


class Handles:
  """qpos / actuator indices of the two robots and the box."""

  def __init__(self, m):
    self.root, self.q, self.v, self.act, self.gyro = [], [], [], [], []
    for n in AGENTS:
      self.root.append(m.joint(next(m.joint(i).name for i in range(m.njnt) if m.joint(i).name.startswith(f"{n}_")
                                    and m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)).qposadr[0])
      self.q.append(np.array([m.joint(f"{n}_{j}").qposadr[0] for j in D.wbc.JOINTS]))
      self.v.append(np.array([m.joint(f"{n}_{j}").dofadr[0] for j in D.wbc.JOINTS]))
      self.act.append(np.array([m.actuator(f"{n}_{j}").id for j in D.wbc.JOINTS]))
      self.gyro.append(m.sensor_adr[m.sensor(f"{n}_imu_ang_vel").id])
    self.box = m.joint("box_joint").qposadr[0]
    # the bodies that carry a visual mesh, the only ones the browser needs
    self.geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1]
    self.bodies = sorted({int(m.geom_bodyid[g]) for g in self.geoms})


def set_pose(d, adr, xy, yaw, z):
  d.qpos[adr:adr + 3] = xy[0], xy[1], z
  d.qpos[adr + 3:adr + 7] = math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)


def get_pose(d, adr):
  w, x, y, z = d.qpos[adr + 3:adr + 7]
  return float(d.qpos[adr]), float(d.qpos[adr + 1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(d, adr):
  w, x, y, z = d.qpos[adr + 3:adr + 7]
  return math.degrees(math.acos(min(1.0, abs(1 - 2 * (x * x + y * y)))))


def body_frame(d, h):
  """Flat world poses of h.bodies: x, y, z, qw, qx, qy, qz per body."""
  return np.round(np.concatenate([d.xpos[h.bodies], d.xquat[h.bodies]], 1), 3).ravel().tolist()


def model_description(m, h):
  """Everything the browser needs to draw the robots: per geom body slot, mesh id, local pose, colour."""
  slot = {b: i for i, b in enumerate(h.bodies)}
  return {"bodies": len(h.bodies), "geoms": [
    {"body": slot[int(m.geom_bodyid[g])], "mesh": int(m.geom_dataid[g]), "pos": m.geom_pos[g].tolist(),
     "quat": m.geom_quat[g].tolist(), "rgba": m.geom_rgba[g].tolist()} for g in h.geoms],
    "meshes": {int(m.geom_dataid[g]): [int(m.mesh_vertnum[m.geom_dataid[g]]), int(m.mesh_facenum[m.geom_dataid[g]])]
               for g in h.geoms}}


def mesh_bytes(m, i):
  v, f = m.mesh_vertadr[i], m.mesh_faceadr[i]
  return (m.mesh_vert[v:v + m.mesh_vertnum[i]].astype(np.float32).tobytes()
          + m.mesh_face[f:f + m.mesh_facenum[i]].astype(np.int32).tobytes())


class Kinematics:
  """Robots standing at HOME at given poses, for showing the scene without simulating it."""

  def __init__(self):
    self.m = build()
    self.h = Handles(self.m)
    self.d = mujoco.MjData(self.m)

  def frame(self, a, b):
    h, d = self.h, self.d
    for k, pose in enumerate((a, b)):
      set_pose(d, h.root[k], pose[:2], pose[2], 0.52)
      d.qpos[h.q[k]] = D.HOME
    mujoco.mj_kinematics(self.m, d)
    return body_frame(d, h)


def rollout(scene, policy, base, max_s=180.0, progress=None, cancelled=lambda: False):
  """scene: a, b, box, goal = (x, y, yaw), half = half sizes, mass, friction. Returns outcome, trace and body frames."""
  half = [float(v) for v in scene["half"]]
  m = build(half, float(scene["mass"]), float(scene["friction"]))
  h, d = Handles(m), mujoco.MjData(m)
  goal = tuple(scene["goal"])
  solo = bool(scene.get("solo"))  # robot b is parked and stands still; the policy sees a ghost partner beside a
  poses = (scene["a"], (*PARKED, 0.0) if solo else scene["b"])
  for k in range(2):
    set_pose(d, h.root[k], poses[k][:2], poses[k][2], 0.62)
    d.qpos[h.q[k]] = d.ctrl[h.act[k]] = D.PREPARE
  set_pose(d, h.box, scene["box"][:2], scene["box"][2], half[2] + 0.005)

  kp, kd = D.KP.copy(), D.KD.copy()
  kp[2:10], kd[2:10] = 40.0, 2.0  # k1_gamepad_deploy --task

  def gains(p, v):
    for act in h.act:
      m.actuator_gainprm[act, 0], m.actuator_biasprm[act, 1], m.actuator_biasprm[act, 2] = p, -p, -v

  walkers = [D.Walker(base) for _ in AGENTS]
  ctrl = PushIKController(policy, half)
  frames, trace, n = [], [], 0
  outcome, detail = "timeout", f"not done after {max_s:.0f} s of pushing"
  t_policy = 1.0 + D.RAMP_S + STAND_S
  history = deque(maxlen=MOCAP_DELAY + 1)
  ramp = np.linspace(D.PREPARE, D.HOME, int(D.RAMP_S / DT))
  gains(D.PREP_KP, D.PREP_KD)
  total = int((t_policy + max_s) / DT)
  cmd, dh, arms = np.zeros((2, 3)), np.zeros(2), np.tile(D.ARM_POSE, (2, 1))
  a, b, box = None, None, None

  def record(t):
    trace.append([round(t, 2)] + [round(v, 3) for p in (a, b, box) for v in p] + np.round(cmd, 2).ravel().tolist())
    frames.append({"t": round(t, 2), "b": body_frame(d, h), "box": np.round(np.r_[d.qpos[h.box:h.box + 7]], 3).tolist()})

  for n in range(total):
    t = n * DT
    if cancelled():
      return {"outcome": "cancelled"}
    a, b, box = (get_pose(d, h.root[0]), get_pose(d, h.root[1]), get_pose(d, h.box))
    if solo:
      b = ghost(a)
    if t < 1.0:
      pass  # Prepare pose held
    elif n - int(1.0 / DT) < len(ramp):
      for k in range(2):
        d.ctrl[h.act[k]] = ramp[n - int(1.0 / DT)]
    else:
      if n - int(1.0 / DT) == len(ramp):
        gains(kp, kd)
        for w in walkers:
          w.reset()
      history.append((a, b, box))
      if t >= t_policy:
        seen = history[0]
        cmd, dh, arms = ctrl.step(seen[0], seen[1], seen[2], goal)
        if solo:
          cmd[1], dh[1], arms[1] = 0.0, 0.0, D.ARM_POSE
        err, yaw_err = box_errors(box, goal)
        if err < SUCCESS_R and yaw_err < SUCCESS_YAW:
          outcome, detail = "success", f"{err:.2f} m, {math.degrees(yaw_err):.1f} deg"
        elif tilt_deg(d, h.box) > TIP_DEG:
          outcome, detail = "box_tip", f"{tilt_deg(d, h.box):.0f} deg"
        elif max(math.hypot(p[0] - goal[0], p[1] - goal[1]) for p in (a, b)) > ARENA_R:
          outcome, detail = "arena", "a robot left the arena"
      for k in range(2):
        inv, grav = d.qpos[h.root[k] + 3:h.root[k] + 7] * np.array((1.0, -1.0, -1.0, -1.0)), np.zeros(3)
        mujoco.mju_rotVecQuat(grav, np.array((0.0, 0.0, -1.0)), inv)
        if grav[2] > -0.5 and not (solo and k == 1):
          outcome, detail = "fell", f"robot {AGENTS[k]}"
        c = np.clip(cmd[k], -np.array(D.CAP), np.array(D.CAP)).astype(np.float32)
        d.ctrl[h.act[k]] = walkers[k](d.sensordata[h.gyro[k]:h.gyro[k] + 3], grav, d.qpos[h.q[k]], d.qvel[h.v[k]], c,
                                      (0.0, 0.0), float(np.clip(dh[k], D.wbc.H_MIN - D.wbc.H_STAND, 0.0)), arms[k])
    if n % 2 == 0:
      record(t)
    if progress and n % 25 == 0:
      progress(n / total)
    if outcome != "timeout":
      break
    for _ in range(DECIM):
      mujoco.mj_step(m, d)
  record(n * DT)
  err, yaw_err = box_errors(box, goal)
  return {"outcome": outcome, "detail": detail, "t_policy": t_policy, "t_end": round(n * DT, 2),
          "push_s": round(max(0.0, n * DT - t_policy), 2), "err_m": round(err, 3), "err_deg": round(math.degrees(yaw_err), 1),
          "end": {"a": list(a), "b": list(b), "box": list(box)}, "trace": trace, "frames": frames,
          "box_z": half[2]}
