"""Browser tool for the 2-K1 box push: track the 3 PhaseSpace caps, set the scene, preview it in sim, then execute.

  ros2 launch phasespace_ros2 phasespace.launch.py rviz:=false profile:="Booster"
  2026-11-02 setup: profile Booster; cap 153F (dhyan-hat, 4 LEDs on top of the head) on A3, cap 1440 (dhyan-hat-2,
  8 LEDs at the box corners) on the box, box 0.381 x 0.432 x 0.495 m (depth x width x height). Heading row on the page
  sets each thing's true yaw once its cap has a yaw reference.
  python push_ik_web.py --head-cap X Y Z              open http://127.0.0.1:8765, nothing is sent to the robots
  python push_ik_web.py --head-cap X Y Z --send       Execute really commands the robots
  python push_ik_web.py --fake                        no PhaseSpace or robots: caps are set from the page
  python push_ik_web.py --no-mocap --pad --send       no PhaseSpace: DS4 drives both robots by hand (rough test)
  python push_ik_web.py --solo --head-cap X Y Z --send one robot (default A3, caps dhyan-hat + dhyan-hat-2): the policy
                                                       is shown a ghost partner fixed sim.GHOST beside it (untrained case)

--pad adds the DS4: hold triangle = start the robot scripts, hold options = start the policy (or return to it),
share = manual takeover (hold it when nothing is running), L1 / R1 = select robot a / b (one while holding the other = both),
sticks drive the selected robot, L2 / R2 held = sticks move its left / right hand by IK, circle = stop both.

Served on localhost only: the page can command robots and has no login. Same UDP protocol and stops as
push_ik_phasespace.py. Steps: docs/k1_phasespace_field_runbook.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import push_ik_sim as sim
from k1_common.paths import models_dir
from k1_runtime.ds4 import Pad
from k1_runtime.gamepad_core import ARM_POSE, ARM_RATE, HAND_RATE, ArmJog
from k1_runtime.push_ik_core import CMD_SCALE, SUCCESS_R, SUCCESS_YAW, PushIKController, box_errors
from k1_teleop.push_ik_phasespace import ARENA_R, DT, MOCAP_STOP_S, MOCAP_ZERO_S, SETTLE_S, Link, object_pose, quat_rot

HERE = Path(__file__).resolve().parent
MODELS = models_dir() / "mjlab_k1"
BASE = MODELS / "walk_still_v2/model_8998.pt"
#: name -> checkpoint, trained full box height range (m). Depth and width are the same for all.
POLICIES = {"v10 (normal boxes)": (MODELS / "push_ik_v10/model_12496.pt", (0.30, 0.60)),
            "v8 (normal boxes)": (MODELS / "push_ik_v8/model_9497.pt", (0.30, 0.60)),
            "v9 (tall boxes)": (MODELS / "push_ik_v9/model_12496.pt", (0.71, 1.90)),
            "v7": (MODELS / "push_ik_v7/model_6998.pt", (0.30, 0.60))}
LIMITS = {"depth": (0.4, 0.7), "width": (0.8, 1.5), "mass": (2.0, 5.0), "friction": (0.3, 0.6), "goal_dist": 2.5}
#: live scene must be this close to what was previewed before Execute is allowed
MATCH_M, MATCH_DEG = 0.30, 20.0
DEADMAN_S = 3.0  # page stopped polling during a run: stop the robots
LATCH_S = 1.0
REPLY_STALE_S = 1.5  # no status from a robot for this long during a run: stop both
DEADZONE = 0.2  # as k1_gamepad_deploy.py: DS4 sticks rest at up to 0.14
SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/booster_k1"), "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=5"]
#: starts the robot script unless it already runs. It only waits for the start press: no motor is commanded yet.
ROBOT_START = ("pgrep -a python3 | grep -q gamepad_deploy && exit 0; source /opt/booster/BoosterRos2Interface/install/setup.bash; "
               "cd ~/k1_mjlab_deploy/deploy && (setsid nohup python3 -u k1_gamepad_deploy.py --policy ../walk_still_v2.pt --udp 9870 --task "
               "--log ~/push_$(date +%H%M%S).npy > ~/push.log 2>&1 < /dev/null &)")
PREVIEW_S = (10.0, 600.0)  # sim seconds of pushing a preview may be given; the robots themselves have no time limit
#: walk to box (no policy, no push): P gains, speed caps (vx fwd, vx back, vy, wz), arrive tolerance, time limit
APPROACH_K, APPROACH_KW = 0.8, 1.2
APPROACH_CAP = (0.30, 0.15, 0.15, 0.5)
APPROACH_TOL_M, APPROACH_TOL_DEG, APPROACH_S = 0.06, 8.0, 60.0
APPROACH_MIN_V = 0.08  # m/s floor on the walk speed until within APPROACH_TOL_M
ROBOT_R = 0.2  # robot footprint radius for the "box in the way" check


def wrap(a):
  return math.atan2(math.sin(a), math.cos(a))


def approach_target(box, u, standoff):
  """Trunk pose standoff m from the box centre on the -u side, facing the box centre along u."""
  return box[0] - u[0] * standoff, box[1] - u[1] * standoff, math.atan2(u[1], u[0])


def approach_cmd(a, tgt):
  """(vx, vy, wz) body frame toward tgt; turns first while the heading is far off. Also (distance, heading error)."""
  ex, ey = tgt[0] - a[0], tgt[1] - a[1]
  c, s = math.cos(a[2]), math.sin(a[2])
  bx, by = c * ex + s * ey, -s * ex + c * ey
  eyaw = wrap(tgt[2] - a[2])
  go = max(0.0, 1.0 - abs(eyaw) / 0.8)
  vx = float(np.clip(APPROACH_K * bx, -APPROACH_CAP[1], APPROACH_CAP[0])) * go
  vy = float(np.clip(APPROACH_K * by, -APPROACH_CAP[2], APPROACH_CAP[2])) * go
  wz = float(np.clip(APPROACH_KW * eyaw, -APPROACH_CAP[3], APPROACH_CAP[3]))
  sp = math.hypot(vx, vy)
  if 1e-6 < sp < APPROACH_MIN_V and math.hypot(ex, ey) > APPROACH_TOL_M:  # the walk base ignores tiny commands
    vx, vy = vx * APPROACH_MIN_V / sp, vy * APPROACH_MIN_V / sp
  return (vx, vy, wz), math.hypot(ex, ey), abs(eyaw)


class Tool:
  def __init__(self, args):
    self.args = args
    self.lock = threading.Lock()
    self.latest, self.oriented = {}, {}
    self.caps = dict(zip(("a", "box") if args.solo else ("a", "b", "box"), args.caps))
    self.yaw_fix = {"a": 0.0, "b": 0.0, "box": 0.0}  # rad added to each cap's yaw, set from the page (set_heading)
    self.kin = sim.Kinematics()
    self.preview = {"status": "none", "progress": 0.0}
    self.previewed = None
    self.form_scene = None  # last form the page sent (Execute, Preview, or each poll)
    self.preview_cancel = threading.Event()
    self.run = {"phase": "idle", "message": "", "t": 0.0, "trace": []}
    self.polled = time.monotonic()
    self.stop_flag = threading.Event()
    self.links = [Link(h) for h in args.robots]
    for link in self.links:  # press counters count up across restarts of this server
      link.a = link.b = int(time.time())
    self.ping = {h: None for h in args.robots}
    self.fake, self.fake_age = {}, 0.0
    self.status = [(0.0, {}) for _ in self.links]
    self.manual, self.ctrl, self.flash, self.pad_msg, self.stop_why = False, None, 0.0, "", None
    self.pad = Pad(self.on_button) if args.pad or args.fake else None
    self.sel, self.sel_flash, self.mocap_held, self.blend = "a", 0.0, False, None
    self.jogs = [ArmJog() for _ in self.links]
    self.approach = None  # walk to box: {"u": side unit vector, "standoff": m, "target": pose}

  # ---------------------------------------------------------------- mocap
  def start_mocap(self):
    if self.args.fake or self.args.no_mocap:
      return
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    node = rclpy.create_node("push_ik_web")

    def on_tf(msg):
      now = time.monotonic()
      for t in msg.transforms:
        if t.child_frame_id in self.args.caps:
          p, q = t.transform.translation, t.transform.rotation
          self.latest[t.child_frame_id] = (now, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    def on_groups(msg):
      self.oriented = {k: v.get("oriented", False) for k, v in json.loads(msg.data)["groups"].items()}

    node.create_subscription(TFMessage, "/tf", on_tf, 100)
    node.create_subscription(String, "/phasespace/groups", on_groups,
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

  def set_fake(self, who, pose):
    """Fake cap: place the thing itself at pose, the cap goes where the real one would be."""
    x, y, yaw = pose
    offset = self.args.head_cap if who != "box" else self.args.box_cap
    c, s = math.cos(yaw), math.sin(yaw)
    p = (x + c * offset[0] - s * offset[1], y + s * offset[0] + c * offset[1], 0.8 + offset[2])
    self.fake[who] = (p, (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))

  def pose_of(self, who):
    """(x, y, yaw, age s) of a robot trunk or the box, or None if never seen."""
    if who == "b" and self.args.solo:
      p = self.pose_of("a")
      return None if p is None else (*sim.ghost(p[:3]), p[3])
    if self.args.fake:
      if who not in self.fake:
        return None
      p, q = self.fake[who]
      age = self.fake_age
    else:
      entry = self.latest.get(self.caps[who])
      if entry is None:
        return None
      age, p, q = time.monotonic() - entry[0], entry[1], entry[2]
    if who == "box":
      x, y, yaw = object_pose(p, q, self.args.box_cap, math.radians(self.args.box_yaw) + self.yaw_fix[who])
    else:
      x, y, yaw = object_pose(p, q, self.args.head_cap, self.yaw_fix[who])
    return x, y, math.atan2(math.sin(yaw), math.cos(yaw)), age

  def box_pose3(self):
    """Full 3D box pose for the 3D view: [x, y, z, qw, qx, qy, qz] of the --box-cap point (the cap, on the top face),
    tilt included; yaw offsets (--box-yaw, heading fix) turn it about world z, as in pose_of."""
    if self.args.fake:
      p, q = self.fake["box"] if "box" in self.fake else (None, None)
    else:
      entry = self.latest.get(self.caps["box"])
      p, q = (entry[1], entry[2]) if entry else (None, None)
    if p is None:
      return None
    o = math.radians(self.args.box_yaw) + self.yaw_fix["box"]
    rz = np.array([[math.cos(o), -math.sin(o), 0.0], [math.sin(o), math.cos(o), 0.0], [0.0, 0.0, 1.0]])
    rc = quat_rot(q)
    r = rz @ rc
    c = np.asarray(p, float) - rc @ np.asarray(self.args.box_cap, float)
    tr = r[0, 0] + r[1, 1] + r[2, 2]  # rotation matrix -> quaternion (Shepperd: largest component first)
    if tr > 0:
      s = 2 * math.sqrt(1 + tr)
      qw, qx, qy, qz = s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
      s = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
      qw, qx, qy, qz = (r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
      s = 2 * math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
      qw, qx, qy, qz = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s
    else:
      s = 2 * math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
      qw, qx, qy, qz = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4
    return [float(c[0]), float(c[1]), float(c[2]), float(qw), float(qx), float(qy), float(qz)]

  def is_oriented(self, who):
    if who == "b" and self.args.solo:
      who = "a"
    return self.args.fake or bool(self.oriented.get(self.caps[who]))

  def things(self):
    """What has a cap: the ghost partner of one-robot mode has none."""
    return ("a", "box") if self.args.solo else ("a", "b", "box")

  def set_heading(self, who, deg):
    """The thing's true heading now is deg (from PhaseSpace +x): its yaw is offset so, and keeps following the cap."""
    if self.run["phase"] in ("settle", "running"):
      raise ValueError("robots are running")
    if who not in self.things():
      raise ValueError(f"'{who}' has no cap")
    if not self.is_oriented(who):
      raise ValueError(f"cap '{who}' has no yaw reference: capture it first (runbook section 1), in any heading")
    now = self.latch(who)["pose"][2]
    fix = self.yaw_fix[who] + math.radians(float(deg)) - now
    self.yaw_fix[who] = math.atan2(math.sin(fix), math.cos(fix))
    return {"fix_deg": math.degrees(self.yaw_fix[who])}

  def latch(self, who):
    """Mean pose of a cap over LATCH_S; the box cap also sets the goal this way, before it goes on the box."""
    samples, t_end = [], time.monotonic() + LATCH_S
    while time.monotonic() < t_end:
      p = self.pose_of(who)
      if p and p[3] < MOCAP_ZERO_S:
        samples.append(p[:3])
      time.sleep(DT)
    if len(samples) < 20:
      raise ValueError(f"cap for '{who}' not tracked: {len(samples)} samples in {LATCH_S} s")
    s = np.array(samples)
    return {"pose": [float(s[:, 0].mean()), float(s[:, 1].mean()), math.atan2(np.sin(s[:, 2]).mean(), np.cos(s[:, 2]).mean())],
            "spread_cm": float(s[:, :2].std(0).max() * 100), "spread_deg": float(math.degrees(s[:, 2].std()))}

  # ---------------------------------------------------------------- scene
  def warnings(self, scene):
    """Things outside what the policy trained on. Previews still run: they are how to see what that does."""
    out, full = [], [2 * float(v) for v in scene["half"]]
    lo, hi = POLICIES[scene["policy"]][1]
    for name, v, (a, b) in (("depth", full[0], LIMITS["depth"]), ("width", full[1], LIMITS["width"]), ("height", full[2], (lo, hi)),
                            ("mass", float(scene["mass"]), LIMITS["mass"]), ("friction", float(scene["friction"]), LIMITS["friction"])):
      if not a <= v <= b:
        out.append(f"{name} {v:.2f} outside trained {a:.2f}-{b:.2f}")
    d = math.hypot(scene["box"][0] - scene["goal"][0], scene["box"][1] - scene["goal"][1])
    if d > LIMITS["goal_dist"]:
      out.append(f"goal {d:.2f} m away, trained up to {LIMITS['goal_dist']} m")
    if self.args.solo:
      out.append("one robot with a ghost partner: never trained")
    elif math.hypot(scene["a"][0] - scene["b"][0], scene["a"][1] - scene["b"][1]) < 0.6:
      out.append("robots closer than 0.6 m")
    return out

  def start_preview(self, scene):
    if self.preview["status"] == "running":
      raise ValueError("a preview is already running")
    if self.run["phase"] in ("settle", "running"):
      raise ValueError("robots are running")
    if self.args.solo:
      scene = {**scene, "solo": True, "b": list(sim.ghost(scene["a"]))}
    self.form_scene = scene
    self.preview, self.previewed = {"status": "running", "progress": 0.0}, None
    self.preview_cancel.clear()
    max_s = min(PREVIEW_S[1], max(PREVIEW_S[0], float(scene.get("max_s", 180.0))))

    def job():
      try:
        res = sim.rollout(scene, str(POLICIES[scene["policy"]][0]), str(BASE), max_s=max_s,
                          progress=lambda f: self.preview.update(progress=f), cancelled=self.preview_cancel.is_set)
        if res["outcome"] == "cancelled":
          self.preview = {"status": "none", "progress": 0.0}
          return
        res["warnings"] = self.warnings(scene)
        self.preview = {"status": "done", "result": res}
        self.previewed = scene
      except Exception as e:  # shown on the page; the server keeps running
        self.preview = {"status": "error", "message": repr(e)}

    threading.Thread(target=job, daemon=True).start()

  def blockers(self, scene=None):
    """Why Execute of scene (the page form; default: the last sent or previewed one) is refused right now, [] = allowed.
    The preview outcome is only a note (preview_notes): a timeout or no preview does not block (2026-11-02)."""
    scene = scene or self.form_scene or self.previewed
    out = []
    if self.run["phase"] in ("settle", "running"):
      out.append("already running")
    if self.args.no_mocap:
      return out + ["--no-mocap: manual only"]
    out += self.not_waiting()
    if not scene:
      return out + ["no scene from the page yet: reload the page"]
    for who in self.things():
      p = self.pose_of(who)
      if p is None or p[3] > MOCAP_ZERO_S:
        out.append(f"cap '{who}' not tracked")
        continue
      if not self.is_oriented(who):
        out.append(f"cap '{who}' has no orientation reference: yaw is not measured (runbook section 1)")
      ref = scene[who]
      d, e = math.hypot(p[0] - ref[0], p[1] - ref[1]), math.degrees(math.atan2(math.sin(p[2] - ref[2]), math.cos(p[2] - ref[2])))
      if d > MATCH_M or abs(e) > MATCH_DEG:
        out.append(f"'{who}' moved {d:.2f} m, {abs(e):.0f} deg from the form: press Cap or tick Track live")
    return out

  # ---------------------------------------------------------------- robots
  def not_waiting(self):
    """Robots whose script is not waiting for the start press. Dry runs send nothing, so nothing can answer."""
    if not self.args.send:
      return []
    out = []
    for name, (t, d) in zip("ab", self.status):
      if time.monotonic() - t > REPLY_STALE_S:
        out.append(f"robot {name}: script not answering (hold triangle or press Start robot scripts)")
      elif d.get("state") != "waiting":
        out.append(f"robot {name}: {d.get('state')} {d.get('why', '')}".strip())
    return out

  def listen(self, k):
    """Status replies of one robot's k1_gamepad_deploy.py."""
    sock = self.links[k].sock
    sock.settimeout(0.5)
    while True:
      try:
        self.status[k] = (time.monotonic(), json.loads(sock.recv(1024)))
      except (TimeoutError, OSError, ValueError):  # OSError: not sending yet, so the socket has no port
        time.sleep(0.05)

  def start_scripts(self):
    if not self.args.send or self.args.fake:
      raise ValueError("needs --send and real robots")
    if self.run["phase"] in ("settle", "running"):
      raise ValueError("robots are running")

    def job(host):
      r = subprocess.run([*SSH, f"booster@{host}", ROBOT_START], capture_output=True, text=True, timeout=30)
      self.pad_msg = f"{host}: script started" if r.returncode == 0 else f"{host}: ssh failed: {r.stderr.strip()[-120:]}"

    for host in self.args.robots:
      threading.Thread(target=job, args=(host,), daemon=True).start()

  def enter_manual(self):
    """Arm IK starts from where each arm is now, so taking over does not move it."""
    for link, jog in zip(self.links, self.jogs):
      arms = link.task[2] if link.task else None
      jog.q = np.array(ARM_POSE if arms is None else arms, dtype=np.float64)
      jog.homing = False
    self.manual = True

  def start_manual(self):
    """No policy: both robots enter Custom, walk base on, sticks command them. For the rough test and for repositioning."""
    if self.run["phase"] in ("settle", "running"):
      raise ValueError("already running")
    blocked = self.not_waiting()
    if blocked:
      raise ValueError("; ".join(blocked))
    if not (self.pad and self.pad.connected):
      raise ValueError("no DS4 connected")
    self.ctrl = None
    self.enter_manual()
    self._launch(None, None, "robots ramping and standing, then manual")

  def take_manual(self):
    if self.run["phase"] not in ("settle", "running"):
      raise ValueError("nothing is running")
    if not (self.pad and self.pad.connected):
      raise ValueError("no DS4 connected")
    self.enter_manual()
    self.run.update(message="manual takeover")

  def resume_policy(self):
    if not self.ctrl:
      raise ValueError("started without a policy: stop, then Execute")
    now = [self.pose_of(w) for w in ("a", "b", "box")]
    if not all(now) or max(p[3] for p in now) > MOCAP_ZERO_S:
      raise ValueError("caps not tracked: staying in manual")
    self.ctrl.hist = None  # policy history must not span the manual part
    self.blend = [jog.q.copy() for jog in self.jogs]  # arms return at the IK rate limit, no jump
    self.manual = False
    self.run.update(message="policy on")

  def start_approach(self, goal, standoff, side, half):
    """Robot a walks to standoff m from the box centre and faces it, then stops. No policy, arms at home, no push.
    side 'push': the side away from goal (ready to push it there); 'near': the side facing robot a now."""
    if self.run["phase"] in ("settle", "running"):
      raise ValueError("already running")
    if self.args.no_mocap:
      raise ValueError("--no-mocap: walk to box needs the caps")
    blocked = self.not_waiting()
    if blocked:
      raise ValueError("; ".join(blocked))
    a, box = self.pose_of("a"), self.pose_of("box")
    for who, p in (("a", a), ("box", box)):
      if p is None or p[3] > MOCAP_ZERO_S:
        raise ValueError(f"cap '{who}' not tracked")
    if not self.is_oriented("a"):
      raise ValueError("cap 'a' has no yaw reference")
    standoff = float(standoff)
    if not 0.35 <= standoff <= 1.5:
      raise ValueError("standoff must be 0.35-1.5 m from the box centre")
    d = (goal[0] - box[0], goal[1] - box[1]) if side == "push" and goal else (box[0] - a[0], box[1] - a[1])
    n = math.hypot(*d)
    if n < 0.1:
      raise ValueError("goal is on the box: pick side 'nearest'" if side == "push" else "robot is on the box")
    u = (d[0] / n, d[1] / n)
    tgt = approach_target(box, u, standoff)
    # straight path from robot to target must not pass through the box
    clear = math.hypot(float(half[0]), float(half[1])) + ROBOT_R
    sx, sy = tgt[0] - a[0], tgt[1] - a[1]
    t = max(0.0, min(1.0, ((box[0] - a[0]) * sx + (box[1] - a[1]) * sy) / max(sx * sx + sy * sy, 1e-9)))
    if math.hypot(a[0] + t * sx - box[0], a[1] + t * sy - box[1]) < clear and math.hypot(sx, sy) > 0.05:
      raise ValueError("box is in the way of the straight path: drive around it first (DS4), or pick side 'nearest'")
    if math.hypot(sx, sy) > ARENA_R:
      raise ValueError(f"target {math.hypot(sx, sy):.1f} m away: too far")
    self.ctrl, self.manual = None, False
    self.approach = {"u": u, "standoff": standoff, "target": tgt}
    self._launch(None, None, "robot ramping and standing, then walk to box")

  def reset(self):
    """Fresh start for the page: preview and run state cleared. Heading offsets and the form stay."""
    if self.run["phase"] in ("settle", "running"):
      raise ValueError("robots are running: press STOP first")
    self.approach = None
    self.preview_cancel.set()
    self.preview, self.previewed = {"status": "none", "progress": 0.0}, None
    self.run = {"phase": "idle", "message": "", "t": 0.0, "trace": []}
    self.manual, self.mocap_held, self.blend, self.ctrl = False, False, None, None

  def request_stop(self, why="Stop pressed"):
    self.stop_flag.set()
    if self.run["phase"] not in ("settle", "running"):
      self.stop_robots(why)

  def on_button(self, name, held):
    try:
      active = self.run["phase"] in ("settle", "running")
      if name == "lost":
        if active:
          self.stop_why = "DS4 lost"
          self.request_stop("DS4 lost")
      elif name in ("l1", "r1") and not held:
        other = self.pad.r1 if name == "l1" else self.pad.l1
        self.sel = "both" if other else "a" if name == "l1" else "b"
        self.sel_flash = time.monotonic() + 0.6
      elif name == "circle" and not held:
        self.request_stop("Stop on DS4")
      elif name == "share" and active and not held:
        self.take_manual()
      elif name == "share" and held and not active:
        self.start_manual()
      elif name == "options" and held:
        self.resume_policy() if active and self.manual else self.execute()
      elif name == "triangle" and held:
        self.start_scripts()
    except ValueError as e:
      self.flash, self.pad_msg = time.monotonic() + 1.0, str(e)

  def drive_manual(self):
    """Sticks go to the selected robot, the other gets zero. L2 / R2 held: sticks move hands by IK (as the robot's own
    jog: right hand mirrors y), walking zero. Height stays where the policy left it."""
    dz = lambda v: 0.0 if abs(v) < DEADZONE else float(v)  # noqa: E731
    p, zero = self.pad, [0.0, 0.0, 0.0]
    arm_mode = p.l2 or p.r2
    walk = zero if arm_mode else [-dz(p.ly) * CMD_SCALE[0], -dz(p.lx) * CMD_SCALE[1], -dz(p.rx) * CMD_SCALE[2]]
    vel = HAND_RATE * np.array([-dz(p.ly), -dz(p.lx), -dz(p.ry)])
    for k, (link, jog) in enumerate(zip(self.links, self.jogs)):
      chosen = self.sel == "both" or self.sel == "ab"[k]
      arms = jog.step(DT, chosen and p.l2, chosen and p.r2, vel)
      link.task = (walk if chosen else zero, link.task[1] if link.task else 0.0, arms)

  def light(self):
    if time.monotonic() < self.flash:
      return "refused"
    phase = self.run["phase"]
    if time.monotonic() < self.sel_flash:  # a selection always shows, also when idle or under the policy
      return f"robot {self.sel}"
    if phase == "settle":
      return "settle"
    if phase == "running":
      if self.manual:
        return f"{'arms' if self.pad.l2 or self.pad.r2 else 'robot'} {self.sel}"
      return "mocap hold" if self.mocap_held else "policy"
    if not self.not_waiting() and self.args.send:
      return "ready"
    return "stopped" if phase == "stopped" else "not ready"

  def light_loop(self):
    while True:
      self.pad.show(self.light())
      time.sleep(0.1)

  def heartbeat(self):
    """50 Hz to both robots, always: stale packets stop them, and the stop counter is in every packet."""
    while True:
      if self.args.send:
        for link in self.links:
          try:
            link.send()
          except OSError:  # robot off the network (ARP timeout): keep the thread, it must resume when the robot is back
            pass
      time.sleep(DT)

  def pinger(self):
    while True:
      for host in self.args.robots:
        self.ping[host] = subprocess.call(["ping", "-c1", "-W1", host], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
      time.sleep(2.0)

  def stop_robots(self, why):
    # zero walk but keep the arms / height in the stop packet: the robot fades from where it is (k1_gamepad_deploy
    # FADE_S), not from its jog pose. Cleared after 1 s, so a later run's settle does not start from these arms.
    for link in self.links:
      link.task, link.b = (link.task and ((0.0, 0.0, 0.0), link.task[1], link.task[2])), link.b + 1
    stopped = [link.b for link in self.links]

    def clear():
      for link, b in zip(self.links, stopped):
        if link.b == b and self.run["phase"] not in ("settle", "running"):
          link.task = None

    threading.Timer(1.0, clear).start()
    self.run.update(phase="stopped", message=why)
    self.manual, self.mocap_held, self.blend, self.approach = False, False, None, None

  def set_form(self, scene):
    """The page's current form: what Execute runs."""
    if self.args.solo:
      scene = {**scene, "solo": True, "b": list(sim.ghost(scene["a"]))}
    self.form_scene = scene

  def preview_notes(self):
    """Not blocking: what the last preview said about this form."""
    if not self.previewed:
      return ["no preview of this form: running unpreviewed"]
    out = []
    if self.preview.get("result", {}).get("outcome") not in (None, "success"):
      out.append(f"last preview ended in {self.preview['result']['outcome']}")
    keys = ("a", "box", "goal", "half", "mass", "friction", "policy")
    if self.form_scene and any(self.form_scene.get(k) != self.previewed.get(k) for k in keys):
      out.append("form changed since the preview")
    return out

  def execute(self, scene=None):
    if scene is not None:
      self.set_form(scene)
    blocked = self.blockers()
    if blocked:
      raise ValueError("; ".join(blocked))
    scene = self.form_scene or self.previewed
    self.ctrl = PushIKController(str(POLICIES[scene["policy"]][0]), [float(v) for v in scene["half"]])
    self.manual, self.approach = False, None
    self._launch(self.ctrl, tuple(scene["goal"]), "robots ramping and standing")

  def _launch(self, ctrl, goal, message):
    self.stop_flag.clear()
    self.stop_why = None
    self.polled = time.monotonic()
    self.run = {"phase": "settle", "message": message, "t": 0.0, "trace": []}
    threading.Thread(target=self._run, args=(ctrl, goal), daemon=True).start()

  def _run(self, ctrl, goal):
    why = "stopped"
    try:
      for link in self.links:
        link.a += 1
      t_end = time.monotonic() + SETTLE_S
      while time.monotonic() < t_end and not self.stop_flag.is_set():
        time.sleep(DT)
      self.run.update(phase="running", message="manual" if self.manual else "walking to box" if self.approach else "policy on")
      t_next, n, held = time.monotonic(), 0, False
      while True:
        if self.stop_flag.is_set():
          why = self.stop_why or "Stop pressed"
          break
        if time.monotonic() - self.polled > DEADMAN_S:
          why = f"page gone for {DEADMAN_S} s"
          break
        if self.args.send:  # a robot that fell or stopped cannot leave its partner pushing
          for name, (t, d) in zip("ab", self.status):
            if d.get("state") == "exit":
              why = f"robot {name} ended: {d.get('why')}"
            elif time.monotonic() - t > REPLY_STALE_S:
              why = f"robot {name} silent for {REPLY_STALE_S} s"
          if why != "stopped":
            break
        if self.manual:
          self.drive_manual()
          n += 1
          t_next += DT
          time.sleep(max(0.0, t_next - time.monotonic()))
          continue
        if self.approach:
          ap = self.approach
          a, box = self.pose_of("a"), self.pose_of("box")
          age = max(a[3], box[3]) if a and box else math.inf
          if age > MOCAP_STOP_S:
            why = f"mocap lost for {age:.1f} s"
            break
          cmd = (0.0, 0.0, 0.0)
          if age <= MOCAP_ZERO_S:
            ap["target"] = tgt = approach_target(box, ap["u"], ap["standoff"])  # follows the box if it is moved
            cmd, d, e = approach_cmd(a[:3], tgt)
            self.run.update(t=n * DT, dist_m=round(d, 3), err_deg=round(math.degrees(e), 1), cmd=[[round(v, 2) for v in cmd]])
            if n % 5 == 0:
              self.run["trace"].append([round(v, 3) for p in (a[:3], a[:3], box[:3]) for v in p])
            if d < APPROACH_TOL_M and math.degrees(e) < APPROACH_TOL_DEG:
              why = f"ARRIVED: {d:.2f} m, {math.degrees(e):.1f} deg from the target after {n * DT:.1f} s"
              break
          self.mocap_held = age > MOCAP_ZERO_S
          if n * DT > APPROACH_S:
            why = f"walk to box timed out after {APPROACH_S:.0f} s"
            break
          for k, link in enumerate(self.links):
            link.task = (cmd if k == 0 else (0.0, 0.0, 0.0), 0.0, ARM_POSE)
          n += 1
          t_next += DT
          time.sleep(max(0.0, t_next - time.monotonic()))
          continue
        now = [self.pose_of(w) for w in ("a", "b", "box")]
        age = max(p[3] for p in now) if all(now) else math.inf
        if age > MOCAP_STOP_S:
          why = f"mocap lost for {age:.1f} s"
          break
        if age > MOCAP_ZERO_S:  # keep the arms, stop walking; reset so history is not stale when tracking returns
          for link in self.links:
            link.task = ((0.0, 0.0, 0.0), link.task[1], link.task[2]) if link.task else None
          held = self.mocap_held = True
        else:
          a, b, box = (p[:3] for p in now)
          if held:
            ctrl.hist, held, self.mocap_held = None, False, False
          cmd, dh, arms = ctrl.step(a, b, box, goal)
          if self.blend:
            arms = [b_ + np.clip(t_ - b_, -ARM_RATE * DT, ARM_RATE * DT) for b_, t_ in zip(self.blend, arms)]
            self.blend = list(arms) if max(np.abs(b_ - t_).max() for b_, t_ in zip(self.blend, arms)) > 1e-4 else None
          for k, link in enumerate(self.links):
            link.task = (cmd[k], dh[k], arms[k])
          d, e = box_errors(box, goal)
          self.run.update(t=n * DT, err_m=d, err_deg=math.degrees(e), cmd=np.round(cmd, 2).tolist(), dh=np.round(dh, 2).tolist())
          if n % 5 == 0:
            self.run["trace"].append([round(v, 3) for p in (a, b, box) for v in p])
          if d < SUCCESS_R and e < SUCCESS_YAW:
            why = f"SUCCESS: {d:.2f} m, {math.degrees(e):.1f} deg after {n * DT:.1f} s"
            break
          far = max(math.hypot(a[0] - goal[0], a[1] - goal[1]), math.hypot(b[0] - goal[0], b[1] - goal[1]))
          if far > ARENA_R:
            why = f"robot {far:.1f} m from the goal"
            break
        n += 1
        t_next += DT
        time.sleep(max(0.0, t_next - time.monotonic()))
    except Exception as e:  # never leave the robots walking on a crash here
      why = f"error: {e!r}"
    finally:
      self.stop_robots(why)

  # ---------------------------------------------------------------- page state
  def state(self):
    self.polled = time.monotonic()
    live = {}
    for who in ("a", "b", "box"):
      p = self.pose_of(who)
      live[who] = None if p is None else {"pose": p[:3], "age": p[3], "oriented": self.is_oriented(who),
                                          "fix_deg": math.degrees(self.yaw_fix[who])}
    if live["box"]:
      live["box"]["pose3"] = self.box_pose3()
    frame = None
    if live["a"] and live["b"]:
      with self.lock:
        frame = self.kin.frame(live["a"]["pose"], live["b"]["pose"])
    run = {k: v for k, v in self.run.items() if k != "trace"}
    now_t = time.monotonic()
    robots = [{"age": round(now_t - t, 1), **d} for t, d in self.status]
    pad = None if not self.pad else {"connected": self.pad.connected, "light": self.light(), "light_ok": self.pad.light_ok, "msg": self.pad_msg}
    return {"solo": self.args.solo, "ghost": sim.GHOST, "robot_status": robots, "manual": self.manual, "sel": self.sel, "pad": pad, "no_mocap": self.args.no_mocap, "live": live, "frame": frame, "run": run, "run_trace": self.run["trace"], "ping": self.ping, "send": self.args.send, "fake": self.args.fake,
            "preview": {k: v for k, v in self.preview.items() if k != "result"}, "blockers": self.blockers(), "notes": self.preview_notes(), "approach": self.approach and {"target": self.approach["target"]}, "robots": self.args.robots,
            "policies": {k: v[1] for k, v in POLICIES.items() if v[0].exists()}, "limits": LIMITS, "match": [MATCH_M, MATCH_DEG]}


def handler(tool):
  class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
      pass

    def reply(self, body, ctype="application/json", code=200):
      if not isinstance(body, bytes):
        body = json.dumps(body).encode()
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      p = self.path.split("?")[0]
      m = tool.kin.m
      if p == "/":
        return self.reply((HERE / "push_ik_web/index.html").read_bytes(), "text/html")
      if p.startswith("/vendor/") and "/" not in p[8:] and (HERE / "push_ik_web/vendor" / p[8:]).is_file():
        return self.reply((HERE / "push_ik_web/vendor" / p[8:]).read_bytes(), "text/javascript")
      if p == "/api/state":
        return self.reply(tool.state())
      if p == "/api/model":
        return self.reply(sim.model_description(m, tool.kin.h))
      if p.startswith("/api/mesh/"):
        return self.reply(sim.mesh_bytes(m, int(p[10:])), "application/octet-stream")
      if p == "/api/preview":
        return self.reply(tool.preview.get("result") or {})
      self.reply({"error": "not found"}, code=404)

    def do_POST(self):
      # a page on another site can POST to localhost; only this page's own origin may command robots
      origin = self.headers.get("Origin")
      if origin and origin != f"http://127.0.0.1:{tool.args.port}" and origin != f"http://localhost:{tool.args.port}":
        return self.reply({"error": "bad origin"}, code=403)
      body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
      try:
        if self.path == "/api/latch":
          return self.reply(tool.latch(body["who"]))
        if self.path == "/api/heading":
          return self.reply(tool.set_heading(body["who"], body["deg"]))
        if self.path == "/api/kinematics":
          with tool.lock:
            return self.reply({"frame": tool.kin.frame(body["a"], body["b"])})
        if self.path == "/api/preview":
          tool.start_preview(body)
          return self.reply({"ok": True})
        if self.path == "/api/preview/cancel":
          tool.preview_cancel.set()
          return self.reply({"ok": True})
        if self.path == "/api/execute":
          tool.execute(body.get("scene"))
          return self.reply({"ok": True})
        if self.path == "/api/form":
          tool.set_form(body)
          return self.reply({"ok": True})
        if self.path == "/api/approach":
          tool.start_approach(body.get("goal"), body["standoff"], body["side"], body["half"])
          return self.reply({"ok": True})
        if self.path == "/api/reset":
          tool.reset()
          return self.reply({"ok": True})
        if self.path == "/api/stop":
          tool.request_stop()
          return self.reply({"ok": True})
        if self.path == "/api/start_scripts":
          tool.start_scripts()
          return self.reply({"ok": True})
        if self.path == "/api/fake" and tool.args.fake:
          if "button" in body:  # DS4 stand-ins for the test
            tool.pad.connected = True
            tool.on_button(body["button"], bool(body.get("held")))
          elif "sticks" in body:
            tool.pad.connected = True
            for k, v in body["sticks"].items():
              setattr(tool.pad, k, v)
          elif "age" in body:  # all caps look this many seconds old
            tool.fake_age = float(body["age"])
          elif body.get("pose") is None:
            tool.fake.pop(body["who"], None)
          else:
            tool.set_fake(body["who"], body["pose"])
          return self.reply({"ok": True})
        self.reply({"error": "not found"}, code=404)
      except ValueError as e:
        self.reply({"error": str(e)}, code=409)

  return H


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robots", nargs="+", help="robot a, robot b IP (--solo: one IP). Default A1 A3, --solo: A3")
  ap.add_argument("--caps", nargs="+", help="robot a, robot b, box caps (--solo: robot, box). "
                  "Default dhyan-hat dhyan-hat-2 dhyan-hat-3, --solo: dhyan-hat dhyan-hat-2")
  ap.add_argument("--solo", action="store_true", help="one robot; the policy sees a ghost partner beside it (untrained)")
  ap.add_argument("--head-cap", type=float, nargs=3, metavar=("X", "Y", "Z"),
                  help="MEASURED cap frame origin in the trunk frame, head at zero (m). Required unless --fake")
  ap.add_argument("--box-cap", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                  help="cap frame origin in the box frame (box centre = 0); z is not used")
  ap.add_argument("--box-yaw", type=float, default=0.0, help="box x axis from the cap x axis, deg")
  ap.add_argument("--send", action="store_true", help="really send to the robots. Without it Execute runs everything but sends nothing")
  ap.add_argument("--fake", action="store_true", help="no PhaseSpace: caps are placed from the page")
  ap.add_argument("--no-mocap", action="store_true", help="no PhaseSpace: manual driving with --pad only")
  ap.add_argument("--pad", action="store_true", help="use the DS4 on this laptop")
  ap.add_argument("--port", type=int, default=8765)
  args = ap.parse_args()
  n = 1 if args.solo else 2
  args.robots = args.robots or (["192.168.50.230"] if args.solo else ["192.168.50.82", "192.168.50.230"])
  args.caps = args.caps or (["dhyan-hat", "dhyan-hat-2"] if args.solo else ["dhyan-hat", "dhyan-hat-2", "dhyan-hat-3"])
  if len(args.robots) != n or len(args.caps) != n + 1:
    ap.error(f"{'--solo takes' if args.solo else 'needs'} {n} robot IP(s) and {n + 1} caps")
  if args.head_cap is None:
    if not args.fake and not args.no_mocap:
      ap.error("--head-cap is required: cap position error becomes robot position error")
    args.head_cap = (0.0, 0.0, 0.0)
  if not args.fake and not args.no_mocap and os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
    raise SystemExit("export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST first: DDS over Wi-Fi OOMs the K1")

  import torch
  torch.set_num_threads(2)
  tool = Tool(args)
  tool.start_mocap()
  threading.Thread(target=tool.heartbeat, daemon=True).start()
  threading.Thread(target=tool.pinger, daemon=True).start()
  for k in range(len(tool.links)):
    threading.Thread(target=tool.listen, args=(k,), daemon=True).start()
  if tool.pad:
    threading.Thread(target=tool.light_loop, daemon=True).start()
  if args.pad:
    tool.pad.start_evdev()
  server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(tool))
  print(f"http://127.0.0.1:{args.port}  {'SENDS TO ROBOTS ' + ' '.join(args.robots) if args.send else 'dry: sends nothing'}", flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    tool.stop_flag.set()
    tool.stop_robots("server closed")
    if args.send:
      for _ in range(25):
        for link in tool.links:
          link.send()
        time.sleep(DT)
    print("stop sent")


if __name__ == "__main__":
  main()
