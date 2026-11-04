"""End to end without hardware: push_ik_web.py --fake --send -> UDP -> the robot's Gamepad parser.
  python tests/e2e_push_ik_web.py   (pytest runs it via tests/test_push_ik_web.py)
Scenario: latch goal with the box cap, preview, refused while the scene moved, execute, commands arrive, stop arrives."""
import json
import math
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
from k1_runtime.ds4 import COLOURS, Pad  # noqa: E402
from k1_runtime.gamepad_core import Gamepad  # noqa: E402

PORT = 8799
URL = f"http://127.0.0.1:{PORT}"


def get(path):
  return json.load(urllib.request.urlopen(URL + path, timeout=10))


def post(path, body=None, headers=None, body_ok=True):
  req = urllib.request.Request(URL + path, json.dumps(body or {}).encode(), headers or {})
  try:
    return json.load(urllib.request.urlopen(req, timeout=30))
  except urllib.error.HTTPError as e:
    if not body_ok:
      raise
    raise AssertionError(f"{path}: {e.read()}") from e


def refused(path, body=None, headers=None):
  try:
    post(path, body, headers, body_ok=False)
  except urllib.error.HTTPError as e:
    return json.load(e)["error"]
  raise AssertionError(f"{path} should have been refused")


def wait_preview():
  while (s := get("/api/state"))["preview"]["status"] == "running":
    time.sleep(0.2)
  assert s["preview"]["status"] == "done", s["preview"]
  return get("/api/preview")


peers = {}  # robot a and b are one local parser here: answer every sender, keep each one's last packet (a came first)


class TapSocket(socket.socket):
  def recvfrom(self, *a):
    data, peer = super().recvfrom(*a)
    peers[peer] = json.loads(data)
    return data, peer


socket.socket = TapSocket
pad = Gamepad(None, 9870)
robot = {"state": "waiting"}  # stands in for run_robot's report thread


def report():
  while not robot.get("silent"):
    if pad.sock:
      for p in list(peers):
        pad.sock.sendto(json.dumps({**robot, "gap_ms": 0}).encode(), p)
    time.sleep(0.02)


threading.Thread(target=report, daemon=True).start()
proc = subprocess.Popen([sys.executable, str(DEPLOY / "push_ik_web.py"), "--fake", "--send", "--port", str(PORT),
                         "--robots", "127.0.0.1", "127.0.0.1"])
try:
  for _ in range(100):
    try:
      s = get("/api/state")
      break
    except (urllib.error.URLError, ConnectionError):
      time.sleep(0.2)
  # no form yet blocks; robots that have not answered yet may too; no preview is only a note (2026-11-02)
  assert s["blockers"][-1] == "no scene from the page yet: reload the page", s["blockers"]
  assert all(b.startswith("robot ") for b in s["blockers"][:-1]), s["blockers"]
  assert s["notes"] == ["no preview of this form: running unpreviewed"], s["notes"]
  model = get("/api/model")
  assert model["bodies"] > 20 and len(model["geoms"]) > 20
  v, f = next(iter(model["meshes"].items()))[1]
  mesh_id = next(iter(model["meshes"]))
  assert len(urllib.request.urlopen(f"{URL}/api/mesh/{mesh_id}").read()) == 12 * v + 12 * f
  assert "bad origin" in refused("/api/execute", {}, {"Origin": "http://evil.example"})

  # 1. cap at goal, 2. cap on box
  post("/api/fake", {"who": "box", "pose": [1.0, 0.2, 0.17]})
  r = post("/api/latch", {"who": "box"})
  assert abs(r["pose"][0] - 1.0) < 1e-6 and abs(r["pose"][2] - 0.17) < 1e-6, r
  goal = r["pose"]
  a, b, box = [-0.75, 0.4, 0.0], [-0.75, -0.4, 0.0], [0.0, 0.0, 0.0]
  for who, p in (("box", box), ("a", a), ("b", b)):
    post("/api/fake", {"who": who, "pose": p})
  scene = {"a": a, "b": b, "box": box, "goal": goal, "half": [0.25, 0.6, 0.2], "mass": 3, "friction": 0.45,
           "policy": "v8 (normal boxes)"}

  # 3. preview
  post("/api/preview", scene)
  res = wait_preview()
  print({k: res[k] for k in ("outcome", "detail", "push_s", "err_m", "err_deg")}, len(res["frames"]), "frames")
  assert res["outcome"] == "success", res["outcome"]
  s = get("/api/state")
  assert s["blockers"] == [] and s["notes"] == [], (s["blockers"], s["notes"])

  # out of range scenes warn
  post("/api/preview", {**scene, "half": [0.25, 0.6, 0.6]})
  assert any("height" in w for w in wait_preview()["warnings"])
  post("/api/preview", scene)
  wait_preview()

  # the preview limit comes from the page: a short one ends in timeout (a note, not a blocker), a long one can be cancelled
  post("/api/preview", {**scene, "max_s": 10})
  res = wait_preview()
  assert res["outcome"] == "timeout" and res["push_s"] < 10.5, (res["outcome"], res["push_s"])
  s = get("/api/state")
  assert "last preview ended in timeout" in s["notes"] and s["blockers"] == [], (s["notes"], s["blockers"])
  post("/api/preview", {**scene, "max_s": 600})
  post("/api/preview/cancel", {})
  for _ in range(50):
    if (s := get("/api/state"))["preview"]["status"] != "running":
      break
    time.sleep(0.2)
  assert s["preview"]["status"] == "none", s["preview"]
  post("/api/preview", scene)
  wait_preview()

  # execute is refused when the scene moved away from the preview
  post("/api/fake", {"who": "a", "pose": [-1.5, 0.4, 0.0]})
  assert "moved" in refused("/api/execute")
  post("/api/fake", {"who": "a", "pose": a})
  # and when a cap vanished
  post("/api/fake", {"who": "b", "pose": None})
  assert "not tracked" in refused("/api/execute")
  post("/api/fake", {"who": "b", "pose": b})

  # 4. execute: start press, then commands
  assert pad.start is False
  post("/api/execute")
  time.sleep(0.5)
  assert pad.start, "start press did not arrive"
  assert pad.stop is False
  for _ in range(100):  # keeps polling, as the page does
    run = get("/api/state")["run"]
    if run["phase"] == "running" and "cmd" in run:
      break
    time.sleep(0.1)
  time.sleep(0.5)
  get("/api/state")
  assert pad.task is not None and pad.task["arms"] is not None, pad.task
  cmd, dh, arms = pad.walk_cmd(), pad.height_step(0.02), pad.arm_step(0.02)
  print("robot side: cmd", cmd, "dh", dh, "arms", arms)
  assert arms.shape == (8,) and -0.13 < dh <= 0.0

  # 5. stop
  post("/api/stop")
  t0 = time.time()
  while not pad.stop and time.time() - t0 < 2.0:
    time.sleep(0.02)
  print(f"stop after {time.time() - t0:.2f} s:", get("/api/state")["run"]["message"])
  # the stop packet zeroes the walk but keeps the arms (the robot fades from where it is), cleared after 1 s
  assert pad.stop and pad.task["arms"] is not None and not any(pad.task["cmd"]), pad.task
  time.sleep(1.3)
  assert pad.task["arms"] is None and not any(pad.task["cmd"]), pad.task

  # 6. deadman: the page stops polling during a run
  pad.stop = False
  post("/api/execute")
  time.sleep(9.0)  # settle + deadman, no polling
  assert pad.stop, "no stop after the page went away"
  assert "page gone" in get("/api/state")["run"]["message"]

  # 7. mocap loss during a run: walk command zero but arms held at 0.25 s, stop at 1 s
  pad.stop = False
  post("/api/execute")
  while (run := get("/api/state")["run"])["phase"] != "running" or "cmd" not in run:
    time.sleep(0.1)
  post("/api/fake", {"age": 0.5})
  time.sleep(0.3)
  s = get("/api/state")
  assert s["run"]["phase"] == "running" and not pad.stop, s["run"]
  assert not pad.walk_cmd().any() and pad.task["arms"] is not None, pad.task
  post("/api/fake", {"age": 2.0})
  time.sleep(0.3)
  assert pad.stop and "mocap lost" in get("/api/state")["run"]["message"]
  post("/api/fake", {"age": 0.0})

  # 8. Stop during the settle: never reaches the policy
  pad.stop = False
  post("/api/execute")
  time.sleep(0.5)
  post("/api/stop")
  time.sleep(0.3)
  assert pad.stop and not any(pad.task["cmd"]) and "cmd" not in get("/api/state")["run"], pad.task
  time.sleep(1.0)  # stop packet arms cleared after 1 s
  assert not pad.task["arms"], pad.task

  # 9. DS4: hold options starts the policy, share takes over, L1 gates the stick, options returns to the policy
  pad.stop, pad.start = False, False
  post("/api/fake", {"button": "options", "held": True})
  while (run := get("/api/state")["run"])["phase"] != "running" or "cmd" not in run:
    time.sleep(0.1)
  assert pad.start and get("/api/state")["pad"]["light"] == "policy"
  policy_arms = pad.task["arms"]
  post("/api/fake", {"button": "share"})
  time.sleep(0.2)
  s = get("/api/state")
  assert s["manual"] and s["pad"]["light"] == "robot a", s["pad"]
  assert not pad.walk_cmd().any(), pad.task  # nothing held: zero
  assert pad.task["arms"] is not None and pad.task["arms"] != [], pad.task
  held_arms = pad.task["arms"]
  time.sleep(0.2)
  assert pad.task["arms"] == held_arms, "arms moved in manual"
  post("/api/fake", {"sticks": {"ly": -1.0, "lx": 0.0, "rx": 0.0}})
  post("/api/fake", {"button": "l1"})
  assert get("/api/state")["pad"]["light"] == "robot a"
  time.sleep(0.2)
  cmd_a, cmd_b = [m["task"]["cmd"] for m in peers.values()]
  print("manual cmd a", cmd_a, "b", cmd_b)
  assert cmd_a == [0.5, 0.0, 0.0] and cmd_b == [0.0, 0.0, 0.0], (cmd_a, cmd_b)
  post("/api/fake", {"button": "r1"})
  assert get("/api/state")["sel"] == "b" and get("/api/state")["pad"]["light"] == "robot b"
  time.sleep(0.2)
  cmd_a, cmd_b = [m["task"]["cmd"] for m in peers.values()]
  assert cmd_a == [0.0, 0.0, 0.0] and cmd_b == [0.5, 0.0, 0.0], (cmd_a, cmd_b)
  # one held while pressing the other = both
  post("/api/fake", {"sticks": {"r1": True}})
  post("/api/fake", {"button": "l1"})
  post("/api/fake", {"sticks": {"r1": False}})
  time.sleep(0.2)
  cmd_a, cmd_b = [m["task"]["cmd"] for m in peers.values()]
  assert get("/api/state")["sel"] == "both" and cmd_a == cmd_b == [0.5, 0.0, 0.0], (cmd_a, cmd_b)
  # L2 on robot b: walking zero, only its left arm moves, then options brings the arms back without a jump
  post("/api/fake", {"button": "r1"})
  before = [m["task"]["arms"] for m in peers.values()]
  post("/api/fake", {"sticks": {"l2": True}})
  time.sleep(0.8)
  assert get("/api/state")["pad"]["light"] == "arms b"
  after = [m["task"]["arms"] for m in peers.values()]
  cmd_a, cmd_b = [m["task"]["cmd"] for m in peers.values()]
  assert cmd_a == cmd_b == [0.0, 0.0, 0.0], (cmd_a, cmd_b)
  assert after[0] == before[0] and after[1][4:] == before[1][4:] and after[1][:4] != before[1][:4], (before, after)
  post("/api/fake", {"sticks": {"ly": 0.0, "l2": False}})
  time.sleep(0.1)
  jogged = [m["task"]["arms"] for m in peers.values()]
  post("/api/fake", {"button": "options", "held": True})
  time.sleep(0.3)
  s = get("/api/state")
  assert not s["manual"] and s["run"]["message"] == "policy on" and policy_arms is not None, s["run"]
  back = [m["task"]["arms"] for m in peers.values()]
  assert max(abs(x - y) for p_, q_ in zip(jogged, back) for x, y in zip(p_, q_)) < 0.8, (jogged, back)
  # 10. circle stops, robot packets end the run for both
  post("/api/fake", {"button": "circle"})
  time.sleep(0.3)
  assert pad.stop and "DS4" in get("/api/state")["run"]["message"] or "Stop" in get("/api/state")["run"]["message"]
  pad.stop = False
  post("/api/execute")
  for _ in range(25):  # as the page does: past the settle, deadman not firing
    time.sleep(0.2)
    get("/api/state")
  robot.update(state="exit", why="fall (gravity_z -0.1)")
  time.sleep(0.3)
  s = get("/api/state")
  assert pad.stop and "fall" in s["run"]["message"], s["run"]
  assert any("exit" in b for b in s["blockers"]), s["blockers"]  # no Execute until the script is back
  robot.update(state="waiting", why="")
  time.sleep(0.2)  # the status packet has to reach the tool
  # 11. a silent robot stops the run, and blocks Execute
  pad.stop = False
  post("/api/execute")
  for _ in range(25):  # as the page does: past the settle, deadman not firing
    time.sleep(0.2)
    get("/api/state")
  robot["silent"] = True
  for _ in range(20):
    time.sleep(0.2)
    s = get("/api/state")
    if s["run"]["phase"] == "stopped":
      break
  assert "silent" in s["run"]["message"] and pad.stop, s["run"]
  assert any("not answering" in b for b in s["blockers"]), s["blockers"]

  # 12. light bar files, and hold fires once on top of the press
  events = []
  p = Pad(lambda *e: events.append(e))
  with tempfile.TemporaryDirectory() as d:
    for f in ("multi_intensity", "brightness"):
      Path(d, f).touch()
    p.use_led(d)
    assert p.light_ok
    p.show("robot b")
    assert Path(d, "multi_intensity").read_text() == " ".join(map(str, COLOURS["robot b"])) and Path(d, "brightness").read_text() == "255"
  p.button("options", True)
  time.sleep(1.3)
  p.button("options", False)
  assert events == [("options", False), ("options", True)], events
finally:
  proc.terminate()
  proc.wait(timeout=10)
print("WEB TEST OK")
