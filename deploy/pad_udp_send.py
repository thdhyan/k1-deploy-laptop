"""Laptop gamepad -> K1 deploy over UDP: run k1_gamepad_deploy.py --udp PORT on the robot, this on the laptop.

Sends the raw pad state at RATE Hz as JSON: sticks lx, ly, rx, ry in [-1, 1] (evdev sign: down/right = +),
D-pad hx, hy, per-button press COUNTERS a, b, y, x, r3, l3 (a lost packet never loses a press) and held L1 / R1 as
l1, r1 (hand IK mode).
The robot side stops the policy (B) when packets stop for more than UDP_STALE.

  PYTHONNOUSERSITE=1 uv run --no-project --with evdev python pad_udp_send.py --host 192.168.50.82
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time

import evdev

e = evdev.ecodes
BUTTONS = {e.BTN_SOUTH: "a", e.BTN_EAST: "b", e.BTN_NORTH: "y", e.BTN_WEST: "x", e.BTN_THUMBR: "r3", e.BTN_THUMBL: "l3"}
HELD = {e.BTN_TL: "l1", e.BTN_TR: "r1"}


def read(dev, state):
  yaw = e.ABS_RX if e.ABS_RX in dict(dev.capabilities()[e.EV_ABS]) else e.ABS_Z
  axes = {e.ABS_X: "lx", e.ABS_Y: "ly", yaw: "rx", e.ABS_RY if yaw == e.ABS_RX else e.ABS_RZ: "ry"}
  for ev in dev.read_loop():
    if ev.type == e.EV_ABS and ev.code in axes:
      info = dev.absinfo(ev.code)
      state[axes[ev.code]] = 2.0 * (ev.value - info.min) / (info.max - info.min) - 1.0
    elif ev.type == e.EV_ABS and ev.code in (e.ABS_HAT0X, e.ABS_HAT0Y):
      state["hx" if ev.code == e.ABS_HAT0X else "hy"] = ev.value
    elif ev.type == e.EV_KEY and ev.code in HELD and ev.value != 2:
      state[HELD[ev.code]] = ev.value
    elif ev.type == e.EV_KEY and ev.value == 1 and ev.code in BUTTONS:
      state[BUTTONS[ev.code]] += 1


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--host", required=True, help="robot IP")
  ap.add_argument("--port", type=int, default=9870)
  ap.add_argument("--rate", type=float, default=50.0)
  a = ap.parse_args()
  dev = next(d for d in map(evdev.InputDevice, evdev.list_devices())
             if e.EV_ABS in d.capabilities() and e.BTN_SOUTH in d.capabilities().get(e.EV_KEY, []))
  state = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, hx=0, hy=0, a=0, b=0, y=0, x=0, r3=0, l3=0, l1=0, r1=0)
  threading.Thread(target=read, args=(dev, state), daemon=True).start()
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  print(f"{dev.name} -> udp {a.host}:{a.port} at {a.rate:.0f} Hz (Ctrl-C to stop; the robot then stops too)")
  while True:
    sock.sendto(json.dumps(state).encode(), (a.host, a.port))
    print(f"lx {state['lx']:+.2f} ly {state['ly']:+.2f} rx {state['rx']:+.2f} | A {state['a']} B {state['b']} L1 {state['l1']} R1 {state['r1']}", end="\r")
    time.sleep(1.0 / a.rate)
