"""Runs the e2e_push_ik_web.py scenario (fake mocap, local UDP robots) in its own process: it patches socket.socket."""
import subprocess
import sys
from pathlib import Path


def test_push_ik_web_e2e():
  r = subprocess.run([sys.executable, str(Path(__file__).with_name("e2e_push_ik_web.py"))],
                     capture_output=True, text=True, timeout=600)
  assert r.returncode == 0 and "WEB TEST OK" in r.stdout, (r.stdout + r.stderr)[-3000:]
