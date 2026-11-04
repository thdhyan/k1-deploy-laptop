#!/usr/bin/env python3
"""Locomotion Node — runs the trained RSL-RL velocity policy for Booster K1.

Backend-agnostic: subscribes the standard fleet endpoints published by every
sim backend (Gazebo Harmonic, MuJoCo, Isaac Sim) and the real robot stack.

  Sub: /{robot_ns}/joint_states     (sensor_msgs/JointState)   @ 50 Hz
  Sub: /{robot_ns}/cmd_vel          (geometry_msgs/Twist)
  Sub: /{robot_ns}/imu              (sensor_msgs/Imu)          [optional]
  Sub: /{robot_ns}/odom             (nav_msgs/Odometry)        [optional]
  Pub: /{robot_ns}/joint_commands   per `command_type` param:
         - 'k1_interfaces/JointCommand'  -> Gazebo (sim_bridge), MuJoCo fleet
         - 'sensor_msgs/JointState'      -> Isaac Sim fleet (bundled stack
                                            cannot import workspace msgs)
  Pub: /{robot_ns}/policy_obs       (k1_interfaces/PolicyObs)  [debug]

Observation layout — MUST match training exactly
(isaac_tasks/k1_velocity/.../velocity_env_cfg.py :: ObservationsCfg.PolicyCfg,
concatenate_terms=True, declaration order):

    [ 0: 3]  base_lin_vel        (base frame; odom or zeros)
    [ 3: 6]  base_ang_vel        (base frame; IMU or zeros)
    [ 6: 9]  projected_gravity   (unit gravity in base frame; IMU or [0,0,-1])
    [ 9:12]  velocity command    (vx, vy, wz from cmd_vel)
    [12:24]  leg joint pos       (pos - default, LEG_JOINTS order)
    [24:36]  leg joint vel
    [36:48]  last raw action
    [48:50]  phase clock         [sin(2*pi*phase), cos(2*pi*phase)]

OBS_DIM is 50 = 48 proprioception + the 2-dim gait phase clock (gait lever 3,
k1_velocity/tasks/velocity/gait_clock.py). Exports made before that change are
48-dim and are REFUSED on load — a clock-less policy fed a clock is a layout
lie, and the retrained export is the only policy that belongs here.

The 10-step observation HISTORY is maintained HERE in the node (rolling
buffer, latest step last) — sim backends send single frames only. The current
policy consumes the latest 50-dim step; the full 500-dim flattened history is
published on `policy_obs` for debugging and for future history-stacked /
recurrent policies.

Action: 12 leg joint position targets = default(0) + action_scale * action.
Upper body (head/arms) is NOT commanded here — regulated separately per
workspace convention.

Params:
  robot_ns:       'k1_0'
  policy_path:    TorchScript policy ('' = idle plumbing mode, no inference)
  control_freq:   50.0 Hz (training decimation 4 x 200 Hz physics)
  action_scale:   0.25 (JointPositionActionCfg.scale, use_default_offset=True)
  command_type:   'k1_interfaces/JointCommand' | 'sensor_msgs/JointState'
  cmd_timeout:    0.5 s without cmd_vel -> zero velocity command
  js_timeout:     0.5 s without joint_states -> stop publishing (safety)
  imu_topic:      '' (disabled) | e.g. 'imu' -> /{ns}/imu
  odom_topic:     '' (disabled) | e.g. 'odom' -> /{ns}/odom
  history_len:    10
  input_mode:     'latest'  -> policy(x) with the newest 50-dim step
                  'stacked' -> policy(x) with 50x10 term-major history stack
                  (oldest->newest per term), matching IsaacLab
                  ObservationManager group-level history_length=10 — use with
                  the distilled student policy (Isaac-Velocity-Distill-K1-v0)
  publish_obs_debug: True

Torch runtime: the node lazy-imports torch (TorchScript). On the system ROS
python install CPU torch with:
    pip install --user torch --index-url https://download.pytorch.org/whl/cpu
or run this node with the venv-isaac interpreter plus the system ROS package
path (both are Python 3.12):
    PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH \
      /home/thakk100/Projects/IsaacLab/.venv-isaac/bin/python3.12 \
      -m k1_locomotion.locomotion_node --ros-args -p robot_ns:=k1_0
"""
import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from k1_interfaces.msg import JointCommand, PolicyObs

# 12 leg joints in EXACT policy order (= K1_LEG_JOINTS in velocity_env_cfg.py)
LEG_JOINTS = [
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll',
]

# Default leg joint positions at training (BOOSTER_K1_CFG init_state: legs all 0)
DEFAULT_LEG_POS = np.zeros(12, dtype=np.float32)

# Gait phase clock: one gait cycle == two steps, so 1.0 Hz == 2.0 steps/s.
# Must match gait_clock.PHASE_FREQUENCY_HZ in velocity_env_cfg's obs group.
PHASE_FREQ_HZ = 1.0

# 48 proprioception + 2 phase clock; matches ObservationsCfg.PolicyCfg.
OBS_DIM = 50


def projected_gravity_from_quat(w, x, y, z) -> np.ndarray:
    """Unit gravity vector expressed in the base frame from a world->base
    orientation quaternion (w, x, y, z)."""
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    # Rotation matrix world<-base (row-major)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    g_world = np.array([0.0, 0.0, -1.0])
    return (R.T @ g_world).astype(np.float32)


class LocomotionNode(Node):
    def __init__(self):
        super().__init__('locomotion_node')
        self._ns = self.declare_parameter('robot_ns', 'k1_0').value
        policy_path = self.declare_parameter('policy_path',
                                             'models/k1_velocity_policy.pt').value
        freq = float(self.declare_parameter('control_freq', 50.0).value)
        self._action_scale = float(self.declare_parameter('action_scale', 0.25).value)
        command_type = self.declare_parameter(
            'command_type', 'k1_interfaces/JointCommand').value
        self._cmd_timeout = float(self.declare_parameter('cmd_timeout', 0.5).value)
        self._js_timeout = float(self.declare_parameter('js_timeout', 0.5).value)
        imu_topic = self.declare_parameter('imu_topic', '').value
        odom_topic = self.declare_parameter('odom_topic', '').value
        history_len = int(self.declare_parameter('history_len', 10).value)
        self._history_len = history_len
        self._input_mode = self.declare_parameter('input_mode', 'latest').value
        if self._input_mode not in ('latest', 'stacked'):
            raise ValueError(f"input_mode must be 'latest' or 'stacked', "
                             f"got {self._input_mode!r}")
        # Width the policy is actually fed at inference. The load-time probe must
        # use this and not a hardcoded OBS_DIM: a distilled student is
        # (history_len * OBS_DIM)-input, so the one mode that can run it
        # ('stacked') is the one the old probe rejected.
        self._policy_input_dim = (history_len * OBS_DIM
                                  if self._input_mode == 'stacked' else OBS_DIM)
        publish_obs_debug = bool(
            self.declare_parameter('publish_obs_debug', True).value)

        self._lock = threading.Lock()
        self._pref = f'/{self._ns}'

        # --- state ---
        self._cmd_vel = np.zeros(3, dtype=np.float32)          # vx, vy, wz
        self._cmd_stamp = None
        self._joint_pos = np.zeros(12, dtype=np.float32)
        self._joint_vel = np.zeros(12, dtype=np.float32)
        self._js_stamp = None
        self._base_lin_vel = np.zeros(3, dtype=np.float32)
        self._base_ang_vel = np.zeros(3, dtype=np.float32)
        self._projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._history = np.zeros((history_len, OBS_DIM), dtype=np.float32)
        # gait phase clock: starts at an arbitrary origin (training re-seeds it
        # every episode), advances once per control loop at PHASE_FREQ_HZ
        self._phase = 0.0
        self._phase_dt = PHASE_FREQ_HZ / freq
        self._warned = {}

        # --- policy ---
        self._policy = None
        if policy_path:
            self._policy = self._load_policy(policy_path)
        else:
            self.get_logger().warn(
                "policy_path is empty — idle mode (no inference, no commands)")

        # --- ROS I/O ---
        sensor_qos = QoSProfile(depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, f'{self._pref}/joint_states',
                                 self._js_cb, sensor_qos)
        self.create_subscription(Twist, f'{self._pref}/cmd_vel',
                                 self._cmd_cb, 10)
        if imu_topic:
            from sensor_msgs.msg import Imu
            self.create_subscription(Imu, f'{self._pref}/{imu_topic.strip("/")}',
                                     self._imu_cb, sensor_qos)
        if odom_topic:
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry,
                                     f'{self._pref}/{odom_topic.strip("/")}',
                                     self._odom_cb, sensor_qos)

        if command_type == 'sensor_msgs/JointState':
            cmd_msg_type = JointState
        elif command_type == 'k1_interfaces/JointCommand':
            cmd_msg_type = JointCommand
        else:
            raise ValueError(f'unknown command_type: {command_type!r}')
        self._cmd_pub = self.create_publisher(cmd_msg_type,
                                              f'{self._pref}/joint_commands', 10)
        self._obs_pub = (self.create_publisher(PolicyObs,
                                               f'{self._pref}/policy_obs', 10)
                         if publish_obs_debug else None)

        self.create_timer(1.0 / freq, self._control_loop)
        self.get_logger().info(
            f'LocomotionNode ready [ns={self._ns}, freq={freq} Hz, '
            f'policy={"loaded" if self._policy else "idle"}, '
            f'cmd_type={command_type}, obs={OBS_DIM}, '
            f'history={history_len} @ node]')

    # ------------------------------------------------------------------ #
    def _load_policy(self, path: str):
        import os
        import torch  # lazy — see module docstring for runtime options

        if not os.path.isfile(path):
            # resolve workspace-relative paths via the module location
            # (<ws>/src/k1_locomotion/k1_locomotion/locomotion_node.py)
            ws_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            candidate = os.path.join(ws_root, path)
            if os.path.isfile(candidate):
                path = candidate
            else:
                raise FileNotFoundError(
                    f'policy not found: {path} (cwd={os.getcwd()}, '
                    f'tried {candidate})')
        policy = torch.jit.load(path)
        policy.eval()
        # Fail fast on layout drift: probe input dim with a dummy forward.
        # Probe at the width this node will actually feed (see
        # _policy_input_dim), not at OBS_DIM. Naming both real candidates matters
        # because the old message blamed 48-dim exports and sent the reader
        # looking for a phase-clock regression that was not the cause.
        want = self._policy_input_dim
        with torch.no_grad():
            try:
                out = policy(torch.zeros(1, want))
            except RuntimeError as exc:
                other = OBS_DIM if want != OBS_DIM else self._history_len * OBS_DIM
                raise RuntimeError(
                    f'policy input layout mismatch: {path} does not accept '
                    f'{want} inputs, which is what input_mode='
                    f'{self._input_mode!r} feeds (history_len='
                    f'{self._history_len} x OBS_DIM={OBS_DIM}). If this export '
                    f'was built for the other layout, it wants {other} — set '
                    f'input_mode accordingly. A distilled student is '
                    f'{self._history_len * OBS_DIM}-input and needs '
                    f"input_mode='stacked'."
                ) from exc
        if out.numel() != 12:
            raise RuntimeError(f'policy output dim {out.numel()} != 12')
        self.get_logger().info(
            f'policy loaded: {path} (in={want}, out=12, '
            f"input_mode={self._input_mode})")
        return policy

    # ------------------------------------------------------------------ #
    def _warn_once(self, key: str, msg: str):
        if not self._warned.get(key):
            self._warned[key] = True
            self.get_logger().warn(msg)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _stale(self, stamp, timeout: float) -> bool:
        return stamp is None or (self._now() - stamp) > timeout

    # ------------------------------------------------------------------ #
    def _js_cb(self, msg: JointState):
        with self._lock:
            for i, jname in enumerate(LEG_JOINTS):
                try:
                    idx = msg.name.index(jname)
                except ValueError:
                    continue
                self._joint_pos[i] = msg.position[idx]
                self._joint_vel[i] = (msg.velocity[idx]
                                      if msg.velocity and len(msg.velocity) > idx
                                      else 0.0)
            self._js_stamp = self._now()

    def _cmd_cb(self, msg: Twist):
        with self._lock:
            self._cmd_vel[:] = [msg.linear.x, msg.linear.y, msg.angular.z]
            self._cmd_stamp = self._now()

    def _imu_cb(self, msg):
        with self._lock:
            q = msg.orientation
            self._projected_gravity = projected_gravity_from_quat(
                q.w, q.x, q.y, q.z)
            av = msg.angular_velocity
            self._base_ang_vel[:] = [av.x, av.y, av.z]

    def _odom_cb(self, msg):
        with self._lock:
            t = msg.twist.twist.linear  # child_frame_id = base link
            self._base_lin_vel[:] = [t.x, t.y, t.z]

    # ------------------------------------------------------------------ #
    def _build_obs(self) -> np.ndarray:
        """50-dim observation, EXACT training term order (see module docstring)."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:3] = self._base_lin_vel
        obs[3:6] = self._base_ang_vel
        obs[6:9] = self._projected_gravity
        obs[9:12] = self._cmd_vel
        obs[12:24] = self._joint_pos - DEFAULT_LEG_POS
        obs[24:36] = self._joint_vel
        obs[36:48] = self._last_action
        # [48:50] phase clock — the policy's time base for stepping. Read at
        # the current phase, then advanced, so obs_t carries t * PHASE_FREQ_HZ
        # of elapsed phase (one full wrap every 1/f PHASE_FREQ_HZ seconds).
        angle = 2.0 * math.pi * self._phase
        obs[48:50] = (math.sin(angle), math.cos(angle))
        self._phase = (self._phase + self._phase_dt) % 1.0
        return obs

    def _control_loop(self):
        with self._lock:
            js_stale = self._stale(self._js_stamp, self._js_timeout)
            cmd_stale = self._stale(self._cmd_stamp, self._cmd_timeout)
            if cmd_stale:
                self._cmd_vel[:] = 0.0
            obs = self._build_obs()
            self._history[:-1] = self._history[1:]
            self._history[-1] = obs

        if js_stale:
            # No feedback -> do not command blind. Hold off until states return.
            self._warn_once('js', f'no joint_states for > {self._js_timeout}s '
                                  '- withholding commands')
            return
        if self._policy is None:
            return

        import torch
        with torch.no_grad():
            if self._input_mode == 'stacked':
                # term-major 50x10 stack, oldest->newest per term — matches
                # ObservationManager group history_length layout
                x = torch.from_numpy(self._history.T.flatten()).unsqueeze(0)
            else:
                # current blind policy consumes the LATEST single-step obs
                x = torch.from_numpy(self._history[-1]).unsqueeze(0)
            assert x.numel() == self._policy_input_dim, (
                f'inference feeds {x.numel()} but the policy was probed at '
                f'{self._policy_input_dim}; history_len and input_mode disagree')
            action = self._policy(x).squeeze(0).numpy()

        with self._lock:
            self._last_action[:] = action
            targets = DEFAULT_LEG_POS + self._action_scale * action

        msg = self._cmd_msg(targets)
        self._cmd_pub.publish(msg)
        if self._obs_pub is not None:
            self._publish_obs()

    def _cmd_msg(self, targets: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        if self._cmd_pub.msg_type is JointState:
            m = JointState()
            m.header.stamp = stamp
            m.name = list(LEG_JOINTS)
            m.position = [float(v) for v in targets]
            return m
        m = JointCommand()
        m.header.stamp = stamp
        m.joint_names = list(LEG_JOINTS)
        m.positions = [float(v) for v in targets]
        m.control_mode = 0  # position
        return m

    def _publish_obs(self):
        with self._lock:
            m = PolicyObs()
            m.header.stamp = self.get_clock().now().to_msg()
            m.obs_vector = self._history.flatten().tolist()
            m.history_len = self._history.shape[0]
            m.obs_dim = OBS_DIM
        self._obs_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = LocomotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
