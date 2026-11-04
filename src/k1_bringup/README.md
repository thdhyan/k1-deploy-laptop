# k1_bringup

Top-level launch entry points for the Booster K1 humanoid robot.

## Overview

This package provides launch wrappers for all K1 simulation and deployment modes:
- **Gazebo Harmonic**: physics-based simulation with ROS2 bridge
- **Isaac Sim 6.0.1**: visual simulation with real-time rendering
- **Real Robot**: hardware bring-up via booster_robotics_sdk
- **Fleet**: multi-robot coordination (Phase 5)
- **Validation**: unified policy validation across all backends

All nodes respect the `robot_ns` parameter (default `k1_0`) for multi-robot operation.

## Quick Start — Policy Validation (NEW)

**Single command to validate any policy on any backend:**

```bash
# Gazebo (CPU, default)
ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo policy:=models/k1_velocity_policy.pt

# Isaac Sim (GPU, photorealistic)
ros2 launch k1_bringup sim_validate.launch.py backend:=isaac policy:=models/k1_velocity_policy.pt usd_path:=/path/to/k1.usd

# MuJoCo (CPU, fastest iteration) — run fleet script separately
ros2 launch k1_bringup sim_validate.launch.py backend:=mujoco policy:=models/k1_velocity_policy.pt

# Multi-robot fleet validation (Gazebo)
ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo n_robots:=3

# With keyboard teleop
ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo teleop:=true
```

See [Sim Validation Guide](#simulation-validation) below for full checklist.

## Launching

### Gazebo Simulation
```bash
ros2 launch k1_bringup sim_gazebo.launch.py robot_ns:=k1_0
```

Brings up:
1. k1_description (robot_state_publisher)
2. Gazebo Harmonic + K1 URDF
3. ros_gz_bridge (joint_states, odom, cmd_vel)
4. k1_control (sim_bridge_node)
5. k1_locomotion (policy inference)

### Isaac Sim Simulation
```bash
ros2 launch k1_bringup sim_isaac.launch.py robot_ns:=k1_0 usd_path:=/path/to/k1.usd
```

Brings up:
1. k1_description (robot_state_publisher)
2. Isaac Sim 6.0.1 + K1 USD asset
3. OmniGraph ROS2 bridge (ActionGraph)
4. k1_control (sim_bridge_node)
5. k1_locomotion (policy inference)

**Note**: Isaac Sim ROS2 is separate from system ROS2 and lives at `~/Projects/IsaacLab/isaac6`.
Do NOT source both in the same shell.

### Real Robot Hardware
```bash
ros2 launch k1_bringup real.launch.py robot_ns:=k1_0 sdk_ip:=192.168.1.100
```

**⚠️ WARNING**: Ensure robot is powered and in a safe, open area (≥ 3m × 3m).
E-stop must be accessible. See `guide_real.md` for complete deployment guide.

Launches:
1. k1_description (robot_state_publisher)
2. k1_control (sdk_bridge_node — **currently a stub**, implement SDK integration)
3. k1_locomotion (policy inference)
4. k1_wbc (whole-body controller, optional via `use_wbc:=true`)

### Multi-Robot Fleet (Phase 5)
```bash
ros2 launch k1_bringup fleet.launch.py n_robots:=6 sim:=gazebo
```

Spawns 6 robots (k1_0..k1_5) with 2m spacing. Backend: `gazebo` or `isaac`.

### Simulation Validation (All Backends)
```bash
ros2 launch k1_bringup sim_validate.launch.py backend:=gazebo policy:=models/k1_velocity_policy.pt
```

Validates a trained policy on Gazebo, Isaac Sim, or MuJoCo with identical ROS2 endpoints.
Supports single-robot and fleet modes, optional keyboard teleop.

## Launch Arguments

### Common (all modes)
| Argument | Default | Description |
|----------|---------|-------------|
| `robot_ns` | `k1_0` | Robot namespace, used in topic names: `/{robot_ns}/topic_name` |
| `policy_path` | `models/k1_velocity_policy.pt` | Path to TorchScript locomotion policy |

### Gazebo-specific
| Argument | Default | Description |
|----------|---------|-------------|
| `world` | `flat` | World name: `flat` or `rough` |
| `headless` | `false` | Disable visualizer |

### Isaac-specific
| Argument | Default | Description |
|----------|---------|-------------|
| `usd_path` | `''` | Path to K1 USD (from booster_assets) |
| `headless` | `false` | Disable visualizer |

### Real robot-specific
| Argument | Default | Description |
|----------|---------|-------------|
| `sdk_ip` | `192.168.1.100` | Robot SDK IP address |
| `use_wbc` | `false` | Enable whole-body controller (k1_wbc) |
| `control_freq` | `50.0` | Control loop frequency (Hz) |

### Fleet-specific
| Argument | Default | Description |
|----------|---------|-------------|
| `n_robots` | `1` | Number of robots (1-6) |
| `sim` | `gazebo` | Backend: `gazebo` or `isaac` |

### Validation-specific (`sim_validate.launch.py`)
| Argument | Default | Description |
|----------|---------|-------------|
| `backend` | `gazebo` | Sim backend: `gazebo` \| `isaac` \| `mujoco` |
| `n_robots` | `1` | Number of robots for fleet validation |
| `teleop` | `false` | Launch keyboard teleop (teleop_twist_keyboard) |

## Available Policies

| Policy File | Description | Obs Dim | Action Scale | Input Mode |
|-------------|-------------|---------|--------------|------------|
| `models/k1_velocity_policy.pt` | P2 velocity student (blind) | 48 | 0.25 | `latest` |
| `models/k1_velocity_student.pt` | P2 velocity distilled (history) | 48×10 | 0.25 | `stacked` |
| `models/p2_move_student.pt` | P2 move student (alt. checkpoint) | 48×10 | 0.25 | `stacked` |
| `models/p1_basic_student.pt` | P1 basic stand/balance | 42 | 0.25 | `latest` |
| `models/k1_partialctrl_base.pt` | Partial control (legs + head) | 68 | 0.25 | `latest` |

**First real-robot runs**: Use `k1_velocity_policy.pt` with `input_mode:=latest`.

## Simulation Validation Checklist

Run `sim_validate.launch.py` on each backend and verify:

- [ ] Robot stands stably (no drift, no collapse)
- [ ] Forward walk: `vx=0.5` → robot moves forward ~0.5 m/s
- [ ] Lateral walk: `vy=0.3` → robot moves laterally
- [ ] Yaw turn: `wz=0.5` → robot rotates in place
- [ ] Combined: `vx=0.3, wz=0.3` → curved trajectory
- [ ] Stop command → robot halts smoothly
- [ ] Policy obs published on `/{ns}/policy_obs` (debug)
- [ ] Joint commands within URDF limits (check `/{ns}/joint_commands`)

## Dependencies

- `k1_description` — URDF/mesh assets
- `k1_control` — joint control bridge nodes (sim_bridge + sdk_bridge)
- `k1_locomotion` — neural policy inference
- `k1_sim_gazebo` — Gazebo integration
- `k1_sim_isaac` — Isaac Sim integration
- `k1_wbc` — whole-body controller (optional)

## Phase Roadmap

- **Phase 1 (v0.1)**: Gazebo sim bringup
- **Phase 2**: Isaac Sim integration
- **Phase 3**: Real robot hardware interface (SDK bridge implementation)
- **Phase 4**: Whole-body control (k1_wbc)
- **Phase 5**: Multi-robot coordination

## Real Robot Deployment Guide

See **[guide_real.md](../guide_real.md)** for complete instructions on:
- Network setup (WiFi, CycloneDDS)
- Policy selection and parameters
- Single-robot and multi-robot bringup
- Safety checklist (mandatory)
- Troubleshooting common issues
- Policy-to-task mapping table

## Notes

- All topics are namespaced: e.g., `/k1_0/joint_states`, `/k1_1/odom`
- Namespace default is `k1_0`; use `robot_ns:=k1_1` for second robot
- Joint states come from Gazebo/Isaac/SDK; commands go back via respective bridges
- Odometry estimated from joint kinematics (Gazebo) or physics engine (Isaac) or SDK (real)
- Isaac Sim uses `sensor_msgs/JointState` for commands; Gazebo/real use `k1_interfaces/JointCommand`

---
> **Network note (2026-10-27):** the robotics router (192.168.50.1) has no internet uplink. The laptop shares its eduroam (`wlo1`) internet to the robots by NAT: `enp2s0` = 192.168.50.250/24 is the robots' gateway, with `ip_forward=1` and nft table `ip k1nat` (masquerade out `wlo1`, source 192.168.50.0/24). It is lost on laptop reboot; robots use DNS 8.8.8.8/1.1.1.1 (router DNS is wrong). Re-create steps: `docs/network_robotics_5G_2026-10-26.md` ("Robot internet via laptop").
