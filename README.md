# k1-deploy-laptop

ROS 2 jazzy workspace, push-IK web/sim, pad sender, sim-to-sim.

| | |
|---|---|
| Layer | **L3**: Apps (processes you start) |
| May import | [k1-common](https://github.com/thdhyan/k1-common), [k1-runtime](https://github.com/thdhyan/k1-runtime), [k1-teleop](https://github.com/thdhyan/k1-teleop), [k1-perception](https://github.com/thdhyan/k1-perception) |
| Files / LOC | 56 files · 5,520 lines (py, sh, cpp) |
| Part of | [k1-workspace](https://github.com/thdhyan/k1-workspace) (architecture and layering rules live there) |

## Install

```bash
uv add git+https://github.com/thdhyan/k1-deploy-laptop
# or, from a checkout
uv sync
```

## Layout

```
deploy/  (8 files)
  README.md
  gamepad_sim.py
  pad_udp_send.py
  push_ik_sim.py
  push_ik_web/  (3 files)
  push_ik_web.py
pyproject.toml
src/  (54 files)
  k1_bringup/  (10 files)
  k1_control/  (9 files)
  k1_fleet_rviz/  (8 files)
  k1_locomotion/  (7 files)
  k1_sim_gazebo/  (14 files)
  k1_wbc/  (6 files)
tests/  (1 files)
  test_push_ik_web.py
```

## Dependencies

- Declared sibling libraries: k1-common, k1-runtime, k1-teleop, k1-perception (see `pyproject.toml`, `[tool.uv.sources]`).
- PyPI: evdev, imageio, mujoco, numpy, opencv-python, pytest, torch.
- Provided by the environment (ROS 2, Isaac, mjlab, robot SDK) or unresolved: `ament_index_python`, `geometry_msgs`, `launch`, `launch_ros`, `nav_msgs`, `push_ik_phasespace`, `push_ik_sim`, `rclpy`, `sensor_msgs`, `setuptools`, `std_msgs`, `tf2_msgs`.

## Hard-coded paths into other repos

These files still name paths that now live in another repo (hard-coded path strings, not imports). Pass them as arguments / config when you touch the file:

| File | Refers to | Owner |
|---|---|---|
| `deploy/gamepad_sim.py` | `mjlab_k1/` | k1-rl-mj |

## Origin

Extracted from `booster_ws` (branch `dev/push-training-wip`, working tree of 2026-11-03) by
`tools/build_workspace.py` in k1-workspace. Old path to new path for every file: [`docs/migration_map.tsv`](https://github.com/thdhyan/k1-workspace/blob/main/docs/migration_map.tsv) (filter on `k1-deploy-laptop`).
History was not carried over: the working tree was copied as is, including previously untracked files.
