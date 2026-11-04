# src/k1_bringup/launch

Part of [k1-deploy-laptop](https://github.com/thdhyan/k1-deploy-laptop).

| File | What it does |
|---|---|
| `fleet.launch.py` | Multi-robot fleet bringup (Phase 5). Usage: ros2 launch k1_bringup fleet.launch.py n_robots:=6 sim:=gazebo |
| `real.launch.py` | Real robot bringup — Booster K1 via booster_robotics_sdk. |
| `sim_gazebo.launch.py` | Top-level Gazebo bringup. Usage: ros2 launch k1_bringup sim_gazebo.launch.py robot_ns:=k1_0 |
| `sim_isaac.launch.py` | Top-level Isaac Sim bringup. Usage: ros2 launch k1_bringup sim_isaac.launch.py robot_ns:=k1_0 |
| `sim_validate.launch.py` | Unified Simulation Validation Launch — Booster K1 |
