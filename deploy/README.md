# deploy

Part of [k1-deploy-laptop](https://github.com/thdhyan/k1-deploy-laptop).

| File | What it does |
|---|---|
| `gamepad_sim.py` | Sim-to-sim for the K1 walk base: MuJoCo K1_22dof_mjlab.xml, scripted stick or live gamepad (viewer), get-up FSM test. |
| `pad_udp_send.py` | Laptop gamepad -> K1 deploy over UDP: run k1_gamepad_deploy.py --udp PORT on the robot, this on the laptop. |
| `push_ik_sim.py` | CPU MuJoCo preview of a PushIK run: real deploy code (PushIKController + walk base) on 2 K1s and a box, from given poses. |
| `push_ik_web.py` | Browser tool for the 2-K1 box push: track the 3 PhaseSpace caps, set the scene, preview it in sim, then execute. |
