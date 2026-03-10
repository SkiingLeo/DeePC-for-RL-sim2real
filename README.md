# DeePC (Data-Enabled Predictive Control) for Reinforcement Learning Transferability on a 7-DoF Robot

**Leo Noth, Lucas Gimeno, Mirko Meboldt**

> Reinforcement learning policies trained in simulation often suffer performance degradation when deployed on real robotic systems due to discrepancies between simulated and physical dynamics. This work proposes a control-theoretic alternative that regulates the sim2real discrepancy during deployment using Data-Enabled Predictive Control (DeePC), without modifying the trained policy itself.

---

## Overview

Deploying RL policies from simulation to real hardware is limited by the **sim2real gap** — the mismatch between simulated and physical dynamics. Common approaches (domain randomization, residual learning, system identification) address this from the *training* side, adding complexity and computational cost.

This project takes a different approach: treat the sim2real gap as an **online control problem**. By running the simulator as a digital twin alongside the real robot, we observe the joint-level discrepancy in real time and use **DeePC** to compute corrective torques that steer the physical system toward simulator-consistent behavior — all without retraining or modifying the policy.

<p align="center">
  <img src="DeePC Sim2Real Setup Diagram.png" alt="System architecture: the same RL policy drives both the real and simulated robot in parallel. The resulting sim2real gap is fed into DeePC, which computes corrective torques to reduce the discrepancy." width="850"/>
</p>
<p align="center"><i>Fig. 1 — The real and simulated systems run the same policy in parallel. The joint-level sim2real gap is actively regulated by DeePC in a feedback control loop.</i></p>

### How It Works

1. An RL policy (trained in [IsaacLab](https://github.com/isaac-sim/IsaacLab)) commands joint positions to the real **Franka Emika Panda** via a low-level PD controller at 500 Hz.
2. A **digital twin** in IsaacLab mirrors the real robot by receiving the same desired joint positions and stepping the simulation at matched frequency.
3. The **sim2real gap** — the difference between simulated and real joint positions — is modeled as an approximately linear time-invariant (LTI) system.
4. **DeePC** solves a data-driven predictive control problem at 10 Hz using a pre-collected Hankel matrix, producing corrective torques that are applied to the real robot.
5. These torques drive the gap toward zero, keeping the real system within the state-action distribution the policy was trained on.

---

## Key Results

Experimental validation on the 7-DoF Franka Emika Panda across 100 repeated reach tasks with randomized target poses:

### Normalized Joint Differences (Sim vs. Real)

| | Mean | Std | Median | RMSE | Max | Min |
|---|---|---|---|---|---|---|
| **Zero-Shot** | 0.5603 | 0.5629 | 0.3395 | 0.7943 | 2.1541 | 0.0303 |
| **With DeePC** | 0.2620 | 0.2553 | 0.1581 | 0.3658 | 1.3059 | 0.0189 |
| **Improvement** | 53.2% | 54.6% | 53.4% | 53.9% | 39.4% | 37.6% |

### Terminal Pose Error (End-Effector RMSE, N=100)

| | Mean \[m\] | Std \[m\] | Median \[m\] | Min \[m\] | Max \[m\] |
|---|---|---|---|---|---|
| **Zero-Shot** | 0.0440 | 0.0507 | 0.0215 | 0.0033 | 0.2321 |
| **With DeePC** | 0.0181 | 0.0267 | 0.0137 | 0.0022 | 0.2035 |
| **Improvement** | 58.7% | 47.3% | 36.3% | 33.3% | 12.3% |

The improvement in mean terminal RMSE from 4.4 cm to 1.8 cm is **statistically significant** (Welch's t-test: *p* = 1.34 × 10⁻⁵, Cohen's *d* = 0.637).

---

## Project Structure

This project spans **two separate environments** that communicate via ZMQ:

| Side | Framework | Python | Runs on |
|---|---|---|---|
| **Franka / ROS 2** | franka_ros2 v0.1.15, ROS Humble | System Python 3.10 | Real-time PC connected to the robot |
| **IsaacLab** | NVIDIA IsaacLab (Isaac Sim) | venv Python 3.11 | GPU workstation (RTX 4090) |

The files in this repository are **not a standalone project** — they slot into the folder structures of franka_ros2 and IsaacLab after those frameworks are installed. The sections below explain where each file goes.

---

## Part 1 — Franka / ROS 2 Side

### 1.1 Install franka_ros2 (v0.1.15)

> **Important:** The latest franka_ros2 version is not compatible. Version **v0.1.15** with **libfranka 0.13.2** must be used. A detailed installation guide is included in this repository as [`Tips on franka_ros2 installation.pdf`](Tips%20on%20franka_ros2%20installation.pdf).

### 1.2 Custom C++ Controllers

A custom real-time controller was written for this project and added to the existing `franka_example_controllers` package:

| Controller | Purpose |
|---|---|
| `RLPolicyJointPDController` | Joint-space PD controller (500 Hz) that tracks desired positions published by the RL policy at 20 Hz. When `enable_py_torque` is set, it also subscribes to `deepc/py_controller` and blends DeePC corrective torques into its output. |

**Files to place** into the existing `~/franka_ros2_ws/src/franka_example_controllers/`:

```
franka_example_controllers/
├── src/
│   └── rl_policy_joint_pd_controller.cpp    ← add
├── include/franka_example_controllers/
│   └── rl_policy_joint_pd_controller.hpp    ← add
├── CMakeLists.txt                           ← modify (add the .cpp to add_library)
└── franka_example_controllers.xml           ← modify (register the plugin)
```

This repository ships **only the custom files and the modified build configs** — not the stock example controllers (those already exist after installing franka_ros2). The modified [`CMakeLists.txt`](franka_example_controllers/CMakeLists.txt) and [`franka_example_controllers.xml`](franka_example_controllers/franka_example_controllers.xml) contain all original entries plus the new one, so they can be used as drop-in replacements. After placing the files, rebuild:

```bash
cd ~/franka_ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.sh
```

### 1.3 Python Scripts (Robot Control & DeePC)

All Python scripts on the Franka side run under the **system Python 3.10** used by ROS Humble. Place the following files into a single directory:

```
~/franka_ros2_ws/src/franka_bringup/scripts/reach/
├── rl_policy_reach.py            # Main RL policy node (20 Hz inference loop)
├── state_preparation.py          # Assembles the 32-dim observation vector from robot state
├── single_inference.py           # Runs inference on the exported TorchScript policy
├── franka_isaac_translation.py   # Joint offset translation between Franka and Isaac frames
├── performance_metrics_publisher.py  # FK-based end-effector pose + live reward computation
├── ros_zmq_bridge.py             # ZMQ bridge: forwards ROS topics to/from the IsaacLab side
├── target_poses.csv              # Pre-generated reach target poses (used by state_preparation)
├── target_positions.csv          # Pre-generated target positions
├── initial_pose.csv              # Starting joint configuration (not included — generate from your setup)
└── exported_policy.pt            # Trained policy as TorchScript (not included — see Part 2)
```

Additional Python dependencies (install into the ROS system Python):

```bash
pip install numpy torch zmq scipy yourdfpy
```

### 1.4 DeePC Controller

The DeePC node also runs on the Franka/ROS 2 side (it needs `rclpy` to subscribe to the digital twin's gap signal). Place the `DeePC/` folder alongside the reach scripts or in its own location:

```
DeePC/
├── run_deepc_for_tuning_with_constraint_working_7_out.py   # DeePC ROS node (10 Hz)
└── 10Hz_1346_p_5_300_hankel_tini5_N15.npz                  # Pre-built Hankel matrix
```

The DeePC solver uses the [PyDeePC](https://github.com/rssalessio/PyDeePC) library by Alessio Russo. Install it along with the required solver:

```bash
pip install pydeepc cvxpy mosek
```

> **Note:** A [MOSEK license](https://www.mosek.com/products/academic-licenses/) is required (free for academic use). MOSEK is the recommended solver; without it, solve times may exceed the 100 ms real-time budget.

A pre-built Hankel matrix (`.npz`) is included in the `DeePC/` folder. To build your own from fresh data, use `Supplementary Scripts/build_hankel_from_csv.py` on a CSV of recorded joint differences and applied torques (see Section 1.5).

### 1.5 Supplementary Scripts

These are utility scripts used during development and data preparation, not part of the live control loop:

| Script | Purpose |
|---|---|
| `Supplementary Scripts/build_hankel_from_csv.py` | Builds Hankel matrices from logged CSV data for DeePC |
| `Supplementary Scripts/p_control_gap.py` | Simple P-controller baseline for gap regulation (used during Hankel data collection) |
| `Supplementary Scripts/play_to_export_model.py` | Exports a trained skrl checkpoint to TorchScript (runs on the IsaacLab side) |

---

## Part 2 — IsaacLab Side

The simulation environment runs inside [IsaacLab](https://github.com/isaac-sim/IsaacLab) and is used for three purposes: **training** the RL policy, **exporting** it to TorchScript, and running the **digital twin** during deployment.

> **Versions used:** Isaac Sim 5.0.0, IsaacLab 2.1.0. Install IsaacLab following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). All IsaacLab scripts run in the IsaacLab **virtual environment** (Python 3.11).

### 2.1 Custom Robot Configuration (Matched Low-Level Controller)

A critical requirement of this project is that the **simulated and real low-level controllers are identical**, so the observed sim2real gap come from dynamics mismatch rather than controller mismatch. To achieve this, a custom Franka actuator configuration (`FRANKA_PANDA_REAL_CFG`) was added to IsaacLab.

Place [`franka.py`](franka.py) into:

```
<isaaclab_root>/source/isaaclab_assets/isaaclab_assets/robots/franka.py
```

This is a modified version of the stock file with `FRANKA_PANDA_REAL_CFG` appended. It uses `IdealPDActuatorCfg` with per-joint PD gains that match the real robot's C++ controller (`RLPolicyJointPDController`):

Stiffness and damping must be kept the exact same between `franka.py` (sim) and the `k_gains` / `d_gains` parameters of the C++ controller (real).

### 2.2 Modified Reach Task

The reach task configuration was modified to encourage smooth, HRC-compatible motion. Replace the stock reach folder:

```
<isaaclab_root>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/
```

with the [`reach/`](reach/) folder from this repository. The key modifications compared to stock IsaacLab:

| File | What changed |
|---|---|
| `reach_env_cfg.py` | Increased `decimation` to 25 (for 20 Hz policy at 500 Hz physics), `episode_length_s` to 120 s, tuned `action_rate` and `joint_vel` penalty weights for compliant motion |
| `config/franka/joint_pos_env_cfg.py` | Uses `FRANKA_PANDA_REAL_CFG` (see Section 2.1) instead of the default `FRANKA_PANDA_CFG` |
| `config/franka/agents/skrl_ppo_cfg.yaml` | PPO hyperparameters for skrl training (64×64 network, 24k timesteps) |

### 2.3 Training

Train the reach policy using skrl:

```bash
cd <isaaclab_root>
python source/isaaclab_rl/scripts/skrl/train.py --task Isaac-Reach-Franka-v0 --headless
```

### 2.4 Export Policy to TorchScript

After training, export the policy (including observation normalization) to a standalone `.pt` file using the provided script:

```bash
python play_to_export_model.py --task Isaac-Reach-Franka-v0 \
    --checkpoint <path_to_checkpoint.pt> \
    --export_path exported_policy.pt --headless
```

Place [`Supplementary Scripts/play_to_export_model.py`](Supplementary%20Scripts/play_to_export_model.py) in the IsaacLab skrl scripts directory:

```
<isaaclab_root>/source/isaaclab_rl/scripts/skrl/play_to_export_model.py
```

The exported `exported_policy.pt` is then copied to the Franka side (see Section 1.3).

### 2.5 Digital Twin

During deployment, the digital twin runs in IsaacLab and communicates with the Franka/ROS 2 side via ZMQ. Place [`run_real_time_from_franka.py`](run_real_time_from_franka.py) in the skrl scripts directory:

```
<isaaclab_root>/source/isaaclab_rl/scripts/skrl/run_real_time_from_franka.py
```

Launch it with:

```bash
python run_real_time_from_franka.py --headless
```

This script:
- Creates a single-environment Reach simulation using `FRANKA_PANDA_REAL_CFG`
- Subscribes to desired joint positions and real joint positions from the Franka side (via ZMQ on port 6001)
- Steps the simulation at matched frequency and computes the joint-level sim2real gap
- Publishes gap signals and simulated torques back to the Franka side (via ZMQ on port 6002)

---
## Additional Notes

- The target_poses.csv file contains randomized reach poses (x,y,z) + (qw,qx,qy,qz) as pose_commands this is what the robot will track
- The target_positions.csv is not used for anything in particular, but is a left over of the lift task and refered to cube positions
- It is heavily recommended to create a bash script that launches several terminal tabs with tmux all at once to get all these commands ready.
- [Rosbag](https://wiki.ros.org/rosbag) is also recommended as it eliminates the need to write custom csv printing or saving functions.
- If rosbag was used,  [plotjuggler](https://github.com/facontidavide/PlotJuggler) is recommended to quickly analyze the generated data.
- A real time kernel can stabilize operations, but here communication frequency was dropped to 500 Hz, so a standard linux kernel was sufficient.
- Chances are that this software will not run out of the box, as newer versions of all the included software are released
- Adjusting the code to run on up-to-date software stacks seems like a good usecase for llms, given the right context
- A better integration of IsaacLab with Ros2 might be possible, but given the simplicity of this task, the ZMQ bridge solution was implemented.
- The flag --headless for running the online simulation is absolutely necessary, as otherwise with rendering, the real time factor would not be below 1
- If used for anything else other than the reach task, this codebase would require large changes in almost all files

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{noth2025deepc,
  title     = {{DeePC} (Data-Enabled Predictive Control) for Reinforcement Learning Transferability on a 7 {DoF} Robot},
  author    = {Noth, Leo and Gimeno, Lucas and Meboldt, Mirko},
  year      = {2026}
  note      = Coming soon, in review.
}
```
