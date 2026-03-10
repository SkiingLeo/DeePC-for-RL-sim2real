"""
Digital twin that mirrors the real Franka arm in Isaac Lab using the
FRANKA_PANDA_REAL_CFG actuators and the Reach environment. This version
removes direct ROS dependencies and exchanges data with the ROS side via
ZMQ envelopes, staying compatible with ros_zmq_bridge.py.
"""

import argparse
import pickle
import sys
import threading
import time
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import zmq

# AppLauncher must be instantiated before importing any Omniverse/Isaac modules that load extensions.
from isaaclab.app import AppLauncher

TASK_NAME = "Isaac-Reach-Franka-v0"
CONTROL_DT = 0.05  # 50 ms real-time cadence
NUM_JOINTS = 7

# ZMQ defaults (compatible with ros_zmq_bridge.py)
ROS_TO_ISAAC_ENDPOINT = "tcp://127.0.0.1:6001"  # bridge PUB binds here; we SUB
ISAAC_TO_ROS_ENDPOINT = "tcp://127.0.0.1:6002"  # bridge SUB connects here; we PUB
PICKLE_PROTOCOL = 5
SNDHWM_DEFAULT = 10
RCVHWM_DEFAULT = 10

ROS_IN_TOPICS = [
    "/rl_policy_controller/desired_joint_positions",
    "/franka_robot_state_broadcaster/robot_state",
]

ISAAC_OUT_TOPICS = [
    "/digital_twin/joint_differences",
    "/digital_twin/sim_joints",
    "/digital_twin/sim_torques",
    "/digital_twin/torque_differences",
]


def make_envelope(topic: str, payload: Dict[str, Any]) -> bytes:
    return pickle.dumps({"topic": topic, "t_mono": time.monotonic(), "payload": payload}, protocol=PICKLE_PROTOCOL)


def parse_envelope(raw: bytes) -> Dict[str, Any]:
    return pickle.loads(raw)


class ZmqTwinBridge:
    """ZMQ bridge for the Isaac (py3.11) side."""

    def __init__(
        self,
        sub_endpoint: str,
        pub_endpoint: str,
        sndhwm: int = SNDHWM_DEFAULT,
        rcvhwm: int = RCVHWM_DEFAULT,
        conflate: bool = False,
        poll_timeout_ms: int = 1,
    ):
        self._ctx = zmq.Context(io_threads=1)
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVHWM, rcvhwm)
        self._sub.setsockopt(zmq.LINGER, 0)
        for topic in ROS_IN_TOPICS:
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._sub.connect(sub_endpoint)

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, sndhwm)
        self._pub.setsockopt(zmq.LINGER, 0)
        if conflate:
            self._pub.setsockopt(zmq.CONFLATE, 1)
        self._pub.bind(pub_endpoint)

        self._poller = zmq.Poller()
        self._poller.register(self._sub, zmq.POLLIN)
        self._poll_timeout_ms = poll_timeout_ms

        self._running = threading.Event()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)

        self._lock = threading.Lock()
        self._latest_desired: Optional[np.ndarray] = None
        self._latest_real: Optional[np.ndarray] = None
        self._latest_real_effort: Optional[np.ndarray] = None

    # --- Inbound (ROS -> Isaac via SUB) ---
    def _recv_loop(self) -> None:
        while self._running.is_set():
            events = dict(self._poller.poll(timeout=self._poll_timeout_ms))
            if self._sub not in events:
                continue

            try:
                topic_b, body = self._sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            except Exception:
                continue

            topic = topic_b.decode()
            try:
                env = parse_envelope(body)
                payload = env.get("payload") or {}
            except Exception:
                continue

            positions = payload.get("position")
            efforts = payload.get("effort")
            if not positions:
                continue

            arr = np.array(positions[:NUM_JOINTS], dtype=np.float32)
            with self._lock:
                if topic == "/rl_policy_controller/desired_joint_positions":
                    self._latest_desired = arr
                elif topic == "/franka_robot_state_broadcaster/robot_state":
                    self._latest_real = arr
                    if efforts:
                        eff_arr = np.array(efforts[:NUM_JOINTS], dtype=np.float32)
                        self._latest_real_effort = eff_arr

    # --- Outbound (Isaac -> ROS via PUB) ---
    def _send(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            self._pub.send_multipart([topic.encode(), make_envelope(topic, payload)], flags=zmq.DONTWAIT)
        except zmq.Again:
            # Drop if queue is full to preserve latency
            pass
        except Exception:
            pass

    def publish_outputs(self, joint_diff: np.ndarray, sim_joints: np.ndarray) -> None:
        self._send("/digital_twin/joint_differences", {"position": joint_diff.tolist()})
        self._send("/digital_twin/sim_joints", {"position": sim_joints.tolist()})
        # Torques are published separately to avoid breaking existing consumers.
        # Expect shape (NUM_JOINTS,) in sim joint order.
        # Caller should ensure CPU numpy array.
        # Topic: /digital_twin/sim_torques

    def publish_torques(self, sim_torques: np.ndarray) -> None:
        self._send("/digital_twin/sim_torques", {"effort": sim_torques.tolist()})

    def publish_torque_differences(self, torque_diff: np.ndarray) -> None:
        self._send("/digital_twin/torque_differences", {"effort": torque_diff.tolist()})

    # --- Control ---
    def start(self) -> None:
        self._running.set()
        self._recv_thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._recv_thread.join(timeout=1.0)
        self._poller.unregister(self._sub)
        self._sub.close(0)
        self._pub.close(0)
        self._ctx.term()

    # --- Accessors ---
    def latest_desired(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_desired is None else self._latest_desired.copy()

    def latest_real(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_real is None else self._latest_real.copy()

    def latest_real_effort(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_real_effort is None else self._latest_real_effort.copy()


def _build_env():
    """Create a single-env Reach simulation with real Franka PD actuators."""
    # Import after AppLauncher instantiation to satisfy Omni extension loading order.
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_assets import FRANKA_PANDA_REAL_CFG
    from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
        FrankaReachEnvCfg,
    )
    import isaaclab_tasks  # noqa: F401 - registers tasks with Gym

    env_cfg = FrankaReachEnvCfg()
    env_cfg.scene.num_envs = 1  # hardcoded single arm
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.robot = FRANKA_PANDA_REAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    return env, torch.device(env_cfg.sim.device), dt

def step_and_average_torque_last_n(env, robot, joint_ids, action_tensor, n_substeps: int): #taking 20 instead of the full 25, for not falling behind RT
    """Apply an action with decimation and return mean applied torque over the last n_substeps."""
    unwrapped = env.unwrapped
    decimation = getattr(unwrapped.cfg, "decimation", 1)
    n = min(n_substeps, decimation)

    torque_accum = torch.zeros(len(joint_ids), device=action_tensor.device, dtype=robot.data.applied_torque.dtype)
    window_start = decimation - n

    unwrapped.action_manager.process_action(action_tensor)
    for step_idx in range(decimation):
        unwrapped.action_manager.apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        unwrapped.scene.update(dt=unwrapped.physics_dt)
        if step_idx >= window_start:
            torque_accum += robot.data.applied_torque[0, joint_ids]

    return torque_accum / float(n)


def main():
    parser = argparse.ArgumentParser(description="Run digital twin for Franka at 20 Hz via ZMQ.")
    parser.add_argument("--ros-to-isaac", default=ROS_TO_ISAAC_ENDPOINT, help="ZMQ endpoint to SUB (from ROS bridge).")
    parser.add_argument("--isaac-to-ros", default=ISAAC_TO_ROS_ENDPOINT, help="ZMQ endpoint to PUB (to ROS bridge).")
    parser.add_argument("--sndhwm", type=int, default=SNDHWM_DEFAULT, help="ZMQ PUB high-water mark.")
    parser.add_argument("--rcvhwm", type=int, default=RCVHWM_DEFAULT, help="ZMQ SUB high-water mark.")
    parser.add_argument("--conflate", action="store_true", help="Enable ZMQ CONFLATE on PUB (latest only).")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    # Prevent leftover Hydra args from interfering
    sys.argv = [sys.argv[0]]

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import Omni-dependent modules after SimulationApp is created.
    # from franka_isaac_translation import *  # noqa: F401,F403

    bridge = ZmqTwinBridge(
        sub_endpoint=args_cli.ros_to_isaac,
        pub_endpoint=args_cli.isaac_to_ros,
        sndhwm=args_cli.sndhwm,
        rcvhwm=args_cli.rcvhwm,
        conflate=args_cli.conflate,
    )
    bridge.start()

    env, device, dt = _build_env()
    robot = env.unwrapped.scene["robot"]
    # Fetch the action term using the API-provided getter.
    arm_action = env.unwrapped.action_manager.get_term("arm_action")
    # Neutralize any built-in scaling/offset so raw actions are absolute Isaac-frame joints.
    arm_action._scale = torch.ones_like(torch.as_tensor(arm_action._scale, device=device))
    arm_action._offset = torch.zeros_like(torch.as_tensor(arm_action._offset, device=device))
    joint_ids = (
        torch.tensor(arm_action._joint_ids, device=device)
        if not isinstance(arm_action._joint_ids, slice)
        else torch.arange(arm_action._asset.data.joint_pos.shape[1], device=device)
    )[:NUM_JOINTS]

    env.reset()

    print(
        f"[INFO] Digital twin started: device={device}, sim_dt={dt}, control_dt={CONTROL_DT}, "
        f"SUB={args_cli.ros_to_isaac}, PUB={args_cli.isaac_to_ros}"
    )

    try:
        while simulation_app.is_running():
            loop_start = time.time()

            desired = bridge.latest_desired()
            if desired is None:
                time.sleep(0.001)
                continue

            desired_isaac = desired #translate_franka_actions_to_isaac(desired), no translation needed, franka publishes absolute joint positions
            desired_tensor = torch.as_tensor(desired_isaac, device=device).unsqueeze(0)
            raw_actions = desired_tensor

            with torch.inference_mode():
                avg_torque = step_and_average_torque_last_n(env, robot, joint_ids, raw_actions, n_substeps=20)

            sim_joints = robot.data.joint_pos[0, joint_ids].cpu().numpy()
            sim_torques = avg_torque.cpu().numpy()
            real_joints = bridge.latest_real()
            real_torques = bridge.latest_real_effort()
            if real_joints is not None:
                real_isaac = real_joints #translate_franka_position_to_isaac(real_joints), no translation needed, this script also uses absolute joint positions
                joint_diff = sim_joints - real_isaac
                bridge.publish_outputs(joint_diff, sim_joints)
            if real_torques is not None:
                torque_diff = sim_torques - real_torques
                bridge.publish_torque_differences(torque_diff)
            bridge.publish_torques(sim_torques)

            elapsed = time.time() - loop_start
            remaining = CONTROL_DT - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[INFO] Received shutdown signal, stopping twin...")
    finally:
        bridge.stop()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
