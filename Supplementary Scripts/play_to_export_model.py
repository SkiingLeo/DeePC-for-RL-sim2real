# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl and export it to a .pt file.

This script loads the trained agent and exports it as a TorchScript module (JIT).
The exported model includes:
1. Observation Normalization (RunningStandardScaler)
2. The Policy Network (Actor)

This ensures that the exported model can be run on the robot without needing skrl
or manually reconstructing the network architecture.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

# Note: We delay torch imports until after AppLauncher to avoid libstdc++ conflicts with Isaac Sim
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl and export it.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--export_path", type=str, default="exported_policy.pt", help="Path to save the exported .pt file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
# IMPORTANT: Import torch and other libraries AFTER launching the app to prevent DLL conflicts
import torch
import torch.nn as nn

import gymnasium as gym
import random
import time

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


class ExportedPolicy(nn.Module):
    """
    Wraps the skrl agent's policy and preprocessor for export.
    Includes:
    - Input Normalization (using running mean/std from training)
    - Policy Inference (Deterministic)
    """
    def __init__(self, agent, checkpoint_dict=None):
        super().__init__()
        self.device = torch.device("cpu") # Force CPU for export
        
        # 1. Extract Normalization Parameters from checkpoint or agent
        self.has_preprocessor = False
        
        # First, try loading from checkpoint dictionary
        if checkpoint_dict is not None and "state_preprocessor" in checkpoint_dict:
            print("[INFO] Found state preprocessor in checkpoint. Exporting normalization parameters.")
            preprocessor_state = checkpoint_dict["state_preprocessor"]
            self.register_buffer("running_mean", torch.tensor(preprocessor_state["running_mean"]).float().to(self.device))
            self.register_buffer("running_variance", torch.tensor(preprocessor_state["running_variance"]).float().to(self.device))
            self.has_preprocessor = True
        # Fallback: try to get from agent's state_preprocessor attribute
        elif hasattr(agent, "state_preprocessor") and agent.state_preprocessor is not None:
            print("[INFO] Found state preprocessor in agent. Exporting normalization parameters.")
            self.register_buffer("running_mean", agent.state_preprocessor.running_mean.float().to(self.device))
            self.register_buffer("running_variance", agent.state_preprocessor.running_variance.float().to(self.device))
            self.has_preprocessor = True
        else:
            print("[WARNING] No state preprocessor found! Inputs will not be normalized.")
            self.register_buffer("running_mean", torch.zeros(1).to(self.device))
            self.register_buffer("running_variance", torch.ones(1).to(self.device))

        self.epsilon = 1e-8

        # 2. Extract Policy Network
        # skrl PPO agent has .policy
        # Move policy to CPU to avoid device mismatch during trace/export
        self.policy = agent.policy.to(self.device)
        self.policy.eval() # Ensure deterministic mode
        
    def forward(self, x):
        # Apply Normalization
        if self.has_preprocessor:
            x = (x - self.running_mean) / torch.sqrt(self.running_variance + self.epsilon)
        
        # Run Policy
        # We assume x is a tensor. skrl policy.act expects a dict {"states": x} typically,
        # but the internal 'net' might take x directly.
        # However, the most robust way to support skrl's architectures (like GaussianMixin)
        # is to call .act() and extract the mean. 
        # But .act() might do other things.
        # GaussianMixin.act returns (actions, log_prob, outputs)
        # where 'actions' is the mean in eval mode (if clip_actions=False).
        
        # We wrap the dictionary creation here to match skrl's expected input
        # Note: Jit Tracing will trace the execution path, so dictionary creation is fine 
        # if the policy supports it.
        
        # Using 'act' with role="policy"
        actions, _, _ = self.policy.act({"states": x}, role="policy")
        
        return actions


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent and export model."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # ---------------------------------------------------------
    # EXPORT LOGIC
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STARTING MODEL EXPORT")
    print("="*50)

    # 1. Analyze Environment for Scale/Offset
    # Try to find the action scaling and default offsets to inform the user
    try:
        # Access inner env
        unwrapped_env = env.unwrapped
        # Depending on wrapper depth, might need to go deeper. 
        # ManagerBasedRLEnv usually has action_manager
        if hasattr(unwrapped_env, "action_manager"):
            print("[INFO] Inspecting Action Manager...")
            # Assuming 'arm_action' is the name
            if "arm_action" in unwrapped_env.action_manager._terms:
                term = unwrapped_env.action_manager._terms["arm_action"]
                print(f"[INFO] arm_action configuration:")
                print(f"       Scale: {term.cfg.scale}")
                if hasattr(term.cfg, "use_default_offset"):
                    print(f"       Use Default Offset: {term.cfg.use_default_offset}")
                
                # Try to print default joints if available
                if hasattr(unwrapped_env.scene, "robot"):
                     print(f"[INFO] Robot Default Joints (first 7): {unwrapped_env.scene.robot.data.default_joint_pos[0, :7]}")
            else:
                 print("[INFO] 'arm_action' term not found in action manager.")
    except Exception as e:
        print(f"[WARNING] Could not inspect environment internals: {e}")

    # 2. Prepare Export Wrapper
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    export_model = ExportedPolicy(runner.agent, checkpoint_dict=checkpoint)
    
    # 3. Trace and Save
    # We need a dummy input.
    # env.reset() returns (obs, info)
    obs, _ = env.reset()
    # obs is a tensor on device, move to CPU for export
    obs = obs.cpu()
    
    print(f"[INFO] Tracing model with input shape: {obs.shape}")
    
    with torch.no_grad():
        # Run once to verify
        out = export_model(obs)
        print(f"[INFO] Model forward pass successful. Output shape: {out.shape}")
        
        # Trace
        # Check if we can strictly trace or need scripting. 
        # skrl policies often involve dictionaries, so strictly tracing might fail if we don't handle it carefully.
        # But our wrapper takes a Tensor and builds the dict inside. Tracing follows the tensor.
        traced_script_module = torch.jit.trace(export_model, obs)
        
        save_path = os.path.join(os.getcwd(), args_cli.export_path)
        traced_script_module.save(save_path)
        
        print(f"[SUCCESS] Model exported to: {save_path}")
        print(f"\n[IMPORTANT] The exported model expects input observations of shape {obs.shape} (normalized automatically internally).")
        print(f"[IMPORTANT] The output is the raw action (typically unscaled delta).")
        print(f"[IMPORTANT] Ensure your robot interface applies the correct scale and offset (default joint positions) to this output.")
        print("="*50 + "\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
