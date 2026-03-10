import numpy as np
import torch
from numpy.typing import NDArray
from typing import Any

from franka_isaac_translation import *

def initialize_model(model_path: str) -> torch.jit.ScriptModule:
    """
    Loads the exported SKRL policy (with observation normalization and action squashing).
    Args:
        model_path: Path to the TorchScript file produced by play_to_export_policy.py
                    e.g. ".../logs/skrl/<run>/exported_policy.pt"
    """
    model = torch.jit.load(model_path, map_location="cpu").eval()
    return model

def single_inference_run(input_vector: NDArray[np.float32], model: Any) -> NDArray[np.float32]: #size of input 32, size of output 7

    #the input_vector is built like this (looking at indices):
    """
    The input to the neural net is now of a different scheme, namely the input vector is:

    joint_pos, shape=(9,)
    joint_vel shape=(9,)
    pose_command shape=(7,)
    actions shape=(7,)

    for a total of 32 input dimensions.
    the output vector is now:
    arm_actions shape=(7,)

    for a total of 7 output dimensions.
    """

    # the output is in isaacframe, so needs to be translated into franka frame.
    
    # the input vector according to the scheme above
    joint_positions = input_vector[0:9]  # 9 joints (7 arm + 2 gripper)
    joint_velocities = input_vector[9:18]  # 9 velocities
    pose_command = input_vector[18:25]  # 7-dimensional pose command
    actions = input_vector[25:32]  # 7-dimensional actions
    
    # Translate only the arm joints/actions (first 7 of each, gripper doesn't need translation)
    arm_positions_translated = translate_franka_position_to_isaac(joint_positions[0:7])    
    arm_actions_translated = translate_franka_actions_to_isaac(actions[0:7])
    
    # Reconstruct the input vector with translated arm components and unchanged gripper components
    translated_input = np.concatenate([
        arm_positions_translated,
        joint_positions[7:9],  # gripper positions unchanged
        joint_velocities[0:7],  # arm velocities
        joint_velocities[7:9],  # gripper velocities unchanged
        pose_command,
        arm_actions_translated
    ])
    
    input_tensor = torch.from_numpy(translated_input).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    output_numpy = output.squeeze().numpy()
    
    # Output is 7-dimensional arm actions only
    arm_actions_output = output_numpy#[0:7]
    arm_actions_translated = translate_isaac_actions_to_franka(arm_actions_output)

    return arm_actions_translated
