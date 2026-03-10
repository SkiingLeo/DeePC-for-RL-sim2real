import numpy as np

def translate_franka_position_to_isaac(joint_positions: np.ndarray) -> np.ndarray:
    panda_offsets = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    return joint_positions - panda_offsets

def translate_isaac_position_to_franka(joint_positions: np.ndarray) -> np.ndarray:
    panda_offsets = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    return joint_positions + panda_offsets

def translate_isaac_actions_to_franka(arm_action: np.ndarray) -> np.ndarray:
    panda_offsets = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    #return arm_action/2 + panda_offsets
    return arm_action/2.0 + panda_offsets

# franka action is just simply a desired point in fraka space
def translate_franka_actions_to_isaac(arm_action: np.ndarray) -> np.ndarray:
    panda_offsets = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    #return (arm_action - panda_offsets)*2
    return (arm_action - panda_offsets)*2.0