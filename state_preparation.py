# this script contains functions that subscribe to various robot states, prepares data and 
# upon calling get_rl_ready_state_lift() should return a 36 dimenstional state vector that can 
# be used by the reinforcement learning neural network. This will be running at a frequency
# of 20Hz, thus, speed is of concern.

# this is all still in franka space, so the neural net calling will do the translation.

# all get_ functions should return neural net ready data, meaning the robot's ros2 output 
# should be translated to the correct format in each function seperately.

# the state will be read with a frequency of 20Hz, as mentioned above

# this reading of the state at 20Hz has to be ensured by the script that calls these functions!

from operator import concat
import numpy as np
import csv  # for target object position, this must be hard coded
import threading
from typing import Optional, Any

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from franka_msgs.msg import FrankaRobotState
from sensor_msgs.msg import JointState

import franka_isaac_translation
import os

# ============================================================================
# Module-level state management for thread-safe ROS2 node
# ============================================================================

_node: Optional[Node] = None
_executor: Optional[SingleThreadedExecutor] = None
_state_lock = threading.Lock()
_latest_robot_state: Optional[FrankaRobotState] = None
_spin_thread: Optional[threading.Thread] = None


# Target positions CSV reader, used for lift
_target_positions_file: Optional[Any] = None
_target_positions_reader: Optional[Any] = None
_target_positions_eof = False


# Target pose CSV reader, used for reach
_target_pose_file: Optional[Any] = None
_target_pose_reader: Optional[Any] = None
_target_pose_eof = False
_target_pose_publisher: Optional[Any] = None


# Previous actions state
_latest_desired_positions: Optional[np.ndarray] = None


def _robot_state_callback(msg: FrankaRobotState) -> None:
    """
    Stores the latest robot state message in a thread-safe manner.
    """
    global _latest_robot_state
    with _state_lock:
        _latest_robot_state = msg


def _desired_positions_callback(msg: JointState) -> None:
    """Callback for desired joint positions subscription."""
    global _latest_desired_positions
    _latest_desired_positions = np.array(msg.position[:7], dtype=np.float64)


def start_state_preparation_node() -> None:
    """
    This function should be called once at the start of the application.
    It creates a ROS2 node, subscribes to the robot state topic, and starts
    spinning in a separate daemon thread to handle callbacks asynchronously.
    
    This allows state_preparation.py to manage its own ROS2 lifecycle
    independently from rl_policy_test.py.
    """
    global _node, _executor, _spin_thread, _target_pose_publisher
    
    if _node is not None:
        return  # Already initialized
    
    # Initialize ROS2 if not already done
    if not rclpy.ok():
        rclpy.init()
    
    # state preparation node
    _node = Node('state_preparation_node')
    _target_pose_publisher = _node.create_publisher(JointState, '/target_pose', 10)
    
    # Subscribe to the robot state topic published by franka_robot_state_broadcaster
    _node.create_subscription(
        FrankaRobotState,
        '/franka_robot_state_broadcaster/robot_state',
        _robot_state_callback,
        10  # QoS depth
    )
    
    # Subscribe to desired positions from RL controller
    _node.create_subscription(
        JointState,
        '/rl_policy_controller/desired_joint_positions',
        _desired_positions_callback,
        10
    )
    
    _node.get_logger().info("State Preparation Node initialized and subscribing to /franka_robot_state_broadcaster/robot_state")
    
    #executor for this node
    _executor = SingleThreadedExecutor()
    _executor.add_node(_node)
    
    # separate daemon thread
    def spin_thread_func():
        _executor.spin()
    
    _spin_thread = threading.Thread(target=spin_thread_func, daemon=True)
    _spin_thread.start()
    
    # Initialize target positions CSV reader
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'target_positions.csv')
    globals()['_target_positions_file'] = open(csv_path, 'r')
    globals()['_target_positions_reader'] = csv.DictReader(globals()['_target_positions_file'])
    
    # Initialize target poses CSV reader
    csv_path_poses = os.path.join(script_dir, 'target_poses.csv')
    globals()['_target_pose_file'] = open(csv_path_poses, 'r')
    globals()['_target_pose_reader'] = csv.DictReader(globals()['_target_pose_file'])

def get_joint_positions() -> np.ndarray:
    """
    Returns a 9-dimensional state vector containing:
    - 7 measured joint positions (from robot state measured_joint_state.position)
    - 2 hardcoded gripper values: [-0.038, -0.038] the standard deviation was found to be 0.00566 in sim
    
    If no robot state has been received yet, returns zeros for the joint positions
    and the hardcoded gripper values.
    
    Returns:
        np.ndarray: 9-dimensional array of float64 joint positions and gripper values
    """
    # Hardcoded gripper values
    gripper_values = np.array([-0.038, -0.038], dtype=np.float64)
    
    with _state_lock:
        if _latest_robot_state is None:
            # No data available yet, return zeros for joints + gripper values
            return np.concatenate([np.zeros(7, dtype=np.float64), gripper_values])
        
        # Extract 7 joint positions from measured_joint_state
        joint_positions = np.array(
            _latest_robot_state.measured_joint_state.position[:7],
            dtype=np.float64
        )
        
        # Concatenate joint positions with gripper values to form 9-dim vector
        return np.concatenate([joint_positions, gripper_values])

def get_joint_velocities() -> np.ndarray:
    """
    Returns a 9-dimensional state vector containing:
    - 7 measured joint velocities (from robot state measured_joint_state.velocity)
    - 2 hardcoded gripper velocities: [-0.2, 0.2] (gripper positions are static)

    -> standard deviation of joint_vel_7 0.00606, joint_vel_8 0.00866 in sim
    
    If no robot state has been received yet, returns all zeros.
    
    Returns:
        np.ndarray: 9-dimensional array of float64 joint velocities
    """
    gripper_velocities = np.array([-0.2, 0.2], dtype=np.float64)
    
    with _state_lock:
        if _latest_robot_state is None:
            # No data available yet, return all zeros
            return np.concatenate([np.zeros(7, dtype=np.float64), gripper_velocities])
        
        # Extract 7 joint velocities from measured_joint_state
        joint_velocities = np.array(
            _latest_robot_state.measured_joint_state.velocity[:7],
            dtype=np.float64
        )        
        return np.concatenate([joint_velocities, gripper_velocities])

def get_object_positions() -> np.ndarray:
    """
    Returns a 3-dimensional position vector (X, Y, Z) corresponding to the
    measured end-effector position in the base frame. This assumes the object
    is always perfectly grasped by the end-effector.
    
    If no robot state has been received yet, returns all zeros.
    
    Returns:
        np.ndarray: 3-dimensional array of float64 (X, Y, Z) coordinates in meters
    """
    with _state_lock:
        if _latest_robot_state is None:
            # No data available yet, return zeros
            return np.zeros(3, dtype=np.float64)
        
        # Extract end-effector position (X, Y, Z) from measured pose in base frame
        ee_position = np.array([
            _latest_robot_state.o_t_ee.pose.position.x,
            _latest_robot_state.o_t_ee.pose.position.y,
            _latest_robot_state.o_t_ee.pose.position.z
        ], dtype=np.float64)
        
        return ee_position

def get_target_object_positions() -> np.ndarray:
    """
    Returns a 7-dimensional state vector:
    - First 3: Target (X, Y, Z) from CSV (one row per call)
    - Last 4: [1, 0, 0, 0] (fixed values)
    
    Reads sequentially through all CSV rows. After EOF, returns all zeros.
    
    Returns:
        np.ndarray: 7-dimensional array of float64
    """
    global _target_positions_eof
    
    fixed_values = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    if _target_positions_eof or _target_positions_reader is None:
        return np.concatenate([np.zeros(3, dtype=np.float64), fixed_values])
    
    try:
        row = next(_target_positions_reader)
        target_pos = np.array([
            float(row['target_object_position_0']),
            float(row['target_object_position_1']),
            float(row['target_object_position_2'])
        ], dtype=np.float64)
        return np.concatenate([target_pos, fixed_values])
    except StopIteration:
        _target_positions_eof = True
        return np.concatenate([np.zeros(3, dtype=np.float64), fixed_values])

def get_target_pose() -> np.ndarray:
    """
    Returns a 7-dimensional state vector with pose commands from CSV.
    Reads columns: pose_command_0 through pose_command_6
    
    Reads sequentially through all CSV rows. After EOF, returns all zeros.
    
    Returns:
        np.ndarray: 7-dimensional array of float64
    """
    global _target_pose_eof
    
    pose_command = np.zeros(7, dtype=np.float64)
    
    if not _target_pose_eof and _target_pose_reader is not None:
        try:
            row = next(_target_pose_reader)
            pose_command = np.array([
                float(row['pose_command_0']),
                float(row['pose_command_1']),
                float(row['pose_command_2']),
                float(row['pose_command_3']),
                float(row['pose_command_4']),
                float(row['pose_command_5']),
                float(row['pose_command_6'])
            ], dtype=np.float64)
        except StopIteration:
            _target_pose_eof = True
    
    if _target_pose_publisher is not None and _node is not None:
        msg = JointState()
        msg.header.stamp = _node.get_clock().now().to_msg()
        msg.position = pose_command.tolist()
        msg.name = [f"pose_command_{i}" for i in range(7)]
        _target_pose_publisher.publish(msg)
    
    return pose_command

def get_previous_actions() -> np.ndarray:
    """
    Returns an 8-dimensional state vector:
    - First 7: Latest desired joint positions from /rl_policy_controller/desired_joint_positions
    - Last 1: Fixed gripper value of -2.5
    
    If no desired positions available yet, returns all zeros.
    
    Returns:
        np.ndarray: 8-dimensional array of float64
    """
    if _latest_desired_positions is None:
        return np.concatenate([np.zeros(7, dtype=np.float64), [-2.5]])
    
    return np.concatenate([_latest_desired_positions, [-2.5]])

def get_rl_ready_state_lift() -> np.ndarray:
    # total of 36 dimensions
    """
    scheme:

    joint pos 9 (7 joints 2 gripper)
    joint vel 9 (7 joints 2 gripper)
    object position 3 (x,y,z of cube)
    target object position 7 (xyz and fixed quaternion)
    last action 8 (7 joint positions, 1 fixed gripper)

    """
    return np.concatenate([
        get_joint_positions(),
        get_joint_velocities(),
        get_object_positions(),
        get_target_object_positions(),
        get_previous_actions()
    ])



def get_rl_ready_state_reach() -> np.ndarray:
    #should be 32 dimensional

    """
    scheme:

    joint pos 9 (7 joints 2 gripper)
    joint vel 9 (7 joints 2 gripper)
    target_pose 7 (x,y,z and quaternion), aka pose_command
    last action 7 (7 joint positions)
    """
    desired_positions = _latest_desired_positions if _latest_desired_positions is not None else np.zeros(7, dtype=np.float64)
    
    return np.concatenate([
        get_joint_positions(),
        get_joint_velocities(),
        get_target_pose(),
        desired_positions,
    ])
