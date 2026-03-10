#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
import numpy as np
import os
from datetime import datetime
from state_preparation import start_state_preparation_node, get_rl_ready_state_reach
from single_inference import initialize_model, single_inference_run
from franka_isaac_translation import translate_isaac_position_to_franka


class RLPolicy(Node):
    """Node that runs RL policy inference for reach task to control robot"""

    def __init__(self):
        super().__init__('rl_policy_reach')

        # Initialize state preparation module
        start_state_preparation_node()

        # Get the directory where this script is located
        script_dir = os.path.dirname(__file__)

        # Initialize neural network model
        model_path = os.path.join(script_dir, 'exported_policy.pt')
        self.get_logger().info(f"Loading model from: {model_path}")
        self.model = initialize_model(model_path)
        self.get_logger().info("Model loaded successfully")
        
        self.joint_limit_scaling_factor = 1
        # Joint limits (min, max) for FR3
        self.joint_limits = [
            (-2.8973, 2.8973),    # panda_joint0
            (-1.7628, 1.7628),    # panda_joint1
            (-2.8973, 2.8973),    # panda_joint2
            (-3.0718, -0.0698),   # panda_joint3
            (-2.8973, 2.8973),    # panda_joint4
            (-0.0175, 3.7525),    # panda_joint5
            (-2.8973, 2.8973)     # panda_joint6
        ]

        # tightening the limits, for more safety/robost behaviour

        self.joint_limits = self.joint_limit_scaling_factor * self.joint_limits

        # State variables
        self.current_position = None
        self.iteration_count = 0
        
        # Load initial pose from CSV
        initial_pose_path = os.path.join(script_dir, 'initial_pose.csv')
        self.get_logger().info(f"Loading initial pose from: {initial_pose_path}")
        with open(initial_pose_path, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            self.initial_pose = np.array([
                float(row['pose_command_0']),
                float(row['pose_command_1']),
                float(row['pose_command_2']),
                float(row['pose_command_3']),
                float(row['pose_command_4']),
                float(row['pose_command_5']),
                float(row['pose_command_6'])
            ], dtype=np.float64)
        self.initial_pose = translate_isaac_position_to_franka(self.initial_pose)
        self.get_logger().info(f"Initial pose loaded: {self.initial_pose}")
        
        # Number of iterations to hold initial pose, 20 seconds
        self.INITIAL_POSE_ITERATIONS = 400

        # Create publisher to RL policy controller
        self.publisher_ = self.create_publisher(
            JointState,
            '/rl_policy_controller/desired_joint_positions',
            10
        )

        # Create CSV logger for RL-ready state and actions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_log_path = os.path.join(script_dir, f'rl_ready_state_{timestamp}.csv')
        self.state_log_file = open(self.state_log_path, 'w', newline='')
        self.state_log_writer = csv.writer(self.state_log_file)
        # Write header with 32 state dimensions + 7 action dimensions
        header = [f'state_{i}' for i in range(32)] + [f'action_{i}' for i in range(7)]
        self.state_log_writer.writerow(header)

        # Create timer - 20Hz publisher (50ms period)
        self.publish_timer = self.create_timer(0.05, self.publish_callback)

        self.get_logger().info("RL Policy Reach Node started, running neural network inference at 20Hz")
        self.get_logger().info(f"Will hold initial pose for {self.INITIAL_POSE_ITERATIONS} iterations before RL control")
        self.get_logger().info(f"Logging state and actions to: {self.state_log_path}")

    def publish_callback(self):
        """Run neural network inference and publish desired joint positions at 20Hz"""
        # Increment iteration counter
        self.iteration_count += 1
        
        # Get current state (32-dimensional)
        rl_state = get_rl_ready_state_reach()
        
        # First 100 iterations: hold initial pose
        if self.iteration_count <= self.INITIAL_POSE_ITERATIONS:
            # Clip initial pose to joint limits
            self.current_position = np.array([
                np.clip(self.initial_pose[i], self.joint_limits[i][0], self.joint_limits[i][1])
                for i in range(7)
            ])
            
            if self.iteration_count == 1:
                self.get_logger().info(f"Holding initial pose for {self.INITIAL_POSE_ITERATIONS} iterations (20 seconds)...")
            elif self.iteration_count == self.INITIAL_POSE_ITERATIONS:
                self.get_logger().info("Initial pose holding complete. Starting RL control.")
        else:
            # After 100 iterations: run RL policy
            # Run neural network inference (returns 7 values)
            arm_actions = single_inference_run(rl_state, self.model)
            
            # Clip to joint limits
            self.current_position = np.array([
                np.clip(arm_actions[i], self.joint_limits[i][0], self.joint_limits[i][1])
                for i in range(7)
            ])
            #overwrite
            self.current_position = arm_actions
        
        # Publish desired current position
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.current_position.tolist()
        msg.velocity = []
        msg.effort = []
        msg.name = ["desired_joint_0","desired_joint_1","desired_joint_2","desired_joint_3","desired_joint_4","desired_joint_5","desired_joint_6"]
        self.publisher_.publish(msg)
        
        # Log RL-ready state and published actions
        log_row = np.concatenate([rl_state, self.current_position])
        self.state_log_writer.writerow(log_row)


def main(args=None):
    rclpy.init(args=args)

    rl_policy = RLPolicy()

    try:
        rclpy.spin(rl_policy)
    except KeyboardInterrupt:
        pass
    finally:
        # Close log file before destroying node
        if hasattr(rl_policy, 'state_log_file'):
            rl_policy.state_log_file.close()

    rl_policy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
