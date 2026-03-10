#!/usr/bin/env python3
"""
Publish end-effector pose for performance metrics.

Current scope:
- Subscribe to Franka measured joint state from /franka_robot_state_broadcaster/robot_state
- Compute forward kinematics using the existing Franka URDF
- Publish PoseStamped with EE position (XYZ) and orientation (quaternion) of panda_hand
"""

import os
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from franka_msgs.msg import FrankaRobotState  # type: ignore[import]
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from scipy.spatial.transform import Rotation as R
import yourdfpy


class PerformanceMetricsPublisher(Node):
    """ROS2 node that publishes the Franka EE pose from joint states."""

    def __init__(self) -> None:
        super().__init__("performance_metrics_publisher")

        self._base_frame = "panda_link0"
        self._ee_frame = "panda_hand"
        self._joint_names = [f"panda_joint{i}" for i in range(1, 8)]

        self._robot: Optional[yourdfpy.URDF] = self._load_urdf()

        self._pose_pub = self.create_publisher(
            PoseStamped, "/performance_metrics/ee_pose", 10
        )

        # reward output (additive; does not affect existing publishing)
        self._reward_pub = self.create_publisher(
            Float64, "/performance_metrics/total_reward", 10
        )

        self._state_sub = self.create_subscription(
            FrankaRobotState,
            "/franka_robot_state_broadcaster/robot_state",
            self._state_callback,
            10,
        )

        # additional inputs for reward computation
        self._target_pose_sub = self.create_subscription(
            JointState,
            "/target_pose",
            self._target_pose_callback,
            10,
        )
        self._desired_positions_sub = self.create_subscription(
            JointState,
            "/rl_policy_controller/desired_joint_positions",
            self._desired_positions_callback,
            10,
        )

        # cached state (updated by callbacks)
        self._ee_pos_xyz: Optional[np.ndarray] = None
        self._ee_quat_wxyz: Optional[np.ndarray] = None
        self._joint_vel_7: Optional[np.ndarray] = None

        self._cmd_pos_xyz: Optional[np.ndarray] = None
        self._cmd_quat_wxyz: Optional[np.ndarray] = None

        self._prev_action_7: Optional[np.ndarray] = None

        # throttle for missing-input logs
        self._last_missing_log_s: float = 0.0

        # reward params (defaults match reach_env_cfg.py)
        self.declare_parameter("reward_weights.pos", -0.2)
        self.declare_parameter("reward_weights.pos_fine", 0.1)
        self.declare_parameter("reward_weights.rot", -0.1)
        self.declare_parameter("reward_weights.action_rate", -0.001)
        self.declare_parameter("reward_weights.joint_vel", -0.001)
        self.declare_parameter("reward_params.pos_fine_std", 0.1)

        self.get_logger().info(
            "performance_metrics_publisher ready: subscribing to "
            "/franka_robot_state_broadcaster/robot_state and publishing "
            "/performance_metrics/ee_pose; additionally computing live reach reward and publishing "
            "/performance_metrics/total_reward"
        )

    def _load_urdf(self) -> Optional[yourdfpy.URDF]:
        """Load the Franka URDF once for FK."""
        script_dir = os.path.dirname(__file__)
        default_path = os.path.join(script_dir, "lula_franka_gen.urdf")
        fallback_path = "lula_franka_gen.urdf"

        for path in (default_path, fallback_path):
            if not os.path.exists(path):
                continue
            try:
                robot = yourdfpy.URDF.load(
                    path, load_meshes=False, load_collision_meshes=False
                )
                self.get_logger().info(f"Loaded URDF from {path}")
                return robot
            except Exception as exc:  # pragma: no cover - defensive
                self.get_logger().error(f"Failed to load URDF at {path}: {exc}")
        self.get_logger().error(
            "Could not load URDF (lula_franka_gen.urdf). FK publishing disabled."
        )
        return None

    def _state_callback(self, msg: FrankaRobotState) -> None:
        """Handle incoming robot state, compute FK, and publish pose."""
        if self._robot is None:
            return

        positions = msg.measured_joint_state.position
        if len(positions) < len(self._joint_names):
            self.get_logger().warn(
                f"Received {len(positions)} joint positions, expected "
                f"{len(self._joint_names)}"
            )
            return

        cfg = {name: positions[i] for i, name in enumerate(self._joint_names)}

        try:
            self._robot.update_cfg(cfg)
            tf_matrix = self._robot.get_transform(self._ee_frame, self._base_frame)
        except Exception as exc:  # pragma: no cover - defensive
            self.get_logger().error(f"FK computation failed: {exc}")
            return

        pos = tf_matrix[:3, 3]
        rot_mat = tf_matrix[:3, :3]

        # scipy returns quaternion as (x, y, z, w); geometry_msgs expects the same order
        quat = R.from_matrix(rot_mat).as_quat()

        # cache EE pose for reward computation (IsaacLab uses wxyz)
        self._ee_pos_xyz = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        self._ee_quat_wxyz = np.array([float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])], dtype=np.float64)

        # cache joint velocities for reward computation (first 7 joints)
        velocities = msg.measured_joint_state.velocity
        if len(velocities) >= 7:
            self._joint_vel_7 = np.array(velocities[:7], dtype=np.float64)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self._base_frame

        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])

        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        self._pose_pub.publish(pose_msg)

    def _target_pose_callback(self, msg: JointState) -> None:
        """Cache the latest target pose command (pos[0:3]=XYZ, pos[3:7]=WXYZ)."""
        if len(msg.position) < 7:
            return
        self._cmd_pos_xyz = np.array(msg.position[:3], dtype=np.float64)
        self._cmd_quat_wxyz = np.array(msg.position[3:7], dtype=np.float64)

    @staticmethod
    def _quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(q))
        if n < eps:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiply in (w,x,y,z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z], dtype=np.float64)

    @staticmethod
    def _axis_angle_from_quat_wxyz(q: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
        """Match isaaclab.utils.math.axis_angle_from_quat (wxyz)."""
        # ensure unique representation (w >= 0)
        if q[0] < 0.0:
            q = -q
        xyz = q[1:]
        mag = float(np.linalg.norm(xyz))
        half_angle = math.atan2(mag, float(q[0]))
        angle = 2.0 * half_angle
        if abs(angle) > eps:
            s = math.sin(half_angle) / angle
        else:
            s = 0.5 - (angle * angle) / 48.0
        if abs(s) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return xyz / s

    @classmethod
    def _quat_error_magnitude_wxyz(cls, q1: np.ndarray, q2: np.ndarray) -> float:
        """Match isaaclab.utils.math.quat_error_magnitude (wxyz)."""
        q1n = cls._quat_normalize_wxyz(q1)
        q2n = cls._quat_normalize_wxyz(q2)
        quat_diff = cls._quat_mul_wxyz(q1n, cls._quat_conjugate_wxyz(q2n))  # q1 * q2^-1
        axis_angle = cls._axis_angle_from_quat_wxyz(quat_diff)
        return float(np.linalg.norm(axis_angle))

    def _desired_positions_callback(self, msg: JointState) -> None:
        """Compute and publish reward at the (slowest) desired-joint update rate (~20Hz)."""
        if len(msg.position) < 7:
            return

        missing = []
        if self._ee_pos_xyz is None or self._ee_quat_wxyz is None:
            missing.append("ee_pose")
        if self._cmd_pos_xyz is None or self._cmd_quat_wxyz is None:
            missing.append("target_pose")
        if self._joint_vel_7 is None:
            missing.append("joint_vel")

        if missing:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s - self._last_missing_log_s > 5.0:
                self.get_logger().warn(
                    f"Reward not published yet; missing inputs: {', '.join(missing)}"
                )
                self._last_missing_log_s = now_s
            return

        # local bindings (helps type checkers and avoids repeated attribute lookups)
        ee_pos_xyz = self._ee_pos_xyz
        ee_quat_wxyz = self._ee_quat_wxyz
        cmd_pos_xyz = self._cmd_pos_xyz
        cmd_quat_wxyz = self._cmd_quat_wxyz
        joint_vel_7 = self._joint_vel_7
        assert ee_pos_xyz is not None
        assert ee_quat_wxyz is not None
        assert cmd_pos_xyz is not None
        assert cmd_quat_wxyz is not None
        assert joint_vel_7 is not None

        # params (read at runtime to allow tuning via ROS params)
        w_pos = float(self.get_parameter("reward_weights.pos").value)
        w_pos_fine = float(self.get_parameter("reward_weights.pos_fine").value)
        w_rot = float(self.get_parameter("reward_weights.rot").value)
        w_action_rate = float(self.get_parameter("reward_weights.action_rate").value)
        w_joint_vel = float(self.get_parameter("reward_weights.joint_vel").value)
        pos_fine_std = float(self.get_parameter("reward_params.pos_fine_std").value)

        # A/B: position errors
        pos_error = float(np.linalg.norm(ee_pos_xyz - cmd_pos_xyz))
        rew_pos = w_pos * pos_error
        rew_pos_fine = w_pos_fine * (1.0 - math.tanh(pos_error / max(pos_fine_std, 1e-9)))

        # C: orientation error (IsaacLab uses quat_error_magnitude)
        rot_error = self._quat_error_magnitude_wxyz(ee_quat_wxyz, cmd_quat_wxyz)
        rew_rot = w_rot * rot_error

        # D: action rate L2 (sum square)
        action_7 = np.array(msg.position[:7], dtype=np.float64)
        if self._prev_action_7 is None:
            self._prev_action_7 = action_7.copy()
        action_rate_l2 = float(np.sum(np.square(action_7 - self._prev_action_7)))
        self._prev_action_7 = action_7
        rew_action_rate = w_action_rate * action_rate_l2

        # E: joint velocity L2 (sum square)
        joint_vel_l2 = float(np.sum(np.square(joint_vel_7)))
        rew_joint_vel = w_joint_vel * joint_vel_l2

        total_reward = rew_pos + rew_pos_fine + rew_rot + rew_action_rate + rew_joint_vel

        out = Float64()
        out.data = float(total_reward)
        self._reward_pub.publish(out)


def main() -> None:
    rclpy.init()
    node = PerformanceMetricsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down performance_metrics_publisher")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()