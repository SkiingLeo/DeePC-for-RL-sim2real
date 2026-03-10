#!/usr/bin/env python3
"""
Simple P-controller that consumes digital twin joint differences and publishes
torques to deepc/py_controller for execution by the PyTorqueRunner bridge.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

NUM_JOINTS = 7
GAIN = 5.0  # tau = GAIN * diff
DT = 0.1  # 10 Hz


class PControlGap(Node):
    def __init__(self) -> None:
        super().__init__("p_control_gap")
        self._latest_diff = [0.0] * NUM_JOINTS
        self._have_diff = False

        self._diff_sub = self.create_subscription(
            JointState,
            "/digital_twin/joint_differences",
            self._diff_cb,
            10,
        )
        self._tau_pub = self.create_publisher(JointState, "deepc/py_controller", 10)
        self._timer = self.create_timer(DT, self._tick)

        self.get_logger().info(
            f"PControlGap running at {1/DT:.1f} Hz, publishing torques to deepc/py_controller"
        )

    def _diff_cb(self, msg: JointState) -> None:
        if len(msg.position) < NUM_JOINTS:
            self.get_logger().warn(
                f"Received diff vector of size {len(msg.position)}, expected {NUM_JOINTS}"
            )
            return
        self._latest_diff = list(msg.position[:NUM_JOINTS])
        self._have_diff = True

    def _tick(self) -> None:
        if not self._have_diff:
            return
        efforts = [GAIN * d for d in self._latest_diff]
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = [f"tau_{i+1}" for i in range(NUM_JOINTS)]
        out.effort = efforts
        self._tau_pub.publish(out)


def main() -> None:
    rclpy.init()
    node = PControlGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down PControlGap...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
