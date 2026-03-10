"""
ZMQ <-> ROS2 bridge to move joint messages between ROS (py3.10) and
IsaacLab (py3.11) without cross-version imports. Runs on the ROS side.
Targets sub-ms latency by using small pickle envelopes over PUB/SUB.
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from typing import Any, Dict, Optional

import rclpy
from franka_msgs.msg import FrankaRobotState  # type: ignore
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
import zmq

# Socket layout:
# - out_pub (PUB): ROS -> IsaacLab (bind)
# - in_sub (SUB): IsaacLab -> ROS (connect)
ROS_TO_ZMQ_ADDR = "tcp://0.0.0.0:6001"
ZMQ_TO_ROS_ADDR = "tcp://127.0.0.1:6002"

PICKLE_PROTOCOL = 5
NUM_JOINTS = 7
SNDHWM = 10
RCVHWM = 10

# Topics
ROS_OUTBOUND_TOPICS = [
    "/rl_policy_controller/desired_joint_positions",  # JointState
    "/franka_robot_state_broadcaster/robot_state",  # FrankaRobotState (positions used)
]

ZMQ_INBOUND_TOPICS = [
    "/digital_twin/joint_differences",  # JointState
    "/digital_twin/sim_joints",  # JointState
    "/digital_twin/sim_torques",  # Float64MultiArray (effort)
]


def make_envelope(topic: str, payload: Dict[str, Any]) -> bytes:
    """Serialize payload with metadata."""
    envelope = {"topic": topic, "t_mono": time.monotonic(), "payload": payload}
    return pickle.dumps(envelope, protocol=PICKLE_PROTOCOL)


def parse_envelope(raw: bytes) -> Dict[str, Any]:
    """Deserialize envelope."""
    return pickle.loads(raw)


def joint_state_to_payload(msg: JointState) -> Dict[str, Any]:
    return {
        "position": list(msg.position[:NUM_JOINTS]),
        "velocity": list(msg.velocity[:NUM_JOINTS]) if msg.velocity else None,
        "effort": list(msg.effort[:NUM_JOINTS]) if msg.effort else None,
    }


def franka_state_to_payload(msg: FrankaRobotState) -> Dict[str, Any]:
    js = msg.measured_joint_state
    return {
        "position": list(js.position[:NUM_JOINTS]),
        "velocity": list(js.velocity[:NUM_JOINTS]) if js.velocity else None,
        "effort": list(js.effort[:NUM_JOINTS]) if js.effort else None,
    }


def payload_to_joint_state(payload: Dict[str, Any], stamp) -> JointState:
    out = JointState()
    out.header.stamp = stamp
    pos = payload.get("position") or []
    vel = payload.get("velocity") or []
    eff = payload.get("effort") or []
    out.position = pos[:NUM_JOINTS]
    if vel:
        out.velocity = vel[:NUM_JOINTS]
    if eff:
        out.effort = eff[:NUM_JOINTS]
    return out


class RosZmqBridge(Node):
    """Bridge ROS topics to ZMQ and back."""

    def __init__(self):
        super().__init__("ros_zmq_bridge")
        self._ctx = zmq.Context(io_threads=1)
        self._out_pub = self._ctx.socket(zmq.PUB)
        self._out_pub.setsockopt(zmq.SNDHWM, SNDHWM)
        self._out_pub.setsockopt(zmq.LINGER, 0)
        self._out_pub.bind(ROS_TO_ZMQ_ADDR)

        self._in_sub = self._ctx.socket(zmq.SUB)
        self._in_sub.setsockopt(zmq.RCVHWM, RCVHWM)
        self._in_sub.setsockopt(zmq.LINGER, 0)
        for topic in ZMQ_INBOUND_TOPICS:
            self._in_sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._in_sub.connect(ZMQ_TO_ROS_ADDR)

        self._poller = zmq.Poller()
        self._poller.register(self._in_sub, zmq.POLLIN)
        self._running = threading.Event()

        # ROS subscriptions (outbound to ZMQ)
        self._desired_sub = self.create_subscription(
            JointState,
            "/rl_policy_controller/desired_joint_positions",
            self._desired_cb,
            10,
        )
        self._real_sub = self.create_subscription(
            FrankaRobotState,
            "/franka_robot_state_broadcaster/robot_state",
            self._real_cb,
            10,
        )

        # ROS publishers (inbound from ZMQ)
        self._diff_pub = self.create_publisher(JointState, "/digital_twin/joint_differences", 10)
        self._sim_pub = self.create_publisher(JointState, "/digital_twin/sim_joints", 10)
        self._torque_pub = self.create_publisher(JointState, "/digital_twin/sim_torques", 10)

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)

        self.get_logger().info(
            f"ros_zmq_bridge ready | PUB {ROS_TO_ZMQ_ADDR} | SUB {ZMQ_TO_ROS_ADDR}"
        )

    # --- Outbound (ROS -> ZMQ) ---
    def _send(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            self._out_pub.send_multipart(
                [topic.encode(), make_envelope(topic, payload)],
                flags=zmq.DONTWAIT,
            )
        except zmq.Again:
            self.get_logger().warning("Outbound ZMQ buffer full; dropping message", throttle_duration_sec=1.0)
        except Exception as exc:  # pragma: no cover - defensive
            self.get_logger().error(f"Failed to publish to ZMQ: {exc}")

    def _desired_cb(self, msg: JointState) -> None:
        self._send("/rl_policy_controller/desired_joint_positions", joint_state_to_payload(msg))

    def _real_cb(self, msg: FrankaRobotState) -> None:
        self._send("/franka_robot_state_broadcaster/robot_state", franka_state_to_payload(msg))

    # --- Inbound (ZMQ -> ROS) ---
    def _recv_loop(self) -> None:
        while self._running.is_set():
            events = dict(self._poller.poll(timeout=1))  # 1 ms poll
            if self._in_sub not in events:
                continue

            try:
                topic_b, body = self._in_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                self.get_logger().error(f"ZMQ recv failed: {exc}")
                continue

            topic = topic_b.decode()
            try:
                env = parse_envelope(body)
                payload = env.get("payload", {})
            except Exception as exc:
                self.get_logger().warning(f"Bad envelope from ZMQ: {exc}")
                continue

            stamp = self.get_clock().now().to_msg()
            if topic == "/digital_twin/joint_differences":
                msg = payload_to_joint_state(payload, stamp)
                msg.name = [f"diff_{i+1}" for i in range(NUM_JOINTS)]
                self._diff_pub.publish(msg)
            elif topic == "/digital_twin/sim_joints":
                msg = payload_to_joint_state(payload, stamp)
                msg.name = [f"sim_joint_{i+1}" for i in range(NUM_JOINTS)]
                self._sim_pub.publish(msg)
            elif topic == "/digital_twin/sim_torques":
                msg = payload_to_joint_state(payload, stamp)
                msg.name = [f"sim_torque_{i+1}" for i in range(NUM_JOINTS)]
                self._torque_pub.publish(msg)
            else:
                self.get_logger().debug(f"Dropping unexpected topic from ZMQ: {topic}")

    def start(self) -> None:
        self._running.set()
        self._recv_thread.start()

    def shutdown_bridge(self) -> None:
        self._running.clear()
        self._recv_thread.join(timeout=1.0)
        self._poller.unregister(self._in_sub)
        self._in_sub.close(0)
        self._out_pub.close(0)
        self._ctx.term()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rclpy.init(args=None)

    node = RosZmqBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    node.start()

    try:
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ros_zmq_bridge...")
    finally:
        node.shutdown_bridge()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()