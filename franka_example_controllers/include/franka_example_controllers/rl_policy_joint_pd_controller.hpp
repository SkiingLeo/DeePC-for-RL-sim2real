// Copyright (c) 2024 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>
#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

class RLPolicyJointPDController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  void desired_positions_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void py_torque_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void updateJointStates();

  // *** SUBSCRIPTION POINT 1: Member variable for ROS2 subscriber ***
  std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::JointState>> desired_positions_sub_;
  std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::JointState>> py_torque_sub_;

  // *** SUBSCRIPTION POINT 2: Desired joint positions from subscriber (thread-safe) ***
  std::vector<double> q_desired_;
  std::mutex q_desired_mutex_;

  // Python torque blending (optional)
  std::vector<double> tau_py_;
  rclcpp::Time tau_py_stamp_;
  std::mutex tau_py_mutex_;
  bool enable_py_torque_{false};
  std::string py_torque_topic_;
  double py_torque_timeout_{1.0};
  double effort_limit_py_torque_{2.0};

  std::string arm_id_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  double effort_limit_;
};

}  // namespace franka_example_controllers
