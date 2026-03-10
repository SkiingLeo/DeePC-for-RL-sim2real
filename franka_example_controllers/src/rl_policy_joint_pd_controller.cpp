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

#include <franka_example_controllers/rl_policy_joint_pd_controller.hpp>

#include <cassert>
#include <exception>
#include <string>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
RLPolicyJointPDController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
RLPolicyJointPDController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type RLPolicyJointPDController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates();

  Vector7d q_desired_eigen = Vector7d::Zero();
  {
    std::lock_guard<std::mutex> lock(q_desired_mutex_);
    if (!q_desired_.empty()) {
      q_desired_eigen = Eigen::Map<Vector7d>(q_desired_.data());
    }
  }

  const double kAlpha = 1;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;

  // *** SUBSCRIPTION POINT 3: Use desired positions from subscriber in control law ***
  Vector7d tau_d_pd =
      k_gains_.cwiseProduct(q_desired_eigen - q_) - d_gains_.cwiseProduct(dq_filtered_);
  // Clip PD contribution
  tau_d_pd = tau_d_pd.cwiseMax(-effort_limit_).cwiseMin(effort_limit_);

  Vector7d tau_cmd = tau_d_pd;

  // Optionally add python-provided torques if fresh
  if (enable_py_torque_) {
    std::vector<double> tau_py_copy;
    rclcpp::Time stamp_copy;
    {
      std::lock_guard<std::mutex> lock(tau_py_mutex_);
      tau_py_copy = tau_py_;
      stamp_copy = tau_py_stamp_;
    }
    if (tau_py_copy.size() >= static_cast<size_t>(num_joints)) {
      const double age = (get_node()->now() - stamp_copy).seconds();
      if (age <= py_torque_timeout_) {
        Vector7d tau_py = Eigen::Map<Vector7d>(tau_py_copy.data());
        tau_py = tau_py.cwiseMax(-effort_limit_py_torque_).cwiseMin(effort_limit_py_torque_);
        for (int i = 0; i < num_joints && i < static_cast<int>(tau_py_copy.size()); ++i) {
          tau_cmd(i) += tau_py(i);
        }
      }
    }
  }

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }

  return controller_interface::return_type::OK;
}

CallbackReturn RLPolicyJointPDController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("effort_limit", 87.0);
    auto_declare<bool>("enable_py_torque", false);
    auto_declare<std::string>("py_torque_topic", "deepc/py_controller");
    auto_declare<double>("py_torque_timeout", 1.0);
    auto_declare<double>("effort_limit_py_torque", 2.0);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn RLPolicyJointPDController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();

  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (k_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size %d but is of size %ld",
                 num_joints, k_gains.size());
    return CallbackReturn::FAILURE;
  }
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (d_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size %d but is of size %ld",
                 num_joints, d_gains.size());
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    k_gains_(i) = k_gains.at(i);
    d_gains_(i) = d_gains.at(i);
  }

  effort_limit_ = get_node()->get_parameter("effort_limit").as_double();
  enable_py_torque_ = get_node()->get_parameter("enable_py_torque").as_bool();
  py_torque_topic_ = get_node()->get_parameter("py_torque_topic").as_string();
  py_torque_timeout_ = get_node()->get_parameter("py_torque_timeout").as_double();
  effort_limit_py_torque_ = get_node()->get_parameter("effort_limit_py_torque").as_double();

  dq_filtered_.setZero();

  // *** SUBSCRIPTION POINT 4: Create ROS2 subscriber in on_configure ***
  desired_positions_sub_ =
      get_node()->create_subscription<sensor_msgs::msg::JointState>(
          "~/desired_joint_positions", 10,
          std::bind(&RLPolicyJointPDController::desired_positions_callback, this,
                    std::placeholders::_1));

  if (enable_py_torque_) {
    py_torque_sub_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
        py_torque_topic_, 10,
        std::bind(&RLPolicyJointPDController::py_torque_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_node()->get_logger(), "Python torque blending enabled from topic: %s",
                py_torque_topic_.c_str());
  } else {
    RCLCPP_INFO(get_node()->get_logger(), "Python torque blending disabled");
  }

  RCLCPP_INFO(get_node()->get_logger(), "RLPolicyJointPDController configured");
  return CallbackReturn::SUCCESS;
}

CallbackReturn RLPolicyJointPDController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();

  {
    std::lock_guard<std::mutex> lock(q_desired_mutex_);
    q_desired_.assign(num_joints, 0.0);
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn RLPolicyJointPDController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  return CallbackReturn::SUCCESS;
}

void RLPolicyJointPDController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

// *** SUBSCRIPTION POINT 5: Subscriber callback receives desired positions ***
void RLPolicyJointPDController::desired_positions_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(q_desired_mutex_);
  if (msg->position.size() >= static_cast<size_t>(num_joints)) {
    q_desired_.assign(msg->position.begin(), msg->position.begin() + num_joints);
  }
}

void RLPolicyJointPDController::py_torque_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  if (msg->effort.size() < static_cast<size_t>(num_joints)) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                         "Received py torque size %zu, expected %d; ignoring.",
                         msg->effort.size(), num_joints);
    return;
  }

  std::lock_guard<std::mutex> lock(tau_py_mutex_);
  tau_py_.assign(msg->effort.begin(), msg->effort.begin() + num_joints);
  tau_py_stamp_ = get_node()->now();
}

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::RLPolicyJointPDController,
                       controller_interface::ControllerInterface)
