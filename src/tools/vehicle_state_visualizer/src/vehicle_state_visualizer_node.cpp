// Copyright 2023 Haoru Xue
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <lmpc_utils/ros_param_helper.hpp>
#include "vehicle_state_visualizer/vehicle_state_visualizer_node.hpp"

namespace lmpc
{
namespace vehicle_state_visualizer
{
VehicleStateVisualizerNode::VehicleStateVisualizerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("vehicle_state_visualizer", options),
  fl_joint_(lmpc::utils::declare_parameter<std::string>(this, "fl_joint")),
  fr_joint_(lmpc::utils::declare_parameter<std::string>(this, "fr_joint"))
  // wheel_radius_(lmpc::utils::declare_parameter<double>(this, "wheel_radius"))
{
  joint_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 1);

  // last_wheel_spin_state_ = std::make_shared<sensor_msgs::msg::JointState>();
  // last_wheel_spin_state_->name = {
  //   "fl_tyre_rotate_joint", "fr_tyre_rotate_joint", "rl_tyre_rotate_joint", "rr_tyre_rotate_joint"};
  // last_wheel_spin_state_->position = {0.0, 0.0, 0.0, 0.0};

  vehicle_actuation_sub_ = this->create_subscription<mpclab_msgs::msg::VehicleActuationMsg>(
    "vehicle_actuation", 1,
    std::bind(
      &VehicleStateVisualizerNode::vehicle_actuation_callback, this,
      std::placeholders::_1));
  
  // vehicle_state_sub_ = this->create_subscription<mpclab_msgs::msg::VehicleStateMsg>(
  //   "vehicle_state", 1,
  //   std::bind(
  //     &VehicleStateVisualizerNode::vehicle_state_callback, this,
  //     std::placeholders::_1));
}

void VehicleStateVisualizerNode::vehicle_actuation_callback(
  const mpclab_msgs::msg::VehicleActuationMsg::SharedPtr msg)
{
  const auto & steer = msg->u_steer;
  const auto left_steer = steer > 0.0 ? steer * 0.8 : steer * 1.2;
  const auto right_steer = steer > 0.0 ? steer * 1.2 : steer * 0.8;

  // send the two front wheel joint states
  sensor_msgs::msg::JointState joint_state;
  joint_state.header.stamp = msg->header.stamp;
  joint_state.name = {fl_joint_, fr_joint_};
  joint_state.position = {left_steer, right_steer};
  joint_publisher_->publish(joint_state);
}

// void VehicleStateVisualizerNode::vehicle_state_callback(
//   const mpclab_msgs::msg::VehicleStateMsg::SharedPtr msg)
// {
//   if (!last_state_)
//   {
//     last_state_ = msg;
//     return;
//   }
//   const auto time_elapsed = msg->t - last_state_->t;
//   const auto v_lon = msg->v.v_long;
//   // convert from m/s to rad/s
//   const auto v_rot = v_lon / wheel_radius_;
//   const auto delta_theta = v_rot * time_elapsed;
//   for (auto & pos : last_wheel_spin_state_->position)
//   {
//     pos = fmod(pos + delta_theta, 2 * M_PI);
//   }
//   last_wheel_spin_state_->header.stamp = msg->header.stamp;
//   joint_publisher_->publish(*last_wheel_spin_state_);
// }
}  // namespace vehicle_state_visualizer
}  // namespace lmpc

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options{};
  auto node = std::make_shared<lmpc::vehicle_state_visualizer::VehicleStateVisualizerNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
