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
{
  joint_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 1);

  vehicle_actuation_sub_ = this->create_subscription<mpclab_msgs::msg::VehicleActuationMsg>(
    "vehicle_actuation", 1,
    std::bind(
      &VehicleStateVisualizerNode::vehicle_actuation_callback, this,
      std::placeholders::_1));
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
