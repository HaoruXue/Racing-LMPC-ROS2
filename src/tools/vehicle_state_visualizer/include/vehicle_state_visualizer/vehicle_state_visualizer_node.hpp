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

#ifndef VEHICLE_STATE_VISUALIZER__VEHICLE_STATE_VISUALIZER_NODE_HPP_
#define VEHICLE_STATE_VISUALIZER__VEHICLE_STATE_VISUALIZER_NODE_HPP_

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <mpclab_msgs/msg/vehicle_actuation_msg.hpp>

#include <lmpc_transform_helper/lmpc_transform_helper.hpp>

namespace lmpc
{
namespace vehicle_state_visualizer
{
class VehicleStateVisualizerNode : public rclcpp::Node
{
public:
  explicit VehicleStateVisualizerNode(const rclcpp::NodeOptions & options);

protected:
  // name of the two front wheel joints
  std::string fl_joint_;
  std::string fr_joint_;

  // subscribers (from controller)
  rclcpp::Subscription<mpclab_msgs::msg::VehicleActuationMsg>::SharedPtr vehicle_actuation_sub_ {};

  // publishers
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_publisher_;

  // callback
  void vehicle_actuation_callback(const mpclab_msgs::msg::VehicleActuationMsg::SharedPtr msg);
};
}  // namespace vehicle_state_visualizer
}  // namespace lmpc
#endif  // VEHICLE_STATE_VISUALIZER__VEHICLE_STATE_VISUALIZER_NODE_HPP_
