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

#ifndef LMPC_UTILS__ROS_CASADI_HELPER_HPP_
#define LMPC_UTILS__ROS_CASADI_HELPER_HPP_

#include <std_msgs/msg/float64_multi_array.hpp>

#include <casadi/casadi.hpp>

namespace lmpc
{
namespace utils
{
casadi::DM ros_array_to_dm(const std_msgs::msg::Float64MultiArray & array);
std_msgs::msg::Float64MultiArray dm_to_ros_array(const casadi::DM & array);
}  // namespace utils
}  // namespace lmpc
#endif  // LMPC_UTILS__ROS_CASADI_HELPER_HPP_
