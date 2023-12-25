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
casadi::DM ros_array_to_dm(const std_msgs::msg::Float64MultiArray & array)
{
  // check if the array is 1D
  if (array.layout.dim.size() == 1) {
    return casadi::DM(array.data);
  } else if (array.layout.dim.size() == 2) {
    return casadi::DM::reshape(
      casadi::DM(array.data), array.layout.dim[0].size,
      array.layout.dim[1].size);
  } else {
    throw std::runtime_error("Cannot convert array with more than 2 dimensions to casadi::DM");
  }
}

std_msgs::msg::Float64MultiArray dm_to_ros_array(const casadi::DM & array)
{
  // for 1D array, only one dimension is needed
  std_msgs::msg::Float64MultiArray ros_array;
  ros_array.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
  ros_array.layout.dim[0].size = array.size1();
  ros_array.layout.dim[0].stride = array.size1() * array.size2();
  ros_array.layout.dim[0].label = "rows";
  ros_array.data = array.get_elements();
  if (array.size2() > 1) {
    ros_array.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
    ros_array.layout.dim[1].size = array.size2();
    ros_array.layout.dim[1].stride = array.size2();
    ros_array.layout.dim[1].label = "cols";
  }
  return ros_array;
}
}  // namespace utils
}  // namespace lmpc
#endif  // LMPC_UTILS__ROS_CASADI_HELPER_HPP_
