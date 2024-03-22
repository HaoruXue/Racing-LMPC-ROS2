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

#ifndef NEURAL_DYNAMICS_MODEL__ROS_PARAM_LOADER_HPP_
#define NEURAL_DYNAMICS_MODEL__ROS_PARAM_LOADER_HPP_

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "neural_dynamics_model/base_neural_dynamics_model.hpp"

namespace lmpc
{
namespace vehicle_model
{
namespace neural_dynamics_model
{
BaseNeuralDynamicsModelConfig::SharedPtr load_parameters(rclcpp::Node * node);
}  // namespace neural_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
#endif  // NEURAL_DYNAMICS_MODEL__ROS_PARAM_LOADER_HPP_
