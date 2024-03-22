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

#include <string>
#include <memory>
#include <vector>

#include <lmpc_utils/ros_param_helper.hpp>

#include "neural_dynamics_model/ros_param_loader.hpp"

namespace lmpc
{
namespace vehicle_model
{
namespace neural_dynamics_model
{
BaseNeuralDynamicsModelConfig::SharedPtr load_parameters(rclcpp::Node * node)
{
  auto declare_double = [&](const char * name) {
      return lmpc::utils::declare_parameter<double>(node, name);
    };
  auto declare_bool = [&](const char * name) {
      return lmpc::utils::declare_parameter<bool>(node, name);
    };
  auto declare_string = [&](const char * name) {
      return lmpc::utils::declare_parameter<std::string>(node, name);
    };
  auto delcare_string_vec = [&](const char * name) {
      return lmpc::utils::declare_parameter<std::vector<std::string>>(node, name);
    };

  const auto keys = delcare_string_vec("base_neural_dynamics.model_names");
  const auto values = delcare_string_vec("base_neural_dynamics.model_paths");
  std::unordered_map<std::string, std::string> model_paths(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    model_paths[keys[i]] = values[i];
  }

  return std::make_shared<BaseNeuralDynamicsModelConfig>(

    BaseNeuralDynamicsModelConfig{
          declare_double("base_neural_dynamics.fd_max"),
          declare_double("base_neural_dynamics.fb_max"),
          declare_double("base_neural_dynamics.td"),
          declare_double("base_neural_dynamics.tb"),
          declare_double("base_neural_dynamics.v_max"),
          declare_double("base_neural_dynamics.p_max"),
          declare_double("base_neural_dynamics.mu"),
          declare_double("base_neural_dynamics.lon_force_scale"),
          declare_bool("base_neural_dynamics.use_cuda"),
          model_paths
        }
  );
}
}  // namespace neural_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
