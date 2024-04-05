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
#include <torch_dynamics_model/base_torch_dynamics_model.hpp>

namespace lmpc
{
namespace vehicle_model
{
namespace torch_dynamics_model
{
BaseTorchDynamicsModel::BaseTorchDynamicsModel(const lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig::SharedPtr & config)
: base_config_(config)
{
}
torch::Tensor BaseTorchDynamicsModel::forward(const torch::Tensor& state, const torch::Tensor& control, const torch::Tensor& param)
{
  throw std::runtime_error("Not implemented");
}
torch::Tensor BaseTorchDynamicsModel::forward_discrete(const torch::Tensor& state, const torch::Tensor& control, const torch::Tensor& param)
{
  if(get_base_config().modeling_config->integrator_type == lmpc::vehicle_model::base_vehicle_model::IntegratorType::EULER)
  {
    const auto dt = param.index({torch::indexing::Slice(), ParamIndex::DT}).unsqueeze(1).expand({-1, 6});
    const auto state_next = forward(state, control, param);
    return state + dt * state_next;
  }
  else if (get_base_config().modeling_config->integrator_type == lmpc::vehicle_model::base_vehicle_model::IntegratorType::RK4)
  {
    // make batchsize * 6 dt
    const auto dt = param.index({torch::indexing::Slice(), ParamIndex::DT}).unsqueeze(1).expand({-1, 6});
    const auto k1 = forward(state, control, param);
    const auto k2 = forward(state + 0.5 * dt * k1, control, param);
    const auto k3 = forward(state + 0.5 * dt * k2, control, param);
    const auto k4 = forward(state + dt * k3, control, param);
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
  }
  else
  {
    throw std::runtime_error("Integrator type not supported");
  }
}
const lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig & BaseTorchDynamicsModel::get_base_config() const
{
  return *base_config_;
}
}  // namespace torch_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
