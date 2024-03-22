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

#include "neural_dynamics_model/base_neural_dynamics_model.hpp"

namespace lmpc
{
namespace vehicle_model
{
namespace neural_dynamics_model
{
BaseNeuralDynamicsModel::BaseNeuralDynamicsModel(
  base_vehicle_model::BaseVehicleModelConfig::SharedPtr base_config,
  BaseNeuralDynamicsModelConfig::SharedPtr base_nn_config)
: base_vehicle_model::BaseVehicleModel(base_config), base_nn_config_(base_nn_config)
{
  using casadi::SX;
  const auto x_sym = SX::sym("x", nx());
    const auto u_derived_sym = SX::sym("u", nu());
    const auto u_base_sym = SX::sym("u", BaseVehicleModel::nu());

    const auto u_base_out = SX::vertcat(
          {
            SX::if_else(
              u_derived_sym(UIndex::LON) > 0.0, u_derived_sym(UIndex::LON),
              0.0),
            SX::if_else(
              u_derived_sym(UIndex::LON) < 0.0, u_derived_sym(UIndex::LON),
              0.0),
            u_derived_sym(UIndex::STEER)
          });
    const auto u_derived_out = SX::vertcat(
          {
            SX::if_else(
              abs(u_base_sym(base_vehicle_model::UIndex::FD)) > abs(u_base_sym(base_vehicle_model::UIndex::FB)),
              u_base_sym(base_vehicle_model::UIndex::FD), u_base_sym(base_vehicle_model::UIndex::FB)),
            u_base_sym(base_vehicle_model::UIndex::STEER)
          });

    to_base_control_ = casadi::Function(
      "to_base_control", {x_sym, u_derived_sym}, {u_base_out}, {"x", "u"}, {"u_out"});
    from_base_control_ = casadi::Function(
      "from_base_control", {x_sym, u_base_sym}, {u_derived_out}, {"x", "u"}, {"u_out"});
    to_base_state_ = casadi::Function(
      "to_base_state", {x_sym, u_derived_sym}, {x_sym}, {"x", "u"}, {"x_out"});
    from_base_state_ = casadi::Function(
      "from_base_state", {x_sym, u_base_sym}, {x_sym}, {"x", "u"}, {"x_out"});
}

const BaseNeuralDynamicsModelConfig & BaseNeuralDynamicsModel::get_base_nn_config() const
{
  return *base_nn_config_.get();
}

size_t BaseNeuralDynamicsModel::nx() const
{
  return 6;
}

size_t BaseNeuralDynamicsModel::nu() const
{
  return 2;
}

}  // namespace neural_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
