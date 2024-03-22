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

#ifndef NEURAL_DYNAMICS_MODEL__BASE_NEURAL_DYNAMICS_MODEL_HPP_
#define NEURAL_DYNAMICS_MODEL__BASE_NEURAL_DYNAMICS_MODEL_HPP_

#include <memory>
#include <unordered_map>
#include <string>

#include <casadi/casadi.hpp>

#include "base_vehicle_model/base_vehicle_model.hpp"

namespace lmpc
{
namespace vehicle_model
{
namespace neural_dynamics_model
{
struct BaseNeuralDynamicsModelConfig
{
  typedef std::shared_ptr<BaseNeuralDynamicsModelConfig> SharedPtr;

  double Fd_max;
  double Fb_max;
  double Td;
  double Tb;
  double v_max;
  double P_max;
  double mu;
  double lon_force_scale;
  bool use_cuda;
  std::unordered_map<std::string, std::string> model_paths;
};

enum XIndex : casadi_int
{
  PX = 0,  // global or frenet x position
  PY = 1,  // global or frenet y position
  YAW = 2,  // global or frenet yaw
  VX = 3,  // body longitudinal velocity
  VY = 4,  // body lateral velocity
  VYAW = 5  // body yaw rate
};

enum UIndex : casadi_int
{
  LON = 0,
  STEER = 1
};

class BaseNeuralDynamicsModel : public base_vehicle_model::BaseVehicleModel
{
public:
  typedef std::shared_ptr<BaseNeuralDynamicsModel> SharedPtr;
  typedef std::unique_ptr<BaseNeuralDynamicsModel> UniquePtr;

  BaseNeuralDynamicsModel(
    base_vehicle_model::BaseVehicleModelConfig::SharedPtr base_config,
    BaseNeuralDynamicsModelConfig::SharedPtr base_nn_config);

  size_t nx() const override;
  size_t nu() const override;

  const BaseNeuralDynamicsModelConfig & get_base_nn_config() const;

protected:
  BaseNeuralDynamicsModelConfig::SharedPtr base_nn_config_ {};
};
}  // namespace neural_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
#endif  // NEURAL_DYNAMICS_MODEL__BASE_NEURAL_DYNAMICS_MODEL_HPP_
