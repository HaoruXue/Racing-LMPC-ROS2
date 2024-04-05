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

#ifndef TORCH_DYNAMICS_MODEL__BASE_TORCH_DYNAMICS_MODEL_HPP_
#define TORCH_DYNAMICS_MODEL__BASE_TORCH_DYNAMICS_MODEL_HPP_

#include <torch/torch.h>

#include <memory>
#include <base_vehicle_model/base_vehicle_model_config.hpp>
#include <base_vehicle_model/base_vehicle_model_state.hpp>

namespace lmpc
{
namespace vehicle_model
{
namespace torch_dynamics_model
{
enum XIndex : int
{
  PX = 0,  // global or frenet x position
  PY = 1,  // global or frenet y position
  YAW = 2,  // global or frenet yaw
  VX = 3,  // body longitudinal velocity
  VY = 4,  // body lateral velocity
  VYAW = 5  // body yaw rate
};

enum UIndex : int
{
  F_LON = 0,
  STEER = 1
};

enum ParamIndex : int
{
  CURV = 0,
  BANK = 1,
  DT = 2
};

class BaseTorchDynamicsModel : public torch::nn::Module
{
public:
    BaseTorchDynamicsModel() = default;
    BaseTorchDynamicsModel(const lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig::SharedPtr & config);
    virtual ~BaseTorchDynamicsModel() = default;
    virtual torch::Tensor forward(const torch::Tensor& state, const torch::Tensor& control, const torch::Tensor& param);
    virtual torch::Tensor forward_discrete(const torch::Tensor& state, const torch::Tensor& control, const torch::Tensor& param);
    const lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig & get_base_config() const;
protected:
    lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig::SharedPtr base_config_;
};
}  // namespace torch_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc

#endif // TORCH_DYNAMICS_MODEL__BASE_TORCH_DYNAMICS_MODEL_HPP_