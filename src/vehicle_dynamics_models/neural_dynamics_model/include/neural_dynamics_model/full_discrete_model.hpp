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

#ifndef NEURAL_DYNAMICS_MODEL__NEURAL_DYNAMICS_MODEL_HPP_
#define NEURAL_DYNAMICS_MODEL__NEURAL_DYNAMICS_MODEL_HPP_

#include <memory>

#include <torch/torch.h>
#include <casadi/casadi.hpp>

#include <torch_casadi_interface/torch_casadi_function.hpp>

#include "neural_dynamics_model/base_neural_dynamics_model.hpp"

namespace lmpc
{
namespace vehicle_model
{
namespace neural_dynamics_model
{

class NeuralDynamicsModel final : public BaseNeuralDynamicsModel
{
public:
  typedef std::shared_ptr<NeuralDynamicsModel> SharedPtr;
  typedef std::unique_ptr<NeuralDynamicsModel> UniquePtr;

  NeuralDynamicsModel(
    base_vehicle_model::BaseVehicleModelConfig::SharedPtr base_config,
    BaseNeuralDynamicsModelConfig::SharedPtr base_nn_config);

  void add_nlp_constraints(casadi::Opti & opti, const casadi::MXDict & in) override;
  void calc_lon_control(
    const casadi::DMDict & in, double & throttle,
    double & brake_kpa) const override;
  void calc_lat_control(const casadi::DMDict & in, double & steering_rad) const override;

private:
  void compile_dynamics();

  std::shared_ptr<torch::jit::script::Module> model_;
  std::shared_ptr<torch_casadi_interface::TorchCasadiEvalFunction> torch_func_;
};
}  // namespace neural_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
#endif  // NEURAL_DYNAMICS_MODEL__NEURAL_DYNAMICS_MODEL_HPP_
