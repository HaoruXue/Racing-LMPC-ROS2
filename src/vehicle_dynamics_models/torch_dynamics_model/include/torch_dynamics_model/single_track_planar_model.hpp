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

#ifndef TORCH_DYNAMICS_MODEL__SINGLE_TRACK_PLANAR_MODEL_HPP_
#define TORCH_DYNAMICS_MODEL__SINGLE_TRACK_PLANAR_MODEL_HPP_

#include <memory>

#include <single_track_planar_model/single_track_planar_model.hpp>
#include <torch_dynamics_model/base_torch_dynamics_model.hpp>

namespace lmpc
{
namespace vehicle_model
{
namespace torch_dynamics_model
{
class SingleTrackPlanarModel : public BaseTorchDynamicsModel
{
public:
    SingleTrackPlanarModel() = default;
    SingleTrackPlanarModel(const lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig::SharedPtr & base_config, const lmpc::vehicle_model::single_track_planar_model::SingleTrackPlanarModelConfig::SharedPtr & dt_config);
    virtual ~SingleTrackPlanarModel() = default;
    virtual torch::Tensor forward(const torch::Tensor& state, const torch::Tensor& control, const torch::Tensor& param);
    const lmpc::vehicle_model::single_track_planar_model::SingleTrackPlanarModelConfig & get_config() const;
protected:
    lmpc::vehicle_model::single_track_planar_model::SingleTrackPlanarModelConfig::SharedPtr dt_config_;

    torch::Tensor b_kd_f_;
    torch::Tensor b_kb_f_;
    torch::Tensor b_m_;
    torch::Tensor b_Jzz_;
    torch::Tensor b_l_;
    torch::Tensor b_lr_;
    torch::Tensor b_lf_;
    torch::Tensor b_fr_;
    torch::Tensor b_hcog_;
    torch::Tensor b_cl_f_;
    torch::Tensor b_cl_r_;
    torch::Tensor b_rho_;
    torch::Tensor b_A_;
    torch::Tensor b_cd_;
    torch::Tensor b_mu_;
    torch::Tensor b_tyre_f_;
    torch::Tensor b_Bf_;
    torch::Tensor b_Cf_;
    torch::Tensor b_tyre_r_;
    torch::Tensor b_Br_;
    torch::Tensor b_Cr_;
};
}  // namespace torch_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
#endif  // TORCH_DYNAMICS_MODEL__SINGLE_TRACK_PLANAR_MODEL_HPP_
