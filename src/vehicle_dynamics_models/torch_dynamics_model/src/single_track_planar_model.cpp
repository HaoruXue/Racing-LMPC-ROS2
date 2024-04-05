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

#include "torch_dynamics_model/single_track_planar_model.hpp"
#include "lmpc_utils/utils.hpp"
#define GRAVITY 9.8

namespace lmpc
{
namespace vehicle_model
{
namespace torch_dynamics_model
{
SingleTrackPlanarModel::SingleTrackPlanarModel(
  const lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig::SharedPtr & base_config, const lmpc::vehicle_model::single_track_planar_model::SingleTrackPlanarModelConfig::SharedPtr & dt_config)
: BaseTorchDynamicsModel(base_config), dt_config_(dt_config)
{
  const auto batch_size = 1000000;
  b_kd_f_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().powertrain_config->kd;
  b_kb_f_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().front_brake_config->bias;
  b_m_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().chassis_config->total_mass;
  b_Jzz_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().chassis_config->moi;
  b_l_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().chassis_config->wheel_base;
  b_lr_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().chassis_config->cg_ratio * get_base_config().chassis_config->wheel_base;
  b_lf_ = b_l_ - b_lr_;
  b_fr_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().chassis_config->fr;
  b_hcog_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().chassis_config->cg_height;
  b_cl_f_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().aero_config->cl_f;
  b_cl_r_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().aero_config->cl_r;
  b_rho_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().aero_config->air_density;
  b_A_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().aero_config->frontal_area;
  b_cd_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().aero_config->drag_coeff;
  b_mu_ = torch::ones({batch_size}).to(torch::kCUDA) * dt_config->mu;
  b_tyre_f_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().front_tyre_config->pacejka_b;
  b_Bf_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().front_tyre_config->pacejka_b;
  b_Cf_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().front_tyre_config->pacejka_c;
  b_tyre_r_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().rear_tyre_config->pacejka_b;
  b_Br_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().rear_tyre_config->pacejka_b;
  b_Cr_ = torch::ones({batch_size}).to(torch::kCUDA) * get_base_config().rear_tyre_config->pacejka_c;
}

torch::Tensor SingleTrackPlanarModel::forward(const torch::Tensor& state, const torch::Tensor& control, const torch::Tensor& param)
{
  // state: [x, y, theta, v_x, v_y, omega] (n * 6)
  // control: [a, delta] (n * 2)
  // output: [x_dot, y_dot, theta_dot, vx_dot, vy_dot, omega_dot] (n * 6)
  // get parameters
  // const auto & kd_f = get_base_config().powertrain_config->kd;
  // const auto & kb_f = get_base_config().front_brake_config->bias;  // front brake force bias
  // const auto & m = get_base_config().chassis_config->total_mass;  // mass of car
  // const auto & Jzz = get_base_config().chassis_config->moi;  // MOI around z axis
  // const auto & l = get_base_config().chassis_config->wheel_base;  // wheelbase
  // const auto lr = get_base_config().chassis_config->cg_ratio * l;  // cg to front axle
  // const auto lf = l - lr;  // cg to rear axle
  // const auto & twf = get_base_config().chassis_config->tw_f;  // front track width
  // const auto & twr = get_base_config().chassis_config->tw_r;  // rear track width
  // const auto & fr = get_base_config().chassis_config->fr;  // rolling resistance coefficient
  // const auto & hcog = get_base_config().chassis_config->cg_height;  // center of gravity height
  // const auto & cl_f = get_base_config().aero_config->cl_f;  // downforce coefficient at front
  // const auto & cl_r = get_base_config().aero_config->cl_r;  // downforce coefficient at rear
  // const auto & rho = get_base_config().aero_config->air_density;  // air density
  // const auto & A = get_base_config().aero_config->frontal_area;  // frontal area
  // const auto & cd = get_base_config().aero_config->drag_coeff;  // drag coefficient
  // const auto & mu = get_config().mu;  // tyre - track friction coefficient
  // const auto & tyre_f = *get_base_config().front_tyre_config;
  // const auto & Bf = tyre_f.pacejka_b;  // magic formula B - front
  // const auto & Cf = tyre_f.pacejka_c;  // magic formula C - front
  // const auto & tyre_r = *get_base_config().rear_tyre_config;
  // const auto & Br = tyre_r.pacejka_b;  // magic formula B - rear
  // const auto & Cr = tyre_r.pacejka_c;  // magic formula C - rear

  const auto fd = torch::nn::functional::relu(control.index({torch::indexing::Slice(), UIndex::F_LON}));  // drive force
  const auto fb = torch::neg(torch::nn::functional::relu(torch::neg(control.index({torch::indexing::Slice(), UIndex::F_LON}))));  // brake force
  const auto delta = control.index({torch::indexing::Slice(), UIndex::STEER});  // steering angle
  const auto curv = param.index({torch::indexing::Slice(), ParamIndex::CURV});
  const auto bank = param.index({torch::indexing::Slice(), ParamIndex::BANK});
  const auto vx = state.index({torch::indexing::Slice(), XIndex::VX});
  const auto vy = state.index({torch::indexing::Slice(), XIndex::VY});
  const auto omega = state.index({torch::indexing::Slice(), XIndex::VYAW});
  const auto py = state.index({torch::indexing::Slice(), XIndex::PY});
  const auto phi = state.index({torch::indexing::Slice(), XIndex::YAW});

  const auto v_sq = vx * vx + vy * vy;
  const auto N = b_m_ * GRAVITY * cos(bank) - b_m_ * v_sq * curv * sin(bank);  // normal force
  const auto Fx_f = 0.5 * b_kd_f_ * fd + 0.5 * b_kb_f_ * fb - 0.5 * b_fr_ * N * b_lr_ / b_l_;
  const auto Fx_r = 0.5 * (1 - b_kd_f_) * fd + 0.5 * (1.0 - b_kb_f_) * fb - 0.5 * b_fr_ * N * b_lf_ / b_l_;
  const auto ax = (fd + fb - 0.5 * b_cd_ * b_rho_ * b_A_ * v_sq - b_fr_ * N) / b_m_;

  const auto Fz_f = 0.5 * N * b_lr_ / (b_lf_ + b_lr_) - 0.5 * b_hcog_ / (b_lf_ + b_lr_) * b_m_ * ax + 0.25 * b_cl_f_ * b_rho_ * b_A_ * v_sq;
  const auto Fz_r = 0.5 * N * b_lf_ / (b_lf_ + b_lr_) + 0.5 * b_hcog_ / (b_lf_ + b_lr_) * b_m_ * ax + 0.25 * b_cl_r_ * b_rho_ * b_A_ * v_sq;
  const auto a_fl = delta - atan((b_lf_ * omega + vy) / (vx));
  const auto a_rl = atan((b_lr_ * omega - vy) / (vx));

  const auto Fy_f = b_mu_ * Fz_f * sin(b_Cf_ * atan(b_Bf_ * a_fl));
  const auto Fy_r = b_mu_ * Fz_r * sin(b_Cr_ * atan(b_Br_ * a_rl));

  const auto omega_dot = 1.0 / b_Jzz_ *
    (-(2 * Fy_r) * b_lr_ + ((2 * Fy_f) * cos(delta) + (2 * Fx_f) * sin(delta)) * b_lf_);
  const auto vx_dot = 1.0 / b_m_ *
    ((2 * Fx_r) + (2 * Fx_f) * cos(delta) - (2 * Fy_f) * sin(delta) -
    0.5 * b_cd_ * b_rho_ * b_A_ * v_sq) + omega * vy;
  const auto vy_dot = 1.0 / b_m_ *
    ((2 * Fy_r) + (2 * Fy_f) * cos(delta) + (2 * Fx_f) * sin(delta)) -
    omega * vx - GRAVITY * sin(bank);

  // global frame acceleration px_dot, py_dot
  auto px_dot = vx * cos(phi) - vy * sin(phi);
  auto py_dot = vx * sin(phi) + vy * cos(phi);
  auto phi_dot = omega;

  if (base_config_->modeling_config->use_frenet) {
    // convert to frenet frame
    const auto k_banked = curv * cos(bank);
    px_dot /= (1 - py * k_banked);
    phi_dot -= k_banked * px_dot;
  } else {
    phi_dot *= cos(bank);
    px_dot = vx * cos(phi) - vy * sin(phi) * cos(bank);
    py_dot = vx * sin(phi) + vy * cos(phi) * cos(bank);
  }
  return torch::stack({px_dot, py_dot, phi_dot, vx_dot, vy_dot, omega_dot}, 1).squeeze();
}

const lmpc::vehicle_model::single_track_planar_model::SingleTrackPlanarModelConfig & SingleTrackPlanarModel::get_config() const
{
  return *dt_config_;
}
}  // namespace torch_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
