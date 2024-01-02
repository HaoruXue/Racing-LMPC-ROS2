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

#include <math.h>
#include <exception>
#include <vector>
#include <iostream>
#include <chrono>

#include "racing_mpc/racing_follow_mpc.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
RacingFollowMPC::RacingFollowMPC(
  RacingMPCConfig::SharedPtr mpc_config,
  BaseVehicleModel::SharedPtr model,
  const bool & full_dynamics)
: RacingMPC(mpc_config, model, full_dynamics),
  opponent_X_ref_(opti_.parameter(model_->nx(), config_->N)),
  follow_distance_(opti_.parameter(1, config_->N)),
  q_follow_(opti_.parameter(1, 1)),
  q_vel_(opti_.parameter(1, 1))
{
  for (casadi_int i = 0; i < static_cast<casadi_int>(config_->N); i++) {
  }
}

bool RacingFollowMPC::init()
{
  build_boundary_constraint(cost_);
  build_following_cost(cost_);
  build_dynamics_constraint();
  build_initial_constraint();
  opti_.minimize(cost_);
  return true;
}

void RacingFollowMPC::init_solve(
  const casadi::DMDict & in, casadi::DMDict & out,
  casadi::Dict & stats)
{
  (void) out;
  (void) stats;
  using casadi::Slice;
  using casadi::DM;

  if (!in.count("opponent_X_ref")) {
    opti_.set_value(opponent_X_ref_, DM::zeros(model_->nx(), config_->N));
    opti_.set_value(follow_distance_, 0.0);
    opti_.set_value(q_follow_, 0.0);
    opti_.set_value(q_vel_, config_->q_vel);
    return;
  }

  const auto & total_length = in.at("total_length");
  const auto x_ic = in.at("x_ic");
  auto opponent_X_ref = in.at("opponent_X_ref");
  opponent_X_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", opponent_X_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  auto x_offset = DM::zeros(model_->nx());
  x_offset(XIndex::PX) = x_ic(XIndex::PX);
  opponent_X_ref = opponent_X_ref - x_offset;
  opti_.set_value(opponent_X_ref_, opponent_X_ref);

  auto X_ref = in.at("X_ref");
  X_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", X_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  X_ref = X_ref - x_offset;

  if (in.count("follow_distance")) {
    const auto target_gap = static_cast<double>(in.at("follow_distance"));
    const auto current_gap = (opponent_X_ref(XIndex::PX, Slice()) - X_ref(XIndex::PX, Slice())).get_elements();
    auto follow_distance = DM::zeros(1, config_->N);
    for (casadi_int i = 0; i < static_cast<casadi_int>(config_->N); i++) {
      // TODO(haoru): make max diff a parameter
      follow_distance(i) = std::clamp(target_gap, current_gap[i] - 1.0, current_gap[i] + 1.0);
    }

    opti_.set_value(follow_distance_, follow_distance);
    opti_.set_value(q_follow_, config_->q_follow);
    opti_.set_value(q_vel_, 0.0);
  } else {
    opti_.set_value(follow_distance_, 0.0);
    opti_.set_value(q_follow_, 0.0);
    opti_.set_value(q_vel_, config_->q_vel);
  }
}

void RacingFollowMPC::build_following_cost(casadi::MX & cost)
{
  using casadi::MX;
  using casadi::Slice;

  // --- MPC stage cost ---
  const auto x0 = X_(Slice(), 0) * scale_x_;
  for (size_t i = 0; i < config_->N - 1; i++) {
    const auto xi = X_(Slice(), i) * scale_x_;
    const auto ui = U_(Slice(), i - 1) * scale_u_;
    const auto dui = dU_(Slice(), i - 1) * scale_u_;
    // xi start with 1 since x0 must equal to x_ic and there is nothing we can do about it
    // const auto d_px =
    // utils::align_abscissa<MX>(xi(XIndex::PX), x0(XIndex::PX), total_length_) - x0(XIndex::PX);
    const auto x_base = model_->to_base_state()(casadi::MXDict{{"x", xi}, {"u", ui}}).at("x_out");
    const auto dv = x_base(XIndex::VX) - vel_ref_(i);
    // const auto dv = x_base(XIndex::VX) - 10.0;
    cost += x_base(XIndex::PY) * x_base(XIndex::PY) * config_->q_contour;
    cost += x_base(XIndex::YAW) * x_base(XIndex::YAW) * config_->q_heading;
    cost += dv * dv * q_vel_;
    cost += x_base(XIndex::VY) * x_base(XIndex::VY) * config_->q_vy;
    cost += x_base(XIndex::VYAW) * x_base(XIndex::VYAW) * config_->q_vyaw;

    cost += MX::mtimes({ui.T(), config_->R, ui});
    cost += MX::mtimes({dui.T(), config_->R_d, dui});

    const auto px_i = X_(XIndex::PX, i) * scale_x_(XIndex::PX);
    const auto px_follow_i = opponent_X_ref_(XIndex::PX, i) - follow_distance_(i);
    const auto d_px_i = px_i - px_follow_i;
    cost += d_px_i * d_px_i * q_follow_;
  }

  // terminal cost
  const auto xN = X_(Slice(), config_->N - 1) * scale_x_;
  const auto uN = U_(Slice(), config_->N - 2) * scale_u_;
  const auto x_base_N = model_->to_base_state()(casadi::MXDict{{"x", xN}, {"u", uN}}).at("x_out");
  const auto dv = x_base_N(XIndex::VX) - vel_ref_(config_->N - 1);
  const auto px_N = X_(XIndex::PX, config_->N - 1) * scale_x_(XIndex::PX);
  const auto px_follow_N = opponent_X_ref_(XIndex::PX, config_->N - 1) - follow_distance_(config_->N - 1);
  const auto d_px_N = px_N - px_follow_N;
  cost += x_base_N(XIndex::PY) * x_base_N(XIndex::PY) * config_->q_contour * 10.0;
  cost += x_base_N(XIndex::YAW) * x_base_N(XIndex::YAW) * config_->q_heading * 10.0;
  cost += dv * dv * q_vel_ * 10.0;
  cost += x_base_N(XIndex::VY) * x_base_N(XIndex::VY) * config_->q_vy * 10.0;
  cost += x_base_N(XIndex::VYAW) * x_base_N(XIndex::VYAW) * config_->q_vyaw * 10.0;
  cost += d_px_N * d_px_N * q_follow_ * 10.0;
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
