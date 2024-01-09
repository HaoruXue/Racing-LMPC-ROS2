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

#include "racing_mpc/racing_mpc_function.hpp"
#include "lmpc_utils/utils.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
RacingMPCFunction::RacingMPCFunction(
  RacingMPCConfig::SharedPtr mpc_config,
  BaseVehicleModel::SharedPtr model,
  const bool & full_dynamics)
: config_(mpc_config), model_(model),
  scale_x_(config_->x_max),
  scale_u_(config_->u_max),
  align_abscissa_(utils::align_abscissa_function(config_->N)),
  solved_(false)
{
  using casadi::MX;
  using casadi::SX;
  using casadi::Slice;

  // furture adjust x scale
  scale_x_(XIndex::PX) = config_->x_max(XIndex::VX);
  scale_x_(XIndex::PY) = config_->average_track_width;
  scale_x_(XIndex::YAW) = M_PI;
  // throw if scale contains inf
  if (static_cast<double>(casadi::DM::sum1(scale_x_)) == std::numeric_limits<double>::infinity()) {
    throw std::runtime_error("scale_x contains inf.");
  }
  if (static_cast<double>(casadi::DM::sum1(scale_u_)) == std::numeric_limits<double>::infinity()) {
    throw std::runtime_error("scale_u contains inf.");
  }
  solver_ = casadi::external("racing_mpc_function", "libracing_mpc_function.so");
}

const RacingMPCConfig & RacingMPCFunction::get_config() const
{
  return *config_;
}

bool RacingMPCFunction::init()
{
    return true;
}

void RacingMPCFunction::solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats)
{
  using casadi::DM;
  using casadi::MX;
  using casadi::Slice;

  const auto & total_length = in.at("total_length");
  const auto & x_ic = in.at("x_ic");
  const auto & u_ic = in.at("u_ic");
  auto X_ref = in.at("X_ref");
  X_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", X_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  const auto U_ref = in.at("U_ref");
  const auto & bound_left = in.at("bound_left");
  const auto & bound_right = in.at("bound_right");
  const auto & curvatures = in.at("curvatures");
  const auto & vel_ref = in.at("vel_ref");
  const auto & bank_angle = in.at("bank_angle");
  const auto & T_ref = in.at("T_ref");

  // build x offset
  auto x_offset = casadi::DM::zeros(model_->nx());
  x_offset(XIndex::PX) = x_ic(XIndex::PX);
  if (!in.count("X_optm_ref")) {
    throw std::runtime_error("No optimal reference given.");
  }
  auto X_optm_ref = in.at("X_optm_ref");
  X_optm_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", X_optm_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  const auto U_optm_ref = in.at("U_optm_ref");
  const auto & dU_optm_ref = in.at("dU_optm_ref");

  // solve problem
  try {
    const auto inputs = casadi::DMDict(
        {
            {"X", (X_optm_ref - x_offset) / scale_x_}, {"U", U_optm_ref / scale_u_}, {"dU", dU_optm_ref / scale_u_}, {"boundary_slack", 0.0},
            {"scale_x_param", scale_x_}, {"scale_u_param", scale_u_},
            {"X_ref", X_ref - x_offset}, {"U_ref", U_ref}, {"T_ref", T_ref},
            {"x_ic", x_ic - x_offset}, {"u_ic", u_ic},
            {"bound_left", bound_left}, {"bound_right", bound_right},
            {"total_length", total_length}, {"bank_angle", bank_angle},
            {"curvatures", curvatures}, {"vel_ref", vel_ref}
        }
    );
    const auto outputs = solver_(inputs);
    out["X_optm"] = outputs.at("X_optm") * scale_x_ + x_offset;
    out["U_optm"] = outputs.at("U_optm") * scale_u_;
    out["dU_optm"] = outputs.at("dU_optm") * scale_u_;
    solved_ = true;
  } catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
  }
}

BaseVehicleModel & RacingMPCFunction::get_model()
{
  return *model_;
}

const bool & RacingMPCFunction::solved() const
{
  return solved_;
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
