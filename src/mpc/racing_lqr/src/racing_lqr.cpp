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

#include "racing_lqr/racing_lqr.hpp"
#include "lmpc_utils/utils.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_lqr
{
RacingLQR::RacingLQR(
  RacingLQRConfig::SharedPtr mpc_config,
  BaseVehicleModel::SharedPtr model)
: config_(mpc_config), model_(model),
  c2d_(utils::c2d_function(model_->nx(), model_->nu(), config_->dt)),
  rk4_(utils::rk4_function(model_->nx(), model_->nu(), model_->dynamics())),
  align_abscissa_(utils::align_abscissa_function(config_->N)),
  lqr_solved_(false)
{
}

const RacingLQRConfig & RacingLQR::get_config() const
{
  return *config_.get();
}

void RacingLQR::solve(const casadi::DMDict & in, casadi::DMDict & out)
{
  using casadi::DM;
  using casadi::MX;
  using casadi::Slice;

  const auto & total_length = in.at("total_length");
  const auto & x_ic = in.at("x_ic");
  auto X_ref = in.at("X_ref");
  X_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", X_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  const auto & U_ref = in.at("U_ref");
  const auto & curvatures = in.at("curvatures");
  const auto & bank_angle = in.at("bank_angle");
  //TODO(David): add curavture and bank angle input

  auto P = casadi::DMVector(config_->N, config_->Qf);
  auto K = casadi::DMVector(config_->N - 1, DM::zeros(model_->nu(), model_->nx()));
  auto As = casadi::DMVector(config_->N - 1, DM::zeros(model_->nx(), model_->nx()));
  auto Bs = casadi::DMVector(config_->N - 1, DM::zeros(model_->nx(), model_->nu()));
  for (int k = config_->N - 2; k >= 0; k--) {
    // obtain linearlized continuous dynamics
    const auto dyn_jac = model_->dynamics_jacobian()(
      casadi::DMDict{{"x", X_ref(Slice(), k)}, {"u", U_ref(Slice(), k)}, {"k", curvatures(Slice(), k)}, {"bank", bank_angle(Slice(), k)}});  // TODO(David): pass curvature and bank angle here
    const auto & Ac = dyn_jac.at("A");
    const auto & Bc = dyn_jac.at("B");

    // convert continuous dynamics to discrete
    const auto dyn_d = c2d_(casadi::DMDict{{"Ac", Ac}, {"Bc", Bc}});
    As[k] = dyn_d.at("A");
    Bs[k] = dyn_d.at("B");
    std::cout << "x_ic in lqr: " << x_ic << std::endl;
    std::cout << "X_ref: " << X_ref << std::endl;
    // std::cout << "R: " << config_->R << std::endl;
    // std::cout << "P[k+1]: " << P[k+1] << std::endl;
    // std::cout << "As[k]: " << As[k] << std::endl;
    // std::cout << "Bs[k]: " << Bs[k] << std::endl;
    // std::cout << "NOTE: I changed R to a 2x2 matrix from a 3x3 matrix.\n";
    // std::cout << "NOTE: simplify_lon_control is true in single_track_planar_model and nu is 2.\n";
    // std::cout << "NOTE: Please remember to change the size of R in ros_param_loader.\n";
    // std::cout << "k: " << k << std::endl;

    // Ricatti
    K[k] =
      DM::solve(
      config_->R + DM::mtimes({Bs[k].T(), P[k + 1], Bs[k]}),
      DM::mtimes({Bs[k].T(), P[k + 1], As[k]}));
    P[k] = config_->Q + DM::mtimes({As[k].T(), P[k + 1], As[k] - DM::mtimes(Bs[k], K[k])});
  }

  // simulate
  auto X_optm = DM::zeros(model_->nx(), config_->N);
  X_optm(Slice(), 0) = x_ic;
  auto U_optm = DM::zeros(model_->nu(), config_->N - 1);
  for (size_t k = 0; k < config_->N - 1; k++) {
    U_optm(Slice(), k) =
      U_ref(Slice(), k) - DM::mtimes(K[k], X_optm(Slice(), k) - X_ref(Slice(), k));
    // TODO(haoru): add frenet support
    X_optm(Slice(), k + 1) = rk4_(
      casadi::DMDict{{"x", X_optm(Slice(), k)}, {"u", U_optm(
            Slice(), k)}, {"dt", config_->dt}, {"k", 0.0}}).at("xip1");
  }

  // calculate control
  out["u"] = U_optm(Slice(), 0);
  out["U_optm"] = U_optm;
  out["X_optm"] = X_optm;
  lqr_solved_ = true;
}

BaseVehicleModel & RacingLQR::get_model()
{
  return *model_;
}

bool RacingLQR::get_solved()
{
  return lqr_solved_;
}

}  // namespace racing_lqr
}  // namespace mpc
}  // namespace lmpc
