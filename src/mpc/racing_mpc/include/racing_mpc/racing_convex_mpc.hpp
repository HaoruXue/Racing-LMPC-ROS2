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

#ifndef RACING_MPC__RACING_CONVEX_MPC_HPP_
#define RACING_MPC__RACING_CONVEX_MPC_HPP_

#include <memory>

#include <casadi/casadi.hpp>

#include <base_mpc/base_mpc.hpp>
#include <vehicle_model_factory/vehicle_model_factory.hpp>

#include "racing_mpc/racing_mpc_config.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
using lmpc::vehicle_model::base_vehicle_model::BaseVehicleModelConfig;
using lmpc::vehicle_model::base_vehicle_model::BaseVehicleModel;
using lmpc::vehicle_model::base_vehicle_model::XIndex;
using lmpc::vehicle_model::base_vehicle_model::UIndex;

struct QPProblem
{
  casadi::SX x;
  casadi::SX f;
  casadi::SX g;
  casadi::SX lbg;
  casadi::SX ubg;
  casadi::SX lbx;
  casadi::SX ubx;
  casadi::SX p;
};

class RacingConvexMPC : public BaseMPC
{
public:
  typedef std::shared_ptr<RacingConvexMPC> SharedPtr;
  typedef std::unique_ptr<RacingConvexMPC> UniquePtr;

  explicit RacingConvexMPC(
    RacingMPCConfig::SharedPtr mpc_config,
    BaseVehicleModel::SharedPtr model,
    const bool & full_dynamics = false);
  const RacingMPCConfig & get_config() const;

  bool init() override;
  void solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats) override;

  BaseVehicleModel & get_model();

  const bool & solved() const;

protected:
  RacingMPCConfig::SharedPtr config_ {};
  BaseVehicleModel::SharedPtr model_ {};

  casadi::DM scale_x_;
  casadi::DM scale_u_;
  casadi::Function align_abscissa_;

  // optimization variables
  casadi::SX X_;  // all the states, scaled
  casadi::SX U_;  // all the inputs, scaled
  casadi::SX dU_;  // all the input deltas, scaled
  casadi::SX boundary_slack_;
  // casadi::SX convex_hull_slack_;
  // casadi::SX convex_combi_;

  // optimization parameters
  casadi::SX X_ref_;  // reference states, unscaled
  casadi::SX U_ref_;  // reference inputs, unscaled
  casadi::SX dU_ref_;  // reference input deltas, unscaled
  casadi::SX T_ref_;  // reference time, unscaled
  casadi::SX x_ic_;  // initial state, unscaled
  casadi::SX u_ic_;  // initial input, unscaled
  casadi::SX bound_left_;
  casadi::SX bound_right_;
  casadi::SX total_length_;
  casadi::SX bank_angle_;
  casadi::SX curvatures_;
  casadi::SX vel_ref_;
  // casadi::SX ss_;
  // casadi::SX ss_costs_;  // J in LMPC paper

  // problem
  QPProblem prob_;
  casadi::Function solver_;

  // flag if the nlp has been solved at least once
  bool solved_;

  bool enable_boundary_slack_;
};
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
#endif  // RACING_MPC__RACING_CONVEX_MPC_HPP_
