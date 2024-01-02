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

#ifndef RACING_MPC__RACING_FOLLOW_MPC_HPP_
#define RACING_MPC__RACING_FOLLOW_MPC_HPP_

#include <memory>

#include "racing_mpc/racing_mpc.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
class RacingFollowMPC : public RacingMPC
{
public:
  typedef std::shared_ptr<RacingFollowMPC> SharedPtr;
  typedef std::unique_ptr<RacingFollowMPC> UniquePtr;

  explicit RacingFollowMPC(
    RacingMPCConfig::SharedPtr mpc_config,
    BaseVehicleModel::SharedPtr model,
    const bool & full_dynamics = false);

  bool init() override;

protected:
  casadi::MX opponent_X_ref_;
  casadi::MX follow_distance_;
  casadi::MX q_follow_;
  casadi::MX q_vel_;

  void init_solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats) override;

  void build_following_cost(casadi::MX & cost);
};
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
#endif  // RACING_MPC__RACING_FOLLOW_MPC_HPP_
