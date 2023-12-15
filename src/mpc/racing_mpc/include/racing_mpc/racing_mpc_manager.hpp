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

#ifndef RACING_MPC__RACING_MPC_MANAGER_HPP_
#define RACING_MPC__RACING_MPC_MANAGER_HPP_

#include <memory>
#include <vector>
#include <future>

#include <casadi/casadi.hpp>

#include "racing_mpc/racing_mpc.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
struct MPCSolution
{
  casadi::DMDict solution;
  casadi::Dict stats;
  size_t solution_age = 0;
};

/**
 * @brief Manages dispatching available MPCs and collecting their solutions.
 * 
 */
class RacingMPCManager
{
public:
  typedef std::shared_ptr<RacingMPCManager> SharedPtr;
  typedef std::unique_ptr<RacingMPCManager> UniquePtr;

  explicit RacingMPCManager(const RacingMPCConfig & config, const BaseVehicleModel & model, const bool & full_dynamics);

  MPCSolution initialize(const casadi::DMDict & in);
  MPCSolution step(const casadi::DMDict & in);
  size_t get_mpc_authority() const;

protected:
    std::vector<RacingMPC::SharedPtr> mpc_list_;
    std::vector<std::future<MPCSolution>> mpc_future_list_;
    std::vector<MPCSolution> mpc_solution_list_;
    size_t mpc_authority_;

    MPCSolution mpc_callback(const casadi::DMDict & in, const size_t & mpc_index);
    size_t next_mpc(const size_t & mpc_index);
};
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
#endif  // RACING_MPC__RACING_MPC_MANAGER_HPP_
