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

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <future>
#include "racing_mpc/multi_mpc_manager.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
MultiMPCManager::MultiMPCManager(const MultiMPCManagerConfig & config)
: num_cycle_to_switch_(config.num_cycle_to_switch),
  mpcs_(config.mpcs),
  mpc_cycle_count_(mpcs_.size(), 0),
  mpc_futures_(mpcs_.size()),
  current_mpc_idx_(0),
  buffer_()
{
}

void MultiMPCManager::solve(
  const casadi::DMDict & in, const size_t & timestamp,
  SolutionCallback callback)
{
  if (!is_mpc_ready(current_mpc_idx_)) {
    mpc_cycle_count_[current_mpc_idx_] += 1;
    // if the number of cycles is not enough, we don't start another MPC
    if (mpc_cycle_count_[current_mpc_idx_] < num_cycle_to_switch_) {
      return;
    }
    // if the current MPC is blocked for too long, we switch to the next one
    for (size_t i = 0; i < mpc_cycle_count_.size(); ++i) {
      if (i == current_mpc_idx_) {
        continue;
      }
      if (is_mpc_ready(i)) {
        current_mpc_idx_ = i;
        mpc_cycle_count_[current_mpc_idx_] = 0;
        goto new_solve;
      }
      return;       // this is only reached when no MPC is ready
    }
  }
new_solve: mpc_futures_[current_mpc_idx_] = std::async(
    std::launch::async,
    &MultiMPCManager::solve_mpc_thread,
    this,
    current_mpc_idx_.load(),
    in,
    timestamp,
    callback);
}

lmpc::utils::MPCSolution MultiMPCManager::get_solution(const size_t & timestamp)
{
  return buffer_.get_mpc_solution(timestamp);
}

bool MultiMPCManager::is_solution_initialized() const
{
  return buffer_.is_initialized();
}

MultiMPCSolution MultiMPCManager::solve_mpc_thread(
  const size_t mpc_idx, const casadi::DMDict in,
  const size_t timestamp,
  SolutionCallback callback)
{
  MultiMPCSolution solution;
  mpcs_[mpc_idx]->solve(in, solution.solution, solution.stats);
  solution.timestamp = timestamp;
  solution.mpc_name = "MPC " + std::to_string(mpc_idx);
  // only update the buffer if the solution is from the current MPC
  if (current_mpc_idx_ == mpc_idx) {
    if (mpcs_[mpc_idx]->is_solve_success(solution.solution, solution.stats)) {
      const auto x = mpcs_[mpc_idx]->get_x(solution.solution);
      const auto u = mpcs_[mpc_idx]->get_u(solution.solution);
      buffer_.set_mpc_solution(x, u, timestamp);
    }
  }
  callback(solution);
  return solution;
}

bool MultiMPCManager::is_mpc_ready(const size_t & mpc_idx)
{
  // an MPC is ready if
  // 1. it has never been solved (so the future is invalid)
  // 2. it has been solved and the future is ready
  return !mpc_futures_[mpc_idx].valid() ||
         mpc_futures_[mpc_idx].wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
