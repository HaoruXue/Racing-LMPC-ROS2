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
#include <chrono>
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
  mpc_mutexes_(mpcs_.size()),
  current_mpc_idx_(0)
{
}

MultiMPCManager::~MultiMPCManager()
{
  // wait for all MPCs to finish
  for (size_t i = 0; i < mpcs_.size(); ++i) {
    std::unique_lock<std::shared_mutex> lock(mpc_mutexes_[i]);
    if (mpc_futures_[i].valid()) {
      mpc_futures_[i].wait();
    }
  }
}

MultiMPCSolution MultiMPCManager::initialize(const casadi::DMDict & in, const size_t & timestamp)
{
  auto solve_async = [&](const size_t & mpc_idx) {
      MultiMPCSolution solution;
      mpcs_[mpc_idx]->solve(in, solution.solution, solution.stats);
      solution.in = in;
      solution.timestamp = timestamp;
      solution.mpc_name = "MPC " + std::to_string(mpc_idx);
      return solution;
    };

  // solve all MPCs asynchronously
  for (size_t i = 0; i < mpcs_.size(); ++i) {
    std::unique_lock<std::shared_mutex> lock(mpc_mutexes_[i]);
    mpc_futures_[i] = std::async(std::launch::async, solve_async, i);
  }

  // wait for all MPCs to finish
  for (size_t i = 0; i < mpcs_.size(); ++i) {
    mpc_futures_[i].wait();
  }

  // put current MPC solution into the buffer
  MultiMPCSolution solution;
  try {
    solution = mpc_futures_[current_mpc_idx_].get();
    if (mpcs_[current_mpc_idx_]->is_solve_success(solution.solution, solution.stats)) {
      solution.result = MultiMPCSolveResult::SUCCESS;
    } else {
      solution.result = MultiMPCSolveResult::FAILURE;
    }
  } catch (const std::exception & e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
    solution.result = MultiMPCSolveResult::FAILURE;
  }
  return solution;
}

MPCSolveScheduleResult MultiMPCManager::solve(
  const casadi::DMDict & in, const size_t & timestamp,
  SolutionCallback callback)
{
  if (!is_mpc_ready(current_mpc_idx_)) {
    mpc_cycle_count_[current_mpc_idx_] += 1;
    // if the number of cycles is not enough, we don't start another MPC
    if (mpc_cycle_count_[current_mpc_idx_] < num_cycle_to_switch_) {
      return MPCSolveScheduleResult::NOT_SCHEDULED_PRIMARY_BUSY;
    }
    // if the current MPC is blocked for too long, we switch to the next one
    for (size_t i = 0; i < mpc_cycle_count_.size(); ++i) {
      if (i == current_mpc_idx_) {
        continue;
      }
      if (is_mpc_ready(i)) {
        current_mpc_idx_ = i;
        goto new_solve;
      }
    }
    // this is only reached when no MPC is ready
    return MPCSolveScheduleResult::NOT_SCHEDULED_NO_MPC_READY;
  }
new_solve: std::unique_lock<std::shared_mutex> lock(mpc_mutexes_[current_mpc_idx_]);
  mpc_cycle_count_[current_mpc_idx_] = 0;
  mpc_futures_[current_mpc_idx_] = std::async(
    std::launch::async,
    &MultiMPCManager::solve_mpc_thread,
    this,
    current_mpc_idx_.load(),
    in,
    timestamp,
    callback);
  return MPCSolveScheduleResult::SCHEDULED;
}

MultiMPCSolution MultiMPCManager::solve_mpc_thread(
  const size_t mpc_idx, const casadi::DMDict in,
  const size_t timestamp,
  SolutionCallback callback)
{
  const auto start_time = std::chrono::high_resolution_clock::now();
  MultiMPCSolution solution;
  solution.in = in;
  mpcs_[mpc_idx]->solve(in, solution.solution, solution.stats);
  const auto end_time = std::chrono::high_resolution_clock::now();
  solution.solve_time_nanosec =
    std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  solution.timestamp = timestamp;
  solution.mpc_name = "MPC " + std::to_string(mpc_idx);
  // only update the buffer if the solution is from the current MPC
  const auto solve_success = mpcs_[mpc_idx]->is_solve_success(solution.solution, solution.stats);
  if (current_mpc_idx_ == mpc_idx) {
    if (solve_success) {
      solution.result = MultiMPCSolveResult::SUCCESS;
    } else {
      solution.result = MultiMPCSolveResult::FAILURE;
    }
  } else {
    if (solve_success) {
      solution.result = MultiMPCSolveResult::SUCCESS_OUTDATED;
    } else {
      solution.result = MultiMPCSolveResult::FAILURE_OUTDATED;
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
  std::shared_lock<std::shared_mutex> lock(mpc_mutexes_[mpc_idx]);
  const auto is_ready = !mpc_futures_[mpc_idx].valid() ||
    mpc_futures_[mpc_idx].wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  return is_ready;
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
