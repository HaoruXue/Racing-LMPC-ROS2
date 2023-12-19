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

#ifndef RACING_MPC__MULTI_MPC_MANAGER_HPP_
#define RACING_MPC__MULTI_MPC_MANAGER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <future>
#include <atomic>
#include <casadi/casadi.hpp>
#include "racing_mpc/racing_mpc_config.hpp"
#include "lmpc_utils/mpc_solution_buffer.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
class MultiMPCInterface
{
public:
  typedef std::shared_ptr<MultiMPCInterface> SharedPtr;
  typedef std::unique_ptr<MultiMPCInterface> UniquePtr;

  virtual ~MultiMPCInterface() = default;
  virtual void solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats) = 0;
  virtual bool is_solve_success(const casadi::DMDict & out, const casadi::Dict & stats) const = 0;
  virtual casadi::DM get_x(const casadi::DMDict & out) const = 0;
  virtual casadi::DM get_u(const casadi::DMDict & out) const = 0;
};

struct MultiMPCManagerConfig
{
  typedef std::shared_ptr<MultiMPCManagerConfig> SharedPtr;
  typedef std::unique_ptr<MultiMPCManagerConfig> UniquePtr;

  size_t num_cycle_to_switch = 0;    // number of cycles to wait before switching to the next MPC
  std::vector<MultiMPCInterface::SharedPtr> mpcs;
};

struct MultiMPCSolution
{
  typedef std::shared_ptr<MultiMPCSolution> SharedPtr;
  typedef std::unique_ptr<MultiMPCSolution> UniquePtr;

  casadi::DMDict solution;
  casadi::Dict stats;
  size_t timestamp;
  std::string mpc_name;
};

enum class MPCSolveScheduleResult : uint8_t
{
  SCHEDULED,
  NOT_SCHEDULED_PRIMARY_BUSY,
  NOT_SCHEDULED_NO_MPC_READY
};

class MultiMPCManager
{
public:
  typedef std::shared_ptr<MultiMPCManager> SharedPtr;
  typedef std::unique_ptr<MultiMPCManager> UniquePtr;
  typedef std::function<void (const MultiMPCSolution &)> SolutionCallback;

  explicit MultiMPCManager(const MultiMPCManagerConfig & config);

  /**
   * @brief Performs a solve for every MPC in the manager.
   * This can initialize some solver behavior such as JIT compilation.
   * This call is blocking untill all MPCs finish solving.
   * It also initializes the solution buffer.
   * The solution of the primary MPC is returned.
   *
   * @param in solver input
   * @param timestamp timestamp of the solver input
   * @return MultiMPCSolution solution of the primary MPC
   */
  MultiMPCSolution initialize(const casadi::DMDict & in, const size_t & timestamp);

  /**
   * @brief Schedule a solve for the primary MPC.
   * If the primary MPC is not ready for up to time defined in the config,
   * it will schedule the solve on the next MPC.
   * If the primary MPC is not ready but the time has not reached,
   * the solve will not be scheduled.
   * If no MPC is ready, the solve will not be scheduled.
   * This call is non-blocking.
   * When a solver finishes solving, the callback function will be called.
   * The solution buffer will be updated only if the solver is still the primary MPC
   * after the solve.
   *
   * @param in solver input
   * @param timestamp timestamp of the solver input
   * @param callback callback function to be called when the primary MPC finishes solving
   *
   * @return SCHEDULED if the solve is scheduled
   * @return NOT_SCHEDULED_PRIMARY_BUSY if the primary MPC is still solving but within the time limit
   * @return NOT_SCHEDULED_NO_MPC_READY if no MPC is ready
   */
  MPCSolveScheduleResult solve(
    const casadi::DMDict & in, const size_t & timestamp,
    SolutionCallback callback);

  /**
   * @brief Get the solution from the solution buffer.
   *
   * @param timestamp timestamp of the solution
   * @return lmpc::utils::MPCSolution solution
   */
  lmpc::utils::MPCSolution get_solution(const size_t & timestamp);

  /**
   * @brief Check if the solution buffer is initialized.
   *
   * @return true if the solution buffer is initialized
   * @return false if the solution buffer is not initialized
   */
  bool is_solution_initialized() const;

protected:
  size_t num_cycle_to_switch_;
  std::vector<MultiMPCInterface::SharedPtr> mpcs_;
  std::vector<size_t> mpc_cycle_count_;
  std::vector<std::future<MultiMPCSolution>> mpc_futures_;
  std::atomic_ulong current_mpc_idx_;
  lmpc::utils::MPCSolutionBuffer buffer_;

  MultiMPCSolution solve_mpc_thread(
    const size_t mpc_idx, const casadi::DMDict in,
    const size_t timestamp, SolutionCallback callback);
  bool is_mpc_ready(const size_t & mpc_idx);
};
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
#endif  // RACING_MPC__MULTI_MPC_MANAGER_HPP_
