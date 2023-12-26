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
#include <shared_mutex>
#include <mutex>
#include <casadi/casadi.hpp>

#include <rclcpp/rclcpp.hpp>
#include <lmpc_msgs/srv/solve_mpc.hpp>

#include "racing_mpc/racing_mpc_config.hpp"
#include "lmpc_utils/mpc_solution_buffer.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
struct MultiMPCSolution
{
  typedef std::shared_ptr<MultiMPCSolution> SharedPtr;
  typedef std::unique_ptr<MultiMPCSolution> UniquePtr;

  casadi::DMDict in;
  casadi::DMDict solution;
  size_t timestamp;
  std::string mpc_name;
  size_t solve_time_nanosec;
  bool success;
  bool outdated;
};

using SolutionCallback = std::function<void (MultiMPCSolution)>;

class MultiMPCInterface
{
public:
  typedef std::shared_ptr<MultiMPCInterface> SharedPtr;
  typedef std::unique_ptr<MultiMPCInterface> UniquePtr;

  virtual ~MultiMPCInterface() = default;
  virtual void solve(
    const casadi::DMDict & in, const size_t & timestamp,
    SolutionCallback callback) = 0;
  virtual bool is_ready() = 0;
};

class MPCSolverNodeInterface : public MultiMPCInterface
{
public:
  typedef std::shared_ptr<MPCSolverNodeInterface> SharedPtr;
  typedef std::unique_ptr<MPCSolverNodeInterface> UniquePtr;
  using Client = rclcpp::Client<lmpc_msgs::srv::SolveMPC>;
  MPCSolverNodeInterface(rclcpp::Node * node, const std::string & name);

  // MultiMPCInterface overrides
  void solve(
    const casadi::DMDict & in, const size_t & timestamp,
    SolutionCallback callback) override;
  bool is_ready() override;

private:
  rclcpp::Node * node_;
  rclcpp::Client<lmpc_msgs::srv::SolveMPC>::SharedPtr client_;
  std::string name_;
  std::atomic_bool is_ready_;

  void service_callback(Client::SharedFutureWithRequest cb, SolutionCallback callback);
};

struct MultiMPCManagerConfig
{
  typedef std::shared_ptr<MultiMPCManagerConfig> SharedPtr;
  typedef std::unique_ptr<MultiMPCManagerConfig> UniquePtr;

  size_t num_cycle_to_switch;    // number of cycles to wait before switching to the next MPC
  size_t max_extrapolate_horizon;  // maximum number of cycles to extrapolate the solution
  std::vector<MultiMPCInterface::SharedPtr> mpcs;
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
  typedef std::function<void (MultiMPCSolution)> SolutionCallback;

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
   * @param callback callback function to be called when the primary MPC finishes solving
   */
  void initialize(const casadi::DMDict & in, const size_t & timestamp, SolutionCallback callback);

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

protected:
  size_t num_cycle_to_switch_;
  std::vector<MultiMPCInterface::SharedPtr> mpcs_;
  std::vector<size_t> mpc_cycle_count_;
  std::atomic_ulong current_mpc_idx_;

  void solution_callback(MultiMPCSolution solution, size_t mpc_idx, SolutionCallback callback);
};
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc

#endif  // RACING_MPC__MULTI_MPC_MANAGER_HPP_
