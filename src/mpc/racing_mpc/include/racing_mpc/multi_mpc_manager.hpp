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

/**
 * @brief An interface for a MPC solver to be
 * managable by the MultiMPCManager.
 * Remember that casadi matrices are not thread safe,
 * and creating matrices on new threads can cause segfaults.
 *
 */
class MultiMPCInterface
{
public:
  typedef std::shared_ptr<MultiMPCInterface> SharedPtr;
  typedef std::unique_ptr<MultiMPCInterface> UniquePtr;

  virtual ~MultiMPCInterface() = default;

  /**
   * @brief Pass the solve request to the solver.
   * This call should be non-blocking.
   * Remember that casadi matrices are not thread safe,
   * and creating matrices on new threads can cause segfaults.
   * If the solver is casadi-based, you need to condense the
   * entire solve process into a casadi::Function.
   * Alternatively, see MPCSolverNodeInterface for a ROS2-based
   * solution to communicate with a casadi-based solver in a
   * separate ROS2 node with service calls.
   * If using other solvers, convert casadi matrices to the
   * corresponding matrix type (e.g. Eigen) before passing to the solver
   * on a new thread.
   *
   * @param in Solver input
   * @param timestamp Cycle number corresponding to the solver input.
   * This should be current cycle number + 1, for example,
   * if solving for the next cycle.
   * @param callback Callback function to be called when the solver finishes solving.
   */
  virtual void solve(
    const casadi::DMDict & in, const size_t & timestamp,
    SolutionCallback callback) = 0;

  /**
   * @brief If the solver is ready to solve.
   *
   * @return true there is no solve in progress
   * @return false there is a solve in progress
   */
  virtual bool is_ready() = 0;
};

/**
 * @brief This class interfaces MPCSolverNode, which is
 * a seperate ROS2 node with a service call to solve MPC.
 * Multiple MPCSolverNodeInterface can be created to
 * communicate with multiple MPCSolverNodes.
 *
 */
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

/**
 * @brief A class for managing multiple MPCs.
 * Often times, we want to have a backup MPC in case the primary MPC solve stalls.
 * This class can schedule solve with the next available MPC.
 * Remember that casadi matrices are not thread safe,
 * and creating matrices on new threads can cause segfaults.
 *
 */
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
