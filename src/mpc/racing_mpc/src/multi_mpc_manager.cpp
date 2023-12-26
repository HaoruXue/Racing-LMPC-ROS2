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

#include <lmpc_utils/ros_casadi_helper.hpp>

#include "racing_mpc/multi_mpc_manager.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
MPCSolverNodeInterface::MPCSolverNodeInterface(rclcpp::Node * node, const std::string & name)
: node_(node),
  name_(name),
  is_ready_(false)
{
  client_ = node_->create_client<lmpc_msgs::srv::SolveMPC>("solve_mpc");
  // wait for the service to be available
  while (!client_->wait_for_service(std::chrono::milliseconds(100))) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(node_->get_logger(), "Interrupted while waiting for the service. Exiting.");
      return;
    }
    RCLCPP_INFO_THROTTLE(
      node_->get_logger(),
      *node_->get_clock(), 1000, "Waiting for MPC service...");
  }
  is_ready_ = true;
}

void MPCSolverNodeInterface::solve(
  const casadi::DMDict & in, const size_t & timestamp,
  SolutionCallback callback)
{
  is_ready_ = false;
  lmpc_msgs::srv::SolveMPC::Request::SharedPtr request =
    std::make_shared<lmpc_msgs::srv::SolveMPC::Request>();
  request->values_in.reserve(in.size());
  request->keys_in.reserve(in.size());
  for (casadi::DMDict::const_iterator it = in.begin(); it != in.end(); ++it) {
    request->keys_in.push_back(it->first);
    request->values_in.push_back(lmpc::utils::dm_to_ros_array(it->second));
  }
  request->header.stamp = node_->now();
  request->time_budget_ns = 1e9;  // TODO(haoru): read this from dt
  request->timestamp = timestamp;
  client_->async_send_request(
    request, [this, callback](Client::SharedFutureWithRequest cb) {
      service_callback(cb, callback);
    }
  );
}

bool MPCSolverNodeInterface::is_ready()
{
  return is_ready_;
}

void MPCSolverNodeInterface::service_callback(
  Client::SharedFutureWithRequest cb,
  SolutionCallback callback)
{
  const auto request_and_response = cb.get();
  const auto request = request_and_response.first;
  const auto response = request_and_response.second;
  MultiMPCSolution solution;
  for (size_t i = 0; i < request->keys_in.size(); i++) {
    solution.in[request->keys_in[i]] = lmpc::utils::ros_array_to_dm(request->values_in[i]);
  }
  for (size_t i = 0; i < response->keys_out.size(); i++) {
    solution.solution[response->keys_out[i]] =
      lmpc::utils::ros_array_to_dm(response->values_out[i]);
  }
  solution.timestamp = request->timestamp;
  solution.mpc_name = name_;
  solution.solve_time_nanosec = response->duration_ns;
  solution.success = response->solved;
  solution.outdated = response->outdated;
  callback(solution);
  is_ready_ = true;
}

MultiMPCManager::MultiMPCManager(const MultiMPCManagerConfig & config)
: num_cycle_to_switch_(config.num_cycle_to_switch),
  mpcs_(config.mpcs),
  mpc_cycle_count_(mpcs_.size(), 0),
  current_mpc_idx_(0)
{
}

void MultiMPCManager::initialize(
  const casadi::DMDict & in, const size_t & timestamp,
  SolutionCallback callback)
{
  // call solve on the first MPC with actual callback
  mpcs_[0]->solve(
    in, timestamp, [this, callback](MultiMPCSolution solution) {
      solution_callback(solution, 0, callback);
    });
  // call solve on ther remaining MPCs with empty callback
  for (size_t i = 1; i < mpcs_.size(); i++) {
    mpcs_[i]->solve(in, timestamp, [](MultiMPCSolution) {});
  }
}

MPCSolveScheduleResult MultiMPCManager::solve(
  const casadi::DMDict & in, const size_t & timestamp,
  SolutionCallback callback)
{
  if (!mpcs_[current_mpc_idx_]->is_ready()) {
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
      if (mpcs_[i]->is_ready()) {
        current_mpc_idx_ = i;
        goto new_solve;
      }
    }
    // this is only reached when no MPC is ready
    return MPCSolveScheduleResult::NOT_SCHEDULED_NO_MPC_READY;
  }
new_solve: mpc_cycle_count_[current_mpc_idx_] = 0;
  size_t current_mpc_idx = current_mpc_idx_;
  mpcs_[current_mpc_idx_]->solve(
    in, timestamp, [this, current_mpc_idx, callback](MultiMPCSolution solution) {
      solution_callback(solution, current_mpc_idx, callback);
    });
  return MPCSolveScheduleResult::SCHEDULED;
}

void MultiMPCManager::solution_callback(
  MultiMPCSolution solution, size_t mpc_idx,
  SolutionCallback callback)
{
  solution.outdated = mpc_idx != current_mpc_idx_;
  callback(solution);
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
