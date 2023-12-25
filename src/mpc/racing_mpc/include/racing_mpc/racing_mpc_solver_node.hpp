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

#ifndef RACING_MPC__RACING_MPC_SOLVER_NODE_HPP_
#define RACING_MPC__RACING_MPC_SOLVER_NODE_HPP_

#include <memory>
#include <shared_mutex>
#include <vector>
#include <string>

#include <casadi/casadi.hpp>

#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <lmpc_msgs/srv/solve_mpc.hpp>

#include <lmpc_utils/cycle_profiler.hpp>

#include "racing_mpc/racing_mpc_config.hpp"
#include "racing_mpc/racing_mpc.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
using lmpc_msgs::srv::SolveMPC;

class RacingMPCSolverNode : public rclcpp::Node
{
public:
  explicit RacingMPCSolverNode(const rclcpp::NodeOptions & options);

protected:
  // mpc solve service
  rclcpp::Service<SolveMPC>::SharedPtr solve_mpc_srv_;
  // mpc solver
  RacingMPC::UniquePtr mpc_ {};
  // profilers
  lmpc::utils::CycleProfiler<double>::UniquePtr profiler_ {};
  lmpc::utils::CycleProfiler<double>::UniquePtr profiler_iter_count_ {};

  // service callback
  void solve_mpc_callback(
    SolveMPC::Request::ConstSharedPtr request,
    SolveMPC::Response::SharedPtr response);

  // diagnostic publisher
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;

  // name
  std::string name_;
};
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
#endif  // RACING_MPC__RACING_MPC_SOLVER_NODE_HPP_
