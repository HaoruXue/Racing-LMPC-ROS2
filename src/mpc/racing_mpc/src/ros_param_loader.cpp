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

#include <string>
#include <memory>
#include <vector>

#include <lmpc_utils/ros_param_helper.hpp>

#include "racing_mpc/ros_param_loader.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
RacingMPCConfig::SharedPtr load_parameters(rclcpp::Node * node)
{
  auto declare_double = [&](const char * name) {
      return lmpc::utils::declare_parameter<double>(node, name);
    };
  auto declare_vec = [&](const char * name) {
      return lmpc::utils::declare_parameter<std::vector<double>>(node, name);
    };
  auto declare_int = [&](const char * name) {
      return lmpc::utils::declare_parameter<int64_t>(node, name);
    };
  auto declare_bool = [&](const char * name) {
      return lmpc::utils::declare_parameter<bool>(node, name);
    };

  return std::make_shared<RacingMPCConfig>(
    RacingMPCConfig{
          declare_double("racing_mpc.max_cpu_time"),
          declare_int("racing_mpc.max_iter"),
          declare_double("racing_mpc.tol"),
          static_cast<size_t>(declare_int("racing_mpc.n")),
          declare_double("racing_mpc.margin"),
          declare_double("racing_mpc.average_track_width"),
          declare_bool("racing_mpc.verbose"),
          casadi::DM(declare_double("racing_mpc.q_contour")),
          casadi::DM(declare_double("racing_mpc.q_heading")),
          casadi::DM(declare_double("racing_mpc.q_vel")),
          casadi::DM::reshape(casadi::DM(declare_vec("racing_mpc.r")), 3, 3),
          casadi::DM(declare_vec("racing_mpc.x_max")),
          casadi::DM(declare_vec("racing_mpc.x_min")),
          casadi::DM(declare_vec("racing_mpc.u_max")),
          casadi::DM(declare_vec("racing_mpc.u_min")),
        }
  );
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
