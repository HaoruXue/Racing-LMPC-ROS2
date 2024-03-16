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

#include "racing_mpc/racing_mpc_solver_node.hpp"
#include "racing_mpc/racing_mpc.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options{};
  auto node = std::make_shared<lmpc::mpc::racing_mpc::RacingMPCSolverNode>(options);
  auto mpc = dynamic_cast<lmpc::mpc::racing_mpc::RacingMPC*>(node->get_mpc());
  auto mpc_function = mpc->to_function();
  mpc_function.generate(
    {
        {"cpp", true},
        {"with_header", true}
    }
  );
  rclcpp::shutdown();
  return 0;
}
