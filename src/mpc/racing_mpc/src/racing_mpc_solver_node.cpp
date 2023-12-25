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
#include "racing_mpc/ros_param_loader.hpp"
#include "lmpc_utils/ros_param_helper.hpp"
#include "lmpc_utils/ros_casadi_helper.hpp"
#include "lmpc_utils/utils.hpp"
#include "vehicle_model_factory/vehicle_model_factory.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
RacingMPCSolverNode::RacingMPCSolverNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("racing_mpc_solver_node", options),
  profiler_(std::make_unique<lmpc::utils::CycleProfiler<double>>(10)),
  profiler_iter_count_(std::make_unique<lmpc::utils::CycleProfiler<double>>(10)),
  diagnostics_pub_(create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", 10)),
  name_(declare_parameter<std::string>("racing_mpc_solver_node.name", "Solver 0"))
{
    // initialize mpc solver with ROS parameters
    auto config = lmpc::mpc::racing_mpc::load_parameters(this);
    auto model = vehicle_model::vehicle_model_factory::load_vehicle_model(
        utils::declare_parameter<std::string>(
        this, "racing_mpc_node.vehicle_model_name"), this
    );
    bool full_dynamics = utils::declare_parameter<bool>(
        this, "racing_mpc_node.full_dynamics", false);
    mpc_ = std::make_unique<RacingMPC>(config, model, full_dynamics);
    
    // initialize mpc solve service
    solve_mpc_srv_ = create_service<SolveMPC>(
        "solve_mpc", std::bind(&RacingMPCSolverNode::solve_mpc_callback, this,
        std::placeholders::_1, std::placeholders::_2));
}

void RacingMPCSolverNode::solve_mpc_callback(
SolveMPC::Request::ConstSharedPtr request,
SolveMPC::Response::SharedPtr response)
{
    static size_t profile_step_count = 0;

    // Unpack the requst into DMDict
    auto sol_in = casadi::DMDict();
    for (size_t i = 0; i < request->keys_in.size(); i++)
    {
        sol_in[request->keys_in[i]] = utils::ros_array_to_dm(request->values_in[i]);
    }

    // Solve the MPC problem
    // TODO(haoru): consider time budget
    auto sol_out = casadi::DMDict{};
    auto stats = casadi::Dict{};
    auto t0 = std::chrono::high_resolution_clock::now();
    mpc_->solve(sol_in, sol_out, stats);
    response->solved = mpc_->is_solve_success(sol_out, stats);
    if (response->solved)
    {
        // Pack the solution into the response
        response->keys_out.reserve(sol_out.size());
        response->values_out.reserve(sol_out.size());
        for (casadi::DMDict::const_iterator it = sol_out.begin(); it != sol_out.end(); ++it)
        {
            response->keys_out.push_back(it->first);
            response->values_out.push_back(utils::dm_to_ros_array(it->second));
        }
    }
    else
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "MPC \"%s\" failed to solve.", name_.c_str());
    }
    response->duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();
    
    // Profiler
    const auto duration_ms = response->duration_ns / 1e6;
    profiler_->add_cycle_stats(duration_ms);
    if (stats.count("iter_count")) {
        profiler_iter_count_->add_cycle_stats(static_cast<double>(stats.at("iter_count")));
    }
    profile_step_count++;
    if (profile_step_count == profiler_->capacity()) {
        auto diagnostics_msg = diagnostic_msgs::msg::DiagnosticArray();
        diagnostics_msg.status.push_back(
        profiler_->profile().to_diagnostic_status(
            "Racing MPC Solve Time", "(ms)", 1000.0));
        diagnostics_msg.status.push_back(
        profiler_iter_count_->profile().to_diagnostic_status(
            "Racing MPC Iteration Count", "Number of Solver Iterations", 50));
        diagnostics_msg.header.stamp = now();
        diagnostics_pub_->publish(diagnostics_msg);
        profile_step_count = 0;
    }
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc

int main (int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options{};
    auto node = std::make_shared<lmpc::mpc::racing_mpc::RacingMPCSolverNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
