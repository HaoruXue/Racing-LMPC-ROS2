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

#include <math.h>
#include <exception>
#include <vector>
#include <iostream>
#include <chrono>

#include "racing_mpc/racing_mpc.hpp"
#include "lmpc_utils/utils.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
RacingMPC::RacingMPC(
  RacingMPCConfig::SharedPtr mpc_config,
  BaseVehicleModel::SharedPtr model,
  const bool & full_dynamics)
: config_(mpc_config), model_(model),
  scale_x_(config_->x_max),
  scale_u_(config_->u_max),
  align_abscissa_(utils::align_abscissa_function(config_->N)),
  opti_(full_dynamics ? casadi::Opti() : casadi::Opti("conic")),
  cost_(casadi::MX::zeros(1)),
  X_(opti_.variable(model_->nx(), config_->N)),
  U_(opti_.variable(model_->nu(), config_->N - 1)),
  dU_(opti_.variable(model_->nu(), config_->N - 1)),
  scale_x_param_(opti_.parameter(model_->nx(), 1)),
  scale_u_param_(opti_.parameter(model_->nu(), 1)),
  X_ref_(opti_.parameter(model_->nx(), config_->N)),
  U_ref_(opti_.parameter(model_->nu(), config_->N - 1)),
  T_ref_(opti_.parameter(1, config_->N - 1)),
  x_ic_(opti_.parameter(model_->nx(), 1)),
  u_ic_(opti_.parameter(model_->nu(), 1)),
  bound_left_(opti_.parameter(1, config_->N)),
  bound_right_(opti_.parameter(1, config_->N)),
  total_length_(opti_.parameter(1, 1)),
  bank_angle_(opti_.parameter(1, config_->N)),
  curvatures_(opti_.parameter(1, config_->N)),
  vel_ref_(opti_.parameter(1, config_->N)),
  solved_(false),
  sol_(nullptr),
  full_dynamics_(full_dynamics)
{
  using casadi::MX;
  using casadi::Slice;

  // further adjust x scale
  scale_x_(XIndex::PX) = config_->x_max(XIndex::VX);
  scale_x_(XIndex::PY) = config_->average_track_width;
  scale_x_(XIndex::YAW) = M_PI;
  // throw if scale contains inf
  if (static_cast<double>(casadi::DM::sum1(scale_x_)) == std::numeric_limits<double>::infinity()) {
    throw std::runtime_error("scale_x contains inf.");
  }
  if (static_cast<double>(casadi::DM::sum1(scale_u_)) == std::numeric_limits<double>::infinity()) {
    throw std::runtime_error("scale_u contains inf.");
  }

  // configure solver
  if (full_dynamics) {
    auto p_opts = casadi::Dict{
      {"expand", true},
      {"print_time", config_->verbose ? true : false},
      {"error_on_fail", true}
    };
    // if (config_->jit) {
    //   p_opts["jit"] = true;
    //   p_opts["jit_options"] = casadi::Dict{{"flags", "-Ofast"}};
    //   p_opts["compiler"] = "shell";
    // }
    const auto s_opts = casadi::Dict{
      {"max_cpu_time", config_->max_cpu_time},
      {"tol", config_->tol},
      {"print_level", config_->verbose ? 5 : 0},
      {"max_iter", static_cast<casadi_int>(config_->max_iter)}
    };
    opti_.solver("ipopt", p_opts, s_opts);
  } else {
    auto p_opts = casadi::Dict{
      {"expand", true},
      {"print_time", config_->verbose ? true : false},
      {"error_on_fail", true},
      {"osqp", casadi::Dict
        {
          {"polish", true},
          {"verbose", config_->verbose ? true : false},
        }
      }
    };
    if (config_->jit) {
      p_opts["jit"] = true;
      p_opts["jit_options"] = casadi::Dict{{"flags", "-Ofast"}};
      p_opts["compiler"] = "shell";
    }
    const auto s_opts = casadi::Dict{};
    opti_.solver("osqp", p_opts, s_opts);
  }
}

const RacingMPCConfig & RacingMPC::get_config() const
{
  return *config_.get();
}

bool RacingMPC::init()
{
  // set up track boundary constraint
  build_boundary_constraint(cost_);

  if (config_->learning) {
    // LMPC cost
    build_lmpc_cost(cost_);
  } else {
    // Tracking MPC cost
    build_tracking_cost(cost_);
  }

  build_dynamics_constraint();
  build_initial_constraint();
  opti_.minimize(cost_);
  return true;
}

void RacingMPC::solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats)
{
  using casadi::DM;
  using casadi::MX;
  using casadi::Slice;

  const auto & total_length = in.at("total_length");
  const auto & x_ic = in.at("x_ic");
  const auto & u_ic = in.at("u_ic");
  auto X_ref = in.at("X_ref");
  X_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", X_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  const auto & U_ref = in.at("U_ref");
  const auto & bound_left = in.at("bound_left");
  const auto & bound_right = in.at("bound_right");
  const auto & curvatures = in.at("curvatures");
  const auto & vel_ref = in.at("vel_ref");
  const auto & bank_angle = in.at("bank_angle");

  if (config_->learning) {
    opti_.set_value(ss_, in.at("ss_x"));
    opti_.set_value(ss_costs_, in.at("ss_j"));
    opti_.set_initial(convex_combi_, in.at("convex_combi_optm_ref"));
  }

  // set up the offsets
  const auto P0 = X_ref(XIndex::PX, Slice());
  auto x_offset = casadi::DM::zeros(model_->nx());
  x_offset(XIndex::PX) = x_ic(XIndex::PX);

  // if optimal reference is given, typically from last MPC solution,
  // initialize with this reference.
  if (in.count("X_optm_ref")) {
    auto X_optm_ref = in.at("X_optm_ref");
    X_optm_ref(XIndex::PX, Slice()) = align_abscissa_(
      casadi::DMDict{{"abscissa_1", X_optm_ref(XIndex::PX, Slice())},
        {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
        {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
    const auto & U_optm_ref = in.at("U_optm_ref");
    const auto & T_optm_ref = in.at("T_optm_ref");
    const auto & dU_optm_ref = in.at("dU_optm_ref");
    opti_.set_initial(X_, (X_optm_ref - x_offset) / scale_x_);
    opti_.set_initial(U_, U_optm_ref / scale_u_);
    opti_.set_initial(dU_, dU_optm_ref / scale_u_);
    opti_.set_value(T_ref_, T_optm_ref);
  } else {
    throw std::runtime_error("No optm reference given.");
  }

  // starting state must match
  opti_.set_value(x_ic_, x_ic - x_offset);
  opti_.set_value(u_ic_, u_ic);

  // initialize other parameters
  opti_.set_value(X_ref_, X_ref - x_offset);
  opti_.set_value(U_ref_, U_ref);
  opti_.set_value(bound_left_, bound_left);
  opti_.set_value(bound_right_, bound_right);
  opti_.set_value(total_length_, total_length);
  opti_.set_value(curvatures_, curvatures);
  opti_.set_value(vel_ref_, vel_ref);
  opti_.set_value(bank_angle_, bank_angle);
  opti_.set_value(scale_x_param_, scale_x_);
  opti_.set_value(scale_u_param_, scale_u_);

  // initialize subclass variables
  init_solve(in, out, stats);

  // solve problem
  try {
    sol_ = std::make_unique<casadi::OptiSol>(opti_.solve_limited());
    // const auto sol = opti.solve();
    solved_ = true;
    out["X_optm"] = sol_->value(X_) * scale_x_ + x_offset;
    out["U_optm"] = sol_->value(U_) * scale_u_;
    out["dU_optm"] = sol_->value(dU_) * scale_u_;
    stats = sol_->stats();
    if (config_->learning) {
      out["convex_combi_optm"] = sol_->value(convex_combi_);
    }
  } catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
  }
}

void RacingMPC::create_warm_start(const casadi::DMDict & in, casadi::DMDict & out)
{
  using casadi::DM;
  using casadi::Slice;

  const auto & P0 = in.at("P0");
  const auto & Yaws = in.at("Yaws");
  const auto & Radii = in.at("Radii");
  const auto & current_vel = in.at("current_vel");
  const auto & target_vel = in.at("target_vel");

  if (static_cast<size_t>(P0.size2()) != config_->N) {
    throw std::length_error("create_warm_start: P0 dimension does not match MPC dimension.");
  }
  if (static_cast<size_t>(Yaws.size2()) != config_->N) {
    throw std::length_error("create_warm_start: Yaws dimension does not match MPC dimension.");
  }
  if (static_cast<double>(current_vel) <= 0.0) {
    throw std::range_error("Current velocity cannot be smaller than or equal to zero.");
  }
  if (static_cast<double>(target_vel) <= 0.0) {
    throw std::range_error("Target velocity cannot be smaller than or equal to zero.");
  }

  auto X_ref = DM::zeros(model_->nx(), config_->N);
  auto U_ref = DM::zeros(model_->nu(), config_->N - 1);
  auto T_ref = DM::zeros(config_->N - 1, 1);

  X_ref(Slice(0, 2), Slice()) = P0;
  X_ref(XIndex::YAW, Slice()) = Yaws;
  X_ref(XIndex::VX, Slice()) = DM::linspace(current_vel, target_vel, config_->N);
  X_ref(XIndex::VYAW, Slice()) = X_ref(XIndex::VX, Slice()) / Radii;

  for (size_t i = 0; i < config_->N - 1; i++) {
    // fill control force with 2nd Newton's law
    const auto v0 = X_ref(XIndex::VX, i);
    const auto v1 = X_ref(XIndex::VX, i + 1);
    const auto d = DM::norm_2(P0(Slice(), i) - P0(Slice(), i + 1));
    const auto a = static_cast<double>((pow(v1, 2) - pow(v0, 2)) / (2 * d));
    const auto f = model_->get_base_config().chassis_config->total_mass * a;
    if (f > 0.0) {
      U_ref(UIndex::FD, i) = f;
    } else {
      U_ref(UIndex::FB, i) = f;
    }
    T_ref(i) = d / v0;

    // fill steering angle with pure pursuit
    {
      U_ref(
        UIndex::STEER,
        i) = atan(model_->get_base_config().chassis_config->wheel_base / Radii(i));
    }
  }
  out["X_ref"] = X_ref;
  out["U_ref"] = U_ref;
}

casadi::Function RacingMPC::to_function()
{
  if (config_->learning)
  {
    throw std::runtime_error("Cannot convert to function for learning MPC.");
  }
  const auto inputs = casadi::MXVector{
    X_, U_, dU_, boundary_slack_,
    scale_x_param_, scale_u_param_, X_ref_, U_ref_, T_ref_,
    x_ic_, u_ic_,
    bound_left_, bound_right_, total_length_, bank_angle_, curvatures_, vel_ref_};
  const auto outputs = casadi::MXVector{X_, U_, dU_, boundary_slack_, cost_};
  const auto input_names = std::vector<std::string>{
    "X", "U", "dU", "boundary_slack",
    "scale_x_param", "scale_u_param", "X_ref", "U_ref", "T_ref",
    "x_ic", "u_ic",
    "bound_left", "bound_right", "total_length", "bank_angle", "curvatures", "vel_ref"};
  const auto output_names = std::vector<std::string>{
    "X_optm", "U_optm", "dU_optm", "boundary_slack_optm", "cost_optm"};
  return opti_.to_function(
    "racing_mpc_function",
    inputs,
    outputs,
    input_names,
    output_names
  );
}

BaseVehicleModel & RacingMPC::get_model()
{
  return *model_;
}

const bool & RacingMPC::solved() const
{
  return solved_;
}

void RacingMPC::build_tracking_cost(casadi::MX & cost)
{
  using casadi::MX;
  using casadi::Slice;

  // --- MPC stage cost ---
  const auto x0 = X_(Slice(), 0) * scale_x_param_;
  for (size_t i = 0; i < config_->N - 1; i++) {
    const auto xi = X_(Slice(), i) * scale_x_param_;
    const auto ui = U_(Slice(), i - 1) * scale_u_;
    const auto dui = dU_(Slice(), i - 1) * scale_u_;
    // xi start with 1 since x0 must equal to x_ic and there is nothing we can do about it
    // const auto d_px =
    // utils::align_abscissa<MX>(xi(XIndex::PX), x0(XIndex::PX), total_length_) - x0(XIndex::PX);
    const auto x_base = model_->to_base_state()(casadi::MXDict{{"x", xi}, {"u", ui}}).at("x_out");
    const auto dv = x_base(XIndex::VX) - vel_ref_(i);
    // const auto dv = x_base(XIndex::VX) - 10.0;
    cost += x_base(XIndex::PY) * x_base(XIndex::PY) * config_->q_contour;
    cost += x_base(XIndex::YAW) * x_base(XIndex::YAW) * config_->q_heading;
    cost += dv * dv * config_->q_vel;
    cost += x_base(XIndex::VY) * x_base(XIndex::VY) * config_->q_vy;
    cost += x_base(XIndex::VYAW) * x_base(XIndex::VYAW) * config_->q_vyaw;

    cost += MX::mtimes({ui.T(), config_->R, ui});
    cost += MX::mtimes({dui.T(), config_->R_d, dui});
  }

  // terminal cost
  const auto xN = X_(Slice(), config_->N - 1) * scale_x_param_;
  const auto uN = U_(Slice(), config_->N - 2) * scale_u_param_;
  const auto x_base_N = model_->to_base_state()(casadi::MXDict{{"x", xN}, {"u", uN}}).at("x_out");
  const auto dv = x_base_N(XIndex::VX) - vel_ref_(config_->N - 1);
  cost += x_base_N(XIndex::PY) * x_base_N(XIndex::PY) * config_->q_contour * 10.0;
  cost += x_base_N(XIndex::YAW) * x_base_N(XIndex::YAW) * config_->q_heading * 10.0;
  cost += dv * dv * config_->q_vel * 10.0;
  cost += x_base_N(XIndex::VY) * x_base_N(XIndex::VY) * config_->q_vy * 10.0;
  cost += x_base_N(XIndex::VYAW) * x_base_N(XIndex::VYAW) * config_->q_vyaw * 10.0;
}

void RacingMPC::build_lmpc_cost(casadi::MX & cost)
{
  using casadi::MX;
  using casadi::Slice;

  convex_combi_ = opti_.variable(config_->num_ss_pts);
  ss_ = opti_.parameter(model_->nx(), config_->num_ss_pts);
  ss_costs_ = opti_.parameter(1, config_->num_ss_pts);
  const auto xN = X_(Slice(), -1) * scale_x_param_;
  const auto xN_combi = MX::mtimes({ss_, convex_combi_});
  // convex combination constraint
  opti_.subject_to(convex_combi_ >= 0.0);
  opti_.subject_to(MX::sum1(convex_combi_) == 1.0);

  bool enable_convex_hull_slack = static_cast<double>(MX::sumsqr(config_->convex_hull_slack)) > 0.0;
  if (enable_convex_hull_slack) {
    convex_hull_slack_ = opti_.variable(model_->nx(), 1);
    opti_.subject_to(xN == xN_combi + convex_hull_slack_);
    cost += MX::mtimes(
      {convex_hull_slack_.T(), MX::diag(
          config_->convex_hull_slack), convex_hull_slack_});
  } else {
    opti_.subject_to(xN_combi == xN);
  }

  cost += MX::mtimes({ss_costs_, convex_combi_});

  // control effort and rate cost
  for (size_t i = 0; i < config_->N - 1; i++) {
    const auto ui = U_(Slice(), i - 1) * scale_u_param_;
    const auto dui = dU_(Slice(), i - 1) * scale_u_param_;
    cost += MX::mtimes({ui.T(), config_->R, ui});
    cost += MX::mtimes({dui.T(), config_->R_d, dui});
  }
}

void RacingMPC::build_boundary_constraint(casadi::MX & cost)
{
  using casadi::MX;
  using casadi::Slice;

  bool enable_boundary_slack = static_cast<double>(config_->q_boundary) > 0.0;
  const auto PY = X_(XIndex::PY, Slice()) * scale_x_param_(XIndex::PY);
  const auto margin = config_->margin + model_->get_base_config().chassis_config->b / 2.0;
  if (enable_boundary_slack) {
    boundary_slack_ = opti_.variable(1, config_->N);
    const auto boundary_slack = boundary_slack_ * scale_x_param_(XIndex::PY);
    opti_.subject_to(
      opti_.bounded(
        bound_right_ + margin, PY + boundary_slack,
        bound_left_ - margin));
    cost += MX::mtimes({boundary_slack, config_->q_boundary, boundary_slack.T()});
  } else {
    opti_.subject_to(opti_.bounded(bound_right_ + margin, PY, bound_left_ - margin));
  }
}

void RacingMPC::build_dynamics_constraint()
{
  using casadi::MX;
  using casadi::Slice;
  // --- model constraints ---
  for (size_t i = 0; i < config_->N - 1; i++) {
    const auto xi = X_(Slice(), i) * scale_x_param_;
    const auto xip1 = X_(Slice(), i + 1) * scale_x_param_;
    const auto ui = U_(Slice(), i) * scale_u_param_;
    const auto ti = T_ref_(i);
    const auto k = curvatures_(i);
    const auto bank_angle = bank_angle_(i);
    const auto dui = dU_(Slice(), i) * scale_u_param_;

    // primal bounds
    opti_.subject_to(opti_.bounded(config_->x_min, xi, config_->x_max));
    opti_.subject_to(opti_.bounded(config_->u_min, ui, config_->u_max));
    opti_.subject_to(opti_.bounded(config_->du_min, dui, config_->du_max));

    if (full_dynamics_) {
      // use full dynamics for dynamics constraints
      const auto xip1_pred =
        model_->discrete_dynamics()(
        {{"x", xi}, {"u", ui}, {"k", k}, {"dt", ti},
          {"bank", bank_angle}}).at("xip1");
      opti_.subject_to(xip1_pred - xip1 == 0);
    } else {
      // or use linearlized dynamics for dynamics constraints
      const auto xi_ref = X_ref_(Slice(), i);
      const auto ui_ref = U_ref_(Slice(), i);
      const auto AB =
        model_->discrete_dynamics_jacobian()(
        {{"x", xi_ref}, {"u", ui_ref}, {"k", k},
          {"dt", ti}, {"bank", bank_angle}});
      const auto x_ref_p1 =
        model_->discrete_dynamics()(
        {{"x", xi_ref}, {"u", ui_ref}, {"k", k}, {"dt", ti},
          {"bank", bank_angle}}).at(
        "xip1");
      const auto & A = AB.at("A");
      const auto & B = AB.at("B");
      const auto & g = AB.at("g");
      opti_.subject_to(xip1 - (MX::mtimes(A, xi) + MX::mtimes(B, ui) + g) == 0);
    }

    // control rate constraints
    MX uim1;
    if (i == 0) {
      uim1 = u_ic_;
    } else {
      uim1 = U_(Slice(), i - 1) * scale_u_param_;
    }
    opti_.subject_to(uim1 + dui * ti == ui);
  }
}
void RacingMPC::build_initial_constraint()
{
  using casadi::MX;
  using casadi::Slice;
  // --- initial state constraint ---
  const auto x0 = X_(Slice(), 0) * scale_x_param_;
  opti_.subject_to(x0 == x_ic_);
}

void RacingMPC::init_solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats)
{
  (void) in;
  (void) out;
  (void) stats;
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
