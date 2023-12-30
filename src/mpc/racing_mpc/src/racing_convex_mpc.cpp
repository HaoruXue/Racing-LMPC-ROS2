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

#include "racing_mpc/racing_convex_mpc.hpp"
#include "lmpc_utils/utils.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
RacingConvexMPC::RacingConvexMPC(
  RacingMPCConfig::SharedPtr mpc_config,
  BaseVehicleModel::SharedPtr model,
  const bool & full_dynamics)
: config_(mpc_config), model_(model),
  scale_x_(casadi::DM{2000.0, 10.0, 0.1, 80.0, 2.0, 2.0}),
  scale_u_(casadi::DM{10.0, 0.3}),
  align_abscissa_(utils::align_abscissa_function(config_->N)),
  X_(casadi::SX::sym("X", model_->nx(), config_->N)),
  U_(casadi::SX::sym("U", model_->nu(), config_->N)),
  dU_(casadi::SX::sym("dU", model_->nu(), config_->N - 1)),
  X_ref_(casadi::SX::sym("X_ref", model_->nx(), config_->N)),
  U_ref_(casadi::SX::sym("U_ref", model_->nu(), config_->N - 1)),
  // dU_ref_(casadi::SX::sym("dU_ref", model_->nu(), config_->N - 1)),
  T_ref_(casadi::SX::sym("T_ref", 1, config_->N - 1)),
  x_ic_(casadi::SX::sym("x_ic", model_->nx(), 1)),
  u_ic_(casadi::SX::sym("u_ic", model_->nu(), 1)),
  bound_left_(casadi::SX::sym("bound_left", 1, config_->N)),
  bound_right_(casadi::SX::sym("bound_right", 1, config_->N)),
  total_length_(casadi::SX::sym("total_length", 1, 1)),
  bank_angle_(casadi::SX::sym("bank", 1, config_->N)),
  curvatures_(casadi::SX::sym("k", 1, config_->N)),
  vel_ref_(casadi::SX::sym("vel_ref", 1, config_->N)),
  solved_(false),
  full_dynamics_(full_dynamics),
  enable_boundary_slack_(static_cast<double>(config_->q_boundary) > 0.0)
{
  using casadi::MX;
  using casadi::SX;
  using casadi::Slice;

  // configure solver
  if (full_dynamics) {
    throw std::runtime_error("Full dynamics is not supported by convex MPC.");
  }
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

  // a helper function that converts the dynamics model to
  // take in du as control
  const auto x_sym = casadi::SX::sym("x", model_->nx(), 1);
  const auto u_sym = casadi::SX::sym("u", model_->nu(), 1);
  const auto du_sym = casadi::SX::sym("du", model_->nu(), 1);
  const auto k_sym = casadi::SX::sym("k", 1, 1);
  const auto dt_sym = casadi::SX::sym("dt", 1, 1);
  const auto bank_sym = casadi::SX::sym("bank", 1, 1);
  const auto uip1_sym = u_sym + du_sym * dt_sym;
  const auto xip1_sym = model_->discrete_dynamics()(
    casadi::SXDict{{"x", x_sym}, {"u", uip1_sym}, {"k", k_sym}, {"dt", dt_sym},
      {"bank", bank_sym}}).at("xip1");
  const auto Ad = SX::jacobian(SX::vertcat({xip1_sym, uip1_sym}), SX::vertcat({x_sym, u_sym}));
  const auto Bd = SX::jacobian(SX::vertcat({xip1_sym, uip1_sym}), du_sym);
  const auto gd = SX::vertcat({xip1_sym, uip1_sym}) -
    SX::mtimes({Ad, SX::vertcat({x_sym, u_sym})}) - SX::mtimes({Bd, u_sym});
  auto dynamics_jac = casadi::Function(
    "dynamics_jac", {x_sym, u_sym, du_sym, k_sym, dt_sym, bank_sym},
    {Ad, Bd, gd}, {"x", "u", "du", "k", "dt", "bank"}, {"A", "B", "g"});

  // objective
  casadi::SX f = 0;
  // prepare some SX vectors to be stacked later
  // we are using casadi's high-level QP interface
  // the variable naming conventions can be found here
  // https://web.casadi.org/docs/#high-level-interface
  casadi::SXVector x_vec;
  casadi::SXVector g_vec;
  casadi::SXVector lbg_vec;
  casadi::SXVector ubg_vec;
  casadi::SXVector lbx_vec;
  casadi::SXVector ubx_vec;
  std::vector<casadi_int> ng;
  std::vector<casadi_int> nu;
  std::vector<casadi_int> nx;

  // initialize slack variables
  const auto margin = config_->margin + model_->get_base_config().chassis_config->b / 2.0;
  if (enable_boundary_slack_) {
    boundary_slack_ = SX::sym("boundary_slack", 1, config_->N - 1);
  }

  for (casadi_int i = 0; i < static_cast<casadi_int>(config_->N); i++) {
    const auto xi = X_(Slice(), i) * SX(scale_x_);
    const auto ui = U_(Slice(), i) * SX(scale_u_);
    SX dui;
    SX ti;
    const auto k = curvatures_(i);
    const auto bank_angle = bank_angle_(i);
    if (i < static_cast<casadi_int>(config_->N - 1)) {
      dui = dU_(Slice(), i) * SX(scale_u_);
      ti = T_ref_(i);
    }

    // add xi, ui, dui to decision variable x.
    // note that our ui is a state, and dui is a control input.
    // according to hpipm requirements, they are ordered like
    // [x0, u0, x1, u1].
    x_vec.push_back(X_(Slice(), i));
    x_vec.push_back(U_(Slice(), i));
    lbx_vec.push_back(config_->x_min / scale_x_);
    lbx_vec.push_back(config_->u_min / scale_u_);
    ubx_vec.push_back(config_->x_max / scale_x_);
    ubx_vec.push_back(config_->u_max / scale_u_);
    nx.push_back(model_->nx() + model_->nu());
    if (i < static_cast<casadi_int>(config_->N - 1)) {
      x_vec.push_back(dU_(Slice(), i));
      // TODO(haoru): hardcoded rate constraints. Replace with config.
      lbx_vec.push_back(config_->u_min * 2 / scale_u_);
      ubx_vec.push_back(config_->u_max * 2 / scale_u_);
      nu.push_back(model_->nu());
      if (enable_boundary_slack_) {
        x_vec.push_back(boundary_slack_(i));
        lbx_vec.push_back(-std::numeric_limits<double>::infinity());
        ubx_vec.push_back(std::numeric_limits<double>::infinity());
        nu.back() += 1;
      }
    }

    // add cost terms
    const auto xi_base =
      model_->to_base_state()(casadi::SXDict{{"x", xi}, {"u", ui}}).at("x_out");
    const auto dv = xi_base(XIndex::VX) - vel_ref_(i);
    SX q_contour = SX(config_->q_contour);
    SX q_heading = SX(config_->q_heading);
    SX q_vel = SX(config_->q_vel);
    SX q_vy = SX(config_->q_vy);
    SX q_vyaw = SX(config_->q_vyaw);
    if (i == static_cast<casadi_int>(config_->N - 1)) {
      q_contour *= 10.0;
      q_heading *= 10.0;
      q_vel *= 10.0;
      q_vy *= 10.0;
      q_vyaw *= 10.0;
    } else {
      f += SX::mtimes({ui.T(), config_->R, ui});
      f += SX::mtimes({dui.T(), config_->R_d, dui});
      if (enable_boundary_slack_) {
        f += SX::mtimes({boundary_slack_(i).T(), config_->q_boundary, boundary_slack_(i)});
      }
    }
    f += xi_base(XIndex::PY) * xi_base(XIndex::PY) * SX(config_->q_contour);
    f += xi_base(XIndex::YAW) * xi_base(XIndex::YAW) * SX(config_->q_heading);
    f += dv * dv * SX(config_->q_vel);
    f += xi_base(XIndex::VY) * xi_base(XIndex::VY) * SX(config_->q_vy);
    f += xi_base(XIndex::VYAW) * xi_base(XIndex::VYAW) * SX(config_->q_vyaw);

    if (i < static_cast<casadi_int>(config_->N - 1)) {
      // add dynamics constraints
      const auto xi_ref = X_ref_(Slice(), i);
      const auto ui_ref = U_ref_(Slice(), i);
      // const auto dui_ref = dU_ref_(Slice(), i);
      const auto xip1 = X_(Slice(), i + 1) * SX(scale_x_);
      const auto uip1 = U_(Slice(), i + 1) * SX(scale_u_);
      const auto ABg = dynamics_jac(
        casadi::SXDict{{"x", xi_ref}, {"u", ui_ref}, {"du", SX::zeros(model_->nu(), 1)}, {"k", k},
          {"dt", ti}, {"bank", bank_angle}});
      const auto & A = ABg.at("A");
      const auto & B = ABg.at("B");
      const auto & g = ABg.at("g");
      g_vec.push_back(
        SX::mtimes(
          A,
          SX::vertcat({xi, ui})) + SX::mtimes(B, dui) + g - SX::vertcat({xip1, uip1}));
      lbg_vec.push_back(SX::zeros(model_->nx() + model_->nu(), 1));
      ubg_vec.push_back(SX::zeros(model_->nx() + model_->nu(), 1));
    }

    // non-dynamic constraints
    casadi_int ng_i = 0;
    // add boundary constraints
    if (i < static_cast<casadi_int>(config_->N - 1)) {
      const auto PY = X_(XIndex::PY, i) * SX(scale_x_(XIndex::PY));
      if (enable_boundary_slack_) {
        g_vec.push_back(PY + boundary_slack_(i));
        lbg_vec.push_back(bound_right_(i) + margin);
        ubg_vec.push_back(bound_left_(i) - margin);
      } else {
        g_vec.push_back(PY);
        lbg_vec.push_back(bound_right_(i) + margin);
        ubg_vec.push_back(bound_left_(i) - margin);
      }
      ng_i += 1;
    }

    // add initial condition constraint
    if (i == 0) {
      g_vec.push_back(xi - x_ic_);
      lbg_vec.push_back(SX::zeros(model_->nx(), 1));
      ubg_vec.push_back(SX::zeros(model_->nx(), 1));

      g_vec.push_back(ui - u_ic_);
      lbg_vec.push_back(SX::zeros(model_->nu(), 1));
      ubg_vec.push_back(SX::zeros(model_->nu(), 1));
      ng_i += model_->nx() + model_->nu();
    }
    ng.push_back(ng_i);
  }

  // save the problem vectors
  prob_.f = f;
  prob_.x = SX::vertcat(x_vec);
  prob_.g = SX::vertcat(g_vec);
  prob_.lbg = SX::vertcat(lbg_vec);
  prob_.ubg = SX::vertcat(ubg_vec);
  prob_.lbx = SX::vertcat(lbx_vec);
  prob_.ubx = SX::vertcat(ubx_vec);
  prob_.p = SX::vertcat(
    {SX::reshape(T_ref_, -1, 1), SX::reshape(x_ic_, -1, 1),
      SX::reshape(u_ic_, -1, 1), SX::reshape(X_ref_, -1, 1), SX::reshape(U_ref_, -1, 1),
      SX::reshape(bound_left_, -1, 1), SX::reshape(bound_right_, -1, 1),
      SX::reshape(total_length_, -1, 1), SX::reshape(curvatures_, -1, 1),
      SX::reshape(vel_ref_, -1, 1), SX::reshape(bank_angle_, -1, 1)});

  // build the solver
  const casadi::SXDict prob({{"f", f}, {"x", prob_.x}, {"g", prob_.g}, {"p", prob_.p}});
  solver_ = casadi::qpsol("mpc", "osqp", prob, p_opts);
}

const RacingMPCConfig & RacingConvexMPC::get_config() const
{
  return *config_.get();
}

bool RacingConvexMPC::init()
{
  return true;
}

void RacingConvexMPC::solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats)
{
  (void) stats;
  using casadi::DM;
  using casadi::MX;
  using casadi::Slice;

  const auto & total_length = in.at("total_length");
  const auto & x_ic = in.at("x_ic");
  const auto & u_ic = in.at("u_ic");
  // const auto & t_ic = in.at("t_ic");
  auto X_ref = in.at("X_ref");
  X_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", X_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  const auto U_ref = in.at("U_ref");
  const auto & bound_left = in.at("bound_left");
  const auto & bound_right = in.at("bound_right");
  const auto & curvatures = in.at("curvatures");
  const auto & vel_ref = in.at("vel_ref");
  const auto & bank_angle = in.at("bank_angle");
  const auto & T_ref = in.at("T_ref");

  // std::cout << "[x_ic]:\n" << x_ic << std::endl;
  // std::cout << "[u_ic]:\n" << u_ic << std::endl;
  // std::cout << "[t_ic]:\n" << t_ic << std::endl;
  // std::cout << "[X_ref]:" << X_ref << std::endl;
  // std::cout << "[U_ref]:" << U_ref << std::endl;
  // std::cout << "[bound_left]\n:" << bound_left << std::endl;
  // std::cout << "[bound_right]\n:" << bound_right << std::endl;
  // std::cout << "[curvatures]\n:" << curvatures << std::endl;
  // std::cout << "[vel_ref]\n:" << vel_ref << std::endl;

  // if (config_->learning) {
  //   opti_.set_value(ss_, in.at("ss_x"));
  //   opti_.set_value(ss_costs_, in.at("ss_j"));
  //   opti_.set_initial(convex_combi_, in.at("convex_combi_optm_ref"));
  // }
  // std::cout << "[ss_j]:\n" << ss_j << std::endl;
  // std::cout << "[ss_x]:\n" << ss_x(XIndex::PX, Slice()) << std::endl;
  if (!in.count("X_optm_ref")) {
    throw std::runtime_error("No optimal reference given.");
  }
  auto X_optm_ref = in.at("X_optm_ref");
  X_optm_ref(XIndex::PX, Slice()) = align_abscissa_(
    casadi::DMDict{{"abscissa_1", X_optm_ref(XIndex::PX, Slice())},
      {"abscissa_2", DM::ones(1, config_->N) * x_ic(XIndex::PX)},
      {"total_distance", DM::ones(1, config_->N) * total_length}}).at("abscissa_1_aligned");
  const auto U_optm_ref = casadi::DM::horzcat({u_ic, in.at("U_optm_ref")});
  const auto & dU_optm_ref = in.at("dU_optm_ref");

  casadi::SXVector sub_v = {
    T_ref_, x_ic_, u_ic_, X_ref_, U_ref_, bound_left_, bound_right_, total_length_,
    curvatures_, vel_ref_, bank_angle_, X_, U_, dU_, boundary_slack_};
  casadi::SXVector sub_vdef = {T_ref, x_ic, u_ic, X_ref, U_ref, bound_left, bound_right,
    total_length, curvatures, vel_ref, bank_angle, X_optm_ref / scale_x_, U_optm_ref / scale_u_,
    dU_optm_ref / scale_u_, 0.0};
  casadi::SXVector sub_ex = {prob_.x, prob_.lbg, prob_.ubg, prob_.lbx, prob_.ubx, prob_.p};
  casadi::SXVector sub_result = casadi::SX::substitute(sub_ex, sub_v, sub_vdef);

  // solve problem
  try {
    // if (config_->learning) {
    //   out["convex_combi_optm"] = sol_->value(convex_combi_);
    //   // std::cout << DM::mtimes(out["ss_x"], out["convex_combi_optm"])(XIndex::VX) << std::endl;
    // }
    // std::cout << "[X_optm]:" << out.at("X_optm") << std::endl;
    // std::cout << "[U_optm]:" << out.at("U_optm") << std::endl;
    // std::cout << "[dU_optm]:" << out.at("dU_optm") << std::endl;
    const auto sol = solver_(
      casadi::DMDict{
            {"x0", sub_result.at(0)},
            {"lbg", sub_result.at(1)}, {"ubg", sub_result.at(2)},
            {"lbx", sub_result.at(3)}, {"ubx", sub_result.at(4)}, {"p", sub_result.at(5)}}).at("x");
    // casadi::SXVector sub_v = {prob_.x};
    // casadi::SXVector sub_vdef = {sol};
    // casadi::SX X = X_, U = U_, dU = dU_;
    // casadi::SXVector sub_ex = {X, U, dU};
    // casadi::SX::substitute_inplace(sub_v, sub_vdef, sub_ex, true);
    auto X_optm = casadi::DM::zeros(model_->nx(), config_->N);
    auto U_optm = casadi::DM::zeros(model_->nu(), config_->N);
    auto dU_optm = casadi::DM::zeros(model_->nu(), config_->N - 1);

    casadi_int i = 0;
    casadi_int k = 0;
    const casadi_int nx = model_->nx();
    const casadi_int nu = model_->nu();
    while (true) {
      X_optm(casadi::Slice(), k) = sol(casadi::Slice(i, i + nx));
      i += nx;
      U_optm(casadi::Slice(), k) = sol(casadi::Slice(i, i + nu));
      i += nu;
      if (k < static_cast<casadi_int>(config_->N - 1)) {
        dU_optm(casadi::Slice(), k) = sol(casadi::Slice(i, i + nu));
        if (enable_boundary_slack_) {
          i += 1;
        }
      }
      i += nu;
      k += 1;
      if (k == static_cast<casadi_int>(config_->N)) {
        break;
      }
    }

    out["X_optm"] = X_optm * scale_x_;
    out["U_optm"] =
      U_optm(
      casadi::Slice(),
      casadi::Slice(1, std::numeric_limits<casadi_int>::max())) * scale_u_;
    out["dU_optm"] = dU_optm * scale_u_;
    solved_ = true;
  } catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    // throw e;
    // out["X_optm"] = opti_.debug().value(X_) * scale_x_;
    // out["U_optm"] = opti_.debug().value(U_) * scale_u_;
    // out["dU_optm"] = opti_.debug().value(dU_) * scale_u_;
    // if (config_->learning) {
    //   out["convex_combi_optm"] = opti_.debug().value(convex_combi_);
    // }
    // stats = opti_.stats();
    // std::cout << "[X_optm]:" << out.at("X_optm") << std::endl;
    // std::cout << "[U_optm]:" << out.at("U_optm") << std::endl;
    // std::cout << "[dU_optm]:" << out.at("dU_optm") << std::endl;
  }
}

void RacingConvexMPC::create_warm_start(const casadi::DMDict & in, casadi::DMDict & out)
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

BaseVehicleModel & RacingConvexMPC::get_model()
{
  return *model_;
}

const bool & RacingConvexMPC::solved() const
{
  return solved_;
}

// void RacingConvexMPC::build_tracking_cost(casadi::MX & cost)
// {
//   using casadi::MX;
//   using casadi::Slice;

//   // --- MPC stage cost ---
//   const auto x0 = X_(Slice(), 0) * scale_x_;
//   for (size_t i = 0; i < config_->N - 1; i++) {
//     const auto xi = X_(Slice(), i) * scale_x_;
//     const auto ui = U_(Slice(), i - 1) * scale_u_;
//     const auto dui = dU_(Slice(), i - 1) * scale_u_;
//     // xi start with 1 since x0 must equal to x_ic and there is nothing we can do about it
//     // const auto d_px =
//     // utils::align_abscissa<MX>(xi(XIndex::PX), x0(XIndex::PX), total_length_) - x0(XIndex::PX);
//     const auto x_base = model_->to_base_state()(casadi::MXDict{{"x", xi}, {"u", ui}}).at("x_out");
//     const auto dv = x_base(XIndex::VX) - vel_ref_(i);
//     // const auto dv = x_base(XIndex::VX) - 10.0;
//     cost += x_base(XIndex::PY) * x_base(XIndex::PY) * config_->q_contour;
//     cost += x_base(XIndex::YAW) * x_base(XIndex::YAW) * config_->q_heading;
//     cost += dv * dv * config_->q_vel;
//     cost += x_base(XIndex::VY) * x_base(XIndex::VY) * config_->q_vy;
//     cost += x_base(XIndex::VYAW) * x_base(XIndex::VYAW) * config_->q_vyaw;

//     cost += MX::mtimes({ui.T(), config_->R, ui});
//     cost += MX::mtimes({dui.T(), config_->R_d, dui});
//   }

//   // terminal cost
//   const auto xN = X_(Slice(), config_->N - 1) * scale_x_;
//   const auto uN = U_(Slice(), config_->N - 2) * scale_u_;
//   const auto x_base_N = model_->to_base_state()(casadi::MXDict{{"x", xN}, {"u", uN}}).at("x_out");
//   const auto dv = x_base_N(XIndex::VX) - vel_ref_(config_->N - 1);
//   cost += x_base_N(XIndex::PY) * x_base_N(XIndex::PY) * config_->q_contour * 10.0;
//   cost += x_base_N(XIndex::YAW) * x_base_N(XIndex::YAW) * config_->q_heading * 10.0;
//   cost += dv * dv * config_->q_vel * 10.0;
// }

// void RacingConvexMPC::build_lmpc_cost(casadi::MX & cost)
// {
//   using casadi::MX;
//   using casadi::Slice;

//   convex_combi_ = opti_.variable(config_->num_ss_pts);
//   ss_ = opti_.parameter(model_->nx(), config_->num_ss_pts);
//   ss_costs_ = opti_.parameter(1, config_->num_ss_pts);
//   const auto xN = X_(Slice(), -1) * scale_x_;
//   const auto xN_combi = MX::mtimes({ss_, convex_combi_});
//   // convex combination constraint
//   opti_.subject_to(convex_combi_ >= 0.0);
//   opti_.subject_to(MX::sum1(convex_combi_) == 1.0);

//   bool enable_convex_hull_slack = static_cast<double>(MX::sumsqr(config_->convex_hull_slack)) > 0.0;
//   if (enable_convex_hull_slack) {
//     convex_hull_slack_ = opti_.variable(model_->nx(), 1);
//     opti_.subject_to(xN == xN_combi + convex_hull_slack_);
//     cost += MX::mtimes(
//       {convex_hull_slack_.T(), MX::diag(
//           config_->convex_hull_slack), convex_hull_slack_});
//   } else {
//     opti_.subject_to(xN_combi == xN);
//   }

//   cost += MX::mtimes({ss_costs_, convex_combi_});

//   // control effort and rate cost
//   for (size_t i = 0; i < config_->N - 1; i++) {
//     const auto ui = U_(Slice(), i - 1) * scale_u_;
//     const auto dui = dU_(Slice(), i - 1) * scale_u_;
//     cost += MX::mtimes({ui.T(), config_->R, ui});
//     cost += MX::mtimes({dui.T(), config_->R_d, dui});

//     // const auto xi = X_(Slice(), i) * scale_x_;
//     // const auto x_base =
//     //   model_->to_base_state()(casadi::MXDict{{"x", xi}, {"u", ui}}).at("x_out");
//     // cost += x_base(XIndex::VY) * x_base(XIndex::VY) * config_->q_vy;
//     // cost += x_base(XIndex::VYAW) * x_base(XIndex::VYAW) * config_->q_vyaw;

//     // cost += MX::mtimes({ui.T(), config_->R, ui});
//     // cost += MX::mtimes({dui.T(), config_->R_d, dui});
//   }
// }

// void RacingConvexMPC::build_boundary_constraint(casadi::MX & cost)
// {
//   using casadi::MX;
//   using casadi::Slice;

//   bool enable_boundary_slack = static_cast<double>(config_->q_boundary) > 0.0;
//   const auto PY = X_(XIndex::PY, Slice()) * scale_x_(XIndex::PY);
//   const auto margin = config_->margin + model_->get_base_config().chassis_config->b / 2.0;
//   if (enable_boundary_slack) {
//     boundary_slack_ = opti_.variable(1);
//     opti_.subject_to(
//       opti_.bounded(
//         bound_right_ + margin - boundary_slack_, PY,
//         bound_left_ - margin + boundary_slack_));
//     opti_.subject_to(boundary_slack_ >= 0.0);
//     cost += MX::mtimes({boundary_slack_.T(), config_->q_boundary, boundary_slack_});
//   } else {
//     opti_.subject_to(opti_.bounded(bound_right_ + margin, PY, bound_left_ - margin));
//   }
// }
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
