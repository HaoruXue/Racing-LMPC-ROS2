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

#include <torch/script.h>

#include "neural_dynamics_model/full_discrete_model.hpp"
#include "lmpc_utils/utils.hpp"

namespace lmpc
{
namespace vehicle_model
{
namespace neural_dynamics_model
{
NeuralDynamicsModel::NeuralDynamicsModel(
  base_vehicle_model::BaseVehicleModelConfig::SharedPtr base_config,
  BaseNeuralDynamicsModelConfig::SharedPtr base_nn_config)
: BaseNeuralDynamicsModel(base_config, base_nn_config),
  model_(std::make_shared<torch::jit::script::Module>(torch::jit::load(base_nn_config_->model_paths["full_discrete"])))
{
  if (base_nn_config_->use_cuda) {
    model_->to(torch::kCUDA);
  }
  torch_func_ = std::make_shared<torch_casadi_interface::TorchCasadiFunction>(
      model_, 5, 6, base_nn_config_->use_cuda);
  compile_dynamics();
}

void NeuralDynamicsModel::add_nlp_constraints(casadi::Opti & opti, const casadi::MXDict & in)
{
  // const auto & u = in.at("u");
  // casadi::MX fd, fb, delta;
  // if (config_->simplify_lon_control) {
  //   fd = u(UIndexSimple::LON) * (casadi::MX::tanh(u(UIndexSimple::LON)) * 0.5 + 0.5) *
  //     config_->lon_force_scale;
  //   fb = u(UIndexSimple::LON) * (casadi::MX::tanh(-u(UIndexSimple::LON)) * 0.5 + 0.5) *
  //     config_->lon_force_scale;
  //   delta = u(UIndexSimple::STEER_SIMPLE);
  // } else {
  //   fd = u(UIndex::FD);
  //   fb = u(UIndex::FB);
  //   delta = u(UIndex::STEER);
  // }
  // const auto & t = in.at("t");
  // const auto & Fd_max = get_config().Fd_max;
  // const auto & Fb_max = get_config().Fb_max;
  // const auto & delta_max = get_base_config().steer_config->max_steer;
  // const auto & Td = get_config().Td;
  // const auto & Tb = get_config().Tb;
  // const auto & max_steer_rate =
  //   get_base_config().steer_config->max_steer_rate;

  // if (in.count("x")) {
  //   const auto & x = in.at("x");
  //   // const auto & xip1 = in.at("xip1");
  //   const auto k =
  //     base_config_->modeling_config->use_frenet ? in.at("k") : casadi::MX::sym("k", 1, 1);
  //   const auto v = x(XIndex::VX);
  //   // const auto & mu = get_config().mu;
  //   // const auto & P_max = get_config().P_max;

  //   // dynamics constraint
  //   // auto xip1_temp = casadi::MX(xip1);
  //   // if (base_config_->modeling_config->use_frenet) {
  //   //   xip1_temp(XIndex::PX) =
  //   //     lmpc::utils::align_abscissa<casadi::MX>(
  //   //     xip1_temp(XIndex::PX), x(XIndex::PX),
  //   //     in.at("track_length"));
  //   // } else {
  //   //   xip1_temp(XIndex::YAW) =
  //   //     lmpc::utils::align_yaw<casadi::MX>(xip1_temp(XIndex::YAW), x(XIndex::YAW));
  //   // }

  //   // const auto out1 = dynamics_({{"x", x}, {"u", u}, {"k", k}});
  //   // const auto xip1_pred =
  //   // discrete_dynamics_({{"x", x}, {"u", u}, {"k", k}, {"dt", t}}).at("xip1");
  //   // opti.subject_to(xip1_pred - xip1_temp == 0);

  //   // tyre constraints
  //   // const auto Fx_ij = out1.at("Fx_ij");
  //   // const auto Fy_ij = out1.at("Fy_ij");
  //   // const auto Fz_ij = out1.at("Fz_ij");
  //   // for (int i = 0; i < 2; i++) {
  //   //   opti.subject_to(pow(Fx_ij(i) / (mu * Fz_ij(i)), 2) +
  //   //   pow(Fy_ij(i) / (mu * Fz_ij(i)), 2) <= 1);
  //   // }

  //   // static actuator cconstraint
  //   // opti.subject_to(v * fd <= P_max);
  //   // opti.subject_to(v >= 0.0);
  //   if (config_->simplify_lon_control) {
  //     opti.subject_to(
  //       opti.bounded(
  //         Fb_max / config_->lon_force_scale, u(UIndexSimple::LON),
  //         Fd_max / config_->lon_force_scale));
  //   } else {
  //     opti.subject_to(pow(fd * fb, 2) <= 100.0);
  //     opti.subject_to(opti.bounded(0.0, fd, Fd_max));
  //     opti.subject_to(opti.bounded(Fb_max, fb, 0.0));
  //   }
  //   opti.subject_to(opti.bounded(-1.0 * delta_max, delta, delta_max));
  // }

  // // dynamic actuator constraint
  // if (in.count("uip1")) {
  //   const auto & uip1 = in.at("uip1");

  //   if (config_->simplify_lon_control) {
  //     opti.subject_to(
  //       opti.bounded(
  //         Fb_max / config_->lon_force_scale / Tb,
  //         (uip1(UIndexSimple::LON) - u(UIndexSimple::LON)) / t,
  //         Fd_max / config_->lon_force_scale / Td));
  //     opti.subject_to(
  //       opti.bounded(
  //         -max_steer_rate, (uip1(UIndexSimple::STEER_SIMPLE) - delta) / t, max_steer_rate));
  //   } else {
  //     opti.subject_to((uip1(UIndex::FD) - fd) / t <= Fd_max / Td);
  //     opti.subject_to((uip1(UIndex::FB) - fb) / t >= Fb_max / Tb);
  //     opti.subject_to(
  //       opti.bounded(
  //         -max_steer_rate, (uip1(UIndex::STEER) - delta) / t, max_steer_rate));
  //   }
  // }

  // if (in.count("dui")) {
  //   const auto & dui = in.at("dui");
  //   if (config_->simplify_lon_control) {
  //     opti.subject_to(
  //       opti.bounded(
  //         Fb_max / config_->lon_force_scale / Tb, dui(UIndexSimple::LON),
  //         Fd_max / config_->lon_force_scale / Td));
  //     opti.subject_to(
  //       opti.bounded(-max_steer_rate, dui(UIndexSimple::STEER_SIMPLE), max_steer_rate));
  //   } else {
  //     opti.subject_to(dui(UIndex::FD) <= Fd_max / Td);
  //     opti.subject_to(dui(UIndex::FB) >= Fb_max / Tb);
  //     opti.subject_to(
  //       opti.bounded(-max_steer_rate, dui(UIndex::STEER), max_steer_rate));
  //   }
  // }
}

void NeuralDynamicsModel::calc_lon_control(
  const casadi::DMDict & in, double & throttle,
  double & brake_kpa) const
{
  const auto u = in.at("u").get_elements();
  const auto fd = u[UIndex::LON] * (tanh(u[UIndex::LON]) * 0.5 + 0.5) * base_nn_config_->lon_force_scale;
  const auto fb = u[UIndex::LON] * (tanh(-u[UIndex::LON]) * 0.5 + 0.5) *
      base_nn_config_->lon_force_scale;

  throttle = 0.0;
  brake_kpa = 0.0;
  if (abs(fd) > abs(fb)) {
    throttle = calc_throttle(fd);
  } else {
    brake_kpa = calc_brake(fb);
  }
}

void NeuralDynamicsModel::calc_lat_control(
  const casadi::DMDict & in,
  double & steering_rad) const
{
  const auto u = in.at("u").get_elements();
  steering_rad = u[UIndex::STEER];
}

void NeuralDynamicsModel::compile_dynamics()
{
  using casadi::MX;

  const auto x = MX::sym("x", nx());
  const auto u = MX::sym("u", nu());
  const auto k = MX::sym("k", 1);  // curvature for frenet frame
  const auto dt = MX::sym("dt", 1);  // time step
  const auto bank = MX::sym("bank", 1);
  const auto px = x(XIndex::PX);
  const auto py = x(XIndex::PY);
  const auto phi = x(XIndex::YAW);  // yaw
  const auto omega = x(XIndex::VYAW);  // yaw rate
  const auto vx = x(XIndex::VX);  // body frame longitudinal velocity
  const auto vy = x(XIndex::VY);  // body frame lateral velocity
  const auto v_sq = vx * vx + vy * vy;
  const auto u_a = u(UIndex::LON) * base_nn_config_->lon_force_scale;
  const auto delta = u(UIndex::STEER);

  // predict body frame states from NN
  const auto nn_input = vertcat(vx, vy, omega, u_a, delta);
  const auto nn_output = torch_func_->operator()(nn_input)[0];
  const auto px_d_body = nn_output(XIndex::PX);
  const auto py_d_body = nn_output(XIndex::PY);
  auto phi_d = nn_output(XIndex::YAW);
  
  // convert px_d_body, py_d_body to global frame
 auto px_d = px_d_body * cos(phi) - py_d_body * sin(phi);
 auto py_d = px_d_body * sin(phi) + py_d_body * cos(phi);

  if (base_config_->modeling_config->use_frenet) {
    // convert to frenet frame
    const auto k_banked = k * cos(bank);
    px_d /= (1 - py * k_banked);
    phi_d -= k_banked * px_d;
  } else {
    // phi_dot *= cos(bank);
    // px_dot = vx * cos(phi) - vy * sin(phi) * cos(bank);
    // py_dot = vx * sin(phi) + vy * cos(phi) * cos(bank);
  }

  const auto x_next = vertcat(px + px_d , py + py_d, phi + phi_d, nn_output(casadi::Slice(XIndex::VX, XIndex::VYAW + 1)));

  dynamics_ = casadi::Function(
    "neural_dynamics_model_dynamics",
    {x, u, k, bank},
    {x_next},
    {"x", "u", "k", "bank"},
    {"x_dot"});

  const auto Ad_Bd = MX::jacobian(x_next, MX::vertcat({x, u}));
  const auto Ad = Ad_Bd(casadi::Slice(), casadi::Slice(0, static_cast<casadi_int>(nx())));
  const auto Bd = Ad_Bd(casadi::Slice(), casadi::Slice(static_cast<casadi_int>(nx()), static_cast<casadi_int>(nx() + nu())));

  dynamics_jacobian_ = casadi::Function(
    "neural_dynamics_model_dynamics_jacobian",
    {x, u, k, bank},
    {Ad, Bd},
    {"x", "u", "k", "bank"},
    {"A", "B"}
  );

  // discretize dynamics
  discrete_dynamics_ = dynamics_;
  const auto gd = x_next - (MX::mtimes(Ad, x) + MX::mtimes(Bd, u));

  discrete_dynamics_jacobian_ = casadi::Function(
    "neural_dynamics_model_discrete_dynamics_jacobian",
    {x, u, k, dt, bank},
    {Ad, Bd, gd},
    {"x", "u", "k", "dt", "bank"},
    {"A", "B", "g"}
  );
}
}  // namespace neural_dynamics_model
}  // namespace vehicle_model
}  // namespace lmpc
