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

#include "kinematic_bicycle_model/kinematic_bicycle_model.hpp"
#include "lmpc_utils/utils.hpp"
#define GRAVITY 9.8

namespace lmpc
{
namespace vehicle_model
{
namespace kinematic_bicycle_model
{
KinematicBicycleModel::KinematicBicycleModel(
  base_vehicle_model::BaseVehicleModelConfig::SharedPtr base_config,
  KinematicBicycleModelConfig::SharedPtr config)
: base_vehicle_model::BaseVehicleModel(base_config), config_(config)
{
  compile_dynamics();
}

const KinematicBicycleModelConfig & KinematicBicycleModel::get_config() const
{
  return *config_.get();
}

size_t KinematicBicycleModel::nx() const
{
  return 4;
}

size_t KinematicBicycleModel::nu() const
{
  return 2;
}

void KinematicBicycleModel::add_nlp_constraints(casadi::Opti & opti, const casadi::MXDict & in)
{
(void) opti;
(void) in;
}

void KinematicBicycleModel::calc_lon_control(
  const casadi::DMDict & in, double & throttle,
  double & brake_kpa) const
{
  const auto u_lon = static_cast<double>(in.at("u")(UIndex::F_LON));
  if (u_lon > 0.0) {
    throttle = calc_throttle(u_lon);
    brake_kpa = 0.0;
  } else {
    throttle = 0.0;
    brake_kpa = calc_brake(u_lon);
  }
}

void KinematicBicycleModel::calc_lat_control(
  const casadi::DMDict & in,
  double & steering_rad) const
{
  const auto u = in.at("u").get_elements();
  steering_rad = u[UIndex::STEER];
}

void KinematicBicycleModel::compile_dynamics()
{
  using casadi::SX;

  const auto x = SX::sym("x", nx());
  const auto u = SX::sym("u", nu());
  const auto k = SX::sym("k", 1);  // curvature for frenet frame
  const auto bank = SX::sym("bank", 1);  // bank angle
  const auto dt = SX::sym("dt", 1);  // time step

  const auto px = x(XIndex::PX);
  const auto py = x(XIndex::PY);
  const auto phi = x(XIndex::YAW);  // yaw
  const auto v = x(XIndex::V);  // body frame velocity magnitude
  const auto f_lon = u(UIndex::F_LON);  // longitudinal control force
  const auto delta = u(UIndex::STEER);  // front wheel angle
  const auto v_sq = v * v;

  const auto & m = get_base_config().chassis_config->total_mass;  // mass of car
  const auto & l = get_base_config().chassis_config->wheel_base;  // wheelbase
  const auto lr = get_base_config().chassis_config->cg_ratio * l;  // cg to front axle
  const auto & fr = get_base_config().chassis_config->fr;  // rolling resistance coefficient
  const auto & cl_f = get_base_config().aero_config->cl_f;  // downforce coefficient at front
  const auto & cl_r = get_base_config().aero_config->cl_r;  // downforce coefficient at rear
  const auto & rho = get_base_config().aero_config->air_density;  // air density
  const auto & A = get_base_config().aero_config->frontal_area;  // frontal area
  const auto & cd = get_base_config().aero_config->drag_coeff;  // drag coefficient
  // const auto & mu = get_config().mu;  // tyre - track friction coefficient

  // magic tyre parameters
  // const auto & tyre_f = *get_base_config().front_tyre_config;
  // const auto & Bf = tyre_f.pacejka_b;  // magic formula B - front
  // const auto & Cf = tyre_f.pacejka_c;  // magic formula C - front
  // const auto & Ef = tyre_f.pacejka_e;  // magic formula E - front
  // const auto & Fz0_f = tyre_f.pacejka_fz0;  // magic formula Fz0 - front
  // const auto & eps_f = tyre_f.pacejka_eps;  // extended magic formula epsilon - front
  // const auto & tyre_r = *get_base_config().rear_tyre_config;
  // const auto & Br = tyre_r.pacejka_b;  // magic formula B - rear
  // const auto & Cr = tyre_r.pacejka_c;  // magic formula C - rear
  // const auto & Er = tyre_r.pacejka_e;  // magic formula E - rear
  // const auto & Fz0_r = tyre_r.pacejka_fz0;  // magic formula Fz0 - rear
  // const auto & eps_r = tyre_r.pacejka_eps;  // extended magic formula epsilon - rear

  // compute kinematics
  const auto beta = atan(lr * tan(delta) / l);
  auto phi_dot = v * cos(beta) * tan(delta) / l;
  const auto global_yaw_rate = phi_dot;
  auto px_dot = v * cos(beta + phi);
  auto py_dot = v * sin(beta + phi);
  const auto ax = (f_lon * 1000.0 - 0.5 * cd * rho * A * v_sq - fr * m * GRAVITY) / m;
  const auto v_dot = ax;

  const auto x_dot = vertcat(px_dot, py_dot, phi_dot, v_dot);

  dynamics_ = casadi::Function(
    "kinematic_bicycle_model",
    {x, u, k, bank},
    {x_dot},
    {"x", "u", "k", "bank"},
    {"x_dot"});

  const auto Ac = SX::jacobian(x_dot, x);
  const auto Bc = SX::jacobian(x_dot, u);

  dynamics_jacobian_ = casadi::Function(
    "kinematic_bicycle_model_jacobian",
    {x, u, k, bank},
    {Ac, Bc},
    {"x", "u", "k", "bank"},
    {"A", "B"}
  );

  // discretize dynamics
  SX xip1;
  const auto & integrator_type = get_base_config().modeling_config->integrator_type;
  if (integrator_type == base_vehicle_model::IntegratorType::RK4) {
    xip1 = utils::rk4_function(nx(), nu(), dynamics_)(
      casadi::SXDict{{"x", x}, {"u", u}, {"k", k}, {"dt", dt}, {"bank", bank}}
    ).at("xip1");
  } else if (integrator_type == base_vehicle_model::IntegratorType::EULER) {
    xip1 = utils::euler_function(nx(), nu(), dynamics_)(
      casadi::SXDict{{"x", x}, {"u", u}, {"k", k}, {"dt", dt}, {"bank", bank}}
    ).at("xip1");
  } else {
    throw std::runtime_error("unsupported integrator type");
  }

  discrete_dynamics_ = casadi::Function(
    "single_track_planar_model_discrete_dynamics",
    {x, u, k, dt, bank},
    {xip1},
    {"x", "u", "k", "dt", "bank"},
    {"xip1"});

  const auto Ad = SX::jacobian(xip1, x);
  const auto Bd = SX::jacobian(xip1, u);

  discrete_dynamics_jacobian_ = casadi::Function(
    "single_track_planar_model_discrete_dynamics_jacobian",
    {x, u, k, dt},
    {Ad, Bd},
    {"x", "u", "k", "dt"},
    {"A", "B"}
  );

  // state conversions
  const auto base_x = SX{vertcat(
      x(kinematic_bicycle_model::XIndex::PX),
      x(kinematic_bicycle_model::XIndex::PY),
      x(kinematic_bicycle_model::XIndex::YAW),
      x(kinematic_bicycle_model::XIndex::V) * cos(beta),
      x(kinematic_bicycle_model::XIndex::V) * sin(beta),
      global_yaw_rate
    )};
  to_base_state_ = casadi::Function(
    "to_base_state", {x, u}, {base_x}, {"x", "u"}, {"x_out"});

  const auto base_x_sym = SX::sym("base_x", BaseVehicleModel::nx());
  const auto base_u_sym = SX::sym("base_u", BaseVehicleModel::nu());
  const auto derived_x = SX{vertcat(
      base_x_sym(base_vehicle_model::XIndex::PX),
      base_x_sym(base_vehicle_model::XIndex::PY),
      base_x_sym(base_vehicle_model::XIndex::YAW),
      hypot(base_x_sym(base_vehicle_model::XIndex::VX), base_x_sym(base_vehicle_model::XIndex::VY))
    )};
  from_base_state_ = casadi::Function(
    "from_base_state", {base_x_sym, base_u_sym}, {derived_x}, {"x", "u"}, {"x_out"});

  const auto u_derived_sym = SX::sym("u", nu());
  const auto u_base_out = SX::vertcat(
        {
          SX::if_else(
            u_derived_sym(UIndex::F_LON) > 0.0, u_derived_sym(UIndex::F_LON),
            0.0),
          SX::if_else(
            u_derived_sym(UIndex::F_LON) < 0.0, u_derived_sym(UIndex::F_LON),
            0.0),
          u_derived_sym(UIndex::STEER)
        });
  const auto u_derived_out = SX::vertcat(
        {
          SX::if_else(
            abs(base_u_sym(base_vehicle_model::UIndex::FD)) >
            abs(base_u_sym(base_vehicle_model::UIndex::FB)),
            base_u_sym(base_vehicle_model::UIndex::FD), base_u_sym(base_vehicle_model::UIndex::FB)),
          base_u_sym(base_vehicle_model::UIndex::STEER)
        });
  to_base_control_ = casadi::Function(
    "to_base_control", {x, u_derived_sym}, {u_base_out}, {"x", "u"}, {"u_out"});
  from_base_control_ = casadi::Function(
    "from_base_control", {base_x_sym, base_u_sym}, {u_derived_out}, {"x", "u"}, {"u_out"});
}
}  // namespace kinematic_bicycle_model
}  // namespace vehicle_model
}  // namespace lmpc
