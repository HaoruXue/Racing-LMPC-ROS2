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

#include <gtest/gtest.h>
#include <chrono>

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "base_vehicle_model/ros_param_loader.hpp"
#include "single_track_planar_model/ros_param_loader.hpp"
#include "torch_dynamics_model/single_track_planar_model.hpp"

TEST(SingleTrackPlanarModelTest, TestSingleTrackPlanarModel) {
  rclcpp::init(0, nullptr);
  const auto base_share_dir = ament_index_cpp::get_package_share_directory("base_vehicle_model");
  const auto share_dir = ament_index_cpp::get_package_share_directory("torch_dynamics_model");
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "--params-file", base_share_dir + "/param/sample_vehicle.param.yaml",
    "--params-file", share_dir + "/param/sample_vehicle.param.yaml",
  });
  auto test_node = rclcpp::Node("test_torch_dynamics_model_node", options);

  auto base_config = lmpc::vehicle_model::base_vehicle_model::load_parameters(&test_node);
  auto config = lmpc::vehicle_model::single_track_planar_model::load_parameters(&test_node);
  auto model = lmpc::vehicle_model::torch_dynamics_model::SingleTrackPlanarModel(
    base_config,
    config);

  rclcpp::shutdown();
  SUCCEED();
}

TEST(SingleTrackPlanarModelTest, TestSingleTrackDynamics) {
  static constexpr int64_t num_samples = 1e6;
  static constexpr double dt = 0.01;
  static constexpr int64_t num_steps = 40;

  rclcpp::init(0, nullptr);
  const auto base_share_dir = ament_index_cpp::get_package_share_directory("base_vehicle_model");
  const auto share_dir = ament_index_cpp::get_package_share_directory("torch_dynamics_model");
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "--params-file", base_share_dir + "/param/sample_vehicle.param.yaml",
    "--params-file", share_dir + "/param/sample_vehicle.param.yaml",
  });
  auto test_node = rclcpp::Node("test_torch_dynamics_model_node", options);

  auto base_config = lmpc::vehicle_model::base_vehicle_model::load_parameters(&test_node);
  base_config->modeling_config->integrator_type = lmpc::vehicle_model::base_vehicle_model::IntegratorType::EULER;
  auto config = lmpc::vehicle_model::single_track_planar_model::load_parameters(&test_node);
  auto model = lmpc::vehicle_model::torch_dynamics_model::SingleTrackPlanarModel(
    base_config,
    config);
  model.eval();
  torch::NoGradGuard no_grad;

  auto device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Running on GPU." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "CUDA is not available! Running on CPU." << std::endl;
  }

  const auto x0 = torch::stack(
    {
      torch::randn({num_samples, 1}),
      torch::randn({num_samples, 1}),
      torch::randn({num_samples, 1}) * 0.1,
      (torch::randn({num_samples, 1}) + 1.0) * 20.0,
      torch::randn({num_samples, 1}),
      torch::randn({num_samples, 1}) * 0.5,
    }, 1
  ).squeeze().to(device);

  const auto u = torch::stack(
    {
      torch::randn({num_samples, 1}) * 5000.0,
      torch::randn({num_samples, 1}) * model.get_base_config().steer_config->max_steer,
    }, 1
  ).squeeze().to(device);

  const auto param = torch::stack(
    {
      torch::randn({num_samples, 1}) * 0.001,
      torch::zeros({num_samples, 1}),
      torch::zeros({num_samples, 1}) + dt,
    }, 1
  ).squeeze().to(device);
  
  auto x = model.forward_discrete(x0, u, param);  // warm up

  for (int64_t i = 0 ; i < 100; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < num_steps; i++) {
      x = model.forward_discrete(x, u, param);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time with " << num_samples << " samples and " << num_steps << " steps: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
  }
  rclcpp::shutdown();




  SUCCEED();
}
