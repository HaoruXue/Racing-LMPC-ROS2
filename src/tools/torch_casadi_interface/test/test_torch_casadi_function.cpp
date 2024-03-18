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
#include <torch/script.h>

#include <chrono>
#include <memory>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <torch_casadi_interface/torch_casadi_function.hpp>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "
const auto share_dir = ament_index_cpp::get_package_share_directory("torch_casadi_interface");
const auto test_model_dir = share_dir + "/test/dummy_model.pt";

TEST(TorchCasadiFunctionTest, TestTorchCasadiFunction)
{
  auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(test_model_dir));
  module->eval();
  if (torch::cuda::is_available()) {
    module->to(torch::kCUDA);
    std::cout << "Module moved to GPU." << std::endl;
  } else {
    std::cout << "CUDA is not available. Module remains on CPU." << std::endl;
  }
  // auto module_optimized = std::make_shared<torch::jit::script::Module>(torch::jit::optimize_for_inference(*module));
  torch_casadi_interface::TorchCasadiFunction eval_function(module, 8, 6);
  casadi::DMVector arg = {casadi::DM::ones(8, 1)};

  // infer once to warm up
  eval_function(arg);

  const auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    eval_function(arg);
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  GTEST_COUT << "Forward Compute Time: " << std::chrono::duration_cast<std::chrono::microseconds>(
    end_time - start_time).count() / 100 << " us" << std::endl;
  const auto result = eval_function(arg)[0].get_elements();

  std::vector<double> ground_truth = {-0.2255, 0.1500, -0.2820, -0.0941, -0.1130, 0.1970};
  for (casadi_int i = 0; i < 6; i++) {
    EXPECT_NEAR(result[i], ground_truth[i], 1e-3);
  }
  SUCCEED();
}

TEST(TorchCasadiFunctionTest, TestTorchCasadiJacobianFunction)
{
  auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(test_model_dir));
  module->eval();
  if (torch::cuda::is_available()) {
    module->to(torch::kCUDA);
    std::cout << "Module moved to GPU." << std::endl;
  } else {
    std::cout << "CUDA is not available. Module remains on CPU." << std::endl;
  }
  // auto module_optimized = std::make_shared<torch::jit::script::Module>(torch::jit::optimize_for_inference(*module));
  torch_casadi_interface::TorchCasadiFunction eval_function(module, 8, 6);
  casadi::DMVector arg = {casadi::DM::ones(8, 1)};
  const auto x_sym = casadi::MX::sym("x", 8, 1);
  const auto sym_out = eval_function({x_sym})[0];
  const auto jacobian_sym = casadi::MX::jacobian(sym_out, x_sym);
  casadi::Function jacobian_function = casadi::Function("jacobian", {x_sym}, {jacobian_sym});

  // infer once to warm up
  eval_function(arg);

  const auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    jacobian_function(arg);
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  GTEST_COUT << "Jacobian Compute Time: " << std::chrono::duration_cast<std::chrono::microseconds>(
    end_time - start_time).count() / 100 << " us" << std::endl;

  const auto result = jacobian_function(arg)[0].get_elements();
  std::vector<std::vector<double>> ground_truth = {
    {0.0110, -0.0011, -0.0166, 0.0097, -0.0083, 0.0180, -0.0117, -0.0318},
    {0.0033, 0.0009, -0.0002, 0.0120, -0.0143, -0.0183, -0.0093, -0.0002},
    {0.0040, 0.0037, 0.0021, 0.0133, -0.0147, -0.0308, -0.0090, 0.0090},
    {0.0125, -0.0344, -0.0108, 0.0055, -0.0071, 0.0246, 0.0131, -0.0212},
    {-0.0116, 0.0113, 0.0192, 0.0076, -0.0205, 0.0478, -0.0119, -0.0309},
    {0.0109, -0.0226, 0.0071, 0.0174, -0.0230, -0.0330, 0.0282, 0.0240},
  };
  for (casadi_int i = 0; i < 6; i++) {
    for (casadi_int j = 0; j < 8; j++) {
      EXPECT_NEAR(result[i * 8 + j], ground_truth[i][j], 1e-3);
    }
  }
  SUCCEED();
}
