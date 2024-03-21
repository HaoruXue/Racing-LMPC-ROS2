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

#include <memory>
#include <chrono>
#include <torch_casadi_interface/torch_casadi_function.hpp>

namespace torch_casadi_interface
{
TorchCasadiEvalFunction::TorchCasadiEvalFunction(
  std::shared_ptr<torch::jit::script::Module> module,
  const casadi_int & in_dim, const casadi_int & out_dim)
: module_(module), in_dim_(in_dim), out_dim_(out_dim)
{
  if (torch::cuda::is_available()) {
    use_cuda_ = true;
  } else {
    use_cuda_ = false;
  }
}

void TorchCasadiEvalFunction::init()
{
  module_->eval();
}

casadi::DMVector TorchCasadiEvalFunction::eval(const casadi::DMVector & arg) const
{
  auto options = torch::TensorOptions().dtype(torch::kDouble);
  auto input = torch::from_blob(
    const_cast<double *>(arg[0].nonzeros().data()), {1,
      arg[0].size1()}, options);
  if (use_cuda_) {
    input = input.to(torch::kCUDA);
  }
  // convert to fp32
  input = input.to(torch::kFloat32);
  auto output = module_->forward({input}).toTensor();
  output = output.to(torch::kDouble);
  if (use_cuda_) {
    output = output.to(torch::kCPU);
  }
  auto output_data = output.const_data_ptr<double>();
  casadi::DMVector result =
  {casadi::DM::reshape(
      casadi::DM(
        std::vector<double>(
          output_data,
          output_data + output.numel())), out_dim_, 1)};
  return result;
}

casadi_int TorchCasadiEvalFunction::get_n_in()
{
  return 1;
}

casadi_int TorchCasadiEvalFunction::get_n_out()
{
  return 1;
}

casadi::Sparsity TorchCasadiEvalFunction::get_sparsity_in(casadi_int i)
{
  return casadi::Sparsity::dense(in_dim_, 1);
}

casadi::Sparsity TorchCasadiEvalFunction::get_sparsity_out(casadi_int i)
{
  return casadi::Sparsity::dense(out_dim_, 1);
}

TorchCasadiJacobianFunction::TorchCasadiJacobianFunction(
  std::shared_ptr<torch::jit::script::Module> module,
  const casadi_int & in_dim, const casadi_int & out_dim)
: TorchCasadiEvalFunction(module, in_dim, out_dim)
{
  construct("torch_casadi_jacobian_function");
}

casadi_int TorchCasadiJacobianFunction::get_n_in()
{
  return 2;
}

casadi::DMVector TorchCasadiJacobianFunction::eval(const casadi::DMVector & arg) const
{
  auto options = torch::TensorOptions().dtype(torch::kDouble);
  auto input = torch::from_blob(
    const_cast<double *>(arg[0].nonzeros().data()), {1,
      arg[0].size1()}, options);
  if (use_cuda_) {
    input = input.to(torch::kCUDA);
  }
  // convert to fp32
  input = input.to(torch::kFloat32);
  // tile the input to match the output size
  input = input.repeat({out_dim_, 1});
  input.requires_grad_(true);
  auto output = module_->forward({input}).toTensor();
  auto grads = torch::eye(out_dim_);
  if (use_cuda_) {
    grads = grads.to(torch::kCUDA);
  }
  auto jacobian = torch::autograd::grad(
    {output}, {input}, {grads}, true, false)[0];
  
  jacobian = jacobian.to(torch::kDouble);
  if (use_cuda_) {
    jacobian = jacobian.to(torch::kCPU);
  }
  auto jacobian_data = jacobian.const_data_ptr<double>();
  casadi::DMVector result =
  {casadi::DM::reshape(
      casadi::DM(
        std::vector<double>(
          jacobian_data,
          jacobian_data + jacobian.numel())), jacobian.sizes()[0], jacobian.sizes()[1])};
  return result;
}

casadi::Sparsity TorchCasadiJacobianFunction::get_sparsity_in(casadi_int i)
{
  if (i == 0) {
    return casadi::Sparsity::dense(in_dim_, 1);
  } else {
    return casadi::Sparsity::dense(out_dim_, 1);
  }
}

casadi::Sparsity TorchCasadiJacobianFunction::get_sparsity_out(casadi_int i)
{
  return casadi::Sparsity::dense(out_dim_, in_dim_);
}

TorchCasadiFunction::TorchCasadiFunction(
  std::shared_ptr<torch::jit::script::Module> module,
  const casadi_int & in_dim, const casadi_int & out_dim)
: TorchCasadiEvalFunction(module, in_dim, out_dim),
  jacobian_function_(TorchCasadiJacobianFunction(module, in_dim, out_dim))
{
  construct("torch_casadi_function");
}

bool TorchCasadiFunction::has_jacobian() const
{
  return true;
}

casadi::Function TorchCasadiFunction::get_jacobian(
  const std::string & name,
  const casadi::StringVector & inames,
  const casadi::StringVector & onames,
  const casadi::Dict & opts) const
{
  return jacobian_function_;
}

}  // namespace torch_casadi_interface
