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

#ifndef TORCH_CASADI_INTERFACE__TORCH_CASADI_FUNCTION_HPP_
#define TORCH_CASADI_INTERFACE__TORCH_CASADI_FUNCTION_HPP_

#include <torch/torch.h>
#include <memory>
#include <string>
#include <casadi/casadi.hpp>

namespace torch_casadi_interface
{
class TorchCasadiEvalFunction : public casadi::Callback
{
public:
  TorchCasadiEvalFunction(
    std::shared_ptr<torch::jit::script::Module> module_,
    const casadi_int & in_dim, const casadi_int & out_dim);
  virtual ~TorchCasadiEvalFunction() = default;

  void init() override;
  casadi::DMVector eval(const casadi::DMVector & arg) const override;
  casadi_int get_n_in() override;
  casadi_int get_n_out() override;
  casadi::Sparsity get_sparsity_in(casadi_int i) override;
  casadi::Sparsity get_sparsity_out(casadi_int i) override;

protected:
  std::shared_ptr<torch::jit::script::Module> module_;
  casadi_int in_dim_;
  casadi_int out_dim_;
  bool use_cuda_;
};

class TorchCasadiJacobianFunction : public TorchCasadiEvalFunction
{
public:
  TorchCasadiJacobianFunction(
    std::shared_ptr<torch::jit::script::Module> module_,
    const casadi_int & in_dim, const casadi_int & out_dim);
  virtual ~TorchCasadiJacobianFunction() = default;
  casadi_int get_n_in() override;
  casadi::Sparsity get_sparsity_in(casadi_int i) override;
  casadi::Sparsity get_sparsity_out(casadi_int i) override;
  casadi::DMVector eval(const casadi::DMVector & arg) const override;
};

class TorchCasadiFunction : public TorchCasadiEvalFunction
{
public:
  TorchCasadiFunction(
    std::shared_ptr<torch::jit::script::Module> module_,
    const casadi_int & in_dim, const casadi_int & out_dim);
  virtual ~TorchCasadiFunction() = default;
  bool has_jacobian() const override;
  Function get_jacobian(
    const std::string & name,
    const casadi::StringVector & inames,
    const casadi::StringVector & onames,
    const casadi::Dict & opts) const override;

protected:
  TorchCasadiJacobianFunction jacobian_function_;
};
}  // namespace torch_casadi_interface

#endif  // TORCH_CASADI_INTERFACE__TORCH_CASADI_FUNCTION_HPP_
