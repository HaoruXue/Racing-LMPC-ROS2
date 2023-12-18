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

#ifndef RACING_MPC__MULTI_MPC_MANAGER_HPP_
#define RACING_MPC__MULTI_MPC_MANAGER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <future>
#include <atomic>
#include <casadi/casadi.hpp>
#include "racing_mpc/racing_mpc_config.hpp"
#include "lmpc_utils/mpc_solution_buffer.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
class MultiMPCInterface
{
public:
  typedef std::shared_ptr<MultiMPCInterface> SharedPtr;
  typedef std::unique_ptr<MultiMPCInterface> UniquePtr;

  virtual ~MultiMPCInterface() = default;
  virtual void solve(const casadi::DMDict in, casadi::DMDict & out, casadi::Dict & stats) = 0;
  virtual bool is_solve_success(casadi::DMDict & out, const casadi::Dict & stats) = 0;
  virtual casadi::DM get_x(casadi::DMDict & out) = 0;
  virtual casadi::DM get_u(casadi::DMDict & out) = 0;
};

struct MultiMPCManagerConfig
{
  typedef std::shared_ptr<MultiMPCManagerConfig> SharedPtr;
  typedef std::unique_ptr<MultiMPCManagerConfig> UniquePtr;

  size_t num_cycle_to_switch = 0;    // number of cycles to wait before switching to the next MPC
  std::vector<MultiMPCInterface::SharedPtr> mpcs;
};

struct MultiMPCSolution
{
  typedef std::shared_ptr<MultiMPCSolution> SharedPtr;
  typedef std::unique_ptr<MultiMPCSolution> UniquePtr;

  casadi::DMDict solution;
  casadi::Dict stats;
  size_t timestamp;
  std::string mpc_name;
};

class MultiMPCManager
{
public:
  typedef std::shared_ptr<MultiMPCManager> SharedPtr;
  typedef std::unique_ptr<MultiMPCManager> UniquePtr;
  typedef std::function<void (const MultiMPCSolution &)> SolutionCallback;

  explicit MultiMPCManager(const MultiMPCManagerConfig & config);

  void solve(const casadi::DMDict & in, const size_t & timestamp, SolutionCallback callback);
  lmpc::utils::MPCSolution get_solution(const size_t & timestamp);
  bool is_solution_initialized() const;

protected:
  size_t num_cycle_to_switch_;
  std::vector<MultiMPCInterface::SharedPtr> mpcs_;
  std::vector<size_t> mpc_cycle_count_;
  std::vector<std::future<MultiMPCSolution>> mpc_futures_;
  std::atomic_ulong current_mpc_idx_;
  lmpc::utils::MPCSolutionBuffer buffer_;

  MultiMPCSolution solve_mpc_thread(
    const size_t mpc_idx, const casadi::DMDict in,
    const size_t timestamp, SolutionCallback callback);
  bool is_mpc_ready(const size_t & mpc_idx);
};
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc
#endif  // RACING_MPC__MULTI_MPC_MANAGER_HPP_
