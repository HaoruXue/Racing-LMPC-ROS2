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

#include "lmpc_utils/mpc_solution_buffer.hpp"

namespace lmpc
{
namespace utils
{
MPCSolutionBuffer::MPCSolutionBuffer()
: initialized_(false)
{
}

void MPCSolutionBuffer::set_mpc_solution(
  const casadi::DM & x, const casadi::DM & u,
  const size_t & start_timestamp)
{
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (!initialized_) {
    throw std::runtime_error("Must call set_current_timestamp before calling set_mpc_solution.");
  }

  if (x.size2() < u.size2()) {
    throw std::runtime_error("x must have at least as many columns as u.");
  }

  solution_.x = x;
  solution_.u = u;
  solution_.timestamp = start_timestamp;
}

MPCSolution MPCSolutionBuffer::get_mpc_solution(const size_t & timestamp)
{
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (!initialized_) {
    throw std::runtime_error("Must call set_current_timestamp before calling get_mpc_solution.");
  }

  const auto delta = timestamp - solution_.timestamp;
  const auto sol_idx = std::clamp<casadi_int>(delta, 0ul, solution_.u.size2() - 1);
  MPCSolution solution;
  solution.x = solution_.x(casadi::Slice(), sol_idx);
  solution.u = solution_.u(casadi::Slice(), sol_idx);
  solution.timestamp = solution_.timestamp + sol_idx;
  solution.status = MPCSolutionStatus::OK;

  if (timestamp < solution_.timestamp) {
    solution.status = MPCSolutionStatus::TOO_EARLY;
  } else if (timestamp >= solution_.timestamp + solution_.u.size2()) {
    solution.status = MPCSolutionStatus::TOO_LATE;
  }
  return solution;
}

bool MPCSolutionBuffer::is_initialized() const
{
  return initialized_;
}
}  // namespace utils
}  // namespace lmpc
