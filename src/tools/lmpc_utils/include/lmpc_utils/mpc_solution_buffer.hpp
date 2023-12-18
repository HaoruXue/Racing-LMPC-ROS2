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

#ifndef LMPC_UTILS__MPC_SOLUTION_BUFFER_HPP_
#define LMPC_UTILS__MPC_SOLUTION_BUFFER_HPP_

#include <mutex>
#include <shared_mutex>
#include <memory>
#include <vector>

#include <casadi/casadi.hpp>

namespace lmpc
{
namespace utils
{
enum class MPCSolutionStatus
{
  OK,
  TOO_EARLY,
  TOO_LATE,
};

struct MPCSolution
{
  casadi::DM x;
  casadi::DM u;
  size_t timestamp = 0;
  MPCSolutionStatus status = MPCSolutionStatus::OK;
};

class MPCSolutionBuffer
{
public:
  typedef std::shared_ptr<MPCSolutionBuffer> SharedPtr;
  typedef std::unique_ptr<MPCSolutionBuffer> UniquePtr;

/**
 * @brief Create a MPC Solution Buffer of a given size
 *
 * @param buffer_size the size of the buffer horizon
 */
  MPCSolutionBuffer();

/**
 * @brief Store a sequence of MPC solutions into the buffer.
 *
 * @param x nx by horizon, the state trajectory
 * @param u nu by horizon - 1, the inputs
 * @param start_timestamp the timestamp of the first state in the trajectory
 */
  void set_mpc_solution(const casadi::DM & x, const casadi::DM & u, const size_t & start_timestamp);

/**
 * @brief Get the MPC solution at a given timestamp.
 * If the solution falls outside the buffer timestamp range,
 * the returned solution will have status TOO_EARLY or TOO_LATE,
 * and the first or last solution in the buffer will be returned accordingly.
 *
 * @param timestamp
 * @return MPCSolution
 */
  MPCSolution get_mpc_solution(const size_t & timestamp);

protected:
  MPCSolution solution_;
  std::shared_mutex mutex_;
  bool initialized_;
};

}  // namespace utils
}  // namespace lmpc
#endif  // LMPC_UTILS__MPC_SOLUTION_BUFFER_HPP_
