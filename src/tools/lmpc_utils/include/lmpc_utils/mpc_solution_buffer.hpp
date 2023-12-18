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
  explicit MPCSolutionBuffer(const size_t & buffer_size);

/**
 * @brief Set the current timestamp.
 * The unit of timestamp is 1 MPC cycle.
 * The sliding window of solutions stored will be shifted accordingly.
 * All solutions with timestamp < current_timestamp will be discarded.
 * All solutions with timestamp >= current_timestamp + buffer_size will be discarded.
 * Must be called before calling add_mpc_solution.
 * Must be called at every MPC cycle.
 *
 * @param timestamp
 */
  void set_current_timestamp(const size_t & timestamp);

/**
 * @brief Get the current timestamp.
 *
 * @return const size_t&
 */
  const size_t & get_current_timestamp() const;

/**
 * @brief Store a sequence of MPC solutions into the buffer.
 * The solutions will be stored at the starting timestamp.
 * The horizon of the solution does not need to match the buffer size.
 * horizon - 1 inputs and states will be stored.
 *
 * @param x nx by horizon (or nx by horizon - 1), the state trajectory
 * @param u nu by horizon - 1, the inputs
 * @param start_timestamp the timestamp of the first state in the trajectory
 */
  void add_mpc_solution(const casadi::DM & x, const casadi::DM & u, const size_t & start_timestamp);

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
  typedef std::vector<MPCSolution> MPCSolutionCircularBuffer;

  MPCSolutionCircularBuffer::iterator get_iterator(const int64_t & delta);

  MPCSolutionCircularBuffer buffer_;
  MPCSolutionCircularBuffer::iterator current_solution_;
  size_t current_timestamp_;
  std::shared_mutex mutex_;
  bool initialized_;
};

}  // namespace utils
}  // namespace lmpc
#endif  // LMPC_UTILS__MPC_SOLUTION_BUFFER_HPP_
