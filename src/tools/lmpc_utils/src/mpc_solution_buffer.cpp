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
MPCSolutionBuffer::MPCSolutionBuffer(const size_t & buffer_size)
: buffer_(buffer_size),
  current_solution_(buffer_.begin()),
  current_timestamp_(0),
  initialized_(false)
{
  if (buffer_size == 0) {
    throw std::invalid_argument("Buffer size cannot be zero.");
  }
}

void MPCSolutionBuffer::set_current_timestamp(const size_t & timestamp)
{
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (!initialized_) {
    current_timestamp_ = timestamp;
    initialized_ = true;
    return;
  }

  const auto delta = timestamp - current_timestamp_;
  if (delta > buffer_.size()) {
    current_timestamp_ = timestamp;
    return;
  }

  // Move the current solution iterator
  current_solution_ = get_iterator(delta);
  current_timestamp_ = timestamp;
}

void MPCSolutionBuffer::add_mpc_solution(
  const casadi::DM & x, const casadi::DM & u,
  const size_t & start_timestamp)
{
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (!initialized_) {
    throw std::runtime_error("Must call set_current_timestamp before calling add_mpc_solution.");
  }

  const auto sol_length = std::min(static_cast<size_t>(u.size2()), buffer_.size());

  // Check if any part of the solution will fall within the buffer
  if (start_timestamp + sol_length <= current_timestamp_ ||
    start_timestamp >= current_timestamp_ + buffer_.size())
  {
    return;
  }

  // Copy the in-range part of solutions into the buffer. Mind the wrap-around.
  const auto delta = static_cast<int64_t>(start_timestamp) -
    static_cast<int64_t>(current_timestamp_);
  const auto start_index = std::max<size_t>(0, -delta);
  const auto end_index =
    std::min<size_t>(
    static_cast<int64_t>(sol_length),
    static_cast<int64_t>(buffer_.size()) - delta);
  auto iter = get_iterator(delta + start_index);
  for (size_t i = start_index; i < end_index; ++i) {
    iter->x = x(casadi::Slice(), i);
    iter->u = u(casadi::Slice(), i);
    iter->status = MPCSolutionStatus::OK;
    iter->timestamp = start_timestamp + i;
    ++iter;
    if (iter == buffer_.end()) {
      iter = buffer_.begin();
    }
  }
}

MPCSolution MPCSolutionBuffer::get_mpc_solution(const size_t & timestamp)
{
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (!initialized_) {
    throw std::runtime_error("Must call set_current_timestamp before calling get_mpc_solution.");
  }

  const auto delta = timestamp - current_timestamp_;
  MPCSolution solution;
  if (timestamp < current_timestamp_) {
    solution = *current_solution_;
    solution.status = MPCSolutionStatus::TOO_EARLY;
  } else if (timestamp >= current_timestamp_ + buffer_.size()) {
    solution = *get_iterator(buffer_.size() - 1);
    solution.status = MPCSolutionStatus::TOO_LATE;
  } else {
    solution = *get_iterator(delta);
  }
  return solution;
}

MPCSolutionBuffer::MPCSolutionCircularBuffer::iterator MPCSolutionBuffer::get_iterator(
  const int64_t & delta)
{
  if (delta == 0) {
    return current_solution_;
  } else if (delta > 0) {
    const auto dist_to_end = std::distance(current_solution_, buffer_.end());
    if (delta <= dist_to_end) {
      return current_solution_ + delta;
    } else {
      return buffer_.begin() + (delta - dist_to_end) % buffer_.size();
    }
  } else {
    const auto dist_to_begin = std::distance(buffer_.begin(), current_solution_);
    if (-delta <= dist_to_begin) {
      return current_solution_ + delta;
    } else {
      return buffer_.end() - (-delta - dist_to_begin) % buffer_.size();
    }
  }
}
}  // namespace utils
}  // namespace lmpc
