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

#ifndef BASE_MPC__BASE_MPC_HPP_
#define BASE_MPC__BASE_MPC_HPP_

#include <memory>
#include <casadi/casadi.hpp>

namespace lmpc
{
namespace mpc
{
class BaseMPC
{
public:
  typedef std::shared_ptr<BaseMPC> SharedPtr;
  typedef std::unique_ptr<BaseMPC> UniquePtr;

  BaseMPC() = default;
  virtual ~BaseMPC() = default;
  virtual bool init() = 0;
  virtual void solve(const casadi::DMDict & in, casadi::DMDict & out, casadi::Dict & stats) = 0;
};
}  // namespace mpc
}  // namespace lmpc
#endif  // BASE_MPC__BASE_MPC_HPP_
