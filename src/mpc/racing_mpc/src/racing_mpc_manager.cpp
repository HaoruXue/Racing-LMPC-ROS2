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

#include "racing_mpc/racing_mpc_manager.hpp"

namespace lmpc
{
namespace mpc
{
namespace racing_mpc
{
RacingMPCManager::RacingMPCManager(const RacingMPCConfig & config, const BaseVehicleModel & model, const bool & full_dynamics)
{
    const auto & num_mpc = config.num_mpc;
    if (num_mpc == 0)
    {
        throw std::invalid_argument("Number of MPCs must be greater than 0.");
    }
    for (size_t i = 0; i < num_mpc; i++)
    {
        mpc_list_.push_back(std::make_shared<RacingMPC>(config, model, full_dynamics));
        mpc_future_list_.push_back(std::future<MPCSolution>());
        mpc_solution_list_.push_back(MPCSolution());
    }
    mpc_authority_ = 0;
}

MPCSolution RacingMPCManager::initialize(const casadi::DMDict & in)
{
    for (size_t i = 0; i < mpc_list_.size(); i++)
    {
        mpc_future_list_[i] = std::async(std::launch::async, &RacingMPCManager::mpc_callback, this, in, i);
    }
    // wait for all MPCs to finish initialization
    for (size_t i = 0; i < mpc_list_.size(); i++)
    {
        mpc_future_list_[i].wait();
    }
    // return the solution from the first MPC
    mpc_solution_list_[mpc_authority_] = mpc_future_list_[mpc_authority_].get();
    mpc_future_list_[mpc_authority_] = std::async(std::launch::async, &RacingMPCManager::mpc_callback, this, in, mpc_authority_);
    return mpc_solution_list_[mpc_authority_];
}

MPCSolution RacingMPCManager::step(const casadi::DMDict & in)
{
    // advance the solution age of all MPCs
    for (size_t i = 0; i < mpc_list_.size(); i++)
    {
        mpc_solution_list_[i].solution_age++;
    }

    // check if the current MPC is ready
    if (mpc_future_list_[mpc_authority_].valid() && mpc_future_list_[mpc_authority_].wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
    {
        mpc_solution_list_[mpc_authority_] = mpc_future_list_[mpc_authority_].get();
        mpc_future_list_[mpc_authority_] = std::async(std::launch::async, &RacingMPCManager::mpc_callback, this, in, mpc_authority_);
    }
    else
    {
        // if the current MPC is not ready, advance to the next available MPC
        auto next_mpc_index = mpc_authority_;
        for (size_t i = 0; i < mpc_list_.size(); i++)
        {
            next_mpc_index = next_mpc(next_mpc_index);
            if (mpc_future_list_[next_mpc_index].valid() && mpc_future_list_[next_mpc_index].wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                mpc_authority_ = next_mpc_index;
                mpc_future_list_[mpc_authority_] = std::async(std::launch::async, &RacingMPCManager::mpc_callback, this, in, mpc_authority_);
                break;
            }
        }
    }

    // return the most recent solution (the one with the smallest solution age)
    size_t min_age = std::numeric_limits<size_t>::max();
    size_t min_age_index = 0;
    for (size_t i = 0; i < mpc_list_.size(); i++)
    {
        if (mpc_solution_list_[i].solution_age < min_age)
        {
            min_age = mpc_solution_list_[i].solution_age;
            min_age_index = i;
        }
    }
    return mpc_solution_list_[min_age_index];
}

size_t RacingMPCManager::get_mpc_authority() const
{
    return mpc_authority_;
}

MPCSolution RacingMPCManager::mpc_callback(const casadi::DMDict & in, const size_t & mpc_index)
{
    MPCSolution solution;
    mpc_list_[mpc_index]->solve(in, solution.solution, solution.stats);
    return solution;
}

size_t RacingMPCManager::next_mpc(const size_t & mpc_index)
{
    return (mpc_index + 1) % mpc_list_.size();
}
}  // namespace racing_mpc
}  // namespace mpc
}  // namespace lmpc