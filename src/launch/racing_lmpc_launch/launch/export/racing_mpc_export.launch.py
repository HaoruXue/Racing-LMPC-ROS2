# Copyright 2023 Haoru Xue
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from lmpc_utils.lmpc_launch_utils import get_share_file, get_sim_time_launch_arg
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    declare_use_sim_time_cmd, use_sim_time = get_sim_time_launch_arg()
    sim_config = get_share_file(
        "racing_lmpc_launch", "param", "racing_simulator", "continuous_simulator.param.yaml"
    )
    dt_model_config = get_share_file(
        "racing_lmpc_launch", "param", "iac_car", "iac_car_single_track.param.yaml"
    )
    base_model_config = get_share_file(
        "racing_lmpc_launch", "param", "LVMS", "LVMS_base.param.yaml"
    )
    mpc_config = get_share_file(
        "racing_lmpc_launch", "param", "racing_mpc", "LVMS_tracking_mpc.param.yaml"
    )
    sim_track_file = get_share_file("racing_trajectory", "test_data", "LVMS", "20_LVMS_optm.txt")
    track_file_folder = get_share_file("racing_trajectory", "test_data", "LVMS")

    vd_model_name = DeclareLaunchArgument(
        "vehicle_model_name",
        default_value="single_track_planar_model",
        description="vehicle model name",
    )

    return LaunchDescription(
        [
            declare_use_sim_time_cmd,
            vd_model_name,
            Node(
                package="racing_mpc",
                executable="racing_mpc_export_exe",
                name="racing_mpc_solver_node",
                output="screen",
                parameters=[
                    mpc_config,
                    dt_model_config,
                    base_model_config,
                    use_sim_time,
                    {
                        "racing_mpc_node.vehicle_model_name": LaunchConfiguration(
                            "vehicle_model_name"
                        ),
                    },
                ],
                remappings=[
                    ("solve_mpc", "mpc_0/solve_mpc"),
                ],
                # prefix=['taskset -c 22,23'],
                emulate_tty=True,
            ),
        ]
    )