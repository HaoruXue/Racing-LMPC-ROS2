# Copyright 2024 David Qiao
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
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from lmpc_utils.lmpc_launch_utils import get_share_file, get_sim_time_launch_arg


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
    lqr_config = get_share_file(
        "racing_lmpc_launch", "param", "racing_lqr", "LVMS_tracking_lqr.param.yaml"
    )
    sim_track_file = get_share_file("racing_trajectory", "test_data", "LVMS", "20_LVMS_optm.txt")
    track_file_folder = get_share_file("racing_trajectory", "test_data", "LVMS")

    return LaunchDescription(
        [
            declare_use_sim_time_cmd,
            Node(
                package="racing_simulator",
                executable="racing_simulator_node_exe",
                name="continuous_racing_simulator_node",
                output="screen",
                parameters=[
                    sim_config,
                    dt_model_config,
                    base_model_config,
                    use_sim_time,
                    {
                        "racing_simulator.race_track_file_path": sim_track_file,
                        "modeling.use_frenet": False,
                        # "racing_simulator.x0": [-100.0, -5.0, 3.14, 15.0, 0.0, 0.0]
                        # "racing_simulator.x0": [50.0, 5.0, 3.14, 15.0, 0.0, 0.0]
                        # "racing_simulator.x0": [-10.0, 2.0, 3.14, 15.0, 0.0, 0.0]
                        "racing_simulator.x0": [-1.0, 2.0, 3.8, 15.0, 0.0, 0.0]
                        # "racing_simulator.x0": [-350.0, -20.0, 3.14, 15.0, 0.0, 0.0]
                        # "racing_simulator.x0": [-67.9, 247.6, -2.61799, 15.0, 0.0, 0.0]
                    },
                ],
                remappings=[
                    ("abscissa_polygon", "/simulation/abscissa_polygon"),
                    ("left_boundary_polygon", "/simulation/left_boundary_polygon"),
                    ("right_boundary_polygon", "/simulation/right_boundary_polygon"),
                ],
                emulate_tty=True,
            ),
            Node(
                package="racing_lqr",
                executable="racing_lqr_node_exe",  # TODO: find out what is this
                name="racing_lqr_node",
                output="screen",
                parameters=[  # TODO: find out how to change this
                    lqr_config,
                    dt_model_config,
                    base_model_config,
                    use_sim_time,
                    {
                        "racing_lqr_node.vehicle_model_name": "single_track_planar_model",
                        "racing_lqr_node.default_traj_idx": 20,
                        "racing_lqr_node.traj_folder": track_file_folder,
                        "racing_lqr_node.velocity_profile_scale": 0.6,
                        "racing_lqr_node.delay_step": 0,
                    },
                ],
                remappings=[],
                # prefix=['taskset -c 22,23'],
                emulate_tty=True,
            ),
        ]
    )
