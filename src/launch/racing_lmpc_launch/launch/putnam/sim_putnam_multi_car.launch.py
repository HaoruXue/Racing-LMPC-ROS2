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
        "racing_lmpc_launch", "param", "iac_car", "iac_car_base.param.yaml"
    )
    mpc_config = get_share_file(
        "racing_lmpc_launch", "param", "racing_mpc", "iac_car_tracking_mpc.param.yaml"
    )
    sim_track_file = get_share_file(
        "racing_trajectory", "test_data", "putnam", "10_putnam_optm.txt"
    )
    track_file_folder = get_share_file("racing_trajectory", "test_data", "putnam")

    vd_model_name = DeclareLaunchArgument(
        "vehicle_model_name",
        default_value="single_track_planar_model",
        description="vehicle model name",
    )

    num_vehicles = 2
    x0s = [
        [-10.0, 2.0, 3.14, 15.0, 0.0, 0.0],
        [-30.0, 2.0, 3.14, 15.0, 0.0, 0.0],
    ]

    launch_descriptions = [declare_use_sim_time_cmd, vd_model_name]

    for i in range(num_vehicles):
        urdf_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                get_share_file("racing_lmpc_launch", "launch", "iac_car", "urdf.launch.py")
            ),
            launch_arguments={
                "namespace": "vehicle_{}".format(i),
            }.items(),
            # condition=IfCondition(LaunchConfiguration("vehicle_model_name") == "iac_car"),
        )
        sim_node = Node(
            package="racing_simulator",
            executable="racing_simulator_node_exe",
            name="continuous_racing_simulator_node",
            namespace="vehicle_{}".format(i),
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
                    # "racing_simulator.x0": [-350.0, -20.0, 3.14, 15.0, 0.0, 0.0]
                    # "racing_simulator.x0": [-67.9, 247.6, -2.61799, 15.0, 0.0, 0.0]
                    "racing_simulator.x0": x0s[i],
                },
            ],
            remappings=[
                ("abscissa_polygon", "/simulation/abscissa_polygon"),
                ("left_boundary_polygon", "/simulation/left_boundary_polygon"),
                ("right_boundary_polygon", "/simulation/right_boundary_polygon"),
            ],
            emulate_tty=True,
        )
        mpc_node = Node(
            package="racing_mpc",
            executable="racing_mpc_node_exe",
            name="racing_mpc_node",
            namespace="vehicle_{}".format(i),
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
                    "racing_mpc_node.default_traj_idx": 10,
                    "racing_mpc_node.traj_folder": track_file_folder,
                    "racing_mpc_node.velocity_profile_scale": 0.9,
                    "racing_mpc_node.delay_step": 0,
                },
            ],
            remappings=[("oppo_prediction", f"/vehicle_{(i+1)%num_vehicles}/ego_prediction")],
            # prefix=['taskset -c 22,23'],
            emulate_tty=True,
        )
        mpc_solver_node = Node(
            package="racing_mpc",
            executable="racing_mpc_solver_node_exe",
            name="racing_mpc_solver_node",
            namespace="vehicle_{}".format(i),
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
        )

        launch_descriptions.append(urdf_launch)
        launch_descriptions.append(sim_node)
        launch_descriptions.append(mpc_node)
        launch_descriptions.append(mpc_solver_node)

    return LaunchDescription(launch_descriptions)