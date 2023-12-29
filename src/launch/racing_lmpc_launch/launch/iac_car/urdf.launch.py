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
from lmpc_utils.lmpc_launch_utils import get_share_file, get_sim_time_launch_arg
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    urdf_file_name = get_share_file("racing_lmpc_launch", "config", "urdf", "iac_car", "av21.urdf")

    vehicle_name_arg = DeclareLaunchArgument(
        name="urdf_path",
        default_value=urdf_file_name,
        description="Path to the vehicle urdf file",
    )

    robot_description = ParameterValue(
        Command(["xacro ", LaunchConfiguration("urdf_path")]), value_type=str
    )

    declare_use_sim_time_cmd, use_sim_time = get_sim_time_launch_arg()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            use_sim_time,
            {
                "robot_description": robot_description,
            },
        ],
    )

    vehicle_state_visualizer_node = Node(
        package="vehicle_state_visualizer",
        executable="vehicle_state_visualizer_node_exe",
        name="vehicle_state_visualizer_node",
        output="screen",
        parameters=[
            use_sim_time,
            {
                "fl_joint": "fl_tyre_joint",
                "fr_joint": "fr_tyre_joint",
            },
        ],
    )

    return LaunchDescription(
        [
            vehicle_name_arg,
            declare_use_sim_time_cmd,
            robot_state_publisher_node,
            vehicle_state_visualizer_node,
        ]
    )
