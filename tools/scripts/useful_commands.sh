ros2 topic pub /vehicle_0/lmpc_trajectory_command lmpc_msgs/msg/TrajectoryCommand \
"{trajectory_index: 10, speed_limit: 100.0, velocity_profile_scale: 0.9, behavior_stratergy: {behavior_stratergy: 128, follow_distance: 15.0}}"

ros2 topic pub /vehicle_1/lmpc_trajectory_command lmpc_msgs/msg/TrajectoryCommand \
"{trajectory_index: 5, speed_limit: 100.0, velocity_profile_scale: 0.9, behavior_stratergy: {behavior_stratergy: 0, follow_distance: 0.0}}"

ros2 topic pub /lmpc_trajectory_command lmpc_msgs/msg/TrajectoryCommand \
"{trajectory_index: 5, speed_limit: 20.0, velocity_profile_scale: 0.9, behavior_stratergy: {behavior_stratergy: 0, follow_distance: 0.0}}"