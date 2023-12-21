std_msgs/Header header

# current trajectory index
int32 trajectory_index 0

# solved?
bool solved false

# total cost
float64 cost 0.0

# trajectory tracking cost
float64 cost_trajectory 0.0

# state predictions
float64[] state

# control predictions
float64[] control

# solve time
float64 solve_time 0.0

# reference trajectory
float64[] reference_trajectory

# reference input
float64[] reference_input

# bank angle list
float64[] banks

# curvature list
float64[] curvatures

# Trajectory Command
lmpc_msgs/TrajectoryCommand trajectory_cmd