# Assign Exclusive CPU Cores to a Process in Linux

This guide will help your MPC application run on specific CPU cores. This is useful for real-time applications where you want to ensure that a process is not interrupted by other processes.

## Before You Begin

This guide assumes that grub is your bootloader. If you have a dual-boot system with two Linux distros, make sure you are editing the grub file for the distro that is selected in BIOS as the boot option.

## Procedure

Main tools involved: `isolcpus`, `taskset`.

Source: <https://unix.stackexchange.com/questions/326579/how-to-ensure-exclusive-cpu-availability-for-a-running-process>

### Step 1: Take some cores off of kernel scheduling

This step prevents the kernel from scheduling other tasks on these dedicated cores.

Determine how many cores you want to dedicate to your process. Most ROS2 nodes are single-threaded, so you only need to dedicate one core. For MPC with OSQP's default linear solver, it is also single-threaded.

Don't go overboard with this. The MPC node and MPC Solver node are the only ones that need to be dedicated to a core. Other nodes can share cores.

Add the following to `GRUB_CMDLINE_LINUX` in `/etc/default/grub`:

```bash
# Block core #8 (zero-indexed)
GRUB_CMDLINE_LINUX="isolcpus=7"

# Alternatively, block cores #7 and #8
GRUB_CMDLINE_LINUX="isolcpus=6,7"

# Alternatively, block cores #5 to #8
GRUB_CMDLINE_LINUX="isolcpus=4-7"
```

Then update grub in terminal and reboot.

```bash
sudo update-grub
sudo reboot now
```

### Step 2: Launch a process on specific cores

For a regular task, run like this

```bash
# Run gedit on cores #7 and #8
taskset -c 6,7 gedit
```

To configure `taskset` in a ROS2 launch file, do the following:

```python
from launch import LaunchDescription
from launch import LaunchService

import launch_ros.actions


def generate_launch_description():
    """Run demo nodes with cores #5 to #8."""
    ld = LaunchDescription([
        launch_ros.actions.Node(
            package='demo_nodes_cpp', node_executable='talker', output='screen',
            remappings=[('chatter', 'my_chatter')],
            prefix=['taskset -c 4-7'], # prefix launch with taskset
        )
    ])
    return ld
```

More on ROS2 launch-prefix: <https://answers.ros.org/question/202712/how-to-bind-a-node-to-a-specific-cpu-core/>

More on taskset: <https://linux.die.net/man/1/taskset>
