#!/bin/bash

# Check if ROS_DISTRO is set
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS_DISTRO is not set. Please source ROS2 first."
    exit 1
fi

# Check if ROS_DISTRO is either 'Humble' or 'Iron'
if [ "$ROS_DISTRO" == "humble" ] || [ "$ROS_DISTRO" == "iron" ]; then
    echo "Found ROS_DISTRO of '$ROS_DISTRO'."
else
    echo "Warning: This Repo was not tested with ROS_DISTRO of '$ROS_DISTRO'. Humble or iron are recommended."
fi

ROS_DISTRO_SOURCE=/opt/ros/${ROS_DISTRO}/setup.bash
if [ -f "${ROS_DISTRO_SOURCE}" ]; then
    source ${ROS_DISTRO_SOURCE}
    echo "Sourced ${ROS_DISTRO_SOURCE}"
fi

REPO_LOCAL=install/setup.bash;
if [ -f "${REPO_LOCAL}" ]; then
    source ${REPO_LOCAL};
    echo "Sourced ${REPO_LOCAL}";
fi;