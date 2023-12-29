cmake_minimum_required(VERSION 3.8)
project(vehicle_state_visualizer)

# Default to C++17.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set ROS_DISTRO macros
if(NOT DEFINED ENV{ROS_DISTRO})
    message(FATAL_ERROR "Environment variable ROS_DISTRO is not defined. Have you sourced your ROS workspace?")
endif()
set(ROS_DISTRO $ENV{ROS_DISTRO})
if(${ROS_DISTRO} STREQUAL "rolling")
  add_compile_definitions(ROS_DISTRO_ROLLING)
elseif(${ROS_DISTRO} STREQUAL "galactic")
  add_compile_definitions(ROS_DISTRO_GALACTIC)
elseif(${ROS_DISTRO} STREQUAL "humble")
  add_compile_definitions(ROS_DISTRO_HUMBLE)
endif()

# Require that dependencies from package.xml be available.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies(REQUIRED
  ${${PROJECT_NAME}_BUILD_DEPENDS}
  ${${PROJECT_NAME}_BUILDTOOL_DEPENDS}
)

set(${PROJECT_NAME}_SRC
  src/vehicle_state_visualizer_node.cpp
)

set(${PROJECT_NAME}_HEADER
  include/vehicle_state_visualizer/vehicle_state_visualizer_node.hpp
)

ament_auto_add_library(${PROJECT_NAME} SHARED
  ${${PROJECT_NAME}_SRC}
  ${${PROJECT_NAME}_HEADER}
)

ament_auto_add_executable(${PROJECT_NAME}_node_exe
  src/vehicle_state_visualizer_node.cpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
#   set(TEST_SOURCES test/test_vehicle_state_visualizer.cpp)
#   set(TEST_EXE test_vehicle_state_visualizer)
#   ament_add_gtest(${TEST_EXE} ${TEST_SOURCES})
#   target_link_libraries(${TEST_EXE} ${PROJECT_NAME})
endif()

# Create & install ament package.
ament_auto_package()
