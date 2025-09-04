cmake_minimum_required(VERSION 3.5)
project(bmw_gtr)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# === Find dependencies ===
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_dev REQUIRED)

# === Include directories ===
include_directories(
  plugins/include
)

# === Build the plugin ===
add_library(car_plugin SHARED
  plugins/src/carlikerobot.cpp
  plugins/src/carlikerobot_ros_plugin.cpp
)

target_include_directories(car_plugin PUBLIC
  plugins/include
)

ament_target_dependencies(car_plugin
  rclcpp
  std_msgs
  geometry_msgs
  gazebo_ros
  gazebo_dev
)

# === Install the plugin library ===
install(TARGETS car_plugin
  LIBRARY DESTINATION lib
)

# === Install other assets ===
install(
  DIRECTORY launch worlds models plugins config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
