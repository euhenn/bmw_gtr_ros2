cmake_minimum_required(VERSION 3.8)
project(simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(msgs REQUIRED)

include_directories(
  plugins/include
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)

# === Plugins ===
add_library(carlikerobot_ros_plugin SHARED
  plugins/src/carlikerobot_ros_plugin.cpp
  plugins/src/carlikerobot.cpp
)
ament_target_dependencies(carlikerobot_ros_plugin
  rclcpp std_msgs geometry_msgs gazebo_ros gazebo_dev msgs
)

add_library(bno055_plugin SHARED
  plugins/src/bno055_plugin.cpp
)
ament_target_dependencies(bno055_plugin
  rclcpp std_msgs geometry_msgs gazebo_ros gazebo_dev msgs
)

add_library(gps_plugin SHARED
  plugins/src/gps_plugin.cpp
)
ament_target_dependencies(gps_plugin
  rclcpp std_msgs geometry_msgs gazebo_ros gazebo_dev msgs
)

# === Install ===
install(TARGETS
  carlikerobot_ros_plugin
  bno055_plugin
  gps_plugin
  LIBRARY DESTINATION lib
)

install(DIRECTORY
  launch worlds models
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
