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
find_package(rosidl_default_generators REQUIRED)

# === Include directories ===
include_directories(plugins/include)

# === Generate custom messages ===
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/IMU.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# === Build plugins ===
add_library(carlikerobot_ros_plugin SHARED plugins/src/carlikerobot_ros_plugin.cpp)
ament_target_dependencies(carlikerobot_ros_plugin
  rclcpp std_msgs geometry_msgs gazebo_ros gazebo_dev
)

add_library(bno055_plugin SHARED plugins/src/bno055_plugin.cpp)
ament_target_dependencies(bno055_plugin
  rclcpp std_msgs geometry_msgs gazebo_ros gazebo_dev
)

# Generated message headers
target_include_directories(carlikerobot_ros_plugin PUBLIC
  plugins/include
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)
target_include_directories(bno055_plugin PUBLIC
  plugins/include
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)

# === Install message files ===
install(DIRECTORY msg
  DESTINATION share/${PROJECT_NAME}
)

# === Install plugin libraries ===
install(TARGETS carlikerobot_ros_plugin bno055_plugin
  LIBRARY DESTINATION lib
)

# === Install other assets ===
install(DIRECTORY launch worlds models plugins config
  DESTINATION share/${PROJECT_NAME}
)

# Export message runtime so other packages can depend on it
ament_export_dependencies(rosidl_default_runtime)

ament_package()
