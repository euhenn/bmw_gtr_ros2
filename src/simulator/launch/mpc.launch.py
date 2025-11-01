from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():


    # Delete entity via script directly
    delete_entity = ExecuteProcess(
        cmd=["python3", "/ros2_ws/src/simulator/scripts/delete_entity_node.py"],
        output="screen"
    )


    # Respawn entity
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_m3_gtr",
        arguments=[
            "-entity", "m3_gtr",
            "-file", PathJoinSubstitution([
                FindPackageShare("simulator"),
                "models/m3_gtr/model.sdf"
            ]),
            "-x", "0.40646722", #"17.4925", 
            "-y", "5.50430996", #"13.3131",
            "-z", "0.0309",
            "-Y", "-1.57"
        ],
        output="screen"
    )

    # Run controller script
    run_controller_script = ExecuteProcess(
        cmd=["python3", "/ros2_ws/src/brain/brain/main_brain.py"],
        output="screen"
    )

    delayed_delete = TimerAction(period=1.0, actions=[delete_entity])
    delayed_spawn = TimerAction(period=4.0, actions=[spawn_entity])
    delayed_script = TimerAction(period=6.0, actions=[run_controller_script])

    return LaunchDescription([
        delayed_delete,
        delayed_spawn,
        delayed_script
    ])
