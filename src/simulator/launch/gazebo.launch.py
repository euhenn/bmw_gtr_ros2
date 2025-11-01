from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():

    # Pre-launch cleanup
    pre_cleanup = ExecuteProcess(
        cmd=['bash', '-c', 'pkill -9 gzserver; pkill -9 gzclient; pkill -9 gazebo; rm -rf /tmp/gazebo* ~/.gazebo/log/*'],
        name='pre_cleanup',
        output='screen'
    )

    # Fixed GUI ini creation - using proper bash syntax
    create_gui_ini = ExecuteProcess(
        cmd=[
            'bash', '-c', 
            '''
            mkdir -p ~/.gazebo
            cat > ~/.gazebo/gui.ini << "EOF"
[geometry]
x=77
y=557
width=890
height=481
EOF
            '''
        ],
        name='create_gui_ini',
        output='screen'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('simulator'),
                'worlds',
                'bfmc_track.world'
            ]),
            'verbose': 'false',  # Add verbose for debugging
            'gdb': 'false'      # Disable gdb for now
        }.items()
    )


    return LaunchDescription([
        pre_cleanup,
        TimerAction(period=2.0, actions=[create_gui_ini]),  # Wait after cleanup
        TimerAction(period=3.0, actions=[gazebo_launch]),   # Wait before starting gazebo
    ])