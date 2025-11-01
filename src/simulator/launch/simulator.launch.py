from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

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
            ])
        }.items()
    )
    
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_m3_gtr',
        arguments=[
            '-entity', 'm3_gtr',
            '-file', PathJoinSubstitution([
                FindPackageShare('simulator'),
                'models/m3_gtr/model.sdf'
            ]),
            '-x', '0.40646722',
            '-y', '6.560996',
            '-z', '0.0309',
            '-Y', '-1.57'
        ],
        output='screen'
    )
    
    delayed_spawn = TimerAction(
        period=5.0,  # seconds
        actions=[spawn_entity]
    )

    return LaunchDescription([
        gazebo_launch,
        delayed_spawn
    ])