from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    # Use gazebo_ros launch file (starts server only because gui=false)
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
            'gui': 'false'  # This will run without the client
        }.items()
    )

    # Spawn the car after Gazebo starts
    spawn_entity = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-entity', 'm3_gtr',
            '-file', '/ros2_ws/src/simulator/models/m3_gtr/model.sdf',
            '-x', '0.40646722',
            '-y', '6.560996',
            '-z', '0.0309',
            '-Y', '-1.57'
        ],
        output='screen'
    )

    spawn_fixed_camera = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-entity', 'fixed_camera',
            '-file', '/ros2_ws/src/simulator/models/fixed_camera/model.sdf'
        ],
        output='screen'
    )

    # Run camera dashboard
    camera_dashboard = ExecuteProcess(
        cmd=['python3', '/ros2_ws/src/bmw_gtr/scripts/camera_dashboard.py'],
        output='screen'
    )

    delayed_spawn = TimerAction(period=5.0, actions=[spawn_entity])
    delayed_fixed_camera = TimerAction(period=6.0, actions=[spawn_fixed_camera])
    delayed_dashboard = TimerAction(period=10.0, actions=[camera_dashboard])

    return LaunchDescription([
        gazebo_launch,
        delayed_spawn,
        delayed_fixed_camera,
        delayed_dashboard
    ])