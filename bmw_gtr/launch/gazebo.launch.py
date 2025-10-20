from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    create_gui_ini = ExecuteProcess(
        cmd=[
            'bash', '-c', '''
            mkdir -p ~/.gazebo
            rm -f ~/.gazebo/gui.ini
            cat <<EOF > ~/.gazebo/gui.ini
[geometry]
x=77
y=557
width=890
height=481
EOF
            '''
        ],
        shell=True
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
                FindPackageShare('bmw_gtr'),
                'worlds',
                'bfmc_track.world'
            ])
        }.items()
    )

    return LaunchDescription([
        create_gui_ini,
        gazebo_launch
    ])