from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_automobile',
            arguments=[
                '-entity', 'automobile',
                '-file', '/ros2_ws/src/bmw_gtr/models/rcCar_assembly/model.sdf',
                '-x', '13.0',
                '-y', '2.0',
                '-z', '0.03'
            ],
            output='screen'
        )
    ])
