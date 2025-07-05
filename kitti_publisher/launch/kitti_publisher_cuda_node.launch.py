import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    depthai_node = Node(
        package='kitti_publisher',
        executable='kitti_publisher_cuda_node',
        name='kitti_publisher_cuda_node',
        output='screen',
        parameters=[{'kitti_path': '/datasets/odometry/data_odometry_color/dataset/sequences/04'}]
    )

    return LaunchDescription([
        depthai_node,
    ])
