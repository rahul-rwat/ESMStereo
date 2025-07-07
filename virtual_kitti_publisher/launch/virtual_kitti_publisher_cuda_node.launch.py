import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # morning rain fog overcast sunset

    depthai_node = Node(
        package='virtual_kitti_publisher',
        executable='virtual_kitti_publisher_cuda_node',
        name='virtual_kitti_publisher_cuda_node',
        output='screen',
        parameters=[{'kitti_path': './../vkitti/vkitti/Scene02/rain/frames/rgb/',
                     'depth_kitti_path': './../vkitti/depth/Scene02/rain/frames/depth/'}]

    )

    return LaunchDescription([
        depthai_node,
    ])
