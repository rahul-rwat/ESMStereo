import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # morning rain fog overcast sunset

    depthai_node = Node(
        package='kitti_publisher_ess',
        executable='kitti_publisher_ess_cuda_node',
        name='kitti_publisher_ess_cuda_node',
        output='screen',
        parameters=[{'kitti_path': './../vkitti/vkitti/Scene01/fog/frames/rgb/',
                     'depth_kitti_path': './../vkitti/depth/Scene01/fog/frames/depth/',
                     'plugin_path': './../../dnn_stereo_disparity/plugins/x86_64/ess_plugins.so'}]
    )

    return LaunchDescription([
        depthai_node,
    ])
