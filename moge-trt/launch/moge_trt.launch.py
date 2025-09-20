# MIT License
#
# Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    moge_trt_dir = get_package_share_directory('moge_trt')
    config_path = os.path.join(moge_trt_dir, "config", "moge_trt.param.yaml")

    zenoh_router_node = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "rmw_zenoh_cpp",
            "rmw_zenohd"
        ],
        output="screen",
    )

    rosbag_play_node = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "/bags/2025-08-13_JBU_ika-hof-pedestrians/rosbag2_2025_08_13-17_19_57_fixed",
            "-l",
            "-r",
            "1",
        ],
        output="screen",
    )

    moge_trt_node = Node(
        package='moge_trt',
        executable='moge_trt_main',
        name='moge_trt',
        output='screen',
        remappings=[
            ('~/input/image', "/drivers/zed_camera/front_center/left/image_rect_color/compressed"),
            ('~/input/camera_info', "/drivers/zed_camera/front_center/left/camera_info"),
            ('~/output/depth_image', "/moge_trt/output/depth_image"),
            ('~/output/point_cloud', "/moge_trt/output/point_cloud")
        ],
        parameters=[config_path]
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=['-d' + os.path.join(get_package_share_directory('moge_trt'), 'config', 'rviz.rviz')]
    )

    ld = LaunchDescription()
    ld.add_action(rosbag_play_node)
    ld.add_action(rviz_node)
    ld.add_action(moge_trt_node)
    ld.add_action(zenoh_router_node)
    return ld
