// MIT License
//
// Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef MOGE_TRT__MOGE_TRT_NODE_HPP_
#define MOGE_TRT__MOGE_TRT_NODE_HPP_

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "moge_trt/tensorrt_moge.hpp"

namespace moge_trt
{
class MogeTrtNode : public rclcpp::Node
{
public:
  explicit MogeTrtNode(const rclcpp::NodeOptions & node_options);

  struct NodeParam
  {
    std::string onnx_path{};
    std::string precision{};
    bool enable_debug{};
    std::string debug_colormap{};
    std::string debug_filepath{};
    bool write_colormap{};
    double debug_colormap_min_depth{};  // Minimum depth value for colormap visualization
    double debug_colormap_max_depth{};  // Maximum depth value for colormap visualization
    int point_cloud_downsample_factor{};  // Only publish every Nth point (1 = no downsampling)
    bool colorize_point_cloud{};  // Add RGB colors from input image to point cloud
  };

private:
  // Synchronized subscribers for compressed image and camera_info
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>> sub_compressed_image_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> sub_camera_info_;
  
  // Use approximate time synchronizer for more flexible timing
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CameraInfo> ApproxSyncPolicy;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;
  
  // Debug subscribers (separate from sync)
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr debug_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr debug_camera_info_sub_;

  // Callbacks
  void onCompressedImageCameraInfo(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);
    
  // Debug callbacks for individual topics
  void onCompressedImageDebug(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg);
  void onCameraInfoDebug(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_debug_;

  // Helper methods
  int getColorMapType(const std::string& colormap_name);

  // Parameter Server
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
  rcl_interfaces::msg::SetParametersResult onSetParam(
    const std::vector<rclcpp::Parameter> & params);

  // Parameter
  NodeParam node_param_{};

  // Core
  std::shared_ptr<TensorRTMoge> tensorrt_moge_;
  bool is_initialized_ = false;
};

} // namespace moge_trt

#endif // MOGE_TRT__MOGE_TRT_NODE_HPP_
