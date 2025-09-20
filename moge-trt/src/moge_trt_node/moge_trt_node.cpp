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

#include "moge_trt/moge_trt_node.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace
{
template <class T>
bool update_param(
  const std::vector<rclcpp::Parameter> & params, const std::string & name, T & value)
{
  const auto itr = std::find_if(
    params.cbegin(), params.cend(),
    [&name](const rclcpp::Parameter & p) { return p.get_name() == name; });

  // Not found
  if (itr == params.cend()) {
    return false;
  }

  value = itr->template get_value<T>();
  return true;
}
} // namespace

namespace moge_trt
{
using namespace std::literals;

MogeTrtNode::MogeTrtNode(const rclcpp::NodeOptions & node_options)
: Node("moge_trt", node_options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  
  // Parameter
  set_param_res_ =
    this->add_on_set_parameters_callback(std::bind(&MogeTrtNode::onSetParam, this, _1));
  
  node_param_.onnx_path = declare_parameter<std::string>(
    "onnx_path", "models/moge-2_vits_normal_291x518.onnx");
  node_param_.precision = declare_parameter<std::string>("precision", "fp16");
  
  // Debug parameters
  node_param_.enable_debug = declare_parameter<bool>("enable_debug", false);
  node_param_.debug_colormap = declare_parameter<std::string>("debug_colormap", "JET");
  node_param_.debug_filepath = declare_parameter<std::string>(
    "debug_filepath", "/tmp/moge_debug/");
  node_param_.write_colormap = declare_parameter<bool>("write_colormap", true);
  node_param_.debug_colormap_min_depth = declare_parameter<double>("debug_colormap_min_depth", 2.0);
  node_param_.debug_colormap_max_depth = declare_parameter<double>("debug_colormap_max_depth", 100.0);
  
  // Point cloud parameters
  node_param_.point_cloud_downsample_factor = declare_parameter<int>("point_cloud_downsample_factor", 10);
  node_param_.colorize_point_cloud = declare_parameter<bool>("colorize_point_cloud", true);
  RCLCPP_INFO(get_logger(), "Point cloud downsampling factor: %d (publishing every %dth point)", 
    node_param_.point_cloud_downsample_factor, node_param_.point_cloud_downsample_factor);

  RCLCPP_INFO(get_logger(), "Using ONNX model: %s", node_param_.onnx_path.c_str());

  // Synchronized subscribers for compressed image and camera_info
  sub_compressed_image_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>(
    this, "~/input/image");
  sub_camera_info_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(
    this, "~/input/camera_info");
  
  // Use approximate time synchronizer with 100ms tolerance
  sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(
    ApproxSyncPolicy(10), *sub_compressed_image_, *sub_camera_info_);
  sync_->registerCallback(std::bind(&MogeTrtNode::onCompressedImageCameraInfo, this, _1, _2));
  
  RCLCPP_INFO(get_logger(), "Using ApproximateTime synchronizer with queue size 10");

  // Debug subscribers to check if individual topics are arriving
  debug_image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
    "~/input/image", rclcpp::SensorDataQoS(),
    std::bind(&MogeTrtNode::onCompressedImageDebug, this, std::placeholders::_1));
  debug_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "~/input/camera_info", rclcpp::SensorDataQoS(),
    std::bind(&MogeTrtNode::onCameraInfoDebug, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "MoGe TRT node initialized successfully");
  RCLCPP_INFO(get_logger(), "Waiting for synchronized messages on:");
  RCLCPP_INFO(get_logger(), "  - Image topic: ~/input/image");
  RCLCPP_INFO(get_logger(), "  - Camera info topic: ~/input/camera_info");

  // Publishers
  pub_depth_image_ = create_publisher<sensor_msgs::msg::Image>("~/output/depth_image", 1);
  pub_point_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/output/point_cloud", 1);
  
  if (node_param_.enable_debug) {
    pub_depth_image_debug_ = create_publisher<sensor_msgs::msg::Image>(
      "~/output/depth_image_debug", 1);
  }

  // Init TensorRT model
  std::string calibType = "MinMax";
  int dla = -1;
  bool first = false;
  bool last = false;
  bool prof = false;
  double clip = 0.0;
  tensorrt_common::BuildConfig build_config(calibType, dla, first, last, prof, clip);

  int batch = 1;
  tensorrt_common::BatchConfig batch_config{1, batch / 2, batch};

  bool use_gpu_preprocess = false;
  std::string calibration_images = "calibration_images.txt";
  const size_t workspace_size = (1 << 30);

  tensorrt_moge_ = std::make_shared<TensorRTMoge>(
    node_param_.onnx_path, node_param_.precision, build_config, use_gpu_preprocess,
    calibration_images, batch_config, workspace_size);
    
  RCLCPP_INFO(get_logger(), "Finish initialize TensorRT MoGe model");
}

void MogeTrtNode::onCompressedImageCameraInfo(
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{

  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  
  const auto width = in_image_ptr->image.cols;
  const auto height = in_image_ptr->image.rows;

  if (!is_initialized_) {
    RCLCPP_INFO(get_logger(), "Initializing TensorRT MoGe preprocessing buffer for %dx%d images", width, height);
    tensorrt_moge_->initPreprocessBuffer(width, height);
    is_initialized_ = true;
    RCLCPP_INFO(get_logger(), "TensorRT MoGe preprocessing buffer initialized");
  }

  std::vector<cv::Mat> input_images;
  input_images.push_back(in_image_ptr->image);
  
  RCLCPP_INFO(get_logger(), "Starting MoGe TensorRT inference...");
  auto start = std::chrono::high_resolution_clock::now();
  bool success = tensorrt_moge_->doInference(input_images, *camera_info_msg, node_param_.point_cloud_downsample_factor, node_param_.colorize_point_cloud);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> inference_duration = end - start;
  
  if (!success) {
    RCLCPP_ERROR(get_logger(), "MoGe inference FAILED!");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "MoGe inference completed successfully in %f seconds", inference_duration.count());

  // Get depth image result
  cv::Mat depth_image = tensorrt_moge_->getDepthImage();
  
  // Publish depth image (32FC1 format for accurate depth values)
  cv_bridge::CvImage cv_img_depth;
  cv_img_depth.image = depth_image;
  cv_img_depth.encoding = "32FC1";
  cv_img_depth.header = image_msg->header;
  
  sensor_msgs::msg::Image depth_image_msgs;
  cv_img_depth.toImageMsg(depth_image_msgs);
  pub_depth_image_->publish(depth_image_msgs);

  // Publish point cloud
  sensor_msgs::msg::PointCloud2 point_cloud = tensorrt_moge_->getPointCloud();
  point_cloud.header = image_msg->header;
  pub_point_cloud_->publish(point_cloud);

  // Publish debug depth image if enabled
  if (node_param_.enable_debug && pub_depth_image_debug_) {
    // Convert depth to visualization format
    cv::Mat depth_vis;
    depth_image.convertTo(depth_vis, CV_32F);
    
    // Normalize for visualization using fixed depth range
    double min_depth = node_param_.debug_colormap_min_depth;
    double max_depth = node_param_.debug_colormap_max_depth;
    
    // Clamp depth values to the specified range
    cv::threshold(depth_vis, depth_vis, max_depth, max_depth, cv::THRESH_TRUNC);
    cv::threshold(depth_vis, depth_vis, min_depth, 0, cv::THRESH_TOZERO);
    
    if (max_depth > min_depth) {
      depth_vis = (depth_vis - min_depth) / (max_depth - min_depth);
    } else {
      depth_vis = cv::Mat::zeros(depth_image.size(), CV_32F);
    }
    
    depth_vis.convertTo(depth_vis, CV_8UC1, 255);
    
    int colormap_type = getColorMapType(node_param_.debug_colormap);
    cv::applyColorMap(depth_vis, depth_vis, colormap_type);
    
    // Add FPS text overlay
    static std::vector<double> inference_times;
    inference_times.push_back(inference_duration.count());
    if (inference_times.size() > 20) {
      inference_times.erase(inference_times.begin());
    }

    double mean_inference_time = std::accumulate(
      inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
    
    // Extract just the filename from the ONNX path
    std::filesystem::path onnx_path(node_param_.onnx_path);
    std::string model_name = onnx_path.filename().string();
    
    std::string inference_time_text = "MoGe-TRT - " + model_name + 
      " - FPS: " + std::to_string(static_cast<int>(1.0 / mean_inference_time));
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    int thickness = 2;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(
      inference_time_text, font_face, font_scale, thickness, &baseline);
    cv::Point text_org((depth_vis.cols - text_size.width) / 2, depth_vis.rows - 10);
    cv::putText(depth_vis, inference_time_text, text_org, font_face, 
                font_scale, cv::Scalar(255, 255, 255), thickness);
    
    // Write to file if enabled
    if (node_param_.write_colormap) {
      // Create debug directory if it doesn't exist
      std::filesystem::create_directories(node_param_.debug_filepath);
      
      int64_t timestamp_sec = image_msg->header.stamp.sec;
      int64_t timestamp_nanosec = image_msg->header.stamp.nanosec;
      std::stringstream ss;
      ss << timestamp_sec << std::setfill('0') << std::setw(9) << timestamp_nanosec;
      std::string filename = node_param_.debug_filepath + "depth_image_" + ss.str() + ".jpg";
      cv::imwrite(filename, depth_vis);
    }
    
    // Publish debug image
    cv_bridge::CvImage cv_img_debug;
    cv_img_debug.image = depth_vis;
    cv_img_debug.encoding = "bgr8";
    cv_img_debug.header = image_msg->header;
    
    sensor_msgs::msg::Image debug_image_msgs;
    cv_img_debug.toImageMsg(debug_image_msgs);
    pub_depth_image_debug_->publish(debug_image_msgs);
  }
}

rcl_interfaces::msg::SetParametersResult MogeTrtNode::onSetParam(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  try {
    {
      auto & p = node_param_;
      update_param(params, "onnx_path", p.onnx_path);
      update_param(params, "precision", p.precision);
      update_param(params, "enable_debug", p.enable_debug);
      update_param(params, "debug_colormap", p.debug_colormap);
      update_param(params, "debug_filepath", p.debug_filepath);
      update_param(params, "write_colormap", p.write_colormap);
      update_param(params, "debug_colormap_min_depth", p.debug_colormap_min_depth);
      update_param(params, "debug_colormap_max_depth", p.debug_colormap_max_depth);
      update_param(params, "point_cloud_downsample_factor", p.point_cloud_downsample_factor);
      update_param(params, "colorize_point_cloud", p.colorize_point_cloud);
    }
  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
    return result;
  }
  result.successful = true;
  result.reason = "success";
  return result;
}

int MogeTrtNode::getColorMapType(const std::string& colormap_name)
{
  if (colormap_name == "JET") return cv::COLORMAP_JET;
  else if (colormap_name == "HOT") return cv::COLORMAP_HOT;
  else if (colormap_name == "COOL") return cv::COLORMAP_COOL;
  else if (colormap_name == "SPRING") return cv::COLORMAP_SPRING;
  else if (colormap_name == "SUMMER") return cv::COLORMAP_SUMMER;
  else if (colormap_name == "AUTUMN") return cv::COLORMAP_AUTUMN;
  else if (colormap_name == "WINTER") return cv::COLORMAP_WINTER;
  else if (colormap_name == "BONE") return cv::COLORMAP_BONE;
  else if (colormap_name == "GRAY") return cv::COLORMAP_BONE;
  else if (colormap_name == "HSV") return cv::COLORMAP_HSV;
  else if (colormap_name == "PARULA") return cv::COLORMAP_PARULA;
  else if (colormap_name == "PLASMA") return cv::COLORMAP_PLASMA;
  else if (colormap_name == "INFERNO") return cv::COLORMAP_INFERNO;
  else if (colormap_name == "VIRIDIS") return cv::COLORMAP_VIRIDIS;
  else if (colormap_name == "MAGMA") return cv::COLORMAP_MAGMA;
  else if (colormap_name == "CIVIDIS") return cv::COLORMAP_CIVIDIS;
  else {
    RCLCPP_WARN(get_logger(), "Unknown colormap '%s', using JET as default", colormap_name.c_str());
    return cv::COLORMAP_JET;
  }
}

void MogeTrtNode::onCompressedImageDebug(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg)
{
  static int image_count = 0;
  image_count++;
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, 
    "[DEBUG] Received compressed image #%d, size: %zu bytes, format: %s, timestamp: %d.%09d", 
    image_count, msg->data.size(), msg->format.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
}

void MogeTrtNode::onCameraInfoDebug(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  static int camera_info_count = 0;
  camera_info_count++;
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
    "[DEBUG] Received camera info #%d, frame: %s, timestamp: %d.%09d", 
    camera_info_count, msg->header.frame_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
}

} // namespace moge_trt

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(moge_trt::MogeTrtNode)
