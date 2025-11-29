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

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "moge_trt/tensorrt_moge.hpp"
#include "cuda_utils/cuda_check_error.hpp"
#include "cuda_utils/cuda_unique_ptr.hpp"

namespace
{
namespace fs = std::filesystem;

static bool fileExists(const std::string & file_name, bool verbose = true)
{
  if (!std::filesystem::exists(std::filesystem::path(file_name))) {
    if (verbose) {
      std::cout << "File does not exist : " << file_name << std::endl;
    }
    return false;
  }
  return true;
}

// Simple depth to point cloud conversion using pre-scaled camera intrinsics
void depthImageToPointCloud(
  const cv::Mat & depth_image,
  float fx, float fy, float cx, float cy,
  sensor_msgs::msg::PointCloud2 & cloud_msg,
  const std::string & frame_id,
  int downsample_factor,
  const cv::Mat & rgb_image)
{
  cloud_msg.header.frame_id = frame_id;
  
  // Calculate downsampled dimensions
  const int downsampled_height = (depth_image.rows + downsample_factor - 1) / downsample_factor;
  const int downsampled_width = (depth_image.cols + downsample_factor - 1) / downsample_factor;
  
  cloud_msg.height = downsampled_height;
  cloud_msg.width = downsampled_width;
  cloud_msg.is_dense = false;
  cloud_msg.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(cloud_msg);
  const bool has_color = !rgb_image.empty();
  if (has_color) {
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
  } else {
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
  }

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  
  // Only create RGB iterators if we have color data
  std::unique_ptr<sensor_msgs::PointCloud2Iterator<uint8_t>> iter_r, iter_g, iter_b;
  if (has_color) {
    iter_r = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "r");
    iter_g = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "g");
    iter_b = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "b");
  }

  float bad_point = std::numeric_limits<float>::quiet_NaN();

  // Downsample by taking every Nth pixel
  for (int v = 0; v < depth_image.rows; v += downsample_factor) {
    for (int u = 0; u < depth_image.cols; u += downsample_factor) {
      float depth = depth_image.at<float>(v, u);
      
      // Skip invalid depths
      if (depth <= 0.0f || !std::isfinite(depth)) {
        *iter_x = *iter_y = *iter_z = bad_point;
        if (has_color) {
          **iter_r = **iter_g = **iter_b = 0;
        }
      } else {
        // Convert pixel coordinates to 3D point
        *iter_x = (u - cx) * depth / fx;
        *iter_y = (v - cy) * depth / fy;  
        *iter_z = depth;
        
        // Add RGB color if available
        if (has_color) {
          cv::Vec3b rgb = rgb_image.at<cv::Vec3b>(v, u);
          **iter_r = rgb[2]; // R
          **iter_g = rgb[1]; // G  
          **iter_b = rgb[0]; // B
        }
      }
      
      ++iter_x; ++iter_y; ++iter_z;
      if (has_color) {
        ++(*iter_r); ++(*iter_g); ++(*iter_b);
      }
    }
  }
}

} // anonymous namespace

namespace moge_trt
{

TensorRTMoge::TensorRTMoge(
  const std::string & model_path, const std::string & precision,
  tensorrt_common::BuildConfig build_config, const bool use_gpu_preprocess,
  std::string /* calibration_image_list_path */, const tensorrt_common::BatchConfig & batch_config,
  const size_t max_workspace_size)
: batch_size_(batch_config[2]), use_gpu_preprocess_(use_gpu_preprocess)
{
  if (!fileExists(model_path)) {
    throw std::runtime_error("Model file does not exist: " + model_path);
  }

  // Initialize TensorRT common
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
    model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
  trt_common_->setup();

  // Get output tensor information
  const auto& output_dims = trt_common_->getBindingDimensions(1); // points output
  points_elem_num_ = 1;
  for (int i = 0; i < output_dims.nbDims; ++i) {
    points_elem_num_ *= output_dims.d[i];
  }
  
  const auto& mask_dims = trt_common_->getBindingDimensions(3); // mask output  
  mask_elem_num_ = 1;
  for (int i = 0; i < mask_dims.nbDims; ++i) {
    mask_elem_num_ *= mask_dims.d[i];
  }
    
  const auto& scale_dims = trt_common_->getBindingDimensions(4); // metric_scale output
  metric_scale_elem_num_ = 1;
  for (int i = 0; i < scale_dims.nbDims; ++i) {
    metric_scale_elem_num_ *= scale_dims.d[i];
  }

  // Allocate GPU memory for outputs
  points_d_ = cuda_utils::make_unique<float[]>(points_elem_num_);
  mask_d_ = cuda_utils::make_unique<float[]>(mask_elem_num_);
  metric_scale_d_ = cuda_utils::make_unique<float[]>(metric_scale_elem_num_);

  // Allocate CPU memory for outputs
  points_h_ = cuda_utils::make_unique_host<float[]>(points_elem_num_, cudaHostAllocDefault);
  mask_h_ = cuda_utils::make_unique_host<float[]>(mask_elem_num_, cudaHostAllocDefault);
  metric_scale_h_ = cuda_utils::make_unique_host<float[]>(metric_scale_elem_num_, cudaHostAllocDefault);

  // Get input dimensions
  const auto input_dims = trt_common_->getBindingDimensions(0);
  const int input_channels = input_dims.d[1];
  input_height_ = input_dims.d[2]; 
  input_width_ = input_dims.d[3];
  
  // Allocate input memory
  const size_t input_elem_num = batch_size_ * input_channels * input_height_ * input_width_;
  input_d_ = cuda_utils::make_unique<float[]>(input_elem_num);
  input_h_.resize(input_elem_num);

  // Initialize preprocessing buffers if needed
  if (use_gpu_preprocess_) {
    // Will be initialized in initPreprocessBuffer
  }
}

TensorRTMoge::~TensorRTMoge()
{
}

void TensorRTMoge::initPreprocessBuffer(int width, int height)
{
  if (use_gpu_preprocess_) {
    const size_t image_size = width * height * 3; // RGB
    image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
      image_size * batch_size_, cudaHostAllocDefault);
    image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(image_size * batch_size_);
  }
}

bool TensorRTMoge::doInference(
  const std::vector<cv::Mat> & images, 
  const sensor_msgs::msg::CameraInfo & camera_info,
  int downsample_factor,
  bool colorize_pointcloud)
{
  if (images.size() != static_cast<size_t>(batch_size_)) {
    std::cerr << "Batch size mismatch. Expected: " << batch_size_ 
              << ", got: " << images.size() << std::endl;
    return false;
  }

  // Store camera info for postprocessing
  current_camera_info_ = camera_info;

  // Preprocess
  if (use_gpu_preprocess_) {
    preprocessGpu(images);
  } else {
    preprocess(images);
  }

  // Run inference
  if (!infer()) {
    return false;
  }

  // Postprocess with downsampling
  cv::Mat rgb_for_pointcloud = colorize_pointcloud ? images[0] : cv::Mat();
  postprocess(camera_info, downsample_factor, rgb_for_pointcloud);
  
  return true;
}

void TensorRTMoge::preprocess(const std::vector<cv::Mat> & images)
{
  for (size_t batch = 0; batch < images.size(); ++batch) {
    const cv::Mat& image = images[batch];
    cv::Mat resized_image;
    
    // Resize to model input size
    cv::resize(image, resized_image, cv::Size(input_width_, input_height_));
    
    // Convert BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
    
    // Normalize to [0, 1]
    rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0);
    
    // Convert HWC to CHW and copy to input buffer
    const int offset = batch * 3 * input_height_ * input_width_;
    const int channel_size = input_height_ * input_width_;
    
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < input_height_; ++h) {
        for (int w = 0; w < input_width_; ++w) {
          const int src_idx = h * input_width_ * 3 + w * 3 + c;
          const int dst_idx = offset + c * channel_size + h * input_width_ + w;
          input_h_[dst_idx] = reinterpret_cast<float*>(rgb_image.data)[src_idx];
        }
      }
    }
  }
  
  // Copy to GPU
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    input_d_.get(), input_h_.data(), 
    input_h_.size() * sizeof(float), cudaMemcpyHostToDevice, *stream_));
}

void TensorRTMoge::preprocessGpu(const std::vector<cv::Mat> & images)
{
  // GPU preprocessing implementation would go here
  // For now, fall back to CPU preprocessing
  preprocess(images);
}

bool TensorRTMoge::infer()
{
  // Set tensor addresses by name instead of index order
  for (int i = 0; i < trt_common_->getNbIOTensors(); ++i) {
    auto const& name = trt_common_->getEngine()->getIOTensorName(i);
    std::string tensor_name(name);
    
    void* buffer_ptr = nullptr;
    if (tensor_name.find("input") != std::string::npos || i == 0) {
      buffer_ptr = input_d_.get();
    } else if (tensor_name.find("points") != std::string::npos) {
      buffer_ptr = points_d_.get();
    } else if (tensor_name.find("mask") != std::string::npos) {
      buffer_ptr = mask_d_.get();
    } else if (tensor_name.find("metric_scale") != std::string::npos || tensor_name.find("scale") != std::string::npos) {
      buffer_ptr = metric_scale_d_.get();
    } else {
      // For unknown tensors (like 'normal'), allocate a dummy buffer
      static std::vector<CudaUniquePtr<float[]>> dummy_buffers;
      auto dims = trt_common_->getBindingDimensions(i);
      size_t elem_count = 1;
      for (int j = 0; j < dims.nbDims; ++j) {
        elem_count *= dims.d[j];
      }
      dummy_buffers.push_back(cuda_utils::make_unique<float[]>(elem_count));
      buffer_ptr = dummy_buffers.back().get();
    }
    
    trt_common_->getContext()->setTensorAddress(name, buffer_ptr);
  }
  
  const bool result = trt_common_->enqueueV3(*stream_);
  
  if (!result) {
    return false;
  }

  // Copy results back to host
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    points_h_.get(), points_d_.get(), 
    points_elem_num_ * sizeof(float), cudaMemcpyDeviceToHost, *stream_));
    
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    mask_h_.get(), mask_d_.get(),
    mask_elem_num_ * sizeof(float), cudaMemcpyDeviceToHost, *stream_));
    
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    metric_scale_h_.get(), metric_scale_d_.get(),
    metric_scale_elem_num_ * sizeof(float), cudaMemcpyDeviceToHost, *stream_));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(*stream_));
  
  return true;
}

float TensorRTMoge::recoverShift(
    const sensor_msgs::msg::CameraInfo & camera_info,
    const cv::Size & model_input_size)
{
    const int height = model_input_size.height;
    const int width = model_input_size.width;

    // Use a downsampled version for efficiency, as in the original implementation
    const cv::Size downsample_size(64, 64);
    const int ds_height = downsample_size.height;
    const int ds_width = downsample_size.width;

    // Adjust camera intrinsics for the model's input size
    const float scale_x = static_cast<float>(width) / camera_info.width;
    const float scale_y = static_cast<float>(height) / camera_info.height;
    const float fx = camera_info.k[0] * scale_x;
    const float fy = camera_info.k[4] * scale_y;
    const float cx = camera_info.k[2] * scale_x;
    const float cy = camera_info.k[5] * scale_y;

    std::vector<float> x, y, z, u, v;

    // Downsample points, mask, and create normalized UV coordinates
    for (int h = 0; h < height; h += height / ds_height) {
        for (int w = 0; w < width; w += width / ds_width) {
            const int mask_idx = h * width + w;
            if (mask_h_[mask_idx] > 0.5f) {
                const int point_idx_x = h * width * 3 + w * 3 + 0;
                const int point_idx_y = point_idx_x + 1;
                const int point_idx_z = point_idx_x + 2;
                
                x.push_back(points_h_[point_idx_x]);
                y.push_back(points_h_[point_idx_y]);
                z.push_back(points_h_[point_idx_z]);

                // Normalized image plane coordinates
                u.push_back((w - cx) / fx);
                v.push_back((h - cy) / fy);
            }
        }
    }
    
    if (x.empty()) {
        return 0.0f; // Not enough valid points
    }

    // Solve for optimal shift 't' by minimizing: sum | (z_i + t) * u_i - x_i | + | (z_i + t) * v_i - y_i |
    // This is a 1D convex optimization problem. We can solve it by finding where the derivative is zero.
    // The derivative is a sum of step functions, so we find the median of the roots.
    std::vector<float> roots;
    for (size_t i = 0; i < x.size(); ++i) {
        float ui_sq_vi_sq = u[i] * u[i] + v[i] * v[i];
        if (std::abs(ui_sq_vi_sq) > 1e-6) {
            roots.push_back((x[i] * u[i] + y[i] * v[i]) / ui_sq_vi_sq - z[i]);
        }
    }

    if (roots.empty()) {
        return 0.0f;
    }

    // The optimal shift is the median of the roots
    std::sort(roots.begin(), roots.end());
    return roots[roots.size() / 2];
}


void TensorRTMoge::postprocess(const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor, const cv::Mat & rgb_image)
{
    const int height = input_height_;
    const int width = input_width_;

    // 1. Recover the optimal z-axis shift
    float shift = recoverShift(camera_info, cv::Size(width, height));

    // 2. Create the depth map from the Z channel of the points tensor
    cv::Mat depth_map(height, width, CV_32F);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const int point_idx_z = h * width * 3 + w * 3 + 2; // Z channel
            const int mask_idx = h * width + w;

            if (mask_h_[mask_idx] > 0.5f) {
                // 3. Apply the recovered shift
                float corrected_depth = points_h_[point_idx_z] + shift;
                depth_map.at<float>(h, w) = (corrected_depth > 0) ? corrected_depth : 0.0f;
            } else {
                depth_map.at<float>(h, w) = 0.0f;
            }
        }
    }

    // 4. Apply the global metric scale
    if (metric_scale_elem_num_ > 0) {
        const float scale = metric_scale_h_[0];
        if (scale > 0) {
            depth_map *= scale;
        }
    }

    // 5. Store depth at model resolution (no upscaling to save compute)
    depth_image_ = std::move(depth_map);

    // 6. Convert the final depth map to a point cloud
    depthToPointCloud(camera_info, downsample_factor, rgb_image);
}


void TensorRTMoge::depthToPointCloud(const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor, const cv::Mat & rgb_image)
{
  const std::string frame_id = camera_info.header.frame_id.empty() ? "camera_link" : camera_info.header.frame_id;
  
  // Scale camera intrinsics from original image size to model output size
  const float scale_x = static_cast<float>(depth_image_.cols) / camera_info.width;
  const float scale_y = static_cast<float>(depth_image_.rows) / camera_info.height;
  const float fx = camera_info.k[0] * scale_x;
  const float fy = camera_info.k[4] * scale_y;
  const float cx = camera_info.k[2] * scale_x;
  const float cy = camera_info.k[5] * scale_y;
  
  // Resize RGB image to match depth if colorizing
  cv::Mat rgb_resized;
  if (!rgb_image.empty()) {
    cv::resize(rgb_image, rgb_resized, depth_image_.size());
  }
  
  depthImageToPointCloud(depth_image_, fx, fy, cx, cy, point_cloud_, frame_id, downsample_factor, rgb_resized);
  point_cloud_.header.stamp = camera_info.header.stamp;
}

cv::Mat TensorRTMoge::getDepthImage()
{
  return depth_image_;
}

sensor_msgs::msg::PointCloud2 TensorRTMoge::getPointCloud()
{
  return point_cloud_;
}

void TensorRTMoge::printProfiling()
{
  trt_common_->printProfiling();
}

} // namespace moge_trt
