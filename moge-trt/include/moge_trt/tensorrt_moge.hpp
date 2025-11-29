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

#ifndef MOGE_TRT__TENSORRT_MOGE_HPP_
#define MOGE_TRT__TENSORRT_MOGE_HPP_

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <vector>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

namespace moge_trt
{
using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

/**
 * @class TensorRTMoge
 * @brief TensorRT MoGeV2 for faster depth estimation and point cloud generation
 */
class TensorRTMoge
{
public:
  /**
   * @brief Construct TensorRTMoge.
   * @param[in] model_path ONNX model_path
   * @param[in] precision precision for inference
   * @param[in] build_config configuration including precision, calibration method, etc.
   * @param[in] use_gpu_preprocess whether use cuda gpu for preprocessing
   * @param[in] calibration_image_list_file path for calibration files
   * @param[in] batch_config configuration for batched execution
   * @param[in] max_workspace_size maximum workspace for building TensorRT engine
   */
  TensorRTMoge(
    const std::string & model_path, const std::string & precision,
    const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
    const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
    const tensorrt_common::BatchConfig & batch_config = {1, 1, 1},
    const size_t max_workspace_size = (1 << 30));

  /**
   * @brief Deconstruct TensorRTMoge
   */
  ~TensorRTMoge();

  /**
   * @brief run inference including pre-process and post-process
   * @param[in] images batched images
   * @param[in] camera_info camera calibration info for point cloud generation
   * @param[in] downsample_factor only publish every Nth point (1 = no downsampling)
   * @param[in] colorize_pointcloud whether to colorize point cloud with RGB
   */
  bool doInference(const std::vector<cv::Mat> & images, const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor = 1, bool colorize_pointcloud = false);

  void initPreprocessBuffer(int width, int height);

  /**
   * @brief output TensorRT profiles for each layer
   */
  void printProfiling(void);

  /**
   * @brief Get the depth image result
   * @return depth image as cv::Mat
   */
  cv::Mat getDepthImage();

  /**
   * @brief Get the point cloud result
   * @return point cloud as ROS2 PointCloud2 message
   */
  sensor_msgs::msg::PointCloud2 getPointCloud();

private:
  /**
   * @brief run preprocess including resizing, letterbox, NHWC2NCHW and toFloat on CPU
   * @param[in] images batching images
   */
  void preprocess(const std::vector<cv::Mat> & images);

  /**
   * @brief run preprocess on GPU
   * @param[in] images batching images
   */
  void preprocessGpu(const std::vector<cv::Mat> & images);

  /**
   * @brief perform TensorRT inference
   */
  bool infer();

  /**
   * @brief postprocess inference results to generate depth and point cloud
   * @param[in] camera_info camera calibration for point cloud generation
   * @param[in] downsample_factor downsampling factor for point cloud
   * @param[in] rgb_image optional RGB image for colorizing point cloud
   */
  void postprocess(const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor = 1, const cv::Mat & rgb_image = cv::Mat());

  /**
   * @brief recover focal length and shift for MoGe model
   */
  float recoverShift(
    const sensor_msgs::msg::CameraInfo & camera_info,
    const cv::Size & model_input_size);

  /**
   * @brief convert depth image to point cloud using camera intrinsics
   * @param[in] camera_info camera calibration parameters
   * @param[in] downsample_factor only publish every Nth point
   * @param[in] rgb_image optional RGB image for colorizing point cloud
   */
  void depthToPointCloud(const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor = 1, const cv::Mat & rgb_image = cv::Mat());

  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;

  // Input/output buffers
  std::vector<float> input_h_;
  CudaUniquePtr<float[]> input_d_;
  
  // Output buffers for MoGe (points, mask, metric_scale)
  CudaUniquePtr<float[]> points_d_;
  CudaUniquePtr<float[]> mask_d_;  
  CudaUniquePtr<float[]> metric_scale_d_;
  
  CudaUniquePtrHost<float[]> points_h_;
  CudaUniquePtrHost<float[]> mask_h_;
  CudaUniquePtrHost<float[]> metric_scale_h_;

  size_t points_elem_num_;
  size_t mask_elem_num_;
  size_t metric_scale_elem_num_;

  StreamUniquePtr stream_{makeCudaStream()};

  int batch_size_;

  // preprocessing parameters
  bool use_gpu_preprocess_;
  CudaUniquePtrHost<unsigned char[]> image_buf_h_;
  CudaUniquePtr<unsigned char[]> image_buf_d_;

  int input_width_ = 518;  // MoGe input width
  int input_height_ = 291; // MoGe input height

  // Results
  cv::Mat depth_image_;
  sensor_msgs::msg::PointCloud2 point_cloud_;
  
  // Current camera info for processing
  sensor_msgs::msg::CameraInfo current_camera_info_;
};

}  // namespace moge_trt

#endif  // MOGE_TRT__TENSORRT_MOGE_HPP_
