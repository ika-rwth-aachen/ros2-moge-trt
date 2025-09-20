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

cmake_minimum_required(VERSION 3.16)
project(moge_trt)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV 4 REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_ONNXPARSER})

# Common utilities are included in the include/ and src/ directories

# Include directories
include_directories(
  include
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Create TensorRT MoGe library
add_library(tensorrt_moge SHARED
  src/tensorrt_moge/tensorrt_moge.cpp
  src/tensorrt_common/tensorrt_common.cpp
  src/tensorrt_common/simple_profiler.cpp
)

target_link_libraries(tensorrt_moge
  ${TENSORRT_LIBRARY}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
)

ament_target_dependencies(tensorrt_moge
  rclcpp
  sensor_msgs
  cv_bridge
)

# Create MoGe node library
add_library(moge_trt_node SHARED
  src/moge_trt_node/moge_trt_node.cpp
)

target_link_libraries(moge_trt_node
  tensorrt_moge
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(moge_trt_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  message_filters
)

# Register component
rclcpp_components_register_nodes(moge_trt_node "moge_trt::MogeTrtNode")

# Create executable
add_executable(moge_trt_main
  src/main.cpp
)

target_link_libraries(moge_trt_main
  moge_trt_node
)

ament_target_dependencies(moge_trt_main
  rclcpp
)

# Create generate_engines executable
add_executable(generate_engines
  src/generate_engines.cpp
  src/tensorrt_common/tensorrt_common.cpp
  src/tensorrt_common/simple_profiler.cpp
)

target_include_directories(generate_engines PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

target_compile_definitions(generate_engines PRIVATE DEPTH_ANYTHING_STANDALONE)

target_link_libraries(generate_engines
  ${TENSORRT_LIBRARY}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
)

# Install targets
install(TARGETS
  tensorrt_moge
  moge_trt_node
  moge_trt_main
  generate_engines
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY config launch models
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_libraries(tensorrt_moge moge_trt_node)
ament_export_include_directories(include)
ament_package()
