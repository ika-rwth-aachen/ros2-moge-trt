# MoGeV2 TensorRT ROS2 Node


A ROS2 node for [MoGeV2](https://github.com/microsoft/MoGe) depth estimation using TensorRT for real-time inference. This node subscribes to camera image and camera info topics and publishes directly both, a metric depth image and `PointCloud2` point cloud.


## Features

- **Real-time metric depth estimation** using MoGeV2 with TensorRT acceleration
- **Point cloud generation** from metric depth image
- **Debug visualization** with colormap options
- **Configurable precision** (FP16/FP32)

> [!IMPORTANT]  
> This repository is open-sourced and maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/).  
> We cover a wide variety of research topics within our [*Vehicle Intelligence & Automated Driving*](https://www.ika.rwth-aachen.de/en/competences/fields-of-research/vehicle-intelligence-automated-driving.html) domain.  
> If you would like to learn more about how we can support your automated driving or robotics efforts, feel free to reach out to us!  
> :email: ***opensource@ika.rwth-aachen.de***


## Dependencies
- Based on image `nvcr.io/nvidia/tensorrt:25.08-py3`
- Ubuntu 24.04, ROS2 Jazzy
- CUDA 13
- TensorRT 10.9

## Topics

### Subscribed Topics

- `~/input/image` (sensor_msgs/Image): Input camera image
- `~/input/camera_info` (sensor_msgs/CameraInfo): Camera calibration info

### Published Topics  

- `~/output/depth_image` (sensor_msgs/Image): Depth image (32FC1 format)
- `~/output/point_cloud` (sensor_msgs/PointCloud2): Generated point cloud
- `~/output/depth_image_debug` (sensor_msgs/Image): Debug depth visualization (if enabled)

## Parameters

### Model Configuration
- `onnx_path` (string): Path to MoGe ONNX model file
- `precision` (string): Inference precision ("fp16" or "fp32")

### Debug Configuration
- `enable_debug` (bool): Enable debug visualization output
- `debug_colormap` (string): Colormap for depth visualization
- `debug_filepath` (string): Directory to save debug images
- `write_colormap` (bool): Save debug images to disk

## Usage

### Basic Launch

```bash
ros2 launch moge_trt moge_trt.launch.py
```

### With Custom Topics

```bash
ros2 launch moge_trt moge_trt.launch.py \
    input_image_topic:=/your_camera/image_raw \
    input_camera_info_topic:=/your_camera/camera_info \
    output_depth_topic:=/moge/depth \
    output_point_cloud_topic:=/moge/points
```

### With Debug Enabled

```bash
ros2 launch moge_trt moge_trt.launch.py \
    params_file:=src/moge_trt/config/debug.param.yaml
```

## Model Preparation

1. **Obtain MoGe ONNX model**: Follow the instructions to export the ONNX model [here](https://github.com/yester31/Monocular_Depth_Estimation_TRT/blob/main/MoGe_2/README.md)
2. **Place model file**: Put the ONNX file in the `models/` directory
3. **Update configuration**: Modify `config/moge_trt.param.yaml` with the correct model path

Expected model input format:
- Input shape: [1, 3, 291, 518] (batch, channels, height, width)
- Input type: float32
- Value range: [0, 1] (normalized)

Expected model outputs:
- `points`: [1, 291, 518, 3] - 3D points
- `mask`: [1, 291, 518] - validity mask
- `metric_scale`: [1] - scale factor


## Docker Image
We precompile and provide a Docker image with all dependencies installed and ready 
to use. You can pull it from Docker Hub:
```bash
docker pull tillbeemelmanns/ros2-moge-trt:latest-dev
```

If you want to run rosbags and visualize the output in rviz2 make sure to install  it
```bash
apt update && apt install -y \
ros-jazzy-rviz2 \
ros-jazzy-rosbag2 \
ros-jazzy-rosbag2-storage-mcap
```

We recommend to use the docker image in combination with our other tools for Docker and ROS.
- [*docker-ros*](https://github.com/ika-rwth-aachen/docker-ros) automatically builds minimal container images of ROS applications <a href="https://github.com/ika-rwth-aachen/docker-ros"><img src="https://img.shields.io/github/stars/ika-rwth-aachen/docker-ros?style=social"/></a>
- [*docker-run*](https://github.com/ika-rwth-aachen/docker-run) is a CLI tool for simplified interaction with Docker images <a href="https://github.com/ika-rwth-aachen/docker-run"><img src="https://img.shields.io/github/stars/ika-rwth-aachen/docker-run?style=social"/></a>



## Building

```bash
# From your ROS2 workspace
colcon build --packages-select moge_trt --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Performance
The node is optimized for real-time performance.

Performance on RTX 6000:
- **VITS**: 37 FPS
- **VITB**: 31 FPS
- **VITL**: 17 FPS

## Architecture

```
Input Image + Camera Info
         ↓
    Preprocessing (CPU/GPU)
         ↓  
    TensorRT Inference (GPU)
         ↓
    Postprocessing (CPU)
         ↓
   Depth Image + Point Cloud
```

## Shift Recovery Algorithm

The MoGe model outputs an affine-invariant point map, which has an unknown scale
and shift. When using a camera with **known intrinsics**, we don't need to 
recover the focal length, but we still must solve for the Z-axis `shift` for 
each frame to align the point cloud correctly.

This is an optimization problem: find the `shift` that results in the lowest 
reprojection error. There are two common ways to measure this error: 
- L2 norm (leading to a least-squares solution) 
- L1 norm (leading to a least absolute deviations solution).

#### 1. L2 Norm (Least Squares)

This approach minimizes the **sum of the squares** of the reprojection errors: `min Σ(error)²`.

-   **Key Characteristic**: It is highly sensitive to outliers. If the model produces a few points with very large errors, the squaring of these errors causes them to dominate the calculation. The resulting `shift` will be heavily skewed to accommodate these bad points, often at the expense of a better fit for the majority of good points.
-   **Implementation**: Typically requires an iterative numerical solver (e.g., Levenberg-Marquardt, used in `scipy.optimize.least_squares`).

#### 2. L1 Norm (Least Absolute Deviations)

This approach minimizes the **sum of the absolute values** of the reprojection errors: `min Σ|error|`.

-   **Key Characteristic**: It is **robust to outliers**. Since the error is not squared, outlier points contribute linearly to the total error and do not dominate the solution. This is analogous to how the median is more robust to extreme values than the mean.
-   **Implementation**: For this specific 1D optimization problem, there is a highly efficient, non-iterative solution: calculating the **median** of the roots of the individual error terms.

### Implementation

For this ROS2 node, we use the **L1 Norm** approach for three reasons:
1.  **Robustness**: Deep learning models can produce erroneous outputs. The L1 method provides stable and reliable `shift` estimates even in the presence of such outliers.
2.  **Performance**: The direct median calculation is significantly faster and more predictable, making it ideal for real-time applications.
3.  **Simplicity**: It avoids adding heavy external dependencies (like Ceres Solver) for a C++ implementation.


## Troubleshooting

### Common Issues

1. **TensorRT engine building fails**:
   - Check CUDA/TensorRT compatibility
   - Verify ONNX model format
   - Increase workspace size

2. **Point cloud appears incorrect**:
   - Verify camera_info calibration
   - Check coordinate frame conventions
   - Validate depth value units

3. **Performance issues**:
   - Enable FP16 precision
   - Check GPU memory usage

### Debug Mode

Enable debug mode to troubleshoot:
```yaml
enable_debug: true
debug_colormap: "JET"
debug_colormap_min_depth: 0.0
debug_colormap_max_depth: 50.0
write_colormap: true
```

This will publish colorized depth images and save them to disk for inspection.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Thanks to the following repositories for inspiration:

- [MoGe](https://github.com/microsoft/MoGe)
- [Monocular_Depth_Estimation_TRT](https://github.com/yester31/Monocular_Depth_Estimation_TRT)
- [DepthAnything-ROS](https://github.com/scepter914/DepthAnything-ROS)
