# MoGe ONNX Models

This directory should contain the MoGe ONNX model files.

## Required Files

Place your MoGe ONNX model file here:
- `moge-2_vits_normal_388x518_dynamo_sim.onnx` (or update the path in config/moge_trt.param.yaml)

## Model Format

Expected model specifications:
- **Input**: RGB image, shape [1, 3, 388, 518], type float32, range [0, 1]
- **Outputs**:
  - `points`: 3D points [1, 388, 518, 3] 
  - `mask`: validity mask [1, 388, 518]
  - `metric_scale`: depth scale factor [1]

## Installation

After building the package, this directory will be installed to:
```
/path/to/install/share/moge_trt/models/
```

The node will look for models at the path specified in the parameter file.