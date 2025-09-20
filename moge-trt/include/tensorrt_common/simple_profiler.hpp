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

// Updated code from https://github.com/scepter914/DepthAnything-ROS

#ifndef TENSORRT_COMMON__SIMPLE_PROFILER_HPP_
#define TENSORRT_COMMON__SIMPLE_PROFILER_HPP_

#include <NvInfer.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace tensorrt_common
{
struct LayerInfo
{
  int in_c;
  int out_c;
  int w;
  int h;
  int k;
  int stride;
  int groups;
  nvinfer1::LayerType type;
};

/**
 * @class Profiler
 * @brief Collect per-layer profile information, assuming times are reported in the same order
 */
class SimpleProfiler : public nvinfer1::IProfiler
{
public:
  struct Record
  {
    float time{0};
    int count{0};
    float min_time{-1.0};
    int index;
  };
  SimpleProfiler(
    std::string name,
    const std::vector<SimpleProfiler> & src_profilers = std::vector<SimpleProfiler>());

  void reportLayerTime(const char * layerName, float ms) noexcept override;

  void setProfDict(nvinfer1::ILayer * layer) noexcept;

  friend std::ostream & operator<<(std::ostream & out, SimpleProfiler & value);

private:
  std::string m_name;
  std::map<std::string, Record> m_profile;
  int m_index;
  std::map<std::string, LayerInfo> m_layer_dict;
};
}  // namespace tensorrt_common
#endif  // TENSORRT_COMMON__SIMPLE_PROFILER_HPP_
