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

#include <tensorrt_common/simple_profiler.hpp>

#include <iomanip>

namespace tensorrt_common
{

SimpleProfiler::SimpleProfiler(std::string name, const std::vector<SimpleProfiler> & src_profilers)
: m_name(name)
{
  float total_time = 0.0;
  m_index = 0;
  for (const auto & src_profiler : src_profilers) {
    for (const auto & rec : src_profiler.m_profile) {
      auto it = m_profile.find(rec.first);
      if (it == m_profile.end()) {
        m_profile.insert(rec);
      } else {
        it->second.time += rec.second.time;
        it->second.count += rec.second.count;
        total_time += rec.second.time;
      }
    }
  }
}

void SimpleProfiler::reportLayerTime(const char * layerName, float ms) noexcept
{
  m_profile[layerName].count++;
  m_profile[layerName].time += ms;
  if (m_profile[layerName].min_time == -1.0) {
    m_profile[layerName].min_time = ms;
    m_profile[layerName].index = m_index;
    m_index++;
  } else if (m_profile[layerName].min_time > ms) {
    m_profile[layerName].min_time = ms;
  }
}

void SimpleProfiler::setProfDict(nvinfer1::ILayer * layer) noexcept
{
  std::string name = layer->getName();
  m_layer_dict[name];
  m_layer_dict[name].type = layer->getType();
  if (layer->getType() == nvinfer1::LayerType::kCONVOLUTION) {
    nvinfer1::IConvolutionLayer * conv = (nvinfer1::IConvolutionLayer *)layer;
    nvinfer1::ITensor * in = layer->getInput(0);
    nvinfer1::Dims dim_in = in->getDimensions();
    nvinfer1::ITensor * out = layer->getOutput(0);
    nvinfer1::Dims dim_out = out->getDimensions();
    nvinfer1::Dims k_dims = conv->getKernelSizeNd();
    nvinfer1::Dims s_dims = conv->getStrideNd();
    int groups = conv->getNbGroups();
    int stride = s_dims.d[0];
    int kernel = k_dims.d[0];
    m_layer_dict[name].in_c = dim_in.d[1];
    m_layer_dict[name].out_c = dim_out.d[1];
    m_layer_dict[name].w = dim_in.d[3];
    m_layer_dict[name].h = dim_in.d[2];
    m_layer_dict[name].k = kernel;
    ;
    m_layer_dict[name].stride = stride;
    m_layer_dict[name].groups = groups;
  }
}

std::ostream & operator<<(std::ostream & out, SimpleProfiler & value)
{
  out << "========== " << value.m_name << " profile ==========" << std::endl;
  float totalTime = 0;
  std::string layerNameStr = "Operation";

  int maxLayerNameLength = static_cast<int>(layerNameStr.size());
  for (const auto & elem : value.m_profile) {
    totalTime += elem.second.time;
    maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
  }

  auto old_settings = out.flags();
  auto old_precision = out.precision();
  // Output header
  {
    out << "index, " << std::setw(12);
    out << std::setw(maxLayerNameLength) << layerNameStr << " ";
    out << std::setw(12) << "Runtime"
        << "%,"
        << " ";
    out << std::setw(12) << "Invocations"
        << " , ";
    out << std::setw(12) << "Runtime[ms]"
        << " , ";
    out << std::setw(12) << "Avg Runtime[ms]"
        << " ,";
    out << std::setw(12) << "Min Runtime[ms]" << std::endl;
  }
  int index = value.m_index;
  for (int i = 0; i < index; i++) {
    for (const auto & elem : value.m_profile) {
      if (elem.second.index == i) {
        out << i << ",   ";
        out << std::setw(maxLayerNameLength) << elem.first << ",";
        out << std::setw(12) << std::fixed << std::setprecision(1)
            << (elem.second.time * 100.0F / totalTime) << "%"
            << ",";
        out << std::setw(12) << elem.second.count << ",";
        out << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.time << ", ";
        out << std::setw(12) << std::fixed << std::setprecision(2)
            << elem.second.time / elem.second.count << ", ";
        out << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.min_time
            << std::endl;
      }
    }
  }
  out.flags(old_settings);
  out.precision(old_precision);
  out << "========== " << value.m_name << " total runtime = " << totalTime
      << " ms ==========" << std::endl;
  return out;
}
}  // namespace tensorrt_common
