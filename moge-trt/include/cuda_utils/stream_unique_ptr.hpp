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

#ifndef CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_
#define CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_

#include <cuda_runtime_api.h>

#include <memory>

namespace cuda_utils
{
struct StreamDeleter
{
  void operator()(cudaStream_t * stream)
  {
    if (stream) {
      cudaStreamDestroy(*stream);
      delete stream;
    }
  }
};

using StreamUniquePtr = std::unique_ptr<cudaStream_t, StreamDeleter>;

inline StreamUniquePtr makeCudaStream(const uint32_t flags = cudaStreamDefault)
{
  StreamUniquePtr stream(new cudaStream_t, StreamDeleter());
  if (cudaStreamCreateWithFlags(stream.get(), flags) != cudaSuccess) {
    stream.reset(nullptr);
  }
  return stream;
}
}  // namespace cuda_utils

#endif  // CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_
