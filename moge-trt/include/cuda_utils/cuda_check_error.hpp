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

#ifndef CUDA_UTILS__CUDA_CHECK_ERROR_HPP_
#define CUDA_UTILS__CUDA_CHECK_ERROR_HPP_

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace cuda_utils
{
template <typename F, typename N>
void cuda_check_error(const ::cudaError_t e, F && f, N && n)
{
  if (e != ::cudaSuccess) {
    std::stringstream s;
    s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": "
      << ::cudaGetErrorString(e);
    throw std::runtime_error{s.str()};
  }
}
}  // namespace cuda_utils

#define CHECK_CUDA_ERROR(e) (cuda_utils::cuda_check_error(e, __FILE__, __LINE__))

#endif  // CUDA_UTILS__CUDA_CHECK_ERROR_HPP_
