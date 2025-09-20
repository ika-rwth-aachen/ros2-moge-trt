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

#ifndef CUDA_UTILS__CUDA_UNIQUE_PTR_HPP_
#define CUDA_UTILS__CUDA_UNIQUE_PTR_HPP_

#include "cuda_utils/cuda_check_error.hpp"

#include <memory>
#include <type_traits>

namespace cuda_utils
{
struct CudaDeleter
{
  void operator()(void * p) const { CHECK_CUDA_ERROR(::cudaFree(p)); }
};
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
typename std::enable_if_t<std::is_array<T>::value, CudaUniquePtr<T>> make_unique(
  const std::size_t n)
{
  using U = typename std::remove_extent_t<T>;
  U * p;
  CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(U) * n));
  return CudaUniquePtr<T>{p};
}

template <typename T>
CudaUniquePtr<T> make_unique()
{
  T * p;
  CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(T)));
  return CudaUniquePtr<T>{p};
}

struct CudaDeleterHost
{
  void operator()(void * p) const { CHECK_CUDA_ERROR(::cudaFreeHost(p)); }
};
template <typename T>
using CudaUniquePtrHost = std::unique_ptr<T, CudaDeleterHost>;

template <typename T>
typename std::enable_if_t<std::is_array<T>::value, CudaUniquePtrHost<T>> make_unique_host(
  const std::size_t n, unsigned int flag)
{
  using U = typename std::remove_extent_t<T>;
  U * p;
  CHECK_CUDA_ERROR(::cudaHostAlloc(reinterpret_cast<void **>(&p), sizeof(U) * n, flag));
  return CudaUniquePtrHost<T>{p};
}

template <typename T>
CudaUniquePtrHost<T> make_unique_host(unsigned int flag = cudaHostAllocDefault)
{
  T * p;
  CHECK_CUDA_ERROR(::cudaHostAlloc(reinterpret_cast<void **>(&p), sizeof(T), flag));
  return CudaUniquePtrHost<T>{p};
}
}  // namespace cuda_utils

#endif  // CUDA_UTILS__CUDA_UNIQUE_PTR_HPP_
