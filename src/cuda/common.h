/**
 *  Copyright (c) 2017-2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/common.h
 * @brief Common utilities for CUDA
 */
#ifndef GNNDLP_CUDA_COMMON_H_
#define GNNDLP_CUDA_COMMON_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include <memory>

namespace gnndlp {
namespace cuda {

/**
 * @brief This class is designed to allocate workspace storage
 * and to get a nonblocking thrust execution policy
 * that uses torch's CUDA memory pool and the current cuda stream:
 *
 * cuda::CUDAWorkspaceAllocator allocator;
 * const auto stream = torch::cuda::getDefaultCUDAStream();
 * const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);
 *
 * Now, one can pass exec_policy to thrust functions
 *
 * To get an integer array of size 1000 whose lifetime is managed by unique_ptr,
 * use:
 *
 * auto int_array = allocator.AllocateStorage<int>(1000);
 *
 * int_array.get() gives the raw pointer.
 */
struct CUDAWorkspaceAllocator {
  // Required by thrust to satisfy allocator requirements.
  using value_type = char;

  explicit CUDAWorkspaceAllocator() { at::globalContext().lazyInitCUDA(); }

  CUDAWorkspaceAllocator& operator=(const CUDAWorkspaceAllocator&) = default;

  void operator()(void* ptr) const {
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
  }

  // Required by thrust to satisfy allocator requirements.
  value_type* allocate(std::ptrdiff_t size) const {
    return reinterpret_cast<value_type*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(size));
  }

  // Required by thrust to satisfy allocator requirements.
  void deallocate(value_type* ptr, std::size_t) const { operator()(ptr); }

  template <typename T>
  std::unique_ptr<T, CUDAWorkspaceAllocator> AllocateStorage(
      std::size_t size) const {
    return std::unique_ptr<T, CUDAWorkspaceAllocator>(
        reinterpret_cast<T*>(allocate(sizeof(T) * size)), *this);
  }
};

inline auto BuildAllocator() { return CUDAWorkspaceAllocator{}; }

template <typename T>
inline bool is_zero(T size) {
  return size == 0;
}

template <>
inline bool is_zero<dim3>(dim3 size) {
  return size.x == 0 || size.y == 0 || size.z == 0;
}

#define CUDA_CALL(func) C10_CUDA_CHECK((func))

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...)            \
  {                                                                           \
    if (!gnndlp::cuda::is_zero((nblks)) && !gnndlp::cuda::is_zero((nthrs))) { \
      (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__);         \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    }                                                                         \
  }

#define GNNDLP_DISPATCH_ELEMENT_SIZES(element_size, name, ...)                \
  [&] {                                                                       \
    switch (element_size) {                                                   \
      case 1: {                                                               \
        using element_size_t = uint8_t;                                       \
        [[maybe_unused]] constexpr auto element_size_v =                      \
            sizeof(element_size_t);                                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 2: {                                                               \
        using element_size_t = uint16_t;                                      \
        [[maybe_unused]] constexpr auto element_size_v =                      \
            sizeof(element_size_t);                                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 4: {                                                               \
        using element_size_t = uint32_t;                                      \
        [[maybe_unused]] constexpr auto element_size_v =                      \
            sizeof(element_size_t);                                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 8: {                                                               \
        using element_size_t = uint64_t;                                      \
        [[maybe_unused]] constexpr auto element_size_v =                      \
            sizeof(element_size_t);                                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 16: {                                                              \
        using element_size_t = float4;                                        \
        [[maybe_unused]] constexpr auto element_size_v =                      \
            sizeof(element_size_t);                                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default:                                                                \
        TORCH_CHECK(false, name, " with the element_size is not supported!"); \
        using element_size_t = uint8_t;                                       \
        [[maybe_unused]] constexpr auto element_size_v =                      \
            sizeof(element_size_t);                                           \
        return __VA_ARGS__();                                                 \
    }                                                                         \
  }()

}  // namespace cuda
}  // namespace gnndlp
#endif  // GNNDLP_CUDA_COMMON_H_
