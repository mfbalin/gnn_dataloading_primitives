/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/sort.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <gnndlp/ops.h>

#include <cub/cub.cuh>

#include "./common.h"

namespace gnndlp {
namespace cuda {
namespace ops {

std::pair<torch::Tensor, torch::Tensor> Sort(
    torch::Tensor input, int num_bits) {
  int64_t num_items = input.size(0);
  auto original_idx =
      torch::arange(num_items, input.options().dtype(torch::kLong));
  auto sorted_array = torch::empty_like(input);
  auto sorted_idx = torch::empty_like(original_idx);
  auto allocator = cuda::BuildAllocator();
  auto stream = c10::cuda::getDefaultCUDAStream();
  if (num_bits == 0) num_bits = input.element_size() * 8;
  AT_DISPATCH_INDEX_TYPES(
      input.scalar_type(), "Sort", ([&] {
        const auto input_keys = input.data_ptr<index_t>();
        const auto input_values = original_idx.data_ptr<int64_t>();
        auto sorted_keys = sorted_array.data_ptr<index_t>();
        auto sorted_values = sorted_idx.data_ptr<int64_t>();
        size_t workspace_size = 0;
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(
            nullptr, workspace_size, input_keys, sorted_keys, input_values,
            sorted_values, num_items, 0, num_bits, stream));
        auto temp = allocator.AllocateStorage<char>(workspace_size);
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(
            temp.get(), workspace_size, input_keys, sorted_keys, input_values,
            sorted_values, num_items, 0, num_bits, stream));
      }));
  return std::make_pair(sorted_array, sorted_idx);
}

}  // namespace ops
}  // namespace cuda
}  // namespace gnndlp
