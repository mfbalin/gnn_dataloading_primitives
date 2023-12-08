/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/index_select_csc.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <gnndlp/ops.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <numeric>

#include "./common.h"
#include "./utils.h"

namespace gnndlp {
namespace cuda {
namespace ops {

constexpr int BLOCK_SIZE = 128;

// Given the in_degree array and a permutation, returns in_degree of the output
// and the permuted and modified in_degree of the input. The modified in_degree
// is modified so that there is slack to be able to align as needed.
template <typename indptr_t, typename indices_t>
struct AlignmentFunc {
  static_assert(GPU_CACHE_LINE_SIZE % sizeof(indices_t) == 0);
  const indptr_t* in_degree;
  const int64_t* perm;
  int64_t num_nodes;
  __host__ __device__ auto operator()(int64_t row) {
    constexpr int num_elements = GPU_CACHE_LINE_SIZE / sizeof(indices_t);
    return thrust::make_tuple(
        in_degree[row],
        // A single cache line has num_elements items, we add num_elements - 1
        // to ensure there is enough slack to move forward or backward by
        // num_elements - 1 items if the performed access is not aligned.
        (indptr_t)(in_degree[perm ? perm[row % num_nodes] : row] + num_elements - 1));
  }
};

template <typename indptr_t, typename indices_t>
__global__ void _CSRRowWiseOneHopExtractorAlignedKernel(
    const indptr_t hop_size, const int64_t num_nodes,
    const indptr_t* const indptr, const indptr_t* const sub_indptr,
    const indptr_t* const sub_indptr_aligned, const indices_t* const indices,
    indices_t* const hop, const int64_t* const perm) {
  indptr_t tx = static_cast<indptr_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  while (tx < hop_size) {
    const auto rpos_ = cuda::UpperBound(sub_indptr_aligned, num_nodes, tx) - 1;
    const auto rpos = perm ? perm[rpos_] : rpos_;
    const auto out_row = sub_indptr[rpos];
    const auto d = sub_indptr[rpos + 1] - out_row;
    const int offset =
        ((size_t)(indices + indptr[rpos] - sub_indptr_aligned[rpos_]) %
         GPU_CACHE_LINE_SIZE) /
        sizeof(indices_t);
    const auto rofs = tx - sub_indptr_aligned[rpos_] - offset;
    if (rofs >= 0 && rofs < d) {
      const auto in_idx = indptr[rpos] + rofs;
      assert((size_t)(indices + in_idx - tx) % GPU_CACHE_LINE_SIZE == 0);
      const auto u = indices[in_idx];
      hop[out_row + rofs] = u;
    }
    tx += stride_x;
  }
}

// Given rows and indptr, computes:
// inrow_indptr[i] = indptr[rows[i]];
// in_degree[i] = indptr[rows[i] + 1] - indptr[rows[i]];
template <typename indptr_t, typename nodes_t>
struct DegreeFunc {
  const nodes_t* rows;
  const indptr_t* indptr;
  indptr_t* in_degree;
  indptr_t* inrow_indptr;
  __host__ __device__ auto operator()(int64_t tIdx) {
    const auto out_row = rows[tIdx];
    const auto indptr_val = indptr[out_row];
    const auto degree = indptr[out_row + 1] - indptr_val;
    in_degree[tIdx] = degree;
    inrow_indptr[tIdx] = indptr_val;
  }
};

struct PairSum {
  template <typename indptr_t>
  __host__ __device__ auto operator()(
      thrust::tuple<indptr_t, indptr_t> a,
      thrust::tuple<indptr_t, indptr_t> b) {
    return thrust::make_tuple(
        thrust::get<0>(a) + thrust::get<0>(b),
        thrust::get<1>(a) + thrust::get<1>(b));
  };
};

template <typename indptr_t>
auto ComputeDegree(
    const indptr_t* const indptr, torch::Tensor nodes, cudaStream_t stream) {
  auto allocator = cuda::BuildAllocator();
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);
  const int64_t num_nodes = nodes.size(0);
  // Read indptr only once in case it is pinned and access is slow.
  auto sliced_indptr = allocator.AllocateStorage<indptr_t>(num_nodes);
  // compute in-degrees
  auto in_degree = allocator.AllocateStorage<indptr_t>(num_nodes + 1);
  thrust::counting_iterator<int64_t> iota(0);
  AT_DISPATCH_INDEX_TYPES(nodes.scalar_type(), "IndexSelectCSCNodes", ([&] {
                            using nodes_t = index_t;
                            thrust::for_each(
                                exec_policy, iota, iota + num_nodes,
                                DegreeFunc<indptr_t, nodes_t>{
                                    nodes.data_ptr<nodes_t>(), indptr,
                                    in_degree.get(), sliced_indptr.get()});
                          }));
  return std::make_pair(std::move(in_degree), std::move(sliced_indptr));
}

template <typename indptr_t, typename indices_t>
std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCCopyIndices(
    torch::Tensor indices, const indptr_t* const sliced_indptr,
    const int64_t num_nodes, const indptr_t* const in_degree,
    const int64_t* const perm, torch::TensorOptions nodes_options,
    torch::ScalarType indptr_scalar_type, cudaStream_t stream) {
  auto allocator = cuda::BuildAllocator();
  thrust::counting_iterator<int64_t> iota(0);

  // Output indptr for the slice indexed by nodes.
  auto sub_indptr =
      torch::empty(num_nodes + 1, nodes_options.dtype(indptr_scalar_type));

  // Actual and modified number of edges.
  indptr_t hop_size, hop_size_aligned;
  auto sub_indptr_aligned = allocator.AllocateStorage<indptr_t>(num_nodes + 1);
  {
    // Returns the actual and modified_indegree as a pair, the
    // latter overestimates the actual indegree for alignment
    // purposes.
    auto modified_in_degree = thrust::make_transform_iterator(
        iota, AlignmentFunc<indptr_t, indices_t>{in_degree, perm, num_nodes});
    auto sub_indptr_pair = thrust::make_zip_iterator(
        sub_indptr.data_ptr<indptr_t>(), sub_indptr_aligned.get());
    thrust::tuple<indptr_t, indptr_t> zero_value{};
    // Compute the prefix sum over actual and modified indegrees.
    size_t workspace_size = 0;
    CUDA_CALL(cub::DeviceScan::ExclusiveScan(
        nullptr, workspace_size, modified_in_degree, sub_indptr_pair, PairSum{},
        zero_value, num_nodes + 1, stream));
    auto temp = allocator.AllocateStorage<char>(workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveScan(
        temp.get(), workspace_size, modified_in_degree, sub_indptr_pair,
        PairSum{}, zero_value, num_nodes + 1, stream));
  }
  // Copy the modified number of edges.
  CUDA_CALL(cudaMemcpyAsync(
      &hop_size_aligned, sub_indptr_aligned.get() + num_nodes,
      sizeof(hop_size_aligned), cudaMemcpyDeviceToHost, stream));
  // Copy the actual total number of edges.
  CUDA_CALL(cudaMemcpyAsync(
      &hop_size, sub_indptr.data_ptr<indptr_t>() + num_nodes, sizeof(hop_size),
      cudaMemcpyDeviceToHost, stream));
  // synchronizes here, we can read hop_size and hop_size_aligned
  CUDA_CALL(cudaStreamSynchronize(stream));
  // Allocate output array with actual number of edges.
  torch::Tensor sub_indices =
      torch::empty(hop_size, nodes_options.dtype(indices.scalar_type()));
  const dim3 block(BLOCK_SIZE);
  const dim3 grid((hop_size_aligned + BLOCK_SIZE - 1) / BLOCK_SIZE);
  // Perform the actual copying, of the indices array into
  // sub_indices in an aligned manner.
  CUDA_KERNEL_CALL(
      _CSRRowWiseOneHopExtractorAlignedKernel, grid, block, 0, stream,
      hop_size_aligned, num_nodes, sliced_indptr,
      sub_indptr.data_ptr<indptr_t>(), sub_indptr_aligned.get(),
      reinterpret_cast<indices_t*>(indices.data_ptr()),
      reinterpret_cast<indices_t*>(sub_indices.data_ptr()), perm);
  return {sub_indptr, sub_indices};
}

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes) {
  // Sorting nodes so that accesses over PCI-e are more regular.
  const auto perm_tensor =
      Sort(nodes, cuda::NumberOfBits(indptr.size(0) - 1)).second;
  auto stream = c10::cuda::getDefaultCUDAStream();
  const int64_t num_nodes = nodes.size(0);

  return AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "UVAIndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        auto [in_degree_ptr, sliced_indptr_ptr] =
            ComputeDegree(indptr.data_ptr<indptr_t>(), nodes, stream);
        auto in_degree = in_degree_ptr.get();
        auto sliced_indptr = sliced_indptr_ptr.get();
        return GNNDLP_DISPATCH_ELEMENT_SIZES(
            indices.element_size(), "UVAIndexSelectCSCCopyIndices", ([&] {
              return UVAIndexSelectCSCCopyIndices<indptr_t, element_size_t>(
                  indices, sliced_indptr, num_nodes, in_degree,
                  perm_tensor.data_ptr<int64_t>(), nodes.options(),
                  indptr.scalar_type(), stream);
            }));
      }));
}

template <typename indptr_t, typename indices_t>
struct IteratorFunc {
  indptr_t* indptr;
  indices_t* indices;
  __host__ __device__ auto operator()(int64_t i) { return indices + indptr[i]; }
};

template <typename indptr_t, typename indices_t>
struct ConvertToBytes {
  const indptr_t* in_degree;
  __host__ __device__ indptr_t operator()(int64_t i) {
    return in_degree[i] * sizeof(indices_t);
  }
};

template <typename indptr_t, typename indices_t>
void IndexSelectCSCCopyIndices(
    const int64_t num_nodes, indices_t* const indices,
    indptr_t* const sliced_indptr, indptr_t* const sub_indptr,
    const indptr_t* const in_degree, indices_t* const sub_indices,
    cudaStream_t stream) {
  auto allocator = cuda::BuildAllocator();
  thrust::counting_iterator<int64_t> iota(0);

  auto input_buffer_it = thrust::make_transform_iterator(
      iota, IteratorFunc<indptr_t, indices_t>{sliced_indptr, indices});
  auto output_buffer_it = thrust::make_transform_iterator(
      iota, IteratorFunc<indptr_t, indices_t>{sub_indptr, sub_indices});
  auto buffer_sizes = thrust::make_transform_iterator(
      iota, ConvertToBytes<indptr_t, indices_t>{in_degree});
  constexpr int64_t max_copy_at_once = std::numeric_limits<int32_t>::max();
  // Performs the copy from indices into sub_indices.
  for (int64_t i = 0; i < num_nodes; i += max_copy_at_once) {
    size_t workspace_size = 0;
    CUDA_CALL(cub::DeviceMemcpy::Batched(
        nullptr, workspace_size, input_buffer_it + i, output_buffer_it + i,
        buffer_sizes + i, std::min(num_nodes - i, max_copy_at_once), stream));
    auto temp = allocator.AllocateStorage<char>(workspace_size);
    CUDA_CALL(cub::DeviceMemcpy::Batched(
        temp.get(), workspace_size, input_buffer_it + i, output_buffer_it + i,
        buffer_sizes + i, std::min(num_nodes - i, max_copy_at_once), stream));
  }
}

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes) {
  auto stream = c10::cuda::getDefaultCUDAStream();
  const int64_t num_nodes = nodes.size(0);
  return AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "IndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        auto [in_degree_ptr, sliced_indptr_ptr] =
            ComputeDegree(indptr.data_ptr<indptr_t>(), nodes, stream);
        auto in_degree = in_degree_ptr.get();
        auto sliced_indptr = sliced_indptr_ptr.get();
        // Output indptr for the slice indexed by nodes.
        torch::Tensor sub_indptr = torch::empty(
            num_nodes + 1, nodes.options().dtype(indptr.scalar_type()));
        {  // Compute the output indptr, sub_indptr.
          size_t workspace_size = 0;
          CUDA_CALL(cub::DeviceScan::ExclusiveSum(
              nullptr, workspace_size, in_degree,
              sub_indptr.data_ptr<indptr_t>(), num_nodes + 1, stream));
          auto allocator = cuda::BuildAllocator();
          auto temp = allocator.AllocateStorage<char>(workspace_size);
          CUDA_CALL(cub::DeviceScan::ExclusiveSum(
              temp.get(), workspace_size, in_degree,
              sub_indptr.data_ptr<indptr_t>(), num_nodes + 1, stream));
        }
        // Number of edges being copied
        indptr_t hop_size;
        CUDA_CALL(cudaMemcpyAsync(
            &hop_size, sub_indptr.data_ptr<indptr_t>() + num_nodes,
            sizeof(hop_size), cudaMemcpyDeviceToHost, stream));
        // blocking read of hop_size
        CUDA_CALL(cudaStreamSynchronize(stream));
        // Allocate output array of size number of copied edges.
        torch::Tensor sub_indices = torch::empty(
            hop_size, nodes.options().dtype(indices.scalar_type()));
        GNNDLP_DISPATCH_ELEMENT_SIZES(
            indices.element_size(), "IndexSelectCSCCopyIndices", ([&] {
              using indices_t = element_size_t;
              IndexSelectCSCCopyIndices<indptr_t, indices_t>(
                  num_nodes, reinterpret_cast<indices_t*>(indices.data_ptr()),
                  sliced_indptr, sub_indptr.data_ptr<indptr_t>(), in_degree,
                  reinterpret_cast<indices_t*>(sub_indices.data_ptr()), stream);
            }));
        return std::make_tuple(sub_indptr, sub_indices);
      }));
}

}  // namespace ops
}  // namespace cuda
}  // namespace gnndlp
