/**
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file gnndlp/ops.h
 * @brief GNN dataloading primitives or operators.
 */
#ifndef GNNDLP_OPS_H_
#define GNNDLP_OPS_H_

#include <torch/script.h>

namespace gnndlp {
namespace cuda {
namespace ops {

std::pair<torch::Tensor, torch::Tensor> Sort(
    torch::Tensor input, int num_bits = 0);

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

torch::Tensor UVAIndexSelect(torch::Tensor input, torch::Tensor nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, int num_bits = 0);

}  // namespace ops
}  // namespace cuda
}  // namespace gnndlp

#endif  // GNNDLP_OPS_H_