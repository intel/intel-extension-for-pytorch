#pragma once

#include <ATen/Tensor.h>
#include <c10/core/CPUAllocator.h>
#include <omp.h>
#include <torch/extension.h>

namespace torch_ipex {
namespace cpu {

using namespace at;

enum PoolingMode { SUM = 0, MEAN = 1 };

struct BatchedHyperCompressedSparseColumn {
  // A data structure to describe how sparse grads got by MergeEmbedingBag
  // should be used to update weights/tables
  // Use 1 table as an  example,
  // We have a table T of shape (5, 2) (num_of_features = 5, feature size = 2,
  // mode=MEAN) If indices = [0, 2, 0, 3, 0, 4, 5], offsets = [0, 2, 4, 7] Then
  // the outputs (or grads for output) should be shape of (3, 2) The first row
  // is the mean value of  T[0] + T[2] The second row is the mean value of  T[0]
  // + T[3] The third row is the mean value of  T[0] + T[4] + T[5]
  //
  // Then for backward, since each index in indices have contributed to output
  // The corresponding weight rows should be updated by grads
  // For first row grad[0], the output is contribute by T[0], T[2] and since
  // mode=MEAN The update for first row should be  T[0] += lr * 0.5* grad[0], T2
  // += lr * 0.5 * grad[0] For second row: T[0] += lr * 0.5* grad[1], T[3] += lr
  // * 0.5* grad[1], For third row: T[0] += lr * 0.33 * grad[1], T[4] += lr *
  // 0.33 * grad[2], T[5] += lr * 0.33 * grad[2] Summary with the weights idx:
  // T[0] += lr * (0.5grad[0] + 0.5grad[1] + 0.33grad[2])
  // T[2] += lr * (0.5grad[0])
  // T[3] += lr * (0.5grad[1])
  // T[4] += lr * (0.33grad[2])
  // T[5] += lr * (0.33grad[2])
  int num_tables; // # of tables

  // Num different indices, it is 5 (0, 2, 3, 4, 5) in the example above
  int uniq_indices = 0;

  // Length of uniq_indices + 1, to record the index for sorted indices
  // sort ([0, 2, 0, 3, 0, 4, 5]) -> [0, 0, 0, 2, 3, 4, 5] -> [0, 3, 4, 5, 6, 7]
  // which "7" =  num of indices
  int* segment_ptr = nullptr;

  // Length of uniq_indices, to record the row ids in weights [0, 2, 3, 4, 5]
  int* segment_indices = nullptr;

  // Length num of indices (7 in the example above), to record the output rows
  // for each indices [0, 1, 2, 0, 1,  2, 2]
  int* output_row_indices = nullptr;

  // Length num of indices, to record the weights while update 1 row
  // [0.5, 0.5, 0.33, 0.5, 0.5, 0.33, 0.33]
  float* weights = nullptr; // length column_ptr[table_ptr[T]]

  ~BatchedHyperCompressedSparseColumn() {
    Allocator* allocator = c10::GetAllocator(c10::DeviceType::CPU);
    if (segment_ptr) {
      allocator->raw_deallocate(segment_ptr);
      allocator->raw_deallocate(segment_indices);
      allocator->raw_deallocate(output_row_indices);
    }
    if (weights) {
      allocator->raw_deallocate(weights);
    }
  };
};

void sort_based_batched_csr2csc_opt(
    BatchedHyperCompressedSparseColumn& batched_csc,
    int B,
    const Tensor& offsets,
    const Tensor& indices,
    std::vector<int64_t> pooling_modes,
    int64_t max_embeddings);

} // namespace cpu
} // namespace torch_ipex
