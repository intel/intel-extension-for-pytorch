#include "csr2csc.h"
#include "radix_sort.h"

namespace torch_ipex {
namespace cpu {
void sort_based_batched_csr2csc_opt(
    BatchedHyperCompressedSparseColumn& batched_csc,
    int B,
    const Tensor& offsets,
    const Tensor& indices,
    std::vector<int64_t> pooling_modes,
    int64_t max_embeddings) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(__FUNCTION__, std::vector<c10::IValue>({}));
#endif
  TensorAccessor<int64_t, 1> offsets_data = offsets.accessor<int64_t, 1>();
  TensorAccessor<int64_t, 1> batched_csr_indices =
      indices.accessor<int64_t, 1>();
  int num_tables = pooling_modes.size();
  batched_csc.num_tables = num_tables;
  int64_t n_indices = indices.numel();
  int64_t n_offsets = offsets.numel() - 1;
  batched_csc.output_row_indices =
      (int*)c10::alloc_cpu(n_indices * sizeof(int));
  for (auto pooling_mode : pooling_modes) {
    if (pooling_mode == MEAN) {
      batched_csc.weights = (float*)c10::alloc_cpu(n_indices * sizeof(float));
      break;
    }
  }

  auto get_table_id = [&](int n) { return n / B; };

  Key_Value_Weight_Tuple<int>* tmpBuf =
      (Key_Value_Weight_Tuple<int>*)c10::alloc_cpu(
          (n_indices) * sizeof(Key_Value_Weight_Tuple<int>));
  Key_Value_Weight_Tuple<int>* tmpBuf1 =
      (Key_Value_Weight_Tuple<int>*)c10::alloc_cpu(
          (n_indices) * sizeof(Key_Value_Weight_Tuple<int>));
#pragma omp parallel for
  for (int n = 0; n < n_offsets; ++n) {
    int64_t pool_begin = offsets_data[n];
    int64_t pool_end = offsets_data[n + 1];
    auto table_id = get_table_id(n);
    float scale_factor =
        pooling_modes[table_id] == MEAN ? 1.0 / (pool_end - pool_begin) : 1;
    for (int64_t p = pool_begin; p < pool_end; ++p) {
      std::get<0>(tmpBuf[p]) = batched_csr_indices[p];
      std::get<1>(tmpBuf[p]) = n;
      if (batched_csc.weights) {
        std::get<2>(tmpBuf[p]) = scale_factor;
      }
    }
  }

  Key_Value_Weight_Tuple<int>* sorted_col_row_index_pairs =
      radix_sort_parallel<int>(
          &tmpBuf[0], &tmpBuf1[0], n_indices, max_embeddings);
  int max_thds = omp_get_max_threads();
  int num_uniq[max_thds][64];

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    num_uniq[tid][0] = 0;
#pragma omp for schedule(static)
    for (int i = 1; i < n_indices; i++) {
      if (std::get<0>(sorted_col_row_index_pairs[i]) !=
          std::get<0>(sorted_col_row_index_pairs[i - 1]))
        num_uniq[tid][0]++;
    }
  }
  num_uniq[0][0] += 1;
  for (int i = 1; i < max_thds; i++)
    num_uniq[i][0] += num_uniq[i - 1][0];
  int U = num_uniq[max_thds - 1][0];

  batched_csc.segment_ptr = (int*)c10::alloc_cpu(U * sizeof(int));
  batched_csc.segment_indices = (int*)c10::alloc_cpu(U * sizeof(int));
  batched_csc.output_row_indices =
      (int*)c10::alloc_cpu(n_indices * sizeof(int));

  batched_csc.segment_ptr[0] = 0;
  batched_csc.output_row_indices[0] =
      std::get<1>(sorted_col_row_index_pairs[0]) % B;
  batched_csc.segment_indices[0] = std::get<0>(sorted_col_row_index_pairs[0]);
  if (batched_csc.weights) {
    batched_csc.weights[0] = std::get<2>(sorted_col_row_index_pairs[0]);
  }
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int* tstart =
        (tid == 0 ? batched_csc.segment_indices + 1
                  : batched_csc.segment_indices + num_uniq[tid - 1][0]);
    int* t_offs =
        (tid == 0 ? batched_csc.segment_ptr + 1
                  : batched_csc.segment_ptr + num_uniq[tid - 1][0]);

#pragma omp for schedule(static)
    for (int i = 1; i < n_indices; i++) {
      batched_csc.output_row_indices[i] =
          std::get<1>(sorted_col_row_index_pairs[i]) % B;
      if (batched_csc.weights) {
        batched_csc.weights[i] = std::get<2>(sorted_col_row_index_pairs[i]);
      }
      if (std::get<0>(sorted_col_row_index_pairs[i]) !=
          std::get<0>(sorted_col_row_index_pairs[i - 1])) {
        *tstart = std::get<0>(sorted_col_row_index_pairs[i]);
        *t_offs = i;
        tstart++;
        t_offs++;
      }
    }
  }
  batched_csc.uniq_indices += U;
  batched_csc.segment_ptr[U] = n_indices;
  c10::free_cpu(tmpBuf);
  c10::free_cpu(tmpBuf1);
}

} // namespace cpu
} // namespace torch_ipex
