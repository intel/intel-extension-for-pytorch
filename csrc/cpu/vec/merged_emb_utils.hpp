#ifndef MERGEDEMB_UTIL_HPP
#define MERGEDEMB_UTIL_HPP
#include <aten/MergedEmbeddingBag.h>
#include "unroll_helper.hpp"
#include "vec.h"

namespace torch_ipex {
namespace cpu {
using namespace at;
template <typename acc_t, typename scalar_t, typename index_t>
void prepare_ccl_buffer(
    std::vector<Tensor>& idx,
    std::vector<Tensor>& val,
    std::vector<Tensor>& ofs,
    std::vector<EmbeddingRowCache<acc_t>>& cache,
    int64_t world_size,
    int64_t inn_size,
    int64_t emb_dim,
    TensorOptions idx_option,
    TensorOptions val_option) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
  // create tensors for idx, ofs, val
  std::vector<scalar_t*> val_ptr(world_size);
  std::vector<index_t*> idx_ptr(world_size);
  std::vector<int64_t*> ofs_ptr(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    ofs[i] = at::empty({inn_size + 1}, torch::kInt64);
    int64_t* ofs_ptr = ofs[i].data_ptr<int64_t>();
    index_t s = 0;
    for (int64_t n = 0; n < inn_size; ++n) {
      ofs_ptr[n] = s;
      s += cache[i * inn_size + n].size();
    }
    ofs_ptr[inn_size] = s;
    val[i] = at::empty({s, emb_dim}, val_option);
    idx[i] = at::empty({s}, idx_option);
  }
  for (int64_t i = 0; i < world_size; ++i) {
    val_ptr[i] = val[i].data_ptr<scalar_t>(); // EMBACC
    idx_ptr[i] = idx[i].data_ptr<index_t>();
    ofs_ptr[i] = ofs[i].data_ptr<int64_t>();
  }
  // copy into idx, ofs, val
#pragma omp parallel for // collapse(2)
  for (int64_t i = 0; i < inn_size; ++i) {
    for (int64_t o = 0; o < world_size; ++o) {
      size_t j = ofs_ptr[o][i];
      auto emb_cache = cache[o * inn_size + i].cache();
      for (auto& [key, value] : emb_cache) {
        idx_ptr[o][j] = key;
        scalar_t* bufPtr = &val_ptr[o][j * emb_dim];
        kernel::move_ker(bufPtr, value, emb_dim);
        j++;
      }
    }
  }
}

} // namespace cpu
} // namespace torch_ipex
#endif
