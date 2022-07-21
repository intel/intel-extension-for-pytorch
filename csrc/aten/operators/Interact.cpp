#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <torch/custom_class.h>
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;
using namespace sycl;

#ifndef TILE_BN
#define TILE_BN 2 // number of batch
#endif

#define TILE_OUTPUT_COL 4

#ifndef TILE_INPUT_COL_HALF
#define TILE_INPUT_COL_HALF 16
#endif

#define TILE_INPUT_COL_FLOAT (TILE_INPUT_COL_HALF / 2)

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

// inline function
inline int DivUp(int a, int b) {
  return (a + b - 1) / b;
}

void interaction_kernel(
    half* __restrict__ input_mlp,
    half* __restrict__ input,
    half* __restrict__ output,
    const int Batch,
    const int64_t Row,
    const int64_t Col) {
  const int Col_fp = Col / 2;
  const int output_numel = Col + Row * (Row - 1) / 2;
  // size of SLM for index
  const int NX = DivUp(Row - 1, TILE_OUTPUT_COL);
  const int working_set = (NX + 1) * NX / 2;

  constexpr int local_size = 32;
  constexpr int total_local_size = TILE_BN * local_size;
  // Computation Mapping
  sycl::range<2> local{TILE_BN, local_size};
  // Virtual Padding for group mapping to HW
  sycl::range<2> global{Batch, local_size};

  // Computation in GPU
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(h) {
    // Local Memory Management
    auto in_local_data =
        sycl::accessor<float, 3, dpcpp_rw_mode, sycl::access::target::local>(
            sycl::range<3>{TILE_BN, 32, TILE_INPUT_COL_FLOAT}, h);
    auto index =
        sycl::accessor<char, 1, dpcpp_rw_mode, sycl::access::target::local>(
            sycl::range<1>{local_size * 2}, h);
    float* in_mlp_ptr = reinterpret_cast<float*>(input_mlp);
    float* in_ptr = reinterpret_cast<float*>(input);

    h.parallel_for(
        sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
          // variables for loading data to SLM
          int group_id = item.get_group(0); // batch-dimension
          int local_id = item.get_local_linear_id();

          // variables for writing back to device
          int local_bn = item.get_local_id(0);
          int local_row = item.get_local_id(1);
          int global_bn = local_bn + group_id * TILE_BN;
          half* local_data_half =
              reinterpret_cast<half*>(&in_local_data[local_bn][0][0]);
          half* output_bn = output + global_bn * output_numel;

          // make index
          for (int i = 0; i < NX; i += TILE_BN) {
            int ni = local_bn + i;
            if (ni >= NX)
              continue;
            for (int j = 0; j <= ni; j += local_size) {
              int nj = local_row + j;
              if (nj <= ni) {
                int idx = (ni + 1) * ni / 2 + nj;
                index[idx * 2 + 0] = static_cast<char>(ni);
                index[idx * 2 + 1] = static_cast<char>(nj);
              }
            }
          }
          item.barrier(dpcpp_local_fence);

          int item_row =
              static_cast<int>(index[2 * local_row]) * TILE_OUTPUT_COL + 1;
          int item_col =
              static_cast<int>(index[2 * local_row + 1]) * TILE_OUTPUT_COL;

          // each thread computes TILE_OUTPUT_COL * TILE_OUTPUT_COL outputs
          float res[TILE_OUTPUT_COL][TILE_OUTPUT_COL] = {{0.f}};
          int cub_numel = (Row - 1) * TILE_BN * TILE_INPUT_COL_FLOAT;
          // Computation in Col direction
          for (int col = 0; col < Col_fp; col += TILE_INPUT_COL_FLOAT) {
            item.barrier(dpcpp_local_fence);

            // Load one-sub-cube data to SLM
            size_t global_offset = group_id * TILE_BN * Col_fp + col;
            /* #pragma unroll
               TODO: the loop boundary is var.
               const int64_t Row
               int cub_numel = (Row - 1) * TILE_BN * TILE_INPUT_COL_FLOAT;
               DivUp(cub_numel, total_local_size);
               Need to change it to const expression if possible. */
            for (int kk = 0; kk < DivUp(cub_numel, total_local_size); ++kk) {
              int cub_offset = local_id + kk * total_local_size;
              int col_idx = cub_offset % TILE_INPUT_COL_FLOAT; // col
              cub_offset = cub_offset / TILE_INPUT_COL_FLOAT;
              int bn_idx = cub_offset % TILE_BN;
              int row_idx = cub_offset / TILE_BN;
              if (row_idx < Row - 1) {
                // input tensor memory format
                // [Row][bs][TILE_BN][col][TILE_INPUT_COL_FLOAT]
                size_t local_offset =
                    (row_idx)*Batch * Col_fp + bn_idx * Col_fp + col_idx;
                size_t offset = global_offset + local_offset;
                in_local_data[bn_idx][row_idx + 1][col_idx] = in_ptr[offset];
              }
            }
            item.barrier(dpcpp_local_fence);

        // Write the first 128 data (TILE_INPUT_COL_HALF for each cycle k)
#pragma unroll
            for (int kk = 0; kk < TILE_INPUT_COL_FLOAT; kk += local_size) {
              int idx = local_row + kk;
              size_t offset = global_offset + local_bn * Col_fp + idx;
              if (idx < TILE_INPUT_COL_FLOAT) {
                in_local_data[local_bn][0][idx] = in_mlp_ptr[offset];
                output_bn[2 * col + 2 * idx] = local_data_half[2 * idx];
                output_bn[2 * col + 2 * idx + 1] = local_data_half[2 * idx + 1];
              }
            }
            item.barrier(dpcpp_local_fence);

            // Compute
            if (local_row >= working_set)
              continue;

            for (int kk = 0; kk < TILE_INPUT_COL_FLOAT; ++kk) {
              // #pragma unroll
              for (int i = 0; i < TILE_OUTPUT_COL; ++i) {
                int global_row = item_row + i;
                float value_row = in_local_data[local_bn][global_row][kk];
                half* value_row_half = reinterpret_cast<half(*)>(&value_row);
                for (int j = 0; j < TILE_OUTPUT_COL; ++j) {
                  int global_col = item_col + j;
                  float value_col = in_local_data[local_bn][global_col][kk];
                  half* value_col_half = reinterpret_cast<half(*)>(&value_col);

                  // 2 times multiple in here
                  res[i][j] += (float)(value_row_half[0] * value_col_half[0]);
                  res[i][j] += (float)(value_row_half[1] * value_col_half[1]);
                }
              }
            }

          } // col-loop

          // Write the tail 351 results to output
          if (local_row < working_set) {
#pragma unroll
            for (int i = 0; i < TILE_OUTPUT_COL; ++i) {
              int global_row = item_row + i;
              if (global_row >= Row)
                continue;
#pragma unroll
              for (int j = 0; j < TILE_OUTPUT_COL; ++j) {
                int global_col = item_col + j;
                if (global_col < global_row) {
                  size_t offset =
                      Col + (global_row - 1) * global_row / 2 + global_col;
                  output_bn[offset] = half(res[i][j]);
                }
              }
            }
          }
        }); // End of parallel_for
  }; // End of cgf
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace impl

Tensor interaction(Tensor& input_mlp, Tensor& input_emb) {
  RECORD_FUNCTION(
      "dpcpp_interaction", std::vector<c10::IValue>({input_mlp, input_emb}));

  // Currently, this op is specific designed for DLRM inference forward path on
  // Terabyte Dataset
  TORCH_CHECK(
      input_emb.dim() == 3 && input_emb.size(0) == 26 &&
          input_emb.size(2) == 128,
      "(26, batch, 128) tensor expected for input_emb, but got: ",
      input_emb.sizes())
  TORCH_CHECK(
      input_emb.scalar_type() == at::ScalarType::Half, "only fp16 supported")

  auto input_mlp_ = input_mlp;
  if (input_mlp.dim() == 2) {
    input_mlp_ = input_mlp.view({1, input_mlp.size(0), input_mlp.size(1)});
  }
  // only the batch value can be changed
  TORCH_CHECK(
      input_mlp_.dim() == 3 && input_mlp_.size(1) == input_emb.size(1) &&
          input_mlp_.size(2) == 128,
      "(1, batch, 128) tensor expected for input_emb, but got: ",
      input_mlp_.sizes())
  TORCH_CHECK(
      input_emb.scalar_type() == at::ScalarType::Half, "only fp16 supported")

  int64_t tensor_num = input_emb.size(0) + 1;
  int64_t batch = input_emb.size(1);
  int64_t in_features = input_emb.size(2);
  auto out_features = in_features + (tensor_num * (tensor_num - 1) / 2);
  std::vector<int64_t> output_size{batch, out_features};
  Tensor output = at::empty(output_size, input_emb.options());

  impl::interaction_kernel(
      (half*)input_mlp_.data_ptr(),
      (half*)input_emb.data_ptr(),
      (half*)output.data_ptr(),
      batch,
      tensor_num,
      in_features);

  return output;
}

} // namespace AtenIpexTypeXPU
} // namespace at
