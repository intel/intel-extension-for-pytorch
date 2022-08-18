#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
#include <core/detail/TensorInfo.h>
#include <intrinsic/intrinsic.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include "comm/RegistrationDeclarations.h"

#include <aten/operators/MemoryAccess.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/SimpleReduce.h"

using namespace dnnl;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {
template <
    int SIMD,
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void group_reduce(
    nd_item_id item_id,
    int sub_group_num,
    accscalar_t& val,
    const local_shared& local_data,
    reduce_op bin_op) {
  auto sg = item_id.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();
  int sg_id = sg.get_group_linear_id();
  int lid = item_id.get_local_id(0);

  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
#pragma unroll
  for (int i = SIMD >> 1; i > 0; i >>= 1) {
    val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
  }
  if (sub_group_num == 1)
    return;

  if (sg_local_id == 0) {
    local_data[sg_id] = val;
  }
  item_id.barrier(dpcpp_local_fence);

  // use one subgroup to reduce WGroupSize/subGroupSize elements
  // into the final result
  // Return a range representing the number of sub-groups in the work-group.
  int range = std::min(SIMD, sub_group_num);
  if (sg_id == 0) {
    if (sg_local_id < range) {
      val = accscalar_t(local_data[sg_local_id]);
      if (sg_local_id + SIMD < sub_group_num) {
        val = bin_op(
            val, static_cast<accscalar_t>(local_data[sg_local_id + SIMD]));
      }
      for (int i = range >> 1; i > 0; i >>= 1) {
        val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
      }
      if (((range >> 1) << 1) < sub_group_num) {
        val += local_data[range - 1];
      }
    }
  }

  // the 0th WI (the 0th WI in the 0th sub_group) generate the final result
  if (lid == 0) {
    local_data[0] = val;
  }
  item_id.barrier(dpcpp_local_fence);
  val = local_data[0];
}

template <
    int vec_size,
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void group_reduce_spatial(
    nd_item_id item_id,
    accscalar_t input[vec_size],
    const local_shared& local_data,
    int block_row,
    reduce_op bin_op) {
  auto local_row_id = item_id.get_local_id(1);
  auto local_col_id = item_id.get_local_id(2);

#pragma unroll(vec_size)
  for (int j = 0; j < vec_size; ++j) {
    local_data[local_row_id][local_col_id][j] = input[j];
  }
  item_id.barrier(dpcpp_local_fence);

  int k = 1;
  while (k < block_row) {
    if (local_row_id % (k << 1) == 0 && local_row_id + k < block_row)
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        local_data[local_row_id][local_col_id][j] = bin_op(
            local_data[local_row_id][local_col_id][j],
            local_data[local_row_id + k][local_col_id][j]);
      }
    k *= 2;
    item_id.barrier(dpcpp_local_fence);
  }
}

// this method help to divide the computation resource for softmax
template <int SIMD>
inline int get_wgroup_size(int vec_size, uint64_t dim_size) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_group_size = dpcppMaxWorkGroupSize(dev_id);
  max_group_size =
      std::min(dim_size / vec_size, static_cast<uint64_t>(max_group_size));
  return (max_group_size + SIMD - 1) / SIMD;
}

// this method help to divide the computation resource for spatial_softmax
template <int vec_size>
static inline void get_wgroup_size_spatial(
    int bs,
    int dim_size,
    int inner_size,
    int& GroupSize,
    int& GroupRow) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int maxWGSize = dpcppMaxWorkGroupSize(dev_id);
  int EUNum = dpcppMaxComputeUnitSize(dev_id);
  constexpr int threadPerEU = 8;
  constexpr int SIMD = 32; // largest SIMD length
  int total_resource = EUNum * threadPerEU * SIMD;

  // set the GroupSize smaller to ensure larger group number
  // smaller GroupSize is friendly to the tail case
  GroupSize = SIMD;
  GroupSize = std::min(GroupSize, int(inner_size));
  auto local_group_num = (inner_size + GroupSize - 1) / GroupSize;

  // enlarge the GroupRow to occupy all the computation resource
  GroupRow = 1;
  while (bs * GroupRow * local_group_num * GroupSize <
         total_resource * vec_size) {
    GroupRow = GroupRow << 1;
    if (GroupRow * SIMD == maxWGSize)
      break;
  }
  GroupRow = std::min(GroupRow, int(dim_size));
}

template <typename scalar_t>
int get_vec_size_helper(
    std::vector<scalar_t*> input,
    int dim_size,
    int inner_size) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto max_wgroup_size = dpcppMaxWorkGroupSize(dev_id);

  int vec_size = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(input[0]));
  for (int i = 1; i < input.size(); ++i) {
    int vec_size_out = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
        getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(input[i]));
    vec_size = std::min(vec_size, vec_size_out);
  }

  // set the min value of in and out as the final vec_size
  int vec_size1 = vec_size;

  // for dispatch_softmax_forward/backward, make sure dim_size % vec_size == 0
  if (inner_size == 1) {
    while (dim_size % vec_size1) {
      vec_size1 = vec_size1 >> 1;
    }
    if (max_wgroup_size * vec_size1 >= dim_size) {
      vec_size = vec_size1;
    }
  }

  // for spatial_softmax_forward/backward, make sure inner_size % vec_size == 0
  if (inner_size != 1) {
    while (inner_size % vec_size) {
      vec_size = vec_size >> 1;
    }
  }
  return vec_size;
}

template <
    int vec_size,
    int SIMD,
    int numel_per_item,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void dispatch_softmax_forward(
    scalar_t* in_data,
    scalar_t* out_data,
    int dim_size,
    int outer_size,
    int sub_group_num) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int loops_end = dim_size / vec_size;
  int local_size = SIMD * sub_group_num;
  sycl::range<1> local_range{local_size};
  sycl::range<1> global_range{outer_size * local_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_max = dpcpp_local_acc_t<accscalar_t>(sub_group_num, cgh);
    auto local_sum = dpcpp_local_acc_t<accscalar_t>(sub_group_num, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>{global_range, local_range},
        [=](cl::sycl::nd_item<1> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          int local_id = item_id.get_local_id(0);
          int group_offset = item_id.get_group(0) * dim_size;
          vec_t* vec_in_data_ptr =
              reinterpret_cast<vec_t*>(in_data + group_offset);
          vec_t* vec_out_data_ptr =
              reinterpret_cast<vec_t*>(out_data + group_offset);

          // load data and get max value
          accscalar_t max_value = std::numeric_limits<accscalar_t>::lowest();
          vec_t reg_in[numel_per_item];
          if (local_id < loops_end) {
            reg_in[0] = vec_in_data_ptr[local_id];
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              max_value = Numerics<accscalar_t>::max(
                  max_value, accscalar_t(reg_in[0][j]));
            }

            if (local_id + local_size < loops_end) {
              reg_in[1] = vec_in_data_ptr[local_id + local_size];
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                max_value = Numerics<accscalar_t>::max(
                    max_value, accscalar_t(reg_in[1][j]));
              }
            }
          }
          group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              max_value,
              local_max,
              [](accscalar_t a, accscalar_t b) {
                return Numerics<accscalar_t>::max(a, b);
              });

          // get sum value
          accscalar_t sum_value = 0;
          if (local_id < loops_end) {
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              sum_value += Numerics<accscalar_t>::exp(reg_in[0][j] - max_value);
              if (local_id + local_size < loops_end) {
                sum_value +=
                    Numerics<accscalar_t>::exp(reg_in[1][j] - max_value);
              }
            }
          }
          group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              sum_value,
              local_sum,
              [](accscalar_t a, accscalar_t b) { return a + b; });
          if (LogSoftMax)
            sum_value = Numerics<accscalar_t>::log(sum_value);
          else
            sum_value = accscalar_t(1) / sum_value;

          // update result
          if (local_id < loops_end) {
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              if (LogSoftMax) {
                reg_in[0][j] =
                    static_cast<scalar_t>(reg_in[0][j] - max_value - sum_value);
              } else {
                reg_in[0][j] = static_cast<scalar_t>(
                    Numerics<accscalar_t>::exp(reg_in[0][j] - max_value) *
                    sum_value);
              }
            }
            vec_out_data_ptr[local_id] = reg_in[0];
            if (local_id + local_size < loops_end) {
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                if (LogSoftMax) {
                  reg_in[1][j] = static_cast<scalar_t>(
                      reg_in[1][j] - max_value - sum_value);
                } else {
                  reg_in[1][j] = static_cast<scalar_t>(
                      Numerics<accscalar_t>::exp(reg_in[1][j] - max_value) *
                      sum_value);
                }
              }
              vec_out_data_ptr[local_id + local_size] = reg_in[1];
            }
          }
        });
  };
  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void softmax_forward_kernel(
    scalar_t* in_data,
    scalar_t* out_data,
    int dim_size,
    int outer_size) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  constexpr int align_bytes = alignof(vec_t);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int local_size = get_wgroup_size<SIMD>(vec_size, dim_size);
  int sub_group_num = local_size / SIMD;
  sycl::range<1> local_range{local_size};
  sycl::range<1> global_range{local_size * outer_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_max = dpcpp_local_acc_t<accscalar_t>(sub_group_num, cgh);
    auto local_sum = dpcpp_local_acc_t<accscalar_t>(sub_group_num, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          int local_id = item_id.get_local_id(0);
          auto group_offset = item_id.get_group(0) * dim_size;
          int start = ((uint64_t)(in_data + group_offset)) % align_bytes /
              sizeof(scalar_t);
          int loops_end = (dim_size + start + vec_size - 1) / vec_size;

          vec_t* vec_in_data_ptr =
              reinterpret_cast<vec_t*>(in_data + group_offset - start);
          vec_t* vec_out_data_ptr =
              reinterpret_cast<vec_t*>(out_data + group_offset - start);

          // get max value
          auto max_value = std::numeric_limits<accscalar_t>::lowest();
          for (int i = local_id; i < loops_end; i += local_size) {
            vec_t in_val = vec_in_data_ptr[i];
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              int linear_idx = i * vec_size + j - start;
              if (linear_idx >= 0 && linear_idx < dim_size) {
                scalar_t in_value = in_val[j];
                max_value = Numerics<accscalar_t>::max(
                    accscalar_t(in_value), max_value);
              }
            }
          }
          group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              max_value,
              local_max,
              [](accscalar_t a, accscalar_t b) {
                return Numerics<accscalar_t>::max(a, b);
              });

          // get sum value
          auto sum_value = accscalar_t(0);
          for (int i = local_id; i < loops_end; i += local_size) {
            vec_t in_val = vec_in_data_ptr[i];
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              int64_t linear_idx = i * vec_size + j - start;
              if (linear_idx >= 0 && linear_idx < dim_size)
                sum_value += Numerics<accscalar_t>::exp(
                    accscalar_t(in_val[j]) - max_value);
            }
          }
          group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              sum_value,
              local_sum,
              [](accscalar_t a, accscalar_t b) { return a + b; });
          if (LogSoftMax)
            sum_value = Numerics<accscalar_t>::log(sum_value);
          else
            sum_value = accscalar_t(1) / sum_value;

          // update result
          for (int i = local_id; i < loops_end; i += local_size) {
            auto remaining = dim_size + start - i * vec_size;
            if ((start > 0 && i == 0) || (remaining < vec_size)) {
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                auto linear_idx = i * vec_size + j - start;
                if (linear_idx >= 0 && linear_idx < dim_size) {
                  if (LogSoftMax)
                    out_data[group_offset + linear_idx] = static_cast<scalar_t>(
                        in_data[group_offset + linear_idx] - max_value -
                        sum_value);
                  else
                    out_data[group_offset + linear_idx] = static_cast<scalar_t>(
                        Numerics<accscalar_t>::exp(
                            in_data[group_offset + linear_idx] - max_value) *
                        sum_value);
                }
              }
            } else {
              vec_t in_val = vec_in_data_ptr[i];
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                if (LogSoftMax)
                  in_val[j] =
                      static_cast<scalar_t>(in_val[j] - max_value - sum_value);
                else
                  in_val[j] = static_cast<scalar_t>(
                      Numerics<accscalar_t>::exp(in_val[j] - max_value) *
                      sum_value);
              }
              vec_out_data_ptr[i] = in_val;
            }
          }
        });
  };

  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void spatial_softmax_forward(
    scalar_t* in_data,
    scalar_t* out_data,
    int dim_size,
    int inner_size,
    int outer_size) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int local_size, block_row;
  get_wgroup_size_spatial<vec_size>(
      outer_size, dim_size, inner_size, local_size, block_row);
  int group_num =
      (inner_size + local_size * vec_size - 1) / (local_size * vec_size);
  sycl::range<3> global_range{outer_size, block_row, group_num * local_size};
  sycl::range<3> local_range{1, block_row, local_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_data = dpcpp_local_acc_t<accscalar_t, dpcpp_rw_mode, 3>(
        sycl::range<3>{block_row, local_size, vec_size}, cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item_id) {
          auto global_col = item_id.get_global_id(2);
          auto local_row_id = item_id.get_local_id(1);
          auto local_col_id = item_id.get_local_id(2);

          auto group_offset = item_id.get_global_id(0) * dim_size * inner_size;
          auto in_ptr = in_data + group_offset;
          auto out_ptr = out_data + group_offset;

          // get max value
          accscalar_t max_value[vec_size] = {
              std::numeric_limits<accscalar_t>::lowest()};
          for (int i = local_row_id; i < dim_size; i += block_row) {
            auto offset = i * inner_size + global_col * vec_size;
            vec_t value = *(reinterpret_cast<vec_t*>(in_ptr + offset));
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j)
              max_value[j] = Numerics<accscalar_t>::max(
                  max_value[j], accscalar_t(value[j]));
          }
          if (block_row > 1) {
            group_reduce_spatial<vec_size, accscalar_t>(
                item_id,
                max_value,
                local_data,
                block_row,
                [](accscalar_t a, accscalar_t b) {
                  return Numerics<accscalar_t>::max(a, b);
                });
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              max_value[j] = local_data[0][local_col_id][j];
            }
            item_id.barrier();
          }

          // get sum value
          accscalar_t sum_value[vec_size] = {accscalar_t(0)};
          for (int i = local_row_id; i < dim_size; i += block_row) {
            auto offset = i * inner_size + global_col * vec_size;
            vec_t value = *(reinterpret_cast<vec_t*>(in_ptr + offset));
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j)
              sum_value[j] +=
                  Numerics<accscalar_t>::exp(value[j] - max_value[j]);
          }
          if (block_row > 1) {
            group_reduce_spatial<vec_size, accscalar_t>(
                item_id,
                sum_value,
                local_data,
                block_row,
                [](accscalar_t a, accscalar_t b) { return a + b; });
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              if (LogSoftMax)
                sum_value[j] =
                    Numerics<accscalar_t>::log(local_data[0][local_col_id][j]);
              else
                sum_value[j] = accscalar_t(1) / local_data[0][local_col_id][j];
            }
          } else {
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              if (LogSoftMax)
                sum_value[j] = Numerics<accscalar_t>::log(sum_value[j]);
              else
                sum_value[j] = accscalar_t(1) / sum_value[j];
            }
          }

          // update result
          if (global_col * vec_size < inner_size) {
            for (int i = local_row_id; i < dim_size; i += block_row) {
              auto offset = i * inner_size + global_col * vec_size;
              vec_t in_val = *(reinterpret_cast<vec_t*>(in_ptr + offset));
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                if (LogSoftMax)
                  in_val[j] = static_cast<scalar_t>(
                      in_val[j] - max_value[j] - sum_value[j]);
                else
                  in_val[j] = static_cast<scalar_t>(
                      Numerics<accscalar_t>::exp(in_val[j] - max_value[j]) *
                      sum_value[j]);
              }
              *(reinterpret_cast<vec_t*>(out_ptr + offset)) = in_val;
            }
          }
        });
  };

  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// softmax = exp(x) / sum(exp(x))
// to ensuare the exp(x) in range of [0, 1], we use exp(x - max_x)
// then softmax = exp(x) / (exp(max_x) * sum(exp(x - max_x)))
template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void vec_softmax_forward_impl(
    scalar_t* input,
    scalar_t* output,
    int outer_size,
    int dim_size,
    int inner_size) {
  if (inner_size == 1) {
    constexpr int SIMD = 32;
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    constexpr int numel_per_item = 2;
    int sub_group_num = (dim_size / vec_size + SIMD - 1) / SIMD;
    sub_group_num = (sub_group_num + numel_per_item - 1) / numel_per_item;
    if (SIMD * sub_group_num <= dpcppMaxWorkGroupSize(dev_id)) {
      dispatch_softmax_forward<
          vec_size,
          SIMD,
          numel_per_item,
          scalar_t,
          accscalar_t,
          LogSoftMax>(input, output, dim_size, outer_size, sub_group_num);
    } else {
      softmax_forward_kernel<vec_size, SIMD, scalar_t, accscalar_t, LogSoftMax>(
          input, output, dim_size, outer_size);
    }
  } else {
    spatial_softmax_forward<vec_size, scalar_t, accscalar_t, LogSoftMax>(
        input, output, dim_size, inner_size, outer_size);
  }
}

template <typename scalar_t, typename accscalar_t, bool LogSoftMax>
void SpatialSoftMaxForward(
    scalar_t* output,
    scalar_t* input,
    int outer_size,
    int dim_size,
    int inner_size) {
  std::vector<scalar_t*> inputs = {input, output};
  int vec_size = get_vec_size_helper(inputs, dim_size, inner_size);

#define VEC_SOFTMAX_FORWARD_IMPL(vec_size)                                 \
  {                                                                        \
    vec_softmax_forward_impl<vec_size, scalar_t, accscalar_t, LogSoftMax>( \
        input, output, outer_size, dim_size, inner_size);                  \
  }
  switch (vec_size) {
    case 8: {
      VEC_SOFTMAX_FORWARD_IMPL(8);
      break;
    }
    case 4: {
      VEC_SOFTMAX_FORWARD_IMPL(4);
      break;
    }
    case 2: {
      VEC_SOFTMAX_FORWARD_IMPL(2);
      break;
    }
    case 1: {
      VEC_SOFTMAX_FORWARD_IMPL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for softmax forward kernel. vec size ",
          vec_size);
  }
#undef VEC_SOFTMAX_FORWARD_IMPL
}

template <
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void dispatch_softmax_backward(
    scalar_t* gradInput,
    const scalar_t* output,
    const scalar_t* gradOutput,
    int dim_size,
    int outer_size,
    int sub_group_num) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int loops_end = dim_size / vec_size;
  int local_size = SIMD * sub_group_num;
  sycl::range<1> local_range{local_size};
  sycl::range<1> global_range{outer_size * local_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_sum = dpcpp_local_acc_t<accscalar_t>(sub_group_num, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          auto local_id = item_id.get_local_id(0);
          auto group_offset = item_id.get_group(0) * dim_size;

          const vec_t* vec_out_data_ptr =
              reinterpret_cast<const vec_t*>(output + group_offset);
          const vec_t* vec_gradout_data_ptr =
              reinterpret_cast<const vec_t*>(gradOutput + group_offset);

          // load data and get max value
          vec_t reg_out = vec_out_data_ptr[local_id];
          vec_t reg_gradout = vec_gradout_data_ptr[local_id];
          accscalar_t sum_value = accscalar_t(0);
          if (local_id < loops_end) {
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              if (LogSoftMax) {
                sum_value += reg_gradout[j];
              } else {
                sum_value += reg_out[j] * reg_gradout[j];
              }
            }
          }
          group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              sum_value,
              local_sum,
              [](accscalar_t a, accscalar_t b) { return a + b; });

          // update result
          if (local_id < loops_end) {
            vec_t* vec_gradin_data_ptr =
                reinterpret_cast<vec_t*>(gradInput + group_offset);
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              if (LogSoftMax) {
                reg_out[j] = static_cast<scalar_t>(
                    reg_gradout[j] -
                    Numerics<accscalar_t>::exp(reg_out[j]) * sum_value);
              } else {
                reg_out[j] = static_cast<scalar_t>(
                    reg_out[j] * (reg_gradout[j] - sum_value));
              }
            }
            vec_gradin_data_ptr[local_id] = reg_out;
          }
        });
  };
  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void softmax_backward_kernel(
    scalar_t* gradInput,
    const scalar_t* output,
    const scalar_t* gradOutput,
    int dim_size,
    int outer_size) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  constexpr int align_bytes = alignof(vec_t);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int local_size = get_wgroup_size<SIMD>(vec_size, dim_size);
  int sub_group_num = local_size / SIMD;
  sycl::range<1> local_range{local_size};
  sycl::range<1> global_range{local_size * outer_size};
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_data = dpcpp_local_acc_t<accscalar_t>(sub_group_num, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          int local_id = item_id.get_local_id(0);
          auto group_offset = item_id.get_group(0) * dim_size;
          int start = ((uint64_t)(output + group_offset)) % align_bytes /
              sizeof(scalar_t);
          int loops_end = (dim_size + start + vec_size - 1) / vec_size;

          vec_t* vec_gradin_data_ptr =
              reinterpret_cast<vec_t*>(gradInput + group_offset - start);
          const vec_t* vec_out_data_ptr =
              reinterpret_cast<const vec_t*>(output + group_offset - start);
          const vec_t* vec_gradout_data_ptr =
              reinterpret_cast<const vec_t*>(gradOutput + group_offset - start);

          // get sum value
          auto sum_value = accscalar_t(0);
          for (int i = local_id; i < loops_end; i += local_size) {
            auto gradout_val = vec_gradout_data_ptr[i];
            if (LogSoftMax) {
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                int64_t linear_idx = i * vec_size + j - start;
                if (linear_idx >= 0 && linear_idx < dim_size) {
                  sum_value += gradout_val[j];
                }
              }
            } else {
              vec_t out_val = vec_out_data_ptr[i];
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                int64_t linear_idx = i * vec_size + j - start;
                if (linear_idx >= 0 && linear_idx < dim_size) {
                  sum_value += out_val[j] * gradout_val[j];
                }
              }
            }
          }
          group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              sum_value,
              local_data,
              [](accscalar_t a, accscalar_t b) { return a + b; });

          // update result
          for (int i = local_id; i < loops_end; i += local_size) {
            // handle the head and tail
            auto remaining = dim_size + start - i * vec_size;
            if ((start > 0 && i == 0) || (remaining < vec_size)) {
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                auto linear_idx = i * vec_size + j - start;
                if (linear_idx >= 0 && linear_idx < dim_size) {
                  auto offset = group_offset + linear_idx;
                  if (LogSoftMax) {
                    gradInput[offset] = gradOutput[offset] -
                        Numerics<accscalar_t>::exp(output[offset]) * sum_value;
                  } else {
                    gradInput[offset] =
                        output[offset] * (gradOutput[offset] - sum_value);
                  }
                }
              }
            } else {
              vec_t grad_val = vec_gradout_data_ptr[i];
              vec_t out_val = vec_out_data_ptr[i];
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                if (LogSoftMax) {
                  out_val[j] = grad_val[j] -
                      Numerics<accscalar_t>::exp(out_val[j]) * sum_value;
                } else {
                  out_val[j] = out_val[j] * (grad_val[j] - sum_value);
                }
              }
              vec_gradin_data_ptr[i] = out_val;
            }
          }
        });
  };

  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void spatial_softmax_backward_kernel(
    scalar_t* gradInput,
    const scalar_t* output,
    const scalar_t* gradOutput,
    int dim_size,
    int inner_size,
    int outer_size) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int local_size, block_row;
  get_wgroup_size_spatial<vec_size>(
      outer_size, dim_size, inner_size, local_size, block_row);
  int group_num =
      (inner_size + local_size * vec_size - 1) / (local_size * vec_size);
  sycl::range<3> global_range{outer_size, block_row, group_num * local_size};
  sycl::range<3> local_range{1, block_row, local_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_data = dpcpp_local_acc_t<accscalar_t, dpcpp_rw_mode, 3>(
        sycl::range<3>{block_row, local_size, vec_size}, cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item_id) {
          auto global_col = item_id.get_global_id(2);
          auto local_row_id = item_id.get_local_id(1);
          auto local_col_id = item_id.get_local_id(2);

          auto group_offset = item_id.get_global_id(0) * dim_size * inner_size;
          auto gradin_ptr = gradInput + group_offset;
          auto out_ptr = output + group_offset;
          auto gradout_ptr = gradOutput + group_offset;

          // get sum value
          accscalar_t sum_value[vec_size] = {accscalar_t(0)};
          for (int i = local_row_id; i < dim_size; i += block_row) {
            auto offset = i * inner_size + global_col * vec_size;
            vec_t gradout_val =
                *(reinterpret_cast<const vec_t*>(gradout_ptr + offset));
            if (LogSoftMax) {
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j)
                sum_value[j] += gradout_val[j];
            } else {
              vec_t out_val =
                  *(reinterpret_cast<const vec_t*>(out_ptr + offset));
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j)
                sum_value[j] += accscalar_t(gradout_val[j]) * out_val[j];
            }
          }
          if (block_row > 1) {
            group_reduce_spatial<vec_size, accscalar_t>(
                item_id,
                sum_value,
                local_data,
                block_row,
                [](accscalar_t a, accscalar_t b) { return a + b; });
#pragma unroll(vec_size)
            for (int j = 0; j < vec_size; ++j) {
              sum_value[j] = local_data[0][local_col_id][j];
            }
          }

          // update result
          if (global_col * vec_size < inner_size) {
            for (int i = local_row_id; i < dim_size; i += block_row) {
              auto offset = i * inner_size + global_col * vec_size;
              vec_t out_val =
                  *(reinterpret_cast<const vec_t*>(out_ptr + offset));
              vec_t gradout_val =
                  *(reinterpret_cast<const vec_t*>(gradout_ptr + offset));
#pragma unroll(vec_size)
              for (int j = 0; j < vec_size; ++j) {
                if (LogSoftMax) {
                  out_val[j] = static_cast<scalar_t>(
                      gradout_val[j] -
                      Numerics<accscalar_t>::exp(out_val[j]) * sum_value[j]);
                } else {
                  out_val[j] = static_cast<scalar_t>(
                      out_val[j] * (gradout_val[j] - sum_value[j]));
                }
              }
              *(reinterpret_cast<vec_t*>(gradin_ptr + offset)) = out_val;
            }
          }
        });
  };

  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void vec_softmax_backward_impl(
    scalar_t* gradInput,
    const scalar_t* output,
    const scalar_t* gradOutput,
    int outer_size,
    int dim_size,
    int inner_size) {
  if (inner_size == 1) {
    constexpr int SIMD = 32;
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int sub_group_num = (dim_size / vec_size + SIMD - 1) / SIMD;
    // if the element number is smaller than max_work_group_size * vec_size,
    // the fast path (dispatch_softmax_backward) will be selected.
    // otherwise, the general path (softmax_backward_kernel) will be selected.
    if (SIMD * sub_group_num <= dpcppMaxWorkGroupSize(dev_id)) {
      dispatch_softmax_backward<
          vec_size,
          SIMD,
          scalar_t,
          accscalar_t,
          LogSoftMax>(
          gradInput, output, gradOutput, dim_size, outer_size, sub_group_num);
    } else {
      softmax_backward_kernel<
          vec_size,
          SIMD,
          scalar_t,
          accscalar_t,
          LogSoftMax>(gradInput, output, gradOutput, dim_size, outer_size);
    }
  } else {
    spatial_softmax_backward_kernel<
        vec_size,
        scalar_t,
        accscalar_t,
        LogSoftMax>(
        gradInput, output, gradOutput, dim_size, inner_size, outer_size);
  }
}

template <typename scalar_t, typename accscalar_t, bool LogSoftMax>
void SpatialSoftMaxBackward(
    scalar_t* gradInput,
    scalar_t* output,
    scalar_t* gradOutput,
    int outer_size,
    int dim_size,
    int inner_size) {
  std::vector<scalar_t*> inputs = {gradInput, output, gradOutput};
  int vec_size = get_vec_size_helper(inputs, dim_size, inner_size);

#define VEC_SOFTMAX_BACKWARD_IMPL(vec_size)                                 \
  {                                                                         \
    vec_softmax_backward_impl<vec_size, scalar_t, accscalar_t, LogSoftMax>( \
        gradInput, output, gradOutput, outer_size, dim_size, inner_size);   \
  }

  switch (vec_size) {
    case 8: {
      VEC_SOFTMAX_BACKWARD_IMPL(8);
      break;
    }
    case 4: {
      VEC_SOFTMAX_BACKWARD_IMPL(4);
      break;
    }
    case 2: {
      VEC_SOFTMAX_BACKWARD_IMPL(2);
      break;
    }
    case 1: {
      VEC_SOFTMAX_BACKWARD_IMPL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for softmax backward kernel. vec size",
          vec_size);
  }
#undef VEC_SOFTMAX_BACKWARD_IMPL
}

} // namespace impl

template <bool LogSoftMax>
Tensor host_softmax(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on DPCPP");
  TORCH_CHECK(
      input_.is_contiguous(),
      "** host_softmax only supports contiguous input tensor");

  Tensor output = at::native::empty_like(input_);
  Tensor input = input_;
  if (input.dim() == 0)
    input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "** dpcpp dim must be non-negative and less than input dimensions");

  // TODO:: handle case larger than 4GB
  if (input.numel() > 0) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        input.scalar_type(),
        "host_softmax",
        [&] {
          auto inner_size = input.stride(dim);
          auto dim_size = input.size(dim);
          auto outer_size = input.numel() / (inner_size * dim_size);
          using accscalar_t = acc_type<scalar_t>;
          impl::SpatialSoftMaxForward<scalar_t, accscalar_t, LogSoftMax>(
              output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              outer_size,
              dim_size,
              inner_size);
        });
  }
  return output;
}

template <bool LogSoftMax>
Tensor host_softmax_backward(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on DPCPP");
  TORCH_CHECK(
      grad_.is_contiguous(),
      "** host_softmax_backward only supports contiguous grad tensor");
  TORCH_CHECK(
      output_.is_contiguous(),
      "** host_softmax_backward only supports contiguous output tensor");

  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  Tensor gI = at::empty_like(grad_);

  if (output_.numel() == 0) {
    return gI;
  }

  Tensor grad = grad_;
  if (grad.dim() == 0)
    grad = grad.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  Tensor output = output_;
  if (output.dim() == 0)
    output = output.view(1);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "host_softmax_backward",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        auto inner_size = output.stride(dim);
        auto dim_size = output.size(dim);
        auto outer_size = output.numel() / (dim_size * inner_size);
        impl::SpatialSoftMaxBackward<scalar_t, accscalar_t, LogSoftMax>(
            gI.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size);
      });
  return gI;
}

// We now use DPCPP softmax fwd kernel instead of oneDNN softmax fwd kernel
Tensor _softmax(
    const Tensor& input_,
    const int64_t dim,
    const bool half_to_float) {
  checkBackend("_softmax", {input_}, Backend::XPU);

  // 1.check the tensors type are supported by oneDNN or not
  // 2.check the tensors are contiguous or not
  // 3.check the tensors are blocked format or not
  // when satify the aformentioned two conditions,
  // the oneDNN path will be selected,
  // all the other cases will go to DPCPP path
  if (xpu::oneDNN::softmax_valid(input_) &&
      xpu::oneDNN::is_onednn_layout(input_)) {
    return xpu::oneDNN::softmax(input_, dim, half_to_float);
  } else {
    Tensor input = to_plain_if_needed(input_).contiguous();
    return host_softmax<false>(input, dim, half_to_float);
  }
}

Tensor _softmax_backward_data(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim,
    const Tensor& input) {
  bool half_to_float = grad_.scalar_type() != input.scalar_type();
  if (half_to_float) {
    TORCH_CHECK(
        !half_to_float,
        "softmax backward with half to float "
        "conversion is not supported on DPCPP");
  }

  // 1.check the tensors type are supported by oneDNN or not
  // 2.check the tensors are contiguous or not
  // 3.check the tensors are blocked format or not
  // when satify the aformentioned conditions,
  // the oneDNN path will be selected,
  // all the other cases will go to DPCPP path
  if (xpu::oneDNN::softmax_backward_valid(grad_, output_, input) &&
      IPEX_ANY(xpu::oneDNN::is_onednn_layout, grad_, output_)) {
    return xpu::oneDNN::softmax_backward(grad_, output_, dim, half_to_float);
  } else {
    auto grad = to_plain_if_needed(grad_).contiguous();
    auto output = to_plain_if_needed(output_).contiguous();
    return host_softmax_backward<false>(grad, output, dim, half_to_float);
  }
}

Tensor _log_softmax(const Tensor& self_, int64_t dim, bool half_to_float) {
  Tensor self = self_.contiguous();
  return host_softmax<true>(self, dim, half_to_float);
}

Tensor _log_softmax_backward_data(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim,
    const Tensor& input) {
  bool half_to_float = grad_.scalar_type() != input.scalar_type();
  if (half_to_float) {
    TORCH_INTERNAL_ASSERT(
        !half_to_float,
        "softmax with half to float conversion is not supported on DPCPP");
  }

  auto grad = grad_.contiguous();
  auto output = output_.contiguous();
  return host_softmax_backward<true>(grad, output, dim, half_to_float);
}

} // namespace AtenIpexTypeXPU
} // namespace at
