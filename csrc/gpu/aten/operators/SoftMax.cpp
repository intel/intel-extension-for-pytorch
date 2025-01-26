#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
#include <core/detail/TensorInfo.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "comm/RegistrationDeclarations.h"

//#include <aten/operators/MemoryAccess.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/SimpleReduce.h"
#include "utils/ComputeEngine.h"
#include "utils/CustomOperatorRegistration.h"
/*
softmax forward and backward follow the same optimization routine, we take
forward as an example here. softmax = exp(x) / sum(exp(x)) to ensuare the exp(x)
in range of [0, 1], we use exp(x - max) to replace exp(x) then softmax = exp(x -
max) / sum(exp(x - max)) Any input tensor for softmax can be viewed as
[outer_size, dim_size, inner_size] If the softmax axis is the last dim (dim=-1),
then the inner_size = 1, and the input tensor can be viewed as [outer_size,
dim_size, 1] If the somftmax axis is not the last dim (dim!=-1), then the input
tensor can be viewed as [outer_size, dim_size, inner_size] Genearally, three
steps are needed to get the softmax result
1. read data and get the max value
2. read data and get the sum value
3. read data and compute element-wise result


***************************************************************************************
dispatch_softmax_forward_kernel is the fast path for softmax forward with
inner_size=1, by reading the input elements only once and keep them in the
registers. When MaxWorkGroupSize (1024 on PVC and ATSM) * INNER_LOOP >=
dim_size, this fast path can be selected

The main steps includes:
1. each workitem load INNER_LOOP [NUM][vec_size] numbers of elements
2. Get max/sum value along dim_size
   if dim_size < 16 and dim_size * sizeof(scalar_t) < sizeof(float16), reduce
happened internal one workitem, otherwise reduced happened internal one subgroup
or group and will be processed by group_reduce function.
3. compute and store the softmax result into the global memory

Configs:
   The vec_size is decided by datatype and dim_size:
   double && dim_size % 2 == 0: vec_size = 2 (sizeof(float4)/sizeof(double))
   float  && dim_size % 4 == 0: vec_size = 4 (sizeof(float4)/sizeof(float))
   bf16/fp16 && dim_size % 8 == 0: vec_size = 8
(sizeof(float4)/sizeof(bf16/fp16)) otherwise, vec_size = 1

   Initial INNER_LOOP = sizeof(float8) / sizeof(scalar_t)
   if dim_size < INNER_LOOP * SIMD16,
      INNER_LOOP = sizeof(float8) / sizeof(scalar_t) * 2
      SIMD=16
   otherwise,
      INNER_LOOP = sizeof(float8) / sizeof(scalar_t)
      SIMD=32

   WorkGroupSize equals to multi times of SIMD covering dim_size / INNER_LOOP
   WorkGroupNum equals to outer_size
   If WorkGroupNum is too large and WorkGroupSize is small, we will enlarge
WorkGroupSize to process multiple dim_size elements.


***************************************************************************************
softmax_forward_kernel is the reference path for softmax forward with
inner_size=1 input data cannot be reused and must be loaded in each step
including: get max, get sum, update result

   Configs:
   double: vec_size = 2 (sizeof(float4)/sizeof(double))
   float: vec_size = 4 (sizeof(float4)/sizeof(float))
   bf16/fp16: vec_size = 8 (sizeof(float4)/sizeof(bf16/fp16))
   The non-alignment will be handled in this kernel and max_vec_size will always
be selected.

   WorkGroupSize equals the MaxWorkGroupSize
   WorkGroupNum equals to outer_size


***************************************************************************************
spatial_softmax_forward used for softmax forward with inner_size != 1
   input tensor [outer_size, dim_size, inner_size]
   workitem space [outer_size] [DIM_NUM][dim_size/DIM_NUM]
[INNER_NUM][inner_size/INNER_NUM]
*/

using namespace dnnl;
using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;
using namespace torch_ipex::xpu::oneDNN;

#define MIN_WG_NUM 32768

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
    int lid_row,
    int sub_group_num,
    accscalar_t& val,
    accscalar_t init,
    const local_shared& local_data,
    reduce_op bin_op) {
  auto sg = item_id.get_sub_group();

  // dynamic get SIMD width result in big performance drop
  // uint32_t SIMD = sg.get_local_range()[0];
#pragma unroll
  for (int i = 1; i < SIMD; i <<= 1) {
    val = bin_op(
        val, static_cast<accscalar_t>(sycl::shift_group_left(sg, val, i)));
  }
  if (sub_group_num == 1) {
    val = sycl::group_broadcast(sg, val, 0);
    return;
  }
  uint32_t sg_local_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();
  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
  int idx = sg_id - (lid_row * sub_group_num);
  if (sg_local_id == 0) {
    local_data[lid_row][idx] = val;
  }
  item_id.barrier(dpcpp_local_fence);

  // use one subgroup to reduce WGroupSize/subGroupSize elements
  // into the final result
  if (idx == 0) {
    val = init;
    if (sg_local_id < sub_group_num) {
      val = accscalar_t(local_data[lid_row][sg_local_id]);
    }
    for (int i = sg_local_id + SIMD; i < sub_group_num; i += SIMD) {
      val = bin_op(val, static_cast<accscalar_t>(local_data[lid_row][i]));
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      val = bin_op(
          val, static_cast<accscalar_t>(sycl::shift_group_left(sg, val, i)));
      if (i >= ((sub_group_num + 1) >> 1))
        break;
    }

    // the 0th WI (the 0th WI in the 0th sub_group) generate the final result
    if (sg_local_id == 0) {
      local_data[lid_row][0] = val;
    }
  }

  item_id.barrier(dpcpp_local_fence);
  val = local_data[lid_row][0];
} // namespace impl

template <int SIMD, int vec_size, int NUM>
static inline void get_wgroup_size(
    uint64_t dim_size,
    int outer_size,
    int& sub_group_num,
    int& range,
    int& global_size_row,
    int& local_size_row,
    int& local_size_col) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int maxWGSize = dpcppMaxWorkGroupSize(dev_id);

  int local_size = (dim_size + NUM * vec_size - 1) / (NUM * vec_size);
  local_size = std::min(local_size, maxWGSize);
  // select the local_size_col to cover the dim_size
  sub_group_num = (local_size + SIMD - 1) / SIMD;
  local_size_col = sub_group_num * SIMD;
  // if one workitem [NUM][vec_size] can cover the dim_size number of elements
  // local_size_col will be 1
  if (dim_size <= vec_size * NUM) {
    local_size_col = 1;
    local_size_row = SIMD;
    global_size_row = (outer_size + local_size_row - 1) / local_size_row;
    return;
  }

  // if outer_size is too large and local_size_col is small,
  // then use one workgroup to handle multi rows (dim_size)
  local_size_row = 1;
  global_size_row = outer_size;
  while ((global_size_row >> 1) > MIN_WG_NUM &&
         (local_size_row << 1) * local_size_col <= maxWGSize &&
         !(global_size_row % 2)) {
    global_size_row = global_size_row >> 1;
    local_size_row = local_size_row << 1;
  }

  // compute the reduce range
  range = SIMD;
  while (sub_group_num <= (range >> 1)) {
    range = range >> 1;
  }
}

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    bool is_masked,
    typename calc_t,
    typename vec_t>
struct DispatchSoftmaxForwardKernelFunctor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    if (local_size == 1 && item_id.get_global_id(0) >= outer_size)
      return;

    uint32_t lid_row = 0;
    uint32_t lid_col = item_id.get_local_id(0);
    uint32_t group_offset = item_id.get_group(0) * dim_size;
    if (local_size_row != 1) {
      lid_row = item_id.get_local_id(0) / local_size;
      lid_col = item_id.get_local_id(0) % local_size;
      group_offset =
          (item_id.get_group(0) * local_size_row + lid_row) * dim_size;
    }
    vec_t reg_in[outer_loop];
    vec_t reg_mask[outer_loop];
    auto lid_offset = lid_col * vec_size;
    auto local_stride = local_size * vec_size;

    // load data and get max value
    accscalar_t max_value = std::numeric_limits<accscalar_t>::lowest();
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size)
        break;

      reg_in[i] = *(reinterpret_cast<vec_t*>(in_data + group_offset + index));
      if constexpr (is_masked) {
        auto vec_offset = group_offset + index;
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          auto linear_idx = vec_offset + j;
          auto mask_offset = input_calc.get(linear_idx)[1];
          reg_mask[i][j] = mask_data[mask_offset];
        }
      }
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (is_masked) {
          if (reg_mask[i][j]) {
            reg_in[i][j] = neginf;
          }
        }
        max_value =
            Numerics<accscalar_t>::max(max_value, accscalar_t(reg_in[i][j]));
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          max_value,
          std::numeric_limits<accscalar_t>::lowest(),
          local_max,
          [](accscalar_t a, accscalar_t b) {
            return Numerics<accscalar_t>::max(a, b);
          });
    }

    // get sum value
    accscalar_t sum_value = 0;
#pragma unroll(outer_loop)
    for (int i = 0;
         i < outer_loop && ((i * local_stride + lid_offset) < dim_size);
         ++i) {
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value += Numerics<accscalar_t>::exp(reg_in[i][j] - max_value);
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          sum_value,
          accscalar_t(0),
          local_sum,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    if constexpr (LogSoftMax)
      sum_value = Numerics<accscalar_t>::log(sum_value);
    else if (sum_value != 0)
      sum_value = accscalar_t(1) / sum_value;

      // update result
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size)
        break;

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (LogSoftMax) {
          reg_in[i][j] =
              static_cast<scalar_t>(reg_in[i][j] - max_value - sum_value);
        } else if (sum_value == 0) {
          reg_in[i][j] = nan;
        } else {
          reg_in[i][j] = static_cast<scalar_t>(
              Numerics<accscalar_t>::exp(reg_in[i][j] - max_value) * sum_value);
        }
      }
      *(reinterpret_cast<vec_t*>(out_data + group_offset + index)) = reg_in[i];
    }
  }
  DispatchSoftmaxForwardKernelFunctor(
      scalar_t* in_data_,
      scalar_t* out_data_,
      int dim_size_,
      int outer_size_,
      bool* mask_data_,
      calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_,
      scalar_t neginf_,
      scalar_t nan_,
      dpcpp_local_acc_t<accscalar_t, 2> local_max_,
      dpcpp_local_acc_t<accscalar_t, 2> local_sum_)
      : in_data(in_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        outer_size(outer_size_),
        mask_data(mask_data_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_),
        neginf(neginf_),
        nan(nan_),
        local_max(local_max_),
        local_sum(local_sum_) {}

 private:
  scalar_t* in_data;
  scalar_t* out_data;
  int dim_size;
  int outer_size;
  bool* mask_data;
  calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
  scalar_t neginf;
  scalar_t nan;
  dpcpp_local_acc_t<accscalar_t, 2> local_max;
  dpcpp_local_acc_t<accscalar_t, 2> local_sum;
};

// replace std::nullptr_t to avoid kernel name in std namespace
struct DummyFunctor {};

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    bool is_masked = false,
    typename calc_t = decltype(nullptr)>
void dispatch_softmax_forward_kernel(
    scalar_t* in_data,
    scalar_t* out_data,
    int dim_size,
    int outer_size,
    bool* mask_data = nullptr,
    calc_t input_calc = nullptr) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int sub_group_num, global_size_row, local_size_row, range, local_size;
  get_wgroup_size<SIMD, vec_size, outer_loop>(
      dim_size,
      outer_size,
      sub_group_num,
      range,
      global_size_row,
      local_size_row,
      local_size);
  sycl::range<1> local_range{local_size_row * local_size};
  sycl::range<1> global_range{global_size_row * local_size_row * local_size};
  scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  scalar_t nan = std::numeric_limits<accscalar_t>::quiet_NaN();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_max = dpcpp_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);
    auto local_sum = dpcpp_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);

    if constexpr (is_masked) {
      DispatchSoftmaxForwardKernelFunctor<
          INNER_LOOP,
          vec_size,
          SIMD,
          scalar_t,
          accscalar_t,
          IndexType,
          LogSoftMax,
          outer_loop,
          is_masked,
          calc_t,
          vec_t>
          kfn(in_data,
              out_data,
              dim_size,
              outer_size,
              mask_data,
              input_calc,
              sub_group_num,
              global_size_row,
              local_size_row,
              range,
              local_size,
              neginf,
              nan,
              local_max,
              local_sum);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>{global_range, local_range}, kfn);
    } else {
      DummyFunctor dummy;
      DispatchSoftmaxForwardKernelFunctor<
          INNER_LOOP,
          vec_size,
          SIMD,
          scalar_t,
          accscalar_t,
          IndexType,
          LogSoftMax,
          outer_loop,
          is_masked,
          DummyFunctor,
          vec_t>
          kfn(in_data,
              out_data,
              dim_size,
              outer_size,
              mask_data,
              dummy,
              sub_group_num,
              global_size_row,
              local_size_row,
              range,
              local_size,
              neginf,
              nan,
              local_max,
              local_sum);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>{global_range, local_range}, kfn);
    }
  };
  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    typename inp_offset_calc_t,
    typename vec_t>
struct DispatchSoftmaxForwardAddKernelFunctor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    if (local_size == 1 && item_id.get_global_id(0) >= outer_size)
      return;

    uint32_t lid_row = 0;
    uint32_t lid_col = item_id.get_local_id(0);
    uint32_t group_offset = item_id.get_group(0) * dim_size;
    if (local_size_row != 1) {
      lid_row = item_id.get_local_id(0) / local_size;
      lid_col = item_id.get_local_id(0) % local_size;
      group_offset =
          (item_id.get_group(0) * local_size_row + lid_row) * dim_size;
    }
    vec_t reg_in[outer_loop];
    vec_t reg_tmp;
    auto lid_offset = lid_col * vec_size;
    auto local_stride = local_size * vec_size;
    // load data and get max value
    accscalar_t max_value = std::numeric_limits<accscalar_t>::lowest();
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size)
        break;

      auto group_batch_offset = group_offset + index;
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        auto linear_offset = group_batch_offset + j;
        scalar_t input_value = in_data[input_calc.get(linear_offset)[0]];
        scalar_t other_value = other_data[input_calc.get(linear_offset)[1]];
        reg_in[i][j] = input_value + alpha * other_value;
      }

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        max_value =
            Numerics<accscalar_t>::max(max_value, accscalar_t(reg_in[i][j]));
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          max_value,
          std::numeric_limits<accscalar_t>::lowest(),
          local_max,
          [](accscalar_t a, accscalar_t b) {
            return Numerics<accscalar_t>::max(a, b);
          });
    }

    // get sum value
    accscalar_t sum_value = 0;
#pragma unroll(outer_loop)
    for (int i = 0;
         i < outer_loop && ((i * local_stride + lid_offset) < dim_size);
         ++i) {
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value += Numerics<accscalar_t>::exp(reg_in[i][j] - max_value);
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          sum_value,
          accscalar_t(0),
          local_sum,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    if constexpr (LogSoftMax)
      sum_value = Numerics<accscalar_t>::log(sum_value);
    else
      sum_value = accscalar_t(1) / sum_value;

      // update result
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size)
        break;

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (LogSoftMax) {
          reg_in[i][j] =
              static_cast<scalar_t>(reg_in[i][j] - max_value - sum_value);
        } else {
          reg_in[i][j] = static_cast<scalar_t>(
              Numerics<accscalar_t>::exp(reg_in[i][j] - max_value) * sum_value);
        }
      }
      *(reinterpret_cast<vec_t*>(out_data + group_offset + index)) = reg_in[i];
    }
  }
  DispatchSoftmaxForwardAddKernelFunctor(
      scalar_t* in_data_,
      scalar_t* other_data_,
      scalar_t* out_data_,
      int dim_size_,
      scalar_t alpha_,
      int outer_size_,
      int other_outer_size_,
      inp_offset_calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_,
      int other_offset_,
      dpcpp_local_acc_t<accscalar_t, 2> local_max_,
      dpcpp_local_acc_t<accscalar_t, 2> local_sum_)
      : in_data(in_data_),
        other_data(other_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        alpha(alpha_),
        outer_size(outer_size_),
        other_outer_size(other_outer_size_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_),
        other_offset(other_offset_),
        local_max(local_max_),
        local_sum(local_sum_) {}

 private:
  scalar_t* in_data;
  scalar_t* other_data;
  scalar_t* out_data;
  int dim_size;
  scalar_t alpha;
  int outer_size;
  int other_outer_size;
  inp_offset_calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
  int other_offset;
  dpcpp_local_acc_t<accscalar_t, 2> local_max;
  dpcpp_local_acc_t<accscalar_t, 2> local_sum;
};

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    typename inp_offset_calc_t>
void dispatch_softmax_forward_add_kernel(
    scalar_t* in_data,
    scalar_t* other_data,
    scalar_t* out_data,
    int dim_size,
    scalar_t alpha,
    int outer_size,
    int other_outer_size,
    inp_offset_calc_t input_calc) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int sub_group_num, global_size_row, local_size_row, range, local_size;
  get_wgroup_size<SIMD, vec_size, outer_loop>(
      dim_size,
      outer_size,
      sub_group_num,
      range,
      global_size_row,
      local_size_row,
      local_size);
  sycl::range<1> local_range{local_size_row * local_size};
  sycl::range<1> global_range{global_size_row * local_size_row * local_size};
  auto other_offset = other_outer_size * dim_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_max = dpcpp_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);
    auto local_sum = dpcpp_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);

    DispatchSoftmaxForwardAddKernelFunctor<
        INNER_LOOP,
        vec_size,
        SIMD,
        scalar_t,
        accscalar_t,
        IndexType,
        LogSoftMax,
        outer_loop,
        inp_offset_calc_t,
        vec_t>
        kfn(in_data,
            other_data,
            out_data,
            dim_size,
            alpha,
            outer_size,
            other_outer_size,
            input_calc,
            sub_group_num,
            global_size_row,
            local_size_row,
            range,
            local_size,
            other_offset,
            local_max,
            local_sum);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>{global_range, local_range}, kfn);
  };
  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename accscalar_t, bool LogSoftMax>
Tensor& MaskedSoftMaxForward(
    Tensor& output,
    Tensor& input,
    int dim,
    const Tensor mask) {
  auto inner_size = input.stride(dim);
  auto dim_size = input.size(dim);
  auto outer_size = input.numel() / (inner_size * dim_size);

  constexpr int float4_size = sizeof(float) * 4;
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int INNER_LOOP = max_vec_size * 2;

  // decide vec_size: max_vec_size or 1
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, max_vec_size>;
  constexpr int align_bytes = alignof(vec_t);
  int input_start =
      ((uint64_t)input.data_ptr()) % align_bytes / sizeof(scalar_t);
  int output_start =
      ((uint64_t)output.data_ptr()) % align_bytes / sizeof(scalar_t);

  // decide indexing range: uint32_t (4GB) or uint64_t (>4GB)
  bool can_use_32bit_index =
      canUse32BitIndexMath(input) && canUse32BitIndexMath(output);

  // decide SIMD: SIMD32 or SIMD16
  auto* dev_prop =
      at::xpu::getDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  auto sub_group_size = dev_prop->sub_group_sizes;
  int SIMD = sub_group_size[1];
  if (SIMD == SIMD32) {
    if (dim_size < SIMD16 * INNER_LOOP)
      SIMD = SIMD16;
  }

#define DISPATCH_MASK_SOFTMAX_FORWARD_IMPL(vec_size, SIMD, outer_loop) \
  {                                                                    \
    dispatch_softmax_forward_kernel<                                   \
        INNER_LOOP,                                                    \
        vec_size,                                                      \
        SIMD,                                                          \
        scalar_t,                                                      \
        accscalar_t,                                                   \
        uint32_t,                                                      \
        LogSoftMax,                                                    \
        outer_loop,                                                    \
        true,                                                          \
        decltype(input_calc)>(                                         \
        input.data_ptr<scalar_t>(),                                    \
        output.data_ptr<scalar_t>(),                                   \
        dim_size,                                                      \
        outer_size,                                                    \
        mask.data_ptr<bool>(),                                         \
        input_calc);                                                   \
  }

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_group_size = dpcppMaxWorkGroupSize(dev_id);
  if (inner_size == 1 && can_use_32bit_index &&
      max_group_size * INNER_LOOP >= dim_size) {
    // if the element number is smaller than max_work_group_size * INNER_LOOP,
    // the fast path (dispatch_softmax_forward) will be selected.
    // otherwise, the general path (softmax_forward_kernel) will be selected.
    // it assumes vec_size * outer_loop * work_group_size >= dim_size
    auto iter = TensorIterator::binary_op(output, input, mask);
    auto input_calc = make_input_offset_calculator<2>(iter);

    if (SIMD == SIMD32) {
      // Ensure input/output tensor are aligned with max_vec_size
      if (input_start == 0 && output_start == 0 &&
          dim_size % max_vec_size == 0) {
        constexpr int outer_loop = INNER_LOOP / max_vec_size;
        DISPATCH_MASK_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ max_vec_size, /*SIMD*/ SIMD32, outer_loop);
      } else {
        constexpr int outer_loop = INNER_LOOP;
        DISPATCH_MASK_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ 1, /*SIMD*/ SIMD32, outer_loop);
      }
    } else {
      if (input_start == 0 && output_start == 0 &&
          dim_size % max_vec_size == 0) {
        if (max_vec_size >= 4 && dim_size <= 4 * SIMD) {
          // if vec_size >= 4 and dim_size <= 4 * SIMD, take smaller vec_size
          // and 1 outer_loop
          constexpr int outer_loop = 1;
          DISPATCH_MASK_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ 4, /*SIMD*/ SIMD16, outer_loop);
        } else if (dim_size <= max_vec_size * SIMD) {
          // if dim_size <= max_vec_size * SIMD , take 1 outer_loop
          constexpr int outer_loop = 1;
          DISPATCH_MASK_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ max_vec_size, /*SIMD*/ SIMD16, outer_loop);
        } else {
          // SIMD16 will use less register numbers than SIMD32
          // if the SIMD = SIMD16, then outer_loop will be enlarged 2x
          constexpr int outer_loop = INNER_LOOP / max_vec_size * 2;
          DISPATCH_MASK_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ max_vec_size, /*SIMD*/ SIMD16, outer_loop);
        }
      } else {
        constexpr int outer_loop = INNER_LOOP * 2;
        DISPATCH_MASK_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ 1, /*SIMD*/ SIMD16, outer_loop);
      }
    }
  } else {
    auto mask_expand = mask.expand(input.sizes());
    output = at::softmax_out(
        output,
        input.masked_fill(
            mask_expand, -std::numeric_limits<scalar_t>::infinity()),
        dim);
  }
  return output;
#undef DISPATCH_MASK_SOFTMAX_FORWARD_IMPL
}

template <typename scalar_t, typename accscalar_t>
Tensor& add_view_softmax_impl(
    const Tensor& input,
    const Tensor& other,
    int64_t dim,
    const Scalar& alpha_scalar,
    Tensor& output,
    IntArrayRef sizes) {
  auto alpha = alpha_scalar.to<scalar_t>();
  auto view_output = input.view(sizes);
  auto inner_size = view_output.stride(dim);
  // decide indexing range: uint32_t (4GB) or uint64_t (>4GB)
  bool can_use_32bit_index =
      canUse32BitIndexMath(view_output) && canUse32BitIndexMath(output);
  auto dim_size = view_output.size(dim);
  auto outer_size = view_output.numel() / (inner_size * dim_size);
  auto other_outer_size = outer_size;

  constexpr int float4_size = sizeof(float) * 4;
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int INNER_LOOP = max_vec_size * 2;

  bool fuse_pattern = false;
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_group_size = dpcppMaxWorkGroupSize(dev_id);
  if (inner_size == 1 && can_use_32bit_index &&
      max_group_size * INNER_LOOP >= dim_size)
    fuse_pattern = true;
  if (fuse_pattern) {
    Tensor add_output = output.view(input.sizes());
    auto iter = TensorIterator::binary_op(add_output, input, other);
    auto input_calc = make_input_offset_calculator<2>(iter);

    // decide vec_size: max_vec_size or 1
    using vec_t =
        at::native::Memory::aligned_vector_loop<scalar_t, max_vec_size>;
    constexpr int align_bytes = alignof(vec_t);
    int input_start =
        ((uint64_t)input.data_ptr()) % align_bytes / sizeof(scalar_t);
    int output_start =
        ((uint64_t)output.data_ptr()) % align_bytes / sizeof(scalar_t);

    // decide SIMD: SIMD32 or SIMD16
    auto* dev_prop =
        at::xpu::getDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
    auto sub_group_size = dev_prop->sub_group_sizes;
    int SIMD = sub_group_size[1];
    if (SIMD == SIMD32) {
      if (dim_size < SIMD16 * INNER_LOOP)
        SIMD = SIMD16;
    }
    // fused kernel
#define DISPATCH_SOFTMAX_FORWARD_IMPL(vec_size, SIMD, outer_loop) \
  {                                                               \
    dispatch_softmax_forward_add_kernel<                          \
        INNER_LOOP,                                               \
        vec_size,                                                 \
        SIMD,                                                     \
        scalar_t,                                                 \
        accscalar_t,                                              \
        uint32_t,                                                 \
        false,                                                    \
        outer_loop>(                                              \
        input.data_ptr<scalar_t>(),                               \
        other.data_ptr<scalar_t>(),                               \
        output.data_ptr<scalar_t>(),                              \
        dim_size,                                                 \
        alpha,                                                    \
        outer_size,                                               \
        other_outer_size,                                         \
        input_calc);                                              \
  }

    // if the element number is smaller than max_work_group_size *
    // INNER_LOOP, the fused path (dispatch_softmax_forward_add) will be
    // selected. otherwise, the general path (add then softmax) will be
    // selected.
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int max_group_size = dpcppMaxWorkGroupSize(dev_id);
    // it assumes vec_size * outer_loop * work_group_size >= dim_size
    if (SIMD == SIMD32) {
      // Ensure input/output tensor are aligned with max_vec_size
      if (input_start == 0 && output_start == 0 &&
          dim_size % max_vec_size == 0) {
        constexpr int outer_loop = INNER_LOOP / max_vec_size;
        DISPATCH_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ max_vec_size, /*SIMD*/ SIMD32, outer_loop);
      } else {
        constexpr int outer_loop = INNER_LOOP;
        DISPATCH_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ 1, /*SIMD*/ SIMD32, outer_loop);
      }
    } else {
      if (input_start == 0 && output_start == 0 &&
          dim_size % max_vec_size == 0) {
        if (max_vec_size >= 4 && dim_size <= 4 * SIMD) {
          // if vec_size >= 4 and dim_size <= 4 * SIMD, take smaller vec_size
          // and 1 outer_loop
          constexpr int outer_loop = 1;
          DISPATCH_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ 4, /*SIMD*/ SIMD16, outer_loop);
        } else if (dim_size <= max_vec_size * SIMD) {
          // if dim_size <= max_vec_size * SIMD , take 1 outer_loop
          constexpr int outer_loop = 1;
          DISPATCH_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ max_vec_size, /*SIMD*/ SIMD16, outer_loop);
        } else {
          // SIMD16 will use less register numbers than SIMD32
          // if the SIMD = SIMD16, then outer_loop will be enlarged 2x
          constexpr int outer_loop = INNER_LOOP / max_vec_size * 2;
          DISPATCH_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ max_vec_size, /*SIMD*/ SIMD16, outer_loop);
        }
      } else {
        constexpr int outer_loop = INNER_LOOP * 2;
        DISPATCH_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ 1, /*SIMD*/ SIMD16, outer_loop);
      }
    }
    return output;
#undef DISPATCH_SOFTMAX_FORWARD_IMPL
  } else {
    Tensor add_out = at::add(input, other, alpha).view(sizes);
    return at::softmax_out(output, add_out, dim);
  }
}

} // namespace impl

bool shape_use_fused_path(const Tensor& input, const Tensor& other) {
  // for add_softmaxi_fusion, we support shapes like:
  // [N, C, H, W], [N1, C1, H1, W1] which X is divisible by X1
  // [N, C, H, W], [C1, H1, W1] which X is divisible by X1
  // [N, C, H, W], [H1, W1] which X is divisible by X1
  // [N, C, H, W], [W1] which X is divisible by X1
  // likewise for 3D and 5D inputs

  if (input.sizes() == other.sizes())
    return true;
  auto a_dim = input.dim();
  auto b_dim = other.dim();
  if (b_dim > a_dim)
    return false;
  auto input_size = input.sizes();
  auto other_size = other.sizes();
  // loop for the smaller shape from end
  for (int i = 1; i <= b_dim; i++) {
    if (input_size[a_dim - i] % other_size[b_dim - i] != 0) {
      return false;
    }
  }
  return true;
}

Tensor add_softmax(
    const Tensor& input,
    const Tensor& other,
    Scalar alpha,
    const int64_t dim,
    c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("torch_ipex::add_softmax", {});

  // fall back to no fuse path for different type inputs or not supported shapes
  if (!shape_use_fused_path(input, other) ||
      (input.scalar_type() != other.scalar_type()) ||
      (dtype.has_value() && (dtype.value() != input.scalar_type()))) {
    return at::softmax(at::add(input, other, alpha), dim, dtype);
  }
  IntArrayRef sizes;
  Tensor output;
  sizes = input.sizes();
  output = at::empty_like(input);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "add_softmax",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        impl::add_view_softmax_impl<scalar_t, accscalar_t>(
            input, other, dim, alpha, output, sizes);
      });
  return output;
}

Tensor add_view(
    const Tensor& input,
    const Tensor& other,
    Scalar alpha,
    IntArrayRef sizes) {
  return at::add(input, other, alpha).view(sizes);
}

Tensor add_scalar_view(
    const Tensor& input,
    Scalar other,
    Scalar alpha,
    IntArrayRef sizes) {
  return at::add(input, other, alpha).view(sizes);
}

Tensor add_view_softmax(
    const Tensor& input,
    const Tensor& other,
    Scalar alpha,
    IntArrayRef sizes,
    const int64_t dim,
    c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("torch_ipex::add_view_softmax", {});
  // fall back to no fuse path for different type inputs or not supported shapes

  if (!shape_use_fused_path(input, other) ||
      (input.scalar_type() != other.scalar_type()) ||
      (dtype.has_value() && dtype.value() != input.scalar_type())) {
    return at::softmax(at::add(input, other, alpha).view(sizes), dim, dtype);
  }

  Tensor output = at::empty_like(input).view(sizes);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "add_view_softmax",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        impl::add_view_softmax_impl<scalar_t, accscalar_t>(
            input, other, dim, alpha, output, sizes);
      });
  return output;
}

Tensor _masked_softmax(
    const Tensor& input_,
    const Tensor& mask_,
    const c10::optional<int64_t> dim_,
    const c10::optional<int64_t> mask_type_) {
  Tensor output = at::empty_like(input_, input_.options());
  TORCH_CHECK(
      mask_.scalar_type() == ScalarType::Bool,
      "Mask should be a boolean tensor");

  TORCH_CHECK(mask_type_.has_value(), "Mask Type should be defined");
  int64_t mask_type = mask_type_.value();
  TORCH_CHECK(
      (mask_type == 0) || (mask_type == 1) || (mask_type == 2),
      "Mask Type should be 0 (src_mask), 1 (src_key_padding_mask), or 2 (default_mask)");

  // If input is [B, H, T, T] and mask is [B, T]
  // we have special fast kernel
  // mask_type == 1 => mask_ is a src_key_padding_mask
  bool is_BxT_mask = (mask_type == 1) &&
      (input_.dim() == 4 && mask_.dim() == 2 &&
       input_.size(0) == mask_.size(0) && input_.size(2) == mask_.size(1) &&
       input_.size(3) == mask_.size(1));

  // If input is [B, H, T, T] and mask is [T, T]
  // expand mask to [B, H, T, T] and treat it like regular mask
  // TODO We should have special fast kernel for TxT mask as well
  // mask_type == 0 => mask_ is a src_mask
  bool is_TxT_mask = (mask_type == 0) && input_.dim() == 4 &&
      mask_.dim() == 2 && input_.size(3) == mask_.size(1) &&
      input_.size(2) == mask_.size(0) && mask_.size(0) == mask_.size(1);
  // If mask_type == 2, then mask_.sizes() must equal input_.sizes()
  TORCH_CHECK(
      mask_.sizes() == input_.sizes() || is_BxT_mask || is_TxT_mask,
      "Mask shape should match input. mask: ",
      mask_.sizes(),
      " input: ",
      input_.sizes());

  auto input = input_.dim() == 0 ? input_.view(1) : input_;
  auto mask = mask_.dim() == 0 ? mask_.view(1) : mask_;
  int64_t dim = dim_.has_value() ? dim_.value() : input.dim() - 1;

  if (is_BxT_mask) {
    mask = mask.view({mask_.size(0), 1, 1, mask_.size(1)});
  }
  // Here assumes that the mask is broadcastable for input
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      input.scalar_type(),
      "masked_softmax",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        impl::MaskedSoftMaxForward<scalar_t, accscalar_t, false>(
            output, input, dim, mask);
      });
  return output;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  IPEX_OP_REGISTER("add_softmax", at::AtenIpexTypeXPU::add_softmax);
  IPEX_OP_REGISTER("add_view", at::AtenIpexTypeXPU::add_view);
  IPEX_OP_REGISTER("add_view.Scalar", at::AtenIpexTypeXPU::add_scalar_view);
  IPEX_OP_REGISTER("add_view_softmax", at::AtenIpexTypeXPU::add_view_softmax);
}

} // namespace
