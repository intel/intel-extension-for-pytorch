#pragma once

#include <stdio.h>

#include <ATen/ATen.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <assert.h>

#include <core/Array.h>
#include <core/Context.h>
#include <core/DPCPP.h>
#include <core/Memory.h>
#include <core/detail/OffsetCalculator.h>

#include "Loops.h"

#include <functional>
#include <iosfwd>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace at::dpcpp;
using at::dpcpp::Array;

DPCPP_DEF_K1(reduce_kernel);

namespace at {
namespace dpcpp {

static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

struct ReduceConfig {
  static constexpr int LANE = 0;
  static constexpr int SUB_GROUP = 1;
  static constexpr int WORK_GROUP = 2;

  ReduceConfig(
      int element_size_bytes,
      int num_outputs,
      int num_inputs,
      int num_total,
      int work_group_size)
      : element_size_bytes(element_size_bytes),
        num_inputs(num_inputs),
        num_outputs(num_outputs),
        num_total(num_total),
        work_group_size(work_group_size) {}

  ReduceConfig(const ReduceConfig& rhs) = default;

  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int num_total;
  int work_group_size;
  int step_input = 1;
  int step_output = 1;
  int wg_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  DPCPP::range<2> get_local_size() const {
    int sg_size =
        DPCPP_SUB_GROUP_SIZE; // to be replaced with real sub_group_size;
    return DPCPP::range<2>(sg_size, work_group_size / sg_size);
  }

  DPCPP::range<2> get_global_size() const {
    return DPCPP::range<2>(
        div_up(num_outputs, step_output) * DPCPP_SUB_GROUP_SIZE,
        wg_per_output * work_group_size / DPCPP_SUB_GROUP_SIZE);
  }

  bool should_sg_reduce() const {
    return input_mult[LANE] != 0;
  }

  bool should_wg_reduce() const {
    return input_mult[SUB_GROUP] != 0;
  }

  bool should_global_reduce() const {
    return input_mult[WORK_GROUP] != 0;
  }

  bool should_store(int output_idx, const DPCPP::nd_item<2>& item_id) const {
    return output_idx < num_outputs &&
        (!should_sg_reduce() || item_id.get_local_id(0) == 0) &&
        (!should_wg_reduce() || item_id.get_local_id(1) == 0);
  }

  int input_idx(const DPCPP::nd_item<2>& item_id) const {
    int lane = item_id.get_local_id(0);
    int sg = item_id.get_local_id(1);
    int wg2 = item_id.get_group(1);
    return (
        lane * input_mult[LANE] + sg * input_mult[SUB_GROUP] +
        wg2 * input_mult[WORK_GROUP]);
  }

  int output_idx(const DPCPP::nd_item<2>& item_id) const {
    int lane = item_id.get_local_id(0);
    int sg = item_id.get_local_id(1);
    int wg1 = item_id.get_group(0);
    return (
        lane * output_mult[LANE] + sg * output_mult[SUB_GROUP] +
        wg1 * step_output);
  }

  int shared_memory_offset(int offset, const DPCPP::nd_item<2>& item_id) const {
    int lane = item_id.get_local_id(0);
    int sg = item_id.get_local_id(1);
    return lane + (sg + offset) * item_id.get_local_range(0);
  }

  int staging_memory_offset(int wg2, const DPCPP::nd_item<2>& item_id) const {
    int offset = wg2 + item_id.get_group(0) * item_id.get_group_range(1);
    if (!should_sg_reduce()) {
      offset = item_id.get_local_id(0) + offset * item_id.get_local_range(0);
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_wg_reduce()) {
      return 0;
    }
    return element_size_bytes * work_group_size;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }

    auto size = (int64_t)element_size_bytes * num_outputs * wg_per_output;
    if (!should_sg_reduce()) {
      size *= get_local_size()[0];
    }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * get_global_size()[0] / DPCPP_SUB_GROUP_SIZE;
  }

  int values_per_thread() const {
    return div_up(num_inputs, step_input);
  }
};

template <typename index_t>
static OffsetCalculator<2, index_t> make_output_calculator(
    const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  int input_index = iter.ntensors() - 1;
  int output_index = 0;
  std::array<const int64_t*, 2> strides = {
      iter.strides(output_index).data() + num_reduce_dims,
      iter.strides(input_index).data() + num_reduce_dims,
  };
  auto shape = iter.shape().data() + num_reduce_dims;
  return OffsetCalculator<2, index_t>(num_output_dims, shape, strides.data());
}

template <typename index_t>
static OffsetCalculator<1, index_t> make_input_calculator(
    const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int input_index = iter.ntensors() - 1;
  std::array<const int64_t*, 1> strides = {
      iter.strides(input_index).data(),
  };
  return OffsetCalculator<1, index_t>(
      num_reduce_dims, iter.shape().data(), strides.data());
}

template <int vt, typename index_t, typename func_t>
void strided_iterate(func_t f, index_t begin, index_t end, index_t stride) {
  if (begin + (vt - 1) * stride < end) {
    //#pragma unroll
    for (index_t i = 0; i < vt; i++) {
      f(i, begin + i * stride);
    }
  } else { //#pragma unroll
    for (index_t i = 0; i < vt; i++) {
      index_t idx = begin + i * stride;
      if (idx < end) {
        f(i, idx);
      }
    }
  }
}

template <int vt, typename index_t, typename type_t, typename foo_t>
Array<type_t, vt> load_memory(
    const dpcpp_global_ptr_pt<type_t>& in,
    index_t begin,
    index_t end,
    index_t stride,
    foo_t foo) {
  Array<type_t, vt> res;
  strided_iterate<vt>(
      [&](index_t i, index_t idx) { res[i] = in[foo(idx)]; },
      begin,
      end,
      stride);
  return res;
}

template <int vt, typename index_t, typename type_t>
Array<type_t, vt> load_memory(
    const dpcpp_global_ptr_pt<type_t>& in,
    index_t begin,
    index_t end,
    index_t stride) {
  return load_memory<vt, index_t, type_t>(
      in, begin, end, stride, [](index_t idx) { return idx; });
}

template <typename out_scalar_t, typename func_t>
struct func_wrapper_t {
  using arg_t = typename binary_function_traits<func_t>::arg2_t;
  func_t reduce;
  func_t combine;
  static inline out_scalar_t project(arg_t arg) {
    return (out_scalar_t)arg;
  }
  static inline arg_t sg_shfl_down(arg_t arg, int offset) {
    // TODO: replace following function with sub_group api when it's available
    // return WARP_SHFL_DOWN(arg, offset);
    return arg;
  }

  func_wrapper_t(const func_t& op) : reduce(op), combine(op) {}
};

template <typename scalar_t, typename func_t>
func_wrapper_t<scalar_t, func_t> func_wrapper(const func_t& op) {
  return func_wrapper_t<scalar_t, func_t>{op};
}

template <
    typename scalar_t,
    typename ops_t,
    typename index_t,
    typename out_scalar_t = scalar_t,
    int vt0 = 4>
struct ReduceOp {
  using traits = binary_function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename std::remove_const<
      typename std::remove_reference<typename traits::arg1_t>::type>::type;
  using out_t = out_scalar_t;

  using InputCalculator = OffsetCalculator<1, index_t>;
  using OutputCalculator = OffsetCalculator<2, index_t>;

  static constexpr bool can_accumulate_in_output =
      std::is_convertible<arg_t, out_scalar_t>::value;

  ops_t ops;
  arg_t ident;
  ReduceConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void* src;
  void* dst0;
  void* dst1;
  void* buffer;
  int* semaphores;
  bool accumulate;
  int noutputs;

  ReduceOp(
      ops_t ops,
      ReduceConfig config,
      InputCalculator input_calc,
      OutputCalculator output_calc,
      const void* src,
      void* dst0,
      void* dst1,
      void* buffer,
      int* semaphores,
      arg_t ident,
      int noutputs)
      : ops(ops),
        ident(ident),
        config(config),
        input_calc(input_calc),
        output_calc(output_calc),
        src(src),
        dst0(dst0),
        dst1(dst1),
        buffer(buffer),
        semaphores(semaphores),
        noutputs(noutputs) {}

  Array<scalar_t, vt0> load_inputs(
      const dpcpp_global_ptr_pt<scalar_t>& data,
      index_t offset) const {
    index_t end = config.num_inputs;
    index_t stride = input_calc.strides_[0][0] / sizeof(scalar_t);
    if (input_calc.dims == 1) {
      return load_memory<vt0, index_t, scalar_t>(
          data, offset, end, config.step_input, [&](index_t idx) {
            return idx * stride;
          });
    } else {
      return load_memory<vt0, index_t, scalar_t>(
          data, offset, end, config.step_input, [&](index_t idx) {
            return input_calc.get(idx)[0] / sizeof(scalar_t);
          });
    }
  }

  arg_t thread_reduce_once(
      const dpcpp_global_ptr_pt<scalar_t>& data,
      index_t offset) const {
    auto values = load_inputs(data, offset);

    arg_t value = ident;
    strided_iterate<vt0, index_t>(
        [&](index_t i, index_t idx) { value = ops.reduce(value, values[i]); },
        offset,
        config.num_inputs,
        config.step_input);

    return value;
  }

  arg_t thread_reduce(
      const dpcpp_global_ptr_pt<scalar_t>& data,
      const DPCPP::nd_item<2>& item_id) const {
    arg_t value = ident;
    index_t idx = config.input_idx(item_id);
    while (idx < static_cast<index_t>(config.num_inputs)) {
      arg_t next = thread_reduce_once(data, idx);
      value = ops.combine(value, next);
      idx += config.step_input * vt0;
    }
    return value;
  }

  arg_t sub_group_reduce(arg_t value) const {
    for (int64_t offset = 1; offset < DPCPP_SUB_GROUP_SIZE; offset <<= 1) {
      arg_t other = ops.sg_shfl_down(value, offset);
      value = ops.combine(value, other);
    }
    return value;
  }

  arg_t work_group_reduce(
      arg_t value,
      const dpcpp_local_ptr_pt<arg_t>& local_ptr,
      const DPCPP::nd_item<2>& item_id) const {
    local_ptr[config.shared_memory_offset(0, item_id)] = value;
    int num_sg = (item_id.get_local_range(0) * item_id.get_local_range(1)) /
        DPCPP_SUB_GROUP_SIZE;
    for (int64_t offset = num_sg / 2; offset > 0; offset >>= 1) {
      item_id.barrier(dpcpp_global_and_local_fence);
      if (static_cast<int64_t>(item_id.get_local_id(1)) < offset &&
          ((static_cast<int64_t>(item_id.get_local_id(1)) + offset) < num_sg)) {
        arg_t other = local_ptr[config.shared_memory_offset(offset, item_id)];
        value = ops.combine(value, other);
        local_ptr[config.shared_memory_offset(0, item_id)] = value;
      }
    }
    return value;
  }

  bool mark_block_finished(
      const dpcpp_local_ptr_pt<int>& last_wg_done_ptr,
      const dpcpp_global_ptr_pt<int>& smem,
      const DPCPP::nd_item<2>& item_id) const {
    item_id.barrier(dpcpp_global_and_local_fence);
    if (item_id.get_local_linear_id() == 0) {
      dpcpp_multi_ptr<int, dpcpp_global_space> sema_multi_ptr(smem);
      DPCPP::atomic<int> at_var(sema_multi_ptr + item_id.get_group(0));
      int prev_blocks_finished = at_var.fetch_add(1);

      last_wg_done_ptr[0] =
          (prev_blocks_finished ==
           static_cast<int>(item_id.get_group_range(1) - 1));
    }
    item_id.barrier(dpcpp_global_and_local_fence);
    bool is_last_block_done = last_wg_done_ptr[0];
    item_id.barrier(dpcpp_global_and_local_fence);
    return is_last_block_done;
  }

  template <bool can_acc>
  arg_t accumulate_in_output(
      const dpcpp_global_ptr_pt<out_scalar_t>& out,
      arg_t value,
      typename std::enable_if<can_acc>::type* = nullptr) const {
    return ops.combine(*out, value);
  }

  // This function should never be called --
  // it's the version of `accumulate_in_output`
  // when accumulation in the output is not possible.
  template <bool can_acc>
  arg_t accumulate_in_output(
      const dpcpp_global_ptr_pt<out_scalar_t>& out,
      arg_t,
      typename std::enable_if<!can_acc>::type* = nullptr) const {
    // TODO: Replace following assert with dpcpp counterparts.
    // assert(false); // can't use TORCH_INTERNAL_ASSERT in Cuda.
    return arg_t{};
  }

  template <class T>
  void set_results(
      const T x,
      const dpcpp_global_ptr_pt<out_scalar_t>& out0,
      const dpcpp_global_ptr_pt<out_scalar_t>& out1) const {
    *out0 = x;
  }

  template <class T>
  void set_results(
      const std::pair<T, T> x,
      const dpcpp_global_ptr_pt<out_scalar_t>& out0,
      const dpcpp_global_ptr_pt<out_scalar_t>& out1) const {
    if (noutputs >= 1) {
      *out0 = x.first;
    }
    if (noutputs >= 2) {
      *out1 = x.second;
    }
  }

  void set_results_to_output(
      arg_t value,
      const dpcpp_global_ptr_pt<out_scalar_t>& out0,
      const dpcpp_global_ptr_pt<out_scalar_t>& out1) const {
    set_results(ops.project(value), out0, out1);
  }

  arg_t global_reduce(
      arg_t value,
      const dpcpp_global_ptr_pt<out_scalar_t>& out0,
      const dpcpp_global_ptr_pt<out_scalar_t>& out1,
      const dpcpp_local_ptr_pt<arg_t>& local_ptr,
      const dpcpp_global_ptr_pt<char>& global_reduce_buf,
      const dpcpp_global_ptr_pt<int>& smem,
      const DPCPP::nd_item<2>& item_id) const {
    arg_t* reduce_buffer = (arg_t*)global_reduce_buf;
    bool should_store =
        config.should_store(config.output_idx(item_id), item_id);
    if (should_store) {
      index_t offset =
          config.staging_memory_offset(item_id.get_group(1), item_id);
      reduce_buffer[offset] = value;
    }

    item_id.barrier(dpcpp_global_and_local_fence);
    bool is_last_block_done =
        mark_block_finished((dpcpp_local_ptr_pt<int>)local_ptr, smem, item_id);

    if (is_last_block_done) {
      value = arg_t{};
      if (config.should_sg_reduce()) {
        index_t input_offset = item_id.get_local_id(0) +
            item_id.get_local_id(1) * item_id.get_local_range(0);
        index_t step = item_id.get_local_range(0) * item_id.get_local_range(1);
        for (; input_offset < static_cast<index_t>(config.wg_per_output);
             input_offset += step) {
          index_t idx = config.staging_memory_offset(input_offset, item_id);
          arg_t next = reduce_buffer[idx];
          value = ops.combine(value, next);
        }
      } else {
        index_t input_offset = item_id.get_local_id(1);
        index_t step = item_id.get_local_range(1);
        for (; input_offset < static_cast<index_t>(config.wg_per_output);
             input_offset += step) {
          index_t idx = config.staging_memory_offset(input_offset, item_id);
          arg_t next = reduce_buffer[idx];
          value = ops.combine(value, next);
        }
      }
      value = work_group_reduce(value, local_ptr, item_id);
      if (config.should_sg_reduce()) {
        value = sub_group_reduce(value);
      }
      if (should_store) {
        if (accumulate) {
          value = accumulate_in_output<can_accumulate_in_output>(out0, value);
        }
        set_results_to_output(value, out0, out1);
      }
    }

    return value;
  }

  void run(
      const dpcpp_global_ptr_pt<char>& input,
      const dpcpp_global_ptr_pt<char>& output0,
      const dpcpp_global_ptr_pt<char>& output1,
      const dpcpp_local_ptr_pt<arg_t>& local_ptr,
      const dpcpp_global_ptr_pt<char>& global_reduce_buf,
      const dpcpp_global_ptr_pt<int>& smem,
      const DPCPP::nd_item<2>& item_id) const {
    index_t output_idx = config.output_idx(item_id);
    index_t input_idx = config.input_idx(item_id);
    auto base_offsets = output_calc.get(output_idx);
    arg_t value = ident;
    if (output_idx < static_cast<index_t>(config.num_outputs) &&
        input_idx < static_cast<index_t>(config.num_inputs)) {
      auto input_slice = (char*)input + base_offsets[1];
      value =
          thread_reduce((dpcpp_global_ptr_pt<scalar_t>)input_slice, item_id);
    }
    bool should_wg_reduce = config.should_wg_reduce();
    if (should_wg_reduce) {
      value = work_group_reduce(value, local_ptr, item_id);
    }
    if (config.should_sg_reduce() &&
        (!should_wg_reduce || item_id.get_local_id(1) == 0)) {
      value = sub_group_reduce(value);
    }

    auto out0 = (out_scalar_t*)(output0 + base_offsets[0]);
    auto out1 = (out_scalar_t*)(output1 + base_offsets[0]);
    if (config.should_global_reduce()) {
      value = global_reduce(
          value,
          (dpcpp_global_ptr_pt<out_scalar_t>)out0,
          (dpcpp_global_ptr_pt<out_scalar_t>)out1,
          local_ptr,
          global_reduce_buf,
          smem,
          item_id);
    } else if (config.should_store(output_idx, item_id)) {
      if (accumulate) {
        value = accumulate_in_output<can_accumulate_in_output>(
            (dpcpp_global_ptr_pt<out_scalar_t>)out0, value);
      }
      set_results_to_output(
          value,
          (dpcpp_global_ptr_pt<out_scalar_t>)out0,
          (dpcpp_global_ptr_pt<out_scalar_t>)out1);
    }
  }
};

template <typename DataType, typename R>
static void launch_reduce_kernel(
    const ReduceConfig& config,
    const R reduction) {
  using acc_t = typename R::arg_t;
  using output_t = typename R::out_t;
  auto queue = dpcppGetCurrentQueue();
  DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));

  // This is a work-around because dpcpp_discard_w_mode doesn't work in some
  // conditions
  dpcppMemsetAsync(reduction.dst0, 0, sizeof(output_t) * config.num_outputs);
  if (reduction.noutputs > 1) {
    dpcppMemsetAsync(reduction.dst1, 0, sizeof(output_t) * config.num_outputs);
  }

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, reduction.src);
    auto out0_acc = DPCPPAccessor<dpcpp_discard_w_mode>(
        cgh, reduction.dst0);
    auto out1_acc = reduction.noutputs <= 1
        ? DPCPPAccessor<dpcpp_discard_w_mode>(cgh, dummy_buffer)
        : // dummy
        DPCPPAccessor<dpcpp_discard_w_mode>(
            cgh, reduction.dst1);
    auto local_acc = dpcpp_local_acc_t<acc_t>(config.work_group_size, cgh);

    auto global_reduce_acc = config.should_global_reduce()
        ? DPCPPAccessor<dpcpp_rw_mode>(cgh, reduction.buffer)
        : DPCPPAccessor<dpcpp_rw_mode>(cgh, dummy_buffer); // dummy
    auto sema_acc = config.should_global_reduce()
        ? DPCPPAccessor<dpcpp_rw_mode>(cgh, reduction.semaphores)
        : DPCPPAccessor<dpcpp_rw_mode>(cgh, dummy_buffer); // dummy

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      auto in_ptr = in_acc.template get_pointer<char>();
      auto out0_ptr = out0_acc.template get_pointer<char>();
      auto out1_ptr = out1_acc.template get_pointer<char>();
      auto local_ptr = (dpcpp_local_ptr_pt<acc_t>)local_acc.get_pointer().get();
      auto global_reduce_ptr = global_reduce_acc.template get_pointer<char>();
      auto sema_ptr = sema_acc.template get_pointer<int>();
      reduction.run(
          in_ptr,
          out0_ptr,
          out1_ptr,
          local_ptr,
          global_reduce_ptr,
          sema_ptr,
          item_id);
    };

    cgh.parallel_for<DPCPP_K(reduce_kernel, DataType, R)>(
        DPCPP::nd_range<2>(config.get_global_size(), config.get_local_size()),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    typename out_scalar_t,
    int vt0 = 4,
    typename ops_t,
    typename ident_t = double>
inline void dpcpp_reduce_kernel(
    TensorIterator& iter,
    const ops_t& ops,
    ident_t ident = 0) {
  TORCH_INTERNAL_ASSERT(
      iter.numel() > 0 && iter.ntensors() - iter.noutputs() == 1 &&
      iter.noutputs() >= 1);

  using traits = binary_function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename traits::arg1_t;
  static constexpr bool can_accumulate_in_output =
      std::is_convertible<arg_t, out_scalar_t>::value;

  bool can_use_32bit_indexing = iter.can_use_32bit_indexing();
  if (can_accumulate_in_output && !can_use_32bit_indexing) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_reduce_kernel<scalar_t, out_scalar_t, vt0>(sub_iter, ops, ident);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  const char* in_data = (char*)iter.data_ptr(iter.ntensors() - 1);
  char* out_data_extra;
  const auto noutputs = iter.noutputs();
  if (noutputs > 1) {
    out_data_extra = (char*)iter.data_ptr(1);
  } else {
    out_data_extra = nullptr;
  }

  auto queue = dpcppGetCurrentQueue();
  int64_t wg_size = dpcppMaxWorkGroupSize(queue);
  // firstly hardcoded; to be replaced with get_sub_group_max_size
  int sg_size = DPCPP_SUB_GROUP_SIZE;
  int sgs_per_wg = wg_size / sg_size;

  // Start by assuming that each thread handles a single output and all
  // the inputs for that output.
  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config = ReduceConfig(
      sizeof(arg_t), num_outputs, inputs_per_output, iter.numel(), wg_size);

  // TODO: Currently subgroup and its corresponding logic doesn't work well.
  // We will re-enable it when subgroup api is available
  if (iter.ndim() == 0 ||
      iter.strides(/*arg=*/input_index)[0] == sizeof(scalar_t)) {
    config.input_mult[0] = config.split_input(sg_size);
  } else {
    // Otherwise split the output across langs in a subgroup
    config.output_mult[0] = config.split_output(sg_size);
  }

  if (config.values_per_thread() >= sgs_per_wg * 16 ||
      config.values_per_thread() >= 256) {
    config.input_mult[1] = config.split_input(sgs_per_wg);
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(sgs_per_wg);
  }

  if (config.values_per_thread() >= 256 && num_outputs <= 4096) {
    config.wg_per_output = div_up(config.values_per_thread(), 16);
    if (config.wg_per_output > 65535) {
      config.wg_per_output = 65535;
    }
    config.input_mult[2] = config.split_input(config.wg_per_output);
  }

  at::DataPtr buffer;
  at::DataPtr semaphores;
  if (config.should_global_reduce()) {
    auto& allocator = *dpcpp::getDPCPPDeviceAllocator();
    buffer = allocator.allocate(config.global_memory_size());
    semaphores = allocator.allocate(config.semaphore_size());
    dpcppMemset(semaphores.get(), 0, config.semaphore_size());
  }

  if (can_use_32bit_indexing) {
    auto output_calc = make_output_calculator<uint32_t>(iter);
    auto input_calc = make_input_calculator<uint32_t>(iter);
    auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, vt0>(
        ops,
        config,
        input_calc,
        output_calc,
        in_data,
        out_data,
        out_data_extra,
        buffer.get(),
        (int*)semaphores.get(),
        ident,
        noutputs);
    reduce.accumulate = iter.should_accumulate();
    launch_reduce_kernel<scalar_t>(config, reduce);
  } else {
    auto output_calc = make_output_calculator<uint64_t>(iter);
    auto input_calc = make_input_calculator<uint64_t>(iter);
    auto reduce = ReduceOp<scalar_t, ops_t, uint64_t, out_scalar_t, vt0>(
        ops,
        config,
        input_calc,
        output_calc,
        in_data,
        out_data,
        out_data_extra,
        buffer.get(),
        (int*)semaphores.get(),
        ident,
        noutputs);
    TORCH_INTERNAL_ASSERT(!iter.should_accumulate());
    reduce.accumulate = false;
    launch_reduce_kernel<scalar_t>(config, reduce);
  }
}

template <typename reduce_op, typename nd_item_id, typename local_shared>
static inline void reduce(
  nd_item_id item_id,
  const local_shared& local_shared_mem,
  reduce_op bin_op) {
  auto local_idx = item_id.get_local_id(0);
  auto group_size = item_id.get_local_range().size();

  decltype(group_size) __k = 1;
  do {
    item_id.barrier(DPCPP::access::fence_space::local_space);
    if (local_idx % (2 * __k) == 0 && local_idx + __k < group_size) {
      local_shared_mem[local_idx] = bin_op(
        local_shared_mem[local_idx], local_shared_mem[local_idx + __k]);
    }
    __k *= 2;
  } while (__k < group_size);
  item_id.barrier(DPCPP::access::fence_space::local_space);
}

} // namespace dpcpp
} // namespace at
