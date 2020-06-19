#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <core/detail/OffsetCalculator.h>
#include <core/detail/TensorInfo.h>

#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>

using namespace at::dpcpp;

namespace at {
namespace dpcpp {

#define MAX_INPUT_TENSOR_NUM 3
#define MAX_TOTAL_TENSOR_NUM 4
// Work around for passing the offsets to the dpcpp kernel instead of using
// OffsetCalculator.
// Need to change it back to OffsetCalculator with dpcpp
template <typename IndexType = uint32_t>
struct SyclOffsetCal {
  int dims;
  // Make the information to be basic data types to avoid compiler issue.
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];

  IndexType get(IndexType linear_idx) const {
    IndexType offset = 0;

    for (int dim = 0; dim < dims; ++dim) {
      // Make the code as naive as possible.
      offset += (linear_idx % sizes[dim]) * strides[dim];
      linear_idx = linear_idx / sizes[dim];
    }
    return offset;
  }
};

template <typename IndexType>
static SyclOffsetCal<IndexType> make_offset_calculator(
    const TensorIterator& iter,
    int n) {
  SyclOffsetCal<IndexType> offset;
  if (n < iter.ntensors()) {
    auto dims = iter.ndim();
    offset.dims = dims;
    auto strides = iter.strides(n);
    auto sizes = iter.shape();
    for (int i = 0; i < dims; i++) {
      offset.sizes[i] = sizes[i];
      offset.strides[i] = strides[i];
    }
  } else {
    offset.dims = 0;
  }
  return offset;
}

template <int N>
static OffsetCalculator<N> make_offset_calculator(const TensorIterator& iter) {
  TORCH_INTERNAL_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data());
}

template <typename traits, typename ptr_t, std::size_t... INDEX>
typename traits::ArgsTuple dereference_impl(
    ptr_t data[],
    const int64_t* strides,
    int64_t i,
    std::index_sequence<INDEX...>) {
  return std::make_tuple(*(typename traits::template arg<
                           INDEX>::type*)(data[INDEX] + i * strides[INDEX])...);
}

template <typename traits, typename ptr_t>
typename traits::ArgsTuple dereference(
    ptr_t data[],
    const int64_t* strides,
    int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

// This is a workaround for the compute cpp's issue and limitation.
// 1. The DPCPPAccessor cannot be put in container because there is no default
// constructor in accessor.
// 2. The DPCPPAccessor cannot be *new* because the dpcpp kernel doesn't accept
// host pointer.
// 3. The DPCPPAccessor cannot be in tuples because the dpcpp kernel doesn't
// accept the std::tuple. (But also variance template class)
// 4. The DPCPPAccessor cannot be passed to dpcpp kernel in array like:
// DPCPPAccessor acc[1] = {DPCPPAccessor(cgh, vptr)}.
//    Because of unknown compute cpp bug, the dpcpp kernel always got random
//    data by passing DPCPPAccessor array.
// To add the repeating accessor and pointer code pair in macro.
#define CHR1(x, y) x##y
#define CHR(x, y) CHR1(x, y)

#define DEC_1 0
#define DEC_2 1
#define DEC_3 2
#define DEC_4 3
#define DEC_5 4
#define DEC_6 5
#define DEC_7 6
#define DEC_8 7
#define DEC_9 8
#define DEC_10 9
#define DEC_11 10
#define DEC_12 11
#define DEC_13 12
#define DEC_14 13
#define DEC_15 14
#define DEC_16 15
#define DEC_17 16
#define DEC_18 17
#define DEC_19 18
#define DEC_20 19
#define DEC_21 20
#define DEC_22 21
#define DEC_23 22
#define DEC_24 23
#define DEC_25 24

#define DEC_(n) DEC_##n
#define DEC(n) DEC_(n)

#define REPEAT_0(n, f) f(n)
#define REPEAT_1(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_2(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_3(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_4(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_5(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_6(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_7(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_8(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_9(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_10(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_11(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_12(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_13(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_14(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_15(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_16(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_17(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_18(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_19(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_20(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_21(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_22(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_23(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_24(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)
#define REPEAT_25(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f) f(n)

#define REPEAT_PATTERN(n, f) CHR(REPEAT_, DEC(n))(DEC(n), f)

DPCPP_DEF_K1(dpcpp_loops_kernel_impl);
template <typename kernel_name, typename func_t>
void dpcpp_loops_kernel_impl(TensorIterator& iter, const func_t f) {
  using traits = function_traits<func_t>;
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == traits::arity + 1);
  TORCH_INTERNAL_ASSERT(
      traits::arity <= MAX_INPUT_TENSOR_NUM,
      "loops kernel for",
      traits::arity,
      " operands is not generated");
  constexpr int n_in_tensors = traits::arity;
  using ret_t = typename traits::result_type;

  int64_t numel = iter.numel();

  void* out_data = (void*)iter.data_ptr(0);
  // Initial the in_data with some dummy valid address.
  at::detail::Array<char*, MAX_INPUT_TENSOR_NUM> in_data((char*)out_data);
  for (int i = 0; i < n_in_tensors; i++) {
    // The life will be much easier in UMS
    in_data[i] = (char*)iter.data_ptr(i + 1);
  }

  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

  using out_accessor_t = DPCPPAccessor<dpcpp_discard_w_mode>;
  using in_accessor_t = DPCPPAccessor<dpcpp_r_mode>;
  using in_ptr_t = typename DPCPP::global_ptr<char>::pointer_t;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    out_accessor_t out_acc = out_accessor_t(__cgh, out_data);

#define ACCESSOR_DEFINE(n) \
  in_accessor_t in_acc_##n = in_accessor_t(__cgh, in_data[n]);
    REPEAT_PATTERN(MAX_INPUT_TENSOR_NUM, ACCESSOR_DEFINE)
#undef ACCESSOR_DEFINE

#define OFFSET_DEFINE(n) \
  SyclOffsetCal<uint32_t> __off_##n = make_offset_calculator<uint32_t>(iter, n);
    REPEAT_PATTERN(MAX_TOTAL_TENSOR_NUM, OFFSET_DEFINE)
#undef OFFSET_DEFINE

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto out_ptr = out_acc.template get_pointer<char>();
      at::detail::Array<in_ptr_t, MAX_INPUT_TENSOR_NUM> in_ptr;
      at::detail::Array<int64_t, MAX_TOTAL_TENSOR_NUM> offsets;

#define ACCESSOR_DEREFER(n) in_ptr[n] = in_acc_##n.template get_pointer<char>();
      REPEAT_PATTERN(MAX_INPUT_TENSOR_NUM, ACCESSOR_DEREFER)
#undef ACCESSOR_DEREFER

      auto linear_idx = item_id.get_id(0);
#define OFFSET_GET(n) offsets[n] = __off_##n.get(linear_idx);
      REPEAT_PATTERN(MAX_TOTAL_TENSOR_NUM, OFFSET_GET)
#undef OFFSET_GET

      ret_t* out = (ret_t*)(out_ptr + offsets[0]);
      *out = c10::guts::apply(
          f, dereference<traits>(&in_ptr.data[0], &offsets.data[1], 1));
    };

    __cgh.parallel_for<DPCPP_K(dpcpp_loops_kernel_impl, kernel_name, ret_t)>(
        DPCPP::range</*dim=*/1>(numel), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename func_t>
void dpcpp_kernel_for_tensor_iter(TensorIterator& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).type() == at::kDPCPP);
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_kernel_for_tensor_iter<scalar_t>(sub_iter, f);
    }
    return;
  }

  dpcpp_loops_kernel_impl<scalar_t>(iter, f);
}

// all three operands contiguous
template <typename traits>
static inline bool is_binary_contiguous(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
      strides[1] == sizeof(typename traits::arg1_t) &&
      strides[2] == sizeof(typename traits::arg2_t);
}

// arg1 is a scalar, output and arg2 are contiguous
template <typename traits>
static inline bool is_binary_contiguous_s1(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
      strides[1] == 0 && strides[2] == sizeof(typename traits::arg2_t);
}

// arg2 is a scalar, output and arg1 are contiguous
template <typename traits>
static inline bool is_binary_contiguous_s2(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
      strides[1] == sizeof(typename traits::arg1_t) && strides[2] == 0;
}

// result is
static inline bool is_reduction(char** data, const int64_t* strides) {
  return strides[0] == 0 && strides[1] == 0 && data[0] == data[1];
}

#define LOOP_HEADER(func_t, data, strides)       \
  using traits = binary_function_traits<func_t>; \
  using arg0_t = typename traits::result_type;   \
  using arg1_t = typename traits::arg1_t;        \
  using arg2_t = typename traits::arg2_t;        \
  char* out_ptr = data[0];                       \
  const char* in1_ptr = data[1];                 \
  const char* in2_ptr = data[2];                 \
  int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];

template <class op_type, typename arg0_t, typename arg1_t, typename arg2_t>
class KernelName {};

template <class op_type, typename func_t>
static inline void dpcpp_binary_loop(
    char** data,
    const int64_t* strides,
    int64_t i,
    int64_t n,
    const func_t op) {
  LOOP_HEADER(func_t, data, strides)
  static const auto write_mode = DPCPP::access::mode::discard_write;
  static const auto read_mode = DPCPP::access::mode::read;
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(n, tileSize, rng, GRange);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in1_Acc = DPCPPAccessor<read_mode>(cgh, in1_ptr);
    auto in2_Acc = DPCPPAccessor<read_mode>(cgh, in2_ptr);
    auto out_Acc = DPCPPAccessor<write_mode>(cgh, out_ptr);
    cgh.parallel_for<KernelName<op_type, arg0_t, arg1_t, arg2_t>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(GRange), DPCPP::range<1>(tileSize)),
        [=](DPCPP::nd_item<1> item) {
          int64_t globalid = item.get_global_linear_id();
          auto input1_ptr = in1_Acc.template get_pointer<arg1_t>();
          auto input2_ptr = in2_Acc.template get_pointer<arg2_t>();
          auto output_ptr = out_Acc.template get_pointer<arg0_t>();

          if (globalid < rng)
            output_ptr[SyclConvertToActualOffset(arg0_t, globalid * s0)] = op(
                input1_ptr[SyclConvertToActualOffset(arg1_t, globalid * s1)],
                input2_ptr[SyclConvertToActualOffset(arg2_t, globalid * s2)]);
        });
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <class op_type, typename func_t>
void dpcpp_binary_kernel(TensorIterator& iter, const func_t& op) {
  using traits = binary_function_traits<func_t>;
  static_assert(
      std::is_same<typename traits::result_type, typename traits::arg1_t>::
          value,
      "all types must match");
  static_assert(
      std::is_same<typename traits::result_type, typename traits::arg2_t>::
          value,
      "all types must match");

  // int64_t n = iter.numel();
  // if (n == 0)
  //   return;

  // auto strides = iter.get_strides();
  // while (strides.size() < 2 * iter.ntensors()) {
  //   strides.push_back(0);
  // }

  // auto data = iter.get_base_ptrs().data();

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_binary_contiguous<traits>(strides)) {
      dpcpp_binary_loop<op_type>(data, strides, 0, n, op);
    } else if (is_binary_contiguous_s1<traits>(strides)) {
      dpcpp_binary_loop<op_type>(data, strides, 0, n, op);
    } else if (is_binary_contiguous_s2<traits>(strides)) {
      dpcpp_binary_loop<op_type>(data, strides, 0, n, op);
    } else {
      dpcpp_binary_loop<op_type>(data, strides, 0, n, op);
    }
  });
}

DPCPP_DEF_K1(dpcpp_index_kernel);
template <typename kernel_name, typename func_t>
void dpcpp_index_kernel(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    const func_t f) {
  auto numel = iter.numel();

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_index_kernel<kernel_name>(sub_iter, index_size, index_stride, f);
    }
    return;
  }

  size_t num_indices = index_size.size();
  TORCH_INTERNAL_ASSERT(num_indices == index_stride.size());
  TORCH_INTERNAL_ASSERT(
      num_indices == static_cast<size_t>(iter.ntensors()) - 2);
  TORCH_INTERNAL_ASSERT(num_indices <= MAX_TENSORINFO_DIMS);

  void* out_data = (void*)iter.data_ptr(0);
  void* in_data = (void*)iter.data_ptr(1);
  auto sizes = at::detail::Array<int64_t, MAX_TENSORINFO_DIMS>(0);
  auto strides = at::detail::Array<int64_t, MAX_TENSORINFO_DIMS>(0);
  // Initial the index_ptrs with some dummy valid address.
  auto index_ptrs =
      at::detail::Array<char*, MAX_TENSORINFO_DIMS>((char*)in_data);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

  using out_accessor_t = DPCPPAccessor<dpcpp_rw_mode>;
  using in_accessor_t = DPCPPAccessor<dpcpp_r_mode>;
  using dpcpp_ptr_t = typename DPCPP::global_ptr<char>::pointer_t;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    out_accessor_t out_acc = out_accessor_t(__cgh, out_data);
    in_accessor_t in_acc = in_accessor_t(__cgh, in_data);

#define ACCESSOR_DEFINE(n) \
  in_accessor_t in_acc_##n = in_accessor_t(__cgh, index_ptrs[n]);
    REPEAT_PATTERN(MAX_TENSORINFO_DIMS, ACCESSOR_DEFINE)
#undef ACCESSOR_DEFINE

    auto offset_calc = make_offset_calculator<3>(iter);

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto out_ptr = out_acc.template get_pointer<char>();
      auto in_ptr = in_acc.template get_pointer<char>();
      at::detail::Array<dpcpp_ptr_t, MAX_TENSORINFO_DIMS> index_ptr;

#define ACCESSOR_DEREFER(n) \
  index_ptr[n] = in_acc_##n.template get_pointer<char>();
      REPEAT_PATTERN(MAX_TENSORINFO_DIMS, ACCESSOR_DEREFER)
#undef ACCESSOR_DEREFER

      auto linear_idx = item_id.get_id(0);
      auto offsets = offset_calc.get(linear_idx);
      out_ptr = out_ptr + offsets[0];
      in_ptr = in_ptr + offsets[1];
      int64_t offset = 0;
      //#pragma unroll
      for (size_t i = 0; i < num_indices; i++) {
        int64_t index = *(int64_t*)(index_ptr[i] + offsets[2]);
        if (index >= -sizes[i] && index < sizes[i]) {
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }
        //        else
        //        assert(index >= -sizes[i] && index < sizes[i] && "index out of
        //        bounds");
      }

      f(out_ptr, in_ptr, offset);
    };

    __cgh.parallel_for<DPCPP_K(dpcpp_index_kernel, kernel_name)>(
        DPCPP::range</*dim=*/1>(numel), kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}
} // namespace dpcpp
} // namespace at
