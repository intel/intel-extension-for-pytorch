#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/core/detail/IndexUtils.h>
#include <aten/operators/comm/Numerics.h>
#include <runtime/Utils.h>
#include "ATen/OpMathType.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

#include <iostream>
#include "ForeachFunctors.h"
#include "Loops.h"
#include "MultiTensorApply.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename T,
    int NormType,
    int depth = 1,
    int r_args_depth = 1,
    int res_arg_index = 0>
struct LpNormFunctor {
  static_assert(
      NormType == 1 || NormType == 2,
      "foreach_norm supports only L1 and L2 norm");
  using opmath_t = acc_type<T>;
  template <typename TLA, typename TLW>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      opmath_t* output_per_tensor,
      const int max_chunks_per_tensor) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;

    T* x = (T*)tlAddress[tensor_loc].addresses[0] + chunk_idx * chunk_size;
    n -= chunk_idx * chunk_size;

    opmath_t vals[kILP];
    T r_x[kILP];
    for (int i = 0; i < kILP; i++) {
      vals[i] = opmath_t(0.0f);
      r_x[i] = T(0.0f);
    }

    if (n % kILP == 0 && (chunk_size & kILP) == 0 && is_aligned(x)) {
      for (int64_t i_start = item_idx;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += item_range) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          opmath_t next = static_cast<opmath_t>(r_x[ii]);
          vals[ii] += NormType == 1
              ? static_cast<opmath_t>(Numerics<opmath_t>::fabs(next))
              : static_cast<opmath_t>(next * next);
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += item_range * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          int i = i_start + item_idx + ii * item_range;
          if (i < n && i < chunk_size) {
            opmath_t next = static_cast<opmath_t>(r_x[ii]);
            vals[ii] += NormType == 1
                ? static_cast<opmath_t>(Numerics<opmath_t>::fabs(next))
                : static_cast<opmath_t>(next * next);
          }
        }
      }
    }

    auto val = opmath_t(0);
    for (int i = 0; i < kILP; i++) {
      val += vals[i];
    }
    auto sum_val = sycl::reduce_over_group(
        item_id.get_group(), val, sycl::plus<opmath_t>());

    if (item_idx == 0) {
      output_per_tensor[tensor_loc * max_chunks_per_tensor + chunk_idx] =
          sum_val;
    }
  }
};

template <typename T, int NormType, typename opmath_t = at::opmath_type<T>>
struct lpnormChunkReduceKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto lid = item_id.get_local_linear_id();
    auto group_id = item_id.get_group(0);

    const opmath_t* output_this_tensor =
        output_per_tensor + group_id * max_chunks_per_tensor;
    opmath_t val = 0;
    for (int i = lid; i < max_chunks_per_tensor; i += wg_size) {
      val += output_this_tensor[i];
    }
    auto sum_val = sycl::reduce_over_group(
        item_id.get_group(), val, sycl::plus<opmath_t>());
    if (lid == 0) {
      ret_per_tensor[group_id] =
          NormType == 1 ? sum_val : Numerics<opmath_t>::sqrt(sum_val);
    }
  }
  lpnormChunkReduceKernelFunctor(
      const opmath_t* output_per_tensor_,
      T* ret_per_tensor_,
      int max_chunks_per_tensor_,
      int wg_size_)
      : output_per_tensor(output_per_tensor_),
        ret_per_tensor(ret_per_tensor_),
        max_chunks_per_tensor(max_chunks_per_tensor_),
        wg_size(wg_size_) {}

 private:
  const opmath_t* output_per_tensor;
  T* ret_per_tensor;
  int max_chunks_per_tensor;
  int wg_size;
};

template <typename T, int NormType, typename opmath_t = at::opmath_type<T>>
void lpnorm_chunk_reduce_kernel(
    const opmath_t* output_per_tensor,
    T* ret_per_tensor,
    int max_chunks_per_tensor,
    int n_tensor) {
  auto& queue = dpcppGetCurrentQueue();
  int wg_size = std::min(max_chunks_per_tensor, int(dpcppMaxWorkItemsPerEU()));
  auto cgf = DPCPP_Q_CGF(__cgh) {
    lpnormChunkReduceKernelFunctor<T, NormType, opmath_t> kfn(
        output_per_tensor, ret_per_tensor, max_chunks_per_tensor, wg_size);
    __cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(n_tensor * wg_size, wg_size), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

std::vector<Tensor> _foreach_norm(TensorList tensors, const Scalar& ord) {
  double p;
  if (ord.isIntegral(false)) {
    p = ord.to<int64_t>();
  } else if (ord.isFloatingPoint()) {
    p = ord.to<double>();
  } else {
    TORCH_CHECK(false, "foreach_norm expects ord to be integer or float");
  }
  at::native::check_foreach_api_restrictions(tensors);
  const bool has_int_or_complex =
      std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
            at::isComplexType(scalar_type);
      });
  if (!at::native::can_use_fast_route(tensors) || has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2))) {
    return at::native::foreach_tensor_norm_slow(tensors, ord);
  }

  const int ntensors = tensors.size();
  int max_chunks_per_tensor = -1;

  int64_t wg_size = xpu::dpcpp::dpcppMaxWorkGroupSize();
  int64_t kChunkSize = kElementPerThread * wg_size;

  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor =
        (tensors[t].numel() + kChunkSize - 1) / kChunkSize;
    if (max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }
  const auto options = tensors[0].options();
  auto output_per_tensor = at::zeros(
      {ntensors * max_chunks_per_tensor},
      options.dtype(toOpMathType(tensors[0].scalar_type())));
  auto ret_per_tensor = at::empty({ntensors}, options);

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};
  if (p == static_cast<double>(1)) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_norm",
        [&]() {
          using opmath_t = typename at::opmath_type<scalar_t>;
          // sum temp val for each chunk
          multi_tensor_apply<1>(
              tensor_lists,
              LpNormFunctor<scalar_t, 1>(),
              output_per_tensor.data_ptr<opmath_t>(),
              max_chunks_per_tensor);
          // sum final val for all chunks
          lpnorm_chunk_reduce_kernel<scalar_t, 1>(
              output_per_tensor.data_ptr<opmath_t>(),
              ret_per_tensor.data_ptr<scalar_t>(),
              max_chunks_per_tensor,
              ntensors);
        });
  } else if (p == static_cast<double>(2)) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_norm",
        [&]() {
          using opmath_t = typename at::opmath_type<scalar_t>;
          multi_tensor_apply<1>(
              tensor_lists,
              LpNormFunctor<scalar_t, 2>(),
              output_per_tensor.data_ptr<opmath_t>(),
              max_chunks_per_tensor);
          lpnorm_chunk_reduce_kernel<scalar_t, 2>(
              output_per_tensor.data_ptr<opmath_t>(),
              ret_per_tensor.data_ptr<scalar_t>(),
              max_chunks_per_tensor,
              ntensors);
        });
  } else {
    TORCH_CHECK(false, "foreach_norm fast path got unexpected ord value: ", p);
  }

  std::vector<Tensor> result;
  result.reserve(ntensors);
  for (const auto& i : c10::irange(ntensors)) {
    result.emplace_back(ret_per_tensor[i]);
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
