#include <ATen/Context.h>
#include <ATen/native/TensorIterator.h>
#include <core/detail/IndexUtils.h>
#include <core/SYCL.h>
#include <utils/Atomics.h>
#include "Loops.h"

template <bool has_weight, typename ...> class histogram_kernel {};

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template<typename input_t, typename IndexType>
static IndexType getBin(input_t bVal, input_t minvalue, input_t maxvalue, int nbins) {
  IndexType bin = (int)((bVal - minvalue) * nbins / (maxvalue - minvalue));
  // (only applicable for histc)
  // while each bin is inclusive at the lower end and exclusive at the higher, i.e. [start, end)
  // the last bin is inclusive at both, i.e. [start, end], in order to include maxvalue if exists
  // therefore when bin == nbins, adjust bin to the last bin
  if (bin == nbins) bin -= 1;
  return bin;
}

/*
  Kernel for computing the histogram of the input.
 */
template <
        typename output_t,
        typename input_t,
        typename IndexType,
        int ADims,
        bool has_weight,
        typename Op>
void kernelHistogram1D(
        sycl::detail::TensorInfo<output_t, IndexType> a, /* output */
        sycl::detail::TensorInfo<input_t, IndexType> b, /* input */
        sycl::detail::TensorInfo<output_t, IndexType> c, /* weight */
        int nbins,
        input_t minvalue,
        input_t maxvalue,
        IndexType totalElements,
        Op getOp) {
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

  using out_accessor_t = c10::sycl::SYCLAccessor<dp_rw_mode>;
  using in_accessor_t = c10::sycl::SYCLAccessor<dp_r_mode>;

  auto cgf = DP_Q_CGF(__cgh) {
    out_accessor_t out_acc = out_accessor_t (__cgh, a.data);
    in_accessor_t in_acc = in_accessor_t (__cgh, b.data);
    in_accessor_t weight_acc = in_accessor_t (__cgh, c.data);

    auto kfn = DP_Q_KFN(DP::item<1> item_id) {
      auto out_ptr = out_acc.template get_pointer<output_t>();
      auto in_ptr = in_acc.template get_pointer<input_t>();
      auto weight_ptr = weight_acc.template get_pointer<output_t>();

      auto linearIndex = item_id.get_id(0);
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
              sycl::detail::IndexToOffset<input_t, IndexType>::get(linearIndex, b);
      const auto bVal = in_ptr[bOffset];
      if (bVal >= minvalue && bVal <= maxvalue) {
        // Use value at `b` as an offset of `a`
        const IndexType bin = getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
        const IndexType aOffset =
                sycl::detail::IndexToOffset<output_t, IndexType, ADims>::get(bin, a);
        atomicAdd(&out_ptr[aOffset], getOp(weight_ptr, linearIndex));
      }
    };

    __cgh.parallel_for<histogram_kernel<has_weight, output_t, input_t, IndexType>>(
            DP::range</*dim=*/1>(totalElements), kfn);
  };

  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

#define HANDLE_CASE(WEIGHTS_OP, WITH_WEIGHT)                   \
  kernelHistogram1D<output_t, input_t, IndexType, 1, WITH_WEIGHT>    \
          (aInfo, bInfo, cInfo, nbins, minvalue, maxvalue, totalElements, WEIGHTS_OP);

/*
  Calculate the frequency of the input values.

  `a` contains the final output or the histogram.
  Input `b` is assumed to be 1-D non-negative int array.
  `c` optionally contains the weight vector.
  See `help torch.bincount` for details on the math.
 */
template <typename output_t, typename input_t, bool HasWeights>
bool dpcpp_tensor_histogram(
        at::Tensor a, /* output */
        at::Tensor b, /* input */
        at::Tensor c, /* weights(optional) */
        int64_t nbins,
        input_t minvalue,
        input_t maxvalue) {
  checkBackend("dpcpp_tensor_histogram", {a, b}, Backend::DPCPP);
  if (HasWeights) {
    checkBackend("dpcpp_tensor_histogram", {c}, Backend::DPCPP);
  }

  auto totalElements = b.numel();

  using IndexType = int64_t;
  auto aInfo = sycl::detail::getTensorInfo<output_t, IndexType>(a);
  auto bInfo = sycl::detail::getTensorInfo<input_t, IndexType>(b);
  if (HasWeights) {
    auto cInfo = sycl::detail::getTensorInfo<output_t, IndexType>(c);
    const auto getWeightsOp = [cInfo] (dp_global_ptr_pt<output_t> cPtr, IndexType cIndex) {
      const IndexType cOffset =
              sycl::detail::IndexToOffset<output_t, IndexType, 1>::get(cIndex, cInfo);
      return cPtr[cOffset];
    };
    HANDLE_CASE(getWeightsOp, true);
  } else {
    sycl::detail::TensorInfo<output_t, IndexType> cInfo;
    // set the dummy cinfo with the ptr to the output
    cInfo.data = aInfo.data;
    static const auto getDummyOp = [] (dp_global_ptr_pt<output_t>, IndexType) { return static_cast<output_t>(1); };
    HANDLE_CASE(getDummyOp, false);
  }

  return true;
}

///////////////// bincount /////////////////
template <typename input_t, typename weights_t>
Tensor bincount_template(
        const Tensor& self,
        const Tensor& weights,
        int64_t minlength) {
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  if (self.dim() == 1 && self.numel() == 0) {
    return native::zeros({minlength}, device(kDPCPP).dtype(kLong));
  }
  if (self.dim() != 1 ||
      (!std::is_same<input_t, uint8_t>::value &&
       *self.min().cpu().data_ptr<input_t>() < 0)) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.size(0) != self.size(0)) {
    AT_ERROR("input and weights should have the same length");
  }

  const int64_t nbins = std::max(*self.max().cpu().data_ptr<input_t>() + (int64_t)1, minlength);
  const input_t minvalue = 0;
  const input_t maxvalue = nbins;
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = native::zeros({nbins}, weights.options());
    auto ret = dpcpp_tensor_histogram<weights_t, input_t, true>(
            output, self, weights, nbins, minvalue, maxvalue);
  } else {
    output = native::zeros({nbins}, device(DeviceType::DPCPP).dtype(kInt));
    auto ret = dpcpp_tensor_histogram<int, input_t, false>(
            output, self, weights, nbins, minvalue, maxvalue);
  }
  return output;
}

} // namespace impl

Tensor bincount(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_sycl", [&] {
    const auto scalar = weights.scalar_type();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return impl::bincount_template<scalar_t, float>(self, weights, minlength);
    else if (scalar == ScalarType::Int)
      return impl::bincount_template<scalar_t, float>(self, weights, minlength);
    TORCH_CHECK(0, "bincount_sycl not implemented for weight type '", toString(scalar), "'");
  });
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
