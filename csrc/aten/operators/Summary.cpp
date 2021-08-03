#include <ATen/Context.h>
#include <ATen/native/TensorIterator.h>

#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/Atomics.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

template <bool has_weight, typename...>
class histogram_kernel {};

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename input_t, typename IndexType>
static IndexType getBin(
    input_t bVal,
    input_t minvalue,
    input_t maxvalue,
    int nbins) {
  IndexType bin = (int)((bVal - minvalue) * nbins / (maxvalue - minvalue));
  // (only applicable for histc)
  // while each bin is inclusive at the lower end and exclusive at the higher,
  // i.e. [start, end)
  // the last bin is inclusive at both, i.e. [start, end], in order to include
  // maxvalue if exists
  // therefore when bin == nbins, adjust bin to the last bin
  if (bin == nbins)
    bin -= 1;
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
    TensorInfo<output_t, IndexType> a, /* output */
    TensorInfo<input_t, IndexType> b, /* input */
    TensorInfo<output_t, IndexType> c, /* weight */
    int nbins,
    input_t minvalue,
    input_t maxvalue,
    IndexType totalElements,
    Op getOp) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = a.data;
    auto in_data = b.data;
    auto weight_data = c.data;

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto out_ptr = out_data;
      auto in_ptr = in_data;
      auto weight_ptr = weight_data;

      auto linearIndex = item_id.get_id(0);
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          IndexToOffset<input_t, IndexType>::get(linearIndex, b);
      const auto bVal = in_ptr[bOffset];
      if (bVal >= minvalue && bVal <= maxvalue) {
        // Use value at `b` as an offset of `a`
        const IndexType bin =
            getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
        const IndexType aOffset =
            IndexToOffset<output_t, IndexType, ADims>::get(bin, a);
        atomicAdd(
            (dpcpp_global_ptr_pt<output_t>)&out_ptr[aOffset],
            getOp(weight_ptr, linearIndex));
      }
    };

    __cgh.parallel_for<
        histogram_kernel<has_weight, output_t, input_t, IndexType>>(
        DPCPP::range</*dim=*/1>(totalElements), kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

#define HANDLE_CASE(WEIGHTS_OP, WITH_WEIGHT)                       \
  kernelHistogram1D<output_t, input_t, IndexType, 1, WITH_WEIGHT>( \
      aInfo,                                                       \
      bInfo,                                                       \
      cInfo,                                                       \
      nbins,                                                       \
      minvalue,                                                    \
      maxvalue,                                                    \
      totalElements,                                               \
      WEIGHTS_OP);

template <typename output_t, typename input_t, bool HasWeights>
bool dpcpp_tensor_histogram(
    at::Tensor a, /* output */
    at::Tensor b, /* input */
    at::Tensor c, /* weights(optional) */
    int64_t nbins,
    input_t minvalue,
    input_t maxvalue) {
  checkBackend("dpcpp_tensor_histogram", {a, b}, Backend::XPU);
  if (HasWeights) {
    checkBackend("dpcpp_tensor_histogram", {c}, Backend::XPU);
  }

  auto totalElements = b.numel();

  using IndexType = int64_t;
  auto aInfo = getTensorInfo<output_t, IndexType>(a);
  auto bInfo = getTensorInfo<input_t, IndexType>(b);
  if (HasWeights) {
    auto cInfo = getTensorInfo<output_t, IndexType>(c);
    const auto getWeightsOp = [cInfo](output_t* cPtr, IndexType cIndex) {
      const IndexType cOffset =
          IndexToOffset<output_t, IndexType, 1>::get(cIndex, cInfo);
      return cPtr[cOffset];
    };
    HANDLE_CASE(getWeightsOp, true);
  } else {
    TensorInfo<output_t, IndexType> cInfo;
    // set the dummy cinfo with the ptr to the output
    cInfo.data = aInfo.data;
    static const auto getDummyOp = [](output_t*, IndexType) {
      return static_cast<output_t>(1);
    };
    HANDLE_CASE(getDummyOp, false);
  }

  return true;
}

template <typename input_t, typename weights_t>
Tensor bincount_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    TORCH_CHECK(0, "minlength should be >= 0");
  }
  if (self.dim() == 1 && self.numel() == 0) {
    return native::zeros({minlength}, device(kXPU).dtype(kLong));
  }
  if (self.dim() != 1 ||
      (!std::is_same<input_t, uint8_t>::value &&
       *self.min().cpu().data_ptr<input_t>() < 0)) {
    TORCH_CHECK(0, "bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.size(0) != self.size(0)) {
    TORCH_CHECK(0, "input and weights should have the same length");
  }

  const int64_t nbins =
      std::max(*self.max().cpu().data_ptr<input_t>() + (int64_t)1, minlength);
  const input_t minvalue = 0;
  const input_t maxvalue = nbins;
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = native::zeros({nbins}, weights.options());
    auto ret = dpcpp_tensor_histogram<weights_t, input_t, true>(
        output, self, weights, nbins, minvalue, maxvalue);
  } else {
    output = native::zeros({nbins}, device(DeviceType::XPU).dtype(kLong));
    auto ret = dpcpp_tensor_histogram<
        typename c10::impl::ScalarTypeToCPPType<kLong>::type,
        input_t,
        false>(output, self, weights, nbins, minvalue, maxvalue);
  }
  return output;
}

template <typename scalar_t, typename input_t>
Tensor histc_template(
    const Tensor& self,
    int64_t nbins,
    input_t min,
    input_t max) {
  input_t minvalue = min;
  input_t maxvalue = max;
  if (min == max) {
    minvalue = self.min().item<input_t>();
    maxvalue = self.max().item<input_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }
  TORCH_CHECK(
      !(std::isinf(minvalue) || std::isinf(maxvalue) || std::isnan(minvalue) ||
        std::isnan(maxvalue)),
      "range of [",
      minvalue,
      ", ",
      maxvalue,
      "] is not finite");
  TORCH_CHECK(minvalue < maxvalue, "max must be larger than min");
  Tensor output = at::zeros({nbins}, self.options());
  auto ret = dpcpp_tensor_histogram<scalar_t, input_t, false>(
      output, self, Tensor(), nbins, minvalue, maxvalue);
  return output;
}

} // namespace impl

Tensor bincount(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return IPEX_DISPATCH_INTEGRAL_TYPES(
      self.scalar_type(), "bincount_dpcpp", [&] {
        const auto scalar = weights.scalar_type();
        if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
          return impl::bincount_template<scalar_t, float>(
              self, weights, minlength);
        return impl::bincount_template<scalar_t, double>(
            self, weights.to(kDouble), minlength);
      });
}

Tensor histc(const Tensor& self, int64_t bins, Scalar min, Scalar max) {
  TORCH_CHECK(bins > 0, "bins should be > 0, but is ", bins, " instead");
  return IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "histc", [&] {
    return impl::histc_template<scalar_t>(
        self, bins, min.to<scalar_t>(), max.to<scalar_t>());
  });
}

Tensor& histc_out(
    Tensor& out,
    const Tensor& self,
    int64_t bins,
    Scalar min,
    Scalar max) {
  Tensor out_tmp = at::AtenIpexTypeXPU::histc(self, bins, min, max);
  out.resize_as_(out_tmp).copy_(out_tmp);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
