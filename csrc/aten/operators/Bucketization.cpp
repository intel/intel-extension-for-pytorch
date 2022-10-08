#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/BucketizationUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <iterator>
#include <tuple>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

// Implement a TF like searchsorted and a bucketize function running on XPU
namespace impl {

template <typename input_t>
const input_t* lower_bound(
    const input_t* start,
    const input_t* end,
    input_t val) {
  while (start < end) {
    const input_t* mid = start + ((end - start) >> 1);
    if (!(*mid >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t, typename output_t>
void searchsorted_dpcpp_contiguous(
    Tensor& result,
    const Tensor& input,
    const Tensor& boundaries,
    const bool& right) {
  int64_t numel_in = input.numel();
  auto& queue = dpcppGetCurrentQueue();
  int64_t rng, grng, tile_size;
  tile_size = dpcppMaxWorkGroupSize();
  rng = numel_in;
  if (rng == 0) {
    rng = static_cast<int64_t>(1);
  }

  grng = rng;
  if (tile_size > grng) {
    tile_size = grng;
  } else if (grng > tile_size) {
    int64_t xMode = static_cast<int64_t>(grng % tile_size);
    if (xMode != 0) {
      grng += static_cast<int64_t>(tile_size - xMode);
    }
  }

  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t* data_in = input.data_ptr<input_t>();
  const input_t* data_bd = boundaries.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  bool is_1d_boundaries = boundaries.dim() == 1;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto data_in_data = input.data_ptr<input_t>();
    auto data_bd_data = boundaries.data_ptr<input_t>();
    auto data_out_data = result.data_ptr<output_t>();

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto data_in = data_in_data;
      auto data_bd = data_bd_data;
      auto data_out = data_out_data;

      for (int64_t i = item.get_global_id(0); i < numel_in;
           i += item.get_global_range()[0]) {
        // If boundaries tensor is 1d, we always search the entire boundary
        // tensor
        int64_t start_bd = is_1d_boundaries ? 0 : i / idim_in * idim_bd;
        const input_t* data_bd_start = &data_bd[start_bd];

        int64_t pos = !right
            ? lower_bound(data_bd_start, data_bd_start + idim_bd, data_in[i]) -
                data_bd_start
            : std::upper_bound(
                  data_bd_start, data_bd_start + idim_bd, data_in[i]) -
                data_bd_start;

        // type conversion might happen here
        data_out[i] = pos;
      }
    };
    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(grng), sycl::range<1>(tile_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

void dispatch(
    Tensor& result,
    const Tensor& input,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  if (!out_int32) {
    IPEX_DISPATCH_ALL_TYPES_AND(
        ScalarType::BFloat16, input.scalar_type(), "searchsorted_out_cpu", [&] {
          searchsorted_dpcpp_contiguous<scalar_t, int64_t>(
              result, input, boundaries, right);
        });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND(
        ScalarType::BFloat16, input.scalar_type(), "searchsorted_out_cpu", [&] {
          searchsorted_dpcpp_contiguous<scalar_t, int>(
              result, input, boundaries, right);
        });
  }
}

} // namespace impl

Tensor& searchsorted_out(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    Tensor& result) {
  at::native::searchsorted_pre_check(sorted_sequence, self, result, out_int32);
  if (result.numel() == 0) {
    result.resize_(self.sizes());
  }
  if (self.numel() == 0) {
    return result;
  }
  if (sorted_sequence.is_contiguous() && self.is_contiguous() &&
      sorted_sequence.dtype() == self.dtype()) {
    at::AtenIpexTypeXPU::impl::dispatch(
        result, self, sorted_sequence, out_int32, right);
    return result;
  }

  Tensor trimmed_input;
  Tensor trimmed_boundaries;
  at::native::searchsorted_maybe_trim_input_tensors(
      trimmed_input, trimmed_boundaries, self, sorted_sequence);
  const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
  const Tensor& final_boundaries =
      trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
  at::AtenIpexTypeXPU::impl::dispatch(
      result, final_input, final_boundaries, out_int32, right);
  return result;
}

Tensor searchsorted(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::AtenIpexTypeXPU::searchsorted_out(
      sorted_sequence, self, out_int32, right, result);
  return result;
}

Tensor searchsorted(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right) {
  return at::AtenIpexTypeXPU::searchsorted(
      sorted_sequence,
      at::native::searchsorted_scalar_tensor(self, sorted_sequence.device()),
      out_int32,
      right);
}

Tensor& bucketize_out(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& result) {
  TORCH_CHECK(
      boundaries.dim() == 1,
      "boundaries tensor must be 1 dimension, but got dim(",
      boundaries.dim(),
      ")");
  at::AtenIpexTypeXPU::searchsorted_out(
      boundaries, self, out_int32, right, result);
  return result;
}

Tensor bucketize(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::AtenIpexTypeXPU::bucketize_out(
      self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize(
    const Scalar& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  return at::AtenIpexTypeXPU::bucketize(
      at::native::searchsorted_scalar_tensor(self, boundaries.device()),
      boundaries,
      out_int32,
      right);
}

} // namespace AtenIpexTypeXPU
} // namespace at
