#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
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

// customized lower_bound func to ensure the low bound of 'nan', 'inf' etc. be
// the end of boundary and we can properly handle a sorter argument
// std::lower_bound can not be used here since its customized comparator need
// strict weak ordering and the customized comparators require both arguments to
// have the same type, which wouldn't happen when comparing val of input_t to an
// indexer value from sorter of int64
template <typename input_t>
int64_t cus_lower_bound(
    int64_t start,
    int64_t end,
    const input_t val,
    const input_t* bd,
    const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

// customized upper_bound func to ensure we can properly handle a sorter
// argument std::upper_bound can not be used here since its customized
// comparator requires both arguments to have the same type, which wouldn't
// happen when comparing val of input_t to an indexer value from sorter of int64
template <typename input_t>
int64_t cus_upper_bound(
    int64_t start,
    int64_t end,
    const input_t val,
    const input_t* bd,
    const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val > val)) {
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
    const bool& right,
    const Tensor& sorter) {
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
  const int64_t* data_st =
      sorter.defined() ? sorter.data_ptr<int64_t>() : nullptr;
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
        int64_t end_bd = start_bd + idim_bd;

        int64_t pos = !right
            ? cus_lower_bound(start_bd, end_bd, data_in[i], data_bd, data_st) -
                start_bd
            : cus_upper_bound(start_bd, end_bd, data_in[i], data_bd, data_st) -
                start_bd;

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
    bool right,
    const Tensor& sorter) {
  if (!out_int32) {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_out",
        [&] {
          searchsorted_dpcpp_contiguous<scalar_t, int64_t>(
              result, input, boundaries, right, sorter);
        });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_out",
        [&] {
          searchsorted_dpcpp_contiguous<scalar_t, int>(
              result, input, boundaries, right, sorter);
        });
  }
}

} // namespace impl

Tensor& searchsorted_out(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter_opt,
    Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> sorter_maybe_owned =
      at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  at::native::searchsorted_pre_check(
      sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  at::native::resize_output(result, self.sizes());

  // we have two inputs to set right, pre_check checks that they aren't set to
  // opposites
  bool is_right = side_opt ? *side_opt == "right" : right;

  if (self.numel() == 0) {
    return result;
  }

  // for non-contiguous result tensors, we write the output to a contiguous copy
  // so we can later copy back, maintaing the original result tensor
  Tensor out = result;
  if (!result.is_contiguous()) {
    out = result.contiguous();
  }
  if (sorted_sequence.is_contiguous() && self.is_contiguous() &&
      sorted_sequence.dtype() == self.dtype() && sorter.is_contiguous()) {
    at::AtenIpexTypeXPU::impl::dispatch(
        out, self, sorted_sequence, out_int32, is_right, sorter);
  } else {
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    at::native::searchsorted_maybe_trim_input_tensors(
        trimmed_input,
        trimmed_boundaries,
        trimmed_sorter,
        self,
        sorted_sequence,
        sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries =
        trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter =
        trimmed_sorter.defined() ? trimmed_sorter : sorter;
    at::AtenIpexTypeXPU::impl::dispatch(
        out, final_input, final_boundaries, out_int32, is_right, final_sorter);
  }

  // if result is non-contiguous, we wrote the answer to a copied version, so we
  // copy back to the original result tensor
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
  return result;
}

Tensor searchsorted(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter_opt) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::AtenIpexTypeXPU::searchsorted_out(
      sorted_sequence, self, out_int32, right, side_opt, sorter_opt, result);
  return result;
}

Tensor searchsorted(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter_opt) {
  const Tensor& scalar_tensor =
      at::native::searchsorted_scalar_tensor(self, sorted_sequence.device());
  return at::AtenIpexTypeXPU::searchsorted(
      sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt);
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
      boundaries, self, out_int32, right, nullopt, nullopt, result);
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
