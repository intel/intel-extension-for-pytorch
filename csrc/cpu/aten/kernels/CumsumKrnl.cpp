#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

#include <aten/Cumsum.h>

#include <immintrin.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at::vec;
using namespace torch_ipex::cpu::kernel;

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename scalar_t>
static inline void cumsum_lastdim_kernel(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim) {
  const auto input_ndim = self.dim();
  TORCH_CHECK(
      dim == input_ndim - 1,
      "cumsum_lastdim_kernel: expect dim to be ",
      input_ndim - 1,
      " got ",
      dim);
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "cumsum_lastdim_kernel: expect same data type for self and result");

  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return;
  }
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  int64_t N = self.size(dim);
  int64_t M = self.numel() / N;
  const scalar_t* self_data = self.data_ptr<scalar_t>();
  scalar_t* result_data = result.data_ptr<scalar_t>();

  int64_t T = at::get_num_threads();

  // bytes per core for each chunk, set to 256KB (L2 cache reside)
  constexpr int64_t CHUNK_SIZE_PER_CORE = 256 * 1024 / sizeof(scalar_t);
  int64_t CHUNK_SIZE = std::max(int64_t(1), CHUNK_SIZE_PER_CORE / M * T);
  int64_t K = divup(N, CHUNK_SIZE);

  // offset value per chunk
  std::vector<scalar_t> outer_offsets(M, scalar_t(0));

  // offset value per thread
  std::vector<scalar_t> inner_offsets(M * T, scalar_t(0));

  for (int64_t k = 0; k < K; k++) {
    int64_t k_begin = k * CHUNK_SIZE;
    int64_t k_end = std::min(k_begin + CHUNK_SIZE, N);

    // Parallel Path I: accumulate locally per thread
    at::parallel_for(k_begin, k_end, 1, [&](int64_t begin, int64_t end) {
      int64_t tid = at::get_thread_num();
      for (int64_t m = 0; m < M; m++) {
        const scalar_t* self_ptr = self_data + m * N + begin;
        scalar_t* result_ptr = result_data + m * N + begin;
        int64_t len = end - begin;

        prefix_sum<scalar_t>(self_ptr, result_ptr, scalar_t(0), len);
        inner_offsets[m * T + tid] = result_ptr[len - 1];
      }
    });

    // update offset value for each thread
    for (int64_t m = 0; m < M; m++) {
      for (int64_t t = T - 1; t >= 0; t--) {
        scalar_t offset = scalar_t(0);
        for (int64_t i = t - 1; i >= 0; i--) {
          offset += inner_offsets[m * T + i];
        }
        inner_offsets[m * T + t] = offset;
      }
    }

    // Parallel Path II: apply offset (result should be in L2)
    at::parallel_for(k_begin, k_end, 1, [&](int64_t begin, int64_t end) {
      int64_t tid = at::get_thread_num();
      for (int64_t m = 0; m < M; m++) {
        scalar_t* result_ptr = result_data + m * N + begin;
        int64_t len = end - begin;

        scalar_t offset = outer_offsets[m] + inner_offsets[m * T + tid];
        at::vec::map(
            [=](Vectorized<scalar_t> x) {
              return x + Vectorized<scalar_t>(offset);
            },
            result_ptr,
            result_ptr,
            len);
      }
    });

    // reinit inner offset value
    std::fill(inner_offsets.begin(), inner_offsets.end(), scalar_t(0));

    // update outer offset value
    for (int64_t m = 0; m < M; m++) {
      outer_offsets[m] = result_data[m * N + k_end - 1];
    }
  }
}

bool cumsum_fast_path(
    const at::Tensor& self,
    const at::Tensor& result,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  // check contiguous
  bool is_contig = self.is_contiguous() && (result.is_contiguous());
  if (!is_contig)
    return false;
  // check dim
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  if (wrap_dim != self.dim() - 1)
    return false;
  // check dtype matched
  auto out_dtype = result.scalar_type();
  if (dtype.has_value() && out_dtype != dtype.value())
    return false;
  // check dtype enabled
  bool is_dtype_enabled = out_dtype == at::ScalarType::Double ||
      out_dtype == at::ScalarType::Float || out_dtype == at::ScalarType::Long;
  if (!is_dtype_enabled)
    return false;
  return true;
}

class NewCumSumOp : public torch::autograd::Function<NewCumSumOp> {
 public:
  static at::Tensor _forward(
      at::Tensor& result,
      const at::Tensor& self,
      int64_t dim,
      c10::optional<at::ScalarType> dtype) {
    RECORD_FUNCTION("IPEXCumSumOp::_forward", c10::ArrayRef<c10::IValue>({}));

    if (result.sizes() != self.sizes()) {
      at::native::resize_output(result, self.sizes());
    }
    if (cumsum_fast_path(result, self, dim, dtype)) {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::Long, self.scalar_type(), "cumsum_lastdim_cpu", [&] {
            cumsum_lastdim_kernel<scalar_t>(result, self, dim);
          });
      return result;
    }
    return at::cumsum_out(result, self, dim, dtype);
  }

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor& result,
      const at::Tensor& self,
      int64_t dim,
      c10::optional<at::ScalarType> dtype) {
    RECORD_FUNCTION("IPEXCumSumOp::forward", c10::ArrayRef<c10::IValue>({}));

    at::AutoDispatchBelowADInplaceOrView g;
    ctx->saved_data["dim"] = dim;
    auto ret = _forward(result, self, dim, dtype);
    return ret;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    RECORD_FUNCTION("IPEXCumSumOp::backward", c10::ArrayRef<c10::IValue>({}));

    at::AutoDispatchBelowADInplaceOrView g;
    int64_t dim = ctx->saved_data["dim"].toInt();

    at::Tensor grad_out = grad_outputs[0];
    at::Tensor grad_self;
    if (grad_out.numel() <= 1 || grad_out.size(dim) == 1) {
      grad_self = grad_out;
    }
    grad_self = grad_out.flip(dim).cumsum(dim).flip(dim);
    return {at::Tensor(), grad_self, at::Tensor(), at::Tensor()};
  }
};

at::Tensor cumsum_kernel_impl(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  if (at::GradMode::is_enabled() && self.requires_grad())
    return NewCumSumOp::apply(result, self, dim, dtype);
  return NewCumSumOp::_forward(result, self, dim, dtype);
}

} // anonymous namespace

REGISTER_DISPATCH(cumsum_kernel_stub, &cumsum_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
