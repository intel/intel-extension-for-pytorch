#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AtenIpexTypeXPU.h>

#include <core/DPCPP.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

// Note: dpcpp compiler does not support uname type in template.
class SyclOpMulAdd {};

static void mul_add_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mul_add",
      [&]() {
        auto alpha = alpha_scalar.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<SyclOpMulAdd>(
            iter, [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
              return a * b + alpha * c;
            });
      });
}

// Basic checking for all input tensors.
static inline void dim_check(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu) {
  int64_t self_ndims = self.ndimension();
  int64_t other_ndims = other.ndimension();
  int64_t accumu_ndims = accumu.ndimension();

  TORCH_CHECK(
      self_ndims == other_ndims || other_ndims == accumu_ndims,
      "The dimensions of three inputs tensor not equal is not supported. ");
}

} // impl

Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha) {
  impl::dim_check(self, other, accumu);
  Tensor _self, _other, _accumu, result;
  if (check_has_opaque_and_no_padding({self, other, accumu})) {
    std::vector<Tensor> inputs;
    inputs.push_back(self);

    // align shape
    if (self.numel() != other.numel())
      inputs.push_back(other.expand_as(self).contiguous());
    else
      inputs.push_back(other);

    if (self.numel() != accumu.numel())
      inputs.push_back(accumu.expand_as(self).contiguous());
    else
      inputs.push_back(accumu);

    // align format
    std::vector<Tensor> _inputs;

    Tensor tar;
    for (int i = 0; i < inputs.size(); ++i) {
      if (DPCPPTensorConvertor::is_opaque_tensor(inputs[i])) {
        tar = inputs[i];
        break;
      }
    }

    auto tar_ctx = AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(tar);

    for (int i = 0; i < inputs.size(); ++i) {
      if (!tar.is_same(inputs[i])) {
        Tensor cur = inputs[i];
        auto cur_ctx =
            AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(cur);
        if (cur_ctx.meta() != tar_ctx.meta()) {
          cur = empty_opaque_tensor(
              tar_ctx.meta(), inputs[i].options(), c10::nullopt);
          AtenIpexTypeXPU::DPCPPTensorConvertor::convert(cur, inputs[i]);
        }
        _inputs.push_back(cur);
      } else {
        _inputs.push_back(tar);
      }
    }
    _self = _inputs.at(0);
    _other = _inputs.at(1);
    _accumu = _inputs.at(2);
    result = empty_opaque_tensor(tar_ctx.meta(), tar.options(), c10::nullopt);
  } else {
    _self = to_plain_if_needed(self);
    _other = to_plain_if_needed(other);
    _accumu = to_plain_if_needed(accumu);
    result = at::empty_like(self);
  }

  auto iter = TensorIteratorConfig()
  .set_check_mem_overlap(true)
  .add_output(result)
  .add_input(_self)
  .add_input(_other)
  .add_input(_accumu)
  .build();
  impl::mul_add_kernel_dpcpp(iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

template <typename T>
class PackedAdd_ker {};

template <typename scalar_t>
static inline void packed_add_kernel(
    unsigned short* __restrict__ w_MSB,
    unsigned short* __restrict__ w_LSB,
    const at::BFloat16* __restrict__ gw,
    int num_elem,
    float lr) {
  union packed_bf16 {
    unsigned short s[2];
    float f;
  };
  static const auto read_mode = DPCPP::access::mode::read;
  static const auto write_mode = DPCPP::access::mode::write;
  static const auto read_write_mode = DPCPP::access::mode::read_write;
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

  auto cgf = DPCPP_Q_CGF(cgh) {
      auto MSB_data = get_buffer<read_write_mode>(cgh, w_MSB);
      auto LSB_data = get_buffer<read_write_mode>(cgh, w_LSB);
      auto gw_data = get_buffer<read_mode>(cgh, gw);

      cgh.parallel_for<PackedAdd_ker<scalar_t>>(
        DPCPP::range<1>(num_elem), [=](DPCPP::item<1> item) {

          int64_t gid = item.get_linear_id();
          auto MSB_p = get_pointer(MSB_data);
          auto LSB_p = get_pointer(LSB_data);
          auto gw_p = get_pointer(gw_data);

          packed_bf16 p16;
          p16.s[0] = LSB_p[gid];
          p16.s[1] = MSB_p[gid];
          p16.f += lr * (float)(gw_p[gid]);
          LSB_p[gid] = p16.s[0];
          MSB_p[gid] = p16.s[1];
      });
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

Tensor packed_add(
    at::Tensor & top_half,
    at::Tensor & bot_half,
    const at::Tensor & grad,
    float alpha) {
  RECORD_FUNCTION("packed_add", std::vector<c10::IValue>({top_half, bot_half, grad}));
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      top_half.scalar_type(),
      "packed_add_kernel",
      [&]() {
        packed_add_kernel<scalar_t>(
            (unsigned short *)top_half.data_ptr<scalar_t>(),
            (unsigned short *)bot_half.data_ptr<scalar_t>(),
            grad.data_ptr<at::BFloat16>(),
            top_half.numel(),
            static_cast<float>(alpha));
      });
  return top_half;
}

} // namespace AtenIpexTypeXPU
} // namespace at
