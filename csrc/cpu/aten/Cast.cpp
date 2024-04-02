#include <ATen/Parallel.h>
#include "csrc/utils/CustomOperatorRegistration.h"
#include "fp8_utils.h"

namespace torch_ipex {
namespace cpu {

using namespace torch_ipex::cpu;

at::ScalarType convert_to_dtype(int64_t format) {
  switch (format) {
    case Float8Format::kFloat8_E5M2:
      return ScalarType::Float8_e5m2;
    case Float8Format::kFloat8_E4M3:
      return ScalarType::Float8_e4m3fn;
    default:
      TORCH_CHECK(false, "undefined format.\n");
  }
}

template <typename scalar_t, typename type_t>
void fp8_quantize_impl(
    at::Tensor& input,
    at::Tensor& scale,
    at::Tensor& amax_history,
    at::Tensor& scale_inv,
    int64_t& fp8_tensor_index,
    at::Tensor& output) {
  RECORD_FUNCTION("fp8_quantize_impl", c10::ArrayRef<c10::IValue>({}));

  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* input_ptr = input_data;
  float* scale_ptr = scale.data_ptr<float>();
  float scale_val = scale_ptr[fp8_tensor_index];
  type_t* output_data = output.data_ptr<type_t>();
  type_t* output_ptr = output_data;

  int64_t fp8_max =
      output.scalar_type() == c10::ScalarType::Float8_e4m3fn ? 448 : 57344;
  int num_threads = at::get_num_threads();
  std::vector<float> max_buffer(num_threads, 0);
  at::parallel_for(
      0,
      input.numel(),
      at::internal::GRAIN_SIZE,
      [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        float local_max = 0;
        for (const auto n : c10::irange(begin, end)) {
          float out = input_ptr[n] * scale_val;
          float out_saturation = out;
          if (fabsf(out) > fp8_max) {
            out_saturation = out > 0 ? fp8_max : (-1.0) * fp8_max;
          }
          output_ptr[n] = static_cast<type_t>(out_saturation);
          local_max = fmaxf(fabsf(input_ptr[n]), local_max);
        }
        max_buffer[tid] = local_max;
      });

  float max = *std::max_element(max_buffer.begin(), max_buffer.end());
  float* scale_inv_ptr = scale_inv.data_ptr<float>();
  scale_inv_ptr[fp8_tensor_index] = 1.0 / scale_val;
  amax_history[fp8_tensor_index] = max;
}

template <typename scalar_t>
void fp8_quantize(
    at::Tensor& input,
    at::Tensor& scale,
    at::Tensor& amax_history,
    at::Tensor& scale_inv,
    int64_t& fp8_tensor_index,
    at::Tensor& output,
    int64_t otype) {
  IPEX_TYPE_SWITCH_FP8ONLY(
      otype,
      type_t,
      fp8_quantize_impl<scalar_t, type_t>(
          input, scale, amax_history, scale_inv, fp8_tensor_index, output););
}

at::Tensor cast_to_fp8(
    at::Tensor& input,
    at::Tensor& scale,
    at::Tensor& amax_history,
    at::Tensor& scale_inv,
    int64_t fp8_tensor_index,
    int64_t otype) {
  at::ScalarType out_type = convert_to_dtype(otype);
  auto output = at::empty_like(input, out_type);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "fp8_quantize",
      [&] {
        fp8_quantize<scalar_t>(
            input,
            scale,
            amax_history,
            scale_inv,
            fp8_tensor_index,
            output,
            otype);
      });
  return output;
}

template <typename scalar_t, typename type_t>
at::Tensor fp8_dequantize_impl(
    at::Tensor& input,
    at::Tensor& scale_inv,
    int64_t fp8_tensor_index,
    at::Tensor& output) {
  RECORD_FUNCTION("fp8_dequantize_impl", c10::ArrayRef<c10::IValue>({}));

  type_t* input_data = input.data_ptr<type_t>();
  type_t* input_ptr = input_data;
  float* scale_inv_ptr = scale_inv.data_ptr<float>();
  float scale_inv_val = scale_inv_ptr[fp8_tensor_index];
  scalar_t* output_data = output.data_ptr<scalar_t>();
  scalar_t* output_ptr = output_data;

  at::parallel_for(
      0,
      input.numel(),
      at::internal::GRAIN_SIZE,
      [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        for (const auto n : c10::irange(begin, end)) {
          output_ptr[n] = static_cast<scalar_t>(input_ptr[n] * scale_inv_val);
        }
      });
  return output;
}

template <typename scalar_t>
void fp8_dequantize(
    at::Tensor& input,
    at::Tensor& scale_inv,
    int64_t fp8_tensor_index,
    at::Tensor& output,
    int64_t itype) {
  IPEX_TYPE_SWITCH_FP8ONLY(itype,
                           type_t,
                           fp8_dequantize_impl<scalar_t, type_t>(
                               input, scale_inv, fp8_tensor_index, output););
}

at::Tensor cast_from_fp8(
    at::Tensor input,
    at::Tensor& scale_inv,
    int64_t fp8_tensor_index,
    int64_t itype,
    ScalarType otype) {
  auto output = at::empty_like(input, otype);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      output.scalar_type(),
      "fp8_dequantize",
      [&] {
        fp8_dequantize<scalar_t>(
            input, scale_inv, fp8_tensor_index, output, itype);
      });
  return output;
}

} // namespace cpu
} // namespace torch_ipex

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "cast_to_fp8", torch_ipex::cpu::cast_to_fp8, c10::DispatchKey::CPU);
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "cast_from_fp8", torch_ipex::cpu::cast_from_fp8, c10::DispatchKey::CPU);
}

} // namespace
