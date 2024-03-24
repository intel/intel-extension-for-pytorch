#if defined(USE_XETLA)
#include "XEGEMM_INT4.h"
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

#define GEMM_QKV_WINT4_XETLA_DISPATCH(has_bias, arch)                         \
  if (!has_bias) {                                                            \
    auto policy =                                                             \
        HGEMMXetla_INT4()                                                     \
            .add_matrix_out(out0)                                             \
            .add_matrix_out(out1)                                             \
            .add_matrix_out(out2)                                             \
            .add_matrix_inp(input)                                            \
            .add_matrix_wei(weight)                                           \
            .add_matrix_scl(weight_scl)                                       \
            .add_matrix_zp(weight_zp)                                         \
            .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SPLIT3)    \
            .add_calib_gz(calib_gz)                                           \
            .add_arch(arch)                                                   \
            .build();                                                         \
    TORCH_CHECK(policy.fallback() == false, "qkv: invalid gemm shape");       \
    policy.run();                                                             \
  } else {                                                                    \
    auto policy =                                                             \
        HGEMMXetla_INT4()                                                     \
            .add_matrix_out(out0)                                             \
            .add_matrix_out(out1)                                             \
            .add_matrix_out(out2)                                             \
            .add_matrix_inp(input)                                            \
            .add_matrix_wei(weight)                                           \
            .add_matrix_scl(weight_scl)                                       \
            .add_matrix_zp(weight_zp)                                         \
            .add_epilogue(bias_.value(), HGEMMXetla_INT4::EpilogueType::BIAS) \
            .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SPLIT3)    \
            .add_calib_gz(calib_gz)                                           \
            .add_arch(arch)                                                   \
            .build();                                                         \
    TORCH_CHECK(policy.fallback() == false, "qkv bias: invalid gemm shape");  \
    policy.run();                                                             \
  }

int8_t GetGpuArchId() noexcept {
  DeviceId curDevID = at::xpu::current_device();
  bool is_2d_block = dpcppGetDeviceHas2DBlock(curDevID);
  bool is_xmx = dpcppGetDeviceHasXMX(curDevID);
  if (is_2d_block && is_xmx) {
    return static_cast<int8_t>(gpu::xetla::gpu_arch::XeHpc);
  } else if (is_xmx) {
    return static_cast<int8_t>(gpu::xetla::gpu_arch::XeHpg);
  } else { // TODO(Yi): distinguish PVC-VG from MTL which supports 2d-block
    return static_cast<int8_t>(gpu::xetla::gpu_arch::XeLpg);
  }
}
static void mm_qkv_out_wint4(
    const Tensor& input_,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const optional<Tensor>& bias_,
    const Tensor& out0_,
    const Tensor& out1_,
    const Tensor& out2_,
    int64_t calib_gz) {
  auto input = input_.flatten(0, -2);
  if (input.scalar_type() == ScalarType::Float)
    input = input.to(at::kHalf);
  auto out0 = out0_.flatten(0, -2);
  auto out1 = out1_.flatten(0, -2);
  auto out2 = out2_.flatten(0, -2);
  // input: m,k; weight: 3,k,n, bias(opt): 3,n
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3);
  TORCH_CHECK(out0.dim() == 2 && out1.dim() == 2 && out2.dim() == 2);
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int n = weight.sizes()[2] * 2;

  bool has_bias = bias_.has_value();
  if (has_bias) {
    auto bias = bias_.value();
    TORCH_CHECK(
        bias.dim() == 2 && bias.sizes()[0] == 3 && bias.sizes()[1] == n);
  }
  TORCH_CHECK(
      out0.sizes()[0] == m && out1.sizes()[0] == m && out2.sizes()[0] == m);
  TORCH_CHECK(
      out0.sizes()[1] == n && out1.sizes()[1] == n && out2.sizes()[1] == n);

  bool is_a_contiguous = input.is_contiguous();
  bool is_b_row_major = weight.is_contiguous();

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(
      input.scalar_type() == kHalf &&
      (weight.scalar_type() == kQUInt8 || weight.scalar_type() == kByte ||
       weight.scalar_type() == kChar));

  GEMM_QKV_WINT4_XETLA_DISPATCH(has_bias, GetGpuArchId());
}

#undef GEMM_QKV_WINT4_XETLA_DISPATCH

static std::tuple<Tensor, Tensor, Tensor> mm_qkv_wint4(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[2] * 2;
  auto out0 = at::empty({m, n}, input.options());
  auto out1 = at::empty({m, n}, input.options());
  auto out2 = at::empty({m, n}, input.options());
  mm_qkv_out_wint4(
      input, weight, weight_scl, weight_zp, bias_, out0, out1, out2, calib_gz);
  auto sizes = input.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return std::forward_as_tuple(
      out0.view_symint(sizes),
      out1.view_symint(sizes),
      out2.view_symint(sizes));
}

static Tensor mm_bias_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);

  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[1] * 2;
  auto bias = bias_.flatten();
  auto output = at::empty({m, n}, input.options());

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input_flat)
                    .add_matrix_wei(weight_flat)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_epilogue(bias, HGEMMXetla_INT4::EpilogueType::BIAS)
                    .add_calib_gz(calib_gz)
                    .add_arch(GetGpuArchId())
                    .build();
  TORCH_CHECK(policy.fallback() == false, "mm bias int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);

  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[1] * 2;
  auto output = at::empty({m, n}, input.options());

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);

  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input_flat)
                    .add_matrix_wei(weight_flat)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_calib_gz(calib_gz)
                    .add_arch(GetGpuArchId())
                    .build();
  TORCH_CHECK(policy.fallback() == false, "mm int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_silu_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());
  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input_flat)
                    .add_matrix_wei(weight_flat)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SILU)
                    .add_calib_gz(calib_gz)
                    .add_arch(GetGpuArchId())
                    .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_resmul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& res,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());
  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL)
          .add_calib_gz(calib_gz)
          .add_arch(GetGpuArchId())
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm resmul int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_gelu_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& bias,
    int64_t calib_gz,
    c10::string_view approximate) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto bias_flat = bias.flatten();
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  TORCH_CHECK(approximate == "tanh");
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());
  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS)
          .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::GELU)
          .add_calib_gz(calib_gz)
          .add_arch(GetGpuArchId())
          .build();
  TORCH_CHECK(
      policy.fallback() == false, "mm bias gelu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_resadd_resadd_int4(
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& res0_,
    const Tensor& res1_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input = input_.flatten(0, -2);
  auto weight = weight_.flatten(0, -2);
  auto bias = bias_.flatten();
  auto res0 = res0_.flatten(0, -2);
  auto res1 = res1_.flatten(0, -2);
  if (input.scalar_type() == ScalarType::Float)
    input = input.to(at::kHalf);
  // a: m x k, b: k x n, bias: n, res0/1: m x n
  TORCH_CHECK(
      input.dim() == 2 && weight.dim() == 2 && bias.dim() == 1 &&
      res0.dim() == 2 && res1.dim() == 2);
  int m = input.sizes()[0];
  int n = weight.sizes()[1] * 2;
  int k = input.sizes()[1];
  auto output = at::empty({m, n}, input.options());
  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input)
                    .add_matrix_wei(weight)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_epilogue(bias, HGEMMXetla_INT4::EpilogueType::BIAS)
                    .add_epilogue(res0, HGEMMXetla_INT4::EpilogueType::RES_ADD)
                    .add_epilogue(res1, HGEMMXetla_INT4::EpilogueType::RES_ADD)
                    .add_calib_gz(calib_gz)
                    .add_arch(GetGpuArchId())
                    .build();
  TORCH_CHECK(
      policy.fallback() == false,
      "mm bias resadd resadd int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input_, output);
}

static Tensor mm_low_bits(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& bias,
    bool has_bias,
    const std::string& compute_dtype,
    const std::string& weight_dtype,
    int64_t calib_gz) {
  return has_bias
      ? mm_bias_int4(input, weight, bias, weight_scl, weight_zp, calib_gz)
      : mm_int4(input, weight, weight_scl, weight_zp, calib_gz);
}

static Tensor mm_silu_mul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SILU)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL)
          .add_calib_gz(calib_gz)
          .add_arch(GetGpuArchId())
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_silu_mul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  auto bias_flat = bias.flatten();
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS)
          .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SILU)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL)
          .add_calib_gz(calib_gz)
          .add_arch(GetGpuArchId())
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_add_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD)
          .add_calib_gz(calib_gz)
          .add_arch(GetGpuArchId())
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm add int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_add_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  auto bias_flat = bias.flatten();
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD)
          .add_calib_gz(calib_gz)
          .add_arch(GetGpuArchId())
          .build();
  TORCH_CHECK(
      policy.fallback() == false, "mm bias add int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

bool is_match_total_mem(
    Tensor tensor,
    IntArrayRef new_sizes,
    c10::ScalarType new_type) {
  auto old_element = tensor.numel() * c10::elementSize(tensor.scalar_type());
  auto new_element = std::accumulate(
      new_sizes.begin(),
      new_sizes.end(),
      static_cast<int64_t>(1),
      std::multiplies<>());
  return old_element == new_element;
}

template <class NewType>
Tensor recast(
    const Tensor& tensor,
    IntArrayRef sizes,
    IntArrayRef strides,
    std::string info = "") {
  auto new_dtype = c10::CppTypeToScalarType<NewType>::value;
  auto device = tensor.device();
  TORCH_CHECK(
      is_match_total_mem(tensor, sizes, new_dtype),
      info,
      "recast failed : The new type must have the same total number of elements as the old type.");
  return at::from_blob(
      (void*)tensor.data_ptr(),
      sizes,
      strides,
      nullptr,
      at::device(device).dtype(new_dtype),
      {device});
}

Tensor _weight_int4pack_mm_xpu(
    const Tensor& A,
    const Tensor& B,
    int64_t qGroupSize,
    const Tensor& qScaleAndZeros) {
  constexpr int64_t kNTileSize = 8;

  auto M = A.size(0);
  auto N = B.size(0) * kNTileSize;
  auto K = A.size(1);

  TORCH_CHECK(
      A.dtype() == kBFloat16, __func__, " : expect A to be bfloat16 tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == kInt, __func__, " : expect B to be int32 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(B.dim() == 4, __func__, " : expect B to 4d tensor.");

  TORCH_CHECK(
      qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
          qGroupSize == 256,
      __func__,
      ": expect qGroupSize to be 32, 64, 128 or 256, got ",
      qGroupSize);

  TORCH_CHECK(
      qScaleAndZeros.dim() == 3 && qScaleAndZeros.size(1) == N &&
          qScaleAndZeros.size(2) == 2,
      __func__,
      ": expect qScaleAndZeros to be 3d tensor with sizes [:, ",
      N,
      ", 2]");
  auto C = at::empty({M, N}, A.options());
  auto q_scale_and_zeros_vec = at::split(qScaleAndZeros, 1, -1);
  auto b_recast = recast<uint8_t>(
      B, {K, N / 2}, {N / 2, 1}, "_weight_int4pack_mm_xpu tensor B");
  auto q_scale = q_scale_and_zeros_vec[0].reshape({-1, N}).contiguous();
  auto q_zeros = q_scale_and_zeros_vec[1].reshape({-1, N}).contiguous();
  TORCH_CHECK(A.dim() == 2 && b_recast.dim() == 2);

  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(C)
          .add_matrix_inp(A)
          .add_matrix_wei(b_recast)
          .add_matrix_scl(q_scale)
          .add_matrix_zp(q_zeros)
          .add_calib_gz(qGroupSize)
          .add_arch(GetGpuArchId())
          .add_quant_mode(
              torch_ipex::xpu::xetla::quant_mode::S4_ASYM_ZERO_NO_DEGRAD)
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(A, C);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "mm_qkv_out_int4.xpu", at::AtenIpexTypeXPU::mm_qkv_out_wint4);
  IPEX_OP_REGISTER("mm_qkv_int4.xpu", at::AtenIpexTypeXPU::mm_qkv_wint4);
  IPEX_OP_REGISTER("mm_int4.xpu", at::AtenIpexTypeXPU::mm_int4);
  IPEX_OP_REGISTER("mm_bias_int4.xpu", at::AtenIpexTypeXPU::mm_bias_int4);
  IPEX_OP_REGISTER("mm_silu_int4.xpu", at::AtenIpexTypeXPU::mm_silu_int4);
  IPEX_OP_REGISTER("mm_resmul_int4.xpu", at::AtenIpexTypeXPU::mm_resmul_int4);
  IPEX_OP_REGISTER(
      "mm_bias_gelu_int4.xpu", at::AtenIpexTypeXPU::mm_bias_gelu_int4);
  IPEX_OP_REGISTER(
      "mm_bias_resadd_resadd_int4.xpu",
      at::AtenIpexTypeXPU::mm_bias_resadd_resadd_int4);
  IPEX_OP_REGISTER("mm_low_bits.xpu", at::AtenIpexTypeXPU::mm_low_bits);
  IPEX_OP_REGISTER(
      "mm_silu_mul_int4.xpu", at::AtenIpexTypeXPU::mm_silu_mul_int4);
  IPEX_OP_REGISTER(
      "mm_bias_silu_mul_int4.xpu", at::AtenIpexTypeXPU::mm_bias_silu_mul_int4);
  IPEX_OP_REGISTER("mm_add_int4.xpu", at::AtenIpexTypeXPU::mm_add_int4);
  IPEX_OP_REGISTER(
      "mm_bias_add_int4.xpu", at::AtenIpexTypeXPU::mm_bias_add_int4);
}

TORCH_LIBRARY_FRAGMENT(aten, m) {
  m.def(
      "_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros) -> Tensor");
  m.impl(
      "_weight_int4pack_mm",
      c10::DispatchKey::XPU,
      at::AtenIpexTypeXPU::_weight_int4pack_mm_xpu);
}
} // namespace
#endif
