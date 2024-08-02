#if defined(USE_XETLA)
#include "XEGEMM_INT4.h"
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "oneDNN/WoqMatmul.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

bool choose_recommand_compute_eng() {
  auto compute_eng = Settings::I().get_compute_eng();
  return compute_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA ||
      compute_eng == torch_ipex::xpu::COMPUTE_ENG::RECOMMEND;
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
    Tensor& out0_,
    Tensor& out1_,
    Tensor& out2_,
    int64_t group_size) {
  if (choose_recommand_compute_eng()) {
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
    int n = weight.sizes()[1];

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
        input.scalar_type() == kHalf || input.scalar_type() == kBFloat16);
    TORCH_CHECK(
        weight.scalar_type() == kQUInt8 || weight.scalar_type() == kByte ||
        weight.scalar_type() == kChar || weight.scalar_type() == kInt)

    auto policy = HGEMMXetla_INT4()
                      .add_matrix_out(out0)
                      .add_matrix_out(out1)
                      .add_matrix_out(out2)
                      .add_matrix_inp(input)
                      .add_matrix_wei(weight)
                      .add_matrix_scl(weight_scl)
                      .add_matrix_zp(weight_zp);
    if (has_bias)
      policy.add_epilogue(bias_.value(), HGEMMXetla_INT4::EpilogueType::BIAS);
    policy.add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SPLIT3)
        .add_group_size(group_size)
        .add_arch(GetGpuArchId())
        .build();
    TORCH_CHECK(
        policy.fallback() == false,
        has_bias ? "qkv bias: " : "qkv: ",
        "invalid gemm shape");
    policy.run();
  } else {
    Attr attr;
    if (bias_.has_value()) {
      auto bias = bias_.value();
      out0_ = torch_ipex::xpu::oneDNN::woq_matmul_int4(
          out0_,
          input_,
          weight[0],
          weight_scl[0],
          weight_zp[0],
          group_size,
          false,
          attr,
          bias[0]);
      out1_ = torch_ipex::xpu::oneDNN::woq_matmul_int4(
          out1_,
          input_,
          weight[1],
          weight_scl[1],
          weight_zp[1],
          group_size,
          false,
          attr,
          bias[1]);
      out2_ = torch_ipex::xpu::oneDNN::woq_matmul_int4(
          out2_,
          input_,
          weight[2],
          weight_scl[2],
          weight_zp[2],
          group_size,
          false,
          attr,
          bias[2]);
    } else {
      out0_ = torch_ipex::xpu::oneDNN::woq_matmul_int4(
          out0_,
          input_,
          weight[0],
          weight_scl[0],
          weight_zp[0],
          group_size,
          false,
          attr);
      out1_ = torch_ipex::xpu::oneDNN::woq_matmul_int4(
          out1_,
          input_,
          weight[1],
          weight_scl[1],
          weight_zp[1],
          group_size,
          false,
          attr);
      out2_ = torch_ipex::xpu::oneDNN::woq_matmul_int4(
          out2_,
          input_,
          weight[2],
          weight_scl[2],
          weight_zp[2],
          group_size,
          false,
          attr);
    }
  }
}

static std::tuple<Tensor, Tensor, Tensor> mm_qkv_wint4(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size) {
  auto input_flat = input.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[1];
  auto out0 = at::empty({m, n}, input.options());
  auto out1 = at::empty({m, n}, input.options());
  auto out2 = at::empty({m, n}, input.options());
  mm_qkv_out_wint4(
      input,
      weight,
      weight_scl,
      weight_zp,
      bias_,
      out0,
      out1,
      out2,
      group_size);
  if (choose_recommand_compute_eng()) {
    auto sizes = input.sym_sizes().vec();
    sizes[sizes.size() - 1] = n;
    return std::forward_as_tuple(
        out0.view_symint(sizes),
        out1.view_symint(sizes),
        out2.view_symint(sizes));
  } else {
    return std::forward_as_tuple(out0, out1, out2);
  }
}

// mlp operators naming convention:
// mlp_[<gate_postop>]..._<binary_op>_[<up_postop>]...[_out]_int4
static inline HGEMMXetla_INT4 mlp_mul_dispatch(
    const Tensor& input_,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    int64_t group_size,
    const std::vector<std::tuple<const Tensor&, HGEMMXetla_INT4::EpilogueType>>&
        gate_post_ops,
    const std::vector<std::tuple<const Tensor&, HGEMMXetla_INT4::EpilogueType>>&
        up_post_ops,
    Tensor* const output) {
  auto input = input_.flatten(0, -2);
  if (input.scalar_type() == ScalarType::Float)
    input = input.to(at::kHalf);
  // input: m,k; gate_up_wei: 2,n,k; gate_proj: n,k
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int n = gate_up_wei.sizes()[1];
  *output = output->defined() ? output->flatten(0, -2)
                              : at::empty({m, n}, input.options());
  TORCH_CHECK(input.is_contiguous() && gate_up_wei.is_contiguous());
  TORCH_CHECK(input.scalar_type() == kHalf || input.scalar_type() == kBFloat16);
  TORCH_CHECK(
      gate_up_wei.scalar_type() == kChar ||
      gate_up_wei.scalar_type() == kByte ||
      gate_up_wei.scalar_type() == kQUInt8 ||
      gate_up_wei.scalar_type() == kInt);

  auto dispatcher = HGEMMXetla_INT4()
                        .add_matrix_out(*output)
                        .add_matrix_inp(input)
                        .add_matrix_wei(gate_up_wei)
                        .add_matrix_scl(gate_up_wei_scl)
                        .add_matrix_zp(gate_up_wei_zp);
  for (auto& [epilogue_, epilogue_type] : gate_post_ops) {
    dispatcher.add_epilogue(epilogue_, epilogue_type);
  }
  dispatcher.add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::GATE_UP_MUL);
  for (auto& [epilogue_, epilogue_type] : up_post_ops) {
    dispatcher.add_epilogue(epilogue_, epilogue_type);
  }
  dispatcher //
      .add_group_size(group_size)
      .add_arch(GetGpuArchId())
      .build();
  TORCH_CHECK(dispatcher.fallback() == false, "mlp_mul: invalid gemm config");
  dispatcher.run();
  return dispatcher;
}
// silu(linear(input, gate_wei)) * linear(input, up_wei)
static Tensor mlp_silu_mul_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    int64_t group_size) {
  Tensor out;
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {},
      &out);
  return resize_as_mat1(input, out);
}
// silu(linear(input, gate_wei)) * linear(input, up_wei)
static void mlp_silu_mul_out_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    Tensor& out,
    int64_t group_size) {
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {},
      &out);
  return;
}
// silu(linear(input, gate_wei) + gate_bias) * linear(input, up_wei)
static Tensor mlp_bias_silu_mul_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    const Tensor& gate_bias,
    int64_t group_size) {
  Tensor out;
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{gate_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS},
       {{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {},
      &out);
  return resize_as_mat1(input, out);
}
// silu(linear(input, gate_wei) + gate_bias) * linear(input, up_wei)
static void mlp_bias_silu_mul_out_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    Tensor& out,
    const Tensor& gate_bias,
    int64_t group_size) {
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{gate_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS},
       {{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {},
      &out);
  return;
}
// silu(linear(input, gate_wei)) * (linear(input, up_wei) + up_bias)
static Tensor mlp_silu_mul_bias_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    const Tensor& up_bias,
    int64_t group_size) {
  Tensor out;
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {{up_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS}},
      &out);
  return resize_as_mat1(input, out);
}
// silu(linear(input, gate_wei)) * (linear(input, up_wei) + up_bias)
static void mlp_silu_mul_bias_out_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    Tensor& out,
    const Tensor& up_bias,
    int64_t group_size) {
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {{up_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS}},
      &out);
  return;
}
// silu(linear(input, gate_wei) + gate_bias) * (linear(input, up_wei) + up_bias)
static Tensor mlp_bias_silu_mul_bias_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    const Tensor& gate_bias,
    const Tensor& up_bias,
    int64_t group_size) {
  Tensor out;
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{gate_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS},
       {{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {{up_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS}},
      &out);
  return resize_as_mat1(input, out);
}
// silu(linear(input, gate_wei) + gate_bias) * (linear(input, up_wei) + up_bias)
static void mlp_bias_silu_mul_bias_out_int4(
    const Tensor& input,
    const Tensor& gate_up_wei,
    const Tensor& gate_up_wei_scl,
    const Tensor& gate_up_wei_zp,
    const Tensor& up_bias,
    Tensor& out,
    const Tensor& gate_bias,
    int64_t group_size) {
  mlp_mul_dispatch(
      input,
      gate_up_wei,
      gate_up_wei_scl,
      gate_up_wei_zp,
      group_size,
      {{gate_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS},
       {{}, HGEMMXetla_INT4::EpilogueType::SILU}},
      {{up_bias.flatten(), HGEMMXetla_INT4::EpilogueType::BIAS}},
      &out);
  return;
}

static inline HGEMMXetla_INT4 mm_int4_dispatch(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size,
    const std::vector<std::tuple<const Tensor&, HGEMMXetla_INT4::EpilogueType>>&
        epilogues,
    Tensor* const output) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  if (input_flat.scalar_type() == ScalarType::Float)
    input_flat = input_flat.to(at::kHalf);
  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight_flat.sizes()[0];
  *output = output->defined() ? output->flatten(0, -2)
                              : at::empty({m, n}, input.options());
  auto dispatcher = HGEMMXetla_INT4()
                        .add_matrix_out(*output)
                        .add_matrix_inp(input_flat)
                        .add_matrix_wei(weight_flat)
                        .add_matrix_scl(weight_scl)
                        .add_matrix_zp(weight_zp)
                        .add_group_size(group_size)
                        .add_arch(GetGpuArchId());
  for (auto& [epilogue_, epilogue_type] : epilogues) {
    dispatcher.add_epilogue(epilogue_, epilogue_type);
  }
  dispatcher.build();
  TORCH_CHECK(dispatcher.fallback() == false, "mm int4: invalid gemm shape");
  dispatcher.run();
  return dispatcher;
}

static Tensor mm_bias_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto bias = bias_.flatten();
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{bias, HGEMMXetla_INT4::EpilogueType::BIAS}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_int4(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        false,
        attr,
        bias_);
    return out;
  }
}

static Tensor mm_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto dispatcher = mm_int4_dispatch(
        input, weight, weight_scl, weight_zp, group_size, {}, &out);
    return resize_as_mat1(input, out);
  } else {
    at::Tensor bias = Tensor();
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_int4(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        false,
        attr,
        bias);
    return out;
  }
}

static void mm_int4_out(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    Tensor& out,
    int64_t group_size) {
  if (choose_recommand_compute_eng()) {
    auto dispatcher = mm_int4_dispatch(
        input, weight, weight_scl, weight_zp, group_size, {}, &out);
    return;
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_int4(
        out, input, weight, weight_scl, weight_zp, group_size, false, attr);
    return;
  }
}

static Tensor mm_silu_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{Tensor(), HGEMMXetla_INT4::EpilogueType::SILU}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    at::Tensor bias = Tensor();
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_silu(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        false,
        attr,
        bias);
    return out;
  }
}

static Tensor mm_resmul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& res,
    int64_t group_size) {
  auto res_flat = res.flatten(0, -2);
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_resmul(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        res,
        group_size,
        false,
        attr);
    return out;
  }
}

static Tensor mm_bias_gelu_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& bias,
    int64_t group_size,
    c10::string_view approximate) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto bias_flat = bias.flatten();
    TORCH_CHECK(approximate == "tanh");
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS},
         {Tensor(), HGEMMXetla_INT4::EpilogueType::GELU}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_bias_gelu(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        approximate,
        false,
        attr,
        bias);
    return out;
  }
}

static Tensor mm_bias_resadd_resadd_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& res0,
    const Tensor& res1,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    std::cout << "xetla path" << std::endl;
    auto bias_flat = bias.flatten();
    auto res0_flat = res0.flatten(0, -2);
    auto res1_flat = res1.flatten(0, -2);
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS},
         {res0_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD},
         {res1_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    std::cout << "onednn path" << std::endl;
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_bias_resadd_resadd(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        res0,
        res1,
        group_size,
        false,
        attr,
        bias);
    return out;
  }
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
    int64_t group_size) {
  return has_bias
      ? mm_bias_int4(input, weight, bias, weight_scl, weight_zp, group_size)
      : mm_int4(input, weight, weight_scl, weight_zp, group_size);
}

static Tensor mm_silu_mul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size,
    const Tensor& res) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto res_flat = res.flatten(0, -2);
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{Tensor(), HGEMMXetla_INT4::EpilogueType::SILU},
         {res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_silu_mul(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        res,
        group_size,
        false,
        attr);
    return out;
  }
}

static Tensor mm_bias_silu_mul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size,
    const Tensor& res) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto res_flat = res.flatten(0, -2);
    auto bias_flat = bias.flatten();
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS},
         {Tensor(), HGEMMXetla_INT4::EpilogueType::SILU},
         {res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_bias_silu_mul_int4(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        res,
        group_size,
        false,
        attr,
        bias);
    return out;
  }
}

static Tensor mm_add_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size,
    const Tensor& res) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto res_flat = res.flatten(0, -2);
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{res_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_add_int4(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        res,
        group_size,
        false,
        attr);
    return out;
  }
}

static Tensor mm_bias_add_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size,
    const Tensor& res) {
  Tensor out;
  if (choose_recommand_compute_eng()) {
    auto res_flat = res.flatten(0, -2);
    auto bias_flat = bias.flatten();
    auto dispatcher = mm_int4_dispatch(
        input,
        weight,
        weight_scl,
        weight_zp,
        group_size,
        {{bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS},
         {res_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD}},
        &out);
    return resize_as_mat1(input, out);
  } else {
    Attr attr;
    torch_ipex::xpu::oneDNN::woq_matmul_bias_add_int4(
        out,
        input,
        weight,
        weight_scl,
        weight_zp,
        res,
        group_size,
        false,
        attr,
        bias);
    return out;
  }
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

Tensor _weight_int4pack_mm(
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
      B, {K, N / 2}, {N / 2, 1}, "_weight_int4pack_mm tensor B");
  auto q_scale = q_scale_and_zeros_vec[0].reshape({-1, N}).contiguous();
  auto q_zeros = q_scale_and_zeros_vec[1].reshape({-1, N}).contiguous();
  TORCH_CHECK(A.dim() == 2 && b_recast.dim() == 2);

  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(C)
                    .add_matrix_inp(A)
                    .add_matrix_wei(b_recast)
                    .add_matrix_scl(q_scale)
                    .add_matrix_zp(q_zeros)
                    .add_group_size(qGroupSize)
                    .add_arch(GetGpuArchId())
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
  IPEX_OP_REGISTER(
      "mlp_silu_mul_out_int4.xpu", at::AtenIpexTypeXPU::mlp_silu_mul_out_int4);
  IPEX_OP_REGISTER(
      "mlp_silu_mul_int4.xpu", at::AtenIpexTypeXPU::mlp_silu_mul_int4);
  IPEX_OP_REGISTER(
      "mlp_bias_silu_mul_out_int4.xpu",
      at::AtenIpexTypeXPU::mlp_bias_silu_mul_out_int4);
  IPEX_OP_REGISTER(
      "mlp_bias_silu_mul_int4.xpu",
      at::AtenIpexTypeXPU::mlp_bias_silu_mul_int4);
  IPEX_OP_REGISTER(
      "mlp_silu_mul_bias_out_int4.xpu",
      at::AtenIpexTypeXPU::mlp_silu_mul_bias_out_int4);
  IPEX_OP_REGISTER(
      "mlp_silu_mul_bias_int4.xpu",
      at::AtenIpexTypeXPU::mlp_silu_mul_bias_int4);
  IPEX_OP_REGISTER(
      "mlp_bias_silu_mul_bias_out_int4.xpu",
      at::AtenIpexTypeXPU::mlp_bias_silu_mul_bias_out_int4);
  IPEX_OP_REGISTER(
      "mlp_bias_silu_mul_bias_int4.xpu",
      at::AtenIpexTypeXPU::mlp_bias_silu_mul_bias_int4);
  IPEX_OP_REGISTER("mm_int4.xpu", at::AtenIpexTypeXPU::mm_int4);
  IPEX_OP_REGISTER("mm_int4_out.xpu", at::AtenIpexTypeXPU::mm_int4_out);
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
} // namespace
#else
#include <ATen/ATen.h>
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor _weight_int4pack_mm(
    const Tensor& A,
    const Tensor& B,
    int64_t qGroupSize,
    const Tensor& qScaleAndZeros) {
  TORCH_CHECK(false, "_weight_int4pack_mm is not supported without XeTLA");
}

} // namespace AtenIpexTypeXPU
} // namespace at
#endif
