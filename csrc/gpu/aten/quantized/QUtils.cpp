#include <ATen/ATen.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/accumulate.h>
#include <torch/custom_class.h>
#include <torch/custom_class_detail.h>

#include <quantized/QUtils.h>

namespace at {
namespace AtenIpexTypeQuantizedXPU {

at::Tensor u8tos8(const at::Tensor& u8) {
  auto s8 = at::_empty_affine_quantized(
      u8.sizes(),
      ScalarType::QInt8,
      c10::nullopt,
      u8.device(),
      c10::nullopt,
      u8.q_scale(),
      0,
      u8.suggest_memory_format());

  auto& dpcpp_queue = xpu::dpcpp::dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    uint8_t* u8_ptr = (uint8_t*)u8.data_ptr();
    int8_t* s8_ptr = (int8_t*)s8.data_ptr();
    cgh.parallel_for(
        cl::sycl::range<1>(u8.numel()), [=](cl::sycl::item<1> item) {
          auto id = item.get_linear_id();
          auto s8_val = (float)Round(static_cast<float>(u8_ptr[id]) / 2.f);
          s8_val =
              Numerics<float>::min(Numerics<float>::max(s8_val, -128.f), 127.f);
          s8_ptr[id] = static_cast<int8_t>(s8_val);
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  return s8;
}

// Note: [Opaque u8 tensor]
// Due to the difference between oneDNN and PyTorch u8 quantization, we quant
// tensor with kQUint8 and 128 zp to memory::data_type::s8 and 0 zp inside. This
// utils is used for checking this kind of QTensor. More details can see
// Quantization.cpp quantizer_tensor_per_tenser_affine function.
bool is_opaque_u8(const Tensor& qx) {
  auto qx_ctx = DPCPPTensorContext::get_tensor_ctx(qx);
  if (!qx_ctx.is_plain()) {
    return (
        (qx.scalar_type() == kQUInt8) &&
        (qx_ctx.meta().data_type() == memory::data_type::s8));
  } else {
    return false;
  }
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at

#ifdef BUILD_JIT_QUANTIZATION_SAVE
// QConv prepack pickling method hacking
template <int kSpatialDim = 2>
torch::class_<ConvPackedParamsBase<kSpatialDim>> register_conv_params();

extern template torch::class_<ConvPackedParamsBase<2>> register_conv_params<
    2>();
extern template torch::class_<ConvPackedParamsBase<3>> register_conv_params<
    3>();

template <int kSpatialDim = 2>
ConvParamsSerializationTypeV2 serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params);
extern template ConvParamsSerializationTypeV2 serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& params);
extern template ConvParamsSerializationTypeV2 serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& params);

template <uint32_t kSpatialDim>
ConvParamsSerializationTypeV3 parse_conv_serialized_state(c10::IValue v);

template <int kSpatialDim>
int redefine_prepack() {
  auto conv_prepack_class = register_conv_params<kSpatialDim>();
  auto clsptr = torch::getCustomClass(
      "__torch__.torch.classes.quantized.Conv" + c10::to_string(kSpatialDim) +
      "dPackedParamsBase");
  clsptr->unsafeRemoveMethod("__getstate__");
  clsptr->unsafeRemoveMethod("__setstate__");
  conv_prepack_class.def_pickle(
      [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params)
          -> ConvParamsSerializationType { // __getstate__
        return serialize_conv<kSpatialDim>(params);
      },
      // __setstate__ takes c10::IValue because we support parsing historical
      // serialization versions.
      [](c10::IValue v) -> c10::intrusive_ptr<
                            ConvPackedParamsBase<kSpatialDim>> { // __setstate__
        ConvParamsSerializationTypeV3 state =
            parse_conv_serialized_state<kSpatialDim>(v);
        return deserialize_conv_dpcpp<kSpatialDim>(state);
      });
  return 0;
}

template int redefine_prepack<2>();
template int redefine_prepack<3>();

// QLinear prepack pickling method hacking
torch::class_<LinearPackedParamsBase> register_linear_params();

int redefine_linear_prepack() {
  auto linear_prepack_class = register_linear_params();
  auto clsptr = torch::getCustomClass(
      "__torch__.torch.classes.quantized.LinearPackedParamsBase");
  clsptr->unsafeRemoveMethod("__getstate__");
  clsptr->unsafeRemoveMethod("__setstate__");
  using SerializationType = std::tuple<at::Tensor, c10::optional<at::Tensor>>;
  linear_prepack_class.def_pickle(
      [](const c10::intrusive_ptr<LinearPackedParamsBase>& params)
          -> SerializationType { // __getstate__
        at::Tensor weight;
        c10::optional<at::Tensor> bias;
        std::tie(weight, bias) = params->unpack();
        return std::make_tuple(std::move(weight), std::move(bias));
      },
      [](SerializationType state)
          -> c10::intrusive_ptr<LinearPackedParamsBase> { // __setstate__
        at::Tensor weight;
        c10::optional<at::Tensor> bias;
        weight = std::move(std::get<0>(state));
        bias = std::move(std::get<1>(state));

        return at::AtenIpexTypeQuantizedXPU::PackedLinearWeightQDPCPP::prepack(
            std::move(weight), std::move(bias));
      });
  return 0;
}

namespace {
static auto conv2d_params = redefine_prepack<2>();
static auto conv3d_params = redefine_prepack<3>();
static auto linear_params = redefine_linear_prepack();
} // namespace
#endif
