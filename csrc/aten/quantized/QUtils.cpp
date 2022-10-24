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
