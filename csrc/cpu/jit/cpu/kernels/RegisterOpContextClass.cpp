#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>

#include "ConvPacked.h"
#include "ConvTransposePacked.h"
#include "LinearMKLPacked.h"
#include "LinearPacked.h"
#include "LinearWoqPacked.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
using detail::conv_transpose::createConvTransposePrePackOpContext;
using detail::convolution::createConvolutionPrePackOpContext;
using detail::linear::createLinearPrePackOpContext;
using detail::mkl_sgemm::createLinearMKLPrePackOpContext;
using detail::woq_linear::createWoqLinearPrePackOpContext;

TORCH_LIBRARY(ipex_prepack, m) {
  m.class_<ConvolutionOpContext>("ConvolutionOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<ConvolutionOpContext>& op_context)
              -> SerializationTypeConvolutionPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeConvolutionPrePack state)
              -> c10::intrusive_ptr<ConvolutionOpContext> { // __setstate__
            return createConvolutionPrePackOpContext(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          })
      .def(
          "get_weight",
          &torch_ipex::cpu::ConvolutionOpContext::get_at_packed_weight)
      .def("get_bias", &torch_ipex::cpu::ConvolutionOpContext::get_at_bias)
      .def("pack", &torch_ipex::cpu::ConvolutionOpContext::pack)
      .def("to_public", &torch_ipex::cpu::ConvolutionOpContext::to_public)
      .def(
          "get_data_handle",
          &torch_ipex::cpu::ConvolutionOpContext::get_data_handle)
      .def(
          "load_from_ctx",
          &torch_ipex::cpu::ConvolutionOpContext::load_from_ctx);
  m.class_<LinearOpContext>("LinearOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<LinearOpContext>& op_context)
              -> SerializationTypeLinearPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeLinearPrePack state)
              -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
            return createLinearPrePackOpContext(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)));
          })
      .def(
          "get_weight", &torch_ipex::cpu::LinearOpContext::get_at_packed_weight)
      .def("get_bias", &torch_ipex::cpu::LinearOpContext::get_at_bias)
      .def("pack", &torch_ipex::cpu::LinearOpContext::pack)
      .def("to_public", &torch_ipex::cpu::LinearOpContext::to_public)
      .def(
          "get_data_handle", &torch_ipex::cpu::LinearOpContext::get_data_handle)
      .def("load_from_ctx", &torch_ipex::cpu::LinearOpContext::load_from_ctx);
  m.class_<MKLOpContext>("MKLOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<MKLOpContext>& op_context)
              -> SerializationTypeMKLPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeMKLPrePack state)
              -> c10::intrusive_ptr<MKLOpContext> { // __setstate__
            return createLinearMKLPrePackOpContext(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)));
          })
      .def("get_weight", &torch_ipex::cpu::MKLOpContext::get_at_packed_weight)
      .def("get_bias", &torch_ipex::cpu::MKLOpContext::get_at_bias)
      .def("pack", &torch_ipex::cpu::MKLOpContext::pack)
      .def("to_public", &torch_ipex::cpu::MKLOpContext::to_public)
      .def("get_data_handle", &torch_ipex::cpu::MKLOpContext::get_data_handle)
      .def("load_from_ctx", &torch_ipex::cpu::MKLOpContext::load_from_ctx);
  m.class_<ConvTransposeOpContext>("ConvTransposeOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<ConvTransposeOpContext>& op_context)
              -> SerializationTypeConvTransposePrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeConvTransposePrePack state)
              -> c10::intrusive_ptr<ConvTransposeOpContext> { // __setstate__
            return createConvTransposePrePackOpContext(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)),
                std::move(std::get<8>(state)));
          })
      .def(
          "get_weight",
          &torch_ipex::cpu::ConvTransposeOpContext::get_at_packed_weight)
      .def("get_bias", &torch_ipex::cpu::ConvTransposeOpContext::get_at_bias)
      .def("pack", &torch_ipex::cpu::ConvTransposeOpContext::pack)
      .def("to_public", &torch_ipex::cpu::ConvTransposeOpContext::to_public)
      .def(
          "get_data_handle",
          &torch_ipex::cpu::ConvTransposeOpContext::get_data_handle)
      .def(
          "load_from_ctx",
          &torch_ipex::cpu::ConvTransposeOpContext::load_from_ctx);
  m.class_<WoqLinearOpContext>("WoqLinearOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<WoqLinearOpContext>& op_context)
              -> SerializationTypeWoqLinearPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeWoqLinearPrePack state)
              -> c10::intrusive_ptr<WoqLinearOpContext> { // __setstate__
            return createWoqLinearPrePackOpContext(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)));
          })
      .def(
          "get_weight",
          &torch_ipex::cpu::WoqLinearOpContext::get_at_packed_weight)
      .def("pack", &torch_ipex::cpu::WoqLinearOpContext::pack)
      .def("to_public", &torch_ipex::cpu::WoqLinearOpContext::to_public)
      .def(
          "get_data_handle",
          &torch_ipex::cpu::WoqLinearOpContext::get_data_handle)
      .def(
          "load_from_ctx", &torch_ipex::cpu::WoqLinearOpContext::load_from_ctx);
  m.def(
      "convolution_prepack(Tensor W, Tensor? B, int[] stride, "
      "int[] padding, int[] dilation, int groups, "
      "bool input_is_channels_last, int[] input_sizes) "
      "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext");
  m.def(
      "linear_prepack(Tensor W, Tensor? B, int? batch_size) "
      "-> __torch__.torch.classes.ipex_prepack.LinearOpContext");
  m.def(
      "mkl_sgemm_prepack(Tensor W, Tensor? B, int? batch_size) "
      "-> __torch__.torch.classes.ipex_prepack.MKLOpContext");
  m.def(
      "conv_transpose_prepack(Tensor W, Tensor? B, int[] stride, "
      "int[] padding, int[] output_padding, int groups, int[] dilation, "
      "bool input_is_channels_last, int[] input_sizes) "
      "-> __torch__.torch.classes.ipex_prepack.ConvTransposeOpContext");
  m.def(
      "weight_only_qlinear_prepack(Tensor W, Tensor? B, int? batch_size, int lowp_mode, int num_concats) "
      "-> __torch__.torch.classes.ipex_prepack.WoqLinearOpContext");
}

TORCH_LIBRARY_IMPL(ipex_prepack, CPU, m) {
  m.impl("convolution_prepack", TORCH_FN(createConvolutionPrePackOpContext));
  m.impl("linear_prepack", TORCH_FN(createLinearPrePackOpContext));
  m.impl("mkl_sgemm_prepack", TORCH_FN(createLinearMKLPrePackOpContext));
  m.impl(
      "conv_transpose_prepack", TORCH_FN(createConvTransposePrePackOpContext));
}
TORCH_LIBRARY_IMPL(ipex_prepack, QuantizedCPU, m) {
  m.impl(
      "weight_only_qlinear_prepack", TORCH_FN(createWoqLinearPrePackOpContext));
}

} // namespace cpu
} // namespace torch_ipex
