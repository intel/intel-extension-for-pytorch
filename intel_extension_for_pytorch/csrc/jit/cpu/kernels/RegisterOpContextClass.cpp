#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>

#include "ConvPacked.h"
#include "ConvTransposePacked.h"
#include "LinearPacked.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
using detail::conv_transpose2d::createConvTransposePrePackOpContext;
using detail::convolution::createConvolutionPrePackOpContext;
using detail::linear::createLinearPrePackOpContext;

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
                std::move(std::get<7>(state)),
                std::move(std::get<8>(state)),
                std::move(std::get<9>(state)),
                std::move(std::get<10>(state)));
          });
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
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)));
          });
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
                std::move(std::get<8>(state)),
                std::move(std::get<9>(state)),
                std::move(std::get<10>(state)),
                std::move(std::get<11>(state)));
          });
  m.def(
      "convolution_prepack(Tensor W, Tensor? B, int[] stride, "
      "int[] padding, int[] dilation, int[] kernel_size, int groups, int "
      "output_channel, "
      "bool input_is_channels_last, bool weight_is_packed, int[] input_sizes) "
      "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext");
  m.def(
      "linear_prepack(Tensor W, Tensor? B, "
      "int out_features, int int_features, int batch_size, bool "
      "weight_is_prepacked) "
      "-> __torch__.torch.classes.ipex_prepack.LinearOpContext");
  m.def(
      "conv_transpose2d_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int groups, int[2] dilation, "
      "int[2] kernel_size,  int output_channel, "
      "bool input_is_channels_last, bool weight_is_prepacked, int[4] "
      "input_sizes) "
      "-> __torch__.torch.classes.ipex_prepack.ConvTransposeOpContext");
}

TORCH_LIBRARY_IMPL(ipex_prepack, AutogradCPU, m) {
  m.impl("convolution_prepack", TORCH_FN(createConvolutionPrePackOpContext));
  m.impl("linear_prepack", TORCH_FN(createLinearPrePackOpContext));
  m.impl(
      "conv_transpose2d_prepack",
      TORCH_FN(createConvTransposePrePackOpContext));
}

} // namespace cpu
} // namespace torch_ipex
