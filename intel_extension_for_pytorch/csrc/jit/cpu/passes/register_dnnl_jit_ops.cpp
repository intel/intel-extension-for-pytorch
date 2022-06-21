#include <c10/util/Exception.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

#include "csrc/aten/cpu/AddLayerNorm.h"
#include "csrc/aten/cpu/ConcatBnRelu.h"
#include "csrc/jit/cpu/kernels/ConvPacked.h"
#include "csrc/jit/cpu/kernels/ConvTransposePacked.h"
#include "csrc/jit/cpu/kernels/Einsum.h"
#include "csrc/jit/cpu/kernels/Embeddingbag.h"
#include "csrc/jit/cpu/kernels/Interaction.h"
#include "csrc/jit/cpu/kernels/LinearPacked.h"
#include "csrc/jit/cpu/kernels/LinearSwishCustomized.h"
#include "csrc/jit/cpu/kernels/Matmul.h"
#include "csrc/jit/cpu/kernels/MaxPool2D.h"
#include "csrc/jit/cpu/kernels/Mha.h"
#include "csrc/jit/cpu/kernels/OpContext.h"
#include "csrc/jit/cpu/kernels/RNN.h"
#include "csrc/jit/cpu/kernels/Shuffle.h"
#include "csrc/jit/cpu/kernels/Softmax.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

at::Tensor toOptionalTensor(const IValue& v) {
  return v.isNone() ? at::Tensor() : v.toTensor();
}

using namespace torch_ipex::cpu;
using namespace torch_ipex::cpu::detail::convolution;
using namespace torch_ipex::cpu::detail::linear;
using namespace torch_ipex::cpu::detail::conv_transpose;

#define CONV_PREPACK_ARGS                         \
  "Tensor W, Tensor? B, "                         \
  "int[] stride, int[] padding, int[] dilation, " \
  "int groups, bool input_is_channels_last, "     \
  "int[] input_sizes"

#define CreateConvUnaryPostOpPrepack(FUSED_OP)                             \
  Operator(                                                                \
      "ipex_prepack::convolution_" #FUSED_OP "_prepack(" CONV_PREPACK_ARGS \
      ") "                                                                 \
      "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext",      \
      [](const Node* node) -> Operation {                                  \
        return [](Stack* stack) {                                          \
          auto result = IpexConvolutionOpContext::create_context(          \
              std::move((std::move(peek(stack, 0, 8))).toTensor()),        \
              std::move(toOptionalTensor(std::move(peek(stack, 1, 8)))),   \
              std::move((std::move(peek(stack, 2, 8))).toIntVector()),     \
              std::move((std::move(peek(stack, 3, 8))).toIntVector()),     \
              std::move((std::move(peek(stack, 4, 8))).toIntVector()),     \
              (std::move(peek(stack, 5, 8))).toInt(),                      \
              (std::move(peek(stack, 6, 8))).toBool(),                     \
              std::move((std::move(peek(stack, 7, 8))).toIntVector()),     \
              ideep::attr_t::fuse_##FUSED_OP());                           \
          drop(stack, 8);                                                  \
          pack(stack, std::move(result));                                  \
          return 0;                                                        \
        };                                                                 \
      },                                                                   \
      aliasAnalysisFromSchema())

#define CreateConvUnaryPostOpRun(FUSED_OP)                         \
  Operator(                                                        \
      "ipex_prepack::convolution_" #FUSED_OP                       \
      "(Tensor input, "                                            \
      "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext " \
      "W_prepack) -> Tensor",                                      \
      [](const Node* node) -> Operation {                          \
        return [](Stack* stack) {                                  \
          auto result = convolution_##FUSED_OP(                    \
              (std::move(peek(stack, 0, 2))).toTensor(),           \
              (std::move(peek(stack, 1, 2)))                       \
                  .toCustomClass<ConvolutionOpContext>());         \
          drop(stack, 2);                                          \
          pack(stack, std::move(result));                          \
          return 0;                                                \
        };                                                         \
      },                                                           \
      aliasAnalysisFromSchema())

#define CreateConvBinaryPostOpPrepack(FUSED_OP, ATTR)                         \
  Operator(                                                                   \
      "ipex_prepack::convolution_" #FUSED_OP "_prepack(" CONV_PREPACK_ARGS    \
      ", *, Scalar? alpha) "                                                  \
      "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext",         \
      [](const Node* node) -> Operation {                                     \
        return [](Stack* stack) {                                             \
          auto alpha1 =                                                       \
              (std::move(peek(stack, 8, 9))).toOptional<at::Scalar>();        \
          auto scale = alpha1.has_value() ? alpha1.value().to<float>() : 1.0; \
          auto result = IpexConvolutionOpContext::create_context(             \
              std::move((std::move(peek(stack, 0, 9))).toTensor()),           \
              std::move(toOptionalTensor(std::move(peek(stack, 1, 9)))),      \
              std::move((std::move(peek(stack, 2, 9))).toIntVector()),        \
              std::move((std::move(peek(stack, 3, 9))).toIntVector()),        \
              std::move((std::move(peek(stack, 4, 9))).toIntVector()),        \
              (std::move(peek(stack, 5, 9))).toInt(),                         \
              (std::move(peek(stack, 6, 9))).toBool(),                        \
              std::move((std::move(peek(stack, 7, 9))).toIntVector()),        \
              ideep::attr_t::ATTR(scale));                                    \
          drop(stack, 9);                                                     \
          pack(stack, std::move(result));                                     \
          return 0;                                                           \
        };                                                                    \
      },                                                                      \
      aliasAnalysisFromSchema())

#define CreateConvBinaryPostOpRun(FUSED_OP)                            \
  Operator(                                                            \
      "ipex_prepack::convolution_" #FUSED_OP                           \
      "(Tensor input, Tensor(a!) accumu, "                             \
      "*, Scalar? alpha, "                                             \
      "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "     \
      "W_prepack) -> Tensor",                                          \
      [](const Node* node) -> Operation {                              \
        return [](Stack* stack) {                                      \
          auto output = (std::move(peek(stack, 1, 4))).toTensor();     \
          auto result = convolution_##FUSED_OP(                        \
              (std::move(peek(stack, 0, 4))).toTensor(),               \
              output,                                                  \
              (std::move(peek(stack, 2, 4))).toOptional<at::Scalar>(), \
              (std::move(peek(stack, 3, 4)))                           \
                  .toCustomClass<ConvolutionOpContext>());             \
          drop(stack, 4);                                              \
          pack(stack, std::move(result));                              \
          return 0;                                                    \
        };                                                             \
      },                                                               \
      aliasAnalysisFromSchema())

#define CreateLinearUnaryPostOpRun(FUSED_OP)                  \
  Operator(                                                   \
      "ipex_prepack::linear_" #FUSED_OP                       \
      "(Tensor input, "                                       \
      "__torch__.torch.classes.ipex_prepack.LinearOpContext " \
      "W_prepack) -> Tensor",                                 \
      [](const Node* node) -> Operation {                     \
        return [](Stack* stack) {                             \
          auto result = linear_##FUSED_OP(                    \
              (std::move(peek(stack, 0, 2))).toTensor(),      \
              (std::move(peek(stack, 1, 2)))                  \
                  .toCustomClass<LinearOpContext>());         \
          drop(stack, 2);                                     \
          pack(stack, std::move(result));                     \
          return 0;                                           \
        };                                                    \
      },                                                      \
      aliasAnalysisFromSchema())

#define CreateConvTransposeUnaryPostOpRun(FUSED_OP)                  \
  Operator(                                                          \
      "ipex_prepack::conv_transpose_" #FUSED_OP                      \
      "(Tensor input, "                                              \
      "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext " \
      "W_prepack) -> Tensor",                                        \
      [](const Node* node) -> Operation {                            \
        return [](Stack* stack) {                                    \
          auto result = conv_transpose_##FUSED_OP(                   \
              (std::move(peek(stack, 0, 2))).toTensor(),             \
              (std::move(peek(stack, 1, 2)))                         \
                  .toCustomClass<ConvTransposeOpContext>());         \
          drop(stack, 2);                                            \
          pack(stack, std::move(result));                            \
          return 0;                                                  \
        };                                                           \
      },                                                             \
      aliasAnalysisFromSchema())

RegisterOperators op({
    CreateConvUnaryPostOpPrepack(relu),
    CreateConvUnaryPostOpPrepack(sigmoid),
    CreateConvUnaryPostOpPrepack(swish),
    CreateConvUnaryPostOpPrepack(tanh),
    CreateConvUnaryPostOpPrepack(mish),
    CreateConvUnaryPostOpPrepack(abs),
    CreateConvUnaryPostOpPrepack(exp),
    CreateConvUnaryPostOpPrepack(hardswish),
    CreateConvUnaryPostOpPrepack(square),
    CreateConvUnaryPostOpPrepack(log),
    CreateConvUnaryPostOpPrepack(round),
    CreateConvUnaryPostOpPrepack(sqrt),
    CreateConvUnaryPostOpPrepack(hardsigmoid),

    CreateConvUnaryPostOpRun(run),
    CreateConvUnaryPostOpRun(relu_run),
    CreateConvUnaryPostOpRun(sigmoid_run),
    CreateConvUnaryPostOpRun(swish_run),
    CreateConvUnaryPostOpRun(tanh_run),
    CreateConvUnaryPostOpRun(mish_run),
    CreateConvUnaryPostOpRun(abs_run),
    CreateConvUnaryPostOpRun(exp_run),
    CreateConvUnaryPostOpRun(hardswish_run),
    CreateConvUnaryPostOpRun(square_run),
    CreateConvUnaryPostOpRun(log_run),
    CreateConvUnaryPostOpRun(round_run),
    CreateConvUnaryPostOpRun(sqrt_run),
    CreateConvUnaryPostOpRun(hardsigmoid_run),

    CreateConvBinaryPostOpPrepack(add, fuse_sum),
    CreateConvBinaryPostOpPrepack(add_relu, residual),
    CreateConvBinaryPostOpRun(add_run),
    CreateConvBinaryPostOpRun(add_relu_run),

    Operator(
        "ipex_prepack::convolution_hardtanh_prepack(" CONV_PREPACK_ARGS
        ", Scalar lower_bound, Scalar upper_bound) "
        "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto lower_bound_value =
                (std::move(peek(stack, 8, 10))).toScalar().to<float>();
            auto upper_bound_value =
                (std::move(peek(stack, 9, 10))).toScalar().to<float>();
            auto result = IpexConvolutionOpContext::create_context(
                std::move((std::move(peek(stack, 0, 10))).toTensor()),
                std::move(toOptionalTensor(std::move(peek(stack, 1, 10)))),
                std::move((std::move(peek(stack, 2, 10))).toIntVector()),
                std::move((std::move(peek(stack, 3, 10))).toIntVector()),
                std::move((std::move(peek(stack, 4, 10))).toIntVector()),
                (std::move(peek(stack, 5, 10))).toInt(),
                (std::move(peek(stack, 6, 10))).toBool(),
                std::move((std::move(peek(stack, 7, 10))).toIntVector()),
                ideep::attr_t::fuse_clamp(
                    lower_bound_value, upper_bound_value));
            drop(stack, 10);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_elu_prepack(" CONV_PREPACK_ARGS
        ", Scalar alpha, Scalar scale, Scalar input_scale) "
        "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto alpha_value =
                (std::move(peek(stack, 8, 11))).toScalar().to<float>();
            auto scale_value =
                (std::move(peek(stack, 9, 11))).toScalar().to<float>();
            auto input_scale_value =
                (std::move(peek(stack, 10, 11))).toScalar().to<float>();
            auto result = IpexConvolutionOpContext::create_context(
                std::move((std::move(peek(stack, 0, 11))).toTensor()),
                std::move(toOptionalTensor(std::move(peek(stack, 1, 11)))),
                std::move((std::move(peek(stack, 2, 11))).toIntVector()),
                std::move((std::move(peek(stack, 3, 11))).toIntVector()),
                std::move((std::move(peek(stack, 4, 11))).toIntVector()),
                (std::move(peek(stack, 5, 11))).toInt(),
                (std::move(peek(stack, 6, 11))).toBool(),
                std::move((std::move(peek(stack, 7, 11))).toIntVector()),
                ideep::attr_t::fuse_elu(
                    scale_value, alpha_value, input_scale_value));
            drop(stack, 11);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_leaky_relu_prepack(" CONV_PREPACK_ARGS
        ", Scalar alpha) "
        "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto alpha_value =
                (std::move(peek(stack, 8, 9))).toScalar().to<float>();
            auto result = IpexConvolutionOpContext::create_context(
                std::move((std::move(peek(stack, 0, 9))).toTensor()),
                std::move(toOptionalTensor(std::move(peek(stack, 1, 9)))),
                std::move((std::move(peek(stack, 2, 9))).toIntVector()),
                std::move((std::move(peek(stack, 3, 9))).toIntVector()),
                std::move((std::move(peek(stack, 4, 9))).toIntVector()),
                (std::move(peek(stack, 5, 9))).toInt(),
                (std::move(peek(stack, 6, 9))).toBool(),
                std::move((std::move(peek(stack, 7, 9))).toIntVector()),
                ideep::attr_t::fuse_relu(1.0, alpha_value));
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_pow_prepack(" CONV_PREPACK_ARGS
        ", Scalar exponent) "
        "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto exponent_value =
                (std::move(peek(stack, 8, 9))).toScalar().to<float>();
            auto result = IpexConvolutionOpContext::create_context(
                std::move((std::move(peek(stack, 0, 9))).toTensor()),
                std::move(toOptionalTensor(std::move(peek(stack, 1, 9)))),
                std::move((std::move(peek(stack, 2, 9))).toIntVector()),
                std::move((std::move(peek(stack, 3, 9))).toIntVector()),
                std::move((std::move(peek(stack, 4, 9))).toIntVector()),
                (std::move(peek(stack, 5, 9))).toInt(),
                (std::move(peek(stack, 6, 9))).toBool(),
                std::move((std::move(peek(stack, 7, 9))).toIntVector()),
                ideep::attr_t::fuse_pow(1.0, exponent_value));
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_gelu_prepack(" CONV_PREPACK_ARGS
        ", str approximate) "
        "-> __torch__.torch.classes.ipex_prepack.ConvolutionOpContext",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto approximate = (std::move(peek(stack, 8, 9))).toStringView();
            dnnl::algorithm gelu_type;
            if (approximate == "none") {
              gelu_type = dnnl::algorithm::eltwise_gelu_erf;
            } else {
              gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
            }
            auto result = IpexConvolutionOpContext::create_context(
                std::move((std::move(peek(stack, 0, 9))).toTensor()),
                std::move(toOptionalTensor(std::move(peek(stack, 1, 9)))),
                std::move((std::move(peek(stack, 2, 9))).toIntVector()),
                std::move((std::move(peek(stack, 3, 9))).toIntVector()),
                std::move((std::move(peek(stack, 4, 9))).toIntVector()),
                (std::move(peek(stack, 5, 9))).toInt(),
                (std::move(peek(stack, 6, 9))).toBool(),
                std::move((std::move(peek(stack, 7, 9))).toIntVector()),
                ideep::attr_t::fuse_gelu(1.f, 0.f, 0.f, gelu_type));
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_hardtanh_run(Tensor input, Scalar "
        "lower_bound, Scalar upper_bound, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_hardtanh_run(
                (std::move(peek(stack, 0, 4))).toTensor(),
                (std::move(peek(stack, 1, 4))).toScalar(),
                (std::move(peek(stack, 2, 4))).toScalar(),
                (std::move(peek(stack, 3, 4)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_elu_run(Tensor input, Scalar alpha, "
        "Scalar scale, Scalar input_scale, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_elu_run(
                (std::move(peek(stack, 0, 5))).toTensor(),
                (std::move(peek(stack, 1, 5))).toScalar(),
                (std::move(peek(stack, 2, 5))).toScalar(),
                (std::move(peek(stack, 3, 5))).toScalar(),
                (std::move(peek(stack, 4, 5)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 5);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_leaky_relu_run(Tensor input, Scalar alpha, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_leaky_relu_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toScalar(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_pow_run(Tensor input, Scalar exponent, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_pow_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toScalar(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex_prepack::convolution_bottleneck_run(Tensor(a!) input, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack1, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack2, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack3"
        ") -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto output = (std::move(peek(stack, 0, 4))).toTensor();
            auto result = convolution_bottleneck_run(
                output,
                (std::move(peek(stack, 1, 4)))
                    .toCustomClass<ConvolutionOpContext>(),
                (std::move(peek(stack, 2, 4)))
                    .toCustomClass<ConvolutionOpContext>(),
                (std::move(peek(stack, 3, 4)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_bottleneck_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack1, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack2, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack3, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack4"
        ") -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_bottleneck_run(
                (std::move(peek(stack, 0, 5))).toTensor(),
                (std::move(peek(stack, 1, 5)))
                    .toCustomClass<ConvolutionOpContext>(),
                (std::move(peek(stack, 2, 5)))
                    .toCustomClass<ConvolutionOpContext>(),
                (std::move(peek(stack, 3, 5)))
                    .toCustomClass<ConvolutionOpContext>(),
                (std::move(peek(stack, 4, 5)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 5);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_gelu_run(Tensor input, str approximate, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_gelu_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toStringView(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    CreateLinearUnaryPostOpRun(run),
    CreateLinearUnaryPostOpRun(relu_run),
    CreateLinearUnaryPostOpRun(sigmoid_run),
    CreateLinearUnaryPostOpRun(swish_run),
    CreateLinearUnaryPostOpRun(tanh_run),
    CreateLinearUnaryPostOpRun(mish_run),
    CreateLinearUnaryPostOpRun(abs_run),
    CreateLinearUnaryPostOpRun(exp_run),
    CreateLinearUnaryPostOpRun(hardswish_run),
    CreateLinearUnaryPostOpRun(square_run),
    CreateLinearUnaryPostOpRun(log_run),
    CreateLinearUnaryPostOpRun(round_run),
    CreateLinearUnaryPostOpRun(sqrt_run),
    CreateLinearUnaryPostOpRun(hardsigmoid_run),

    Operator(
        "ipex_prepack::linear_leaky_relu_run(Tensor input, Scalar alpha, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_leaky_relu_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toScalar(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_hardtanh_run(Tensor input, Scalar "
        "lower_bound, Scalar upper_bound, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_hardtanh_run(
                (std::move(peek(stack, 0, 4))).toTensor(),
                (std::move(peek(stack, 1, 4))).toScalar(),
                (std::move(peek(stack, 2, 4))).toScalar(),
                (std::move(peek(stack, 3, 4)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_elu_run(Tensor input, Scalar alpha, "
        "Scalar scale, Scalar input_scale, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_elu_run(
                (std::move(peek(stack, 0, 5))).toTensor(),
                (std::move(peek(stack, 1, 5))).toScalar(),
                (std::move(peek(stack, 2, 5))).toScalar(),
                (std::move(peek(stack, 3, 5))).toScalar(),
                (std::move(peek(stack, 4, 5)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 5);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_pow_run(Tensor input, Scalar exponent, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_pow_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toScalar(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_gelu_run(Tensor input, str approximate, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_gelu_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toStringView(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_add_run(Tensor input, Tensor(a!) accumu, *, "
        "Scalar? alpha, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext W_prepack) "
        "-> Tensor(a!)",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto output = (std::move(peek(stack, 1, 4))).toTensor();
            auto result = linear_add_run(
                (std::move(peek(stack, 0, 4))).toTensor(),
                output,
                (std::move(peek(stack, 2, 4))).toOptional<at::Scalar>(),
                (std::move(peek(stack, 3, 4)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    // ConvTranspose fusion run OP
    CreateConvTransposeUnaryPostOpRun(run),
    CreateConvTransposeUnaryPostOpRun(relu_run),
    CreateConvTransposeUnaryPostOpRun(sigmoid_run),
    CreateConvTransposeUnaryPostOpRun(swish_run),
    CreateConvTransposeUnaryPostOpRun(tanh_run),
    CreateConvTransposeUnaryPostOpRun(mish_run),
    CreateConvTransposeUnaryPostOpRun(abs_run),
    CreateConvTransposeUnaryPostOpRun(exp_run),
    CreateConvTransposeUnaryPostOpRun(hardswish_run),
    CreateConvTransposeUnaryPostOpRun(square_run),
    CreateConvTransposeUnaryPostOpRun(log_run),
    CreateConvTransposeUnaryPostOpRun(round_run),
    CreateConvTransposeUnaryPostOpRun(sqrt_run),
    CreateConvTransposeUnaryPostOpRun(hardsigmoid_run),

    Operator(
        "ipex_prepack::conv_transpose_gelu_run(Tensor input, str approximate, "
        "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = conv_transpose_gelu_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toStringView(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<ConvTransposeOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::conv_transpose_leaky_relu_run(Tensor input, Scalar alpha, "
        "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = conv_transpose_leaky_relu_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toScalar(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<ConvTransposeOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::conv_transpose_hardtanh_run(Tensor input, Scalar "
        "lower_bound, Scalar upper_bound, "
        "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = conv_transpose_hardtanh_run(
                (std::move(peek(stack, 0, 4))).toTensor(),
                (std::move(peek(stack, 1, 4))).toScalar(),
                (std::move(peek(stack, 2, 4))).toScalar(),
                (std::move(peek(stack, 3, 4)))
                    .toCustomClass<ConvTransposeOpContext>());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::conv_transpose_elu_run(Tensor input, Scalar alpha, "
        "Scalar scale, Scalar input_scale, "
        "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = conv_transpose_elu_run(
                (std::move(peek(stack, 0, 5))).toTensor(),
                (std::move(peek(stack, 1, 5))).toScalar(),
                (std::move(peek(stack, 2, 5))).toScalar(),
                (std::move(peek(stack, 3, 5))).toScalar(),
                (std::move(peek(stack, 4, 5)))
                    .toCustomClass<ConvTransposeOpContext>());
            drop(stack, 5);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::conv_transpose_pow_run(Tensor input, Scalar exponent, "
        "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = conv_transpose_pow_run(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toScalar(),
                (std::move(peek(stack, 2, 3)))
                    .toCustomClass<ConvTransposeOpContext>());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::max_pool2d(Tensor input, int[2] kernel_size, int[2] stride, "
        "int[2] padding, int[2] dilation, bool ceil_mode) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_max_pool2d(
                (std::move(peek(stack, 0, 6))).toTensor(),
                (std::move(peek(stack, 1, 6))).toIntVector(),
                (std::move(peek(stack, 2, 6))).toIntVector(),
                (std::move(peek(stack, 3, 6))).toIntVector(),
                (std::move(peek(stack, 4, 6))).toIntVector(),
                (std::move(peek(stack, 5, 6))).toBool());
            drop(stack, 6);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex::matmul_div(Tensor left, Tensor right, Tensor(a!) out_opt, Tensor "
        "div_input) -> Tensor(a!)",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_matmul_div(
                (std::move(peek(stack, 0, 4))).toTensor(),
                (std::move(peek(stack, 1, 4))).toTensor(),
                toOptionalTensor(std::move(peek(stack, 2, 4))),
                (std::move(peek(stack, 3, 4))).toTensor());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::matmul_div(Tensor left, Tensor right, Tensor(a!) out_opt, Scalar "
        "div_input) -> Tensor(a!)",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_matmul_div(
                (std::move(peek(stack, 0, 4))).toTensor(),
                (std::move(peek(stack, 1, 4))).toTensor(),
                toOptionalTensor(std::move(peek(stack, 2, 4))),
                (std::move(peek(stack, 3, 4))).toScalar());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::matmul_div(Tensor left, Tensor right,  Tensor div_input) -> "
        "Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_matmul_div(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toTensor(),
                at::Tensor(),
                (std::move(peek(stack, 2, 3))).toTensor());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::matmul_div(Tensor left, Tensor right,  Scalar div_input) -> "
        "Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_matmul_div(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toTensor(),
                at::Tensor(),
                (std::move(peek(stack, 2, 3))).toScalar());
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::bmm_add(Tensor input, Tensor batch1, Tensor batch2, Scalar alpha) -> "
        "Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_bmm_add(
                (std::move(peek(stack, 0, 4))).toTensor(),
                (std::move(peek(stack, 1, 4))).toTensor(),
                (std::move(peek(stack, 2, 4))).toTensor(),
                (std::move(peek(stack, 3, 4))).toScalar());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::mha_scores_calc(Tensor q, Tensor k, Tensor rel_qk, Scalar "
        "alpha, "
        "Scalar dim_per_head, int softmax_dim, ScalarType ? dtype) -> Tensor",
        [](Stack& stack) {
          auto result = dil_mha_scores_calc(
              peek(stack, 0, 7).toTensor(),
              peek(stack, 1, 7).toTensor(),
              peek(stack, 2, 7).toTensor(),
              peek(stack, 3, 7).toScalar(),
              peek(stack, 4, 7).toScalar(),
              peek(stack, 5, 7).toInt(),
              peek(stack, 6, 7));
          drop(stack, 7);
          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::distil_mha_scores_calc(Tensor q, Tensor k, Tensor mask_qk, "
        "int[] mask_qk_reshp, int transpose_dim_a, int transpose_dim_b, "
        "Scalar fill, Scalar dim_per_head) "
        "-> Tensor",
        [](Stack& stack) {
          auto result = dil_distil_mha_scores_calc(
              peek(stack, 0, 8).toTensor(),
              peek(stack, 1, 8).toTensor(),
              peek(stack, 2, 8).toTensor(),
              peek(stack, 3, 8).toIntVector(),
              peek(stack, 4, 8).toInt(),
              peek(stack, 5, 8).toInt(),
              peek(stack, 6, 8).toScalar(),
              peek(stack, 7, 8).toScalar());
          drop(stack, 8);

          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::maskedfill_softmax(Tensor qk, Tensor mask_qk, "
        "int[] mask_qk_reshp, "
        "Scalar fill) "
        "-> Tensor",
        [](Stack& stack) {
          auto result = dil_maskedfill_softmax(
              peek(stack, 0, 4).toTensor(),
              peek(stack, 1, 4).toTensor(),
              peek(stack, 2, 4).toIntVector(),
              peek(stack, 3, 4).toScalar());
          drop(stack, 4);

          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::linear_swish_customized(Tensor x, Tensor weight, Tensor ? bias) -> Tensor",
        [](Stack& stack) {
          auto result = dil_linear_swish_customized(
              peek(stack, 0, 3).toTensor(),
              peek(stack, 1, 3).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 3))));
          drop(stack, 3);
          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::linear_swish_customized(Tensor x, Tensor weight, Tensor ? bias) -> Tensor",
        [](Stack& stack) {
          auto result = dil_linear_swish_customized(
              peek(stack, 0, 3).toTensor(),
              peek(stack, 1, 3).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 3))));
          drop(stack, 3);

          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::softmax(Tensor self, int dim, ScalarType ? dtype) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_softmax(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toInt(),
                (std::move(peek(stack, 2, 3))));
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::softmax_(Tensor(a!) self, int dim, ScalarType ? dtype) -> Tensor(a!)",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            // here the return value (Tensor) is alias of the input "self"
            auto output = (peek(stack, 0, 3).toTensor());
            auto result = dil_softmax_(
                output, peek(stack, 1, 3).toInt(), peek(stack, 2, 3));
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = at::batch_norm(
                (std::move(peek(stack, 0, 9))).toTensor(),
                toOptionalTensor(std::move(peek(stack, 1, 9))),
                toOptionalTensor(std::move(peek(stack, 2, 9))),
                toOptionalTensor(std::move(peek(stack, 3, 9))),
                toOptionalTensor(std::move(peek(stack, 4, 9))),
                (std::move(peek(stack, 5, 9))).toBool(),
                (std::move(peek(stack, 6, 9))).toDouble(),
                (std::move(peek(stack, 7, 9))).toDouble(),
                (std::move(peek(stack, 8, 9))).toBool());
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::qembedding_bag(Tensor weight, Tensor indices, Tensor offsets, "
        "bool sparse, bool include_last_offset, "
        "float o_scale, int o_zp, ScalarType o_dtype) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_qembeddingbag(
                (std::move(peek(stack, 0, 8))).toTensor(),
                (std::move(peek(stack, 1, 8))).toTensor(),
                (std::move(peek(stack, 2, 8))).toTensor(),
                (std::move(peek(stack, 3, 8))).toBool(),
                (std::move(peek(stack, 4, 8))).toBool(),
                (std::move(peek(stack, 5, 8))).toDouble(),
                (std::move(peek(stack, 6, 8))).toInt(),
                (std::move(peek(stack, 7, 8))).toScalarType());
            drop(stack, 8);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::qinteraction(Tensor[] tensors,  float o_scale, int o_zp, "
        "ScalarType o_dtype) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_qinteraction(
                (std::move(peek(stack, 0, 4))).toTensorVector(),
                (std::move(peek(stack, 1, 4))).toDouble(),
                (std::move(peek(stack, 2, 4))).toInt(),
                (std::move(peek(stack, 3, 4))).toScalarType());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::quantized_lstm(Tensor quantized_input, Tensor[] hx, Tensor [] quantized_weights, bool has_biases, int num_layers, float dropout_p, bool train, bool bidirectional, bool batch_first, float scale, int zp, int dtype) -> (Tensor, Tensor, Tensor)",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = quantized_lstm(
                (std::move(peek(stack, 0, 12))).toTensor(),
                (std::move(peek(stack, 1, 12))).toTensorList(),
                (std::move(peek(stack, 2, 12))).toTensorList(),
                (std::move(peek(stack, 3, 12))).toBool(),
                (std::move(peek(stack, 4, 12))).toInt(),
                (std::move(peek(stack, 5, 12))).toDouble(),
                (std::move(peek(stack, 6, 12))).toBool(),
                (std::move(peek(stack, 7, 12))).toBool(),
                (std::move(peek(stack, 8, 12))).toBool(),
                (std::move(peek(stack, 9, 12))).toDouble(),
                (std::move(peek(stack, 10, 12))).toInt(),
                (std::move(peek(stack, 11, 12))).toInt());
            drop(stack, 12);

            pack(stack, std::move(std::get<0>(result)));
            pack(stack, std::move(std::get<1>(result)));
            pack(stack, std::move(std::get<2>(result)));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

    Operator(
        "ipex::shuffle_2d("
        "  Tensor input,"
        "  int[5] view_shape,"
        "  int trans_dim0,"
        "  int trans_dim1) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_shuffle(
                (std::move(peek(stack, 0, 4))).toTensor(),
                (std::move(peek(stack, 1, 4))).toIntVector(),
                (std::move(peek(stack, 2, 4))).toInt(),
                (std::move(peek(stack, 3, 4))).toInt());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex::add_layernorm(Tensor a, Tensor b, int alpha, int[] "
        "normalized_shape, Tensor ? "
        "weight_opt, Tensor ? bias_opt, float eps, bool cuda_enable) -> "
        "Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = dil_add_layernorm(
                (std::move(peek(stack, 0, 8))).toTensor(),
                (std::move(peek(stack, 1, 8))).toTensor(),
                (std::move(peek(stack, 2, 8))).toInt(),
                (std::move(peek(stack, 3, 8))).toIntVector(),
                toOptionalTensor(std::move(peek(stack, 4, 8))),
                toOptionalTensor(std::move(peek(stack, 5, 8))),
                (std::move(peek(stack, 6, 8))).toDouble(),
                (std::move(peek(stack, 7, 8))).toBool());
            drop(stack, 8);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex::concat_bn_relu(Tensor[] a, Tensor bn_scale, Tensor bn_beta, "
        "Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled, int dim) -> "
        "Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = ConcatBnRelu(
                (std::move(peek(stack, 0, 12))).toTensorList(),
                (std::move(peek(stack, 1, 12))).toTensor(),
                (std::move(peek(stack, 2, 12))).toTensor(),
                toOptionalTensor(std::move(peek(stack, 3, 12))),
                toOptionalTensor(std::move(peek(stack, 4, 12))),
                toOptionalTensor(std::move(peek(stack, 5, 12))),
                toOptionalTensor(std::move(peek(stack, 6, 12))),
                (std::move(peek(stack, 7, 12))).toBool(),
                (std::move(peek(stack, 8, 12))).toDouble(),
                (std::move(peek(stack, 9, 12))).toDouble(),
                (std::move(peek(stack, 10, 12))).toBool(),
                (std::move(peek(stack, 11, 12))).toInt());
            drop(stack, 12);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex::einsum_binary(str equation, Tensor[] tensors, Tensor add_arg, Scalar alpha) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto other_arg = std::move(peek(stack, 2, 4));
            if (other_arg.isDouble()) {
              auto other_arg_st = at::ones(1).fill_(other_arg.toDouble());
              auto result = einsum_binary(
                  (std::move(peek(stack, 0, 4))).toStringView(),
                  (std::move(peek(stack, 1, 4))).toTensorList(),
                  other_arg_st,
                  (std::move(peek(stack, 3, 4))).toScalar());
              drop(stack, 4);
              pack(stack, std::move(result));
            } else {
              auto result = einsum_binary(
                  (std::move(peek(stack, 0, 4))).toStringView(),
                  (std::move(peek(stack, 1, 4))).toTensorList(),
                  other_arg.toTensor(),
                  (std::move(peek(stack, 3, 4))).toScalar());
              drop(stack, 4);
              pack(stack, std::move(result));
            }
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex::einsum_binary(str equation, Tensor[] tensors, Scalar add_arg, Scalar alpha) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto other_arg = std::move(peek(stack, 2, 4));
            auto other_arg_st = at::ones(1).fill_(other_arg.toScalar());
            auto result = einsum_binary(
                (std::move(peek(stack, 0, 4))).toStringView(),
                (std::move(peek(stack, 1, 4))).toTensorList(),
                other_arg_st,
                (std::move(peek(stack, 3, 4))).toScalar());

            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex::einsum_binary(str equation, Tensor[] tensors, double add_arg, Scalar alpha) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto other_ard =
                at::ones(1).fill_((std::move(peek(stack, 2, 4))).toDouble());
            auto result = einsum_binary(
                (std::move(peek(stack, 0, 4))).toStringView(),
                (std::move(peek(stack, 1, 4))).toTensorList(),
                other_ard,
                (std::move(peek(stack, 3, 4))).toScalar());

            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

});
} // namespace jit
} // namespace torch
