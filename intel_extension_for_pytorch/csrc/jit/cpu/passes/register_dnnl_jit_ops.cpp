#include <c10/util/Exception.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

#include "csrc/jit/cpu/kernels/AddLayerNorm.h"
#include "csrc/jit/cpu/kernels/ConvPacked.h"
#include "csrc/jit/cpu/kernels/ConvTransposePacked.h"
#include "csrc/jit/cpu/kernels/Embeddingbag.h"
#include "csrc/jit/cpu/kernels/Interaction.h"
#include "csrc/jit/cpu/kernels/LinearPacked.h"
#include "csrc/jit/cpu/kernels/Matmul.h"
#include "csrc/jit/cpu/kernels/MaxPool2D.h"
#include "csrc/jit/cpu/kernels/Mha.h"
#include "csrc/jit/cpu/kernels/OpContext.h"
#include "csrc/jit/cpu/kernels/Shuffle.h"
#include "csrc/jit/cpu/kernels/Softmax.h"

#include "csrc/aten/cpu/Pooling.h"
#include "csrc/utils/utils.h"

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
using namespace torch_ipex::cpu::detail::conv_transpose2d;

RegisterOperators op({
    Operator(
        "ipex_prepack::convolution_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 2);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_relu_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_relu_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 2);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_sigmoid_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_sigmoid_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 2);
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
        "ipex_prepack::convolution_swish_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = convolution_swish_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 2);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_add_run(Tensor input, Tensor(a!) "
        "accumu, *, Scalar? alpha, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor(a!)",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto output = (std::move(peek(stack, 1, 4))).toTensor();
            auto result = convolution_add_run(
                (std::move(peek(stack, 0, 4))).toTensor(),
                output,
                (std::move(peek(stack, 2, 4))).toOptional<at::Scalar>(),
                (std::move(peek(stack, 3, 4)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::convolution_add_relu_run(Tensor input, Tensor(a!) "
        "accumu, *, Scalar? alpha, "
        "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext "
        "W_prepack) -> Tensor(a!)",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto output = (std::move(peek(stack, 1, 4))).toTensor();
            auto result = convolution_add_relu_run(
                (std::move(peek(stack, 0, 4))).toTensor(),
                output,
                (std::move(peek(stack, 2, 4))).toOptional<at::Scalar>(),
                (std::move(peek(stack, 3, 4)))
                    .toCustomClass<ConvolutionOpContext>());
            drop(stack, 4);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::conv_transpose2d_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext "
        "W_prepack) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = conv_transpose2d_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<ConvTransposeOpContext>());
            drop(stack, 2);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext W_prepack) "
        "-> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 2);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_relu_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext W_prepack) "
        "-> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_relu_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 2);
            pack(stack, std::move(result));
            return 0;
          };
        },
        aliasAnalysisFromSchema()),
    Operator(
        "ipex_prepack::linear_gelu_run(Tensor input, "
        "__torch__.torch.classes.ipex_prepack.LinearOpContext W_prepack) "
        "-> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack* stack) {
            auto result = linear_gelu_run(
                (std::move(peek(stack, 0, 2))).toTensor(),
                (std::move(peek(stack, 1, 2)))
                    .toCustomClass<LinearOpContext>());
            drop(stack, 2);
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
        "ipex::matmul_div(Tensor left, Tensor right, Tensor? out_opt, Tensor "
        "div_input) -> Tensor",
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
        "ipex::matmul_div(Tensor left, Tensor right, Tensor? out_opt, Scalar "
        "div_input) -> Tensor",
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

});
} // namespace jit
} // namespace torch
