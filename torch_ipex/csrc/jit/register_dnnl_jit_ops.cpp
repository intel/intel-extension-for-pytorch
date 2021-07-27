#include <c10/util/Exception.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

#include "torch_ipex/csrc/cpu/CustomOPs.h"
#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/cpu/Pooling.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

at::Tensor toOptionalTensor(const IValue& v) {
  return v.isNone() ? at::Tensor() : v.toTensor();
}

using namespace torch_ipex::cpu;


// Convolution Fusion Ops

#define CONV_ARGS \
  "Tensor input, Tensor weight, Tensor? bias, " \
  "int[2] stride, int[2] padding, int[2] dilation, int groups"

#define CONV_ARGS_ND_WEIGHT                                                    \
  "Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, " \
  "int[2] dilation, int[2] kernel_size, "                                      \
  "int groups, int output_channel, bool weight_channels_last, bool "           \
  "weight_prepacked"

#define CreateConvEltwiseOperator(ND, FUSED_OP)                        \
  Operator(                                                            \
      "ipex::conv" #ND "_" #FUSED_OP "(" CONV_ARGS ") -> Tensor",      \
      [](const Node* node) -> Operation {                              \
          return [](Stack* stack) {                                    \
            auto result = AtenIpexJITDev::dil_convolution_##FUSED_OP(  \
                (std::move(peek(stack, 0, 7))).toTensor(),             \
                (std::move(peek(stack, 1, 7))).toTensor(),             \
                toOptionalTensor(std::move(peek(stack, 2, 7))),        \
                (std::move(peek(stack, 3, 7))).toIntVector(),          \
                (std::move(peek(stack, 4, 7))).toIntVector(),          \
                (std::move(peek(stack, 5, 7))).toIntVector(),          \
                (std::move(peek(stack, 6, 7))).toInt());               \
            drop(stack, 7);                                            \
            pack(stack, std::move(result));                            \
            return 0;                                                  \
          };                                                           \
      },                                                               \
      aliasAnalysisFromSchema())

#define _CreateConvSumEltwiseOperator(ND, ...)                                \
  Operator(                                                                   \
      "ipex::conv" #ND "_sum" #__VA_ARGS__ "(" CONV_ARGS ", Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)", \
      [](const Node* node) -> Operation {                                     \
          return [](Stack* stack) {                                           \
            auto output = (std::move(peek(stack, 7, 9))).toTensor();          \
            auto result = AtenIpexJITDev::dil_convolution_sum##__VA_ARGS__(   \
                (std::move(peek(stack, 0, 9))).toTensor(),                    \
                (std::move(peek(stack, 1, 9))).toTensor(),                    \
                toOptionalTensor(std::move(peek(stack, 2, 9))),               \
                (std::move(peek(stack, 3, 9))).toIntVector(),                 \
                (std::move(peek(stack, 4, 9))).toIntVector(),                 \
                (std::move(peek(stack, 5, 9))).toIntVector(),                 \
                (std::move(peek(stack, 6, 9))).toInt(),                       \
                output,                                                       \
                (std::move(peek(stack, 8, 9))).toScalar());                   \
            drop(stack, 9);                                                   \
            pack(stack, std::move(result));                                   \
            return 0;                                                         \
          };                                                                  \
      },                                                                      \
      aliasAnalysisFromSchema())

#define CreateConvEltwiseOperatorNDWeight(ND, FUSED_OP)                        \
  Operator(                                                                    \
      "ipex::conv" #ND "_" #FUSED_OP "(" CONV_ARGS_ND_WEIGHT ") -> Tensor",    \
      [](const Node *node) -> Operation {                                      \
        return [](Stack *stack) {                                              \
          auto result = AtenIpexJITDev::dil_convolution_nd_weight_##FUSED_OP(  \
              (std::move(peek(stack, 0, 11))).toTensor(),                      \
              (std::move(peek(stack, 1, 11))).toTensor(),                      \
              toOptionalTensor(std::move(peek(stack, 2, 11))),                 \
              (std::move(peek(stack, 3, 11))).toIntVector(),                   \
              (std::move(peek(stack, 4, 11))).toIntVector(),                   \
              (std::move(peek(stack, 5, 11))).toIntVector(),                   \
              (std::move(peek(stack, 6, 11))).toIntVector(),                   \
              (std::move(peek(stack, 7, 11))).toInt(),                         \
              (std::move(peek(stack, 8, 11))).toInt(),                         \
              (std::move(peek(stack, 9, 11))).toBool(),                        \
              (std::move(peek(stack, 10, 11))).toBool());                      \
          drop(stack, 11);                                                     \
          pack(stack, std::move(result));                                      \
          return 0;                                                            \
        };                                                                     \
      },                                                                       \
      aliasAnalysisFromSchema())

#define _CreateConvSumEltwiseOperatorNDWeight(ND, ...)                         \
  Operator(                                                                    \
      "ipex::conv" #ND "_sum" #__VA_ARGS__ "(" CONV_ARGS_ND_WEIGHT             \
      ", Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",                   \
      [](const Node *node) -> Operation {                                      \
        return [](Stack *stack) {                                              \
          auto output = (std::move(peek(stack, 11, 13))).toTensor();           \
          auto result =                                                        \
              AtenIpexJITDev::dil_convolution_nd_weight_sum##__VA_ARGS__(      \
                  (std::move(peek(stack, 0, 13))).toTensor(),                  \
                  (std::move(peek(stack, 1, 13))).toTensor(),                  \
                  toOptionalTensor(std::move(peek(stack, 2, 13))),             \
                  (std::move(peek(stack, 3, 13))).toIntVector(),               \
                  (std::move(peek(stack, 4, 13))).toIntVector(),               \
                  (std::move(peek(stack, 5, 13))).toIntVector(),               \
                  (std::move(peek(stack, 6, 13))).toIntVector(),               \
                  (std::move(peek(stack, 7, 13))).toInt(),                     \
                  (std::move(peek(stack, 8, 13))).toInt(),                     \
                  (std::move(peek(stack, 9, 13))).toBool(),                    \
                  (std::move(peek(stack, 10, 13))).toBool(), output,           \
                  (std::move(peek(stack, 12, 13))).toScalar());                \
          drop(stack, 13);                                                     \
          pack(stack, std::move(result));                                      \
          return 0;                                                            \
        };                                                                     \
      },                                                                       \
      aliasAnalysisFromSchema())

#define CreateConvSumOperator(ND) _CreateConvSumEltwiseOperator(ND)
#define CreateConvSumEltwiseOperator(ND, FUSED_OP) \
  _CreateConvSumEltwiseOperator(ND, _##FUSED_OP)

#define CreateConvSumOperatorNDWeight(ND)                                      \
  _CreateConvSumEltwiseOperatorNDWeight(ND)
#define CreateConvSumEltwiseOperatorNDWeight(ND, FUSED_OP)                     \
  _CreateConvSumEltwiseOperatorNDWeight(ND, _##FUSED_OP)

// Linear Fusion Ops

#define CreateLinearEltwiseOperator(FUSED_OP)                                 \
  Operator(                                                                   \
      "ipex::linear_" #FUSED_OP "(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", \
      [](const Node* node) -> Operation {                                     \
          return [](Stack* stack) {                                           \
            auto result = AtenIpexJITDev::dil_linear_fuse_eltwise(            \
                (std::move(peek(stack, 0, 3))).toTensor(),                    \
                (std::move(peek(stack, 1, 3))).toTensor(),                    \
                toOptionalTensor(std::move(peek(stack, 2, 3))),               \
                ideep::attr_t::fuse_##FUSED_OP());                            \
            drop(stack, 3);                                                   \
            pack(stack, std::move(result));                                   \
            return 0;                                                         \
          };                                                                  \
      },                                                                      \
      aliasAnalysisFromSchema())

RegisterOperators op(
    {CreateConvEltwiseOperator(2d, base), CreateConvEltwiseOperator(2d, relu),
     CreateConvEltwiseOperator(2d, sigmoid),
     CreateConvEltwiseOperator(2d, swish), CreateConvEltwiseOperator(3d, relu),
     CreateConvSumOperator(2d), CreateConvSumOperator(3d),
     CreateConvSumEltwiseOperator(2d, relu),
     CreateConvSumEltwiseOperator(3d, relu), CreateLinearEltwiseOperator(relu),
     CreateLinearEltwiseOperator(gelu),
     // n-d weight case
     CreateConvEltwiseOperatorNDWeight(2d, base),
     CreateConvEltwiseOperatorNDWeight(2d, relu),
     CreateConvEltwiseOperatorNDWeight(2d, sigmoid),
     CreateConvEltwiseOperatorNDWeight(2d, swish),
     CreateConvSumOperatorNDWeight(2d),
     CreateConvSumEltwiseOperatorNDWeight(2d, relu),
     // TODO: 3d n-d weight case.
     Operator(
         "ipex::conv2d_clamp(" CONV_ARGS
         ", float lower_bound=-1.0, float upper_bound=1.0) -> Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_convolution_clamp(
                 (std::move(peek(stack, 0, 9))).toTensor(),
                 (std::move(peek(stack, 1, 9))).toTensor(),
                 toOptionalTensor(std::move(peek(stack, 2, 9))),
                 (std::move(peek(stack, 3, 9))).toIntVector(),
                 (std::move(peek(stack, 4, 9))).toIntVector(),
                 (std::move(peek(stack, 5, 9))).toIntVector(),
                 (std::move(peek(stack, 6, 9))).toInt(),
                 (std::move(peek(stack, 7, 9))).toDouble(),
                 (std::move(peek(stack, 8, 9))).toDouble());
             drop(stack, 9);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema()),
     Operator(
         "ipex::conv2d_elu(" CONV_ARGS ", float alpha=1.0, Scalar scale=1.0, "
         "Scalar input_scale=1.0) -> Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_convolution_elu(
                 (std::move(peek(stack, 0, 10))).toTensor(),
                 (std::move(peek(stack, 1, 10))).toTensor(),
                 toOptionalTensor(std::move(peek(stack, 2, 10))),
                 (std::move(peek(stack, 3, 10))).toIntVector(),
                 (std::move(peek(stack, 4, 10))).toIntVector(),
                 (std::move(peek(stack, 5, 10))).toIntVector(),
                 (std::move(peek(stack, 6, 10))).toInt(),
                 (std::move(peek(stack, 7, 10))).toDouble(),
                 (std::move(peek(stack, 8, 10))).toScalar(),
                 (std::move(peek(stack, 9, 10))).toScalar());
             drop(stack, 10);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema()),

     Operator(
         "ipex::conv2d_clamp(" CONV_ARGS_ND_WEIGHT
         ", float lower_bound=-1.0, float upper_bound=1.0) -> Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_convolution_nd_weight_clamp(
                 (std::move(peek(stack, 0, 13))).toTensor(),
                 (std::move(peek(stack, 1, 13))).toTensor(),
                 toOptionalTensor(std::move(peek(stack, 2, 13))),
                 (std::move(peek(stack, 3, 13))).toIntVector(),
                 (std::move(peek(stack, 4, 13))).toIntVector(),
                 (std::move(peek(stack, 5, 13))).toIntVector(),
                 (std::move(peek(stack, 6, 13))).toIntVector(),
                 (std::move(peek(stack, 7, 13))).toInt(),
                 (std::move(peek(stack, 8, 13))).toInt(),
                 (std::move(peek(stack, 9, 13))).toBool(),
                 (std::move(peek(stack, 10, 13))).toBool(),
                 (std::move(peek(stack, 11, 13))).toDouble(),
                 (std::move(peek(stack, 12, 13))).toDouble());
             drop(stack, 13);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema()),

     Operator(
         "ipex::conv2d_elu(" CONV_ARGS_ND_WEIGHT
         ", float alpha=1.0, Scalar scale=1.0, Scalar input_scale=1.0) -> "
         "Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_convolution_nd_weight_elu(
                 (std::move(peek(stack, 0, 14))).toTensor(),
                 (std::move(peek(stack, 1, 14))).toTensor(),
                 toOptionalTensor(std::move(peek(stack, 2, 14))),
                 (std::move(peek(stack, 3, 14))).toIntVector(),
                 (std::move(peek(stack, 4, 14))).toIntVector(),
                 (std::move(peek(stack, 5, 14))).toIntVector(),
                 (std::move(peek(stack, 6, 14))).toIntVector(),
                 (std::move(peek(stack, 7, 14))).toInt(),
                 (std::move(peek(stack, 8, 14))).toInt(),
                 (std::move(peek(stack, 9, 14))).toBool(),
                 (std::move(peek(stack, 10, 14))).toBool(),
                 (std::move(peek(stack, 11, 14))).toDouble(),
                 (std::move(peek(stack, 12, 14))).toScalar(),
                 (std::move(peek(stack, 13, 14))).toScalar());
             drop(stack, 14);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema()),

     Operator(
         "ipex::max_pool2d(Tensor input, int[2] kernel_size, int[2] stride, "
         "int[2] padding, int[2] dilation, bool ceil_mode) -> Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_max_pool2d(
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
         "ipex::linear(Tensor input, Tensor weight, Tensor? bias=None) -> "
         "Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_linear(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(),
                 toOptionalTensor(std::move(peek(stack, 2, 3))));
             drop(stack, 3);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema()),
     Operator(
         "ipex::linear_add(Tensor input, Tensor weight, Tensor? bias, "
         "Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto output = (std::move(peek(stack, 3, 5))).toTensor();
             auto result = AtenIpexJITDev::dil_linear_add(
                 (std::move(peek(stack, 0, 5))).toTensor(),
                 (std::move(peek(stack, 1, 5))).toTensor(),
                 toOptionalTensor(std::move(peek(stack, 2, 5))), output,
                 (std::move(peek(stack, 4, 5))).toScalar());
             drop(stack, 5);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema()),

     Operator(
         "ipex::matmul_div(Tensor left, Tensor right, Tensor? out_opt, Tensor "
         "div_input) -> Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_matmul_div(
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
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_matmul_div(
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
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_matmul_div(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(), at::Tensor(),
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
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_matmul_div(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toTensor(), at::Tensor(),
                 (std::move(peek(stack, 2, 3))).toScalar());
             drop(stack, 3);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema()),

     Operator(
         "ipex::softmax(Tensor self, int dim, ScalarType ? dtype) -> Tensor",
         [](const Node *node) -> Operation {
           return [](Stack *stack) {
             auto result = AtenIpexJITDev::dil_softmax(
                 (std::move(peek(stack, 0, 3))).toTensor(),
                 (std::move(peek(stack, 1, 3))).toInt(),
                 (std::move(peek(stack, 2, 3))));
             drop(stack, 3);
             pack(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisFromSchema())

    });

} // namespace jit
} // namespace torch
