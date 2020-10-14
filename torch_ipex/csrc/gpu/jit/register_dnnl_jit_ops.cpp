#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include "dpcpp_ops.h"
#include "accelerated_ops.h"
#include "graph_ext.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

using namespace at::native;

RegisterOperators op({
    Operator(
      "dnnl::reorder(Tensor self) -> Tensor",
      [](const Node* node) -> Operation {
        return [node] (Stack& stack) {
          auto* enode = reinterpret_cast<const NodeExt *>(node);
          auto from = enode->inputFormat(0);
          auto to = enode->inputFormat(1);
          auto groups = enode->getGroupInfo();

          auto result = torch::jit::dpcpp::reorder(
              (std::move(peek(stack, 0, 1))).toTensor(), from, to, groups);
          drop(stack, 1);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::conv2d_relu(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto result = torch::jit::dpcpp::conv2d_relu(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 7))),
              (std::move(peek(stack, 3, 7))).toIntVector(),
              (std::move(peek(stack, 4, 7))).toIntVector(),
              (std::move(peek(stack, 5, 7))).toIntVector(),
              (std::move(peek(stack, 6, 7))).toInt());
          drop(stack, 7);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::conv2d_sigmoid(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto result = torch::jit::dpcpp::conv2d_sigmoid(
              (std::move(peek(stack, 0, 7))).toTensor(),
              (std::move(peek(stack, 1, 7))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 7))),
              (std::move(peek(stack, 3, 7))).toIntVector(),
              (std::move(peek(stack, 4, 7))).toIntVector(),
              (std::move(peek(stack, 5, 7))).toIntVector(),
              (std::move(peek(stack, 6, 7))).toInt());
          drop(stack, 7);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto result = torch::jit::dpcpp::batch_norm(
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
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::fold_weight(Tensor weight, Tensor? bn_weight, Tensor? running_var, float eps) -> Tensor",
      [] (const Node* node) -> Operation {
        return [] (Stack& stack) {
          auto result = torch::jit::dpcpp::fold_weight(
              (std::move(peek(stack, 0, 4))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 1, 4))),
              toOptionalTensor(std::move(peek(stack, 2, 4))),
              (std::move(peek(stack, 3, 4))).toDouble());
          drop(stack, 4);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::fold_bias(Tensor weight, Tensor? bias, Tensor? bn_weight, Tensor? bn_bias, Tensor? running_mean, Tensor? running_var, float eps) -> Tensor",
      [] (const Node* node) -> Operation{
        return [] (Stack& stack) {
          auto result = torch::jit::dpcpp::fold_bias(
              (std::move(peek(stack, 0, 7))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 1, 7))),
              toOptionalTensor(std::move(peek(stack, 2, 7))),
              toOptionalTensor(std::move(peek(stack, 3, 7))),
              toOptionalTensor(std::move(peek(stack, 4, 7))),
              toOptionalTensor(std::move(peek(stack, 5, 7))),
              (std::move(peek(stack, 6, 7))).toDouble());
          drop(stack, 7);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::conv2d_sum(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto output = (std::move(peek(stack, 7, 9))).toTensor();
          auto result = torch::jit::dpcpp::conv2d_sum(
              output,
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 9))),
              (std::move(peek(stack, 3, 9))).toIntVector(),
              (std::move(peek(stack, 4, 9))).toIntVector(),
              (std::move(peek(stack, 5, 9))).toIntVector(),
              (std::move(peek(stack, 6, 9))).toInt(),
              (std::move(peek(stack, 8, 9))).toScalar()
          );
          drop(stack, 9);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::conv2d_sum_relu(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto output = (std::move(peek(stack, 7, 9))).toTensor();
          auto result = torch::jit::dpcpp::conv2d_sum_relu(
              output,
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 9))),
              (std::move(peek(stack, 3, 9))).toIntVector(),
              (std::move(peek(stack, 4, 9))).toIntVector(),
              (std::move(peek(stack, 5, 9))).toIntVector(),
              (std::move(peek(stack, 6, 9))).toInt(),
              (std::move(peek(stack, 8, 9))).toScalar()
          );
          drop(stack, 9);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::mul_add(Tensor self, Tensor other, Tensor accumu, Scalar alpha) -> Tensor",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto result = torch::jit::dpcpp::mul_add(
              (std::move(peek(stack, 0, 4))).toTensor(),
              (std::move(peek(stack, 1, 4))).toTensor(),
              (std::move(peek(stack, 2, 4))).toTensor(),
              (std::move(peek(stack, 3, 4))).toScalar()
          );
          drop(stack, 4);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dpcpp::q_conv2d_sum_relu(Tensor input, Tensor packed_weight, int[2] stride, int[2] padding, int[2] dilation, int groups, float conv_scale, int conv_zpoint, Tensor(a!) accumu, *, float sum_scale, int sum_zpoint) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto output = (std::move(peek(stack, 8, 11))).toTensor();
          auto result = torch::jit::dpcpp::q_conv2d_sum_relu(
              output,
              (std::move(peek(stack, 0, 11))).toTensor(),
              (std::move(peek(stack, 1, 11))).toTensor(),
              (std::move(peek(stack, 2, 11))).toIntVector(),
              (std::move(peek(stack, 3, 11))).toIntVector(),
              (std::move(peek(stack, 4, 11))).toIntVector(),
              (std::move(peek(stack, 5, 11))).toInt(),
              (std::move(peek(stack, 6, 11))).toDouble(),
              (std::move(peek(stack, 7, 11))).toInt(),
              (std::move(peek(stack, 9, 11))).toDouble(),
              (std::move(peek(stack, 10, 11))).toInt()
          );
          drop(stack, 11);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    });
}
}
