#include "torch/csrc/jit/runtime/operator.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "accelerated_ops.h"
#include "graph_ext.h"
#include "cpu/DevOPs.h"
//#include "dnnl_ops.h"


namespace torch {
namespace jit {

c10::AliasAnalysisKind  aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

using namespace torch_ipex::cpu;

RegisterOperators op({
    Operator(
      "dnnl::reorder(Tensor self) -> Tensor",
      [](const Node* node) -> Operation {
        return [node] (Stack& stack) {
          auto* enode = reinterpret_cast<const NodeExt *>(node);
          auto from = enode->inputFormat(0);
          auto to = enode->inputFormat(1);
          auto groups = enode->getGroupInfo();

          // auto result = dnnl_reorder(
          //    (std::move(peek(stack, 0, 1))).toTensor(), from, to, groups);
          auto result = at::Tensor();
          drop(stack, 1);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::relu(Tensor self) -> Tensor",
      [](const Node* node) -> Operation {
        return [] (Stack& stack) {
          auto result = AtenIpexCPUDev::dil_relu((std::move(peek(stack, 0, 1))).toTensor());
          drop(stack, 1);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::relu_(Tensor(a!) self) -> Tensor(a!)",
      [] (const Node* node) -> Operation {
        return [] (Stack& stack) {
          at::Tensor input;
          pop(stack, input);
          auto result = AtenIpexCPUDev::dil_relu_(input);
          push(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      [] (const Node* node) -> Operation {
        return [] (Stack& stack) {
          auto result = AtenIpexCPUDev::dil_convolution(
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
      "dnnl::conv2d_relu(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto result = AtenIpexCPUDev::dil_convolution_relu(
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
      "dnnl::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          /*
          auto result = dnnl_batch_norm(
              (std::move(peek(stack, 0, 9))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 1, 9))),
              toOptionalTensor(std::move(peek(stack, 2, 9))),
              toOptionalTensor(std::move(peek(stack, 3, 9))),
              toOptionalTensor(std::move(peek(stack, 4, 9))),
              (std::move(peek(stack, 5, 9))).toBool(),
              (std::move(peek(stack, 6, 9))).toDouble(),
              (std::move(peek(stack, 7, 9))).toDouble(),
              (std::move(peek(stack, 8, 9))).toBool());
          */
          auto result = at::Tensor();
          drop(stack, 9);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::fold_weight(Tensor weight, Tensor? bn_weight, Tensor? running_var, float eps) -> Tensor",
      [] (const Node* node) -> Operation {
        return [] (Stack& stack) {
          /*
          auto result = dnnl_fold_weight(
              (std::move(peek(stack, 0, 4))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 1, 4))),
              toOptionalTensor(std::move(peek(stack, 2, 4))),
              (std::move(peek(stack, 3, 4))).toDouble());
          */
          auto result = at::Tensor();
          drop(stack, 4);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::fold_bias(Tensor weight, Tensor? bias, Tensor? bn_weight, Tensor? bn_bias, Tensor? running_mean, Tensor? running_var, float eps) -> Tensor",
      [] (const Node* node) -> Operation{
        return [] (Stack& stack) {
          /*
          auto result = dnnl_fold_bias(
              (std::move(peek(stack, 0, 7))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 1, 7))),
              toOptionalTensor(std::move(peek(stack, 2, 7))),
              toOptionalTensor(std::move(peek(stack, 3, 7))),
              toOptionalTensor(std::move(peek(stack, 4, 7))),
              toOptionalTensor(std::move(peek(stack, 5, 7))),
              (std::move(peek(stack, 6, 7))).toDouble());
          */
          auto result = at::Tensor();
          drop(stack, 7);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::sum(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          /*
          auto result = dnnl_sum(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar()
          );
          */
          auto result = at::Tensor();
          drop(stack, 3);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::sum_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
      [] (const Node* node) ->Operation{
        return [](Stack &stack) {
          auto self = (std::move(peek(stack, 0, 3))).toTensor();
          /*
          auto result = dnnl_sum_(
              self,
              (std::move(peek(stack, 1, 3))).toTensor(),
              (std::move(peek(stack, 2, 3))).toScalar());
          */
          auto result = at::Tensor();
          drop(stack, 3);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    /*
    Operator(
      "dnnl::conv2d_sum(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1, Tensor(a!) accumu, *, Scalar alpha=1) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto output = (std::move(peek(stack, 7, 9))).toTensor();
          auto result = dnnl_conv2d_sum(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 9))),
              (std::move(peek(stack, 3, 9))).toIntVector(),
              (std::move(peek(stack, 4, 9))).toIntVector(),
              (std::move(peek(stack, 5, 9))).toIntVector(),
              (std::move(peek(stack, 6, 9))).toInt(),
              output,
              (std::move(peek(stack, 8, 9))).toScalar()
          );
          auto result = at::Tensor();
          drop(stack, 9);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::conv2d_sum_relu(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1, Tensor(a!) accumu, *, Scalar alpha=1) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto output = (std::move(peek(stack, 7, 9))).toTensor();
          auto result = dnnl_conv2d_sum_relu(
              (std::move(peek(stack, 0, 9))).toTensor(),
              (std::move(peek(stack, 1, 9))).toTensor(),
              toOptionalTensor(std::move(peek(stack, 2, 9))),
              (std::move(peek(stack, 3, 9))).toIntVector(),
              (std::move(peek(stack, 4, 9))).toIntVector(),
              (std::move(peek(stack, 5, 9))).toIntVector(),
              (std::move(peek(stack, 6, 9))).toInt(),
              output,
              (std::move(peek(stack, 8, 9))).toScalar()
          );
          auto result = at::Tensor();
          drop(stack, 9);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),*/
    Operator(
      "dnnl::pooling_max_2d(Tensor input, int[2] kernel_size, int[2] stride=1, int[2] padding=0, int[2] dilation=1, bool ceil_mode=0) -> Tensor(a!)",
      [] (const Node *node) ->Operation {
        return [] (Stack& stack) {
          /*
          auto result = dnnl_pooling_max_2d(
              (std::move(peek(stack, 0, 6))).toTensor(),      // Input tensor
              (std::move(peek(stack, 1, 6))).toIntVector(),  // Kernel size
              (std::move(peek(stack, 2, 6))).toIntVector(),  // Stride
              (std::move(peek(stack, 3, 6))).toIntVector(),  // Padding
              (std::move(peek(stack, 4, 6))).toIntVector(),  // Dilation
              (std::move(peek(stack, 5, 6))).toBool());       // Ceil mode
          */
          auto result = at::Tensor();
          drop(stack, 6);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "dnnl::pooling_avg_2d(Tensor input, int[2] kernel_size, int[2] stride=1, int[2] padding=0, bool ceil_mode=0, bool count_include_pad=True, int? divisor_override=None) -> Tensor(a!)",
      [] (const Node *node) ->Operation {
        return [] (Stack& stack) {
          /*
          auto result = dnnl_pooling_avg_2d(
              (std::move(peek(stack, 0, 7))).toTensor(),      // Input tensor
              (std::move(peek(stack, 1, 7))).toIntVector(),  // Kernel size
              (std::move(peek(stack, 2, 7))).toIntVector(),  // Stride
              (std::move(peek(stack, 3, 7))).toIntVector(),  // Padding
              (std::move(peek(stack, 4, 7))).toBool());       // Ceil mode
          */
          auto result = at::Tensor();
          drop(stack, 7);
          pack(stack, std::move(result));
          return 0;
        };
      },
      aliasAnalysisFromSchema()
      ),
    });
}
}
