#include <c10/util/Exception.h>

#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "torch_ipex/csrc/utils.h"
#include "cpu/FusionOPs.h"


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
      "dnnl::conv2d_relu(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      [] (const Node* node) ->Operation {
        if (torch_ipex::check_auto_dnnl()) {
          return [] (Stack& stack) {
            auto result = AtenIpexJITDev::dil_convolution_relu(
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
        } else {
          TORCH_CHECK(false, "PyTorch native path not support convolution relu fusion now")
        }
      },
      aliasAnalysisFromSchema()
      )
    /*
    Operator(
      "dnnl::conv2d_sum(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1, Tensor(a!) accumu, *, Scalar alpha=1) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        return [] (Stack& stack) {
          auto output = (std::move(peek(stack, 7, 9))).toTensor();
          auto result = AtenIpexCPUDev::conv2d_sum(
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
          auto result = AtenIpexCPUDev::conv2d_sum_relu(
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
    });
}
}
