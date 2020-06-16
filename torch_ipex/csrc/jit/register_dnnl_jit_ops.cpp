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
      "ipex::conv2d_relu(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
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
          TORCH_CHECK(false, "PyTorch native path not support convolution relu fusion now for 2d case");
        }
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "ipex::conv3d_relu(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor",
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
          TORCH_CHECK(false, "PyTorch native path not support convolution relu fusion now for 3d case");
        }
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "ipex::conv2d_sum(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        if (torch_ipex::check_auto_dnnl()) {
          return [] (Stack& stack) {
            auto output = (std::move(peek(stack, 7, 9))).toTensor();
            auto result = AtenIpexJITDev::dil_convolution_sum(
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
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        } else {
          TORCH_CHECK(false, "PyTorch native path not support convolution sum fusion now for 2d case");
        }
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "ipex::conv3d_sum(Tensor input, Tensor weight, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        if (torch_ipex::check_auto_dnnl()) {
          return [] (Stack& stack) {
            auto output = (std::move(peek(stack, 7, 9))).toTensor();
            auto result = AtenIpexJITDev::dil_convolution_sum(
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
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        } else {
          TORCH_CHECK(false, "PyTorch native path not support convolution sum fusion now for 3d case");
        }
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "ipex::conv2d_sum_relu(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        if (torch_ipex::check_auto_dnnl()) {
          return [] (Stack& stack) {
            auto output = (std::move(peek(stack, 7, 9))).toTensor();
            auto result = AtenIpexJITDev::dil_convolution_sum_relu(
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
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        } else {
          TORCH_CHECK(false, "PyTorch native path not support convolution sum relu fusion now for 2d case");
        }
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "ipex::conv3d_sum_relu(Tensor input, Tensor weight, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        if (torch_ipex::check_auto_dnnl()) {
          return [] (Stack& stack) {
            auto output = (std::move(peek(stack, 7, 9))).toTensor();
            auto result = AtenIpexJITDev::dil_convolution_sum_relu(
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
            drop(stack, 9);
            pack(stack, std::move(result));
            return 0;
          };
        } else {
          TORCH_CHECK(false, "PyTorch native path not support convolution sum relu fusion now for 3d case");
        }
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "ipex::prepack_weight(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups) -> Tensor(a!)",
      [] (const Node* node) ->Operation {
        if (torch_ipex::check_auto_dnnl()) {
          return [] (Stack& stack) {
            return 0;
          };
        } else {
          TORCH_CHECK(false, "PyTorch native path not support prepack weight now");
        }
      },
      aliasAnalysisFromSchema()
      ),
    Operator(
      "ipex::linear_relu(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
      [] (const Node* node) ->Operation {
        if (torch_ipex::check_auto_dnnl()) {
          return [] (Stack& stack) {
            auto result = AtenIpexJITDev::dil_linear_fuse_relu(
                (std::move(peek(stack, 0, 3))).toTensor(),
                (std::move(peek(stack, 1, 3))).toTensor(),
                toOptionalTensor(std::move(peek(stack, 2, 3)))
            );
            drop(stack, 3);
            pack(stack, std::move(result));
            return 0;
          };
        } else {
          TORCH_CHECK(false, "PyTorch native path not support linear relu fusion now");
        }
      },
      aliasAnalysisFromSchema()
      )
    });
}
}
