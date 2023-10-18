#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {
namespace utils {

bool isViewOp(torch::jit::Node* n);

bool isBinaryOp(torch::jit::Node* n);

bool isEltwiseOp(torch::jit::Node* n);

bool isSupportedAsInputToDequant(torch::jit::Node* n);

std::vector<int64_t> IntZeroDimTensorToVector(const at::Tensor& tensor);

double getScale(torch::jit::Node* input_node);

std::vector<int64_t> getZPSVector(torch::jit::Node* input_node);

bool isZeroPointSupported(torch::jit::Value* zps);

bool isScaleSupported(torch::jit::Value* scale);

bool compareConstValue(torch::jit::Value* v, double d);

void mark_original_output_dtype(torch::jit::Node* node);

void convertInputTo0DTensor(
    torch::jit::Node* node,
    int input_index,
    at::ScalarType dtype);

void modifyDtypeOfNode(torch::jit::Node* node, at::ScalarType dtype);

void insertTypeCast(
    torch::jit::Node* node,
    int input_index,
    at::ScalarType dtype);

void mayModifyOutputDtype(torch::jit::Node* node);

} // namespace utils
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
