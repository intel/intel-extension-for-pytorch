/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once

#include <fmt/ostream.h>
#include <hl_logger/hllog.hpp>
#include <synapse_ir_types.h>
#include <torch/csrc/jit/ir/ir.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "logging_utils.h"

struct SynIRGraph;
struct SynIRNode;
struct SynIRNodeArray;

struct _SynIRTensor;
using SynIRTensor = _SynIRTensor;

struct SynIRTensorArray;

struct _SynIRTensorGeometry;
using SynIRTensorGeometry = _SynIRTensorGeometry;

struct Uint8Array;
struct Uint64Array;

std::ostream& operator<<(std::ostream& os, const SynIRGraph& graph);
std::ostream& operator<<(std::ostream& os, const SynIRNode& node);
std::ostream& operator<<(std::ostream& os, const SynIRNodeArray& array);
std::ostream& operator<<(std::ostream& os, const SynIRTensor& tensor);
std::ostream& operator<<(std::ostream& os, const SynIRTensorArray& array);
std::ostream& operator<<(std::ostream& os, const SynIRTensorGeometry&);
std::ostream& operator<<(std::ostream& os, const Uint8Array& array);
std::ostream& operator<<(std::ostream& os, const Uint64Array& array);

template <typename T>
std::string array_to_str(const T* ptr, size_t size) {
  if (0 == size || ptr == nullptr) {
    return {"[]"};
  }

  std::stringstream ss{};
  ss << "[";
  for (uint32_t i = 0; i < size - 1; ++i)
    ss << ptr[i] << ",";
  ss << ptr[size - 1] << "]";
  return ss.str();
};

#define CREATE_OSTREAM_FORMATTER(type) \
  template <>                          \
  struct fmt::formatter<type> : ostream_formatter {};

CREATE_OSTREAM_FORMATTER(torch::jit::Graph);
CREATE_OSTREAM_FORMATTER(torch::jit::Node);
CREATE_OSTREAM_FORMATTER(at::ScalarType);
CREATE_OSTREAM_FORMATTER(std::vector<int64_t>);
CREATE_OSTREAM_FORMATTER(c10::Device);
CREATE_OSTREAM_FORMATTER(c10::OperatorName);

#define BRACED_PARAM(p) "{}"
#define FORMAT_AND_MSG(...) \
  HLLOG_APPLY(HLLOG_EMPTY, BRACED_PARAM, ##__VA_ARGS__), ##__VA_ARGS__

#define HLLOG_ENUM_TYPE_NAME IPEXLogger::LoggerTypes
HLLOG_DECLARE_MODULE_LOGGER()

#define IPEX_MOD_TRACE(MOD, ...) HLLOG_TRACE(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_DEBUG(MOD, ...) HLLOG_DEBUG(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_INFO(MOD, ...) HLLOG_INFO(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_WARN(MOD, ...) HLLOG_WARN(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_ERROR(MOD, ...) HLLOG_ERR(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_FATAL(MOD, ...) HLLOG_FATAL(MOD, FORMAT_AND_MSG(__VA_ARGS__));