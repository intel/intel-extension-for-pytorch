/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "logging_hl.h"
#include <synapse_ir_types.h>
#include <vector>
#include "logging_utils.h"

namespace {
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vect) {
  for (auto& elem : vect) {
    os << elem << " ";
  }
  return os;
}

} // namespace

std::ostream& operator<<(
    std::ostream& os,
    const SynIRTensorGeometry& geometry) {
  os << "SynIRTensorGeometry dim_t: " << geometry.dims
     << "\tsizes: " << array_to_str(geometry.sizes, geometry.dims)
     << "\tstrides: " << array_to_str(geometry.strides, geometry.dims);
  return os;
}
std::ostream& operator<<(std::ostream& os, const SynIRTensor& tensor) {
  os << "SynIRTensor id: " << tensor.id
     << "\ttensorDataType: " << tensor.tensorDataType
     << "\tbufferDataType: " << tensor.bufferDataType
     << "\tdata: " << tensor.data << "\tdataSize: " << tensor.dataSize
     << "\ttype: " << tensor.type << "\tisPersistent: " << tensor.isPersistent
     << std::endl
     << "\tgeometry: " << tensor.geometry;

  return os;
}
std::ostream& operator<<(std::ostream& os, const SynIRTensorArray& array) {
  os << "SynIRTensorArray size: " << array.tensorArrSize << std::endl;
  for (uint32_t i = 0; i < array.tensorArrSize; ++i) {
    os << *array.tensorArr[i] << std::endl;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Uint8Array& array) {
  os << array_to_str(array.uintArr, array.arrSize);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Uint64Array& array) {
  os << array_to_str(array.uintArr, array.arrSize);
  return os;
}

std::ostream& operator<<(std::ostream& os, const SynIRNode& node) {
  os << "SynIRNode "
     << "guid: " << node.guid << "\tnodeName: "
     << node.name
     //  << "\tblockingNodesIDs: " << node.blockingNodesIDs TODO:
     //  blockingNodesIDs field contains garbage for WS nodes
     << "\tnodeType: " << node.type << "\tid: "
     << node.id
     //  << "\tpermutation: " << node.permutation[MAX_HABANA_DIM] << std::endl
     //  // TODO:
     << std::endl
     << "\tinputTensors: " << node.inputTensors << std::endl
     << "\toutputTensors: " << node.outputTensors << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const SynIRNodeArray& array) {
  os << "SynIRNodeArray size: " << array.nodeArrSize << std::endl;
  for (uint32_t i = 0; i < array.nodeArrSize; ++i) {
    os << *array.nodeArr[i];
  }
  os << std::endl;
  return os;
}

std::ostream& operator<<(std::ostream& os, const SynIRGraph& graph) {
  os << "SynIRGraph\tdeviceType: " << graph.deviceType
     << "\ttrainingMode: " << graph.trainingMode << std::endl
     << graph.nodes;
  return os;
}
namespace IPEXLogger {
// create loggers (all the log files are created immediately when the module is
// loaded)
static void createModuleLoggers(LoggerTypes) {}

static void createModuleLoggersOnDemand(LoggerTypes) {
  hl_logger::LoggerCreateParams default_params;
  default_params.logFileName = "ipex_log.txt";
  default_params.logFileAmount = 5;
  default_params.defaultLoggingLevel = HLLOG_LEVEL_WARN;
  hl_logger::createLoggersOnDemand(
      {LoggerTypes::IPEX_OPS,
       LoggerTypes::IPEX_MEMORY,
       LoggerTypes::IPEX_RUNTIME,
       LoggerTypes::IPEX_SYNGRAPH},
      default_params);
}

} // namespace IPEXLogger

HLLOG_DEFINE_MODULE_LOGGER(
    IPEX_OPS,
    IPEX_MEMORY,
    IPEX_RUNTIME,
    IPEX_SYNGRAPH,
    LOG_MAX)