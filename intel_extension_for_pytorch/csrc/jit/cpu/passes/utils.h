#pragma once
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/jit/ir/ir.h>

// Check if the memory format of the tensor is ChannelsLast(3d)
bool is_channelslast(c10::TensorType tensor);
// Check if the memory format of the tensor is Contiguous
bool is_contiguous(c10::TensorTypePtr tensor);
