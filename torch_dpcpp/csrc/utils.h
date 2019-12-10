#pragma once

#include <ATen/Tensor.h>

#include <c10/util/Exception.h>

namespace torch_ipex {

enum DPCPPSubDev {
    CPU,
    GPU
};

bool check_device(const at::Tensor& tensor, DPCPPSubDev sub_dev) {
  auto dev_idx = tensor.get_device();
  TORCH_CHECK(dev_idx >= 0);
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx > 0 ? false : true;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
  }
}

bool check_device(const at::TensorOptions& tensor_options, DPCPPSubDev sub_dev) {
  TORCH_CHECK(tensor_options.has_device());
  TORCH_CHECK(((sub_dev == DPCPPSubDev::CPU) || (sub_dev == DPCPPSubDev::GPU)));
  auto dev = tensor_options.device();
  int64_t dev_idx = 0;
  if (dev.has_index())
    dev_idx = dev.index();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx > 0 ? false : true;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
  }
}

bool check_device(const at::Device& device, DPCPPSubDev sub_dev) {
  int64_t dev_idx = 0;
  if (device.has_index())
    dev_idx = device.index();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx > 0 ? false : true;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
  }
}

}