#include <iostream>

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/passes/pass_manager.h>

#include "accelerated_ops.h"
#include "op_rewrite.h"
#include "format_analysis.h"
#include "fusion_pass.h"
#include "dnnl_ops.h"

namespace py = pybind11;
using namespace torch::jit;

static bool pyrys_enabled = false;

PYBIND11_MODULE(pyrys, m) {
  m.doc() = "A DO fusion backend for Pytorch JIT";

  RegisterPass pass_1([](std::shared_ptr<Graph>& g) {
    if (pyrys_enabled) {
      torch::jit::OpRewritePass(g);
    }
  });
  RegisterPass pass_2([](std::shared_ptr<Graph>& g) {
    if (pyrys_enabled) {
      torch::jit::FormatOptimize(g);
    }
  });
  RegisterPass pass_3([](std::shared_ptr<Graph>& g) {
    if (pyrys_enabled) {
      torch::jit::FusionPass(g);
    }
  });

  m.def("enable", []() { pyrys_enabled = true; });
  m.def("disable", []() { pyrys_enabled = false; });
  m.def("dnnl_conv2d", at::native::dnnl_conv2d, "A conv2d function of dnnl");
  m.def("dnnl_conv2d_relu", at::native::dnnl_conv2d_relu, "A conv2d_relu function of dnnl");
  m.def("dnnl_relu", at::native::dnnl_relu, "A relu function of dnnl");
  m.def("dnnl_relu_", at::native::dnnl_relu_, "A relu_ function of dnnl");
  m.def("dnnl_batch_norm", at::native::dnnl_batch_norm, "A batch_norm function of dnnl");
  m.def("dnnl_pooling_max_2d", at::native::dnnl_pooling_max_2d, "A max-pooling-2d funtion of dnnl");
  m.def("dnnl_pooling_avg_2d", at::native::dnnl_pooling_avg_2d, "An avg-pooling-2d funtion of dnnl");
}
