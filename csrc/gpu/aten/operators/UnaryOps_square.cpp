#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <core/detail/TensorInfo.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>

#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& square_out(const Tensor& self, Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_square>(
      TensorIterator::unary_op, out, self, [=](TensorIteratorBase& iter) {
        at::AtenIpexTypeXPU::pow_out(
            iter.tensor(1), 2, const_cast<Tensor&>(iter.tensor(0)));
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at
