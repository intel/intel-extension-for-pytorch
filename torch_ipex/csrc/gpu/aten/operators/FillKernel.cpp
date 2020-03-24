#include <ATen/Dispatch.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>

#include <core/ApplyUtils.h>
#include <core/detail/IndexUtils.h>

#include "Loops.h"

using namespace at::dpcpp::detail;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  inline void operator()(T& v) const {
    v = val;
  }

  const T val;
};

void fill_kernel_dpcpp(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(), "fill_dpcpp", [&] {
    at::dpcpp::DPCPP_tensor_apply1<scalar_t>(
        iter.tensor(0), TensorFillOp<scalar_t>(value.to<scalar_t>()));
  });
}

template <typename IndexType, int Dim>
class fill_slice_dpcpp_ker {};

template <typename IndexType, int Dim>
void fillSliceWithIndex(
    TensorInfo<int64_t, IndexType> out,
    IndexType totalSlices,
    IndexType sliceSize,
    IndexType sliceStride) {
  auto& queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t local_size =
      queue.get_device().template get_info<dpcpp_dev_max_wgroup_size>();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_acc = DPCPPAccessor<dpcpp_w_mode>(cgh, out.data);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      IndexType local_id = item_id.get_local_id(0);
      IndexType slice = item_id.get_group_linear_id();
      const uint64_t offset =
          IndexToOffset<int64_t, IndexType, Dim>::get(slice, out);
      int64_t* base = out_acc.template get_pointer<int64_t>() + offset;

      for (IndexType i = local_id; i < sliceSize;
           i += item_id.get_local_range(0)) {
        // Torch indices are 1-based (hence the +1)
        base[i * sliceStride] = i /* + TH_INDEX_BASE */;
      }
    };
    cgh.parallel_for<fill_slice_dpcpp_ker<IndexType, Dim>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(totalSlices * local_size),
            DPCPP::range<1>(local_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

} // impl

Tensor& fill_out(Tensor& self, Scalar value) {
  auto iter = TensorIterator::nullary_op(self);
  impl::fill_kernel_dpcpp(iter, value);
  return self;
}

Tensor& fill_(Tensor& self, Scalar value) {
  return fill_out(self, value);
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "fill_ only supports 0-dimension value tensor but got tensor with ",
      value.dim(),
      " dimensions.");
  return fill_out(self, value.item());
}

Tensor& zero_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::fill_(self, 0);
}

Tensor& fill_slice_with_index(Tensor& t, int dim) {
  int64_t dims = t.dim() == 0 ? 1 : t.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(
      t.scalar_type() == at::kLong || t.scalar_type() == at::kInt,
      "non integer tensor");

  ptrdiff_t inElements = t.numel();
  if (inElements > 0) {
    int64_t sliceSize = t.dim() == 0 ? 1 : t.size(dim);
    ptrdiff_t numSlices = inElements / sliceSize;

#define FILL_INDEX(T, DIM)          \
  impl::fillSliceWithIndex<T, DIM>( \
      info, numSlices, sliceSize, info.strides[collapseDim])

    if (canUse32BitIndexMath(t)) {
      TensorInfo<int64_t, uint32_t> info =
          getTensorInfo<int64_t, unsigned int>(t);
      info.reduceDim(dim);
      int collapseDim = info.collapseDims(dim);
      if (info.isContiguous()) {
        FILL_INDEX(unsigned int, -2);
      } else {
        if (info.dims == 1) {
          FILL_INDEX(unsigned int, 1);
        } else if (info.dims == 2) {
          FILL_INDEX(unsigned int, 2);
        } else {
          FILL_INDEX(unsigned int, -1);
        }
      }
    } else {
      TensorInfo<int64_t, uint64_t> info = getTensorInfo<int64_t, uint64_t>(t);
      info.reduceDim(dim);
      int collapseDim = info.collapseDims(dim);

      // catch-all implementation
      FILL_INDEX(uint64_t, -1);
    }
  }

  return t;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
