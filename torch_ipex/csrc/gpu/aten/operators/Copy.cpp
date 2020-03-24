#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/native/TensorIterator.h>

#include <core/ApplyUtils.h>
#include <core/Exception.h>
#include <core/Guard.h>
#include <core/Memory.h>
#include <core/Stream.h>

using namespace at;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename T>
struct inter_copy_type {
  using type = T;
};

template <>
struct inter_copy_type<uint8_t> {
  using type = int64_t;
};

template <typename T>
using inter_copy_type_t = typename inter_copy_type<T>::type;

template <typename dst_T, typename src_T>
class copy_functor {
 public:
  copy_functor() {}
  void operator()(dst_T& dst_val, const src_T& src_val) const {
    dst_val =
        static_cast<dst_T>(static_cast<inter_copy_type_t<dst_T>>(src_val));
  }
};

#define BUILD_TENSOR_ITER(dst, src, iter) \
  auto iter = TensorIterator();           \
  iter.add_output(dst);                   \
  iter.add_input(src);                    \
  iter.dont_resize_outputs();             \
  iter.dont_compute_common_dtype();       \
  iter.build();

// Copy operator for the pointerwise apply kernel
template <typename dst_T, typename src_T>
struct CopyOp {
  static void apply(Tensor& dst, const Tensor& src) {
    DPCPP_tensor_apply2<dst_T, src_T>(dst, src, copy_functor<dst_T, src_T>());
  }
};

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter, bool non_blocking) {
  auto numel = iter.numel();
  if (numel == 0) {
    return;
  }

  // We can memcpy the memory if:
  // -both tensors are contiguous; or,
  // -there is only one element to copy; or,
  // -FIXME: if both tensors have matching size and stride arrays, and no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is
  // contiguous).
  // -AND: both tensors have the same type.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool memcpy_eligible = same_type && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);
  DPCPPGuard device_guard(src_device);
  // FIXME:: figure out how to copy buffer between two device
  TORCH_CHECK(src_device == dst_device, "device not match");

  if (src_device != dst_device) {
    // FIXME
  }

  if (memcpy_eligible) {
    dpcppMemcpyAsync(
        iter.data_ptr(0),
        iter.data_ptr(1),
        numel * iter.element_size(0),
        DeviceToDevice);
  } else {
    // auto src_contig = at::empty_like(iter.tensor(0),
    //      iter.tensor(1).options().dtype(iter.tensor(0).dtype()));
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(0), "copy_", [&] {
      using dst_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(1), "copy_", [&] {
        CopyOp<dst_t, scalar_t>::apply(iter.tensor(0), iter.tensor(1));
      });
    });
  }

  if (src_device != dst_device) {
    // FixMe
  }
}

void copy_from_cpu(TensorIterator& iter, bool non_blocking) {
  Tensor& dst = iter.tensor(0);
  Tensor& src = iter.tensor(1);

  Tensor dst_contig = dst.contiguous();
  Tensor src_contig = src.contiguous();

  dpcppMemcpy(
      dst_contig.data_ptr(),
      src_contig.data_ptr(),
      src.numel() * src.dtype().itemsize(),
      HostToDevice);
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      src.scalar_type(),
      "copy_from_cpu",
      [&]() {
        // FIXME: [Remove Me] Make ComputeCPP happy
        scalar_t dummy = 0;
        dummy = dummy;
        BUILD_TENSOR_ITER(dst, dst_contig, _iter);
        copy_device_to_device(_iter, non_blocking);
      });
}

void copy_to_cpu(TensorIterator& iter, bool non_blocking) {
  Tensor& dst = iter.tensor(0);
  Tensor& src = iter.tensor(1);

  Tensor dst_contig = dst.contiguous();
  Tensor src_contig = src.contiguous();

  DPCPPGuard device_guard(src.device());
  dpcppMemcpy(
      dst_contig.data_ptr(),
      src_contig.data_ptr(),
      src.numel() * src.dtype().itemsize(),
      DeviceToHost);
  // DispatchStub is not exposed by torch
  // BUILD_TENSOR_ITER(dst, dst_contig, _iter);
  // copy_stub(kCPU, _iter, non_blocking);
  at::native::copy_(dst, dst_contig);
}

void copy_from_cpu_async_(TensorIterator& iter) {
  Tensor& dst = iter.tensor(0);
  Tensor& src = iter.tensor(1);

  TORCH_CHECK(dst.is_contiguous(), "Target tensor must be contiguous.");
  TORCH_CHECK(src.is_contiguous(), "Source tensor must be contiguous.");

  if (dst.numel() == 0) {
    return;
  }

  DPCPPGuard device_guard(dst.device());
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      src.scalar_type(),
      "copy_from_cpu_async",
      [&]() {
        dpcppMemcpyAsync(
            dst.data_ptr<scalar_t>(),
            src.data_ptr<scalar_t>(),
            src.numel() * sizeof(scalar_t),
            HostToDevice);
      });
}

void copy_to_cpu_async_(TensorIterator& iter) {
  Tensor& dst = iter.tensor(0);
  Tensor& src = iter.tensor(1);

  TORCH_CHECK(dst.is_contiguous(), "Target tensor must be contiguous.");
  TORCH_CHECK(src.is_contiguous(), "Source tensor must be contiguous.");

  if (dst.numel() == 0) {
    return;
  }

  DPCPPGuard device_guard(src.device());

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      src.scalar_type(),
      "copy_to_cpu_async",
      [&]() {
        dpcppMemcpyAsync(
            dst.data_ptr<scalar_t>(),
            src.data_ptr<scalar_t>(),
            src.numel() * sizeof(scalar_t),
            DeviceToHost);
      });
}

void copy_kernel_dpcpp(TensorIterator& iter, bool non_blocking);

template <typename dst_T>
void _copy__dpcpp(TensorIterator& iter, bool non_blocking) {
  Tensor& dst = iter.tensor(0);
  Tensor& src = iter.tensor(1);

  TORCH_CHECK(dst.numel() == src.numel(), "sizes do not match");
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      src.scalar_type(),
      "_copy_dpcpp",
      [&]() {
        if (dst.device().type() == at::kDPCPP &&
            src.device().type() == at::kDPCPP) {
          copy_device_to_device(iter, non_blocking);
        } else {
          if (dst.device().type() == at::kDPCPP) {
            if (std::is_same<dst_T, scalar_t>::value) {
              if (non_blocking) {
                copy_from_cpu_async_(iter);
              } else {
                copy_from_cpu(iter, non_blocking);
              }
            } else {
              // Do a dtype converting copy on the CPU,
              // then copy to device
              Tensor srcf =
                  at::empty_like(src, src.options().dtype(dst.dtype()));
              // DispatchStub is not exposed by torch
              // BUILD_TENSOR_ITER(srcf, src, iter1)
              // copy_stub(kCPU, iter1, non_blocking);
              at::native::copy_(srcf, src);
              BUILD_TENSOR_ITER(dst, srcf, iter2)
              copy_from_cpu(iter2, non_blocking);
            }
          } else {
            if (std::is_same<dst_T, scalar_t>::value) {
              if (non_blocking) {
                copy_to_cpu_async_(iter);
              } else {
                copy_to_cpu(iter, non_blocking);
              }
            } else {
              // Copy to CPU as the same dtype, then do a
              // dtype converting copy
              Tensor srcf =
                  at::empty_like(src, dst.options().dtype(src.dtype()));
              BUILD_TENSOR_ITER(srcf, src, iter1)
              copy_to_cpu(iter1, non_blocking);
              BUILD_TENSOR_ITER(dst, srcf, iter2)
              copy_kernel_dpcpp(iter2, non_blocking);
            }
          }
        }
      });
}

void copy_kernel_dpcpp(TensorIterator& iter, bool non_blocking) {
  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Half,
      ScalarType::Bool,
      iter.tensor(0).scalar_type(),
      "_copy__dpcpp",
      [&]() { _copy__dpcpp<scalar_t>(iter, non_blocking); });
}

} // impl

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  // TODO: valid check

  BUILD_TENSOR_ITER(self, src, iter);

  if (iter.numel() == 0) {
    return self;
  }

  impl::copy_kernel_dpcpp(iter, non_blocking);

  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
