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

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.type() == c10::DeviceType::DPCPP &&
      src_device.type() == c10::DeviceType::DPCPP );
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use sycl copy
    return false;
  } else if (dst_device.type() == c10::DeviceType::DPCPP &&
             src_device.type() == c10::DeviceType::DPCPP ) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GPU.
    return true;
  }
}

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  // no p2p so far
  return false;
}

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter, bool non_blocking) {
  auto numel = iter.numel();
  if (numel == 0) {
    return;
  }

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
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
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBFloat16, kBool, iter.dtype(0), "copy_", [&] {
          using dst_t = scalar_t;
          AT_DISPATCH_ALL_TYPES_AND3(
              kHalf, kBFloat16, kBool, iter.dtype(1), "copy_", [&] {
                CopyOp<dst_t, scalar_t>::apply(iter.tensor(0), iter.tensor(1));
              });
        });
  }

  if (src_device != dst_device) {
    // FixMe
  }
}

void copy_kernel_dpcpp(TensorIterator& iter, bool non_blocking) {
  AT_ASSERT(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it invovles the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // Type conversions are performed on the CPU for CPU-GPU copies and on
    // the src device for GPU-GPU copies.
    if (iter.device_type(0) == kDPCPP) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1));
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.type() == c10::DeviceType::DPCPP &&
      src_device.type() == c10::DeviceType::DPCPP) {
    copy_device_to_device(iter, non_blocking);
    return;
  }

  // Copy between CPU and GPU
  OptionalDPCPPGuard device_guard;
  dpcppMemcpyKind kind;
  if (dst_device.type() == c10::DeviceType::DPCPP &&
      src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = HostToDevice;
  } else if (dst_device.is_cpu() &&
             src_device.type() == c10::DeviceType::DPCPP) {
    device_guard.set_device(src_device);
    kind = DeviceToHost;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);

  dpcppMemcpyAsync(dst, src, nbytes, kind);

  if (non_blocking) {
    // here do the cuda copy synchronisation.
    // we use a very simple version for the singleton sycl queue.
    // TODO: enhance this for the multi-queue.
    // void* ptr = (dst_device == kCPU ? dst : src);
    // AT_CUDA_CHECK(THCCachingHostAllocator_recordEvent(ptr, stream));
  } else {
    auto& queue = getCurrentDPCPPStream().dpcpp_queue();
    queue.wait_and_throw();
  }
}

} // namespace impl

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
