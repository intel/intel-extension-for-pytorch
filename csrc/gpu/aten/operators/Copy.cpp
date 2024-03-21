#include <ATen/ATen.h>

#include <ATen/core/TensorBody.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include <ATen/native/Resize.h>
#include <c10/core/ScalarType.h>
#include <core/Event.h>
#include <core/Guard.h>
#include <core/Memory.h>
#include <core/Stream.h>
#include <core/detail/TensorInfo.h>
#include <quantized/QTensor.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>

#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/Utils.h>
#include <oneDNN/oneDNN.h>
#include "Loops.h"

using namespace at;
using namespace xpu::dpcpp;

namespace at {
namespace impl {

#define BUILD_TENSOR_ITER(dst, src, iter)       \
  auto iter = TensorIteratorConfig()            \
                  .set_check_mem_overlap(true)  \
                  .add_output(self)             \
                  .add_input(src)               \
                  .resize_outputs(false)        \
                  .check_all_same_dtype(false)  \
                  .check_all_same_device(false) \
                  .build();

template <typename scalar_t>
struct direct_copy_kernel_gpu_functor {
  scalar_t operator()(scalar_t src_val) const {
    return src_val;
  }
};

void direct_copy_kernel_gpu(TensorIteratorBase& iter) {
  ScalarType dtype = iter.common_dtype();
  if (isQIntType(dtype)) {
    IPEX_DISPATCH_QINT_TYPES(dtype, "direct_copy_kernel_gpu", [&] {
      direct_copy_kernel_gpu_functor<scalar_t> f;
      dpcpp_fast_mode_kernel_for_tensor_iter(iter, f);
    });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(
        kBool,
        kHalf,
        kBFloat16,
        kComplexHalf,
        kFloat8_e4m3fn,
        kFloat8_e5m2,
        dtype,
        "direct_copy_kernel_gpu",
        [&] {
          direct_copy_kernel_gpu_functor<scalar_t> f;
          dpcpp_fast_mode_kernel_for_tensor_iter(iter, f);
        });
  }
}

template <typename scalar_t>
struct conj_kernel_gpu_functor {
  scalar_t operator()(scalar_t src_val) const {
    return std::conj(src_val);
  }
};

void conj_kernel_gpu(TensorIterator& iter) {
  AT_DISPATCH_SWITCH(
      iter.common_dtype(),
      "conj_kernel_gpu",
      AT_DISPATCH_CASE_ALL_TYPES_AND3(
          kBool,
          kBFloat16,
          kHalf,
          [&] {
            // Conj is a no-op for non-complex types
            direct_copy_kernel_gpu(iter);
          })
          AT_DISPATCH_CASE_COMPLEX_TYPES_AND(
              kComplexHalf, iter.common_dtype(), "conj_kernel_gpu", [&] {
                conj_kernel_gpu_functor<scalar_t> f;
                dpcpp_fast_mode_kernel_for_tensor_iter(iter, f);
              }));
}

template <typename scalar_t>
struct neg_conj_kernel_gpu_functor {
  scalar_t operator()(scalar_t src_val) const {
    return std::conj(-src_val);
  }
};

void neg_conj_kernel_gpu(TensorIteratorBase& iter) {
  IPEX_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_kernel_gpu", [&] {
    neg_conj_kernel_gpu_functor<scalar_t> f;
    dpcpp_fast_mode_kernel_for_tensor_iter(iter, f);
  });
}

template <typename scalar_t>
struct neg_kernel_gpu_functor {
  scalar_t operator()(scalar_t src_val) const {
    return -src_val;
  }
};

void neg_kernel_gpu(TensorIteratorBase& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf,
      kBFloat16,
      kComplexHalf,
      iter.common_dtype(),
      "neg_kernel_gpu",
      [&] {
        neg_kernel_gpu_functor<scalar_t> f;
        dpcpp_fast_mode_kernel_for_tensor_iter(iter, f);
      });
}

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_xpu() && src_device.is_xpu());
    return false;
  } else if (
      dst_device.is_xpu() && src_device.is_xpu() &&
      (dst_device != src_device)) {
    // Across device copies need temporaries if p2p not enabled
    return !p2p_enabled;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use memcpyAsync
    return false;
  } else if (dst_device.is_xpu() && src_device.is_xpu()) {
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

  auto dst_queue =
      (sycl::queue*)getCurrentDPCPPStream(dst_device.index()).queue();
  auto src_queue =
      (sycl::queue*)getCurrentDPCPPStream(src_device.index()).queue();
  auto dst_dev = dst_queue->get_device();
  auto src_dev = src_queue->get_device();
  return src_dev.ext_oneapi_can_access_peer(
      dst_dev, sycl::ext::oneapi::peer_access::access_supported);
}

template <typename func_t>
void dpcpp_loops_memcpy_kernel(TensorIteratorBase& iter, const func_t& f) {
  xpu::dpcpp::Array<char*, 2> data;
  data[0] = (char*)iter.data_ptr(0);
  data[1] = (char*)iter.data_ptr(1);
  int vec_size = at::native::Memory::can_vectorize_up_to_loop<func_t>(
      dpcppGetDeviceIdOfCurrentQueue(), data);
  auto ic = TrivialOffsetCalculator<1>();
  launch_vectorized_kernel(iter.numel(), f, data, ic, vec_size);
}

template <typename func_t>
void dpcpp_kernel_loops_memcpy_for_tensor_iter(
    TensorIteratorBase& iter,
    const func_t& f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_kernel_loops_memcpy_for_tensor_iter<func_t>(sub_iter, f);
    }
    return;
  }

  dpcpp_loops_memcpy_kernel<func_t>(iter, f);
}

template <typename scalar_t>
struct memcpyAsync_functor {
  scalar_t operator()(scalar_t src_val) const {
    return src_val;
  }
};

void memcpyAsync(
    TensorIteratorBase& iter,
    DPCPPStream& copy_stream,
    bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);
  auto dtype = iter.dtype(0);
  if (dst_device == src_device) {
    if (isQIntType(dtype)) {
      IPEX_DISPATCH_QINT_TYPES(dtype, "copy_loops_memcpy", [&] {
        memcpyAsync_functor<scalar_t> f;
        dpcpp_kernel_loops_memcpy_for_tensor_iter(iter, f);
      });
    } else {
      IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(
          kBool,
          kHalf,
          kBFloat16,
          kComplexHalf,
          kFloat8_e4m3fn,
          kFloat8_e5m2,
          dtype,
          "copy_loops_memcpy",
          [&] {
            memcpyAsync_functor<scalar_t> f;
            dpcpp_kernel_loops_memcpy_for_tensor_iter(iter, f);
          });
    }
  } else {
    TORCH_INTERNAL_ASSERT(p2p_enabled == true);
    auto dst = (char*)iter.data_ptr(0);
    auto src = (char*)iter.data_ptr(1);
    size_t size = iter.numel() * iter.element_size(0);
    auto q = (sycl::queue*)copy_stream.queue();
    q->copy(src, dst, size);
  }
}

void copy_device_to_device(
    TensorIterator& iter,
    bool non_blocking,
    bool p2p_enabled) {
  auto numel = iter.numel();
  if (numel == 0) {
    return;
  }

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible =
      same_type && same_conj && same_neg && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  DPCPPGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy.
  DPCPPStream copy_stream = getCurrentDPCPPStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    DPCPPEvent dst_ready;
    device_guard.set_device(dst_device);
    dst_ready.record(getCurrentDPCPPStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    // SYCL queue.memcpy performance is worse than SYCL copy kernel
    // implementation. JIRA:
    // https://jira.devtools.intel.com/browse/CMPLRLLVM-41292
    memcpyAsync(iter, copy_stream, p2p_enabled);
  } else {
    if (same_neg) {
      if (!same_conj) {
        conj_kernel_gpu(iter);
      } else {
        direct_copy_kernel_gpu(iter);
      }
    } else {
      if (!same_conj) {
        neg_conj_kernel_gpu(iter);
      } else {
        neg_kernel_gpu(iter);
      }
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.
    // Still on src_device, record stream event
    DPCPPEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready.block(getCurrentDPCPPStream(dst_device.index()));
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

    bool requires_across_device_temporaries =
        (iter.device(0) != iter.device(1)) &&
        iter.device(0).type() == c10::DeviceType::XPU &&
        iter.device(1).type() == c10::DeviceType::XPU;

    // If non_blocking is true - type conversions are performed on the GPU
    // for CPU-GPU copies, otherwise type conversions are performed on the CPU.
    // Type conversions are performed on the src device for GPU-GPU copies.
    if (iter.device_type(0) == kXPU || non_blocking) {
      if (requires_across_device_temporaries) {
        dst_contig = at::empty(dst.sizes(), dst.options().device(kCPU));
      } else {
        dst_contig = dst.is_contiguous()
            ? dst
            : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      }
      if (dst.is_quantized()) {
        src_contig =
            expand_as_quantized_dpcpp(iter.tensor(1).to(iter.dtype(0)), dst)
                .contiguous();
      } else {
        src_contig =
            iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
      }
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type)
          ? dst
          : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(iter.tensor(1).is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(iter.tensor(1).is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      // TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.is_xpu() && src_device.is_xpu()) {
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    return;
  }

  // Copy between CPU and GPU
  OptionalDPCPPGuard device_guard;
  dpcppMemcpyKind kind;
  if (dst_device.type() == c10::DeviceType::XPU && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = HostToDevice;
  } else if (dst_device.is_cpu() && src_device.type() == c10::DeviceType::XPU) {
    device_guard.set_device(src_device);
    kind = DeviceToHost;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);

  if (non_blocking) {
    // here do the dpcpp copy synchronisation.
    // we use a very simple version for the singleton sycl queue.
    // TODO: enhance the full functionality in multi-queue scenario.
    dpcppMemcpyAsync(dst, src, nbytes, kind);
  } else {
    dpcppMemcpy(dst, src, nbytes, kind);
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
    iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
    iter.tensor(0).neg_();
  }
}

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  // TODO: valid check
  if (self.is_same(src)) {
    return self;
  }

  if (self.is_quantized() && src.is_quantized()) {
    auto mfmt = self.is_contiguous(at::MemoryFormat::ChannelsLast)
        ? at::MemoryFormat::ChannelsLast
        : at::MemoryFormat::Contiguous;
    if (src.qscheme() == kPerTensorAffine) {
      self = _empty_affine_quantized(
          self.sizes(),
          self.options(),
          src.q_scale(),
          src.q_zero_point(),
          mfmt);
      set_quantizer_(self, src.quantizer());
    } else {
      self = _empty_per_channel_affine_quantized(
          self.sizes(),
          src.q_per_channel_scales().to(self.device()),
          src.q_per_channel_zero_points().to(self.device()),
          src.q_per_channel_axis(),
          self.options(),
          mfmt);
      set_quantizer_(self, src.quantizer());
    }
  }

  BUILD_TENSOR_ITER(self, src, iter);

  if (iter.numel() == 0) {
    return self;
  }

  impl::copy_kernel_dpcpp(iter, non_blocking);

  return self;
}
} // namespace impl

namespace AtenIpexTypeXPU {
Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  return impl::copy_(self, src, non_blocking);
}

Tensor _to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  return at::native::_to_copy(
      self,
      dtype,
      layout,
      device,
      pin_memory,
      non_blocking,
      optional_memory_format);
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  return impl::copy_(self, src, non_blocking);
}

Tensor _to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  return at::native::_to_copy(
      self,
      dtype,
      layout,
      device,
      pin_memory,
      non_blocking,
      optional_memory_format);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
