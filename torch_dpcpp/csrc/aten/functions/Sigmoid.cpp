#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <core/SYCLContext.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>

#include <utils/Numerics.h>
#include <functions/Resize.h>

namespace at {
namespace native {

template <typename scalar_t>
class sigmoid_sycl_ker {};

template <typename scalar_t>
static inline void sigmoid_sycl(Tensor & output, const Tensor & self) {
  static const auto write_mode = cl::sycl::access::mode::discard_write;
  static const auto read_mode = cl::sycl::access::mode::read;
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  int64_t rng, grng, tile_size, size;
  c10::sycl::parallel_for_setup(self.numel(), tile_size, rng, grng);
  size = self.numel() * sizeof(scalar_t);
  resize_sycl_(output, self.sizes().vec());

  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    auto in_acc =
        c10::sycl::SYCLAccessor<read_mode>(cgh, self.data_ptr<scalar_t>(), size);
    auto out_acc =
        c10::sycl::SYCLAccessor<write_mode>(cgh, output.data_ptr<scalar_t>(), size);
    cgh.parallel_for<sigmoid_sycl_ker<scalar_t>>(cl::sycl::nd_range<1>(
        cl::sycl::range<1>(grng), cl::sycl::range<1>(tile_size)),
        [=](cl::sycl::nd_item<1> item) {
      size_t id = item.get_global_linear_id();
      auto in_ptr = in_acc.template get_pointer<scalar_t>();
      auto out_ptr = out_acc.template get_pointer<scalar_t>();
      if (id < size / sizeof(scalar_t))
        out_ptr[id] = 1 / (1 + Numerics<scalar_t>::exp(-static_cast<scalar_t>(in_ptr[id])));
    });
  });
}

Tensor & _sigmoid_out_sycl(Tensor & output, const Tensor & self) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "_sigmoid_out_sycl",
      [&] () {
        sigmoid_sycl<scalar_t>(output, self);
      }
  );
  return output;
}

}
}
