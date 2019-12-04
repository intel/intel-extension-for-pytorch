#include <c10/dpcpp/SYCL.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Indexing.h>

#include <functions/Loops.h>
#include <utils/Atomics.h>


DP_DEF_K1(index_kernel);
DP_DEF_K1(index_put_kernel);

namespace at { namespace native {

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <int N> struct alignas(N) OpaqueType { char data[N]; };

static void index_kernel_sycl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "index", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    sycl_index_kernel<DP_K(index_kernel, scalar_t)>(iter, index_size, index_stride,
            // This lambda function only works in sycl kernel.
                                                    [](dp_global_ptr_pt<char> out_data, dp_global_ptr_pt<char> in_data, int64_t offset) {
                                                      *(dtype*)out_data = *(dtype*)(in_data + offset);
                                                    }
    );
  });
}



#define AT_DISPATCH_ALL_ATOMIC_TYPES(TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

static void index_put_kernel_sycl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  if (accumulate) {
    AT_DISPATCH_ALL_ATOMIC_TYPES(iter.dtype(), "index_put", [&] {
      sycl_index_kernel<DP_K(index_put_kernel, scalar_t, /*accumulate=*/bool)>(iter, index_size, index_stride,
              // This lambda function only works in sycl kernel.
                                                          [](dp_global_ptr_pt<char> out_data, dp_global_ptr_pt<char> in_data, int64_t offset) {
                                                            dp_global_ptr_pt<scalar_t> out_ptr = (dp_global_ptr_pt<scalar_t>) (out_data + offset);
                                                            auto in = *(scalar_t*)in_data;
                                                            atomicAdd(out_ptr, in);
                                                          }
      );
    });
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(), "index_put", [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      sycl_index_kernel<DP_K(index_put_kernel, scalar_t)>(iter, index_size, index_stride,
              // This lambda function only works in sycl kernel.
                                                          [](dp_global_ptr_pt<char> out_data, dp_global_ptr_pt<char> in_data, int64_t offset) {
                                                            *(dtype*)(out_data + offset) = *(dtype*)in_data;
                                                          }
      );
    });
  }
}

}} // namespace at::native
