#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>

#include <aten_ipex_tensor_type.h>

#include <core/Context.h>
#include <core/TensorImplUtils.h>
#include <utils/Numerics.h>
#include <ATen/aten_ipex_type_dpcpp.h>


using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

Tensor empty_strided_dpcpp(IntArrayRef size, IntArrayRef stride, const TensorOptions& options) {
  check_size_nonnegative(size);
  auto t = at::AtenIpexTypeDPCPP::empty({0}, options, c10::nullopt);
  TensorImpl_resizeImpl(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

Tensor& eye_out_dpcpp(Tensor& result, int64_t n, int64_t m) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);

  if(m < 0) {
    m = n;
  }

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

Tensor& eye_out_dpcpp(Tensor& result, int64_t n) {
  return eye_out_dpcpp(result, n, /*m=*/-1);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace triangle_dpcpp {
// To find the max integer that does not exceed the root of an int64_t variable,
// we could use a loop to test one bit at a time, which takes up to 31
// iterations. This would give the accurate result, but is relatively slow and
// is an overkill for most cases where double's precision suffice.
//
// If we directly use sqrt to calculate the root, the convertion from int64_t
// to double would lose 11 bits precision.
//
// The following solution uses sqrt directly for most cases, and would only
// special handle it if there is indeed precision loss.
DP_DEVICE
inline int64_t resolve_root_int(int64_t b, int64_t cX4, int64_t x, int32_t sign) {

  int64_t bXb_cX4 = b*b - cX4;
  // potential precision loss could occur here when casting int64_t (63 bits
  // precision) to double (52 bits precision)
  double sr = DP::sqrt((double)bXb_cX4);
  //
  // TODO: CUDA uses ::__double2ll_rd. No corresponding API in DPCPP.
  // uses std::llround or std::ceil or std::float will cause error:
  // terminate called after throwing an instance of 'DP::compile_program_error'.
  //
  int64_t res = static_cast<int64_t>((-b + sign * sr)/2);

  // have to cast double to int64_t, otherwise it would only compare up to the
  // precision of a double variable, ignoring the precision loss
  if (bXb_cX4 != (int64_t) (sr * sr)) {
    // TODO:CUDA uses ::__double2ll_rd && ::__double2ll_ru. No corresponding API in DPCPP.
  }

  return res;
}

DP_DEVICE
inline void get_coordinate_in_triu_trapezoid(
    int64_t f, int64_t x, int64_t & row, int64_t & col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = -1 - f;
  auto cX4 = x << 3; // 4 * c = 4 * (2x) = 8x;
  row = resolve_root_int(b, cX4, x, -1);
  col = x - ((f - row + 1) * row >> 1) + row;
}

DP_DEVICE
inline void get_coordinate_in_tril_trapezoid(
    int64_t f, int64_t x, int64_t & row, int64_t & col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = f - 1;
  auto cX4 = - (x << 3); // 4 * c = 4 * (-2x) = -8x;
  row = resolve_root_int(b, cX4, x, 1);
  col = x - ((f + row - 1) * row >> 1);
}

} // end of namespace

DP_DEF_K1(triuIndicesSycl);
DP_DEF_K1(trilIndicesSycl);

template <typename scalar_t>
void
triu_indices_dpcpp_kernel(scalar_t * tensor,
                         int64_t col_offset,
                         int64_t m_first_row,
                         int64_t col,
                         int64_t rectangle_size,
                         int64_t triu_size) {

  auto queue         = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(queue);
  auto totalElements = triu_size;
  auto num_groups    = CeilDiv(totalElements, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto acc = DPCPPAccessor<dp_w_mode>(cgh, tensor);

    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto tensor_ptr = acc.template get_pointer<scalar_t>();
      int64_t r, c;
      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
        if (linearIndex < rectangle_size) {
          // the coordinate is within the top rectangle
          r = linearIndex / col;
          c = linearIndex % col;
        } else {
          // the coordinate falls in the bottom trapezoid
          triangle_dpcpp::get_coordinate_in_triu_trapezoid(
            m_first_row, linearIndex - rectangle_size, r, c);
          r += rectangle_size / col;
        }
        c += col_offset;
        tensor_ptr[linearIndex] = r;
        tensor_ptr[linearIndex + triu_size] = c;
      }
    };
    // kick off kernel
    cgh.parallel_for<DP_K(triuIndicesSycl, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void
tril_indices_dpcpp_kernel(scalar_t * tensor,
                         int64_t row_offset,
                         int64_t m_first_row,
                         int64_t col,
                         int64_t trapezoid_size,
                         int64_t tril_size) {

  auto queue         = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(queue);
  auto totalElements = tril_size;
  auto num_groups    = CeilDiv(totalElements, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto acc = DPCPPAccessor<dp_w_mode>(cgh, tensor);

    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto tensor_ptr = acc.template get_pointer<scalar_t>();
      int64_t r, c;
      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
        if (linearIndex < trapezoid_size) {
          // the coordinate is within the top trapezoid
          triangle_dpcpp::get_coordinate_in_tril_trapezoid(m_first_row, linearIndex, r, c);
        } else {
          // the coordinate falls in the bottom rectangle
          auto surplus = linearIndex - trapezoid_size;
          // add the height of trapezoid: m_last_row (col) - m_first_row + 1
          r = surplus / col + col - m_first_row + 1;
          c = surplus % col;
        }
        r += row_offset;
        tensor_ptr[linearIndex] = r;
        tensor_ptr[linearIndex + tril_size] = c;
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(trilIndicesSycl, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

Tensor triu_indices_dpcpp(int64_t row, int64_t col, int64_t offset, const TensorOptions& options) {

  check_args(row, col, options);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);
  auto tensor = at::AtenIpexTypeDPCPP::empty({2, triu_size}, options, c10::nullopt);

  if (triu_size > 0) {
    // # of triu elements in the first row
    auto m_first_row = (offset > 0) ?
      std::max<int64_t>(col - offset, 0) : // upper bounded by col
      col;

    // size of the top rectangle
    int64_t rectangle_size = 0;
    if (offset < 0) {
      rectangle_size = std::min<int64_t>(row, -offset) * col;
    }

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, tensor.scalar_type(), "triu_indices_cuda", [&] {
      triu_indices_dpcpp_kernel<scalar_t>(
        tensor.data_ptr<scalar_t>(),
        std::max<int64_t>(0, offset),
        m_first_row,
        col,
        rectangle_size,
        triu_size);
    });
  }

  return tensor;
}

Tensor tril_indices_dpcpp(int64_t row, int64_t col, int64_t offset, const TensorOptions& options) {
  check_args(row, col, options);

  auto tril_size = get_tril_size(row, col, offset);
  auto tensor = at::AtenIpexTypeDPCPP::empty({2, tril_size}, options, c10::nullopt);

  if (tril_size > 0) {
    auto m_first_row = (offset > 0) ?
      std::min<int64_t>(col, 1 + offset) : // upper bounded by col
      (row + offset > 0); // either 0 or 1
    auto trapezoid_row_offset = std::max<int64_t>(0, -offset);
    auto rectangle_row_offset = trapezoid_row_offset + col - m_first_row + 1;

    int64_t rectangle_size = 0;
    if (rectangle_row_offset < row) {
      rectangle_size = (row - rectangle_row_offset) * col;
    }

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, tensor.scalar_type(), "tril_indices_dpcpp", [&] {
      tril_indices_dpcpp_kernel<scalar_t>(
        tensor.data_ptr<scalar_t>(),
        trapezoid_row_offset,
        m_first_row,
        col,
        tril_size - rectangle_size,
        tril_size);
    });
  }

  return tensor;
}

} // namespace impl

Tensor empty(IntArrayRef size, const TensorOptions& options, c10::optional<MemoryFormat> optional_memory_format) {
  AT_ASSERT(options.backend() == at::Backend::DPCPP);
  // AT_ASSERT(!options.is_variable()); // is_variable should have been "unpacked"

  auto* allocator = at::dpcpp::getDPCPPDeviceAllocator();
  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  auto storage_impl =  c10::make_intrusive<StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(storage_impl, at::torch_ipex::DPCPPTensorId());
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options){
  return impl::empty_strided_dpcpp(size, stride, options);
}
Tensor & eye_out(Tensor & out, int64_t n){
  impl::eye_out_dpcpp(out, n);
  return out;
}
Tensor & eye_out(Tensor & out, int64_t n, int64_t m){
  impl::eye_out_dpcpp(out, n, m);
  return out;
}
Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options){
  return impl::tril_indices_dpcpp(row, col, offset, options);
}
Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options){
  return impl::triu_indices_dpcpp(row, col, offset, options);
}
} // namespace AtenIpexTypeDPCPP
} // namespace at
