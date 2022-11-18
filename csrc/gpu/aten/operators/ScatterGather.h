#include <core/Memory.h>
#include <core/Stream.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"

#include "Loops.h"
#include "ReduceOpsUtils.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

enum class SCATTER_GATHER_OP : uint8_t { REDUCE_ADD, REDUCE_MULTIPLY };

SCATTER_GATHER_OP get_operator_enum(const c10::string_view reduce) {
  if (reduce == "add") {
    return SCATTER_GATHER_OP::REDUCE_ADD;
  } else if (reduce == "multiply") {
    return SCATTER_GATHER_OP::REDUCE_MULTIPLY;
  } else {
    TORCH_CHECK(false, "reduce argument must be either add or multiply.");
  }
}

template <typename T, typename ReduceStub, typename FillStub>
void scatter_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const T& src,
    const Tensor& out,
    ReduceStub& reduce_stub,
    FillStub& fill_stub,
    const c10::optional<c10::string_view> reduce = nullopt) {
  if (index.numel() == 0)
    return;
  dim = at::maybe_wrap_dim(dim, self.dim());
  auto mut_out = const_cast<Tensor&>(out);

  if (!self.is_same(mut_out)) {
    mut_out.copy_(self);
  }

  if (reduce.has_value()) {
    auto op = get_operator_enum(reduce.value());
    reduce_stub(mut_out, dim, index, src, op);
  } else {
    fill_stub(mut_out, dim, index, src);
  }
}

class ReduceMultiply {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicMul(
        (dpcpp_global_ptr_pt<scalar_t>)(self_data_start + index), *src_data);
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicMul((dpcpp_global_ptr_pt<scalar_t>)self_data, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicAdd(
        (dpcpp_global_ptr_pt<scalar_t>)(self_data_start + index), *src_data);
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicAdd((dpcpp_global_ptr_pt<scalar_t>)self_data, *src_data);
  }
};
static ReduceAdd reduce_add;

class TensorAssign {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    (void)numel; // suppress unused warning
    *(self_data_start + index) = *src_data;
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

// The kernels are implemented on an opaque,
// self-aligned type of the correct size,
// to avoid redundant kernels for different types
// of the same size.
template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename func_t>
static void _launch_scatter_gather_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);

  int outputSize = N;
  int work_group_size = outputSize > max_wg_size ? max_wg_size : outputSize;
  const auto target_global_size = dpcppMaxWorkItemsPerTile();
  // Each work group size is work_group_size, one full device launch is
  // target_global_size, so we can calculate max work group num as below
  const int max_work_group_num = target_global_size / work_group_size;
  int work_group_num = outputSize / work_group_size < max_work_group_num
      ? outputSize / work_group_size
      : max_work_group_num;
  int draft_work_group_num =
      (outputSize + work_group_size - 1) / work_group_size;

  int thread_work_size = draft_work_group_num / work_group_num + 1;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int nv = work_group_size * thread_work_size;
      auto wg_id = item.get_group_linear_id();
      auto local_id = item.get_local_linear_id();
      int idx = nv * wg_id + local_id;
      for (int i = 0; i < thread_work_size; ++i) {
        if (idx < N) {
          f(idx);
          idx += work_group_size;
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_size * work_group_num),
            sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
struct _dpcpp_scatter_fill_internal_kernel {
  template <typename func_t>
  void operator()(
      TensorIterator& iter,
      scalar_t src_val,
      int64_t index_size,
      int64_t index_stride,
      const func_t& f) {
    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        _dpcpp_scatter_fill_internal_kernel<scalar_t>()(
            sub_iter, src_val, index_size, index_stride, f);
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* index_ptr = (char*)iter.data_ptr(1);

    auto offset_calc = make_offset_calculator<2>(iter);
    auto loop = [=](int i) {
      auto offsets = offset_calc.get(i);

      int64_t idx_dim = *(int64_t*)(index_ptr + offsets[1]);

      char* self_data = self_ptr + offsets[0];

      f((scalar_t*)self_data + idx_dim * index_stride, (scalar_t*)&src_val);
    };

    _launch_scatter_gather_kernel(iter.numel(), loop);
  }
}; // struct _dpcpp_scatter_fill_internal_kernel

template <bool is_scatter_like, typename scalar_t>
struct _dpcpp_scatter_gather_internal_kernel {
  template <typename func_t>
  void operator()(
      TensorIterator& iter,
      int64_t index_size,
      int64_t index_stride,
      int64_t numel,
      const func_t& f) {
    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        _dpcpp_scatter_gather_internal_kernel<is_scatter_like, scalar_t>()(
            sub_iter, index_size, index_stride, numel, f);
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    auto offset_calc = make_offset_calculator<3>(iter);
    auto loop = [=] C10_DEVICE(int i) {
      auto offsets = offset_calc.get(i);

      int64_t idx_dim = *(int64_t*)(index_ptr + offsets[2]);
      SYCL_KERNEL_ASSERT(
          idx_dim >= 0 && idx_dim < index_size && "index out of bounds");

      f((scalar_t*)(self_ptr + offsets[0]),
        is_scatter_like ? idx_dim * index_stride : 0,
        numel,
        (scalar_t*)(src_ptr + offsets[1]) +
            (is_scatter_like ? 0 : idx_dim * index_stride));
    };

    _launch_scatter_gather_kernel(iter.numel(), loop);
  }
}; // struct _dpcpp_scatter_fill_internal_kernel

template <bool cast_to_opaque = true>
struct dpcpp_scatter_fill_base_kernel {
  template <typename func_t>
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      Scalar src,
      const std::string& method_name,
      const func_t& f) {
    at::assert_no_internal_overlap(self);
    dim = (dim < 0) ? (self.ndimension() + dim) : dim;

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(index)
                    .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_fill_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          auto src_scalar_val = src.to<scalar_t>();
          auto src_val = *(dtype*)&src_scalar_val;

          _dpcpp_scatter_fill_internal_kernel<dtype>()(
              iter, src_val, index_size, index_stride, f);
        });
  }

  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      Scalar src,
      const std::string& method_name,
      const ReduceMultiply& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(index)
                    .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_fill_base_kernel_reduce_multiply",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          auto src_scalar_val = src.to<scalar_t>();
          auto src_val = *(dtype*)&src_scalar_val;

          _dpcpp_scatter_fill_internal_kernel<dtype>()(
              iter, src_val, index_size, index_stride, f);
        });
  }
}; // struct dpcpp_scatter_fill_base_kernel

template <bool is_scatter_like = true, bool cast_to_opaque = true>
struct dpcpp_scatter_gather_base_kernel {
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const ReduceAdd& f) {
    at::assert_no_internal_overlap(self);

    dim = (dim < 0) ? (src.ndimension() + dim) : dim;
    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(src_restrided)
                    .add_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_gather_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          _dpcpp_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, self.numel(), f);
        });
  }

  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const TensorAssign& f) {
    at::assert_no_internal_overlap(self);

    dim = (dim < 0) ? (src.ndimension() + dim) : dim;
    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(src_restrided)
                    .add_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_gather_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          _dpcpp_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, self.numel(), f);
        });
  }

  template <typename func_t>
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const func_t& f) {
    at::assert_no_internal_overlap(self);

    dim = (dim < 0) ? (src.ndimension() + dim) : dim;
    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(src_restrided)
                    .add_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_gather_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          _dpcpp_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, self.numel(), f);
        });
  }

  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const ReduceMultiply& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(src_restrided)
                    .add_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_gather_base_kernel_reduce_multiply",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          _dpcpp_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, self.numel(), f);
        });
  }
};

} // namespace AtenIpexTypeXPU
} // namespace at
