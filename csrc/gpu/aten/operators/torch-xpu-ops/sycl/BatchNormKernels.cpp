#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ops/from_blob.h>
#include <ATen/xpu/XPUContext.h>
#include "aten/operators/torch-xpu-ops/comm/SYCLContext.h"
#include "aten/operators/torch-xpu-ops/comm/XPUMathCompat.h"
#include "aten/operators/torch-xpu-ops/comm/xpu_aten.h"
#include "aten/operators/torch-xpu-ops/sycl/Loops.h"
#include "aten/operators/torch-xpu-ops/sycl/Reduce.h"
#include "aten/operators/torch-xpu-ops/sycl/ResizeKernel.h"

#include "aten/operators/ReduceOpStdVar.h"

#include "aten/operators/torch-xpu-ops/sycl/BatchNormKernels.h"

namespace at {
namespace native {
namespace xpu {

#define SIMD32 32
#define SIMD16 16

// ========================== batch_norm utils ==========================

ScalarType first_type() {
  return ScalarType::Undefined;
}

template <typename... Args>
ScalarType first_type(const Tensor& arg, const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}

// A transform is mixed type if the parameters are higher precision than the
// input
template <typename... Args>
bool is_mixed_type(const Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  return (
      (parameter_type != ScalarType::Undefined) &&
      (parameter_type != input.scalar_type()));
}

inline bool batch_norm_use_channels_last_kernels(const at::Tensor& self) {
  return (
      self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      self.is_contiguous(at::MemoryFormat::ChannelsLast3d) ||
      (self.is_contiguous() && self.strides()[1] == 1));
}

enum class Impl {
  Contiguous,
  ChannelsLast,
  General,
};

inline Impl batch_norm_choose_impl(const Tensor& self) {
  if (!canUse32BitIndexMath(self)) {
    return Impl::General;
  }

  if (self.is_contiguous()) {
    return self.strides()[1] == 1 ? Impl::ChannelsLast : Impl::Contiguous;
  }

  if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return Impl::ChannelsLast;
  }

  return Impl::General;
}

inline Impl batch_norm_choose_impl(const Tensor& in1, const Tensor& in2) {
  auto imp1 = batch_norm_choose_impl(in1);
  if (imp1 == Impl::General) {
    return imp1;
  }
  auto imp2 = batch_norm_choose_impl(in2);
  return imp1 == imp2 ? imp1 : Impl::General;
}

template <
    typename scalar_t,
    int64_t dim,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>
get_packed_accessor(const Tensor& t, c10::string_view var_name) {
  constexpr auto expect_type = c10::CppTypeToScalarType<
      typename std::remove_const<scalar_t>::type>::value;
  const auto actual_type = t.scalar_type();
  TORCH_CHECK(
      actual_type == expect_type,
      "Expected ",
      var_name,
      " to have type ",
      expect_type,
      " but got ",
      actual_type);
  return t.generic_packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

template <
    typename scalar_t,
    int64_t dim,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>
packed_accessor_or_dummy(const Tensor& t, c10::string_view var_name) {
  if (!t.defined()) {
    const std::array<index_t, dim> zeros{{0}};
    return GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(
        nullptr, zeros.data(), zeros.data());
  }
  return get_packed_accessor<scalar_t, dim, PtrTraits, index_t>(t, var_name);
}

struct InvStd {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    T invstd = 0.0f;
    if (var != static_cast<T>(0.0f) || epsilon != static_cast<T>(0.0f)) {
      invstd = static_cast<T>(1.0f) / std::sqrt(var + static_cast<T>(epsilon));
    }
    return invstd;
  }
};

struct Var {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    return var;
  }
};

template <class KernelClass>
int get_max_group_size(int simd = SIMD32) {
  // The max work group size required by batch_norm needs to ensure that the two
  // subgroup reduces can obtain correct results.
  int max_size = syclMaxWorkGroupSize<KernelClass>();
  int shfl2_restricted_size = simd * simd;
  return max_size > shfl2_restricted_size ? shfl2_restricted_size : max_size;
}

template <class KernelClass>
int get_num_threads(int nelem, int restricted_simd = SIMD32) {
  int max_size = get_max_group_size<KernelClass>(restricted_simd);
  int thread_sizes[5] = {32, 64, 128, 256, max_size};
  for (int i = 0; i < 5; ++i) {
    if (nelem <= thread_sizes[i]) {
      return thread_sizes[i];
    }
  }
  return max_size;
}

int get_dev_max_group_size(int simd = SIMD32) {
  // The max work group size required by batch_norm needs to ensure that the two
  // subgroup reduces can obtain correct results.
  int max_size = syclDeviceMaxWorkGroupSize();
  int shfl2_restricted_size = simd * simd;
  return max_size > shfl2_restricted_size ? shfl2_restricted_size : max_size;
}

int get_num_threads_by_dev_max_group_size(
    int nelem,
    int restricted_simd = SIMD32) {
  int max_size = get_dev_max_group_size(restricted_simd);
  int thread_sizes[5] = {32, 64, 128, 256, max_size};
  for (int i = 0; i < 5; ++i) {
    if (nelem <= thread_sizes[i]) {
      return thread_sizes[i];
    }
  }
  return max_size;
}

int get_prefer_simd(int numPlane, int nHw) {
  // decide SIMD: SIMD32 or SIMD16

  auto dev_id = at::xpu::getDeviceIndexOfCurrentQueue();

  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  auto sub_group_size = dev_prop->sub_group_sizes;
  int simd = sub_group_size[1];
  if (simd <= SIMD16)
    return simd;

  // if max supported simd >16
  if (nHw <= SIMD16)
    return SIMD16;
  if (simd >= SIMD32 && nHw <= SIMD32)
    return SIMD32;

  int64_t target_tile_size = syclMaxWorkItemsPerTile(dev_id);
  // for work group barrier perf
  int64_t wg_size = syclMaxWorkItemsPerEU(dev_id);
  if (simd == SIMD32) {
    // when setting wg_size 256 can achieve high occupancy, use SIMD16
    if (wg_size * numPlane >= target_tile_size)
      return SIMD16;
    // for latency case
    if (nHw <= 1024 && numPlane > 128 && SIMD16 * SIMD16 >= wg_size) {
      return SIMD16;
    }
  }
  return simd;
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  Float2() {}

  Float2(scalar_t v1, scalar_t v2)
      : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  Float2(int v)
      : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }

  friend Float2 operator+(Float2 a, const Float2& b) {
    a += b;
    return a;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  GradOp(accscalar_t m, const PTA& i, const PTA& g)
      : mean(m), input(i), grad_output(g) {}
  Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) const {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

template <
    int SIMD,
    typename accscalar_t,
    typename reduce_op,
    typename item_t,
    typename local_shared_t>
static inline void group_reduce(
    item_t item,
    int sub_group_num,
    accscalar_t& val,
    accscalar_t init,
    const local_shared_t& local_data,
    reduce_op bin_op) {
  auto sg = item.get_sub_group();
  uint32_t lane_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();

  // dynamic get SIMD width result in big performance drop
  // uint32_t SIMD = sg.get_local_range()[0];
#pragma unroll
  for (int i = 1; i < SIMD; i <<= 1) {
    val = bin_op(
        val, static_cast<accscalar_t>(sycl::shift_group_left(sg, val, i)));
  }
  if (sub_group_num == 1) {
    if (lane_id == 0) {
      local_data[0] = val;
    }
    item.barrier(sycl_local_fence);
    val = local_data[0];

    return;
  }

  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
  if (lane_id == 0) {
    local_data[sg_id] = val;
  }
  item.barrier(sycl_local_fence);

  // use one subgroup to reduce WGroupSize/subGroupSize elements
  // into the final result
  if (sg_id == 0) {
    val = init;
    if (lane_id < sub_group_num) {
      val = accscalar_t(local_data[lane_id]);
    }
    for (int i = lane_id + SIMD; i < sub_group_num; i += SIMD) {
      val = bin_op(val, static_cast<accscalar_t>(local_data[i]));
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      val = bin_op(
          val, static_cast<accscalar_t>(sycl::shift_group_left(sg, val, i)));
      if (i >= ((sub_group_num + 1) >> 1))
        break;
    }

    // the 0th WI (the 0th WI in the 0th sub_group) generate the final
    // result
    if (lane_id == 0) {
      local_data[0] = val;
    }
  }

  item.barrier(sycl_local_fence);
  val = local_data[0];
}

template <
    int SIMD,
    typename scalar_t,
    typename item_t,
    typename Op,
    typename PTA,
    typename local_shared_t>
scalar_t plane_reduce(
    item_t item,
    Op grad_op,
    PTA tensor,
    int plane,
    int sub_group_num,
    const local_shared_t& shared) {
  // first the reductions each thread does separately
  scalar_t sum_value = 0;
  for (int batch = item.get_local_id(0); batch < tensor.size(0);
       batch += item.get_local_range(0)) {
    for (int x = item.get_local_id(1); x < tensor.size(2);
         x += item.get_local_range(1)) {
      auto res = grad_op(batch, plane, x);
      sum_value += res;
    }
  }
  group_reduce<SIMD, scalar_t>(
      item,
      sub_group_num,
      sum_value,
      scalar_t(0),
      shared,
      [](scalar_t a, scalar_t b) { return a + b; });
  if (item.get_local_linear_id() == 0) {
    shared[0] = sum_value;
  }
  item.barrier(sycl_local_fence);
  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

constexpr int ELEMENTS_PER_ITER =
    4; // enables concurrency within each thread to hide latency
constexpr int ELEMENTS_PER_WORK_ITEM = 16;

std::tuple<sycl::range<2>, sycl::range<2>> get_adaptive_launch_config(
    const int reduction,
    const int stride,
    const bool coop_flag = false,
    const int loops_per_item = 1) {
  int max_wg_size = syclMaxWorkItemsPerEU();
  int group_x = std::min(last_pow2(stride), 32);
  int group_y = std::min(
      last_pow2(div_up(reduction, loops_per_item)), max_wg_size / group_x);
  if (group_x * group_y != max_wg_size) {
    group_x = std::min(last_pow2(stride), max_wg_size / group_y);
  }

  int nwg_x = div_up(stride, group_x);
  int nwg_y = std::min(
      div_up(reduction, group_y * loops_per_item),
      int(syclMaxWorkItemsPerTile()) / (nwg_x * group_x) / (group_y));
  nwg_y = std::max(nwg_y, 1);

  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not
    // big enough
    nwg_y = nwg_y < 8 ? 1 : nwg_y;
  }

  sycl::range<2> local_range(group_y, group_x);
  sycl::range<2> global_range(nwg_y * group_y, nwg_x * group_x);

  return std::make_tuple(global_range, local_range);
}

template <typename T, typename C>
inline void welford_merge_element(
    C& count,
    T& mean,
    T& m2n,
    const C& count_new,
    const T& mean_new,
    const T& m2n_new) {
  T factor = T(1.0) / std::max(1, (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

// ========================== batch_norm_stats ==========================

template <
    int SIMD,
    typename VarTransform,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
struct BatchNormCollectStatisticsKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    int plane = item.get_group(1);
    int tid = item.get_local_linear_id();

    auto sg = item.get_sub_group();
    auto sg_lid = sg.get_local_linear_id();
    auto sg_id = sg.get_group_linear_id();

    // Compute the mean and variance across (batch, x/y/z)
    // this uses the Welford (in the for loop)/parallel algorithm (to sum
    // across the group)
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    // and the parallel algorithm on the same page.
    // We use two shuffles to reduce across the entire group.

    // first the reductions each thread does separately
    stat_accscalar_t avg = 0;
    stat_accscalar_t var_n = 0;
    int n = 0;
    for (int batch = item.get_local_id(0); batch < input_.size(0);
         batch += item.get_local_range(0)) {
      for (int x = item.get_local_id(1); x < input_.size(2);
           x += item.get_local_range(1)) {
        stat_accscalar_t v = input_[batch][plane][x];
        stat_accscalar_t d1 = v - avg;
        n++;
        avg += d1 / n;
        var_n += d1 * (v - avg);
      }
    }

    // first subgroupSum to get one value per thread to
    // one value per subgroup
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      stat_accscalar_t o_avg = sycl::permute_group_by_xor(sg, avg, i);
      int o_n = sycl::permute_group_by_xor(sg, n, i);
      stat_accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
      var_n += sycl::permute_group_by_xor(sg, var_n, i) +
          (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
      avg = (n * avg + o_n * o_avg) * factor;
      n += o_n;
    }

    // this writes each subgroups item into shared memory
    if (sg_lid == 0) {
      shared_n_[sg_id] = n;
      shared_avg_var_[sg_id * 2] = avg;
      shared_avg_var_[sg_id * 2 + 1] = var_n;
    }
    item.barrier(sycl_local_fence);
    // now have a second subgroupSum to reduce the intermediate values
    // from shared memory to a single number. The very first
    // thread writes it to shared memory.
    int num_sg = item.get_local_range(1) * item.get_local_range(0) / SIMD;
    if (tid < num_sg) {
      n = shared_n_[tid];
      avg = shared_avg_var_[2 * tid];
      var_n = shared_avg_var_[2 * tid + 1];
    } else {
      n = 0;
      avg = stat_accscalar_t(0);
      var_n = stat_accscalar_t(0);
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      stat_accscalar_t o_avg = sycl::permute_group_by_xor(sg, avg, i);
      int o_n = sycl::permute_group_by_xor(sg, n, i);
      stat_accscalar_t factor = 1.0f / fmaxf(1.0f, n + o_n);
      var_n += sycl::permute_group_by_xor(sg, var_n, i) +
          (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
      avg = (n * avg + o_n * o_avg) * factor;
      n += o_n;
    }

    // Save the mean, variance, and moving averages
    auto save_mean = save_mean_;
    auto save_transformed_var = save_transformed_var_;
    if (tid == 0) {
      if (save_mean_.data() != NULL) {
        save_mean[plane] = avg;
      }
      if (save_transformed_var_.data() != NULL) {
        save_transformed_var[plane] =
            VarTransform{}(var_n / (input_.size(0) * input_.size(2)), epsilon_);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_n_ = sycl_local_acc_t<int, 1>(sycl::range<1>{(size_t)SIMD}, cgh);
    shared_avg_var_ = sycl_local_acc_t<stat_accscalar_t, 1>(
        sycl::range<1>{(size_t)SIMD * 2 * 2}, cgh);
  }

  BatchNormCollectStatisticsKernelFunctor(
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          RestrictPtrTraits,
          index_t> input,
      const stat_accscalar_t epsilon,
      const stat_accscalar_t momentum,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          RestrictPtrTraits,
          index_t> save_mean,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          RestrictPtrTraits,
          index_t> save_transformed_var)
      : input_(input),
        epsilon_(epsilon),
        momentum_(momentum),
        save_mean_(save_mean),
        save_transformed_var_(save_transformed_var) {}

 private:
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      RestrictPtrTraits,
      index_t>
      input_;
  const stat_accscalar_t epsilon_;
  const stat_accscalar_t momentum_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>
      save_mean_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>
      save_transformed_var_;
  sycl_local_acc_t<int, 1> shared_n_;
  sycl_local_acc_t<stat_accscalar_t, 1> shared_avg_var_;
};

template <typename scalar_t, typename index_t, typename VarTransform>
void batch_norm_stats_template(
    const Tensor& out_mean,
    const Tensor& out_invstd,
    const Tensor& input_,
    double epsilon) {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions

  at::native::resize_output(out_mean, {n_input});
  at::native::resize_output(out_invstd, {n_input});

  auto input =
      get_packed_accessor<const scalar_t, 3, RestrictPtrTraits, index_t>(
          input_reshaped, "input");

  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  auto mean =
      packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(
          out_mean, "out_mean");
  auto invstd =
      packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(
          out_invstd, "out_invstd");

  auto& queue = getCurrentSYCLQueue();
  int simd = get_prefer_simd(input.size(1), input.size(0) * input.size(2));

  if (simd == SIMD32) {
    using KernelClass = BatchNormCollectStatisticsKernelFunctor<
        SIMD32,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>;

    auto kfn = KernelClass(input, epsilon, 0.0, mean, invstd);

    int max_group_size = get_max_group_size<KernelClass>(simd);
    int tf = get_num_threads<KernelClass>(input.size(2), simd);
    int64_t work_group_size_x = tf;
    int64_t work_group_size_y = std::max(1, max_group_size / tf);
    int64_t global_size_x = input.size(1) * work_group_size_x;
    int64_t global_size_y = 1 * work_group_size_y;

    sycl_kernel_submit(
        sycl::range<2>(global_size_y, global_size_x),
        sycl::range<2>(work_group_size_y, work_group_size_x),
        queue,
        kfn);
  } else {
    using KernelClass = BatchNormCollectStatisticsKernelFunctor<
        SIMD16,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>;

    auto kfn = KernelClass(input, epsilon, 0.0, mean, invstd);

    int max_group_size = get_max_group_size<KernelClass>(simd);
    int tf = get_num_threads<KernelClass>(input.size(2), simd);
    int64_t work_group_size_x = tf;
    int64_t work_group_size_y = std::max(1, max_group_size / tf);
    int64_t global_size_x = input.size(1) * work_group_size_x;
    int64_t global_size_y = 1 * work_group_size_y;

    sycl_kernel_submit(
        sycl::range<2>(global_size_y, global_size_x),
        sycl::range<2>(work_group_size_y, work_group_size_x),
        queue,
        kfn);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    typename vec_y,
    int vec_size,
    bool two_pass_reduce>
struct BatchNormReduceSumChannelsLastKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    // int plane = item.get_group(0);
    // int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset_base = item.get_global_id(1) * vec_size;

    int thread_idx_y = item.get_local_id(0);
    // int thread_idx_x = item.get_local_id(1);
    int group_idx_y = item.get_group(0);
    // int group_idx_x = item.get_group(1);

    int address_base = m_offset * stride_ + c_offset_base;
    int inner_loop_stride = global_range_y_;
    int address_increment = inner_loop_stride * stride_;

    accscalar_t x_sum[vec_size] = {0.0f};
    accscalar_t x_sq_sum[vec_size] = {0.0f};
    // thread reduction
    for (int i = 0; i < loop_count_; i++) {
      vec_t x_math_vec = *(reinterpret_cast<vec_t*>(input_ptr_ + address_base));
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        auto c_offset = c_offset_base + j;

        if (c_offset < stride_ && m_offset < reduction_size_) {
          // scalar_t arr = input_ptr_[address_base + j];
          auto x_math = x_math_vec[j];
          x_sum[j] += x_math;
          x_sq_sum[j] += x_math * x_math;
        }
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      vec_y value;
      value[0] = x_sum[j];
      value[1] = x_sq_sum[j];

      value = group_y_reduce(
          item, shared_, value, [](accscalar_t a, accscalar_t b) {
            return a + b;
          });

      x_sum[j] = value[0];
      x_sq_sum[j] = value[1];

      item.barrier(sycl_local_fence);
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      auto c_offset = c_offset_base + j;
      // global_reduciton
      if (thread_idx_y == 0 && c_offset < stride_) {
        if constexpr (two_pass_reduce) {
          // write to temp[c][group_idx_y]
          // int offset = c_offset * group_num_y_ + group_idx_y;
          temp_sum_ptr_[c_offset * group_num_y_ + group_idx_y] = x_sum[j];
          temp_sum_sq_ptr_[c_offset * group_num_y_ + group_idx_y] = x_sq_sum[j];
        } else {
          out_mean_ptr_[c_offset] = x_sum[j];
          out_invstd_ptr_[c_offset] = x_sq_sum[j];
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<vec_y, 1>(sycl::range<1>{(size_t)wg_size_}, cgh);
  }

  BatchNormReduceSumChannelsLastKernelFunctor(
      const int reduction_size,
      const int stride,
      int global_range_y,
      int local_range_y,
      int group_num_x,
      int group_num_y,
      accscalar_t* temp_sum_ptr,
      accscalar_t* temp_sum_sq_ptr,
      int wg_size,
      scalar_t* input_ptr,
      accscalar_t* out_mean_ptr,
      accscalar_t* out_invstd_ptr,
      int loop_count)
      : reduction_size_(reduction_size),
        stride_(stride),
        global_range_y_(global_range_y),
        local_range_y_(local_range_y),
        group_num_x_(group_num_x),
        group_num_y_(group_num_y),
        temp_sum_ptr_(temp_sum_ptr),
        temp_sum_sq_ptr_(temp_sum_sq_ptr),
        wg_size_(wg_size),
        input_ptr_(input_ptr),
        out_mean_ptr_(out_mean_ptr),
        out_invstd_ptr_(out_invstd_ptr),
        loop_count_(loop_count) {}

 private:
  const int reduction_size_;
  const int stride_;
  int global_range_y_;
  int local_range_y_;
  int group_num_x_;
  int group_num_y_;
  accscalar_t* temp_sum_ptr_;
  accscalar_t* temp_sum_sq_ptr_;
  int wg_size_;
  scalar_t* input_ptr_;
  accscalar_t* out_mean_ptr_;
  accscalar_t* out_invstd_ptr_;
  int loop_count_;
  sycl_local_acc_t<vec_y, 1> shared_;
};

template <typename accscalar_t>
struct BatchNormReduceSumChannelsLastTwoPassKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto local_id = item.get_local_linear_id();
    // auto global_id = item.get_global_linear_id();
    auto c_offset = item.get_group_linear_id();

    accscalar_t temp_sum_val = 0.0f;
    accscalar_t temp_sum_sq_val = 0.0f;
    for (int i = local_id; i < group_num_y_; i += wg_size_) {
      int offset = c_offset * group_num_y_ + i;
      temp_sum_val += temp_sum_ptr_[offset];
      temp_sum_sq_val += temp_sum_sq_ptr_[offset];
    }
    auto total_sum = sycl::reduce_over_group(
        item.get_group(), temp_sum_val, sycl::plus<accscalar_t>());
    auto total_sum_sq = sycl::reduce_over_group(
        item.get_group(), temp_sum_sq_val, sycl::plus<accscalar_t>());
    if (local_id == 0) {
      out_mean_ptr_[c_offset] = total_sum;
      out_invstd_ptr_[c_offset] = total_sum_sq;
    }
  }
  BatchNormReduceSumChannelsLastTwoPassKernelFunctor(
      int group_num_y,
      accscalar_t* temp_sum_ptr,
      accscalar_t* temp_sum_sq_ptr,
      int wg_size,
      accscalar_t* out_mean_ptr,
      accscalar_t* out_invstd_ptr)
      : group_num_y_(group_num_y),
        temp_sum_ptr_(temp_sum_ptr),
        temp_sum_sq_ptr_(temp_sum_sq_ptr),
        wg_size_(wg_size),
        out_mean_ptr_(out_mean_ptr),
        out_invstd_ptr_(out_invstd_ptr) {}

 private:
  int group_num_y_;
  accscalar_t* temp_sum_ptr_;
  accscalar_t* temp_sum_sq_ptr_;
  int wg_size_;
  accscalar_t* out_mean_ptr_;
  accscalar_t* out_invstd_ptr_;
};

template <typename T, typename C, typename TACC, typename CACC, typename item_t>
inline void welford_merge_group_vertical(
    item_t item,
    C& count,
    T& mean,
    T& m2n,
    CACC& shmem_count,
    TACC& shmem_mean,
    TACC& shmem_m2n) {
  // write to shared memory
  auto address_base = item.get_local_linear_id();

#pragma unroll
  for (int offset = item.get_local_range(0) / 2; offset > 0; offset >>= 1) {
    if (item.get_local_id(0) < offset * 2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    item.barrier(sycl_local_fence);
    if (item.get_local_id(0) < offset &&
        item.get_local_id(0) + offset < item.get_local_range(0)) {
      auto address = address_base + offset * item.get_local_range(1);
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, count_new, mean_new, m2n_new);
    }
  }
}

template <
    typename VarTransform,
    typename scalar_t,
    typename accscalar_t,
    int PARALLEL_LOADS>
struct BatchNormCollectStatisticsChannelsLastKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    accscalar_t x_mean[PARALLEL_LOADS];
    accscalar_t m_2_n[PARALLEL_LOADS];
    int count[PARALLEL_LOADS];

#pragma unroll
    for (int i = 0; i < PARALLEL_LOADS; i++) {
      x_mean[i] = accscalar_t(0);
      m_2_n[i] = accscalar_t(0);
      count[i] = accscalar_t(0);
    }

    // loop along m dimension
    int inner_loop_stride = item.get_local_range(0) * item.get_group_range(0);

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset = item.get_global_id(1);

    int loop_count =
        1 + (reduction_size_ - 1) / (inner_loop_stride * PARALLEL_LOADS);
    int address_base = m_offset * stride_ + c_offset;
    int address_increment = inner_loop_stride * stride_;

    for (int i = 0; i < loop_count; i++) {
      accscalar_t x_math[PARALLEL_LOADS];
      accscalar_t x_count_inv[PARALLEL_LOADS];
      accscalar_t is_valid[PARALLEL_LOADS];

      // load multiple data in
#pragma unroll
      for (int j = 0; j < PARALLEL_LOADS; j++) {
        if (c_offset < stride_ && m_offset < reduction_size_) {
          x_math[j] = input_[address_base];
          count[j]++;
          x_count_inv[j] = accscalar_t(1) / count[j];
          is_valid[j] = accscalar_t(1);
        } else {
          x_math[j] = accscalar_t(0);
          x_count_inv[j] = accscalar_t(0);
          is_valid[j] = accscalar_t(0);
        }
        m_offset += inner_loop_stride;
        address_base += address_increment;
      }

      // calculate mean/m2n with welford
#pragma unroll
      for (int j = 0; j < PARALLEL_LOADS; j++) {
        accscalar_t delta0 = x_math[j] - x_mean[j];
        x_mean[j] += delta0 * x_count_inv[j];
        accscalar_t delta1 = x_math[j] - x_mean[j];
        m_2_n[j] += delta0 * delta1 * is_valid[j];
      }
    }

    // thread reduction to accumulate mean/m_2_n/count between PARALLEL_LOADS
#pragma unroll
    for (int j = 1; j < PARALLEL_LOADS; j++) {
      welford_merge_element(
          count[0], x_mean[0], m_2_n[0], count[j], x_mean[j], m_2_n[j]);
    }

    // release x_mean / m_2_n
    auto mean_th = x_mean[0];
    auto m2_th = m_2_n[0];
    auto count_th = count[0];

    welford_merge_group_vertical(
        item, count_th, mean_th, m2_th, shmem_count_, shmem_mean_, shmem_m2n_);

    if (item.get_group_range(0) > 1) {
      volatile accscalar_t* staging_mean = staging_data_;
      volatile accscalar_t* staging_m2n =
          &staging_data_[stride_ * item.get_group_range(0)];
      volatile int* staging_count = reinterpret_cast<volatile int*>(
          &staging_m2n[stride_ * item.get_group_range(0)]);

      address_base = c_offset + item.get_group(0) * stride_;
      // write data to staging_data;
      if (item.get_local_id(0) == 0 && c_offset < stride_) {
        staging_mean[address_base] = mean_th;
        staging_m2n[address_base] = m2_th;
        staging_count[address_base] = count_th;
      }

      item.barrier(sycl_local_fence);

      // mark group done
      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> count(
            semaphores_[item.get_group(1)]);
        int old = count.fetch_add(
            1, sycl_mem_odr_acq_rel
            /* , default memory scope is device */);
        is_last_group_done_[0] = (old == (item.get_group_range(0) - 1));
      }

      item.barrier(sycl_local_fence);

      // check that all data is now available in global memory
      if (is_last_group_done_[0]) {
        count_th = 0;
        mean_th = accscalar_t(0.0);
        m2_th = accscalar_t(0.0);

        for (int y = item.get_local_id(0); y < item.get_group_range(0);
             y += item.get_local_range(0)) {
          address_base = c_offset + y * stride_;
          int count_new = c_offset < stride_ ? staging_count[address_base] : 0;
          accscalar_t mean_new = c_offset < stride_ ? staging_mean[address_base]
                                                    : accscalar_t(0.0);
          accscalar_t m2n_new =
              c_offset < stride_ ? staging_m2n[address_base] : accscalar_t(0.0);

          welford_merge_element(
              count_th, mean_th, m2_th, count_new, mean_new, m2n_new);
        }

        welford_merge_group_vertical(
            item,
            count_th,
            mean_th,
            m2_th,
            shmem_count_,
            shmem_mean_,
            shmem_m2n_);
        if (item.get_local_id(0) == 0 && c_offset < stride_) {
          out_mean_[c_offset] = static_cast<accscalar_t>(mean_th);
          out_invstd_[c_offset] = VarTransform{}(m2_th / count_th, epsilon_);
        }
      }
    } else {
      if (item.get_group(0) == 0 && item.get_local_id(0) == 0 &&
          c_offset < stride_) {
        out_mean_[c_offset] = static_cast<accscalar_t>(mean_th);
        out_invstd_[c_offset] = VarTransform{}(m2_th / count_th, epsilon_);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shmem_mean_ =
        sycl_local_acc_t<accscalar_t>(sycl::range<1>{(size_t)wg_size_}, cgh);
    shmem_m2n_ =
        sycl_local_acc_t<accscalar_t>(sycl::range<1>{(size_t)wg_size_}, cgh);
    shmem_count_ = sycl_local_acc_t<int>(sycl::range<1>{(size_t)wg_size_}, cgh);
    is_last_group_done_ = sycl_local_acc_t<bool>(sycl::range<1>{1}, cgh);
  }

  BatchNormCollectStatisticsChannelsLastKernelFunctor(
      const scalar_t* RESTRICT input,
      accscalar_t* RESTRICT out_mean,
      accscalar_t* RESTRICT out_invstd,
      volatile accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride,
      accscalar_t epsilon,
      int wg_size)
      : input_(input),
        out_mean_(out_mean),
        out_invstd_(out_invstd),
        staging_data_(staging_data),
        semaphores_(semaphores),
        reduction_size_(reduction_size),
        stride_(stride),
        epsilon_(epsilon),
        wg_size_(wg_size) {}

 private:
  const scalar_t* RESTRICT input_;
  accscalar_t* RESTRICT out_mean_;
  accscalar_t* RESTRICT out_invstd_;
  volatile accscalar_t* staging_data_;
  int* semaphores_;
  const int reduction_size_;
  const int stride_;
  accscalar_t epsilon_;
  int wg_size_;
  sycl_local_acc_t<accscalar_t> shmem_mean_;
  sycl_local_acc_t<accscalar_t> shmem_m2n_;
  sycl_local_acc_t<int> shmem_count_;
  sycl_local_acc_t<bool> is_last_group_done_;
};

template <typename scalar_t, typename VarTransform>
void batch_norm_stats_channels_last_template(
    Tensor& out_mean,
    Tensor& out_invstd,
    const Tensor& input,
    double epsilon) {
  using accscalar_t = acc_type_device<scalar_t, kXPU>;

  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  at::native::resize_output(out_mean, {stride});
  at::native::resize_output(out_invstd, {stride});
  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  auto config = get_adaptive_launch_config(
      reduction_size, stride, true, ELEMENTS_PER_WORK_ITEM);
  auto global_range = std::get<0>(config);
  auto local_range = std::get<1>(config);

  at::Tensor staging_data;
  at::Tensor semaphores;
  auto wg_size_y = local_range[0];
  auto wg_size_x = local_range[1];
  auto nwg_y = global_range[0] / wg_size_y;
  auto nwg_x = global_range[1] / wg_size_x;
  if (nwg_y > 1) {
    staging_data = at::empty({(long)(4 * stride * nwg_y)}, out_mean.options());
    semaphores = at::zeros({(long)nwg_x}, input.options().dtype(at::kInt));
  }
  accscalar_t* staging_data_ptr =
      nwg_y > 1 ? staging_data.mutable_data_ptr<accscalar_t>() : nullptr;
  int* semaphores_ptr =
      nwg_y > 1 ? semaphores.mutable_data_ptr<int>() : nullptr;

  auto kfn = BatchNormCollectStatisticsChannelsLastKernelFunctor<
      VarTransform,
      scalar_t,
      accscalar_t,
      ELEMENTS_PER_ITER>(
      input.const_data_ptr<scalar_t>(),
      out_mean.mutable_data_ptr<accscalar_t>(),
      out_invstd.mutable_data_ptr<accscalar_t>(),
      staging_data_ptr,
      semaphores_ptr,
      reduction_size,
      stride,
      epsilon,
      wg_size_y * wg_size_x);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

// ========================== batch_norm_elemt ==========================

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    bool train,
    typename index_t>
struct BatchNormTransformInputKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    index_t plane = item.get_group(1);

    if (plane >= input_.size(1)) {
      return;
    }

    stat_accscalar_t gamma = weight_.size(0) > 0
        ? static_cast<stat_accscalar_t>(weight_[plane])
        : static_cast<stat_accscalar_t>(1);
    stat_accscalar_t beta = bias_.size(0) > 0
        ? static_cast<stat_accscalar_t>(bias_[plane])
        : static_cast<stat_accscalar_t>(0);
    stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
    stat_accscalar_t invstd;
    if constexpr (train) {
      invstd = var_or_invstd_[plane];
    } else {
      invstd =
          static_cast<stat_accscalar_t>(1) /
          std::sqrt(
              static_cast<stat_accscalar_t>(var_or_invstd_[plane]) + epsilon_);
    }

    index_t bs = input_.size(0);
    index_t fs = input_.size(2);

    index_t bstep = item.get_local_range(0) * item.get_group_range(0);
    for (index_t batch = item.get_global_id(0); batch < bs; batch += bstep) {
      auto o = output_[batch][plane];
      auto i = input_[batch][plane];
      for (index_t feature = item.get_local_id(1); feature < fs;
           feature += item.get_local_range(1)) {
        o[feature] = static_cast<input_scalar_t>(
            gamma * (i[feature] - mean) * invstd + beta);
      }
    }
  }

  BatchNormTransformInputKernelFunctor(
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          RestrictPtrTraits,
          index_t> input,
      GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t>
          output,
      const GenericPackedTensorAccessor<
          typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::
              type,
          1,
          RestrictPtrTraits,
          index_t> mean,
      const GenericPackedTensorAccessor<
          typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::
              type,
          1,
          RestrictPtrTraits,
          index_t> var_or_invstd,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          RestrictPtrTraits,
          index_t> weight,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          RestrictPtrTraits,
          index_t> bias,
      stat_accscalar_t epsilon)
      : input_(input),
        output_(output),
        mean_(mean),
        var_or_invstd_(var_or_invstd),
        weight_(weight),
        bias_(bias),
        epsilon_(epsilon) {}

 private:
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      RestrictPtrTraits,
      index_t>
      input_;
  GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t>
      output_;
  const GenericPackedTensorAccessor<
      typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type,
      1,
      RestrictPtrTraits,
      index_t>
      mean_;
  const GenericPackedTensorAccessor<
      typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type,
      1,
      RestrictPtrTraits,
      index_t>
      var_or_invstd_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>
      weight_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>
      bias_;
  stat_accscalar_t epsilon_;
};

template <
    int VEC_SIZE,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    bool train,
    typename index_t>
struct BatchNormTransformInputVectorizedKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    index_t plane = item.get_group(1);

    if (plane >= input_.size(1)) {
      return;
    }

    stat_accscalar_t gamma = weight_.size(0) > 0
        ? static_cast<stat_accscalar_t>(weight_[plane])
        : static_cast<stat_accscalar_t>(1);
    stat_accscalar_t beta = bias_.size(0) > 0
        ? static_cast<stat_accscalar_t>(bias_[plane])
        : static_cast<stat_accscalar_t>(0);
    stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
    stat_accscalar_t invstd;
    if constexpr (train) {
      invstd = var_or_invstd_[plane];
    } else {
      invstd =
          static_cast<stat_accscalar_t>(1) /
          std::sqrt(
              static_cast<stat_accscalar_t>(var_or_invstd_[plane]) + epsilon_);
    }

    index_t bs = input_.size(0);
    index_t fs = input_.size(2);

    index_t bstep = item.get_local_range(0) * item.get_group_range(0);
    for (index_t batch = item.get_global_id(0); batch < bs; batch += bstep) {
      auto o = output_[batch][plane];
      auto i = input_[batch][plane];

      for (index_t feature_vec_begin = item.get_local_id(1) * VEC_SIZE;
           feature_vec_begin < fs;
           feature_vec_begin += VEC_SIZE * item.get_local_range(1)) {
        using vec_t = memory::aligned_vector<input_scalar_t, VEC_SIZE>;
        vec_t vec;
#pragma unroll
        for (int vt = 0; vt < VEC_SIZE; ++vt) {
          index_t feature = feature_vec_begin + vt;
          vec[vt] = static_cast<input_scalar_t>(
              gamma * (i[feature] - mean) * invstd + beta);
        }
        input_scalar_t* write_ptr = &o[feature_vec_begin];
        *(reinterpret_cast<vec_t*>(write_ptr)) = vec;
      }
    }
  }

  BatchNormTransformInputVectorizedKernelFunctor(
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          RestrictPtrTraits,
          index_t> input,
      GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t>
          output,
      const GenericPackedTensorAccessor<
          typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::
              type,
          1,
          RestrictPtrTraits,
          index_t> mean,
      const GenericPackedTensorAccessor<
          typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::
              type,
          1,
          RestrictPtrTraits,
          index_t> var_or_invstd,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          RestrictPtrTraits,
          index_t> weight,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          RestrictPtrTraits,
          index_t> bias,
      stat_accscalar_t epsilon)
      : input_(input),
        output_(output),
        mean_(mean),
        var_or_invstd_(var_or_invstd),
        weight_(weight),
        bias_(bias),
        epsilon_(epsilon) {}

 private:
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      RestrictPtrTraits,
      index_t>
      input_;
  GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t>
      output_;
  const GenericPackedTensorAccessor<
      typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type,
      1,
      RestrictPtrTraits,
      index_t>
      mean_;
  const GenericPackedTensorAccessor<
      typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type,
      1,
      RestrictPtrTraits,
      index_t>
      var_or_invstd_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>
      weight_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>
      bias_;
  stat_accscalar_t epsilon_;
};

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
void batch_norm_elemt_template(
    const Tensor& output_,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& mean_,
    const Tensor& invstd_) {
  using stat_accscalar_t = acc_type_device<stat_scalar_t, kXPU>;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  auto input =
      get_packed_accessor<const input_scalar_t, 3, RestrictPtrTraits, index_t>(
          input_reshaped, "input");
  auto output =
      get_packed_accessor<input_scalar_t, 3, RestrictPtrTraits, index_t>(
          output_reshaped, "output");
  auto weight = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>(weight_, "weight");
  auto bias = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>(bias_, "bias");
  auto mean =
      packed_accessor_or_dummy<stat_accscalar_t, 1, RestrictPtrTraits, index_t>(
          mean_, "mean");
  auto invstd =
      packed_accessor_or_dummy<stat_accscalar_t, 1, RestrictPtrTraits, index_t>(
          invstd_, "invstd");
  auto& queue = getCurrentSYCLQueue();

  // NOTE: We use transform_input_kernel in training mode, which ignores
  // epsilon
  const double dummy_epsilon = 1e-5;

  int tf = std::max<int>(
      get_num_threads_by_dev_max_group_size(input.size(2) / 4),
      std::min<int>(get_num_threads_by_dev_max_group_size(input.size(2)), 64));
  int tb = std::max<int>(64 / tf, 1);
  sycl::range<2> local_range(tb, tf);
  int nwg_x = input.size(1);
  int nwg_y = std::max<int>(
      1,
      std::min<int>(
          (256 * 1024) / input.size(1), (input.size(0) + tb - 1) / tb));
  nwg_y = std::min<int>(nwg_y, syclMaxWorkItemsPerTile() / (tf * tb));
  sycl::range<2> global_range(nwg_y * tb, nwg_x * tf);

  auto output_ptr = (char*)output_reshaped.data_ptr();
  if (output_reshaped.is_contiguous() &&
      memory::can_vectorize_up_to<input_scalar_t>(output_ptr) >= 4 &&
      sizeof(input_scalar_t) < sizeof(float) && input.size(2) % 4 == 0) {
    auto kfn = BatchNormTransformInputVectorizedKernelFunctor<
        4,
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        true,
        index_t>(input, output, mean, invstd, weight, bias, dummy_epsilon);
    sycl_kernel_submit(global_range, local_range, queue, kfn);
  } else {
    auto kfn = BatchNormTransformInputKernelFunctor<
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        true,
        index_t>(input, output, mean, invstd, weight, bias, dummy_epsilon);
    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int PARALLEL_LOADS>
struct BatchNormTransformInputChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // tensor dimension (m,c)
    // loop along m dimension
    int inner_loop_stride = item.get_local_range(0) * item.get_group_range(0);

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset = item.get_global_id(1);

    if (c_offset >= stride_ || m_offset >= reduction_size_) {
      return;
    }

    auto m_c = mean_[c_offset];
    auto inv_std_c = static_cast<accscalar_t>(inv_std_[c_offset]);
    auto w_c = weight_ == nullptr ? accscalar_t(1.0)
                                  : static_cast<accscalar_t>(weight_[c_offset]);
    auto s_c = shift_ == nullptr ? accscalar_t(0.0)
                                 : static_cast<accscalar_t>(shift_[c_offset]);

    int loop_count =
        1 + (reduction_size_ - 1) / (inner_loop_stride * PARALLEL_LOADS);
    int address_base = m_offset * stride_ + c_offset;
    int address_increment = inner_loop_stride * stride_;

    for (int i = 0; i < loop_count; i++) {
#pragma unroll
      for (int j = 0; j < PARALLEL_LOADS; j++) {
        if (c_offset < stride_ && m_offset < reduction_size_) {
          auto tmp = w_c *
                  (static_cast<accscalar_t>(input_[address_base]) - m_c) *
                  inv_std_c +
              s_c;
          if (z_ != nullptr) {
            tmp += z_[address_base];
          }
          out_[address_base] =
              (fuse_relu_ && tmp <= accscalar_t(0.0)
                   ? scalar_t(0.0)
                   : static_cast<scalar_t>(tmp));
        }
        m_offset += inner_loop_stride;
        address_base += address_increment;
      }
    }
  }

  BatchNormTransformInputChannelsLastKernelFunctor(
      const scalar_t* RESTRICT input,
      const scalar_t* RESTRICT z,
      const accscalar_t* RESTRICT mean,
      const accscalar_t* RESTRICT inv_std,
      const layerscalar_t* RESTRICT weight,
      const layerscalar_t* RESTRICT shift,
      scalar_t* RESTRICT out,
      const int reduction_size,
      const int stride,
      const bool fuse_relu)
      : input_(input),
        z_(z),
        mean_(mean),
        inv_std_(inv_std),
        weight_(weight),
        shift_(shift),
        out_(out),
        reduction_size_(reduction_size),
        stride_(stride),
        fuse_relu_(fuse_relu) {}

 private:
  const scalar_t* RESTRICT input_;
  const scalar_t* RESTRICT z_;
  const accscalar_t* RESTRICT mean_;
  const accscalar_t* RESTRICT inv_std_;
  const layerscalar_t* RESTRICT weight_;
  const layerscalar_t* RESTRICT shift_;
  scalar_t* RESTRICT out_;
  const int reduction_size_;
  const int stride_;
  const bool fuse_relu_;
};

void batch_norm_elemt_channels_last_template(
    const at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& shift, // bias of BN
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::optional<at::Tensor>& z = c10::nullopt, // bias after BN
    const bool fuse_relu = false) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;
  auto config = get_adaptive_launch_config(
      reduction_size, stride, false, ELEMENTS_PER_WORK_ITEM);
  auto global_range = std::get<0>(config);
  auto local_range = std::get<1>(config);
  auto& queue = getCurrentSYCLQueue();
  const auto second_dtype = weight.defined()
      ? weight.scalar_type()
      : (shift.defined() ? shift.scalar_type() : input.scalar_type());

  if (input.scalar_type() != second_dtype) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward_xpu", [&] {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          auto kfn = BatchNormTransformInputChannelsLastKernelFunctor<
              scalar_t,
              accscalar_t,
              accscalar_t,
              ELEMENTS_PER_ITER>(
              input.const_data_ptr<scalar_t>(),
              z.has_value() ? z.value().const_data_ptr<scalar_t>() : nullptr,
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              weight.defined() ? weight.const_data_ptr<accscalar_t>() : nullptr,
              shift.defined() ? shift.const_data_ptr<accscalar_t>() : nullptr,
              output.mutable_data_ptr<scalar_t>(),
              reduction_size,
              stride,
              fuse_relu);
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(
          input.scalar_type() == weight.scalar_type(),
          "batchnorm_forward: input.scalar_type() ",
          input.scalar_type(),
          " is not supported with weight.scalar_type() ",
          weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward_xpu", [&] {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          auto kfn = BatchNormTransformInputChannelsLastKernelFunctor<
              scalar_t,
              accscalar_t,
              scalar_t,
              ELEMENTS_PER_ITER>(
              input.const_data_ptr<scalar_t>(),
              z.has_value() ? z.value().const_data_ptr<scalar_t>() : nullptr,
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              weight.defined() ? weight.const_data_ptr<scalar_t>() : nullptr,
              shift.defined() ? shift.const_data_ptr<scalar_t>() : nullptr,
              output.mutable_data_ptr<scalar_t>(),
              reduction_size,
              stride,
              fuse_relu);
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        });
  }
}

template <typename scalar_t, typename acc_t>
struct BatchNormElementwiseLoopsFunctor {
  scalar_t operator()(
      scalar_t input,
      acc_t weight,
      acc_t bias,
      acc_t mean,
      acc_t invstd) const {
    volatile acc_t res = ((acc_t)input - mean) * weight * invstd + bias;
    return res;
  }
};

// ====================== batch_norm_backward_reduce ======================

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
struct BatchNormBackwardReduceKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    index_t plane = item.get_group(1);

    stat_accscalar_t r_mean = mean_[plane];
    stat_accscalar_t factor = invstd_[plane];

    GradOp<
        input_scalar_t,
        stat_accscalar_t,
        GenericPackedTensorAccessor<
            input_scalar_t,
            3,
            DefaultPtrTraits,
            index_t>>
        g(r_mean, input_, grad_output_);
    int num_sg = item.get_local_range(1) * item.get_local_range(0) / SIMD;
    auto res = plane_reduce<SIMD, Float2<input_scalar_t, stat_accscalar_t>>(
        item, g, grad_output_, plane, num_sg, local_sum_);

    if (item.get_local_id(1) == 0) {
      if (grad_weight_.size(0) > 0) {
        auto grad_weight = grad_weight_;
        grad_weight[plane] = static_cast<stat_scalar_t>(res.v2 * factor);
      }
      if (grad_bias_.size(0) > 0) {
        auto grad_bias = grad_bias_;
        grad_bias[plane] = static_cast<stat_scalar_t>(res.v1);
      }
      if (sum_dy_.size(0) > 0) {
        auto sum_dy = sum_dy_;
        sum_dy[plane] = static_cast<stat_accscalar_t>(res.v1);
      }
      if (sum_dy_xmu_.size(0) > 0) {
        auto sum_dy_xmu = sum_dy_xmu_;
        sum_dy_xmu[plane] = static_cast<stat_accscalar_t>(res.v2);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_sum_ = sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>>(
        sycl::range<1>{(size_t)wg_size_ / SIMD}, cgh);
  }

  BatchNormBackwardReduceKernelFunctor(
      const GenericPackedTensorAccessor<
          input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> input,
      const GenericPackedTensorAccessor<
          input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> grad_output,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> mean,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> invstd,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> sum_dy,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> sum_dy_xmu,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_weight,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_bias,
      int wg_size)
      : input_(input),
        grad_output_(grad_output),
        mean_(mean),
        invstd_(invstd),
        sum_dy_(sum_dy),
        sum_dy_xmu_(sum_dy_xmu),
        grad_weight_(grad_weight),
        grad_bias_(grad_bias),
        wg_size_(wg_size) {}

 private:
  const GenericPackedTensorAccessor<
      input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      input_;
  const GenericPackedTensorAccessor<
      input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      grad_output_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      mean_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      invstd_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      sum_dy_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      sum_dy_xmu_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_weight_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_bias_;
  int wg_size_;
  sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>> local_sum_;
};

// supports CF and CL
template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& weight_,
    const bool input_g,
    const bool weight_g,
    const bool bias_g) {
  using stat_accscalar_t = acc_type_device<stat_scalar_t, kXPU>;
  int64_t n_input = input_.size(1);
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;

  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (input_g) {
    sum_dy_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    sum_dy_xmu_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (weight_g) {
    grad_weight_ = at::empty({n_input}, weight_.options());
  }
  if (bias_g) {
    grad_bias_ = at::empty({n_input}, weight_.options());
  }

  auto input =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          input_reshaped, "input");
  auto grad_output =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_output_reshaped, "grad_output");
  auto grad_weight =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_weight_, "grad_weight");
  auto grad_bias =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_bias_, "grad_bias");
  auto mean =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          mean_, "mean");
  auto invstd =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          invstd_, "invstd");
  auto sum_dy =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy_, "sum_dy");
  auto sum_dy_xmu =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy_xmu_, "sum_dy_xmu");

  auto batch_size = input_reshaped.size(0);
  auto feature_size = input_reshaped.size(2);
  auto& queue = getCurrentSYCLQueue();
  int simd = get_prefer_simd(
      input_reshaped.size(1), input_reshaped.size(0) * input_reshaped.size(1));

  if (simd == SIMD32) {
    using KernelClass = BatchNormBackwardReduceKernelFunctor<
        SIMD32,
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        index_t>;

    int max_wg_size = get_max_group_size<KernelClass>(simd);
    int wg_size_y = std::min<int>(last_pow2(batch_size), max_wg_size / simd);
    int wg_size_x = std::min<int>(
        std::max<int>(get_num_threads<KernelClass>(feature_size, simd), simd),
        max_wg_size / wg_size_y);
    sycl::range<2> local_range(wg_size_y, wg_size_x);
    sycl::range<2> global_range(1 * wg_size_y, n_input * wg_size_x);

    auto kfn = KernelClass(
        input,
        grad_output,
        mean,
        invstd,
        sum_dy,
        sum_dy_xmu,
        grad_weight,
        grad_bias,
        wg_size_y * wg_size_x);

    sycl_kernel_submit(global_range, local_range, queue, kfn);
  } else {
    using KernelClass = BatchNormBackwardReduceKernelFunctor<
        SIMD16,
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        index_t>;

    int max_wg_size = get_max_group_size<KernelClass>(simd);
    int wg_size_y = std::min<int>(last_pow2(batch_size), max_wg_size / simd);
    int wg_size_x = std::min<int>(
        std::max<int>(get_num_threads<KernelClass>(feature_size, simd), simd),
        max_wg_size / wg_size_y);
    sycl::range<2> local_range(wg_size_y, wg_size_x);
    sycl::range<2> global_range(1 * wg_size_y, n_input * wg_size_x);

    auto kfn = KernelClass(
        input,
        grad_output,
        mean,
        invstd,
        sum_dy,
        sum_dy_xmu,
        grad_weight,
        grad_bias,
        wg_size_y * wg_size_x);

    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
  return std::make_tuple(sum_dy_, sum_dy_xmu_, grad_weight_, grad_bias_);
}

template <typename T, typename TACC, typename item_t>
inline void merge_group_vertical_backward(
    item_t item,
    T& sum_dy,
    T& sum_dy_xmu,
    TACC shmem_sum_dy,
    TACC shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = item.get_local_linear_id();
  int local_id_y = item.get_local_id(0);

#pragma unroll
  for (int offset = item.get_local_range(0) / 2; offset > 0; offset >>= 1) {
    if (local_id_y < offset * 2) {
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
    item.barrier(sycl_local_fence);
    if (local_id_y < offset && local_id_y + offset < item.get_local_range(0)) {
      auto address = address_base + offset * item.get_local_range(1);

      sum_dy += shmem_sum_dy[address];
      sum_dy_xmu += shmem_sum_dy_xmu[address];
    }
  }
}

template <
    int PARALLEL_LOADS,
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t>
struct BatchNormBackwardReduceChannelsLastKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    // hide latency with concurrency
    accscalar_t sum_dy[PARALLEL_LOADS];
    accscalar_t sum_dy_xmu[PARALLEL_LOADS];

#pragma unroll
    for (int i = 0; i < PARALLEL_LOADS; i++) {
      sum_dy[i] = accscalar_t(0);
      sum_dy_xmu[i] = accscalar_t(0);
    }
    // tensor dimension (m,c)

    // loop along m dimension
    int inner_loop_stride = item.get_local_range(0) * item.get_group_range(0);

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset = item.get_global_id(1);

    int loop_count =
        1 + (reduction_size_ - 1) / (inner_loop_stride * PARALLEL_LOADS);
    int address_base = m_offset * stride_ + c_offset;
    int address_increment = inner_loop_stride * stride_;

    auto r_mean = c_offset < stride_ ? mean_[c_offset] : accscalar_t(0);
    auto factor = c_offset < stride_ ? inv_std_[c_offset] : accscalar_t(0);

    for (int i = 0; i < loop_count; i++) {
      accscalar_t x_input[PARALLEL_LOADS];
      accscalar_t x_grad_output[PARALLEL_LOADS];

      // load multiple data in
#pragma unroll
      for (int j = 0; j < PARALLEL_LOADS; j++) {
        if (c_offset < stride_ && m_offset < reduction_size_) {
          x_input[j] = input_[address_base];
          x_grad_output[j] = grad_output_[address_base];
        } else {
          x_input[j] = accscalar_t(0);
          x_grad_output[j] = accscalar_t(0);
        }
        m_offset += inner_loop_stride;
        address_base += address_increment;
      }

      // calculate sum_dy / sum_dy_xmu
#pragma unroll
      for (int j = 0; j < PARALLEL_LOADS; j++) {
        sum_dy[j] += x_grad_output[j];
        sum_dy_xmu[j] += x_grad_output[j] * (x_input[j] - r_mean);
      }
    }

#pragma unroll
    for (int j = 1; j < PARALLEL_LOADS; j++) {
      sum_dy[0] += sum_dy[j];
      sum_dy_xmu[0] += sum_dy_xmu[j];
    }

    // release array of registers
    auto sum_dy_th = sum_dy[0];
    auto sum_dy_xmu_th = sum_dy_xmu[0];

    merge_group_vertical_backward(
        item, sum_dy_th, sum_dy_xmu_th, shmem_sum_dy_, shmem_sum_dy_xmu_);

    auto nwg_y = item.get_group_range(0);
    int tid_y = item.get_local_id(0);

    if (nwg_y > 1) {
      volatile accscalar_t* staging_sum_dy = staging_data_;
      volatile accscalar_t* staging_sum_dy_xmu =
          &staging_data_[stride_ * nwg_y];

      address_base = c_offset + item.get_group(0) * stride_;
      // write data to staging_data;
      if (tid_y == 0 && c_offset < stride_) {
        staging_sum_dy[address_base] = sum_dy_th;
        staging_sum_dy_xmu[address_base] = sum_dy_xmu_th;
      }

      item.barrier(sycl_local_fence);

      // mark group done
      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> count(
            semaphores_[item.get_group(1)]);
        int old = count.fetch_add(
            1, sycl_mem_odr_acq_rel
            /* , default memory scope is device */);
        is_last_group_done_[0] = (old == (nwg_y - 1));
      }

      item.barrier(sycl_local_fence);

      // check that all data is now available in global memory
      if (is_last_group_done_[0]) {
        sum_dy_th = accscalar_t(0.0);
        sum_dy_xmu_th = accscalar_t(0.0);

        for (int y = tid_y; y < nwg_y; y += item.get_local_range(0)) {
          address_base = c_offset + y * stride_;
          sum_dy_th +=
              (c_offset < stride_ ? staging_sum_dy[address_base]
                                  : accscalar_t(0.0));
          sum_dy_xmu_th +=
              (c_offset < stride_ ? staging_sum_dy_xmu[address_base]
                                  : accscalar_t(0.0));
        }

        merge_group_vertical_backward(
            item, sum_dy_th, sum_dy_xmu_th, shmem_sum_dy_, shmem_sum_dy_xmu_);
        if (tid_y == 0 && c_offset < stride_) {
          if (grad_bias_ != nullptr) {
            grad_bias_[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
          }
          if (grad_weight_ != nullptr) {
            grad_weight_[c_offset] =
                static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
          }
          sum_dy_o_[c_offset] = sum_dy_th;
          sum_dy_xmu_o_[c_offset] = sum_dy_xmu_th;
        }
      }
    } else {
      if (item.get_group(0) == 0 && tid_y == 0 && c_offset < stride_) {
        if (grad_bias_ != nullptr) {
          grad_bias_[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
        }
        if (grad_weight_ != nullptr) {
          grad_weight_[c_offset] =
              static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
        }
        sum_dy_o_[c_offset] = sum_dy_th;
        sum_dy_xmu_o_[c_offset] = sum_dy_xmu_th;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shmem_sum_dy_ =
        sycl_local_acc_t<accscalar_t>(sycl::range<1>{(size_t)wg_size_}, cgh);
    shmem_sum_dy_xmu_ =
        sycl_local_acc_t<accscalar_t>(sycl::range<1>{(size_t)wg_size_}, cgh);
    is_last_group_done_ = sycl_local_acc_t<bool>(sycl::range<1>{1}, cgh);
  }

  BatchNormBackwardReduceChannelsLastKernelFunctor(
      const scalar_t* RESTRICT input,
      const scalar_t* RESTRICT grad_output,
      const accscalar_t* RESTRICT mean,
      const accscalar_t* RESTRICT inv_std,
      accscalar_t* RESTRICT sum_dy_o,
      accscalar_t* RESTRICT sum_dy_xmu_o,
      layerscalar_t* RESTRICT grad_weight,
      layerscalar_t* RESTRICT grad_bias,
      volatile accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride,
      const int wg_size)
      : input_(input),
        grad_output_(grad_output),
        mean_(mean),
        inv_std_(inv_std),
        sum_dy_o_(sum_dy_o),
        sum_dy_xmu_o_(sum_dy_xmu_o),
        grad_weight_(grad_weight),
        grad_bias_(grad_bias),
        staging_data_(staging_data),
        semaphores_(semaphores),
        reduction_size_(reduction_size),
        stride_(stride),
        wg_size_(wg_size) {}

 private:
  const scalar_t* RESTRICT input_;
  const scalar_t* RESTRICT grad_output_;
  const accscalar_t* RESTRICT mean_;
  const accscalar_t* RESTRICT inv_std_;
  accscalar_t* RESTRICT sum_dy_o_;
  accscalar_t* RESTRICT sum_dy_xmu_o_;
  layerscalar_t* RESTRICT grad_weight_;
  layerscalar_t* RESTRICT grad_bias_;
  volatile accscalar_t* staging_data_;
  int* semaphores_;
  const int reduction_size_;
  const int stride_;
  const int wg_size_;
  sycl_local_acc_t<accscalar_t> shmem_sum_dy_;
  sycl_local_acc_t<accscalar_t> shmem_sum_dy_xmu_;
  sycl_local_acc_t<bool> is_last_group_done_;
};

std::tuple<Tensor, Tensor, Tensor, Tensor>
batch_norm_backward_reduce_channels_last_template(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::Tensor& weight,
    const bool input_g,
    const bool weight_g,
    const bool bias_g) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  at::Tensor sumn_dy = at::zeros({stride}, mean.options());
  at::Tensor sum_dy_xmu = at::zeros({stride}, mean.options());

  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (weight.defined()) {
    grad_weight = at::zeros({stride}, weight.options());
    grad_bias = at::zeros({stride}, weight.options());
  } else {
    // because I cannot return an uninitialized at::Tensor
    grad_weight = at::empty({0}, mean.options());
    grad_bias = at::empty({0}, mean.options());
  }

  auto config = get_adaptive_launch_config(
      reduction_size, stride, false, ELEMENTS_PER_WORK_ITEM);
  auto global_range = std::get<0>(config);
  auto local_range = std::get<1>(config);
  auto wg_size_y = local_range[0];
  auto wg_size_x = local_range[1];
  auto nwg_y = global_range[0] / wg_size_y;
  auto nwg_x = global_range[1] / wg_size_x;

  at::Tensor staging_data;
  at::Tensor semaphores;
  if (nwg_y > 1) {
    staging_data = at::empty({(long)(2 * stride * nwg_y)}, mean.options());
    semaphores = at::zeros({(long)nwg_x}, input.options().dtype(at::kInt));
  }
  auto& queue = getCurrentSYCLQueue();

  if (weight.defined() && input.scalar_type() != weight.scalar_type()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_reduce_xpu",
        [&] {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          accscalar_t* staging_data_ptr = nwg_y > 1
              ? staging_data.mutable_data_ptr<accscalar_t>()
              : nullptr;
          int* semaphores_ptr =
              nwg_y > 1 ? semaphores.mutable_data_ptr<int>() : nullptr;
          auto kfn = BatchNormBackwardReduceChannelsLastKernelFunctor<
              ELEMENTS_PER_ITER,
              scalar_t,
              accscalar_t,
              accscalar_t>(
              input.const_data_ptr<scalar_t>(),
              grad_output.const_data_ptr<scalar_t>(),
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              sumn_dy.mutable_data_ptr<accscalar_t>(),
              sum_dy_xmu.mutable_data_ptr<accscalar_t>(),
              grad_weight.mutable_data_ptr<accscalar_t>(),
              grad_bias.mutable_data_ptr<accscalar_t>(),
              staging_data_ptr,
              semaphores_ptr,
              reduction_size,
              stride,
              wg_size_y * wg_size_x);
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(
          input.scalar_type() == weight.scalar_type(),
          "batchnorm_backward_reduce: input.scalar_type() ",
          input.scalar_type(),
          " is not supported with weight.scalar_type() ",
          weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_reduce_xpu",
        [&] {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          accscalar_t* staging_data_ptr = nwg_y > 1
              ? staging_data.mutable_data_ptr<accscalar_t>()
              : nullptr;
          int* semaphores_ptr =
              nwg_y > 1 ? semaphores.mutable_data_ptr<int>() : nullptr;

          auto kfn = BatchNormBackwardReduceChannelsLastKernelFunctor<
              ELEMENTS_PER_ITER,
              scalar_t,
              accscalar_t,
              scalar_t>(
              input.const_data_ptr<scalar_t>(),
              grad_output.const_data_ptr<scalar_t>(),
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              sumn_dy.mutable_data_ptr<accscalar_t>(),
              sum_dy_xmu.mutable_data_ptr<accscalar_t>(),
              weight.defined() ? grad_weight.mutable_data_ptr<scalar_t>()
                               : nullptr,
              weight.defined() ? grad_bias.mutable_data_ptr<scalar_t>()
                               : nullptr,
              staging_data_ptr,
              semaphores_ptr,
              reduction_size,
              stride,
              wg_size_y * wg_size_x);

          sycl_kernel_submit(global_range, local_range, queue, kfn);
        });
  }

  return std::make_tuple(sumn_dy, sum_dy_xmu, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight_opt,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (canUse32BitIndexMath(grad_output) &&
      batch_norm_use_channels_last_kernels(grad_output) &&
      batch_norm_use_channels_last_kernels(input) &&
      (!weight.defined() || weight.is_contiguous()) && mean.is_contiguous() &&
      invstd.is_contiguous()) {
    return batch_norm_backward_reduce_channels_last_template(
        grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g);
  }
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad_output.scalar_type(),
      "batch_norm_backward_reduce_xpu",
      [&] {
        auto mean_st = mean.dtype();
        auto invstd_st = invstd.dtype();
        TORCH_CHECK(
            mean_st == invstd_st,
            "mean and invstd need to have the same data types");
        const bool mixed_type = is_mixed_type(input, weight);
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        if (canUse32BitIndexMath(grad_output)) {
          if (mixed_type) {
            return batch_norm_backward_reduce_template<
                scalar_t,
                accscalar_t,
                int32_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          } else {
            return batch_norm_backward_reduce_template<
                scalar_t,
                scalar_t,
                int32_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          }
        } else {
          if (mixed_type) {
            return batch_norm_backward_reduce_template<
                scalar_t,
                accscalar_t,
                int64_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          } else {
            return batch_norm_backward_reduce_template<
                scalar_t,
                scalar_t,
                int64_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          }
        }
      });
}

// ====================== batch_norm_backward_elemt ======================

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t,
    bool USE_COUNTS = false>
struct BatchNormBackwardElemtKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    stat_accscalar_t norm_fct;
    if constexpr (USE_COUNTS) {
      int64_t total_numel = 0;
      for (int i = 0; i < world_size_; i++) {
        total_numel += numel_[i];
      }
      norm_fct = static_cast<stat_accscalar_t>(1) /
          static_cast<stat_accscalar_t>(total_numel);
    } else {
      norm_fct = norm_fct_;
    }

    index_t plane = item.get_group(1);

    if (plane >= input_.size(1)) {
      return;
    }

    stat_accscalar_t m_c = mean_[plane];
    stat_accscalar_t m_dy_c = sum_dy_[plane] * norm_fct;
    stat_accscalar_t factor_1_c = invstd_[plane];
    stat_accscalar_t factor_2_c = weight_.size(0) > 0
        ? static_cast<stat_accscalar_t>(weight_[plane])
        : stat_accscalar_t(1);
    factor_2_c *= factor_1_c;
    factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu_[plane] * norm_fct;

    index_t bs = input_.size(0);
    index_t fs = input_.size(2);

    index_t bstep = item.get_local_range(0) * item.get_group_range(0);
    for (index_t batch = item.get_global_id(0); batch < bs; batch += bstep) {
      auto g_i = grad_input_[batch][plane];
      auto g_o = grad_output_[batch][plane];
      auto i = input_[batch][plane];
      for (index_t feature = item.get_local_id(1); feature < fs;
           feature += item.get_local_range(1)) {
        g_i[feature] = static_cast<input_scalar_t>(
            (g_o[feature] - m_dy_c - (i[feature] - m_c) * factor_1_c) *
            factor_2_c);
      }
    }
  }
  BatchNormBackwardElemtKernelFunctor(
      const GenericPackedTensorAccessor<
          input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> input,
      const GenericPackedTensorAccessor<
          input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> grad_output,
      const GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> mean,
      const GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> invstd,
      const GenericPackedTensorAccessor<
          stat_scalar_t,
          1,
          DefaultPtrTraits,
          index_t> weight,
      const GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> sum_dy,
      const GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> sum_dy_xmu,
      GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>
          grad_input,
      const stat_accscalar_t norm_fct,
      const int* RESTRICT numel = nullptr,
      const int world_size = 0)
      : input_(input),
        grad_output_(grad_output),
        mean_(mean),
        invstd_(invstd),
        weight_(weight),
        sum_dy_(sum_dy),
        sum_dy_xmu_(sum_dy_xmu),
        grad_input_(grad_input),
        norm_fct_(norm_fct),
        numel_(numel),
        world_size_(world_size) {}

 private:
  const GenericPackedTensorAccessor<
      input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      input_;
  const GenericPackedTensorAccessor<
      input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      grad_output_;
  const GenericPackedTensorAccessor<
      stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      mean_;
  const GenericPackedTensorAccessor<
      stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      invstd_;
  const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      weight_;
  const GenericPackedTensorAccessor<
      stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      sum_dy_;
  const GenericPackedTensorAccessor<
      stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      sum_dy_xmu_;
  GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>
      grad_input_;
  const stat_accscalar_t norm_fct_;
  const int* RESTRICT numel_;
  const int world_size_;
};

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& weight_,
    const Tensor& sum_dy_,
    const Tensor& sum_dy_xmu_) {
  using stat_accscalar_t = acc_type_device<stat_scalar_t, kXPU>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());
  auto grad_input_reshaped =
      at::empty_like(input_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto input =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          input_reshaped, "input");
  auto grad_input =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_input_reshaped, "grad_input");
  auto grad_output =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_output_reshaped, "grad_output");
  auto mean =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          mean_, "mean");
  auto invstd =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          invstd_, "invstd");
  auto weight =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          weight_, "weight");
  auto sum_dy =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy_, "sum_dy");
  auto sum_dy_xmu =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy_xmu_, "sum_dy_xmu");

  auto& queue = getCurrentSYCLQueue();
  int tf = std::max<int>(
      get_num_threads_by_dev_max_group_size(input.size(2) / 4),
      std::min<int>(get_num_threads_by_dev_max_group_size(input.size(2)), 64));
  int tb = std::max<int>(64 / tf, 1);
  int nwg_x = input.size(1);
  int nwg_y = std::max<int>(
      1,
      std::min<int>(
          (256 * 1024) / input.size(1), (input.size(0) + tb - 1) / tb));
  nwg_y = std::min<int>(nwg_y, syclMaxWorkItemsPerTile() / (tf * tb));
  auto reduction_size = input_.numel() / n_input;
  auto norm_fct = static_cast<stat_accscalar_t>(1.0 / reduction_size);

  sycl::range<2> local_range(tb, tf);
  sycl::range<2> global_range(nwg_y * tb, nwg_x * tf);

  auto kfn = BatchNormBackwardElemtKernelFunctor<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t>(
      input,
      grad_output,
      mean,
      invstd,
      weight,
      sum_dy,
      sum_dy_xmu,
      grad_input,
      norm_fct);
  sycl_kernel_submit(global_range, local_range, queue, kfn);

  return grad_input_reshaped.view(input_.sizes());
}

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& weight_,
    const Tensor& sum_dy_,
    const Tensor& sum_dy_xmu_,
    const Tensor& count) {
  using stat_accscalar_t = at::acc_type_device<stat_scalar_t, kXPU>;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());
  auto grad_input_reshaped =
      at::empty_like(input_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto input =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          input_reshaped, "input");
  auto grad_input =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_input_reshaped, "grad_input");
  auto grad_output =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_output_reshaped, "grad_output");
  auto mean =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          mean_, "mean");
  auto invstd =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          invstd_, "invstd");
  auto weight =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          weight_, "weight");
  auto sum_dy =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy_, "sum_dy");
  auto sum_dy_xmu =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy_xmu_, "sum_dy_xmu");

  auto& queue = getCurrentSYCLQueue();

  int tf = std::max<int>(
      get_num_threads_by_dev_max_group_size(input.size(2) / 4),
      std::min<int>(get_num_threads_by_dev_max_group_size(input.size(2)), 64));
  int tb = std::max<int>(64 / tf, 1);
  int nwg_x = input.size(1);
  int nwg_y = std::max<int>(
      1,
      std::min<int>(
          (256 * 1024) / input.size(1), (input.size(0) + tb - 1) / tb));
  nwg_y = std::min<int>(nwg_y, syclMaxWorkItemsPerTile() / (tf * tb));

  sycl::range<2> local_range(tb, tf);
  sycl::range<2> global_range(nwg_y * tb, nwg_x * tf);

  auto kfn = BatchNormBackwardElemtKernelFunctor<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t,
      true>(
      input,
      grad_output,
      mean,
      invstd,
      weight,
      sum_dy,
      sum_dy_xmu,
      grad_input,
      0,
      count.const_data_ptr<int>(),
      count.numel());
  sycl_kernel_submit(global_range, local_range, queue, kfn);

  return grad_input_reshaped.view(input_.sizes());
}

template <
    int PARALLEL_LOADS,
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    bool USE_COUNTS = false>
struct BatchNormBackwardElemtChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    accscalar_t norm_fct;
    if constexpr (USE_COUNTS) {
      int64_t total_numel = 0;
      for (int i = 0; i < world_size_; i++) {
        total_numel += numel_[i];
      }
      norm_fct =
          static_cast<accscalar_t>(1) / static_cast<accscalar_t>(total_numel);
    } else {
      norm_fct = norm_fct_;
    }

    // tensor dimension (m,c)
    // loop along m dimension
    int inner_loop_stride = item.get_local_range(0) * item.get_group_range(0);

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset = item.get_global_id(1);

    if (c_offset >= stride_ || m_offset >= reduction_size_) {
      return;
    }

    auto m_c = mean_[c_offset];
    auto m_dy_c = sum_dy_[c_offset] * norm_fct;
    auto factor_1_c = inv_std_[c_offset];
    auto factor_2_c =
        (weight_ == nullptr ? accscalar_t(1.0)
                            : static_cast<accscalar_t>(weight_[c_offset])) *
        factor_1_c;
    factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu_[c_offset] * norm_fct;

    int loop_count =
        1 + (reduction_size_ - 1) / (inner_loop_stride * PARALLEL_LOADS);
    int address_base = m_offset * stride_ + c_offset;
    int address_increment = inner_loop_stride * stride_;

    for (int i = 0; i < loop_count; i++) {
#pragma unroll
      for (int j = 0; j < PARALLEL_LOADS; j++) {
        if (c_offset < stride_ && m_offset < reduction_size_) {
          grad_input_[address_base] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(grad_output_[address_base]) - m_dy_c -
               (static_cast<accscalar_t>(input_[address_base]) - m_c) *
                   factor_1_c) *
              factor_2_c);
        }
        m_offset += inner_loop_stride;
        address_base += address_increment;
      }
    }
  }

  BatchNormBackwardElemtChannelsLastKernelFunctor(
      const scalar_t* RESTRICT grad_output,
      const scalar_t* RESTRICT input,
      const accscalar_t* RESTRICT mean,
      const accscalar_t* RESTRICT inv_std,
      const layerscalar_t* RESTRICT weight,
      const accscalar_t* RESTRICT sum_dy,
      const accscalar_t* RESTRICT sum_dy_xmu,
      scalar_t* RESTRICT grad_input,
      const accscalar_t norm_fct,
      const int reduction_size,
      const int stride,
      const int* RESTRICT numel = nullptr,
      const int64_t world_size = 0)
      : grad_output_(grad_output),
        input_(input),
        mean_(mean),
        inv_std_(inv_std),
        weight_(weight),
        sum_dy_(sum_dy),
        sum_dy_xmu_(sum_dy_xmu),
        grad_input_(grad_input),
        norm_fct_(norm_fct),
        reduction_size_(reduction_size),
        stride_(stride),
        numel_(numel),
        world_size_(world_size) {}

 private:
  const scalar_t* RESTRICT grad_output_;
  const scalar_t* RESTRICT input_;
  const accscalar_t* RESTRICT mean_;
  const accscalar_t* RESTRICT inv_std_;
  const layerscalar_t* RESTRICT weight_;
  const accscalar_t* RESTRICT sum_dy_;
  const accscalar_t* RESTRICT sum_dy_xmu_;
  scalar_t* RESTRICT grad_input_;
  const accscalar_t norm_fct_;
  const int reduction_size_;
  const int stride_;
  const int* RESTRICT numel_;
  const int64_t world_size_;
};

at::Tensor batch_norm_backward_elemt_channels_last_template(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::Tensor& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;
  auto norm_fct = 1.0 / reduction_size;

  // Input is guarunteed to be channels-last compatible
  at::Tensor grad_input = at::empty_like(input);

  auto config = get_adaptive_launch_config(
      reduction_size, stride, true, ELEMENTS_PER_WORK_ITEM);
  auto global_range = std::get<0>(config);
  auto local_range = std::get<1>(config);
  auto& queue = getCurrentSYCLQueue();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "batchnorm_backward_element_xpu",
      [&] {
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

        if (weight.defined() && weight.scalar_type() != input.scalar_type()) {
          auto kfn = BatchNormBackwardElemtChannelsLastKernelFunctor<
              ELEMENTS_PER_ITER,
              scalar_t,
              accscalar_t,
              accscalar_t>(
              grad_output.const_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              weight.const_data_ptr<accscalar_t>(),
              sum_dy.const_data_ptr<accscalar_t>(),
              sum_dy_xmu.const_data_ptr<accscalar_t>(),
              grad_input.mutable_data_ptr<scalar_t>(),
              static_cast<accscalar_t>(norm_fct),
              reduction_size,
              stride);
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        } else {
          auto kfn = BatchNormBackwardElemtChannelsLastKernelFunctor<
              ELEMENTS_PER_ITER,
              scalar_t,
              accscalar_t,
              scalar_t>(
              grad_output.const_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              weight.defined() ? weight.const_data_ptr<scalar_t>() : nullptr,
              sum_dy.const_data_ptr<accscalar_t>(),
              sum_dy_xmu.const_data_ptr<accscalar_t>(),
              grad_input.mutable_data_ptr<scalar_t>(),
              static_cast<accscalar_t>(norm_fct),
              reduction_size,
              stride);
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        }
      });

  return grad_input;
}

at::Tensor batch_norm_backward_elemt_channels_last_template(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::Tensor& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  // Input is guarunteed to be channels-last compatible
  at::Tensor grad_input = at::empty_like(input);

  auto config = get_adaptive_launch_config(
      reduction_size, stride, false, ELEMENTS_PER_WORK_ITEM);
  auto global_range = std::get<0>(config);
  auto local_range = std::get<1>(config);
  auto& queue = getCurrentSYCLQueue();

  if (weight.defined() && weight.scalar_type() != input.scalar_type()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_element_xpu",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          auto kfn = BatchNormBackwardElemtChannelsLastKernelFunctor<
              ELEMENTS_PER_ITER,
              scalar_t,
              accscalar_t,
              accscalar_t,
              true>(
              grad_output.const_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              weight.const_data_ptr<accscalar_t>(),
              sum_dy.const_data_ptr<accscalar_t>(),
              sum_dy_xmu.const_data_ptr<accscalar_t>(),
              grad_input.mutable_data_ptr<scalar_t>(),
              0,
              reduction_size,
              stride,
              count.const_data_ptr<int>(),
              count.numel());
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(
          input.scalar_type() == weight.scalar_type(),
          "batchnorm_backward_element: input.scalar_type() ",
          input.scalar_type(),
          " is not supported with weight.scalar_type() ",
          weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "batchnorm_backward_element_xpu",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          auto kfn = BatchNormBackwardElemtChannelsLastKernelFunctor<
              ELEMENTS_PER_ITER,
              scalar_t,
              accscalar_t,
              scalar_t,
              true>(
              grad_output.const_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              mean.const_data_ptr<accscalar_t>(),
              inv_std.const_data_ptr<accscalar_t>(),
              weight.defined() ? weight.const_data_ptr<scalar_t>() : nullptr,
              sum_dy.const_data_ptr<accscalar_t>(),
              sum_dy_xmu.const_data_ptr<accscalar_t>(),
              grad_input.mutable_data_ptr<scalar_t>(),
              0,
              reduction_size,
              stride,
              count.const_data_ptr<int>(),
              count.numel());
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        });
  }

  return grad_input;
}

// ====================== batch_norm_update_stats ======================

void batch_norm_mean_var(
    const Tensor& self,
    Tensor& save_mean,
    Tensor& save_var) {
  // NOTE: Epsilon is only used for InvStd, not Var. The value here is ignored.
  const double dummy_epsilon = 1e-5;
  switch (batch_norm_choose_impl(self)) {
    case Impl::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats_xpu", [&] {
            batch_norm_stats_template<scalar_t, int32_t, Var>(
                save_mean, save_var, self, dummy_epsilon);
          });
      return;
    }
    case Impl::ChannelsLast: {
      if ((!save_mean.defined() || save_mean.is_contiguous()) &&
          (!save_var.defined() || save_var.is_contiguous())) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats_xpu", [&] {
              batch_norm_stats_channels_last_template<scalar_t, Var>(
                  save_mean, save_var, self, dummy_epsilon);
            });
        return;
      }
      [[fallthrough]];
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector reduce_dims(ndim - 1);
      reduce_dims[0] = 0;
      for (int64_t i = 2; i < ndim; ++i) {
        reduce_dims[i - 1] = i;
      }
      // For some reason this isn't an actual operator but it exists anyway...
      // at::native::var_mean_out(save_var,save_mean,
      // self,/*dims=*/reduce_dims,/*unbiased=*/false,/*keepdim=*/false);
      at::AtenIpexTypeXPU::std_var_mean_out(
          "var_mean", save_var, save_mean, self, reduce_dims, 0, false, false);
      return;
    }
  }
}

// ====================== native_batch_norm ======================

template <typename scalar_t, typename acc_t>
struct BatchNormUpdateStatsAndInvertFunctor {
  std::tuple<scalar_t, scalar_t, acc_t> operator()(
      acc_t mean,
      acc_t var,
      scalar_t running_mean,
      scalar_t running_var) const {
    const acc_t unbiased_var = var * bessel_correction_factor_;
    volatile acc_t a = mean * momentum_ + (1 - momentum_) * (acc_t)running_mean;
    volatile acc_t b =
        unbiased_var * momentum_ + (1 - momentum_) * (acc_t)running_var;
    volatile acc_t c = c10::xpu::compat::rsqrt(var + eps_);
    return std::tuple<scalar_t, scalar_t, acc_t>{a, b, c};
  }

  BatchNormUpdateStatsAndInvertFunctor(
      const acc_t bessel_correction_factor,
      const acc_t eps,
      const acc_t momentum)
      : bessel_correction_factor_(bessel_correction_factor),
        eps_(eps),
        momentum_(momentum) {}

 private:
  const acc_t bessel_correction_factor_;
  const acc_t eps_;
  const acc_t momentum_;
};

void batch_norm_update_stats_and_invert(
    const Tensor& save_mean,
    const Tensor& save_var,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum_,
    double epsilon,
    int64_t N) {
  auto iter = TensorIteratorConfig()
                  .add_output(running_mean)
                  .add_output(running_var)
                  .add_output(save_var)
                  .add_const_input(save_mean)
                  .add_input(save_var)
                  .add_input(running_mean)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      running_mean.scalar_type(),
      "batch_norm_update_stats_and_invert_xpu",
      [&] {
        using acc_t = acc_type_device<scalar_t, kXPU>;
        const auto bessel_correction_factor = static_cast<acc_t>(
            static_cast<double>(N) / static_cast<double>(N - 1));
        const auto eps = static_cast<acc_t>(epsilon);
        const auto momentum = static_cast<acc_t>(momentum_);
        BatchNormUpdateStatsAndInvertFunctor<scalar_t, acc_t> f(
            bessel_correction_factor, eps, momentum);
        gpu_kernel_multiple_outputs(iter, f);
      });
}

template <typename scalar_t, typename acc_t>
struct BatchNormCalcInvstdFunctor {
  acc_t operator()(scalar_t var) const {
    volatile acc_t v = var + eps_;
    return c10::xpu::compat::rsqrt(v);
  }

  BatchNormCalcInvstdFunctor(acc_t eps) : eps_(eps) {}

 private:
  acc_t eps_;
};

void batch_norm_calc_invstd(
    const Tensor& out_invstd,
    const Tensor& running_var,
    double epsilon) {
  auto iter = TensorIteratorConfig()
                  .add_output(out_invstd)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      running_var.scalar_type(),
      "batch_norm_invert_std_xpu",
      [&] {
        using acc_t = at::acc_type_device<scalar_t, kXPU>;
        auto eps = static_cast<acc_t>(epsilon);
        BatchNormCalcInvstdFunctor<scalar_t, acc_t> f(eps);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t, typename acc_t>
struct BatchNormElementwiseFunctor {
  scalar_t operator()(
      scalar_t input,
      acc_t weight,
      acc_t bias,
      acc_t mean,
      acc_t invstd) const {
    return ((input - mean) * invstd) * weight + bias;
  }
};

void batch_norm_elementwise(
    const Tensor& out,
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const Tensor& mean_,
    const Tensor& invstd_) {
  switch (batch_norm_choose_impl(self)) {
    case Impl::Contiguous: {
      c10::MaybeOwned<Tensor> weight =
          at::borrow_from_optional_tensor(weight_opt);
      c10::MaybeOwned<Tensor> bias = at::borrow_from_optional_tensor(bias_opt);
      at::native::resize_output(out, self.sizes());
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
            const bool mixed_type = is_mixed_type(self, *weight, *bias);
            if (mixed_type) {
              batch_norm_elemt_template<scalar_t, accscalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            } else {
              batch_norm_elemt_template<scalar_t, scalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            }
          });
      return;
    }
    case Impl::ChannelsLast: {
      auto weight = at::borrow_from_optional_tensor(weight_opt);
      auto bias = at::borrow_from_optional_tensor(bias_opt);

      if (resize_output_check(out, self.sizes())) {
        resize_impl_xpu_(
            out.unsafeGetTensorImpl(), self.sizes(), self.strides());
      }
      if ((out.strides() == self.strides()) &&
          (!weight->defined() || weight->is_contiguous()) &&
          (!bias->defined() || bias->is_contiguous()) &&
          (!mean_.defined() || mean_.is_contiguous()) &&
          (!invstd_.defined() || invstd_.is_contiguous())) {
        batch_norm_elemt_channels_last_template(
            out, self, *weight, *bias, mean_, invstd_);
        return;
      }
      [[fallthrough]];
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector sizes(ndim, 1), strides(ndim, 0);
      // Helper to convert 1d tensors to an nd tensor that broadcasts with input
      // All elements go into the channel dimension
      auto as_nd = [&](const Tensor& t) {
        TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
        sizes[1] = t.sizes()[0];
        strides[1] = t.strides()[0];
        return t.as_strided(sizes, strides);
      };

      auto weight = weight_opt.has_value() && weight_opt->defined()
          ? as_nd(*weight_opt)
          : at::scalar_tensor(1, mean_.options());
      auto bias = bias_opt.has_value() && bias_opt->defined()
          ? as_nd(*bias_opt)
          : at::scalar_tensor(0, mean_.options());
      auto mean = as_nd(mean_);
      auto invstd = as_nd(invstd_);

      auto iter = TensorIteratorConfig()
                      .add_output(out)
                      .add_input(self)
                      .add_input(weight)
                      .add_input(bias)
                      .add_input(mean)
                      .add_input(invstd)
                      .check_all_same_dtype(false)
                      .promote_inputs_to_common_dtype(false)
                      .build();

      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using acc_t = at::acc_type_device<scalar_t, kXPU>;
            BatchNormElementwiseFunctor<scalar_t, acc_t> f;
            gpu_kernel(iter, f);
          });
      return;
    }
  }
}

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_kernel(
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double epsilon,
    Tensor& output,
    Tensor& save_mean,
    Tensor& save_invstd) {
  const bool has_running_mean =
      (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var =
      (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK(has_running_mean == has_running_var);

  if (train) {
    batch_norm_mean_var(self, save_mean, save_invstd);
    if (has_running_mean) {
      const int64_t N = self.numel() / save_mean.numel();
      batch_norm_update_stats_and_invert(
          save_mean,
          save_invstd,
          *running_mean_opt,
          *running_var_opt,
          momentum,
          epsilon,
          N);
    } else {
      batch_norm_calc_invstd(save_invstd, save_invstd, epsilon);
    }
  } else {
    TORCH_CHECK(has_running_mean);
    at::native::resize_output(save_mean, running_mean_opt->sizes());
    save_mean.copy_(*running_mean_opt, /*non_blocking=*/true);
    batch_norm_calc_invstd(save_invstd, running_var_opt.value(), epsilon);
  }
  batch_norm_elementwise(
      output, self, weight_opt, bias_opt, save_mean, save_invstd);
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_invstd);
}

// ====================== native_batch_norm_bw ======================

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
struct BatchNormBackwardKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    index_t plane = item.get_group(1);
    index_t N = grad_output_.size(0) * grad_output_.size(2);

    stat_accscalar_t mean, invstd;
    if (train_) {
      mean = save_mean_[plane];
      invstd = save_invstd_[plane];
    } else {
      mean = static_cast<stat_accscalar_t>(running_mean_[plane]);
      invstd =
          static_cast<stat_accscalar_t>(1) /
          std::sqrt(
              static_cast<stat_accscalar_t>(running_var_[plane]) + epsilon_);
    }

    stat_accscalar_t weight_val = weight_.size(0) > 0
        ? static_cast<stat_accscalar_t>(weight_[plane])
        : stat_accscalar_t(1);
    stat_accscalar_t norm = stat_accscalar_t(1) / N;

    // Compute two values across (batch, x/y/z) in one pass:
    // 1. Sum(grad_output)
    // 2. DotProduct(input - mean, grad_output)
    GradOp<
        input_scalar_t,
        stat_accscalar_t,
        GenericPackedTensorAccessor<
            const input_scalar_t,
            3,
            DefaultPtrTraits,
            index_t>>
        g(mean, input_, grad_output_);
    int num_sg = item.get_local_range(1) * item.get_local_range(0) / SIMD;
    auto res = plane_reduce<SIMD, Float2<input_scalar_t, stat_accscalar_t>>(
        item, g, grad_output_, plane, num_sg, local_sum_);

    stat_accscalar_t grad_output_sum = res.v1;
    stat_accscalar_t dot_p = res.v2;

    stat_accscalar_t grad_mean = grad_output_sum * norm;
    stat_accscalar_t proj_scale = dot_p * norm * invstd * invstd;
    stat_accscalar_t grad_scale = invstd * weight_val;

    auto grad_input = grad_input_;
    if (grad_input_.data() != NULL) {
      for (int batch = item.get_local_id(0); batch < grad_output_.size(0);
           batch += item.get_local_range(0)) {
        for (int x = item.get_local_id(1); x < grad_output_.size(2);
             x += item.get_local_range(1)) {
          input_scalar_t go = grad_output_[batch][plane][x];
          if (train_) {
            stat_accscalar_t inp = input_[batch][plane][x];
            stat_accscalar_t proj = (inp - mean) * proj_scale;
            grad_input[batch][plane][x] = static_cast<input_scalar_t>(
                (go - proj - grad_mean) * grad_scale);
          } else {
            grad_input[batch][plane][x] =
                static_cast<input_scalar_t>(go * grad_scale);
          }
        }
      }
    }

    if (grad_weight_.size(0) > 0) {
      if (item.get_local_id(1) == 0) {
        auto grad_weight = grad_weight_;
        grad_weight[plane] = static_cast<stat_scalar_t>(dot_p * invstd);
      }
    }

    if (grad_bias_.size(0) > 0) {
      if (item.get_local_id(1) == 0) {
        auto grad_bias = grad_bias_;
        grad_bias[plane] = static_cast<stat_scalar_t>(grad_output_sum);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_sum_ = sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>>(
        sycl::range<1>{(size_t)wg_size_}, cgh);
  }

  BatchNormBackwardKernelFunctor(
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> input,
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> grad_output,
      GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>
          grad_input,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_weight,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_bias,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          DefaultPtrTraits,
          index_t> weight,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          DefaultPtrTraits,
          index_t> running_mean,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          DefaultPtrTraits,
          index_t> running_var,
      const GenericPackedTensorAccessor<
          const stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> save_mean,
      const GenericPackedTensorAccessor<
          const stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> save_invstd,
      bool train,
      stat_accscalar_t epsilon,
      int wg_size)
      : input_(input),
        grad_output_(grad_output),
        grad_input_(grad_input),
        grad_weight_(grad_weight),
        grad_bias_(grad_bias),
        weight_(weight),
        running_mean_(running_mean),
        running_var_(running_var),
        save_mean_(save_mean),
        save_invstd_(save_invstd),
        train_(train),
        epsilon_(epsilon),
        wg_size_(wg_size) {}

 private:
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      input_;
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      grad_output_;
  GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>
      grad_input_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_weight_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_bias_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      weight_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      running_mean_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      running_var_;
  const GenericPackedTensorAccessor<
      const stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      save_mean_;
  const GenericPackedTensorAccessor<
      const stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      save_invstd_;
  bool train_;
  stat_accscalar_t epsilon_;
  int wg_size_;
  sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>> local_sum_;
};

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& running_mean_,
    const Tensor& running_var_,
    const Tensor& save_mean_,
    const Tensor& save_invstd_,
    bool train,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  using accscalar_t = acc_type_device<stat_scalar_t, kXPU>;
  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  auto input =
      get_packed_accessor<const input_scalar_t, 3, DefaultPtrTraits, index_t>(
          input_reshaped, "input");
  auto grad_output =
      get_packed_accessor<const input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_output_reshaped, "grad_output");
  auto grad_input =
      packed_accessor_or_dummy<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_input_reshaped, "grad_input");
  auto weight = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>(weight_, "weight");
  auto grad_weight =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_weight_, "grad_weight");
  auto grad_bias =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_bias_, "grad_bias");
  auto running_mean = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>(running_mean_, "running_mean");
  auto running_var = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>(running_var_, "running_var");
  auto save_mean =
      packed_accessor_or_dummy<const accscalar_t, 1, DefaultPtrTraits, index_t>(
          save_mean_, "save_mean");
  auto save_invstd =
      packed_accessor_or_dummy<const accscalar_t, 1, DefaultPtrTraits, index_t>(
          save_invstd_, "save_invstd");

  int simd = get_prefer_simd(
      input_reshaped.size(1), input_reshaped.size(0) * input_reshaped.size(1));

  auto& queue = getCurrentSYCLQueue();

  if (simd == SIMD32) {
    using KernelClass = BatchNormBackwardKernelFunctor<
        SIMD32,
        input_scalar_t,
        stat_scalar_t,
        accscalar_t,
        index_t>;

    int max_group_size = get_max_group_size<KernelClass>(simd);
    int tf = get_num_threads<KernelClass>(input.size(2), simd);
    int wg_sz_y = std::max<int>(1, max_group_size / tf);
    sycl::range<2> local_range(wg_sz_y, tf);
    sycl::range<2> global_range(1 * wg_sz_y, input.size(1) * tf);

    auto kfn = KernelClass(
        input,
        grad_output,
        grad_input,
        grad_weight,
        grad_bias,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        epsilon,
        wg_sz_y * tf);

    sycl_kernel_submit(global_range, local_range, queue, kfn);
  } else {
    using KernelClass = BatchNormBackwardKernelFunctor<
        SIMD16,
        input_scalar_t,
        stat_scalar_t,
        accscalar_t,
        index_t>;

    int max_group_size = get_max_group_size<KernelClass>(simd);
    int tf = get_num_threads<KernelClass>(input.size(2), simd);
    int wg_sz_y = std::max<int>(1, max_group_size / tf);
    sycl::range<2> local_range(wg_sz_y, tf);
    sycl::range<2> global_range(1 * wg_sz_y, input.size(1) * tf);

    auto kfn = KernelClass(
        input,
        grad_output,
        grad_input,
        grad_weight,
        grad_bias,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        epsilon,
        wg_sz_y * tf);

    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

template <typename scalar_t, typename accscalar_t>
struct BatchNormElementwiseBackwardTrainFunctor {
  scalar_t operator()(
      scalar_t gO,
      scalar_t input,
      accscalar_t weight,
      accscalar_t mean,
      accscalar_t invstd,
      accscalar_t xmu,
      accscalar_t dy) const {
    auto factor_1_c = invstd * invstd * xmu * norm_fct_;
    auto factor_2_c = weight * invstd;
    auto m_dy_c = dy * norm_fct_;
    volatile accscalar_t res =
        ((accscalar_t)gO - m_dy_c - ((accscalar_t)input - mean) * factor_1_c) *
        factor_2_c;
    return res;
  }

  BatchNormElementwiseBackwardTrainFunctor(accscalar_t norm_fct)
      : norm_fct_(norm_fct) {}

 private:
  accscalar_t norm_fct_;
};

Tensor batch_norm_elementwise_backward_train(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu) {
  switch (batch_norm_choose_impl(input, grad_out)) {
    case Impl::Contiguous: {
      return AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          input.scalar_type(),
          "batch_norm_backward_elemt_xpu",
          [&] {
            using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
            const bool mixed_type = is_mixed_type(input, weight);
            if (mixed_type) {
              return batch_norm_backward_elemt_template<
                  scalar_t,
                  accscalar_t,
                  int32_t>(
                  grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
            } else {
              return batch_norm_backward_elemt_template<
                  scalar_t,
                  scalar_t,
                  int32_t>(
                  grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
            }
          });
    }
    case Impl::ChannelsLast: {
      if ((!weight.defined() || weight.is_contiguous()) &&
          mean.is_contiguous() && invstd.is_contiguous()) {
        return batch_norm_backward_elemt_channels_last_template(
            grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
      }
      [[fallthrough]];
    }
    case Impl::General: {
      const auto ndim = input.dim();
      DimVector sizes(ndim, 1), strides(ndim, 0);
      auto as_nd = [&](const Tensor& t) {
        TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
        sizes[1] = t.sizes()[0];
        strides[1] = t.strides()[0];
        return t.as_strided(sizes, strides);
      };
      auto invstd_nd = as_nd(invstd);
      auto mean_nd = as_nd(mean);
      auto sum_dy_nd = as_nd(sum_dy);
      auto sum_dy_xmu_nd = as_nd(sum_dy_xmu);
      auto weight_nd = weight.defined()
          ? as_nd(weight)
          : at::scalar_tensor(1.0, input.options().dtype(mean.scalar_type()));

      Tensor grad_input = at::empty(
          input.sizes(),
          grad_out.options().memory_format(input.suggest_memory_format()));
      auto iter = TensorIteratorConfig()
                      .add_output(grad_input)
                      .add_input(grad_out)
                      .add_input(input)
                      .add_input(weight_nd)
                      .add_input(mean_nd)
                      .add_input(invstd_nd)
                      .add_input(sum_dy_xmu_nd)
                      .add_input(sum_dy_nd)
                      .check_all_same_dtype(false)
                      .promote_inputs_to_common_dtype(false)
                      .build();

      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          grad_out.scalar_type(),
          "batch_norm_eval_backward_xpu",
          [&] {
            using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
            auto norm_fct =
                static_cast<accscalar_t>(1.0 / (input.numel() / input.size(1)));
            BatchNormElementwiseBackwardTrainFunctor<scalar_t, accscalar_t> f(
                norm_fct);
            gpu_kernel(iter, f);
          });
      return grad_input;
    }
  }
  TORCH_INTERNAL_ASSERT(false);
}

template <typename scalar_t, typename accscalar_t>
struct BatchNormElementwiseBackwardEvalWithWeightfunctor {
  scalar_t operator()(scalar_t gO, accscalar_t invstd, accscalar_t weight)
      const {
    volatile accscalar_t res = (accscalar_t)gO * weight * invstd;
    return res;
  }
};

template <typename scalar_t, typename accscalar_t>
struct BatchNormElementwiseBackwardEvalfunctor {
  scalar_t operator()(scalar_t gO, accscalar_t invstd) const {
    volatile accscalar_t res = (accscalar_t)gO * invstd;
    return res;
  }
};

Tensor batch_norm_elementwise_backward_eval(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& invstd,
    const Tensor& weight) {
  const auto ndim = input.dim();
  DimVector shape(ndim, 1), strides(ndim, 0);
  shape[1] = invstd.sizes()[0];
  strides[1] = invstd.strides()[0];
  auto invstd_nd = invstd.as_strided(shape, strides);
  Tensor grad_input = at::empty(input.sizes(), grad_out.options());

  if (weight.defined()) {
    strides[1] = weight.strides()[0];
    auto weight_nd = weight.as_strided(shape, strides);
    auto iter = TensorIteratorConfig()
                    .add_output(grad_input)
                    .add_const_input(grad_out)
                    .add_const_input(invstd_nd)
                    .add_const_input(weight_nd)
                    .check_all_same_dtype(false)
                    .promote_inputs_to_common_dtype(false)
                    .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        grad_out.scalar_type(),
        "batch_norm_eval_backward_xpu",
        [&] {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          BatchNormElementwiseBackwardEvalWithWeightfunctor<
              scalar_t,
              accscalar_t>
              f;
          gpu_kernel(iter, f);
        });
  } else {
    auto iter = TensorIteratorConfig()
                    .add_output(grad_input)
                    .add_const_input(grad_out)
                    .add_const_input(invstd_nd)
                    .check_all_same_dtype(false)
                    .promote_inputs_to_common_dtype(false)
                    .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        grad_out.scalar_type(),
        "batch_norm_eval_backward_xpu",
        [&] {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          BatchNormElementwiseBackwardEvalfunctor<scalar_t, accscalar_t> f;
          gpu_kernel(iter, f);
        });
  }
  return grad_input;
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    const c10::optional<Tensor>& save_mean_opt,
    const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> save_mean =
      at::borrow_from_optional_tensor(save_mean_opt);
  c10::MaybeOwned<Tensor> save_invstd =
      at::borrow_from_optional_tensor(save_invstd_opt);
  c10::MaybeOwned<Tensor> running_mean =
      at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var =
      at::borrow_from_optional_tensor(running_var_opt);

  const bool needs_reduction =
      train || grad_input_mask[1] || grad_input_mask[2];

  // Fused reduction & elementwise kernel
  if (needs_reduction && grad_input_mask[0] &&
      !batch_norm_use_channels_last_kernels(input) &&
      canUse32BitIndexMath(input) && canUse32BitIndexMath(grad_out)) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batch_norm_backward_xpu", [&] {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          const bool mixed_type =
              is_mixed_type(input, *weight, *running_mean, *running_var);
          if (mixed_type) {
            return batch_norm_backward_template<scalar_t, accscalar_t, int32_t>(
                grad_out,
                input,
                *weight,
                *running_mean,
                *running_var,
                *save_mean,
                *save_invstd,
                train,
                epsilon,
                grad_input_mask);
          } else {
            return batch_norm_backward_template<scalar_t, scalar_t, int32_t>(
                grad_out,
                input,
                *weight,
                *running_mean,
                *running_var,
                *save_mean,
                *save_invstd,
                train,
                epsilon,
                grad_input_mask);
          }
        });
  }

  const auto acc_type = at::toAccumulateType(input.scalar_type(), true);
  Tensor mean;
  TORCH_INTERNAL_ASSERT(
      save_mean->defined(), "save_mean should always be defined\n");
  if (save_mean->numel() != 0) {
    mean = *save_mean;
  } else if (needs_reduction) {
    TORCH_CHECK(!train && running_mean->defined());
    mean = (running_mean->scalar_type() == acc_type)
        ? *running_mean
        : running_mean->to(acc_type);
  }

  Tensor invstd;
  TORCH_INTERNAL_ASSERT(
      save_invstd->defined(), "save_invstd should always be defined\n");
  if (save_invstd->numel() != 0) {
    invstd = *save_invstd;
  } else {
    TORCH_CHECK(!train && running_var->defined());
    auto n_channels = input.sizes()[1];
    invstd = at::empty({n_channels}, input.options().dtype(acc_type));
    batch_norm_calc_invstd(invstd, *running_var, epsilon);
  }

  Tensor sum_dy, sum_dy_xmu, grad_weight, grad_bias;
  if (needs_reduction) {
    std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias) =
        batch_norm_backward_reduce_kernel(
            grad_out,
            input,
            mean,
            invstd,
            *weight,
            grad_input_mask[0],
            grad_input_mask[1],
            grad_input_mask[2]);
  }

  Tensor grad_input;
  if (grad_input_mask[0]) {
    if (train) {
      // NOTE: sum_dy and sum_dy_xmy are defined, as train implies
      // needs_reduction
      grad_input = batch_norm_elementwise_backward_train(
          grad_out, input, mean, invstd, *weight, sum_dy, sum_dy_xmu);
    } else {
      grad_input = batch_norm_elementwise_backward_eval(
          grad_out, input, invstd, *weight);
    }
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct BatchNormReduceStatisticsKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int feature_size = vec_mean_.size(1);
    int world_size = vec_mean_.size(0);

    int bid = item.get_group(0);
    int tid = item.get_local_id(0);
    int group_size_x = item.get_local_range(0);

    auto mean = mean_;
    auto invstd = invstd_;
    auto running_mean = running_mean_;
    auto running_var = running_var_;

    // first the reductions each thread does separately
    for (int i = bid * group_size_x + tid; i < feature_size;
         i += item.get_group_range(0) * group_size_x) {
      accscalar_t avg = 0;
      accscalar_t var_n = 0;
      index_t n = 0;
      for (int j = 0; j < world_size; j++) {
        scalar_t count = counts_[j];
        accscalar_t m = vec_mean_[j][i];
        accscalar_t v = accscalar_t(1.0) / (vec_invstd_[j][i]);
        v = (v * v - epsilon_) * count;
        accscalar_t factor = 1.0 / (n + count);
        var_n += v + (avg - m) * (avg - m) * n * count * factor;
        avg = n * factor * avg + count * factor * m;
        n += count;
      }
      mean[i] = avg;
      invstd[i] = static_cast<accscalar_t>(1) / std::sqrt(var_n / n + epsilon_);
      if (running_mean.data() != NULL) {
        running_mean[i] = static_cast<scalar_t>(
            (1 - momentum_) * running_mean[i] + momentum_ * avg);
      }
      accscalar_t unbiasedVar = var_n / (n - 1);
      if (running_var.data() != NULL) {
        running_var[i] = static_cast<scalar_t>(
            (1 - momentum_) * running_var[i] + momentum_ * unbiasedVar);
      }
    }
  }
  BatchNormReduceStatisticsKernelFunctor(
      const GenericPackedTensorAccessor<
          accscalar_t,
          2,
          RestrictPtrTraits,
          index_t> vec_mean,
      const GenericPackedTensorAccessor<
          accscalar_t,
          2,
          RestrictPtrTraits,
          index_t> vec_invstd,
      GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t>
          mean,
      GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t>
          invstd,
      GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t>
          running_mean,
      GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t>
          running_var,
      const accscalar_t epsilon,
      const accscalar_t momentum,
      const GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t>
          counts)
      : vec_mean_(vec_mean),
        vec_invstd_(vec_invstd),
        mean_(mean),
        invstd_(invstd),
        running_mean_(running_mean),
        running_var_(running_var),
        epsilon_(epsilon),
        momentum_(momentum),
        counts_(counts) {}

 private:
  const GenericPackedTensorAccessor<accscalar_t, 2, RestrictPtrTraits, index_t>
      vec_mean_;
  const GenericPackedTensorAccessor<accscalar_t, 2, RestrictPtrTraits, index_t>
      vec_invstd_;
  GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t> mean_;
  GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t>
      invstd_;
  GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t>
      running_mean_;
  GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t>
      running_var_;
  const accscalar_t epsilon_;
  const accscalar_t momentum_;
  const GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t>
      counts_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_gather_stats_kernel_template(
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& running_mean_,
    const Tensor& running_var_,
    double momentum,
    double epsilon,
    const Tensor& counts_) {
  Tensor save_mean_;
  Tensor save_invstd_;

  auto features = mean_.size(1);
  auto input_options = mean_.options();
  if (mean_.scalar_type() == at::ScalarType::Half ||
      mean_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  save_mean_ = at::empty({features}, input_options);
  save_invstd_ = at::empty({features}, input_options);

  auto mean =
      packed_accessor_or_dummy<accscalar_t, 2, RestrictPtrTraits, index_t>(
          mean_, "mean");
  auto invstd =
      packed_accessor_or_dummy<accscalar_t, 2, RestrictPtrTraits, index_t>(
          invstd_, "invstd");
  auto running_mean =
      packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(
          running_mean_, "running_mean");
  auto running_var =
      packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(
          running_var_, "running_mean");
  auto counts =
      packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(
          counts_, "counts");

  auto save_mean =
      get_packed_accessor<accscalar_t, 1, RestrictPtrTraits, index_t>(
          save_mean_, "save_mean");
  auto save_invstd =
      get_packed_accessor<accscalar_t, 1, RestrictPtrTraits, index_t>(
          save_invstd_, "save_invstd");

  using KernelClass =
      BatchNormReduceStatisticsKernelFunctor<scalar_t, accscalar_t, index_t>;

  int group_size_x = get_num_threads<KernelClass>(features);
  sycl::range<1> local_range(group_size_x);
  sycl::range<1> global_range(
      group_size_x * std::max<int>(1, features / group_size_x));

  auto caller = KernelClass(
      mean,
      invstd,
      save_mean,
      save_invstd,
      running_mean,
      running_var,
      epsilon,
      momentum,
      counts);
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), caller);

  return std::make_tuple(save_mean_, save_invstd_);
}

} // namespace xpu
} // namespace native
} // namespace at
