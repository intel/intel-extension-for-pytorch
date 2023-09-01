#include <ATen/ATen.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/Resize.h>
#include <core/detail/IndexUtils.h>
#include <oneDNN/oneDNN.h>
#include "Loops.h"
#include "Reduce.h"
#include "ReduceOpStdVar.h"
#include "Resize.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/RegistrationDeclarations.h"
#include "core/MemoryFormat.h"
#include "utils/ComputeEngine.h"
#include "utils/DPCPP.h"
using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {

struct Var {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    return var;
  }
};

struct InvStd {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    T invstd = 0.0f;
    if (var != static_cast<T>(0.0f) || epsilon != static_cast<T>(0.0f)) {
      invstd = static_cast<T>(1.0f) /
          Numerics<T>::sqrt(var + static_cast<T>(epsilon));
    }
    return invstd;
  }
};

inline bool batch_norm_use_channels_last_kernels(const at::Tensor& self) {
  return (
      self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      (self.is_contiguous() && self.strides()[1] == 1) ||
      is_smf_channels_last(self));
}

static int getNumThreads(int nElem, int max_size) {
  int threadSizes[6] = {16, 32, 64, 128, 256, max_size};
  for (int i = 0; i < 6; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return max_size;
}

int get_prefer_simd(int numPlane, int nHw) {
  // decide SIMD: SIMD32 or SIMD16

  auto* dev_prop = dpcppGetDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  auto sub_group_size = dev_prop->subgroup_sizes;
  int simd = sub_group_size[1];
  if (simd <= SIMD16)
    return simd;

  // if max supported simd >16
  if (nHw <= SIMD16)
    return SIMD16;
  if (simd >= SIMD32 && nHw <= SIMD32)
    return SIMD32;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t target_tile_size = dpcppMaxWorkItemsPerTile(dev_id);
  // for work group barrier perf
  int64_t wg_size = dpcppMaxWorkItemsPerEU(dev_id);
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

int get_prefer_wg_size(unsigned int nHw, int simd) {
  if (nHw < simd)
    return simd;
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto size_problem = getNumThreads(nHw, simd * simd);
  auto wg_size = dpcppMaxWorkGroupSize(dev_id);
  return std::min(int64_t(size_problem), wg_size);
}

template <typename scalar_t>
int inline get_nhwc_suggest_vec_size(
    const Tensor input,
    int reduction_size,
    int channels) {
  if (!batch_norm_use_channels_last_kernels(input))
    return 1;
  // no need to vectorize if channels < 16
  if (channels < 16)
    return 1;
  // if small reduction size, make no vectorization for higher occupancy
  if (reduction_size < 8 * dpcppMaxWorkGroupSize())
    return 1;

  // just to load/store data
  auto func = [](scalar_t a) { return a + static_cast<scalar_t>(1.0f); };
  xpu::dpcpp::Array<char*, 1> data;
  data[0] = (char*)input.data_ptr();

  int vec_size = at::native::Memory::can_vectorize_up_to_loop<decltype(func)>(
      dpcppGetDeviceIdOfCurrentQueue(), data);

  // for resnet50 shape, bf16 type, vec 4 have better performance
  if (vec_size == 8 && reduction_size == 256 * 56 * 56 &&
      (channels == 128 || channels == 256))
    return 4;

  return vec_size;
}

inline int rnd_up(const int a, const int b) {
  return (div_up(a, b) * b);
}

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

  if (self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      is_smf_channels_last(self)) {
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

std::tuple<sycl::range<2>, sycl::range<2>> flexible_launch_configs(
    const int reduction,
    const int stride,
    const int loops_per_item = 1,
    const bool coop_flag = false) {
  int wg_size = dpcppMaxWorkItemsPerEU();
  int group_x = std::min(last_pow2(stride), 32);
  int group_y = std::min(
      last_pow2(CeilDiv(reduction, loops_per_item)), wg_size / group_x);
  if (group_x * group_y != wg_size) {
    group_x = std::min(last_pow2(stride), wg_size / group_y);
  }

  int grid_x = CeilDiv(stride, group_x);

  int grid_y = std::min(
      CeilDiv(reduction, group_y * loops_per_item),
      int(dpcppMaxWorkItemsPerTile()) / (grid_x * group_x) /
          (group_y * loops_per_item));
  grid_y = std::max(grid_y, 1);

  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not
    // big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  sycl::range<2> local_range(group_y, group_x);
  sycl::range<2> global_range(grid_y * group_y, grid_x * group_x);

  return std::make_tuple(global_range, local_range);
}

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    bool train,
    typename index_t>
void batch_norm_transform_input_kernel(
    const Tensor input,
    Tensor& output,
    const Tensor& mean_,
    const Tensor& var_or_invstd,
    const Tensor& weight,
    const Tensor& bias,
    stat_accscalar_t epsilon) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int numPlane = input.size(1);
  int64_t target_tile_size = dpcppMaxWorkItemsPerTile(dev_id);
  int64_t wg_size = dpcppMaxWorkItemsPerEU(dev_id); // for work group barrier
  if (wg_size * numPlane < target_tile_size) {
    wg_size = dpcppMaxWorkGroupSize(); // for higher occupancy
  }

  int bs = input.size(0);
  int fs = input.size(2);
  int weight_size = weight.size(0);
  int bias_size = bias.size(0);

  int tf = getNumThreads(fs, wg_size);
  int tb = std::max<int>(wg_size / tf, 1);
  sycl::range<2> local_range(tb, tf);
  sycl::range<2> global_range((bs + tb - 1) / tb * tb, numPlane * tf);

  auto input_ptr = input.data_ptr<input_scalar_t>();
  auto output_ptr = output.data_ptr<input_scalar_t>();
  auto weight_ptr =
      weight.defined() ? weight.data_ptr<stat_scalar_t>() : nullptr;
  auto bias_ptr = bias.defined() ? bias.data_ptr<stat_scalar_t>() : nullptr;
  auto mean_ptr = mean_.data_ptr<stat_accscalar_t>();
  auto var_or_invstd_ptr = var_or_invstd.data_ptr<stat_accscalar_t>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto bias_ptr_ = bias_ptr;
      auto weight_ptr_ = weight_ptr;
      auto group_idx_x = item.get_group().get_group_id(1);
      index_t plane = group_idx_x;

      if (plane >= numPlane) {
        return;
      }

      stat_accscalar_t gamma = weight_ptr != nullptr
          ? static_cast<stat_accscalar_t>(weight_ptr[plane])
          : static_cast<stat_accscalar_t>(1);
      stat_accscalar_t beta = bias_ptr != nullptr
          ? static_cast<stat_accscalar_t>(bias_ptr[plane])
          : static_cast<stat_accscalar_t>(0);

      stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_ptr[plane]);
      stat_accscalar_t invstd;
      if constexpr (train) {
        invstd = var_or_invstd_ptr[plane];
      } else {
        invstd = static_cast<stat_accscalar_t>(1) /
            Numerics<stat_accscalar_t>::sqrt(
                     static_cast<stat_accscalar_t>(var_or_invstd_ptr[plane]) +
                     epsilon);
      }

      index_t bstep = item.get_global_range(0);
      for (index_t batch = item.get_global_id(0); batch < bs; batch += bstep) {
        auto batch_offset = batch * numPlane * fs + plane * fs;
        for (index_t feature = item.get_local_id(1); feature < fs;
             feature += item.get_local_range(1)) {
          output_ptr[batch_offset + feature] = static_cast<input_scalar_t>(
              gamma * (input_ptr[batch_offset + feature] - mean) * invstd +
              beta);
        }
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int vec_size>
void batch_norm_transform_input_channels_last_kernel(
    scalar_t* input_ptr,
    const scalar_t* z_ptr,
    accscalar_t* mean_ptr,
    accscalar_t* inv_std_ptr,
    const layerscalar_t* weight_ptr,
    const layerscalar_t* shift_ptr,
    scalar_t* output_ptr,
    const int reduction_size,
    const int stride,
    const bool fuse_relu) {
  // tensor dimension (m,c)
  // loop along m dimension
  int64_t total_num = reduction_size * stride;
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using vec_s_t =
      at::native::Memory::aligned_vector_loop<accscalar_t, vec_size>;
  auto& queue = dpcppGetCurrentQueue();
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto group_idx_x = item.get_group().get_group_id(1);

      // int inner_loop_stride = item.get_global_range(0);
      // offset along m dimension
      int m_offset = item.get_global_id(0);
      int c_offset_base = item.get_global_id(1) * vec_size;

      if (c_offset_base >= stride || m_offset >= reduction_size) {
        return;
      }

      vec_s_t m_c = *(reinterpret_cast<vec_s_t*>(mean_ptr + c_offset_base));
      vec_s_t inv_vec =
          *(reinterpret_cast<vec_s_t*>(inv_std_ptr + c_offset_base));
      vec_s_t w_c;
      vec_s_t s_c;
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        if (weight_ptr != nullptr) {
          w_c[j] = static_cast<accscalar_t>(weight_ptr[c_offset_base + j]) *
              inv_vec[j];
        } else {
          w_c[j] = (inv_vec[j]);
        }
        if (shift_ptr != nullptr) {
          s_c[j] = shift_ptr[c_offset_base + j];
        } else {
          s_c[j] = static_cast<accscalar_t>(0.0f);
        }
      }

      int address_base = m_offset * stride + c_offset_base;
      int address_increment = item.get_global_range(0) * stride;

      vec_t output_vec;
      for (; address_base < total_num; address_base += address_increment) {
        vec_t x_math_vec =
            *(reinterpret_cast<vec_t*>(input_ptr + address_base));
#pragma unroll
        for (int j = 0; j < vec_size; j++) {
          auto c_offset = c_offset_base + j;

          output_vec[j] =
              w_c[j] * (static_cast<accscalar_t>(x_math_vec[j]) - m_c[j]) +
              s_c[j];
        }
        *(reinterpret_cast<vec_t*>(output_ptr + address_base)) = output_vec;
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
void batch_norm_elemt_channels_first_template(
    Tensor& output_,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& mean_,
    const Tensor& invstd_) {
  using stat_accscalar_t = acc_type<stat_scalar_t>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  // NOTE: We use transform_input_kernel in training mode, which ignores
  // epsilon
  const double dummy_epsilon = 1e-5;

  batch_norm_transform_input_kernel<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      true,
      index_t>(
      input_reshaped,
      output_reshaped,
      mean_,
      invstd_,
      weight_,
      bias_,
      dummy_epsilon);
}

void batch_norm_elemt_channels_last_template(
    Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& shift, // bias of BN
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::optional<at::Tensor>& z = c10::nullopt, // bias after BN
    const bool fuse_relu = false) {
  const auto second_dtype = weight.defined()
      ? weight.scalar_type()
      : (shift.defined() ? shift.scalar_type() : input.scalar_type());
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

#define DISPATCH_TRANSFORM_INPUT_IMPL(vec_size)                   \
  {                                                               \
    batch_norm_transform_input_channels_last_kernel<              \
        scalar_t,                                                 \
        accscalar_t,                                              \
        scalar_t,                                                 \
        vec_size>(                                                \
        input.data_ptr<scalar_t>(),                               \
        z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr, \
        mean.data_ptr<accscalar_t>(),                             \
        inv_std.data_ptr<accscalar_t>(),                          \
        weight.defined() ? weight.data_ptr<scalar_t>() : nullptr, \
        shift.defined() ? shift.data_ptr<scalar_t>() : nullptr,   \
        output.data_ptr<scalar_t>(),                              \
        reduction_size,                                           \
        stride,                                                   \
        fuse_relu);                                               \
  }

#define DISPATCH_TRANSFORM_ACC_INPUT_IMPL(vec_size)                  \
  {                                                                  \
    batch_norm_transform_input_channels_last_kernel<                 \
        scalar_t,                                                    \
        accscalar_t,                                                 \
        accscalar_t,                                                 \
        vec_size>(                                                   \
        input.data_ptr<scalar_t>(),                                  \
        z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr,    \
        mean.data_ptr<accscalar_t>(),                                \
        inv_std.data_ptr<accscalar_t>(),                             \
        weight.defined() ? weight.data_ptr<accscalar_t>() : nullptr, \
        shift.defined() ? shift.data_ptr<accscalar_t>() : nullptr,   \
        output.data_ptr<scalar_t>(),                                 \
        reduction_size,                                              \
        stride,                                                      \
        fuse_relu);                                                  \
  }

  if (input.scalar_type() != second_dtype) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward", [&] {
          using accscalar_t = acc_type<scalar_t>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);
          switch (suggest_vec_size) {
            case 8: {
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(8);
              break;
            }
            case 4: {
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(4);
              break;
            }
            default:
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(1);
          }
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
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward", [&] {
          using accscalar_t = acc_type<scalar_t>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);
          switch (suggest_vec_size) {
            case 8: {
              DISPATCH_TRANSFORM_INPUT_IMPL(8);
              break;
            }
            case 4: {
              DISPATCH_TRANSFORM_INPUT_IMPL(4);
              break;
            }
            default:
              DISPATCH_TRANSFORM_INPUT_IMPL(1);
          }
        });
  }
#undef DISPATCH_TRANSFORM_INPUT_IMPL
#undef DISPATCH_TRANSFORM_ACC_INPUT_IMPL
}

void batch_norm_elementwise(
    Tensor& out,
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
      resize_output(out, self.sizes());
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16, kHalf, self.scalar_type(), "batch_norm_elementwise", [&] {
            using accscalar_t = acc_type<scalar_t>;
            const bool mixed_type = is_mixed_type(self, *weight, *bias);
            if (mixed_type) {
              batch_norm_elemt_channels_first_template<
                  scalar_t,
                  accscalar_t,
                  int32_t>(out, self, *weight, *bias, mean_, invstd_);
            } else {
              batch_norm_elemt_channels_first_template<
                  scalar_t,
                  scalar_t,
                  int32_t>(out, self, *weight, *bias, mean_, invstd_);
            }
          });
      return;
    }
    case Impl::ChannelsLast: {
      auto weight = at::borrow_from_optional_tensor(weight_opt);
      auto bias = at::borrow_from_optional_tensor(bias_opt);

      if (resize_output_check(out, self.sizes())) {
        resize_impl(out.unsafeGetTensorImpl(), self.sizes(), self.strides());
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
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector sizes(ndim, 1), strides(ndim, 0);
      // Helper to convert 1d tensors to an nd tensor that broadcasts with
      // input All elements go into the channel dimension
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

      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16, kHalf, self.scalar_type(), "batch_norm_elementwise", [&] {
            using acc_t = acc_type<scalar_t>;
            dpcpp_kernel_for_tensor_iter(
                iter,
                [](scalar_t input,
                   acc_t weight,
                   acc_t bias,
                   acc_t mean,
                   acc_t invstd) -> scalar_t {
                  return ((input - mean) * invstd) * weight + bias;
                });
          });
      return;
    }
  }
}

template <
    int SIMD,
    typename VarTransform,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_collect_statistics_kernel(
    int N,
    int numPlane,
    int Hw,
    const input_scalar_t* input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    stat_accscalar_t* save_mean,
    stat_accscalar_t* save_transformed_var) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t wg_size = get_prefer_wg_size(N * Hw, SIMD);
  int64_t work_group_size_x = getNumThreads(Hw, wg_size);
  int64_t work_group_size_y = std::max(int64_t(1), wg_size / work_group_size_x);
  work_group_size_y = std::min(int64_t(N), work_group_size_y);
  int64_t sg_num = work_group_size_x * work_group_size_y / SIMD;
  auto batch_stride = numPlane * Hw;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto shared_n =
        dpcpp_local_acc_t<stat_accscalar_t, 1>(sycl::range<1>{sg_num}, cgh);
    auto shared_avg_var = dpcpp_local_acc_t<stat_accscalar_t, 1>(
        sycl::range<1>{sg_num * 2 + 2}, cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item)
        [[intel::reqd_sub_group_size(SIMD)]] {
      int plane = item.get_group(0);
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
      // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a
      // description.

      // first the reductions each thread does separately
      stat_accscalar_t avg = 0;
      stat_accscalar_t var_n = 0;
      int n = 0;
      for (int batch = item.get_local_id(0); batch < N;
           batch += item.get_local_range(0)) {
        for (int x = item.get_local_id(1); x < Hw;
             x += item.get_local_range(1)) {
          auto offset = batch * batch_stride + plane * Hw + x;
          stat_accscalar_t v = input[offset];
          stat_accscalar_t d1 = v - avg;
          n++;
          avg += d1 / n;
          var_n += d1 * (v - avg);
        }
      }

      // first warpSum to get one value per thread to
      // one value per warp
#pragma unroll
      for (int i = 1; i < SIMD; i <<= 1) {
        stat_accscalar_t o_avg = sg.shuffle_xor(avg, i);
        int o_n = sg.shuffle_xor(n, i);
        stat_accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
        var_n += sg.shuffle_xor(var_n, i) +
            (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
      }

      // this writes each warps item into shared memory

      if (sg_lid == 0) {
        shared_n[sg_id] = n;
        shared_avg_var[sg_id * 2] = avg;
        shared_avg_var[sg_id * 2 + 1] = var_n;
      }
      item.barrier(dpcpp_local_fence);
      // now have a second warpSum to reduce the intermediate values
      // from shared memory to a single number. The very first
      // thread writes it to shared memory.

      if (tid < sg_num) {
        n = shared_n[tid];
        avg = shared_avg_var[2 * tid];
        var_n = shared_avg_var[2 * tid + 1];
      } else {
        n = 0;
        avg = stat_accscalar_t(0);
        var_n = stat_accscalar_t(0);
      }
#pragma unroll
      for (int i = 1; i < SIMD; i <<= 1) {
        stat_accscalar_t o_avg = sg.shuffle_xor(avg, i);
        int o_n = sg.shuffle_xor(n, i);
        stat_accscalar_t factor = 1.0f / fmaxf(1.0f, n + o_n);
        auto o_var_n = sg.shuffle_xor(var_n, i);
        var_n += sg.shuffle_xor(var_n, i) +
            (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
      }

      // Save the mean, variance, and moving averages
      if (tid == 0) {
        if (save_mean != nullptr) {
          save_mean[plane] = avg;
        }
        if (save_transformed_var != nullptr) {
          save_transformed_var[plane] =
              VarTransform{}(var_n / (N * Hw), epsilon);
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(numPlane * work_group_size_y, work_group_size_x),
            sycl::range<2>(work_group_size_y, work_group_size_x)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T, typename C>
void inline welford_merge_element(
    C& count,
    T& mean,
    T& m2n,
    const C& count_new,
    const T& mean_new,
    const T& m2n_new) {
  T factor = T(1.0f) / std::max(1, (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

template <typename T, typename C, typename C_slm_ptr, typename T_slm_ptr>
void inline welford_merge_group_vertical(
    sycl::nd_item<2> item,
    C& count,
    T& mean,
    T& m2n,
    C_slm_ptr& shmem_count,
    T_slm_ptr& shmem_mean,
    T_slm_ptr& shmem_m2n) {
  // write to shared memory
  auto address_base = item.get_local_linear_id();
  auto local_range_y = item.get_local_range(0);
  auto local_range_x = item.get_local_range(1);
  auto liy = item.get_local_id(0);
  auto lix = item.get_local_id(1);

  for (int offset = local_range_y / 2; offset > 0; offset >>= 1) {
    if (liy < offset * 2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    item.barrier(dpcpp_local_fence);
    if (liy < offset && liy + offset < local_range_y) {
      auto address = address_base + offset * local_range_x;
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, count_new, mean_new, m2n_new);
    }
  }
}

template <typename VarTransform, typename scalar_t, typename stat_accscalar_t>
void batch_norm_update_mean_var_kernel(
    scalar_t* mean_,
    scalar_t* var_,
    int channel_num,
    scalar_t factor,
    stat_accscalar_t epsilon) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t wg_size = std::min(
      int64_t(channel_num),
      dpcppMaxWorkItemsPerEU(dev_id)); // for work group barrier

  sycl::range<1> local_range(wg_size);
  sycl::range<1> global_range((channel_num + wg_size - 1) / wg_size * wg_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto c_offset = item.get_global_linear_id();
      if (c_offset < channel_num) {
        scalar_t mean = mean_[c_offset] * factor;

        mean_[c_offset] = mean;
        var_[c_offset] =
            VarTransform{}(var_[c_offset] * factor - mean * mean, epsilon);
      }
    };
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

// sum x and x^2 in channels
template <
    typename scalar_t,
    typename accscalar_t,
    int vec_size,
    bool two_pass_reduce>
void batch_norm_reduce_sum_channels_last_kernel(
    const Tensor input,
    Tensor& out_mean,
    Tensor& out_invstd,
    const int reduction_size,
    const int stride) {
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size, true);
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& queue = dpcppGetCurrentQueue();
  auto global_range_y = global_range[0];
  auto local_range_y = local_range[0];
  int group_num_x = global_range[1] / local_range[1];
  int group_num_y = global_range[0] / local_range[0];
  Tensor temp_sum, temp_sum_sq;
  accscalar_t* temp_sum_ptr = nullptr;
  accscalar_t* temp_sum_sq_ptr = nullptr;
  if constexpr (two_pass_reduce) {
    out_mean.zero_();
    out_invstd.zero_();
    temp_sum = at::empty({group_num_y * stride}, out_mean.options());
    temp_sum_sq = at::empty({group_num_y * stride}, out_mean.options());
    temp_sum_ptr = temp_sum.data_ptr<accscalar_t>();
    temp_sum_sq_ptr = temp_sum_sq.data_ptr<accscalar_t>();
  }
  int wg_size = local_range[0] * local_range[1];

  auto input_ptr = input.data_ptr<scalar_t>();
  auto out_mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto out_invstd_ptr = out_invstd.data_ptr<accscalar_t>();

  int loop_count = 1 + (reduction_size - 1) / (global_range_y);
  using vec_y = at::detail::Array<accscalar_t, 2>;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto shared = dpcpp_local_acc_t<vec_y, 1>(sycl::range<1>{wg_size}, cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      int plane = item.get_group(0);
      int tid = item.get_local_linear_id();
      auto sg = item.get_sub_group();

      // offset along m dimension
      int m_offset = item.get_global_id(0);
      int c_offset_base = item.get_global_id(1) * vec_size;

      int thread_idx_y = item.get_local_id(0);
      int thread_idx_x = item.get_local_id(1);
      int group_idx_y = item.get_group(0);
      int group_idx_x = item.get_group(1);

      int address_base = m_offset * stride + c_offset_base;
      int inner_loop_stride = global_range_y;
      int address_increment = inner_loop_stride * stride;

      accscalar_t x_sum[vec_size] = {0.0f};
      accscalar_t x_sq_sum[vec_size] = {0.0f};
      // thread reduction
      for (int i = 0; i < loop_count; i++) {
        vec_t x_math_vec =
            *(reinterpret_cast<vec_t*>(input_ptr + address_base));
#pragma unroll
        for (int j = 0; j < vec_size; j++) {
          auto c_offset = c_offset_base + j;

          if (c_offset < stride && m_offset < reduction_size) {
            scalar_t arr = input_ptr[address_base + j];
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
            item, shared, value, [](accscalar_t a, accscalar_t b) {
              return a + b;
            });

        x_sum[j] = value[0];
        x_sq_sum[j] = value[1];

        item.barrier(dpcpp_local_fence);
      }

#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        auto c_offset = c_offset_base + j;
        // global_reduciton
        if (thread_idx_y == 0 && c_offset < stride) {
          if constexpr (two_pass_reduce) {
            // write to temp[c][group_idx_y]
            int offset = c_offset * group_num_y + group_idx_y;
            temp_sum_ptr[c_offset * group_num_y + group_idx_y] = x_sum[j];
            temp_sum_sq_ptr[c_offset * group_num_y + group_idx_y] = x_sq_sum[j];
          } else {
            out_mean_ptr[c_offset] = x_sum[j];
            out_invstd_ptr[c_offset] = x_sq_sum[j];
          }
        }
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);

  // reduce temp sum
  if constexpr (two_pass_reduce) {
    int wg_size = std::min(group_num_y, int(dpcppMaxWorkItemsPerEU()));
    auto cgf_2 = DPCPP_Q_CGF(__cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
        auto local_id = item_id.get_local_linear_id();
        auto global_id = item_id.get_global_linear_id();
        auto c_offset = item_id.get_group_linear_id();

        accscalar_t temp_sum_val = 0.0f;
        accscalar_t temp_sum_sq_val = 0.0f;
        for (int i = local_id; i < group_num_y; i += wg_size) {
          int offset = c_offset * group_num_y + i;
          temp_sum_val += temp_sum_ptr[offset];
          temp_sum_sq_val += temp_sum_sq_ptr[offset];
        }
        auto total_sum = sycl::reduce_over_group(
            item_id.get_group(), temp_sum_val, sycl::plus<accscalar_t>());
        auto total_sum_sq = sycl::reduce_over_group(
            item_id.get_group(), temp_sum_sq_val, sycl::plus<accscalar_t>());
        if (local_id == 0) {
          out_mean_ptr[c_offset] = total_sum;
          out_invstd_ptr[c_offset] = total_sum_sq;
        }
      };
      __cgh.parallel_for(sycl::nd_range<1>(stride * wg_size, wg_size), kfn);
    };
    DPCPP_Q_SUBMIT(queue, cgf_2);
  }
}

template <typename scalar_t, typename index_t, typename VarTransform>
void batch_norm_stats_channels_first_template(
    const Tensor& out_mean,
    const Tensor& out_invstd,
    const Tensor& input_,
    double epsilon) {
  using accscalar_t = acc_type<scalar_t>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  int N = input_reshaped.size(0);
  int C = input_reshaped.size(1);
  int Hw = input_reshaped.size(2);

  at::native::resize_output(out_mean, {n_input});
  at::native::resize_output(out_invstd, {n_input});
  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  auto input_ptr = input_reshaped.data_ptr<scalar_t>();
  auto mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto invstd_ptr = out_invstd.data_ptr<accscalar_t>();
  int simd = get_prefer_simd(C, N * Hw);
  if (simd == SIMD32) {
    batch_norm_collect_statistics_kernel<
        SIMD32,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>(N, C, Hw, input_ptr, epsilon, 0.0, mean_ptr, invstd_ptr);
  } else {
    batch_norm_collect_statistics_kernel<
        SIMD16,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>(N, C, Hw, input_ptr, epsilon, 0.0, mean_ptr, invstd_ptr);
  }
}

template <typename scalar_t, typename VarTransform>
void batch_norm_stats_channels_last_template(
    Tensor& out_mean,
    Tensor& out_invstd,
    const Tensor& input,
    double epsilon) {
  using accscalar_t = acc_type<scalar_t>;

  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  resize_output(out_mean, {stride});
  resize_output(out_invstd, {stride});
  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  int suggest_vec_size =
      get_nhwc_suggest_vec_size<scalar_t>(input, reduction_size, stride);

#define DISPATCH_REDUCE_2_PASS_IMPL(vec_size)                       \
  {                                                                 \
    batch_norm_reduce_sum_channels_last_kernel<                     \
        scalar_t,                                                   \
        accscalar_t,                                                \
        vec_size,                                                   \
        true>(input, out_mean, out_invstd, reduction_size, stride); \
  }

#define DISPATCH_REDUCE_IMPL(vec_size)                               \
  {                                                                  \
    batch_norm_reduce_sum_channels_last_kernel<                      \
        scalar_t,                                                    \
        accscalar_t,                                                 \
        vec_size,                                                    \
        false>(input, out_mean, out_invstd, reduction_size, stride); \
  }
  sycl::range<2> global_range(1, 1), local_range(1, 1);

  if (stride % suggest_vec_size == 0 && suggest_vec_size == 8) {
    constexpr int vec_size = 8;

    std::tie(global_range, local_range) =
        flexible_launch_configs(reduction_size, stride / vec_size, true);
    int group_num_y = global_range[0] / local_range[0];
    if (group_num_y > 1) {
      DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
    } else {
      DISPATCH_REDUCE_IMPL(vec_size);
    }
  } else if (stride % suggest_vec_size == 0 && suggest_vec_size == 4) {
    constexpr int vec_size = 4;

    std::tie(global_range, local_range) =
        flexible_launch_configs(reduction_size, stride / vec_size, true);
    int group_num_y = global_range[0] / local_range[0];
    if (group_num_y > 1) {
      DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
    } else {
      DISPATCH_REDUCE_IMPL(vec_size);
    }
  } else {
    constexpr int vec_size = 1;

    std::tie(global_range, local_range) =
        flexible_launch_configs(reduction_size, stride / vec_size, true);
    int group_num_y = global_range[0] / local_range[0];
    if (group_num_y > 1) {
      DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
    } else {
      DISPATCH_REDUCE_IMPL(vec_size);
    }
  }

  auto out_mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto out_invstd_ptr = out_invstd.data_ptr<accscalar_t>();
  const auto factor = static_cast<accscalar_t>(1.0f / reduction_size);
  batch_norm_update_mean_var_kernel<VarTransform>(
      out_mean_ptr, out_invstd_ptr, stride, factor, epsilon);
#undef DISPATCH_REDUCE_2_PASS_IMPL
#undef DISPATCH_REDUCE_IMPL
}

void batch_norm_mean_var(
    const Tensor& self,
    Tensor& save_mean,
    Tensor& save_var) {
  // NOTE: Epsilon is only used for InvStd, not Var. The value here is
  // ignored.
  const double dummy_epsilon = 1e-5;
  switch (batch_norm_choose_impl(self)) {
    case Impl::Contiguous: {
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats", [&] {
            batch_norm_stats_channels_first_template<scalar_t, int32_t, Var>(
                save_mean, save_var, self, dummy_epsilon);
          });
      return;
    }
    case Impl::ChannelsLast: {
      if ((!save_mean.defined() || save_mean.is_contiguous()) &&
          (!save_var.defined() || save_var.is_contiguous())) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats", [&] {
              batch_norm_stats_channels_last_template<scalar_t, Var>(
                  save_mean, save_var, self, dummy_epsilon);
            });
        return;
      }
      C10_FALLTHROUGH;
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector reduce_dims(ndim - 1);
      reduce_dims[0] = 0;
      for (int64_t i = 2; i < ndim; ++i) {
        reduce_dims[i - 1] = i;
      }

      at::AtenIpexTypeXPU::std_var_mean_out(
          "var_mean", save_var, save_mean, self, reduce_dims, 0, false, false);
    }
  }
}

static inline at::Tensor condition_contiguous(const at::Tensor& t) {
  auto ndim = t.ndimension();
  if (!t.defined()) {
    return t;
  }

  if (t.defined() && !is_smf_channels_last(t)) {
    return t.contiguous();
  }

  auto cl_tag = get_cl_tag_by_ndim(t.ndimension());
  if (CHANNELSLAST1D_DPCPP == cl_tag) {
    auto tmp = t.contiguous();
    return convert_tensor_to_channels_last_1d(tmp);
  }

  return t.contiguous(cl_tag);
}

void batch_norm_update_stats(
    const Tensor& save_mean,
    const Tensor& save_var,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum_,
    int64_t N) {
  auto iter = TensorIteratorConfig()
                  .add_output(running_mean)
                  .add_output(running_var)
                  .add_input(save_mean)
                  .add_input(save_var)
                  .add_input(running_mean)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      running_mean.scalar_type(),
      "batch_norm_update_stats",
      [&] {
        using acc_t = acc_type<scalar_t>;
        const auto bessel_correction_factor = static_cast<acc_t>(
            static_cast<double>(N) / static_cast<double>(N - 1));
        const auto momentum = static_cast<acc_t>(momentum_);

        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter,
            [=](acc_t mean,
                acc_t var,
                scalar_t running_mean,
                scalar_t running_var) -> std::tuple<scalar_t, scalar_t> {
              const auto unbiased_var = var * bessel_correction_factor;
              return std::tuple<scalar_t, scalar_t>{
                  mean * momentum + (1 - momentum) * running_mean,
                  unbiased_var * momentum + (1 - momentum) * running_var,
              };
            });
      });
}

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
                  .add_input(save_mean)
                  .add_input(save_var)
                  .add_input(running_mean)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      running_mean.scalar_type(),
      "batch_norm_update_stats_and_invert",
      [&] {
        using acc_t = acc_type<scalar_t>;
        const auto bessel_correction_factor = static_cast<acc_t>(
            static_cast<double>(N) / static_cast<double>(N - 1));
        const auto eps = static_cast<acc_t>(epsilon);
        const auto momentum = static_cast<acc_t>(momentum_);

        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter,
            [=](acc_t mean,
                acc_t var,
                scalar_t running_mean,
                scalar_t running_var) -> std::tuple<scalar_t, scalar_t, acc_t> {
              const auto unbiased_var = var * bessel_correction_factor;
              return std::tuple<scalar_t, scalar_t, scalar_t>{
                  mean * momentum + (1 - momentum) * running_mean,
                  unbiased_var * momentum + (1 - momentum) * running_var,
                  Numerics<acc_t>::rsqrt(var + eps)};
            });
      });
}

void batch_norm_calc_invstd(
    const Tensor& out_invstd,
    const Tensor& running_var,
    double epsilon) {
  auto iter = TensorIteratorConfig()
                  .add_output(out_invstd)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      running_var.scalar_type(),
      "batch_norm_invert_std",
      [&] {
        using acc_t = acc_type<scalar_t>;
        auto eps = static_cast<acc_t>(epsilon);
        dpcpp_kernel_for_tensor_iter(iter, [eps](scalar_t var) -> acc_t {
          return Numerics<acc_t>::rsqrt(var + eps);
        });
      });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(
    const Tensor& input,
    const optional<at::Tensor>& weight_opt,
    const optional<at::Tensor>& bias_opt,
    const optional<at::Tensor>& running_mean_opt,
    const optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double epsilon,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_invstd) {
  xpu::COMPUTE_ENG real_eng;

  if (input.is_quantized()) {
    real_eng = xpu::COMPUTE_ENG::ONEDNN;
  } else {
    real_eng = choose_compute_eng(xpu::COMPUTE_ENG::BASIC, input);
  }

  if (xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    c10::MaybeOwned<Tensor> weight_maybe_owned =
        at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_maybe_owned;

    c10::MaybeOwned<Tensor> bias_maybe_owned =
        at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;

    c10::MaybeOwned<Tensor> running_mean_maybe_owned =
        at::borrow_from_optional_tensor(running_mean_opt);
    const Tensor& running_mean = *running_mean_maybe_owned;

    c10::MaybeOwned<Tensor> running_var_maybe_owned =
        at::borrow_from_optional_tensor(running_var_opt);
    const Tensor& running_var = *running_var_maybe_owned;

    if (running_mean.defined() && running_var.defined()) {
      checkBackend(
          "batch_norm",
          {input, weight, bias, running_mean, running_var},
          Backend::XPU);
    } else {
      checkBackend("batch_norm", {input, weight, bias}, Backend::XPU);
    }

    if (input.scalar_type() != at::ScalarType::Float &&
        input.scalar_type() != at::ScalarType::Half &&
        input.scalar_type() != at::ScalarType::BFloat16) {
      std::stringstream ss;
      ss << "DPCPP batch_norm backend got unsupported type="
         << input.scalar_type();
      TORCH_CHECK(0, ss.str());
    }

    xpu::oneDNN::batch_normalization(
        condition_contiguous(input),
        condition_contiguous(weight),
        condition_contiguous(bias),
        condition_contiguous(running_mean),
        condition_contiguous(running_var),
        training,
        momentum,
        epsilon,
        out,
        save_mean,
        save_invstd);

    // Update running_mean and running_var
    if (training && running_mean.defined() && running_var.defined()) {
      const int64_t N = input.size(1);
      batch_norm_update_stats(
          save_mean, save_invstd, running_mean, running_var, momentum, N);
    }

  } else {
    const bool has_running_mean =
        (running_mean_opt.has_value() && running_mean_opt->defined());
    const bool has_running_var =
        (running_var_opt.has_value() && running_var_opt->defined());
    TORCH_CHECK(has_running_mean == has_running_var);
    if (training) {
      batch_norm_mean_var(input, save_mean, save_invstd);
      if (has_running_mean) {
        const int64_t N = input.numel() / save_mean.numel();
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
      resize_output(save_mean, running_mean_opt->sizes());
      save_mean.copy_(*running_mean_opt, /*non_blocking=*/true);
      batch_norm_calc_invstd(save_invstd, running_var_opt.value(), epsilon);
    }
    batch_norm_elementwise(
        out, input, weight_opt, bias_opt, save_mean, save_invstd);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& src,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double epsilon) {
  Tensor dst = at::empty_like(src);
  int n_input = src.size(1);
  auto options = src.options().dtype(toAccumulateType(src.scalar_type()));
  Tensor save_mean = at::empty({n_input}, options);
  Tensor save_var = at::empty({n_input}, options);

  native_batch_norm_out(
      src,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      training,
      momentum,
      epsilon,
      dst,
      save_mean,
      save_var);
  return std::tuple<Tensor&, Tensor&, Tensor&>(dst, save_mean, save_var);
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

template <
    typename scalar_t,
    typename accscalar_t,
    typename tensor_ptr,
    typename tensor_strides>
struct GradOp {
  GradOp(
      accscalar_t m,
      const tensor_ptr& i,
      const tensor_ptr& g,
      const tensor_strides& i_batch,
      const tensor_strides& i_plane,
      const tensor_strides& i_feature,
      const tensor_strides& go_batch,
      const tensor_strides& go_plane,
      const tensor_strides& go_feature)
      : mean(m),
        input(i),
        grad_output(g),
        i_batch_stride(i_batch),
        i_plane_stride(i_plane),
        i_feature_stride(i_feature),
        go_batch_stride(go_batch),
        go_plane_stride(go_plane),
        go_feature_stride(go_feature) {}
  inline Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output
        [batch * go_batch_stride + plane * go_plane_stride +
         n * go_feature_stride];
    accscalar_t c = static_cast<accscalar_t>(
                        input
                            [batch * i_batch_stride + plane * i_plane_stride +
                             n * i_feature_stride]) -
        mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const tensor_ptr& input;
  const tensor_ptr& grad_output;
  const tensor_strides& i_batch_stride;
  const tensor_strides& i_plane_stride;
  const tensor_strides& i_feature_stride;
  const tensor_strides& go_batch_stride;
  const tensor_strides& go_plane_stride;
  const tensor_strides& go_feature_stride;
};

template <
    int SIMD,
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void group_reduce(
    nd_item_id item_id,
    int sub_group_num,
    accscalar_t& val,
    accscalar_t init,
    const local_shared& local_data,
    reduce_op bin_op) {
  auto sg = item_id.get_sub_group();
  uint32_t lane_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();

  // dynamic get SIMD width result in big performance drop
  // uint32_t SIMD = sg.get_local_range()[0];
#pragma unroll
  for (int i = 1; i < SIMD; i <<= 1) {
    val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
  }
  if (sub_group_num == 1) {
    if (lane_id == 0) {
      local_data[0] = val;
    }
    item_id.barrier(dpcpp_local_fence);
    val = local_data[0];

    return;
  }

  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
  if (lane_id == 0) {
    local_data[sg_id] = val;
  }
  item_id.barrier(dpcpp_local_fence);

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
      val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
      if (i >= ((sub_group_num + 1) >> 1))
        break;
    }

    // the 0th WI (the 0th WI in the 0th sub_group) generate the final
    // result
    if (lane_id == 0) {
      local_data[0] = val;
    }
  }

  item_id.barrier(dpcpp_local_fence);
  val = local_data[0];
}

template <
    int SIMD,
    typename scalar_t,
    typename item_t,
    typename Op,
    typename local_shared>
scalar_t plane_reduce(
    item_t item,
    Op grad_op,
    int batch_size,
    int hw,
    int plane,
    int sub_group_num,
    const local_shared& shared) {
  // first the reductions each thread does separately
  scalar_t sum_value = static_cast<scalar_t>(0);
  for (int batch = item.get_local_id(0); batch < batch_size;
       batch += item.get_local_range(0)) {
    for (int x = item.get_local_id(1); x < hw; x += item.get_local_range(1)) {
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
  item.barrier(dpcpp_local_fence);
  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_backward_channels_first_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor save_invstd,
    bool train,
    stat_accscalar_t epsilon) {
  using accscalar_t = acc_type<stat_scalar_t>;
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto N = grad_output.size(0);
  auto numPlane = grad_output.size(1);
  auto Hw = grad_output.size(2);
  index_t numel = grad_output.size(0) * grad_output.size(2);

  int64_t wg_size = get_prefer_wg_size(N * Hw, SIMD);

  int64_t work_group_size_x = getNumThreads(Hw, wg_size);
  int64_t work_group_size_y = std::max(int64_t(1), wg_size / work_group_size_x);
  int sg_num = work_group_size_x * work_group_size_y / SIMD;
  index_t gi_batch_stride = grad_input.size(0) > 0 ? grad_input.stride(0) : 0;
  index_t gi_plane_stride = grad_input.size(0) > 0 ? grad_input.stride(1) : 0;
  index_t go_batch_stride = grad_output.stride(0);
  index_t go_plane_stride = grad_output.stride(1);
  index_t go_feature_stride = grad_output.stride(2);
  index_t input_batch_stride = input.stride(0);
  index_t input_plane_stride = input.stride(1);
  index_t input_feature_stride = input.stride(2);

  auto input_ptr = input.data_ptr<input_scalar_t>();
  auto grad_output_ptr = grad_output.data_ptr<input_scalar_t>();
  input_scalar_t* grad_input_ptr =
      grad_input.numel() > 0 ? grad_input.data_ptr<input_scalar_t>() : nullptr;
  stat_scalar_t* weight_ptr =
      weight.numel() > 0 ? weight.data_ptr<stat_scalar_t>() : nullptr;
  stat_scalar_t* grad_weight_ptr =
      grad_weight.numel() > 0 ? grad_weight.data_ptr<stat_scalar_t>() : nullptr;
  stat_scalar_t* grad_bias_ptr =
      grad_bias.numel() > 0 ? grad_bias.data_ptr<stat_scalar_t>() : nullptr;
  stat_scalar_t* running_mean_ptr =
      running_mean.defined() ? running_mean.data_ptr<stat_scalar_t>() : nullptr;
  stat_scalar_t* running_var_ptr =
      running_var.defined() ? running_var.data_ptr<stat_scalar_t>() : nullptr;
  accscalar_t* save_mean_ptr =
      save_mean.defined() ? save_mean.data_ptr<accscalar_t>() : nullptr;
  accscalar_t* save_invstd_ptr =
      save_invstd.defined() ? save_invstd.data_ptr<accscalar_t>() : nullptr;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_sum =
        dpcpp_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>, 1>(
            sycl::range<1>{work_group_size_x * work_group_size_y}, cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item)
        [[intel::reqd_sub_group_size(SIMD)]] {
      index_t plane = item.get_group(0);
      auto lix = item.get_local_id(1);
      auto liy = item.get_local_id(0);
      auto local_range_y = item.get_local_range(0);
      auto local_range_x = item.get_local_range(1);

      stat_accscalar_t mean, invstd;
      if (train) {
        mean = save_mean_ptr[plane];
        invstd = save_invstd_ptr[plane];
      } else {
        mean = static_cast<stat_accscalar_t>(running_mean_ptr[plane]);
        invstd = static_cast<stat_accscalar_t>(1) /
            Numerics<stat_accscalar_t>::sqrt(
                     static_cast<stat_accscalar_t>(running_var_ptr[plane]) +
                     epsilon);
      }

      stat_accscalar_t weight_val = weight_ptr != nullptr
          ? static_cast<stat_accscalar_t>(weight_ptr[plane])
          : stat_accscalar_t(1);
      stat_accscalar_t norm = stat_accscalar_t(1) / numel;

      // Compute two values across (batch, x/y/z) in one pass:
      // 1. Sum(grad_output)
      // 2. DotProduct(input - mean, grad_output)
      GradOp<
          input_scalar_t,
          stat_accscalar_t,
          decltype(input_ptr),
          decltype(input_batch_stride)>
          g(mean,
            input_ptr,
            grad_output_ptr,
            input_batch_stride,
            input_plane_stride,
            input_feature_stride,
            go_batch_stride,
            go_plane_stride,
            go_feature_stride);
      auto res = plane_reduce<SIMD, Float2<input_scalar_t, stat_accscalar_t>>(
          item, g, N, Hw, plane, sg_num, local_sum);
      stat_accscalar_t grad_output_sum = res.v1;
      stat_accscalar_t dot_p = res.v2;

      stat_accscalar_t grad_mean = grad_output_sum * norm;
      stat_accscalar_t proj_scale = dot_p * norm * invstd * invstd;
      stat_accscalar_t grad_scale = invstd * weight_val;

      if (grad_input_ptr != nullptr) {
        for (int batch = liy; batch < N; batch += local_range_y) {
          for (int x = lix; x < Hw; x += local_range_x) {
            input_scalar_t go = grad_output_ptr
                [batch * go_batch_stride + plane * go_plane_stride + x];
            if (train) {
              stat_accscalar_t inp = input_ptr
                  [batch * input_batch_stride + plane * input_plane_stride + x];
              stat_accscalar_t proj = (inp - mean) * proj_scale;
              grad_input_ptr
                  [batch * gi_batch_stride + plane * gi_plane_stride + x] =
                      static_cast<input_scalar_t>(
                          (go - proj - grad_mean) * grad_scale);
            } else {
              grad_input_ptr
                  [batch * gi_batch_stride + plane * gi_plane_stride + x] =
                      static_cast<input_scalar_t>(go * grad_scale);
            }
          }
        }
      }

      if (grad_weight_ptr != nullptr) {
        if (lix == 0) {
          grad_weight_ptr[plane] = static_cast<stat_scalar_t>(dot_p * invstd);
        }
      }

      if (grad_bias_ptr != nullptr) {
        if (lix == 0) {
          grad_bias_ptr[plane] = static_cast<stat_scalar_t>(grad_output_sum);
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(numPlane * work_group_size_y, work_group_size_x),
            sycl::range<2>(work_group_size_y, work_group_size_x)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

// only for channels first
template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_channels_first_template(
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
  using accscalar_t = acc_type<stat_scalar_t>;
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

  int simd = get_prefer_simd(
      input_reshaped.size(1), input_reshaped.size(0) * input_reshaped.size(1));
  if (simd == SIMD32) {
    batch_norm_backward_channels_first_kernel<
        SIMD32,
        input_scalar_t,
        stat_scalar_t,
        accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        grad_input_reshaped,
        grad_weight_,
        grad_bias_,
        weight_,
        running_mean_,
        running_var_,
        save_mean_,
        save_invstd_,
        train,
        epsilon);
  } else {
    batch_norm_backward_channels_first_kernel<
        SIMD16,
        input_scalar_t,
        stat_scalar_t,
        accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        grad_input_reshaped,
        grad_weight_,
        grad_bias_,
        weight_,
        running_mean_,
        running_var_,
        save_mean_,
        save_invstd_,
        train,
        epsilon);
  }
  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_backward_reduce_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& mean,
    const Tensor& invstd,
    Tensor& sum_dy,
    Tensor& sum_dy_xmu,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  int numPlane = input.size(1);
  auto i_batch_size = input.size(0);
  auto i_feature_size = input.size(2);
  auto o_batch_size = grad_output.size(0);
  auto o_feature_size = grad_output.size(2);

  auto& queue = dpcppGetCurrentQueue();
  int64_t wg_size = get_prefer_wg_size(
      i_batch_size * i_feature_size, SIMD); // for higher occupancy

  int tx = getNumThreads(i_feature_size, wg_size);
  int ty = std::min(int64_t(last_pow2(i_batch_size)), wg_size / tx);
  ty = std::max(1, ty);
  sycl::range<2> local_range(ty, tx);
  sycl::range<2> global_range(numPlane * ty, tx);
  auto input_ptr = input.data_ptr<input_scalar_t>();
  auto grad_output_ptr = grad_output.data_ptr<input_scalar_t>();
  auto mean_ptr = mean.data_ptr<stat_accscalar_t>();
  auto invstd_ptr = invstd.data_ptr<stat_accscalar_t>();
  stat_scalar_t* grad_weight_ptr =
      grad_weight.size(0) > 0 ? grad_weight.data_ptr<stat_scalar_t>() : nullptr;
  stat_scalar_t* grad_bias_ptr =
      grad_bias.size(0) > 0 ? grad_bias.data_ptr<stat_scalar_t>() : nullptr;
  stat_accscalar_t* sum_dy_ptr =
      sum_dy.size(0) > 0 ? sum_dy.data_ptr<stat_accscalar_t>() : nullptr;
  stat_accscalar_t* sum_dy_xmu_ptr = sum_dy_xmu.size(0) > 0
      ? sum_dy_xmu.data_ptr<stat_accscalar_t>()
      : nullptr;

  index_t go_batch_stride = grad_output.stride(0);
  index_t go_plane_stride = grad_output.stride(1);
  index_t go_feature_stride = grad_output.stride(2);
  index_t i_batch_stride = input.stride(0);
  index_t i_plane_stride = input.stride(1);
  index_t i_feature_stride = input.stride(2);

  int sg_num = tx * ty / SIMD;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_sum =
        dpcpp_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>, 1>(
            sycl::range<1>{tx * ty}, cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item)
        [[intel::reqd_sub_group_size(SIMD)]] {
      index_t plane = item.get_group(0);
      auto lidx = item.get_local_id(1);
      auto lidy = item.get_local_id(0);

      if (plane >= numPlane) {
        return;
      }

      stat_accscalar_t r_mean = mean_ptr[plane];
      stat_accscalar_t factor = invstd_ptr[plane];
      GradOp<
          input_scalar_t,
          stat_accscalar_t,
          decltype(input_ptr),
          decltype(i_batch_stride)>
          g(r_mean,
            input_ptr,
            grad_output_ptr,
            i_batch_stride,
            i_plane_stride,
            i_feature_stride,
            go_batch_stride,
            go_plane_stride,
            go_feature_stride);
      auto res = plane_reduce<SIMD, Float2<input_scalar_t, stat_accscalar_t>>(
          item, g, o_batch_size, o_feature_size, plane, sg_num, local_sum);

      if (lidx == 0) {
        if (grad_weight_ptr != nullptr) {
          grad_weight_ptr[plane] = static_cast<stat_scalar_t>(res.v2 * factor);
        }
        if (grad_bias_ptr != nullptr) {
          grad_bias_ptr[plane] = static_cast<stat_scalar_t>(res.v1);
        }
        if (sum_dy_ptr != nullptr) {
          sum_dy_ptr[plane] = static_cast<stat_accscalar_t>(res.v1);
        }
        if (sum_dy_xmu_ptr != nullptr) {
          sum_dy_xmu_ptr[plane] = static_cast<stat_accscalar_t>(res.v2);
        }
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T, typename item_t, typename shared_t>
inline void merge_group_vertical_backward(
    item_t item,
    T& sum_dy,
    T& sum_dy_xmu,
    shared_t& shmem_sum_dy,
    shared_t& shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = item.get_local_linear_id();
  auto local_range_y = item.get_local_range(0);
  auto local_range_x = item.get_local_range(1);
  auto global_range_y = item.get_global_range(0);
  auto lidx = item.get_local_id(1);
  auto lidy = item.get_local_id(0);

  for (int offset = local_range_y / 2; offset > 0; offset >>= 1) {
    if (lidy < offset * 2) {
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
    item.barrier(dpcpp_local_fence);
    if (lidy < offset && lidy + offset < local_range_y) {
      auto address = address_base + offset * local_range_x;

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
void batch_norm_backward_reduce_channels_last_kernel(
    const Tensor& input,
    const scalar_t* grad_output,
    const Tensor& mean,
    const accscalar_t* inv_std,
    accscalar_t* sum_dy_o,
    accscalar_t* sum_dy_xmu_o,
    layerscalar_t* grad_weight,
    layerscalar_t* grad_bias,
    const int reduction_size,
    const int stride) {
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride, PARALLEL_LOADS, false);
  int loop_count =
      1 + (reduction_size - 1) / (global_range[0] * PARALLEL_LOADS);
  int group_num_y = global_range[0] / local_range[0];
  int group_num_x = global_range[1] / local_range[1];
  int global_range_y = global_range[0];
  int local_range_y = local_range[0];
  int global_range_x = global_range[1];
  auto wg_size = local_range[0] * local_range[1];
  auto& queue = dpcppGetCurrentQueue();
  at::Tensor staging_data;
  at::Tensor semaphores;

  scalar_t* input_ptr = input.data_ptr<scalar_t>();
  accscalar_t* mean_ptr = mean.data_ptr<accscalar_t>();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto shmem_sum_dy =
        dpcpp_local_acc_t<accscalar_t, 1>(sycl::range<1>{wg_size}, cgh);
    auto shmem_sum_dy_xmu =
        dpcpp_local_acc_t<accscalar_t, 1>(sycl::range<1>{wg_size}, cgh);

    auto is_last_group_done =
        dpcpp_local_acc_t<bool, 1>(sycl::range<1>{1}, cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
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
      int inner_loop_stride = item.get_global_range(0);

      // offset along m dimension
      int m_offset = item.get_global_id(0);
      int c_offset = item.get_global_id(1);
      auto lidx = item.get_local_id(1);
      auto lidy = item.get_local_id(0);
      auto group_id_y = item.get_group().get_group_id(0);
      auto group_id_x = item.get_group().get_group_id(1);

      int loop_count =
          1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
      int address_base = m_offset * stride + c_offset;
      int address_increment = inner_loop_stride * stride;

      accscalar_t r_mean = 0;
      accscalar_t factor = 1;
      if (c_offset < stride) {
        r_mean = mean_ptr[c_offset];
        factor = inv_std[c_offset];
      }
      for (int i = 0; i < loop_count; i++) {
        accscalar_t x_input[PARALLEL_LOADS];
        accscalar_t x_grad_output[PARALLEL_LOADS];

        // load multiple data in
#pragma unroll
        for (int j = 0; j < PARALLEL_LOADS; j++) {
          if (c_offset < stride && m_offset < reduction_size) {
            x_input[j] = input_ptr[address_base];
            x_grad_output[j] = grad_output[address_base];
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

      // thread reduction to accumulate sum_dy / sum_dy_xmu between
      // PARALLEL_LOADS
#pragma unroll
      for (int j = 1; j < PARALLEL_LOADS; j++) {
        sum_dy[0] += sum_dy[j];
        sum_dy_xmu[0] += sum_dy_xmu[j];
      }

      // release array of registers
      auto sum_dy_th = sum_dy[0];
      auto sum_dy_xmu_th = sum_dy_xmu[0];

      // group-wise reduction with shared memory (since reduction cannot be
      // done within a warp)

      merge_group_vertical_backward(
          item, sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);
      if (group_num_y > 1) {
        if (lidy == 0 && c_offset < stride) {
          atomicAdd(
              (dpcpp_global_ptr_pt<accscalar_t>)&sum_dy_o[c_offset], sum_dy_th);
          atomicAdd(
              (dpcpp_global_ptr_pt<accscalar_t>)&sum_dy_xmu_o[c_offset],
              sum_dy_xmu_th);
          if (grad_bias != nullptr) {
            atomicAdd(
                (dpcpp_global_ptr_pt<layerscalar_t>)&grad_bias[c_offset],
                sum_dy_th);
          }
          if (grad_weight != nullptr) {
            atomicAdd(
                (dpcpp_global_ptr_pt<layerscalar_t>)&grad_weight[c_offset],
                sum_dy_xmu_th * factor);
          }
        }
      } else {
        if (group_id_y == 0 && lidy == 0 && c_offset < stride) {
          if (grad_bias != nullptr) {
            grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
          }
          if (grad_weight != nullptr) {
            grad_weight[c_offset] =
                static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
          }
          // mean_dy[c_offset] = sum_dy_th / reduction_size;
          // mean_dy_xmu[c_offset] = sum_dy_xmu_th / reduction_size;
          sum_dy_o[c_offset] = sum_dy_th;
          sum_dy_xmu_o[c_offset] = sum_dy_xmu_th;
        }
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

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

  if (weight.defined() && input.scalar_type() != weight.scalar_type()) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_reduce",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          constexpr int ELEMENTS_PER_ITER =
              sizeof(float) * 4 / sizeof(scalar_t);
          batch_norm_backward_reduce_channels_last_kernel<ELEMENTS_PER_ITER>(
              input,
              grad_output.data_ptr<scalar_t>(),
              mean,
              inv_std.data_ptr<accscalar_t>(),
              sumn_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              grad_weight.data_ptr<accscalar_t>(),
              grad_bias.data_ptr<accscalar_t>(),
              reduction_size,
              stride);
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
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_reduce",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          constexpr int ELEMENTS_PER_ITER =
              sizeof(float) * 4 / sizeof(scalar_t);
          batch_norm_backward_reduce_channels_last_kernel<ELEMENTS_PER_ITER>(
              input,
              grad_output.data_ptr<scalar_t>(),
              mean,
              inv_std.data_ptr<accscalar_t>(),
              sumn_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr,
              weight.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr,
              reduction_size,
              stride);
        });
  }
  return std::make_tuple(sumn_dy, sum_dy_xmu, grad_weight, grad_bias);
}

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_backward_elemt_channels_first_kernel_impl(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu,
    Tensor& grad_input,
    const stat_accscalar_t norm_fct) {
  auto Hw = input.size(2);
  auto N = input.size(0);
  auto numPlane = input.size(1);
  int weight_size = weight.size(0);

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t target_tile_size = dpcppMaxWorkItemsPerTile(dev_id);
  int64_t wg_size = dpcppMaxWorkItemsPerEU(dev_id); // for work group barrier
  if (wg_size * numPlane < target_tile_size) {
    wg_size = dpcppMaxWorkGroupSize(); // for higher occupancy
  }
  auto i_batch_stride = input.size(1) * input.size(2);
  auto gi_batch_stride = grad_input.size(1) * grad_input.size(2);
  auto go_batch_stride = grad_output.size(1) * grad_output.size(2);
  auto gi_Hw = grad_input.size(2);
  auto go_Hw = grad_output.size(2);

  int tf = getNumThreads(Hw, wg_size);
  int tb = std::max<int>(wg_size / tf, 1);
  sycl::range<2> local_range(tb, tf);
  sycl::range<2> global_range((N + tb - 1) / tb * tb, tf * numPlane);

  int global_range_y = global_range[0];
  int local_range_x = local_range[1];
  auto input_ptr = input.data_ptr<input_scalar_t>();
  auto grad_output_ptr = grad_output.data_ptr<input_scalar_t>();
  auto grad_input_ptr = grad_input.data_ptr<input_scalar_t>();

  auto mean_ptr = mean.data_ptr<stat_accscalar_t>();
  auto invstd_ptr = invstd.data_ptr<stat_accscalar_t>();
  auto weight_ptr =
      weight.defined() ? weight.data_ptr<stat_scalar_t>() : nullptr;
  auto sum_dy_ptr = sum_dy.data_ptr<stat_accscalar_t>();
  auto sum_dy_xmu_ptr = sum_dy_xmu.data_ptr<stat_accscalar_t>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      index_t plane = item.get_group(1);

      if (plane >= numPlane) {
        return;
      }

      stat_accscalar_t m_c = mean_ptr[plane];
      stat_accscalar_t m_dy_c = sum_dy_ptr[plane] * norm_fct;
      stat_accscalar_t factor_1_c = invstd_ptr[plane];
      stat_accscalar_t factor_2_c = weight_ptr != nullptr
          ? static_cast<stat_accscalar_t>(weight_ptr[plane])
          : stat_accscalar_t(1);
      factor_2_c *= factor_1_c;
      factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu_ptr[plane] * norm_fct;

      index_t bstep = global_range_y;
      for (index_t batch = item.get_global_id(0); batch < N; batch += bstep) {
        auto g_i_offset = batch * gi_batch_stride + plane * gi_Hw;
        auto g_o_offset = batch * go_batch_stride + plane * go_Hw;
        auto i_offset = batch * i_batch_stride + plane * Hw;
        for (index_t feature = item.get_local_id(1); feature < Hw;
             feature += local_range_x) {
          grad_input_ptr[g_i_offset + feature] = static_cast<input_scalar_t>(
              (grad_output_ptr[g_o_offset + feature] - m_dy_c -
               (input_ptr[i_offset + feature] - m_c) * factor_1_c) *
              factor_2_c);
        }
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_channels_first_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& weight_,
    const Tensor& sum_dy_,
    const Tensor& sum_dy_xmu_) {
  using stat_accscalar_t = acc_type<stat_scalar_t>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());
  auto grad_input_reshaped =
      at::empty_like(input_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto reduction_size = input_.numel() / n_input;
  auto norm_fct = static_cast<stat_accscalar_t>(1.0 / reduction_size);
  batch_norm_backward_elemt_channels_first_kernel_impl<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t>(
      input_reshaped,
      grad_output_reshaped,
      mean_,
      invstd_,
      weight_,
      sum_dy_,
      sum_dy_xmu_,
      grad_input_reshaped,
      norm_fct);

  return grad_input_reshaped.view(input_.sizes());
}

template <typename scalar_t, typename accscalar_t, typename layerscalar_t>
void batch_norm_backward_elemt_channels_last_kernel_impl(
    const scalar_t* grad_output,
    const scalar_t* input,
    const accscalar_t* mean,
    const accscalar_t* inv_std,
    const layerscalar_t* weight,
    const accscalar_t* sum_dy,
    const accscalar_t* sum_dy_xmu,
    scalar_t* grad_input,
    const accscalar_t norm_fct,
    const int reduction_size,
    const int stride) {
  auto& queue = dpcppGetCurrentQueue();
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      // tensor dimension (m,c)
      // loop along m dimension
      int inner_loop_stride = item.get_global_range(0);

      // offset along m dimension
      int m_offset = item.get_global_id(0);
      int c_offset = item.get_global_id(1);

      if (c_offset >= stride || m_offset >= reduction_size) {
        return;
      }

      auto m_c = mean[c_offset];
      auto m_dy_c = sum_dy[c_offset] * norm_fct;
      auto factor_1_c = inv_std[c_offset];
      auto factor_2_c =
          (weight == nullptr ? accscalar_t(1.0)
                             : static_cast<accscalar_t>(weight[c_offset])) *
          factor_1_c;
      factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu[c_offset] * norm_fct;

      int address_base = m_offset * stride + c_offset;
      int address_increment = inner_loop_stride * stride;

      for (int m_offset_loop = item.get_global_id(0);
           m_offset_loop < reduction_size;
           m_offset_loop += inner_loop_stride) {
        grad_input[address_base] = static_cast<scalar_t>(
            (static_cast<accscalar_t>(grad_output[address_base]) - m_dy_c -
             (static_cast<accscalar_t>(input[address_base]) - m_c) *
                 factor_1_c) *
            factor_2_c);
        address_base += address_increment;
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename accscalar_t, typename layerscalar_t>
void batch_norm_backward_elemt_channels_last_kernel(
    const scalar_t* grad_output,
    const scalar_t* input,
    const accscalar_t* mean,
    const accscalar_t* inv_std,
    const layerscalar_t* weight,
    const accscalar_t* sum_dy,
    const accscalar_t* sum_dy_xmu,
    const int* numel,
    scalar_t* grad_input,
    const int64_t world_size,
    const int reduction_size,
    const int stride) {
  int64_t total_numel = 0;
  for (int i = 0; i < world_size; i++) {
    total_numel += numel[i];
  }

  auto norm_fct =
      static_cast<accscalar_t>(1) / static_cast<accscalar_t>(total_numel);
  batch_norm_backward_elemt_channels_last_kernel_impl(
      grad_output,
      input,
      mean,
      inv_std,
      weight,
      sum_dy,
      sum_dy_xmu,
      grad_input,
      norm_fct,
      reduction_size,
      stride);
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

  if (weight.defined() && weight.scalar_type() != input.scalar_type()) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_element",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          batch_norm_backward_elemt_channels_last_kernel(
              grad_output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              mean.data_ptr<accscalar_t>(),
              inv_std.data_ptr<accscalar_t>(),
              weight.data_ptr<accscalar_t>(),
              sum_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              count.data_ptr<int>(),
              grad_input.data_ptr<scalar_t>(),
              count.numel(),
              reduction_size,
              stride);
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
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "batchnorm_backward_element",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          batch_norm_backward_elemt_channels_last_kernel(
              grad_output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              mean.data_ptr<accscalar_t>(),
              inv_std.data_ptr<accscalar_t>(),
              weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
              sum_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              count.data_ptr<int>(),
              grad_input.data_ptr<scalar_t>(),
              count.numel(),
              reduction_size,
              stride);
        });
  }

  return grad_input;
}

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

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "batchnorm_backward_element", [&] {
        using accscalar_t = acc_type<scalar_t>;

        if (weight.defined() && weight.scalar_type() != input.scalar_type()) {
          batch_norm_backward_elemt_channels_last_kernel_impl(
              grad_output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              mean.data_ptr<accscalar_t>(),
              inv_std.data_ptr<accscalar_t>(),
              weight.data_ptr<accscalar_t>(),
              sum_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              grad_input.data_ptr<scalar_t>(),
              static_cast<accscalar_t>(norm_fct),
              reduction_size,
              stride);
        } else {
          batch_norm_backward_elemt_channels_last_kernel_impl(
              grad_output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              mean.data_ptr<accscalar_t>(),
              inv_std.data_ptr<accscalar_t>(),
              weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
              sum_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              grad_input.data_ptr<scalar_t>(),
              static_cast<accscalar_t>(norm_fct),
              reduction_size,
              stride);
        }
      });

  return grad_input;
}

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
      return IPEX_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          input.scalar_type(),
          "batch_norm_backward_elemt",
          [&] {
            using accscalar_t = acc_type<scalar_t>;
            const bool mixed_type = is_mixed_type(input, weight);
            if (mixed_type) {
              return batch_norm_backward_elemt_channels_first_template<
                  scalar_t,
                  accscalar_t,
                  int32_t>(
                  grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
            } else {
              return batch_norm_backward_elemt_channels_first_template<
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

      Tensor grad_input = at::empty(input.sizes(), grad_out.options());
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

      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          grad_out.scalar_type(),
          "batch_norm_eval_backward",
          [&] {
            using accscalar_t = acc_type<scalar_t>;
            auto norm_fct =
                static_cast<accscalar_t>(1.0 / (input.numel() / input.size(1)));
            dpcpp_kernel_for_tensor_iter(
                iter,
                [norm_fct](
                    scalar_t gO,
                    scalar_t input,
                    accscalar_t weight,
                    accscalar_t mean,
                    accscalar_t invstd,
                    accscalar_t xmu,
                    accscalar_t dy) -> scalar_t {
                  auto factor_1_c = invstd * invstd * xmu * norm_fct;
                  auto factor_2_c = weight * invstd;
                  auto m_dy_c = dy * norm_fct;
                  return (gO - m_dy_c - (input - mean) * factor_1_c) *
                      factor_2_c;
                });
          });
      return grad_input;
    }
  }
}

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
                    .add_input(grad_out)
                    .add_input(invstd_nd)
                    .add_input(weight_nd)
                    .check_all_same_dtype(false)
                    .promote_inputs_to_common_dtype(false)
                    .build();

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        grad_out.scalar_type(),
        "batch_norm_eval_backward",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          dpcpp_kernel_for_tensor_iter(
              iter,
              [](scalar_t gO, accscalar_t invstd, accscalar_t weight)
                  -> scalar_t { return gO * weight * invstd; });
        });
  } else {
    auto iter = TensorIteratorConfig()
                    .add_output(grad_input)
                    .add_input(grad_out)
                    .add_input(invstd_nd)
                    .check_all_same_dtype(false)
                    .promote_inputs_to_common_dtype(false)
                    .build();

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        grad_out.scalar_type(),
        "batch_norm_eval_backward",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          dpcpp_kernel_for_tensor_iter(
              iter, [](scalar_t gO, accscalar_t invstd) -> scalar_t {
                return gO * invstd;
              });
        });
  }
  return grad_input;
}

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
  using stat_accscalar_t = acc_type<stat_scalar_t>;
  int64_t n_input = input_.size(1);
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto new_shape = {
      input_.size(0),
      input_.size(1),
      input_.numel() / input_.size(0) / input_.size(1)};
  auto new_stride = {input_.stride(0), input_.stride(1), input_.stride(-1)};

  auto input_reshaped = at::as_strided(input_, new_shape, new_stride);
  auto grad_output_reshaped = at::as_strided(grad_out_, new_shape, new_stride);

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

  int simd = get_prefer_simd(
      input_reshaped.size(1), input_reshaped.size(0) * input_reshaped.size(1));
  if (simd == SIMD32) {
    batch_norm_backward_reduce_kernel<
        SIMD32,
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        mean_,
        invstd_,
        sum_dy_,
        sum_dy_xmu_,
        grad_weight_,
        grad_bias_);
  } else {
    batch_norm_backward_reduce_kernel<
        SIMD16,
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        mean_,
        invstd_,
        sum_dy_,
        sum_dy_xmu_,
        grad_weight_,
        grad_bias_);
  }
  return std::make_tuple(sum_dy_, sum_dy_xmu_, grad_weight_, grad_bias_);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_dispatch(
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
  return IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad_output.scalar_type(),
      "batch_norm_backward_reduce",
      [&] {
        auto mean_st = mean.dtype();
        auto invstd_st = invstd.dtype();
        TORCH_CHECK(
            mean_st == invstd_st,
            "mean and invstd need to have the same data types");
        const bool mixed_type = is_mixed_type(input, weight);
        using accscalar_t = acc_type<scalar_t>;

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

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    const c10::optional<at::Tensor>& save_mean_opt,
    const c10::optional<at::Tensor>& save_var_opt,
    bool training,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  xpu::COMPUTE_ENG real_eng;
  real_eng = choose_compute_eng(xpu::COMPUTE_ENG::BASIC, input);

  if (xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    c10::MaybeOwned<Tensor> weight_maybe_owned =
        at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_maybe_owned;

    c10::MaybeOwned<Tensor> running_mean_maybe_owned =
        at::borrow_from_optional_tensor(running_mean_opt);
    const Tensor& running_mean = *running_mean_maybe_owned;

    c10::MaybeOwned<Tensor> running_var_maybe_owned =
        at::borrow_from_optional_tensor(running_var_opt);
    const Tensor& running_var = *running_var_maybe_owned;

    c10::MaybeOwned<Tensor> save_mean_maybe_owned =
        at::borrow_from_optional_tensor(save_mean_opt);
    const Tensor& save_mean = *save_mean_maybe_owned;

    c10::MaybeOwned<Tensor> save_var_maybe_owned =
        at::borrow_from_optional_tensor(save_var_opt);
    const Tensor& save_var = *save_var_maybe_owned;

    if (save_mean.defined() && save_var.defined()) {
      checkBackend(
          "batch_norm",
          {input, weight, grad_output, save_mean, save_var},
          Backend::XPU);
    } else {
      checkBackend("batch_norm", {input, weight, grad_output}, Backend::XPU);
    }

    if (input.scalar_type() != at::ScalarType::Float &&
        input.scalar_type() != at::ScalarType::Half &&
        input.scalar_type() != at::ScalarType::BFloat16) {
      std::stringstream ss;
      ss << "DPCPP batch_norm backend got unsupported type="
         << input.scalar_type();
      TORCH_CHECK(0, ss.str());
    } else {
      return xpu::oneDNN::batch_normalization_backward(
          condition_contiguous(grad_output),
          condition_contiguous(input),
          condition_contiguous(weight),
          condition_contiguous(running_mean),
          condition_contiguous(running_var),
          condition_contiguous(save_mean),
          condition_contiguous(save_var),
          training,
          epsilon,
          grad_input_mask);
    }
  } else {
    // SYCL implementation
    c10::MaybeOwned<Tensor> weight =
        at::borrow_from_optional_tensor(weight_opt);
    c10::MaybeOwned<Tensor> save_mean =
        at::borrow_from_optional_tensor(save_mean_opt);
    c10::MaybeOwned<Tensor> save_invstd =
        at::borrow_from_optional_tensor(save_var_opt);
    c10::MaybeOwned<Tensor> running_mean =
        at::borrow_from_optional_tensor(running_mean_opt);
    c10::MaybeOwned<Tensor> running_var =
        at::borrow_from_optional_tensor(running_var_opt);

    const bool needs_reduction =
        training || grad_input_mask[1] || grad_input_mask[2];

    // Fused reducion & elementwise kernel
    if (needs_reduction && grad_input_mask[0] &&
        !batch_norm_use_channels_last_kernels(input) &&
        canUse32BitIndexMath(input) && canUse32BitIndexMath(grad_output)) {
      return IPEX_DISPATCH_FLOATING_TYPES_AND2(
          kHalf, kBFloat16, input.scalar_type(), "batch_norm_backward", [&] {
            using accscalar_t = acc_type<scalar_t>;
            const bool mixed_type =
                is_mixed_type(input, *weight, *running_mean, *running_var);
            if (mixed_type) {
              return batch_norm_backward_channels_first_template<
                  scalar_t,
                  accscalar_t,
                  int32_t>(
                  grad_output,
                  input,
                  *weight,
                  *running_mean,
                  *running_var,
                  *save_mean,
                  *save_invstd,
                  training,
                  epsilon,
                  grad_input_mask);
            } else {
              return batch_norm_backward_channels_first_template<
                  scalar_t,
                  scalar_t,
                  int32_t>(
                  grad_output,
                  input,
                  *weight,
                  *running_mean,
                  *running_var,
                  *save_mean,
                  *save_invstd,
                  training,
                  epsilon,
                  grad_input_mask);
            }
          });
    }
    auto acc_type = at::AtenIpexTypeXPU::toAccumulateType(input.scalar_type());
    Tensor mean;
    TORCH_INTERNAL_ASSERT(
        save_mean->defined(), "save_mean should always be defined\n");
    if (save_mean->numel() != 0) {
      mean = *save_mean;
    } else if (needs_reduction) {
      TORCH_CHECK(!training && running_mean->defined());
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
      TORCH_CHECK(!training && running_var->defined());
      auto n_channels = input.sizes()[1];
      invstd = at::empty({n_channels}, input.options().dtype(acc_type));
      batch_norm_calc_invstd(invstd, *running_var, epsilon);
    }

    Tensor sum_dy, sum_dy_xmu, grad_weight, grad_bias;
    if (needs_reduction) {
      std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias) =
          batch_norm_backward_reduce_dispatch(
              grad_output,
              input,
              mean,
              invstd,
              *weight,
              grad_input_mask[0],
              grad_input_mask[1],
              grad_input_mask[2]);
    }
    Tensor grad_input = at::empty_like(grad_output);
    if (grad_input_mask[0]) {
      if (training) {
        // NOTE: sum_dy and sum_dy_xmy are defined, as train implies
        grad_input = batch_norm_elementwise_backward_train(
            grad_output, input, mean, invstd, *weight, sum_dy, sum_dy_xmu);
      } else {
        grad_input = batch_norm_elementwise_backward_eval(
            grad_output, input, invstd, *weight);
      }
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }
}

// For native_batch_norm, we don't use this batch_norm_elemt_out
// Because oneDNN could handle it automatically.
at::Tensor& batch_norm_elemt_out(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps,
    at::Tensor& out) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  checkBackend("batch_norm", {input, weight, bias, mean, invstd}, Backend::XPU);

  // The check follows native batch norm
  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half &&
      input.scalar_type() != at::ScalarType::BFloat16) {
    std::stringstream ss;
    ss << "batch_norm backend got unsupported type=" << input.scalar_type();
    TORCH_CHECK(0, ss.str());
  }

  // Don't need these two, thus use dummy tensor.
  // In current stat, the oneDNN batch norm flag should be
  // inference + use_global_stats.
  Tensor dummy_mean;
  Tensor dummy_var;

  // don't need momentum, epsilon, thus use dummy data
  xpu::oneDNN::batch_normalization(
      condition_contiguous(input),
      condition_contiguous(weight),
      condition_contiguous(bias),
      condition_contiguous(mean),
      condition_contiguous(invstd),
      /* training*/ false,
      /* momentum */ 1.0,
      /*epsilon , dummy epsilon*/ 1e-5,
      out,
      dummy_mean,
      dummy_var);
  return out;
}

at::Tensor batch_norm_elemt(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps) {
  // Empty tensor, it will be initialized in batch_norm_elemt_out
  Tensor out;
  batch_norm_elemt_out(input, weight, bias, mean, invstd, eps, out);
  return out;
}

template <typename scalar_t, typename accscalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_gather_stats_xpu_template(
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

  auto mean = mean_.data_ptr<accscalar_t>();
  auto invstd = invstd_.data_ptr<accscalar_t>();
  auto running_mean =
      running_mean_.defined() ? running_mean_.data_ptr<scalar_t>() : nullptr;
  auto running_var =
      running_var_.defined() ? running_var_.data_ptr<scalar_t>() : nullptr;
  auto counts = counts_.data_ptr<scalar_t>();
  auto save_mean = save_mean_.data_ptr<accscalar_t>();
  auto save_invstd = save_invstd_.data_ptr<accscalar_t>();

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (features + wgroup_size - 1) / wgroup_size;

  int world_size = mean_.size(0);
  // Avoid double issues in ATSM
  float momentum_ = momentum;
  float epsilon_ = epsilon;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(ngroups * wgroup_size, wgroup_size),
        [=](sycl::nd_item<1> itemId) {
          auto tid = itemId.get_global_id(0);

          // first the reductions each thread does separately
          if (tid < features) {
            accscalar_t avg = 0;
            accscalar_t var_n = 0;
            index_t n = 0;
            for (int j = 0; j < world_size; j++) {
              scalar_t count = counts[j];
              accscalar_t m = mean[j * features + tid];
              accscalar_t v = accscalar_t(1.0f) / (invstd[j * features + tid]);
              v = (v * v - epsilon_) * count;
              accscalar_t factor = 1.0f / (n + count);
              var_n += v + (avg - m) * (avg - m) * n * count * factor;
              avg = n * factor * avg + count * factor * m;
              n += count;
            }
            save_mean[tid] = avg;
            save_invstd[tid] = static_cast<accscalar_t>(1) /
                Numerics<accscalar_t>::sqrt(var_n / n + epsilon_);
            if (running_mean != nullptr) {
              running_mean[tid] = static_cast<scalar_t>(
                  (1 - momentum_) * running_mean[tid] + momentum_ * avg);
            }
            accscalar_t unbiasedVar = var_n / (n - 1);
            if (running_var != nullptr) {
              running_var[tid] = static_cast<scalar_t>(
                  (1 - momentum_) * running_var[tid] + momentum_ * unbiasedVar);
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  return std::make_tuple(save_mean_, save_invstd_);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_xpu(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    double momentum,
    double epsilon,
    const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  auto scalar_type =
      running_mean.defined() ? running_mean.scalar_type() : self.scalar_type();
  return IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "batch_norm_update_stats_xpu",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        if (xpu::dpcpp::detail::canUse32BitIndexMath(self)) {
          return batch_norm_gather_stats_xpu_template<
              scalar_t,
              accscalar_t,
              int32_t>(
              mean,
              invstd,
              running_mean,
              running_var,
              momentum,
              epsilon,
              counts);
        } else {
          return batch_norm_gather_stats_xpu_template<
              scalar_t,
              accscalar_t,
              int64_t>(
              mean,
              invstd,
              running_mean,
              running_var,
              momentum,
              epsilon,
              counts);
        }
      });
}

// accepting input(self) here to determine template data types, since
// running_mean/running_var are optional
std::tuple<Tensor, Tensor> batch_norm_gather_stats(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    double momentum,
    double epsilon,
    int64_t count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  Tensor counts_ = at::empty(
      mean.size(0),
      self.options().dtype(
          running_mean.defined() ? running_mean.dtype() : self.dtype()));
  counts_.fill_(count);
  return batch_norm_gather_stats_with_counts_xpu(
      self,
      mean,
      invstd,
      running_mean,
      running_var,
      momentum,
      epsilon,
      counts_);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    double momentum,
    double epsilon,
    const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  return batch_norm_gather_stats_with_counts_xpu(
      self,
      mean,
      invstd,
      running_mean_opt,
      running_var_opt,
      momentum,
      epsilon,
      counts);
}

std::tuple<Tensor, Tensor> batch_norm_stats(
    const Tensor& self,
    double epsilon) {
  auto options = self.options().dtype(toAccumulateType(self.scalar_type()));
  auto n_channels = self.size(1);
  auto save_mean = at::empty({n_channels}, options);
  auto save_invstd = at::empty({n_channels}, options);

  batch_norm_stats_out(self, epsilon, save_mean, save_invstd);
  return std::tuple<Tensor, Tensor>(save_mean, save_invstd);
}

std::tuple<Tensor&, Tensor&> batch_norm_stats_out(
    const Tensor& self,
    double epsilon,
    Tensor& save_mean,
    Tensor& save_invstd) {
  bool use_channels_last_kernel = batch_norm_use_channels_last_kernels(self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_stats",
      [&] {
        if (use_channels_last_kernel) {
          batch_norm_stats_channels_last_template<scalar_t, InvStd>(
              save_mean, save_invstd, self, epsilon);
        } else if (canUse32BitIndexMath(self)) {
          batch_norm_stats_channels_first_template<scalar_t, int32_t, InvStd>(
              save_mean, save_invstd, self, epsilon);
        } else {
          batch_norm_stats_channels_first_template<scalar_t, int64_t, InvStd>(
              save_mean, save_invstd, self, epsilon);
        }
      });
  return std::tuple<Tensor&, Tensor&>(save_mean, save_invstd);
}

} // namespace AtenIpexTypeXPU
} // namespace at
