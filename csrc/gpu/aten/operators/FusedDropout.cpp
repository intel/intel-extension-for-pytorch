#include <ATen/ATen.h>

#include <ATen/record_function.h>
#include <core/Generator.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

#include <aten/operators/MemoryAccess.h>

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

const int UNROLL = 4;

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int vec_size,
    typename mask_t,
    typename item_t>
inline void fused_dropout_kernel_vec(
    item_t& item,
    TensorInfo<scalar_t, IndexType> a,
    TensorInfo<scalar_t, IndexType> b,
    TensorInfo<mask_t, IndexType> c,
    IndexType totalElements,
    accscalar_t p,
    PhiloxState philox_args) {
  // make sure we don't break assumption that we can't have > 4 elements /
  // thread
  static_assert(vec_size <= 4, "Value of vec_size must be in [2, 4]");

  using LoadT = native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using MaskLoadT = native::Memory::aligned_vector_loop<mask_t, vec_size>;

  auto thread_idx = item.get_local_id(0);
  auto thread_range = item.get_local_range(0);
  auto group_idx = item.get_group(0);
  auto group_range = item.get_group_range(0);

  auto seeds = philox_unpack(philox_args);
  IndexType idx = group_idx * thread_range + thread_idx;
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  // Helps align the total number of times rand_uniform4 is called by each
  // thread for the same totalElements in the vec=2 and vec=4 cases.
  bool gxvec_loop_state = 0;
  accscalar_t scale = 1.0f / p;

  float4 rand;

  // Note: Vectorized loads means we'll stride each thread by an additional
  // vec_size factor, as we'll load vec_size elements at a time
  for (IndexType linearIndex = idx * vec_size; linearIndex < totalElements;
       linearIndex += group_range * thread_range * vec_size) {
    // local storage
    scalar_t src[vec_size];
    // We'll use this to actually cause vectorized loads later
    LoadT* value = reinterpret_cast<LoadT*>(&src);

    // Note: need a new set of random values per 4 elements -- we'll handle
    // vec_size elements in this thread, so need ceil(vec_size / 4) sets of
    // rand.
    if ((vec_size == 4) || (gxvec_loop_state == 0)) {
      rand = rand_uniform4(&state);
    } else {
      // sets up the last two values we generated last iteration to be used this
      // iteration.
      rand.x = rand.z;
      rand.y = rand.w;
      gxvec_loop_state ^= 1;
    }

    rand.x = rand.x < p;
    rand.y = rand.y < p;
    if (vec_size == 4) {
      rand.z = rand.z < p;
      rand.w = rand.w < p;
    }

    // Note: We explicitly check for is_contiguous() before launching the
    // vectorized kernel and replace IndexToOffset call with linearIndex to
    // allow vectorization of NHWC (or other) ordering. Single vectorized load
    *value = *reinterpret_cast<LoadT*>(&a.data[linearIndex]);

    scalar_t r[vec_size];
    mask_t mask[vec_size];

// Perform the actual computation
#pragma unroll
    for (int ii = 0; ii < vec_size; ii++) {
      r[ii] = src[ii] * (&rand.x)[ii] * scale;
      mask[ii] = (mask_t)(&rand.x)[ii];
    }
    // Vectorized writes for both mask & result
    *(reinterpret_cast<LoadT*>(&b.data[linearIndex])) =
        *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&c.data[linearIndex])) =
        *reinterpret_cast<MaskLoadT*>(&mask[0]);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    typename mask_t,
    typename item_t>
inline void fused_dropout_kernel(
    item_t& item,
    TensorInfo<scalar_t, IndexType> a,
    TensorInfo<scalar_t, IndexType> b,
    TensorInfo<mask_t, IndexType> c,
    IndexType totalElements,
    accscalar_t p,
    PhiloxState philox_args) {
  auto seeds = philox_unpack(philox_args);
  IndexType idx = item.get_linear_id();
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);
  accscalar_t scale = 1.0f / p;

  float rand = _rand_uniform(state.output.x);
  rand = rand < p;
  scalar_t src = a.data[idx];
  b.data[idx] = src * rand * scale;
  c.data[idx] = (mask_t)rand;
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    typename mask_t,
    typename item_t>
inline void fused_dropout_kernel_unroll(
    item_t& item,
    TensorInfo<scalar_t, IndexType> a,
    TensorInfo<scalar_t, IndexType> b,
    TensorInfo<mask_t, IndexType> c,
    IndexType totalElements,
    accscalar_t p,
    PhiloxState philox_args) {
  auto thread_idx = item.get_local_id(0);
  auto thread_range = item.get_local_range(0);
  auto group_idx = item.get_group(0);
  auto group_range = item.get_group_range(0);

  auto seeds = philox_unpack(philox_args);
  IndexType idx = group_idx * thread_range + thread_idx;
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);
  accscalar_t scale = 1.0f / p;

  IndexType rounded_size =
      ((totalElements - 1) / (thread_range * group_range * UNROLL) + 1) *
      thread_range * group_range * UNROLL;
  for (IndexType linearIndex = idx; linearIndex < rounded_size;
       linearIndex += group_range * thread_range * UNROLL) {
    float4 rand = rand_uniform4(&state);
    scalar_t src[UNROLL];
    rand.x = rand.x < p;
    rand.y = rand.y < p;
    rand.z = rand.z < p;
    rand.w = rand.w < p;
#pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + thread_range * group_range * ii;
      if (li < totalElements) {
        // Convert `linearIndex` into an offset of `a`
        const IndexType aOffset =
            IndexToOffset<scalar_t, IndexType>::get(li, a);
        src[ii] = a.data[aOffset];
      }
    }
#pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + thread_range * group_range * ii;
      if (li < totalElements) {
        // Convert `linearIndex` into an offset of `b`
        const IndexType bOffset =
            IndexToOffset<scalar_t, IndexType>::get(li, b);
        b.data[bOffset] = src[ii] * (&rand.x)[ii] * scale;
        c.data[bOffset] = (mask_t)(&rand.x)[ii];
      }
    }
  }
}

template <typename scalar_t>
int get_vector_size(at::Tensor self, at::Tensor ret, at::Tensor mask) {
  int vec_size = 4;
  // get the vector size
  if (!self.is_non_overlapping_and_dense() ||
      !ret.is_non_overlapping_and_dense() ||
      !mask.is_non_overlapping_and_dense()) {
    vec_size = 1;
  } else {
    vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
        dpcppGetDeviceIdOfCurrentQueue(), (char*)self.data_ptr());
  }

  // check that we'd have no remainders - prefer a smaller vector size with no
  // remainders over a larger vector and remainder.
  bool can_vectorize = true;
  do {
    can_vectorize = self.numel() % vec_size == 0 &&
        ret.numel() % vec_size == 0 && mask.numel() % vec_size == 0;
    if (!can_vectorize)
      vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

template <typename mask_t>
std::tuple<Tensor, Tensor> dropout_template(
    DPCPPGeneratorImpl* gen,
    const Tensor& self,
    double p) {
  Tensor mask = at::empty_like(
      self, self.options().dtype(c10::CppTypeToScalarType<mask_t>::value));

  // empty tensors should not get here, but just in case, avoid FPE
  // non-training shot-cut
  int64_t nelem = self.numel();
  if (nelem == 0)
    return std::tuple<Tensor, Tensor>(self.clone(), mask);
  Tensor ret = at::empty_like(self);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "fused_dropout",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        using index_t = int64_t;

        accscalar_t pa = (accscalar_t)(p);

        bool need_not_rand4 = nelem < dpcppMaxWorkItemsPerTile();
        int vec_size =
            need_not_rand4 ? 1 : get_vector_size<scalar_t>(self, ret, mask);
        uint64_t counter_offset;
        uint32_t group_range, group_size;
        std::tie(counter_offset, group_range, group_size) =
            calc_execution_policy(nelem / vec_size);
        auto glb_range = group_range * group_size;
        auto loc_range = group_size;

        // trivial offset calculate for small workload
        // 1. due to bounding in latency, avoid offset calculation
        // 2. additional copy for contiguous is cheap
        auto self_ = need_not_rand4 ? self.contiguous() : self;
        auto self_info = getTensorInfo<scalar_t, index_t>(self_);
        auto ret_info = getTensorInfo<scalar_t, index_t>(ret);
        auto mask_info = getTensorInfo<mask_t, index_t>(mask);
        self_info.collapseDims();
        ret_info.collapseDims();
        mask_info.collapseDims();

        std::pair<uint64_t, uint64_t> seeds;
        {
          // See Note [Acquire lock when using random generators]
          std::lock_guard<std::mutex> lock(gen->mutex_);
          seeds = gen->philox_engine_inputs(counter_offset);
        }
        PhiloxState rng_engine_inputs(std::get<0>(seeds), std::get<1>(seeds));

        auto& q = dpcppGetCurrentQueue();
        if (need_not_rand4) {
          auto cgf = DPCPP_Q_CGF(cgh) {
            auto kfn = DPCPP_Q_KFN(sycl::item<1> item) {
              fused_dropout_kernel<scalar_t, accscalar_t, index_t>(
                  item,
                  self_info,
                  ret_info,
                  mask_info,
                  nelem,
                  pa,
                  rng_engine_inputs);
            };
            cgh.parallel_for(sycl::range<1>(nelem), kfn);
          };
          DPCPP_Q_SUBMIT(q, cgf);
        } else {
          // 1. batch random number creation
          // 2. batch/vectorized memory access
          switch (vec_size) {
            case 16:
            case 8:
            case 4: {
              auto cgf = DPCPP_Q_CGF(cgh) {
                auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
                  fused_dropout_kernel_vec<scalar_t, accscalar_t, index_t, 4>(
                      item,
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
                };
                cgh.parallel_for(sycl::nd_range<1>(glb_range, loc_range), kfn);
              };
              DPCPP_Q_SUBMIT(q, cgf);
            } break;
            case 2: {
              auto cgf = DPCPP_Q_CGF(cgh) {
                auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
                  fused_dropout_kernel_vec<scalar_t, accscalar_t, index_t, 2>(
                      item,
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
                };
                cgh.parallel_for(sycl::nd_range<1>(glb_range, loc_range), kfn);
              };
              DPCPP_Q_SUBMIT(q, cgf);
            } break;
            case 1: {
              auto cgf = DPCPP_Q_CGF(cgh) {
                auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
                  fused_dropout_kernel_unroll<scalar_t, accscalar_t, index_t>(
                      item,
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
                };
                cgh.parallel_for(sycl::nd_range<1>(glb_range, loc_range), kfn);
              };
              DPCPP_Q_SUBMIT(q, cgf);
            } break;
          };
        }
      });

  return std::tuple<Tensor, Tensor>(ret, mask);
}

template <typename mask_t, typename scalar_t, typename accscalar_t>
void masked_scale_kernel(
    at::Tensor& ret,
    const at::Tensor& src,
    const at::Tensor& mask,
    accscalar_t scale) {
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(ret)
                  .add_input(src)
                  .add_input(mask)
                  .build();

  dpcpp_kernel_for_tensor_iter(
      iter, [=](const scalar_t src_val, const mask_t mask_val) -> scalar_t {
        return (float)mask_val * src_val * scale;
      });
}

template <typename mask_t>
Tensor dropout_backward_dpcpp(
    const Tensor& grad,
    const Tensor& mask,
    double scale) {
  Tensor ret = at::empty_like(grad, grad.suggest_memory_format());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      ret.scalar_type(),
      "masked_scale",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        masked_scale_kernel<mask_t, scalar_t>(
            ret, grad, mask, (accscalar_t)scale);
      });
  return ret;
}

} // namespace impl

std::tuple<Tensor, Tensor> native_dropout(
    const Tensor& self,
    double p,
    c10::optional<bool> train) {
  // short-cut for train == false
  if (train.has_value() && !train.value()) {
    return std::make_tuple(
        self.clone(),
        at::ones_like(
            self, self.options().dtype(c10::CppTypeToScalarType<bool>::value)));
  }
  // short-cut
  if (p == 1) {
    // native_dropout is in file yaml, so we don't need to add data
    // dependency from output to input for autograd
    auto ret = at::zeros_like(self);
    auto mask = at::zeros_like(
        self, self.options().dtype(c10::CppTypeToScalarType<bool>::value));
    return std::tuple<Tensor, Tensor>(ret, mask);
  }

  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      c10::nullopt, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  double p1m = 1. - p;
  return impl::dropout_template<bool>(gen, self, p1m);
}

// NOTE: _fused_dropout will be removed, see PR #63937
std::tuple<Tensor, Tensor> _fused_dropout(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  return impl::dropout_template<uint8_t>(gen, self, p);
}

Tensor native_dropout_backward(
    const Tensor& grad,
    const Tensor& mask,
    double scale) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "Mask should be Bool Scalar Type",
      mask.scalar_type());
  return impl::dropout_backward_dpcpp<bool>(grad, mask, scale);
}

// NOTE: _masked_scale will be removed, see PR #63937
Tensor _masked_scale(const Tensor& self, const Tensor& mask, double scale) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype");
  return impl::dropout_backward_dpcpp<uint8_t>(self, mask, scale);
}

} // namespace AtenIpexTypeXPU
} // namespace at
