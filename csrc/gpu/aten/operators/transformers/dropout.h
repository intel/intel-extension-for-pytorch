#include <ATen/ATen.h>

#include <ATen/record_function.h>
#include <core/Generator.h>
#include <runtime/Utils.h>

#include "../DistributionTemplates.h"
#include "../RandomEngine.h"

#include <aten/operators/MemoryAccess.h>

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

const int UNROLL = 4;
namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
int get_vector_size(Tensor mask) {
  int vec_size = 4;
  // get the vector size
  if (!mask.is_non_overlapping_and_dense()) {
    vec_size = 1;
  } else {
    vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
        dpcppGetDeviceIdOfCurrentQueue(), (char*)mask.data_ptr());
  }

  // check that we'd have no remainders - prefer a smaller vector size with no
  // remainders over a larger vector and remainder.
  bool can_vectorize = true;
  do {
    can_vectorize = mask.numel() % vec_size == 0;
    if (!can_vectorize)
      vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

template <typename mask_t, typename item_t>
inline void dropout_mask_only_kernel_unroll(
    item_t& item,
    mask_t* out,
    int64_t totalElements,
    double p,
    PhiloxState philox_args) {
  auto thread_idx = item.get_local_id(0);
  auto thread_range = item.get_local_range(0);
  auto group_idx = item.get_group(0);
  auto group_range = item.get_group_range(0);

  auto seeds = philox_unpack(philox_args);
  int64_t idx = group_idx * thread_range + thread_idx;
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  int64_t rounded_size =
      ((totalElements - 1) / (thread_range * group_range * UNROLL) + 1) *
      thread_range * group_range * UNROLL;
  for (int64_t linearIndex = idx; linearIndex < rounded_size;
       linearIndex += group_range * thread_range * UNROLL) {
    float4 rand = rand_uniform4(&state);
    rand.x = rand.x > p;
    rand.y = rand.y > p;
    rand.z = rand.z > p;
    rand.w = rand.w > p;

#pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      int64_t li = linearIndex + thread_range * group_range * ii;
      if (li < totalElements) {
        out[li] = (mask_t)(&rand.x)[ii];
      }
    }
  }
}

template <typename mask_t>
struct DropoutMaskOnlyKernelFunctor2 {
  void operator()(sycl::nd_item<1> item) const {
    dropout_mask_only_kernel_unroll<mask_t>(
        item, mask, nelem, p, rng_engine_inputs);
  }
  DropoutMaskOnlyKernelFunctor2(
      mask_t* mask_,
      int64_t nelem_,
      double p_,
      PhiloxState rng_engine_inputs_)
      : mask(mask_),
        nelem(nelem_),
        p(p_),
        rng_engine_inputs(rng_engine_inputs_) {}

 private:
  mask_t* mask;
  int64_t nelem;
  double p;
  PhiloxState rng_engine_inputs;
};

template <typename mask_t, int vec_size, typename item_t>
inline void dropout_mask_only_kernel_vec(
    item_t& item,
    mask_t* out,
    int64_t totalElements,
    double p,
    PhiloxState philox_args) {
  // make sure we don't break assumption that we can't have > 4 elements /
  // thread
  static_assert(vec_size <= 4, "Value of vec_size must be in [2, 4]");

  using MaskLoadT = native::Memory::aligned_vector_loop<mask_t, vec_size>;

  auto thread_idx = item.get_local_id(0);
  auto thread_range = item.get_local_range(0);
  auto group_idx = item.get_group(0);
  auto group_range = item.get_group_range(0);

  auto seeds = philox_unpack(philox_args);
  int64_t idx = group_idx * thread_range + thread_idx;
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  bool gxvec_loop_state = 0;
  float4 rand;

  // Note: Vectorized loads means we'll stride each thread by an additional
  // vec_size factor, as we'll load vec_size elements at a time
  for (int64_t linearIndex = idx * vec_size; linearIndex < totalElements;
       linearIndex += group_range * thread_range * vec_size) {
    if ((vec_size == 4) || (gxvec_loop_state == 0)) {
      rand = rand_uniform4(&state);
    } else {
      // sets up the last two values we generated last iteration to be used this
      // iteration.
      rand.x = rand.z;
      rand.y = rand.w;
      gxvec_loop_state ^= 1;
    }

    rand.x = rand.x > p;
    rand.y = rand.y > p;
    if (vec_size == 4) {
      rand.z = rand.z > p;
      rand.w = rand.w > p;
    }

    mask_t mask[vec_size];

// Perform the actual computation
#pragma unroll
    for (int ii = 0; ii < vec_size; ii++) {
      mask[ii] = (mask_t)(&rand.x)[ii];
    }
    *(reinterpret_cast<MaskLoadT*>(&out[linearIndex])) =
        *reinterpret_cast<MaskLoadT*>(&mask[0]);
  }
}

template <typename mask_t, int Num>
struct DropoutMaskOnlyKernelFunctor1 {
  void operator()(sycl::nd_item<1> item) const {
    dropout_mask_only_kernel_vec<mask_t, Num>(
        item, mask, nelem, p, rng_engine_inputs);
  }
  DropoutMaskOnlyKernelFunctor1(
      mask_t* mask_,
      int64_t nelem_,
      double p_,
      PhiloxState rng_engine_inputs_)
      : mask(mask_),
        nelem(nelem_),
        p(p_),
        rng_engine_inputs(rng_engine_inputs_) {}

 private:
  mask_t* mask;
  int64_t nelem;
  double p;
  PhiloxState rng_engine_inputs;
};

template <typename mask_t>
Tensor& dropout_mask_only(Tensor& mask, double p) {
  TORCH_CHECK(mask.is_contiguous(), "Only generate contiguous mask tensor");
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      c10::nullopt, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  int64_t nelem = mask.numel();
  int vec_size = get_vector_size<mask_t>(mask);
  uint64_t counter_offset;
  uint32_t group_range, group_size;
  std::tie(counter_offset, group_range, group_size) =
      calc_execution_policy(nelem / vec_size);
  auto glb_range = group_range * group_size;
  auto loc_range = group_size;

  std::pair<uint64_t, uint64_t> seeds;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    seeds = gen->philox_engine_inputs(counter_offset);
  }
  PhiloxState rng_engine_inputs(std::get<0>(seeds), std::get<1>(seeds));

  auto& q = dpcppGetCurrentQueue();

  switch (vec_size) {
    case 16:
    case 8:
    case 4: {
      auto cgf = DPCPP_Q_CGF(cgh) {
        constexpr int Num = 4;
        DropoutMaskOnlyKernelFunctor1<mask_t, Num> kfn(
            mask.data_ptr<mask_t>(), nelem, p, rng_engine_inputs);
        cgh.parallel_for<decltype(kfn)>(
            sycl::nd_range<1>(glb_range, loc_range), kfn);
      };
      DPCPP_Q_SUBMIT(q, cgf);
    } break;
    case 2: {
      auto cgf = DPCPP_Q_CGF(cgh) {
        constexpr int Num = 2;
        DropoutMaskOnlyKernelFunctor1<mask_t, Num> kfn(
            mask.data_ptr<mask_t>(), nelem, p, rng_engine_inputs);
        cgh.parallel_for<decltype(kfn)>(
            sycl::nd_range<1>(glb_range, loc_range), kfn);
      };
      DPCPP_Q_SUBMIT(q, cgf);
    } break;
    case 1: {
      auto cgf = DPCPP_Q_CGF(cgh) {
        DropoutMaskOnlyKernelFunctor2<mask_t> kfn(
            mask.data_ptr<mask_t>(), nelem, p, rng_engine_inputs);
        cgh.parallel_for<decltype(kfn)>(
            sycl::nd_range<1>(glb_range, loc_range), kfn);
      };
      DPCPP_Q_SUBMIT(q, cgf);
    } break;
  };

  return mask;
}
} // namespace AtenIpexTypeXPU
} // namespace at