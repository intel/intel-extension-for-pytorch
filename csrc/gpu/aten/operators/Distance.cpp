#include <ATen/ATen.h>
#include <ATen/native/Distance.h>

#include <core/Memory.h>
#include <core/Stream.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "BatchKernel.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <torch/custom_class.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
static scalar_t device_sqrt(scalar_t val) {
  return Numerics<scalar_t>::sqrt(val);
};

template <typename scalar_t>
class dists {
 public:
  static scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }

  // Zero norm
  struct zero {
    static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
      agg += diff != 0.0;
    }
    static scalar_t finish(const scalar_t agg, const scalar_t p) {
      return agg;
    }
    static void agg(scalar_t& update, const scalar_t other) {
      update += other;
    }
  };

  // One norm
  struct one {
    static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
      agg += diff;
    }
    static scalar_t finish(const scalar_t agg, const scalar_t p) {
      return agg;
    }
    static void agg(scalar_t& update, const scalar_t other) {
      update += other;
    }
    static scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return grad * sign(diff);
    }
  };

  // Special case backward when p is less than two
  struct lt_two {
    static scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return dist == 0.0 ? static_cast<scalar_t>(0)
                         : sign(diff) *
              Numerics<scalar_t>::pow(Numerics<scalar_t>::abs(diff), p - 1) *
              grad / Numerics<scalar_t>::pow(dist, p - 1);
    }
  };

  // Two norm
  struct two {
    static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
      agg += diff * diff;
    }
    static scalar_t finish(const scalar_t agg, const scalar_t p) {
      return device_sqrt<scalar_t>(agg);
    }
    static void agg(scalar_t& update, const scalar_t other) {
      update += other;
    }
    static scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return dist == 0.0f ? static_cast<scalar_t>(0) : grad * diff / dist;
    }
  };

  // General p norm
  struct p {
    static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
      // TODO:
      // Here had an unknown bug which affected other code segments
      // unexpectedly. See Jira:
      // https://jira.devtools.intel.com/browse/PYTORCHDGQ-958 for details Below
      // is what code we wrote before and will trigger the bug
      //
      // agg += Numerics<scalar_t>::pow(diff, p);
      agg += static_cast<scalar_t>(std::pow(static_cast<scalar_t>(diff), p));
    }
    static scalar_t finish(const scalar_t agg, const scalar_t p) {
      // TODO:
      // Here had an unknown bug which affected other code segments
      // unexpectedly. See Jira:
      // https://jira.devtools.intel.com/browse/PYTORCHDGQ-958 for details Below
      // is what code we wrote before and will trigger the bug
      //
      // return Numerics<scalar_t>::pow(agg, static_cast<scalar_t>(1) / p);
      return static_cast<scalar_t>(
          std::pow(static_cast<scalar_t>(agg), 1.0f / p));
    }
    static void agg(scalar_t& update, const scalar_t other) {
      update += other;
    }
    static scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return dist == 0.0 ? static_cast<scalar_t>(0)
                         : diff *
              Numerics<scalar_t>::pow(Numerics<scalar_t>::abs(diff), p - 2) *
              grad / Numerics<scalar_t>::pow(dist, p - 1);
    }
  };

  // Inf norm
  struct inf {
    static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
      if (diff > agg) {
        agg = diff;
      }
    }
    static scalar_t finish(const scalar_t agg, const scalar_t p) {
      return agg;
    }
    static void agg(scalar_t& update, const scalar_t other) {
      if (other > update) {
        update = other;
      }
    }
    static scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return grad * sign(diff) * (Numerics<scalar_t>::abs(diff) == dist);
    }
  };
};

template <int SG_SIZE, typename scalar_t, typename F, typename nd_item>
DPCPP_DEVICE scalar_t subgroup_reduce_agg_impl(nd_item item, scalar_t value) {
  const auto sg = item.get_sub_group();

#pragma unroll
  for (int offset = (SG_SIZE >> 1); offset > 0; offset >>= 1) {
    F::agg(value, sg.shuffle_down(value, offset));
  }
  return value;
}

template <typename scalar_t, typename F, typename nd_item>
DPCPP_DEVICE scalar_t
subgroup_reduce_agg(nd_item item, scalar_t value, const int sg_size) {
  scalar_t ret;
  switch (sg_size) {
    case 8:
      ret = subgroup_reduce_agg_impl<8, scalar_t, F, nd_item>(item, value);
      break;
    case 16:
      ret = subgroup_reduce_agg_impl<16, scalar_t, F, nd_item>(item, value);
      break;
    case 32:
      ret = subgroup_reduce_agg_impl<32, scalar_t, F, nd_item>(item, value);
      break;
    case 64:
      ret = subgroup_reduce_agg_impl<64, scalar_t, F, nd_item>(item, value);
      break;
    default:
      SYCL_KERNEL_ASSERT(false);
  }
  return ret;
}

template <
    typename scalar_t,
    typename F,
    typename nd_item,
    typename local_shared>
static inline scalar_t reduce_agg(
    scalar_t agg,
    nd_item item,
    const local_shared& local_shared_mem) {
  const auto sg = item.get_sub_group();
  const int sg_size = sg.get_local_range()[0];

  const int group_size = item.get_local_range(0);
  const int sg_num = group_size / sg_size;

  const int local_id = item.get_local_id(0);
  const int lane_id = local_id % sg_size;
  const int sg_id = local_id / sg_size;
  agg = subgroup_reduce_agg<scalar_t, F, nd_item>(item, agg, sg_size);
  item.barrier(dpcpp_local_fence);
  if (0 == lane_id) {
    local_shared_mem[sg_id] = agg;
  }
  item.barrier(dpcpp_local_fence);
  agg = (local_id < sg_num) ? local_shared_mem[lane_id] : (scalar_t)0.0f;
  if (0 == sg_id) {
    agg = subgroup_reduce_agg<scalar_t, F, nd_item>(item, agg, sg_size);
  }

  return agg;
}

Tensor _euclidean_dist(const Tensor& x1, const Tensor& x2) {
  Tensor x1_norm = x1.pow(2).sum(-1, true);
  Tensor x1_pad = at::ones_like(x1_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor x2_norm = x2.pow(2).sum(-1, true);
  Tensor x2_pad = at::ones_like(x2_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor x1_ = at::cat({x1.mul(-2), x1_norm, x1_pad}, -1);
  Tensor x2_ = at::cat({x2, x2_pad, x2_norm}, -1);
  Tensor result = x1_.matmul(x2_.transpose(-2, -1));
  result.clamp_min_(0).sqrt_();
  return result;
}

std::tuple<Tensor, Tensor> _euclidean_dist_backward(
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const Tensor& res) {
  if (!grad.defined()) {
    return std::tuple<Tensor, Tensor>(Tensor(), Tensor());
  }
  // handle case at 0 where we return a subgradient containing 0
  Tensor ratio = grad / res;
  ratio.masked_fill_(res == 0, 0);
  return std::tuple<Tensor, Tensor>{
      x1 * ratio.sum(-1, true) - ratio.matmul(x2),
      x2 * ratio.sum(-2, false).unsqueeze(-1) - ratio.mT().matmul(x1)};
}

template <typename scalar_t, typename F, int p_tpye>
static void pdist_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n,
    const int64_t m,
    const double p,
    const double n2,
    const double n2_squared_minus_1) {
  const auto ngroups = result.numel();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  while (wgroup_size >> 1 >= m && wgroup_size >> 1 >= 32 /* sg_size */) {
    wgroup_size >>= 1;
  }
  using accscalar_t = acc_type<scalar_t>;
  auto p_val = static_cast<accscalar_t>(p);
  auto n2_val = static_cast<accscalar_t>(n2);
  auto n2_squared_minus_1_val = static_cast<accscalar_t>(n2_squared_minus_1);

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = result.data_ptr<scalar_t>();
    auto in_data = self.data_ptr<scalar_t>();
    // Create the local shared memory for reducing
    dpcpp_local_acc_t<scalar_t, 1> shared(wgroup_size, __cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto out_ptr = out_data;
      auto in_ptr = in_data;

      const size_t k = item_id.get_group_linear_id();
      const size_t stride = item_id.get_local_range().size();

      int64_t i = static_cast<int64_t>(
          (n2_val - device_sqrt<accscalar_t>(n2_squared_minus_1_val - 2 * k)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

      const scalar_t* const start = in_ptr + i * m;
      const scalar_t* const end = start + m;
      const scalar_t* a = start + item_id.get_local_linear_id();
      const scalar_t* b = in_ptr + j * m + item_id.get_local_linear_id();
      scalar_t agg = 0.0f;
      for (; a < end; a += stride, b += stride) {
        F::inc(
            agg,
            Numerics<scalar_t>::abs(
                static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
            p_val);
      }

      agg = reduce_agg<scalar_t, F>(agg, item_id, shared);
      if (item_id.get_local_linear_id() == 0) {
        out_ptr[k] = F::finish(agg, p_val);
      }
    };

    __cgh.parallel_for(
        sycl::nd_range</*dim=*/1>(ngroups * wgroup_size, wgroup_size), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename F, int p_type>
static void pdist_backward_kernel_impl(
    Tensor& buffer,
    const Tensor& grad,
    const Tensor& self,
    const Tensor& dist,
    int64_t gs,
    const int64_t n,
    const int64_t m,
    const int64_t combs,
    const double p,
    const double n2,
    const double n2_squared_minus_1) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  static constexpr int val_per_wi = 8;
  BatchKernelConfig cfg = {dist.numel(), m / val_per_wi, 1, dist.numel(), true};
  sycl::nd_range<2> work_load(cfg.global_size(), cfg.group_size());
  using accscalar_t = acc_type<scalar_t>;
  auto p_val = static_cast<accscalar_t>(p);
  auto n2_val = static_cast<accscalar_t>(n2);
  auto n2_squared_minus_1_val = static_cast<accscalar_t>(n2_squared_minus_1);

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_ptr = buffer.data_ptr<scalar_t>();
    auto in_ptr = self.data_ptr<scalar_t>();
    auto grad_ptr = grad.data_ptr<scalar_t>();
    auto dist_ptr = dist.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item_id) {
      auto desc = cfg.get_item_desc(item_id);
      const int k = desc.glb_batch;
      const int stride = desc.chunk_num * desc.chunk_size;
      const int init = desc.chunk * desc.chunk_size + desc.chunk_off;

      if (k >= combs) {
        return;
      }

      // select row i, j depending on k
      int64_t i = static_cast<int64_t>(
          (n2_val - device_sqrt<accscalar_t>(n2_squared_minus_1_val - 2 * k)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
      int64_t ib = j - i - 1;
      int64_t jb = n - 2 - i;

      const scalar_t grad_k = grad_ptr[k * gs];
      const scalar_t dist_k = dist_ptr[k];

      const scalar_t* const start = in_ptr + i * m;
      const scalar_t* const end = start + m;
      const scalar_t* self_i = start + init;
      const scalar_t* self_j = in_ptr + j * m + init;
      scalar_t* buff_i = out_ptr + (ib * n + i) * m + init;
      scalar_t* buff_j = out_ptr + (jb * n + j) * m + init;

      for (; self_i < end; self_i += stride,
                           self_j += stride,
                           buff_i += stride,
                           buff_j += stride) {
        const scalar_t res =
            F::backward(*self_i - *self_j, grad_k, dist_k, p_val);
        *buff_i = res;
        *buff_j = -res;
      }
    };

    __cgh.parallel_for(work_load, kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

void pdist_forward(Tensor& result, const Tensor& self, double p) {
  int64_t n = self.size(0);
  int64_t m = self.size(1);
  // https://github.com/pytorch/pytorch/issues/15511 demonstrated we need to do
  // some math in fp64 -- this is just minimizing the amount of fp64 math we do
  // on the device.
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "pdist_dpcpp",
      [&] {
        if (p == 0.0) {
          pdist_kernel_impl<scalar_t, dists<scalar_t>::zero, 0>(
              result, self, n, m, p, n2, n2_squared_minus_1);
        } else if (p == 1.0) {
          pdist_kernel_impl<scalar_t, dists<scalar_t>::one, 1>(
              result, self, n, m, p, n2, n2_squared_minus_1);
        } else if (p == 2.0) {
          pdist_kernel_impl<scalar_t, dists<scalar_t>::two, 2>(
              result, self, n, m, p, n2, n2_squared_minus_1);
        } else if (Numerics<scalar_t>::isinf(p)) {
          pdist_kernel_impl<scalar_t, dists<scalar_t>::inf, 3>(
              result, self, n, m, p, n2, n2_squared_minus_1);
        } else {
          pdist_kernel_impl<scalar_t, dists<scalar_t>::p, 4>(
              result, self, n, m, p, n2, n2_squared_minus_1);
        }
      });
}

void pdist_backward(
    Tensor& result,
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& dist) {
  if (p == 0.0 || grad.numel() == 0 || self.numel() == 0) {
    result.fill_(0);
    return;
  }
  const int64_t n = result.size(0);
  const int64_t m = self.size(1);
  // https://github.com/pytorch/pytorch/issues/15511 demonstrated we need to do
  // some math in fp64 -- this is just minimizing the amount of fp64 math we do
  // on the device.
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  Tensor buffer =
      at::empty({n - 1, result.size(0), result.size(1)}, result.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, self.scalar_type(), "pdist_backward", [&] {
        if (p == 1.0) {
          pdist_backward_kernel_impl<scalar_t, dists<scalar_t>::one, 0>(
              buffer,
              grad,
              self,
              dist,
              grad.stride(0),
              n,
              m,
              dist.numel(),
              p,
              n2,
              n2_squared_minus_1);
        } else if (p < 2.0) {
          pdist_backward_kernel_impl<scalar_t, dists<scalar_t>::lt_two, 1>(
              buffer,
              grad,
              self,
              dist,
              grad.stride(0),
              n,
              m,
              dist.numel(),
              p,
              n2,
              n2_squared_minus_1);
        } else if (p == 2.0) {
          pdist_backward_kernel_impl<scalar_t, dists<scalar_t>::two, 2>(
              buffer,
              grad,
              self,
              dist,
              grad.stride(0),
              n,
              m,
              dist.numel(),
              p,
              n2,
              n2_squared_minus_1);
        } else if (Numerics<scalar_t>::isinf(p)) {
          pdist_backward_kernel_impl<scalar_t, dists<scalar_t>::inf, 3>(
              buffer,
              grad,
              self,
              dist,
              grad.stride(0),
              n,
              m,
              dist.numel(),
              p,
              n2,
              n2_squared_minus_1);
        } else {
          pdist_backward_kernel_impl<scalar_t, dists<scalar_t>::p, 4>(
              buffer,
              grad,
              self,
              dist,
              grad.stride(0),
              n,
              m,
              dist.numel(),
              p,
              n2,
              n2_squared_minus_1);
        }
      });

  at::sum_out(result, buffer, 0);
}

template <typename scalar_t, typename F, int p_type>
static void cdist_forward_kernel_impl(
    Tensor& result,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const int64_t r1,
    const int64_t r2,
    const int64_t m,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size) {
  const auto ngroups = result.numel();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = 32;
  using accscalar_t = acc_type<scalar_t>;
  auto p_val = static_cast<accscalar_t>(p);

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = result.data_ptr<scalar_t>();
    auto x1_data = x1.data_ptr<scalar_t>();
    auto x2_data = x2.data_ptr<scalar_t>();
    // Create the local shared memory for reducing
    dpcpp_local_acc_t<scalar_t, 1> shared(wgroup_size, __cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto out_ptr = out_data;
      auto x1_ptr = x1_data;
      auto x2_ptr = x2_data;

      const int64_t group_id = item_id.get_group_linear_id();
      const int64_t local_id = item_id.get_local_linear_id();
      const int64_t l = group_id / r_size;
      const int64_t k = group_id % r_size;
      const int64_t i = k / r2;
      const int64_t j = k % r2;
      const size_t stride = item_id.get_local_range().size();

      scalar_t* start = x1_ptr + l * l1_size + i * m;
      scalar_t* end = start + m;
      scalar_t* a = start + local_id;
      scalar_t* b = x2_ptr + l * l2_size + j * m + local_id;

      scalar_t agg = 0.0f;
      for (; a < end; a += stride, b += stride) {
        F::inc(
            agg,
            Numerics<scalar_t>::abs(
                static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
            p_val);
      }
      agg = reduce_agg<scalar_t, F>(agg, item_id, shared);
      if (local_id == 0) {
        out_ptr[group_id] = F::finish(agg, p_val);
      }
    };

    __cgh.parallel_for(
        sycl::nd_range</*dim=*/1>(ngroups * wgroup_size, wgroup_size), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

static Tensor cdist_forward(
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode) {
  int64_t mode = compute_mode.value_or(0);
  int64_t r1 = x1.size(-2);
  int64_t r2 = x2.size(-2);
  int64_t m = x1.size(-1);
  int64_t dim1 = x1.dim();
  int64_t dim2 = x2.dim();
  IntArrayRef batchsize1(x1.sizes().data(), dim1 - 2);
  IntArrayRef batchsize2(x2.sizes().data(), dim2 - 2);
  std::vector<int64_t> expand_batchsize =
      at::infer_size(batchsize1, batchsize2);
  std::vector<int64_t> x1_expand_size(expand_batchsize);
  x1_expand_size.insert(x1_expand_size.end(), {r1, m});
  std::vector<int64_t> x2_expand_size(expand_batchsize);
  x2_expand_size.insert(x2_expand_size.end(), {r2, m});

  int expand_batch_product = std::accumulate(
      expand_batchsize.begin(),
      expand_batchsize.end(),
      1,
      std::multiplies<int64_t>());
  std::vector<int64_t> x1_view{expand_batch_product, r1, m};
  std::vector<int64_t> x2_view{expand_batch_product, r2, m};

  Tensor x1_expanded = x1.expand(x1_expand_size).contiguous().view(x1_view);
  Tensor x2_expanded = x2.expand(x2_expand_size).contiguous().view(x2_view);

  std::vector<int64_t> output_shape(expand_batchsize);
  output_shape.insert(output_shape.end(), {r1, r2});

  Tensor result;
  if (r1 == 0 || r2 == 0) {
    result = at::empty(output_shape, x1.options());
  } else if (m == 0) {
    result = at::zeros(output_shape, x1.options());
  } else if (p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))) {
    Tensor dist = (expand_batch_product == 1)
        ? impl::_euclidean_dist(x1, x2)
        : impl::_euclidean_dist(x1_expanded, x2_expanded);
    result = dist.view(output_shape);
  } else {
    result = at::empty(output_shape, x1.options());
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x1.scalar_type(),
        "cdist_forward_dpcpp",
        [&] {
          if (p == 0.0) {
            cdist_forward_kernel_impl<scalar_t, dists<scalar_t>::zero, 0>(
                result,
                x1_expanded,
                x2_expanded,
                p,
                r1,
                r2,
                m,
                r1 * r2,
                r1 * m,
                r2 * m);
          } else if (p == 1.0) {
            cdist_forward_kernel_impl<scalar_t, dists<scalar_t>::one, 1>(
                result,
                x1_expanded,
                x2_expanded,
                p,
                r1,
                r2,
                m,
                r1 * r2,
                r1 * m,
                r2 * m);
          } else if (p == 2.0) {
            cdist_forward_kernel_impl<scalar_t, dists<scalar_t>::two, 2>(
                result,
                x1_expanded,
                x2_expanded,
                p,
                r1,
                r2,
                m,
                r1 * r2,
                r1 * m,
                r2 * m);
          } else if (Numerics<scalar_t>::isinf(p)) {
            cdist_forward_kernel_impl<scalar_t, dists<scalar_t>::inf, 3>(
                result,
                x1_expanded,
                x2_expanded,
                p,
                r1,
                r2,
                m,
                r1 * r2,
                r1 * m,
                r2 * m);
          } else {
            cdist_forward_kernel_impl<scalar_t, dists<scalar_t>::p, 4>(
                result,
                x1_expanded,
                x2_expanded,
                p,
                r1,
                r2,
                m,
                r1 * r2,
                r1 * m,
                r2 * m);
          }
        });
  }
  return result;
}

template <typename scalar_t, typename F, int p_type>
static void cdist_backward_kernel_impl(
    Tensor& buffer,
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const Tensor& dist,
    int64_t gs,
    const double p,
    const int64_t r1,
    const int64_t r2,
    const int64_t m,
    const int64_t count,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
  auto batch = (x1.dim() > 2) ? x1.size(0) : 1;
  const int group_size_x = 256 > wgroup_size ? wgroup_size : 256;
  const int group_size_y = wgroup_size / group_size_x;
  const int group_num_x = (m + group_size_x * 32 - 1) / (group_size_x * 32);
  using accscalar_t = acc_type<scalar_t>;
  auto p_val = static_cast<accscalar_t>(p);

  const int64_t group_num_temp = (count + group_size_y - 1) / group_size_y;

  const int group_num_y = (group_num_temp - 1) / 65535 + 1;
  const int group_num_z = (group_num_temp - 1) / group_num_y + 1;

  sycl::range<3> global_range(
      group_size_x * group_num_x, group_size_y * group_num_y, 1 * group_num_z);
  sycl::range<3> local_range(group_size_x, group_size_y, 1);
  sycl::nd_range<3> work_load(global_range, local_range);

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto buff_data = buffer.data_ptr<scalar_t>();
    auto grad_data = grad.data_ptr<scalar_t>();
    auto dist_data = dist.data_ptr<scalar_t>();
    auto x1_data = x1.data_ptr<scalar_t>();
    auto x2_data = x2.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto buff_ptr = buff_data;
      auto grad_ptr = grad_data;
      auto dist_ptr = dist_data;
      auto x1_ptr = x1_data;
      auto x2_ptr = x2_data;

      const int y =
          (item.get_group(1) * group_num_z + item.get_group(2)) * group_size_y +
          item.get_local_id(1);
      const int init = item.get_group(0) * group_size_x + item.get_local_id(0);
      if (y >= count || init >= m) {
        return;
      }

      const int l = y / r_size;
      const int k = y % r_size;
      const int stride = group_size_x * group_num_x;
      const int l_size = r_size * m;

      int64_t i = k / r2;
      int64_t j = k % r2;

      const scalar_t grad_k = grad_ptr[y];
      const scalar_t dist_k = dist_ptr[y];

      const scalar_t* const start = x1_ptr + l * l1_size + i * m;
      const scalar_t* const end = start + m;
      const scalar_t* self_i = start + init;
      const scalar_t* self_j = x2_ptr + l * l2_size + j * m + init;

      scalar_t* buff_i = buff_ptr + l * l_size + (r1 * j + i) * m + init;

      for (; self_i < end;
           self_i += stride, self_j += stride, buff_i += stride) {
        const scalar_t res = F::backward(
            static_cast<scalar_t>(*self_i) - static_cast<scalar_t>(*self_j),
            grad_k,
            dist_k,
            p_val);
        *buff_i = res;
      }
    };

    __cgh.parallel_for(work_load, kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

static Tensor cdist_backward(
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const Tensor& cdist) {
  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);
  const int64_t count = cdist.numel();
  const int64_t gs = 1;
  const int64_t batch = (x1.dim() > 2) ? x1.size(0) : 1;
  Tensor result =
      at::empty_like(x1, x1.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (p == 0.0 || grad.numel() == 0 || x1.numel() == 0 || x2.numel() == 0) {
    result.fill_(0);
    return result;
  }

  if (2.0 == p && (r1 > 25 || r2 > 25)) {
    std::tuple<Tensor, Tensor> edist_tuple;
    edist_tuple = _euclidean_dist_backward(grad, x1, x2, cdist);
    result = std::get<0>(edist_tuple);
    return result;
  }

  Tensor buffer = (x1.dim() > 2)
      ? at::empty({batch, r2, r1, m}, result.options())
      : at::empty({r2, r1, m}, result.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, x1.scalar_type(), "cdist_backward_dpcpp", [&] {
        if (p == 1.0) {
          cdist_backward_kernel_impl<scalar_t, dists<scalar_t>::one, 0>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else if (p < 2.0) {
          cdist_backward_kernel_impl<scalar_t, dists<scalar_t>::lt_two, 1>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else if (p == 2.0) {
          cdist_backward_kernel_impl<scalar_t, dists<scalar_t>::two, 2>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else if (Numerics<scalar_t>::isinf(p)) {
          cdist_backward_kernel_impl<scalar_t, dists<scalar_t>::inf, 3>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else {
          cdist_backward_kernel_impl<scalar_t, dists<scalar_t>::p, 4>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        }
      });
  if (x1.dim() > 2) {
    at::sum_out(result, buffer, 1);
  } else {
    at::sum_out(result, buffer, 0);
  }
  return result;
}

} // namespace impl

Tensor _pdist_forward(const Tensor& self, const double p) {
  TORCH_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
  auto device = self.device().type();
  Tensor result = at::empty({0}, self.options());
  if (self.size(0) <= 1) {
    result.resize_({0});
  } else {
    int64_t n = self.size(0);
    int64_t c = n * (n - 1) / 2;
    result.resize_({c});
    if (self.size(1) == 0) {
      result.fill_(0);
    } else {
      impl::pdist_forward(result, self, p);
    }
  }
  return result;
}

Tensor _pdist_backward(
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& pdist) {
  TORCH_CHECK(
      self.is_contiguous(), "_pdist_backward requires self to be contiguous");
  TORCH_CHECK(
      pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");
  auto device = self.device().type();
  Tensor result = at::empty_like(self);
  impl::pdist_backward(result, grad, self, p, pdist);
  return result;
}

Tensor _cdist_forward(
    const Tensor& x1,
    const Tensor& x2,
    double p,
    c10::optional<int64_t> compute_mode) {
  TORCH_CHECK(
      x1.dim() >= 2,
      "cdist only supports at least 2D tensors, X1 got: ",
      x1.dim(),
      "D");
  TORCH_CHECK(
      x2.dim() >= 2,
      "cdist only supports at least 2D tensors, X2 got: ",
      x2.dim(),
      "D");
  TORCH_CHECK(
      x1.size(-1) == x2.size(-1),
      "X1 and X2 must have the same number of columns. X1: ",
      x1.size(-1),
      " X2: ",
      x2.size(-1));
  TORCH_CHECK(
      at::isFloatingType(x1.scalar_type()),
      "cdist only supports floating-point dtypes, but X1 got: ",
      x1.scalar_type());
  TORCH_CHECK(
      at::isFloatingType(x2.scalar_type()),
      "cdist only supports floating-point dtypes, but X2 got: ",
      x2.scalar_type());
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
  TORCH_CHECK(
      !x1.is_xpu() || x1.get_device() == x2.get_device(),
      "device of X1 (",
      x1.get_device(),
      ") must match device of X2 (",
      x2.get_device(),
      ")");
  return impl::cdist_forward(x1, x2, p, compute_mode);
}

Tensor _cdist_backward(
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    double p,
    const Tensor& cdist) {
  TORCH_CHECK(
      x1.is_contiguous(), "_cdist_backward requires X1 to be contiguous");
  TORCH_CHECK(
      x2.is_contiguous(), "_cdist_backward requires X2 to be contiguous");
  TORCH_CHECK(
      cdist.is_contiguous(), "_cdist_backward requires dist to be contiguous");
  TORCH_CHECK(
      grad.is_contiguous(), "_cdist_backward requires grad to be contiguous");
  return impl::cdist_backward(grad, x1, x2, p, cdist);
}

Tensor cdist(
    const Tensor& x1,
    const Tensor& x2,
    double p,
    c10::optional<int64_t> compute_mode) {
  TORCH_CHECK(
      x1.dim() >= 2,
      "cdist only supports at least 2D tensors, X1 got: ",
      x1.dim(),
      "D");
  TORCH_CHECK(
      x2.dim() >= 2,
      "cdist only supports at least 2D tensors, X2 got: ",
      x2.dim(),
      "D");
  TORCH_CHECK(
      x1.size(-1) == x2.size(-1),
      "X1 and X2 must have the same number of columns. X1: ",
      x1.size(-1),
      " X2: ",
      x2.size(-1));
  int64_t r1 = x1.size(-2);
  int64_t r2 = x2.size(-2);
  int64_t mode = compute_mode.value_or(0);
  return at::_cdist_forward(x1, x2, p, compute_mode);
}

} // namespace AtenIpexTypeXPU
} // namespace at
