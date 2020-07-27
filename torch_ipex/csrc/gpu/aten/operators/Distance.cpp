#include <ATen/ATen.h>
#include <ATen/native/Distance.h>

#include <core/DPCPP.h>
#include <core/Memory.h>
#include <core/Stream.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
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
      return dist == 0.0 ? static_cast<scalar_t>(0) : grad * diff / dist;
    }
  };

  // General p norm
  struct p {
    static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
      agg += Numerics<scalar_t>::pow(diff, p);
    }
    static scalar_t finish(const scalar_t agg, const scalar_t p) {
      return Numerics<scalar_t>::pow(agg, static_cast<scalar_t>(1) / p);
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

template <
    typename scalar_t,
    typename F,
    typename nd_item_id,
    typename local_shared>
static inline scalar_t reduce_agg(
    scalar_t agg,
    nd_item_id item_id,
    const local_shared& local_shared_mem) {
  auto local_idx = item_id.get_local_id(0);
  auto group_size = item_id.get_local_range().size();

  local_shared_mem[local_idx] = agg;
  decltype(group_size) __k = 1;
  do {
    item_id.barrier(DPCPP::access::fence_space::local_space);
    if (local_idx % (2 * __k) == 0 && local_idx + __k < group_size) {
      F::agg(local_shared_mem[local_idx], local_shared_mem[local_idx + __k]);
    }
    __k *= 2;
  } while (__k < group_size);
  return local_shared_mem[local_idx];
}

template <int p_tpye, typename... T>
class DPCPPOpPdist {};

template <typename scalar_t, typename F, int p_tpye>
static void pdist_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n,
    const int64_t m,
    const scalar_t p,
    const double n2,
    const double n2_squared_minus_1) {
  const auto ngroups = result.numel();
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dpcpp_queue);

  // TODO: this is not optimized if the m is smaller than 256. The work item is
  // wasted (m-256).
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = get_buffer<dpcpp_discard_w_mode>(__cgh, result.data_ptr<scalar_t>());
    auto in_data = get_buffer<dpcpp_r_mode>(__cgh, self.data_ptr<scalar_t>());
    // Create the local shared memory for reducing
    DPCPP::accessor<scalar_t, 1, dpcpp_rw_mode, DPCPP::access::target::local>
        shared(wgroup_size, __cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto out_ptr = get_pointer(out_data);
      auto in_ptr = get_pointer(in_data);

      const size_t k = item_id.get_group_linear_id();
      const size_t stride = item_id.get_local_range().size();

      // The -1 accounts for floating point truncation issues
      int64_t i = static_cast<int64_t>(
          (n2 - device_sqrt<double>(n2_squared_minus_1 - 2 * k)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

      const scalar_t* const start = in_ptr + i * m;
      const scalar_t* const end = start + m;
      const scalar_t* a = start + item_id.get_local_linear_id();
      const scalar_t* b = in_ptr + j * m + item_id.get_local_linear_id();
      scalar_t agg = 0.0;
      for (; a < end; a += stride, b += stride) {
        F::inc(
            agg,
            Numerics<scalar_t>::abs(
                static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
            p);
      }

      agg = reduce_agg<scalar_t, F>(agg, item_id, shared);
      if (item_id.get_local_linear_id() == 0) {
        out_ptr[k] = F::finish(agg, p);
      }
    };

    __cgh.parallel_for<DPCPPOpPdist<p_tpye, scalar_t>>(
        DPCPP::nd_range</*dim=*/1>(ngroups * wgroup_size, wgroup_size), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <int p_type, typename... T>
class DPCPPOpPdistBackward {};

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
    const scalar_t p,
    const double n2,
    const double n2_squared_minus_1) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dpcpp_queue);

  // TODO: this is not optimized if the m is smaller than 256. The work item is
  // wasted (m-256).
  int64_t m_round = ((m + wgroup_size - 1) / (wgroup_size));
  DPCPP::range<2> global_range(
      dist.numel() /**wgroup_size*/, m_round * wgroup_size);
  DPCPP::range<2> local_range(/*wgroup_size*/ 1, wgroup_size);
  DPCPP::nd_range<2> work_load(global_range, local_range);

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = get_buffer<dpcpp_discard_w_mode>(__cgh, buffer.data_ptr<scalar_t>());
    auto in_data = get_buffer<dpcpp_r_mode>(__cgh, self.data_ptr<scalar_t>());
    auto grad_data = get_buffer<dpcpp_r_mode>(__cgh, grad.data_ptr<scalar_t>());
    auto dist_data = get_buffer<dpcpp_r_mode>(__cgh, dist.data_ptr<scalar_t>());

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      auto out_ptr = get_pointer(out_data);
      auto in_ptr = get_pointer(in_data);
      auto grad_ptr = get_pointer(grad_data);
      auto dist_ptr = get_pointer(dist_data);

      const int k = item_id.get_global_id(0);
      const int init = item_id.get_group(1) * item_id.get_local_range(1) +
          item_id.get_local_id(1);
      const int stride = item_id.get_local_range(1);

      if (k >= combs) {
        return;
      }

      int64_t i = static_cast<int64_t>(
          (n2 - device_sqrt<double>(n2_squared_minus_1 - 2 * k)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
      int64_t ib = j - 1;
      int64_t jb = i;

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
        const scalar_t res = F::backward(
            static_cast<scalar_t>(*self_i) - static_cast<scalar_t>(*self_j),
            grad_k,
            dist_k,
            p);
        *buff_i = res;
        *buff_j = -res;
      }
    };

    __cgh.parallel_for<DPCPPOpPdistBackward<p_type, scalar_t>>(work_load, kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
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
        } else if (std::isinf(p)) {
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
        } else if (std::isinf(p)) {
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
} // namespace AtenIpexTypeDPCPP
} // namespace at
