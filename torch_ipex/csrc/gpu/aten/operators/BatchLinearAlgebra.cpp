#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include <core/ApplyUtils.h>
#include <core/Context.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

#ifdef USE_ONEMKL
#include <mkl_sycl.hpp>
#include <mkl.h>
#endif

using namespace at::dpcpp;

DPCPP_DEF_K2(triuTrilSycl, typename scalar_t, typename IndexType, bool upper);

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t, typename IndexType, bool upper>
void triu_tril_dpcpp_kernel(
    scalar_t* result,
    const scalar_t* src,
    const IndexType stride0,
    const IndexType stride1,
    const int64_t k,
    const int64_t N) {
  auto queue = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(queue);
  auto num_groups = CeilDiv(N, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto result_data = get_buffer<dpcpp_w_mode>(cgh, result);
    auto src_data = get_buffer<dpcpp_r_mode>(cgh, src);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto result_ptr = get_pointer(result_data);
      auto src_ptr = get_pointer(src_data);

      for (size_t linearIndex = item.get_global_id(0); linearIndex < (size_t)N;
           linearIndex += item.get_global_range()[0]) {
        IndexType row, col;

        if (stride0 > stride1) {
          row = (IndexType)(linearIndex / stride0);
          col = (IndexType)(linearIndex % stride0 / stride1);
        } else {
          row = (IndexType)((linearIndex % stride1) / stride0);
          col = (IndexType)(linearIndex / stride1);
        }

        bool mask = upper ? (col - row >= k) : (col - row <= k);
        result_ptr[linearIndex] = mask ? src_ptr[linearIndex] : scalar_t(0);
      }
    };

    // kick off kernel
    cgh.parallel_for<DPCPP_K(triuTrilSycl, scalar_t, IndexType, upper)>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <bool upper>
Tensor& triu_tril_dpcpp_template(
    Tensor& result,
    const Tensor& self,
    int64_t k,
    const char* name) {
  int64_t N = self.numel();

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      self.scalar_type(),
      name,
      [&] {
        if (dpcpp::detail::canUse32BitIndexMath(self)) {
          auto self_info =
              dpcpp::detail::getTensorInfo<scalar_t, int32_t>(self);
          triu_tril_dpcpp_kernel<scalar_t, int32_t, upper>(
              result.data_ptr<scalar_t>(),
              self.data_ptr<scalar_t>(),
              self_info.strides[0],
              self_info.strides[1],
              k,
              N);
        } else {
          auto self_info =
              dpcpp::detail::getTensorInfo<scalar_t, int64_t>(self);
          triu_tril_dpcpp_kernel<scalar_t, int64_t, upper>(
              result.data_ptr<scalar_t>(),
              self.data_ptr<scalar_t>(),
              self_info.strides[0],
              self_info.strides[1],
              k,
              N);
        }
      });

  return result;
}

Tensor& tril_dpcpp_out(Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }

  return triu_tril_dpcpp_template<false>(result, self, k, "tril");
}

Tensor& tril_dpcpp_(Tensor& self, int64_t k) {
  return tril_dpcpp_out(self, self, k);
}

Tensor& triu_dpcpp_out(Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  return triu_tril_dpcpp_template<true>(result, self, k, "triu");
}

Tensor& triu_dpcpp_(Tensor& self, int64_t k) {
  return triu_dpcpp_out(self, self, k);
}

template<typename scalar_t>
static void apply_lu_dpcpp_(
    Tensor& self_,
    Tensor& pivots_,
    Tensor& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

  auto batch_size = native::batchCount(self_);
  auto m_ = self_.size(-2);
  auto n_ = self_.size(-1);
  Tensor self = self_.view({batch_size, m_, n_});
  Tensor pivots = pivots_.view({batch_size, m_});
  Tensor infos = infos_.view({batch_size});

  std::vector<std::int64_t> m(batch_size, m_);
  std::vector<std::int64_t> n(batch_size, n_);
  std::vector<std::int64_t> lda(batch_size, m_);
  std::vector< DPCPP::buffer<scalar_t,1>> A_buf;
  std::vector< DPCPP::buffer<int64_t,1>> ipiv_buf;
  std::vector< DPCPP::buffer<int64_t,1>> info_buf;

  for (std::int64_t i = 0; i < batch_size; i++)
  {
    auto a = make_buffer<scalar_t>(self[i].data_ptr());
    A_buf.emplace_back(a);
    auto ipiv = make_buffer<int64_t>(pivots[i].data_ptr());
    ipiv_buf.emplace_back(ipiv);
    auto info = make_buffer<int64_t>(infos[i].data_ptr());
    info_buf.emplace_back(info);
  }
  mkl::lapack::getrf_batch(dpcpp_queue, m, n, A_buf, lda, ipiv_buf, info_buf);
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}


} // namespace impl

Tensor& triu_out(Tensor& out, const Tensor& self, int64_t diagonal) {
  impl::triu_dpcpp_out(out, self, diagonal);
  return out;
}

Tensor& tril_out(Tensor& out, const Tensor& self, int64_t diagonal) {
  impl::tril_dpcpp_out(out, self, diagonal);
  return out;
}

Tensor& tril_(Tensor& self, int64_t diagonal) {
  return at::AtenIpexTypeDPCPP::tril_out(self, self, diagonal);
}

Tensor& triu_(Tensor& self, int64_t diagonal) {
  return at::AtenIpexTypeDPCPP::triu_out(self, self, diagonal);
}

std::tuple<Tensor,Tensor,Tensor> _lu_with_info(const Tensor & self, bool pivot, bool check_errors) {
  TORCH_CHECK(pivot, "lu without pivoting is not implemented on the DPCPP");
  TORCH_CHECK(self.dim() >= 2,
              "expected tensor with 2 or more dimensions, got size: ", self.sizes(),
              " instead");
  native::squareCheckInputs(self);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  auto pivots_tensor = at::empty(req_size, self.options().dtype(kLong));
  req_size.pop_back();
  auto infos_tensor = at::zeros(req_size, self.options().dtype(kLong));

  Tensor self_working_copy;
  if (self.numel() == 0) {
    self_working_copy = at::empty_like(self);
  } else {
    self_working_copy = native::cloneBatchedColumnMajor(self);
    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lu_dpcpp", [&]{
      impl::apply_lu_dpcpp_<scalar_t>(self_working_copy, pivots_tensor, infos_tensor);
    });
  }
  if (check_errors) {
    if (self.dim() > 2) {
      native::batchCheckErrors(infos_tensor, "lu");
    } else {
      native::singleCheckErrors(infos_tensor.item<int64_t>(), "lu");
    }
  }
  return std::make_tuple(self_working_copy, pivots_tensor, infos_tensor);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
