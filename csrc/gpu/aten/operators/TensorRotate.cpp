#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <core/MemoryFormat.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename scalar_t,
    int N,
    typename index_type,
    bool unsigned_index,
    template <int, typename, bool>
    class OffsetCalculator>
void apply_rotary_embedding_impl(
    scalar_t* tensor,
    scalar_t* sin,
    scalar_t* cos,
    scalar_t* out,
    OffsetCalculator<N, index_type, unsigned_index> offset_calc,
    int64_t problem_size,
    int64_t total_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t wg_size = std::min(max_wg_size, problem_size);
  int64_t work_group_num = (total_size + problem_size - 1) / problem_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto item_idx = item_id.get_local_id(0);
      auto item_range = item_id.get_local_range(0);
      auto group_idx = item_id.get_group(0);
      auto sg = item_id.get_sub_group();
      scalar_t tensor_val;
      for (int i = item_idx; i < problem_size; i += item_range) {
        auto global_offset = group_idx * problem_size + i;
        const auto offset = offset_calc.get(global_offset);
        tensor_val = *(tensor + offset[1]);
        scalar_t scale = i % 2 == 0 ? -1 : 1;
        scalar_t shift_val = sg.shuffle_xor(tensor_val, 1) * scale;
        scalar_t sin_val = *(sin + offset[2]);
        scalar_t cos_val = *(cos + offset[3]);
        scalar_t out_val = shift_val * sin_val + tensor_val * cos_val;
        scalar_t* out_ptr = out + offset[0];
        *out_ptr = out_val;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_num * wg_size), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor apply_rotary_embedding(
    const Tensor& tensor,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& out) {
  int64_t tensor_ndim = tensor.ndimension();
  int64_t sin_dim = sin.ndimension();
  int64_t cos_dim = cos.ndimension();
  int64_t out_dim = out.ndimension();
  TORCH_CHECK(
      tensor_ndim == sin_dim && cos_dim == sin_dim && out_dim == sin_dim,
      "The dimensions of all tensor should be equal");
  int64_t numel = tensor.numel();
  int64_t problem_size = tensor.size(tensor_ndim - 1);
  TORCH_CHECK(
      tensor.size(tensor_ndim - 1) == sin.size(tensor_ndim - 1) &&
          sin.size(tensor_ndim - 1) == cos.size(tensor_ndim - 1) &&
          cos.size(tensor_ndim - 1) == out.size(tensor_ndim - 1),
      "The problem size of all tensor should be equal");
  TORCH_CHECK(
      tensor.size(tensor_ndim - 1) % 2 == 0,
      "The problem size should be divisible by 2")
  auto iter = TensorIteratorConfig()
                  .add_output(out)
                  .add_input(tensor)
                  .add_input(sin)
                  .add_input(cos)
                  .build();
  auto offset_calc = make_element_offset_calculator<4>(iter);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensor.scalar_type(),
      "apply_rotary_embedding",
      [&]() {
        apply_rotary_embedding_impl(
            static_cast<scalar_t*>(tensor.data_ptr()),
            static_cast<scalar_t*>(sin.data_ptr()),
            static_cast<scalar_t*>(cos.data_ptr()),
            static_cast<scalar_t*>(out.data_ptr()),
            offset_calc,
            problem_size,
            numel);
      });
  return out;
}

template <
    typename scalar_t,
    int N,
    typename index_type,
    bool unsigned_index,
    template <int, typename, bool>
    class OffsetCalculator>
void apply_rotary_embedding_two_kernel(
    scalar_t* query,
    scalar_t* key,
    float* sin,
    float* cos,
    scalar_t* query_out,
    scalar_t* key_out,
    OffsetCalculator<N, index_type, unsigned_index> offset_calc,
    int64_t problem_size,
    int64_t total_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t wg_size = std::min(max_wg_size, problem_size);
  int64_t work_group_num = (total_size + problem_size - 1) / problem_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item_id) {
      auto item_idx = item_id.get_local_id(1);
      auto item_range = item_id.get_local_range(1);
      auto group_idx = item_id.get_group(1);
      auto group_id = item_id.get_group(0);
      auto sg = item_id.get_sub_group();

      if (group_id == 0) {
        for (int i = item_idx; i < problem_size; i += item_range) {
          auto global_offset = group_idx * problem_size + i;
          const auto offset = offset_calc.get(global_offset);
          scalar_t query_val = *(query + offset[2]);
          scalar_t scale = i % 2 == 0 ? -1 : 1;
          scalar_t shift_val = sg.shuffle_xor(query_val, 1) * scale;
          float sin_val = *(sin + offset[4]);
          float cos_val = *(cos + offset[5]);
          *(query_out + offset[0]) =
              (scalar_t)((float)shift_val * sin_val + (float)query_val * cos_val);
        }
      } else {
        for (int i = item_idx; i < problem_size; i += item_range) {
          auto global_offset = group_idx * problem_size + i;
          const auto offset = offset_calc.get(global_offset);
          scalar_t key_val = *(key + offset[3]);
          scalar_t scale = i % 2 == 0 ? -1 : 1;
          scalar_t shift_val = sg.shuffle_xor(key_val, 1) * scale;
          float sin_val = *(sin + offset[4]);
          float cos_val = *(cos + offset[5]);
          *(key_out + offset[1]) =
              (scalar_t)((float)shift_val * sin_val + (float)key_val * cos_val);
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>({2, work_group_num * wg_size}),
            sycl::range<2>({1, wg_size})),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename scalar_t,
    int N,
    typename index_type,
    bool unsigned_index,
    template <int, typename, bool>
    class OffsetCalculator>
void apply_rotary_embedding_half_kernel(
    scalar_t* query,
    scalar_t* key,
    scalar_t* sin,
    scalar_t* cos,
    scalar_t* query_out,
    scalar_t* key_out,
    OffsetCalculator<N, index_type, unsigned_index> offset_calc,
    int64_t problem_size,
    int64_t total_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  // int64_t wg_size = std::min(max_wg_size, problem_size);
  int64_t wg_size = max_wg_size;
  int64_t work_group_num = (total_size + problem_size - 1) / problem_size;
  int64_t problem_half = problem_size / 2;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item_id) {
      auto item_idx = item_id.get_local_id(1);
      auto item_range = item_id.get_local_range(1);
      auto group_idx = item_id.get_group(1);
      auto group_id = item_id.get_group(0);
      auto sg = item_id.get_sub_group();

      if (group_id == 0) {
        for (int i = item_idx; i < problem_size; i += item_range) {
          auto global_offset1 = group_idx * problem_size + i;
          auto global_offset2 = i < problem_half
              ? group_idx * problem_size + problem_half + i
              : group_idx * problem_size - problem_half + i;
          float scale = i < problem_half ? -1 : 1;
          const auto offset1 = offset_calc.get(global_offset1);
          const auto offset2 = offset_calc.get(global_offset2);
          float query_val1 = *(query + offset1[2]);
          float query_val2 = *(query + offset2[2]);
          float sin_val = *(sin + offset1[4]);
          float cos_val = *(cos + offset1[5]);
          *(query_out + offset1[0]) = static_cast<scalar_t>(
              scale * query_val2 * sin_val + query_val1 * cos_val);
        }
      } else {
        for (int i = item_idx; i < problem_size; i += item_range) {
          auto global_offset1 = group_idx * problem_size + i;
          auto global_offset2 = i < problem_half
              ? group_idx * problem_size + problem_half + i
              : group_idx * problem_size - problem_half + i;
          float scale = i < problem_half ? -1 : 1;
          const auto offset1 = offset_calc.get(global_offset1);
          const auto offset2 = offset_calc.get(global_offset2);
          float key_val1 = *(key + offset1[3]);
          float key_val2 = *(key + offset2[3]);
          float sin_val = *(sin + offset1[4]);
          float cos_val = *(cos + offset1[5]);
          *(key_out + offset1[1]) = static_cast<scalar_t>(
              scale * key_val2 * sin_val + key_val1 * cos_val);
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>({2, work_group_num * wg_size}),
            sycl::range<2>({1, wg_size})),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

void apply_rotary_embedding_qk(
    const Tensor& query,
    const Tensor& key,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& query_out,
    Tensor& key_out,
    bool rotary_half) {
  int64_t query_ndim = query.ndimension();
  int64_t key_ndim = key.ndimension();
  int64_t sin_dim = sin.ndimension();
  int64_t cos_dim = cos.ndimension();
  int64_t query_out_dim = query_out.ndimension();
  int64_t key_out_dim = key_out.ndimension();
  TORCH_CHECK(
      query_ndim == sin_dim && cos_dim == sin_dim && query_out_dim == sin_dim,
      "The dimensions of all tensor should be equal");
  TORCH_CHECK(
      key_ndim == sin_dim && cos_dim == sin_dim && key_out_dim == sin_dim,
      "The dimensions of all tensor should be equal");
  int64_t numel = query.numel();
  int64_t problem_size = query.size(query_ndim - 1);
  TORCH_CHECK(
      query.size(query_ndim - 1) == sin.size(query_ndim - 1) &&
          sin.size(query_ndim - 1) == cos.size(query_ndim - 1) &&
          cos.size(query_ndim - 1) == query_out.size(query_ndim - 1),
      "The problem size of all tensor should be equal");
  TORCH_CHECK(
      key.size(key_ndim - 1) == sin.size(key_ndim - 1) &&
          sin.size(key_ndim - 1) == cos.size(key_ndim - 1) &&
          cos.size(key_ndim - 1) == key_out.size(key_ndim - 1),
      "The problem size of all tensor should be equal");
  TORCH_CHECK(
      query.size(query_ndim - 1) % 2 == 0,
      "The problem size should be divisible by 2")
  TORCH_CHECK(
      key.size(key_ndim - 1) % 2 == 0,
      "The problem size should be divisible by 2")
  auto iter = TensorIteratorConfig()
                  .add_output(query_out)
                  .add_output(key_out)
                  .add_input(query)
                  .add_input(key)
                  .add_input(sin)
                  .add_input(cos)
                  .check_all_same_dtype(false)
                  .build();
  auto offset_calc = make_element_offset_calculator<6>(iter);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "apply_rotary_embedding",
      [&]() {
        if (rotary_half) {
          apply_rotary_embedding_half_kernel(
              static_cast<scalar_t*>(query.data_ptr()),
              static_cast<scalar_t*>(key.data_ptr()),
              static_cast<scalar_t*>(sin.data_ptr()),
              static_cast<scalar_t*>(cos.data_ptr()),
              static_cast<scalar_t*>(query_out.data_ptr()),
              static_cast<scalar_t*>(key_out.data_ptr()),
              offset_calc,
              problem_size,
              numel);
        } else {
          apply_rotary_embedding_two_kernel(
              static_cast<scalar_t*>(query.data_ptr()),
              static_cast<scalar_t*>(key.data_ptr()),
              static_cast<float*>(sin.data_ptr()),
              static_cast<float*>(cos.data_ptr()),
              static_cast<scalar_t*>(query_out.data_ptr()),
              static_cast<scalar_t*>(key_out.data_ptr()),
              offset_calc,
              problem_size,
              numel);
        }
      });
}

void apply_rotary_embedding_two(
    const Tensor& query,
    const Tensor& key,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& query_out,
    Tensor& key_out) {
  apply_rotary_embedding_qk(query, key, sin, cos, query_out, key_out, false);
}

void apply_rotary_embedding_half(
    const Tensor& query,
    const Tensor& key,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& query_out,
    Tensor& key_out) {
  apply_rotary_embedding_qk(query, key, sin, cos, query_out, key_out, true);
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding", apply_rotary_embedding, c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding_two",
      apply_rotary_embedding_two,
      c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding_half",
      apply_rotary_embedding_half,
      c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at
