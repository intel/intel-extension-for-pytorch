#include <ATen/ATen.h>
#include <runtime/Utils.h>
#include <sycl/sycl.hpp>
#include "MemoryAccess.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "utils/CustomOperatorRegistration.h"
#include "xetla/sgmv.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename output_t, typename input_t, uint32_t vec_size>
class BgmvShrinkKernelFunctor {
 public:
  BgmvShrinkKernelFunctor(
      output_t* outputs,
      input_t* inputs,
      input_t* weights,
      int64_t* indices,
      const uint32_t hidden,
      const uint32_t rank,
      const float scale)
      : outputs(outputs),
        inputs(inputs),
        weights(weights),
        indices(indices),
        hidden(hidden),
        rank(rank),
        scale(scale) {}

  using accscalar_t = acc_type<input_t>;
  using vec_t = at::native::Memory::aligned_vector_loop<input_t, vec_size>;
  void operator()(sycl::nd_item<1> item) const {
    uint32_t local_id = item.get_local_linear_id();
    uint32_t group_id = item.get_group_linear_id();
    uint32_t group_size = item.get_local_range(0);
    uint32_t batch_id = group_id / rank;
    uint32_t rank_id = group_id % rank;
    int64_t index = indices[batch_id];
    if (index < 0) {
      return;
    }
    input_t* input_ptr = inputs + batch_id * hidden;
    input_t* weight_ptr = weights + index * rank * hidden + rank_id * hidden;
    uint32_t offset = local_id * vec_size;

    accscalar_t local_result = 0;
    while (offset < hidden) {
      input_t* input_base = input_ptr + offset;
      input_t* weight_base = weight_ptr + offset;
      vec_t input_vec;
      vec_t weight_vec;
      input_vec = *reinterpret_cast<vec_t*>(input_base);
      weight_vec = *reinterpret_cast<vec_t*>(weight_base);

#pragma unroll(vec_size)
      for (uint32_t i = 0; i < vec_size; i++) {
        if (offset + i >= hidden) {
          break;
        }
        local_result += static_cast<accscalar_t>(input_vec[i]) *
            static_cast<accscalar_t>(weight_vec[i]);
      }
      offset += group_size * vec_size;
    }

    accscalar_t group_result = sycl::reduce_over_group(
        item.get_group(), local_result, sycl::plus<accscalar_t>());

    if (local_id == 0) {
      outputs[batch_id * rank + rank_id] +=
          static_cast<output_t>(group_result * scale);
    }
  }

 private:
  output_t* outputs;
  input_t* inputs;
  input_t* weights;
  int64_t* indices;
  const uint32_t hidden;
  const uint32_t rank;
  const float scale;
};

template <
    typename output_t,
    typename input_t,
    uint32_t vec_size,
    uint32_t subgroup_size,
    bool use_aligned_vector>
class BgmvExpandKernelFunctor {
 public:
  BgmvExpandKernelFunctor(
      output_t* outputs,
      input_t* inputs,
      output_t* weights,
      int64_t* indices,
      const uint32_t batch_size,
      const uint32_t rank,
      const uint32_t hidden,
      const uint32_t output_hidden,
      const uint32_t slice_offset,
      const bool add_to_output,
      dpcpp_local_acc_t<acc_type<output_t>> slm,
      const uint32_t workitem_per_hidden,
      const uint32_t hidden_per_subgroup,
      const uint32_t subgroup_num,
      const uint32_t sg_per_wg)
      : outputs(outputs),
        inputs(inputs),
        weights(weights),
        indices(indices),
        batch_size(batch_size),
        rank(rank),
        hidden(hidden),
        output_hidden(output_hidden),
        slice_offset(slice_offset),
        add_to_output(add_to_output),
        slm(slm),
        workitem_per_hidden(workitem_per_hidden),
        hidden_per_subgroup(hidden_per_subgroup),
        subgroup_num(subgroup_num),
        sg_per_wg(sg_per_wg) {}

  using accscalar_t = acc_type<output_t>;
  using input_vec_t =
      at::native::Memory::aligned_vector_loop<input_t, vec_size>;
  using weight_vec_t =
      at::native::Memory::aligned_vector_loop<output_t, vec_size>;
  [[intel::reqd_sub_group_size(subgroup_size)]] void operator()(
      sycl::nd_item<1> item) const {
    sycl::group<1> g = item.get_group();
    sycl::sub_group sg = item.get_sub_group();
    uint32_t group_id = g.get_group_linear_id();
    uint32_t subgroup_id = sg.get_group_linear_id() + group_id * sg_per_wg;
    if (subgroup_id >= subgroup_num) {
      return;
    }

    uint32_t item_id = g.get_local_linear_id();
    uint32_t line_id = sg.get_local_linear_id();
    uint32_t hidden_id_in_subgroup = line_id / workitem_per_hidden;
    uint32_t vec_id = line_id % workitem_per_hidden;
    uint32_t hidden_linear_id =
        subgroup_id * hidden_per_subgroup + hidden_id_in_subgroup;
    uint32_t batch_id = hidden_linear_id / hidden;
    uint32_t hidden_id = hidden_linear_id % hidden;
    int64_t index = indices[batch_id];
    if (hidden_id_in_subgroup < hidden_per_subgroup &&
        hidden_linear_id < batch_size * hidden && index >= 0) {
      input_t* input_ptr = inputs + batch_id * rank;
      output_t* weight_ptr = weights + index * hidden * rank + hidden_id * rank;

      accscalar_t local_result = 0;
      input_vec_t input_vec;
      weight_vec_t weight_vec;
      uint32_t offset = vec_id * vec_size;
      while (offset < rank) {
        input_t* input_base = input_ptr + offset;
        output_t* weight_base = weight_ptr + offset;
        if constexpr (use_aligned_vector) {
          input_vec = *reinterpret_cast<input_vec_t*>(input_base);
          weight_vec = *reinterpret_cast<weight_vec_t*>(weight_base);
        } else {
#pragma unroll(vec_size)
          for (uint32_t i = 0; i < vec_size; i++) {
            input_vec[i] = input_base[i];
            weight_vec[i] = weight_base[i];
          }
        }
#pragma unroll(vec_size)
        for (uint32_t i = 0; i < vec_size; i++) {
          if (offset + i >= rank) {
            break;
          }
          local_result += static_cast<accscalar_t>(input_vec[i]) *
              static_cast<accscalar_t>(weight_vec[i]);
        }
        offset += workitem_per_hidden * vec_size;
      }
      slm[item_id] = local_result;
    }

    sycl::group_barrier(sg);

    if (vec_id == 0 && hidden_id_in_subgroup < hidden_per_subgroup &&
        hidden_linear_id < batch_size * hidden && index >= 0) {
      accscalar_t result = 0;
      for (uint32_t i = 0; i < workitem_per_hidden; i++) {
        result += slm[item_id + i];
      }
      if (add_to_output) {
        outputs[batch_id * output_hidden + slice_offset + hidden_id] +=
            static_cast<output_t>(result);
      } else {
        outputs[batch_id * output_hidden + slice_offset + hidden_id] =
            static_cast<output_t>(result);
      }
    }
  }

 private:
  output_t* outputs;
  input_t* inputs;
  output_t* weights;
  int64_t* indices;
  const uint32_t batch_size;
  const uint32_t rank;
  const uint32_t hidden;
  const uint32_t output_hidden;
  const uint32_t slice_offset;
  const bool add_to_output;
  dpcpp_local_acc_t<accscalar_t> slm;
  const uint32_t workitem_per_hidden;
  const uint32_t hidden_per_subgroup;
  const uint32_t subgroup_num;
  const uint32_t sg_per_wg;
};

template <typename output_t, typename input_t>
void launch_bgmv_shrink(
    output_t* outputs,
    input_t* inputs,
    input_t* weights,
    int64_t* indices,
    const uint32_t batch_size,
    const uint32_t hidden,
    const uint32_t rank,
    const float scale) {
  uint32_t vec_bytes = 16;
  while ((hidden * sizeof(input_t)) % vec_bytes != 0 ||
         reinterpret_cast<uintptr_t>(inputs) % vec_bytes != 0 ||
         reinterpret_cast<uintptr_t>(weights) % vec_bytes != 0) {
    if (vec_bytes <= sizeof(input_t)) {
      vec_bytes = sizeof(input_t);
      break;
    }
    vec_bytes /= 2;
  }

  uint32_t vec_size = vec_bytes / sizeof(input_t);
  bool use_aligned_vector = (hidden % vec_size == 0);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const uint32_t max_workgroup_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t workgroup_size = 128;
  while (workgroup_size * vec_size < hidden &&
         workgroup_size * 2 <= max_workgroup_size) {
    workgroup_size *= 2;
  }

  uint32_t workgroup_num = batch_size * rank;
  sycl::range<1> local_range{workgroup_size};
  sycl::range<1> global_range{workgroup_num * workgroup_size};

#define submit_kernel_functor(vec_size)                                 \
  {                                                                     \
    BgmvShrinkKernelFunctor<output_t, input_t, vec_size> kfn(           \
        outputs, inputs, weights, indices, hidden, rank, scale);        \
    cgh.parallel_for<decltype(kfn)>(                                    \
        sycl::nd_range<1>(                                              \
            sycl::range<1>(global_range), sycl::range<1>(local_range)), \
        kfn);                                                           \
  }

  auto cgf = DPCPP_Q_CGF(cgh) {
    switch (vec_size) {
      case 8:
        submit_kernel_functor(8);
        break;
      case 4:
        submit_kernel_functor(4);
        break;
      case 2:
        submit_kernel_functor(2);
        break;
      case 1:
        submit_kernel_functor(1);
        break;
      default:
        TORCH_CHECK(false, "Unsupported vector size: ", vec_size);
    }
  };
#undef submit_kernel_functor
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename output_t, typename input_t>
void launch_bgmv_expand_with_slice(
    output_t* outputs,
    input_t* inputs,
    output_t* weights,
    int64_t* indices,
    const uint32_t batch_size,
    const uint32_t rank,
    const uint32_t hidden,
    const uint32_t output_hidden,
    const uint32_t slice_offset,
    const bool add_to_output) {
  constexpr uint32_t vec_size = 16 / sizeof(input_t);
  constexpr uint32_t subgroup_size = 32;
  bool use_aligned_vector =
      (rank % vec_size == 0 && reinterpret_cast<uintptr_t>(inputs) % 16 == 0 &&
       reinterpret_cast<uintptr_t>(weights) % 16 == 0);

  // Use several workitems to write one element to output with [batch_size,
  // hidden]. Use at most one subgroup to process one element.
  const uint32_t workitem_per_hidden =
      std::min((rank + vec_size - 1) / vec_size, subgroup_size);

  const uint32_t hidden_per_subgroup = subgroup_size / workitem_per_hidden;
  const uint32_t subgroup_num =
      (batch_size * hidden + hidden_per_subgroup - 1) / hidden_per_subgroup;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const uint32_t max_workgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const uint32_t workgroup_size =
      std::min(max_workgroup_size, subgroup_num * subgroup_size);
  const uint32_t sg_per_wg = workgroup_size / subgroup_size;
  const uint32_t workgroup_num = (subgroup_num + sg_per_wg - 1) / sg_per_wg;

  sycl::range<1> local_range{workgroup_size};
  sycl::range<1> global_range{workgroup_num * workgroup_size};
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<acc_type<output_t>> slm(sycl::range(workgroup_size), cgh);
    if (use_aligned_vector) {
      BgmvExpandKernelFunctor<output_t, input_t, vec_size, subgroup_size, true>
          kfn(outputs,
              inputs,
              weights,
              indices,
              batch_size,
              rank,
              hidden,
              output_hidden,
              slice_offset,
              add_to_output,
              slm,
              workitem_per_hidden,
              hidden_per_subgroup,
              subgroup_num,
              sg_per_wg);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>(
              sycl::range<1>(global_range), sycl::range<1>(local_range)),
          kfn);
    } else {
      BgmvExpandKernelFunctor<output_t, input_t, vec_size, subgroup_size, false>
          kfn(outputs,
              inputs,
              weights,
              indices,
              batch_size,
              rank,
              hidden,
              output_hidden,
              slice_offset,
              add_to_output,
              slm,
              workitem_per_hidden,
              hidden_per_subgroup,
              subgroup_num,
              sg_per_wg);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>(
              sycl::range<1>(global_range), sycl::range<1>(local_range)),
          kfn);
    }
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

void bgmv_shrink(
    Tensor& outputs,
    const Tensor& inputs,
    const Tensor& weights,
    const Tensor& indices,
    const double scale) {
  uint32_t batch_size = inputs.size(0);
  uint32_t hidden = inputs.size(1);
  uint32_t rank = outputs.size(1);
  float scale_ = static_cast<float>(scale);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      outputs.scalar_type(),
      "bgmv_shrink",
      [&]() {
        switch (inputs.scalar_type()) {
          case at::ScalarType::Half:
            launch_bgmv_shrink<scalar_t, at::Half>(
                outputs.data_ptr<scalar_t>(),
                inputs.data_ptr<at::Half>(),
                weights.data_ptr<at::Half>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                hidden,
                rank,
                scale_);
            return;
          case at::ScalarType::BFloat16:
            launch_bgmv_shrink<scalar_t, at::BFloat16>(
                outputs.data_ptr<scalar_t>(),
                inputs.data_ptr<at::BFloat16>(),
                weights.data_ptr<at::BFloat16>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                hidden,
                rank,
                scale_);
            return;
          default:
            TORCH_CHECK(
                false, "Unsupported input type: ", inputs.scalar_type());
        }
      });
}

void bgmv_expand_with_slice(
    Tensor& outputs,
    const Tensor& inputs,
    const Tensor& weights,
    const Tensor& indices,
    const int64_t slice_offset,
    const bool add_to_output) {
  uint32_t batch_size = inputs.size(0);
  uint32_t rank = inputs.size(1);
  uint32_t hidden = weights.size(1);
  uint32_t output_hidden = outputs.size(1);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      inputs.scalar_type(),
      "bgmv_expand_with_slice",
      [&]() {
        switch (outputs.scalar_type()) {
          case at::ScalarType::Half:
            launch_bgmv_expand_with_slice<at::Half, scalar_t>(
                outputs.data_ptr<at::Half>(),
                inputs.data_ptr<scalar_t>(),
                weights.data_ptr<at::Half>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                rank,
                hidden,
                output_hidden,
                slice_offset,
                add_to_output);
            return;
          case at::ScalarType::BFloat16:
            launch_bgmv_expand_with_slice<at::BFloat16, scalar_t>(
                outputs.data_ptr<at::BFloat16>(),
                inputs.data_ptr<scalar_t>(),
                weights.data_ptr<at::BFloat16>(),
                indices.data_ptr<int64_t>(),
                batch_size,
                rank,
                hidden,
                output_hidden,
                slice_offset,
                add_to_output);
            return;
          default:
            TORCH_CHECK(
                false, "Unsupported output type: ", outputs.scalar_type());
        }
      });
}

void sgmv_shrink(
    Tensor& outputs,
    const Tensor& inputs,
    const Tensor& weights,
    const Tensor& seq_start_locs,
    const Tensor& seq_lens,
    const Tensor& lora_indices,
    const int64_t batches,
    const int64_t max_seq_len,
    const double scale) {
  uint32_t gemm_k = inputs.size(1);
  uint32_t gemm_n = weights.size(1);
  float scale_ = static_cast<float>(scale);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

#if defined(USE_XETLA)
  TORCH_CHECK(
      dpcppGetDeviceHasXMX(),
      "sgmv shrink kernel requires XMX, but the current platform has no XMX ...");

  TORCH_CHECK(
      dpcppGetDeviceHas2DBlock(),
      "sgmv shrink kernel requires 2DBlock, but the current platform has no 2DBlock ...");

  static gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_device_gpu_arch();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      outputs.scalar_type(),
      "sgmv_shrink",
      [&]() {
        if constexpr (std::is_same_v<scalar_t, double>) {
          TORCH_CHECK(
              false,
              "Unsupported output type: ",
              outputs.scalar_type(),
              ", only support float, half and bfloat16");
        }

        switch (inputs.scalar_type()) {
          case at::ScalarType::Half: {
            torch_ipex::xpu::xetla::cgf_t cfg =
                torch_ipex::xpu::xetla::sgmv_shrink<scalar_t, at::Half>(
                    arch_tag,
                    outputs.data_ptr<scalar_t>(),
                    inputs.data_ptr<at::Half>(),
                    weights.data_ptr<at::Half>(),
                    seq_start_locs.data_ptr<int64_t>(),
                    seq_lens.data_ptr<int64_t>(),
                    lora_indices.data_ptr<int64_t>(),
                    batches,
                    max_seq_len,
                    gemm_k,
                    gemm_n,
                    scale_);
            DPCPP_Q_SUBMIT(dpcpp_queue, cfg);
            return;
          }
          case at::ScalarType::BFloat16: {
            torch_ipex::xpu::xetla::cgf_t cfg =
                torch_ipex::xpu::xetla::sgmv_shrink<scalar_t, at::BFloat16>(
                    arch_tag,
                    outputs.data_ptr<scalar_t>(),
                    inputs.data_ptr<at::BFloat16>(),
                    weights.data_ptr<at::BFloat16>(),
                    seq_start_locs.data_ptr<int64_t>(),
                    seq_lens.data_ptr<int64_t>(),
                    lora_indices.data_ptr<int64_t>(),
                    batches,
                    max_seq_len,
                    gemm_k,
                    gemm_n,
                    scale_);
            DPCPP_Q_SUBMIT(dpcpp_queue, cfg);
            return;
          }
          default:
            TORCH_CHECK(
                false, "Unsupported input type: ", inputs.scalar_type());
        }
      });
#else
  AT_ERROR("sgmv_shrink: xetla library not found in compilation");
#endif
}

void sgmv_expand_with_slice(
    Tensor& outputs,
    const Tensor& inputs,
    const Tensor& weights,
    const Tensor& seq_start_locs,
    const Tensor& seq_lens,
    const Tensor& lora_indices,
    const int64_t batches,
    const int64_t max_seq_len,
    const int64_t slice_offset,
    const bool add_to_output) {
  uint32_t gemm_k = inputs.size(1);
  uint32_t gemm_n = weights.size(1);
  uint32_t output_hidden = outputs.size(1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

#if defined(USE_XETLA)
  TORCH_CHECK(
      dpcppGetDeviceHasXMX(),
      "sgmv expand kernel requires XMX, but the current platform has no XMX ...");

  TORCH_CHECK(
      dpcppGetDeviceHas2DBlock(),
      "sgmv expand kernel requires 2DBlock, but the current platform has no 2DBlock ...");

  static gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_device_gpu_arch();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      inputs.scalar_type(),
      "sgmv_expand_with_slice",
      [&]() {
        if constexpr (std::is_same_v<scalar_t, double>) {
          TORCH_CHECK(
              false,
              "Unsupported input type: ",
              inputs.scalar_type(),
              ", only support float, half and bfloat16");
        }

        switch (outputs.scalar_type()) {
          case at::ScalarType::Half: {
            if constexpr (
                !std::is_same_v<scalar_t, float> &&
                !std::is_same_v<scalar_t, at::Half>) {
              TORCH_CHECK(
                  false,
                  "Unsupported input type: ",
                  inputs.scalar_type(),
                  ", XeTLA only support float, half as input type when output type is half");
            } else {
              torch_ipex::xpu::xetla::cgf_t cfg = torch_ipex::xpu::xetla::
                  sgmv_expand_with_slice<at::Half, scalar_t>(
                      arch_tag,
                      outputs.data_ptr<at::Half>(),
                      inputs.data_ptr<scalar_t>(),
                      weights.data_ptr<at::Half>(),
                      seq_start_locs.data_ptr<int64_t>(),
                      seq_lens.data_ptr<int64_t>(),
                      lora_indices.data_ptr<int64_t>(),
                      batches,
                      max_seq_len,
                      gemm_k,
                      gemm_n,
                      slice_offset,
                      output_hidden,
                      add_to_output);
              DPCPP_Q_SUBMIT(dpcpp_queue, cfg);
            }
            return;
          }
          case at::ScalarType::BFloat16: {
            if constexpr (
                !std::is_same_v<scalar_t, float> &&
                !std::is_same_v<scalar_t, at::BFloat16>) {
              TORCH_CHECK(
                  false,
                  "Unsupported input type: ",
                  inputs.scalar_type(),
                  ", XeTLA only support float, bfloat16 as input type when output type is bfloat16");
            } else {
              torch_ipex::xpu::xetla::cgf_t cfg = torch_ipex::xpu::xetla::
                  sgmv_expand_with_slice<at::BFloat16, scalar_t>(
                      arch_tag,
                      outputs.data_ptr<at::BFloat16>(),
                      inputs.data_ptr<scalar_t>(),
                      weights.data_ptr<at::BFloat16>(),
                      seq_start_locs.data_ptr<int64_t>(),
                      seq_lens.data_ptr<int64_t>(),
                      lora_indices.data_ptr<int64_t>(),
                      batches,
                      max_seq_len,
                      gemm_k,
                      gemm_n,
                      slice_offset,
                      output_hidden,
                      add_to_output);
              DPCPP_Q_SUBMIT(dpcpp_queue, cfg);
            }
            return;
          }
          default:
            TORCH_CHECK(
                false, "Unsupported output type: ", outputs.scalar_type());
        }
      });
#else
  AT_ERROR("sgmv_expand: xetla library not found in compilation");
#endif
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "bgmv_shrink", at::AtenIpexTypeXPU::bgmv_shrink, c10::DispatchKey::XPU)

  IPEX_OP_REGISTER_DISPATCH(
      "bgmv_expand_with_slice",
      at::AtenIpexTypeXPU::bgmv_expand_with_slice,
      c10::DispatchKey::XPU)

  IPEX_OP_REGISTER_DISPATCH(
      "sgmv_shrink", at::AtenIpexTypeXPU::sgmv_shrink, c10::DispatchKey::XPU)

  IPEX_OP_REGISTER_DISPATCH(
      "sgmv_expand_with_slice",
      at::AtenIpexTypeXPU::sgmv_expand_with_slice,
      c10::DispatchKey::XPU)
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at
