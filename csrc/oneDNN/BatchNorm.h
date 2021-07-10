#pragma once

#include <ATen/ATen.h>

#include <runtime/Utils.h>
#include <oneDNN/Runtime.h>
#include <oneDNN/LRUCache.h>
#include <quantized/Quantizer.h>
#include <tensor/Context.h>
#include <operators/MemoryHelpers.h>
#include <operators/comm/ATDispatch.h>
#include "Utils.h"
#include "Reorder.h"

#include <oneapi/dnnl/dnnl.hpp>


using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static inline memory::format_tag bn_src_format(const at::Tensor& t) {
  auto is_channels_last = t.is_contiguous(at::MemoryFormat::ChannelsLast);
  auto ndim = t.ndimension();
  if (ndim == 2) {
    return memory::format_tag::nc;
  } else if (ndim == 3) {
    return is_channels_last ?
           memory::format_tag::nwc :
           memory::format_tag::ncw;
  } else if (ndim == 4) {
    return is_channels_last ?
           memory::format_tag::nhwc :
           memory::format_tag::nchw;
  } else if (ndim == 5) {
    return is_channels_last ?
           memory::format_tag::ndhwc :
           memory::format_tag::ncdhw;
  } else {
    std::stringstream ss;
    ss << "SYCL batch_norm backend got shape=" << t.sizes()
       << ", expected input with rank 2 [n, c], rank 3 [n, c, l], rank 4 [n, "
          "c, h, w] or rank 5 [n, c, d, h, w] shape ";
    AT_ERROR(ss.str());
  }
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_normalization(
    const at::Tensor& src,
    const at::Tensor& wgh_option,
    const at::Tensor& bia_option,
    const at::Tensor& running_mean_option,
    const at::Tensor& running_var_option,
    bool training,
    double momentum,
    double epsilon) {
  auto engine = GpuEngineManager::Instance().get_engine({at::kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  at::Tensor wgh = wgh_option;
  at::Tensor bia = bia_option;
  at::Tensor running_mean = running_mean_option;
  at::Tensor running_var = running_var_option;

  auto prop = training ?
              prop_kind::forward_training :
              prop_kind::forward_inference;
  auto flag = normalization_flags::use_scale |
              normalization_flags::use_shift;

  auto feature_num = src.size(1);
  auto feature_size = src.numel() / feature_num;

  if (!wgh.defined())
    wgh = at::ones(feature_num, wgh.options());

  if (!bia.defined())
    bia = at::zeros(feature_num, wgh.options());

  if (!training && running_mean.defined() && running_var.defined())
    flag |= normalization_flags::use_global_stats;

  auto src_tz = get_onednn_dims(src);
  auto src_dt = get_onednn_dtype(src);
  auto src_fmt = bn_src_format(src);

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_md = src_ctx.is_plain() ?
                memory::desc({src_tz}, src_dt, src_fmt) :
                src_ctx.meta();
  auto scl_md = memory::desc(
      {feature_num}, memory::data_type::f32, memory::format_tag::a);
  auto sft_md = memory::desc(
      {feature_num}, memory::data_type::f32, memory::format_tag::a);

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, src_md, epsilon, flag);
#endif

  auto bn_fwd_desc =
      batch_normalization_forward::desc(prop, src_md, epsilon, flag);

#ifdef USE_SCRATCHPAD_MODE
  primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto bn_fwd_pd =
      batch_normalization_forward::primitive_desc(bn_fwd_desc, attr, engine);
#else
  auto bn_fwd_pd =
      batch_normalization_forward::primitive_desc(bn_fwd_desc, engine);
#endif

  at::Tensor dst;
  auto dst_md = bn_fwd_pd.dst_desc();
  if (!src_ctx.is_plain()) {
    dst = src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
          empty_opaque_tensor(dst_md, src.options(),
                              at::MemoryFormat::ChannelsLast) :
          empty_opaque_tensor(dst_md, src.options(), c10::nullopt);
  } else {
    dst = src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
          at::empty_like(src, src.options(), at::MemoryFormat::ChannelsLast) :
          at::empty_like(src);
  }

  auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  auto dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

#ifdef USE_PRIMITIVE_CACHE
  auto bn_fwd = fetch_or_create_m<batch_normalization_forward>(key, bn_fwd_pd);
#else
  auto bn_fwd = batch_normalization_forward(bn_fwd_pd);
#endif

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m},
      {DNNL_ARG_DST, dst_m},
  };

  if (wgh.scalar_type() == ScalarType::Half ||
      wgh.scalar_type() == ScalarType::BFloat16) {
    wgh = wgh.to(at::kFloat);
  }

  if (bia.scalar_type() == ScalarType::Half ||
      bia.scalar_type() == ScalarType::BFloat16) {
    bia = bia.to(at::kFloat);
  }

  auto scl_m = dpcpp_onednn_memory(scl_md, engine, wgh.data_ptr());
  auto sft_m = dpcpp_onednn_memory(sft_md, engine, bia.data_ptr());
  args.insert({DNNL_ARG_SCALE, scl_m});
  args.insert({DNNL_ARG_SHIFT, sft_m});

  at::Tensor save_mean = at::empty(feature_num, wgh.options().dtype(at::kFloat));
  at::Tensor save_var = at::empty(feature_num, wgh.options().dtype(at::kFloat));

  void* mean_data = nullptr;
  void* var_data = nullptr;
  if ((bool)(flag & normalization_flags::use_global_stats)) {
    if (running_mean.scalar_type() == ScalarType::Half ||
        running_mean.scalar_type() == ScalarType::BFloat16)
      running_mean = running_mean.to(ScalarType::Float);

    if (running_var.scalar_type() == ScalarType::Half ||
        running_var.scalar_type() == ScalarType::BFloat16)
      running_var = running_var.to(ScalarType::Float);

    mean_data = running_mean.data_ptr();
    var_data = running_var.data_ptr();
  } else {
    mean_data = save_mean.data_ptr();
    var_data = save_var.data_ptr();
  }

  auto mean_m = dpcpp_onednn_memory(bn_fwd_pd.mean_desc(), engine, mean_data);
  auto var_m = dpcpp_onednn_memory(bn_fwd_pd.variance_desc(), engine, var_data);

  args.insert({DNNL_ARG_MEAN, mean_m});
  args.insert({DNNL_ARG_VARIANCE, var_m});

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = bn_fwd_pd.scratchpad_desc().get_size() / src.dtype().itemsize();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty({scratchpad_size}, src.options(), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(bn_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
#endif

  DPCPP_ONEDNN_EXEC(bn_fwd, strm, args);

  if (training && running_mean.defined() && running_var.defined()) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        running_mean.scalar_type(), "mScale1", [&]() {
            dpcppMemoryScale1(
                running_mean.data_ptr<scalar_t>(),
                save_mean.data_ptr<float>(),
                feature_num,
                momentum);
        }
    );
    size_t orig_size = feature_size;
    size_t adjust_size = orig_size - 1;
    float adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        running_var.scalar_type(), "mScale2", [&]() {
            dpcppMemoryScale2(
                running_var.data_ptr<scalar_t>(),
                save_var.data_ptr<float>(),
                feature_num,
                adjust_factor,
                momentum);
        }
    );
  }

  return {dst, save_mean, save_var};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_normalization_backward(
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    bool training,
    double epsilon,
    std::array<bool, 3> diff_src_mask) {
  auto engine = GpuEngineManager::Instance().get_engine({at::kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  at::Tensor diff_src, diff_wgh, diff_bia;
  if (diff_src_mask[0])
    diff_src = at::empty_like(src);
  if (diff_src_mask[1])
    diff_wgh = at::empty_like(wgh);
  if (diff_src_mask[2])
    diff_bia = at::empty_like(wgh);

  auto flags = normalization_flags::use_scale_shift;

  if (!(diff_src_mask[1] && diff_src_mask[2]))
    flags &= ~normalization_flags::use_scale_shift;

  auto src_tz = get_onednn_dims(src);
  auto src_dt = get_onednn_dtype(src);
  auto src_fmt = bn_src_format(src);

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_md = src_ctx.is_plain() ?
                memory::desc({src_tz}, src_dt, src_fmt) :
                src_ctx.meta();
  auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  auto diff_dst_ctx = DPCPPTensorContext::get_tensor_ctx(diff_dst);
  auto diff_dst_md = diff_dst_ctx.is_plain() ?
                     memory::desc({src_tz}, src_dt, src_fmt) :
                     diff_dst_ctx.meta();
  auto diff_dst_usr_m = dpcpp_onednn_memory(	
      diff_dst_md, engine, diff_dst.data_ptr());

  batch_normalization_forward::desc bn_fwd_desc(
      prop_kind::forward_training, src_md, epsilon, flags);
  auto bn_fwd_pd = batch_normalization_forward::primitive_desc(
      bn_fwd_desc, engine);

  at::Tensor diff_dst_;
  auto diff_dst_m = diff_dst_usr_m;
  if (diff_dst_ctx.is_plain() && (!src_ctx.is_plain())) {
    auto expected_dst_md = bn_fwd_pd.dst_desc();
    diff_dst_ = empty_opaque_tensor(expected_dst_md, src.options(), c10::nullopt);
    diff_dst_m = dpcpp_onednn_memory(expected_dst_md, engine, diff_dst_.data_ptr());
    diff_dst_md = expected_dst_md;
    xpu::oneDNN::reorder(diff_dst, diff_dst_);
  }

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, diff_dst_md, src_md, epsilon, flags);
#endif

  prop_kind p_kind;
  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    p_kind = prop_kind::backward;
  } else {
    p_kind = prop_kind::backward_data;
  }

  auto bwd_desc = batch_normalization_backward::desc(
      p_kind, diff_dst_md, src_md, epsilon, flags);

  auto bn_bwd_pd = batch_normalization_backward::primitive_desc(
      bwd_desc, engine, bn_fwd_pd);

  memory mean_m, var_m;
  if (training) {
    mean_m = dpcpp_onednn_memory(
        bn_fwd_pd.mean_desc(), engine, save_mean.data_ptr());
    var_m = dpcpp_onednn_memory(
        bn_fwd_pd.variance_desc(), engine, save_var.data_ptr());
  } else {
    mean_m = dpcpp_onednn_memory(
        bn_fwd_pd.mean_desc(), engine, running_mean.data_ptr());
    var_m = dpcpp_onednn_memory(
        bn_fwd_pd.variance_desc(), engine, running_var.data_ptr());
  }

  auto diff_src_md = memory::desc({src_tz, src_dt, src_fmt});
  auto expected_diff_src_md = bn_bwd_pd.diff_src_desc();
  if (diff_src_md != expected_diff_src_md) {
    diff_src = empty_opaque_tensor(
        expected_diff_src_md, diff_dst.options(), c10::nullopt);
  }
  auto diff_src_m = dpcpp_onednn_memory(
      expected_diff_src_md, engine, diff_src.data_ptr());

#ifdef USE_PRIMITIVE_CACHE
  auto bn_bwd = fetch_or_create_m<
      dnnl::batch_normalization_backward>(key, bn_bwd_pd);
#else
  auto bn_bwd = dnnl::batch_normalization_backward(bn_bwd_pd);
#endif

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC,        src_m       },
      {DNNL_ARG_DIFF_DST,   diff_dst_m  },
      {DNNL_ARG_MEAN,       mean_m      },
      {DNNL_ARG_VARIANCE,   var_m       },
      {DNNL_ARG_DIFF_SRC,   diff_src_m  },
  };

  size_t feature_num = src.size(1);

  at::Tensor diff_wgh_bia;
  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    auto wgh_bia = at::empty(2 * feature_num, wgh.options().dtype(at::kFloat));
    auto wgh_bia_m = dpcpp_onednn_memory(
        bn_fwd_pd.weights_desc(), engine, wgh_bia.data_ptr());
    if (wgh.scalar_type() == ScalarType::BFloat16) {
      dtype_convert_by_scalar(
          wgh_bia.data_ptr<float>(),
          wgh.data_ptr<at::BFloat16>(),
          feature_num);
    } else {
      dtype_convert_by_scalar(
          wgh_bia.data_ptr<float>(),
          wgh.data_ptr<float>(),
          feature_num);
    }
    dpcppMemsetAsync(
      static_cast<uint8_t*>(wgh_bia.data_ptr()) + feature_num * sizeof(float),
      0,
      feature_num * sizeof(float));
    diff_wgh_bia = at::empty(2 * feature_num, wgh.options().dtype(at::kFloat));
    auto diff_wgh_bia_m = dpcpp_onednn_memory(
        bn_bwd_pd.diff_weights_desc(), engine, diff_wgh_bia.data_ptr());

    args.insert({DNNL_ARG_SCALE_SHIFT, wgh_bia_m});
    args.insert({DNNL_ARG_DIFF_SCALE_SHIFT, diff_wgh_bia_m});
  }

  DPCPP_ONEDNN_EXEC(bn_bwd, strm, args);

  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    if (wgh.scalar_type() == ScalarType::BFloat16) {
      dtype_convert_by_scalar(
          diff_wgh.data_ptr<at::BFloat16>(),
          diff_wgh_bia.data_ptr<float>(),
          feature_num);
      dtype_convert_by_scalar(
          diff_bia.data_ptr<at::BFloat16>(),
          diff_wgh_bia.data_ptr<float>() + feature_num,
          feature_num);
    } else {
      dtype_convert_by_scalar(
          diff_wgh.data_ptr<float>(),
          diff_wgh_bia.data_ptr<float>(),
          feature_num);
      dtype_convert_by_scalar(
          diff_bia.data_ptr<float>(),
          diff_wgh_bia.data_ptr<float>() + feature_num,
          feature_num);
    }
  }

  return {diff_src, diff_wgh, diff_bia};
}

}}