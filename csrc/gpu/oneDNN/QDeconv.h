#include <oneapi/dnnl/dnnl_debug.h>
#include "Deconv.h"

namespace xpu {
namespace oneDNN {
namespace {

static std::tuple<
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc>
qdeconv_get_blocked_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int groups) {
  // create memory desc from the src/wgh/dst tensors
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  auto ndim = src.ndimension();
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto fmt_src = deconv_src_fmt(ndim);
  if (src_ctx.is_plain()) {
    auto src_tz = src.sizes().vec();
    auto src_data_t = is_opaque_u8(src) ? memory::data_type::s8
                                        : get_onednn_dtype_include_double(src);
    src_usr_md = memory::desc(src_tz, src_data_t, fmt_src);
  } else {
    src_usr_md = src_ctx.meta();
  }

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  if (dst_ctx.is_plain()) {
    auto dst_tz = dst.sizes().vec();
    auto dst_data_t = get_onednn_dtype_include_double(dst);
    dst_usr_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  } else {
    dst_usr_md = dst_ctx.meta();
  }

  auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
  if (wgh_ctx.is_plain()) {
    auto ic = src.size(1);
    auto oc = dst.size(1);
    memory::dims wgh_tz =
        deconv_compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
    memory::data_type wei_data_t = memory::data_type::s8;
    auto fmt_wgh = deconv_wgh_fmt(wgh, ndim, wgh_tz, groups != 1);
    // auto fmt_wgh = deconv_wgh_fmt(ndim, groups != 1);
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  } else {
    wgh_usr_md = wgh_ctx.meta();
  }

  // create memory desc for deconv primitive, block fmt use fmt_any
  memory::desc src_md, wgh_md, dst_md;
  auto fmt_any = memory::format_tag::any;
  src_md = src.size(1) == 3
      ? src_usr_md
      : memory::desc(src_usr_md.dims(), src_usr_md.data_type(), fmt_any);
  wgh_md = memory::desc(wgh_usr_md.dims(), memory::data_type::s8, fmt_any);
  dst_md = memory::desc(dst_usr_md.dims(), dst_usr_md.data_type(), fmt_any);
  return {src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md};
}

static std::tuple<
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc>
qdeconv_get_plain_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups,
    bool is_channels_last_suggested) {
  auto ndim = src.ndimension();
  auto src_data_t = is_opaque_u8(src) ? memory::data_type::s8
                                      : get_onednn_dtype_include_double(src);

  // TODO: support channels last
  auto fmt_src = deconv_src_fmt(ndim, is_channels_last_suggested);
  auto src_usr_md = memory::desc(src.sizes().vec(), src_data_t, fmt_src);

  auto dst_data_t = get_onednn_dtype_include_double(dst);
  auto dst_usr_md = memory::desc(dst.sizes().vec(), dst_data_t, fmt_src);

  memory::desc wgh_usr_md, wgh_md;
  auto ic = src.size(1);
  auto oc = dst.size(1);
  memory::dims wgh_tz =
      deconv_compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  auto wei_data_t = memory::data_type::s8;
  // auto fmt_wgh = deconv_wgh_fmt(ndim, groups != 1);
  auto fmt_wgh = deconv_wgh_fmt(
      wgh, ndim, wgh_tz, groups != 1, is_channels_last_suggested);
  if (is_channels_last_suggested) {
    // TODO: In future, fmt_wgh would be nhwc instead of format_any
    auto fmt_any = memory::format_tag::any;
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
    wgh_md = memory::desc(wgh_tz, wei_data_t, fmt_any);
  } else {
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
    wgh_md = wgh_usr_md;
  }

  return {src_usr_md, wgh_usr_md, dst_usr_md, src_usr_md, wgh_md, dst_usr_md};
}

static inline void get_deconv_scale(
    const Tensor& src,
    const Tensor& wgh,
    const Tensor& dst,
    const memory::desc& src_usr_md,
    int oc,
    std::vector<float>& wgh_scales,
    std::vector<float>& conv_scale,
    int& mask_conv) {
  if (src.is_quantized()) {
    auto wgh_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(wgh);
    if (wgh.qscheme() == kPerTensorAffine) {
      wgh_scales.push_back(static_cast<float>(wgh.q_scale()));
    } else {
      for (int i = 0; i < oc; i++) {
        // GPU to CPU, may be has perf issue
        wgh_scales.push_back(wgh.q_per_channel_scales()[i].item<float>());
      }
    }
  }

  float src_scale;
  // auto src_data_t = src_usr_md.data_type();
  auto src_data_t = is_opaque_u8(src) ? memory::data_type::s8
                                      : get_onednn_dtype_include_double(src);
  src_scale = (src_data_t == memory::data_type::u8 && src.q_zero_point() == 128)
      ? src.q_scale() / 2
      : src.q_scale();
  conv_scale.clear();

  // See Note: [Convolution requantization]
  float dst_sc = dst.q_scale();
  for (int i = 0; i < wgh_scales.size(); i++) {
    conv_scale.push_back(1.0 / (dst_sc / (src_scale * wgh_scales[i])));
    // conv_scale.push_back(1.0);
  }
  mask_conv = wgh_scales.size() > 1 ? 1 << 1 : 0;
}

static memory qdeconv_get_expected_src_memory(
    const at::Tensor& src,
    at::Tensor& src_blocked,
    memory::desc& src_usr_md,
    memory::desc& expected_src_md,
    dnnl::engine& engine) {
  memory src_m;
  if (src_usr_md != expected_src_md) {
    // avoid reorder in case of, [n][C][1][1][16c] <==> [n][c][1][1]
    // this follows qconv
    // TODO: check this
    if (src.sizes().size() == 4 && src.size(2) == 1 && src.size(3) == 1) {
      src_m = dpcpp_onednn_memory(expected_src_md, engine, src.data_ptr());
    } else {
      // See Note [empty_opaque_tensor for qtensor creation]
      src_blocked =
          empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
      src_m =
          dpcpp_onednn_memory(expected_src_md, engine, src_blocked.data_ptr());
      xpu::oneDNN::reorder(src, src_blocked);
    }
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
  }
  return src_m;
}

static memory qdeconv_get_expected_wgh_memory(
    const at::Tensor& wgh,
    at::Tensor& wgh_blocked,
    memory::desc& wgh_usr_md,
    memory::desc& expected_wgh_md,
    std::vector<float>& wgh_scales,
    dnnl::engine& engine,
    bool weight_cache_optimization) {
  memory wgh_m;
  if (wgh_usr_md != expected_wgh_md) {
    wgh_blocked =
        empty_opaque_tensor(expected_wgh_md, wgh.options(), c10::nullopt);
    wgh_m =
        dpcpp_onednn_memory(expected_wgh_md, engine, wgh_blocked.data_ptr());

    auto reshaped_wgh = wgh;
    // reshape for group convolution weight
    if (wgh_blocked.ndimension() == 5 && wgh.ndimension() == 4) {
      reshaped_wgh = share_storage_and_set_strided_as(
          wgh,
          wgh_blocked.sizes(),
          /*compatible with different strides of weight (including contiguous,
             channels_last and non-contiguous) */
          compatible_groups_conv_strides(wgh, wgh_blocked.sizes().vec()),
          c10::nullopt);
    }
    xpu::oneDNN::reorder(reshaped_wgh, wgh_blocked);

    if (weight_cache_optimization) {
      auto wgh_opt_ctx = DPCPPTensorContext::release_tensor_ctx(wgh_blocked);
      wgh_opt_ctx.set_aten_meta(
          {reshaped_wgh.sizes().vec(), reshaped_wgh.strides().vec()});
      DPCPPTensorContext::set_tensor_ctx(wgh, std::move(wgh_opt_ctx));
    }
  } else {
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
  }
  return wgh_m;
}

static memory qdeconv_get_block_dst_memory(
    at::Tensor& dst,
    at::Tensor& dst_blocked,
    memory::desc& dst_usr_md,
    memory::desc& expected_dst_md,
    dnnl::engine& engine) {
  memory dst_m;
  if (dst_usr_md != expected_dst_md) {
    auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
        (get_onednn_dtype_include_double(dst) == memory::data_type::u8 &&
         dst.q_zero_point() == 128)
            ? dst.q_scale() / 2
            : dst.q_scale(),
        0,
        typeMetaToScalarType(dst.options().dtype()));
    dst_blocked =
        empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
    dst_m =
        dpcpp_onednn_memory(expected_dst_md, engine, dst_blocked.data_ptr());

    // TODO: qdeconv currently does not post ops
    // if (attr.with_sum())
    //   xpu::oneDNN::reorder(dst, dst_blocked);
  } else {
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  }
  return dst_m;
}

static Tensor quantized_deconvolution(
    Tensor& dst,
    const Tensor& src_r,
    const Tensor& wgh_r,
    const Tensor& bia,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups) {
  auto src = src_r;
  auto wgh = wgh_r;

  bool with_bias = bia.defined();
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv = get_memory_layout_for_conv(src, wgh);

  bool is_onednn_layout_suggested =
      (memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked);

  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  if (is_onednn_layout_suggested) {
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        qdeconv_get_blocked_md(src, wgh, dst, groups);
  } else {
    bool is_channels_last_suggested =
        (memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast);
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        qdeconv_get_plain_md(src, wgh, dst, groups, is_channels_last_suggested);
  }

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  auto ndim = src.ndimension();
  auto dst_tz = deconv_dst_tz(
      src.sizes(), wgh.sizes(), padding, stride, dilation, dst_padding, groups);
  auto wgh_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(wgh);
  if (is_onednn_layout_suggested && wgh_ctx.is_plain()) {
    // Transpose wgh
    wgh = wgh.transpose(0, 1);
  }

  auto oc = dst_tz[1];
  memory::dims bia_tz = {oc};

  memory::data_type bia_dt =
      with_bias ? get_onednn_dtype(bia) : memory::data_type::undef;
  memory::format_tag bia_fmt = memory::format_tag::x;

  auto bia_md =
      with_bias ? memory::desc(bia_tz, bia_dt, bia_fmt) : memory::desc();

  auto deconv_fwd_desc = deconvolution_forward::desc(
      prop_kind::forward,
      algorithm::deconvolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding);

  // quant only
  primitive_attr pattr;
  std::vector<float> wgh_scales, conv_scale = {1};

  int conv_zero_point = 0, mask_conv;
  get_deconv_scale(
      src, wgh, dst, src_usr_md, oc, wgh_scales, conv_scale, mask_conv);
  pattr.set_output_scales(mask_conv, conv_scale);
  pattr.set_zero_points(DNNL_ARG_DST, mask_conv, {conv_zero_point});

  auto deconv_fwd_pd =
      deconvolution_forward::primitive_desc(deconv_fwd_desc, pattr, engine);

  auto weight_cache_optimization = [&]() {
    // TODO:: remove ChannelsLast option after oneDNN fix accuracy issue
    return (memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked ||
            memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast) &&
        !at::GradMode::is_enabled();
  }();

  memory src_m, wgh_m, dst_m;
  // Tensor src_, wgh_, bia_;
  Tensor src_blocked, wgh_blocked, dst_blocked = dst;

  if (is_onednn_layout_suggested) {
    auto expected_src_md = deconv_fwd_pd.src_desc();
    auto expected_wgh_md = deconv_fwd_pd.weights_desc();
    auto expected_dst_md = deconv_fwd_pd.dst_desc();
    src_m = qdeconv_get_expected_src_memory(
        src, src_blocked, src_usr_md, expected_src_md, engine);
    wgh_m = qdeconv_get_expected_wgh_memory(
        wgh,
        wgh_blocked,
        wgh_usr_md,
        expected_wgh_md,
        wgh_scales,
        engine,
        weight_cache_optimization);
    dst_m = qdeconv_get_block_dst_memory(
        dst, dst_blocked, dst_usr_md, expected_dst_md, engine);
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    if (memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast) {
      // TODO: Should remove after oneDNN fix channelslast accuracy issue
      auto expected_wgh_md = deconv_fwd_pd.weights_desc();
      wgh_m = qdeconv_get_expected_wgh_memory(
          wgh,
          wgh_blocked,
          wgh_usr_md,
          expected_wgh_md,
          wgh_scales,
          engine,
          weight_cache_optimization);
    }
  }

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m}, {DNNL_ARG_WEIGHTS, wgh_m}, {DNNL_ARG_DST, dst_m}};

  if (with_bias) {
    auto bia_m = dpcpp_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }

  auto deconv_fwd = deconvolution_forward(deconv_fwd_pd);

  DPCPP_ONEDNN_EXEC(deconv_fwd, strm, args);
  if (is_onednn_layout_suggested && dst_blocked.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_blocked);
    blk_ctx.set_aten_meta({dst.sizes().vec(), dst.strides().vec()});
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  return dst;
}

} // namespace
} // namespace oneDNN
} // namespace xpu
