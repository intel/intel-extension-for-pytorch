#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <core/MemoryFormat.h>

#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Attr.h"
#include "ConvUtils.h"
#include "Reorder.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace torch_ipex::xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace torch_ipex::xpu {
namespace oneDNN {

static std::tuple<memory::desc, memory::desc, memory::desc> conv_get_usr_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups,
    int memory_layout) {
  // create memory desc from the src/wgh/dst tensors
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  auto ndim = src.ndimension();
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto fmt_src =
      conv_src_fmt(ndim, memory_layout == MEMORY_LAYOUT_FOR_CONV::ChannelsLast);
  if (src_ctx.is_plain()) {
    auto src_tz = src.sizes().vec();
    auto src_data_t = get_onednn_dtype_include_double(src);
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
    auto wei_data_t = get_onednn_dtype_include_double(wgh);
    memory::dims wgh_tz =
        compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
    auto fmt_wgh = conv_wgh_fmt(
        ndim,
        groups != 1,
        memory_layout == MEMORY_LAYOUT_FOR_CONV::ChannelsLast);
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  } else {
    wgh_usr_md = wgh_ctx.meta();
  }

  return {src_usr_md, wgh_usr_md, dst_usr_md};
}

static std::tuple<memory::desc, memory::desc, memory::desc> conv_get_md(
    memory::desc src_usr_md,
    memory::desc wgh_usr_md,
    memory::desc dst_usr_md,
    int memory_layout) {
  // create memory desc for conv primitive and query the blocked format
  memory::desc src_md, wgh_md, dst_md;
  if (memory_layout == MEMORY_LAYOUT_FOR_CONV::Blocked) {
    auto fmt_any = memory::format_tag::any;
    src_md = src_usr_md.get_dims()[1] == 3
        ? src_usr_md
        : memory::desc(
              src_usr_md.get_dims(), src_usr_md.get_data_type(), fmt_any);
    wgh_md = memory::desc(
        wgh_usr_md.get_dims(), wgh_usr_md.get_data_type(), fmt_any);
    dst_md = memory::desc(
        dst_usr_md.get_dims(), dst_usr_md.get_data_type(), fmt_any);
  } else {
    src_md = src_usr_md;
    wgh_md = wgh_usr_md;
    dst_md = dst_usr_md;
  }

  return {src_md, wgh_md, dst_md};
}

static memory conv_get_expected_src_memory(
    const at::Tensor& src,
    at::Tensor& src_blocked,
    memory::desc& src_usr_md,
    memory::desc& expected_src_md,
    dnnl::engine& engine,
    bool need_reorder = true) {
  memory src_m;
  if (src_usr_md != expected_src_md) {
    src_blocked =
        empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
    src_m =
        dpcpp_onednn_memory(expected_src_md, engine, src_blocked.data_ptr());
    if (need_reorder)
      torch_ipex::xpu::oneDNN::reorder(src, src_blocked);
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    src_blocked = src;
  }
  return src_m;
}

static memory conv_get_expected_wgh_memory(
    const at::Tensor& wgh,
    at::Tensor& wgh_blocked,
    memory::desc& wgh_usr_md,
    memory::desc& expected_wgh_md,
    dnnl::engine& engine,
    bool weight_cache_optimization,
    bool need_reorder = true) {
  memory wgh_m;
  if (wgh_usr_md != expected_wgh_md) {
    wgh_blocked =
        empty_opaque_tensor(expected_wgh_md, wgh.options(), c10::nullopt);
    wgh_m =
        dpcpp_onednn_memory(expected_wgh_md, engine, wgh_blocked.data_ptr());

    if (need_reorder) {
      auto reshaped_wgh = wgh;
      // reshape for group convolution weight
      if (wgh_blocked.ndimension() > wgh.ndimension()) {
        // for groups conv case:
        // expected_wgh will be 5-D Tensor based on expected_wgh_md:
        // g/o/i/h/w or g/o/h/w/i
        // wgh will be 4-D Tensor based on PyTorch
        // (g)o/i/h/w or (g)o/h/w/i
        // we need to manually reshape 4-D wgh to 5-D,
        // consistent with expected_wgh
        reshaped_wgh = share_storage_and_set_strided_as(
            wgh,
            wgh_blocked.sizes(),
            /*compatible with different strides of weight (including contiguous,
               channels_last and non-contiguous) */
            compatible_groups_conv_strides(wgh, wgh_blocked.sizes().vec()),
            c10::nullopt);
      }
      torch_ipex::xpu::oneDNN::reorder(reshaped_wgh, wgh_blocked);

      if (weight_cache_optimization) {
        auto wgh_opt_ctx = DPCPPTensorContext::release_tensor_ctx(wgh_blocked);
        wgh_opt_ctx.set_aten_meta(
            {reshaped_wgh.sizes().vec(), reshaped_wgh.strides().vec()});
        DPCPPTensorContext::set_tensor_ctx(wgh, std::move(wgh_opt_ctx));
      }
    }
  } else {
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    wgh_blocked = wgh;
  }
  return wgh_m;
}

static memory conv_get_expected_dst_memory(
    const at::Tensor& dst,
    at::Tensor& dst_blocked,
    memory::desc& dst_usr_md,
    memory::desc& expected_dst_md,
    dnnl::engine& engine,
    bool need_reorder = true) {
  memory dst_m;
  if (dst_usr_md != expected_dst_md) {
    dst_blocked =
        empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    dst_m =
        dpcpp_onednn_memory(expected_dst_md, engine, dst_blocked.data_ptr());

    if (need_reorder)
      torch_ipex::xpu::oneDNN::reorder(dst, dst_blocked);
  } else {
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
    dst_blocked = dst;
  }
  return dst_m;
}

sycl::event convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& bia,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr,
    const std::vector<sycl::event>& deps) {
  at::Device curDevice = at::Device(kXPU, at::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  // Indicate on which device the engine is created
  auto engine_index = curDevice.index();
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv =
      get_memory_layout_for_conv(src, wgh, /*is_transposed*/ false);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;

  // create usr_md for tensors, and md for conv primitive
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  std::tie(src_usr_md, wgh_usr_md, dst_usr_md) =
      conv_get_usr_md(src, wgh, dst, groups, memory_layout_for_conv);

  memory::dims src_dims = src.sizes().vec();
  memory::dims wgh_dims = wgh.sizes().vec();
  memory::dims dst_dims = dst.sizes().vec();
  auto src_data_t = get_onednn_dtype_include_double(src);
  auto wgh_data_t = get_onednn_dtype_include_double(wgh);
  auto dst_data_t = get_onednn_dtype_include_double(dst);

  memory::dims bia_dims;
  memory::data_type bia_data_t;
  memory::desc bia_md;

  if (bia.defined()) {
    bia_dims = bia.sizes().vec();
    bia_data_t = get_onednn_dtype_include_double(bia);
    bia_md = memory::desc({dst.size(1)}, bia_data_t, memory::format_tag::x);
  }

  // create conv primitive descriptor
  memory::dims _stride = stride.vec();
  memory::dims _dilation = compatible_dilation(dilation);
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();

  lru_key_t key_primitive;

  bool use_deterministic_algorithm = globalContext().deterministicAlgorithms();
  bool onednn_deterministic_enabled =
      Settings::I().is_onednn_deterministic_enabled();

#ifdef USE_PRIMITIVE_CACHE
  create_key(
      key_primitive,
      engine_index,
      src_dims,
      wgh_dims,
      bia_dims,
      dst_dims,
      src_data_t,
      wgh_data_t,
      bia_data_t,
      dst_data_t,
      groups,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      is_onednn_layout_suggested,
      memory_layout_for_conv,
      use_deterministic_algorithm,
      onednn_deterministic_enabled,
      attr);
#endif

  convolution_forward::primitive_desc conv_fwd_pd;
  convolution_forward conv_fwd;

#ifdef USE_PRIMITIVE_CACHE
  bool load_from_cache = find_key<convolution_forward>(key_primitive);
#else
  bool load_from_cache = false;
#endif

  if (load_from_cache) {
    conv_fwd = fetch_m<convolution_forward>(key_primitive);
    auto conv_fwd_pd_t = conv_fwd.get_primitive_desc();
    conv_fwd_pd = convolution_forward::primitive_desc(
        const_cast<dnnl_primitive_desc_t>(conv_fwd_pd_t));
  } else {
    // extract post ops
    primitive_attr pattr;
    post_ops po;
    attr.extract_post_ops(po, dst);
    pattr.set_post_ops(po);

    if (use_deterministic_algorithm || onednn_deterministic_enabled)
      pattr.set_deterministic(true);

#ifdef USE_SCRATCHPAD_MODE
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

    if (src_data_t == memory::data_type::f32) {
      pattr.set_fpmath_mode(torch_ipex::xpu::oneDNN::get_onednn_fpmath_mode());
    }

    memory::desc src_md, wgh_md, dst_md;
    std::tie(src_md, wgh_md, dst_md) =
        conv_get_md(src_usr_md, wgh_usr_md, dst_usr_md, memory_layout_for_conv);

    conv_fwd_pd = convolution_forward::primitive_desc(
        engine,
        prop_kind::forward,
        algorithm::convolution_direct,
        src_md,
        wgh_md,
        bia_md,
        dst_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right,
        pattr);

#ifdef USE_PRIMITIVE_CACHE
    conv_fwd =
        create_and_fetch_m<convolution_forward>(key_primitive, conv_fwd_pd);
#else
    conv_fwd = convolution_forward(conv_fwd_pd);
#endif
  }

  auto weight_cache_optimization = [&]() {
    return memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked &&
        !at::GradMode::is_enabled();
  }();

  memory src_m, wgh_m, dst_m, bia_m;
  Tensor src_blocked, wgh_blocked, dst_blocked = dst;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = conv_fwd_pd.src_desc();
    auto expected_wgh_md = conv_fwd_pd.weights_desc();
    auto expected_dst_md = conv_fwd_pd.dst_desc();
    src_m = conv_get_expected_src_memory(
        src, src_blocked, src_usr_md, expected_src_md, engine);
    wgh_m = conv_get_expected_wgh_memory(
        wgh,
        wgh_blocked,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        weight_cache_optimization);
    dst_m = conv_get_expected_dst_memory(
        dst, dst_blocked, dst_usr_md, expected_dst_md, engine, attr.with_sum());
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  }

  std::unordered_map<int, memory> args;
  if (bia.defined()) {
    bia_m = dpcpp_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }

  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, args);

  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, wgh_m});
  args.insert({DNNL_ARG_DST, dst_m});

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  DPCPP_ONEDNN_EXEC_WITH_EVENT(conv_fwd, strm, args, deps);

  if (is_onednn_layout_suggested && dst_blocked.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_blocked);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  return e;
}

sycl::event convolution_backward_weights(
    at::Tensor& diff_wgh,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef diff_wgh_aten_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    const std::vector<sycl::event>& deps) {
  at::Device curDevice = at::Device(kXPU, at::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  // Indicate on which device the engine is created
  auto engine_index = curDevice.index();
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv =
      get_memory_layout_for_conv(src, diff_dst, /*is_transposed=*/false);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;

  // create memory desc
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  std::tie(src_usr_md, wgh_usr_md, dst_usr_md) =
      conv_get_usr_md(src, diff_wgh, diff_dst, groups, memory_layout_for_conv);

  memory::dims diff_dst_dims = diff_dst.sizes().vec();
  memory::dims src_dims = src.sizes().vec();
  auto diff_dst_data_t = get_onednn_dtype_include_double(diff_dst);
  auto src_data_t = get_onednn_dtype_include_double(src);

  memory::dims bia_dims;
  memory::data_type bia_data_t;
  memory::desc bia_md;

  if (diff_bia.defined()) {
    bia_dims = diff_bia.sizes().vec();
    bia_data_t = get_onednn_dtype_include_double(diff_bia);
    bia_md =
        memory::desc({diff_dst.size(1)}, src_data_t, memory::format_tag::x);
  }

  // create fwd primitive hint
  memory::dims _stride = stride.vec();
  memory::dims _dilation = compatible_dilation(dilation);
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();

  bool use_deterministic_algorithm = globalContext().deterministicAlgorithms();
  bool onednn_deterministic_enabled =
      Settings::I().is_onednn_deterministic_enabled();

  lru_key_t key_primitive;

#ifdef USE_PRIMITIVE_CACHE
  create_key(
      key_primitive,
      engine_index,
      diff_dst_dims,
      src_dims,
      bia_dims,
      diff_dst_data_t,
      src_data_t,
      bia_data_t,
      groups,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      memory_layout_for_conv,
      is_onednn_layout_suggested,
      use_deterministic_algorithm,
      onednn_deterministic_enabled);
#endif

  convolution_backward_weights::primitive_desc conv_bwd_w_pd;
  dnnl::convolution_backward_weights conv_bwd_w;

#ifdef USE_PRIMITIVE_CACHE
  bool load_from_cache =
      find_key<dnnl::convolution_backward_weights>(key_primitive);
#else
  bool load_from_cache = false;
#endif

  if (load_from_cache) {
    conv_bwd_w = fetch_m<dnnl::convolution_backward_weights>(key_primitive);
    auto conv_bwd_w_pd_t = conv_bwd_w.get_primitive_desc();
    conv_bwd_w_pd = convolution_backward_weights::primitive_desc(
        const_cast<dnnl_primitive_desc_t>(conv_bwd_w_pd_t));
  } else {
    primitive_attr pattr;
    if (src_data_t == memory::data_type::f32) {
      pattr.set_fpmath_mode(torch_ipex::xpu::oneDNN::get_onednn_fpmath_mode());
    }
#ifdef USE_SCRATCHPAD_MODE
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

    if (use_deterministic_algorithm || onednn_deterministic_enabled)
      pattr.set_deterministic(true);

    memory::desc src_md, wgh_md, dst_md;
    std::tie(src_md, wgh_md, dst_md) =
        conv_get_md(src_usr_md, wgh_usr_md, dst_usr_md, memory_layout_for_conv);

    auto conv_fwd_pd = convolution_forward::primitive_desc(
        engine,
        prop_kind::forward,
        algorithm::convolution_direct,
        src_md,
        wgh_md,
        bia_md,
        dst_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right,
        pattr);

    // create bwd weight primitive
    conv_bwd_w_pd = convolution_backward_weights::primitive_desc(
        engine,
        algorithm::convolution_direct,
        src_md,
        wgh_md,
        bia_md,
        dst_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right,
        conv_fwd_pd,
        pattr);

#ifdef USE_PRIMITIVE_CACHE
    conv_bwd_w = create_and_fetch_m<dnnl::convolution_backward_weights>(
        key_primitive, conv_bwd_w_pd);
#else
    conv_bwd_w = dnnl::convolution_backward_weights(conv_bwd_w_pd);
#endif
  }

  // create bwd memory
  Tensor expected_src, expected_diff_dst, expected_diff_wgh;
  memory src_m, diff_dst_m, diff_wgh_m;

  if (is_onednn_layout_suggested) {
    auto expected_src_md = conv_bwd_w_pd.src_desc();
    auto expected_dst_md = conv_bwd_w_pd.diff_dst_desc();
    auto expected_wgh_md = conv_bwd_w_pd.diff_weights_desc();
    src_m = conv_get_expected_src_memory(
        src, expected_src, src_usr_md, expected_src_md, engine);
    diff_wgh_m = conv_get_expected_wgh_memory(
        diff_wgh,
        expected_diff_wgh,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        false, // weight_cache
        false); // need_reorder
    diff_dst_m = conv_get_expected_dst_memory(
        diff_dst, expected_diff_dst, dst_usr_md, expected_dst_md, engine);

  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    diff_dst_m = dpcpp_onednn_memory(dst_usr_md, engine, diff_dst.data_ptr());
    diff_wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, diff_wgh.data_ptr());
  }

  // insert args
  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_wgh_m});
  if (diff_bia.defined()) {
    memory diff_bia_m =
        dpcpp_onednn_memory(bia_md, engine, diff_bia.data_ptr());
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }
#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = conv_bwd_w_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      conv_bwd_w_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  // execute primitive
  DPCPP_ONEDNN_EXEC_WITH_EVENT(conv_bwd_w, strm, args, deps);

  if (is_onednn_layout_suggested && diff_wgh_m.get_desc() != wgh_usr_md) {
    // expected_diff_wgh contains the result of gw backward in blk format.
    // In training mode, plain gw output is expected for sgd update
    // Thus, we need one additional reorder here to make diff_wgh plain.
    auto reshaped_diff_wgh = diff_wgh;
    if (expected_diff_wgh.ndimension() > diff_wgh.ndimension()) {
      // for groups conv case:
      // expected_diff_wgh will be 5-D Tensor based on expected_diff_wgh_md:
      // g/o/i/h/w or g/o/h/w/i
      // diff_wgh will be 4-D Tensor based on PyTorch
      // (g)o/i/h/w or (g)o/h/w/i
      // we need to manually reshape 5-D expected_diff_wgh to 4-D,
      // consistent with PyTorch diff_wgh
      reshaped_diff_wgh = share_storage_and_set_strided_as(
          diff_wgh,
          expected_diff_wgh.sizes(),
          compatible_groups_conv_strides(
              diff_wgh, expected_diff_wgh.sizes().vec()),
          c10::nullopt);
    }
    torch_ipex::xpu::oneDNN::reorder(expected_diff_wgh, reshaped_diff_wgh);
  }

  // e is a sycl::event defined in DPCPP_ONEDNN_EXEC_WITH_EVENT
  return e;
}

sycl::event convolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    const std::vector<sycl::event>& deps) {
  at::Device curDevice = at::Device(kXPU, at::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  // Indicate on which device the engine is created
  auto engine_index = curDevice.index();
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv =
      get_memory_layout_for_conv(diff_dst, weight, /*is_transposed=*/false);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;

  // create memory desc
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  std::tie(src_usr_md, wgh_usr_md, dst_usr_md) = conv_get_usr_md(
      diff_src, weight, diff_dst, groups, memory_layout_for_conv);

  memory::dims diff_dst_dims = diff_dst.sizes().vec();
  memory::dims wgh_dims = weight.sizes().vec();
  auto diff_dst_data_t = get_onednn_dtype_include_double(diff_dst);
  auto wgh_data_t = get_onednn_dtype_include_double(weight);

  memory::format_tag bia_fmt = memory::format_tag::x;
  auto bia_md = bias_defined
      ? memory::desc({diff_dst.size(1)}, wgh_data_t, bia_fmt)
      : memory::desc();

  bool use_deterministic_algorithm = globalContext().deterministicAlgorithms();
  bool onednn_deterministic_enabled =
      Settings::I().is_onednn_deterministic_enabled();

  memory::dims _stride = stride.vec();
  memory::dims _dilation = compatible_dilation(dilation);
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();

  lru_key_t key_primitive;

#ifdef USE_PRIMITIVE_CACHE
  create_key(
      key_primitive,
      engine_index,
      diff_dst_dims,
      wgh_dims,
      diff_dst_data_t,
      wgh_data_t,
      groups,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      memory_layout_for_conv,
      is_onednn_layout_suggested,
      use_deterministic_algorithm,
      onednn_deterministic_enabled);
#endif

  convolution_backward_data::primitive_desc conv_bwd_data_pd;
  dnnl::convolution_backward_data conv_bwd_data;

#ifdef USE_PRIMITIVE_CACHE
  bool load_from_cache =
      find_key<dnnl::convolution_backward_data>(key_primitive);
#else
  bool load_from_cache = false;
#endif

  if (load_from_cache) {
    conv_bwd_data = fetch_m<dnnl::convolution_backward_data>(key_primitive);
    auto conv_bwd_data_pd_t = conv_bwd_data.get_primitive_desc();
    conv_bwd_data_pd = convolution_backward_data::primitive_desc(
        const_cast<dnnl_primitive_desc_t>(conv_bwd_data_pd_t));
  } else {
    memory::desc src_md, wgh_md, dst_md;
    std::tie(src_md, wgh_md, dst_md) =
        conv_get_md(src_usr_md, wgh_usr_md, dst_usr_md, memory_layout_for_conv);

    // create fwd primitive desc hint
    primitive_attr pattr;
    if (dst_usr_md.get_data_type() == memory::data_type::f32) {
      pattr.set_fpmath_mode(torch_ipex::xpu::oneDNN::get_onednn_fpmath_mode());
    }

    if (use_deterministic_algorithm || onednn_deterministic_enabled)
      pattr.set_deterministic(true);

#ifdef USE_SCRATCHPAD_MODE
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

    auto conv_fwd_pd = convolution_forward::primitive_desc(
        engine,
        prop_kind::forward,
        algorithm::convolution_direct,
        src_md,
        wgh_md,
        bia_md,
        dst_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right,
        pattr);

    conv_bwd_data_pd = convolution_backward_data::primitive_desc(
        engine,
        algorithm::convolution_direct,
        src_md,
        wgh_md,
        dst_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right,
        conv_fwd_pd,
        pattr);

#ifdef USE_PRIMITIVE_CACHE
    conv_bwd_data = create_and_fetch_m<dnnl::convolution_backward_data>(
        key_primitive, conv_bwd_data_pd);
#else
    conv_bwd_data = dnnl::convolution_backward_data(conv_bwd_data_pd);
#endif
  }

  // create memory
  Tensor expected_src, expected_wei, expected_dst;
  memory diff_dst_m, wei_m, diff_src_m;

  if (is_onednn_layout_suggested) {
    auto expected_src_md = conv_bwd_data_pd.diff_src_desc();
    auto expected_wgh_md = conv_bwd_data_pd.weights_desc();
    auto expected_dst_md = conv_bwd_data_pd.diff_dst_desc();
    diff_src_m = conv_get_expected_src_memory(
        diff_src, expected_src, src_usr_md, expected_src_md, engine, false);
    wei_m = conv_get_expected_wgh_memory(
        weight,
        expected_wei,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        false); // weight_cache
    diff_dst_m = conv_get_expected_dst_memory(
        diff_dst, expected_dst, dst_usr_md, expected_dst_md, engine);
  } else {
    diff_src_m = dpcpp_onednn_memory(src_usr_md, engine, diff_src.data_ptr());
    wei_m = dpcpp_onednn_memory(wgh_usr_md, engine, weight.data_ptr());
    diff_dst_m = dpcpp_onednn_memory(dst_usr_md, engine, diff_dst.data_ptr());
  }

  // insert args
  std::unordered_map<int, memory> args;
#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = conv_bwd_data_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, diff_dst.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(
      conv_bwd_data_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
#endif
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_WEIGHTS, wei_m});
  args.insert({DNNL_ARG_DIFF_SRC, diff_src_m});

  // execute primitive
  DPCPP_ONEDNN_EXEC_WITH_EVENT(conv_bwd_data, strm, args, deps);

  // propagate blk format
  if (is_onednn_layout_suggested &&
      diff_src.data_ptr() != expected_src.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(expected_src);
    DPCPPTensorContext::set_tensor_ctx(diff_src, std::move(blk_ctx));
  }

  // e is a sycl::event defined in DPCPP_ONEDNN_EXEC_WITH_EVENT
  return e;
}

} // namespace oneDNN
} // namespace torch_ipex::xpu
