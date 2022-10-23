#pragma once

#include <ATen/ATen.h>
#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace xpu::dpcpp;

namespace xpu {
namespace oneDNN {
namespace {

static void valid_dims(
    const Tensor& first,
    const Tensor& second,
    int dimension) {
  TORCH_CHECK(
      first.ndimension() == second.ndimension(),
      "Tensors must have same number of dimensions");

  for (int dim = 0; dim < first.ndimension(); dim++) {
    if (dim == dimension)
      continue;

    TORCH_CHECK(
        first.size(dim) == second.size(dim),
        "Sizes of tensors must match except in dimension");
  }
}

} // namespace

static void concat(Tensor& dst, const TensorList srcs, int dimension) {
  Tensor not_skip; // non-owning reference
  int nsrc = srcs.size();

  auto should_skip = [](const Tensor& t) {
    return t.dim() == 1 && at::native::size(t, 0) == 0;
  };

  int ndim = 0;
  for (int i = 0; i < nsrc; i++) {
    if (should_skip(srcs[i])) {
      continue;
    }
    ndim = srcs[i].dim();
    not_skip = srcs[i];
  }

  if (!not_skip.defined()) {
    return;
  }

  TORCH_CHECK(nsrc > 0, "invalid number of srcs");
  TORCH_CHECK(dimension >= 0, "invalid dimension");

  bool is_channels_last = is_smf_channels_last(srcs[0]);
  auto smf = suggest_memory_format_dpcpp(srcs[0]);

  // prepare srcs
  std::vector<Tensor> valid_srcs;
  int64_t cat_sz = 0;
  for (int i = 0; i < nsrc; i++) {
    const Tensor& cur = srcs[i];
    if (should_skip(cur)) {
      continue;
    }
    valid_dims(srcs[0], cur, dimension);
    cat_sz += cur.size(dimension);
    valid_srcs.push_back(cur);
  }

  // prepare dst
  memory::dims dst_tz;
  for (int dim = 0; dim < ndim; dim++) {
    if (dim == dimension) {
      dst_tz.push_back(cat_sz);
    } else {
      dst_tz.push_back(valid_srcs[0].size(dim));
    }
  }

  if (CHANNELSLAST1D_DPCPP == smf) {
    dst.resize_(dst_tz, at::MemoryFormat::Contiguous);
    dst = convert_tensor_to_channels_last_1d(dst);
  } else {
    dst.resize_(dst_tz, smf);
  }

  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});

  std::vector<memory::desc> srcs_md;
  std::vector<memory> srcs_m;
  bool is_block = false;
  for (size_t i = 0; i < valid_srcs.size(); i++) {
    auto src = valid_srcs[i];
    auto src_tz = get_onednn_dims(src);
    auto src_dt = get_onednn_dtype(src);

    // 1. CL case: Non-ChannelsLast tensors have been CL contiguous after
    // 'prepare srcs'.
    // 2. cont case: choose a real memory format
    // 3. block case: using underlying oneDNN md
    // FIXME: https://jira.devtools.intel.com/browse/MFDNN-7771
    // WA case, shape:4x1x4, stride:4x1x1, regarding it as `abc` fmt.
    memory::desc plain_md;
    auto ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    if (is_smf_channels_last(src) && src.size(1) == 1) {
      auto src_fmt = get_dnnl_default_format(ndim, false);
      plain_md = memory::desc(src_tz, src_dt, src_fmt);
    } else {
      auto src_st = get_onednn_strides(src);
      plain_md = memory::desc(src_tz, src_dt, src_st);
    }

    memory::desc src_md = ctx.is_plain() ? plain_md : ctx.meta();
    srcs_md.push_back(src_md);

    auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    srcs_m.push_back(src_m);
    if (!is_block)
      is_block = !ctx.is_plain();
  }

  auto dst_dt = get_onednn_dtype(valid_srcs[0]);
  auto dst_fmt = is_channels_last || !is_block
      ? get_dnnl_default_format(ndim, is_channels_last)
      : memory::format_tag::any;
  auto dst_md = memory::desc(dst_tz, dst_dt, dst_fmt);
  auto concat_pd = concat::primitive_desc(
      dst_md, static_cast<int>(dimension), srcs_md, engine);

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, dst_dims, static_cast<int>(dimension), cat_tensors_md);
#endif

  memory dst_m;
  auto expected_dst_md = concat_pd.dst_desc();
  if (dst_md != expected_dst_md) {
    Tensor dst_opt;
    if (dst.is_quantized()) {
      // oneDNN s8/i8 memory includs some opaque segements.
      // Generaly, even if a s8/i8 dst shows plain format, but the condition,
      // `dst_md != expected_dst_md`, will be true, due to opaque segements.
      // You can check `md.get_size()` for the difference.
      dst_opt = at::AtenIpexTypeXPU::empty_opaque_qtensor(
          expected_dst_md, c10::nullopt, dst.quantizer());
    } else {
      dst_opt = at::AtenIpexTypeXPU::empty_opaque_tensor(
          expected_dst_md, valid_srcs[0].options(), c10::nullopt);
    }
    auto dst_opt_ctx = DPCPPTensorContext::release_tensor_ctx(dst_opt);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(dst_opt_ctx));
    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst.data_ptr());
  } else {
    dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
  }

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_DST, dst_m},
  };
  for (int i = 0; i < (int)valid_srcs.size(); i++) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs_m[i]});
  }

  auto strm = GpuStreamManager::Instance().get_stream();

#ifdef USE_PRIMITIVE_CACHE
  auto concat_p = fetch_or_create_m<dnnl::concat>(key, concat_pd);
#else
  auto concat_p = dnnl::concat(concat_pd);
#endif

  DPCPP_ONEDNN_EXEC(concat_p, strm, args);
}

static void concat(
    Tensor& dst,
    MaterializedITensorListRef srcs,
    int dimension) {
  Tensor not_skip; // non-owning reference
  int nsrc = srcs.size();

  auto should_skip = [](const Tensor& t) {
    return t.dim() == 1 && at::native::size(t, 0) == 0;
  };

  int ndim = 0;
  for (int i = 0; i < nsrc; i++) {
    if (should_skip(srcs[i].get())) {
      continue;
    }
    ndim = srcs[i].get().dim();
    not_skip = srcs[i].get();
  }

  if (!not_skip.defined()) {
    return;
  }

  TORCH_CHECK(nsrc > 0, "invalid number of srcs");
  TORCH_CHECK(dimension >= 0, "invalid dimension");

  bool is_channels_last = is_smf_channels_last(srcs[0].get());
  auto smf = suggest_memory_format_dpcpp(srcs[0].get());

  // prepare srcs
  std::vector<Tensor> valid_srcs;
  int64_t cat_sz = 0;
  for (int i = 0; i < nsrc; i++) {
    const Tensor& cur = srcs[i].get();
    if (should_skip(cur)) {
      continue;
    }
    valid_dims(srcs[0].get(), cur, dimension);
    cat_sz += cur.size(dimension);
    valid_srcs.push_back(cur);
  }

  // prepare dst
  memory::dims dst_tz;
  for (int dim = 0; dim < ndim; dim++) {
    if (dim == dimension) {
      dst_tz.push_back(cat_sz);
    } else {
      dst_tz.push_back(valid_srcs[0].size(dim));
    }
  }

  if (CHANNELSLAST1D_DPCPP == smf) {
    dst.resize_(dst_tz, at::MemoryFormat::Contiguous);
    dst = convert_tensor_to_channels_last_1d(dst);
  } else {
    dst.resize_(dst_tz, smf);
  }

  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});

  std::vector<memory::desc> srcs_md;
  std::vector<memory> srcs_m;
  bool is_block = false;
  for (size_t i = 0; i < valid_srcs.size(); i++) {
    auto src = valid_srcs[i];
    auto src_tz = get_onednn_dims(src);
    auto src_dt = get_onednn_dtype(src);

    // 1. CL case: Non-ChannelsLast tensors have been CL contiguous after
    // 'prepare srcs'.
    // 2. cont case: choose a real memory format
    // 3. block case: using underlying oneDNN md
    // FIXME: https://jira.devtools.intel.com/browse/MFDNN-7771
    // WA case, shape:4x1x4, stride:4x1x1, regarding it as `abc` fmt.
    memory::desc plain_md;
    auto ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    if (is_smf_channels_last(src) && src.size(1) == 1) {
      auto src_fmt = get_dnnl_default_format(ndim, false);
      plain_md = memory::desc(src_tz, src_dt, src_fmt);
    } else {
      auto src_st = get_onednn_strides(src);
      plain_md = memory::desc(src_tz, src_dt, src_st);
    }

    memory::desc src_md = ctx.is_plain() ? plain_md : ctx.meta();
    srcs_md.push_back(src_md);

    auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    srcs_m.push_back(src_m);
    if (!is_block)
      is_block = !ctx.is_plain();
  }

  auto dst_dt = get_onednn_dtype(valid_srcs[0]);
  auto dst_fmt = is_channels_last || !is_block
      ? get_dnnl_default_format(ndim, is_channels_last)
      : memory::format_tag::any;
  auto dst_md = memory::desc(dst_tz, dst_dt, dst_fmt);
  auto concat_pd = concat::primitive_desc(
      dst_md, static_cast<int>(dimension), srcs_md, engine);

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, dst_dims, static_cast<int>(dimension), cat_tensors_md);
#endif

  memory dst_m;
  auto expected_dst_md = concat_pd.dst_desc();
  if (dst_md != expected_dst_md) {
    Tensor dst_opt;
    if (dst.is_quantized()) {
      // oneDNN s8/i8 memory includs some opaque segements.
      // Generaly, even if a s8/i8 dst shows plain format, but the condition,
      // `dst_md != expected_dst_md`, will be true, due to opaque segements.
      // You can check `md.get_size()` for the difference.
      dst_opt = at::AtenIpexTypeXPU::empty_opaque_qtensor(
          expected_dst_md, c10::nullopt, dst.quantizer());
    } else {
      dst_opt = at::AtenIpexTypeXPU::empty_opaque_tensor(
          expected_dst_md, valid_srcs[0].options(), c10::nullopt);
    }
    auto dst_opt_ctx = DPCPPTensorContext::release_tensor_ctx(dst_opt);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(dst_opt_ctx));
    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst.data_ptr());
  } else {
    dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
  }

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_DST, dst_m},
  };
  for (int i = 0; i < (int)valid_srcs.size(); i++) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs_m[i]});
  }

  auto strm = GpuStreamManager::Instance().get_stream();

#ifdef USE_PRIMITIVE_CACHE
  auto concat_p = fetch_or_create_m<dnnl::concat>(key, concat_pd);
#else
  auto concat_p = dnnl::concat(concat_pd);
#endif

  DPCPP_ONEDNN_EXEC(concat_p, strm, args);
}

} // namespace oneDNN
} // namespace xpu
