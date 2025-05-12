#pragma once

#include <ATen/ATen.h>

#include <utils/LRUCache.h>
#include "Attr.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

using namespace torch_ipex::xpu::oneDNN;
using namespace torch_ipex::xpu::dpcpp;

namespace std {

template <>
struct hash<dnnl::memory::dims> {
  size_t operator()(dnnl::memory::dims const& vec) const {
    size_t seed = vec.size();
    for (auto& i : vec) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

} // namespace std

namespace torch_ipex {
namespace xpu {
namespace oneDNN {

template <typename T>
T concat(const T& t1, at::ScalarType d) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back((int64_t)d);

  return t;
}

template <typename T>
T concat(const T& t1, bool b) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back(b);

  return t;
}

template <typename T>
T concat(const T& t1, int b) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back(b);

  return t;
}

template <typename T>
T concat(const T& t1, const T& t2) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.insert(t.end(), t2.begin(), t2.end());

  return t;
}

template <typename T1, typename T2, typename... Ts>
T1 concat(const T1& t1, const T2& t2, const Ts&... ts) {
  return concat(concat(t1, t2), ts...);
}

} // namespace oneDNN
} // namespace xpu
} // namespace torch_ipex

namespace dnnl {

enum class joint_dtypes_t {
  _f32 = 0,
  _f16,
  _bf16,
  _int8,
  _f16_int4,
  _bf16_int4,
  _f16_f8_e5m2,
  _bf16_f8_e5m2,
};

enum class trans_type_t { _nn = 0, _nt, _tn, _tt };

enum class bias_type_t { _none = 0, _scalar, _m, _n, _mn };

template <joint_dtypes_t Ts>
struct onednn_types_mapper;

template <>
struct onednn_types_mapper<joint_dtypes_t::_f16_int4> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(
        dnnl::memory::data_type::f16, dnnl::memory::data_type::u4);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::_bf16_int4> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(
        dnnl::memory::data_type::bf16, dnnl::memory::data_type::u4);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::_f16_f8_e5m2> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(
        dnnl::memory::data_type::f16, dnnl::memory::data_type::f8_e5m2);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::_bf16_f8_e5m2> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(
        dnnl::memory::data_type::bf16, dnnl::memory::data_type::f8_e5m2);
  }
};

// TODO: bias types maybe not right
static inline dnnl::memory::dims get_bias_type(
    bias_type_t b_dims,
    const int m,
    const int n) {
  switch (b_dims) {
    case bias_type_t::_none:
      return {0};
    case bias_type_t::_scalar:
      return {1, 1};
    case bias_type_t::_m:
      return {m, 1};
    case bias_type_t::_n:
      return {1, n};
    case bias_type_t::_mn:
      return {m, n};
    default:
      throw std::runtime_error("unsupported bias type ...");
  }
}

static constexpr int cache_capacity = 512;

class primitive_ext : public primitive {
  static constexpr int max_args = 12;

 public:
  primitive_ext(const primitive& base) : primitive(base) {}
  primitive_ext(primitive&& base) : primitive(std::move(base)) {}

  /// Returns a memory descriptor.
  ///
  /// @note
  ///     There are also convenience methods
  ///     #dnnl::primitive_desc_base::src_desc(),
  ///     #dnnl::primitive_desc_base::dst_desc(), and others.
  ///
  /// @param what The kind of parameter to query; can be
  ///     #dnnl::query::src_md, #dnnl::query::dst_md, etc.
  /// @param idx Index of the parameter. For example, convolution bias can
  ///     be queried with what = #dnnl::query::weights_md and idx = 1.
  /// @returns The requested memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     parameter of the specified kind or index.
  const_dnnl_memory_desc_t query_md(query what, int idx = 0) const {
    std::vector<query> valid_q{
        query::src_md,
        query::diff_src_md,
        query::weights_md,
        query::diff_weights_md,
        query::dst_md,
        query::diff_dst_md,
        query::workspace_md,
        query::scratchpad_md,
        query::exec_arg_md};
    if (!std::any_of(valid_q.cbegin(), valid_q.cend(), [=](query q) {
          return what == q;
        }))
      DNNL_THROW_ERROR(
          dnnl_invalid_arguments, "memory descriptor query is invalid");

    const_dnnl_memory_desc_t cdesc = dnnl_primitive_desc_query_md(
        this->get_primitive_desc(), dnnl::convert_to_c(what), idx);

    return cdesc ? cdesc : nullptr;
  }

  /// Returns a source memory descriptor.
  /// @param idx Source index.
  /// @returns Source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     source parameter with index @p idx.
  const_dnnl_memory_desc_t src_desc(int idx) const {
    return query_md(query::src_md, idx);
  }

  /// Returns a destination memory descriptor.
  /// @param idx Destination index.
  /// @returns Destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     destination parameter with index @p idx.
  const_dnnl_memory_desc_t dst_desc(int idx) const {
    return query_md(query::dst_md, idx);
  }

  /// Returns a weights memory descriptor.
  /// @param idx Weights index.
  /// @returns Weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     weights parameter with index @p idx.
  const_dnnl_memory_desc_t weights_desc(int idx) const {
    return query_md(query::weights_md, idx);
  }

  /// Returns a diff source memory descriptor.
  /// @param idx Diff source index.
  /// @returns Diff source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff source parameter with index @p idx.
  const_dnnl_memory_desc_t diff_src_desc(int idx) const {
    return query_md(query::diff_src_md, idx);
  }

  /// Returns a diff destination memory descriptor.
  /// @param idx Diff destination index.
  /// @returns Diff destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff destination parameter with index @p idx.
  const_dnnl_memory_desc_t diff_dst_desc(int idx) const {
    return query_md(query::diff_dst_md, idx);
  }

  /// Returns a diff weights memory descriptor.
  /// @param idx Diff weights index.
  /// @returns Diff weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff weights parameter with index @p idx.
  const_dnnl_memory_desc_t diff_weights_desc(int idx) const {
    return query_md(query::diff_weights_md, idx);
  }

  const_dnnl_memory_desc_t exec_arg_desc(int idx) const {
    return query_md(query::exec_arg_md, idx);
  }

  // Separate versions without the index argument for documentation
  // purposes.

  /// Returns a source memory descriptor.
  /// @returns Source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     source parameter.
  const_dnnl_memory_desc_t src_desc() const {
    return src_desc(0);
  }

  /// Returns a destination memory descriptor.
  /// @returns Destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     destination parameter.
  const_dnnl_memory_desc_t dst_desc() const {
    return dst_desc(0);
  }

  /// Returns a weights memory descriptor.
  /// @returns Weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     weights parameter.
  const_dnnl_memory_desc_t weights_desc() const {
    return weights_desc(0);
  }

  /// Returns a diff source memory descriptor.
  /// @returns Diff source memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff source memory with.
  const_dnnl_memory_desc_t diff_src_desc() const {
    return diff_src_desc(0);
  }

  /// Returns a diff destination memory descriptor.
  /// @returns Diff destination memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff destination parameter.
  const_dnnl_memory_desc_t diff_dst_desc() const {
    return diff_dst_desc(0);
  }

  /// Returns a diff weights memory descriptor.
  /// @returns Diff weights memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not have a
  ///     diff weights parameter.
  const_dnnl_memory_desc_t diff_weights_desc() const {
    return diff_weights_desc(0);
  }

  /// Returns the workspace memory descriptor.
  /// @returns Workspace memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not require
  ///     workspace parameter.
  const_dnnl_memory_desc_t workspace_desc() const {
    return query_md(query::workspace_md, 0);
  }

  /// Returns the scratchpad memory descriptor.
  /// @returns scratchpad memory descriptor.
  /// @returns A zero memory descriptor if the primitive does not require
  ///     scratchpad parameter.
  /// @sa @ref dev_guide_attributes_scratchpad
  const_dnnl_memory_desc_t scratchpad_desc() const {
    return query_md(query::scratchpad_md, 0);
  }

  inline memory make_memory(
      const_dnnl_memory_desc_t md_t,
      const engine& aengine,
      void* handle = DNNL_MEMORY_ALLOCATE) const {
    sycl_interop::memory_kind kind = dnnl::sycl_interop::memory_kind::usm;
    dnnl_memory_t c_memory;
    error::wrap_c_api(
        dnnl_sycl_interop_memory_create(
            &c_memory, md_t, aengine.get(), convert_to_c(kind), handle),
        "could not create a memory");
    return memory(c_memory);
  }

  memory make_src(const engine& aengine, void* handle = DNNL_MEMORY_ALLOCATE)
      const {
    return make_memory(src_desc(), aengine, handle);
  }

  memory make_weight(const engine& aengine, void* handle = DNNL_MEMORY_ALLOCATE)
      const {
    return make_memory(weights_desc(), aengine, handle);
  }

  memory make_bias(const engine& aengine, void* handle = DNNL_MEMORY_ALLOCATE)
      const {
    return make_memory(weights_desc(1), aengine, handle);
  }

  memory make_dst(const engine& aengine, void* handle = DNNL_MEMORY_ALLOCATE)
      const {
    return make_memory(dst_desc(), aengine, handle);
  }

  memory make_scratchpad(
      const engine& aengine,
      void* handle = DNNL_MEMORY_ALLOCATE) const {
    return make_memory(scratchpad_desc(), aengine, handle);
  }

  size_t get_scratchpad_size() const {
    return dnnl_memory_desc_get_size(scratchpad_desc());
  }

  memory make_args(int arg_class, const engine& aengine, void* handle) const {
    switch (arg_class) {
      case DNNL_ARG_SRC:
        return make_src(aengine, handle);
      case DNNL_ARG_WEIGHTS:
        return make_weight(aengine, handle);
      case DNNL_ARG_SCRATCHPAD:
        return make_scratchpad(aengine, handle);
      case DNNL_ARG_DST:
        return make_dst(aengine, handle);
      case DNNL_ARG_BIAS:
        return make_bias(aengine, handle);
      default:
        throw std::exception();
    }
  }

  template <typename M>
  void set_attribute(int slot, int arg_class, void* handle, M constructor) {
    if (m[slot])
      m[slot].set_data_handle(handle);
    else {
      m[slot] = constructor();
      c_args[slot].arg = arg_class;
      c_args[slot].memory = m[slot].get();
    }
  }

  sycl::event execute(
      const stream& astream,
      const engine& aengine,
      std::vector<std::pair<int, void*>>&& handles,
      int slot_off = 2) {
    auto off = slot_off;
    for (const auto& p : handles) {
      auto& m_arg = m[off];
      if (m_arg)
        m_arg.set_data_handle(p.second);
      else {
        m_arg = make_args(p.first, aengine, p.second);
        c_args[off].arg = p.first;
        c_args[off].memory = m_arg.get();
      }
      ++off;
    }

    sycl::event return_event;
    std::vector<sycl::event> deps{};
    error::wrap_c_api(
        dnnl_sycl_interop_primitive_execute(
            this->get(), astream.get(), off, c_args, &deps, &return_event),
        "could not execute a primitive");
    return return_event;
  }

 private:
  memory m[max_args];
  dnnl_exec_arg_t c_args[max_args];
};

template <trans_type_t Tt>
static inline void get_strides(
    memory::dims& src_strides,
    memory::dims& wei_strides,
    memory::dims& dst_strides,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc) {}

template <>
static inline void get_strides<trans_type_t::_nt>(
    memory::dims& src_strides,
    memory::dims& wei_strides,
    memory::dims& dst_strides,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc) {
  src_strides = {lda, 1};
  wei_strides = {1, ldb};
  dst_strides = {ldc, 1};
}

template <>
static inline void get_strides<trans_type_t::_nn>(
    memory::dims& src_strides,
    memory::dims& wei_strides,
    memory::dims& dst_strides,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc) {
  src_strides = {lda, 1};
  wei_strides = {ldb, 1};
  dst_strides = {ldc, 1};
}

using primitive_cache = lru_cache<memory::dims, primitive_ext>;

template <trans_type_t Tt, joint_dtypes_t Ts, typename F>
struct create_matmul {
  static inline primitive_ext& get(
      const int m,
      const int n,
      const int k,
      const int64_t lda,
      const int64_t ldb,
      const int64_t ldc,
      const bias_type_t
          b_dims, // for shapeless bias, not put it into template parameter
      const int device_id,
      F f_attr) {
    auto& cached = get_cache(device_id);
    memory::dims src_strides, wei_strides, dst_strides;
    get_strides<Tt>(src_strides, wei_strides, dst_strides, lda, ldb, ldc);
    auto pri_key = torch_ipex::xpu::oneDNN::concat(
        src_strides, wei_strides, m, n, k, int(b_dims));
    auto iter = cached.find(pri_key);
    if (iter == cached.end()) {
      auto [src_dt, wei_dt] = onednn_types_mapper<Ts>::get();

      auto src_md = memory::desc({m, k}, src_dt, src_strides);
      auto wei_md = memory::desc({k, n}, wei_dt, wei_strides);
      // TODO: dst data type is not same as src data type?
      auto dst_md = memory::desc({m, n}, src_dt, dst_strides);
      auto bias_format = b_dims == bias_type_t::_none
          ? dnnl::memory::format_tag::undef
          : dnnl::memory::format_tag::ab;
      auto bias_md = memory::desc(
          get_bias_type(b_dims, m, n), src_dt, bias_format); // {m, n} or {1, n}

      primitive_attr pattr;
      f_attr(pattr);

      dnnl::matmul::primitive_desc matmul_pd;
      at::Device curDevice = at::Device(at::kXPU, device_id);
      auto aengine = GpuEngineManager::Instance().get_engine(curDevice);
      if (b_dims == bias_type_t::_none) {
        matmul_pd = dnnl::matmul::primitive_desc(
            aengine, src_md, wei_md, dst_md, pattr);
      } else {
        matmul_pd = dnnl::matmul::primitive_desc(
            aengine, src_md, wei_md, bias_md, dst_md, pattr);
      }

      return cached.insert({pri_key, primitive_ext(dnnl::matmul(matmul_pd))})
          .first->second;
    } else {
      return iter->second;
    }
  }

 private:
  static constexpr int max_cache_capacity = 512;
  // if default constructor of primitive cache could read the environment
  // variable then it'll save a lot of trouble
  static inline thread_local std::array<primitive_cache, 16> mappings;

  // this won't be needed if primitive_cache have good default constructor
  static inline primitive_cache& get_cache(const int device_id) {
    auto& mapping = mappings[device_id];
    if (mapping.max_size() == 0) {
      mapping.resize(max_cache_capacity);
    }
    return mapping;
  }
};

template <joint_dtypes_t Ts, typename F>
static inline primitive_ext& dnnlMatmulCreatePrimitive(
    const trans_type_t Tt,
    const bias_type_t b_dims,
    const int m,
    const int n,
    const int k,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int device_id,
    F attr) {
  switch (Tt) {
    case trans_type_t::_nt:
      return create_matmul<trans_type_t::_nt, Ts, F>::get(
          m, n, k, lda, ldb, ldc, b_dims, device_id, attr);
    case trans_type_t::_nn:
      return create_matmul<trans_type_t::_nn, Ts, F>::get(
          m, n, k, lda, ldb, ldc, b_dims, device_id, attr);
    default:
      throw std::runtime_error("unsupported trans type ...");
  }
}

template <typename F>
static inline primitive_ext& dnnlMatmulCreatePrimitive(
    const joint_dtypes_t Ts,
    const trans_type_t Tt,
    const bias_type_t b_dims,
    const int m,
    const int n,
    const int k,
    const int64_t lda,
    const int64_t ldb, // is weight ldb necessary?
    const int64_t ldc,
    const int device_id,
    F attr) {
  switch (Ts) {
    case joint_dtypes_t::_f16_int4:
      return dnnlMatmulCreatePrimitive<joint_dtypes_t::_f16_int4, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr);
    case joint_dtypes_t::_bf16_int4:
      return dnnlMatmulCreatePrimitive<joint_dtypes_t::_bf16_int4, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr);
    case joint_dtypes_t::_f16_f8_e5m2:
      return dnnlMatmulCreatePrimitive<joint_dtypes_t::_f16_f8_e5m2, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr);
    case joint_dtypes_t::_bf16_f8_e5m2:
      return dnnlMatmulCreatePrimitive<joint_dtypes_t::_bf16_f8_e5m2, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr);
    default:
      throw std::runtime_error("Only support int4 and fp8 gemm ...");
  }
}

template <typename F>
static inline primitive_ext& dnnlMatmulCreatePrimitive(
    const Tensor& src,
    const Tensor& wei,
    const c10::optional<Tensor>& bias,
    Tensor& dst,
    const Tensor& scale,
    const Tensor& zp,
    const int group_size,
    const engine& aengine,
    F attr) {
  static thread_local primitive_cache cached(cache_capacity);

  auto src_sz = src.sizes();
  auto wei_sz = wei.sizes();

  int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  int n = wei_sz[1]; // presume channel last format
  int k = *(src_sz.end() - 1);

  memory::dims src_dims{m, k};
  memory::dims src_stride = {src.stride(0), src.stride(1)};
  memory::dims wei_dims(wei_sz.begin(), wei_sz.end());
  memory::dims zp_dims(zp.sizes().begin(), zp.sizes().end());

  bool is_intype_bf16 = src.scalar_type() == at::ScalarType::BFloat16;
  auto pri_key = torch_ipex::xpu::oneDNN::concat(
      src_dims,
      wei_dims,
      src_stride,
      zp_dims,
      is_intype_bf16,
      (bool)bias,
      group_size);

  auto iter = cached.find(pri_key);
  if (iter == cached.end()) {
    auto flat_src = src.flatten(0, -2);
    // slow path
    auto src_dt = get_onednn_dtype(flat_src);
    auto src_md = get_onednn_md(flat_src);
    auto wei_md =
        memory::desc({k, n}, memory::data_type::u4, memory::format_tag::ba);

    primitive_attr pattr;
#ifdef USE_SCRATCHPAD_MODE
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {group_size, 1},
        get_onednn_dtype(scale));

    if (zp.dim() == 1) {
      pattr.set_zero_points(
          DNNL_ARG_WEIGHTS,
          /* mask */ 0,
          {},
          memory::data_type::s8);
    } else {
      const uint64_t num_groups = (uint64_t)(k / group_size);
      auto zp_md = memory::desc(
          {num_groups, n},
          memory::data_type::u4,
          memory::format_tag::ab); // {n, 1}
      pattr.set_zero_points(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {group_size, 1},
          memory::data_type::u4);
    }
    pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

    attr(pattr);

    dnnl::matmul::primitive_desc matmul_pd;
    auto dst_md = get_onednn_md(dst.flatten(0, -2));

    if (bias) {
      auto b = bias.value();
      if (b.dim() == 1) {
        TORCH_CHECK(
            b.size(0) == n || b.size(0) == 1,
            "matmul supports [n] or [1] when bias dim is 1 ...");
        b = b.expand({1, n});
      } else if (b.dim() == 2) {
        TORCH_CHECK(
            (b.size(0) == m && b.size(1) == n) ||
                (b.size(0) == 1 && b.size(1) == n) ||
                (b.size(0) == m && b.size(1) == 1) ||
                (b.size(0) == 1 && b.size(1) == 1),
            "matmul supports [m, n] or [1, n] or [m, 1] or [1, 1] when bias dim is 2 ...");
        if (b.size(0) == 1 && b.size(1) == 1)
          b = b.expand({1, n}).contiguous();
      } else if (b.dim() == 3) {
        TORCH_CHECK(
            are_expandable({1, m, n}, b.sizes()),
            "matmul bias must be expandable to:",
            dst.sizes(),
            " but got:",
            b.sizes());
        b = b.expand({1, m, n}).contiguous();
      } else if (b.dim() == 0) {
        TORCH_CHECK(
            b.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
        if (flat_src.dim() == 3) {
          b = b.expand({1, m, n}).contiguous();
        } else {
          b = b.expand({1, n}).contiguous();
        }
      } else {
        TORCH_CHECK(0, "unsupported bias dim in matmul ...");
      }

      auto bias_md = get_onednn_md(b);
      matmul_pd = dnnl::matmul::primitive_desc(
          aengine, src_md, wei_md, bias_md, dst_md, pattr);
    } else {
      matmul_pd =
          dnnl::matmul::primitive_desc(aengine, src_md, wei_md, dst_md, pattr);
    }

    return cached.insert({pri_key, primitive_ext(dnnl::matmul(matmul_pd))})
        .first->second;
  } else {
    return iter->second;
  }
}

} // namespace dnnl
