/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// @file
/// C++ API

#pragma once

#include <common/core/base_ops.hpp>
#include <common/core/base_types.hpp>
#include <common/core/common.hpp>
#include <common/core/explicit_conv.hpp>
#include <common/utils/limitation.hpp>

namespace gpu::xetla {

namespace detail {

/// @brief lookup table for cache hint.
///
///
constexpr auto get_cache_hint(gpu::xetla::cache_hint ch) {
  switch (ch) {
#if __INTEL_LLVM_COMPILER >= 20240100
    case gpu::xetla::cache_hint::none:
      return __ESIMD_NS::cache_hint::none;
    case gpu::xetla::cache_hint::uncached:
      return __ESIMD_NS::cache_hint::uncached;
    case gpu::xetla::cache_hint::cached:
      return __ESIMD_NS::cache_hint::cached;
    case gpu::xetla::cache_hint::write_back:
      return __ESIMD_NS::cache_hint::write_back;
    case gpu::xetla::cache_hint::write_through:
      return __ESIMD_NS::cache_hint::write_through;
    case gpu::xetla::cache_hint::streaming:
      return __ESIMD_NS::cache_hint::streaming;
    case gpu::xetla::cache_hint::read_invalidate:
      return __ESIMD_NS::cache_hint::read_invalidate;
    case gpu::xetla::cache_hint::const_cached:
      return __ESIMD_NS::cache_hint::const_cached;
#else
    case gpu::xetla::cache_hint::none:
      return __ESIMD_ENS::cache_hint::none;
    case gpu::xetla::cache_hint::uncached:
      return __ESIMD_ENS::cache_hint::uncached;
    case gpu::xetla::cache_hint::cached:
      return __ESIMD_ENS::cache_hint::cached;
    case gpu::xetla::cache_hint::write_back:
      return __ESIMD_ENS::cache_hint::write_back;
    case gpu::xetla::cache_hint::write_through:
      return __ESIMD_ENS::cache_hint::write_through;
    case gpu::xetla::cache_hint::streaming:
      return __ESIMD_ENS::cache_hint::streaming;
    case gpu::xetla::cache_hint::read_invalidate:
      return __ESIMD_ENS::cache_hint::read_invalidate;
#endif
  }
}

/// @brief lookup table for data size.
///
///
constexpr __ESIMD_ENS::lsc_data_size get_data_size(gpu::xetla::data_size ds) {
  switch (ds) {
    case gpu::xetla::data_size::default_size:
      return __ESIMD_ENS::lsc_data_size::default_size;
    case gpu::xetla::data_size::u8:
      return __ESIMD_ENS::lsc_data_size::u8;
    case gpu::xetla::data_size::u16:
      return __ESIMD_ENS::lsc_data_size::u16;
    case gpu::xetla::data_size::u32:
      return __ESIMD_ENS::lsc_data_size::u32;
    case gpu::xetla::data_size::u64:
      return __ESIMD_ENS::lsc_data_size::u64;
    case gpu::xetla::data_size::u8u32:
      return __ESIMD_ENS::lsc_data_size::u8u32;
    case gpu::xetla::data_size::u16u32:
      return __ESIMD_ENS::lsc_data_size::u16u32;
    case gpu::xetla::data_size::u16u32h:
      return __ESIMD_ENS::lsc_data_size::u16u32h;
  }
}

/// @brief lookup table for memory kind.
///
///
constexpr auto get_memory_kind(gpu::xetla::memory_kind mk) {
  switch (mk) {
#if __INTEL_LLVM_COMPILER >= 20240100
    case gpu::xetla::memory_kind::untyped_global:
      return __ESIMD_NS::memory_kind::global;
    case gpu::xetla::memory_kind::typed_global:
      return __ESIMD_NS::memory_kind::image;
    case gpu::xetla::memory_kind::shared_local:
      return __ESIMD_NS::memory_kind::local;
#else // legacy experimental api
    case gpu::xetla::memory_kind::untyped_global:
      return __ESIMD_ENS::lsc_memory_kind::untyped_global;
    case gpu::xetla::memory_kind::typed_global:
      return __ESIMD_ENS::lsc_memory_kind::typed_global;
    case gpu::xetla::memory_kind::shared_local:
      return __ESIMD_ENS::lsc_memory_kind::shared_local;
#endif
  }
}

/// @brief lookup table for fence op.
///
///
constexpr auto get_fence_op(gpu::xetla::fence_op fo) {
  switch (fo) {
#if __INTEL_LLVM_COMPILER >= 20240100
    case gpu::xetla::fence_op::none:
      return __ESIMD_NS::fence_flush_op::none;
    case gpu::xetla::fence_op::evict:
      return __ESIMD_NS::fence_flush_op::evict;
    case gpu::xetla::fence_op::invalidate:
      return __ESIMD_NS::fence_flush_op::invalidate;
    case gpu::xetla::fence_op::clean:
      return __ESIMD_NS::fence_flush_op::clean;
#else // legacy experimental api
    case gpu::xetla::fence_op::none: //
      return __ESIMD_ENS::lsc_fence_op::none;
    case gpu::xetla::fence_op::evict:
      return __ESIMD_ENS::lsc_fence_op::evict;
    case gpu::xetla::fence_op::invalidate:
      return __ESIMD_ENS::lsc_fence_op::invalidate;
    case gpu::xetla::fence_op::clean:
      return __ESIMD_ENS::lsc_fence_op::clean;
#endif
  }
}

/// @brief lookup table for fence scope.
///
///
constexpr auto get_fence_scope(gpu::xetla::fence_scope fs) {
  switch (fs) {
#if __INTEL_LLVM_COMPILER >= 20240100
    case gpu::xetla::fence_scope::group:
      return __ESIMD_NS::fence_scope::group;
    case gpu::xetla::fence_scope::local:
      return __ESIMD_NS::fence_scope::local;
    case gpu::xetla::fence_scope::tile:
      return __ESIMD_NS::fence_scope::tile;
    case gpu::xetla::fence_scope::gpu: //
      return __ESIMD_NS::fence_scope::gpu;
    case gpu::xetla::fence_scope::gpus:
      return __ESIMD_NS::fence_scope::gpus;
    case gpu::xetla::fence_scope::system:
      return __ESIMD_NS::fence_scope::system;
    case gpu::xetla::fence_scope::sysacq:
      return __ESIMD_NS::fence_scope::system_acquire;
#else // legacy experimental api
    case gpu::xetla::fence_scope::group:
      return __ESIMD_ENS::lsc_scope::group;
    case gpu::xetla::fence_scope::local:
      return __ESIMD_ENS::lsc_scope::local;
    case gpu::xetla::fence_scope::tile: //
      return __ESIMD_ENS::lsc_scope::tile;
    case gpu::xetla::fence_scope::gpu: //
      return __ESIMD_ENS::lsc_scope::gpu;
    case gpu::xetla::fence_scope::gpus: //
      return __ESIMD_ENS::lsc_scope::gpus;
    case gpu::xetla::fence_scope::system:
      return __ESIMD_ENS::lsc_scope::system;
    case gpu::xetla::fence_scope::sysacq:
      return __ESIMD_ENS::lsc_scope::sysacq;
#endif
  }
}

/// @brief lookup table for atomic op.
///
///
constexpr __ESIMD_NS::atomic_op get_atomic_op(gpu::xetla::atomic_op ao) {
  switch (ao) {
    case gpu::xetla::atomic_op::iinc:
      return __ESIMD_NS::atomic_op::inc;
    case gpu::xetla::atomic_op::idec:
      return __ESIMD_NS::atomic_op::dec;
    case gpu::xetla::atomic_op::iadd:
      return __ESIMD_NS::atomic_op::add;
    case gpu::xetla::atomic_op::isub:
      return __ESIMD_NS::atomic_op::sub;
    case gpu::xetla::atomic_op::smin:
      return __ESIMD_NS::atomic_op::smin;
    case gpu::xetla::atomic_op::smax:
      return __ESIMD_NS::atomic_op::smax;
    case gpu::xetla::atomic_op::umin:
      return __ESIMD_NS::atomic_op::umin;
    case gpu::xetla::atomic_op::umax:
      return __ESIMD_NS::atomic_op::umax;
    case gpu::xetla::atomic_op::cmpxchg:
      return __ESIMD_NS::atomic_op::cmpxchg;
    case gpu::xetla::atomic_op::fadd:
      return __ESIMD_NS::atomic_op::fadd;
    case gpu::xetla::atomic_op::fsub:
      return __ESIMD_NS::atomic_op::fsub;
    case gpu::xetla::atomic_op::fmin:
      return __ESIMD_NS::atomic_op::fmin;
    case gpu::xetla::atomic_op::fmax:
      return __ESIMD_NS::atomic_op::fmax;
    case gpu::xetla::atomic_op::fcmpxchg:
      return __ESIMD_NS::atomic_op::fcmpxchg;
    case gpu::xetla::atomic_op::bit_and:
      return __ESIMD_NS::atomic_op::bit_and;
    case gpu::xetla::atomic_op::bit_or:
      return __ESIMD_NS::atomic_op::bit_or;
    case gpu::xetla::atomic_op::bit_xor:
      return __ESIMD_NS::atomic_op::bit_xor;
    case gpu::xetla::atomic_op::load:
      return __ESIMD_NS::atomic_op::load;
    case gpu::xetla::atomic_op::store:
      return __ESIMD_NS::atomic_op::store;
  }
}
} // namespace detail

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-pf-1)
/// void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (usm-pf-2)
///
/// The next 2 functions are similar to the above and were added for
/// convenience. They assume the VS parameter is set to 1 and do not require
/// specifying the template parameters <T, N, VS> at function calls.
/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask,
///                   PropertyListT props = {});                   // (usm-pf-3)
/// void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (usm-pf-4)
/// The next 2 functions are variations of the first 2 above (usm-pf-1,2)
/// and were added only to support simd_view instead of simd for byte_offsets
/// operand.
/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-pf-5)
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             PropertyListT props = {});                        // (usm-pf-6)
///
/// The next functions perform transposed 1-channel prefetch.
/// template <typename T, int VS = 1, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetT byte_offset, simd_mask<1> mask,
///                   PropertyListT props = {});                   // (usm-pf-7)
/// void prefetch(const T *p, OffsetT byte_offset,
///                   PropertyListT props = {});                   // (usm-pf-8)
/// template <typename T, int VS = 1,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd_mask<1> mask,
///                   PropertyListT props = {});                   // (usm-pf-9)
/// void prefetch(const T *p, PropertyListT props = {});           //(usm-pf-10)
///

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-pf-1)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, to the cache.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the prefetch from
/// (p + byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T,
    int N,
    int VS,
    cache_hint L1H = cache_hint::cached,
    cache_hint L2H = cache_hint::cached>
__XETLA_API void xetla_prefetch_global(
    T* p,
    xetla_vector<uint32_t, N / VS> byte_offsets,
    xetla_mask<N / VS> mask = 1) {
#if __INTEL_LLVM_COMPILER >= 20240200
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>};
  __ESIMD_NS::prefetch<T, N, VS>(p, byte_offsets, mask, props);
#else
  constexpr data_size DS = data_size::default_size;
  __ESIMD_ENS::lsc_prefetch<
      T,
      VS,
      gpu::xetla::detail::get_data_size(DS),
      gpu::xetla::detail::get_cache_hint(L1H),
      gpu::xetla::detail::get_cache_hint(L2H),
      N / VS>(p, byte_offsets, mask);
#endif
}

/// template <typename T, int VS = 1, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetT byte_offset,
///                   PropertyListT props = {});                   // (usm-pf-8)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from continuous memory location
/// addressed by the base pointer \p p, and offset \p byte_offset and the length
/// \p VS elements into the cache.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param p The base address.
/// @param byte_offset offset from the base address
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T,
    int VS = 1,
    cache_hint L1H = cache_hint::cached,
    cache_hint L2H = cache_hint::cached>
__XETLA_API void xetla_prefetch_global(T* p, uint64_t byte_offset = 0) {
#if __INTEL_LLVM_COMPILER >= 20240200
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>};
  __ESIMD_NS::prefetch<T, VS>(p, byte_offset, props);
#else
  constexpr data_size DS = data_size::default_size;
  __ESIMD_ENS::lsc_prefetch<
      T,
      VS,
      gpu::xetla::detail::get_data_size(DS),
      gpu::xetla::detail::get_cache_hint(L1H),
      gpu::xetla::detail::get_cache_hint(L2H),
      1>(p, (byte_offset / sizeof(T)));
#endif
}

/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       props={});  // (usm-bl-2)
/// This function loads a contiguous memory block from address referenced
/// by USM pointer \p ptr and the given \p byte_offset.
///
/// There may be temporary restrictions depending on L1, L2 cache hints,
/// See details in the 'Restrictions' section below. The restrictions will be
/// relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is 4-bytes for 4-byte or smaller elements
/// and 8-bytes for 8-byte elements. The address may be element-size aligned
/// even for byte- and word-elements, but in such case the smaller alignment
/// property must explicitly passed to this function. Extra restrictions
/// may be in place - see Restrictions/R1 below.
///
/// Restrictions - cache hint imposed - temporary:
/// If L1 or L2 cache hint is passed, then:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <
    typename T,
    int N,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none,
    int alignment = 16>
__XETLA_API xetla_vector<T, N> xetla_load_global(
    const T* ptr,
    size_t byte_offset) {
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>,
      __ESIMD_NS::alignment<alignment>};
  if constexpr (sizeof(T) * N < sizeof(uint32_t) || N == 1) {
    xetla_vector<T, N> ret;
#pragma unroll
    for (uint32_t i = 0; i < N; i++) {
      ret[i] = ptr[i + byte_offset / sizeof(T)];
    }
    return ret;
  } else {
    return __ESIMD_NS::block_load<T, N>(ptr, byte_offset, props);
  }
}
/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (usm-ga-1)
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-ga-2)
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (usm-ga-3)
///
/// The next 3 functions are similar to the above and were added for
/// convenience. They assume the VS parameter is set to 1 and do not require
/// specifying the template parameters <T, N, VS> at function calls.
/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (usm-ga-4)
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props = {});// (usm-ga-5)
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (usm-ga-6)
///
/// The next 3 functions are variations of the first 3 above (usm-ga-1,2,3)
/// and were added only to support simd_view instead of simd for byte_offsets
/// and/or pass_thru operands.
/// template <typename T, int N, int VS = 1, typename OffsetObjT,
///           typename OffsetRegionT, typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});                         // (usm-ga-7)
/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-ga-8)
/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             PropertyListT props = {});                         // (usm-ga-9)
#ifndef __ESIMD_GATHER_SCATTER_LLVM_IR
/// Supported platforms: DG2, PVC only - Temporary restriction for the variant
/// with pass_thru operand.
#endif // __ESIMD_GATHER_SCATTER_LLVM_IR
/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <
    typename T,
    int N,
    int VS,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none,
    int alignment = 16,
    typename OffsetT = uint32_t>
__XETLA_API xetla_vector<T, N> xetla_load_global(
    T* p,
    xetla_vector<OffsetT, N / VS> byte_offsets,
    xetla_mask<N / VS> mask,
    xetla_vector<T, N> pass_thru) {
#if __INTEL_LLVM_COMPILER >= 20240200
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>,
      __ESIMD_NS::alignment<alignment>};

  return __ESIMD_NS::gather<T, N, VS>(p, byte_offsets, mask, pass_thru, props);
#else
  constexpr data_size DS = data_size::default_size;
  return __ESIMD_ENS::lsc_gather<
      T,
      VS,
      gpu::xetla::detail::get_data_size(DS),
      gpu::xetla::detail::get_cache_hint(L1H),
      gpu::xetla::detail::get_cache_hint(L2H),
      N / VS>(p, byte_offsets, mask, pass_thru);
#endif
}

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-ga-2)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element of the
/// returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
template <
    typename T,
    int N,
    int VS,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none,
    int alignment = 16,
    typename OffsetT = uint32_t>
__XETLA_API xetla_vector<T, N> xetla_load_global(
    T* p,
    xetla_vector<OffsetT, N / VS> byte_offsets,
    xetla_mask<N / VS> mask = 1) {
#if __INTEL_LLVM_COMPILER >= 20240200
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>,
      __ESIMD_NS::alignment<alignment>};
  return __ESIMD_NS::gather<T, N, VS>(p, byte_offsets, mask, props);
#else
  constexpr data_size DS = data_size::default_size;
  return __ESIMD_ENS::lsc_gather<
      T,
      VS,
      gpu::xetla::detail::get_data_size(DS),
      gpu::xetla::detail::get_cache_hint(L1H),
      gpu::xetla::detail::get_cache_hint(L2H),
      N / VS>(p, byte_offsets, mask);
#endif
}

/// template <typename T, int N, int VS = 1, typename OffsetT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-sc-1)

/// template <typename T, int N, int VS = 1, typename OffsetT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
/// 	PropertyListT props = {});                         // (usm-sc-2)

/// The next two functions are similar to usm-sc-{1,2} with the 'byte_offsets'
/// parameter represerented as 'simd_view'.

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-sc-3)

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
///      PropertyListT props = {});                         // (usm-sc-4)

/// template <typename T, int N, int VS = 1, typename OffsetT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-sc-1)
///
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T,
    int N,
    int VS = 1,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none,
    int alignment = 16,
    typename OffsetT = uint32_t>
__XETLA_API void xetla_store_global(
    T* p,
    xetla_vector<OffsetT, N / VS> byte_offsets,
    xetla_vector<T, N> vals,
    xetla_mask<N / VS> mask = 1) {
#if __INTEL_LLVM_COMPILER >= 20240200
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>,
      __ESIMD_NS::alignment<alignment>};
  __ESIMD_NS::scatter<T, N, VS>(p, byte_offsets, vals, mask, props);
#else
  constexpr data_size DS = data_size::default_size;
  __ESIMD_ENS::lsc_scatter<
      T,
      VS,
      gpu::xetla::detail::get_data_size(DS),
      gpu::xetla::detail::get_cache_hint(L1H),
      gpu::xetla::detail::get_cache_hint(L2H),
      N / VS>((T*)p, byte_offsets, vals, mask);
#endif
}

/// void block_store(T* ptr, size_t byte_offset,         // (usm-bs-2)
///                          simd<T, N> vals, props={});
/// This function stores a contiguous memory block to USM pointer \p ptr and
/// byte-offset \p byte_offset with data specified by \p vals.
///
/// There may be temporary restrictions depending on L1, L2 cache hints,
/// See details in the 'Restrictions' section below. The restrictions will be
/// relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is 16 bytes if \p props does not specify any
/// L1 or L2 cache hints, and the minimally required element-size
/// alignment otherwise. Note that additional/temporary restrictions may apply
/// (see Restrictions below).
///
/// Restrictions - cache hint imposed - temporary:
/// If L1 or L2 cache hint is passed, then:
/// R1: The pointer plus byte offset must be at least 4-byte aligned for
/// elements of 4-bytes or smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <
    typename T,
    int N,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none,
    int alignment = 16>
__XETLA_API void xetla_store_global(
    T* ptr,
    size_t byte_offset,
    xetla_vector<T, N> vals) {
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>,
      __ESIMD_NS::alignment<alignment>};

  if constexpr (sizeof(T) * N < sizeof(uint32_t) || N == 1) {
#pragma unroll
    for (uint32_t i = 0; i < N; i++) {
      ptr[i + byte_offset / sizeof(T)] = vals[i];
    }
  } else {
    __ESIMD_NS::block_store<T, N>(ptr, byte_offset, vals, props);
  }
}

/// @addtogroup sycl_esimd_memory_atomics
/// @{

/// @anchor usm_atomic_update0
/// @brief No-argument variant of the atomic update operation.
///
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-1)
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               props = {});                                  /// (usm-au0-2)
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, RegionTy> byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-3)
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, RegionTy> byte_offset,
///               props = {});                                  /// (usm-au0-4)
///
/// Usage of cache hints or non-standard operation width N requires DG2 or PVC.
///
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-1)
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has no arguments in addition to the value at the memory location.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
///   \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes
///  (zero-based).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <
    atomic_op Op,
    typename T,
    int N,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(
    T* p,
    xetla_vector<uint32_t, N> offsets,
    xetla_mask<N> mask) {
  static_assert(
      !(is_internal_type<T>::value),
      "The internal types are not yet supported!");

  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>};
  return __ESIMD_NS::atomic_update<gpu::xetla::detail::get_atomic_op(Op), T, N>(
      p, offsets, mask, props);
}

/// @anchor usm_atomic_update1
/// @brief Single-argument variant of the atomic update operation.
///
/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd_mask<N> mask, props = {});//(usm-au1-1)
/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, props = {});                  // (usm-au1-2)
///
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               simd_mask<N> mask, props = {});                // (usm-au1-3)
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               props = {});                                   // (usm-au1-4)
///

/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd_mask<N> mask, props = {});//(usm-au1-1)
///
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 1 additional argument.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::fadd, \c
/// atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <
    atomic_op Op,
    typename T,
    int N,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(
    T* p,
    xetla_vector<uint32_t, N> offsets,
    xetla_vector<T, N> src0,
    xetla_mask<N> mask) {
  static_assert(
      !(is_internal_type<T>::value),
      "The internal types are not yet supported!");
  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>};
  return __ESIMD_NS::atomic_update<gpu::xetla::detail::get_atomic_op(Op), T, N>(
      p, offsets, src0, mask, props);
}

/// @anchor usm_atomic_update2
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});               // (usm-au2-1)
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {});                                  // (usm-au2-2)
///
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {})                // (usm-au2-3)
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {})                                   // (usm-au2-4)
///

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});               // (usm-au2-1)
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <
    atomic_op Op,
    typename T,
    int N,
    cache_hint L1H = cache_hint::none,
    cache_hint L2H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(
    T* p,
    xetla_vector<uint32_t, N> offsets,
    xetla_vector<T, N> src0,
    xetla_vector<T, N> src1,
    xetla_mask<N> mask) {
  static_assert(
      !(is_internal_type<T>::value),
      "The internal types are not yet supported!");

  __ESIMD_NS::properties props{
      __ESIMD_NS::cache_hint_L1<gpu::xetla::detail::get_cache_hint(L1H)>,
      __ESIMD_NS::cache_hint_L2<gpu::xetla::detail::get_cache_hint(L2H)>};

  // 2-argument xetla_atomic_global arguments order matches the standard one -
  // expected value first, then new value. But atomic_update uses reverse
  // order, hence the src1/src0 swap.
  return __ESIMD_NS::atomic_update<gpu::xetla::detail::get_atomic_op(Op), T, N>(
      p, offsets, src1, src0, mask, props);
}
/// @brief Declare per-work-group slm size.
/// @tparam SLMSize  Shared Local Memory (SLM) size (in Bytes).
template <uint32_t SLMSize>
__XETLA_API void xetla_local_init() {
  if constexpr (SLMSize != 0) {
    __ESIMD_NS::slm_init(SLMSize);
  }
}

/// @brief SLM scattered load.
/// Collects elements located at slm and returns them as a single \ref
/// xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e.
/// vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param offsets [in] is the zero-based offsets for SLM buffer in bytes.
/// @param pred    [in] is predicates.
/// @return is a xetla_vector of type T and size N * NElts.
///
template <
    typename Ty,
    int NElts = 1,
    data_size DS = data_size::default_size,
    int N>
__XETLA_API xetla_vector<Ty, N * NElts> xetla_load_local(
    xetla_vector<uint32_t, N> offsets,
    xetla_mask<N> pred = 1) {
  using T = native_type_t<Ty>;

  return __ESIMD_ENS::
      lsc_slm_gather<T, NElts, gpu::xetla::detail::get_data_size(DS), N>(
          xetla_cvt<uint64_t, uint32_t>(offsets), pred);
}

/// Loads a contiguous block of SLM memory referenced by the given byte-offset
/// \p offset, then returns the loaded data as a simd object.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient gather is generated. If the loaded vector is too long
/// for 1 flat-load GPU instruction, then a series of flat-loads and/or gathers
/// may be generated.
/// @tparam T Element type.
/// @tparam N Number of elements to load.
/// @tparam Flags The alignment specifier type tag.
/// @param byte_offset The byte-offset to load from.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename Ty, int NElts = 1, data_size DS = data_size::default_size>
__XETLA_API xetla_vector<Ty, NElts> xetla_load_local(uint32_t offset) {
  using T = native_type_t<Ty>;

  return __ESIMD_NS::slm_block_load<T, NElts>(offset);
}

/// @brief SLM scattered store.
/// Scatters elements located to slm.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e.
/// vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param offsets [in] is the zero-based offsets for SLM buffer in bytes.
/// @param vals    [in] is values to store.
/// @param pred    [in] is predicates.
///
template <
    typename Ty,
    int NElts = 1,
    data_size DS = data_size::default_size,
    int N>
__XETLA_API void xetla_store_local(
    xetla_vector<uint32_t, N> offsets,
    xetla_vector<Ty, N * NElts> vals,
    xetla_mask<N> pred = 1) {
  using T = native_type_t<Ty>;

  __ESIMD_ENS::
      lsc_slm_scatter<T, NElts, gpu::xetla::detail::get_data_size(DS), N>(
          offsets, vals, pred);
}

/// Stores elements of the vector \p vals to a contiguous block of SLM memory
/// at the given byte-offset \p offset.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient scatter is generated. If the stored vector is too long
/// for 1 flat-store GPU instruction, then a series of flat-store and/or
/// scatters may be generated.
/// @tparam T Element type.
/// @tparam N Number of elements to store.
/// @tparam Flags The alignment specifier type tag.
/// @param offset The byte-offset to store at.
/// @param vals The vector to store.
/// @param Flags Specifies the alignment.
///
template <typename Ty, int NElts = 1, data_size DS = data_size::default_size>
__XETLA_API void xetla_store_local(
    uint32_t offset,
    xetla_vector<Ty, NElts> vals) {
  __ESIMD_NS::slm_block_store<Ty, NElts>(offset, vals);
}

/// @brief SLM scattered atomic (0 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param pred    [in] is predicates.
///
template <
    atomic_op Op,
    typename T,
    int N,
    data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
    xetla_vector<uint32_t, N> offsets,
    xetla_mask<N> pred) {
  static_assert(
      !(is_internal_type<T>::value),
      "The internal types are not yet supported!");
  return __ESIMD_ENS::lsc_slm_atomic_update<
      gpu::xetla::detail::get_atomic_op(Op),
      T,
      N,
      gpu::xetla::detail::get_data_size(DS)>(offsets, pred);
}

/// @brief SLM scattered atomic (1 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param pred    [in] is predicates.
///
template <
    atomic_op Op,
    typename T,
    int N,
    data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
    xetla_vector<uint32_t, N> offsets,
    xetla_vector<T, N> src0,
    xetla_mask<N> pred) {
  static_assert(
      !(is_internal_type<T>::value),
      "The internal types are not yet supported!");
  return __ESIMD_ENS::lsc_slm_atomic_update<
      gpu::xetla::detail::get_atomic_op(Op),
      T,
      N,
      gpu::xetla::detail::get_data_size(DS)>(offsets, src0, pred);
}

/// @brief SLM scattered atomic (2 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param src1    [in] is the second atomic operand.
/// @param pred    [in] is predicates.
///
template <
    atomic_op Op,
    typename T,
    int N,
    data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
    xetla_vector<uint32_t, N> offsets,
    xetla_vector<T, N> src0,
    xetla_vector<T, N> src1,
    xetla_mask<N> pred) {
  static_assert(
      !(is_internal_type<T>::value),
      "The internal types are not yet supported!");
  return __ESIMD_ENS::lsc_slm_atomic_update<
      gpu::xetla::detail::get_atomic_op(Op),
      T,
      N,
      gpu::xetla::detail::get_data_size(DS)>(offsets, src0, src1, pred);
}

/// @brief Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param pred is predicates.
template <
    memory_kind Kind = memory_kind::untyped_global,
    fence_op FenceOp = fence_op::none,
    fence_scope Scope = fence_scope::group,
    int N = 16>
__XETLA_API void xetla_fence() {
#if __INTEL_LLVM_COMPILER >= 20240100
  __ESIMD_NS::fence<
      gpu::xetla::detail::get_memory_kind(Kind),
      gpu::xetla::detail::get_fence_op(FenceOp),
      gpu::xetla::detail::get_fence_scope(Scope)>();
#else
  __ESIMD_ENS::lsc_fence<
      gpu::xetla::detail::get_memory_kind(Kind),
      gpu::xetla::detail::get_fence_op(FenceOp),
      gpu::xetla::detail::get_fence_scope(Scope),
      N>(xetla_mask<N>(1));
#endif
}

/// @} xetla_core_memory

} // namespace gpu::xetla
