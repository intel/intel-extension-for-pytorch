#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>
#include <type_traits>

namespace esimd {

//
// Arch
//

inline void SW_BARRIER() {
#if __INTEL_LLVM_COMPILER >= 20250000
#if defined(__SYCL_DEVICE_ONLY__)
  __asm__ volatile("fence_sw" : : :);
#endif // __SYCL_DEVICE_ONLY__
#else
  __ESIMD_NS::fence<__ESIMD_NS::fence_mask::sw_barrier>();
#endif // __INTEL_LLVM_COMPILER >= 20250000
}

struct ArchConfig {
  static constexpr uint32_t kMaxLoadBytes = 256;
  static constexpr uint32_t kMaxStoreBytes = 256;
  static constexpr uint32_t kMaxVecSizeForSLMLoadStore = 64;
  static constexpr uint32_t kCachelineSize = 64; // bytes
};

template <uint32_t kMax, uint32_t kCurVal = 2>
inline constexpr uint32_t get_max_power_of_2() {
  static_assert(kCurVal < kMax, "kCurVal is too large");
  constexpr uint32_t kDoubleVal = kCurVal * 2;
  if constexpr (kDoubleVal < kMax) {
    return get_max_power_of_2<kMax, kDoubleVal>();
  } else if (kDoubleVal == kMax) {
    return kDoubleVal;
  } else {
    return kCurVal;
  }
}

//
// Alias
//

template <class T, int N>
using Vec = __ESIMD_NS::simd<T, N>;
template <int N>
using Mask = __ESIMD_NS::simd_mask<N>;

//
// Memory
//

template <class T, int kNumElems>
inline void prefetch_global(T* p) {
  __ESIMD_ENS::lsc_prefetch<
      T,
      kNumElems,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>(p);
}

template <class T, int kNumElems, int kNumSg>
inline void cooperative_prefetch_global(T* p, uint64_t offset, uint32_t sid) {
  constexpr uint32_t kMinElems = ArchConfig::kCachelineSize / sizeof(T);

  for (int i = sid * kMinElems; i < kNumElems; i += kNumSg * kMinElems) {
    __ESIMD_ENS::lsc_prefetch<
        uint64_t,
        kMinElems * sizeof(T) / sizeof(uint64_t),
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint64_t*)(p + offset + i));
  }
}

template <class T, int kNumElems>
inline Vec<T, kNumElems> load_global(T* p, uint64_t offset) {
  return __ESIMD_ENS::lsc_block_load<
      T,
      kNumElems,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>(p + offset);
}

template <class T, int kNumElems>
inline Vec<T, kNumElems> load_global_unaligned(T* p, uint64_t offset) {
  static_assert(sizeof(T) <= 4, "Only support types with size <= 4 bytes");
  constexpr int kMinAlignedBytes = 4;
  constexpr int kMinAligned = kMinAlignedBytes / sizeof(T);
  uint64_t aligned_offset =
      (offset * sizeof(T) / kMinAlignedBytes) * kMinAligned;
  auto vals = __ESIMD_ENS::lsc_block_load<
      T,
      kNumElems + kNumElems,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>(p + aligned_offset);
  return vals.template select<kNumElems, 1>(offset - aligned_offset);
}

template <class T, int kNumElems>
inline void store_global(T* p, uint64_t offset, Vec<T, kNumElems>& vals) {
  __ESIMD_ENS::lsc_block_store<
      T,
      kNumElems,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>(p + offset, vals);
}

template <class T, int kNumElems>
inline Vec<T, kNumElems> load_local(uint64_t offset) {
  return __ESIMD_ENS::lsc_slm_block_load<
      T,
      kNumElems,
      __ESIMD_ENS::lsc_data_size::default_size>(offset);
}

template <class T, int kNumElems, typename Arch = ArchConfig>
inline std::enable_if_t<(kNumElems <= Arch::kMaxVecSizeForSLMLoadStore), void>
store_local(uint32_t offset, Vec<T, kNumElems>& vals) {
  return __ESIMD_ENS::lsc_slm_block_store<
      T,
      kNumElems,
      __ESIMD_ENS::lsc_data_size::default_size>(offset, vals);
}

template <class T, int kNumElems, typename Arch = ArchConfig>
inline std::enable_if_t<
    (kNumElems > Arch::kMaxVecSizeForSLMLoadStore &&
     kNumElems % Arch::kMaxVecSizeForSLMLoadStore == 0),
    void>
store_local(uint32_t offset, Vec<T, kNumElems>& vals) {
  constexpr uint32_t MaxVecSize = Arch::kMaxVecSizeForSLMLoadStore;
  for (int i = 0; i < kNumElems / MaxVecSize; i++) {
    Vec<T, MaxVecSize> tmp =
        vals.template select<MaxVecSize, 1>(i * MaxVecSize);
    __ESIMD_ENS::lsc_slm_block_store<
        T,
        MaxVecSize,
        __ESIMD_ENS::lsc_data_size::default_size>(
        offset + i * MaxVecSize * sizeof(T), tmp);
  }
}

inline void barrier() {
  __ESIMD_NS::barrier();
}

//
// Math
//

// template <class T, int kNumElems>
// inline Vec<T, kNumElems> max(Vec<T, kNumElems> src0, Vec<T, kNumElems> src1)
// {
//   return __ESIMD_NS::max<T, kNumElems>(src0, src1);
// }

// template <class T, int kNumElems>
// inline Vec<T, kNumElems> exp(Vec<T, kNumElems> src) {
//   return __ESIMD_NS::exp<T, kNumElems>(src);
// }

//
// Reduce
//

enum class BinaryOp { MAX, SUM, OR };

namespace detail {

template <BinaryOp kOp, class T, int kNumElems>
inline typename std::enable_if_t<kOp == BinaryOp::SUM, Vec<T, kNumElems>>
reduce_impl(Vec<T, kNumElems> a, Vec<T, kNumElems> b) {
  return a + b;
}

template <BinaryOp kOp, class T, int kNumElems>
inline typename std::enable_if_t<kOp == BinaryOp::OR, Vec<T, kNumElems>>
reduce_impl(Vec<T, kNumElems> a, Vec<T, kNumElems> b) {
  return a | b;
}

template <BinaryOp kOp, class T, int kNumElems>
inline typename std::enable_if_t<kOp == BinaryOp::MAX, Vec<T, kNumElems>>
reduce_impl(Vec<T, kNumElems> a, Vec<T, kNumElems> b) {
  Vec<T, kNumElems> out;
  Mask<kNumElems> mask = a > b;
  out.merge(a, b, mask);
  return out;
}

} // namespace detail

template <BinaryOp kOp, class T, int kRow, int kCol>
inline typename std::enable_if_t<kCol == 1, Vec<T, kRow>> row_reduce(
    Vec<T, kRow> vec) {
  return vec;
}
template <BinaryOp kOp, class T, int kRow, int kCol>
inline typename std::enable_if_t<(kCol > 1), Vec<T, kRow>> row_reduce(
    Vec<T, kRow * kCol> vec) {
  static_assert(((kCol) & (kCol - 1)) == 0, "kCol should be power of 2");

  auto vec_2d = vec.template bit_cast_view<T, kRow, kCol>();

  auto tmp = detail::reduce_impl<kOp, T, kRow * kCol / 2>(
      vec_2d.template select<kRow, 1, kCol / 2, 1>(0, 0),
      vec_2d.template select<kRow, 1, kCol / 2, 1>(0, kCol / 2));

  return row_reduce<kOp, T, kRow, kCol / 2>(tmp);
}

template <BinaryOp kOp, class T, int kRow, int kCol>
inline typename std::enable_if_t<kRow == 1, Vec<T, kCol>> col_reduce(
    Vec<T, kCol> vec) {
  return vec;
}
template <BinaryOp kOp, class T, int kRow, int kCol>
inline typename std::enable_if_t<(kRow > 1), Vec<T, kCol>> col_reduce(
    Vec<T, kRow * kCol> vec) {
  static_assert(((kRow) & (kRow - 1)) == 0, "kRow should be power of 2");

  auto tmp = detail::reduce_impl<kOp, T, kRow * kCol / 2>(
      vec.template select<kRow * kCol / 2, 1>(0),
      vec.template select<kRow * kCol / 2, 1>(kRow * kCol / 2));

  return col_reduce<kOp, T, kRow / 2, kCol>(tmp);
}

} // namespace esimd
