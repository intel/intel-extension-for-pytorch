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

#include <CL/sycl.hpp>
#include <common/core/common_types.hpp>
#include <ext/intel/esimd.hpp>
#include <version.hpp>

template <class T>
using remove_const_t = typename std::remove_const<T>::type;

/// @addtogroup xetla_core
/// @{

/// @brief KERNEL_MAIN macro.
/// Alias to ESIMD `"SYCL_ESIMD_KERNEL"`.
///
#define KERNEL_MAIN SYCL_ESIMD_KERNEL

/// @brief KERNEL_FUNC macro.
/// Alias to ESIMD `"SYCL_ESIMD_FUNCTION"`.
///
#define KERNEL_FUNC SYCL_ESIMD_FUNCTION

/// @} xetla_core

#define __XETLA_API inline

#ifndef __ESIMD_ENS
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#endif

#ifndef __ESIMD_NS
#define __ESIMD_NS sycl::ext::intel::esimd
#endif

#define XETLA_MARKER(message) [[deprecated(message)]]
#define XETLA_WARNING(msg) __SYCL_WARNING(msg)

template <auto val>
XETLA_MARKER("Help function to print value")
inline constexpr void XETLA_PRINT() {}
template <typename type>
XETLA_MARKER("Help function to print type")
inline constexpr void XETLA_PRINT() {}

__XETLA_API int32_t xetla_get_hw_thread_id() {
  return __ESIMD_ENS::get_hw_thread_id();
}

__XETLA_API int32_t xetla_get_subdevice_id() {
  return __ESIMD_ENS::get_subdevice_id();
}

namespace gpu::xetla {
enum class mem_space : uint8_t { global = 0, local = 1 };
enum class msg_type : uint8_t {
  block_2d = 0,
  block_1d = 1,
  scatter = 2,
  atomic_add = 3,
  unaligned_2d = 4
  // prefetch_2d = 4,
  // prefetch_1d = 5
};

/// L1, L2 or L3 cache hints.
enum class cache_hint : uint8_t {
  none = 0,
  /// load/store/atomic: do not cache data to cache;
  uncached = 1,

  // load: cache data to cache;
  cached = 2,

  /// store: write data into cache level and mark the cache line as "dirty".
  /// Upon eviction, the "dirty" data will be written into the furthest
  /// subsequent cache;
  write_back = 3,

  /// store: immediately write data to the subsequent furthest cache, marking
  /// the cache line in the current cache as "not dirty";
  write_through = 4,

  /// load: cache data to cache using the evict-first policy to minimize cache
  /// pollution caused by temporary streaming data that may only be accessed
  /// once or twice;
  /// store/atomic: same as write-through, but use the evict-first policy
  /// to limit cache pollution by streaming;
  streaming = 5,

  /// load: asserts that the cache line containing the data will not be read
  /// again until it’s overwritten, therefore the load operation can invalidate
  /// the cache line and discard "dirty" data. If the assertion is violated
  /// (the cache line is read again) then behavior is undefined.
  read_invalidate = 6,

  // TODO: Implement the verification of this enum in check_cache_hint().
  /// load, L2 cache only, next gen GPU after Xe required: asserts that
  /// the L2 cache line containing the data will not be written until all
  /// invocations of the shader or kernel execution are finished.
  /// If the assertion is violated (the cache line is written), the behavior
  /// is undefined.
  const_cached = 7
};

/// Data size or format to read or store
enum class data_size : uint8_t {
  default_size = 0,
  u8 = 1,
  u16 = 2,
  u32 = 3,
  u64 = 4,
  u8u32 = 5, /// load 8b, zero extend to 32b; store the opposite
  u16u32 = 6, /// load 16b, zero extend to 32b; store the opposite
  u16u32h = 7, /// load 16b into high 16 of each 32b; store the high 16
};

/// The specific LSC shared function to fence with xetla_fence
enum class memory_kind : uint8_t {
  untyped_global = 0, /// untyped global memory
  // "1" reserved for low-priority untyped global memory
  typed_global = 2, /// typed global memory
  shared_local = 3, /// shared local memory
};

/// The xetla_fence operation to apply to caches
enum class fence_op : uint8_t {
  none = 0, /// no operation
  evict = 1, /// dirty lines evicted and invalidated from L1
  invalidate = 2, /// invalidate all clean lines

  clean = 4, /// dirty lines are written to memory, but retained in cache
};
/// The scope that xetla_fence operation should apply to
enum class fence_scope : uint8_t {
  group = 0, /// flush out to the threadgroup's scope
  local = 1, /// flush out to the local scope
  tile = 2, /// tile, flush out to several DSSs
  gpu = 3, /// entire GPU, flush out to the GPUs LLC
  gpus = 4, /// all GPUs in the system, flush out to memory shared by all GPUs
  system = 5, /// the entire system memory space
  sysacq = 6, /// the entire system memory space with system-acquire semantics
};

/// Represents an atomic operation. Operations always return the old value(s) of
/// the target memory location(s) as it was before the operation was applied.
enum class atomic_op : uint8_t {
  /// Atomic increment of memory data and return the old value.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53538)
  iinc = 0x0,
  /// Atomic decrement of memory data and return the old value.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53539)
  idec = 0x1,
  /// Atomic signed int add of src1 from memory data and return the old value.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53542)
  iadd = 0x2,
  /// Atomic signed int subtract of src1 from memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53543)
  isub = 0x3,
  /// Atomic store the signed int min of src1 and memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53544)
  smin = 0x4,
  /// Atomic store the signed int max of src1 and memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53545)
  smax = 0x5,
  /// Atomic bit-compare src1_X and memory data and replace if equal with
  /// src1_Y. Returns the old value.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53555)
  cmpxchg = 0x6,
  /// Atomic float add of src1 from memory data and return the old value.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53548)
  fadd = 0x7,
  /// Atomic float subtract of src1 from memory data and return the old value.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53549)
  fsub = 0x8,
  /// Atomic store the float min of src1 and memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53550)
  fmin = 0x9,
  /// Atomic store the float max of src1 and memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53551)
  fmax = 0xa,
  /// Atomic float compare src1_X and memory data and replace if equal with
  /// src1_Y. Returns the old value.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53556)
  fcmpxchg = 0xb,
  /// Atomic store the unsigned int min of src1 and memory data and return the
  /// old value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53546)
  umin = 0xc,
  /// Atomic store the unsigned int max of src1 and memory data and return the
  /// old value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53547)
  umax = 0xd,
  /// Atomic store the bitwise AND of src1 and memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53552)
  bit_and = 0xe,
  /// Atomic store the bitwise OR of src1 and memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53553)
  bit_or = 0xf,
  /// Atomic store the bitwise XOR of src1 and memory data and return the old
  /// value. [see](https://gfxspecs.intel.com/Predator/Home/Index/53554)
  bit_xor = 0x10,
  /// Atomic read of the memory data value, without modifying the data.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53540)
  load = 0x11,
  /// Atomic store untyped data to memory.
  /// [see](https://gfxspecs.intel.com/Predator/Home/Index/53541)
  store = 0x12
};

/// xetla dpas argument typ
enum class argument_type : uint8_t {
  Invalid = 0,
#if __INTEL_LLVM_COMPILER >= 20240200
  U1 __SYCL_DEPRECATED("u1 is reserved/unsupported") = 1, // unsigned 1 bit
  S1 __SYCL_DEPRECATED("s1 is reserved/unsupported") = 2, // signed 1 bit
#else
  U1 = 1, // unsigned 1 bit
  S1 = 2, // signed 1 bit
#endif
  U2 = 3, // unsigned 2 bits
  S2 = 4, // signed 2 bits
  U4 = 5, // unsigned 4 bits
  S4 = 6, // signed 4 bits
  U8 = 7, // unsigned 8 bits
  S8 = 8, // signed 8 bits
  BF16 = 9, // bfloat 16
  FP16 = 10, // half float
  TF32 = 12, // tensorfloat 32
  DF = 13, // double (64bits)
  NUM_ARG_TYPES = 14
};

// Saturation tag
class xetla_saturation_on_tag {
 public:
  using sat_tag = typename __ESIMD_NS::saturation_on_tag;
  static constexpr sat_tag value = {};
};

class xetla_saturation_off_tag {
 public:
  using sat_tag = typename __ESIMD_NS::saturation_off_tag;
  static constexpr sat_tag value = {};
};

template <typename T>
using is_xetla_scalar = typename __ESIMD_DNS::is_esimd_scalar<T>;

/// xetla reduce op
enum class reduce_op : uint8_t {
  sum = 0, // performance reduce_sum
  prod = 1, // performance reduce_prod
  min = 2, // performance reduce_min
  max = 3, // performance reduce_max
};

/// SW_BARRIER, insert software scheduling barrier, for better code control
///

#define SW_BARRIER() __ESIMD_NS::fence<__ESIMD_NS::fence_mask::sw_barrier>()

__XETLA_API void xetla_wait(uint16_t val) {
  __ESIMD_ENS::wait(__ESIMD_NS::simd<uint16_t, 1>(val));
}

} // namespace gpu::xetla
