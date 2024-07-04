/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
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

#pragma once

#include <sycl/sycl.hpp>
#include "ds_kernel_utils.h"

/////////////////////////////// Memory Access Utils
//////////////////////////////////
namespace mem_access {

enum class LoadPolicy {
  CacheAll, // Cache at all levels
  CacheGlobal, // Cache at L2 only
  CacheStreaming // Cache with evict first policy
};

enum class StorePolicy {
  Writeback, // Cache in L1, write-back on eviction
  CacheGlobal, // Bypass L1, write-back on eviction
  CacheStreaming // Allocate cache line with evict first policy
};

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
DS_D_INLINE void load_global(void* dst, const void* src);

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
DS_D_INLINE void load_global(void* dst, const void* src, bool do_access);

// Shared accesses have no cache policy
template <int AccessSize>
DS_D_INLINE void load_shared(void* dst, const void* src);

template <int AccessSize>
DS_D_INLINE void load_shared(void* dst, const void* src, bool do_access);

template <int AccessSize, StorePolicy policy = StorePolicy::Writeback>
DS_D_INLINE void store_global(void* dst, const void* src);

// Shared accesses have no cache policy
template <int AccessSize>
DS_D_INLINE void store_shared(void* dst, const void* src);

// Util for tracking pipeline buffers
template <int max>
class BufferTracker {
 public:
  int current_state;

  DS_D_INLINE BufferTracker() : current_state(0) {}

  DS_D_INLINE int get() {
    int return_val = current_state++;
    current_state = (current_state == max ? 0 : current_state);
    return return_val;
  }
};

DS_D_INLINE uint32_t lane_id() {
  return sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_id(2) &
      (sycl::ext::oneapi::experimental::this_sub_group().get_local_range().get(
           0) -
       1); // Portable
}

/////////// Load Global ///////////
template <>
DS_D_INLINE void load_global<16>(void* dst, const void* src) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<16>(void* dst, const void* src, bool do_access) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
    data[0].z() = 0;
    data[0].w() = 0;
  }
}

template <>
DS_D_INLINE void load_global<16, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<16, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src,
    bool do_access) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
    data[0].z() = 0;
    data[0].w() = 0;
  }
}

template <>
DS_D_INLINE void load_global<16, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<16, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src,
    bool do_access) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
    data[0].z() = 0;
    data[0].w() = 0;
  }
}

template <>
DS_D_INLINE void load_global<8>(void* dst, const void* src) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<8>(void* dst, const void* src, bool do_access) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
  }
}

template <>
DS_D_INLINE void load_global<8, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<8, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src,
    bool do_access) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
  }
}

template <>
DS_D_INLINE void load_global<8, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<8, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src,
    bool do_access) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
  }
}

template <>
DS_D_INLINE void load_global<4>(void* dst, const void* src) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<4>(void* dst, const void* src, bool do_access) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0] = 0;
  }
}

template <>
DS_D_INLINE void load_global<4, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<4, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src,
    bool do_access) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0] = 0;
  }
}

template <>
DS_D_INLINE void load_global<4, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<4, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src,
    bool do_access) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0] = 0;
  }
}

template <>
DS_D_INLINE void load_global<2>(void* dst, const void* src) {
  int16_t* data = reinterpret_cast<int16_t*>(dst);
  const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<2>(void* dst, const void* src, bool do_access) {
  int16_t* data = reinterpret_cast<int16_t*>(dst);
  const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0] = 0;
  }
}

template <>
DS_D_INLINE void load_global<2, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src) {
  int16_t* data = reinterpret_cast<int16_t*>(dst);
  const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<2, LoadPolicy::CacheGlobal>(
    void* dst,
    const void* src,
    bool do_access) {
  int16_t* data = reinterpret_cast<int16_t*>(dst);
  const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0] = 0;
  }
}

template <>
DS_D_INLINE void load_global<2, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src) {
  int16_t* data = reinterpret_cast<int16_t*>(dst);
  const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_global<2, LoadPolicy::CacheStreaming>(
    void* dst,
    const void* src,
    bool do_access) {
  int16_t* data = reinterpret_cast<int16_t*>(dst);
  const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0] = 0;
  }
}

/////////// Load Shared ///////////

template <>
DS_D_INLINE void load_shared<16>(void* dst, const void* src) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_shared<16>(void* dst, const void* src, bool do_access) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
    data[0].z() = 0;
    data[0].w() = 0;
  }
}

template <>
DS_D_INLINE void load_shared<8>(void* dst, const void* src) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_shared<8>(void* dst, const void* src, bool do_access) {
  sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
  const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
  }
}

template <>
DS_D_INLINE void load_shared<4>(void* dst, const void* src) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  data[0] = src_cast[0];
}

template <>
DS_D_INLINE void load_shared<4>(void* dst, const void* src, bool do_access) {
  int32_t* data = reinterpret_cast<int32_t*>(dst);
  const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0] = 0;
  }
}

/////////// Store Global ///////////

template <>
DS_D_INLINE void store_global<16>(void* dst, const void* src) {
  const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
  sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<16, StorePolicy::CacheGlobal>(
    void* dst,
    const void* src) {
  const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
  sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<16, StorePolicy::CacheStreaming>(
    void* dst,
    const void* src) {
  const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
  sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<8>(void* dst, const void* src) {
  const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
  sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<8, StorePolicy::CacheGlobal>(
    void* dst,
    const void* src) {
  const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
  sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<8, StorePolicy::CacheStreaming>(
    void* dst,
    const void* src) {
  const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
  sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<4>(void* dst, const void* src) {
  const int32_t* data = reinterpret_cast<const int32_t*>(src);
  int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<4, StorePolicy::CacheGlobal>(
    void* dst,
    const void* src) {
  const int32_t* data = reinterpret_cast<const int32_t*>(src);
  int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_global<4, StorePolicy::CacheStreaming>(
    void* dst,
    const void* src) {
  const int32_t* data = reinterpret_cast<const int32_t*>(src);
  int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
  dst_cast[0] = data[0];
}

/////////// Store Shared ///////////

template <>
DS_D_INLINE void store_shared<16>(void* dst, const void* src) {
  const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
  sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_shared<8>(void* dst, const void* src) {
  const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
  sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
  dst_cast[0] = data[0];
}

template <>
DS_D_INLINE void store_shared<4>(void* dst, const void* src) {
  const int32_t* data = reinterpret_cast<const int32_t*>(src);
  int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
  dst_cast[0] = data[0];
}

} // namespace mem_access
