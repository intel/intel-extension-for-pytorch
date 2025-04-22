/******************************************************************************
BSD 3-Clause License

Copyright (c) 2024, Tri Dao.
Modified by Di Bao, 2025
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/
// adapted from
// https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d.h
#pragma once

#include <sycl/sycl.hpp>
////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace sycl;
using namespace at;

struct ConvParamsBase {
  using index_t = uint32_t;

  int batch, dim, seqlen, width;
  int64_t pad_slot_id;
  bool silu_activation;

  index_t x_batch_stride;
  index_t x_c_stride;
  index_t x_l_stride;
  index_t weight_c_stride;
  index_t weight_width_stride;
  index_t out_batch_stride;
  index_t out_c_stride;
  index_t out_l_stride;

  int conv_state_len;
  index_t conv_state_batch_stride;
  index_t conv_state_c_stride;
  index_t conv_state_l_stride;

  // Common data pointers.
  void* __restrict__ x_ptr;
  void* __restrict__ weight_ptr;
  void* __restrict__ bias_ptr;
  void* __restrict__ out_ptr;

  void* __restrict__ conv_state_ptr;
  void* __restrict__ query_start_loc_ptr;
  void* __restrict__ has_initial_state_ptr;
  void* __restrict__ cache_indices_ptr;
  int32_t* __restrict__ cache_seqlens;

  // For the continuous batching case. Makes it so that the mamba state for
  // the current batch doesn't need to be a contiguous tensor.
  int32_t* __restrict__ conv_state_indices_ptr;

  void* __restrict__ seq_idx_ptr;

  // No __restrict__ since initial_states could be the same as final_states.
  void* initial_states_ptr;
  index_t initial_states_batch_stride;
  index_t initial_states_l_stride;
  index_t initial_states_c_stride;

  void* final_states_ptr;
  index_t final_states_batch_stride;
  index_t final_states_l_stride;
  index_t final_states_c_stride;

  void* conv_states_ptr;
  index_t conv_states_batch_stride;
  index_t conv_states_l_stride;
  index_t conv_states_c_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
