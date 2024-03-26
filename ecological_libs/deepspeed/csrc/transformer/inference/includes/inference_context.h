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
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <dpct/blas_utils.h>
#include <dpct/dpct.h>
#include <ipex.h>
#include <sycl/sycl.hpp>
#include <cassert>
#include <iostream>
#include <vector>

#include <array>
#include <cmath>
#include <unordered_map>

#include <chrono>

namespace at {
namespace sycl {
inline dpct::queue_ptr getCurrentSYCLStream() {
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);
  return &queue;
}

inline dpct::queue_ptr getStreamFromPool(bool) {
  // not implemented
  return nullptr;
}
} // namespace sycl
} // namespace at

#define MEGABYTE (1024 * 1024)
#define GIGABYTE (1024 * 1024 * 1024)

// TODO: refactor out
#define WARP_SIZE 32

#define SYCL_CHECK(callstr)                                                 \
  {                                                                         \
    syclError_t error_code = callstr;                                       \
    if (error_code != syclSuccess) {                                        \
      std::cerr << "SYCL error " << error_code << " at " << __FILE__ << ":" \
                << __LINE__;                                                \
      assert(0);                                                            \
    }                                                                       \
  }

#define SYCL_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define SYCL_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define DS_SYCL_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + DS_SYCL_NUM_THREADS - 1) / DS_SYCL_NUM_THREADS,
          DS_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since SYCL does not allow empty block
      1);
}

class InferenceContext {
 public:
  InferenceContext() try
      : _workspace(nullptr),
        _seed(42),
        _curr_offset(0),
        _stream(&dpct::get_in_order_queue()),
        _free_memory_size(0),
        _num_tokens(1),
        _attention_unfused_workspace_offset(0),
        _workSpaceSize(0) {
    _workSpaceSize = 0;
    _workspace = 0;

    int stat = DPCT_CHECK_ERROR(_mklHandle = &dpct::get_in_order_queue());
    if (stat != 0) {
      // It would be nice to use mklGetStatusName and
      // mklGetStatusString, but they were only added in SYCL 11.4.2.
      auto message =
          std::string("Failed to create mkl handle: mklStatus_t was ") +
          std::to_string(stat);
      std::cerr << message << std::endl;
      throw std::runtime_error(message);
    }
    _comp1_event = new sycl::event();
    _comp2_event = new sycl::event();
    _comp_event = new sycl::event();
    _comm_event = new sycl::event();
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  virtual ~InferenceContext() {
    _mklHandle = nullptr;
    sycl::free(_workspace, dpct::get_in_order_queue());
    dpct::destroy_event(_comp1_event);
    dpct::destroy_event(_comp2_event);
    dpct::destroy_event(_comp_event);
    dpct::destroy_event(_comm_event);
  }

  static InferenceContext& Instance() {
    static InferenceContext _ctx;
    return _ctx;
  }

  void GenWorkSpace(
      const unsigned& num_layers,
      const unsigned& num_heads,
      const size_t& batch_size,
      const size_t& prompt_len,
      const size_t& hidden_dim,
      const unsigned& mp_size,
      const bool& external_cache,
      const size_t& elem_size,
      const unsigned& rank,
      unsigned max_out_tokens,
      unsigned min_out_tokens) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.in_order_queue();
    size_t total_size;
    /*
    DPCT1106:1: 'syclMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    _free_memory_size = 21474836480;
    if (!_free_memory_size) {
      dpct::get_current_device().get_memory_info(_free_memory_size, total_size);
    }

    // Flash attention requires padded heads and we'll conservatively allocate
    // for that here. Flash attention is only enabled for head size <= 128 right
    // now
    const int head_size = hidden_dim / num_heads;
    const int padded_head_size =
        head_size <= 32 ? 32 : (head_size <= 64 ? 64 : 128);
    const int effective_head_size =
        (head_size > 128) ? head_size : padded_head_size;

    size_t activation_size =
        10 * (num_heads * effective_head_size) * batch_size;
    // Other sequence length dimension is added when the final workSpaceSize is
    // calculated
    size_t temp_size = batch_size * (num_heads / mp_size) * max_out_tokens;
    size_t cache_size = num_layers * batch_size *
        ((num_heads * effective_head_size) / mp_size) * 2;
    size_t minimal_requirements =
        temp_size + (_free_memory_size > GIGABYTE ? 500 : 100) * MEGABYTE;
    if (_free_memory_size < minimal_requirements) {
      printf(
          "Requested:\t%lu\nFree:\t%lu\nTotal:\t%lu\n",
          minimal_requirements,
          _free_memory_size,
          total_size);
      throw std::runtime_error(
          "Workspace can't be allocated, no enough memory.");
    }

    _max_seq_len = ((_free_memory_size - minimal_requirements) / elem_size) /
        (activation_size + temp_size + cache_size);
    _max_seq_len = std::min((size_t)max_out_tokens, _max_seq_len);
    size_t workSpaceSize =
        ((external_cache ? (activation_size + temp_size)
                         : (activation_size + temp_size + cache_size))) *
        _max_seq_len * elem_size;
    temp_size *= _max_seq_len * elem_size;

    if (_max_seq_len < min_out_tokens) {
      printf(
          "Allocatable workspace available (%ld tokens) is less than minimum requested "
          "workspace (%d tokens)\n",
          _max_seq_len,
          min_out_tokens);
      throw std::runtime_error(
          "Workspace can't be allocated, not enough memory");
    }

    if (!_workspace) {
      assert(_workspace == nullptr);
      _workspace = (void*)sycl::malloc_device(workSpaceSize, q_ct1);
    } else if (_workSpaceSize < workSpaceSize) {
      sycl::free(_workspace, q_ct1);
      _workspace = (void*)sycl::malloc_device(workSpaceSize, q_ct1);
    }
    if (rank == 0 && (!_workspace || _workSpaceSize < workSpaceSize))
      printf(
          "------------------------------------------------------\n"
          "Free memory : %f (GigaBytes)  \n"
          "Total memory: %f (GigaBytes)  \n"
          "Requested memory: %f (GigaBytes) \n"
          "Setting maximum total tokens (input + output) to %lu \n"
          "WorkSpace: %p \n"
          "------------------------------------------------------\n",
          (float)_free_memory_size / GIGABYTE,
          (float)total_size / GIGABYTE,
          (float)workSpaceSize / GIGABYTE,
          _max_seq_len,
          _workspace);

    if (!_workspace) {
      printf(
          "Requested:\t%lu\nFree:\t%lu\nTotal:\t%lu\n",
          workSpaceSize,
          _free_memory_size,
          total_size);
      throw std::runtime_error("Workspace is null.");
    }
    _workSpaceSize = workSpaceSize;
    _attention_unfused_workspace_offset = workSpaceSize - temp_size;
  }
  inline int GetMaxTokenLength() const {
    return _max_seq_len;
  }

  dpct::event_ptr GetCompEvent(int id) {
    return id == 1 ? _comp1_event : _comp2_event;
  }

  size_t get_workspace_size() const {
    return _workSpaceSize;
  }
  void* GetWorkSpace() {
    return _workspace;
  }
  void* GetAttentionUnfusedWorkspace() {
    return (char*)_workspace + _attention_unfused_workspace_offset;
  }

  inline unsigned new_token(unsigned layer_id) {
    if (layer_id == 0)
      _token_length++;
    return _token_length;
  }

  inline void reset_tokens(unsigned initial_tokens = 1) {
    _num_tokens = initial_tokens;
  } //_token_length = 0; }

  inline unsigned current_tokens() const {
    return _num_tokens;
  }

  inline void advance_tokens() {
    _num_tokens++;
  }

  dpct::queue_ptr GetCommStream(bool async_op = false) {
    if (!_comm_stream)
      _comm_stream = async_op ? at::sycl::getStreamFromPool(true)
                              : at::sycl::getCurrentSYCLStream();
    return _comm_stream;
  }
  dpct::queue_ptr GetCurrentStream(bool other_stream = false) {
    // get current pytorch stream.
    if (other_stream) {
      if (!_stream)
        _stream = at::sycl::getStreamFromPool(true);
      return _stream;
    }
    dpct::queue_ptr stream = at::sycl::getCurrentSYCLStream();
    return stream;
  }

  void release_workspace() {
    sycl::free(_workspace, dpct::get_in_order_queue());
    _workspace = nullptr;
  }
  bool retake_workspace() {
    if (_workspace != nullptr || _workSpaceSize == 0)
      return true;
    _workspace =
        (void*)sycl::malloc_device(_workSpaceSize, dpct::get_in_order_queue());
    return _workspace != nullptr;
  }
  dpct::queue_ptr GetCublasHandle() {
    return _mklHandle;
  }

  std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc) {
    uint64_t offset = _curr_offset;
    _curr_offset += offset_inc;
    return std::pair<uint64_t, uint64_t>(_seed, offset);
  }

  void SetSeed(uint64_t new_seed) {
    _seed = new_seed;
  }

  const std::vector<std::array<int, 3>>& GetGemmAlgos() const {
    return _gemm_algos;
  }

  inline void SynchComp() {
    /*
    DPCT1012:2: Detected kernel execution time measurement pattern and generated
    an initial code for time measurements in SYCL. You can change the way time
    is measured depending on your goals.
    */
    _comp_event_ct1 = std::chrono::steady_clock::now();
    *_comp_event = _comp_stream->ext_oneapi_submit_barrier();
    _comm_stream->ext_oneapi_submit_barrier({*_comp_event});
  }
  inline void SynchComm() {
    /*
    DPCT1012:3: Detected kernel execution time measurement pattern and generated
    an initial code for time measurements in SYCL. You can change the way time
    is measured depending on your goals.
    */
    _comm_event_ct1 = std::chrono::steady_clock::now();
    *_comm_event = _comm_stream->ext_oneapi_submit_barrier();
    _comp_stream->ext_oneapi_submit_barrier({*_comm_event});
  }

 private:
  dpct::queue_ptr _mklHandle;

  dpct::event_ptr _comp_event;
  std::chrono::time_point<std::chrono::steady_clock> _comp_event_ct1;
  dpct::event_ptr _comm_event;
  std::chrono::time_point<std::chrono::steady_clock> _comm_event_ct1;

  void* _workspace;
  // offset from _workspace for attention unfused memory
  size_t _attention_unfused_workspace_offset;
  uint64_t _seed;
  uint64_t _curr_offset;

  size_t _workSpaceSize;
  size_t _free_memory_size;

  size_t _max_seq_len;

  dpct::event_ptr _comp1_event;
  dpct::event_ptr _comp2_event;

  dpct::queue_ptr _stream;

  unsigned _token_length;
  unsigned _num_tokens;
  std::vector<std::array<int, 3>> _gemm_algos;

  dpct::queue_ptr _comp_stream;
  dpct::queue_ptr _comm_stream;

  std::unordered_map<int, int> _world_sizes;
};
