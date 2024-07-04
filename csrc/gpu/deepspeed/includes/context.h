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
#include <cassert>
#include <iostream>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>

#include <torch/library.h>
#include "rng_utils.h"

#ifndef BF16_AVAILABLE
#define BF16_AVAILABLE
#endif

// define OP register macro

namespace at {
namespace ds {

template <typename Func>
void construct_function_schema_and_register(
    const char* name,
    Func&& func,
    torch::Library& m) {
  std::unique_ptr<c10::FunctionSchema> infer_schema =
      c10::detail::inferFunctionSchemaFromFunctor<std::decay_t<Func>>();
  auto parse_name = torch::jit::parseSchemaOrName(name);
  c10::OperatorName op_name =
      std::get<c10::OperatorName>(std::move(parse_name));
  c10::FunctionSchema s = infer_schema->cloneWithName(
      std::move(op_name.name), std::move(op_name.overload_name));
  s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
  c10::OperatorName find_name = s.operator_name();
  const auto found = c10::Dispatcher::realSingleton().findOp(find_name);
  if (found == c10::nullopt) {
    m.def(std::move(s));
  }
}

template <typename T, int N>
struct TypeSelector {
  template <typename... Args>
  void extract_type(Args... args) {
    return;
  }

  template <typename... Args>
  void extract_type(T& type, Args... args) {
    container_.push_back(type);
    extract_type(args...);
  }

  template <typename U, typename... Args>
  void extract_type(U type, Args... args) {
    extract_type(args...);
  }

  at::ArrayRef<T> retrive_types() {
    return at::ArrayRef<T>(container_.begin(), container_.end());
  }

  at::SmallVector<T, N> container_;
};

template <typename Signature, Signature* Func, typename Ret, typename TypeList>
struct DSFunctionWarpper_ {};

template <typename Signature, Signature* Func, typename Ret, typename... Args>
struct DSFunctionWarpper_<
    Signature,
    Func,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    TypeSelector<at::Tensor, sizeof...(args)> selector;
    selector.extract_type(args...);
    const at::OptionalDeviceGuard dev_guard(
        device_of(selector.retrive_types()));
    return (*Func)(args...);
  }
};

template <typename Signature, Signature* Func>
struct DSFunctionWarpper {
  using type = DSFunctionWarpper_<
      Signature,
      Func,
      typename guts::function_traits<Signature>::return_type,
      typename guts::function_traits<Signature>::parameter_types>;
};
} // namespace ds
} // namespace at

#define DS_LIBRARY_FRAGMENT() TORCH_LIBRARY_FRAGMENT(torch_ipex, m)

#define DS_OP_REGISTER(NAME, Func, Dispatchkey)            \
  at::ds::construct_function_schema_and_register(          \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME), Func, m); \
  m.impl(                                                  \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME),           \
      Dispatchkey,                                         \
      &at::ds::DSFunctionWarpper<decltype(Func), &Func>::type::call);

#ifndef SYCL_CUDA_STREAM
#define SYCL_CUDA_STREAM
namespace at {
inline sycl::queue* getCurrentSYCLStream() {
  auto& dpcpp_queue = at::xpu::getCurrentXPUStream().queue();
  return &dpcpp_queue;
}

inline sycl::queue* getStreamFromPool(bool) {
  // not implemented
  return nullptr;
}

inline sycl::queue* getStreamFromPool() {
  // not implemented
  return nullptr;
}
} // namespace at
#endif

namespace ds {
enum error_code { success = 0, default_error = 999 };
}

#define DS_CHECK_ERROR(expr)              \
  [&]() {                                 \
    try {                                 \
      expr;                               \
      return ds::success;                 \
    } catch (std::exception const& e) {   \
      std::cerr << e.what() << std::endl; \
      return ds::default_error;           \
    }                                     \
  }()

#define MEGABYTE (1024 * 1024)
#define GIGABYTE (1024 * 1024 * 1024)

#define WARP_SIZE 32
#define MAX_REG 256

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
  return (std::max)(
      (std::min)(
          (N + DS_SYCL_NUM_THREADS - 1) / DS_SYCL_NUM_THREADS,
          DS_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since SYCL does not allow empty block
      1);
}

class TrainingContext {
 public:
  TrainingContext() try : _workspace(nullptr), _seed(42), _curr_offset(0) {
    _gen = ds::rng::create_host_rng(ds::rng::random_engine_type::mcg59);
    _gen->set_seed(123);
    int stat = DS_CHECK_ERROR(_mklHandle = at::getCurrentSYCLStream());
    if (stat != 0) {
      // It would be nice to use mklGetStatusName and
      // mklGetStatusString, but they were only added in SYCL 11.4.2.
      auto message =
          std::string("Failed to create mkl handle: mklStatus_t was ") +
          std::to_string(stat);
      std::cerr << message << std::endl;
      throw std::runtime_error(message);
    }
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  virtual ~TrainingContext() {
    _mklHandle = nullptr;
    sycl::free(_workspace, *at::getCurrentSYCLStream());
  }

  static TrainingContext& Instance() {
    static TrainingContext _ctx;
    return _ctx;
  }

  void SetWorkSpace(void* workspace) {
    if (!workspace) {
      throw std::runtime_error("Workspace is null.");
    }
    _workspace = workspace;
  }

  void* GetWorkSpace() {
    return _workspace;
  }

  ds::rng::host_rng_ptr& GetRandGenerator() {
    return _gen;
  }

  sycl::queue* GetCurrentStream() {
    // get current pytorch stream.
    sycl::queue* stream = at::getCurrentSYCLStream();
    return stream;
  }

  sycl::queue* GetNewStream() {
    return at::getStreamFromPool();
  }

  sycl::queue* GetCublasHandle() {
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

 private:
  ds::rng::host_rng_ptr _gen;
  sycl::queue* _mklHandle;
  void* _workspace;
  uint64_t _seed;
  uint64_t _curr_offset;
  std::vector<std::array<int, 3>> _gemm_algos;
};

class InferenceContext {
 public:
  InferenceContext() try
      : _workspace(nullptr),
        _seed(42),
        _curr_offset(0),
        _stream(at::getCurrentSYCLStream()),
        _free_memory_size(0),
        _num_tokens(1),
        _attention_unfused_workspace_offset(0),
        _workSpaceSize(0) {
    _workSpaceSize = 0;
    _workspace = 0;

    int stat = DS_CHECK_ERROR(_mklHandle = at::getCurrentSYCLStream());
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
    sycl::free(_workspace, *at::getCurrentSYCLStream());
    delete _comp1_event;
    delete _comp2_event;
    delete _comp_event;
    delete _comm_event;
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
    sycl::queue& q_ct1 = *at::getCurrentSYCLStream();
    size_t total_size;
    _free_memory_size = 21474836480;

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
  } //_token_length = 0;

  inline unsigned current_tokens() const {
    return _num_tokens;
  }

  inline void advance_tokens() {
    _num_tokens++;
  }

  sycl::queue* GetCommStream(bool async_op = false) {
    if (!_comm_stream)
      _comm_stream =
          async_op ? at::getStreamFromPool(true) : at::getCurrentSYCLStream();
    return _comm_stream;
  }
  sycl::queue* GetCurrentStream(bool other_stream = false) {
    // get current pytorch stream.
    if (other_stream) {
      if (!_stream)
        _stream = at::getStreamFromPool(true);
      return _stream;
    }
    sycl::queue* stream = at::getCurrentSYCLStream();
    return stream;
  }

  void release_workspace() {
    sycl::free(_workspace, *at::getCurrentSYCLStream());
    _workspace = nullptr;
  }
  bool retake_workspace() {
    if (_workspace != nullptr || _workSpaceSize == 0)
      return true;
    _workspace =
        (void*)sycl::malloc_device(_workSpaceSize, *at::getCurrentSYCLStream());
    return _workspace != nullptr;
  }
  sycl::queue* GetCublasHandle() {
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
    _comp_event_ct1 = std::chrono::steady_clock::now();
    *_comp_event = _comp_stream->ext_oneapi_submit_barrier();
    _comm_stream->ext_oneapi_submit_barrier({*_comp_event});
  }
  inline void SynchComm() {
    _comm_event_ct1 = std::chrono::steady_clock::now();
    *_comm_event = _comm_stream->ext_oneapi_submit_barrier();
    _comp_stream->ext_oneapi_submit_barrier({*_comm_event});
  }

 private:
  sycl::queue* _mklHandle;

  sycl::event* _comp_event;
  std::chrono::time_point<std::chrono::steady_clock> _comp_event_ct1;
  sycl::event* _comm_event;
  std::chrono::time_point<std::chrono::steady_clock> _comm_event_ct1;

  void* _workspace;
  // offset from _workspace for attention unfused memory
  size_t _attention_unfused_workspace_offset;
  uint64_t _seed;
  uint64_t _curr_offset;

  size_t _workSpaceSize;
  size_t _free_memory_size;

  size_t _max_seq_len;

  sycl::event* _comp1_event;
  sycl::event* _comp2_event;

  sycl::queue* _stream;

  unsigned _token_length;
  unsigned _num_tokens;
  std::vector<std::array<int, 3>> _gemm_algos;

  sycl::queue* _comp_stream;
  sycl::queue* _comm_stream;

  std::unordered_map<int, int> _world_sizes;
};
