#ifndef IDEEP_ABSTRACT_TYPES_HPP
#define IDEEP_ABSTRACT_TYPES_HPP

#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <cstdlib>
#include <functional>
#include <dnnl.h>
#include <dnnl.hpp>
#include "allocators.hpp"

namespace ideep {

#ifdef _WIN32
#define IDEEP_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define IDEEP_EXPORT __attribute__((__visibility__("default")))
#else
#define IDEEP_EXPORT
#endif

using error = dnnl::error;
using memory = dnnl::memory;
using format_tag = memory::format_tag;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using query = dnnl::query;
using kind = dnnl::primitive::kind;
using prop_kind = dnnl::prop_kind;
using algorithm = dnnl::algorithm;
using batch_normalization_flag = dnnl::normalization_flags;
using query = dnnl::query;
using scale_t = std::vector<float>;
using exec_args = std::unordered_map<int, memory>;

// for computation cache
using key_t = std::string;

#ifndef NDEBUG
#define IDEEP_ENFORCE(condition, message) \
  do {  \
    error::wrap_c_api((condition) \
        ? dnnl_success : dnnl_invalid_arguments, (message));  \
  } while(false)
#else
#define IDEEP_ENFORCE(condition, message)
#endif

const scale_t IDEEP_DEF_SCALE {1.0f};

enum lowp_kind {
  u8s8 = 0,
  s8s8 = 1,
  LOWP_U8S8 = u8s8,
  LOWP_S8S8 = s8s8,
};

enum rnn_kind {
  RNN_RELU = 0,
  RNN_TANH = 1,
  LSTM = 2,
  GRU = 3
};

static bool has_bf16_type_support() {
  // for v1.8
  // static bool support_bf16 = isa >= dnnl::cpu_isa::avx512_core
  //                           && isa != dnnl::cpu_isa::avx2_vnni;
  static bool support_bf16 = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core;
  return support_bf16;
}

/// cpu execution engine only.
struct engine : public dnnl::engine {
  friend class tensor;

  /// Singleton CPU engine for all primitives
  static IDEEP_EXPORT engine& cpu_engine();

  /// Singleton GPU engine for all primitives
  static IDEEP_EXPORT engine& gpu_engine();

  engine(kind akind = kind::cpu, size_t index = 0)
      : dnnl::engine(akind, index),
        malloc(utils::allocator::malloc),
        free(utils::allocator::free) {}

  void set_allocator(const std::function<void*(size_t)>& malloc,
                     const std::function<void(void*)>& free) {
    this->malloc = malloc;
    this->free = free;
  }

 private:
  std::function<void*(size_t)> malloc;
  std::function<void(void*)> free;
};

/// A default stream
struct stream : public dnnl::stream {
  static dnnl::stream& default_stream() {
    static dnnl::stream s(engine::cpu_engine());
    return s;
  }
};
}

#endif
