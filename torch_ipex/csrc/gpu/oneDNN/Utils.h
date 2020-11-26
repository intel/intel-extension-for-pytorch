#pragma once

#include <ATen/ATen.h>
#include <dnnl.hpp>

#ifdef USE_PRIMITIVE_CACHE
#include <string>
#include <vector>
#endif

using namespace dnnl;

namespace at {
namespace dpcpp {
namespace oneDNN {

static inline memory::data_type
get_onednn_dtype(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
  case at::ScalarType::Byte:
    return memory::data_type::u8;
  case at::ScalarType::Char:
    return memory::data_type::s8;
  case at::ScalarType::Int:
    return memory::data_type::s32;
  case at::ScalarType::Half:
    return memory::data_type::f16;
  case at::ScalarType::Float:
    return memory::data_type::f32;
  case at::ScalarType::BFloat16:
    return memory::data_type::bf16;
  default:
    return memory::data_type::undef;
  };
}

static inline memory::dims
get_onednn_dims(const at::Tensor& tensor) {
  memory::dims dims;
  for (int i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

static inline memory::dims
get_onednn_strides(const at::Tensor& tensor) {
  memory::dims strides;
  for (int i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

}}}

#ifdef USE_PRIMITIVE_CACHE
// Shallow copied vector
template <class T, class Alloc = std::allocator<T>>
class s_vector {
public:
  using size_type = typename std::vector<T, Alloc>::size_type;
  using reference = typename std::vector<T, Alloc>::reference;
  using const_reference = typename std::vector<T, Alloc>::const_reference;

  s_vector() : n_elems_(0), storage_() {}
  explicit s_vector(size_type count, const Alloc& alloc = Alloc())
    : n_elems_(count) {
    Alloc dup_alloc(alloc);

    storage_.reset(new (dup_alloc.allocate(count)) T [count] (),
       [dup_alloc, count](T *p) mutable {
      for (int i =0; i < count; i ++)
        p[i].~T();
      dup_alloc.deallocate(p, count);
    });
  }
  s_vector(std::initializer_list<T> init, const Alloc& alloc = Alloc())
    : storage_(init.size(), alloc) {
      auto arr = storage_.get();
      auto src = init.begin();
      for (int i = 0; i < init.size(); i ++)
        arr[i] = src[i];
  }

  s_vector(const s_vector& other) : n_elems_(other.n_elems_),
    storage_(other.storage_) {}
  s_vector(s_vector &&other) noexcept : n_elems_(other.n_elems_),
    storage_(std::move(other.storage_)) {}

  s_vector& operator=(const s_vector &other) {
    storage_ = other.storage_;
    n_elems_ = other.n_elems_;
    return *this;
  }
  s_vector& operator=(s_vector&& other) noexcept {
    storage_ = std::move(other.storage_);
    n_elems_ = other.n_elems_;
    return *this;
  }

  reference operator[]( size_type pos ) {
    return storage_.get()[pos];
  }
  const_reference operator[] (size_type pos) const {
    return storage_.get()[pos];
  }

  size_type size() const noexcept {
    return n_elems_;
  }
protected:
  size_type n_elems_;
  std::shared_ptr<T> storage_;
};

// Fast alternative to heavy string method
using bytestring = std::string;

template <typename T>
inline typename std::enable_if<!std::is_enum<T>::value, void>::type
    _to_bytes(bytestring& bytes, T arg) {}

template <typename T>
inline typename std::enable_if<std::is_enum<T>::value, void>::type
_to_bytes(bytestring& bytes, T arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  bytes.append(as_cstring, sizeof(long));
}

template <>
inline void _to_bytes<long>(bytestring& bytes, long arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  bytes.append(as_cstring, sizeof(long));
}

template <>
inline void _to_bytes<int>(bytestring& bytes, int arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  bytes.append(as_cstring, sizeof(int));
}

template <>
inline void _to_bytes<float>(bytestring& bytes, float arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  bytes.append(as_cstring, sizeof(float));
}

template <>
inline void _to_bytes<uint64_t>(bytestring& str, uint64_t arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  str.append(as_cstring, sizeof(uint64_t));
}

template <>
inline void _to_bytes<std::vector<long>>(bytestring& bytes, std::vector<long> arg) {
  if (arg.size() > 0) {
    for (int elems : arg) {
      _to_bytes(bytes, elems);
      bytes.append(1, 'x');
    }
    bytes.pop_back();
  } else {
    bytes.append(1, 'x');
  }
}

template <>
inline void _to_bytes<std::vector<float>>(bytestring& bytes, std::vector<float> arg) {
  if (arg.size() > 0) {
    for (float elems : arg) {
      _to_bytes(bytes, elems);
      bytes.append(1, 'x');
    }
    bytes.pop_back();
  } else {
    bytes.append(1, 'x');
  }
}

template <typename T>
inline void to_bytes(bytestring& bytes, T arg) {
  _to_bytes(bytes, arg);
}
#endif

