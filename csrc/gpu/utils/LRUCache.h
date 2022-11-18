#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace xpu {
namespace dpcpp {

using bytestring = std::string;
using lru_key_t = bytestring;

// A class that provides the destructor for LRU handler
template <typename T>
class lru_traits {};

template <typename T, typename traits = lru_traits<T>>
class lru_handle
    : public std::shared_ptr<typename std::remove_pointer<T>::type> {
  using super = std::shared_ptr<typename std::remove_pointer<T>::type>;

 public:
  /// Constructs a C handle wrapper.
  /// @param t The C handle to wrap.
  /// @param weak A flag to specify whether to construct a weak wrapper.
  lru_handle(T t = nullptr, bool weak = false)
      : super(t, [weak]() {
          auto dummy = [](T) { return decltype(traits::destructor(0))(0); };
          return weak ? dummy : traits::destructor;
        }()) {}

  using super::super;

  /// Resets the value of a C handle.
  /// @param t The new value of the C handle.
  /// @param weak A flag to specify whether the wrapper should be weak.
  void reset(T t, bool weak = false) {
    auto dummy_destructor = [](T) {
      return decltype(traits::destructor(0))(0);
    };
    super::reset(t, weak ? dummy_destructor : traits::destructor);
  }
};

template <
    class key_t,
    class value_t,
    template <typename...> class map = std::unordered_map>
class lru_cache {
 public:
  class node_t;

  using value_type = typename std::pair<key_t, value_t>;

  // Only need opaque node_t pointer, it'll compile
  using iterator = typename std::list<node_t>::iterator;
  using const_iterator = typename std::list<node_t>::const_iterator;

  using map_it = typename map<key_t, iterator>::iterator;
  using const_map_it = typename map<key_t, iterator>::const_iterator;

  // Only class possible, we can't use typedef or using. Or can we?
  class node_t : public std::pair<map_it, value_t> {
   public:
    node_t(const std::pair<map_it, value_t>& l)
        : std::pair<map_it, value_t>(l) {}
    node_t(std::pair<map_it, value_t>&& l)
        : std::pair<map_it, value_t>(std::move(l)) {}
  };

  using size_type = typename std::list<node_t>::size_type;

  lru_cache(size_type capacity) : capacity_(capacity) {}

  size_type size() const {
    map_.size();
  }
  size_type max_size() const {
    return capacity_;
  }
  void resize(size_type new_capacity) {
    capacity_ = new_capacity;

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last--;
      map_.erase(last->first);
      vlist_.pop_back();
    }
  }

  iterator begin() noexcept {
    auto it = map_.begin();
    if (it == map_.end()) {
      return vlist_.end();
    }
    return it->second;
  }
  const_iterator begin() const noexcept {
    const auto it = map_.begin();
    if (it == map_.end()) {
      return vlist_.end();
    }
    return it->second;
  }
  iterator end() noexcept {
    return vlist_.end();
  }
  const_iterator end() const noexcept {
    return vlist_.end();
  }

  iterator find(const key_t& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  // Is this feasible?
  const_iterator find(const key_t& key) const {
    const auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  bool empty() const noexcept {
    return vlist_.empty();
  }

  void clear() noexcept {
    vlist_.clear();
    map_.clear();
  }

  // Can we?
  // template <class... Args>
  // std::pair<iterator, bool> emplace(Args&&... args) {
  // }

  std::pair<iterator, bool> insert(const value_type& value) {
    auto map_it = map_.find(value.first);

    if (map_it == map_.end()) {
      vlist_.push_front(std::make_pair(map_it, value.second));
      auto list_it = vlist_.begin();
      auto updated = map_.insert(map_it, std::make_pair(value.first, list_it));
      // Update node to pointer to new map position
      list_it->first = updated;
    } else
      return std::make_pair(map_it->second, false);

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last--;
      map_.erase(last->first);
      vlist_.pop_back();
    }

    return std::make_pair(vlist_.begin(), true);
  }

  iterator erase(iterator pos) {
    auto map_pos = pos->first;
    map_.erase(map_pos);
    return vlist_.erase(pos);
  }

  // Warning: carefully check iterator validity
  void swap(lru_cache& other) {
    std::swap(vlist_, other.vlist_);
    std::swap(map_, other.map_);
    std::swap(capacity_, other.capacity_);
  }

 private:
  std::list<node_t> vlist_;
  map<key_t, iterator> map_;
  size_type capacity_;
};

template <class value_t, size_t capacity = 128, class key_t = std::string>
class computation_cache {
 public:
  using iterator = typename lru_cache<key_t, value_t>::iterator;

 protected:
  template <typename... Ts>
  static inline iterator create(const key_t& key, Ts&&... args) {
    auto it = t_store().insert(
        std::make_pair(key, value_t(std::forward<Ts>(args)...)));
    return it.first;
  }

  static inline value_t& fetch(iterator it) {
    return it->second;
  }

  static inline void update(value_t& val, iterator it) {
    it->second = val;
  }

  static inline iterator find(const key_t& key) {
    return t_store().find(key);
  }

  static inline iterator end() {
    return t_store().end();
  }

 public:
  template <typename... Ts>
  static inline value_t& fetch_or_create(const key_t& key, Ts&&... args) {
    return fetch(create(key, std::forward<Ts>(args)...));
  }

  static inline void release(const key_t& key, const value_t& computation) {
    // Empty
  }

  static inline void release(const key_t& key, value_t&& computation) {
    // Empty
  }

  static inline lru_cache<key_t, value_t>& t_store() {
    static thread_local lru_cache<key_t, value_t> t_store_(capacity);
    return t_store_;
  }
};

template <class value_t, class key_t>
struct lru_cache_standalone : public computation_cache<value_t, 128, key_t> {
 public:
  using cache = computation_cache<value_t, 128, key_t>;

  template <typename... Ts>
  inline value_t fetch_or_create_m(key_t key, Ts&&... args) {
    auto it = cache::find(key);
    auto op = it == cache::end()
        ? cache::fetch(cache::create(key, std::forward<Ts>(args)...))
        : cache::fetch(it);
    return op;
  }
};

template <class value_t, class key_t, typename... Ts>
inline value_t fetch_or_create_m(key_t key, Ts&&... args) {
  return lru_cache_standalone<value_t, key_t>().fetch_or_create_m(
      key, std::forward<Ts>(args)...);
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, void>::type to_bytes(
    bytestring& bytes,
    T arg) {
  if (arg == 0)
    return;
  auto len = sizeof(T);
  if constexpr (sizeof(T) >= 4) {
    len = sizeof(T) - (__builtin_clz(arg) / 8);
  }
  auto as_cstring = reinterpret_cast<char*>(&arg);
  bytes.append(as_cstring, len);
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, void>::type
to_bytes(bytestring& bytes, T arg) {
  auto as_cstring = reinterpret_cast<char*>(&arg);
  bytes.append(as_cstring, sizeof(T));
}

template <typename T>
inline typename std::enable_if<std::is_enum<T>::value, void>::type to_bytes(
    bytestring& bytes,
    T arg) {
  to_bytes(bytes, static_cast<int>(arg));
}

template <typename T>
inline typename std::enable_if<std::is_class<T>::value, void>::type to_bytes(
    bytestring& bytes,
    T& arg) {
  arg.to_bytes(bytes);
}

template <typename T>
inline void to_bytes(bytestring& bytes, std::vector<T>& arg) {
  if (arg.size() > 0) {
    for (T& elems : arg) {
      to_bytes(bytes, elems);
      bytes.append(1, 'v');
    }
    bytes.pop_back();
  } else {
    bytes.append(1, 'v');
  }
}

template <>
inline void to_bytes(bytestring& bytes, dnnl::memory::desc& adesc) {
  auto desc = adesc.data;
  for (int i = 0; i < desc.ndims; i++) {
    to_bytes(bytes, desc.dims[i]);
    to_bytes(bytes, desc.padded_dims[i]);
    to_bytes(bytes, desc.padded_offsets[i]);
    to_bytes(bytes, desc.format_desc.blocking.strides[i]);
  }

  for (int i = 0; i < desc.format_desc.blocking.inner_nblks; i++) {
    to_bytes(bytes, desc.format_desc.blocking.inner_blks[i]);
    to_bytes(bytes, desc.format_desc.blocking.inner_idxs[i]);
  }

  to_bytes(bytes, desc.data_type);
  to_bytes(bytes, desc.offset0);
  to_bytes(bytes, desc.format_kind);
}

template <>
inline void to_bytes(bytestring& bytes, bool arg) {
  to_bytes(bytes, arg ? 1 : 0);
  bytes.append(1, 'b');
}

template <typename T, typename... Ts>
inline void to_bytes(bytestring& bytes, T&& arg, Ts&&... args) {
  to_bytes(bytes, std::forward<T>(arg));
  bytes.append(1, '*');
  to_bytes(bytes, std::forward<Ts>(args)...);
}

template <typename... Ts>
inline void create_key(bytestring& key_to_create, Ts&&... args) {
  to_bytes(key_to_create, std::forward<Ts>(args)...);
}

} // namespace dpcpp
} // namespace xpu
