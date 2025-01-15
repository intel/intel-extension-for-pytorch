#pragma once

#include <ATen/Context.h>

namespace at::xpu {

template <typename T1, typename T2>
struct pair {
  typedef T1 first_type;
  typedef T2 second_type;

  first_type first;
  second_type second;

  // default constructor
  pair(void) : first(), second() {}

  inline pair(const T1& x, const T2& y) : first(x), second(y) {}

  template <typename U1, typename U2>
  inline pair(const pair<U1, U2>& p) : first(p.first), second(p.second) {}

  template <typename U1, typename U2>
  pair(const std::pair<U1, U2>& p) : first(p.first), second(p.second) {}
};

template <typename T1, typename T2>
bool operator==(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return x.first == y.first && x.second == y.second;
}

template <typename T1, typename T2>
inline bool operator<(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return x.first < y.first || (!(y.first < x.first) && x.second < y.second);
}

template <typename T1, typename T2>
inline bool operator!=(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return !(x == y);
}

template <typename T1, typename T2>
inline bool operator>(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return y < x;
}

template <typename T1, typename T2>
bool operator<=(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return !(y < x);
}

template <typename T1, typename T2>
bool operator>=(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return !(x < y);
}

template <typename T1, typename T2>
inline pair<T1, T2> make_pair(T1 x, T2 y) {
  return pair<T1, T2>(x, y);
}

template <unsigned int N, typename T1, typename T2>
inline auto& get(pair<T1, T2>& p) {
  if constexpr (N == 0)
    return p.first;
  else
    return p.second;
}

template <unsigned int N, typename T1, typename T2>
inline const auto& get(const pair<T1, T2>& p) {
  if constexpr (N == 0)
    return p.first;
  else
    return p.second;
}

} // namespace at::xpu
