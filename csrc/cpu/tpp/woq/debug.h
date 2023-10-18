#ifndef _TLA_DEBUG_H_
#define _TLA_DEBUG_H_

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>

template <int maxlen>
class SafePrint {
 public:
  SafePrint() {}
  template <typename... Types>
  int operator()(Types... vars) {
    if (len < maxlen) {
      int l = snprintf(&buf[len], 2 * maxlen - len, vars...);
      len += l;
      if (len > maxlen) {
        print();
      }
      return l;
    }
    return 0;
  }
  void print() {
    printf("%s", buf);
    len = 0;
    buf[0] = 0;
  }

 private:
  char buf[2 * maxlen];
  int len = 0;
};

template <typename T>
inline void print_matrix(
    T* mat,
    int m,
    int n,
    int ldm,
    const char* name = nullptr,
    int ldn = 1) {
  if (omp_get_thread_num() != 0)
    return;
  std::cout << "\"" << (name ? name : (const char*)("noname")) << "\""
            << "\n";
  for (int i = 0; i < m; i++) {
    std::cout << "[";
    for (int j = 0; j < n; j++) {
      // for floating point, keep four digits after decimal point
      // for integers, print as is but with the same width by adding leading
      // spaces
      if (std::is_integral<T>::value) {
        std::cout << std::setw(5) << (long)(mat[i * ldm + j * ldn]);
      } else {
        std::cout << std::fixed << std::setprecision(4) << std::setw(10)
                  << (float)(mat[i * ldm + j * ldn]);
      }
    }
    std::cout << "]\n";
  }
}

inline void print_matrix_int4(
    uint8_t* mat,
    int m,
    int n,
    int ldm,
    const char* name = nullptr,
    int ldn = 1) {
  std::cout << "\"" << (name ? name : (const char*)("noname")) << "\""
            << "\n";
  for (int i = 0; i < m; i++) {
    std::cout << "[";
    for (int j = 0; j < n; j += 2) {
      // for floating point, keep four digits after decimal point
      // for integers, print as is but with the same width by adding leading
      // spaces
      std::cout << std::setw(5) << (long)(mat[i * ldm / 2 + j * ldn / 2] & 0xf);
      std::cout << std::setw(5) << (long)(mat[i * ldm / 2 + j * ldn / 2] >> 4);
    }
    std::cout << "]\n";
  }
}

#endif