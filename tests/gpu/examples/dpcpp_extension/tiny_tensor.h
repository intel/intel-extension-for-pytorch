#ifndef TINY_TENSOR_H_
#define TINY_TENSOR_H_

#include <stddef.h>
#include <iostream>
#include "device_memory.h"

class TinyTensor {
 public:
  TinyTensor(int n, int c, int h, int w) {
    this->N = n;
    this->C = c;
    this->H = h;
    this->W = w;
    this->data = g_devMemMgr.alloc(count());
  }
  size_t count() const {
    return N * C * H * W;
  }
  void print(bool showdata) const {
    std::cout << "NCHW: (" << N << ", " << C << ", " << H << ", " << W << ")"
              << std::endl;
    if (!showdata) {
      return;
    }
#ifdef USE_HOST_MEMORY
    for (int ni = 0; ni < N; ++ni) {
      for (int ci = 0; ci < C; ++ci) {
        std::cout << "n: " << ni << ",  c: " << ci << std::endl;
        for (int hi = 0; hi < H; ++hi) {
          for (int wi = 0; wi < W; ++wi) {
            std::cout << data[wi + hi * W + ci * H * W + ni * C * H * W]
                      << ", ";
          }
          std::cout << std::endl << std::endl;
        }
      }
    }
#endif
  }
  // NCHW for the demo
  int N;
  int C;
  int H;
  int W;
  float* data;
};

#endif