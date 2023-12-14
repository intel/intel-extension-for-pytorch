#ifndef _THREADED_LOOPS_H_
#define _THREADED_LOOPS_H_

#include <stdio.h>
#include <array>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "jit_compile.h"
#include "par_loop_generator.h"

namespace torch_ipex {
namespace tpp {

typedef std::function<void()> init_func;
typedef std::function<void()> fini_func;
typedef std::function<void(int*)> loop_func;

constexpr int MAX_BLOCKING_LEVELS = 5;
constexpr int MAX_LOGICAL_LOOPS = 10;
constexpr int MAX_LOOPS = MAX_LOGICAL_LOOPS * MAX_BLOCKING_LEVELS;

static std::string code_str = R"(
#include <stdio.h>
#include <cassert>
#include <functional>
#include <initializer_list>
#include <string>

constexpr int MAX_BLOCKING_LEVELS = 5;
class LoopSpecs {
 public:
  LoopSpecs(long end, std::initializer_list<long> block_sizes = {}) : LoopSpecs(0L, end, 1L, block_sizes) {}
  LoopSpecs(long end, bool isParallel, std::initializer_list<long> block_sizes = {}) : LoopSpecs(0L, end, 1L, isParallel, block_sizes) {}
  LoopSpecs(long start, long end, std::initializer_list<long> block_sizes = {}) : LoopSpecs(start, end, 1L, block_sizes) {}
  LoopSpecs(long start, long end, bool isParallel, std::initializer_list<long> block_sizes = {}) : LoopSpecs(start, end, 1L, isParallel, block_sizes) {}
  LoopSpecs(long start, long end, long step, std::initializer_list<long> block_sizes = {}) :  LoopSpecs(start, end, step, true, block_sizes) {}
  LoopSpecs(long start, long end, long step, bool isParallel, std::initializer_list<long> block_sizes = {}) : start(start), end(end), step(step), isParallel(isParallel), nBlockingLevels(block_sizes.size()), block_size{0} {
    assert(nBlockingLevels <= MAX_BLOCKING_LEVELS);
    int i = 0;
    for (auto x : block_sizes) block_size[i++] = x;
  }
  long start;
  long end;
  long step;
  bool isParallel;
  long nBlockingLevels;
  long block_size[MAX_BLOCKING_LEVELS];
};

using loop_rt_spec_t = LoopSpecs;

)";

class LoopSpecs {
 public:
  // Add this default constructor to fix the issue in the constructor of
  // ThreadedLoop See the comments there for more details
  LoopSpecs() : LoopSpecs(0L, 0L, 1L, {}) {}
  LoopSpecs(long end, std::initializer_list<long> block_sizes = {})
      : LoopSpecs(0L, end, 1L, block_sizes) {}
  LoopSpecs(
      long end,
      bool isParallel,
      std::initializer_list<long> block_sizes = {})
      : LoopSpecs(0L, end, 1L, isParallel, block_sizes) {}
  LoopSpecs(long start, long end, std::initializer_list<long> block_sizes = {})
      : LoopSpecs(start, end, 1L, block_sizes) {}
  LoopSpecs(
      long start,
      long end,
      bool isParallel,
      std::initializer_list<long> block_sizes = {})
      : LoopSpecs(start, end, 1L, isParallel, block_sizes) {}
  LoopSpecs(
      long start,
      long end,
      long step,
      std::initializer_list<long> block_sizes = {})
      : LoopSpecs(start, end, step, true, block_sizes) {}
  LoopSpecs(
      long start,
      long end,
      long step,
      bool isParallel,
      std::initializer_list<long> block_sizes = {})
      : start(start),
        end(end),
        step(step),
        isParallel(isParallel),
        nBlockingLevels(block_sizes.size()),
        block_size{0} {
    assert(nBlockingLevels <= MAX_BLOCKING_LEVELS);
    int i = 0;
    for (auto x : block_sizes)
      block_size[i++] = x;
  }
  long start;
  long end;
  long step;
  bool isParallel;
  long nBlockingLevels;
  long block_size[MAX_BLOCKING_LEVELS];
};

typedef void (*par_loop_kernel)(
    LoopSpecs* loopSpecs,
    std::function<void(int*)>,
    std::function<void()>,
    std::function<void()>);

extern std::unordered_map<std::string, par_loop_kernel> pre_defined_loops;

#if 0
void par_nested_loops(LoopSpecs *loopSpecs, std::function<void(int*)> body_func, std::function<void()> init_func, std::function<void()> fini_func)
{
#pragma omp parallel
  {
    if (init_func) init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end; a0 += loopSpecs[0].step) {
#pragma omp for collapse(2) nowait
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end; b0 += loopSpecs[1].step) {
        for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end; c0 += loopSpecs[2].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func) fini_func();
  }
}
#endif

class LoopingScheme {
 public:
  LoopingScheme(std::string scheme)
      : scheme(scheme),
        nLogicalLoops(0),
        nLoops(0),
        barrierAfter(0),
        ompforBefore(-1),
        nCollapsed(0),
        nLLBL{0},
        test_kernel(NULL) {
    int curLoop = 0;
    for (int i = 0; i < (int)scheme.length() - 1; i++) {
      char c = scheme[i];
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
        int l;
        assert(curLoop < MAX_LOOPS);
        if ((i >= 1) && (scheme[i - 1] == '{')) {
          printf(
              "LoopingScheme: '%s': Ignoring unknown scheme character: '%c' at position %d\n",
              scheme.c_str(),
              scheme[i],
              i);
        } else {
          if (c >= 'a' && c <= 'z') {
            isParallel[curLoop] = false;
            l = c - 'a';
          } else {
            isParallel[curLoop] = true;
            l = c - 'A';
            if (ompforBefore == -1)
              ompforBefore = curLoop;
            if (ompforBefore + nCollapsed == curLoop)
              nCollapsed++;
          }
          p2lMap[curLoop] = l;
          curLoop++;
        }
      } else if (c == '|') {
        barrierAfter = curLoop;
      } else {
        printf(
            "LoopingScheme: '%s': Ignoring unknown scheme character: '%c' at position %d\n",
            scheme.c_str(),
            scheme[i],
            i);
      }
    }
    nLoops = curLoop;
    for (int i = 0; i < nLoops; i++) {
      int ll = p2lMap[i];
      assert(ll < MAX_LOGICAL_LOOPS);
      if (nLogicalLoops <= ll)
        nLogicalLoops = ll + 1;
      nLLBL[ll]++;
    }
    for (int i = 0; i < nLogicalLoops; i++) {
      assert(nLLBL[i] > 0);
    }
    auto search = pre_defined_loops.find(scheme);
    if (search != pre_defined_loops.end()) {
      test_kernel = search->second;
    } else {
      std::string gen_code = loop_generator(scheme.c_str());
      std::ofstream ofs("debug.cpp", std::ofstream::out);
      ofs << code_str + gen_code;
      ofs.close();
      std::cout << "Scheme: " << scheme << std::endl;
      std::cout << "Generated code:" << std::endl << gen_code;

      test_kernel = (par_loop_kernel)jit_from_str(
          code_str + gen_code, " -fopenmp ", "par_nested_loops");
    }
  }

  void call(
      LoopSpecs* loopSpecs,
      std::function<void(int*)> body_func,
      std::function<void()> init_func,
      std::function<void()> fini_func) {
    test_kernel(loopSpecs, body_func, init_func, fini_func);
  }

  const std::string getKernelCode() {
    return "test";
  }
  std::string scheme;
  int nLogicalLoops;
  int nLoops;
  int barrierAfter;
  int ompforBefore;
  int nCollapsed;
  int nLLBL[MAX_LOGICAL_LOOPS]; // LogicalLoopBlockingLevels - 1 as no blocking
  bool isParallel[MAX_LOOPS];
  int p2lMap[MAX_LOOPS];
  par_loop_kernel test_kernel;
};

inline LoopingScheme* getLoopingScheme(std::string scheme) {
  static std::unordered_map<std::string, LoopingScheme*> kernel_cache;

  LoopingScheme* kernel = NULL;
  auto search = kernel_cache.find(scheme);
  if (search != kernel_cache.end())
    kernel = search->second;
  if (kernel == NULL) {
    kernel = new LoopingScheme(scheme);
    kernel_cache[scheme] = kernel;
  }
  return kernel;
}

template <int N>
class ThreadedLoop {
 public:
  /*
  Originally the constructor put the bounds in the initializer list:
  ```
  ThreadedLoop(const LoopSpecs (&bounds)[N], std::string scheme = "")
      : bounds(bounds), scheme(scheme) {
  ```
  But this causes error when building with clang:
  error: array initializer must be an initializer list
  So, now this->bounds is initialized by copy elements one by one
  This change leads to another problem: bounds is an array of LoopSpecs,
  but LoopSpecs does not have a default consturctor. So, we added a
  default constructor for LoopSpecs.
  */
  ThreadedLoop(const LoopSpecs (&bounds)[N], std::string scheme = "")
      : scheme(scheme) {
    for (size_t i = 0; i < N; ++i) {
      this->bounds[i] = bounds[i];
    }
    if (scheme == "")
      scheme = getDefaultScheme();
    loopScheme = getLoopingScheme(scheme);
  }

  template <class T>
  void operator()(T func) {
    loopScheme->call(bounds, func, NULL, NULL);
  }
  template <class T, class Ti, class Tf>
  void operator()(T func, Ti init, Tf fini) {
    loopScheme->call(bounds, func, init, fini);
  }

  std::string getDefaultScheme() {
    std::string scheme;
    for (int i = 0; i < N; i++) {
      if (bounds[i].isParallel)
        scheme.append(std::to_string('A' + i));
      else
        scheme.append(std::to_string('a' + i));
    }
    return scheme;
  }

 private:
  LoopSpecs bounds[N];
  std::string scheme;
  LoopingScheme* loopScheme;
};
} // namespace tpp
} // namespace torch_ipex

#endif // _THREADED_LOOPS_H_
