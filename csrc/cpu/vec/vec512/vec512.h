#include "vec512_bfloat16.h"
#include "vec512_int8.h"

#include "perf_kernel/kernel.h"

namespace torch_ipex {
namespace cpu {
// A class for forced loop unrolling at compile time
// These macro utils and the small gemm intrinsics kernels are implemented
// based on the initial code by pujiang.he@intel.com.
template <int i>
struct compile_time_for {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    compile_time_for<i - 1>::op(function, args...);
    function(std::integral_constant<int, i - 1>{}, args...);
  }
};
template <>
struct compile_time_for<1> {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    function(std::integral_constant<int, 0>{}, args...);
  }
};
template <>
struct compile_time_for<0> {
  // 0 loops, do nothing
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {}
};

} // namespace cpu
} // namespace torch_ipex
