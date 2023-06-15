#include "../../GEMM.h"
#include "hgemm_splitk.h"

namespace xpu {
namespace xetla {

void gemm(
    sycl::queue& queue,
    float* acc,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  if (m == 1 && n == 4096 && k == 4096)
    hgemm_splitk<
        sycl::half,
        /*WG_M*/ 8,
        /*WG_N*/ 32,
        /*SG_M*/ 8,
        /*SG_N*/ 16,
        /*KS*/ 8,
        /*KN*/ 16>(queue, acc, a, b, m, n, k);
  else if (m == 1 && n == 4096 && k == 16384)
    hgemm_splitk<
        sycl::half,
        /*WG_M*/ 8,
        /*WG_N*/ 128,
        /*SG_M*/ 8,
        /*SG_N*/ 16,
        /*KS*/ 8,
        /*KN*/ 64>(queue, acc, a, b, m, n, k);
  else if (m == 1 && n == 16384 && k == 4096)
    hgemm_splitk<
        sycl::half,
        /*WG_M*/ 8,
        /*WG_N*/ 32,
        /*SG_M*/ 8,
        /*SG_N*/ 16,
        /*KS*/ 8,
        /*KN*/ 16>(queue, acc, a, b, m, n, k);
  else if (m == 1 && n == 32000 && k == 4096)
    hgemm_splitk<
        sycl::half,
        /*WG_M*/ 8,
        /*WG_N*/ 32,
        /*SG_M*/ 8,
        /*SG_N*/ 16,
        /*KS*/ 8,
        /*KN*/ 16>(queue, acc, a, b, m, n, k);
}

} // namespace xetla
} // namespace xpu
