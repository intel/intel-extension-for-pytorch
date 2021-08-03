#include <core/Device.h>
#include <core/Stream.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>

#include <utils/Helpers.h>
#include <utils/Settings.h>

#include <cmath>

namespace xpu {
namespace dpcpp {

void parallel_for_setup(
    int64_t n,
    int64_t& tileSize,
    int64_t& rng,
    int64_t& GRange) {
  tileSize = dpcppMaxWorkGroupSize();
  rng = n;
  if (rng == 0) {
    rng = static_cast<int64_t>(1);
  }

  GRange = rng;
  if (tileSize > GRange) {
    tileSize = GRange;
  } else if (GRange > tileSize) {
    int64_t xMode = static_cast<int64_t>(GRange % tileSize);
    if (xMode != 0) {
      GRange += static_cast<int64_t>(tileSize - xMode);
    }
  }
}

void parallel_for_setup(
    int64_t dim0,
    int64_t dim1,
    int64_t& tileSize0,
    int64_t& tileSize1,
    int64_t& rng0,
    int64_t& rng1,
    int64_t& GRange0,
    int64_t& GRange1) {
  int64_t max_workgroup_Size = dpcppMaxWorkGroupSize();
  int64_t pow_of_2 = static_cast<int64_t>(std::log2(max_workgroup_Size));
  tileSize1 =
      static_cast<int64_t>(std::pow(2, static_cast<int64_t>(pow_of_2 / 2)));
  rng1 = dim1;
  if (rng1 == 0) {
    rng1 = static_cast<int64_t>(1);
  }

  GRange1 = rng1;
  if (tileSize1 > GRange1) {
    tileSize1 = GRange1;
  } else if (GRange1 > tileSize1) {
    int64_t xMode = static_cast<int64_t>(GRange1 % tileSize1);
    if (xMode != 0) {
      GRange1 += static_cast<int64_t>(tileSize1 - xMode);
    }
  }

  tileSize0 = static_cast<int64_t>(max_workgroup_Size / tileSize1);
  rng0 = dim0;
  if (rng0 == 0) {
    rng0 = static_cast<int64_t>(1);
  }

  GRange0 = rng0;
  if (tileSize0 > GRange0) {
    tileSize0 = GRange0;
  } else if (GRange0 > tileSize0) {
    int64_t xMode = static_cast<int64_t>(GRange0 % tileSize0);
    if (xMode != 0) {
      GRange0 += static_cast<int64_t>(tileSize0 - xMode);
    }
  }
}

void parallel_for_setup(
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t& tileSize0,
    int64_t& tileSize1,
    int64_t& tileSize2,
    int64_t& rng0,
    int64_t& rng1,
    int64_t& rng2,
    int64_t& GRange0,
    int64_t& GRange1,
    int64_t& GRange2) {
  int64_t max_workgroup_Size = dpcppMaxWorkGroupSize();
  int64_t pow_of_2 = static_cast<int64_t>(std::log2(max_workgroup_Size));
  tileSize2 =
      static_cast<int64_t>(std::pow(2, static_cast<int64_t>(pow_of_2 / 3)));
  rng2 = dim2;
  if (rng2 == 0) {
    rng1 = static_cast<int64_t>(1);
  }

  GRange2 = rng2;
  if (tileSize2 > GRange2) {
    tileSize2 = GRange2;
  } else if (GRange2 > tileSize2) {
    int64_t xMode = static_cast<int64_t>(GRange2 % tileSize2);
    if (xMode != 0)
      GRange2 += static_cast<int64_t>(tileSize2 - xMode);
  }

  pow_of_2 = static_cast<int64_t>(
      std::log2(static_cast<int64_t>(max_workgroup_Size / tileSize2)));
  tileSize1 =
      static_cast<int64_t>(std::pow(2, static_cast<int64_t>(pow_of_2 / 2)));

  rng1 = dim1;
  if (rng1 == 0) {
    rng1 = static_cast<int64_t>(1);
  }

  GRange1 = rng1;
  if (tileSize1 > GRange1) {
    tileSize1 = GRange1;
  } else if (GRange1 > tileSize1) {
    int64_t xMode = static_cast<int64_t>(GRange1 % tileSize1);
    if (xMode != 0) {
      GRange1 += static_cast<int64_t>(tileSize1 - xMode);
    }
  }

  tileSize0 =
      static_cast<int64_t>(max_workgroup_Size / (tileSize1 * tileSize2));
  rng0 = dim0;
  if (rng0 == 0) {
    rng0 = static_cast<int64_t>(1);
  }

  GRange0 = rng0;
  if (tileSize0 > GRange0) {
    tileSize0 = GRange0;
  } else if (GRange0 > tileSize0) {
    int64_t xMode = static_cast<int64_t>(GRange0 % tileSize0);
    if (xMode != 0) {
      GRange0 += static_cast<int64_t>(tileSize0 - xMode);
    }
  }
}

} // namespace dpcpp
} // namespace xpu
