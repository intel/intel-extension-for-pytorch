#include <core/detail/IndexUtils.h>
#include <vector>

namespace at {
namespace dpcpp {
namespace detail {

struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

int compareSizeAndStride(const void *a, const void *b) {
  const SizeAndStride *aS = (const SizeAndStride *)a;
  const SizeAndStride *bS = (const SizeAndStride *)b;

  if (aS->stride < bS->stride)
    return -1;
  if (aS->stride == bS->stride)
    return 0;
  return 1;
}

bool maybeOverlappingIndices(const Tensor &t) {
  std::vector<SizeAndStride> info(t.dim());
  int dims = t.dim();
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    int64_t size = t.size(i);
    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = t.stride(i);

      if (info[nonSize1Dims].stride < 1) {
        return true;
      }

      ++nonSize1Dims;
    }
  }

  if (nonSize1Dims == 0) {
    return false;
  }

  qsort(info.data(), nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  for (int i = 0; i < (nonSize1Dims - 1); ++i) {
    if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
      return true;
    }
  }

  return false;
}

bool canUse32BitIndexMath(const Tensor &t, int64_t max_elem) {
  int64_t elements = t.numel();

  if (elements == 0) {
    return true;
  }

  if (elements >= max_elem) {
    return false;
  }

  int64_t offset = 0;
  int64_t linearId = elements - 1;

  for (int i = t.dim() - 1; i >= 0; --i) {
    int64_t curDimIndex = linearId % t.size(i);
    int64_t curDimOffset = curDimIndex * t.stride(i);
    offset += curDimOffset;
    linearId /= t.size(i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

} // detail
} // dpcpp
} // at
