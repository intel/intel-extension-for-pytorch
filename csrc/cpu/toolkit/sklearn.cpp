#include "sklearn.h"
#include <ATen/Dispatch.h>
#ifdef _WIN32
#include <ppl.h>
#define IPEX_PARALLEL_SORT concurrency::parallel_sort
#else
#include <parallel/algorithm>
#define IPEX_PARALLEL_SORT __gnu_parallel::sort
#endif

namespace toolkit {

// This function is semantically equivalent to python lib sklearn toolkit's
// sklearn.metrics.roc_auc_score() & sklearn.metrics.accuracy_score() function.
// But in sklearn, these two function evaluate the auc score and accuracy
// using single thread. We implemented the multi-thread version of them.
template <typename T>
std::vector<double> roc_auc_score_(
    at::Tensor self,
    at::Tensor other,
    int size,
    bool only_score = true) {
  T* actual = self.data_ptr<T>();
  T* prediction = other.data_ptr<T>();
  std::vector<T> predictedRank(size, 0.0);
  int nPos = 0, nNeg = 0;
#pragma omp parallel for reduction(+ : nPos)
  for (int i = 0; i < size; i++)
    nPos += (int)actual[i];

  nNeg = size - nPos;

  std::vector<std::pair<T, int>> v_sort(size);
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    v_sort[i] = std::make_pair(prediction[i], i);
  }

  IPEX_PARALLEL_SORT(v_sort.begin(), v_sort.end(), [](auto& left, auto& right) {
    return left.first < right.first;
  });

  int r = 1;
  int n = 1;
  size_t i = 0;
  while (i < size) {
    size_t j = i;
    while ((j < (v_sort.size() - 1)) &&
           (v_sort[j].first == v_sort[j + 1].first)) {
      j++;
    }
    n = j - i + 1;
    for (size_t j = 0; j < n; ++j) {
      int idx = v_sort[i + j].second;
      predictedRank[idx] = r + ((n - 1) * 0.5);
    }
    r += n;
    i += n;
  }

  double filteredRankSum = 0;
#pragma omp parallel for reduction(+ : filteredRankSum)
  for (size_t i = 0; i < size; ++i) {
    if (actual[i] == 1) {
      filteredRankSum += predictedRank[i];
    }
  }
  double score = (filteredRankSum - ((double)nPos * ((nPos + 1.0) / 2.0))) /
      ((double)nPos * nNeg);
  double log_loss = 0.0;
  double accuracy = 0.0;
  if (only_score == false) {
    double acc = 0.0;
    double loss = 0.0;
#pragma omp parallel for reduction(+ : acc, loss)
    for (int i = 0; i < size; i++) {
      auto rpred = std::roundf(prediction[i]);
      if (actual[i] == rpred)
        acc += 1;
      loss += (actual[i] * std::log(prediction[i])) +
          ((1 - actual[i]) * std::log(1 - prediction[i]));
    }
    accuracy = acc / size;
    log_loss = -loss / size;
  }

  return {score, log_loss, accuracy};
}

std::vector<double> roc_auc_score(at::Tensor self, at::Tensor other) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.dim() == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.dim() == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.dtype() == other.dtype());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.numel() == other.numel());

  return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "roc_auc_score", [&]() {
    return roc_auc_score_<scalar_t>(self, other, self.numel());
  });
}

std::vector<double> roc_auc_score_all(at::Tensor self, at::Tensor other) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.dim() == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.dim() == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.dtype() == other.dtype());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.numel() == other.numel());

  return AT_DISPATCH_FLOATING_TYPES(
      self.scalar_type(), "roc_auc_score_all", [&]() {
        return roc_auc_score_<scalar_t>(self, other, self.numel(), false);
      });
}

} // namespace toolkit
