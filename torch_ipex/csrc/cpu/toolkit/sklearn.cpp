#include "sklearn.h"
#include <vector>
#include <type_traits>
#include <parallel/algorithm>

namespace toolkit {

template <typename T>
std::vector<double> roc_auc_score_(at::Tensor self, at::Tensor other, int size) {
  T *actual = self.data_ptr<T>();
  T *prediction = other.data_ptr<T>();
  std::vector<T> predictedRank(size, 0.0);
  int nPos = 0, nNeg = 0;
#pragma omp parallel for reduction(+:nPos)
  for(int i = 0; i < size; i++)
    nPos += (int)actual[i];

  nNeg = size - nPos;
  double acc = 0.0;
  double loss = 0.0;
#pragma omp parallel for reduction(+:acc,loss)
  for(int i = 0; i < size; i++) {
    auto rpred = std::roundf(prediction[i]);
    if(actual[i] == rpred) acc += 1;
    loss += (actual[i] * std::log(prediction[i])) + ((1 - actual[i]) * std::log(1 - prediction[i]));
  }

  double accuracy = acc / size;
  double log_loss = -loss / size;

  std::vector<std::pair<T, int> > v_sort(size);
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    v_sort[i] = std::make_pair(prediction[i], i);
  }

  __gnu_parallel::sort(v_sort.begin(), v_sort.end(), [](auto &left, auto &right) {
    return left.first < right.first;
  });

  int r = 1;
  int n = 1;
  size_t i = 0;
  while (i < size) {
    size_t j = i;
    while ((j < (v_sort.size() - 1)) && (v_sort[j].first == v_sort[j + 1].first)) {
      j++;
    }
    n = j - i + 1;
    for (size_t j = 0; j < n; ++j) {
      int idx = v_sort[i+j].second;
      predictedRank[idx] = r + ((n - 1) * 0.5);
    }
    r += n;
    i += n;
  }

  double filteredRankSum = 0;
#pragma omp parallel for reduction(+:filteredRankSum)
  for (size_t i = 0; i < size; ++i) {
    if (actual[i] == 1) {
      filteredRankSum += predictedRank[i];
    }
  }
  double score = (filteredRankSum - ((double)nPos * ((nPos + 1.0) / 2.0))) / ((double)nPos * nNeg);
  return {score, log_loss, accuracy};
}

std::vector<double> roc_auc_score(at::Tensor self, at::Tensor other) {
  if (self.dim() != 1 || other.dim() != 1) {
    throw std::runtime_error("Dims of all inputs must be 1");
  }

  if (self.dtype() != other.dtype()) {
    throw std::runtime_error("DataType of all inputs must be consistent");
  }

  if (self.numel() != other.numel()) {
    throw std::runtime_error("Shapes of all inputs must be consistent");
  }

  return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "roc_auc_score", [&]() {
    return roc_auc_score_<scalar_t>(self, other, self.numel());
  });
}

}
