
#pragma once

#include <omp.h>
#include <cstdint>
#include <utility>
#include "csrc/utils/ipex_op_profile.h"

namespace torch_ipex {
namespace cpu {

template <typename T>
using Key_Value_Weight_Tuple = std::tuple<T, T, float>;
// histogram size per thread
const int HIST_SIZE = 256;

template <typename T>
Key_Value_Weight_Tuple<T>* radix_sort_parallel(
    Key_Value_Weight_Tuple<T>* inp_buf,
    Key_Value_Weight_Tuple<T>* tmp_buf,
    int64_t elements_count,
    int64_t max_value) {
  IPEX_RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
  int maxthreads = omp_get_max_threads();
  alignas(64) int histogram[HIST_SIZE * maxthreads],
      histogram_ps[HIST_SIZE * maxthreads + 1];
  if (max_value == 0)
    return inp_buf;
  int num_bits = sizeof(T) * 8 - __builtin_clz(max_value);
  int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int* local_histogram = &histogram[HIST_SIZE * tid];
    int* local_histogram_ps = &histogram_ps[HIST_SIZE * tid];
    int elements_count_4 = elements_count / 4 * 4;
    Key_Value_Weight_Tuple<T>* input = inp_buf;
    Key_Value_Weight_Tuple<T>* output = tmp_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      /* Step 1: compute histogram
         Reset histogram */
      for (int i = 0; i < HIST_SIZE; i++)
        local_histogram[i] = 0;

#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        T val_1 = std::get<0>(input[i]);
        T val_2 = std::get<0>(input[i + 1]);
        T val_3 = std::get<0>(input[i + 2]);
        T val_4 = std::get<0>(input[i + 3]);

        local_histogram[(val_1 >> (pass * 8)) & 0xFF]++;
        local_histogram[(val_2 >> (pass * 8)) & 0xFF]++;
        local_histogram[(val_3 >> (pass * 8)) & 0xFF]++;
        local_histogram[(val_4 >> (pass * 8)) & 0xFF]++;
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          T val = std::get<0>(input[i]);
          local_histogram[(val >> (pass * 8)) & 0xFF]++;
        }
      }
#pragma omp barrier
      /* Step 2: prefix sum */
      if (tid == 0) {
        int sum = 0, prev_sum = 0;
        for (int bins = 0; bins < HIST_SIZE; bins++)
          for (int t = 0; t < nthreads; t++) {
            sum += histogram[t * HIST_SIZE + bins];
            histogram_ps[t * HIST_SIZE + bins] = prev_sum;
            prev_sum = sum;
          }
        histogram_ps[HIST_SIZE * nthreads] = prev_sum;
        if (prev_sum != elements_count) {
          /* printf("Error1!\n"); exit(123); */
        }
      }
#pragma omp barrier

      /* Step 3: scatter */
#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        T val_1 = std::get<0>(input[i]);
        T val_2 = std::get<0>(input[i + 1]);
        T val_3 = std::get<0>(input[i + 2]);
        T val_4 = std::get<0>(input[i + 3]);
        T bin_1 = (val_1 >> (pass * 8)) & 0xFF;
        T bin_2 = (val_2 >> (pass * 8)) & 0xFF;
        T bin_3 = (val_3 >> (pass * 8)) & 0xFF;
        T bin_4 = (val_4 >> (pass * 8)) & 0xFF;
        int pos;
        pos = local_histogram_ps[bin_1]++;
        output[pos] = input[i];
        pos = local_histogram_ps[bin_2]++;
        output[pos] = input[i + 1];
        pos = local_histogram_ps[bin_3]++;
        output[pos] = input[i + 2];
        pos = local_histogram_ps[bin_4]++;
        output[pos] = input[i + 3];
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          T val = std::get<0>(input[i]);
          int pos = local_histogram_ps[(val >> (pass * 8)) & 0xFF]++;
          output[pos] = input[i];
        }
      }

      Key_Value_Weight_Tuple<T>* temp = input;
      input = output;
      output = temp;
#pragma omp barrier
    }
  }
  return (num_passes % 2 == 0 ? inp_buf : tmp_buf);
}

} // namespace cpu
} // namespace torch_ipex
