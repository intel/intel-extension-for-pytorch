#pragma once
#include <stdint.h>
#include <vector>
#include "aten/aten.hpp"
#include "xsmm/libxsmm_utils.h"
#include "bf16/vec/bf16_vec_kernel.h"
#include "int8/vec/int8_vec_kernel.h"

namespace torch_ipex {
    template <typename T>
    static inline void flat_triangle(const T *in, T *out, size_t size) {
#if defined(IPEX_PROFILE_OP)
        RECORD_FUNCTION("ExtendOps::_flat_triangle", std::vector<c10::IValue>({}));
#endif
        size_t offset = 0;
#pragma unroll(2)
        for (int i = 1; i < size; i++) {
            move_ker(&out[offset], &in[i * size], i);
            offset += i;
        }
    }

    template <typename T>
    static inline void cat(const T *in1, const T *in2, T *out, size_t in1_size,
                           size_t in2_size) {
#if defined(IPEX_PROFILE_OP)
        RECORD_FUNCTION("ExtendOps::_cat", std::vector<c10::IValue>({}));
#endif
        move_ker(out, in1, in1_size);
        move_ker(&out[in1_size], in2, in2_size);
    }

    template <typename T>
    static inline void cat(T *out, const std::vector<T *> &in,
                           const std::vector<uint32_t> &feature_sizes, int64_t bs) {
#if defined(IPEX_PROFILE_OP)
        RECORD_FUNCTION("ExtendOps::_cat_array", std::vector<c10::IValue>({}));
#endif
        size_t offset = 0;
        for (int j = 0; j < feature_sizes.size(); j++) {
            move_ker(&out[offset], &in[j][bs * feature_sizes[j]], feature_sizes[j]);
            offset += feature_sizes[j];
        }
    }

    template <typename T>
    inline at::Tensor _interaction_forward(const std::vector<at::Tensor> &input) {
#if defined(IPEX_PROFILE_OP)
        RECORD_FUNCTION("_interaction_forward", std::vector<c10::IValue>({}));
#endif
        uint32_t total_feature_size = 0;
        int64_t batch_size = input[0].sizes()[0];
        uint32_t vector_size = input[0].sizes()[1];
        uint32_t input_size = input.size();
        std::vector<uint32_t> feature_sizes(input_size);
        std::vector<T *> input_data(input_size);
        for (int i = 0; i < input_size; i++) {
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].device().is_xpu());
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
            feature_sizes[i] = input[i].sizes()[1];
            total_feature_size += input[i].sizes()[1];
            input_data[i] = input[i].data_ptr<T>();
        }
        auto vector_nums = total_feature_size / vector_size;
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
        auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
        auto out_data_line_len = interact_feature_size + vector_size;
        auto tr_vector_size = sizeof(T) == 4 ? vector_size : vector_size / 2;
        auto out = at::empty({batch_size, out_data_line_len}, input[0].options());
        auto out_data = out.data_ptr<T>();

        auto mm_kernel = get_mm_kernel<T>(vector_nums, vector_nums, vector_size);
        auto tr_kernel = get_tr_kernel(tr_vector_size, vector_nums, vector_nums);

        at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
            T cat_buf[vector_nums * vector_size] __attribute__((aligned(64)));
            T tr_buf[vector_nums * vector_size] __attribute__((aligned(64)));
            T mm_buf[vector_nums * vector_nums] __attribute__((aligned(64)));
            for (int64_t i = start; i < end; i++) {
                cat<T>(cat_buf, input_data, feature_sizes, i);
                tr_kernel(cat_buf, &tr_vector_size, tr_buf, &vector_nums);
                mm_kernel((xsmm_dtype<T> *)tr_buf, (xsmm_dtype<T> *)cat_buf,
                          (xsmm_dtype<T> *)mm_buf);
                move_ker(&out_data[i * out_data_line_len], &input_data[0][i * vector_size], vector_size);
                T* flat_buf = (T*)(&out_data[i * out_data_line_len] + vector_size);
                flat_triangle<T>(mm_buf, flat_buf, vector_nums);
            }
        });

        return out;
    }
}
