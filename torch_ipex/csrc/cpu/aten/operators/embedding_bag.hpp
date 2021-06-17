#pragma once

#include <ATen/Tensor.h>
#include <ATen/Parallel.h>
#include <c10/core/ScalarType.h>

#include <vector>


namespace torch_ipex {
    namespace cpu {
        namespace aten {
            namespace embedding_bag {

                at::Tensor embedding_bag_impl(const at::Tensor & weight, const at::Tensor & indices,
                                              const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse,
                                              const at::Tensor & per_sample_weights, bool include_last_offset);

                at::Tensor embedding_bag_int8_impl(const at::Tensor & weight,
                                                   const at::Tensor & indices,
                                                   const at::Tensor & offsets,
                                                   bool include_last_offset);

                template<typename T>
                inline at::Tensor _embedding_bag_index_add_select_fast(const at::Tensor select_indices,
                                                                       const at::Tensor src, const at::Tensor offsets,  bool include_last_offset) {
                    int64_t ddim = src.size(1);
                    auto* src_data = src.data_ptr<T>();
                    int64_t output_size = offsets.numel() - 1;
                    int64_t* offsets_data = offsets.data_ptr<int64_t>();
                    std::vector<int64_t> offsets_include_last;
                    if (!include_last_offset) {
                        output_size = offsets.numel();
                        offsets_include_last.resize(output_size + 1);
                        int64_t* offsets_include_last_data = offsets_include_last.data();
                        int64_t iter_time = (output_size >> 5);
                        int64_t align32_size = (iter_time << 5);
                        int64_t left_size = output_size - align32_size;
                        //std::memcpy(offsets_include_last.data(), offsets_data, sizeof(int64_t) * output_size);
                        at::parallel_for(0, iter_time, 16, [&](int64_t start, int64_t end) {
                            for (int64_t i = start; i < end; i += 1) {
                                auto start_offset = i << 5;
                                move_ker(&offsets_include_last_data[start_offset], &offsets_data[start_offset], 32);
                            }
                        });
                        if (left_size > 0) {
                            move_ker(&offsets_include_last_data[align32_size], &offsets_data[align32_size], left_size);
                        }
                        offsets_include_last[output_size] = select_indices.numel();
                        offsets_data = offsets_include_last.data();
                    }

                    at::Tensor output = at::empty({output_size, src.size(1)}, src.options());
                    auto* output_data = output.data_ptr<T>();
                    auto indices_accessor = select_indices.accessor<int64_t, 1>();
                    at::parallel_for(0, output_size, 16, [&](int64_t start, int64_t end) {
                        for (int64_t i = start; i < end; i++) {
                            auto* out_data_ptr = &output_data[i * ddim];
                            auto inputs_start = offsets_data[i];
                            auto inputs_end = offsets_data[i + 1];
                            if (inputs_start >= inputs_end) {
                                zero_ker((T*)out_data_ptr, ddim);
                            } else {
                                T* select_data_ptr = &src_data[indices_accessor[inputs_start] * ddim];
                                move_ker((T *)out_data_ptr, (T *)select_data_ptr, ddim);
                            }
                            for (int64_t s = (inputs_start + 1); s < inputs_end; s++) {
                                T* select_data_ptr = &src_data[indices_accessor[s] * ddim];
                                add_ker((T *)out_data_ptr, (T *)select_data_ptr, ddim);
                            }
                        }
                    });

                    return output;
                }

                at::Tensor embedding_bag_impl(const at::Tensor & weight, const at::Tensor & indices,
                                              const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse,
                                              const at::Tensor & per_sample_weights, bool include_last_offset);

                at::Tensor embedding_bag_int8_impl(const at::Tensor & weight,
                                                   const at::Tensor & indices,
                                                   const at::Tensor & offsets,
                                                   bool include_last_offset);

                at::Tensor embedding_bag_backward_impl(const at::Tensor & grad, const at::Tensor & indices,
                                                       const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices,
                                                       int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
                                                       const at::Tensor & per_sample_weights);

                at::Tensor embedding_bag_get_offset2bag(const at::Tensor indices, const at::Tensor & offsets, const at::Tensor & offset2bag);

                bool embedding_bag_backward_fast_path_sum(const at::Tensor grad, const at::Tensor indices, const at::Tensor offset2bag, const at::Tensor per_sample_weights, bool scale_grad_by_freq, int64_t mode);

                bool embedding_bag_fast_path_sum(const at::Tensor weight, const at::Tensor per_sample_weights, int64_t mode);

            }  // namespace embedding_bag
        }  // namespace aten
    }  // namespace cpu
}  // namespace torch_ipex
