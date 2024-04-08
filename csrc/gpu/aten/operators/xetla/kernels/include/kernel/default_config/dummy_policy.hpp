/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include <kernel/default_config/common.hpp>

namespace gpu::xetla {
template <param_optimizer_tag tag_, typename dict_t_, typename... candidates_t>
struct dummy_optimizer : param_optimizer_base {
  struct impl {
    enum class eval_tag : uint8_t { et_type, et_delta, et_linear, et_square };

    template <
        auto key_,
        typename T,
        typename U,
        eval_tag eval_tag_,
        typename = void>
    struct param_distance_eval_fcn;

    template <auto key_, typename T, typename U>
    struct param_distance_eval_fcn<key_, T, U, eval_tag::et_type> {
      static constexpr int value = []() constexpr {
        return (std::is_same<T, U>::value) ? 0 : 1;
      }
      ();
    };

    template <auto key_, typename T, typename U>
    struct param_distance_eval_fcn<key_, T, U, eval_tag::et_delta> {
      static constexpr int value = []() constexpr {
        auto l = T::template find_elem_v<key_>;
        auto r = U::template find_elem_v<key_>;
        return (l == r) ? 0 : 1;
      }
      ();
    };

    template <typename T>
    static constexpr T const_max(const T& l, const T& r) {
      return (l > r) ? l : r;
    }

    template <typename T>
    static constexpr T const_min(const T& l, const T& r) {
      return (l > r) ? r : l;
    }

    template <auto key_, typename T, typename U>
    struct param_distance_eval_fcn<key_, T, U, eval_tag::et_linear> {
      static constexpr int value = []() constexpr {
        auto l = T::template find_elem_v<key_>;
        auto r = U::template find_elem_v<key_>;
        return (const_max(l, r) - const_min(l, r));
      }
      ();
    };

    template <auto key_, typename T, typename U>
    struct param_distance_eval_fcn<key_, T, U, eval_tag::et_square> {
      static constexpr int value = []() constexpr {
        auto l = T::template find_elem_v<key_>;
        auto r = U::template find_elem_v<key_>;
        auto ret = (l - r);
        return ret * ret;
      }
      ();
    };

    template <auto key_, typename T, typename U>
    struct param_distance_eval {
      static constexpr int value = []() constexpr {
        using eval_fcn = typename std::conditional<
            ((key_ == tune_key::data_type_a) ||
             (key_ == tune_key::data_type_b) ||
             (key_ == tune_key::data_type_c) ||
             (key_ == tune_key::data_type_acc) ||
             (key_ == tune_key::epilogue_policy)),
            param_distance_eval_fcn<key_, T, U, eval_tag::et_type>,
            typename std::conditional<
                ((key_ == tune_key::global_kslicing_ratio) ||
                 (key_ == tune_key::local_kslicing_ratio) ||
                 (key_ == tune_key::wg_tile_k) ||
                 (key_ == tune_key::prefetch_distance) ||
                 (key_ == tune_key::periodic_sync_interval)),
                param_distance_eval_fcn<key_, T, U, eval_tag::et_linear>,
                param_distance_eval_fcn<key_, T, U, eval_tag::et_delta>>::
                type>::type;
        switch (key_) {
          case tune_key::wg_tile_k:
            return 10 * eval_fcn::value;
          case tune_key::global_kslicing_ratio:
          case tune_key::local_kslicing_ratio:
            return 1000 * eval_fcn::value;
          case tune_key::data_type_acc:
            return 10 * eval_fcn::value;
          default:
            return 10000000 * eval_fcn::value;
        }
        return 0;
      }
      ();
    };

    template <typename T, typename U>
    struct param_distance_eval<tune_key::wg_tile_shape, T, U> {
      static constexpr int value = []() constexpr {
        using T_L =
            typename T::template find_elem_t<tune_key::wg_tile_shape>::type;
        using T_R =
            typename T::template find_elem_t<tune_key::wg_tile_shape>::type;

        int l_x = T_L::template dim<0>();
        int l_y = T_L::template dim<1>();

        int r_x = T_R::template dim<0>();
        int r_y = T_R::template dim<1>();

        return (const_max(l_x, r_x) - const_min(l_x, r_x)) +
            (const_max(l_y, r_y) - const_min(l_y, r_y));
      }
      ();
    };

    template <typename T, typename U>
    struct param_distance_eval<tune_key::sg_tile_shape, T, U> {
      static constexpr int value = []() constexpr {
        using T_L =
            typename T::template find_elem_t<tune_key::sg_tile_shape>::type;
        using T_R =
            typename T::template find_elem_t<tune_key::sg_tile_shape>::type;

        int l_x = T_L::template dim<0>();
        int l_y = T_L::template dim<1>();

        int r_x = T_R::template dim<0>();
        int r_y = T_R::template dim<1>();

        return (const_max(l_x, r_x) - const_min(l_x, r_x)) +
            (const_max(l_y, r_y) - const_min(l_y, r_y));
      }
      ();
    };

    template <typename T, typename U>
    struct param_distance {
      static constexpr int value = []() constexpr {
        int sum = 0;
        sum += param_distance_eval<tune_key::data_type_a, T, U>::value;
        sum += param_distance_eval<tune_key::memory_layout_a, T, U>::value;
        sum += param_distance_eval<tune_key::memory_alignment_a, T, U>::value;
        sum += param_distance_eval<tune_key::data_type_b, T, U>::value;
        sum += param_distance_eval<tune_key::memory_layout_b, T, U>::value;
        sum += param_distance_eval<tune_key::memory_alignment_b, T, U>::value;
        sum += param_distance_eval<tune_key::data_type_c, T, U>::value;
        sum += param_distance_eval<tune_key::memory_layout_c, T, U>::value;
        sum += param_distance_eval<tune_key::memory_alignment_c, T, U>::value;
        sum += param_distance_eval<tune_key::data_type_acc, T, U>::value;
        if constexpr (tag_ == param_optimizer_tag::work_group) {
          sum += param_distance_eval<tune_key::memory_space_a, T, U>::value;
          sum += param_distance_eval<tune_key::memory_space_b, T, U>::value;
          sum += param_distance_eval<tune_key::memory_space_c, T, U>::value;
        }
        if constexpr (tag_ == param_optimizer_tag::kernel) {
          sum +=
              param_distance_eval<tune_key::global_kslicing_ratio, T, U>::value;
          sum +=
              param_distance_eval<tune_key::local_kslicing_ratio, T, U>::value;
        }
        sum += param_distance_eval<tune_key::wg_tile_shape, T, U>::value;
        sum += param_distance_eval<tune_key::wg_tile_k, T, U>::value;
        if constexpr (tag_ == param_optimizer_tag::work_group) {
          sum += param_distance_eval<tune_key::sg_tile_shape, T, U>::value;
          sum += param_distance_eval<tune_key::pre_processing, T, U>::value;
        }
        sum += param_distance_eval<tune_key::prefetch_distance, T, U>::value;
        sum +=
            param_distance_eval<tune_key::periodic_sync_interval, T, U>::value;
        sum += param_distance_eval<tune_key::mma_engine, T, U>::value;
        sum += param_distance_eval<tune_key::gpu_arch, T, U>::value;
        sum += param_distance_eval<tune_key::epilogue_policy, T, U>::value;
        if constexpr (tag_ == param_optimizer_tag::kernel) {
          sum += param_distance_eval<tune_key::dispatch_policy, T, U>::value;
          sum +=
              param_distance_eval<tune_key::group_swizzle_policy, T, U>::value;
        }
        return sum;
      }
      ();
    };

    template <int opt_val_, typename opt_t_, typename... elems>
    struct finder_impl;

    template <int opt_val_, typename opt_t_>
    struct finder_impl<opt_val_, opt_t_> {
      using type = opt_t_;
      static constexpr int value = opt_val_;
    };

    template <int opt_val_, typename opt_t_, typename elem_, typename... elems>
    struct finder_impl<opt_val_, opt_t_, elem_, elems...> {
      static constexpr int can_val = param_distance<dict_t_, elem_>::value;
      using cur_opt_t =
          typename std::conditional<(can_val < opt_val_), elem_, opt_t_>::type;
      static constexpr int cur_opt_val = const_min(opt_val_, can_val);

      using nxt_result = finder_impl<cur_opt_val, cur_opt_t, elems...>;

      using type = typename nxt_result::type;
      static constexpr int value = nxt_result::value;
    };

    template <typename opt_t_, typename... elems>
    struct finder_impl_helper : finder_impl<
                                    param_distance<dict_t_, opt_t_>::value,
                                    opt_t_,
                                    elems...> {};

    using type = typename finder_impl_helper<candidates_t...>::type;
    using fallback_type = fallback_optimizer<dict_t_, type>;
  };
  static constexpr bool use_fallback =
      !(param_optimizer_base::
            template valid_attribute_v<dict_t_, typename impl::type>);
  using type = typename std::
      conditional<use_fallback, typename impl::fallback_type, impl>::type::type;
};

} // namespace gpu::xetla
