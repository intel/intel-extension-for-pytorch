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

/// @file
/// C++ API

#pragma once

#include <common/core/base_ops.hpp>
#include <common/core/base_types.hpp>
#include <common/core/common.hpp>
#include <common/core/math_general.hpp>

namespace gpu::xetla {

/// @addtogroup xetla_core_conv
/// @{

// template <typename T_dst, typename T_src, int N>
//__XETLA_API xetla_vector<T_dst, N> xetla_cvt(xetla_vector<T_src, N> src) {}

/// @brief xetla explicit data conversion for standard data
/// types(integer,float,half)
/// @tparam T_dst is the destination data type.
/// @tparam T_src is the source data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    !(is_internal_type<T_dst>::value) && !(is_internal_type<T_src>::value),
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
  xetla_vector<T_dst, N> dst = src;
  return dst;
}

/// @brief xetla explicit data conversion, fp32->bf16.
/// @tparam T_dst is the float32 data type.
/// @tparam T_src is the bfloat16 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, bf16>::value && std::is_same<T_src, float>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
  // xetla_vector<int32_t, N> a = src.template bit_cast_view<int32_t>();
  // xetla_vector<int16_t, N> c = a >> 16;
  // return c.xetla_format<bf16>();
  xetla_vector<T_dst, N> dst = src;
  return dst;
}

/// @brief xetla explicit data conversion, bf16->fp16.
/// @tparam T_dst is the float16 data type.
/// @tparam T_src is the bfloat16 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, fp16>::value && std::is_same<T_src, bf16>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
  xetla_vector<T_dst, N> dst = src;
  return dst;
}

/// @brief xetla explicit data conversion, bf16->fp32.
/// @tparam T_dst is the bfloat16 data type.
/// @tparam T_src is the float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, float>::value && std::is_same<T_src, bf16>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
  // xetla_vector<int16_t, N> a = src.template bit_cast_view<int16_t>();
  // xetla_vector<int32_t, N> b = a;
  // auto c = b << 16;
  // return c.xetla_format<float>();
  xetla_vector<T_dst, N> dst = src;
  return dst;
}

/// @brief xetla explicit data conversion, fp32->tf32.
/// @tparam T_dst is the float32 data type.
/// @tparam T_src is the tensor_float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, tf32>::value && std::is_same<T_src, float>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
  xetla_vector<T_dst, N> dst = src;
  return dst;
}

/// @brief xetla explicit data conversion, tf32->fp32.
/// @tparam T_dst is the tensor_float32 data type.
/// @tparam T_src is the float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, float>::value && std::is_same<T_src, tf32>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
  return src.xetla_format<float>();
}

/// @brief xetpp explicit data conversion with scaling, int32->fp16.
/// @tparam T_dst is the half data type.
/// @tparam T_src is the int32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, fp16>::value && std::is_same<T_src, int32_t>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src, float scaling_value) {
  xetla_vector<T_dst, N> dst = scaling_value * src;
  return dst;
}

/// @brief xetpp explicit data conversion with re-quantization, int32->int8.
/// @tparam T_dst is the int32 data type.
/// @tparam T_src is the int8 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, int8_t>::value && std::is_same<T_src, int32_t>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src, float scaling_value) {
  auto tmp = xetla_rnde<float>(scaling_value * src);
  auto dst = __ESIMD_NS::saturate<T_dst, float, N>(tmp);
  return dst;
}

/// @brief xetpp explicit data conversion with scaling and quantization,
/// float32->int8.
/// @tparam T_dst is the int8 data type.
/// @tparam T_src is the float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, int8_t>::value && std::is_same<T_src, float>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src, float scaling_value) {
  auto tmp = xetla_rnde<float>(scaling_value * src);
  auto dst = __ESIMD_NS::saturate<T_dst, float, N>(tmp);
  return dst;
}

/// @brief xetla explicit data conversion, fp16->mx_fp4.
/// @tparam T_src is the float16 data type.
/// @tparam T_dst is the mx_fp4(E2M1) data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, mx_fp4>::value && std::is_same<T_src, fp16>::value,
    xetla_vector<T_dst, N / get_packed_num<T_dst>::value>>
xetla_cvt(
    xetla_vector<T_src, N> src,
    xetla_vector<uint16_t, N> rand_vec = 0x100) {
  /*********prepare, 4 instructions******/
  xetla_vector<T_src, N> src_abs;
  src_abs.xetla_format<uint16_t>() = src.xetla_format<uint16_t>() & 0x7fff;
  xetla_vector<uint16_t, N> sign =
      (src.xetla_format<uint16_t>() & 0x8000) >> 12;
  // only compare 9bits mantissa
  rand_vec = rand_vec & 0x1ff;

  xetla_vector<T_src, N> src_abs_carried;
  // if src_abs is 0.3, then it only has 30% possibility to carry (rand_vec
  // >=0.7) inf If it is 0_11110_1xxxxxxxxx, and carry, then it will become
  // 0_11111_0, and finally round to 6 If it is 0_11111_0000000000, no  carry,
  // then it will become 0_11111_0, and finally round to 6 nan If it is
  // 0_11111_1xxxxxxxxx, and carry, then it will become 1_00000_0, and finally
  // round to 0 If it is 0_11111_1xxxxxxxxx, no carry,  then it will become
  // 0_11111_1, and finally round to 6 If it is 0_11111_0xxxxxxxxx, and carry,
  // then it will become 0_11111_1, and finally round to 6 If it is
  // 0_11111_0xxxxxxxxx, no  carry, then it will become 0_11111_0, and finally
  // round to 4 subnormal If it is 0_00000_1xxxxxxxxx, and carry, then it will
  // become 0_00001_0, and finally round to 0

  // clean the low 9bits to make sure inf will not become nan
  /*********rounding, 1 instruction******/
  src_abs_carried.xetla_format<uint16_t>() =
      src_abs.xetla_format<uint16_t>() + rand_vec.xetla_format<uint16_t>();

  // dst = 0_00_0
  // if src_abs_carried >= 0.5(0_01110_0) => dst = 0_00_1
  // if src_abs_carried >= 1  (0_01111_0) => dst = 0_01_0
  // if src_abs_carried >= 1.5(0_01111_1) => dst = 0_01_1
  // if src_abs_carried >= 2  (0_10000_0) => dst = 0_10_0
  // if src_abs_carried >= 3  (0_10000_1) => dst = 0_10_1
  // if src_abs_carried >= 4  (0_10001_0) => dst = 0_11_0
  // if src_abs_carried >= 6  (0_10001_1) => dst = 0_11_1
  /*********common path, 5 instructions******/
  xetla_vector<uint16_t, N> dst =
      ((src_abs_carried.xetla_format<uint16_t>() >> 9) & 3) |
      ((src_abs_carried.xetla_format<uint16_t>() >> 12) & 4);

  /*********handle conner case, 6 instructions******/
  // if src_abs_carried is nan, there flags should all be false, only go with
  // common path still srnd for subnormal case
  xetla_mask<N> zero_flag = src_abs_carried < 0.5;
  xetla_mask<N> subnormal_flag = src_abs_carried < 1;
  xetla_mask<N> saturate_flag = src_abs >= 6;
  dst.xetla_merge(0x7, saturate_flag);
  // subnormal_flag should prior to zero_flag
  dst.xetla_merge(0x1, subnormal_flag);
  dst.xetla_merge(0x0, zero_flag);

  /*********add sign bit, 1 instructions******/
  dst |= sign;

  /*********pack data, 7 instructions******/
  // T_dst is uint8_t, get_packed_num<T_dst>::value is 2
  xetla_vector<T_dst, N / get_packed_num<T_dst>::value> out;
  auto out_u16 = out.xetla_format<uint16_t>();
  out_u16 = dst.xetla_select<N / 4, 4>(0);
  out_u16 |= dst.xetla_select<N / 4, 4>(1) << 4;
  out_u16 |= dst.xetla_select<N / 4, 4>(2) << 8;
  out_u16 |= dst.xetla_select<N / 4, 4>(3) << 12;

  return out;
}

/// @brief xetla explicit data conversion, fp32->mx_fp4.
/// @tparam T_src is the float32 data type.
/// @tparam T_dst is the mx_fp4(E2M1) data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, mx_fp4>::value && std::is_same<T_src, float>::value,
    xetla_vector<T_dst, N / get_packed_num<T_dst>::value>>
xetla_cvt(
    xetla_vector<T_src, N> src,
    xetla_vector<uint16_t, N> rand_vec = 0x100) {
  xetla_vector<fp16, N> src_f16 = src;
  return xetla_cvt<T_dst, fp16>(src_f16, rand_vec);
}

/// @brief xetla explicit data conversion, same type.
/// @tparam T_dst is the dst data type.
/// @tparam T_src is the src data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<
    std::is_same<T_dst, T_src>::value && is_internal_type<T_src>::value,
    xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
  return src;
}

/// @} xetla_core_conv

} // namespace gpu::xetla
