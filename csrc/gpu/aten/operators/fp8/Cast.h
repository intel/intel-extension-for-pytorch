/*******************************************************************************
 * Copyright (C) 2024 Intel Corporation
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission. This software and the related documents are provided as is,
 * with no express or implied warranties, other than those that are expressly
 * stated in the License.
 *******************************************************************************
 */
#include <ATen/ATen.h>
#include "FP8Utils.h"

namespace at {
namespace AtenIpexTypeXPU {

at::ScalarType convert_to_dtype(int64_t format);

void fp8_quantize_op(
    const Tensor& input,
    Tensor& input_fp8,
    int fp8_format,
    void* amax,
    const void* scale,
    void* scale_inv,
    bool is_SR = false,
    bool is_amax = true,
    bool is_quantize = true);

void fp8_dequantize_op(
    const Tensor& input,
    Tensor& input_c,
    int64_t itype,
    const void* scale_inv,
    bool is_dequantize = true);

} // namespace AtenIpexTypeXPU
} // namespace at
