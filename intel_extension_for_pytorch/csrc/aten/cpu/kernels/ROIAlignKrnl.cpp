// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <csrc/aten/cpu/ROIAlign.h>
#include <torch/library.h>
#include "csrc/autocast/autocast_mode.h"
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

// use float as accumulation type for BFloat16
template <typename scalar_t>
struct AccType {
  using type = scalar_t;
};
template <>
struct AccType<at::BFloat16> {
  using type = float;
};

namespace torch_ipex {
namespace cpu {

namespace {

// This helper computes the interpolation weights (w1, w2...) for every sampling
// point of a given box. There are pool_height * pool_width * roi_bin_grid_h *
// roi_bin_grid_w such sampling points.
//
// The weights (w1, w2...) are computed as the areas in this figure:
// https://en.wikipedia.org/wiki/Bilinear_interpolation#/media/File:Bilinear_interpolation_visualisation.svg
// and pos1, pos2 etc correspond to the indices of their respective pixels.
//
// Note: the weights and indices are shared across all channels, which is why
// they are pre-calculated prior to the main loop in the RoIAlign kernel.
// implementation taken from Caffe2
template <typename T>
void pre_calc_for_bilinear_interpolate(
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T, typename ACC_T>
inline void roi_align_single_framework_forward(
    const T* input,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* output) {
  for (int c = 0; c < channels; c++) {
    const T* offset_input = input + c * height * width;
    int pre_calc_index = 0;

    for (int ph = 0; ph < pooled_height; ph++) {
      for (int pw = 0; pw < pooled_width; pw++) {
        int index = c * pooled_height * pooled_width + ph * pooled_width + pw;

        ACC_T output_val = 0.;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            PreCalc<ACC_T> pc = pre_calc[pre_calc_index];
            output_val += pc.w1 * static_cast<ACC_T>(offset_input[pc.pos1]) +
                pc.w2 * static_cast<ACC_T>(offset_input[pc.pos2]) +
                pc.w3 * static_cast<ACC_T>(offset_input[pc.pos3]) +
                pc.w4 * static_cast<ACC_T>(offset_input[pc.pos4]);

            pre_calc_index += 1;
          }
        }
        output_val /= count; // Average pooling

        output[index] = output_val;
      } // for pw
    } // for ph
  } // for c
}

template <typename T, typename ACC_T>
inline void roi_align_single_framework_channels_last_forward(
    const T* input,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* output) {
  // for 'normal' size of channels, should be L1 fit;
  // otherwise consider blocking on channels.
  using Vec = at::vec::Vectorized<T>;

  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      T* out = output + (ph * pooled_width + pw) * channels;

      // pass I: zero the out lane
      int64_t d1 = 0;
      for (; d1 < channels - (channels % Vec::size()); d1 += Vec::size()) {
        Vec out_vec = Vec(T(0));
        out_vec.store(out + d1);
      }
      // TODO: optimize with masked intrinsics.
      for (; d1 < channels; d1++) {
        out[d1] = T(0);
      }

      // pass II: do accumulation
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          PreCalc<ACC_T> pc = pre_calc[pre_calc_index];
          const T* in1 = input + pc.pos1 * channels;
          const T* in2 = input + pc.pos2 * channels;
          const T* in3 = input + pc.pos3 * channels;
          const T* in4 = input + pc.pos4 * channels;

          Vec w1_vec = Vec(pc.w1);
          Vec w2_vec = Vec(pc.w2);
          Vec w3_vec = Vec(pc.w3);
          Vec w4_vec = Vec(pc.w4);
          int64_t d2 = 0;
          // TODO: optimize with FMA.
          for (; d2 < channels - (channels % Vec::size()); d2 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d2) + w1_vec * Vec::loadu(in1 + d2) +
                w2_vec * Vec::loadu(in2 + d2) + w3_vec * Vec::loadu(in3 + d2) +
                w4_vec * Vec::loadu(in4 + d2);
            out_vec.store(out + d2);
          }
          // TODO: optimize with masked intrinsics.
          for (; d2 < channels; d2++) {
            out[d2] += pc.w1 * in1[d2] + pc.w2 * in2[d2] + pc.w3 * in3[d2] +
                pc.w4 * in4[d2];
          }
          pre_calc_index += 1;
        }
      }

      // pass III: do average
      int64_t d3 = 0;
      Vec count_vec = Vec(count);
      for (; d3 < channels - (channels % Vec::size()); d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / count_vec;
        out_vec.store(out + d3);
      }
      // TODO: optimize with masked intrinsics.
      for (; d3 < channels; d3++) {
        out[d3] /= static_cast<T>(count);
      }
    } // for pw
  } // for ph
}

template <>
inline void roi_align_single_framework_channels_last_forward<
    at::BFloat16,
    float>(
    const at::BFloat16* input,
    const float count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<float>>& pre_calc,
    at::BFloat16* output) {
  // for 'normal' size of channels, should be L1 fit;
  // otherwise consider blocking on channels.
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  // temp buffer for sum, use float as accumulation type
  // can't reuse output buffer to store sum since it is BFloat16
  std::unique_ptr<float[]> sum_arr(new float[channels]);
  float* sum = sum_arr.get();

  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      at::BFloat16* out = output + (ph * pooled_width + pw) * channels;

      // pass I: zero the sum lane
      int64_t d1 = 0;
      for (; d1 < channels - (channels % fVec::size()); d1 += fVec::size()) {
        fVec sum_fvec = fVec(float(0));
        sum_fvec.store(sum + d1);
      }
      // TODO: optimize with masked intrinsics.
      for (; d1 < channels; d1++) {
        sum[d1] = float(0);
      }

      // pass II: do accumulation
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          PreCalc<float> pc = pre_calc[pre_calc_index];
          const at::BFloat16* in1 = input + pc.pos1 * channels;
          const at::BFloat16* in2 = input + pc.pos2 * channels;
          const at::BFloat16* in3 = input + pc.pos3 * channels;
          const at::BFloat16* in4 = input + pc.pos4 * channels;

          fVec w1_fvec = fVec(pc.w1);
          fVec w2_fvec = fVec(pc.w2);
          fVec w3_fvec = fVec(pc.w3);
          fVec w4_fvec = fVec(pc.w4);
          int64_t d2 = 0;
          for (; d2 < channels - (channels % bVec::size());
               d2 += bVec::size()) {
            bVec in1_bvec = bVec::loadu(in1 + d2);
            bVec in2_bvec = bVec::loadu(in2 + d2);
            bVec in3_bvec = bVec::loadu(in3 + d2);
            bVec in4_bvec = bVec::loadu(in4 + d2);
            fVec in1_fvec0, in1_fvec1, in2_fvec0, in2_fvec1, in3_fvec0,
                in3_fvec1, in4_fvec0, in4_fvec1;
            std::tie(in1_fvec0, in1_fvec1) = convert_bfloat16_float(in1_bvec);
            std::tie(in2_fvec0, in2_fvec1) = convert_bfloat16_float(in2_bvec);
            std::tie(in3_fvec0, in3_fvec1) = convert_bfloat16_float(in3_bvec);
            std::tie(in4_fvec0, in4_fvec1) = convert_bfloat16_float(in4_bvec);
            // TODO: optimize with FMA.
            fVec sum_fvec0 = fVec::loadu(sum + d2) + w1_fvec * in1_fvec0 +
                w2_fvec * in2_fvec0 + w3_fvec * in3_fvec0 + w4_fvec * in4_fvec0;
            fVec sum_fvec1 = fVec::loadu(sum + d2 + fVec::size()) +
                w1_fvec * in1_fvec1 + w2_fvec * in2_fvec1 +
                w3_fvec * in3_fvec1 + w4_fvec * in4_fvec1;
            sum_fvec0.store(sum + d2);
            sum_fvec1.store(sum + d2 + fVec::size());
          }
          // TODO: optimize with masked intrinsics.
          for (; d2 < channels; d2++) {
            sum[d2] += pc.w1 * static_cast<float>(in1[d2]) +
                pc.w2 * static_cast<float>(in2[d2]) +
                pc.w3 * static_cast<float>(in3[d2]) +
                pc.w4 * static_cast<float>(in4[d2]);
          }
          pre_calc_index += 1;
        }
      }

      // pass III: do average
      int64_t d3 = 0;
      fVec count_fvec = fVec(count);
      for (; d3 < channels - (channels % bVec::size()); d3 += bVec::size()) {
        fVec out_fvec0 = fVec::loadu(sum + d3) / count_fvec;
        fVec out_fvec1 = fVec::loadu(sum + d3 + fVec::size()) / count_fvec;

        bVec out_bvec = convert_float_bfloat16(out_fvec0, out_fvec1);
        out_bvec.store(out + d3);
      }
      // TODO: optimize with masked intrinsics.
      for (; d3 < channels; d3++) {
        out[d3] = static_cast<at::BFloat16>(sum[d3] / count);
      }
    } // for pw
  } // for ph
}

template <typename T, typename ACC_T>
void roi_align_forward_kernel_body(
    int n_rois,
    const T* input,
    const ACC_T& spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const ACC_T* rois,
    T* output,
    bool is_channels_last) {
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  at::parallel_for(0, n_rois, 1, [&](int begin, int end) {
    for (int n = begin; n < end; n++) {
      const ACC_T* offset_rois = rois + n * 5;
      int roi_batch_ind = offset_rois[0];

      // Do not using rounding; this implementation detail is critical
      ACC_T offset = aligned ? (ACC_T)0.5 : (ACC_T)0.0;
      ACC_T roi_start_w = offset_rois[1] * spatial_scale - offset;
      ACC_T roi_start_h = offset_rois[2] * spatial_scale - offset;
      ACC_T roi_end_w = offset_rois[3] * spatial_scale - offset;
      ACC_T roi_end_h = offset_rois[4] * spatial_scale - offset;

      ACC_T roi_width = roi_end_w - roi_start_w;
      ACC_T roi_height = roi_end_h - roi_start_h;
      if (!aligned) {
        // Force malformed ROIs to be 1x1
        roi_width = std::max(roi_width, (ACC_T)1.);
        roi_height = std::max(roi_height, (ACC_T)1.);
      }

      ACC_T bin_size_h =
          static_cast<ACC_T>(roi_height) / static_cast<ACC_T>(pooled_height);
      ACC_T bin_size_w =
          static_cast<ACC_T>(roi_width) / static_cast<ACC_T>(pooled_width);

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
          ? sampling_ratio
          : ceil(roi_height / pooled_height); // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
          ? sampling_ratio
          : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      // When the grid is empty, output zeros.
      const ACC_T count =
          std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

      // we want to precalculate indices and weights shared by all channels,
      // this is the key point of optimization
      std::vector<PreCalc<ACC_T>> pre_calc(
          roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
      pre_calc_for_bilinear_interpolate(
          height,
          width,
          pooled_height,
          pooled_width,
          roi_start_h,
          roi_start_w,
          bin_size_h,
          bin_size_w,
          roi_bin_grid_h,
          roi_bin_grid_w,
          pre_calc);

      if (is_channels_last) {
        roi_align_single_framework_channels_last_forward<T, ACC_T>(
            input + roi_batch_ind * height * width * channels,
            count,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            pre_calc,
            output + n * pooled_width * pooled_height * channels);
      } else {
        roi_align_single_framework_forward<T, ACC_T>(
            input + roi_batch_ind * channels * height * width,
            count,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            pre_calc,
            output + n * channels * pooled_width * pooled_height);
      }
    } // for n
  });
}

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename T, typename ACC_T>
inline void roi_align_single_framework_backward(
    const T* grad_output,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* grad_input) {
  for (int c = 0; c < channels; c++) {
    T* offset_grad_input = grad_input + c * height * width;
    const T* offset_grad_output =
        grad_output + c * pooled_height * pooled_width;
    int pre_calc_index = 0;

    for (int ph = 0; ph < pooled_height; ph++) {
      for (int pw = 0; pw < pooled_width; pw++) {
        const ACC_T grad_output_this_bin =
            offset_grad_output[ph * pooled_width + pw];

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            PreCalc<ACC_T> pc = pre_calc[pre_calc_index];
            T g1 = grad_output_this_bin * pc.w1 / count;
            T g2 = grad_output_this_bin * pc.w2 / count;
            T g3 = grad_output_this_bin * pc.w3 / count;
            T g4 = grad_output_this_bin * pc.w4 / count;

            add(offset_grad_input + pc.pos1, static_cast<T>(g1));
            add(offset_grad_input + pc.pos2, static_cast<T>(g2));
            add(offset_grad_input + pc.pos3, static_cast<T>(g3));
            add(offset_grad_input + pc.pos4, static_cast<T>(g4));
            pre_calc_index += 1;
          } // ix
        } // iy
      } // pw
    } // ph
  } // c
}

template <typename T, typename ACC_T>
inline void roi_align_single_framework_channels_last_backward(
    const T* grad_output,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* grad_input) {
  // for 'normal' size of channels, should be L1 fit;
  // otherwise consider blocking on channels.
  using Vec = at::vec::Vectorized<T>;

  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      const T* g_out = grad_output + (ph * pooled_width + pw) * channels;

      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          PreCalc<ACC_T> pc = pre_calc[pre_calc_index];
          T* g_in1 = grad_input + pc.pos1 * channels;
          T* g_in2 = grad_input + pc.pos2 * channels;
          T* g_in3 = grad_input + pc.pos3 * channels;
          T* g_in4 = grad_input + pc.pos4 * channels;

          Vec w1_vec = Vec(static_cast<T>(pc.w1 / count));
          Vec w2_vec = Vec(static_cast<T>(pc.w2 / count));
          Vec w3_vec = Vec(static_cast<T>(pc.w3 / count));
          Vec w4_vec = Vec(static_cast<T>(pc.w4 / count));
          int64_t d2 = 0;
          for (; d2 < channels - (channels % Vec::size()); d2 += Vec::size()) {
            Vec g_in1_vec =
                Vec::loadu(g_in1 + d2) + Vec::loadu(g_out + d2) * w1_vec;
            g_in1_vec.store(g_in1 + d2);
            Vec g_in2_vec =
                Vec::loadu(g_in2 + d2) + Vec::loadu(g_out + d2) * w2_vec;
            g_in2_vec.store(g_in2 + d2);
            Vec g_in3_vec =
                Vec::loadu(g_in3 + d2) + Vec::loadu(g_out + d2) * w3_vec;
            g_in3_vec.store(g_in3 + d2);
            Vec g_in4_vec =
                Vec::loadu(g_in4 + d2) + Vec::loadu(g_out + d2) * w4_vec;
            g_in4_vec.store(g_in4 + d2);
          }
          for (; d2 < channels; d2++) {
            g_in1[d2] += g_out[d2] * pc.w1 / count;
            g_in2[d2] += g_out[d2] * pc.w2 / count;
            g_in3[d2] += g_out[d2] * pc.w3 / count;
            g_in4[d2] += g_out[d2] * pc.w4 / count;
          }
          pre_calc_index += 1;
        } // ix
      } // iy
    } // pw
  } // ph
}

template <typename T, typename ACC_T>
void roi_align_backward_kernel_body(
    int n_rois,
    const T* grad_output,
    const ACC_T& spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    T* grad_input,
    const ACC_T* rois,
    bool is_channels_last) {
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // at::parallel_for(0, n_rois, 1, [&](int begin, int end) {
  for (int n = 0; n < n_rois; n++) {
    const ACC_T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    ACC_T offset = aligned ? (ACC_T)0.5 : (ACC_T)0.0;
    ACC_T roi_start_w = offset_rois[1] * spatial_scale - offset;
    ACC_T roi_start_h = offset_rois[2] * spatial_scale - offset;
    ACC_T roi_end_w = offset_rois[3] * spatial_scale - offset;
    ACC_T roi_end_h = offset_rois[4] * spatial_scale - offset;

    ACC_T roi_width = roi_end_w - roi_start_w;
    ACC_T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, (ACC_T)1.);
      roi_height = std::max(roi_height, (ACC_T)1.);
    }

    ACC_T bin_size_h =
        static_cast<ACC_T>(roi_height) / static_cast<ACC_T>(pooled_height);
    ACC_T bin_size_w =
        static_cast<ACC_T>(roi_width) / static_cast<ACC_T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const ACC_T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc<ACC_T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

    if (is_channels_last) {
      roi_align_single_framework_channels_last_backward<T, ACC_T>(
          grad_output + n * channels * pooled_height * pooled_width,
          count,
          channels,
          height,
          width,
          pooled_height,
          pooled_width,
          roi_bin_grid_h,
          roi_bin_grid_w,
          pre_calc,
          grad_input + roi_batch_ind * channels * height * width);
    } else {
      roi_align_single_framework_backward<T, ACC_T>(
          grad_output + n * channels * pooled_height * pooled_width,
          count,
          channels,
          height,
          width,
          pooled_height,
          pooled_width,
          roi_bin_grid_h,
          roi_bin_grid_w,
          pre_calc,
          grad_input + roi_batch_ind * channels * height * width);
    }
  } // for n
  // });
}

at::Tensor roi_align_forward_kernel_impl(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::ROIAlign_forward\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::ROIAlign_forward", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(rois.device().is_cpu(), "rois must be a CPU tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_forward_kernel_impl";
  // at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto memory_format = input.suggest_memory_format();
  bool is_channels_last = memory_format == at::MemoryFormat::ChannelsLast;
  at::Tensor output = at::empty(
      {num_rois, channels, pooled_height, pooled_width},
      input.options().memory_format(memory_format));

  if (output.numel() == 0)
    return output;

  auto input_ = input.contiguous(memory_format), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "roi_align_forward_kernel_impl",
      [&] {
        using accscalar_t = typename AccType<scalar_t>::type;
        roi_align_forward_kernel_body<scalar_t, accscalar_t>(
            num_rois,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            rois_.data_ptr<accscalar_t>(),
            output.data_ptr<scalar_t>(),
            is_channels_last);
      });
  return output;
}

at::Tensor roi_align_backward_kernel_impl(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned,
    bool is_channels_last) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::ROIAlign_backward\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::ROIAlign_backward", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(grad.device().is_cpu(), "grad must be a CPU tensor");
  TORCH_CHECK(rois.device().is_cpu(), "rois must be a CPU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_backward_kernel_impl";
  // at::checkAllSameType(c, {grad_t, rois_t});

  auto memory_format = is_channels_last ? at::MemoryFormat::ChannelsLast
                                        : at::MemoryFormat::Contiguous;
  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor grad_input = at::empty(
                              {batch_size, channels, height, width},
                              grad.options().memory_format(memory_format))
                              .zero_();

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  auto grad_ = grad.contiguous(memory_format), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "roi_align_backward_kernel_impl",
      [&] {
        using accscalar_t = typename AccType<scalar_t>::type;
        roi_align_backward_kernel_body<scalar_t, accscalar_t>(
            grad_.size(0),
            grad_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<accscalar_t>(),
            is_channels_last);
      });
  return grad_input;
}

} // anonymous namespace

REGISTER_DISPATCH(
    roi_align_forward_kernel_stub,
    &roi_align_forward_kernel_impl);
REGISTER_DISPATCH(
    roi_align_backward_kernel_stub,
    &roi_align_backward_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
