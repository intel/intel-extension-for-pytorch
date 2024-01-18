#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "UpSample.h"

using namespace xpu::dpcpp;

template <typename T>
struct Im2colKernelFunctor {
  void operator()(sycl::item<1> itemId) const {
    auto in_ptr = in_data;
    auto out_ptr = out_data;
    auto id = itemId.get_id(0);

    int64_t w_out = id % width_col;
    id /= width_col;

    int64_t h_out = id % height_col;
    int64_t channel_in = id / height_col;
    int64_t channel_out = channel_in * kernel_h * kernel_w;
    int64_t h_in = h_out * stride_h - pad_h;
    int64_t w_in = w_out * stride_w - pad_w;

    out_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    in_ptr += (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_h; ++i) {
      for (int64_t j = 0; j < kernel_w; ++j) {
        int64_t h = h_in + i * dilation_h;
        int64_t w = w_in + j * dilation_w;
        *out_ptr = (h >= 0 && w >= 0 && h < height && w < width)
            ? in_ptr[i * dilation_h * width + j * dilation_w]
            : static_cast<T>(0);
        ;
        out_ptr += height_col * width_col;
      }
    }
  }
  Im2colKernelFunctor(
      const T* in_data_,
      const int64_t channels_,
      const int64_t height_,
      const int64_t width_,
      const int64_t height_col_,
      const int64_t width_col_,
      const int64_t kernel_h_,
      const int64_t kernel_w_,
      const int64_t pad_h_,
      const int64_t pad_w_,
      const int64_t stride_h_,
      const int64_t stride_w_,
      const int64_t dilation_h_,
      const int64_t dilation_w_,
      T* out_data_)
      : in_data(in_data_),
        channels(channels_),
        height(height_),
        width(width_),
        height_col(height_col_),
        width_col(width_col_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        dilation_h(dilation_h_),
        dilation_w(dilation_w_),
        out_data(out_data_) {}

 private:
  const T* in_data;
  const int64_t channels;
  const int64_t height;
  const int64_t width;
  const int64_t height_col;
  const int64_t width_col;
  const int64_t kernel_h;
  const int64_t kernel_w;
  const int64_t pad_h;
  const int64_t pad_w;
  const int64_t stride_h;
  const int64_t stride_w;
  const int64_t dilation_h;
  const int64_t dilation_w;
  T* out_data;
};

template <typename T>
static void im2col_kernel(
    const T* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_col) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = channels * output_width * output_height;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = data_im;
    auto out_data = data_col;
    Im2colKernelFunctor<T> kfn(
        in_data,
        channels,
        height,
        width,
        height_col,
        width_col,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        out_data);
    cgh.parallel_for<decltype(kfn)>(sycl::range<1>(total_threads), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
struct Col2imKernelFunctor {
  void operator()(sycl::item<1> itemId) const {
    auto in_ptr = in_data;
    auto out_ptr = out_data;
    auto id = itemId.get_id(0);

    T val = static_cast<T>(0);
    const int64_t w_im = id % width + pad_w;
    const int64_t h_im = (id / width) % height + pad_h;
    const int64_t c_im = id / (width * height);
    int64_t kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int64_t kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int64_t w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int64_t w_col_end = min(w_im / stride_w + 1, output_width);
    const int64_t h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int64_t h_col_end = min(h_im / stride_h + 1, output_height);

    for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int64_t h_k = (h_im - h_col * stride_h);
        int64_t w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int64_t data_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * output_height +
               h_col) *
                  output_width +
              w_col;
          val += in_ptr[data_col_index];
        }
      }
    }
    out_ptr[id] = static_cast<T>(val);
  }
  Col2imKernelFunctor(
      const T* in_data_,
      const int64_t channels_,
      const int64_t height_,
      const int64_t width_,
      const int64_t output_height_,
      const int64_t output_width_,
      const int64_t kernel_h_,
      const int64_t kernel_w_,
      const int64_t pad_h_,
      const int64_t pad_w_,
      const int64_t stride_h_,
      const int64_t stride_w_,
      const int64_t dilation_h_,
      const int64_t dilation_w_,
      T* out_data_)
      : in_data(in_data_),
        channels(channels_),
        height(height_),
        width(width_),
        output_height(output_height_),
        output_width(output_width_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        dilation_h(dilation_h_),
        dilation_w(dilation_w_),
        out_data(out_data_) {}

 private:
  const T* in_data;
  const int64_t channels;
  const int64_t height;
  const int64_t width;
  const int64_t output_height;
  const int64_t output_width;
  const int64_t kernel_h;
  const int64_t kernel_w;
  const int64_t pad_h;
  const int64_t pad_w;
  const int64_t stride_h;
  const int64_t stride_w;
  const int64_t dilation_h;
  const int64_t dilation_w;
  T* out_data;
};

template <typename T>
static void col2im_kernel(
    const T* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_im) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = channels * width * height;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = data_col;
    auto out_data = data_im;
    Col2imKernelFunctor<T> kfn(
        in_data,
        channels,
        height,
        width,
        output_height,
        output_width,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        out_data);
    cgh.parallel_for<decltype(kfn)>(sycl::range<1>(total_threads), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}
