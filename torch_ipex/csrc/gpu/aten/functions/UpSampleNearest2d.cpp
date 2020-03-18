#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <core/DPCPP.h>
#include <core/Memory.h>
#include <functions/UpSample.h>


using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
static void upsample_nearest2d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    int64_t onum) {
  const float height_scale = (float) input_height / (float) output_height;
  const float width_scale = (float) input_width / (float) output_width;

  auto dpcpp_queue = dpcppGetCurrentQueue();
  int64_t rng, grng, tile_size;
  parallel_for_setup(onum, tile_size, rng, grng);

  // command group functions
  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = DPCPPAccessor<dp_r_mode>(cgh, idata);
    auto out_acc = DPCPPAccessor<dp_w_mode>(cgh, odata);
    const int n = output_height * output_width;

    // kernel function per work-item
    auto kfn = DP_Q_KFN(DP::nd_item<1> item) {
      auto in_ptr = in_acc.template get_pointer<scalar_t>();
      auto out_ptr = out_acc.template get_pointer<scalar_t>();
      int global_id = item.get_global_linear_id();
      if (global_id < n) {
        const int w2 = global_id % output_width;
        const int h2 = global_id / output_width;
        // special case: just copy
        if (input_height == output_height && input_width == output_width) {
          const int h1 = h2;
          const int w1 = w2;
          for (int n = 0; n < nbatch; n++) {
            for (int c = 0; c < channels; ++c) {
              auto val = in_ptr[n * input_height * input_width * channels
                  + c * input_height * input_width + h1 * input_width + w1];
              out_ptr[n * output_height * output_width * channels
                  + c * output_height * output_width + h2 * output_width + w2] = val;
            }
          }
          return;
        }
        const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, input_height);
        const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, input_width);
        for (int n = 0; n < nbatch; n++) {
          for (int c = 0; c < channels; ++c) {
            const scalar_t val = in_ptr[n * input_height * input_width * channels
                + c * input_height * input_width + h1 * input_width + w1];
            out_ptr[n * output_height * output_width * channels
                + c * output_height * output_width + h2 * output_width + w2] = val;
          }
        }
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(nearest_neighbor_4d_dpcpp_kernel, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(grng), DP::range<1>(tile_size)), kfn);
  };

  // submit to DPCPP queue
  DP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

/*template <typename scalar_t>
static void upsample_nearest2d_backward_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    int64_t onum) {
  const float height_scale = (float) input_height / (float) output_height;
  const float width_scale = (float) input_width / (float) output_width;

  auto dpcpp_queue = dpcppGetCurrentQueue();
  int64_t rng, grng, tile_size;
  parallel_for_setup(onum, tile_size, rng, grng);

  // command group functions
  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = DPCPPAccessor<dp_w_mode>(cgh, idata); 
    auto out_acc = DPCPPAccessor<dp_r_mode>(cgh, odata); 
    auto in_acc_read = DPCPPAccessor<dp_r_mode>(cgh, odata); 

    // kernel function per work-item
    auto kfn = DP_Q_KFN(DP::nd_item<1> item) {
      auto in_ptr = in_acc.template get_pointer<scalar_t>();
      auto out_ptr = out_acc.template get_pointer<scalar_t>();
      auto in_read_ptr = in_acc_read.template get_pointer<scalar_t>();

      int global_id = item.get_global_linear_id();
      if (global_id < onum) {
        const int w2 = global_id % output_width; // 0:width2-1
        const int h2 = global_id / output_width; // 0:height2-1
        // special case: just copy
        if (input_height == output_height && input_width == output_width) {
          const int h1 = h2;
          const int w1 = w2;
          for (int n = 0; n < nbatch; n++) {
            for (int c = 0; c < channels; ++c) {
              auto val = out_ptr[n * output_height * output_width * channels
                  + c * output_height * output_width + h2 * output_width + w2];
              in_ptr[n * input_height * input_width * channels
                  + c * input_height * input_width + h1 * input_width + w1] = val;
            }
          }
          return;
        }

        const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, input_height);
        const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, input_width);
        for (int n = 0; n < nbatch; n++) {
          for (int c = 0; c < channels; ++c) {
            auto d2val = out_ptr[n * output_height * output_width * channels
                + c * output_height * output_width + h2 * output_width + w2];
            auto d2val_in = in_read_ptr[n * input_height * input_width * channels
                + c * input_height * input_width + h1 * input_width + w1];
            in_ptr[n * input_height * input_width * channels
                + c * input_height * input_width + h1 * input_width + w1] = d2val+d2val_in;
          }
        }
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(nearest_neighbor_4d_bwd_dpcpp_kernel, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(grng), DP::range<1>(tile_size)), kfn);
  };

  // submit to DPCPP queue
  DP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}*/

static void upsample_nearest2d_out_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);

  upsample_2d_shape_check(
      input_,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_height, output_width});
  output.zero_();

  AT_ASSERT(input_width > 0 && output_width > 0);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_nearest2d", [&] {
    auto* idata = input.data_ptr<scalar_t>();
    auto* odata = output.data_ptr<scalar_t>();
    auto onum = output.numel();

    upsample_nearest2d_out_frame<scalar_t>(
        odata,
        idata,
        input_height,
        input_width,
        output_height,
        output_width,
        nbatch,
        channels,
        onum);
  });
}

static void upsample_nearest2d_backward_out_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  upsample_2d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_input.zero_();

  auto grad_output = grad_output_.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_nearest2d_backward", [&] {
        /*
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        scalar_t* odata = grad_output.data_ptr<scalar_t>();
        auto onum = grad_output.numel();

        upsample_nearest2d_backward_out_frame<scalar_t>(
            odata,
            idata,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            onum);*/
       printf("Backward is depending on the atomic float op implementation, will enable it later!!!\n");

      });
}

} // namespace impl

Tensor& upsample_nearest2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  impl::upsample_nearest2d_backward_out_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

Tensor upsample_nearest2d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  return at::AtenIpexTypeDPCPP::upsample_nearest2d_backward_out(
      grad_input, grad_output, output_size, input_size);
}


Tensor & upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size){
  impl::upsample_nearest2d_out_template(out, self, output_size);
  return out;
}
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size){
  auto output = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::upsample_nearest2d_out(output, self, output_size);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
