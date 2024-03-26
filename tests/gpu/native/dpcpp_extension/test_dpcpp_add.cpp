#include <ipex.h> // SYCL header
#include <torch/extension.h>

const int BLOCKWIDTH = 16;
const int BLOCKHEIGHT = 16;

template <typename scalar_t>
void add_kernel_dpcpp(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* c,
    int height,
    int width) {
  int width_pad = (width + BLOCKWIDTH - 1) / BLOCKWIDTH * BLOCKWIDTH;
  int height_pad = (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT * BLOCKHEIGHT;
  sycl::range<2> gws(height_pad, width_pad);
  sycl::range<2> lws(BLOCKHEIGHT, BLOCKWIDTH);

  c10::xpu::XPUStream stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item) {
      int x = item.get_global_id(1);
      int y = item.get_global_id(0);
      if (x < width && y < height) {
        int i = y * width + x;
        c[i] = a[i] + b[i];
      }
    });
  });
}

void add_dpcpp(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  int height = a.size(0);
  int width = a.size(1);

  AT_DISPATCH_FLOATING_TYPES(a.type(), "add_dpcpp", ([&] {
                               add_kernel_dpcpp<scalar_t>(
                                   a.data_ptr<scalar_t>(),
                                   b.data_ptr<scalar_t>(),
                                   c.data_ptr<scalar_t>(),
                                   height,
                                   width);
                             }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add_dpcpp, "Simple tensor add(XPU) for all floating types");
}
