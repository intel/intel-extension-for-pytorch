#pragma once

#include <ATen/ATen.h>
#include <c10/dpcpp/SYCLGuard.h>

namespace at { namespace native {

Tensor& resize_sycl_(Tensor& self, IntArrayRef size);
Tensor& resize_as_sycl_(Tensor& self, const Tensor& the_template);

}} // namepsace at::native
