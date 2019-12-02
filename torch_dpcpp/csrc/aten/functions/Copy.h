#pragma once

#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

void copy_kernel_sycl(TensorIterator& iter, bool non_blocking=true);

}} // namepsace at::native
