#pragma once

#include <CL/sycl.hpp>
#include <c10/core/Stream.h>

namespace xpu {

IPEX_API sycl::queue& get_queue_from_stream(c10::Stream stream);

} // namespace xpu
