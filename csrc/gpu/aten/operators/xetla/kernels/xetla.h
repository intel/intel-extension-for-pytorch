#pragma once

#include <sycl/sycl.hpp>
#include <xetla.hpp>
#include <cmath>

#define DEVICE_MEM_ALIGNMENT (64)

using namespace sycl;
using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;
