/*******************************************************************************
 * Copyright 2019-2021 Intel Corporation.
 *
 * This software and the related documents are Intel copyrighted  materials, and
 * your use of  them is  governed by the  express license  under which  they
 *were provided to you (License).  Unless the License provides otherwise, you
 *may not use, modify, copy, publish, distribute,  disclose or transmit this
 *software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents  are provided as  is,  with no
 *express or implied  warranties,  other  than those  that are  expressly stated
 *in the License.
 *******************************************************************************/

#define _USE_MATH_DEFINES
#include <CL/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include <mkl.h>
#include <cfloat>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

constexpr int SUCCESS = 0;
constexpr int FAILURE = 1;
constexpr float TWOPI = 6.2831853071795864769f;

// Compute (K*L)%M accurately
static float moda(int K, int L, int M) {
  return (float)(((long long)K * L) % M);
}

// Initialize array data(N) to produce unit peaks at data(H) and data(N-H)
static void init_data(float* data, int N1, int N2, int BATCH, int H1, int H2) {
  // Generalized strides for row-major addressing of data
  int S1 = 1, S2 = N1, S3 = N1 * N2;

  for (int k = 0; k < BATCH; k++) {
    for (int n2 = 0; n2 < N2; ++n2) {
      for (int n1 = 0; n1 < N1; ++n1) {
        double phase = TWOPI * (moda(n1, H1, N1) / N1 + moda(n2, H2, N2) / N2);
        int index = k * 2 * N1 * N2 + 2 * (S3 + n2 * S2 + n1 * S1);
        data[index + 0] = cos(phase) / (N2 * N1);
        data[index + 1] = sin(phase) / (N2 * N1);
      }
    }
  }
}

template <
    oneapi::mkl::dft::precision prec,
    oneapi::mkl::dft::domain signal_type>
int run_dft(cl::sycl::device& dev) {
  int batch = 2;
  int64_t signal_ndim = 2;
  std::vector<int64_t> mkl_signal_sizes = {72, 72};
  std::vector<int64_t> istrides = {5184, 72, 1};
  std::vector<int64_t> ostrides = {5184, 72, 1};
  bool inverse = false;

  cl::sycl::queue dpcpp_queue(dev);
  float* in_data = cl::sycl::malloc_shared<float>((2 * 72 * 72), dpcpp_queue);
  float* out_data = cl::sycl::malloc_shared<float>((2 * 72 * 72), dpcpp_queue);
  init_data(in_data, 72, 72, 2, -1, 2);

  oneapi::mkl::dft::descriptor<prec, signal_type> desc(mkl_signal_sizes);
  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch);
  int64_t idist = istrides[0];
  int64_t odist = ostrides[0];
  desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist);
  std::vector<int64_t> mkl_istrides(1 + signal_ndim, 0),
      mkl_ostrides(1 + signal_ndim, 0);
  for (int64_t i = 1; i <= signal_ndim; i++) {
    mkl_istrides[i] = istrides[i];
    mkl_ostrides[i] = ostrides[i];
  }
  desc.set_value(
      oneapi::mkl::dft::config_param::INPUT_STRIDES, mkl_istrides.data());
  desc.set_value(
      oneapi::mkl::dft::config_param::OUTPUT_STRIDES, mkl_ostrides.data());
  bool complex_input = true, complex_output = true;
  if (!complex_input || !complex_output) {
    desc.set_value(
        oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
        DFTI_COMPLEX_COMPLEX);
  }

  // rescale if requested
  int64_t normalization = 2;
  if (normalization) {
    int64_t signal_numel = 1;
    for (int i = 0; i < signal_ndim; ++i) {
      signal_numel *= mkl_signal_sizes[i];
    }
    double double_scale = double_scale =
        1.0 / static_cast<double>(signal_numel);
    desc.set_value(
        inverse ? oneapi::mkl::dft::config_param::BACKWARD_SCALE
                : oneapi::mkl::dft::config_param::FORWARD_SCALE,
        prec == oneapi::mkl::dft::precision::DOUBLE
            ? double_scale
            : static_cast<float>(double_scale));
  }

  desc.commit(dpcpp_queue);

  if (!inverse) {
    oneapi::mkl::dft::compute_forward(desc, in_data, out_data);
  } else {
    oneapi::mkl::dft::compute_backward(desc, in_data, out_data);
  }
  dpcpp_queue.wait();

  free(in_data, dpcpp_queue.get_context());
  free(out_data, dpcpp_queue.get_context());
  return SUCCESS;
}

//
// Description of example setup, apis used and supported floating point type
// precisions
//
void print_example_banner() {
  std::cout << "" << std::endl;
  std::cout << "###############################################################"
               "#########"
            << std::endl;
  std::cout << "# 2D FFT Real-Complex Single-Precision Example: " << std::endl;
  std::cout << "# " << std::endl;
  std::cout << "# Using apis:" << std::endl;
  std::cout << "#   dft" << std::endl;
  std::cout << "# " << std::endl;
  std::cout << "# Supported floating point type precisions:" << std::endl;
  std::cout << "#   float" << std::endl;
  std::cout << "#   std::complex<float>" << std::endl;
  std::cout << "# " << std::endl;
  std::cout << "###############################################################"
               "#########"
            << std::endl;
  std::cout << std::endl;
}

int main() {
  print_example_banner();

  int returnCode = 0;
  try {
    cl::sycl::device my_dev = cl::sycl::device(cl::sycl::gpu_selector());
    std::cout << "Platform: "
              << sycl::platform(cl::sycl::gpu_selector())
                     .get_info<sycl::info::platform::version>()
              << std::endl;
    std::cout << "Device: " << my_dev.get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Driver: "
              << my_dev.get_info<sycl::info::device::driver_version>()
              << std::endl;

    std::cout << "\tRunning with single precision real-to-complex 2-D FFT:"
              << std::endl;
    int status;
    status = run_dft<
        oneapi::mkl::dft::precision::SINGLE,
        oneapi::mkl::dft::domain::COMPLEX>(my_dev);
    if (status != SUCCESS) {
      std::cout << "\tTest Failed" << std::endl << std::endl;
      returnCode = status;
    } else {
      std::cout << "\tTest Passed" << std::endl << std::endl;
    }
  } catch (std::exception const& e) {
    std::cout << "Cannot select a GPU\n" << e.what() << "\n";
    returnCode = 1;
  }

  return returnCode;
}
