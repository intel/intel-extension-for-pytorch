#include <THDP/THSYCLSleep.h>

#include <c10/dpcpp/SYCLStream.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>
#include <c10/dpcpp/SYCLException.h>

#include <time.h>

void THSYCL_sleep(THSYCLState* state, int64_t cycles)
{
  c10::DeviceIndex device_idx;
  C10_SYCL_CHECK(c10::sycl::syclGetDevice(&device_idx));
  c10::sycl::SYCLStream cur_sycl_stream = c10::sycl::getCurrentSYCLStream(device_idx);
  auto cur_sycl_queue = cur_sycl_stream.sycl_queue();
  cur_sycl_queue.wait();

  cur_sycl_queue.submit([&](cl::sycl::handler &cgh){
    int64_t start_clock = clock();
    int64_t clock_offset = 0;
    while (clock_offset < cycles)
    {
      clock_offset = clock() - start_clock;
    }
  });
}
