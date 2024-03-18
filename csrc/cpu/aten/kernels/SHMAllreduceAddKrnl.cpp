#ifdef USE_CCL
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <aten/ShmAllReduceAdd.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

enum shm_state { INIT = 0, RANK0_COPY = 1, RANKX_COPY_ADD = 2, BROADCAST = 3 };
enum shm_block_state { INIT_BLOCK = 0, COPY_ADD_DONE_BLOCK = 1 };

inline void wait_state_until(
    int* states_ptr,
    const int index,
    enum shm_state state) {
  volatile int* state_ptr = states_ptr + index;
  while (*state_ptr != state)
    _mm_pause();
}

inline void wait_block_until(
    uint8_t* block_states_ptr,
    const int index,
    enum shm_block_state state) {
  volatile uint8_t* state_ptr = block_states_ptr + index;
  while (*state_ptr != state)
    _mm_pause();
}

template <typename DST_T, typename SRC_T>
static inline void multiThreadCopy(DST_T* dst, SRC_T* src, int size) {
  RECORD_FUNCTION("multiThreadCopy", c10::ArrayRef<c10::IValue>({}));
  constexpr int sizePerSplit = 512;
  int splits = (size + sizePerSplit - 1) / sizePerSplit;
#pragma omp parallel for
  for (int i = 0; i < splits; ++i) {
    int block_size =
        (i == splits - 1) ? (size - i * sizePerSplit) : sizePerSplit;
    torch_ipex::cpu::kernel::move_ker<DST_T, SRC_T>(
        dst + i * sizePerSplit, src + i * sizePerSplit, block_size);
  }
}

/**
 * @brief Performs reduction operation by adding the elements in the send
 * buffer in every rank into the shared memory buffer and storing the result
 * in the receive buffer. Firstly, the elements in the send buffer are
 * copied into the shared memory buffer. Then, the elements in the shared
 * memory buffer are added together and stored in the receive buffer. They
 * are 4 state to be maintained in the shared memory buffer:  0: ready for
 * all-reduce, e.g, initialized or last round all-reduce finished; 1: rank-0
 * copy ready; 2: finish add for other ranks; 3: finish broadcast
 * @tparam T The data type of the elements in the buffers.
 * @param sendBuf Pointer to the send buffer.
 * @param recvBuf Pointer to the receive buffer.
 * @param t_address The tensor of the shared memory buffer.
 * @param t_state The tensor of the state.
 * @param t_blockState The tensor of the block state.
 * @param shm_block_size The size of each block in the shared memory buffer.
 * @param size The number of elements in the buffers.
 * @param element_size The size of each element in bytes.
 * @param rank The rank of the current process.
 * @param rankSize The total number of processes.
 */
template <typename T>
void reduceAdd_impl(
    T* sendBuf,
    T* recvBuf,
    at::Tensor t_address,
    at::Tensor t_state,
    at::Tensor t_blockState,
    int shm_block_size,
    unsigned long size,
    int element_size,
    int rank,
    int rankSize) {
  int nbytes = size * element_size;
  int nBlockBytes = shm_block_size * element_size;
  int nblocks = (size + shm_block_size - 1) / shm_block_size;
  int nthreads = std::min(nblocks, omp_get_max_threads());
  float* address = (float*)t_address.data_ptr();
  uint8_t* block_states_ptr = (uint8_t*)t_blockState.data_ptr();
  int* states_ptr = t_state.data_ptr<int>();
  {
    RECORD_FUNCTION(
        "ipex::shm_all_reduce_add::rank0_copy", c10::ArrayRef<c10::IValue>({}));
    if (rank == 0) {
      for (int i = 1; i < rankSize; i++) {
        wait_state_until(states_ptr, i, INIT);
      }
      multiThreadCopy<float, T>(address, sendBuf, size);

    } else {
      wait_state_until(states_ptr, rank, INIT);
      wait_state_until(states_ptr, 0, RANK0_COPY);
    }
  }
  std::atomic_thread_fence(std::memory_order_release);
  states_ptr[rank] = RANK0_COPY;
  {
    RECORD_FUNCTION(
        "ipex::shm_all_reduce_add::copy_add_rankx",
        c10::ArrayRef<c10::IValue>({}));
    if (rank != 0) {
#pragma omp parallel for num_threads(nthreads)
      for (int blockIndex = 0; blockIndex < nblocks; blockIndex++) {
        auto lSendBuf = sendBuf + shm_block_size * blockIndex;
        auto lAddrBuf = address + shm_block_size * blockIndex;
        int realBlockSize =
            (blockIndex == (nblocks - 1)
                 ? (size - shm_block_size * (nblocks - 1))
                 : shm_block_size);

        if (rank != 1) {
          wait_block_until(
              block_states_ptr,
              blockIndex * rankSize + rank - 1,
              COPY_ADD_DONE_BLOCK);
        }
        torch_ipex::cpu::kernel::add_ker<float, T>(
            lAddrBuf, lSendBuf, realBlockSize);
        std::atomic_thread_fence(std::memory_order_release);
        block_states_ptr[blockIndex * rankSize + rank - 1] = INIT_BLOCK;
        std::atomic_thread_fence(std::memory_order_release);
        block_states_ptr[blockIndex * rankSize + rank] = COPY_ADD_DONE_BLOCK;
      }
      std::atomic_thread_fence(std::memory_order_release);
      states_ptr[rank] = RANKX_COPY_ADD;
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::shm_all_reduce_add::broadcast", c10::ArrayRef<c10::IValue>({}));
    wait_state_until(states_ptr, rankSize - 1, RANKX_COPY_ADD);
    multiThreadCopy<T, float>(recvBuf, address, size);
    if (rank == rankSize - 1) {
      for (int i = 0; i < rankSize - 1; i++) {
        wait_state_until(states_ptr, i, BROADCAST);
      }

      for (int i = 0; i < rankSize; i++) {
        std::atomic_thread_fence(std::memory_order_release);
        states_ptr[i] = INIT;
      }
    } else {
      std::atomic_thread_fence(std::memory_order_release);
      states_ptr[rank] = BROADCAST;
    }
  }
}

at::Tensor shm_all_reduce_add_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_address,
    at::Tensor& t_state,
    at::Tensor& t_blockState,
    int64_t shm_block_size,
    int64_t rank,
    int64_t world_size) {
  RECORD_FUNCTION("ipex::shm_all_reduce_add", c10::ArrayRef<c10::IValue>({}));
  // torch_ipex::cpu::shm_all_reduce_add_kernel_stub(kCPU, t_in);
  auto dtype = t_in.scalar_type();
  if (dtype == at::ScalarType::BFloat16) {
    reduceAdd_impl(
        (at::BFloat16*)t_in.data_ptr(),
        (at::BFloat16*)t_in.data_ptr(),
        t_address,
        t_state,
        t_blockState,
        shm_block_size,
        t_in.numel(),
        sizeof(at::BFloat16),
        rank,
        world_size);
  } else if (dtype == at::ScalarType::Half) {
    reduceAdd_impl(
        (at::Half*)t_in.data_ptr(),
        (at::Half*)t_in.data_ptr(),
        t_address,
        t_state,
        t_blockState,
        shm_block_size,
        t_in.numel(),
        sizeof(at::Half),
        rank,
        world_size);
  } else if (dtype == at::ScalarType::Float) {
    reduceAdd_impl(
        (float*)t_in.data_ptr(),
        (float*)t_in.data_ptr(),
        t_address,
        t_state,
        t_blockState,
        shm_block_size,
        t_in.numel(),
        sizeof(float),
        rank,
        world_size);
  } else if (dtype == at::ScalarType::Int) {
    reduceAdd_impl(
        (int*)t_in.data_ptr(),
        (int*)t_in.data_ptr(),
        t_address,
        t_state,
        t_blockState,
        shm_block_size,
        t_in.numel(),
        sizeof(int),
        rank,
        world_size);
  } else if (dtype == at::ScalarType::Long) {
    reduceAdd_impl(
        (int64_t*)t_in.data_ptr(),
        (int64_t*)t_in.data_ptr(),
        t_address,
        t_state,
        t_blockState,
        shm_block_size,
        t_in.numel(),
        sizeof(int64_t),
        rank,
        world_size);
  } else {
    TORCH_CHECK(
        false,
        "Data Type %s is not supported in SHM based all-reduce!\n",
        typeid(dtype).name());
    exit(-1);
  }
  return t_in;
}
} // namespace

IPEX_REGISTER_DISPATCH(
    shm_all_reduce_add_kernel_stub,
    &shm_all_reduce_add_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
#endif