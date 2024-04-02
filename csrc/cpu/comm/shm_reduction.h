#pragma once
#include <ATen/ATen.h>
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include "aten/ShmAllReduceAdd.h"

namespace torch_ipex {
namespace cpu {
#define SHM_NAME "ipex_shm_buffer"
#define SHM_BLOCK_SIZE_L (5120)
#define SHM_BLOCK_SIZE_S (16 * 5120)
#define MAX_SHM_BLOCK_COUNT 4096
#define MAX_SHM_SIZE (SHM_BLOCK_SIZE_S * MAX_SHM_BLOCK_COUNT * sizeof(float))

struct ShmContext {
  const char* name;
  int fp;
  int pid_fd[2];
  int* state;
  at::Tensor t_state;
  uint8_t* blockState;
  at::Tensor t_blockState;
  void* address;
  at::Tensor t_address;
  size_t nstates;
  size_t nblocks;
  size_t nbytes;
};

inline void connect_shm(ShmContext* ctx) {
  char fd_path[64];
  snprintf(
      fd_path,
      sizeof(fd_path),
      "/proc/%d/fd/%d",
      ctx->pid_fd[0],
      ctx->pid_fd[1]);
  ctx->fp = open(fd_path, O_RDWR);
  if (ctx->fp == -1) {
    perror("Bad file descriptor.");
    exit(-1);
  }

  const int total_size =
      ctx->nstates * sizeof(int) + ctx->nbytes + ctx->nblocks * ctx->nstates;

  // Map the shared memory into the address space of the process
  void* shm_ptr =
      mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fp, 0);
  if (shm_ptr == MAP_FAILED) {
    perror("shm mmap failed.");
    exit(-1);
  }
  ctx->state = (int*)shm_ptr;
  ctx->t_state =
      at::from_blob((void*)ctx->state, {(signed long)ctx->nstates}, at::kInt)
          .to(at::kCPU);
  ctx->blockState = (uint8_t*)((int*)shm_ptr + ctx->nstates);
  ctx->t_blockState =
      at::from_blob(
          (void*)ctx->blockState,
          {(signed long)ctx->nblocks, (signed long)ctx->nstates},
          at::kByte)
          .to(at::kCPU);
  ctx->address =
      (void*)((uint8_t*)ctx->blockState + ctx->nblocks * ctx->nstates);
  ctx->t_address = at::from_blob(
                       (void*)ctx->address,
                       {(signed long)(ctx->nbytes / sizeof(float))},
                       at::kFloat)
                       .to(at::kCPU);
}

inline void create_shm(ShmContext* ctx) {
  ctx->fp = shm_open(ctx->name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);

  if (ctx->fp == -1) {
    perror("shm open failed.");
    exit(-1);
  }
  const int total_size =
      ctx->nstates * sizeof(int) + ctx->nbytes + ctx->nblocks * ctx->nstates;
  // Truncate the shared memory to the desired size
  if (ftruncate(ctx->fp, total_size) == -1) {
    perror("shm ftruncate failed.");
    exit(-1);
  }

  // Map the shared memory into the address space of the process
  void* shm_ptr =
      mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fp, 0);
  if (shm_ptr == MAP_FAILED) {
    perror("shm mmap failed.");
    exit(-1);
  }
  ctx->pid_fd[0] = getpid();
  ctx->pid_fd[1] = ctx->fp;
  ctx->state = (int*)shm_ptr;
  ctx->t_state =
      at::from_blob((void*)ctx->state, {(signed long)ctx->nstates}, at::kInt)
          .to(at::kCPU);
  ctx->blockState = (uint8_t*)((int*)shm_ptr + ctx->nstates);
  ctx->t_blockState =
      at::from_blob(
          (void*)ctx->blockState,
          {(signed long)ctx->nblocks, (signed long)ctx->nstates},
          at::kByte)
          .to(at::kCPU);
  ctx->address =
      (void*)((uint8_t*)ctx->blockState + ctx->nblocks * ctx->nstates);
  ctx->t_address = at::from_blob(
                       (void*)ctx->address,
                       {(signed long)(ctx->nbytes / sizeof(float))},
                       at::kFloat)
                       .to(at::kCPU);
}

inline void close_shm(ShmContext* ctx) {
  const int total_size = ctx->nstates * sizeof(int) + ctx->nbytes;
  if (ctx->fp != -1) {
    munmap(ctx->address, total_size);
    shm_unlink(ctx->name);
  }
}

} // namespace cpu
} // namespace torch_ipex

class ShmReduction {
 public:
  ShmReduction(int rank, int size, std::function<void(int*, size_t)> callback)
      : rank_(rank), rank_size_(size) {
    shmCtx_.name = SHM_NAME;
    shmCtx_.nstates = size;
    shmCtx_.nbytes = MAX_SHM_SIZE;
    shmCtx_.nblocks = MAX_SHM_BLOCK_COUNT;
    if (rank_ == 0) {
      torch_ipex::cpu::create_shm(&shmCtx_);
      memset(shmCtx_.state, 0, shmCtx_.nstates * sizeof(int));
      memset((void*)shmCtx_.blockState, 0, shmCtx_.nstates * shmCtx_.nblocks);
    }

    callback(shmCtx_.pid_fd, 2);

    if (rank != 0) {
      torch_ipex::cpu::connect_shm(&shmCtx_);
    }
  }

  ~ShmReduction() {
    torch_ipex::cpu::close_shm(&shmCtx_);
  }

  int getSHMSize() {
    return MAX_SHM_SIZE;
  }

  void reduceAdd(at::Tensor& t_in) {
    bool is_small = t_in.numel() < 51200;
    auto block_size = is_small ? SHM_BLOCK_SIZE_S : SHM_BLOCK_SIZE_L;
    torch_ipex::cpu::shm_all_reduce_add_kernel_stub(
        kCPU,
        t_in,
        shmCtx_.t_address,
        shmCtx_.t_state,
        shmCtx_.t_blockState,
        block_size,
        rank_,
        rank_size_);
  }

  int rank_;
  int rank_size_;

 private:
  torch_ipex::cpu::ShmContext shmCtx_;
};
