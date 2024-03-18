#pragma once
#ifdef USE_CCL
#include <mpi.h>

#include <torch/all.h>
#include <cstdlib>
#include <iostream>
#include "oneapi/ccl.hpp"
#ifdef USE_SHM
#include "shm_reduction.h"
#endif

class Messenger {
 private:
  Messenger() {
    // User has set the SINGLE_INSTANCE environment variable
    // or program is not with MPI.
    if (std::getenv("SINGLE_INSTANCE") != nullptr || !withMpirun()) {
      std::cout << "[INFO] SINGLE_INSTANCE MODE." << std::endl;
      this->pcomm = nullptr;
#ifdef USE_SHM
      this->pshm = nullptr;
#endif
      this->rank = 0;
      this->size = 1;
      return;
    }

    int flag = 0;
    MPI_Initialized(&flag);
    if (flag) {
      MPI_Finalize();
    }
    ccl::init();
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(Messenger::mpi_finalize);

    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      main_addr = kvs->get_address();
      MPI_Bcast(
          (void*)main_addr.data(),
          main_addr.size(),
          MPI_BYTE,
          0,
          MPI_COMM_WORLD);
    } else {
      MPI_Bcast(
          (void*)main_addr.data(),
          main_addr.size(),
          MPI_BYTE,
          0,
          MPI_COMM_WORLD);
      kvs = ccl::create_kvs(main_addr);
    }

    pcomm = new ccl::communicator(ccl::create_communicator(size, rank, kvs));

    rank = pcomm->rank();
    size = pcomm->size();

#ifdef USE_SHM
    char my_hostname[MPI_MAX_PROCESSOR_NAME];
    char all_hostnames[MPI_MAX_PROCESSOR_NAME * MPI_MAX_PROCESSOR_NAME];
    int hostname_len;

    // Check ranks are on the same physical machine
    MPI_Get_processor_name(my_hostname, &hostname_len);
    MPI_Allgather(
        my_hostname,
        MPI_MAX_PROCESSOR_NAME,
        MPI_CHAR,
        all_hostnames,
        MPI_MAX_PROCESSOR_NAME,
        MPI_CHAR,
        MPI_COMM_WORLD);

    int same_hostnames = 1;
    for (int i = 1; i < size; i++) {
      if (strcmp(my_hostname, &all_hostnames[i * MPI_MAX_PROCESSOR_NAME]) !=
          0) {
        same_hostnames = 0;
        break;
      }
    }

    if (same_hostnames) {
      pshm = new ShmReduction(rank, size, [this](int* pid_fd, size_t count) {
        this->broadcast(pid_fd, count);
      });
    } else {
      pshm = nullptr;
    }
#endif
  }

  ~Messenger() {
    delete pcomm;
#ifdef USE_SHM
    if (pshm != nullptr)
      delete pshm;
#endif
  }

  ccl::datatype get_ccl_dtype(at::ScalarType dtype) {
    if (dtype == at::ScalarType::BFloat16) {
      return ccl::datatype::bfloat16;
    } else if (dtype == at::ScalarType::Half) {
      return ccl::datatype::float16;
    } else if (dtype == at::ScalarType::Float) {
      return ccl::datatype::float32;
    } else if (dtype == at::ScalarType::Int) {
      return ccl::datatype::int32;
    } else if (dtype == at::ScalarType::Long) {
      return ccl::datatype::int64;
    } else {
      printf("Type %s not supported!\n", typeid(dtype).name());
      exit(-1);
    }
  }

  void ccl_allreduce_add(at::Tensor& t_in) {
    auto ccl_dtype = get_ccl_dtype(t_in.scalar_type());
    ccl::allreduce(
        t_in.data_ptr(),
        t_in.data_ptr(),
        (size_t)t_in.numel(),
        ccl_dtype,
        ccl::reduction::sum,
        *pcomm)
        .wait();
  }

 public:
  static Messenger& getInstance() {
    static Messenger instance;
    return instance;
  }

  bool isMaster() {
    return rank == 0;
  }

  int getRank() {
    return rank;
  }

  int getSize() {
    return size;
  }

  /**
   * Performs a reduction operation by adding the elements of the input tensor.
   * If USE_SHM is defined and the size of the tensor exceeds the shared memory
   * size or local ranks flag is false, the reduction is performed using the
   * ccl_allreduce_add method. Otherwise, the reduction is performed using the
   * reduceAdd method of the pshm object which used SHM. If USE_SHM is not
   * defined, the reduction is always performed using the ccl_allreduce_add
   * method.
   *
   * @param t_in The input tensor to be reduced.
   */
  void reduceAdd(at::Tensor& t_in) {
#ifdef USE_SHM
    if (t_in.numel() * sizeof(float) > pshm->getSHMSize() || pshm == nullptr) {
      this->ccl_allreduce_add(t_in);
    } else {
      pshm->reduceAdd(t_in);
    }
#else
    this->ccl_allreduce_add(t_in);
#endif
  }

  at::Tensor allgather(
      at::Tensor data,
      const std::vector<at::Tensor>& vec_data_out) {
    std::vector<size_t> recvCounts;
    std::transform(
        vec_data_out.begin(),
        vec_data_out.end(),
        std::back_inserter(recvCounts),
        [](const at::Tensor& t) { return t.numel(); });
    std::vector<void*> recvBufs;
    std::transform(
        vec_data_out.begin(),
        vec_data_out.end(),
        std::back_inserter(recvBufs),
        [](const at::Tensor& t) { return t.data_ptr(); });
    {
      RECORD_FUNCTION("ccl::allgatherv", std::vector<c10::IValue>());
      ccl::allgatherv(
          data.data_ptr(),
          (size_t)data.numel(),
          recvBufs,
          recvCounts,
          get_ccl_dtype(data.scalar_type()),
          *pcomm)
          .wait();
    }
    return at::cat(vec_data_out, -1);
  }

  void barrier() {
    if (check()) {
      ccl::barrier(*pcomm);
    }
  }

  void broadcast(int* pid_fd, size_t count) {
    if (check()) {
      ccl::broadcast(pid_fd, count, ccl::datatype::int32, 0, *pcomm).wait();
    }
  }

  bool withMpirun() {
    return (
        std::getenv("MPI_LOCALRANKID") || std::getenv("MPI_LOCALNRANKS") ||
        std::getenv("PMI_RANK") || std::getenv("PMI_SIZE") ||
        std::getenv("PMIX_RANK"));
  }

 private:
  Messenger(const Messenger& messenger) = delete;
  Messenger& operator=(const Messenger& messenger) = delete;

  static void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
      MPI_Finalize();
    }
  }

  // Check if indeed need to communicate
  bool check() {
    return size > 1;
  }

 private:
  int size;
  int rank;

  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;

  ccl::communicator* pcomm;

#ifdef USE_SHM
  ShmReduction* pshm;
#endif
};
#endif