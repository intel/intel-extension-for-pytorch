#pragma once

#include <utils/DPCPP.h>

template <typename func_t, typename scalar_t>
struct EltwiseBinaryNaiveKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto off = item_id.get_global_linear_id();
    if (off < nelem)
      op(res_ptr + off, op1_ptr + off, op2_ptr + off);
  }
  EltwiseBinaryNaiveKernelFunctor(
      int nelem_,
      const func_t op_,
      scalar_t* res_ptr_,
      scalar_t* op1_ptr_,
      scalar_t* op2_ptr_)
      : nelem(nelem_),
        op(op_),
        res_ptr(res_ptr_),
        op1_ptr(op1_ptr_),
        op2_ptr(op2_ptr_) {}

 private:
  int nelem;
  const func_t op;
  scalar_t* res_ptr;
  scalar_t* op1_ptr;
  scalar_t* op2_ptr;
};

// requirements
// 1. same strides (contiguous)
// 2. same format
template <typename func_t, typename scalar_t>
void eltwise_binary_naive_kernel(
    scalar_t* res_ptr,
    scalar_t* op1_ptr,
    scalar_t* op2_ptr,
    int nelem,
    const func_t& op) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  uint64_t subgp_size = 32;
  uint64_t wg_num = nelem / subgp_size + 1;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    EltwiseBinaryNaiveKernelFunctor<func_t, scalar_t> knf(
        nelem, op, res_ptr, op1_ptr, op2_ptr);
    __cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(wg_num * subgp_size), sycl::range<1>(subgp_size)),
        knf);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}
