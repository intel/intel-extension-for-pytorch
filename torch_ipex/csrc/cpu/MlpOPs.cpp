#include "MlpOPs.h"

#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#pragma message "Using OpenMP"
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif
#include <libxsmm.h>

namespace torch_ipex {

const at::ScalarType dt_map[] = {at::kDouble, at::kFloat, at::kBFloat16, at::kInt, at::kShort, at::kChar, at::kByte/*"UNK"*/};

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

void libxsmm_dnn_fullyconnected_set_ptr_helper(
    libxsmm_dnn_fullyconnected *handle,
    const libxsmm_dnn_tensor_type type,
    const at::Tensor &pt_tensor,
    char *desc) {
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  void *ptr;
  if(pt_tensor.scalar_type() == at::kFloat) ptr = (void*)pt_tensor.data_ptr<float>();
  else if(pt_tensor.scalar_type() == at::kBFloat16) ptr = (void*)pt_tensor.data_ptr<at::BFloat16>();
  else ptr = NULL;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fullyconnected_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(!tensor) {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
    tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( handle, tensor, type ) );
  } else {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(tensor, ptr) );
  }
}

void libxsmm_dnn_fullyconnected_release_tensor_helper(
    libxsmm_dnn_fullyconnected *handle,
    const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fullyconnected_get_tensor(handle, type, &status);
  CHKERR_LIBXSMM_DNN( status );
  if(tensor) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor(tensor) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( handle, type ) );
  }
}

at::Tensor AtenIpexTypeMLPExt::forward(
    void *libxsmm_handle_,
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias) {
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_fullyconnected* libxsmm_handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;
  auto nbn = input.size(0);
  auto bn = input.size(2);
  auto nbk = weight.size(0);
  auto bk = weight.size(3);
  auto output = at::empty({nbn, nbk, bn, bk}, input.options());
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BIAS, bias.view({nbk, bk}), "Bias");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");
  {
    RECORD_FUNCTION("ipex_mm_fwd", std::vector<c10::IValue>(/*input, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
      int tid = omp_get_thread_num();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
    }
  }
  return output;
}


std::vector<at::Tensor> AtenIpexTypeMLPExt::backward(
    void *libxsmm_handle_,
    const at::Tensor &grad_output,
    const at::Tensor &input,
    const at::Tensor &weight) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(libxsmm_handle_ != nullptr);
  libxsmm_dnn_err_t global_status;
  auto nbn = input.size(0);
  auto nbc = input.size(1);
  auto bn = input.size(2);
  auto bc = input.size(3);
  auto nbk = weight.size(0);
  auto bk = weight.size(3);

  auto grad_input = at::empty(input.sizes(), input.options());
  auto grad_weight = at::empty(weight.sizes(), weight.options());
  auto grad_bias = at::empty({nbk * bk}, weight.options());

  libxsmm_dnn_fullyconnected* libxsmm_handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS, grad_bias.view({nbk, bk}), "GradBias");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "GradOutput");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "GradInput");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, grad_weight, "GradWeight");

  RECORD_FUNCTION("ipex_mm_bwdupd", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
  #ifdef _OPENMP
  #pragma omp parallel
  #endif
  {
    int tid = omp_get_thread_num();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid) );
  }
  return {grad_input, grad_weight, grad_bias};
}


void *AtenIpexTypeMLPExt::create_handle(int N, int C, int K, int bn, int bc, int bk, int dtype, int fuse_bias, int act_type) {
  libxsmm_dnn_fullyconnected_desc fullyconnected_desc;
  libxsmm_dnn_fullyconnected* libxsmm_handle;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  fullyconnected_desc.N = N;
  fullyconnected_desc.C = C;
  fullyconnected_desc.K = K;
  fullyconnected_desc.bn = bn;
  fullyconnected_desc.bk = bk;
  fullyconnected_desc.bc = bc;
  fullyconnected_desc.threads = omp_get_max_threads();
  fullyconnected_desc.datatype_in = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fullyconnected_desc.datatype_out = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fullyconnected_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED;
  fullyconnected_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;
  fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;
  if(fuse_bias == 1) {
    if(act_type == 0)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS;
    else if(act_type == 1)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU;
    else if(act_type == 2)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID;
    else
      { printf("Unknown activation type (%d)\n", act_type); exit(1); }
  } else {
     if(act_type == 0)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;
    else if(act_type == 1)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU;
    else if(act_type == 2)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID;
    else
      { printf("Unknown activation type (%d)\n", act_type); exit(1); }
 }

  libxsmm_handle = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto scratch_size = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle, scratch ) );
  return (void *)libxsmm_handle;
}

at::Tensor AtenIpexTypeMLPExt::set_relu_mask(void *libxsmm_handle_) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(libxsmm_handle_ != nullptr);
  libxsmm_dnn_fullyconnected *handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(handle, LIBXSMM_DNN_RELU_MASK, &status); CHKERR_LIBXSMM_DNN( status );
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout != nullptr);
  std::vector<long> dim_size;
  for (int i = layout->num_dims - 1; i >= 0; i--) {
    dim_size.push_back(layout->dim_size[i]);
  }
  at::Tensor pt_tensor = at::empty(dim_size, at::TensorOptions().dtype(dt_map[layout->datatype]));
  void *ptr = pt_tensor.data_ptr();
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( layout );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( handle, tensor, LIBXSMM_DNN_RELU_MASK ) );
  return pt_tensor;
}

void AtenIpexTypeMLPExt::release_handle(void* libxsmm_handle_) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(libxsmm_handle_ != nullptr);
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fullyconnected* libxsmm_handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;

  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT);
  size_t scratch_size = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle, &status );
  if(scratch_size > 0) {
    void *scratch = libxsmm_dnn_fullyconnected_get_scratch_ptr(libxsmm_handle, &status);
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_fullyconnected_release_scratch(libxsmm_handle));
    if(scratch) libxsmm_free(scratch);
  }

  CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_fullyconnected(libxsmm_handle));
}

} // namespace torch_ipex
