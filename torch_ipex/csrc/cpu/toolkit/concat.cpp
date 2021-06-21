#include "concat.h"
#include <vector>
#include <type_traits>
#include <parallel/algorithm>
#include "cpu/bf16/vec/bf16_vec_kernel.h"
#include "cpu/int8/vec/int8_vec_kernel.h"
//#include <iostream>

namespace toolkit {
struct InputMeta {
  void* data_ptr;
  int64_t inner_size;

  InputMeta() {};

  InputMeta(const at::Tensor& t, int64_t dim, int64_t inner)
    : data_ptr(t.data_ptr()) , inner_size(t.size(dim) * inner) {}
};


template<typename scalar_t>
void _concat_all_continue(at::Tensor& result, const std::vector<at::Tensor>& input_tensors, int64_t dim) {
  const int VEC_NUM = 64;
  const int VEC_NUM_LOG2 = 6;
  auto result_dim_stride = result.stride(dim);
  uint32_t input_size = input_tensors.size();
  std::vector<InputMeta> inputs(input_size);
  int64_t max_inner_size = inputs[0].inner_size;
  at::parallel_for(0, input_size, 64, [&](int64_t start, int64_t end) {
    for (int64_t j = start; j < end; j++) {
      InputMeta input_meta(input_tensors[j], dim, result_dim_stride);
      inputs[j] = input_meta; 
      max_inner_size = (inputs[j].inner_size > max_inner_size) ? inputs[j].inner_size : max_inner_size; 
    }
  });

  int64_t outer = result.numel() / (result.size(dim) * result_dim_stride);
  //std::cout << "sizeof (scalar_t) is " << sizeof(scalar_t) << ",  outer is " << outer << std::endl;
  scalar_t* result_data = result.data_ptr<scalar_t>();

  if (outer == 1) {
    scalar_t* result_ptr = result_data;
    //std::cout << "max_inner_size is  " << max_inner_size << ", input_size is " << input_size << std::endl;
    if ((input_size > 16) && (max_inner_size < input_size)) {
      std::vector<scalar_t*> results_addr_array(input_size);
      results_addr_array[0] = result_ptr;
      result_ptr += inputs[0].inner_size;
      for (int64_t j = 1; j < input_size; j++) {
        results_addr_array[j] = result_ptr;
        result_ptr += inputs[j].inner_size;
      }
      at::parallel_for(0, input_size, 0, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; j++) {
          int64_t local_inner = inputs[j].inner_size;
          scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr);
          scalar_t* result_ptr = results_addr_array[j];
	  if (local_inner < 8) {
	    for (auto k = 0; k < local_inner; k++) {
              result_ptr[k] = input_ptr[k];
	    }
	  } else {
            move_ker(result_ptr, input_ptr, local_inner);
	  }
        }
      });
    } else {
      for (int64_t j = 0; j < input_size; j++) {
        int64_t local_inner = inputs[j].inner_size;
        scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr);
        at::parallel_for(0, local_inner, 0, [&](int64_t start, int64_t end) {
          move_ker(result_ptr, input_ptr, end - start);
        });
        result_ptr += local_inner;
      }
    }
  } else {
    auto total_addrs = outer * input_size;
    std::vector<scalar_t*> results_addr_array(total_addrs);
    std::vector<scalar_t*> inputs_addr_array(total_addrs);
    std::vector<int64_t> local_inner_array(total_addrs);
    scalar_t* result_ptr = result_data;
    for (int64_t i = 0; i < outer; ++i) {
      for (int64_t j = 0; j < input_size; j++) {
	int64_t off = i * input_size + j;
	int64_t local_inner = inputs[j].inner_size;
	local_inner_array[off] = local_inner;
	results_addr_array[off] = result_ptr;
	inputs_addr_array[off] = (scalar_t*)(inputs[j].data_ptr) + i * local_inner;
        result_ptr += local_inner;
      }
    }

    //std::cout << "max_inner_size is  " << max_inner_size << ", total_addrs is " << total_addrs << std::endl;
    if (max_inner_size < total_addrs) {
      at::parallel_for(0, total_addrs, 64, [&](int64_t start, int64_t end) {
        for (int64_t off = start; off < end; off++) {
          move_ker(results_addr_array[off], inputs_addr_array[off], local_inner_array[off]);
        }
      });
    } else {
      for (int64_t off = 0; off < total_addrs; off++) {
	int64_t local_inner = local_inner_array[off];
	scalar_t * result_ptr = results_addr_array[off];
	scalar_t * input_ptr = inputs_addr_array[off];
        at::parallel_for(0, local_inner, 0, [&](int64_t start, int64_t end) {
          move_ker(result_ptr, input_ptr, end - start);
        });
      }
    }
  }
}

at::Tensor concat_all_continue(std::vector<at::Tensor> input_tensors, int64_t dim) {
  
  auto scalar_type = input_tensors[0].scalar_type();
  auto first_tensor_mem_format = input_tensors[0].suggest_memory_format();
  uint32_t input_size = input_tensors.size();
  int64_t cat_dim_size = 0;
  for (int i = 0; i < input_size; i++) {
    auto const &tensor = input_tensors[i];
    if (!tensor.is_contiguous(first_tensor_mem_format)) {
      input_tensors[i] = tensor.contiguous();
    }
    cat_dim_size += tensor.size(dim);
  }

  at::Tensor result = at::empty({0}, input_tensors[0].options().dtype());
  auto result_size = input_tensors[0].sizes().vec();
  result_size[dim] = cat_dim_size;
  result.resize_(result_size, first_tensor_mem_format);
  if (scalar_type == at::kFloat) {
    _concat_all_continue<float>(result, input_tensors, dim);
    return result;
  } else if (scalar_type == at::kBFloat16) {
    _concat_all_continue<at::BFloat16>(result, input_tensors, dim);
    return result;
  } else if (scalar_type == at::kLong) {
    _concat_all_continue<int64_t>(result, input_tensors, dim);
    return result;
  } else if (scalar_type == at::kInt) {
    _concat_all_continue<int32_t>(result, input_tensors, dim);
    return result;
  } else if (scalar_type == at::kByte) {
    _concat_all_continue<uint8_t>(result, input_tensors, dim);
    return result;
  } else if (scalar_type == at::kChar) {
    _concat_all_continue<int8_t>(result, input_tensors, dim);
    return result;
  } else if (scalar_type == at::kBool) {
    _concat_all_continue<bool>(result, input_tensors, dim);
    return result;
  } else if (scalar_type == at::kShort) {
    _concat_all_continue<int16_t>(result, input_tensors, dim);
    return result;
  } else {
    return at::cat(input_tensors, dim);
  }
}

} //toolkit


