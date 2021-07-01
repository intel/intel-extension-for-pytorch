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
at::Tensor _concat_all_continue(std::vector<at::Tensor>& input_tensors, int64_t dim) {
  auto first_tensor_mem_format = input_tensors[0].suggest_memory_format();
  uint32_t input_size = input_tensors.size();
  int64_t cat_dim_size = 0;
  for (int i = 0; i < input_size; i++) {
    auto const &tensor = input_tensors[i];
    assert(tensor.is_contiguous(first_tensor_mem_format));
    cat_dim_size += tensor.size(dim);
  }
  at::Tensor result = at::empty({0}, input_tensors[0].options().dtype());
  auto result_size = input_tensors[0].sizes().vec();
  result_size[dim] = cat_dim_size;
  result.resize_(result_size, first_tensor_mem_format);
  int64_t outer = 1;
  for (int i = 0; i < dim; ++i) {
    outer *= result_size[i];
  }

  std::vector<InputMeta> inputs(input_size);
  int64_t max_inner_size = 0;
  auto result_dim_stride = result.stride(dim);
  at::parallel_for(0, input_size, 16, [&](int64_t start, int64_t end) {
    for (int64_t j = start; j < end; j++) {
      InputMeta input_meta(input_tensors[j], dim, result_dim_stride);
      inputs[j] = input_meta;
      max_inner_size = (inputs[j].inner_size > max_inner_size) ? inputs[j].inner_size : max_inner_size; 
    }
  });

  //std::cout << "sizeof (scalar_t) is " << sizeof(scalar_t) << ",  outer is " << outer << std::endl;
  scalar_t* result_data = result.data_ptr<scalar_t>();
  auto half_vec_size = (64 / sizeof(scalar_t)) >> 1;

  if (outer == 1) {
    //std::cout << "max_inner_size is  " << max_inner_size << ", input_size is " << input_size << std::endl;
    if (max_inner_size < (input_size << 1)) {
      scalar_t* result_ptr = result_data;
      std::vector<scalar_t*> results_addr_array(input_size);
      results_addr_array[0] = result_ptr;
      result_ptr += inputs[0].inner_size;
      for (int64_t j = 1; j < input_size; j++) {
        results_addr_array[j] = result_ptr;
        result_ptr += inputs[j].inner_size;
      }
      at::parallel_for(0, input_size, 64, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; j++) {
          int64_t local_inner = inputs[j].inner_size;
          scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr);
          scalar_t* result_ptr = results_addr_array[j];
	  if (local_inner < half_vec_size) {
	    for (auto k = 0; k < local_inner; k++) {
              result_ptr[k] = input_ptr[k];
	    }
	  } else {
            move_ker(result_ptr, input_ptr, local_inner);
	  }
        }
      });
    } else {
      scalar_t* result_ptr = result_data;
      for (int64_t j = 0; j < input_size; j++) {
        int64_t local_inner = inputs[j].inner_size;
        scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr);
        at::parallel_for(0, local_inner, 64, [&](int64_t start, int64_t end) {
          move_ker(result_ptr + start, input_ptr + start, end - start);
        });
        result_ptr += local_inner;
      }
    }
  } else {
    if (outer > 16){
      int64_t inner = 1;
      for (int i = dim + 1; i < int(result_size.size()); ++i) {
        inner *= result_size[i];
      }
      auto outer_stride = inner * cat_dim_size;
      //std::cout << "outer is " << outer << ", outer_stride is " << outer_stride << ", input_size is " << input_size  <<std::endl;
      at::parallel_for(0, outer, 1, [&](int64_t start, int64_t end) {
        for (int o = start; o < end; ++o) {
          scalar_t* result_ptr = result_data + o * outer_stride;
          for (int j = 0; j < input_size; ++j) {
            int64_t local_inner = inputs[j].inner_size;
            scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + o * local_inner;
	    if (local_inner == 1) {
              result_ptr[0] = input_ptr[0];
	    } else {
              move_ker(result_ptr, input_ptr, local_inner);
	    }
	    result_ptr += local_inner;
	  }
        }
      });
    } else {
      //std::cout << "outer is " << outer << std::endl;
      int64_t offset = 0;
      for (int o = 0; o < outer; ++o) {
        for (int64_t j = 0; j < input_size; j++) {
          int64_t local_inner = inputs[j].inner_size;
          scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + o * local_inner;
          scalar_t* result_ptr = result_data + offset;
          at::parallel_for(0, local_inner, 64, [&](int64_t start, int64_t end) {
            move_ker(result_ptr + start, input_ptr + start, end - start);
          });
	  offset += local_inner;
        }
      }
    }
  }
  return result;
}

at::Tensor concat_all_continue(std::vector<at::Tensor> input_tensors, int64_t dim) {
  
  auto scalar_type = input_tensors[0].scalar_type();
  if (scalar_type == at::kFloat) {
    //std::cout << "Float typei !\n";
    return _concat_all_continue<float>(input_tensors, dim);
  } else if (scalar_type == at::kBFloat16) {
    //std::cout << "BFloat16 type !\n";
    return _concat_all_continue<at::BFloat16>(input_tensors, dim);
  } else if (scalar_type == at::kLong) {
    //std::cout << "Long type !\n";
    return _concat_all_continue<int64_t>(input_tensors, dim);
  } else if (scalar_type == at::kInt) {
    //std::cout << "Int type !\n";
    return _concat_all_continue<int32_t>(input_tensors, dim);
  } else if (scalar_type == at::kByte) {
    //std::cout << "Byte type !\n";
    return _concat_all_continue<uint8_t>(input_tensors, dim);
  } else if (scalar_type == at::kChar) {
    //std::cout << "Char type !\n";
    return _concat_all_continue<int8_t>(input_tensors, dim);
  } else if (scalar_type == at::kBool) {
    //std::cout << "Bool type !\n";
    return _concat_all_continue<bool>(input_tensors, dim);
  } else if (scalar_type == at::kShort) {
    //std::cout << "Short type !\n";
    return _concat_all_continue<int16_t>(input_tensors, dim);
  } else {
    //std::cout << "Unknown type !\n";
    return at::cat(input_tensors, dim);
  }
}

} //toolkit


