#include "DNNLChecker.h"

#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/auto_opt_config.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace chk {

bool dnnl_support_the_tensors(const std::vector<at::Tensor> &tensor_vec) {
  return dnnl_tensor_has_data(tensor_vec) &&
         dnnl_support_the_dimension_of(tensor_vec);
}

bool dnnl_inplace_support_the_tensors(const std::vector<at::Tensor> &tensor_vec) {
  return dnnl_tensor_has_data(tensor_vec) &&
         dnnl_support_the_data_type_of(tensor_vec) &&
         dnnl_support_the_memory_layout_of(tensor_vec);
}

bool dnnl_support_the_memory_layout_of(const std::vector<at::Tensor> &tensor_vec) {
  for (auto it = tensor_vec.begin(); it != tensor_vec.end(); ++it) {
    if (! (dnnl_support_the_memory_layout_of(*it))) {
      return false;
    }
  }
  return true;
}

bool dnnl_support_the_memory_layout_of(const at::Tensor& tensor) {
  return tensor.layout() == at::Layout::Strided;
}

bool dnnl_support_the_data_type_of(const std::vector<at::Tensor> &tensor_vec) {
  for (auto it = tensor_vec.begin(); it != tensor_vec.end(); ++it) {
    if (torch_ipex::get_dil_data_type(it->scalar_type()) ==  dil::data_type::undef) {
      return false;
    }
  }

  return true;
}

bool dnnl_support_the_dimension_of(const std::vector<at::Tensor> &tensor_vec) {
  for (auto it = tensor_vec.begin(); it != tensor_vec.end(); ++it) {
    if (it->dim() <= 0 && !is_scalar_tensor(*it)) {
      return false;
    }
  }

  return true;
}

bool dnnl_tensor_has_data(const std::vector<at::Tensor> &tensor_vec) {
  for (auto it = tensor_vec.begin(); it != tensor_vec.end(); ++it)
    if (it->numel() == 0)
      return false;

  return true;
}

bool all_is_dpcpp(const std::vector<at::Tensor> &tensor_vec) {
  for (auto it = tensor_vec.begin(); it != tensor_vec.end(); ++it) {
    if (!it->device().is_dpcpp()) {
      // Not DPCPP device
      if (is_scalar_tensor(*it) || !it->defined() || it->numel() == 0) {
        // If the tensor is scalar tensor or the tensor does not contains any elements, do nothing
      } else {
        return false;
      }
    }
  }

  return true;
}

}  // namespace chk
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
