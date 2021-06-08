#pragma once
#include <ATen/Tensor.h>
namespace torch_ipex{
    namespace mlperf {
        namespace dlrm {
            at::Tensor fuseembint_forward(at::Tensor &lS_o, at::Tensor &lS_i,
                                          std::vector<at::Tensor> &emb,
                                          at::Tensor &densex);
        }
    }
}
