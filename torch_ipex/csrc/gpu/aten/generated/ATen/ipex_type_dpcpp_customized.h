#ifndef IPEX_TYPE_DPCPP_CUSTOMIZED_H
#define IPEX_TYPE_DPCPP_CUSTOMIZED_H

#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeDPCPP {

at::Tensor & fill_slice_with_index(at::Tensor & t, int dim);

}
}

#endif
