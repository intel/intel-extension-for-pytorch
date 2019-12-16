#ifndef THDPNN_INNERPRODUCT_H
#define THDPNN_INNERPRODUCT_H

#include <legacy/THSYCLTensor.hpp>
#include <legacy/THSYCLGeneral.h>

void dnnl_vec_inner_product_forward(
    int K,
    THSYCLTensor *input1,
    THSYCLTensor *input2,
    THSYCLTensor *output);

#endif
