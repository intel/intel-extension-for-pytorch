#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMoreMath.cpp"
#else

#include <ATen/dpcpp/Runtime.h>

using namespace mkldnn;

void THSYCLTensor_(cat)(THSYCLState *state,
    THSYCLTensor *r_, THSYCLTensor *ta, THSYCLTensor *tb, int dimension)
{
  THSYCLTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THSYCLTensor_(catArray)(state, r_, inputs, 2, dimension);
}

/*
static void THSYCLTensor_(catArray_mkldnn__)(THSYCLState *state,
    THSYCLTensor *result, THSYCLTensor **inputs, int numInputs, int dimension)
{
  // TODO: execption handler
  // (skips, dimension check, empty check, contiguous check)
  // Compute size of the result in the cat dimension
  THSYCLTensor *refTensor = inputs[0];
  int64_t nDims = refTensor->dim();
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    THSYCLTensor *tensor = inputs[i];
    cat_dim_size += tensor->size(dimension);
  }

  // Compute the size of the result
  std::vector<int64_t> size(nDims);
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = refTensor->size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }
  THSYCLTensor_(resize)(state, result, size, {});

  // mkldnn setup
  // TODO: scalar_t => mkldnn data type
  at::Device curDevice =
      at::Device(at::kDPCPP, c10::sycl::current_device());
  auto mkldnn_data_t = memory::data_type::f32;
  auto mkldnn_format = [&]() -> memory::format_tag {
    switch(nDims) {
    case 1:
      return memory::format_tag::x;
    case 2:
      return memory::format_tag::nc;
    case 3:
      return memory::format_tag::ntc;
    case 4:
      return memory::format_tag::nchw;
    default:
      return memory::format_tag::undef;
    };
  } ();

  if (mkldnn_format == memory::format_tag::undef) {
    // TODO: general implementation on sycl
    THArgError(1,
        "sycl backend does not support array with a dimension larger than 4 for now");
  }

  auto mkldnn_engine =
      at::native::GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = at::native::Stream::Instance().get_stream();

  // inputs memory
  std::vector<memory::desc> inputs_md;
  std::vector<memory> inputs_m;
  // FIXME: hold memory primitive not to release internel dynamic member
  // untyped_sycl_buffer_t::buf_ptr_
  std::vector<std::shared_ptr<memory>> inputs_mptr;
  for (int i = 0; i < numInputs; i++) {
    memory::dims input_dims;
    for (int dim = 0; dim < nDims; dim++)
      input_dims.push_back(inputs[i]->size(dim));
    // primitive format conform to the first input tensor
    auto input_md = memory::desc(input_dims, mkldnn_data_t, mkldnn_format);
    std::shared_ptr<memory> input_mptr(new memory(input_md, mkldnn_engine));
    at::native::sycl_set_mkldnn_buffer(
        (void *)inputs[i]->data<scalar_t>(), *input_mptr.get());

    inputs_md.push_back(input_md);
    inputs_m.push_back(*input_mptr.get());
    inputs_mptr.push_back(input_mptr);
  }

  // output memory
  memory::dims output_dims;
  for (int dim = 0; dim < nDims; dim++)
    output_dims.push_back((int)size[dim]);
  auto output_md = memory::desc(output_dims, mkldnn_data_t, mkldnn_format);
  auto output_m = memory(output_md, mkldnn_engine);
  at::native::sycl_set_mkldnn_buffer(
      (void *)result->data<scalar_t>(), output_m);

  std::unordered_map<int, memory> args = {
      {MKLDNN_ARG_DST, output_m}};
  for (int i = 0; i < numInputs; i++)
    args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, inputs_m[i]});

  // concat primitive
  auto concat_pd = concat::primitive_desc(
      output_md, dimension, inputs_md, mkldnn_engine);
  auto concat_p = concat(concat_pd);
  concat_p.execute(strm, args);

  std::vector<primitive> mkldnn_tasks = {concat_p};
}
*/

void THSYCLTensor_(check_shape_except_dim)(THSYCLState *state,
    THSYCLTensor *first, THSYCLTensor *second, int dimension)
{
  int first_dims = first->dim();
  int second_dims = second->dim();
  THArgCheck(first_dims == second_dims, 0,
      "Tensors must have same number of dimensions: got %d and %d",
      first_dims, second_dims);
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = THSYCLTensor_(size)(state, first, dim);
    int64_t second_dim_size = THSYCLTensor_(size)(state, second, dim);
    THArgCheck(first_dim_size == second_dim_size, 0,
        "Sizes of tensors must match except in dimension %d. Got %lld and %lld in dimension %d",
        dimension, (long long)first_dim_size, (long long)second_dim_size, dim);
  } 
}

static void THSYCLTensor_(catArray_reference__)(THSYCLState *state, 
                                                THSYCLTensor *result, THSYCLTensor **inputs, 
                                                int numInputs, int dimension)
{
  // previously, size [0] tensors were the only possible empty tensors; thus, it wasn't possible
  // to cat empty tensors unless all the other tensors were 1-dimensional, so we allowed these tensors
  // to be "skipped".  We maintain this behavior for backwards compatibility, but only for this specific
  // size (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  int i, j;
  int64_t offset;
  bool hasSkippedInput = false;
  THSYCLTensor *notSkippedTensor = NULL;  // non-owning reference
  auto should_skip = [](THSYCLTensor *t) { return t->is_empty() && t->dim() == 1; };
  int nDims = 0;

  for (i = 0; i < numInputs; i++)
  {
    if (should_skip(inputs[i])) {
      hasSkippedInput = true;
      continue;
    }
    nDims = inputs[i]->dim();
    notSkippedTensor = inputs[i];
  }

  // If all inputs are empty tensors, return an empty tensor
  if (notSkippedTensor == NULL) {
    return;
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(dimension >= 0, 4, "invalid dimension %d", dimension);

  std::vector<int64_t> size(nDims);

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    THSYCLTensor *tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    THSYCLTensor_(check_shape_except_dim)(state, notSkippedTensor, tensor, dimension);
    cat_dim_size += THSYCLTensor_(size)(state, tensor, dimension);
  }

  // Compute the size of the result
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = THSYCLTensor_(size)(state, notSkippedTensor, dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }
  THSYCLTensor_(resize)(state, result, size, {});

  offset = 0;
  for (j = 0; j < numInputs; j++) {
    if (should_skip(inputs[j])) continue;
    int64_t dimSize = THSYCLTensor_(size)(state, inputs[j], dimension);
    THSYCLTensor *nt = THSYCLTensor_(newWithTensor)(state, result);
    THSYCLTensor_(narrow)(state, nt, NULL, dimension, offset, dimSize);
    THSYCLTensor_(copy)(state, nt, inputs[j]);
    THSYCLTensor_(free)(state, nt);
    offset += dimSize;
  }
}

void THSYCLTensor_(catArray)(THSYCLState *state,
    THSYCLTensor *result, THSYCLTensor **inputs, int numInputs, int dimension)
{
  // TODO: Currently, mkldnn cat primitives has bug in 3 dimension (MFDNN-1901)
  // We will switch from reference version to mkldnn version after the bug is fixed.
  // THSYCLTensor_(catArray_mkldnn__)(state, result, inputs, numInputs, dimension);
  THSYCLTensor_(catArray_reference__)(state, result, inputs, numInputs, dimension);
}
#endif
