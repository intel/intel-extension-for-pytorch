// This file contains all native_functions supported by DPCPP:GPU,
// that can be registered to and the schema string that they should be registered with

Tensor & abs_out(Tensor & out, const Tensor & self); // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & acos_out(Tensor & out, const Tensor & self); // aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor add(const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha); // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)

Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)

Tensor add(const Tensor & self, Scalar other, Scalar alpha); // aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor & add_(Tensor & self, Scalar other, Scalar alpha); // aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)

Tensor & arange_out(Tensor & out, Scalar start, Scalar end, Scalar step); // aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)

Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset); // aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)

Tensor & asin_out(Tensor & out, const Tensor & self); // aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & atan_(Tensor & self); // aten::atan_(Tensor(a!) self) -> Tensor(a!)

Tensor & atan_out(Tensor & out, const Tensor & self); // aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor bitwise_not(const Tensor & self); // aten::bitwise_not(Tensor self) -> Tensor

Tensor & bitwise_not_(Tensor & self); // aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)

Tensor & bitwise_not_out(Tensor & out, const Tensor & self); // aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor logical_not(const Tensor & self); // aten::logical_not(Tensor self) -> Tensor

Tensor & logical_not_(Tensor & self); // aten::logical_not_(Tensor(a!) self) -> Tensor(a!)

Tensor & logical_not_out(Tensor & out, const Tensor & self); // aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & ceil_out(Tensor & out, const Tensor & self); // aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)

Tensor & clamp_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor & clamp_max_(Tensor & self, Scalar max); // aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)

Tensor & clamp_max_out(Tensor & out, const Tensor & self, Scalar max); // aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)

Tensor & clamp_min_(Tensor & self, Scalar min); // aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)

Tensor & clamp_min_out(Tensor & out, const Tensor & self, Scalar min); // aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)

Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups); // aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor

std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask); // aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)

Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking); // aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)

Tensor & cos_out(Tensor & out, const Tensor & self); // aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & cosh_out(Tensor & out, const Tensor & self); // aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor div(const Tensor & self, const Tensor & other); // aten::div.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & div_(Tensor & self, const Tensor & other); // aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format); // aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

Tensor & resize_(Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format); // aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options); // aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

Tensor & erf_(Tensor & self); // aten::erf_(Tensor(a!) self) -> Tensor(a!)

Tensor & erf_out(Tensor & out, const Tensor & self); // aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & erfc_(Tensor & self); // aten::erfc_(Tensor(a!) self) -> Tensor(a!)

Tensor & erfc_out(Tensor & out, const Tensor & self); // aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & exp_(Tensor & self); // aten::exp_(Tensor(a!) self) -> Tensor(a!)

Tensor & exp_out(Tensor & out, const Tensor & self); // aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & expm1_out(Tensor & out, const Tensor & self); // aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & eye_out(Tensor & out, int64_t n); // aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)

Tensor & eye_out(Tensor & out, int64_t n, int64_t m); // aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)

Tensor & fill_(Tensor & self, Scalar value); // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)

Tensor & fill_(Tensor & self, const Tensor & value); // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)

Tensor & floor_out(Tensor & out, const Tensor & self); // aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & linspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps); // aten::linspace.out(Scalar start, Scalar end, int steps=100, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log_out(Tensor & out, const Tensor & self); // aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log10_out(Tensor & out, const Tensor & self); // aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log1p_out(Tensor & out, const Tensor & self); // aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log2_out(Tensor & out, const Tensor & self); // aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & logspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base); // aten::logspace.out(Scalar start, Scalar end, int steps=100, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)

Tensor _log_softmax(const Tensor & self, int64_t dim, bool half_to_float); // aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor

std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim); // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)

std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim); // aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)

std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim); // aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)

std::tuple<Tensor &,Tensor &> min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)

Tensor mul(const Tensor & self, const Tensor & other); // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & mul_(Tensor & self, const Tensor & other); // aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor mul(const Tensor & self, Scalar other); // aten::mul.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & mul_(Tensor & self, Scalar other); // aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)

std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps); // aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)

Tensor & range_out(Tensor & out, Scalar start, Scalar end, Scalar step); // aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)

Tensor & neg_out(Tensor & out, const Tensor & self); // aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & round_out(Tensor & out, const Tensor & self); // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & relu_(Tensor & self); // aten::relu_(Tensor(a!) self) -> Tensor(a!)

Tensor & rsqrt_out(Tensor & out, const Tensor & self); // aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor sigmoid(const Tensor & self); // aten::sigmoid(Tensor self) -> Tensor

Tensor & sigmoid_(Tensor & self); // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)

Tensor & sigmoid_out(Tensor & out, const Tensor & self); // aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & sin_out(Tensor & out, const Tensor & self); // aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & sinh_out(Tensor & out, const Tensor & self); // aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor sum(const Tensor & self, c10::optional<ScalarType> dtype); // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor

Tensor sum(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor

Tensor & sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)

Tensor & sqrt_out(Tensor & out, const Tensor & self); // aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & tan_(Tensor & self); // aten::tan_(Tensor(a!) self) -> Tensor(a!)

Tensor & tan_out(Tensor & out, const Tensor & self); // aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & tanh_(Tensor & self); // aten::tanh_(Tensor(a!) self) -> Tensor(a!)

Tensor & tanh_out(Tensor & out, const Tensor & self); // aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor threshold(const Tensor & self, Scalar threshold, Scalar value); // aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor

Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value); // aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)

Tensor & threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value); // aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)

Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims); // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor

Tensor & trunc_out(Tensor & out, const Tensor & self); // aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor norm(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype); // aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor

Tensor norm(const Tensor & self, Scalar p); // aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor

Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor

Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor

Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)

Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)

Tensor & resize_as_(Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format); // aten::resize_as_(Tensor(a!) self, Tensor the_template, *, int? memory_format=None) -> (Tensor(a!))

Tensor & pow_out(Tensor & out, const Tensor & self, Scalar exponent); // aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)

Tensor pow(const Tensor & self, Scalar exponent); // aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor

Tensor & zero_(Tensor & self); // aten::zero_(Tensor(a!) self) -> Tensor(a!)

Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)

Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor & sub_(Tensor & self, const Tensor & other, Scalar alpha); // aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)

Tensor sub(const Tensor & self, Scalar other, Scalar alpha); // aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor & sub_(Tensor & self, Scalar other, Scalar alpha); // aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)

Tensor rsub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor rsub(const Tensor & self, Scalar other, Scalar alpha); // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

Scalar _local_scalar_dense(const Tensor & self); // aten::_local_scalar_dense(Tensor self) -> Scalar

Tensor & set_(Tensor & self, Storage source); // aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)

Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride); // aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)

Tensor & set_(Tensor & self, const Tensor & source); // aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)

Tensor & set_(Tensor & self); // aten::set_(Tensor(a!) self) -> Tensor(a!)

Tensor view(const Tensor & self, IntArrayRef size); // aten::view(Tensor(a) self, int[] size) -> Tensor(a)

Tensor __and__(const Tensor & self, Scalar other); // aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor

Tensor __and__(const Tensor & self, const Tensor & other); // aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & __iand__(Tensor & self, Scalar other); // aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)

Tensor & __iand__(Tensor & self, const Tensor & other); // aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor __or__(const Tensor & self, Scalar other); // aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor

Tensor __or__(const Tensor & self, const Tensor & other); // aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & __ior__(Tensor & self, Scalar other); // aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)

Tensor & __ior__(Tensor & self, const Tensor & other); // aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & tril_(Tensor & self, int64_t diagonal); // aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)

Tensor & triu_(Tensor & self, int64_t diagonal); // aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)

Tensor & pow_(Tensor & self, Scalar exponent); // aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)

Tensor & pow_(Tensor & self, const Tensor & exponent); // aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)

Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)

Tensor & triu_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)

Tensor & tril_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)

Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options); // aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options); // aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

Tensor & ne_out(Tensor & out, const Tensor & self, Scalar other); // aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor ne(const Tensor & self, Scalar other); // aten::ne.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor ne(const Tensor & self, const Tensor & other); // aten::ne.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & eq_out(Tensor & out, const Tensor & self, Scalar other); // aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor eq(const Tensor & self, Scalar other); // aten::eq.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor eq(const Tensor & self, const Tensor & other); // aten::eq.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & ge_out(Tensor & out, const Tensor & self, Scalar other); // aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor ge(const Tensor & self, Scalar other); // aten::ge.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor ge(const Tensor & self, const Tensor & other); // aten::ge.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & le_out(Tensor & out, const Tensor & self, Scalar other); // aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor le(const Tensor & self, Scalar other); // aten::le.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor le(const Tensor & self, const Tensor & other); // aten::le.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & gt_out(Tensor & out, const Tensor & self, Scalar other); // aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor gt(const Tensor & self, Scalar other); // aten::gt.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor gt(const Tensor & self, const Tensor & other); // aten::gt.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & lt_out(Tensor & out, const Tensor & self, Scalar other); // aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor lt(const Tensor & self, Scalar other); // aten::lt.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor lt(const Tensor & self, const Tensor & other); // aten::lt.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)

Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)

Tensor & addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)

Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

Tensor & min_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor min(const Tensor & self, const Tensor & other); // aten::min.other(Tensor self, Tensor other) -> Tensor

Tensor min(const Tensor & self); // aten::min(Tensor self) -> Tensor

Tensor & max_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor max(const Tensor & self, const Tensor & other); // aten::max.other(Tensor self, Tensor other) -> Tensor

Tensor max(const Tensor & self); // aten::max(Tensor self) -> Tensor

std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted); // aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)

std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted); // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)

Tensor & pow_out(Tensor & out, const Tensor & self, const Tensor & exponent); // aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)

Tensor pow(const Tensor & self, const Tensor & exponent); // aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor

Tensor & pow_out(Tensor & out, Scalar self, const Tensor & exponent); // aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)

Tensor pow(Scalar self, const Tensor & exponent); // aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor

std::tuple<Tensor,Tensor> _max(const Tensor & self, int64_t dim, bool keepdim); // aten::_max(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)

std::tuple<Tensor &,Tensor &> _max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::_max.max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> _min(const Tensor & self, int64_t dim, bool keepdim); // aten::_min(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)

std::tuple<Tensor &,Tensor &> _min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::_min.min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor &,Tensor &> nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)

Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor

Tensor & softplus_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold); // aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)

Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold); // aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor

Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output); // aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output); // aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor

Tensor & adaptive_avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)

Tensor adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor

Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size); // aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor

Tensor _adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self); // aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor

Tensor & avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor

std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

Tensor & max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor

Tensor & upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)

Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest2d(Tensor self, int[2] output_size) -> Tensor

Tensor & upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size) -> Tensor

Tensor & sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output); // aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output); // aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor

Tensor & tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output); // aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor tanh_backward(const Tensor & grad_output, const Tensor & output); // aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor



