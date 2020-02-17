// This file contains all native_functions supported by DPCPP:GPU,
// that can be registered to and the schema string that they should be registered with

Tensor add(const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha); // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)

Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)

Tensor add(const Tensor & self, Scalar other, Scalar alpha); // aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor & add_(Tensor & self, Scalar other, Scalar alpha); // aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)

Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset); // aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)

Tensor bitwise_not(const Tensor & self); // aten::bitwise_not(Tensor self) -> Tensor

Tensor & bitwise_not_(Tensor & self); // aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)

Tensor & bitwise_not_out(Tensor & out, const Tensor & self); // aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor logical_not(const Tensor & self); // aten::logical_not(Tensor self) -> Tensor

Tensor & logical_not_(Tensor & self); // aten::logical_not_(Tensor(a!) self) -> Tensor(a!)

Tensor & logical_not_out(Tensor & out, const Tensor & self); // aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups); // aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor

Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking); // aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)

Tensor div(const Tensor & self, const Tensor & other); // aten::div.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & div_(Tensor & self, const Tensor & other); // aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format); // aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

Tensor & resize_(Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format); // aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)

Tensor & fill_(Tensor & self, Scalar value); // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)

Tensor & fill_(Tensor & self, const Tensor & value); // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)

Tensor _log_softmax(const Tensor & self, int64_t dim, bool half_to_float); // aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor

Tensor mul(const Tensor & self, const Tensor & other); // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & mul_(Tensor & self, const Tensor & other); // aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor mul(const Tensor & self, Scalar other); // aten::mul.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & mul_(Tensor & self, Scalar other); // aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)

std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps); // aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)

Tensor neg(const Tensor & self); // aten::neg(Tensor self) -> Tensor

Tensor & neg_(Tensor & self); // aten::neg_(Tensor(a!) self) -> Tensor(a!)

Tensor & neg_out(Tensor & out, const Tensor & self); // aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & relu_(Tensor & self); // aten::relu_(Tensor(a!) self) -> Tensor(a!)

Tensor threshold(const Tensor & self, Scalar threshold, Scalar value); // aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor

Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value); // aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)

Tensor & threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value); // aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)

Tensor & resize_as_(Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format); // aten::resize_as_(Tensor(a!) self, Tensor the_template, *, int? memory_format=None) -> (Tensor(a!))

Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)

Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor & sub_(Tensor & self, const Tensor & other, Scalar alpha); // aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)

Tensor sub(const Tensor & self, Scalar other, Scalar alpha); // aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor & sub_(Tensor & self, Scalar other, Scalar alpha); // aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)

Tensor rsub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor rsub(const Tensor & self, Scalar other, Scalar alpha); // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

Tensor view(const Tensor & self, IntArrayRef size); // aten::view(Tensor(a) self, int[] size) -> Tensor(a)

Tensor & tril_(Tensor & self, int64_t diagonal); // aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)

Tensor & triu_(Tensor & self, int64_t diagonal); // aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)

Tensor & triu_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)

Tensor & tril_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)

Tensor & avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)



