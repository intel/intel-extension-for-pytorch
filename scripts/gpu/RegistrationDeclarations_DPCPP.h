// This file contains all native_functions supported by DPCPP:GPU,
// that can be registered to and the schema string that they should be registered with

Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size); // aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor

Tensor _adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self); // aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor

Tensor _cat(TensorList tensors, int64_t dim); // aten::_cat(Tensor[] tensors, int dim=0) -> Tensor

Tensor & _cat_out(Tensor & out, TensorList tensors, int64_t dim); // aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> _ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity); // aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)

Tensor _ctc_loss_backward(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity); // aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor

Tensor _cumprod(const Tensor & self, int64_t dim); // aten::_cumprod(Tensor self, int dim) -> Tensor

Tensor & _cumprod_out(Tensor & out, const Tensor & self, int64_t dim); // aten::_cumprod.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)

Tensor _cumsum(const Tensor & self, int64_t dim); // aten::_cumsum(Tensor self, int dim) -> Tensor

Tensor & _cumsum_out(Tensor & out, const Tensor & self, int64_t dim); // aten::_cumsum.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset); // aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)

Tensor _embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights); // aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor

Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe); // aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)

Scalar _local_scalar_dense(const Tensor & self); // aten::_local_scalar_dense(Tensor self) -> Scalar

Tensor _log_softmax(const Tensor & self, int64_t dim, bool half_to_float); // aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor

Tensor _log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self); // aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor

std::tuple<Tensor,Tensor> _max(const Tensor & self, int64_t dim, bool keepdim); // aten::_max(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)

std::tuple<Tensor &,Tensor &> _max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::_max.max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> _min(const Tensor & self, int64_t dim, bool keepdim); // aten::_min(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)

std::tuple<Tensor &,Tensor &> _min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::_min.min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!), Tensor(b!))

Tensor _pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist); // aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor

Tensor _pdist_forward(const Tensor & self, double p); // aten::_pdist_forward(Tensor self, float p=2) -> Tensor

Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other); // aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor

Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float); // aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor

Tensor _softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self); // aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor

Tensor _var(const Tensor & self, bool unbiased); // aten::_var(Tensor self, bool unbiased=True) -> Tensor

Tensor & abs_out(Tensor & out, const Tensor & self); // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & acos_out(Tensor & out, const Tensor & self); // aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor

Tensor & adaptive_avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)

Tensor adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor

Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self); // aten::adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor

Tensor & adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self); // aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & adaptive_avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)

Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor

Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)

Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor

Tensor & adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

Tensor add(const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor add(const Tensor & self, Scalar other, Scalar alpha); // aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha); // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)

Tensor & add_(Tensor & self, Scalar other, Scalar alpha); // aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)

Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)

Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)

Tensor & addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)

Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)

Tensor & addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)

Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)

Tensor all(const Tensor & self, int64_t dim, bool keepdim); // aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor

Tensor all(const Tensor & self); // aten::all(Tensor self) -> Tensor

Tensor & all_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim); // aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)

Tensor any(const Tensor & self, int64_t dim, bool keepdim); // aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor

Tensor any(const Tensor & self); // aten::any(Tensor self) -> Tensor

Tensor & any_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim); // aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)

Tensor & arange_out(Tensor & out, Scalar start, Scalar end, Scalar step); // aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)

Tensor argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim); // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor

Tensor argmin(const Tensor & self, c10::optional<int64_t> dim, bool keepdim); // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor

Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset); // aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)

Tensor & asin_out(Tensor & out, const Tensor & self); // aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor atan2(const Tensor & self, const Tensor & other); // aten::atan2(Tensor self, Tensor other) -> Tensor

Tensor & atan2_(Tensor & self, const Tensor & other); // aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & atan2_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & atan_(Tensor & self); // aten::atan_(Tensor(a!) self) -> Tensor(a!)

Tensor & atan_out(Tensor & out, const Tensor & self); // aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor

Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor

Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)

Tensor & baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)

Tensor & bernoulli_(Tensor & self, const Tensor & p, Generator * generator); // aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)

Tensor & bernoulli_(Tensor & self, double p, Generator * generator); // aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)

Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor

Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor

Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & binary_cross_entropy_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)

Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength); // aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor

Tensor & bitwise_and_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & bitwise_and_out(Tensor & out, const Tensor & self, Scalar other); // aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor bitwise_not(const Tensor & self); // aten::bitwise_not(Tensor self) -> Tensor

Tensor & bitwise_not_(Tensor & self); // aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)

Tensor & bitwise_not_out(Tensor & out, const Tensor & self); // aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & bitwise_or_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & bitwise_or_out(Tensor & out, const Tensor & self, Scalar other); // aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor bmm(const Tensor & self, const Tensor & mat2); // aten::bmm(Tensor self, Tensor mat2) -> Tensor

Tensor & bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2); // aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)

Tensor & ceil_out(Tensor & out, const Tensor & self); // aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor

Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)

Tensor & clamp_max_(Tensor & self, Scalar max); // aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)

Tensor & clamp_max_out(Tensor & out, const Tensor & self, Scalar max); // aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)

Tensor & clamp_min_(Tensor & self, Scalar min); // aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)

Tensor & clamp_min_out(Tensor & out, const Tensor & self, Scalar min); // aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)

Tensor & clamp_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format); // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor

Tensor col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor

Tensor col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor

Tensor & col2im_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im_backward.grad_input(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & col2im_out(Tensor & out, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)

Tensor & conj_out(Tensor & out, const Tensor & self); // aten::conj.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask); // aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)

Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups); // aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor

Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking); // aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)

Tensor & cos_(Tensor & self); // aten::cos_(Tensor(a!) self) -> Tensor(a!)

Tensor & cos_out(Tensor & out, const Tensor & self); // aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & cosh_out(Tensor & out, const Tensor & self); // aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim); // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor

Tensor & cross_out(Tensor & out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim); // aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor diag(const Tensor & self, int64_t diagonal); // aten::diag(Tensor self, int diagonal=0) -> Tensor

Tensor & diag_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)

Tensor digamma(const Tensor & self); // aten::digamma(Tensor self) -> Tensor

Tensor & digamma_(Tensor & self); // aten::digamma_(Tensor(a!) self) -> Tensor(a!)

Tensor & digamma_out(Tensor & out, const Tensor & self); // aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor div(const Tensor & self, const Tensor & other); // aten::div.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & div_(Tensor & self, const Tensor & other); // aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale); // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor

Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale); // aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)

Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output); // aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor

Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output); // aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & elu_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale); // aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)

Tensor embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq); // aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor

Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format); // aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options); // aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

Tensor eq(const Tensor & self, Scalar other); // aten::eq.Scalar(Tensor self, Scalar other) -> Tensor

Tensor eq(const Tensor & self, const Tensor & other); // aten::eq.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & eq_out(Tensor & out, const Tensor & self, Scalar other); // aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

bool equal(const Tensor & self, const Tensor & other); // aten::equal(Tensor self, Tensor other) -> bool

Tensor & erf_(Tensor & self); // aten::erf_(Tensor(a!) self) -> Tensor(a!)

Tensor & erf_out(Tensor & out, const Tensor & self); // aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & erfc_(Tensor & self); // aten::erfc_(Tensor(a!) self) -> Tensor(a!)

Tensor & erfc_out(Tensor & out, const Tensor & self); // aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor erfinv(const Tensor & self); // aten::erfinv(Tensor self) -> Tensor

Tensor & erfinv_(Tensor & self); // aten::erfinv_(Tensor(a!) self) -> Tensor(a!)

Tensor & erfinv_out(Tensor & out, const Tensor & self); // aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & exp_(Tensor & self); // aten::exp_(Tensor(a!) self) -> Tensor(a!)

Tensor & exp_out(Tensor & out, const Tensor & self); // aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & expm1_out(Tensor & out, const Tensor & self); // aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & exponential_(Tensor & self, double lambd, Generator * generator); // aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)

Tensor & eye_out(Tensor & out, int64_t n); // aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)

Tensor & eye_out(Tensor & out, int64_t n, int64_t m); // aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)

Tensor & fill_(Tensor & self, Scalar value); // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)

Tensor & fill_(Tensor & self, const Tensor & value); // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)

Tensor & floor_out(Tensor & out, const Tensor & self); // aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor fmod(const Tensor & self, Scalar other); // aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor

Tensor fmod(const Tensor & self, const Tensor & other); // aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & fmod_(Tensor & self, Scalar other); // aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)

Tensor & fmod_(Tensor & self, const Tensor & other); // aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & fmod_out(Tensor & out, const Tensor & self, Scalar other); // aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & fmod_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)

Tensor fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor

Tensor & fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> fractional_max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)

Tensor fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor

Tensor & fractional_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor &,Tensor &> fractional_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad); // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor

Tensor & gather_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad); // aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)

Tensor ge(const Tensor & self, Scalar other); // aten::ge.Scalar(Tensor self, Scalar other) -> Tensor

Tensor ge(const Tensor & self, const Tensor & other); // aten::ge.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & ge_out(Tensor & out, const Tensor & self, Scalar other); // aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor gelu(const Tensor & self); // aten::gelu(Tensor self) -> Tensor

Tensor gelu_backward(const Tensor & grad, const Tensor & self); // aten::gelu_backward(Tensor grad, Tensor self) -> Tensor

Tensor glu(const Tensor & self, int64_t dim); // aten::glu(Tensor self, int dim=-1) -> Tensor

Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim); // aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor

Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim); // aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & glu_out(Tensor & out, const Tensor & self, int64_t dim); // aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)

Tensor gt(const Tensor & self, Scalar other); // aten::gt.Scalar(Tensor self, Scalar other) -> Tensor

Tensor gt(const Tensor & self, const Tensor & other); // aten::gt.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & gt_out(Tensor & out, const Tensor & self, Scalar other); // aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor hardshrink(const Tensor & self, Scalar lambd); // aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor

Tensor hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd); // aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor

Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor

Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)

Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor

Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & hardtanh_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)

Tensor im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor

Tensor im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor

Tensor & im2col_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col_backward.grad_input(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & im2col_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)

Tensor index(const Tensor & self, TensorList indices); // aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor

Tensor & index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source); // aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value); // aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value); // aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)

Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index); // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor

Tensor & index_select_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index); // aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)

bool is_set_to(const Tensor & self, const Tensor & tensor); // aten::is_set_to(Tensor self, Tensor tensor) -> bool

Tensor isnan(const Tensor & self); // aten::isnan(Tensor self) -> Tensor

std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim); // aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)

Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor

Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)

Tensor le(const Tensor & self, Scalar other); // aten::le.Scalar(Tensor self, Scalar other) -> Tensor

Tensor le(const Tensor & self, const Tensor & other); // aten::le.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & le_out(Tensor & out, const Tensor & self, Scalar other); // aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor leaky_relu(const Tensor & self, Scalar negative_slope); // aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor

Tensor & leaky_relu_(Tensor & self, Scalar negative_slope); // aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)

Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope, bool self_is_result); // aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor

Tensor & leaky_relu_out(Tensor & out, const Tensor & self, Scalar negative_slope); // aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)

Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight); // aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor

Tensor lerp(const Tensor & self, const Tensor & end, const Tensor & weight); // aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor

Tensor & lerp_(Tensor & self, const Tensor & end, Scalar weight); // aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)

Tensor & lerp_(Tensor & self, const Tensor & end, const Tensor & weight); // aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)

Tensor & lerp_out(Tensor & out, const Tensor & self, const Tensor & end, Scalar weight); // aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)

Tensor & lerp_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight); // aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)

Tensor & linspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps); // aten::linspace.out(Scalar start, Scalar end, int steps=100, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log10_out(Tensor & out, const Tensor & self); // aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log1p_out(Tensor & out, const Tensor & self); // aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log2_out(Tensor & out, const Tensor & self); // aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & log_out(Tensor & out, const Tensor & self); // aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer); // aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor

Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer); // aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self); // aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)

std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self); // aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))

Tensor logical_not(const Tensor & self); // aten::logical_not(Tensor self) -> Tensor

Tensor & logical_not_(Tensor & self); // aten::logical_not_(Tensor(a!) self) -> Tensor(a!)

Tensor & logical_not_out(Tensor & out, const Tensor & self); // aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & logical_xor_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & logspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base); // aten::logspace.out(Scalar start, Scalar end, int steps=100, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)

Tensor lt(const Tensor & self, Scalar other); // aten::lt.Scalar(Tensor self, Scalar other) -> Tensor

Tensor lt(const Tensor & self, const Tensor & other); // aten::lt.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & lt_out(Tensor & out, const Tensor & self, Scalar other); // aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & masked_fill_(Tensor & self, const Tensor & mask, Scalar value); // aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)

Tensor & masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value); // aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)

Tensor & masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source); // aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)

Tensor masked_select(const Tensor & self, const Tensor & mask); // aten::masked_select(Tensor self, Tensor mask) -> Tensor

Tensor & masked_select_out(Tensor & out, const Tensor & self, const Tensor & mask); // aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim); // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)

Tensor max(const Tensor & self, const Tensor & other); // aten::max.other(Tensor self, Tensor other) -> Tensor

Tensor max(const Tensor & self); // aten::max(Tensor self) -> Tensor

std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim); // aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)

Tensor & max_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

Tensor max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor

Tensor & max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor

Tensor & max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))

Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor

Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor

Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & max_unpool2d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)

Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor

Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor

Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & max_unpool3d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)

Tensor mean(const Tensor & self, c10::optional<ScalarType> dtype); // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor

Tensor mean(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor

Tensor & mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)

Tensor median(const Tensor & self); // aten::median(Tensor self) -> Tensor

std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim); // aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)

Tensor min(const Tensor & self, const Tensor & other); // aten::min.other(Tensor self, Tensor other) -> Tensor

Tensor min(const Tensor & self); // aten::min(Tensor self) -> Tensor

std::tuple<Tensor &,Tensor &> min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)

Tensor & min_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor mm(const Tensor & self, const Tensor & mat2); // aten::mm(Tensor self, Tensor mat2) -> Tensor

Tensor & mm_out(Tensor & out, const Tensor & self, const Tensor & mat2); // aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)

Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor

Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & mse_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)

Tensor mul(const Tensor & self, const Tensor & other); // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor

Tensor mul(const Tensor & self, Scalar other); // aten::mul.Scalar(Tensor self, Scalar other) -> Tensor

Tensor & mul_(Tensor & self, const Tensor & other); // aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & mul_(Tensor & self, Scalar other); // aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)

Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor

Tensor multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor

Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & multi_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)

Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target); // aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor

Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target); // aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)

std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))

Tensor & multilabel_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)

Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator); // aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor

Tensor & multinomial_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator); // aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)

Tensor narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length); // aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor

std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps); // aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)

std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_invstd, bool train, double eps, std::array<bool,3> output_mask); // aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

std::tuple<Tensor,Tensor,Tensor> native_layer_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t M, int64_t N, double eps); // aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)

std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t M, int64_t N, std::array<bool,3> output_mask); // aten::native_layer_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

Tensor ne(const Tensor & self, Scalar other); // aten::ne.Scalar(Tensor self, Scalar other) -> Tensor

Tensor ne(const Tensor & self, const Tensor & other); // aten::ne.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & ne_out(Tensor & out, const Tensor & self, Scalar other); // aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & neg_out(Tensor & out, const Tensor & self); // aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor

Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)

std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)

std::tuple<Tensor &,Tensor &> nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))

Tensor nonzero(const Tensor & self); // aten::nonzero(Tensor self) -> Tensor

Tensor & nonzero_out(Tensor & out, const Tensor & self); // aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor norm(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype); // aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor

Tensor norm(const Tensor & self, Scalar p); // aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor

Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor

Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor

Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)

Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)

Tensor & normal_(Tensor & self, double mean, double std, Generator * generator); // aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)

Tensor pow(const Tensor & self, Scalar exponent); // aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor

Tensor pow(const Tensor & self, const Tensor & exponent); // aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor

Tensor pow(Scalar self, const Tensor & exponent); // aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor

Tensor & pow_(Tensor & self, Scalar exponent); // aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)

Tensor & pow_(Tensor & self, const Tensor & exponent); // aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)

Tensor & pow_out(Tensor & out, const Tensor & self, Scalar exponent); // aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)

Tensor & pow_out(Tensor & out, const Tensor & self, const Tensor & exponent); // aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)

Tensor & pow_out(Tensor & out, Scalar self, const Tensor & exponent); // aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)

Tensor prelu(const Tensor & self, const Tensor & weight); // aten::prelu(Tensor self, Tensor weight) -> Tensor

std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight); // aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)

Tensor prod(const Tensor & self, c10::optional<ScalarType> dtype); // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor

Tensor prod(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor

Tensor & prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)

Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate); // aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)

Tensor & range_out(Tensor & out, Scalar start, Scalar end, Scalar step); // aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)

Tensor & reciprocal_out(Tensor & out, const Tensor & self); // aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor relu(const Tensor & self); // aten::relu(Tensor self) -> Tensor

Tensor & relu_(Tensor & self); // aten::relu_(Tensor(a!) self) -> Tensor(a!)

Tensor remainder(const Tensor & self, Scalar other); // aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor

Tensor remainder(const Tensor & self, const Tensor & other); // aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor

Tensor & remainder_(Tensor & self, Scalar other); // aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)

Tensor & remainder_(Tensor & self, const Tensor & other); // aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

Tensor & remainder_out(Tensor & out, const Tensor & self, Scalar other); // aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)

Tensor & remainder_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm); // aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor

Tensor & renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm); // aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)

Tensor & renorm_out(Tensor & out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm); // aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)

Tensor repeat_interleave(const Tensor & repeats); // aten::repeat_interleave.Tensor(Tensor repeats) -> Tensor

Tensor & resize_(Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format); // aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)

Tensor & resize_as_(Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format); // aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)

Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims); // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor

Tensor & round_out(Tensor & out, const Tensor & self); // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)

Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor

Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)

Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, bool self_is_result); // aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor

Tensor & rrelu_with_noise_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor & rsqrt_out(Tensor & out, const Tensor & self); // aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor rsub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor rsub(const Tensor & self, Scalar other, Scalar alpha); // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src); // aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value); // aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src); // aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)

Tensor & set_(Tensor & self, Storage source); // aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)

Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride); // aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)

Tensor & set_(Tensor & self, const Tensor & source); // aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)

Tensor & set_(Tensor & self); // aten::set_(Tensor(a!) self) -> Tensor(a!)

Tensor sigmoid(const Tensor & self); // aten::sigmoid(Tensor self) -> Tensor

Tensor & sigmoid_(Tensor & self); // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)

Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output); // aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor

Tensor & sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output); // aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & sigmoid_out(Tensor & out, const Tensor & self); // aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor sign(const Tensor & self); // aten::sign(Tensor self) -> Tensor

Tensor & sign_(Tensor & self); // aten::sign_(Tensor(a!) self) -> Tensor(a!)

Tensor & sign_out(Tensor & out, const Tensor & self); // aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & sin_out(Tensor & out, const Tensor & self); // aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor & sinh_out(Tensor & out, const Tensor & self); // aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor

Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & smooth_l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)

Tensor soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

Tensor soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor

Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & soft_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)

Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold); // aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor

Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output); // aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor

Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output); // aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & softplus_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold); // aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)

Tensor softshrink(const Tensor & self, Scalar lambd); // aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor

Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd); // aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor

Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd); // aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & softshrink_out(Tensor & out, const Tensor & self, Scalar lambd); // aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending); // aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)

std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending); // aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)

Tensor & sqrt_out(Tensor & out, const Tensor & self); // aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> std_mean(const Tensor & self, bool unbiased); // aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)

std::tuple<Tensor,Tensor> std_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)

Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

Tensor sub(const Tensor & self, Scalar other, Scalar alpha); // aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

Tensor & sub_(Tensor & self, const Tensor & other, Scalar alpha); // aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)

Tensor & sub_(Tensor & self, Scalar other, Scalar alpha); // aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)

Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)

Tensor sum(const Tensor & self, c10::optional<ScalarType> dtype); // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor

Tensor sum(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor

Tensor & sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)

Tensor take(const Tensor & self, const Tensor & index); // aten::take(Tensor self, Tensor index) -> Tensor

Tensor & take_out(Tensor & out, const Tensor & self, const Tensor & index); // aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)

Tensor & tan_(Tensor & self); // aten::tan_(Tensor(a!) self) -> Tensor(a!)

Tensor & tan_out(Tensor & out, const Tensor & self); // aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor tanh(const Tensor & self); // aten::tanh(Tensor self) -> Tensor

Tensor & tanh_(Tensor & self); // aten::tanh_(Tensor(a!) self) -> Tensor(a!)

Tensor tanh_backward(const Tensor & grad_output, const Tensor & output); // aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor

Tensor & tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output); // aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & tanh_out(Tensor & out, const Tensor & self); // aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor threshold(const Tensor & self, Scalar threshold, Scalar value); // aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor

Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value); // aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)

Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold); // aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor

Tensor & threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value); // aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)

std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted); // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)

std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted); // aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)

Tensor trace(const Tensor & self); // aten::trace(Tensor self) -> Tensor

Tensor & tril_(Tensor & self, int64_t diagonal); // aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)

Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options); // aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

Tensor & tril_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)

Tensor & triu_(Tensor & self, int64_t diagonal); // aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)

Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options); // aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

Tensor & triu_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)

Tensor & trunc_out(Tensor & out, const Tensor & self); // aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step); // aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)

Tensor & uniform_(Tensor & self, double from, double to, Generator * generator); // aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)

Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w); // aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor

Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w); // aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor

Tensor & upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w); // aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)

Tensor & upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w); // aten::upsample_nearest2d.out(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)

Tensor var(const Tensor & self, bool unbiased); // aten::var(Tensor self, bool unbiased=True) -> Tensor

Tensor var(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor

std::tuple<Tensor,Tensor> var_mean(const Tensor & self, bool unbiased); // aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)

std::tuple<Tensor,Tensor> var_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)

Tensor view(const Tensor & self, IntArrayRef size); // aten::view(Tensor(a) self, int[] size) -> Tensor(a)

Tensor & zero_(Tensor & self); // aten::zero_(Tensor(a!) self) -> Tensor(a!)



