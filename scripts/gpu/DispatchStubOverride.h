static Tensor all(const Tensor & self); // aten::all(Tensor self) -> Tensor
static Tensor all(const Tensor & self, int64_t dim, bool keepdim); // aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
static Tensor & all_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim); // aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static Tensor any(const Tensor & self); // aten::any(Tensor self) -> Tensor
static Tensor any(const Tensor & self, int64_t dim, bool keepdim); // aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
static Tensor & any_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim); // aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking); // aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
static Tensor & fill_(Tensor & self, Scalar value); // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
static Tensor & fill_(Tensor & self, const Tensor & value); // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
static Tensor threshold(const Tensor & self, Scalar threshold, Scalar value); // aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
static Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value); // aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
static Tensor & threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value); // aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
static Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold); // aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
static Tensor bitwise_not(const Tensor & self); // aten::bitwise_not(Tensor self) -> Tensor
static Tensor & bitwise_not_(Tensor & self); // aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
static Tensor & bitwise_not_out(Tensor & out, const Tensor & self); // aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor logical_not(const Tensor & self); // aten::logical_not(Tensor self) -> Tensor
static Tensor & logical_not_(Tensor & self); // aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
static Tensor & logical_not_out(Tensor & out, const Tensor & self); // aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & neg_out(Tensor & out, const Tensor & self); // aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & round_out(Tensor & out, const Tensor & self); // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & abs_out(Tensor & out, const Tensor & self); // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & cos_out(Tensor & out, const Tensor & self); // aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & cosh_out(Tensor & out, const Tensor & self); // aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & sin_out(Tensor & out, const Tensor & self); // aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & sinh_out(Tensor & out, const Tensor & self); // aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & acos_out(Tensor & out, const Tensor & self); // aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & asin_out(Tensor & out, const Tensor & self); // aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & atan_out(Tensor & out, const Tensor & self); // aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & tan_out(Tensor & out, const Tensor & self); // aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & atan2_(Tensor & self, const Tensor & other); // aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static Tensor atan2(const Tensor & self, const Tensor & other); // aten::atan2(Tensor self, Tensor other) -> Tensor
static Tensor & atan2_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & floor_out(Tensor & out, const Tensor & self); // aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & expm1_out(Tensor & out, const Tensor & self); // aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & ceil_out(Tensor & out, const Tensor & self); // aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & frac_out(Tensor & out, const Tensor & self); // aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & trunc_out(Tensor & out, const Tensor & self); // aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & erf_out(Tensor & out, const Tensor & self); // aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & erfc_out(Tensor & out, const Tensor & self); // aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & exp_out(Tensor & out, const Tensor & self); // aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & log_out(Tensor & out, const Tensor & self); // aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & log10_out(Tensor & out, const Tensor & self); // aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & log1p_out(Tensor & out, const Tensor & self); // aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & log2_out(Tensor & out, const Tensor & self); // aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor add(const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha); // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
static Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static Tensor add(const Tensor & self, Scalar other, Scalar alpha); // aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static Tensor & add_(Tensor & self, Scalar other, Scalar alpha); // aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
static Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static Tensor & sub_(Tensor & self, const Tensor & other, Scalar alpha); // aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
static Tensor sub(const Tensor & self, Scalar other, Scalar alpha); // aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static Tensor & sub_(Tensor & self, Scalar other, Scalar alpha); // aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
static Tensor rsub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static Tensor rsub(const Tensor & self, Scalar other, Scalar alpha); // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static Tensor div(const Tensor & self, const Tensor & other); // aten::div.Tensor(Tensor self, Tensor other) -> Tensor
static Tensor & div_(Tensor & self, const Tensor & other); // aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor mul(const Tensor & self, const Tensor & other); // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
static Tensor & mul_(Tensor & self, const Tensor & other); // aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor mul(const Tensor & self, Scalar other); // aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
static Tensor & mul_(Tensor & self, Scalar other); // aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static Tensor & addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
static Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
static Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
static Tensor & addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
static Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
static Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
static Tensor sum(const Tensor & self, c10::optional<ScalarType> dtype); // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
static Tensor sum(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static Tensor & sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static Tensor norm(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype); // aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
static Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
static Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
static Tensor norm(const Tensor & self, Scalar p); // aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
static Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
static Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static Tensor digamma(const Tensor & self); // aten::digamma(Tensor self) -> Tensor
static Tensor & digamma_out(Tensor & out, const Tensor & self); // aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & digamma_(Tensor & self); // aten::digamma_(Tensor(a!) self) -> Tensor(a!)
static Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe); // aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
static Tensor index(const Tensor & self, TensorList indices); // aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
static Tensor mean(const Tensor & self, c10::optional<ScalarType> dtype); // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
static Tensor mean(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static Tensor & mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static Tensor prod(const Tensor & self, c10::optional<ScalarType> dtype); // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
static Tensor prod(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static Tensor & prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static Tensor _pdist_forward(const Tensor & self, double p); // aten::_pdist_forward(Tensor self, float p=2) -> Tensor
static Tensor _pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist); // aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor
static Tensor & cross_out(Tensor & out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim); // aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
static Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim); // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
static Tensor & real_out(Tensor & out, const Tensor & self); // aten::real.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static Tensor & conj_out(Tensor & out, const Tensor & self); // aten::conj.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static std::tuple<Tensor,Tensor> std_mean(const Tensor & self, bool unbiased); // aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
static std::tuple<Tensor,Tensor> std_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
static std::tuple<Tensor,Tensor> var_mean(const Tensor & self, bool unbiased); // aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
static std::tuple<Tensor,Tensor> var_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
static Tensor & reciprocal_out(Tensor & out, const Tensor & self); // aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
