#ifndef IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP

namespace ideep {

struct matmul_forward : public dnnl::matmul,
                        utils::computation_cache<dnnl::matmul::primitive_desc> {
  using super = dnnl::matmul;

  static void compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const std::vector<tensor>& bin_post_params = {},
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src,
        weights,
        bias,
        dst,
        dst_coeff,
        sum_coeff,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        bin_post_params,
        dst_type,
        alowp_kind,
        aengine);
  }

  static void compute(
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const std::vector<tensor>& bin_post_params = {},
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src,
        weights,
        dummy_bias,
        dst,
        dst_coeff,
        sum_coeff,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        bin_post_params,
        dst_type,
        alowp_kind,
        aengine);
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    auto ndims = weights_dims.size();
    auto x_dims = weights_dims;
    x_dims[ndims - 2] = 1;
    x_dims[ndims - 1] = weights_dims[ndims - 2];
    auto y_dims = {x_dims[0], weights_dims[1]};
    if (ndims == 3)
      y_dims = {x_dims[0], x_dims[1], weights_dims[2]};
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(
        x_dims.size() == weights_dims.size(),
        "Invalid dims for data and weights");
    tensor::desc x_desc(x_dims, x_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc y_desc(y_dims, y_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc weights_desc(
        weights_dims, dtype, ndims == 2 ? tag::ab : tag::abc);
    auto pd = primitive_desc({x_desc, weights_desc, y_desc}, aengine);
    return pd.weights_desc();
  }

 private:
  template <bool with_bias>
  static void compute_impl(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const std::vector<tensor>& bin_post_params = {},
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(
        src.ndims() == weights.ndims(), "Invalid dims in src or weights");

    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    auto dst_data_type = data_type::f32;

    tensor::dims dst_dims = {src.get_dim(0)};
    auto ndims = weights.ndims();
    for (auto i = 1; i < ndims - 1; i++) {
      dst_dims.push_back(src.get_dim(i));
    }
    dst_dims.push_back(weights.get_dim(ndims - 1));

    op_attr = attr;

    // We intentionally didn't set weight desc to format `any` so DNNL
    // wouldn't have to determine weight format for us. Because the weight
    // tensor from pytorch may have a transposed format (say `ba`). However,
    // DNNL would choose plain format for it by default (`ab` in this case),
    // which would introduces *an extra reorder* afterwards. Here we keep the
    // weight format untouched thanks to optimizations for both plain and
    // transposed formats in DNNL.
    IDEEP_ENFORCE(
        weights.get_data_type() == data_type::f32 ||
            weights.get_data_type() == data_type::bf16,
        "Incorrect data type in weights");
    dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                           : data_type::f32;
    src_desc = src.get_desc().to_type(dst_data_type);
    weights_desc = weights.get_desc().to_type(dst_data_type);
    if (with_bias) {
      IDEEP_ENFORCE(
          bias.get_data_type() == data_type::f32 ||
              bias.get_data_type() == data_type::bf16,
          "Incorrect data type in bias");
      bias_desc = bias.get_desc().to_format_any();
      auto bias_scales = scale_t(1, 1.0 / dst_coeff);
      bias_attr = {utils::tensor_scale_mask(1, false), bias_scales};
    }

    if (attr.has_op_kind(kind::sum)) {
      op_attr = attr_t::fuse_sum(sum_coeff);
    }
    int scale_size = 1;
    op_attr.set_output_scales(
        utils::op_scale_mask(scale_size), std::vector<float>(1, dst_coeff));

    op_attr.set_fpmath_mode();

    // Use user mode scratchpad
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dst_data_type = dst_type == data_type::undef ? dst_data_type : dst_type;
    tensor::desc dst_desc(dst_dims, dst_data_type, tag::any);
    if (!dst.is_empty())
      dst_desc = dst.get_desc().to_type(dst_data_type);
    auto key = utils::create_key(
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        op_attr,
        with_bias,
        omp_get_max_threads());
    auto pd = fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc(
            {src_desc, weights_desc, bias_desc, dst_desc}, op_attr, aengine);
      } else {
        return primitive_desc(
            {src_desc, weights_desc, dst_desc}, op_attr, aengine);
      }
    });
    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights =
        weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    dst.reinit_if_possible(pd.dst_desc());

    std::unordered_map<int, memory> primitive_args;
    primitive_args.insert({DNNL_ARG_SRC, expected_src});
    primitive_args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    primitive_args.insert({DNNL_ARG_DST, dst});
    for (int i = 0; i < bin_post_params.size(); i++) {
      primitive_args.insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
           bin_post_params[i]});
    }

    if (with_bias) {
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
      primitive_args.insert({DNNL_ARG_BIAS, expected_bias});
    }
    tensor scratchpad(pd.scratchpad_desc());
    primitive_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    super(pd).execute(stream::default_stream(), primitive_args);
  }
};

} // namespace ideep

#endif