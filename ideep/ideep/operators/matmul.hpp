#ifndef IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP

namespace ideep {

struct matmul_forward : public dnnl::matmul {

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
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(src, weights, bias, dst, dst_coeff, sum_coeff,
                                     src_scales, weights_scales, dst_scales,
                                     attr, dst_type, alowp_kind, aengine);
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
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(src, weights, dummy_bias, dst, dst_coeff,
                                      sum_coeff, src_scales, weights_scales,
                                      dst_scales, attr, dst_type, alowp_kind, aengine);
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    auto ndims = weights_dims.size();
    auto x_dims = weights_dims;
    x_dims[ndims-2] = 1;
    x_dims[ndims-1] = weights_dims[ndims-2];
    auto y_dims = {x_dims[0], weights_dims[1]};
    if (ndims == 3) 
        y_dims = {x_dims[0], x_dims[1], weights_dims[2]};
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc x_desc(x_dims, x_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc y_desc(y_dims, y_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc weights_desc(weights_dims , dtype, ndims == 2 ? tag::ab : tag::abc);
    auto pd = primitive_desc({x_desc, weights_desc, y_desc}, aengine);
    return pd.weights_desc();
  }

private:
  template <bool with_bias>
 static void compute_impl(const tensor& src,
                          const tensor& weights,
                          const tensor& bias,
                          tensor& dst,
                          const float dst_coeff = 1.0f,
                          const float sum_coeff = 1.0f,
                          const scale_t& src_scales = scale_t(),
                          const scale_t& weights_scales = scale_t(),
                          const scale_t& dst_scales = scale_t(),
                          const attr_t& attr = attr_t(),
                          const data_type dst_type = data_type::undef,
                          const lowp_kind alowp_kind = u8s8,
                          const engine& aengine = engine::cpu_engine()) {
   IDEEP_ENFORCE(src.ndims() == weights.ndims(), "Invalid dims in src or weights");

   tensor::desc src_desc, weights_desc, bias_desc;
   attr_t op_attr, src_attr, weights_attr, bias_attr;
   scale_t dst_scales_in;
   auto dst_data_type = data_type::f32;

   tensor::dims dst_dims = {src.get_dim(0), weights.get_dim(1)};
   auto ndims = weights.ndims();
   if (ndims == 3) 
       dst_dims = {src.get_dim(0), src.get_dim(1), weights.get_dim(2)};

   auto weights_scales_in =
       weights.has_scale() ? weights.get_scale() : weights_scales;
   tensor scales_m, src_zero_point_m, wei_zero_point_m, dst_zero_point_m;
   if (!weights_scales_in.empty()) {
     IDEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                   "Unsupported lowp kind");

     auto src_scales_in = 
         src.has_scale() ? src.get_scale()
                         : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);
     src_desc = {src.get_dims(),
                 alowp_kind == u8s8 ? data_type::u8 : data_type::s8,
                 tag::any};
     if (src.get_data_type() == data_type::f32) {
       src_attr = {0, src_scales_in};
     }

     int scale_size = (weights_scales_in.size() > 1) ? weights.get_dim(1) : 1;
     weights_desc = weights.get_desc();
     if (weights.get_data_type() == data_type::f32) {
       weights_attr = {utils::tensor_scale_mask(scale_size, false), 
                       weights_scales_in};
     }
     
     // determine dst data type
     if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
       dst_data_type = data_type::f32;
     } else {
       dst_data_type = data_type::u8;
     }

     // fill primitive attr
     scale_t op_scales(scale_size), bias_scales(scale_size);
     dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32) 
                          ? IDEEP_DEF_SCALE 
                          : dst_scales;
     auto src_zero_point = src.has_zero_point()
                           ? src.get_zero_point() : std::vector<int32_t>(1);
     auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
     auto dst_zero_point = dst.has_zero_point()
                           ? dst.get_zero_point() : std::vector<int32_t>(1);
     auto dst_zero_point_size = static_cast<dim>(dst_zero_point.size());
     IDEEP_ENFORCE(src_zero_point_size == 1 && dst_zero_point_size == 1, 
                   "DNNL only support 1-dim zero_point");
     auto wei_zero_point = weights.has_zero_point()
                           ? weights.get_zero_point() : std::vector<int32_t>(1);
     dim wei_zero_point_size = 1;
    
     if (attr.has_op_kind(kind::sum)) {
       float sum_scale = 
           sum_coeff * dst_scales_in[0] / (dst.has_scale() ? dst.get_scale()[0] : 1.0f); 
       op_attr = attr_t::fuse_sum(sum_scale);
     }

     auto bias_scales_in =
         bias.has_scale() ? bias.get_scale() : IDEEP_DEF_SCALE;
     bias_scales_in = bias_scales_in.size() == 1 ? 
	     std::vector<float>(scale_size, bias_scales_in[0]) : bias_scales_in; 
     bool flag_runtime = false;
     if (flag_runtime) {
       op_attr.set_output_scales(utils::op_scale_mask(scale_size), {DNNL_RUNTIME_F32_VAL});
       tensor::desc scales_desc = {{scale_size}, data_type::f32, {1}};
       scales_m.init(scales_desc, aengine);
       auto s = reinterpret_cast<float *>(scales_m.get_data_handle());
       for (memory::dim i = 0; i < scale_size; ++i) {
         bias_scales[i] = src_scales_in[0] * weights_scales_in[i] 
		 / (dst_coeff * bias_scales_in[i]);
         s[i] = dst_coeff * dst_scales_in[0] / (src_scales_in[0] * weights_scales_in[i]);
       }
       
       op_attr.set_zero_points(DNNL_ARG_SRC, utils::tensor_zp_mask(1), {DNNL_RUNTIME_S32_VAL});
       tensor::desc src_zero_point_desc = {{src_zero_point_size}, data_type::s32, {1}};
       src_zero_point_m.init(src_zero_point_desc, aengine);
       auto src_z = reinterpret_cast<int32_t *>(src_zero_point_m.get_data_handle());
       for (memory::dim i = 0; i < src_zero_point_size; ++i)
         src_z[i] = src_zero_point[i];
         
       op_attr.set_zero_points(DNNL_ARG_WEIGHTS, utils::tensor_zp_mask(1), {DNNL_RUNTIME_S32_VAL});
       tensor::desc wei_zero_point_desc = {{wei_zero_point_size}, data_type::s32, {1}};
       wei_zero_point_m.init(wei_zero_point_desc, aengine);
       auto wei_z = reinterpret_cast<int32_t *>(wei_zero_point_m.get_data_handle());
       for (memory::dim i = 0; i < wei_zero_point_size; ++i)
         wei_z[i] = wei_zero_point[i];
     
       if (dst_data_type != data_type::f32) {
         op_attr.set_zero_points(DNNL_ARG_DST, utils::tensor_zp_mask(1), {DNNL_RUNTIME_S32_VAL});
         tensor::desc dst_zero_point_desc = {{dst_zero_point_size}, data_type::s32, {1}};
         dst_zero_point_m.init(dst_zero_point_desc, aengine);
         auto dst_z = reinterpret_cast<int32_t *>(dst_zero_point_m.get_data_handle());
         for (memory::dim i = 0; i < dst_zero_point_size; ++i)
           dst_z[i] = dst_zero_point[i];
       }
     } else {
       for (int i = 0; i < scale_size; i++) {
         bias_scales[i] = src_scales_in[0] * weights_scales_in[i] 
		 / (dst_coeff * bias_scales_in[i]);
         op_scales[i] = dst_coeff * dst_scales_in[0] / (src_scales_in[0] * weights_scales_in[i]);
       }
       op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);
       op_attr.set_zero_points(DNNL_ARG_SRC, 
                               utils::tensor_zp_mask(src_zero_point.size()), src_zero_point);
       op_attr.set_zero_points(DNNL_ARG_WEIGHTS, 
                               utils::tensor_zp_mask(1), std::vector<int32_t>(1,wei_zero_point[0]));
       if (dst_data_type != data_type::f32) {
         op_attr.set_zero_points(DNNL_ARG_DST, 
                                 utils::tensor_zp_mask(dst_zero_point.size()), dst_zero_point);
       }
     }

     if (with_bias) {
       tag bia_tag = bias.get_dims().size() == 2 ? tag::ab : tag::abc;
       bias_desc = {bias.get_dims(), data_type::s32, bia_tag};
       if (bias.get_data_type() != data_type::s32) {
         auto ndims = bias.get_dims().size(); 
         int mask = scale_size > 1 ? 1 << (ndims - 1) : 0;
         bias_attr = {mask, bias_scales};
       }
     }
   } else {
     op_attr = attr;
     if (src.has_scale()) {
       auto src_scale = src.get_scale();
       src_scale[0] = 1.0f / src_scale[0];
       src_attr = {0, src_scale};
     }

     // We intentionally didn't set weight desc to format `any` so DNNL wouldn't
     // have to determine weight format for us. Because the weight tensor from
     // pytorch may have a transposed format (say `ba`). However, DNNL would
     // choose plain format for it by default (`ab` in this case), which would
     // introduces *an extra reorder* afterwards. Here we keep the weight format
     // untouched thanks to optimizations for both plain and transposed formats
     // in DNNL.
     IDEEP_ENFORCE(weights.get_data_type() == data_type::f32 ||
		   weights.get_data_type() == data_type::bf16,
                   "Incorrect data type in weights");
     dst_data_type = src.get_data_type() == data_type::bf16 ? 
                     data_type::bf16 : data_type::f32;
     src_desc = src.get_desc().to_type(dst_data_type);
     weights_desc = weights.get_desc().to_type(dst_data_type);
     if (with_bias) {
       IDEEP_ENFORCE(bias.get_data_type() == data_type::f32 ||
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
     op_attr.set_output_scales(utils::op_scale_mask(scale_size), 
		               std::vector<float>(1, dst_coeff));
   }

   dst_data_type = dst_type == data_type::undef ? dst_data_type : dst_type;   
   tensor::desc dst_desc(dst_dims, dst_data_type, tag::any);
   auto pd = with_bias
       ? primitive_desc({src_desc, weights_desc, bias_desc, dst_desc},
                         op_attr, aengine)
       : primitive_desc({src_desc, weights_desc, dst_desc},
                         op_attr, aengine);
   auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
   auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
   dst.reinit_if_possible(pd.dst_desc());
   if (!dst_scales.empty() && dst_data_type != data_type::f32) {
     dst.set_scale(dst_scales_in);
   }
   if (with_bias){
     auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
     super(pd).execute(stream::default_stream(),
                       {{DNNL_ARG_SRC, expected_src},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_BIAS, expected_bias},
                        {DNNL_ARG_DST, dst},
                        {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_point_m}});
   } else {
     super(pd).execute(stream::default_stream(),
                       {{DNNL_ARG_SRC, expected_src},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_DST, dst},
                        {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_point_m}});
   }
  }
};

}  // namespace ideep

#endif
