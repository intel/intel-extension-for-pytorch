#ifndef IDEEP_OPERATORS_DROPOUT_HPP
#define IDEEP_OPERATORS_DROPOUT_HPP

namespace ideep {

struct dropout_forward {
  static void compute(
      const tensor& src,
      float ratio,
      tensor& dst,
      tensor& mask) {
    switch (src.get_data_type()) {
      case data_type::f32:
        compute_impl<float>(src, ratio, dst, mask);
        break;
      case data_type::s32:
        compute_impl<int32_t>(src, ratio, dst, mask);
        break;
      case data_type::s8:
        compute_impl<int8_t>(src, ratio, dst, mask);
        break;
      case data_type::u8:
        compute_impl<uint8_t>(src, ratio, dst, mask);
        break;
      default:
        throw error(dnnl_invalid_arguments, "Unsupported dnnl data type");
    }
  }

 private:
  template <typename T>
  static void compute_impl(
      const tensor& src,
      float ratio,
      tensor& dst,
      tensor& mask) {
    mask.reinit_if_possible(src.get_desc());
    dst.reinit_if_possible(src.get_desc());
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }

    const auto scale = 1.0 / (1.0 - ratio);
    const auto size = src.get_size() / sizeof(T);
    std::unique_ptr<int[]> bernouli_nums(new int[size]);
    utils::bernoulli_generate(size, 1.0 - ratio, bernouli_nums.get());

    const auto src_data = static_cast<T*>(src.get_data_handle());
    const auto mask_data = static_cast<T*>(mask.get_data_handle());
    const auto dst_data = static_cast<T*>(dst.get_data_handle());
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd
#else
#pragma omp parallel for schedule(static)
#endif
#endif
    for (auto i = 0; i < size; i++) {
      mask_data[i] = bernouli_nums[i] * scale;
      dst_data[i] = mask_data[i] * src_data[i];
    }
  }
};

struct dropout_backward {
  static void compute(
      const tensor& mask,
      const tensor& diff_dst,
      tensor& diff_src) {
    switch (diff_dst.get_data_type()) {
      case data_type::f32:
        compute_impl<float>(mask, diff_dst, diff_src);
        break;
      case data_type::s32:
        compute_impl<int32_t>(mask, diff_dst, diff_src);
        break;
      case data_type::s8:
        compute_impl<int8_t>(mask, diff_dst, diff_src);
        break;
      case data_type::u8:
        compute_impl<uint8_t>(mask, diff_dst, diff_src);
        break;
      default:
        throw error(dnnl_invalid_arguments, "Unsupported dnnl data type!");
    }
  }

 private:
  template <typename T>
  static void compute_impl(
      const tensor& mask,
      const tensor& diff_dst,
      tensor& diff_src) {
    diff_src.reinit_if_possible(diff_dst.get_desc());

    const auto size = mask.get_size() / sizeof(T);
    const auto mask_data = static_cast<T*>(mask.get_data_handle());
    const auto diff_dst_data = static_cast<T*>(diff_dst.get_data_handle());
    const auto diff_src_data = static_cast<T*>(diff_src.get_data_handle());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (auto i = 0; i < size; i++) {
      diff_src_data[i] = mask_data[i] * diff_dst_data[i];
    }
  }
};

} // namespace ideep

#endif