#include <csrc/aten/cpu/optimizer/optimizer.h>
#include "csrc/cpu/vec/vec.h"

#include <torch/csrc/autograd/function.h>
#include <torch/extension.h>
namespace torch_ipex {
namespace cpu {

namespace {

using namespace at::vec;

template <typename scalar_t>
static inline scalar_t acc_vec(const at::vec::Vectorized<scalar_t>& v) {
  const int64_t K = at::vec::Vectorized<scalar_t>::size();
  std::array<scalar_t, K> arr;
  v.store(arr.data());
  return std::accumulate(arr.cbegin(), arr.cend(), scalar_t(0));
}

template <typename scalar_t, typename grad_t>
void lamb_fused_step_kernel(
    const at::Tensor& param,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& grad,
    const at::Tensor& param2,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* exp_avg_data = exp_avg.data_ptr<scalar_t>();
  scalar_t* exp_avg_sq_data = exp_avg_sq.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);

  int num_threads = at::get_num_threads();
  scalar_t param_norm_acc[num_threads];
  scalar_t rtw_norm_acc[num_threads];
  std::fill_n(&param_norm_acc[0], num_threads, scalar_t(0));
  std::fill_n(&rtw_norm_acc[0], num_threads, scalar_t(0));

  using Vec = at::vec::Vectorized<scalar_t>;

  int64_t grain_size = 512;

  // update momentum vt and mt
  // also accumulate sum of param_norm and rtw_norm
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();

        // local pointers
        scalar_t* param_ptr = param_data + begin;
        scalar_t* exp_avg_ptr = exp_avg_data + begin;
        scalar_t* exp_avg_sq_ptr = exp_avg_sq_data + begin;
        scalar_t* grad_ptr = grad_data + begin;

        const int64_t size = end - begin;

        // local sum for param_norm and rtw_norm
        Vec sum1_vec = Vec(scalar_t(0));
        Vec sum2_vec = Vec(scalar_t(0));
        scalar_t sum1_val = scalar_t(0);
        scalar_t sum2_val = scalar_t(0);

        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec grad_vec = Vec::loadu(grad_ptr + d);
          Vec exp_avg_vec = Vec::loadu(exp_avg_ptr + d) * Vec(scalar_t(beta1)) +
              grad_vec * Vec(scalar_t(1 - beta1));
          Vec exp_avg_sq_vec =
              Vec::loadu(exp_avg_sq_ptr + d) * Vec(scalar_t(beta2)) +
              grad_vec * grad_vec * Vec(scalar_t(1 - beta2));
          Vec adam_step_vec = exp_avg_vec / Vec(scalar_t(bias_correction1)) /
              ((exp_avg_sq_vec / Vec(scalar_t(bias_correction2))).sqrt() +
               Vec(scalar_t(eps)));

          exp_avg_vec.store(exp_avg_ptr + d);
          exp_avg_sq_vec.store(exp_avg_sq_ptr + d);

          Vec param_vec = Vec::loadu(param_ptr + d);
          adam_step_vec =
              adam_step_vec + param_vec * Vec(scalar_t(weight_decay));
          // reuse grad to store adam_step
          adam_step_vec.store(grad_ptr + d);

          sum1_vec = sum1_vec + param_vec * param_vec;
          sum2_vec = sum2_vec + adam_step_vec * adam_step_vec;
        }
        for (; d < size; d++) {
          exp_avg_ptr[d] = exp_avg_ptr[d] * beta1 + grad_ptr[d] * (1 - beta1);
          exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * beta2 +
              grad_ptr[d] * grad_ptr[d] * (1 - beta2);
          scalar_t adam_step_val = (exp_avg_ptr[d] / bias_correction1) /
              (std::sqrt(exp_avg_sq_ptr[d] / bias_correction2) + eps);

          adam_step_val += param_ptr[d] * weight_decay;
          // reuse grad to store adam_step
          grad_ptr[d] = adam_step_val;

          sum1_val += param_ptr[d] * param_ptr[d];
          sum2_val += adam_step_val * adam_step_val;
        }
        sum1_val += acc_vec(sum1_vec);
        sum2_val += acc_vec(sum2_vec);

        param_norm_acc[tid] = sum1_val;
        rtw_norm_acc[tid] = sum2_val;
      });

  // synchronize before update true_ratio
  //
  // [Note]: we could use #pragma omp barrier so that finish within a single omp
  // session
  //   but at::parallel_for partition rule will not guarantee ALL threads in the
  //   same team will be used, so the unused thread will keep on waiting since
  //   it never reaches the barrier.
  //
  scalar_t param_norm_sum = scalar_t(0);
  scalar_t rtw_norm_sum = scalar_t(0);
  for (int64_t tid = 0; tid < num_threads; tid++) {
    param_norm_sum += param_norm_acc[tid];
    rtw_norm_sum += rtw_norm_acc[tid];
  }
  scalar_t true_ratio = std::sqrt(param_norm_sum) / std::sqrt(rtw_norm_sum);

  // update param
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        scalar_t* param_ptr = param_data + begin;
        scalar_t* grad_ptr = grad_data + begin;

        const int64_t size = end - begin;

        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec param_vec = Vec::loadu(param_ptr + d) -
              Vec::loadu(grad_ptr + d) *
                  Vec(scalar_t(learning_rate * true_ratio));
          param_vec.store(param_ptr + d);
        }
        for (; d < size; d++) {
          param_ptr[d] -= grad_ptr[d] * learning_rate * true_ratio;
        }
      });
}

template <>
void lamb_fused_step_kernel<at::BFloat16, at::BFloat16>(
    const at::Tensor& param,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& grad,
    const at::Tensor& param2,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  TORCH_CHECK(
      param.scalar_type() == at::kBFloat16,
      "lamb_fused_step_kernel: expect param to be at::BFloat16");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "lamb_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      exp_avg.scalar_type() == at::kFloat,
      "lamb_fused_step_kernel: expect exp_avg to be float32");
  TORCH_CHECK(
      exp_avg_sq.scalar_type() == at::kFloat,
      "lamb_fused_step_kernel: expect exp_avg_sq to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "lamb_fused_step_kernel: expect param2 to be at::BFloat16");

  at::BFloat16* param_data = param.data_ptr<at::BFloat16>();
  float* exp_avg_data = exp_avg.data_ptr<float>();
  float* exp_avg_sq_data = exp_avg_sq.data_ptr<float>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);

  int num_threads = at::get_num_threads();
  float param_norm_acc[num_threads];
  float rtw_norm_acc[num_threads];
  std::fill_n(&param_norm_acc[0], num_threads, float(0));
  std::fill_n(&rtw_norm_acc[0], num_threads, float(0));

  // for float32 path, we can reuse grad to store adam_step
  // but for bfloat16 path, this can't be done since grad is in bfloat16
  // and we want to keep adam_step to be float32
  int64_t numel = param.numel();
  at::Tensor workspace = at::empty({numel}, exp_avg.options());
  float* workspace_data = workspace.data_ptr<float>();

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;

  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();

    // local pointers
    at::BFloat16* param_ptr = param_data + begin;
    float* exp_avg_ptr = exp_avg_data + begin;
    float* exp_avg_sq_ptr = exp_avg_sq_data + begin;
    at::BFloat16* grad_ptr = grad_data + begin;
    at::BFloat16* param2_ptr = param2_data + begin;
    float* workspace_ptr = workspace_data + begin;

    const int64_t size = end - begin;

    // local sum for param_norm and rtw_norm
    fVec sum1_fvec = fVec(float(0));
    fVec sum2_fvec = fVec(float(0));
    float sum1_val = float(0);
    float sum2_val = float(0);

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec grad_bvec = bVec::loadu(grad_ptr + d);
      fVec grad_fvec, grad_fvec2;
      std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

      fVec exp_avg_fvec = fVec::loadu(exp_avg_ptr + d) * fVec(float(beta1)) +
          grad_fvec * fVec(float(1 - beta1));
      fVec exp_avg_sq_fvec =
          fVec::loadu(exp_avg_sq_ptr + d) * fVec(float(beta2)) +
          grad_fvec * grad_fvec * fVec(float(1 - beta2));
      fVec adam_step_fvec = exp_avg_fvec / fVec(float(bias_correction1)) /
          ((exp_avg_sq_fvec / fVec(float(bias_correction2))).sqrt() +
           fVec(float(eps)));

      fVec exp_avg_fvec2 =
          fVec::loadu(exp_avg_ptr + d + fVec::size()) * fVec(float(beta1)) +
          grad_fvec2 * fVec(float(1 - beta1));
      fVec exp_avg_sq_fvec2 =
          fVec::loadu(exp_avg_sq_ptr + d + fVec::size()) * fVec(float(beta2)) +
          grad_fvec2 * grad_fvec2 * fVec(float(1 - beta2));
      fVec adam_step_fvec2 = exp_avg_fvec2 / fVec(float(bias_correction1)) /
          ((exp_avg_sq_fvec2 / fVec(float(bias_correction2))).sqrt() +
           fVec(float(eps)));

      exp_avg_fvec.store(exp_avg_ptr + d);
      exp_avg_fvec2.store(exp_avg_ptr + d + fVec::size());
      exp_avg_sq_fvec.store(exp_avg_sq_ptr + d);
      exp_avg_sq_fvec2.store(exp_avg_sq_ptr + d + fVec::size());

      bVec param_bvec = bVec::loadu(param_ptr + d);
      bVec param2_bvec = bVec::loadu(param2_ptr + d);
      fVec param_fvec, param_fvec2;
      std::tie(param_fvec, param_fvec2) =
          at::vec::pack_bfloat16_float(param_bvec, param2_bvec);

      adam_step_fvec = adam_step_fvec + param_fvec * fVec(float(weight_decay));
      adam_step_fvec2 =
          adam_step_fvec2 + param_fvec2 * fVec(float(weight_decay));
      adam_step_fvec.store(workspace_ptr + d);
      adam_step_fvec2.store(workspace_ptr + d + fVec::size());

      sum1_fvec += param_fvec * param_fvec;
      sum1_fvec += param_fvec2 * param_fvec2;
      sum2_fvec += adam_step_fvec * adam_step_fvec;
      sum2_fvec += adam_step_fvec2 * adam_step_fvec2;
    }
    for (; d < size; d++) {
      float grad_val = float(grad_ptr[d]);
      exp_avg_ptr[d] = exp_avg_ptr[d] * beta1 + grad_val * (1 - beta1);
      exp_avg_sq_ptr[d] =
          exp_avg_sq_ptr[d] * beta2 + grad_val * grad_val * (1 - beta2);
      float adam_step_val = (exp_avg_ptr[d] / bias_correction1) /
          (std::sqrt(exp_avg_sq_ptr[d] / bias_correction2) + eps);

      float param_val =
          at::vec::pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
      adam_step_val += param_val * weight_decay;
      workspace_ptr[d] = adam_step_val;

      sum1_val += param_val * param_val;
      sum2_val += adam_step_val * adam_step_val;
    }
    sum1_val += acc_vec(sum1_fvec);
    sum2_val += acc_vec(sum2_fvec);

    param_norm_acc[tid] = sum1_val;
    rtw_norm_acc[tid] = sum2_val;
  });

  float param_norm_sum = float(0);
  float rtw_norm_sum = float(0);
  for (int64_t tid = 0; tid < num_threads; tid++) {
    param_norm_sum += param_norm_acc[tid];
    rtw_norm_sum += rtw_norm_acc[tid];
  }
  float true_ratio = std::sqrt(param_norm_sum) / std::sqrt(rtw_norm_sum);

  // update param
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    at::BFloat16* param_ptr = param_data + begin;
    at::BFloat16* param2_ptr = param2_data + begin;
    float* workspace_ptr = workspace_data + begin;

    const int64_t size = end - begin;

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec param_bvec = bVec::loadu(param_ptr + d);
      bVec param2_bvec = bVec::loadu(param2_ptr + d);
      fVec param_fvec, param_fvec2;
      std::tie(param_fvec, param_fvec2) =
          at::vec::pack_bfloat16_float(param_bvec, param2_bvec);

      param_fvec -= fVec::loadu(workspace_ptr + d) *
          fVec(float(learning_rate * true_ratio));
      param_fvec2 -= fVec::loadu(workspace_ptr + d + fVec::size()) *
          fVec(float(learning_rate * true_ratio));

      std::tie(param_bvec, param2_bvec) =
          at::vec::unpack_float_bfloat16(param_fvec, param_fvec2);
      param_bvec.store(param_ptr + d);
      param2_bvec.store(param2_ptr + d);
    }
    for (; d < size; d++) {
      float param_val =
          at::vec::pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
      param_val -= workspace_ptr[d] * learning_rate * true_ratio;
      std::tie(param_ptr[d], param2_ptr[d]) =
          at::vec::unpack_float_bfloat16(param_val);
    }
  });
}

template <>
void lamb_fused_step_kernel<float, at::BFloat16>(
    const at::Tensor& param,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& grad,
    const at::Tensor& param2,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  TORCH_CHECK(
      param.scalar_type() == at::kFloat,
      "lamb_fused_step_kernel: expect param to be at::Float");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "lamb_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      exp_avg.scalar_type() == at::kFloat,
      "lamb_fused_step_kernel: expect exp_avg to be float32");
  TORCH_CHECK(
      exp_avg_sq.scalar_type() == at::kFloat,
      "lamb_fused_step_kernel: expect exp_avg_sq to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "lamb_fused_step_kernel: expect param2 to be at::BFloat16");

  float* param_data = param.data_ptr<float>();
  float* exp_avg_data = exp_avg.data_ptr<float>();
  float* exp_avg_sq_data = exp_avg_sq.data_ptr<float>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);

  int num_threads = at::get_num_threads();
  float param_norm_acc[num_threads];
  float rtw_norm_acc[num_threads];
  std::fill_n(&param_norm_acc[0], num_threads, float(0));
  std::fill_n(&rtw_norm_acc[0], num_threads, float(0));

  // for float32 path, we can reuse grad to store adam_step
  // but for bfloat16 path, this can't be done since grad is in bfloat16
  // and we want to keep adam_step to be float32
  int64_t numel = param.numel();
  at::Tensor workspace = at::empty({numel}, exp_avg.options());
  float* workspace_data = workspace.data_ptr<float>();

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;

  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();

    // local pointers
    float* param_ptr = param_data + begin;
    float* exp_avg_ptr = exp_avg_data + begin;
    float* exp_avg_sq_ptr = exp_avg_sq_data + begin;
    at::BFloat16* grad_ptr = grad_data + begin;
    float* workspace_ptr = workspace_data + begin;

    const int64_t size = end - begin;

    // local sum for param_norm and rtw_norm
    fVec sum1_fvec = fVec(float(0));
    fVec sum2_fvec = fVec(float(0));
    float sum1_val = float(0);
    float sum2_val = float(0);

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec grad_bvec = bVec::loadu(grad_ptr + d);
      fVec grad_fvec, grad_fvec2;
      std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

      fVec exp_avg_fvec = fVec::loadu(exp_avg_ptr + d) * fVec(float(beta1)) +
          grad_fvec * fVec(float(1 - beta1));
      fVec exp_avg_sq_fvec =
          fVec::loadu(exp_avg_sq_ptr + d) * fVec(float(beta2)) +
          grad_fvec * grad_fvec * fVec(float(1 - beta2));
      fVec adam_step_fvec = exp_avg_fvec / fVec(float(bias_correction1)) /
          ((exp_avg_sq_fvec / fVec(float(bias_correction2))).sqrt() +
           fVec(float(eps)));

      fVec exp_avg_fvec2 =
          fVec::loadu(exp_avg_ptr + d + fVec::size()) * fVec(float(beta1)) +
          grad_fvec2 * fVec(float(1 - beta1));
      fVec exp_avg_sq_fvec2 =
          fVec::loadu(exp_avg_sq_ptr + d + fVec::size()) * fVec(float(beta2)) +
          grad_fvec2 * grad_fvec2 * fVec(float(1 - beta2));
      fVec adam_step_fvec2 = exp_avg_fvec2 / fVec(float(bias_correction1)) /
          ((exp_avg_sq_fvec2 / fVec(float(bias_correction2))).sqrt() +
           fVec(float(eps)));

      exp_avg_fvec.store(exp_avg_ptr + d);
      exp_avg_fvec2.store(exp_avg_ptr + d + fVec::size());
      exp_avg_sq_fvec.store(exp_avg_sq_ptr + d);
      exp_avg_sq_fvec2.store(exp_avg_sq_ptr + d + fVec::size());

      fVec param_fvec = fVec::loadu(param_ptr + d);
      fVec param_fvec2 = fVec::loadu(param_ptr + d + fVec::size());

      adam_step_fvec = adam_step_fvec + param_fvec * fVec(float(weight_decay));
      adam_step_fvec2 =
          adam_step_fvec2 + param_fvec2 * fVec(float(weight_decay));
      adam_step_fvec.store(workspace_ptr + d);
      adam_step_fvec2.store(workspace_ptr + d + fVec::size());

      sum1_fvec += param_fvec * param_fvec;
      sum1_fvec += param_fvec2 * param_fvec2;
      sum2_fvec += adam_step_fvec * adam_step_fvec;
      sum2_fvec += adam_step_fvec2 * adam_step_fvec2;
    }
    for (; d < size; d++) {
      float grad_val = float(grad_ptr[d]);
      exp_avg_ptr[d] = exp_avg_ptr[d] * beta1 + grad_val * (1 - beta1);
      exp_avg_sq_ptr[d] =
          exp_avg_sq_ptr[d] * beta2 + grad_val * grad_val * (1 - beta2);
      float adam_step_val = (exp_avg_ptr[d] / bias_correction1) /
          (std::sqrt(exp_avg_sq_ptr[d] / bias_correction2) + eps);

      float param_val = param_ptr[d];
      adam_step_val += param_val * weight_decay;
      workspace_ptr[d] = adam_step_val;

      sum1_val += param_val * param_val;
      sum2_val += adam_step_val * adam_step_val;
    }
    sum1_val += acc_vec(sum1_fvec);
    sum2_val += acc_vec(sum2_fvec);

    param_norm_acc[tid] = sum1_val;
    rtw_norm_acc[tid] = sum2_val;
  });

  float param_norm_sum = float(0);
  float rtw_norm_sum = float(0);
  for (int64_t tid = 0; tid < num_threads; tid++) {
    param_norm_sum += param_norm_acc[tid];
    rtw_norm_sum += rtw_norm_acc[tid];
  }
  float true_ratio = std::sqrt(param_norm_sum) / std::sqrt(rtw_norm_sum);

  // update param
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    float* param_ptr = param_data + begin;
    at::BFloat16* param2_ptr = param2_data + begin;
    float* workspace_ptr = workspace_data + begin;

    const int64_t size = end - begin;

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      fVec param_fvec = fVec::loadu(param_ptr + d);
      fVec param_fvec2 = fVec::loadu(param_ptr + d + fVec::size());

      param_fvec -= fVec::loadu(workspace_ptr + d) *
          fVec(float(learning_rate * true_ratio));
      param_fvec2 -= fVec::loadu(workspace_ptr + d + fVec::size()) *
          fVec(float(learning_rate * true_ratio));

      param_fvec.store(param_ptr + d);
      param_fvec2.store(param_ptr + d + fVec::size());
      // sync float param to bfloat16
      bVec param2_bvec = convert_float_bfloat16(param_fvec, param_fvec2);
      param2_bvec.store(param2_ptr + d);
    }
    for (; d < size; d++) {
      float param_val = param_ptr[d];
      param_val -= workspace_ptr[d] * learning_rate * true_ratio;
      param_ptr[d] = param_val;
      param2_ptr[d] = at::BFloat16(param_val);
    }
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lamb_fused_step_kernel_impl(
    const at::Tensor& param_,
    const at::Tensor& exp_avg_,
    const at::Tensor& exp_avg_sq_,
    const at::Tensor& grad_,
    const at::Tensor& param2_,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  auto param = param_.contiguous();
  auto exp_avg = exp_avg_.contiguous();
  auto exp_avg_sq = exp_avg_sq_.contiguous();
  auto grad = grad_.contiguous();
  auto param2 = param2_.contiguous();

  auto grad_dtype = grad_.scalar_type();
  auto param_dtype = param_.scalar_type();
  if (at::ScalarType::Float == grad_dtype) {
    lamb_fused_step_kernel<float, float>(
        param,
        exp_avg,
        exp_avg_sq,
        grad,
        param2,
        step,
        beta1,
        beta2,
        learning_rate,
        weight_decay,
        eps);
  } else if (at::ScalarType::Double == grad_dtype) {
    lamb_fused_step_kernel<double, double>(
        param,
        exp_avg,
        exp_avg_sq,
        grad,
        param2,
        step,
        beta1,
        beta2,
        learning_rate,
        weight_decay,
        eps);
  } else if (
      at::ScalarType::BFloat16 == grad_dtype &&
      at::ScalarType::BFloat16 == param_dtype) {
    lamb_fused_step_kernel<at::BFloat16, at::BFloat16>(
        param,
        exp_avg,
        exp_avg_sq,
        grad,
        param2,
        step,
        beta1,
        beta2,
        learning_rate,
        weight_decay,
        eps);
  } else if (
      at::ScalarType::BFloat16 == grad_dtype &&
      at::ScalarType::Float == param_dtype) {
    lamb_fused_step_kernel<float, at::BFloat16>(
        param,
        exp_avg,
        exp_avg_sq,
        grad,
        param2,
        step,
        beta1,
        beta2,
        learning_rate,
        weight_decay,
        eps);
  } else {
    TORCH_CHECK(false, "expect bfloat16 or float or double param");
  }

  if (!param_.is_contiguous()) {
    param_.copy_(param);
  }
  if (!exp_avg_.is_contiguous()) {
    exp_avg_.copy_(exp_avg);
  }
  if (!exp_avg_sq_.is_contiguous()) {
    exp_avg_sq_.copy_(exp_avg_sq);
  }
  if (!param2_.is_contiguous()) {
    param2_.copy_(param2);
  }

  return std::make_tuple(param_, exp_avg_, exp_avg_sq_);
}

} // anonymous namespace

REGISTER_DISPATCH(lamb_fused_step_kernel_stub, &lamb_fused_step_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
