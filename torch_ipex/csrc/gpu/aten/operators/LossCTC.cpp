#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/Context.h>
#include <core/DPCPP.h>
#include <core/DPCPPTensorUtils.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <utils/ATDispatch.h>
#include <utils/Atomics.h>
#include <utils/Numerics.h>

using namespace at::dpcpp;

template <typename...>
class ctc_loss_alpha_kernel {};
template <typename...>
class ctc_loss_beta_kernel {};
template <typename...>
class ctc_loss_collect_nonblank_kernel {};
template <typename...>
class ctc_loss_collect_kernel {};
template <typename...>
class ctc_loss_zero_padded_kernel {};

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1])
// so if l is l_0 l_1 ... l_(tl-1) then this looks up idx in
// l' = BLANK l_0 BLANK l_1 BLANK ... BLANK l_(tl-1) BLANK
// - note that no bound-checking is done
// - it is important to only call it witth idx == 0 if the target length is 0
template <typename target_t>
static inline int64_t get_target_prime(
    const target_t* target,
    int64_t offset,
    int64_t stride,
    int64_t idx,
    int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

std::tuple<DPCPP::range<2>, DPCPP::range<2>> get_work_range(
    DPCPP::queue& sycl_queue,
    int64_t work_in_batch_size,
    int64_t batch_size) {
  size_t work_group_size = dpcppMaxWorkGroupSize(sycl_queue);
  int intra_batch_size = work_group_size;
  while (intra_batch_size / 2 >= work_in_batch_size && intra_batch_size > 1) {
    intra_batch_size /= 2;
  }

  int inter_batch_size =
      std::min((int)(work_group_size / intra_batch_size), (int)batch_size);

  DPCPP::range<2> global_range(
      std::max<int>(
          (work_in_batch_size + intra_batch_size - 1) / intra_batch_size, 1) *
          intra_batch_size, // round up to intra_batch_size
      ((batch_size + inter_batch_size - 1) / inter_batch_size) *
          inter_batch_size // round up to inter_batch_size
      );
  DPCPP::range<2> local_range(intra_batch_size, inter_batch_size);

  return std::make_tuple(global_range, local_range);
}

// this kernel is a relatively straightforward implementation of the alpha
// calculation in the forward backward algorithm (section 4.1).
// A (minor) twist is that we are using log-calculations to enhance numerical
// stability (log_probs and log_alpha).
// In total it would be more efficient to compute the beta in the same kernel
// (e.g. cudnn does this). While the beta are not
// needed for the loss itself (just the grad), we can return log_alpha+log_beta
// (so same space as currently) and the overhead
// is small and the use-case for loss without grad is relatively limited.
// We parallelize by batch and target sequence. Empirically, it is faster to
// loop over the input (log probs) sequence  and do
// target in parallel, even if it means more frequent __syncthreads.
// In contrast to the cuDNN implementation, we allow large target lengths. For
// this we need that all previous `s` have been
// computed when we start a new block_s. This is why we have our own for loop
// here.
template <typename scalar_t, typename target_t>
void ctc_loss_log_alpha_kernel(
    Tensor& log_alpha,
    const Tensor& log_probs,
    const Tensor& input_lengths,
    int64_t __max_input_length,
    const Tensor& targets,
    const Tensor& target_lengths,
    int64_t __max_target_length,
    Tensor& neg_log_likelihood,
    const Tensor& tg_batch_offsets,
    int64_t __tg_target_stride,
    int64_t __batch_size,
    int64_t __BLANK) {
  constexpr scalar_t neginf = -INFINITY;

  int64_t __lp_input_stride = log_probs.stride(0);
  int64_t __lp_batch_stride = log_probs.stride(1);
  int64_t __lp_char_stride = log_probs.stride(2);
  int64_t __la_batch_stride = log_alpha.stride(0);
  int64_t __la_input_stride = log_alpha.stride(1);
  int64_t __la_target_stride = log_alpha.stride(2);

  auto& sycl_queue = getCurrentDPCPPStream().dpcpp_queue();
  DPCPP::range<2> global_range, local_range;
  std::tie(global_range, local_range) =
      get_work_range(sycl_queue, 2 * __max_input_length + 1, __batch_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto log_alpha_acc =
        DPCPPAccessor<dpcpp_rw_mode>(cgh, log_alpha.data_ptr());
    auto log_probs_acc =
        DPCPPAccessor<dpcpp_discard_w_mode>(cgh, log_probs.data_ptr());
    auto input_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, input_lengths.data_ptr());
    auto targets_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, targets.data_ptr());
    auto target_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, target_lengths.data_ptr());
    auto neg_log_likelihood_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, neg_log_likelihood.data_ptr());
    auto tg_batch_offsets_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, tg_batch_offsets.data_ptr());
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      size_t intra_batch_id = item_id.get_local_id(0);
      size_t intra_batch_size = item_id.get_local_range(0);
      scalar_t* log_alpha_data = log_alpha_acc.template get_pointer<scalar_t>();
      scalar_t* log_probs_data = log_probs_acc.template get_pointer<scalar_t>();
      target_t* targets_data = targets_acc.template get_pointer<target_t>();
      scalar_t* neg_log_likelihood_data =
          neg_log_likelihood_acc.template get_pointer<scalar_t>();
      int64_t* input_lengths_data =
          input_lengths_acc.template get_pointer<int64_t>();
      int64_t* target_lengths_data =
          target_lengths_acc.template get_pointer<int64_t>();
      int64_t* tg_batch_offsets_data =
          tg_batch_offsets_acc.template get_pointer<int64_t>();

      // bookkeeping
      int64_t b = item_id.get_local_id(1) +
          item_id.get_group(1) * item_id.get_local_range(1);
      // This is a workaround for unknown compute cpp issue.
      int64_t max_input_length = __max_input_length;
      int64_t max_target_length = __max_target_length;
      int64_t lp_input_stride = __lp_input_stride;
      int64_t lp_batch_stride = __lp_batch_stride;
      int64_t lp_char_stride = __lp_char_stride;
      int64_t la_batch_stride = __la_batch_stride;
      int64_t la_input_stride = __la_input_stride;
      int64_t la_target_stride = __la_target_stride;
      int64_t batch_size = __batch_size;
      int64_t BLANK = __BLANK;
      int64_t tg_target_stride = __tg_target_stride;

      int64_t input_length = input_lengths_data[b];
      int64_t target_length = target_lengths_data[b];
      int64_t lp_batch_offset = b * lp_batch_stride;
      int64_t la_batch_offset = b * la_batch_stride;
      int64_t tg_batch_offset = tg_batch_offsets_data[b];

      if (b < batch_size) {
        // first row (t=0), the three equations for alpha_1 above eq (6)
        for (int64_t s = intra_batch_id; s < 2 * max_target_length + 1;
             s += intra_batch_size) {
          scalar_t la;
          switch (s) {
            case 0:
              la = log_probs_data[lp_batch_offset + lp_char_stride * BLANK];
              break;
            case 1:
              la = target_length == 0
                  ? neginf
                  : log_probs_data[lp_batch_offset +
                                   lp_char_stride * get_target_prime(
                                                        targets_data,
                                                        tg_batch_offset,
                                                        tg_target_stride,
                                                        1,
                                                        BLANK)];
              break;
            default:
              la = neginf;
          }
          log_alpha_data[la_batch_offset +
                         /* la_input_stride * 0 */ +la_target_stride * s] = la;
        }
      }

      for (int64_t block_s = 0; block_s < 2 * max_target_length + 1;
           block_s += intra_batch_size) {
        int64_t s = intra_batch_id + block_s;
        // These two only depend on s, so we can cache them.
        int64_t current_char; // l_s in eq (6)
        bool have_three; // flag which of the two cases in eq (6) we have
        if (b < batch_size && s < 2 * target_length + 1 && target_length > 0) {
          current_char = get_target_prime(
              targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
          have_three =
              ((s > 1) && (get_target_prime(
                               targets_data,
                               tg_batch_offset,
                               tg_target_stride,
                               s - 2,
                               BLANK) != current_char));
        } else {
          current_char = BLANK;
          have_three = false;
        }

        for (int64_t t = 1; t < max_input_length; t++) {
          item_id.barrier(DPCPP::access::fence_space::global_space);
          if ((b < batch_size) && (t < input_length) &&
              (s < 2 * target_length + 1)) {
            // only for valid t, s. This is equation (6) and (7), la1, la2, la3
            // are the three summands,
            // lamax is the maximum for the logsumexp trick.
            scalar_t la1 =
                log_alpha_data[la_batch_offset + la_input_stride * (t - 1) +
                               la_target_stride * s];
            scalar_t lamax = la1;
            scalar_t la2, la3;
            if (s > 0) {
              la2 = log_alpha_data[la_batch_offset + la_input_stride * (t - 1) +
                                   la_target_stride * (s - 1)];
              if (la2 > lamax)
                lamax = la2;
            } else {
              la2 = neginf;
            }
            if (have_three) {
              la3 = log_alpha_data[la_batch_offset + la_input_stride * (t - 1) +
                                   la_target_stride * (s - 2)];
              if (la3 > lamax)
                lamax = la3;
            } else {
              la3 = neginf;
            }
            if (lamax == neginf) // when all are neginf. (then the whole thing
              // is neginf, but we can pretend)
              lamax = 0;

            scalar_t tmp = Numerics<scalar_t>::log(
                Numerics<scalar_t>::exp(la1 - lamax) +
                Numerics<scalar_t>::exp(la2 - lamax) +
                Numerics<scalar_t>::exp(la3 - lamax));
            log_alpha_data[la_batch_offset + la_input_stride * t +
                           la_target_stride * s] = tmp + lamax +
                log_probs_data[lp_batch_offset + t * lp_input_stride +
                               lp_char_stride * current_char];
          } else {
            // otherwise we just set to neginf
            if (b < batch_size && s < 2 * max_target_length + 1)
              log_alpha_data[la_batch_offset + la_input_stride * t +
                             la_target_stride * s] = neginf;
          }
        }
      }
      item_id.barrier(DPCPP::access::fence_space::global_space);

      if (b >= batch_size)
        return;

      // compute the loss (eq (8))
      if (intra_batch_id == 0) {
        // the likelihood is the the sum of the last two alphas, eq (8), the
        // loss is the negative log likelihood
        if (target_length == 0) {
          // if the target is empty then there is no preceding BLANK state and
          // hence there is no path to merge
          neg_log_likelihood_data[b] =
              -log_alpha_data[la_batch_offset +
                              la_input_stride * (input_length -
                                                 1) /*+ la_target_stride * 0*/];
        } else {
          scalar_t l1 = log_alpha_data[la_batch_offset +
                                       la_input_stride * (input_length - 1) +
                                       la_target_stride * (target_length * 2)];
          scalar_t l2 = target_length > 0
              ? log_alpha_data[la_batch_offset +
                               la_input_stride * (input_length - 1) +
                               la_target_stride * (target_length * 2 - 1)]
              : neginf;
          scalar_t m = ((l1 > l2) ? l1 : l2);
          m = ((m == neginf) ? 0 : m);
          scalar_t log_likelihood = Numerics<scalar_t>::log(
                                        Numerics<scalar_t>::exp(l1 - m) +
                                        Numerics<scalar_t>::exp(l2 - m)) +
              m;
          neg_log_likelihood_data[b] = -log_likelihood;
        }
      }
    };

    cgh.parallel_for<ctc_loss_alpha_kernel<scalar_t, target_t>>(
        DPCPP::nd_range<2>(global_range, local_range), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

// The second (backward) half of the forward backward algorithm, (10) and (11).
// This is parallel to the
// alpha kernel above. (As mentioned above, it might make sense do the
// calculation in the alpha kernel.)
template <typename scalar_t, typename target_t>
void ctc_loss_backward_log_beta_kernel(
    Tensor& log_beta,
    const Tensor& log_probs,
    const Tensor& input_lengths,
    int64_t __max_input_length,
    const Tensor& targets,
    const Tensor& target_lengths,
    int64_t __max_target_length,
    const Tensor& tg_batch_offsets,
    int64_t __tg_target_stride,
    int64_t __batch_size,
    int64_t __BLANK) {
  constexpr scalar_t neginf = -INFINITY;

  int64_t __lp_input_stride = log_probs.stride(0);
  int64_t __lp_batch_stride = log_probs.stride(1);
  int64_t __lp_char_stride = log_probs.stride(2);
  int64_t __lb_batch_stride = log_beta.stride(0);
  int64_t __lb_input_stride = log_beta.stride(1);
  int64_t __lb_target_stride = log_beta.stride(2);

  auto& sycl_queue = getCurrentDPCPPStream().dpcpp_queue();
  DPCPP::range<2> global_range, local_range;
  std::tie(global_range, local_range) =
      get_work_range(sycl_queue, 2 * __max_input_length + 1, __batch_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto log_beta_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, log_beta.data_ptr());
    auto log_probs_acc =
        DPCPPAccessor<dpcpp_discard_w_mode>(cgh, log_probs.data_ptr());
    auto input_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, input_lengths.data_ptr());
    auto targets_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, targets.data_ptr());
    auto target_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, target_lengths.data_ptr());
    auto tg_batch_offsets_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, tg_batch_offsets.data_ptr());
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      size_t intra_batch_id = item_id.get_local_id(0);
      size_t intra_batch_size = item_id.get_local_range(0);
      scalar_t* log_beta_data = log_beta_acc.template get_pointer<scalar_t>();
      scalar_t* log_probs_data = log_probs_acc.template get_pointer<scalar_t>();
      target_t* targets_data = targets_acc.template get_pointer<target_t>();
      int64_t* input_lengths_data =
          input_lengths_acc.template get_pointer<int64_t>();
      int64_t* target_lengths_data =
          target_lengths_acc.template get_pointer<int64_t>();
      int64_t* tg_batch_offsets_data =
          tg_batch_offsets_acc.template get_pointer<int64_t>();

      // bookkeeping
      int64_t b = item_id.get_local_id(1) +
          item_id.get_group(1) * item_id.get_local_range(1);
      // This is a workaround for unknown compute cpp issue.
      int64_t max_input_length = __max_input_length;
      int64_t max_target_length = __max_target_length;
      int64_t lp_input_stride = __lp_input_stride;
      int64_t lp_batch_stride = __lp_batch_stride;
      int64_t lp_char_stride = __lp_char_stride;
      int64_t lb_batch_stride = __lb_batch_stride;
      int64_t lb_input_stride = __lb_input_stride;
      int64_t lb_target_stride = __lb_target_stride;
      int64_t batch_size = __batch_size;
      int64_t BLANK = __BLANK;
      int64_t tg_target_stride = __tg_target_stride;

      int64_t input_length = input_lengths_data[b];
      int64_t target_length = target_lengths_data[b];
      int64_t lp_batch_offset = b * lp_batch_stride;
      int64_t lb_batch_offset = b * lb_batch_stride;
      int64_t tg_batch_offset = tg_batch_offsets_data[b];

      if (b < batch_size) {
        // "first" row, the beta initiaization before eq (10) (t=target_length -
        // differes per batch)
        int64_t last_s_offset =
            2 * max_target_length - (2 * max_target_length % intra_batch_size);
        for (int64_t s = last_s_offset + intra_batch_id; s >= 0;
             s -= intra_batch_size) {
          scalar_t lb;
          if (s == 2 * target_length) {
            lb = log_probs_data[lp_batch_offset +
                                (input_length - 1) * lp_input_stride +
                                lp_char_stride * BLANK];
          } else if (s == 2 * target_length - 1) { // false for target_length ==
            // 0
            int64_t current_target_prime = get_target_prime(
                targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
            lb = log_probs_data[lp_batch_offset +
                                (input_length - 1) * lp_input_stride +
                                lp_char_stride * current_target_prime];
          } else {
            lb = neginf;
          }
          if (s < 2 * max_target_length + 1) {
            log_beta_data[lb_batch_offset +
                          (input_length - 1) * lb_input_stride +
                          lb_target_stride * s] = lb;
          }
        }
      }

      // go backward in s
      for (int64_t block_s = 2 * max_target_length -
               (2 * max_target_length % intra_batch_size);
           block_s >= 0;
           block_s -= intra_batch_size) {
        int64_t s = intra_batch_id + block_s;
        int64_t current_target_prime;
        bool have_three;
        if (b < batch_size && s < 2 * target_length + 1 && target_length > 0) {
          current_target_prime = get_target_prime(
              targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
          have_three =
              ((s < 2 * target_length - 1) &&
               (get_target_prime(
                    targets_data,
                    tg_batch_offset,
                    tg_target_stride,
                    s + 2,
                    BLANK) != current_target_prime));
        } else {
          current_target_prime = BLANK;
          have_three = false;
        }
        // now go backward in t. Note that we need to skip the last timestep
        // that we did above.
        for (int64_t t = max_input_length - 2; t >= 0; t--) {
          item_id.barrier(DPCPP::access::fence_space::global_space);
          if ((b < batch_size) && (t < input_length - 1) &&
              (s < 2 * target_length + 1)) {
            scalar_t lb1 =
                log_beta_data[lb_batch_offset + lb_input_stride * (t + 1) +
                              lb_target_stride * s];
            scalar_t lbmax = lb1;
            scalar_t lb2, lb3;

            if (s < 2 * target_length) {
              lb2 = log_beta_data[lb_batch_offset + lb_input_stride * (t + 1) +
                                  lb_target_stride * (s + 1)];
              if (lb2 > lbmax)
                lbmax = lb2;
            } else {
              lb2 = neginf;
            }
            if (have_three) {
              lb3 = log_beta_data[lb_batch_offset + lb_input_stride * (t + 1) +
                                  lb_target_stride * (s + 2)];
              if (lb3 > lbmax)
                lbmax = lb3;
            } else {
              lb3 = neginf;
            }
            if (lbmax == neginf)
              lbmax = 0;

            scalar_t lb = Numerics<scalar_t>::log(
                              Numerics<scalar_t>::exp(lb1 - lbmax) +
                              Numerics<scalar_t>::exp(lb2 - lbmax) +
                              Numerics<scalar_t>::exp(lb3 - lbmax)) +
                lbmax + log_probs_data[lp_batch_offset + t * lp_input_stride +
                                       lp_char_stride * current_target_prime];

            log_beta_data[lb_batch_offset + lb_input_stride * t +
                          lb_target_stride * s] = lb;
          } else if (
              (b < batch_size) && (s < 2 * max_target_length + 1) &&
              (((target_length == 0) && (s > 0)) ||
               (s >= 2 * target_length + 1) || (t >= input_length))) {
            log_beta_data[lb_batch_offset + lb_input_stride * t +
                          lb_target_stride * s] = neginf;
          }
        }
      }
    };

    cgh.parallel_for<ctc_loss_beta_kernel<scalar_t, target_t>>(
        DPCPP::nd_range<2>(global_range, local_range), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

// This implements the subtrahend of equation (16) for all *nonblank*
// characters.
// It assumes you have probs in gradient_data when called
// and it modifies gradient_data to be, the gradient.
// In order to facilitate this inplace update, We don't actually do this in
// logspace.
// (The other variant implemented uses log_space and the differences seem to be
//  not so problematic at least with unit normal distributed test activations.)
// Internally this uses atomicAdd because different threads may write to the
// same
// gradient position.
// This is parallelised over b and s again.
// Note that for us, the Z of eqn (16) is actually constant for all t and it is
// the
// likelihood - this is why we use the negative log likelihood below.
// We also multiply by the input gradient to keep with standard autograd style.
// I took this trick from [2], for moderate alphabet sizes a log-space
// calculation (with an atomic log add) is similarly in performance, but for
// large
// alphabets the inplace nature is a considerable advantage.
template <typename scalar_t, typename target_t>
void ctc_loss_backward_collect_nonblank_kernel(
    Tensor& gradient,
    const Tensor& grad_out,
    int64_t grad_out_batch_stride,
    const Tensor& log_alpha,
    const Tensor& log_beta,
    const Tensor& log_probs,
    const Tensor& input_lengths,
    int64_t max_input_length,
    const Tensor& targets,
    const Tensor& target_lengths,
    int64_t max_target_length,
    const Tensor& neg_log_likelihood,
    const Tensor& tg_batch_offsets,
    int64_t tg_target_stride,
    int64_t batch_size,
    int64_t num_labels,
    int64_t BLANK,
    bool zero_infinity) {
  int64_t gr_input_stride = gradient.stride(0);
  int64_t gr_batch_stride = gradient.stride(1);
  int64_t gr_char_stride = gradient.stride(2);
  int64_t lp_input_stride = log_probs.stride(0);
  int64_t lp_batch_stride = log_probs.stride(1);
  int64_t lp_char_stride = log_probs.stride(2);
  int64_t la_batch_stride = log_alpha.stride(0);
  int64_t la_input_stride = log_alpha.stride(1);
  int64_t la_target_stride = log_alpha.stride(2);
  int64_t lb_batch_stride = log_beta.stride(0);
  int64_t lb_input_stride = log_beta.stride(1);
  int64_t lb_target_stride = log_beta.stride(2);

  auto& sycl_queue = getCurrentDPCPPStream().dpcpp_queue();
  DPCPP::range<2> global_range, local_range;
  std::tie(global_range, local_range) =
      get_work_range(sycl_queue, max_target_length, batch_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradient_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, gradient.data_ptr());
    auto grad_out_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, grad_out.data_ptr());
    auto log_alpha_acc =
        DPCPPAccessor<dpcpp_rw_mode>(cgh, log_alpha.data_ptr());
    auto log_beta_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, log_beta.data_ptr());
    auto log_probs_acc =
        DPCPPAccessor<dpcpp_discard_w_mode>(cgh, log_probs.data_ptr());
    auto input_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, input_lengths.data_ptr());
    auto targets_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, targets.data_ptr());
    auto target_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, target_lengths.data_ptr());
    auto neg_log_likelihood_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, neg_log_likelihood.data_ptr());
    auto tg_batch_offsets_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, tg_batch_offsets.data_ptr());
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      scalar_t* gradient_data = gradient_acc.template get_pointer<scalar_t>();
      scalar_t* grad_out_data = grad_out_acc.template get_pointer<scalar_t>();
      scalar_t* log_alpha_data = log_alpha_acc.template get_pointer<scalar_t>();
      scalar_t* log_beta_data = log_beta_acc.template get_pointer<scalar_t>();
      scalar_t* log_probs_data = log_probs_acc.template get_pointer<scalar_t>();
      target_t* targets_data = targets_acc.template get_pointer<target_t>();
      scalar_t* neg_log_likelihood_data =
          neg_log_likelihood_acc.template get_pointer<scalar_t>();
      int64_t* input_lengths_data =
          input_lengths_acc.template get_pointer<int64_t>();
      int64_t* target_lengths_data =
          target_lengths_acc.template get_pointer<int64_t>();
      int64_t* tg_batch_offsets_data =
          tg_batch_offsets_acc.template get_pointer<int64_t>();

      int64_t b = item_id.get_local_id(1) +
          item_id.get_group(1) * item_id.get_local_range(1);
      int64_t s = item_id.get_local_id(0) +
          item_id.get_group(0) * item_id.get_local_range(0);
      // note, this directly indexes into targets, no targets prime!

      if (b >= batch_size)
        return;

      int64_t input_length = input_lengths_data[b];
      int64_t target_length = target_lengths_data[b];
      int64_t gr_batch_offset = b * gr_batch_stride;
      int64_t lp_batch_offset = b * lp_batch_stride;
      int64_t la_batch_offset = b * la_batch_stride;
      int64_t lb_batch_offset = b * lb_batch_stride;
      int64_t tg_batch_offset = tg_batch_offsets_data[b];

      if (s >= target_length)
        return;

      int64_t target = targets_data[tg_batch_offset + s * tg_target_stride];
      scalar_t nll = neg_log_likelihood_data[b];
      scalar_t gr = grad_out_data[b * grad_out_batch_stride];

      if (zero_infinity && nll == INFINITY)
        return;

      for (int64_t t = 0; t < input_length; t++) {
        scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride +
                                     lp_char_stride * target];
        atomicAdd(
            &gradient_data[gr_batch_offset + t * gr_input_stride +
                           gr_char_stride * target],
            -Numerics<scalar_t>::exp(
                log_alpha_data[la_batch_offset + la_input_stride * t +
                               la_target_stride * (s * 2 + 1)] +
                log_beta_data[lb_batch_offset + lb_input_stride * t +
                              lb_target_stride * (s * 2 + 1)] +
                nll - lp) *
                gr);
      }
    };

    cgh.parallel_for<ctc_loss_collect_nonblank_kernel<scalar_t, target_t>>(
        DPCPP::nd_range<2>(global_range, local_range), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

// This is the naive implementation of equation (16). It is parallelised in
// batch and input timestep.
// It appears to be faster than the above method for small batch sizes.
template <typename scalar_t, typename target_t>
void ctc_loss_backward_collect_kernel(
    Tensor& gradient,
    const Tensor& grad_out,
    int64_t grad_out_batch_stride,
    const Tensor& log_alpha,
    const Tensor& log_beta,
    const Tensor& log_probs,
    const Tensor& input_lengths,
    int64_t max_input_length,
    const Tensor& targets,
    const Tensor& target_lengths,
    int64_t max_target_length,
    const Tensor& neg_log_likelihood,
    const Tensor& tg_batch_offsets,
    int64_t tg_target_stride,
    int64_t batch_size,
    int64_t num_labels,
    int64_t BLANK,
    bool zero_infinity) {
  constexpr scalar_t neginf = -INFINITY;

  int64_t gr_input_stride = gradient.stride(0);
  int64_t gr_batch_stride = gradient.stride(1);
  int64_t gr_char_stride = gradient.stride(2);
  int64_t lp_input_stride = log_probs.stride(0);
  int64_t lp_batch_stride = log_probs.stride(1);
  int64_t lp_char_stride = log_probs.stride(2);
  int64_t la_batch_stride = log_alpha.stride(0);
  int64_t la_input_stride = log_alpha.stride(1);
  int64_t la_target_stride = log_alpha.stride(2);
  int64_t lb_batch_stride = log_beta.stride(0);
  int64_t lb_input_stride = log_beta.stride(1);
  int64_t lb_target_stride = log_beta.stride(2);

  auto& sycl_queue = getCurrentDPCPPStream().dpcpp_queue();
  DPCPP::range<2> global_range, local_range;
  std::tie(global_range, local_range) =
      get_work_range(sycl_queue, log_probs.size(0), batch_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradient_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, gradient.data_ptr());
    auto grad_out_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, grad_out.data_ptr());
    auto log_alpha_acc =
        DPCPPAccessor<dpcpp_rw_mode>(cgh, log_alpha.data_ptr());
    auto log_beta_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, log_beta.data_ptr());
    auto log_probs_acc =
        DPCPPAccessor<dpcpp_discard_w_mode>(cgh, log_probs.data_ptr());
    auto input_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, input_lengths.data_ptr());
    auto targets_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, targets.data_ptr());
    auto target_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, target_lengths.data_ptr());
    auto neg_log_likelihood_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, neg_log_likelihood.data_ptr());
    auto tg_batch_offsets_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, tg_batch_offsets.data_ptr());
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      scalar_t* gradient_data = gradient_acc.template get_pointer<scalar_t>();
      scalar_t* grad_out_data = grad_out_acc.template get_pointer<scalar_t>();
      scalar_t* log_alpha_data = log_alpha_acc.template get_pointer<scalar_t>();
      scalar_t* log_beta_data = log_beta_acc.template get_pointer<scalar_t>();
      scalar_t* log_probs_data = log_probs_acc.template get_pointer<scalar_t>();
      target_t* targets_data = targets_acc.template get_pointer<target_t>();
      scalar_t* neg_log_likelihood_data =
          neg_log_likelihood_acc.template get_pointer<scalar_t>();
      int64_t* input_lengths_data =
          input_lengths_acc.template get_pointer<int64_t>();
      int64_t* target_lengths_data =
          target_lengths_acc.template get_pointer<int64_t>();
      int64_t* tg_batch_offsets_data =
          tg_batch_offsets_acc.template get_pointer<int64_t>();

      int64_t b = item_id.get_local_id(1) +
          item_id.get_group(1) * item_id.get_local_range(1);
      int64_t t = item_id.get_local_id(0) +
          item_id.get_group(0) * item_id.get_local_range(0);

      if ((t >= max_input_length) || (b >= batch_size))
        return;

      int64_t input_length = input_lengths_data[b];
      int64_t target_length = target_lengths_data[b];
      int64_t gr_batch_offset = b * gr_batch_stride;
      int64_t lp_batch_offset = b * lp_batch_stride;
      int64_t la_batch_offset = b * la_batch_stride;
      int64_t lb_batch_offset = b * lb_batch_stride;
      int64_t tg_batch_offset = tg_batch_offsets_data[b];

      // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
      for (int s = 0; s < 2 * max_target_length + 1; s++) {
        if (s < 2 * target_length + 1) { // if target_length == 0, s == 0
          int64_t current_target_prime = get_target_prime(
              targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
          scalar_t log_alpha_beta =
              (log_alpha_data[la_batch_offset + la_input_stride * t +
                              la_target_stride * s] +
               log_beta_data[lb_batch_offset + lb_input_stride * t +
                             lb_target_stride * s]);
          scalar_t& lcab = gradient_data[gr_batch_offset + t * gr_input_stride +
                                         gr_char_stride * current_target_prime];
          if (lcab == neginf) {
            lcab = log_alpha_beta;
          } else {
            scalar_t max = ((lcab > log_alpha_beta) ? lcab : log_alpha_beta);
            lcab = Numerics<scalar_t>::log(
                       Numerics<scalar_t>::exp(lcab - max) +
                       Numerics<scalar_t>::exp(log_alpha_beta - max)) +
                max;
          }
        }
      }

      scalar_t nll = neg_log_likelihood_data[b];
      scalar_t gr = grad_out_data[b * grad_out_batch_stride];

      for (int64_t c = 0; c < num_labels; c++) {
        scalar_t& res = gradient_data[gr_batch_offset + t * gr_input_stride +
                                      gr_char_stride * c];
        if (t < input_length && (!zero_infinity || nll != INFINITY)) {
          scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride +
                                       lp_char_stride * c];
          res = (Numerics<scalar_t>::exp(lp) -
                 Numerics<scalar_t>::exp(res + nll - lp)) *
              gr;
        } else {
          res = 0.;
        }
      }
    };

    cgh.parallel_for<ctc_loss_collect_kernel<scalar_t, target_t>>(
        DPCPP::nd_range<2>(global_range, local_range), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

// This is to zero gradients which corresponding to the out-of-sequence position
// Those gradients should not be used in any model update since the input
// elements are padded
template <typename scalar_t>
void ctc_loss_zero_padded_gradients(
    Tensor& gradient, /* (T, B, D) layout */
    const Tensor& input_lengths, /* (B, ) layout */
    int64_t gr_timestep_stride,
    int64_t gr_batch_stride,
    int64_t gr_label_stride,
    int64_t max_input_length, /* T */
    int64_t batch_size, /* B */
    int64_t num_labels /* D */) {
  auto& sycl_queue = getCurrentDPCPPStream().dpcpp_queue();

  DPCPP::range<2> global_range, local_range;
  std::tie(global_range, local_range) =
      get_work_range(sycl_queue, max_input_length, batch_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradient_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, gradient.data_ptr());
    auto input_lengths_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, input_lengths.data_ptr());
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      scalar_t* gradient_data = gradient_acc.template get_pointer<scalar_t>();
      int64_t* input_lengths_data =
          input_lengths_acc.template get_pointer<int64_t>();

      int64_t b = item_id.get_local_id(1) +
          item_id.get_group(1) * item_id.get_local_range(1);
      int64_t t = item_id.get_local_id(0) +
          item_id.get_group(0) * item_id.get_local_range(0);

      if (b >= batch_size || t >= max_input_length) {
        return;
      }

      scalar_t input_length = input_lengths_data[b];
      if (t >= input_length) {
        for (int l = 0; l < num_labels; l++)
          gradient_data[t * gr_timestep_stride + b * gr_batch_stride +
                        l * gr_label_stride] = 0.0f;
      }
    };

    cgh.parallel_for<ctc_loss_zero_padded_kernel<scalar_t>>(
        DPCPP::nd_range<2>(global_range, local_range), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

// The forward computation. Lot's of admin and a call to the alpha kernel.
// Note: we do not check that the labels are in the valid range. As we use
// them for indexing in the kernels, you'll see memory errors when you
// pass corrupt labels.
// We support both a 2-dimensional tensor as targets (one set of targets in each
// row) and
// a 1-dimensional tensor where all targets are concatenated (and we use
// target_lengths
// to figure out where they begin).
// We return log_alpha (currently, might change to (log_alpha+log_beta) to be
// passed to the
// backward. The dispatch function will only return the loss.
template <typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> ctc_loss_template(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)
  CheckedFrom c = "ctc_loss_sycl";
  using target_t =
      typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  checkBackend(c, {log_probs, targets}, Backend::DPCPP);

  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  checkScalarType(c, targets_arg, target_scalar_type);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK(
      (0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK(
      input_lengths.size() == batch_size,
      "input_lengths must be of size batch_size");
  TORCH_CHECK(
      target_lengths.size() == batch_size,
      "target_lengths must be of size batch_size");

  int64_t lp_input_stride = log_probs.stride(0);
  int64_t lp_char_stride = log_probs.stride(2);
  int64_t tg_target_stride;

  int64_t max_target_length = 0;
  auto tg_batch_offsets =
      at::empty({batch_size}, at::device(at::kCPU).dtype(at::kLong));
  auto tg_batch_offsets_data = tg_batch_offsets.data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
    checkSize(c, targets_arg, 0, pos);
  } else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(1);
    checkSize(c, targets_arg, 0, batch_size);
    TORCH_CHECK(
        targets.size(1) >= max_target_length,
        "Expected tensor to have size at least ",
        max_target_length,
        " at dimension 1, but got size ",
        targets.size(1),
        " for ",
        targets_arg,
        " (while checking arguments for ",
        c,
        ")");
  }
  int64_t max_input_length = log_probs.size(0);
  for (int64_t b = 0; b < batch_size; b++) {
    TORCH_CHECK(
        input_lengths[b] <= max_input_length,
        "Expected tensor to have size at least ",
        max_input_length,
        " at dimension 1, but got size ",
        targets.size(0),
        " for ",
        targets_arg,
        " (while checking arguments for ",
        c,
        ")");
  }

  auto target_lengths_t =
      at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t =
      at::tensor(input_lengths, targets.options().dtype(kLong));
  tg_batch_offsets = tg_batch_offsets.to("dpcpp");

  Tensor log_alpha = at::empty(
      {batch_size, log_probs.size(0), 2 * max_target_length + 1},
      log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  ctc_loss_log_alpha_kernel<scalar_t, target_t>(
      log_alpha,
      log_probs,
      input_lengths_t,
      log_probs.size(0),
      targets,
      target_lengths_t,
      max_target_length,
      neg_log_likelihood,
      tg_batch_offsets,
      tg_target_stride,
      batch_size,
      BLANK);

  return std::make_tuple(neg_log_likelihood, log_alpha);
}

// The backward. It essentially computes eq 16 by using the above kernels.
// We don't do a lot of checking as we envision this to be called only when
// backpropagating through a (well-checked) forward.
template <typename scalar_t, ScalarType target_scalar_type>
Tensor ctc_loss_backward_template(
    const Tensor& grad_out,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity) {
  constexpr scalar_t neginf = -INFINITY;
  using target_t =
      typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  int64_t lp_input_stride = log_probs.stride(0);
  int64_t lp_char_stride = log_probs.stride(2);
  int64_t tg_target_stride;

  int64_t max_target_length;
  auto tg_batch_offsets =
      at::empty({batch_size}, TensorOptions(at::CPU(kLong)));
  auto tg_batch_offsets_data = tg_batch_offsets.data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
  } else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length =
        log_alpha.size(2) / 2; // targets.size(1) might be larger
  }
  auto target_lengths_t =
      at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t =
      at::tensor(input_lengths, targets.options().dtype(kLong));
  tg_batch_offsets = tg_batch_offsets.to("dpcpp");

  Tensor log_beta = at::empty_like(log_alpha);
  log_beta.fill_(neginf);

  Tensor grad = at::full_like(
      log_probs, neginf); // initialization for log(sum (alpha beta))

  // As above, there may be better configurations to use.
  ctc_loss_backward_log_beta_kernel<scalar_t, target_t>(
      log_beta,
      log_probs,
      input_lengths_t,
      log_probs.size(0),
      targets,
      target_lengths_t,
      max_target_length,
      tg_batch_offsets,
      tg_target_stride,
      batch_size,
      BLANK);

  // Very crude heuristic for what is a small problem., based on linearly
  // regressing problem dimensions on
  // the (capped) difference of timings.
  // Note that for OK problems target length <= input length, so we
  // only consider input length.
  bool is_large = (2 * log_probs.size(0) + (24 * batch_size) / 10 +
                   (2 * num_labels) / 10) > 450;
  if (is_large) { // large alphabet, large batch
    // this computes the probs, minuend in (16)
    exp_out(grad, log_probs);
    // now we compute the subtrahend for the blanks. It is a straightforward
    // reduction because we know that
    // blanks are in every other position.
    // maybe we should kernelize this, too.
    auto grad_blank = grad.narrow(2, BLANK, 1);
    grad_blank -=
        (at::logsumexp(
             log_alpha.as_strided(
                 {batch_size, log_alpha.size(1), max_target_length + 1},
                 {log_alpha.stride(0),
                  log_alpha.stride(1),
                  log_alpha.stride(2) * 2}) +
                 log_beta.as_strided(
                     {batch_size, log_beta.size(1), max_target_length + 1},
                     {log_beta.stride(0),
                      log_beta.stride(1),
                      log_beta.stride(2) * 2}),
             2,
             true)
             .permute({1, 0, 2})
             .add_(neg_log_likelihood.view({1, batch_size, 1}))
             .sub_(log_probs.narrow(2, BLANK, 1))
             .exp_());
    // scale by output gradient (blanks and first summand of non-blanks)
    grad *= grad_out.view({1, batch_size, 1});
    if (zero_infinity) {
      grad = at::where(
          neg_log_likelihood.view({1, batch_size, 1}) == Scalar(INFINITY),
          at::zeros({}, grad.options()),
          grad);
    }

    ctc_loss_backward_collect_nonblank_kernel<scalar_t, target_t>(
        grad,
        grad_out,
        grad_out.stride(0),
        log_alpha,
        log_beta,
        log_probs,
        input_lengths_t,
        log_probs.size(0),
        targets,
        target_lengths_t,
        max_target_length,
        neg_log_likelihood,
        tg_batch_offsets,
        tg_target_stride,
        batch_size,
        num_labels,
        BLANK,
        zero_infinity);
  } else { // small problem, use naive algorithm
    // Still no block/grid configuration guru...
    ctc_loss_backward_collect_kernel<scalar_t, target_t>(
        grad,
        grad_out,
        grad_out.stride(0),
        log_alpha,
        log_beta,
        log_probs,
        input_lengths_t,
        log_probs.size(0),
        targets,
        target_lengths_t,
        max_target_length,
        neg_log_likelihood,
        tg_batch_offsets,
        tg_target_stride,
        batch_size,
        num_labels,
        BLANK,
        zero_infinity);
  }

  // zero those invalid graident elements due to padding
  ctc_loss_zero_padded_gradients<scalar_t>(
      grad,
      input_lengths_t,
      grad.stride(0),
      grad.stride(1),
      grad.stride(2),
      grad.size(0),
      grad.size(1),
      grad.size(2));

  return grad;
}
} // namespace impl

std::tuple<Tensor, Tensor> _ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t blank,
    bool zero_infinity) {
  (void)zero_infinity; // only used for backward
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss", [&] {
    if (targets.scalar_type() == kLong) {
      return impl::ctc_loss_template<scalar_t, kLong>(
          log_probs, targets, input_lengths, target_lengths, blank);
    } else {
      return impl::ctc_loss_template<scalar_t, kInt>(
          log_probs, targets, input_lengths, target_lengths, blank);
    }
  });
}

Tensor _ctc_loss_backward(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity) {
  return AT_DISPATCH_ALL_ATOMIC_TYPES(
      log_probs.scalar_type(), "ctc_loss_backward", [&] {
        if (targets.scalar_type() == kLong) {
          return impl::ctc_loss_backward_template<scalar_t, kLong>(
              grad,
              log_probs,
              targets,
              input_lengths,
              target_lengths,
              neg_log_likelihood,
              log_alpha,
              blank,
              zero_infinity);
        } else {
          return impl::ctc_loss_backward_template<scalar_t, kInt>(
              grad,
              log_probs,
              targets,
              input_lengths,
              target_lengths,
              neg_log_likelihood,
              log_alpha,
              blank,
              zero_infinity);
        }
      });
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
