#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <core/MemoryFormat.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include <ATen/record_function.h>
#include <torch/library.h>
#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
static inline scalar_t logSumExp(scalar_t a, scalar_t b) {
  return (a >= b)
      ? a + Numerics<scalar_t>::log(1 + Numerics<scalar_t>::exp(b - a))
      : b + Numerics<scalar_t>::log(1 + Numerics<scalar_t>::exp(a - b));
}

template <typename scalar_t, typename acc_t>
void transducer_loss_forward_kernel(
    const scalar_t* x,
    const int* label,
    const int* audLen,
    const int* txtLen,
    const int64_t* batchOffset,
    int64_t dictSize,
    int64_t blankIdx,
    int64_t maxFLen,
    int64_t maxGLen,
    bool packedInput,
    const int batchSize,
    acc_t* alpha,
    acc_t* beta,
    scalar_t* loss) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  local_size = std::min(local_size, maxGLen);
  sycl::range<3> local_range{1, 1, local_size};
  sycl::range<3> global_range{2, batchSize, local_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<int> local_label(maxGLen, cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(global_range), sycl::range<3>(local_range)),
        [=](sycl::nd_item<3> item_id) {
          auto local_id = item_id.get_local_id(2);
          int batch = item_id.get_group(1);
          int path = item_id.get_group(0);

          const auto myFLen = audLen[batch];
          const auto myGLen = txtLen[batch] + 1;
          const auto myLabel = label + batch * (maxGLen - 1);
          const int64_t myBatchOffset = packedInput
              ? (batch == 0 ? 0 : batchOffset[batch - 1])
              : batch * maxFLen * maxGLen;
          const int64_t myStrideT = packedInput ? myGLen : maxGLen;

          const scalar_t* myX = x + myBatchOffset * dictSize;

          for (int i = local_id; i < maxGLen; i += local_size) {
            local_label[i] = myLabel[i];
          }
          item_id.barrier();

          if (path == 0) {
            acc_t* myAlpha = alpha + batch * maxFLen * maxGLen;
            if (local_id == 0)
              myAlpha[0] = 0;
            item_id.barrier();

            for (int64_t step = 1; step < myFLen + myGLen - 1; ++step) {
              for (int64_t u = local_id; u < myGLen; u += local_size) {
                int64_t t = step - u;
                if (t >= 0 && t < myFLen && u >= 0 && u < myGLen) {
                  if (u == 0) {
                    // alpha(t, u) = alpha(t-1, u) * null(t-1, u)
                    myAlpha[t * maxGLen] = myAlpha[(t - 1) * maxGLen] +
                        myX[((t - 1) * myStrideT) * dictSize + blankIdx];
                  } else if (t == 0) {
                    // alpha(t, u-1) = alpha(t, u-1) * y(t, u-1)
                    myAlpha[u] = myAlpha[u - 1] +
                        myX[(u - 1) * dictSize + local_label[u - 1]];
                  } else {
                    // alpha(t, u) = alpha(t-1, u) * null(t-1, u) + alpha(t,
                    // u-1) * y(t, u-1)
                    acc_t current = myAlpha[(t - 1) * maxGLen + u] +
                        myX[((t - 1) * myStrideT + u) * dictSize + blankIdx];
                    acc_t next = myAlpha[t * maxGLen + u - 1] +
                        myX[(t * myStrideT + u - 1) * dictSize +
                            local_label[u - 1]];
                    myAlpha[t * maxGLen + u] = logSumExp<acc_t>(next, current);
                  }
                }
              }
              item_id.barrier();
            }
          } else {
            // beta path
            acc_t* myBeta = beta + batch * maxFLen * maxGLen;
            if (local_id == 0) {
              myBeta[(myFLen - 1) * maxGLen + myGLen - 1] =
                  myX[((myFLen - 1) * myStrideT + myGLen - 1) * dictSize +
                      blankIdx];
            }
            item_id.barrier();
            for (int64_t step = myFLen + myGLen - 3; step >= 0; --step) {
              for (int64_t u = local_id; u < myGLen; u += local_size) {
                int64_t t = step - u;
                if (t >= 0 and t < myFLen and u >= 0 and u < myGLen) {
                  if (u == myGLen - 1) {
                    // beta(t,u) = beta(t+1,u) * null(t,u)
                    myBeta[t * maxGLen + u] = myBeta[(t + 1) * maxGLen + u] +
                        myX[(t * myStrideT + u) * dictSize + blankIdx];
                  } else if (t == myFLen - 1) {
                    // beta(t,u) = beta(t,u+1) * y(t,u)
                    myBeta[t * maxGLen + u] = myBeta[t * maxGLen + u + 1] +
                        myX[(t * myStrideT + u) * dictSize + local_label[u]];
                  } else {
                    // beta(t,u) = beta(t+1,u)*null(t,u) + beta(t,u+1)*y(t,u)
                    int64_t offset1 = (t + 1) * maxGLen + u;
                    int64_t offset2 = (t * myStrideT + u) * dictSize + blankIdx;
                    int64_t offset3 = t * maxGLen + u + 1;
                    int64_t offset4 =
                        (t * myStrideT + u) * dictSize + local_label[u];
                    acc_t current = myBeta[(t + 1) * maxGLen + u] +
                        myX[(t * myStrideT + u) * dictSize + blankIdx];
                    acc_t next = myBeta[t * maxGLen + u + 1] +
                        myX[(t * myStrideT + u) * dictSize + local_label[u]];
                    myBeta[int64_t(t * maxGLen + u)] =
                        logSumExp<acc_t>(next, current);
                  }
                }
              }
              item_id.barrier();
            }
            if (local_id == 0) {
              loss[batch] = -myBeta[0];
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> transducer_loss_forward(
    at::Tensor x,
    at::Tensor label,
    at::Tensor audLen,
    at::Tensor txtLen,
    at::Tensor batchOffset,
    int64_t maxFLen,
    int64_t blankIdx,
    int64_t opt,
    bool packedInput) {
  auto scalarType = x.scalar_type();
  auto tensorOpt = x.options();
  const int batchSize = label.size(0);
  const int maxGLen = label.size(1) + 1;
  const int dictSize = x.size(-1);

  TORCH_CHECK(
      blankIdx >= 0 and blankIdx < dictSize,
      "Expected blank index to be in the range of 0 to ",
      dictSize - 1,
      ", but got ",
      blankIdx);
  TORCH_CHECK(
      opt == -1 or opt == 0 or opt == 1,
      "Got an invalid optimization level ",
      opt);

  at::Tensor alpha;
  at::Tensor beta;
  at::Tensor loss = at::empty({batchSize}, tensorOpt);
  const auto batchOffsetPtr =
      packedInput ? batchOffset.data_ptr<int64_t>() : nullptr;
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalarType,
      "transducer_loss_forward",
      [&]() {
        using acc_t = acc_type<scalar_t>;
        const auto kAccType = (scalarType == kHalf || scalarType == kBFloat16)
            ? kFloat
            : scalarType;
        alpha =
            at::empty({batchSize, maxFLen, maxGLen}, tensorOpt.dtype(kAccType));
        beta =
            at::empty({batchSize, maxFLen, maxGLen}, tensorOpt.dtype(kAccType));

        transducer_loss_forward_kernel<scalar_t, acc_t>(
            x.data_ptr<scalar_t>(),
            label.data_ptr<int>(),
            audLen.data_ptr<int>(),
            txtLen.data_ptr<int>(),
            batchOffsetPtr,
            dictSize,
            blankIdx,
            maxFLen,
            maxGLen,
            packedInput,
            batchSize,
            alpha.data_ptr<acc_t>(),
            beta.data_ptr<acc_t>(),
            loss.data_ptr<scalar_t>());
      });
  return std::make_tuple(alpha, beta, loss);
}

template <typename scalar_t, typename acc_t>
void transducer_loss_backward_kernel(
    const scalar_t* x,
    const scalar_t* lossGrad,
    const int* audLen,
    const int* txtLen,
    const int* label,
    const acc_t* alpha,
    const acc_t* beta,
    const int64_t* batchOffset,
    int64_t dictSize,
    int64_t blankIdx,
    int64_t maxFLen,
    int64_t maxGLen,
    bool packedInput,
    const int batchSize,
    scalar_t* xGrad) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  local_size = std::min(local_size, maxGLen);
  sycl::range<3> local_range{1, 1, local_size};
  sycl::range<3> global_range{batchSize, maxFLen, local_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<int> local_label(maxGLen, cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(global_range), sycl::range<3>(local_range)),
        [=](sycl::nd_item<3> item_id) {
          const int local_id = item_id.get_local_id(2);
          const int t = item_id.get_group(1);
          const int batch = item_id.get_group(0);
          const int64_t myFLen = audLen[batch];
          const int64_t myGLen = txtLen[batch] + 1;
          const int64_t myBatchOffset = packedInput
              ? (batch == 0 ? 0 : batchOffset[batch - 1])
              : batch * maxFLen * maxGLen;
          const int64_t myStrideT = packedInput ? myGLen : maxGLen;
          auto myX = x + (myBatchOffset + t * myStrideT) * dictSize;
          auto myAlpha = alpha + batch * maxFLen * maxGLen;
          auto myBeta = beta + batch * maxFLen * maxGLen;
          auto myXGrad = xGrad + (myBatchOffset + t * myStrideT) * dictSize;
          auto myLabel = label + batch * (maxGLen - 1);

          int64_t u = local_id;
          while (t < myFLen && u < myGLen) {
            acc_t grad = Numerics<acc_t>::log(lossGrad[batch]) +
                myAlpha[t * maxGLen + u] - myBeta[0];
            if (u != myGLen - 1)
              myXGrad[u * dictSize + myLabel[u]] = -Numerics<acc_t>::exp(
                  grad + myBeta[t * maxGLen + u + 1] +
                  myX[u * dictSize + myLabel[u]]);
            if (t == myFLen - 1 && u == myGLen - 1)
              myXGrad[u * dictSize + blankIdx] =
                  -Numerics<acc_t>::exp(grad + myX[u * dictSize + blankIdx]);
            else if (t != myFLen - 1)
              myXGrad[u * dictSize + blankIdx] = -Numerics<acc_t>::exp(
                  grad + myBeta[(t + 1) * maxGLen + u] +
                  myX[u * dictSize + blankIdx]);

            u += local_size;
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor transducer_loss_backward(
    Tensor x,
    Tensor lossGrad,
    Tensor alpha,
    Tensor beta,
    Tensor audLen,
    Tensor txtLen,
    Tensor label,
    Tensor batchOffset,
    int64_t maxFLen,
    int64_t blankIdx,
    int64_t opt,
    bool fuseSoftmaxBackward,
    bool packedInput) {
  auto scalarType = x.scalar_type();
  Tensor xGrad;
  const int batchSize = label.size(0);
  const int maxGLen = label.size(1) + 1;
  const int dictSize = x.size(-1);
  const auto batchOffsetPtr =
      packedInput ? batchOffset.data_ptr<int64_t>() : nullptr;
  // for non-fused kernel, the gradients need to be writtern are very sparse,
  // hence initialize the tensor with all zeros.
  xGrad = at::zeros_like(x);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalarType,
      "transducer_loss_backward",
      [&]() {
        using acc_t = acc_type<scalar_t>;
        transducer_loss_backward_kernel<scalar_t, acc_t>(
            x.data_ptr<scalar_t>(),
            lossGrad.data_ptr<scalar_t>(),
            audLen.data_ptr<int>(),
            txtLen.data_ptr<int>(),
            label.data_ptr<int>(),
            alpha.data_ptr<acc_t>(),
            beta.data_ptr<acc_t>(),
            batchOffsetPtr,
            dictSize,
            blankIdx,
            maxFLen,
            maxGLen,
            packedInput,
            batchSize,
            xGrad.data_ptr<scalar_t>());
      });
  return xGrad;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "transducer_loss_forward(Tensor x, Tensor label, Tensor fLen, Tensor yLen, Tensor batchOffset, int maxFLen, int blankIdx, int opt, bool packedInput) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "transducer_loss_forward",
      c10::DispatchKey::XPU,
      at::AtenIpexTypeXPU::transducer_loss_forward);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "transducer_loss_backward(Tensor x, Tensor lossGrad, Tensor alpha, Tensor beta, Tensor autoLen, Tensor txtLen, Tensor label, Tensor batchOffset, int maxFLen, int blankIdx, int opt, bool fuseSoftmaxBackward, bool packedInput) -> Tensor");
  m.impl(
      "transducer_loss_backward",
      c10::DispatchKey::XPU,
      at::AtenIpexTypeXPU::transducer_loss_backward);
}

} // namespace
