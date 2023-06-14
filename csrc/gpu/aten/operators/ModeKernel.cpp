#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/record_function.h>
#include <c10/macros/Macros.h>

#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "BitonicMergeSort.h"
#include "ReduceOpsUtils.h"
#include "SortingCommon.h"
#include "SortingRadixSelect.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::native;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

// this helper's meaning:
// The [status] contains the judgement result comapred with the adjecent
// elements's equivalence. In reduce, the [status] is used to help to record the
// max appearance times's index after scan. The [value] is the initial number
// according to the status after comparison. It is used in the following scan.
struct ModeOpHelper {
  // why not int64_t: ModeOpHelper is used for the condition that problem
  // size is equal to or smaller than the work group max size, so the
  // accumulation value will not exceed the int32_t range
  int32_t status;
  int32_t value;
};

// this is used to record the sorted value(T) and the associated index(int64_t)
// for the fused mode kernel
template <typename T>
struct ModeOpValueIndex {
  T value;
  int64_t index;
};

// problem size >> array size
template <typename T, class Functor, typename item_t>
static inline void ConditionalInclusiveScanForMode(
    T* memStatus,
    T* memValue,
    Functor functor,
    const int64_t problem_size,
    const int64_t outer_offset,
    const int64_t inner_limit,
    const item_t& item) {
  auto id = item.get_local_id(0);
  auto group_size = item.get_local_range(0);

  for (auto inner_id = id; inner_id < inner_limit; inner_id += group_size) {
    // x x x x | x x x x | x x x x | x x x x
    //       ^   ^ --- one item solely compute these two values exclude the
    //       first one
    auto global_id = outer_offset + inner_id;
    if (id == 0 && inner_id != 0 && inner_id < problem_size) {
      std::tie(memStatus[global_id], memValue[global_id]) = functor(
          memStatus[global_id - 1],
          memValue[global_id - 1],
          memStatus[global_id],
          memValue[global_id]);
    }
    item.barrier(dpcpp_global_fence);

    // x x x x | x x x x | x x x x | x x x x
    // ^ ^ ^ ^ --- one group scan one piece of the full data
    for (auto stride = 1; stride < group_size; stride <<= 1) {
      T PreStatus = 0;
      T PreValue = 0;
      T CurStatus = 0;
      T CurValue = 0;
      if (inner_id < problem_size && id >= stride) {
        PreStatus = memStatus[global_id - stride];
        PreValue = memValue[global_id - stride];
        CurStatus = memStatus[global_id];
        CurValue = memValue[global_id];
      }
      item.barrier(dpcpp_global_fence);
      if (inner_id < problem_size && id >= stride) {
        std::tie(memStatus[global_id], memValue[global_id]) =
            functor(PreStatus, PreValue, CurStatus, CurValue);
      }
      item.barrier(dpcpp_global_fence);
    }
  }
}

// problem size = array size = slm size
// No implement it as down sweep and up sweep
template <typename T, class Functor, typename item_t>
static inline void ConditionalInclusiveScanForFusedMode(
    T* memHelper,
    Functor functor,
    const int64_t problem_size,
    const item_t& item) {
  auto id = item.get_local_id(0);
  for (auto stride = 1; stride < problem_size; stride <<= 1) {
    T PreElem;
    if (id >= stride) {
      PreElem = memHelper[id - stride];
    }
    item.barrier(dpcpp_local_fence);
    if (id >= stride) {
      memHelper[id] = functor(PreElem, memHelper[id]);
    }
    item.barrier(dpcpp_local_fence);
  }
}

// reduce, problem size >> array size
template <typename T, class Functor, typename item_t>
static inline void ReduceHelperForMode(
    T* mem,
    Functor functor,
    const int64_t problem_size,
    const int64_t outer_offset,
    const int64_t group_size,
    const item_t& item) {
  auto id = item.get_local_id(0);

  // first add
  for (auto inner_id = id + group_size; inner_id < problem_size;
       inner_id += group_size) {
    mem[outer_offset + id] =
        functor(mem[outer_offset + id], mem[outer_offset + inner_id]);
  }
  item.barrier(dpcpp_global_fence);

  // naive tree
  for (auto stride = group_size / 2; stride > 0; stride >>= 1) {
    if (id < stride) {
      auto tree_id = outer_offset + id;
      mem[tree_id] = functor(mem[tree_id], mem[tree_id + stride]);
    }
    item.barrier(dpcpp_global_fence);
  }
}

// reduce, problem size = array size = slm size
template <class Functor, typename item_t>
static inline void ReduceHelperForFusedMode(
    ModeOpHelper* mem,
    Functor functor,
    const int64_t group_size,
    const item_t& item) {
  auto id = item.get_local_id(0);

  // naive tree
  for (auto stride = group_size / 2; stride > 0; stride >>= 1) {
    if (id < stride) {
      mem[id].value = functor(mem[id], mem[id + stride]);
    }
    item.barrier(dpcpp_local_fence);

    // odd stride
    if (stride % 2 == 1) {
      if (id == 0) {
        mem[0].value = functor(mem[0], mem[stride - 1]);
      }
    }
    item.barrier(dpcpp_local_fence);
  }

  // odd group_size
  if (group_size % 2 == 1) {
    if (id == 0) {
      mem[0].value = functor(mem[0], mem[group_size - 1]);
    }
  }
  item.barrier(dpcpp_local_fence);
}

template <typename item_t>
static inline void ReduceGetMaxElemIndexForFusedMode(
    ModeOpHelper* mem,
    const int64_t group_size,
    const item_t& item) {
  auto id = item.get_local_id(0);
  // value : 0 1 0 1 [2] 0 0 1
  // status: 0 1 0 1  2  0 0 1
  // reduce to find the [maximal] value, it means the most appear times of Mode
  ReduceHelperForFusedMode(
      mem,
      [&](const ModeOpHelper& a, const ModeOpHelper& b) {
        return (a.value > b.value) ? (a.value) : (b.value);
      },
      group_size,
      item);
  item.barrier(dpcpp_local_fence);

  auto max_appearance_time = mem[0].value;
  item.barrier(dpcpp_local_fence);

  // id:          0  1 2 3  4  5 6 7
  // value :     [2] x x x  x  x x x (x means ignore value)
  // status:      0  1 0 1  2  0 0 1
  // new value:   M  M M M [4] M M M (M means the max value)
  mem[id].value = (mem[id].status == max_appearance_time)
      ? (id)
      : (std::numeric_limits<int32_t>::max());
  item.barrier(dpcpp_local_fence);

  // reduce again to find the [minimal] index and put it into mem[0].value
  // value: M M M M [4] M M M
  ReduceHelperForFusedMode(
      mem,
      [&](const ModeOpHelper& a, const ModeOpHelper& b) {
        return (a.value < b.value) ? (a.value) : (b.value);
      },
      group_size,
      item);
}

template <
    typename scalar_t,
    typename value_info_t,
    typename indice_info_t,
    typename item_t>
void mode_impl(
    const scalar_t* problem_values_ptr,
    const int64_t* problem_indices_ptr,
    value_info_t answer_values,
    indice_info_t answer_indices,
    scalar_t* slm_ptr,
    int64_t* scratch_status_ptr,
    int64_t* scratch_value_ptr,
    const int64_t problem_time,
    const int64_t problem_size,
    const int64_t wg_number,
    const int64_t wg_size,
    const int64_t inner_limit,
    const item_t& item) {
  auto group_id = item.get_group_linear_id();
  auto item_id = item.get_local_id(0);

  // outer loop, problem time level
  for (auto outer_id = group_id; outer_id < problem_time;
       outer_id += wg_number) {
    auto outer_offset = outer_id * problem_size;
    // inner loop, problem size level
    for (auto inner_id = item_id; inner_id < inner_limit; inner_id += wg_size) {
      auto global_index = outer_offset + inner_id;

      // load piece of data into slm
      if (inner_id < problem_size) {
        slm_ptr[item_id] = problem_values_ptr[global_index];
      }
      item.barrier(dpcpp_local_fence);

      // compare and record the status using true and false into scratch pad
      // buffer. 0 means begin a new sequence, 1 means the duplicated values.
      // sorted      values: 0 0 1 1 1 3 4 4 (here problem value is sorted)
      // associated indices: 4 6 0 5 7 2 1 3
      // scratch status:     0 1 0 1 1 0 0 1
      // scratch value:      0 1 0 1 1 0 0 1
      // the first value is always status 0 and value 0
      if (inner_id == 0) {
        scratch_status_ptr[outer_offset] = 0;
        scratch_value_ptr[outer_offset] = 0;
      } else {
        if (inner_id < problem_size) {
          // kick out the first item
          auto judgeEqual = false;
          if (item_id == 0) {
            // for the first one, its pre value is not in slm
            // slm:          0 1 1
            // global mem: 0 ^ ---- the pre one is in global mem
            judgeEqual =
                bool(problem_values_ptr[global_index - 1] == slm_ptr[item_id]);
          } else {
            judgeEqual = bool(slm_ptr[item_id - 1] == slm_ptr[item_id]);
          }
          scratch_status_ptr[global_index] = (judgeEqual) ? (1) : (0);
          scratch_value_ptr[global_index] = scratch_status_ptr[global_index];
        }
      }
      item.barrier(dpcpp_global_fence);
    }

    // index:               0 1 2 3 4 5 6 7
    // scratch status:      0 1 0 1 1 0 0 1
    // scratch value:       0 1 0 1 1 0 0 1
    // conditional scan rule:
    // 1. according to current status, if 0, keep the current value. If 1, do
    // accumulation
    // 2. new status = previous scratch status & current scratch status to
    // record if there is duplicated value.
    // The conditional inclusive scan result value should be: 0 1 0 1 2 0 0 1
    ConditionalInclusiveScanForMode(
        scratch_status_ptr,
        scratch_value_ptr,
        [&](const int64_t& PreStatus,
            const int64_t& PreValue,
            const int64_t& CurStatus,
            const int64_t& CurValue) {
          auto TempValue = CurStatus ? (PreValue + CurValue) : (CurValue);
          auto TempStatus = PreStatus & CurStatus;
          return std::make_tuple(TempStatus, TempValue);
        },
        problem_size,
        outer_offset,
        inner_limit,
        item);
    item.barrier(dpcpp_global_fence);

    // copy scratch value into status
    // scratch status:      0 1 0 1 2 0 0 1
    // scratch value:       0 1 0 1 2 0 0 1
    // now the status is changed to be used to record the scan result
    for (auto inner_id = item_id; inner_id < problem_size;
         inner_id += wg_size) {
      auto global_index = outer_offset + inner_id;
      scratch_status_ptr[global_index] = scratch_value_ptr[global_index];
    }
    item.barrier(dpcpp_global_fence);

    // scratch status:      0 1 0 1  2  0 0 1
    // scratch value:       0 1 0 1 [2] 0 0 1
    // reduce scratch value to find the max number
    ReduceHelperForMode(
        scratch_value_ptr,
        [&](const int64_t& a, const int64_t& b) { return (a < b) ? (b) : (a); },
        problem_size,
        outer_offset,
        wg_size,
        item);

    // the reduced maximal number is stored in the first slot
    auto most_appearance_time = scratch_value_ptr[outer_offset];
    item.barrier(dpcpp_global_fence);

    // update the value by comparing that if the status is equal to the found
    // maximul appearance times, if equal, assign the global index scratch
    // index:                    0 1 2 3  4  5 6 7
    // max appearance: [2]
    // scratch status:           0 1 0 1 [2] 0 0 1
    // update scratch value:     M M M M  4  M M M
    // M is max int number
    for (auto inner_id = item_id; inner_id < problem_size;
         inner_id += wg_size) {
      auto global_index = outer_offset + inner_id;
      scratch_value_ptr[global_index] =
          (scratch_status_ptr[global_index] == most_appearance_time)
          ? (inner_id)
          : (std::numeric_limits<int64_t>::max());
    }
    item.barrier(dpcpp_global_fence);

    // appearance in scratch value
    // scratch value:       M M M M [4] M M M
    // reduce scratch value to find the min number
    ReduceHelperForMode(
        scratch_value_ptr,
        [&](const int64_t& a, const int64_t& b) { return (a < b) ? (a) : (b); },
        problem_size,
        outer_offset,
        wg_size,
        item);

    // only one elem operation
    if (item_id == 0) {
      // the reduced minimal number is stored in the first index
      auto reduce_min_index = scratch_value_ptr[outer_offset];

      // index:                0 1 2 3 [4] 5 6 7
      // sorted indices:       4 6 0 5 [7] 2 1 3
      // find out the first-appeared and most-appeared element's original
      // indices, it is 7
      auto answer_mode_index =
          problem_indices_ptr[outer_offset + reduce_min_index];

      // index:                0 1 2 3 [4] 5 6 7
      // sorted values:        0 0 1 1 [1] 3 4 4
      // find out the most-appeared value, it is 1
      auto answer_mode_value =
          problem_values_ptr[outer_offset + reduce_min_index];

      // write back
      auto output_index =
          IndexToOffset<scalar_t, int64_t>::get(outer_id, answer_values);
      answer_values.data[output_index] = answer_mode_value;
      answer_indices.data[output_index] = answer_mode_index;
    }
    item.barrier(dpcpp_global_fence);
  }
}

template <
    typename scalar_t,
    typename value_info_t,
    typename indice_info_t,
    typename item_t>
void mode_fused_impl(
    const scalar_t* problem_values_ptr,
    value_info_t answer_values,
    indice_info_t answer_indices,
    ModeOpHelper* slm_helper_ptr,
    ModeOpValueIndex<scalar_t>* slm_value_indice_ptr,
    std::byte* sort_scratch_pointer,
    const int64_t sort_scratch_memory_size,
    const int64_t problem_time,
    const int64_t problem_size,
    const int64_t wg_number,
    const int64_t wg_size,
    const item_t& item) {
  // read problem values into slm of the group
  auto group_id = item.get_group_linear_id();
  auto item_id = item.get_local_id(0);

  // outer loop, problem time level
  for (auto outer_id = group_id; outer_id < problem_time;
       outer_id += wg_number) {
    auto global_index = outer_id * problem_size + item_id;

    // load values and record indices into slm
    // slm value   1 4 3 4 0 1 0 1
    // slm indices 0 1 2 3 4 5 6 7
    slm_value_indice_ptr[item_id].value = problem_values_ptr[global_index];
    slm_value_indice_ptr[item_id].index = item_id;
    item.barrier(dpcpp_local_fence);

    // sort
    slm_value_indice_ptr[item_id] =
        sycl::ext::oneapi::experimental::sort_over_group(
            sycl::ext::oneapi::experimental::group_with_scratchpad(
                item.get_group(),
                sycl::span{sort_scratch_pointer, sort_scratch_memory_size}),
            slm_value_indice_ptr[item_id],
            [&](const ModeOpValueIndex<scalar_t>& A,
                const ModeOpValueIndex<scalar_t>& B) {
              return A.value < B.value;
            });
    item.barrier(dpcpp_local_fence);

    // compare and compute the status/value using 0/1.
    // sorted values:        0 0 1 1 1 3 4 4
    // sorted indices:       4 6 0 5 7 2 1 3
    // slm helper status:    0 1 0 1 1 0 0 1
    // slm helper value:     0 1 0 1 1 0 0 1
    // 0 means a new sequence. 1 means the value is duplicated with the pre one.
    // kick out the first one
    if (item_id == 0) {
      slm_helper_ptr[item_id].status = 0;
      slm_helper_ptr[item_id].value = 0;
    } else {
      auto judgeEqual = bool(
          slm_value_indice_ptr[item_id - 1].value ==
          slm_value_indice_ptr[item_id].value);
      slm_helper_ptr[item_id].status = (judgeEqual) ? (1) : (0);
      slm_helper_ptr[item_id].value = slm_helper_ptr[item_id].status;
    }
    item.barrier(dpcpp_local_fence);

    // index:                  0 1 2 3 4 5 6 7
    // slm helper status:      0 1 0 1 1 0 0 1
    // slm helper value:       0 1 0 1 1 0 0 1
    // conditional scan rule:
    // 1. according to current status, if 0, keep the current value. If 1, do
    // accumulation
    // 2. new status = previous scratch status & current scratch status to
    // record if there is duplicated value.
    // The conditional inclusive scan result value should be: 0 1 0 1 2 0 0 1
    ConditionalInclusiveScanForFusedMode(
        slm_helper_ptr,
        [&](const ModeOpHelper& Pre, const ModeOpHelper& Cur) {
          ModeOpHelper Temp;
          Temp.value = Cur.status ? (Pre.value + Cur.value) : (Cur.value);
          Temp.status = Pre.status & Cur.status;
          return Temp;
        },
        problem_size,
        item);
    item.barrier(dpcpp_local_fence);

    // truncate the status with value and use it for reduce
    // status: 0 1 0 1 2 0 0 1
    // value:  0 1 0 1 2 0 0 1
    // [watch out] status is used for following reduce and now has totally same
    // number with value
    slm_helper_ptr[item_id].status = slm_helper_ptr[item_id].value;
    item.barrier(dpcpp_local_fence);

    // index:                             0 1 2 3 [4] 5 6 7
    // conditional inclusive scan result: 0 1 0 1  2  0 0 1
    // [watch out] reduce:                         ^ <- reduce to get index 4
    // reduce_min_index means this position contains the first-appeared and
    // most-appeared element's index.
    ReduceGetMaxElemIndexForFusedMode(slm_helper_ptr, wg_size, item);
    item.barrier(dpcpp_local_fence);

    // only one elem operation
    if (item_id == 0) {
      auto reduce_min_index = slm_helper_ptr[0].value;

      // index:                0 1 2 3 [4] 5 6 7
      // sorted indices:       4 6 0 5 [7] 2 1 3
      // according to the reduce_min_index, find out the first-appeared and
      // most-appeared element's original indices, is 7
      auto answer_mode_index = slm_value_indice_ptr[reduce_min_index].index;

      // index:                0 1 2 3 [4] 5 6 7
      // sorted values:        0 0 1 1 [1] 3 4 4
      // find out the most-appeared value is 1
      auto answer_mode_value = slm_value_indice_ptr[reduce_min_index].value;

      // write back
      auto output_index =
          IndexToOffset<scalar_t, int64_t>::get(outer_id, answer_values);
      answer_values.data[output_index] = answer_mode_value;
      answer_indices.data[output_index] = answer_mode_index;
    }
    item.barrier(dpcpp_local_fence);
  }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
The answer rule of the cornor condition is:
1. indice need to be the max indice of the most-appeared value
2. if values appear same times, the returned value should be the smaller one

The implementation idea overview:
1. sort the input values
2. compare the adjecent value and record the status when checking equality
3. conditional scan to calculate the appear times for each kind of value
4. reduce to get the most-appeared value's most appear times
5. reduce again to get the most-appeared value's minimal indice
6. get the answer from the sorted value/indice according to the second reduce
result
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename scalar_t>
static void mode_xpu_kernel(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto& queue = dpcppGetCurrentQueue();
  RECORD_FUNCTION(
      "torch_ipex::mode_xpu_kernel", c10::ArrayRef<c10::IValue>({}));

  if (!keepdim) {
    if (values.ndimension() >= dim) {
      values.unsqueeze_(dim);
    }
    if (indices.ndimension() >= dim) {
      indices.unsqueeze_(dim);
    }
  }

  auto self_sizes = self.sizes().vec();
  // make sure the passed dim does make sense
  TORCH_CHECK(
      (0 <= dim && static_cast<size_t>(dim) < self_sizes.size()),
      "The chosen dim should be between [0, ",
      self_sizes.size() - 1,
      "], but got unexpected ",
      dim);

  auto ndim = self.dim();

  // problem size, the element size of the tensor at this dim
  auto problem_size = self_sizes[dim];
  // calculation times needed for each problem
  auto problem_time = self.numel() / problem_size;

  // problem dim suqeeze to 1
  self_sizes[dim] = 1;

  // Resize output value, index Tensors to sizes after execution
  resize_output(values, self_sizes);
  resize_output(indices, self_sizes);

  // If sliceSize is 1, it means the chosen dim has one value,
  // then copy input to values and set indices to 0
  if (problem_size == 1) {
    values.copy_(self);
    indices.fill_(0);
    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze_(dim);
    }
    return;
  }

  // exchange the problem dim to the last dim for mem coalescing
  auto transposed = self.transpose(dim, ndim - 1);
  auto contiguous = transposed.contiguous();
  auto values_transposed = values.transpose(dim, ndim - 1);
  auto indices_transposed = indices.transpose(dim, ndim - 1);

  // max wg size
  auto max_WG_Size =
      queue.get_device().template get_info<dpcpp_dev_max_work_group_size>();

  // one wg is responsible for one problem batch
  auto group_number = problem_time;

  // When the problem size is larger than the max wg size,
  // the wg is set the upper limitation of a wg size
  if (problem_size > max_WG_Size) {
    auto group_size = max_WG_Size;

    // sorted values and associated indices
    auto sort_tuple_ret =
        at::sort(contiguous, /*stable*/ true, /*dim*/ -1, /*descending*/ false);
    auto problem_values = std::get<0>(sort_tuple_ret);
    auto problem_indices = std::get<1>(sort_tuple_ret);

    auto scratch_status_tensor =
        at::zeros_like(self, TensorOptions(ScalarType::Long));
    auto scratch_value_tensor =
        at::zeros_like(self, TensorOptions(ScalarType::Long));

    auto values_info = getTensorInfo<scalar_t, int64_t>(values_transposed);
    auto indices_info = getTensorInfo<int64_t, int64_t>(indices_transposed);

    // be used to set the limitation for the inner loop wg
    auto problem_upper_limit = ((problem_size % group_size) == 0)
        ? (problem_size)
        : ((problem_size / group_size + 1) * group_size);

    auto cgf = DPCPP_Q_CGF(cgh) {
      // SLM(group size) is used for adjecent element comparing
      dpcpp_local_acc_t<scalar_t, 1> slm(group_size, cgh);
      auto problem_values_ptr = problem_values.data_ptr<scalar_t>();
      auto problem_indices_ptr = problem_indices.data_ptr<int64_t>();
      auto scratch_status_ptr = scratch_status_tensor.data_ptr<int64_t>();
      auto scratch_value_ptr = scratch_value_tensor.data_ptr<int64_t>();
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        mode_impl(
            problem_values_ptr,
            problem_indices_ptr,
            values_info,
            indices_info,
            (scalar_t*)(IPEXGetLocalAccPointer(slm)),
            scratch_status_ptr,
            scratch_value_ptr,
            problem_time,
            problem_size,
            group_number,
            group_size,
            problem_upper_limit,
            item);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(group_number * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(queue, cgf);
  } else {
    // problem_size <= max_WG_Size, wg size is set the problem size
    auto group_size = problem_size;

    // scratch memory size needed by built-in sort
    auto sort_scratch_memory_size = sycl::ext::oneapi::experimental::
        default_sorter<std::greater<scalar_t>>::template memory_required<
            ModeOpValueIndex<scalar_t>>(
            sycl::memory_scope::work_group, sycl::range<1>{group_size});

    auto values_info = getTensorInfo<scalar_t, int64_t>(values_transposed);
    auto indices_info = getTensorInfo<int64_t, int64_t>(indices_transposed);

    auto cgf = DPCPP_Q_CGF(cgh) {
      // SLM used for record status for mode
      dpcpp_local_acc_t<ModeOpHelper, 1> slm_helper(group_size, cgh);

      // SLM used for store value and its associated indice
      dpcpp_local_acc_t<ModeOpValueIndex<scalar_t>, 1> slm_value_indice(
          group_size, cgh);

      // SLM used for sort
      dpcpp_local_acc_t<std::byte, 1> sort_scratch(
          sort_scratch_memory_size, cgh);
      auto problem_values_ptr = contiguous.data_ptr<scalar_t>();
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        mode_fused_impl(
            problem_values_ptr,
            values_info,
            indices_info,
            (ModeOpHelper*)(IPEXGetLocalAccPointer(slm_helper)),
            (ModeOpValueIndex<scalar_t>*)(IPEXGetLocalAccPointer(
                slm_value_indice)),
            (std::byte*)(IPEXGetLocalAccPointer(sort_scratch)),
            sort_scratch_memory_size,
            problem_time,
            problem_size,
            group_number,
            group_size,
            item);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(group_number * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(queue, cgf);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}
} // namespace impl

std::tuple<Tensor&, Tensor&> mode_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "mode only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      self.device() == values.device(),
      "expected device '",
      self.device(),
      "' but got '",
      values.device(),
      "' for values output");
  TORCH_CHECK(
      self.device() == indices.device(),
      "expected device '",
      self.device(),
      "' but got '",
      indices.device(),
      "' for indices output");
  TORCH_CHECK(
      self.scalar_type() == values.scalar_type(),
      "expected scalar type '",
      self.scalar_type(),
      "' but got '",
      values.scalar_type(),
      "' for values output");
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "expected scalar type '",
      ScalarType::Long,
      "' but got '",
      indices.scalar_type(),
      "' for indices output");

  dim = maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    auto sizes = get_zero_numel_tensor_size(self, dim, keepdim, "mode()");
    resize_output(values, sizes);
    resize_output(indices, sizes);
    return std::tie(values, indices);
  } else if (at::native::_dimreduce_return_trivial_no_ident(
                 values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Bool,
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "mode_xpu",
        [&]() {
          impl::mode_xpu_kernel<scalar_t>(values, indices, self, dim, keepdim);
        });
    auto result = std::tuple<Tensor&, Tensor&>{values, indices};
    namedinference::propagate_names_for_reduction(
        std::get<0>(result), self, dim, keepdim);
    namedinference::propagate_names_for_reduction(
        std::get<1>(result), self, dim, keepdim);
    return result;
  }
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return mode_out(self, dim, keepdim, values, indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
