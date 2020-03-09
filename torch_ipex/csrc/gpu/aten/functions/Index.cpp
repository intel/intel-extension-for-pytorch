#include <ATen/ATen.h>

#include <core/SYCL.h>
#include <core/SYCLStream.h>
#include <core/SYCLMemory.h>
#include <core/detail/TensorInfo.h>


using namespace at::sycl::detail;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DP_DEF_K1(index_select_ker);
template <typename scalar_t>
void indexSelect(Tensor & dst, const Tensor & src, int dim, const Tensor & indices) {
  int srcDims = src.dim() == 0 ? 1 : src.dim();
  int dstDims = dst.dim() == 0 ? 1 : dst.dim();
  int idxDims = indices.dim() == 0 ? 1 : indices.dim();

  TORCH_CHECK(srcDims <= MAX_SYCLTORCH_DIMS, SYCLTORCH_DIM_WARNING);
  TORCH_CHECK(dstDims <= MAX_SYCLTORCH_DIMS, SYCLTORCH_DIM_WARNING);
  TORCH_CHECK(idxDims <= MAX_SYCLTORCH_DIMS, SYCLTORCH_DIM_WARNING);
  TORCH_CHECK(idxDims <= 1,
           "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(idxDims < srcDims, "Indexing dim is out of bounds");
  TORCH_CHECK(srcDims > 0, "Source tensor is empty");

  TensorInfo<int64_t, unsigned int> indices_info =
    getTensorInfo<int64_t, unsigned int>(indices);
  indices_info.collapseDims();

  auto new_size = src.sizes().vec();
  new_size[dim] = indices_info.sizes[0];
  dst.resize_(new_size);

  ptrdiff_t dst_num_elem = dst.numel();
  if (dst_num_elem == 0) {
    return;
  }

  TensorInfo<scalar_t, unsigned int> dst_info =
      getTensorInfo<scalar_t, unsigned int>(dst);
  int dst_select_dim = dst_info.collapseDims(dim);
  dst_info.reduceDim(dst_select_dim);

  TensorInfo<scalar_t, unsigned int> src_info =
      getTensorInfo<scalar_t, unsigned int>(src);
  int src_select_dim = src_info.collapseDims(dim);
  src_info.reduceDim(src_select_dim);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  // TODO: if the slice number is to large. Need to balance the work group and work item number.
  // Make the work balance based on the MCU number.
  // auto __mcu = sycl_queue.get_device().template get_info<dp_dev_max_units>();
  uint64_t num_slices = indices.numel();

  auto slice_size = dst_num_elem / num_slices;

  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

  auto wgroup_size = sycl_queue.get_device().template \
          get_info<dp_dev_max_wgroup_size>();

  wgroup_size = std::min(decltype(wgroup_size)(slice_size), wgroup_size);

  auto n_work_item_iter = (slice_size + wgroup_size - 1) / wgroup_size;

  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();
  auto idx_data = indices.data_ptr<int64_t>();
  auto src_size = src.nbytes();
  auto dst_size = dst.nbytes();
  auto idx_size = dst.nbytes();

  auto cgf = DP_Q_CGF(__cgh) {
    auto src_acc = c10::sycl::SYCLAccessor<dp_r_mode>(__cgh, src_data, src_size);
    auto dst_acc = c10::sycl::SYCLAccessor<dp_discard_w_mode>(__cgh, dst_data, dst_size);
    auto idx_acc = c10::sycl::SYCLAccessor<dp_r_mode>(__cgh, idx_data, idx_size);

    __cgh.parallel_for_work_group<DP_K(index_select_ker, scalar_t)>(
      DP::range</*dim=*/1>(num_slices),
      DP::range</*dim=*/1>(wgroup_size),
      [=](DP::group<1> group_id) {
        auto src_ptr = src_acc.template get_pointer<scalar_t>();
        auto dst_ptr = dst_acc.template get_pointer<scalar_t>();
        auto idx_ptr = idx_acc.template get_pointer<long>();

        auto dst_slice_id = group_id.get_id()[0];

        auto slice_off = IndexToOffset<int64_t, unsigned int>::get(dst_slice_id, indices_info);
        auto src_slice_id = idx_ptr[slice_off]/* - TH_INDEX_BASE*/;

        auto g_src_ptr = src_ptr + src_slice_id * src_info.strides[src_select_dim];
        auto g_dst_ptr = dst_ptr + dst_slice_id * dst_info.strides[dst_select_dim];

        group_id.parallel_for_work_item([=](DP::h_item<1> item_id) {

          auto ii_ = item_id.get_logical_local_id()[0];
          auto src_offset_ =
                  IndexToOffset<scalar_t, unsigned int>::get(ii_, src_info);
          auto dst_offset_ =
                  IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);

          g_dst_ptr[ dst_offset_ ] = g_src_ptr[ src_offset_ ];

          for (int iter = 1; iter < n_work_item_iter;iter++) {
            auto __inner_idx = iter * wgroup_size + ii_;
            if (__inner_idx < slice_size) {
              src_offset_ = IndexToOffset<scalar_t, unsigned int>::get(__inner_idx, src_info);
              dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(__inner_idx, dst_info);

              g_dst_ptr[ dst_offset_ ] = g_src_ptr[ src_offset_ ];
            }
          }
        });
      }
    );
  };

  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
  return;
}
} // namespace impl

Tensor & index_select_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "indexSelect", [&]() {
    impl::indexSelect<scalar_t>(out, self, dim, index);
  });
  return out;
}

Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) {
  auto out = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::index_select_out(out, self, dim, index);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
