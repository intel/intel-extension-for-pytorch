#include "aten_ipex_bridge.h"

#include <map>
#include <string>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/UniqueVoidPtr.h>

#include "ipex_tensor_impl.h"
#include "ipex_sparse_tensor_impl.h"
#include "cpu/dbl/Common.h"
#include "cpu/ShadeDataContext.h"
#include "cpu/bf16/Converter.h"
#include "utils.h"

namespace torch_ipex {
namespace bridge {

#if defined(_DEBUG)
#define CHECK_TENSOR(a, b) \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.numel() == b.numel()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.dtype() == b.dtype()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->sizes() == b.unsafeGetTensorImpl()->sizes()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->dtype() == b.unsafeGetTensorImpl()->dtype()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->is_contiguous() == b.unsafeGetTensorImpl()->is_contiguous()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->is_contiguous(at::MemoryFormat::ChannelsLast) == b.unsafeGetTensorImpl()->is_contiguous(at::MemoryFormat::ChannelsLast)); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->is_strides_like_channels_last() == b.unsafeGetTensorImpl()->is_strides_like_channels_last()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->is_non_overlapping_and_dense() == b.unsafeGetTensorImpl()->is_non_overlapping_and_dense()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->is_wrapped_number() == b.unsafeGetTensorImpl()->is_wrapped_number()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->version_counter().current_version() == b.unsafeGetTensorImpl()->version_counter().current_version()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->allow_tensor_metadata_change() == b.unsafeGetTensorImpl()->allow_tensor_metadata_change())
#else
#define CHECK_TENSOR(a, b) ((void) 0)
#endif

#if defined(_DEBUG)
#define CHECK_TENSOR_CRITICAL(a, b, may_alias) \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!may_alias || a.data_ptr() == b.data_ptr()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->strides() == b.unsafeGetTensorImpl()->strides()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.unsafeGetTensorImpl()->storage_offset() == b.unsafeGetTensorImpl()->storage_offset()); \
  CHECK_TENSOR(a, b)
#else
#define CHECK_TENSOR_CRITICAL(a, b, may_alias) ((void) 0)
#endif

#if defined(_DEBUG)
#define CHECK_SPARSE_TENSOR_CRITICAL(a, b, may_alias) \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!may_alias || a._indices().data_ptr() == b._indices().data_ptr()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!may_alias || a._values().data_ptr() == b._values().data_ptr()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.sparse_dim() == b.sparse_dim()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.dense_dim() == b.dense_dim()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a._nnz() == b._nnz()); \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.is_coalesced() == b.is_coalesced()); \
  CHECK_TENSOR(a._indices(), b._indices()); \
  CHECK_TENSOR(a._values(), b._values())
#else
#define CHECK_SPARSE_TENSOR_CRITICAL(a, b, may_alias) ((void) 0)
#endif

at::Tensor shallowFallbackToCPUTensorImpl(const at::Tensor& ipexTensor);

void reorderDilTensorToPublic(const at::Tensor& ipexTensor) {
  void *data_ctx = ipexTensor.unsafeGetTensorImpl()->storage().data_ptr().get_context();
  cpu::ShadeDataContext *shade_data_context = (cpu::ShadeDataContext*)data_ctx;
#if defined(_DEBUG)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! (shade_data_context->dil_tensor->is_empty()));
#endif
  dil::tensor &dil_tensor = *shade_data_context->dil_tensor;

  if (dil_tensor.is_public_format()) {
#if defined(_DEBUG)
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_raw_data == shade_data_context->dil_tensor->get_data_handle());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_raw_data != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_del_fun != nullptr);
#endif
  } else {
#if defined(_DEBUG)
    auto& data_ptr = ipexTensor.storage().unsafeGetStorageImpl()->data_ptr();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data_ptr.get_deleter() == &(cpu::ShadeDataContext::freeShadeDataContext));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_del_fun == nullptr);
#endif
    auto pub_tensor = dil_tensor.to_public(nullptr, dil_tensor.get_data_type());

    cpu::ShadeDataContext *new_shade_data_context = cpu::ShadeDataContext::allocShadeDataContext();
    new_shade_data_context->data_type = cpu::SHADE_DATA_TYPE::DIL;
    new_shade_data_context->dil_tensor = pub_tensor;
    // Share with DNNL raw data because it is plain format now
    new_shade_data_context->cpu_raw_data = pub_tensor.get_data_handle();
    // Cannot free CPU data because the the data is owned by DNNL
    new_shade_data_context->cpu_del_fun = &(c10::detail::deleteNothing);

    // Create a new DataPtr instances because the DataPtr class does not support set
    // its data or context directly
    c10::DataPtr shade_data_ptr(
      pub_tensor.get_data_handle(),
      new_shade_data_context,
      &(cpu::ShadeDataContext::freeShadeDataContext),
      ipexTensor.device().type());

    ipexTensor.unsafeGetTensorImpl()->storage().set_data_ptr(std::move(shade_data_ptr));
    cpu::dbl::comm::sync_shape_from_dil_to_aten(ipexTensor, pub_tensor);
  }
}


void attachShadeDataContext(const at::Tensor& tensor) {
  auto tensor_storage_impl = tensor.storage().unsafeGetStorageImpl();
  auto& data_ptr = tensor_storage_impl->data_ptr();

  // Has contained shade context
  if (check_tensor_own_shade_context(tensor))
    return;

  auto cur_del_fn = data_ptr.get_deleter();
  bool res = data_ptr.compare_exchange_deleter(cur_del_fn, &(c10::detail::deleteNothing));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(res);
  // Make sure that does not triger free resource for set_ptr
  cpu::ShadeDataContext *shade_data_context = cpu::ShadeDataContext::allocShadeDataContext();
  shade_data_context->cpu_raw_data = data_ptr.get();
  shade_data_context->cpu_del_fun = cur_del_fn;
  shade_data_context->data_type = cpu::SHADE_DATA_TYPE::CPU_RAW;
  c10::DataPtr shade_data_ptr(
    data_ptr.get(),
    shade_data_context,
    cpu::ShadeDataContext::freeShadeDataContext,
    tensor.device().type());
  tensor.unsafeGetTensorImpl()->storage().set_data_ptr(std::move(shade_data_ptr));
}


// Unpack CPU tensor from ipex tensor and return to caller directly
//at::Tensor shallowFallbackToCPUShadeTensor(const at::Tensor& ipexTensor) {
at::Tensor shallowFallbackToCPUTensor(const at::Tensor& ipexTensor) {
  if (!(ipexTensor.defined())) {
    return ipexTensor;
  }

  if (ipexTensor.device().is_cpu()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! (ipexTensor.key_set().has(at::DispatchKey::DPCPPTensorId)));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! (ipexTensor.key_set().has(at::DispatchKey::SparseDPCPPTensorId)));
    return ipexTensor;
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.device().is_dpcpp());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
    ipexTensor.key_set().has(at::DispatchKey::DPCPPTensorId) ||
    ipexTensor.key_set().has(at::DispatchKey::SparseDPCPPTensorId));

  // Brnach 1: Sparse Tensor
  if (ipexTensor.is_sparse()) {
    return shallowFallbackToCPUTensorImpl(ipexTensor);
  }

  // Branch 2: Dense Tensor
  
  // Branch 2.0: Dense + CPU Tensor + w/o context.
  // Supposing only Aten inplace op w/ Resize internally will run into this branch,
  // since new DataPtr has replaced orignal one, then DPCPP tensor loses context info.
  // e.g. Sparse add_()
  void *data_ptr = ipexTensor.unsafeGetTensorImpl()->storage().data_ptr().get();
  void *data_ctx = ipexTensor.unsafeGetTensorImpl()->storage().data_ptr().get_context();
  if (data_ptr == data_ctx) {
    return shallowFallbackToCPUTensorImpl(ipexTensor);
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data_ctx != nullptr);
  cpu::ShadeDataContext *shade_data_context = (cpu::ShadeDataContext*)data_ctx;
  // Branch 2.1: Dense + Dil Tensor
  if (cpu::ShadeDataContext::isDilTensor(ipexTensor)) {
    reorderDilTensorToPublic(ipexTensor);
  }

  // Branch 2.2: Dense + CPU Tensor
  return shallowFallbackToCPUTensorImpl(ipexTensor);
}


// Fallback CPU tensor to DPCPP Tensor with shallow copy
// It will create an new CPU tensor but shares DPCPP tensor buffer
at::Tensor shallowFallbackToCPUTensorImpl(const at::Tensor& ipexTensor) {
  if (!(ipexTensor.defined())) {
    return ipexTensor;
  }

  if (ipexTensor.device().is_cpu()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! (ipexTensor.key_set().has(at::DispatchKey::DPCPPTensorId)));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! (ipexTensor.key_set().has(at::DispatchKey::SparseDPCPPTensorId)));
    return ipexTensor;
  }

  if (ipexTensor.is_sparse()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.layout() == c10::kSparse);
    // [NOTE]: Use _indices and _values interfaces to bypass non-coalesced check
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor._indices().layout() == c10::kStrided);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor._values().layout() == c10::kStrided);

    auto&& cpu_indices = shallowFallbackToCPUTensorImpl(ipexTensor._indices());
    auto&& cpu_values = shallowFallbackToCPUTensorImpl(ipexTensor._values());
    // Create cpu sparse tensor and copy meta data from ipex cpu sparse tensor
    auto _tensor = at::detail::make_tensor<IPEXSparseTensorImpl>(
      at::DispatchKeySet(at::DispatchKey::SparseCPUTensorId), ipexTensor.dtype());
    auto cpu_sparse_impl = IPEXSparseTensorImpl::get_ipex_sparse_impl(_tensor);
    auto ipex_sparse_impl = IPEXSparseTensorImpl::get_ipex_sparse_impl(ipexTensor);
    cpu_sparse_impl->copy_meta_info(ipex_sparse_impl);
    // Copy indices and values
    cpu_sparse_impl->copy_indices_and_values(cpu_indices, cpu_values);
    CHECK_SPARSE_TENSOR_CRITICAL(_tensor, ipexTensor, true);
    return _tensor;
  } else {
    auto *ipex_tensor_impl = ipexTensor.unsafeGetTensorImpl();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipex_tensor_impl != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipex_tensor_impl->has_storage());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.layout() == c10::kStrided);
    auto ipex_tensor_storage = ipex_tensor_impl->storage().unsafeGetStorageImpl();
    ipex_tensor_storage->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::CPU));
    auto _tensor = at::detail::make_tensor<IPEXTensorImpl>(ipexTensor.storage(), at::DispatchKey::CPUTensorId);
    IPEXTensorImpl* cur_ipex_impl = (IPEXTensorImpl *)_tensor.unsafeGetTensorImpl();
    cur_ipex_impl->copy_meta_info(ipexTensor.unsafeGetTensorImpl());
    cur_ipex_impl->copy_auto_grad(ipexTensor.unsafeGetTensorImpl());
    CHECK_TENSOR_CRITICAL(_tensor, ipexTensor, true);
    // TODO: Cannot reserved_
    //       dest_impl->reserved_ = src_impl->reserved_;
    return _tensor;
  }
}


// Upgrade CPU tensor to DPCPP Tensor with shallow copy
// It will create an new DPCPP tensor but shares CPU tensor buffer
// [NOTE]: Device info of Dense CPU tensor is polluted.
at::Tensor shallowUpgradeToDPCPPTensor(const at::Tensor& cpuTensor) {
  if (!(cpuTensor.defined())) {
    return at::Tensor();
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.device().type() == at::DeviceType::CPU);
  if (cpuTensor.is_sparse()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.layout() == c10::kSparse);
    // [NOTE]: Use _indices and _values interfaces to bypass non-coalesced check
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor._indices().layout() == c10::kStrided);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor._values().layout() == c10::kStrided);
    auto&& ipex_indices = shallowUpgradeToDPCPPTensor(cpuTensor._indices());
    auto&& ipex_values = shallowUpgradeToDPCPPTensor(cpuTensor._values());
    // Create ipex sparse tensor and copy meta data from cpu sparse tensor
    auto _tensor = at::detail::make_tensor<IPEXSparseTensorImpl>(
        at::DispatchKeySet(at::DispatchKey::SparseDPCPPTensorId), cpuTensor.dtype());
    auto ipex_sparse_impl = IPEXSparseTensorImpl::get_ipex_sparse_impl(_tensor);
    auto cpu_sparse_impl = at::sparse::get_sparse_impl(cpuTensor);
    ipex_sparse_impl->copy_meta_info(cpu_sparse_impl);
    // Copy indices and values
    ipex_sparse_impl->copy_indices_and_values(ipex_indices, ipex_values);
    CHECK_SPARSE_TENSOR_CRITICAL(_tensor, cpuTensor, true);
    return _tensor;
  } else {
    auto *cpu_tensor_impl = cpuTensor.unsafeGetTensorImpl();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpu_tensor_impl != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpu_tensor_impl->has_storage());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.layout() == c10::kStrided);

    auto cpu_storage = cpu_tensor_impl->storage().unsafeGetStorageImpl();
    // [NOTE]: If the deleter of DPCPP::CPU is different form CPU deleter, we need to call
    //         compare_exchange_deleter of DataPtr to update deleter
    cpu_storage->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));
    auto _tensor =  at::detail::make_tensor<IPEXTensorImpl>(cpuTensor.storage(), at::DispatchKey::DPCPPTensorId);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_tensor.device().type() == at::DeviceType::DPCPP);
    IPEXTensorImpl* ipex_impl = (IPEXTensorImpl *)_tensor.unsafeGetTensorImpl();
    ipex_impl->copy_meta_info(cpu_tensor_impl);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! cpuTensor.requires_grad());
    CHECK_TENSOR_CRITICAL(_tensor, cpuTensor, true);
    //TODO: Cannot set reserved_ 
    //      dest_impl->reserved_ = src_impl->reserved_;
    attachShadeDataContext(_tensor);
    return _tensor;
  }
}


at::Tensor shallowUpgradeToDPCPPTensorA(const at::Tensor& ipexTensor, const at::Tensor& cpuTensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!ipexTensor.is_sparse());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!cpuTensor.is_sparse());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.device().type() == at::DeviceType::DPCPP);

  ipexTensor.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.storage().device_type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.device().type() == at::DeviceType::CPU);

  // The ipexTensor and cpuTensor shares same storage.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.storage().device_type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.storage().data() == cpuTensor.storage().data());

  auto _tensor = at::detail::make_tensor<IPEXTensorImpl>(at::Storage(ipexTensor.storage()), at::DispatchKey::DPCPPTensorId);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_tensor.device().type() == at::DeviceType::DPCPP);
  IPEXTensorImpl* ipex_impl = (IPEXTensorImpl *)_tensor.unsafeGetTensorImpl();
  ipex_impl->copy_meta_info(cpuTensor.unsafeGetTensorImpl());
  ipex_impl->copy_auto_grad(cpuTensor.unsafeGetTensorImpl());
  CHECK_TENSOR_CRITICAL(_tensor, cpuTensor, true);

  attachShadeDataContext(_tensor);
  return _tensor;
}


// Upgrade CPU tensor to DPCPP Tensor with shallow copy
// It will not create an new DPCPP tensor but shares CPU tensor buffer
const at::Tensor& shallowUpgradeToDPCPPTensorAW(const at::Tensor& ipexTensor, const at::Tensor& cpuTensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.defined());

  // The dispatch priority of DPCPPTensorId is higher than other CPU tensor ids. So if a tensor is CPU and
  // another tensor is DPCPP, it still will be disptached to DPCPP OPs.
  //   ex, a = tensor(1, device='dpcpp')), a.to('cpu')
  // The above code will call AtenIpexCPUDefault::copy_ and "self" parameter is cpu tensor and "src" parameter is dpcpp tensor.
  if (ipexTensor.device().type() == cpuTensor.device().type()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.device().type() == at::DeviceType::CPU);
    return ipexTensor;
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.device().type() == at::DeviceType::DPCPP);

  if (ipexTensor.is_sparse()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.is_sparse());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.layout() == c10::kSparse);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.layout() == c10::kSparse);
    // NOTICE:
    // In PyTorch, alias semantics is `may alias`, not `must alias`.
    // e.g. some sparse 'alias marked' ops are not alias actually,
    // so following data_ptr will be different for those cases.
    // TODO: whether nnz checking is enough or exact.
    auto may_alias = ipexTensor._nnz() == cpuTensor._nnz();

    auto&& ipex_indices = ipexTensor._indices();
    auto&& ipex_values = ipexTensor._values();
    auto&& cpu_indices = cpuTensor._indices();
    auto&& cpu_values = cpuTensor._values();
    if (!may_alias) {
      ipex_indices = shallowUpgradeToDPCPPTensor(cpu_indices);
      ipex_values = shallowUpgradeToDPCPPTensor(cpu_values);
    } else {
      ipex_indices = shallowUpgradeToDPCPPTensorAW(ipex_indices, cpu_indices);
      ipex_values = shallowUpgradeToDPCPPTensorAW(ipex_values, cpu_values);
    }
    auto ipex_sparse_impl = IPEXSparseTensorImpl::get_ipex_sparse_impl(ipexTensor);
    auto cpu_sparse_impl = at::sparse::get_sparse_impl(cpuTensor);
    ipex_sparse_impl->copy_meta_info(cpu_sparse_impl);
    ipex_sparse_impl->copy_indices_and_values(ipex_indices, ipex_values);
    CHECK_SPARSE_TENSOR_CRITICAL(ipexTensor, cpuTensor, may_alias);
    return ipexTensor;
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!ipexTensor.is_sparse());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!cpuTensor.is_sparse());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.layout() == c10::kStrided);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpuTensor.layout() == c10::kStrided);

    auto ipex_tensor_storage_impl = ipexTensor.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
    auto cpu_tensor_storage_impl = cpuTensor.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

    // Some inplace OPs replace its storage but not modify its raw data. (ex. set_)
    if (ipex_tensor_storage_impl != cpu_tensor_storage_impl) {
      TORCH_WARN("An in-place OP implements its semantic by replace storage!");
      cpuTensor.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));
      ipexTensor.unsafeGetTensorImpl()->set_storage(cpuTensor.storage());
    }

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.data_ptr() == cpuTensor.data_ptr());

    // NOTE: Cannot set storage data_ptr by set_data_ptr.
    //       set_data_ptr will release caller tensor's original data_ptr. It is wrong here because
    //       the ipexTensor and cpuTensor share same buffer here.
    //
    // [Wrong code]:
    //   void* tensor_raw_data = cpuTensor.unsafeGetTensorImpl()->storage().data();
    //   c10::DataPtr dpcpp_data_ptr(tensor_raw_data, at::DeviceType::DPCPP);
    //   ipexTensor.storage().set_data_ptr(std::move(dpcpp_data_ptr));
    //

    ipexTensor.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));

    IPEXTensorImpl* ipex_tensor_impl = (IPEXTensorImpl *)ipexTensor.unsafeGetTensorImpl();
    ipex_tensor_impl->copy_meta_info(cpuTensor.unsafeGetTensorImpl());
    ipex_tensor_impl->copy_auto_grad(cpuTensor.unsafeGetTensorImpl());
    CHECK_TENSOR_CRITICAL(ipexTensor, cpuTensor, true);
    attachShadeDataContext(ipexTensor);
    return ipexTensor;
  }
}


std::vector<at::Tensor> shallowFallbackToCPUTensorList(const at::TensorList& tensor_list) {
  std::vector<at::Tensor> dpcpp_tensor_vec(tensor_list.size());
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const at::Tensor& tensor = tensor_list[i];
    if (tensor.defined()) {
      dpcpp_tensor_vec[i] = shallowFallbackToCPUTensorImpl(tensor);
    }
  }
  return dpcpp_tensor_vec;
}


void reorderTensorToScalarTypeForDNNL(const at::Tensor& ipexTensor, at::ScalarType dstScalarType) {
  TORCH_CHECK(dstScalarType == at::kBFloat16 || dstScalarType == at::kFloat);
  auto tensor_dtype = ipexTensor.scalar_type();
  if ((tensor_dtype != at::kBFloat16 && tensor_dtype != at::kFloat) || tensor_dtype == dstScalarType)
    return;

  if (check_tensor_own_shade_context(ipexTensor)) {
    // Shade data context has been attached
    if (cpu::ShadeDataContext::isDilTensor(ipexTensor)) {
      cpu::ShadeDataContext *shade_context = (cpu::ShadeDataContext*)(ipexTensor.storage().data_ptr().get_context());
      shade_context->dil_tensor->to_type(get_dil_data_type(dstScalarType));
      IPEXTensorImpl* ipex_tensor_impl = (IPEXTensorImpl *)ipexTensor.unsafeGetTensorImpl();
      ipex_tensor_impl->reset_data_type(dstScalarType);
      ipex_tensor_impl->storage().unsafeGetStorageImpl()->set_dtype(at::scalarTypeToTypeMeta(dstScalarType));
      ipex_tensor_impl->storage().unsafeGetStorageImpl()->set_numel(shade_context->dil_tensor->get_nelems());
      return;
    }
  }

  return reorderTensorToScalaraType(ipexTensor, dstScalarType);
}


void reorderTensorToScalaraType(const at::Tensor& ipexTensor, at::ScalarType dstScalarType) {
  if (!(ipexTensor.defined()))
    return;

  auto tensor_dtype = ipexTensor.scalar_type();
  if ((tensor_dtype != at::kBFloat16 && tensor_dtype != at::kFloat) || tensor_dtype == dstScalarType)
    return;

  if (ipexTensor.is_sparse()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipexTensor.layout() == c10::kSparse);
    auto&& ipex_values = ipexTensor._values();
    reorderTensorToScalaraType(ipex_values, dstScalarType);
  }

  if (!check_tensor_own_whole_storage(ipexTensor)) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return;
  }

  if (check_tensor_own_shade_context(ipexTensor)) {
    // Shade data context has been attached
    if (cpu::ShadeDataContext::isDilTensor(ipexTensor)) {
      reorderDilTensorToPublic(ipexTensor);
    }
  }

  auto* allocator = c10::GetAllocator(c10::DeviceType::DPCPP);
  int64_t nelements = ipexTensor.numel();
  auto dtype = c10::scalarTypeToTypeMeta(dstScalarType);
  int64_t data_size = nelements * c10::elementSize(dstScalarType);
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(data_size),
    allocator,
    /*resizeable=*/true);

  void *data_ptr = ipexTensor.unsafeGetTensorImpl()->storage().data_ptr().get();
  if (dstScalarType == at::kBFloat16) {
    torch_ipex::cpu::bf16::converter::fp32_to_bf16(storage_impl->data_ptr().get(), data_ptr, nelements);
  } else {
    torch_ipex::cpu::bf16::converter::bf16_to_fp32(storage_impl->data_ptr().get(), data_ptr, nelements);
  }

  ipexTensor.unsafeGetTensorImpl()->set_storage(storage_impl);
  attachShadeDataContext(ipexTensor);
}


std::vector<at::Tensor> shallowUpgradeToDPCPPTensorVec(const std::vector<at::Tensor> &tensor_vec) {
  std::vector<at::Tensor> ret_dpcpp_tensor_vec;
  for (size_t i = 0; i < tensor_vec.size(); i++) {
    auto&& cur_tensor = tensor_vec[i];
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cur_tensor.defined());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cur_tensor.layout() == c10::kStrided);
    auto&& cur_dpcpp_tensor = shallowUpgradeToDPCPPTensor(cur_tensor);
    ret_dpcpp_tensor_vec.push_back(cur_dpcpp_tensor);
  }
  return ret_dpcpp_tensor_vec;
}

}  // namespace bridge
}  // namespace torch_ipex
