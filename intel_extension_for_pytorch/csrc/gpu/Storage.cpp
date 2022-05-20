#include <torch/csrc/THP.h>
#include <torch/csrc/utils.h>

#include <c10/core/ScalarType.h>
#include <core/Allocator.h>
#include <core/Memory.h>
#include <core/Stream.h>
#include <intrinsic/intrinsic.h>
#include <structmember.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/copy_utils.h>

template <>
struct THPUtils_typeTraits<c10::quint8> {
  static constexpr const char* python_type_str = "int";
};

template <>
struct THPUtils_typeTraits<c10::qint8> {
  static constexpr const char* python_type_str = "int";
};

template <typename scalar_t>
scalar_t THPUtils_unpackReal(PyObject*) {
  throw std::runtime_error("Could not parse Scalar");
}

template <typename scalar_t>
PyObject* THPUtils_newReal(scalar_t) {
  throw std::runtime_error("Could not new Scalar");
}

template <typename scalar_t>
bool THPUtils_checkReal(PyObject*) {
  throw std::runtime_error("Could not check Scalar");
}

#define DECLEAR_UNPACK(s, n) \
  template <>                \
  s THPUtils_unpackReal<s>(PyObject * obj);
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DECLEAR_UNPACK)
#undef DECLEAR_UNPACK

#define DECLEAR_NEW(s, n) \
  template <>             \
  PyObject* THPUtils_newReal<s>(s value);
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DECLEAR_NEW)
#undef DECLEAR_NEW

#define DECLEAR_CHECK(s, n) \
  template <>               \
  bool THPUtils_checkReal<s>(PyObject*);
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DECLEAR_CHECK)
#undef DECLEAR_CHECK

struct THXP_Storage {
  PyObject_HEAD at::StorageImpl* cdata;
};

using THXPStorage_Ptr = THPPointer<THXP_Storage>;
using THStorageImpl_Ptr = THPPointer<at::StorageImpl>;

template <at::ScalarType scalarType>
class THXPStorage_Bridge {
 public:
  using scalar_t = typename c10::impl::ScalarTypeToCPPType<scalarType>::type;
  static constexpr auto LocalDispatchKey = at::DispatchKey::XPU;
  THXPStorage_Bridge(PyObject* module, PyTypeObject* type) {}

 public:
#define TH_XPU_STORAGE_IMPLEMENT_COPYTO(TYPEC)        \
  void TH_CONCAT_4(TH, TYPEC, Storage_copyXPU, Real)( \
      TH##TYPEC##Storage * self, struct THXStorage * src) {}

  template <at::ScalarType cpuType>
  static void Storage_copyToCPU(
      at::StorageImpl* cpu,
      at::StorageImpl* xpu_src) {
    using src_scalar_t = typename c10::impl::ScalarTypeToCPPType<cpuType>::type;
    at::TensorImpl* selfTensor = THTensor_newWithStorage1d<cpuType>(
        cpu, 0, cpu->nbytes() / sizeof(scalar_t), 1, at::DispatchKey::CPU);
    at::TensorImpl* srcTensor = THTensor_newWithStorage1d<scalarType>(
        xpu_src,
        0,
        xpu_src->nbytes() / sizeof(src_scalar_t),
        1,
        at::DispatchKey::XPU);
    THTensor_copy(selfTensor, srcTensor);
    THTensor_free(selfTensor);
    THTensor_free(srcTensor);
  }

  template <>
  static void Storage_copyToCPU<scalarType>(
      at::StorageImpl* cpu,
      at::StorageImpl* xpu_src) {
    THArgCheck(cpu->nbytes() == xpu_src->nbytes(), 2, "size does not match");
    xpu::dpcpp::dpcppMemcpy(
        cpu->data(),
        xpu_src->data(),
        cpu->nbytes(),
        xpu::dpcpp::dpcppMemcpyKind::DeviceToHost);
  }

  static void THTensor_setStorage(
      at::TensorImpl* self,
      at::StorageImpl* storage_,
      ptrdiff_t storageOffset_,
      at::IntArrayRef size_,
      at::IntArrayRef stride_) {
    c10::raw::intrusive_ptr::incref(storage_);
    //    THTensor_wrap(self).set_(
    //        at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage_)),
    //        storageOffset_,
    //        size_,
    //        stride_);
  }

  template <at::ScalarType T>
  static at::TensorImpl* THTensor_newWithStorage1d(
      at::StorageImpl* storage,
      ptrdiff_t storageOffset,
      int64_t size0,
      int64_t stride0,
      at::DispatchKey key) {
    using scalar_t = typename c10::impl::ScalarTypeToCPPType<T>::type;
    c10::raw::intrusive_ptr::incref(storage);
    at::TensorImpl* self =
        c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
            c10::intrusive_ptr<at::StorageImpl>::reclaim(storage),
            key,
            caffe2::TypeMeta::Make<scalar_t>())
            .release();
    THTensor_setStorage(self, storage, storageOffset, {size0}, {stride0});

    return self;
  }

  static void THTensor_copy(at::TensorImpl* dst, at::TensorImpl* src) {
    if (dst == src)
      return;
    at::Tensor dst_wrap = THTensor_wrap(dst);
    at::Tensor src_wrap = THTensor_wrap(src);
    xpu::dpcpp::direct_copy(dst_wrap, src_wrap, false);
  }

  template <at::ScalarType cpuType>
  static void Storage_copyFromCPU(
      at::StorageImpl* xpu,
      at::StorageImpl* cpu_src) {
    using src_scalar_t = typename c10::impl::ScalarTypeToCPPType<cpuType>::type;
    at::TensorImpl* selfTensor = THTensor_newWithStorage1d<scalarType>(
        xpu, 0, xpu->nbytes() / sizeof(scalar_t), 1, at::DispatchKey::XPU);
    at::TensorImpl* srcTensor = THTensor_newWithStorage1d<cpuType>(
        cpu_src,
        0,
        cpu_src->nbytes() / sizeof(src_scalar_t),
        1,
        at::DispatchKey::CPU);
    THTensor_copy(selfTensor, srcTensor);
    THTensor_free(selfTensor);
    THTensor_free(srcTensor);
  }

  template <>
  static void Storage_copyFromCPU<scalarType>(
      at::StorageImpl* xpu,
      at::StorageImpl* cpu_src) {
    //    THArgCheck(xpu->nbytes() == cpu_src->nbytes(), 2, "size does not
    //    match");
    xpu::dpcpp::dpcppMemcpy(
        xpu->data(),
        cpu_src->data(),
        xpu->nbytes(),
        xpu::dpcpp::dpcppMemcpyKind::HostToDevice);
  }

  template <at::ScalarType srcType>
  static void Storage_copyFromXPU(
      at::StorageImpl* xpu,
      at::StorageImpl* xpu_src) {
    using src_scalar_t = typename c10::impl::ScalarTypeToCPPType<srcType>::type;
    at::TensorImpl* selfTensor = THTensor_newWithStorage1d<scalarType>(
        xpu, 0, xpu->nbytes() / sizeof(scalar_t), 1, at::DispatchKey::XPU);
    at::TensorImpl* srcTensor = THTensor_newWithStorage1d<srcType>(
        xpu_src,
        0,
        xpu_src->nbytes() / sizeof(src_scalar_t),
        1,
        at::DispatchKey::XPU);
    THTensor_copy(selfTensor, srcTensor);
    THTensor_free(selfTensor);
    THTensor_free(srcTensor);
  }

  template <>
  static void Storage_copyFromXPU<scalarType>(
      at::StorageImpl* xpu,
      at::StorageImpl* xpu_src) {
    //    THArgCheck(xpu->nbytes() == xpu_src->nbytes(), 2, "size does not
    //    match");
    xpu::dpcpp::dpcppMemcpy(
        xpu->data(),
        xpu_src->data(),
        xpu->nbytes(),
        xpu::dpcpp::dpcppMemcpyKind::DeviceToDevice);
  }

  static void Storage_retain(at::StorageImpl* self) {
    THStorage_retain(self);
  }

  static void Storage_free(at::StorageImpl* self) {
    THStorage_free(self);
  }
  // save_save is necessary since the old eager format saved storages as
  // [size + data], but the v1.5 eager format removes this since size is saved
  // in the filesize.
  template <class io>
  static void Storage_writeFileRaw(
      at::StorageImpl* self,
      io fd,
      bool save_size) {
    at::DeviceGuard guard(self->device());

    scalar_t* data;
    int64_t size_bytes = self->nbytes();
    int64_t numel = size_bytes / sizeof(scalar_t);
    std::unique_ptr<char[]> cpu_data(new char[size_bytes]);
    data = (scalar_t*)cpu_data.get();
    xpu::dpcpp::dpcppMemcpy(
        data,
        Storage_data(self),
        size_bytes,
        xpu::dpcpp::dpcppMemcpyKind::DeviceToHost);
    if (save_size) {
      if (torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN)
        doWrite(fd, &numel, sizeof(int64_t));
      else {
        int64_t nsize; // convert big endian cpu to little endian storage
        torch::utils::THP_encodeInt64Buffer(
            (uint8_t*)&nsize,
            (const int64_t*)&numel,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            1);
        doWrite(fd, &nsize, sizeof(int64_t));
      }
    }
    // fast track for bytes and little endian
    if (sizeof(scalar_t) == 1 ||
        torch::utils::THP_nativeByteOrder() ==
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
      doWrite(fd, data, size_bytes);
    } else {
      int64_t buffer_size = std::min(numel, (int64_t)5000);
      std::unique_ptr<uint8_t[]> le_buffer(
          new uint8_t[buffer_size * sizeof(scalar_t)]);
      for (int64_t i = 0; i < numel; i += buffer_size) {
        size_t to_convert = std::min(numel - i, buffer_size);
        if (sizeof(scalar_t) == 2) {
          torch::utils::THP_encodeInt16Buffer(
              (uint8_t*)le_buffer.get(),
              (const int16_t*)data + i,
              torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
              to_convert);
        } else if (sizeof(scalar_t) == 4) {
          torch::utils::THP_encodeInt32Buffer(
              (uint8_t*)le_buffer.get(),
              (const int32_t*)data + i,
              torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
              to_convert);
        } else if (sizeof(scalar_t) == 8) {
          torch::utils::THP_encodeInt64Buffer(
              (uint8_t*)le_buffer.get(),
              (const int64_t*)data + i,
              torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
              to_convert);
        }
        doWrite(fd, le_buffer.get(), to_convert * sizeof(scalar_t));
      }
    }
  }

  template <class io>
  static at::StorageImpl* Storage_readFileRaw(
      io file,
      at::StorageImpl* _storage) {
    at::OptionalDeviceGuard guard;
    if (_storage != nullptr) {
      guard.reset_device(_storage->device());
    }

    scalar_t* data;
    int64_t size;
    doRead(file, &size, sizeof(int64_t));
    if (torch::utils::THP_nativeByteOrder() ==
        torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
      int64_t nsize; // convert little endian storage to big endian cpu
      nsize = size;
      torch::utils::THP_decodeInt64Buffer(
          &size,
          (const uint8_t*)&nsize,
          torch::utils::THP_nativeByteOrder(),
          1);
    }
    THStorageImpl_Ptr storage;
    if (_storage == nullptr) {
      storage = Storage_newWithSize(size);
    } else {
      int64_t _storage_numel = _storage->nbytes() / sizeof(scalar_t);
      THPUtils_assert(
          _storage_numel == size,
          "storage has wrong size: expected %ld got %ld",
          size,
          _storage_numel);
      storage = _storage;
    }

    std::unique_ptr<char[]> cpu_data(new char[size * sizeof(scalar_t)]);
    data = (scalar_t*)cpu_data.get();

    // fast track for bytes and little endian
    if (sizeof(scalar_t) == 1 ||
        torch::utils::THP_nativeByteOrder() ==
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
      doRead(file, data, storage->nbytes());
    } else {
      int64_t buffer_size = std::min(size, (int64_t)5000);
      std::unique_ptr<uint8_t[]> le_buffer(
          new uint8_t[buffer_size * sizeof(scalar_t)]);

      for (int64_t i = 0; i < size; i += buffer_size) {
        size_t to_convert = std::min(size - i, buffer_size);
        doRead(file, le_buffer.get(), sizeof(scalar_t) * to_convert);

        if (sizeof(scalar_t) == 2) {
          torch::utils::THP_decodeInt16Buffer(
              (int16_t*)data + i,
              le_buffer.get(),
              torch::utils::THP_nativeByteOrder(),
              to_convert);
        } else if (sizeof(scalar_t) == 4) {
          torch::utils::THP_decodeInt32Buffer(
              (int32_t*)data + i,
              le_buffer.get(),
              torch::utils::THP_nativeByteOrder(),
              to_convert);
        } else if (sizeof(scalar_t) == 8) {
          torch::utils::THP_decodeInt64Buffer(
              (int64_t*)data + i,
              le_buffer.get(),
              torch::utils::THP_nativeByteOrder(),
              to_convert);
        }
      }
    }

    xpu::dpcpp::dpcppMemcpy(
        Storage_data(storage),
        data,
        size * sizeof(scalar_t),
        xpu::dpcpp::dpcppMemcpyKind::HostToDevice);

    return storage.release();
  }

  static at::StorageImpl* Storage_newWithMapping(
      const char* fileName,
      ptrdiff_t size,
      int isShared) {
    //    THError("not available yet for THCStorage");
    return NULL;
  }

  static at::StorageImpl* Storage_new() {
    at::StorageImpl* storage = c10::make_intrusive<at::StorageImpl>(
                                   c10::StorageImpl::use_byte_size_t(),
                                   0,
                                   xpu::dpcpp::getDeviceAllocator(),
                                   true)
                                   .release();
    return storage;
  }

  static PyObject* Storage_New(at::StorageImpl* ptr) {
    AT_ASSERT(ptr);
    PyTypeObject* type = (PyTypeObject*)THXPStorage_Class;
    PyObject* obj = type->tp_alloc(type, 0);
    if (obj) {
      ((THPStorage*)obj)->cdata = ptr;
    } else {
      Storage_free(ptr);
    }
    return obj;
  }

  static void Storage_resizeBytes(at::StorageImpl* self, ptrdiff_t size_bytes) {
    THArgCheck(size_bytes >= 0, 2, "invalid size");
    THAssert(self->allocator() != nullptr);
    int device = 0;
    // TODO: add current device
    // xpu::dpcpp::getCurrentDevice

    //    if (!self->resizable())
    //      THError("Trying to resize storage that is not resizable");

    if (size_bytes == 0) {
      self->set_data_ptr(
          at::DataPtr(nullptr, at::Device(at::DeviceType::XPU, device)));
      self->set_nbytes(0);
    } else {
      at::DataPtr data = self->allocator()->allocate(size_bytes);

      if (self->data_ptr()) {
        xpu::dpcpp::dpcppMemcpyAsync(
            data.get(),
            self->data(),
            std::min(self->nbytes(), (size_t)size_bytes),
            xpu::dpcpp::dpcppMemcpyKind::DeviceToDevice);
      }

      // Destructively overwrite data_ptr
      self->set_data_ptr(std::move(data));
      self->set_nbytes(size_bytes);
    }
  }

  static void Storage_fill(at::StorageImpl* storage, scalar_t value) {
    auto dpcpp_queue = xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue();
    scalar_t* data_ptr = Storage_data(storage);
    dpcppFill(data_ptr, value, storage->nbytes() / sizeof(scalar_t));
  }

  static void Storage_set(
      at::StorageImpl* self,
      ptrdiff_t index,
      scalar_t value) {
    //    THArgCheck(
    //        (index >= 0) && (index < (self->nbytes() / sizeof(scalar_t))),
    //        2,
    //        "index out of bounds");
    dpcppMemcpy(
        self->data<scalar_t>() + index,
        &value,
        sizeof(scalar_t),
        xpu::dpcpp::dpcppMemcpyKind::HostToDevice);
  }

  static scalar_t Storage_get(const at::StorageImpl* self, ptrdiff_t index) {
    THArgCheck(
        (index >= 0) && (index < (self->nbytes() / sizeof(scalar_t))),
        2,
        "index out of bounds");
    scalar_t value;
    dpcppMemcpy(
        &value,
        self->data<scalar_t>() + index,
        sizeof(scalar_t),
        xpu::dpcpp::dpcppMemcpyKind::DeviceToHost);
    return value;
  }

  static scalar_t* Storage_data(const at::StorageImpl* self) {
    return self->data<scalar_t>();
  }

  static at::StorageImpl* Storage_newWithAllocator(
      ptrdiff_t size,
      at::Allocator* allocator) {
    at::StorageImpl* storage =
        c10::make_intrusive<at::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            size *
                sizeof(
                    typename c10::impl::ScalarTypeToCPPType<scalarType>::type),
            allocator,
            true)
            .release();
    return storage;
  }

  static at::StorageImpl* Storage_newWithSize(ptrdiff_t size) {
    THStorage* storage =
        c10::make_intrusive<at::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            size *
                sizeof(
                    typename c10::impl::ScalarTypeToCPPType<scalarType>::type),
            xpu::dpcpp::getDeviceAllocator(),
            true)
            .release();
    return storage;
  }

  static PyObject* THXPStorage_new(PyObject* self, PyObject* noargs) {
    HANDLE_TH_ERRORS
    THStorageImpl_Ptr new_storage(Storage_new());
    PyObject* _ret = Storage_New(new_storage);
    new_storage.release();
    return _ret;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_copy_(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs) {
    HANDLE_TH_ERRORS
    return NULL;
    //    return THPStorageCopyMethod(
    //        THXPStorage_copy_functions, (PyObject*)self, args, kwargs);
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_elementSize(PyObject* self, PyObject* noargs) {
    HANDLE_TH_ERRORS
    return PyLong_FromLong(sizeof(scalar_t));
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_resize_(PyObject* _self, PyObject* number_arg) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    THPUtils_assert(
        THPUtils_checkLong(number_arg),
        "resize_ expects an int, "
        "but got %s",
        THPUtils_typename(number_arg));
    int64_t newsize = THPUtils_unpackLong(number_arg);
    Storage_resizeBytes(self->cdata, newsize * sizeof(scalar_t));
    Py_INCREF(self);
    return (PyObject*)self;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_fill_(PyObject* _self, PyObject* number_arg) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    THPUtils_assert(
        THPUtils_checkReal<scalar_t>(number_arg),
        "fill_ expects %s, "
        "but got %s",
        //        THPUtils_typeTraits<scalar_t>::python_type_str,
        THPUtils_typename(number_arg));
    Storage_fill(self->cdata, THPUtils_unpackReal<scalar_t>(number_arg));
    Py_INCREF(self);
    return (PyObject*)self;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_getDevice(PyObject* _self, PyObject* noargs) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    return PyLong_FromLong(self->cdata->device().index());
    END_HANDLE_TH_ERRORS
  }

  static void THXPStorage_dealloc(PyObject* _self) {
    auto self = (THXP_Storage*)_self;
    Storage_free(self->cdata);
    Py_TYPE(self)->tp_free((PyObject*)self);
  }

  static PyObject* THXPStorage_device(PyObject* _self, void* unused) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    return THPDevice_New(self->cdata->device());
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_dtype(PyObject* _self, void* unused) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    return torch::autograd::utils::wrap(torch::getTHPDtype(
        at::typeMetaToScalarType(caffe2::TypeMeta::Make<scalar_t>())));
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_get(PyObject* _self, PyObject* index) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    /* Integer index */
    if (THPUtils_checkLong(index)) {
      int64_t nindex = THPUtils_unpackLong(index);
      if (nindex < 0)
        nindex += (self->cdata->nbytes() / sizeof(scalar_t));
      if (nindex < 0 || nindex >= (self->cdata->nbytes() / sizeof(scalar_t))) {
        PyErr_SetString(
            PyExc_IndexError, "index is out of range for the storage size");
        return nullptr;
      }
      scalar_t value = Storage_get(self->cdata, nindex);
      return THPUtils_newReal<scalar_t>(value);
      /* Slice index */
    } else if (PySlice_Check(index)) {
      Py_ssize_t start, stop, slicelength, step;
      int64_t len = self->cdata->nbytes() / sizeof(scalar_t);
      if (!THPUtils_parseSlice(index, len, &start, &stop, &step, &slicelength))
        return nullptr;
      if (step != 1) {
        THPUtils_setError(
            "Trying to slice with a step of %lld, but only a step of "
            "1 is supported",
            (long long)step);
        return nullptr;
      }

      scalar_t* data = Storage_data(self->cdata);

      at::StorageImpl* old_storage = self->cdata;
      c10::raw::intrusive_ptr::incref(old_storage);
      at::Storage new_storage(c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          slicelength * sizeof(scalar_t),
          at::DataPtr(
              static_cast<void*>(data + start),
              old_storage,
              [](void* s) {
                c10::raw::intrusive_ptr::decref(
                    static_cast<at::StorageImpl*>(s));
              },
              old_storage->device()),
          old_storage->allocator(),
          /* resizable */ false));

      PyObject* _ret = Storage_New(new_storage.unsafeReleaseStorageImpl());
      return _ret;
    }
    PyErr_Format(
        PyExc_TypeError,
        "can't index a " THPStorageStr " with %s",
        THPUtils_typename(index));
    return nullptr;
    END_HANDLE_TH_ERRORS
  }

  static int THXPStorage_set(
      PyObject* _self,
      PyObject* index,
      PyObject* value) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    if (!THPUtils_checkReal<scalar_t>(value)) {
      THPUtils_setError(
          "can only set storage content with a %s, but got "
          "%s instead",
          //          THPUtils_typeTraits<scalar_t>::python_type_str,
          THPUtils_typename(value));
      return -1;
    }

    scalar_t rvalue = THPUtils_unpackReal<scalar_t>(value);
    if (THPUtils_checkLong(index)) {
      int64_t nindex = THPUtils_unpackLong(index);
      Storage_set(self->cdata, nindex, rvalue);
      return 0;
    } else if (PySlice_Check(index)) {
      Py_ssize_t start, stop, slicelength, step;
      int64_t len = self->cdata->nbytes() / sizeof(scalar_t);
      if (!THPUtils_parseSlice(index, len, &start, &stop, &step, &slicelength))
        return -1;
      if (step != 1) {
        THPUtils_setError(
            "Trying to slice with a step of %lld, but only a step of "
            "1 is supported",
            (long long)step);
        return 0;
      }
      // TODO: check the bounds only once
      // TODO: fill?
      for (; start < stop; start++)
        Storage_set(self->cdata, start, rvalue);
      return 0;
    }
    THPUtils_setError(
        "can't index a " THPStorageStr " with %s", THPUtils_typename(index));
    return -1;
    END_HANDLE_TH_ERRORS_RET(-1)
  }

  static Py_ssize_t THXPStorage_length(PyObject* _self) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    return self->cdata->nbytes() / sizeof(scalar_t);
    END_HANDLE_TH_ERRORS_RET(-1)
  }

  static PyObject* THXPStorage_size(PyObject* _self, PyObject* noargs) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    return PyLong_FromLong(self->cdata->nbytes() / sizeof(scalar_t));
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_dataPtr(PyObject* _self, PyObject* noargs) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    return PyLong_FromVoidPtr(Storage_data(self->cdata));
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_isPinned(PyObject* _self, PyObject* noargs) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    // TODO:
    Py_RETURN_FALSE;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_writeFile(PyObject* _self, PyObject* args) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    PyObject* file = PyTuple_GET_ITEM(args, 0);
    bool is_real_file = PyTuple_GET_ITEM(args, 1) == Py_True;
    bool save_size = PyTuple_GET_ITEM(args, 2) == Py_True;

    if (!is_real_file) {
      Storage_writeFileRaw<PyObject*>(self->cdata, file, save_size);
      Py_RETURN_NONE;
    }

    int fd = PyObject_AsFileDescriptor(file);
    THPUtils_assert(
        fd != -1,
        "_write_file couldn't retrieve a file descriptor "
        "from given object");
    Storage_writeFileRaw<int>(self->cdata, fd, save_size);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_newWithFile(PyObject* _unused, PyObject* file) {
    HANDLE_TH_ERRORS
    int fd = PyObject_AsFileDescriptor(file);
    THPUtils_assert(
        fd != -1,
        "_new_with_file couldn't retrieve a file "
        "descriptor from given object");
    at::StorageImpl* storage = Storage_readFileRaw<int>(fd, nullptr);
    if (storage == nullptr)
      return nullptr;
    PyObject* result = Storage_New(storage);
    return result;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_setFromFile(PyObject* _self, PyObject* args) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    PyObject* file = PyTuple_GET_ITEM(args, 0);
    PyObject* offset = PyTuple_GET_ITEM(args, 1);
    bool is_real_file = PyTuple_GET_ITEM(args, 2) == Py_True;

    if (!is_real_file) {
      // offset can be implemented with a call to the Python object's seek()
      // but it is currently unnecessary to support this.
      THPUtils_assert(
          offset == Py_None,
          "_set_from_file: offset is NYI for filelike objects");
      at::StorageImpl* storage =
          Storage_readFileRaw<PyObject*>(file, self->cdata);
      if (storage == nullptr) {
        return nullptr;
      }
      Py_INCREF(self);
      return (PyObject*)self;
    }

    // file is backed by a fd
    const int fd = PyObject_AsFileDescriptor(file);
    const auto fd_original_pos = lseek(fd, 0, SEEK_CUR);
    if (offset != Py_None) {
      lseek(fd, THPUtils_unpackLong(offset), SEEK_SET);
    }
    THPUtils_assert(
        fd != -1,
        "_set_from_file couldn't retrieve a file "
        "descriptor from given object");
    auto storage = THXStorage_readFileRaw<int>(fd, self->cdata);
    if (storage == nullptr)
      return nullptr;
    Py_INCREF(self);

    // the file descriptor is returned to original position and
    // the file handle at python call-site needs updating to the
    // advanced position
    const auto fd_current_pos = lseek(fd, 0, SEEK_CUR);
    lseek(fd, fd_original_pos, SEEK_SET);
    const auto seek_return =
        PyObject_CallMethod(file, "seek", "Li", (long long)fd_current_pos, 0);
    if (seek_return == nullptr) {
      return nullptr;
    }
    Py_DECREF(seek_return);

    return (PyObject*)self;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_fromFile(
      PyObject* _unused,
      PyObject* args,
      PyObject* keywds) {
    HANDLE_TH_ERRORS
    const char* filename;
    Py_ssize_t size = 0;
    int shared = 0;
    static const char* kwlist[] = {"filename", "shared", "size", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywds,
            "s|in",
            const_cast<char**>(kwlist),
            &filename,
            &shared,
            &size)) {
      return nullptr;
    }

    if (shared) {
      fprintf(
          stderr, "TODO: map frontend `shared` to native dpc++ `shared`.\n");
    }
    at::StorageImpl* storage = Storage_newWithMapping(filename, size, shared);
    return (PyObject*)Storage_New(storage);
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage__setCdata(PyObject* _self, PyObject* new_cdata) {
    HANDLE_TH_ERRORS
    auto self = (THXP_Storage*)_self;
    THPUtils_assert(
        THPUtils_checkLong(new_cdata),
        "given an invalid argument to "
        "_set_cdata - expected an int or long, but got %s",
        THPUtils_typename(new_cdata));
    at::StorageImpl* ptr = (at::StorageImpl*)PyLong_AsVoidPtr(new_cdata);
    Storage_retain(ptr);
    Storage_free(self->cdata);
    self->cdata = ptr;
    Py_INCREF(self);
    return (PyObject*)self;
    END_HANDLE_TH_ERRORS
  }

  static PyObject* THXPStorage_pynew(
      PyTypeObject* type,
      PyObject* args,
      PyObject* kwargs) {
    HANDLE_TH_ERRORS
    Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;

    THXPStorage_Ptr self((THXP_Storage*)type->tp_alloc(type, 0));
    THPUtils_assert(self, "failed to allocate a " THPStorageStr " object");
    c10::Allocator* allocator = nullptr;

    // Internally we allow constructing with a keywoard only argument cdata
    if (kwargs != nullptr) {
      PyObject* allocator_ptr = PyDict_GetItemString(kwargs, "allocator");
      if (allocator_ptr) {
        THPUtils_assert(THPUtils_checkLong(allocator_ptr), "invalid allocator");
        allocator =
            static_cast<c10::Allocator*>(PyLong_AsVoidPtr(allocator_ptr));
        PyDict_DelItemString(kwargs, "allocator");
      }

      Py_ssize_t num_kwargs = PyDict_Size(kwargs);
      if (num_args == 0) {
        PyObject* cdata_ptr = PyDict_GetItemString(kwargs, "cdata");
        if (num_kwargs == 1 && cdata_ptr && THPUtils_checkLong(cdata_ptr)) {
          at::StorageImpl* ptr = (at::StorageImpl*)PyLong_AsVoidPtr(cdata_ptr);
          self->cdata = ptr;
          return (PyObject*)self.release();
        }
      }
      THPUtils_assert(
          num_kwargs == 0, THPStorageStr "(): invalid keyword arguments");
    }

    // torch.Storage()
    if (num_args == 0) {
      if (allocator) {
        self->cdata = Storage_newWithAllocator(0, allocator);
      } else {
        self->cdata = Storage_new();
      }
      return (PyObject*)self.release();
    }

    PyObject* first_arg = PyTuple_GET_ITEM(args, 0);

    // torch.Storage(size)
    if (num_args == 1 && THPUtils_checkLong(first_arg)) {
      int64_t size = THPUtils_unpackLong(first_arg);
      if (allocator) {
        self->cdata = Storage_newWithAllocator(size, allocator);
      } else {
        self->cdata = Storage_newWithSize(size);
      }
      return (PyObject*)self.release();
    }

    // torch.Storage(view_source, [offset, [size]])
    if (num_args < 4 &&
        PyObject_IsInstance(first_arg, THXPStorage_Bridge::THXPStorage_Class)) {
      THPUtils_setError("storage views not supported");
      return nullptr;
    }

    // torch.Storage(sequence)
    if (num_args == 1 && PySequence_Check(first_arg)) {
      Py_ssize_t length = PySequence_Length(first_arg);
      THPUtils_assert(
          length >= 0,
          "couldn't obtain the length of %s",
          THPUtils_typename(first_arg));
      self->cdata = Storage_newWithSize(length);
      THPObjectPtr item;
      try {
        for (Py_ssize_t i = 0; i < length; i++) {
          item = PySequence_GetItem(first_arg, i);
          scalar_t value = THPUtils_unpackReal<scalar_t>(item.get());
          // TODO: this might be slow - consider batched updates?
          Storage_set(self->cdata, i, value);
        }
      } catch (const std::exception& e) {
        THPUtils_setError(
            "tried to construct a storage from a sequence (%s), "
            "but one of the items was of type %s instead of %s",
            THPUtils_typename(first_arg),
            THPUtils_typename(item.get())/*,*/
            /*THPUtils_typeTraits<typename c10::impl::ScalarTypeToCPPType<
                scalarType>::type>::python_type_str*/);
        return nullptr;
      }
      return (PyObject*)self.release();
    }

    THPUtils_invalidArguments(
        args,
        kwargs,
        THPStorageStr " constructor",
        6,
        "no arguments",
        "(int size)",
        "(Sequence data)",
        "(" THPStorageStr " view_source)",
        "(" THPStorageStr " view_source, int offset)",
        "(" THPStorageStr " view_source, int offset, int size)");
    return nullptr;
    END_HANDLE_TH_ERRORS
  }

  static void THXPStorage_initCopyMethods() {}
  static PyTypeObject* THXPStorage_Type;
  static PyObject* THXPStorage_Class;
  static THPCopyList THXPStorage_copy_functions;
};

template <at::ScalarType s>
PyTypeObject* THXPStorage_Bridge<s>::THXPStorage_Type;
template <at::ScalarType s>
PyObject* THXPStorage_Bridge<s>::THXPStorage_Class;
template <at::ScalarType s>
THPCopyList THXPStorage_Bridge<s>::THXPStorage_copy_functions;

template <at::ScalarType scalarType>
bool THXPStorage_init(PyObject* module) {
  static PyMappingMethods THXPStorage_mappingmethods = {
      (lenfunc)THXPStorage_Bridge<scalarType>::THXPStorage_length,
      (binaryfunc)THXPStorage_Bridge<scalarType>::THXPStorage_get,
      (objobjargproc)THXPStorage_Bridge<scalarType>::THXPStorage_set};

  static PyTypeObject THXPStorage_Type = {
      PyVarObject_HEAD_INIT(nullptr, 0) nullptr, /* tp_name */
      sizeof(THPStorage), /* tp_basicsize */
      0, /* tp_itemsize */
      &THXPStorage_Bridge<scalarType>::THXPStorage_dealloc, /* tp_dealloc */
      0, /* tp_vectorcall_offset */
      nullptr, /* tp_getattr */
      nullptr, /* tp_setattr */
      nullptr, /* tp_reserved */
      nullptr, /* tp_repr */
      nullptr, /* tp_as_number */
      nullptr, /* tp_as_sequence */
      &THXPStorage_mappingmethods, /* tp_as_mapping */
      nullptr, /* tp_hash  */
      nullptr, /* tp_call */
      nullptr, /* tp_str */
      nullptr, /* tp_getattro */
      nullptr, /* tp_setattro */
      nullptr, /* tp_as_buffer */
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
      nullptr, /* tp_doc */
      nullptr, /* tp_traverse */
      nullptr, /* tp_clear */
      nullptr, /* tp_richcompare */
      0, /* tp_weaklistoffset */
      nullptr, /* tp_iter */
      nullptr, /* tp_iternext */
      nullptr,
      /* will be assigned in init */ /* tp_methods */
      nullptr,
      /* will be assigned in init */ /* tp_members */
      nullptr, /* tp_getset */
      nullptr, /* tp_base */
      nullptr, /* tp_dict */
      nullptr, /* tp_descr_get */
      nullptr, /* tp_descr_set */
      0, /* tp_dictoffset */
      nullptr, /* tp_init */
      nullptr, /* tp_alloc */
      &THXPStorage_Bridge<scalarType>::THXPStorage_pynew, /* tp_new */
  };

  static std::string module_name;
  module_name.append(toString(scalarType));
  module_name.append("StorageBase");
  static std::string class_name("intel_extension_for_pytorch._C.");
  class_name.append(module_name);

  THXPStorage_Type.tp_name = class_name.c_str();

  static std::vector<PyMethodDef> methods;
  static PyMethodDef THXPStorage_methods[] = {
      {"copy_",
       (PyCFunction)(
           void (*)(void))THXPStorage_Bridge<scalarType>::THXPStorage_copy_,
       METH_VARARGS | METH_KEYWORDS,
       nullptr},
      {"element_size",
       THXPStorage_Bridge<scalarType>::THXPStorage_elementSize,
       METH_NOARGS,
       nullptr},
      {"fill_",
       THXPStorage_Bridge<scalarType>::THXPStorage_fill_,
       METH_O,
       nullptr},
      {"new",
       THXPStorage_Bridge<scalarType>::THXPStorage_new,
       METH_NOARGS,
       nullptr},
      {"resize_",
       THXPStorage_Bridge<scalarType>::THXPStorage_resize_,
       METH_O,
       nullptr},
      {"size",
       THXPStorage_Bridge<scalarType>::THXPStorage_size,
       METH_NOARGS,
       nullptr},
      {"data_ptr",
       THXPStorage_Bridge<scalarType>::THXPStorage_dataPtr,
       METH_NOARGS,
       nullptr},
      {"is_pinned",
       THXPStorage_Bridge<scalarType>::THXPStorage_isPinned,
       METH_NOARGS,
       nullptr},
      {"_write_file",
       THXPStorage_Bridge<scalarType>::THXPStorage_writeFile,
       METH_VARARGS,
       nullptr},
      {"_new_with_file",
       THXPStorage_Bridge<scalarType>::THXPStorage_newWithFile,
       METH_O | METH_STATIC,
       nullptr},
      {"_set_from_file",
       THXPStorage_Bridge<scalarType>::THXPStorage_setFromFile,
       METH_VARARGS,
       nullptr},
      {"from_file",
       (PyCFunction)(void (*)(void))(
           THXPStorage_Bridge<scalarType>::THXPStorage_fromFile),
       METH_VARARGS | METH_KEYWORDS | METH_STATIC,
       nullptr},
      {"get_device",
       THXPStorage_Bridge<scalarType>::THXPStorage_getDevice,
       METH_NOARGS,
       nullptr},
      {"_set_cdata",
       THXPStorage_Bridge<scalarType>::THXPStorage__setCdata,
       METH_O,
       nullptr},
      {nullptr}};

  static PyMethodDef THXPStorage_sharingMethods[] = {
      //    {"_new_with_weak_ptr", THPStorage_(newWithWeakPtr), METH_O |
      //    METH_CLASS, nullptr},
      //#ifdef THC_GENERIC_FILE
      //    {"_share_xpu_", THPStorage_(shareXPU), METH_NOARGS, nullptr},
      //    {"_new_shared_xpu", THPStorage_(newSharedXPU), METH_VARARGS |
      //    METH_STATIC, nullptr},
      //    {"_release_ipc_counter", THPStorage_(releaseIPCCounter),
      //    METH_VARARGS | METH_STATIC, nullptr},
      //#endif
      //    {"_weak_ref", THPStorage_(weakRef), METH_NOARGS, nullptr},
      //    {"_free_weak_ref", THPStorage_(freeWeakRef), METH_O | METH_STATIC,
      //    nullptr},
      //    {"_expired", THPStorage_(expired), METH_O | METH_STATIC, nullptr},
      //    {"_shared_decref", THPStorage_(sharedDecref), METH_NOARGS, nullptr},
      //    {"_shared_incref", THPStorage_(sharedIncref), METH_NOARGS, nullptr},
      //    {"_get_shared_fd", THPStorage_(sharedFd), METH_NOARGS, nullptr},
      //    {"is_shared", THPStorage_(isShared), METH_NOARGS, nullptr},
      {nullptr}};

  static struct PyMemberDef THXPStorage_members[] = {
      {(char*)"_cdata",
       T_ULONGLONG,
       offsetof(THXP_Storage, cdata),
       READONLY,
       nullptr},
      {nullptr}};

  static struct PyGetSetDef THXPStorage_properties[] = {
      {"device",
       (getter)THXPStorage_Bridge<scalarType>::THXPStorage_device,
       nullptr,
       nullptr,
       nullptr},
      {"dtype",
       (getter)THXPStorage_Bridge<scalarType>::THXPStorage_dtype,
       nullptr,
       nullptr,
       nullptr},
      {nullptr}};

  THPUtils_addPyMethodDefs(methods, THXPStorage_methods);
  THPUtils_addPyMethodDefs(methods, THXPStorage_sharingMethods);

  THXPStorage_Type.tp_methods = methods.data();
  THXPStorage_Type.tp_members = THXPStorage_members;
  THXPStorage_Type.tp_getset = THXPStorage_properties;
  if (PyType_Ready(&THXPStorage_Type) < 0)
    return false;
  Py_INCREF(&THXPStorage_Type);
  PyModule_AddObject(module, module_name.c_str(), (PyObject*)&THXPStorage_Type);
  THXPStorage_Bridge<scalarType>::THXPStorage_Type = &THXPStorage_Type;

  return true;
}

PyObject* THPStorageClass = nullptr;

template <at::ScalarType scalarType>
void THPStorage_postInit(PyObject* module) {
  // Initialize the copy method.
  //  THXPStorage_Bridge<scalarType>::THXPStorage_initCopyMethods();

  THPStorageClass = PyObject_GetAttrString(module, "_UntypedStorage");
  if (!THPStorageClass)
    throw python_error();

  at::Backend backend = at::Backend::XPU;

  if (scalarType == at::ScalarType::QInt8 ||
      scalarType == at::ScalarType::QUInt8)
    backend = at::Backend::QuantizedXPU;

  torch::registerStoragePyTypeObject(
      (PyTypeObject*)THPStorageClass, backend, scalarType);
}

template <>
void THPPointer<at::StorageImpl>::free() {
  if (ptr) {
    //    THStorage_free(ptr);
  }
}

template <>
void THPPointer<THXP_Storage>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

#define DEFINE_THPUTLIS_NEW(s, n)          \
  template <>                              \
  PyObject* THPUtils_newReal<s>(s value) { \
    return THP##n##Utils_newReal(value);   \
  };
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_THPUTLIS_NEW);
#undef DEFINE_THPUTLIS_NEW

template <>
PyObject* THPUtils_newReal<c10::quint8>(c10::quint8 value) {
  return PyLong_FromLong(value.val_);
};

template <>
PyObject* THPUtils_newReal<c10::qint8>(c10::qint8 value) {
  return PyLong_FromLong(value.val_);
};

#define DEFINE_THPUTLIS_UNPACK(s, n)         \
  template <>                                \
  s THPUtils_unpackReal<s>(PyObject * obj) { \
    return THP##n##Utils_unpackReal(obj);    \
  };
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_THPUTLIS_UNPACK);
#undef DEFINE_THPUTLIS_NEW

template <>
c10::quint8 THPUtils_unpackReal<c10::quint8>(PyObject* obj) {
  auto value = (int)THPUtils_unpackReal_INT(obj);
  return c10::quint8{value};
};

template <>
c10::qint8 THPUtils_unpackReal<c10::qint8>(PyObject* obj) {
  auto value = (int)THPUtils_unpackReal_INT(obj);
  return c10::qint8{value};
};

#define DEFINE_THPUTLIS_CHECK(s, n)            \
  template <>                                  \
  bool THPUtils_checkReal<s>(PyObject * obj) { \
    return THP##n##Utils_checkReal(obj);       \
  };
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_THPUTLIS_CHECK);
#undef DEFINE_THPUTLIS_CHECK

template <>
bool THPUtils_checkReal<c10::quint8>(PyObject* obj) {
  return THPUtils_checkReal_INT(obj);
};

template <>
bool THPUtils_checkReal<c10::qint8>(PyObject* obj) {
  return THPUtils_checkReal_INT(obj);
};

PyObject* THDPStorage_postInitExtension(PyObject* module) {
  HANDLE_TH_ERRORS

  // Register Storage Python objects with DynamicTypes.cpp
  //  THPStorage_postInit<at::kBool>(module);
  THPStorage_postInit<at::kByte>(module);
  //  THPStorage_postInit<at::kHalf>(module);
  //  THPStorage_postInit<at::kShort>(module);
  //  THPStorage_postInit<at::kInt>(module);
  //  THPStorage_postInit<at::kLong>(module);
  //  THPStorage_postInit<at::kFloat>(module);
  //  THPStorage_postInit<at::kDouble>(module);
  //  THPStorage_postInit<at::kBFloat16>(module);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
    return module;

PyObject* THDPStorage_init(PyObject* module) {
  HANDLE_TH_ERRORS
  ASSERT_TRUE(THXPStorage_init<at::kByte>(module));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
