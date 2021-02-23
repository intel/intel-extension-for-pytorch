#include <torch/csrc/THP.h>
#include <torch/csrc/utils.h>
#include <Storage.h>

template<>
void THPPointer<at::StorageImpl>::free() {
  if (ptr) {
    THStorage_free(ptr);
  }
}

template<>
void THPPointer<THXP_Storage>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

#define DEFINE_THPUTLIS_NEW(s, n) template <> \
                             PyObject* THPUtils_newReal<s>(s value) \
                             {        \
                                return THP##n##Utils_newReal(value); \
                             };
AT_FORALL_SCALAR_TYPES_AND(BFloat16,DEFINE_THPUTLIS_NEW);
#undef DEFINE_THPUTLIS_NEW

#define DEFINE_THPUTLIS_UNPACK(s, n) template <> \
                             s THPUtils_unpackReal<s>(PyObject* obj) \
                             {        \
                                return THP##n##Utils_unpackReal(obj); \
                             };
AT_FORALL_SCALAR_TYPES_AND(BFloat16,DEFINE_THPUTLIS_UNPACK);
#undef DEFINE_THPUTLIS_NEW
