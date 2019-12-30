#include <Python.h>
#include <ATen/aten_ipex_type_default.h>


// static PyObject* M_PyInstanceMethod_New(PyObject *self, PyObject *func)
// {
//   return PyInstanceMethod_New(func);
// }
// 
// static PyMethodDef ModuleMethods[] = {
// 	 { (char *)"M_PyInstanceMethod_New", (PyCFunction)M_PyInstanceMethod_New, METH_O, NULL},
// 	 { NULL, NULL, 0, NULL }
// };

#ifdef __cplusplus
// extern "C"{
void torch_ipex_init() {
  // TODO:
  printf("loading _torch_ipex.so ++\n");
  torch_ipex::RegisterAtenTypeFunctions();
  printf("loading _torch_ipex.so --\n");
}
// PyObject* PyInit__torch_ipex() {
//   // TODO:
//   static struct PyModuleDef module = {
//     PyModuleDef_HEAD_INIT,
//     "torch_ipex",
//     NULL,
//     -1,
//     ModuleMethods,
//     NULL,
//     NULL,
//     NULL,
//     NULL
//   };
// 
//   PyObject* m = PyModule_Create(&module);
//   PyTypeObject* m_ = m;
//   m_->tp_mro;
//   return m;
// }
// }
#endif
