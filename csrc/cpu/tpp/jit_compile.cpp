#include "jit_compile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#else
#include <stdexcept>
#endif
namespace torch_ipex {
namespace tpp {
void* jit_compile_and_load(
    const std::string filename,
    const std::string flags) {
#ifndef _WIN32
  char libname[] = "/tmp/ppx_XXXXXX";
  int fd = mkstemp(libname);
  unlink(libname);
  char fdname[50];
  sprintf(fdname, "/proc/self/fd/%d", fd);
  auto cmd = std::string("g++ -shared -fPIC -x c++ ") + flags;
  cmd = cmd + " -o " + fdname + " " + filename;
  printf("JIT COMPILE: %s\n", cmd.c_str());
  int ret = system(cmd.c_str());
  if (ret != 0)
    return NULL;
  auto handle = dlopen(fdname, RTLD_LAZY | RTLD_NODELETE);
  if (!handle) {
    fputs(dlerror(), stderr);
    return NULL;
  }
  return handle;
#else
  throw std::runtime_error("not implemented.");
  return NULL;
#endif
}

void* jit_from_file(
    const std::string filename,
    const std::string flags,
    const std::string func_name) {
#ifndef _WIN32
  void* handle = jit_compile_and_load(filename, flags);
  if (handle == NULL)
    return NULL;
  void* func = dlsym(handle, func_name.c_str());
  if (func == NULL) {
    printf("Unable to find '%s' symbol in JIT COMPILE\n", func_name.c_str());
  }
  dlclose(handle);
  return func;
#else
  throw std::runtime_error("not implemented.");
  return NULL;
#endif
}

void* jit_from_str(
    const std::string src,
    const std::string flags,
    const std::string func_name) {
#ifndef _WIN32
  char filename[] = "/tmp/ppx_XXXXXX";
  int fd = mkstemp(filename);
  unlink(filename);
  char fdname[50];
  sprintf(fdname, "/proc/self/fd/%d", fd);
  write(fd, src.c_str(), src.length());
  return jit_from_file(fdname, flags, func_name);
#else
  throw std::runtime_error("not implemented.");
  return NULL;
#endif
}
} // namespace tpp
} // namespace torch_ipex