#include "module_version.h"
#include <libxsmm.h>
#include "ideep.hpp"
#include "mkl.h"

namespace torch_ipex {
namespace utils {

void get_mkl_version() {
  MKLVersion Version;
  mkl_get_version(&Version);
  printf("Major version:           %d\n", Version.MajorVersion);
  printf("Minor version:           %d\n", Version.MinorVersion);
  printf("Update version:          %d\n", Version.UpdateVersion);
  printf("Product status:          %s\n", Version.ProductStatus);
  printf("Build:                   %s\n", Version.Build);
  printf("Platform:                %s\n", Version.Platform);
  printf("Processor optimization:  %s\n", Version.Processor);
}

void get_libxsmm_version() {
#if 1
  printf(
      "Not avaliable yet, due to libxsmm CMake build not generate version info.\n");
#else
  printf("libxsmm config version:  %s\n", LIBXSMM_CONFIG_VERSION);
  printf("Config branch:           %s\n", LIBXSMM_CONFIG_BRANCH);
  printf("Major version:           %d\n", LIBXSMM_CONFIG_VERSION_MAJOR);
  printf("Minor version:           %d\n", LIBXSMM_CONFIG_VERSION_MINOR);
  printf("Update version:          %d\n", LIBXSMM_CONFIG_VERSION_UPDATE);
  printf("Patch version:           %d\n", LIBXSMM_CONFIG_VERSION_PATCH);
  printf("Build date:              %d\n", LIBXSMM_CONFIG_BUILD_DATE);
#endif
}

void get_ideep_version() {
  printf("Major version:           %d\n", IDEEP_VERSION_MAJOR);
  printf("Minor version:           %d\n", IDEEP_VERSION_MINOR);
  printf("Patch version:           %d\n", IDEEP_VERSION_PATCH);
  printf("Revision version:        %d\n", IDEEP_VERSION_REVISION);
}
} // namespace utils
} // namespace torch_ipex
