FILE(GLOB aten_ops_src "operators/*.cpp" "operators/sparse/*.cpp")

# Reset the order of src files to compile slow files first
list(APPEND slow_patterns
  Pow
  MathCompare
  Activation
  UnaryOps
  BinaryOps
  TopK
  Sort
  )

set(slow_srcs)
foreach (fn_pattern ${slow_patterns})
  foreach (sfile ${aten_ops_src})
    string(REGEX MATCHALL "${fn_pattern}[_a-zA-Z0-9]*\.cpp$" FN_LIST ${sfile})
    if (FN_LIST)
      list(APPEND slow_srcs ${sfile})
    endif()
  endforeach()
endforeach()

set(fast_srcs)
foreach(sfile ${aten_ops_src})
  if (NOT ${sfile} IN_LIST slow_srcs)
    list(APPEND fast_srcs ${sfile})
  endif()
endforeach()

set(IPEX_ATEN_OPS_SRCS ${slow_srcs} ${fast_srcs} PARENT_SCOPE)

# Add oneDNN runtime src here to avoid adding
# any unexpected kernel src in oneDNN folder
FILE(GLOB aten_core_src "core/*.cpp" "core/detail/*.cpp" "../oneDNN/Runtime.cpp")
FILE(GLOB aten_src "amp/*.cpp")
set(IPEX_ATEN_CORE_SRCS ${aten_core_src} ${aten_src} PARENT_SCOPE)

list(APPEND aten_core_headers
  ${IPEX_GPU_ATEN_CORE_DIR}/Allocator.h
  ${IPEX_GPU_ATEN_CORE_DIR}/AllocationInfo.h
  ${IPEX_GPU_ATEN_CORE_DIR}/Device.h
  ${IPEX_GPU_ATEN_CORE_DIR}/DeviceProp.h
  ${IPEX_GPU_ATEN_CORE_DIR}/Event.h
  ${IPEX_GPU_ATEN_CORE_DIR}/Generator.h
  ${IPEX_GPU_ATEN_CORE_DIR}/Guard.h
  ${IPEX_GPU_ATEN_CORE_DIR}/GuardImpl.h
  ${IPEX_GPU_ATEN_CORE_DIR}/Memory.h
  ${IPEX_GPU_ATEN_CORE_DIR}/Stream.h)
set(IPEX_ATEN_HEADERS ${aten_core_headers} PARENT_SCOPE)
