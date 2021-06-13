FILE(GLOB aten_functions_src "operators/*.cpp")

# Reset the order of src files to compile slow files first
list(APPEND aten_slow_src
  LogicalXor.cpp
  TopK.cpp
  UnaryOps_1.cpp
  UnaryOps_2.cpp
  UnaryOps_3.cpp
  UnaryOps_4.cpp)
foreach (sfile ${aten_functions_src})
  set(find_in_slow_list FALSE)
  foreach (ssrc ${aten_slow_src})
    string(FIND ${sfile} ${ssrc} pos)
    if (${pos} GREATER -1)
      set(find_in_slow_list TRUE)
      list(APPEND _SLOW_SRCS ${sfile})
      break()
    endif()
  endforeach()
  if (NOT find_in_slow_list)
    list(APPEND _FAST_SRCS ${sfile})
  endif()
endforeach()
list(APPEND aten_ops_src ${_SLOW_SRCS} ${_FAST_SRCS})
set(IPEX_ATEN_OPS_SRCS ${aten_ops_src} PARENT_SCOPE)

FILE(GLOB aten_core_src "core/*.cpp" "core/detail/*.cpp")
set(IPEX_ATEN_CORE_SRCS ${aten_core_src} PARENT_SCOPE)
