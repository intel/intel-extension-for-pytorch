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
list(APPEND aten_reorg_src ${_SLOW_SRCS} ${_FAST_SRCS})

FILE(GLOB aten_core_src "utils/*.cpp" "core/*.cpp" "core/detail/*.cpp")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/core" "${CMAKE_CURRENT_SOURCE_DIR}/utils"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

set(aten_src)
list(APPEND aten_src ${aten_reorg_src})
list(APPEND aten_src ${aten_core_src})
list(APPEND aten_src ${gpu_generated_src})

set(IPEX_ATEN_SRCS ${aten_src} PARENT_SCOPE)
