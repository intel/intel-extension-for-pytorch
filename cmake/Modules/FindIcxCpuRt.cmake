if(ICX_CPU_RT_FOUND)
  return()
endif()

set(ICX_CPU_RT_FOUND OFF)
set(INTEL_ICX_RT_LIBS "")

function(get_intel_compiler_rt_list libpath_list)
  if(MSVC)
    message( FATAL_ERROR "Not support Windows now." )
  else()
    set(intel_rt_list "libiomp5.so" "libintlc.so" "libintlc.so.5" "libimf.so" "libsvml.so" "libirng.so")
    set(libimf_name "libimf.so")
  endif()

  set(intel_rt_path_list "")
  execute_process(
      COMMAND bash "-c" "${CMAKE_CXX_COMPILER}  --print-file-name=${libimf_name}"
      OUTPUT_VARIABLE intel_imf_path
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  get_filename_component(intel_compiler_rt_install_dir "${intel_imf_path}" DIRECTORY)
  foreach(lib  ${intel_rt_list}) 
    list(APPEND intel_rt_path_list ${intel_compiler_rt_install_dir}/${lib})
  endforeach()
  set(${libpath_list} "${intel_rt_path_list}" PARENT_SCOPE)
endfunction()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
  get_intel_compiler_rt_list(INTEL_ICX_RT_LIBS)
  foreach (intel_lib_item ${INTEL_ICX_RT_LIBS})
    message("Found Intel icx runtime lib: ${intel_lib_item}")
  endforeach()

  string(COMPARE EQUAL "${INTEL_ICX_RT_LIBS}" "" result_empty)
  if(NOT result_empty)
    set(ICX_CPU_RT_FOUND ON)
    message("Intel icx cpu runtime found.")
  endif()
endif()