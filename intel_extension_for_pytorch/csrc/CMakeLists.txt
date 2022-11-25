set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fhonor-nans")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fhonor-infinities")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-associative-math")

add_subdirectory(cpu)
list(APPEND PY_CPP_OBJS $<TARGET_OBJECTS:PY_CPU_OBJ>)

if(BUILD_WITH_XPU)
  add_subdirectory(xpu)
  list(APPEND PY_CPP_OBJS $<TARGET_OBJECTS:PY_XPU_OBJ>)
endif()

add_library(intel-ext-pt-python SHARED ${PY_CPP_OBJS})

if(BUILD_STRIPPED_BIN)
  set_target_properties(intel-ext-pt-python PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

if(NOT BUILD_SKIP_CPU)
  target_link_libraries(intel-ext-pt-python PUBLIC intel-ext-pt-cpu)
endif()

if(BUILD_WITH_XPU)
  target_link_directories(intel-ext-pt-python PRIVATE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
  target_link_libraries(intel-ext-pt-python PUBLIC intel-ext-pt-gpu)
endif()

install(TARGETS intel-ext-pt-python
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
