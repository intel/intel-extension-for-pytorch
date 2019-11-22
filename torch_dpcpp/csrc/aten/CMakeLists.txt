FILE(GLOB aten_src "*.cpp")

list(APPEND DPCPP_SRCS ${aten_src})
message(${DPCPP_SRCS})
