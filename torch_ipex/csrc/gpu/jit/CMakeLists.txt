FILE(GLOB jit_src "fusion_pass.cpp" "graph_ext.cpp" "register_dnnl_jit_ops.cpp" "dpcpp_ops.cpp")

set(DPCPP_JIT_SRCS ${jit_src} PARENT_SCOPE)
