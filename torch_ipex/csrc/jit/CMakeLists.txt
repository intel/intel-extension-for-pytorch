LIST(APPEND DPCPP_JIT_SRCS
    ${DPCPP_ROOT}/jit/fusion_pass.cpp
    ${DPCPP_ROOT}/jit/register_dnnl_jit_ops.cpp
    ${DPCPP_ROOT}/jit/graph_rewrite.cpp

)

# Pass to parent
set(DPCPP_JIT_SRCS ${DPCPP_JIT_SRCS} PARENT_SCOPE)
