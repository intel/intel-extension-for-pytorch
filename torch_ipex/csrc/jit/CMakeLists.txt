LIST(APPEND DPCPP_JIT_SRCS
    ${DPCPP_ROOT}/jit/codegen/onednn/subgraph_dtype_setter.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/graph_fuser.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/graph_helper.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/interface.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/kernel.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/defer_size_check.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/layout_propagation.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/prepare_binary.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/prepare_dequant.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/runtime.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/fusion_group_name.cpp
    ${DPCPP_ROOT}/jit/fusion_pass.cpp
    ${DPCPP_ROOT}/jit/register_dnnl_jit_ops.cpp
    ${DPCPP_ROOT}/jit/graph_rewrite.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/register_interface.cpp
    ${DPCPP_ROOT}/jit/codegen/onednn/guard_shape.cpp
)

# Pass to parent
set(DPCPP_JIT_SRCS ${DPCPP_JIT_SRCS} PARENT_SCOPE)
