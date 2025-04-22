FILE(GLOB mamba_src "causal_conv1d.cpp" "selective_scan_fwd.cpp")

set(IPEX_MAMBA_SRCS ${mamba_src} PARENT_SCOPE)
