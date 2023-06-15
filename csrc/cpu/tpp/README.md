# Overview
This directory mainly includes the tpp related tools and fused kernel implementation based on tpp optimization. 

```
├── bert #fused kernel based on tpp 
│   ├── fused_bert.cpp
│   ├── fused_dense_dropout_layernorm_bwd_tmpl.h #backard for fused linear+dropout+layernorm
│   ├── fused_dense_dropout_layernorm_fwd_tmpl.h #forward for fused linear+dropout+layernorm
│   ├── fused_dense_gelu_bwd_tmpl.h #backward for fused linear+gelu
│   ├── fused_dense_gelu_fwd_tmpl.h #forward for fused linear+gelu
│   ├── fused_embedding_layernorm_dropout_bwd_tmpl.h #forward for fused embeeding+add+layernorm+dropout 
│   ├── fused_embedding_layernorm_dropout_fwd_tmpl.h #backard for fused embeeding+add+layernorm+dropout 
│   ├── fused_self_attention_bwd_tmpl.h #fused backward self-attention 
│   └── fused_self_attention_fwd_tmpl.h #fused forward self-attention
├── CMakeLists.txt
├── common_loops.cpp #loops generation and tuning 
├── ext_tpp.h
├── init.cpp
├── jit_compile.cpp
├── jit_compile.h
├── optim.cpp
├── optim.h
├── par_loop_generator.cpp #loops generation and tuning 
├── par_loop_generator.h #loops generation and tuning 
├── rtm.h
├── tensor_helper.h
├── threaded_loops.h
├── timing.h
├── utils.h
└── xsmm_functors.h #the tpp definition based on libxsmm
```
