### We do not register all native function for DPCPP:GPU backend.
### We accord with current dispatch solution from PyTorch design, and register
### those functions which are backend specific.
### E.G.
### layer_norm, the Op provides common rountine for all backends.
### native_layer_norm, the Op is specific among backends.
###
### 1. Create SYCLType.h from DPCPP-GPU development repo.
### 2. Use SYCLType.h and RegistrationDeclarations.h to abstract registrations for DPCPP:GPU.

### STEP1
`python gen-gpu-decl.py --gpu_decl=./ DPCPPGPUType.h DedicateType.h DispatchStubOverride.h RegistrationDeclarations.h`

### STEP2
`python gen-gpu-ops.py --output_folder=./ DPCPPGPUType.h RegistrationDeclarations_DPCPP.h Functions_DPCPP.h`

### OUTPUTS
aten_ipex_type_default.cpp.in // registers
aten_ipex_type_default.h.in // exposed APIs
aten_ipex_type_dpcpp.h.in // dpcpp type Ops APIs
