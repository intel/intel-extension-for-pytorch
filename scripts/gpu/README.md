### We do not register all native function for DPCPP:GPU backend.
### We accord with current dispatch solution from PyTorch design, and register
### those functions which are backend specific.
### E.G.
### layer_norm, the Op provides common rountine for all backends.
### native_layer_norm, the Op is specific among backends.
###
### 1. Create SYCLType.h from DPCPP-GPU development repo.
### 2. Use SYCLType.h and RegistrationDeclarations.h to abstract registrations for DPCPP:GPU.

`python gen-gpu-decl.py --gpu_decl=./ ./DPCPPGPUType.h RegistrationDeclarations.h`
