#ifndef THSYCLNN_COMMON_H
#define THSYCLNN_COMMON_H


#define THSYCLNN_assertSameGPU(...) THAssertMsg(THSYCLTensor_(checkGPU)(__VA_ARGS__), \
      "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

#define THSYCLNN_check_dim_size(STATE, T, DIM, DIM_SIZE, SIZE) \
    if (THSYCLTensor_(nDimensionLegacyNoScalars)(STATE, T) != DIM ||             \
              THSYCLTensor_(sizeLegacyNoScalars)(STATE, T, DIM_SIZE) != SIZE) {        \
            THSYCLDescBuff s1 = THSYCLTensor_(sizeDesc)(state, T);       \
            THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"        \
                              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
        }

#define THSYCLNN_check_dim_size_indices(STATE, T, DIM, DIM_SIZE, SIZE)  \
    if (THSYCLIndexTensor_(nDimensionLegacyNoScalars)(STATE, T) != DIM ||                 \
              THSYCLIndexTensor_(sizeLegacyNoScalars)(STATE, T, DIM_SIZE) != SIZE) {            \
            THSYCLDescBuff s1 = THSYCLIndexTensor_(sizeDesc)(state, T);           \
            THError("Need " #T " of dimension %d and " #T ".size[%d] == %d" \
                              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
        }





#endif
