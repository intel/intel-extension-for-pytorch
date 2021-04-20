#include <immintrin.h>
#include <iostream>


static inline void add_ker(int8_t *inout, int8_t *in, int64_t len) {
    for (int64_t i = 0; i < len; ++i) {
        inout[i] += in[i];
    }
}

static inline void zero_ker(int8_t *out, int64_t len) {
    for (int64_t i = 0; i < len; ++i) {
        out[i] = 0;
    }
}
