#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathPointwise.h"
#else

THSYCL_API void THSYCLTensor_(sign)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);

THSYCL_API void THSYCLTensor_(cbitand)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(cbitor)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(cbitxor)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src1, THSYCLTensor* src2);

THSYCL_API void THSYCLTensor_(cmax)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(cmin)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(cmaxValue)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(cminValue)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, scalar_t value);

#if !defined(THSYCL_REAL_IS_BOOL)

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)

THSYCL_API void THSYCLTensor_(sigmoid)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(digamma)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(polygamma)(THSYCLState* state, THSYCLTensor* self_, int64_t n, THSYCLTensor* src);
THSYCL_API void THSYCLTensor_(erfinv)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(exp)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(expm1)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(log)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(log10)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(log1p)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(log2)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(cos)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(acos)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(cosh)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(sin)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(asin)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(sinh)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(tan)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(atan)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(atan2)(THSYCLState *state, THSYCLTensor *r_, THSYCLTensor *tx, THSYCLTensor *ty);
THSYCL_API void THSYCLTensor_(tanh)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(erf)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(erfc)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(sqrt)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(rsqrt)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(ceil)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(floor)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(round)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(trunc)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(frac)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(lerp)(THSYCLState *state, THSYCLTensor *result, THSYCLTensor *a, THSYCLTensor *b, scalar_t w);
THSYCL_API void THSYCLTensor_(cinv)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
#endif

THSYCL_API void THSYCLTensor_(neg)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(abs)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);

THSYCL_API void THSYCLTensor_(clamp)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, scalar_t min_value, scalar_t max_value);
THSYCL_API void THSYCLTensor_(cfmod)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2);

THSYCL_API void THSYCLTensor_(cremainder)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2);

THSYCL_API void THSYCLTensor_(addcmul)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor* t, scalar_t value, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(addcdiv)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor* t, scalar_t value, THSYCLTensor *src1, THSYCLTensor *src2);

THSYCL_API void THSYCLTensor_(cmul)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(cpow)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(pow)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(tpow)(THSYCLState *state, THSYCLTensor *self_, scalar_t value, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(clshift)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src1, THSYCLTensor* src2);
THSYCL_API void THSYCLTensor_(crshift)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src1, THSYCLTensor* src2);

#endif

#endif
