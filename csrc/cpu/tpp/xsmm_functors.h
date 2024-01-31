#ifndef _XSMM_FUNCTORS_H_
#define _XSMM_FUNCTORS_H_

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <string>
#include <unordered_map>

namespace torch_ipex {
namespace tpp {

#define PCL_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)
#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))

typedef at::BFloat16 bfloat16;
typedef at::Half half;
typedef struct bfloat8 {
  uint8_t data;
} bfloat8;
inline float upconvert_to_float(float val) {
  return val;
}
inline float upconvert_to_float(bfloat16 val) {
  return (float)val;
}
inline float upconvert_to_float(half val) {
  return (float)val;
}
template <typename T>
inline libxsmm_datatype XsmmDtype();
template <>
inline libxsmm_datatype XsmmDtype<int64_t>() {
  return LIBXSMM_DATATYPE_I64;
}
template <>
inline libxsmm_datatype XsmmDtype<int32_t>() {
  return LIBXSMM_DATATYPE_I32;
}
template <>
inline libxsmm_datatype XsmmDtype<int8_t>() {
  return LIBXSMM_DATATYPE_I8;
}
template <>
inline libxsmm_datatype XsmmDtype<uint8_t>() {
  return LIBXSMM_DATATYPE_U8;
}
template <>
inline libxsmm_datatype XsmmDtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}
template <>
inline libxsmm_datatype XsmmDtype<half>() {
  return LIBXSMM_DATATYPE_F16;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat8>() {
  return LIBXSMM_DATATYPE_BF8;
}

#ifdef __AVX512F__
inline __m512 _mm512_loadu_ps_auto(float const* mem_addr) {
  return _mm512_loadu_ps(mem_addr);
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, float const* mem_addr) {
  return _mm512_maskz_loadu_ps(k, mem_addr);
}
inline void _mm512_storeu_ps_auto(float* mem_addr, __m512 a) {
  _mm512_storeu_ps(mem_addr, a);
}
inline void _mm512_mask_storeu_ps_auto(float* mem_addr, __mmask16 k, __m512 a) {
  _mm512_mask_storeu_ps(mem_addr, k, a);
}

inline __m512 _mm512_loadu_ps_auto(half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(half* mem_addr, __m512 a) {
  _mm256_storeu_si256(
      (__m256i*)mem_addr,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
inline void _mm512_mask_storeu_ps_auto(half* mem_addr, __mmask16 k, __m512 a) {
  _mm256_mask_storeu_epi16(
      (__m256i*)mem_addr,
      k,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline __m512 _mm512_convert_bf_ps(__m256i a) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a), 16));
}
inline __m256i _mm256_convert_ps_bf(__m512 a) {
  return _mm512_cvtepi32_epi16(
      _mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a), 16));
}

inline __m512 _mm512_loadu_ps_auto(bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(
    __mmask16 k,
    bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat16* mem_addr, __m512 a) {
  _mm256_storeu_si256((__m256i*)mem_addr, _mm256_convert_ps_bf(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat16* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm256_mask_storeu_epi16((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a));
}

inline __m512 _mm512_split_loadu_ps(bfloat16 const* hi, bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline __m512 _mm512_maskz_split_loadu_ps(
    __mmask16 k,
    bfloat16 const* hi,
    bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline void _mm512_split_storeu_ps(bfloat16* hi, bfloat16* lo, __m512 a) {
  //_mm512_storeu_ps_auto(hi, a);
  _mm256_storeu_si256(
      (__m256i*)hi,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_storeu_si256(
      (__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline void _mm512_mask_split_storeu_ps(
    bfloat16* hi,
    bfloat16* lo,
    __mmask16 k,
    __m512 a) {
  //_mm512_mask_storeu_ps_auto(hi, k, a);
  _mm256_mask_storeu_epi16(
      (__m256i*)hi,
      k,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_mask_storeu_epi16(
      (__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline __m512 _mm512_convert_bf8_ps(__m128i a) {
  return _mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepi8_epi16(a), 8));
}
inline __m128i _mm_convert_ps_bf8(__m512 a) {
  return _mm256_cvtepi16_epi8(_mm256_srai_epi16(
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 8));
}

inline __m512 _mm512_loadu_ps_auto(bfloat8 const* mem_addr) {
  return _mm512_convert_bf8_ps(_mm_loadu_si128((__m128i const*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, bfloat8 const* mem_addr) {
  return _mm512_convert_bf8_ps(
      _mm_maskz_loadu_epi8(k, (__m128i const*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat8* mem_addr, __m512 a) {
  _mm_storeu_si128((__m128i*)mem_addr, _mm_convert_ps_bf8(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat8* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm_mask_storeu_epi8((__m128i*)mem_addr, k, _mm_convert_ps_bf8(a));
}
#endif

inline void debug_print_eqn_tree(libxsmm_blasint eqn_no) {
  if (false) {
    libxsmm_matrix_eqn_tree_print(eqn_no);
    libxsmm_matrix_eqn_rpn_print(eqn_no);
  }
}

inline int xsmm_get_vnni_block_size(libxsmm_datatype dtype) {
  int bs = libxsmm_cpuid_dot_pack_factor(dtype);
  if (bs <= 0) {
    throw std::invalid_argument("Unsupported datatype");
  }
  return bs;
}
inline libxsmm_datatype convert_dtype_pt2xsmm(at::ScalarType dtype) {
  static const std::map<at::ScalarType, libxsmm_datatype> pt2xsmmDtypes = {
      {at::kDouble, LIBXSMM_DATATYPE_F64},
      {at::kFloat, LIBXSMM_DATATYPE_F32},
      {at::kHalf, LIBXSMM_DATATYPE_F16},
      {at::kBFloat16, LIBXSMM_DATATYPE_BF16},
      {at::kByte, LIBXSMM_DATATYPE_I8},
      {at::kChar, LIBXSMM_DATATYPE_I8},
      {at::kShort, LIBXSMM_DATATYPE_I16},
      {at::kInt, LIBXSMM_DATATYPE_I32},
      {at::kLong, LIBXSMM_DATATYPE_I64}};

  return pt2xsmmDtypes.at(dtype);
}
inline int get_vnni_block_size(at::ScalarType dtype) {
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

template <typename T>
inline int get_vnni_block_size() {
  auto xsmm_dtype = XsmmDtype<T>();
  return xsmm_get_vnni_block_size(xsmm_dtype);
}
inline int meqn_push_arg(
    const libxsmm_blasint idx,
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint ld,
    const libxsmm_blasint in_pos,
    const libxsmm_blasint offs_in_pos,
    const libxsmm_datatype dtype) {
  // This "singular" type dictates that the arg is a regular tensor (and not a
  // set of tensors)
  libxsmm_matrix_arg_attributes arg_singular_attr =
      libxsmm_create_matrix_arg_attributes(
          LIBXSMM_MATRIX_ARG_TYPE_SINGULAR,
          LIBXSMM_MATRIX_ARG_SET_TYPE_NONE,
          0,
          0);
  // Arg metadata include equation id and pos in arg array at runtime
  libxsmm_matrix_eqn_arg_metadata arg_metadata =
      libxsmm_create_matrix_eqn_arg_metadata(idx, in_pos);
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, ld, dtype);
  return libxsmm_matrix_eqn_push_back_arg_v2(
      arg_metadata, arg_shape, arg_singular_attr);
}

inline libxsmm_matrix_eqn_function meqn_dispatch(
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint* ldo,
    const libxsmm_datatype out_type,
    const unsigned int idx) {
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, *ldo, out_type);
  return libxsmm_dispatch_matrix_eqn_v2(idx, arg_shape);
}

inline int meqn_push_unary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_unary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  // OP metadata include equation id and an integer dictating where the op
  // metadata at runtime (if any) are located in the op arg array. -1 dictates
  // there are no op metadata needed
  libxsmm_matrix_eqn_op_metadata op_metadata =
      libxsmm_create_matrix_eqn_op_metadata(idx, -1);
  return libxsmm_matrix_eqn_push_back_unary_op_v2(
      op_metadata, type, dtype, flags);
}
inline int meqn_push_binary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_binary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_BINARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_matrix_eqn_op_metadata op_metadata =
      libxsmm_create_matrix_eqn_op_metadata(idx, -1);
  return libxsmm_matrix_eqn_push_back_binary_op_v2(
      op_metadata, type, dtype, flags);
}
inline int meqn_push_ternary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_ternary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_TERNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_matrix_eqn_op_metadata op_metadata =
      libxsmm_create_matrix_eqn_op_metadata(idx, -1);
  return libxsmm_matrix_eqn_push_back_ternary_op_v2(
      op_metadata, type, dtype, flags);
}

template <int N>
inline uint64_t string_to_hash_int(
    const std::string& str,
    const std::array<int, N>& params) {
  // Using FNV-1a algorithm
  uint64_t hash_value = 14695981039346656037ULL; // Initial hash value
  // Hash the string
  for (char c : str) {
    hash_value ^= static_cast<uint64_t>(c);
    hash_value *= 1099511628211ULL; // FNV prime
  }
  // Hash the vector of integers
  for (int intValue : params) {
    hash_value ^= static_cast<uint64_t>(intValue);
    hash_value *= 1099511628211ULL; // FNV prime
  }
  return hash_value;
}

class BaseTPP {
 public:
  void* get_kernel() {
    auto& kernel_cache = get_kernel_cache();
    void* kernel = NULL;
    if (hash == 0)
      hash = hash_int();
    auto search = kernel_cache.find(hash);
    if (search != kernel_cache.end())
      kernel = search->second;
    if (kernel == NULL) {
      kernel = build_kernel();
      if (kernel == NULL) {
        print_error();
        exit(1);
      }
      // printf("TPP: %s @ %p\n", hash.c_str(), kernel);
      kernel_cache[hash] = kernel;
    }
    return kernel;
  }

 protected:
  std::unordered_map<uint64_t, void*>& get_kernel_cache() {
    static std::unordered_map<uint64_t, void*> kernel_cache;
    return kernel_cache;
  }
  virtual uint64_t hash_int() = 0;
  virtual void* build_kernel() = 0;
  virtual void print_error() = 0;
  uint64_t hash = 0;
  bool initialized = false;
};

class UnaryTPP : public BaseTPP {
 public:
  UnaryTPP() {}
  UnaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_unary_type type)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        dt_in(dt_in),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_unary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    kernel(&unary_param);
  }
  void operator()(void* in, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }
  void operator()(void* in, void* in2, void* in3, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

  void operator()(
      void* in,
      void* in2,
      void* in3,
      void* op,
      void* op2,
      void* op3,
      void* out,
      void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.op.primary = op;
    unary_param.op.secondary = op2;
    unary_param.op.tertiary = op3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

 protected:
  uint64_t hash_int() override {
    std::array<int, 9> params = {
        rows, cols, ldi, ldo, dt_in, dt_out, dt_compute, (int)flags, type};
    uint64_t hash_value = string_to_hash_int<9>("unary", params);
    return hash_value;
  }
  void* build_kernel() override {
    libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
        cols, rows, ldi, ldo, dt_in, dt_out, dt_compute);
    return (void*)libxsmm_dispatch_meltw_unary_v2(type, shape, flags);
  }
  void print_error() override {
    fprintf(
        stderr,
        "Unable to get JIT kernel for unary. Params: rows=%d, cols=%d, ldi=%d, ldo=%d, dt_in=%d, dt_out=%d, dt_compute=%d, flags=%d, type=%d\n",
        rows,
        cols,
        ldi,
        ldo,
        dt_in,
        dt_out,
        dt_compute,
        (int)flags,
        type);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi = 0;
  libxsmm_blasint ldo = 0;
  libxsmm_datatype dt_in = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_out = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_compute = LIBXSMM_DATATYPE_F32;
  libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_meltwfunction_unary kernel = NULL;
};

class BinaryTPP : public BaseTPP {
 public:
  BinaryTPP() {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : BinaryTPP(
            rows,
            cols,
            ldi,
            ldi,
            ldo,
            dt_in,
            dt_in,
            dt_out,
            dt_compute,
            flags,
            type) {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi0,
      libxsmm_blasint ldi1,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in0,
      libxsmm_datatype dt_in1,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldo(ldo),
        dt_in0(dt_in0),
        dt_in1(dt_in1),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_binary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in0, void* in1, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_binary_param binary_param;
    binary_param.in0.primary = in0;
    binary_param.in1.primary = in1;
    binary_param.out.primary = out;
    kernel(&binary_param);
  }

 protected:
  uint64_t hash_int() override {
    std::array<int, 11> params = {
        rows,
        cols,
        ldi0,
        ldi1,
        ldo,
        dt_in0,
        dt_in1,
        dt_out,
        dt_compute,
        (int)flags,
        type};
    uint64_t hash_value = string_to_hash_int<11>("binary", params);
    return hash_value;
  }
  void* build_kernel() override {
    libxsmm_meltw_binary_shape shape = libxsmm_create_meltw_binary_shape(
        cols, rows, ldi0, ldi1, ldo, dt_in0, dt_in1, dt_out, dt_compute);
    return (void*)libxsmm_dispatch_meltw_binary_v2(type, shape, flags);
  }
  void print_error() override {
    fprintf(
        stderr,
        "Unable to get JIT kernel for binary. Params: rows=%d, cols=%d, ldi0=%d, ldi1=%d, ldo=%d, dt_in0=%d, dt_in1=%d, dt_out=%d, dt_compute=%d, flags=%d, type=%d\n",
        rows,
        cols,
        ldi0,
        ldi1,
        ldo,
        dt_in0,
        dt_in1,
        dt_out,
        dt_compute,
        (int)flags,
        type);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi0;
  libxsmm_blasint ldi1;
  libxsmm_blasint ldo;
  libxsmm_datatype dt_in0;
  libxsmm_datatype dt_in1;
  libxsmm_datatype dt_out;
  libxsmm_datatype dt_compute;
  libxsmm_bitfield flags;
  libxsmm_meltw_binary_type type;
  libxsmm_meltwfunction_binary kernel = NULL;
};

template <typename T>
class SetZeroTPP {
 public:
  SetZeroTPP() {}
  SetZeroTPP(int N) : SetZeroTPP(1, N) {}
  SetZeroTPP(int rows, int cols) : SetZeroTPP(rows, cols, cols) {}
  SetZeroTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldo,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_XOR) {}
  void operator()(T* buf) {
    kernel((void*)buf, (void*)buf);
  }
  void ref(T* buf) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        buf[i * ldo + j] = 0;
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class ConvertTPP {
 public:
  ConvertTPP() {}
  ConvertTPP(int N) : ConvertTPP(1, N) {}
  ConvertTPP(int rows, int cols) : ConvertTPP(rows, cols, cols, cols) {}
  ConvertTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY),
        init_done(true) {}
  void operator()(Tin* in, Tout* out) {
    if (!(XsmmDtype<Tin>() == LIBXSMM_DATATYPE_F32 &&
          XsmmDtype<Tout>() == LIBXSMM_DATATYPE_F32) ||
        ((void*)in != (void*)out))
      kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i * ldi + j];
      }
    }
  }
  bool initialized() {
    return init_done;
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
  bool init_done = false;
};

template <typename T>
class CpyTPP {
 public:
  CpyTPP() {}
  CpyTPP(int N) : CpyTPP(1, N) {}
  CpyTPP(int rows, int cols) : CpyTPP(rows, cols, cols, cols) {}
  CpyTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename T>
class PadTPP {
 public:
  PadTPP() {}
  PadTPP(int in_rows, int in_cols, int out_rows, int out_cols)
      : PadTPP(in_rows, in_cols, out_rows, out_cols, in_cols, out_cols) {}
  PadTPP(int in_rows, int in_cols, int out_rows, int out_cols, int ldi, int ldo)
      : in_rows(in_rows),
        in_cols(in_cols),
        out_rows(out_rows),
        out_cols(out_cols),
        ldi(ldi),
        ldo(ldo),
        cpy(),
        zero() {
    if (out_rows > in_rows || out_cols > in_cols) {
      PCL_ASSERT(
          out_rows == in_rows || out_cols == in_cols,
          "PadTPP can pad only 1 dim at a time");
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, ldo);
      if (out_rows > in_rows) {
        zero = SetZeroTPP<T>(out_rows - in_rows, out_cols, ldo);
        zero_offset = in_rows * ldo;
      } else {
        zero = SetZeroTPP<T>(out_rows, out_cols - in_cols, ldo);
        zero_offset = in_cols;
      }
    }
  }
  void operator()(T* in, T* out) {
    cpy(in, out);
    zero(out);
  }
  void ref(T* in, T* out) {
    cpy.ref(in, out);
    zero.ref(out);
  }

 private:
  int in_rows = 0;
  int in_cols = 0;
  int out_rows = 0;
  int out_cols = 0;
  int ldi;
  int ldo;
  int zero_offset = 0;
  CpyTPP<T> cpy;
  SetZeroTPP<T> zero;
};

template <typename Tin, typename Tout = Tin>
class CpyBiasTPP {
 public:
  CpyBiasTPP() {}
  CpyBiasTPP(int rows, int cols) : CpyBiasTPP(rows, cols, cols) {}
  CpyBiasTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            cols,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBcastTPP {
 public:
  CpyBcastTPP() {}
  CpyBcastTPP(int rows, int cols) : CpyBcastTPP(rows, cols, cols) {}
  CpyBcastTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};
template <typename T>
class AddBiasTPP {
 public:
  AddBiasTPP() {}
  AddBiasTPP(int rows, int cols) : AddBiasTPP(rows, cols, cols) {}
  AddBiasTPP(int rows, int cols, int ld)
      : rows(rows),
        cols(cols),
        ld(ld),
        kernel(
            rows,
            cols,
            ld,
            ld,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_ADD),
        cvt() {
    if (!std::is_same<T, float>::value)
      cvt = ConvertTPP<T, float>(1, cols);
  }
  void operator()(T* in, float* out) {
    if (std::is_same<T, float>::value) {
      kernel((void*)in, (void*)out, (void*)out);
    } else {
      float tmp[cols];
      cvt(in, tmp);
      kernel((void*)tmp, (void*)out, (void*)out);
    }
  }
  void ref(T* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ld + c] += (float)in[c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ld;
  BinaryTPP kernel;
  ConvertTPP<T, float> cvt;
};

template <typename Tin, typename Tout = Tin>
class AddTPP {
 public:
  AddTPP() {}
  AddTPP(int N) : AddTPP(1, N) {}
  AddTPP(int rows, int cols) : AddTPP(rows, cols, cols, cols) {}
  AddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_ADD) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] + (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin>
class GradBiasTPP {
 public:
  GradBiasTPP() {}
  GradBiasTPP(int rows, int cols) : GradBiasTPP(rows, cols, cols) {}
  GradBiasTPP(int rows, int cols, int ldi)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(cols) {}
  void operator()(Tin* in, float* out) {
    float tmp[cols];
    reduce((void*)in, (void*)tmp);
    add(tmp, out, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;

  UnaryTPP reduce;
  AddTPP<float, float> add;
};

template <typename T1, typename T2 = T1, typename T3 = T1>
class MulReduceTPP : public BaseTPP {
 public:
  MulReduceTPP() {}
  MulReduceTPP(int N, int M) : N(N), M(M) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }

  void operator()(T1* in0, T2* in1, T3* out) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[2];
    arg_array[0].primary = (void*)in0;
    arg_array[1].primary = (void*)in1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)out;

    kernel(&eqn_param);
  }

  void ref(T1* in0, T2* in1, T3* out) {
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < M; c++) {
        out[r] += (float)in0[r * M + c] * (float)in1[r * M + c];
      }
    }
  }

 protected:
  uint64_t hash_int() override {
    std::array<int, 5> params = {
        N, M, XsmmDtype<T1>(), XsmmDtype<T2>(), XsmmDtype<T3>()};
    uint64_t hash_value = string_to_hash_int<5>("mul_reduce_eqn", params);
    return hash_value;
  }
  void* build_kernel() override {
    auto dt1 = XsmmDtype<T1>();
    auto dt2 = XsmmDtype<T2>();
    auto dt3 = XsmmDtype<T3>();
    libxsmm_blasint ld = 1;
    libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
    meqn_push_unary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
        LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
        LIBXSMM_DATATYPE_F32);
    // libxsmm_matrix_eqn_push_back_arg(my_eqn0, M, N, M, 0, 0, dt1);
    meqn_push_binary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_BINARY_MUL,
        LIBXSMM_MELTW_FLAG_BINARY_NONE,
        LIBXSMM_DATATYPE_F32);
    meqn_push_arg(my_eqn0, M, N, M, 0, 0, dt1);
    meqn_push_arg(my_eqn0, M, N, M, 1, 0, dt2);
    debug_print_eqn_tree(my_eqn0);
    return (void*)meqn_dispatch(1, N, &ld, dt3, my_eqn0);
  }
  void print_error() override {
    fprintf(
        stderr,
        "Unable to get JIT kernel for mul_reduce_eqn. Params: N=%d, M=%d, dt1=%d, dt2=%d, dt3=%d\n",
        N,
        M,
        XsmmDtype<T1>(),
        XsmmDtype<T2>(),
        XsmmDtype<T3>());
  }

 private:
  int N = 0;
  int M = 0;
  libxsmm_matrix_eqn_function kernel = NULL;
};

template <typename Tin, typename Tout = Tin>
class ReduceAddColTPP {
 public:
  ReduceAddColTPP() {}
  ReduceAddColTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        reduce(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {}
  void operator()(Tin* in, float* out) {
    reduce(in, out);
  }
  void ref(Tin* in, float* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (r == 0)
          out[c] = 0;
        out[c] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi, ldo;

  UnaryTPP reduce;
};

template <typename Tin, typename Tout = Tin>
class ReduceAddRowTPP {
 public:
  ReduceAddRowTPP() {}
  ReduceAddRowTPP(int rows, int cols, bool acc)
      : ReduceAddRowTPP(rows, cols, cols, acc) {}
  ReduceAddRowTPP(int rows, int cols, int ldi, bool acc)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        acc(acc),
        reduce(
            rows,
            cols,
            ldi,
            cols,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        add(rows) {}
  void operator()(Tin* in, Tout* out) {
    if (acc) {
      Tout tmp[rows];
      reduce((void*)in, (void*)tmp);
      add(tmp, out, out);
    } else {
      reduce((void*)in, (void*)out);
    }
  }
  void ref(Tin* in, Tout* out) {
    for (int r = 0; r < rows; r++) {
      if (!acc) {
        out[r] = 0;
      }
      for (int c = 0; c < cols; c++) {
        out[r] += (float)in[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  bool acc;
  UnaryTPP reduce;
  AddTPP<Tout, Tout> add;
};

template <typename Tin, typename Tout = Tin>
class MulTPP {
 public:
  MulTPP() {}
  MulTPP(int N) : MulTPP(1, N) {}
  MulTPP(int rows, int cols) : MulTPP(rows, cols, cols, cols) {}
  MulTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] * (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class BCastMulTPP {
 public:
  BCastMulTPP() {}
  BCastMulTPP(int rows, int cols) : BCastMulTPP(rows, cols, cols, cols) {}
  BCastMulTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0, // Broadcast in Row
                                                      // Dimension
            LIBXSMM_MELTW_TYPE_BINARY_MUL) // Multiplication
  {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * ldo + r] = (Tin)in0[r] * in1[c * ldi + r];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

// ############################# Broadcast & Multiplication Addition TPP
// #####################################
template <typename Tin, typename Tout = Tin>
class BCastMulAddTPP {
 public:
  BCastMulAddTPP() {}
  BCastMulAddTPP(int rows, int cols) : BCastMulAddTPP(rows, cols, cols, cols) {}
  BCastMulAddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0, // Broadcast in Row
                                                      // Dimension
            LIBXSMM_MELTW_TYPE_BINARY_MULADD) // Multiplication
  {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }

  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * ldo + r] += (Tin)in0[r] * (Tin)in1[c * ldi + r];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout>
class ScaleTPP {
 public:
  ScaleTPP() {}
  ScaleTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    Tin alpha = scale;
    for (int i = 0; i < N; i++) {
      out[i] = (float)in[i] * (float)alpha;
    }
  }

 private:
  int N = 0;
  BinaryTPP kernel;
};

template <typename T, typename TN = float>
class Norm2TPP {
 public:
  Norm2TPP() {}
  Norm2TPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) {}
  void operator()(T* in, TN* sum) {
    float lsum = 0.0f;
    kernel((void*)in, (void*)&lsum);
    *sum += (TN)lsum;
  }
  void ref(T* in, TN* sum) {
    float lsum = 0.0f;
    for (int i = 0; i < N; i++) {
      lsum += (float)in[i] * (float)in[i];
    }
    *sum += (TN)lsum;
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpTPP {
 public:
  RecpTPP() {}
  RecpTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / in[i];
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename T>
class RecpSqrtTPP {
 public:
  RecpSqrtTPP() {}
  RecpSqrtTPP(int N)
      : N(N),
        kernel(
            1,
            N,
            N,
            N,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < N; i++)
      out[i] = 1.0 / sqrt(in[i]);
  }

 private:
  int N = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MulNormTPP {
 public:
  MulNormTPP() {}
  MulNormTPP(int rows, int cols) : MulNormTPP(rows, cols, cols, cols) {}
  MulNormTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1, // ldi0
            ldi, // ldi1
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in, Tin* in2, Tout* out) {
    kernel((void*)in, (void*)in2, (void*)out);
  }
  void ref(Tin* in, Tin* in2, Tout* out) {
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        out[r * ldo + c] = in[r] * in2[r * ldi + c];
  }

 private:
  int rows, cols;
  int ldi, ldo;
  BinaryTPP kernel;
};

template <typename Tin, typename Tout>
class ScaleAddTPP {
 public:
  ScaleAddTPP() {}
  ScaleAddTPP(int N) : ScaleAddTPP(1, N) {}
  ScaleAddTPP(int rows, int cols) : ScaleAddTPP(rows, cols, cols, cols) {}
  ScaleAddTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            1,
            ldi,
            ldo,
            XsmmDtype<float>(),
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0,
            LIBXSMM_MELTW_TYPE_BINARY_MULADD) {}
  void operator()(Tin* in, Tout* out, float scale) {
    float alpha = scale;
    kernel((void*)&alpha, (void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out, float scale) {
    float alpha = scale;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] += (float)in[i * ldi + j] * (float)alpha;
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};

class XformTPP {
 public:
  XformTPP() {}
  XformTPP(
      libxsmm_blasint rows_i,
      libxsmm_blasint cols_i,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dtype,
      libxsmm_meltw_unary_type type)
      : rows(rows_i),
        cols(cols_i),
        ldi(ldi),
        ldo(ldo),
        dtype(dtype),
        type(type),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            dtype,
            dtype,
            dtype,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            type) {}
  void operator()(void* in, void* out) {
    kernel(in, out);
  }
  typedef enum XFORM_TYPE {
    XFORM_NONE_TPP = 0,
    XFORM_XPOSE_TPP = 1,
    XFORM_N2V_TPP = 2,
    XFORM_XPOSE_N2V_TPP = 3,
    XFORM_XPOSE_V2V_TPP = 4
  } XFORM_TYPE;

 private:
  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_datatype dtype;
  libxsmm_meltw_unary_type type;
  UnaryTPP kernel;
};

template <typename T>
class XformExtTPP {
 public:
  XformExtTPP() {}
  XformExtTPP(
      /* rows and cols as for input tensor */
      int rows,
      int cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            rows,
            cols,
            (xtype == XformTPP::XFORM_N2V_TPP ? rows : cols),
            (xtype == XformTPP::XFORM_N2V_TPP ? cols : rows),
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            in_rows,
            in_cols,
            out_rows,
            out_cols,
            in_cols,
            out_cols,
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      int ldi,
      int ldo,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : in_rows(in_rows),
        in_cols(in_cols),
        out_rows(out_rows),
        out_cols(out_cols),
        ldi(ldi),
        ldo(ldo),
        xtype(xtype),
        dtype(XsmmDtype<T>()),
        kernel(),
        cvt(),
        cpy(),
        zero() {
    libxsmm_meltw_unary_type unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
    if (ignore_vnni_for_fp32 == false) {
      PCL_ASSERT(
          (xtype == XformTPP::XFORM_XPOSE_TPP || dtype != LIBXSMM_DATATYPE_F32),
          "Only Transpose Xofrm supportd for FP32 datatype, specified %d\n",
          (int)xtype);
    }
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_N2V_TPP) {
      in_rows_p = out_rows;
      in_cols_p = out_cols;
      PCL_ASSERT(in_rows_p % BS == 0, "N2VTPP: unaligned number of rows\n");
      if (BS == 1) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
      } else if (BS == 2) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
      } else if (BS == 4) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
      } else {
        PCL_ASSERT(false, "N2VTPP: unsupported packing size (%d)\n", BS);
      }
    } else {
      in_rows_p = out_cols;
      in_cols_p = out_rows;
      if (dtype != LIBXSMM_DATATYPE_F32) {
        if (xtype == XformTPP::XFORM_XPOSE_TPP) {
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
        } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
          // unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT;
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
          PCL_ASSERT(
              in_cols_p % BS == 0, "XposeN2VTPP: uneven number of cols\n");
        } else {
          if (BS == 2) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T;
          } else if (BS == 4) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T;
          } else {
            PCL_ASSERT(false, "V2VTPP: unsupported packing size (%d)\n", BS);
          }
          PCL_ASSERT(in_rows % BS == 0, "XposeV2VTPP: uneven number of rows\n");
          PCL_ASSERT(
              in_cols_p % BS == 0, "XposeV2VTPP: uneven number of cols\n");
        }
      } else {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
      }
    }
    PCL_ASSERT(
        (in_rows_p >= in_rows && in_cols_p >= in_cols),
        "Invalid output rows or cols value\n");
    PCL_ASSERT(
        in_rows_p == in_rows || in_cols_p == in_cols,
        "Padding can only be done in rows or cols\n");

    if (xtype != XformTPP::XFORM_XPOSE_N2V_TPP) {
      int ld = (in_rows_p != in_rows || in_cols_p != in_cols) ? in_cols_p : ldi;
      kernel = XformTPP(in_rows_p, in_cols_p, ld, ldo, dtype, unary_type);
    } else {
      // LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT not implemented so use
      // workaround...
      kernel = XformTPP(
          in_rows_p,
          in_cols_p / BS,
          ldi / BS,
          ldo,
          ((dtype == LIBXSMM_DATATYPE_BF16 && BS == 4) ||
           (dtype == LIBXSMM_DATATYPE_BF8 && BS == 8))
              ? LIBXSMM_DATATYPE_F64
              : LIBXSMM_DATATYPE_F32,
          unary_type);
    }

    if ((xtype == XformTPP::XFORM_N2V_TPP ||
         xtype == XformTPP::XFORM_XPOSE_TPP) &&
        in_rows_p != in_rows) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols);
      zero = SetZeroTPP<T>(in_rows_p - in_rows, in_cols);
      zero_offset = in_rows * in_cols;
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP && in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols_p);
      zero = SetZeroTPP<T>(in_rows, in_cols_p - in_cols, in_cols_p);
      zero_offset = in_cols;
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP && in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows / BS, in_cols * BS, ldi * BS, in_cols_p * BS);
      zero = SetZeroTPP<T>(
          in_rows / BS, (in_cols_p - in_cols) * BS, in_cols_p * BS);
      zero_offset = in_cols * BS;
    }
    if (std::is_same<T, bfloat16>::value)
      cvt = ConvertTPP<float, bfloat16>(in_rows, in_cols);
  }
  void operator()(T* in, T* out) {
    if (in != out) {
      if (in_rows_p != in_rows || in_cols_p != in_cols) {
        T tmp[in_rows_p * in_cols_p];
        cpy(in, tmp);
        zero(tmp + zero_offset);
        kernel((void*)tmp, (void*)out);
      } else {
        kernel((void*)in, (void*)out);
      }
    }
  }
  void ref(T* in, T* out) {
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < in_rows / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols && i * BS + k < out_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      PCL_ASSERT(false, "Should not come here\n");
    }
  }

  void operator()(float* in, bfloat16* out) {
    bfloat16 tmp2[in_rows * in_cols];
    cvt(in, tmp2);
    if (in_rows_p != in_rows || in_cols_p != in_cols) {
      T tmp[in_rows_p * in_cols_p];
      cpy(tmp2, tmp);
      zero(tmp + zero_offset);
      kernel((void*)tmp, (void*)out);
    } else {
      kernel((void*)tmp2, (void*)out);
    }
  }
  void ref(float* in, bfloat16* out) {
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < out_cols / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      PCL_ASSERT(false, "Should not come here\n");
    }
  }
  void operator()(int count, int64_t str_in, int64_t str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  void ref(int count, int64_t str_in, int64_t str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }
  void operator()(
      int count,
      int64_t str_in,
      int64_t str_out,
      float* in,
      bfloat16* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  void ref(
      int count,
      int64_t str_in,
      int64_t str_out,
      float* in,
      bfloat16* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }

 private:
  libxsmm_blasint in_rows = 0;
  libxsmm_blasint in_cols = 0;
  libxsmm_blasint out_rows = 0;
  libxsmm_blasint out_cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int in_rows_p = 0;
  int in_cols_p = 0;
  XformTPP::XFORM_TYPE xtype;
  libxsmm_datatype dtype;
  int zero_offset = 0;
  XformTPP kernel;
  ConvertTPP<float, bfloat16> cvt;
  CpyTPP<T> cpy;
  SetZeroTPP<T> zero;
};

template <typename Tin, typename Tout>
class BrgemmTPP {
 public:
  BrgemmTPP() {}
  BrgemmTPP(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t str_a,
      int64_t str_b,
      float beta = 1.0,
      int a_trans = 0,
      int unroll_hint = 0)
      : BrgemmTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            beta,
            a_trans,
            unroll_hint) {}
  BrgemmTPP(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t str_a,
      int64_t str_b,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      float beta,
      int a_trans,
      int unroll_hint,
      int b_vnni = 1)
      : M(M),
        N(N),
        K(K),
        str_a(str_a),
        str_b(str_b),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        beta(beta),
        a_trans(a_trans),
        unroll_hint(unroll_hint),
        b_vnni(b_vnni),
        k_gemm_with_tc(this, 0),
        k_cfg(this, 1),
        k_rls(this, 2),
        k_gemm_no_tc(this, 3) {}
  void config() {
    k_cfg(NULL);
  }
  void release() {
    k_rls(NULL);
  }
  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      uint64_t count,
      bool no_tile_cfg = false) {
    libxsmm_gemm_param gemm_param;
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.op.tertiary = &count;
    gemm_param.c.primary = (void*)C;
    gemm_param.a.primary = (void*)B;
    gemm_param.b.primary = (void*)A;
    if (!no_tile_cfg) {
      k_gemm_with_tc(&gemm_param);
    } else {
      k_gemm_no_tc(&gemm_param);
    }
  }
  void ref(Tin* A, Tin* B, Tout* C, uint64_t count, bool no_tile_cfg = false) {
    auto dtype = XsmmDtype<Tin>();
    for (uint64_t c = 0; c < count; c++) {
      auto A_ = &A[c * str_a];
      auto B_ = &B[c * str_b];
      if (std::is_same<Tin, float>::value || b_vnni == 0) {
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            if (beta == 0.0 && c == 0)
              C[i * N + j] = 0.0;
            for (int k = 0; k < K; k++) {
              if (a_trans == 1) {
                C[i * ldc + j] += A_[k * lda + i] * B_[k * ldb + j];
              } else {
                C[i * ldc + j] += A_[i * lda + k] * B_[k * ldb + j];
              }
            }
          }
        }
      } else {
        const int BS = xsmm_get_vnni_block_size(dtype);
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            float sum =
                ((beta == 0.0 && c == 0) ? 0.0f : (float)C[i * ldc + j]);
            for (int k = 0; k < K / BS; k++) {
              for (int b = 0; b < BS; b++) {
                if (a_trans == 1) {
                  sum += (float)A_[k * lda * BS + i * BS + b] *
                      (float)B_[k * ldb * BS + j * BS + b];
                } else {
                  sum += (float)A_[i * lda + k * BS + b] *
                      (float)B_[k * ldb * BS + j * BS + b];
                }
              }
            }
            C[i * ldc + j] = (Tout)sum;
          }
        }
      }
    }
  }

  int64_t flops() {
    return 2L * M * N * K;
  }

  class BrgemmKernel : public BaseTPP {
   public:
    BrgemmKernel() {}
    BrgemmKernel(BrgemmTPP* p, int config) : p(p), config(config) {
      auto dt_in = XsmmDtype<Tin>();
      auto dt_out = XsmmDtype<Tout>();
      int64_t type = -1;
      if (dt_in == LIBXSMM_DATATYPE_F32) {
        PCL_ASSERT(dt_out == LIBXSMM_DATATYPE_F32, "BRGEMM Assert\n");
        type = 0;
      } else if (dt_out == LIBXSMM_DATATYPE_F32) {
        if (dt_in == LIBXSMM_DATATYPE_F16) {
          type = 1;
        } else if (dt_in == LIBXSMM_DATATYPE_BF16) {
          type = 2;
        }
      } else if (dt_in == LIBXSMM_DATATYPE_F16) {
        type = 3;
      } else {
        type = 4;
      }
      // if (type != 0)
      //   PCL_ASSERT(
      //       p->a_trans == 0, "A Transpose supported only for FP32 BRGEMM\n");
      brgemm_type = type;
      kernel.gemm = (libxsmm_gemmfunction)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_gemm_param* gemm_param) {
      if (!initialized)
        return;
      kernel.gemm(gemm_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 14> params = {
          p->M,
          p->N,
          p->K,
          p->str_a,
          p->str_b,
          brgemm_type,
          (int)p->beta,
          p->a_trans,
          p->unroll_hint,
          p->lda,
          p->ldb,
          p->ldc,
          config,
          p->b_vnni};
      uint64_t hash_value = string_to_hash_int<14>("brgemm", params);
      return hash_value;
    }
    void* build_kernel() override {
      // float alpha = 1.0;
      libxsmm_gemm_shape l_shape;
      libxsmm_gemm_batch_reduce_config l_brconfig;
      libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
      libxsmm_bitfield l_prefetch_flags = 0;
      libxsmm_xmmfunction l_test_jit = {NULL};

      if (p->a_trans == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
      if (brgemm_type != 0) {
        if (p->b_vnni)
          l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
        if (p->a_trans == 1) {
          l_flags |= LIBXSMM_GEMM_FLAG_VNNI_B;
        }
      }
      if (p->beta == 0)
        l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

      // config = 0 - normal
      // config = 1 - no tile release
      // config = 2 - no tile config
      // config = 3 - brgemm with no tile config or release
      if (config == 1) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
      } else if (config == 2) {
        l_flags |= LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
      } else if (config == 3) {
        l_flags |=
            (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
             LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
      }

      /* setting update GEMM struct */
      l_shape.m = p->N;
      l_shape.n = p->M;
      l_shape.k = p->K;
      l_shape.lda = p->ldb;
      l_shape.ldb = p->lda;
      l_shape.ldc = p->ldc;
      l_shape.a_in_type = XsmmDtype<Tin>();
      l_shape.b_in_type = XsmmDtype<Tin>();
      l_shape.comp_type = LIBXSMM_DATATYPE_F32;
      // TODO(jgong5): we should not always assume u8*i8 for int8 gemm
      if (std::is_same<Tin, int8_t>()) {
        l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED;
        l_shape.comp_type = LIBXSMM_DATATYPE_I32;
      }
      l_shape.out_type = XsmmDtype<Tout>();

      l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
      l_brconfig.br_stride_a_hint = p->str_b * sizeof(Tin);
      l_brconfig.br_stride_b_hint = p->str_a * sizeof(Tin);
      l_brconfig.br_unroll_hint = p->unroll_hint;

      l_test_jit.gemm = libxsmm_dispatch_brgemm_v2(
          l_shape, l_flags, l_prefetch_flags, l_brconfig);

      return (void*)l_test_jit.gemm;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for brgemm. Params: M=%lld, N=%lld, K=%lld, str_a=%lld, str_b=%lld, brgemm_type=%lld, beta=%d, a_trans=%d, unroll_hint=%d, lda=%d, ldb=%d, ldc=%d, config=%d, b_vnni=%d",
          p->M,
          p->N,
          p->K,
          p->str_a,
          p->str_b,
          brgemm_type,
          (int)p->beta,
          p->a_trans,
          p->unroll_hint,
          p->lda,
          p->ldb,
          p->ldc,
          config,
          p->b_vnni);
    }

   private:
    BrgemmTPP* p;
    int config;
    libxsmm_xmmfunction kernel;
    int64_t brgemm_type = -1;
  };

 private:
  int64_t M, N, K, str_a, str_b;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  float beta;
  int a_trans;
  int64_t brgemm_type = -1;
  int unroll_hint;
  int b_vnni;
  BrgemmKernel k_gemm_with_tc;
  BrgemmKernel k_cfg;
  BrgemmKernel k_rls;
  BrgemmKernel k_gemm_no_tc;
};

template <typename Tin, typename Tout = Tin>
class GeluFwdTPP {
 public:
  GeluFwdTPP() {}
  GeluFwdTPP(int N) : GeluFwdTPP(1, N) {}
  GeluFwdTPP(int M, int N) : GeluFwdTPP(M, N, N, N) {}
  GeluFwdTPP(int M, int N, int ldi, int ldo)
      : M(M),
        N(N),
        ldi(ldi),
        ldo(ldo),
        kernel(
            M,
            N,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_GELU) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
#ifdef __AVX512F__
    for (int j = 0; j < M; j++) {
      int i;
      for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
        auto vin = _mm512_loadu_ps_auto(&in[j * ldi + i]);
        // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
        auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
        _mm512_storeu_ps_auto(&out[j * ldo + i], vout);
      }
      if (i < N) {
        int rem = N - i;
        __mmask16 mask = (1 << rem) - 1;
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[j * ldi + i]);
        // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
        auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
        _mm512_mask_storeu_ps_auto(&out[j * ldo + i], mask, vout);
      }
    }
#else
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i++) {
        float x = in[j * ldi + i];
        out[j * ldo + i] = (erff(x / sqrtf(2.0)) + 1.0) * 0.5 * x;
      }
    }
#endif
  }

 private:
  int M = 0;
  int N = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class GeluTanhFwdTPP {
 public:
  GeluTanhFwdTPP() {}
  GeluTanhFwdTPP(int N) : GeluTanhFwdTPP(1, N) {}
  GeluTanhFwdTPP(int M, int N) : GeluTanhFwdTPP(M, N, N, N) {}
  GeluTanhFwdTPP(int M, int N, int ldi, int ldo)
      : M(M), N(N), ldi(ldi), ldo(ldo) {}

  void operator()(Tin* in, Tout* out) {
#ifdef __AVX512F__
    const __m512 c1 = _mm512_set1_ps((float)0.7978846);
    const __m512 c2 = _mm512_set1_ps((float)0.0356814);
    const __m512 c_half = _mm512_set1_ps((float)0.5);
    for (int j = 0; j < M; j++) {
      int i;
      for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
        auto vin = _mm512_loadu_ps_auto(&in[j * ldi + i]);
        __m512 x_half = _mm512_mul_ps(vin, c_half);
        __m512 x_sq = _mm512_mul_ps(vin, vin);
        __m512 poly_x1 = _mm512_mul_ps(vin, _mm512_fmadd_ps(x_sq, c2, c1));
        __m512 tanh_poly_x = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX3(poly_x1);
        __m512 vout = _mm512_fmadd_ps(tanh_poly_x, x_half, x_half);
        _mm512_storeu_ps_auto(&out[j * ldo + i], vout);
      }
      if (i < N) {
        int rem = N - i;
        __mmask16 mask = (1 << rem) - 1;
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[j * ldi + i]);
        __m512 x_half = _mm512_mul_ps(vin, c_half);
        __m512 x_sq = _mm512_mul_ps(vin, vin);
        __m512 poly_x1 = _mm512_mul_ps(vin, _mm512_fmadd_ps(x_sq, c2, c1));
        __m512 tanh_poly_x = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX3(poly_x1);
        __m512 vout = _mm512_fmadd_ps(tanh_poly_x, x_half, x_half);
        _mm512_mask_storeu_ps_auto(&out[j * ldo + i], mask, vout);
      }
    }
#else
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i++) {
        float x = in[j * ldi + i];
        out[j * ldo + i] =
            ((tanh(sqrt(2 / M_PI) * (x + 0.044715 * std::pow(x, 3)))) + 1) * x *
            0.5;
      }
    }
#endif
  }

  void ref(Tin* in, Tout* out) {
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i++) {
        float x = in[j * ldi + i];
        out[j * ldo + i] =
            ((tanh(sqrt(2 / M_PI) * (x + 0.044715 * std::pow(x, 3)))) + 1) * x *
            0.5;
      }
    }
  }

 private:
  int M = 0;
  int N = 0;
  int ldi = 0;
  int ldo = 0;
};

template <typename T1, typename T2 = T1, typename T3 = T1>
class GeluBwdTPP : public BaseTPP {
 public:
  GeluBwdTPP() {}
  GeluBwdTPP(int N) : N(N) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  void operator()(T1* gout, T2* in, T3* gin) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[2];
    arg_array[0].primary = (void*)gout;
    arg_array[1].primary = (void*)in;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)gin;

    kernel(&eqn_param);
  }
  void ref(T1* gout, T2* in, T3* gin) {
#ifdef __AVX512F__
    int i;
    for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
      auto vgout = _mm512_loadu_ps_auto(&gout[i]);
      auto vin_gelu = _mm512_loadu_ps_auto(&in[i]);
      auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
      // auto vgin_gelu =
      // LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
      auto vout = _mm512_mul_ps(vgin_gelu, vgout);
      _mm512_storeu_ps_auto(&gin[i], vout);
    }
    if (i < N) {
      int rem = N - i;
      __mmask16 mask = (1 << rem) - 1;
      auto vgout = _mm512_maskz_loadu_ps_auto(mask, &gout[i]);
      auto vin_gelu = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
      auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
      // auto vgin_gelu =
      // LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
      auto vout = _mm512_mul_ps(vgin_gelu, vgout);
      _mm512_mask_storeu_ps_auto(&gin[i], mask, vout);
    }
#else
    constexpr float PI = 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
      float x = in[i];
      gin[i] = (float)gout[i] *
          (0.5 + 0.5 * erff(x / sqrtf(2.0)) +
           x / (sqrtf(2.0 * PI)) * expf(-0.5 * x * x));
    }
#endif
  }

 protected:
  uint64_t hash_int() override {
    std::array<int, 4> params = {
        XsmmDtype<T1>(), XsmmDtype<T2>(), XsmmDtype<T3>(), N};
    uint64_t hash_value = string_to_hash_int<4>("gelu_bwd_eqn", params);
    return hash_value;
  }
  void* build_kernel() override {
    auto dt1 = XsmmDtype<T1>();
    auto dt2 = XsmmDtype<T2>();
    auto dt3 = XsmmDtype<T3>();
    libxsmm_blasint ld = N;
    libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, N, 1, N, 0, 0, dt1);
    meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_GELU_INV);
    meqn_push_arg(my_eqn0, N, 1, N, 1, 0, dt2);
    debug_print_eqn_tree(my_eqn0);
    return (void*)meqn_dispatch(N, 1, &ld, dt3, my_eqn0);
  }
  void print_error() override {
    fprintf(
        stderr,
        "Unable to get JIT kernel for gelu_bwd_eqn. Params: dt1=%d, dt2=%d, dt3=%d, N=%d",
        XsmmDtype<T1>(),
        XsmmDtype<T2>(),
        XsmmDtype<T3>(),
        N);
  }

 private:
  int N = 0;
  libxsmm_matrix_eqn_function kernel = NULL;
};

template <typename Tin, typename Tout = Tin>
class ReLUFwdTPP {
 public:
  ReLUFwdTPP() {}
  ReLUFwdTPP(int N, bool bm) : ReLUFwdTPP(1, N, bm) {}
  ReLUFwdTPP(int rows, int cols, bool bm)
      : ReLUFwdTPP(rows, cols, cols, cols, bm) {}
  ReLUFwdTPP(int rows, int cols, int ldi, int ldo, bool bm)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ReLUBwdTPP {
 public:
  ReLUBwdTPP() {}
  ReLUBwdTPP(int N, bool bm) : ReLUBwdTPP(1, N, bm) {}
  ReLUBwdTPP(int rows, int cols, bool bm)
      : ReLUBwdTPP(rows, cols, cols, cols, bm) {}
  ReLUBwdTPP(int rows, int cols, int ldi, int ldo, bool bm)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        bm(bm),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {}
  void operator()(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, bm ? (void*)mask : (void*)in2, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, bm ? (void*)mask : (void*)in2, NULL, out, NULL);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  bool bm;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ELUFwdTPP {
 public:
  ELUFwdTPP() {}
  ELUFwdTPP(int N, float alpha) : ELUFwdTPP(1, N, alpha) {}
  ELUFwdTPP(int rows, int cols, float alpha)
      : ELUFwdTPP(rows, cols, cols, cols, alpha) {}
  ELUFwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_ELU) {}
  void operator()(Tin* in, Tout* out) {
    kernel(in, NULL, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out) {
    Tin a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        out[i * ldo + j] = in[i * ldi + j] > 0 ? in[i * ldi + j]
                                               : a * (exp(in[i * ldi + j]) - 1);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ELUBwdTPP {
 public:
  ELUBwdTPP() {}
  ELUBwdTPP(int N, float alpha) : ELUBwdTPP(1, N, alpha) {}
  ELUBwdTPP(int rows, int cols, float alpha)
      : ELUBwdTPP(rows, cols, cols, cols, alpha) {}
  ELUBwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) {}
  void operator()(Tin* in, Tin* in2, Tout* out) {
    kernel(in, in2, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tin* in2, Tout* out) {
    Tin a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        out[i * ldo + j] = in2[i * ldi + j] > 0
            ? in[i * ldi + j]
            : in[i * ldi + j] * in2[i * ldi + j] + a * in[i * ldi + j];
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class LeakyReLUFwdTPP {
 public:
  LeakyReLUFwdTPP() {}
  LeakyReLUFwdTPP(int N, float alpha) : LeakyReLUFwdTPP(1, N, alpha) {}
  LeakyReLUFwdTPP(int rows, int cols, float alpha)
      : LeakyReLUFwdTPP(rows, cols, cols, cols, alpha) {}
  LeakyReLUFwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    kernel(in, NULL, NULL, &alpha, NULL, NULL, out, mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    float a = alpha;
    // std::cout << " op: " << out << " inp: "<< in << std::endl;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j] > 0 ? (Tout)in[i * ldi + j]
                                               : (Tout)(a * (in[i * ldi + j]));
      }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class LeakyReLUBwdTPP {
 public:
  LeakyReLUBwdTPP() {}
  LeakyReLUBwdTPP(int N, float alpha) : LeakyReLUBwdTPP(1, N, alpha) {}
  LeakyReLUBwdTPP(int rows, int cols, float alpha)
      : LeakyReLUBwdTPP(rows, cols, cols, cols, alpha) {}
  LeakyReLUBwdTPP(int rows, int cols, int ldi, int ldo, float alpha)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        alpha(alpha),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) {}
  void operator()(Tin* in, Tout* out, Tin* in2 = NULL, short* mask = NULL) {
    kernel(in, mask, NULL, &alpha, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, Tin* in2, short* mask = NULL) {
    float a = alpha;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        float grad_out = in[i * ldi + j];
        out[i * ldo + j] =
            in2[i * ldi + j] > 0 ? (Tout)grad_out : (Tout)(a * grad_out);
      }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float alpha;
  UnaryTPP kernel;
};

template <typename T>
class SiLUFwdTPP {
 public:
  SiLUFwdTPP() {}
  SiLUFwdTPP(int N) : SiLUFwdTPP(1, N) {}
  SiLUFwdTPP(int rows, int cols) : SiLUFwdTPP(rows, cols, cols, cols) {}
  SiLUFwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        sigmoid(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_SIGMOID),
        mul(rows,
            cols,
            ldi,
            ldo,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(T* in, T* out, T* sigout = nullptr) {
    T tmp[rows * ldo];
    if (sigout == nullptr)
      sigout = tmp;
    sigmoid((void*)in, (void*)sigout);
    mul((void*)in, (void*)sigout, (void*)out);
  }
  void ref(T* in, T* out, T* sigout = nullptr) {
    T tmp[rows * ldo];
    if (sigout == nullptr)
      sigout = tmp;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        sigout[i * ldo + j] = 1. / (1. + exp(-in[i * ldi + j]));
        out[i * ldo + j] = in[i * ldi + j] * sigout[i * ldo + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP sigmoid;
  BinaryTPP mul;
};

template <typename Tin, typename Tout = Tin>
class SiLUBwdTPP : public BaseTPP {
 public:
  SiLUBwdTPP() {}
  SiLUBwdTPP(int N) : SiLUBwdTPP(1, N) {}
  SiLUBwdTPP(int rows, int cols) : SiLUBwdTPP(rows, cols, cols, cols) {}
  SiLUBwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows), cols(cols), ldi(ldi), ldo(ldo) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  void operator()(Tin* in, Tin* in2, Tin* in3, Tout* out) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    float one = 1.;
    arg_array[0].primary = (void*)in;
    arg_array[1].primary = (void*)in2;
    arg_array[2].primary = (void*)in3;
    arg_array[3].primary = (void*)&one;
    arg_array[4].primary = (void*)in2;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)out;

    kernel(&eqn_param);
  }
  void ref(Tin* in, Tin* in2, Tin* in3, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        float grad_out = in[i * ldi + j];
        float si = in2[i * ldi + j];
        float fout = in3[i * ldi + j];

        out[i] = grad_out * (si + fout * (1 - si));
      }
    }
  }

 protected:
  uint64_t hash_int() override {
    std::array<int, 2> params = {rows, cols};
    uint64_t hash_value = string_to_hash_int<2>("silu_bwd_eqn", params);
    return hash_value;
  }
  void* build_kernel() override {
    libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 0, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 1, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 2, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_BINARY_SUB,
        LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 3, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_arg(my_eqn0, cols, rows, ldo, 4, 0, LIBXSMM_DATATYPE_F32);

    auto func0 = meqn_dispatch(cols, rows, &ldo, XsmmDtype<Tout>(), my_eqn0);
    return (void*)func0;
  }
  void print_error() override {
    fprintf(
        stderr,
        "Unable to get JIT kernel for silu_bwd_eqn. Params: rows=%d, cols=%d",
        rows,
        cols);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  libxsmm_matrix_eqn_function kernel = NULL;
};

template <typename Tin, typename Tout = Tin>
class DropOutFwdTPP {
 public:
  DropOutFwdTPP() {}
  DropOutFwdTPP(int N, float p) : DropOutFwdTPP(1, N, p) {}
  DropOutFwdTPP(int rows, int cols, float p)
      : DropOutFwdTPP(rows, cols, cols, cols, p) {}
  DropOutFwdTPP(int rows, int cols, int ldi, int ldo, float p)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        p(p),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {}
  void operator()(Tin* in, void* rng_state, Tout* out, short* mask) {
    kernel(in, NULL, NULL, &p, rng_state, NULL, out, mask);
  }
  void ref(Tin* in, void* rng_state, Tout* out, short* mask) {
    kernel(in, NULL, NULL, &p, rng_state, NULL, out, mask);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float p;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class DropOutBwdTPP {
 public:
  DropOutBwdTPP() {}
  DropOutBwdTPP(int N, float p) : DropOutBwdTPP(1, N, p) {}
  DropOutBwdTPP(int rows, int cols, float p)
      : DropOutBwdTPP(rows, cols, cols, cols, p) {}
  DropOutBwdTPP(int rows, int cols, int ldi, int ldo, float p)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        p(p),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT,
            LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {}
  void operator()(Tin* in, Tout* out, short* mask) {
    kernel(in, mask, NULL, &p, NULL, NULL, out, NULL);
  }
  void ref(Tin* in, Tout* out, short* mask) {
    kernel(in, mask, NULL, &p, NULL, NULL, out, NULL);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  float p;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class SoftMaxFwdTPP {
 public:
  SoftMaxFwdTPP() {}
  SoftMaxFwdTPP(int S1, int S2, int S3)
      : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3), eqn1(S1, S2, S3) {}
  void operator()(Tin* in, Tout* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      eqn0(&in[s2 * S3], tmp);
      eqn1(tmp, &out[s2 * S3]);
    }
  }
  void ref(Tin* pinp, Tout* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, Tin, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, Tout, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      __m512 vmax = _mm512_set1_ps(max);
      __m512 vsum = _mm512_setzero_ps();

      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          vmax = _mm512_max_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          vmax = _mm512_mask_max_ps(
              vmax,
              mask,
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
      }
      max = _mm512_reduce_max_ps(vmax);
      vmax = _mm512_set1_ps(max);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_storeu_ps(&tmp[s1][s3], vz);
          vsum = _mm512_add_ps(vsum, vz);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
          vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      sum = 1.0 / sum;
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          _mm512_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          _mm512_mask_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          if (max < cur)
            max = cur;
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          float z = expf(cur - max);
          tmp[s1][s3] = z;
          sum += z;
        }
      }
      sum = 1.0 / sum;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = tmp[s1][s3] * sum;
          // libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out,
          // s1, s2, s3, S2, S3), 1 );
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = cur;
        }
      }
    }
#endif
  }
  class Eqn0 : BaseTPP {
   public:
    Eqn0() {}
    Eqn0(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(Tin* in, float* out) {
      if (!initialized)
        return;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[1];
      arg_array[0].primary = (void*)in;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)out;

      kernel(&eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 5> params = {
          XsmmDtype<Tin>(), LIBXSMM_DATATYPE_F32, S1, S2, S3};
      uint64_t hash_value = string_to_hash_int<5>("softmax_fwd_eqn0", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto dt_in = XsmmDtype<Tin>();
      libxsmm_blasint tmp_ld = S2;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
      meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_EXP);
      meqn_push_binary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_BINARY_SUB,
          LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, dt_in);
      meqn_push_unary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
      meqn_push_unary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, dt_in);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(
          S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn0);
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for softmax_fwd_eqn0. Params: dt_in=%d, dt_out=%d, S1=%d, S2=%d, S3=%d",
          XsmmDtype<Tin>(),
          LIBXSMM_DATATYPE_F32,
          S1,
          S2,
          S3);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

  class Eqn1 : BaseTPP {
   public:
    Eqn1() {}
    Eqn1(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(float* in, Tout* out) {
      if (!initialized)
        return;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[1];
      arg_array[0].primary = (void*)in;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)out;

      kernel(&eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 5> params = {
          LIBXSMM_DATATYPE_F32, XsmmDtype<Tout>(), S1, S2, S3};
      uint64_t hash_value = string_to_hash_int<5>("softmax_fwd_eqn1", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto dt_out = XsmmDtype<Tout>();
      libxsmm_blasint tmp_ld = S2;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
      meqn_push_binary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_BINARY_MUL,
          LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
      meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL);
      meqn_push_unary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
      meqn_push_unary_op(
          my_eqn1,
          LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
          LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
      meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
      /*debug_print_eqn_tree( my_eqn1 );*/
      return (void*)meqn_dispatch(S3, S1, &ld, dt_out, my_eqn1);
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for softmax_fwd_eqn1. Params: dt_in=%d, dt_out=%d, S1=%d, S2=%d, S3=%d",
          LIBXSMM_DATATYPE_F32,
          XsmmDtype<Tout>(),
          S1,
          S2,
          S3);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn0 eqn0;
  Eqn1 eqn1;
};

template <typename T1, typename T2, typename T3>
class SoftMaxBwdTPP {
 public:
  SoftMaxBwdTPP() {}
  SoftMaxBwdTPP(int S1, int S2, int S3)
      : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3, 0), eqn1(S1, S2, S3, 1) {}
  void operator()(T1* gin, T2* gout, T3* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[2];
      arg_array[0].primary = (void*)&gout[s2 * S3];
      arg_array[1].primary = (void*)&out[s2 * S3];
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)tmp;
      eqn0(&eqn_param);

      arg_array[0].primary = (void*)tmp;
      eqn_param.output.primary = (void*)&gin[s2 * S3];
      eqn1(&eqn_param);
    }
  }
  void ref(T1* pgradinp, T2* pgradout, T3* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T1, ginp, pgradinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T2, gout, pgradout, S2, S3);
    LIBXSMM_VLA_DECL(3, T3, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      __m512 vsum = _mm512_setzero_ps();
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vgo =
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_loadu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgo = _mm512_maskz_loadu_ps(
              mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_maskz_loadu_ps_auto(
              mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 tmp = _mm512_sub_ps(
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              _mm512_mul_ps(
                  _mm512_loadu_ps_auto(
                      &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 tmp = _mm512_sub_ps(
              _mm512_maskz_loadu_ps(
                  mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_mask_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(
                  _mm512_maskz_loadu_ps_auto(
                      mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) *
              upconvert_to_float(
                     LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) =
              upconvert_to_float(
                  LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) *
              (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
        }
      }
    }
#endif
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 7> params = {
          eqn_no,
          XsmmDtype<T1>(),
          XsmmDtype<T2>(),
          LIBXSMM_DATATYPE_F32,
          S1,
          S2,
          S3};
      uint64_t hash_value = string_to_hash_int<7>("softmax_bwd_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto dt_1 = XsmmDtype<T1>();
      auto dt_2 = XsmmDtype<T2>();
      auto dt_3 = XsmmDtype<T3>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn2, S3, S1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn2, S3, S1, ld, 1, 0, dt_3);
        debug_print_eqn_tree(my_eqn2); // printf
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 1) {
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
#if 1
        meqn_push_ternary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_TERNARY_NMULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn3, S3, S1, ld, 1, 0, dt_3);
#else
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn3, S3, S1, ld, 1, 0, dt_3);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_unary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32);
#endif
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(S3, S1, &ld, dt_1, my_eqn3);
      } else {
        PCL_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for softmax_bwd_eqn. Params: eqn_no=%d, dt_1=%d, dt_2=%d, dt_3=%d, S1=%d, S2=%d, S3=%d",
          eqn_no,
          XsmmDtype<T1>(),
          XsmmDtype<T2>(),
          LIBXSMM_DATATYPE_F32,
          S1,
          S2,
          S3);
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn eqn0, eqn1;
};

template <typename Tin, typename Tout>
class VarSoftMaxFwdTPP {
 public:
  VarSoftMaxFwdTPP() {}
  VarSoftMaxFwdTPP(int S2, int S3)
      : S2(S2),
        S3(S3),
        kmax(
            1,
            S3,
            S3,
            S3,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX),
        ksub(
            1,
            S3,
            S3,
            S3,
            XsmmDtype<Tin>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_SUB),
        kexp(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_EXP),
        ksum(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        kmul(
            1,
            S3,
            S3,
            S3,
            LIBXSMM_DATATYPE_F32,
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(int S1, Tin* in, Tout* out) {
    LIBXSMM_ALIGNED(float tmp[S1 * S3], 64);
    for (int s2 = 0; s2 < S2; s2++) {
      Tin max = in[s2 * S3];
      float sum = 0.0f;
      for (int s1 = 0; s1 < S1; s1++) {
        float rmax = 0;
        kmax(&in[s1 * S2 * S3 + s2 * S3], &rmax);
        if (max < rmax)
          max = rmax;
      }
      for (int s1 = 0; s1 < S1; s1++) {
        LIBXSMM_ALIGNED(float tmp2[S3], 64);
        ksub(&in[s1 * S2 * S3 + s2 * S3], &max, tmp2);
        kexp(tmp2, &tmp[s1 * S3]);
        float lsum;
        ksum(&tmp[s1 * S3], &lsum);
        sum += lsum;
      }
      sum = 1.0 / sum;
      for (int s1 = 0; s1 < S1; s1++) {
        kmul(&tmp[s1 * S3], &sum, &out[s1 * S2 * S3 + s2 * S3]);
      }
    }
  }
  void ref(int S1, Tin* pinp, Tout* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, Tin, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, Tout, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      __m512 vmax = _mm512_set1_ps(max);
      __m512 vsum = _mm512_setzero_ps();

      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          vmax = _mm512_max_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          vmax = _mm512_mask_max_ps(
              vmax,
              mask,
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax);
        }
      }
      max = _mm512_reduce_max_ps(vmax);
      vmax = _mm512_set1_ps(max);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_loadu_ps_auto(
                  &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_storeu_ps(&tmp[s1][s3], vz);
          vsum = _mm512_add_ps(vsum, vz);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(
              _mm512_maskz_loadu_ps_auto(
                  mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)),
              vmax));
          _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
          vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      sum = 1.0 / sum;
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          _mm512_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          _mm512_mask_storeu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
        }
      }
    }
#else
    // #warning "Not using AVX512 path for VarSoftMax"
    for (s2 = 0; s2 < S2; s2++) {
      float tmp[S1][S3];
      float max =
          upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          if (max < cur)
            max = cur;
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = upconvert_to_float(
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
          float z = expf(cur - max);
          tmp[s1][s3] = z;
          sum += z;
        }
      }
      sum = 1.0 / sum;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          float cur = tmp[s1][s3] * sum;
          // libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out,
          // s1, s2, s3, S2, S3), 1 );
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = cur;
        }
      }
    }
#endif
  }

 private:
  int S2, S3;
  UnaryTPP kmax;
  BinaryTPP ksub;
  UnaryTPP kexp;
  UnaryTPP ksum;
  BinaryTPP kmul;
};

template <typename T1, typename T2, typename T3>
class VarSoftMaxBwdTPP {
 public:
  VarSoftMaxBwdTPP() {}
  VarSoftMaxBwdTPP(int S2, int S3) : S2(S2), S3(S3), eqn0(S3, 0), eqn1(S3, 1) {}
  void operator()(int S1, T1* gin, T2* gout, T3* out) {
    int64_t S23 = S2 * S3;
    for (int s2 = 0; s2 < S2; s2++) {
      float tmp = 0.0f;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[3];
      arg_array[2].primary = (void*)&tmp;
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = (void*)&tmp;
      for (int s1 = 0; s1 < S1; s1++) {
        int64_t ind = s1 * S23 + s2 * S3;
        arg_array[0].primary = (void*)&gout[ind];
        arg_array[1].primary = (void*)&out[ind];
        eqn0(&eqn_param);
      }
      for (int s1 = 0; s1 < S1; s1++) {
        int64_t ind = s1 * S23 + s2 * S3;
        arg_array[0].primary = (void*)&gout[ind];
        arg_array[1].primary = (void*)&out[ind];
        eqn_param.output.primary = (void*)&gin[ind];
        eqn1(&eqn_param);
      }
    }
  }
  void ref(int S1, T1* pgradinp, T2* pgradout, T3* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T1, ginp, pgradinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T2, gout, pgradout, S2, S3);
    LIBXSMM_VLA_DECL(3, T3, out, pout, S2, S3);
#if defined(__AVX512F__)
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      __m512 vsum = _mm512_setzero_ps();
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 vgo =
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_loadu_ps_auto(
              &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgo = _mm512_maskz_loadu_ps(
              mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
          __m512 vo = _mm512_maskz_loadu_ps_auto(
              mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
          vsum = _mm512_fmadd_ps(vgo, vo, vsum);
        }
      }
      sum = _mm512_reduce_add_ps(vsum);
      vsum = _mm512_set1_ps(sum);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3 += 16) {
          __m512 tmp = _mm512_sub_ps(
              _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              _mm512_mul_ps(
                  _mm512_loadu_ps_auto(
                      &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
        if (s3 < S3) {
          int rem = S3 - s3;
          __mmask16 mask = (1 << rem) - 1;
          __m512 tmp = _mm512_sub_ps(
              _mm512_maskz_loadu_ps(
                  mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)),
              vsum);
          _mm512_mask_storeu_ps(
              &LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3),
              mask,
              _mm512_mul_ps(
                  _mm512_maskz_loadu_ps_auto(
                      mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)),
                  tmp));
        }
      }
    }
#else
    for (s2 = 0; s2 < S2; s2++) {
      float sum = 0.0;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) *
              upconvert_to_float(
                     LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        }
      }
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) =
              upconvert_to_float(
                  LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) *
              (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
        }
      }
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S3, int eqn_no) : S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 5> params = {
          eqn_no, XsmmDtype<T1>(), XsmmDtype<T2>(), XsmmDtype<T3>(), S3};
      uint64_t hash_value = string_to_hash_int<5>("varsoftmax_bwd_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto dt_1 = XsmmDtype<T1>();
      auto dt_2 = XsmmDtype<T2>();
      auto dt_3 = XsmmDtype<T3>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint ld = S3;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_arg(my_eqn2, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn2, S3, 1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn2, S3, 1, ld, 1, 0, dt_3);
        debug_print_eqn_tree(my_eqn2); // printf
        func = meqn_dispatch(S3, 1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 1) {
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn3, S3, 1, ld, 1, 0, dt_3);
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_SUB,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn3, S3, 1, ld, 0, 0, dt_2);
        meqn_push_arg(my_eqn3, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn3);
        func = meqn_dispatch(S3, 1, &ld, dt_1, my_eqn3);
      } else {
        PCL_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for varsoftmax_bwd_eqn. Params: eqn_no=%d, dt_1=%d, dt_2=%d, dt_3=%d, S3=%d",
          eqn_no,
          XsmmDtype<T1>(),
          XsmmDtype<T2>(),
          XsmmDtype<T3>(),
          S3);
    }

   private:
    int S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S2, S3;
  Eqn eqn0, eqn1;
};

template <typename T>
class LayerNormFwdTPP {
 public:
  LayerNormFwdTPP() {}
  LayerNormFwdTPP(int S1, int S2, int S3, float eps)
      : S1(S1),
        S2(S2),
        S3(S3),
        eps(eps),
        reduce_cols_kernel(
            S1,
            S3,
            S2 * S3,
            S3,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD),
        reduce_rows_kernel(
            1,
            S3,
            S3,
            1,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) {}
  void operator()(T* inp, T* gamma, T* beta, float* mean, float* var, T* out) {
    LIBXSMM_ALIGNED(float tmp[2 * S3], 64);
    const float c = 1.0 / ((float)S1 * S3);
    float m, v, s, b;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    eqn_param.inputs = arg_array;
    arg_array[1].primary = &s;
    arg_array[2].primary = &b;
    arg_array[3].primary = (void*)gamma;
    arg_array[4].primary = (void*)beta;
    for (int s2 = 0; s2 < S2; s2++) {
      reduce_cols_kernel((void*)&inp[s2 * S3], (void*)tmp);
      reduce_rows_kernel((void*)tmp, (void*)&m);
      reduce_rows_kernel((void*)&tmp[S3], (void*)&v);
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      s = v;
      b = -1.0 * v * m;
      arg_array[0].primary = (void*)&inp[s2 * S3];
      eqn_param.output.primary = (void*)&out[s2 * S3];
      eqn(&eqn_param);
    }
  }
  void ref(T* pinp, T* pgamma, T* pbeta, float* mean, float* var, T* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
    LIBXSMM_VLA_DECL(2, T, gamma, pgamma, S3);
    LIBXSMM_VLA_DECL(2, T, beta, pbeta, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float m = 0;
      float v = 0;
      float c = 1.0 / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        }
      }
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      float s = v;
      float b = -1.0 * v * m;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) =
              (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) *
                  LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) +
              LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3);
        }
      }
    }
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 5> params = {XsmmDtype<T>(), S1, S2, S3};
      uint64_t hash_value = string_to_hash_int<5>("layernorm_fwd_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint tmp_ld2 = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, S3, S1, tmp_ld2, 3, 0, in_dt);
      meqn_push_arg(my_eqn0, S3, S1, tmp_ld2, 4, 0, in_dt);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(S3, S1, &ld, out_dt, my_eqn0);
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for layernorm_fwd_eqn. Params: dt_1=%dS1=%d, S2=%d, S3=%d",
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  float eps;
  UnaryTPP reduce_cols_kernel;
  UnaryTPP reduce_rows_kernel;
  Eqn eqn;
};

template <typename T>
class LayerNormBwdTPP {
 public:
  LayerNormBwdTPP() {}
  LayerNormBwdTPP(int S1, int S2, int S3)
      : S1(S1),
        S2(S2),
        S3(S3),
        dgamma_func(S1, S2, S3, 1),
        dbeta_func(S1, S2, S3, 2),
        db_func(S1, S2, S3, 3),
        ds_func(S1, S2, S3, 4),
        din_func(S1, S2, S3, 5) {}
  void operator()(
      T* dout,
      T* inp,
      float* mean,
      float* var,
      T* gamma,
      T* din,
      float* dgamma,
      float* dbeta) {
    float a, b, c, db, ds;
    const float scale = 1.0f / ((float)S1 * S3);
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];
    eqn_param.inputs = arg_array;

    arg_array[1].primary = &a;
    arg_array[2].primary = &b;
    arg_array[4].primary = (void*)dgamma;
    arg_array[5].primary = (void*)dbeta;
    arg_array[6].primary = (void*)gamma;
    arg_array[7].primary = &c;

    for (int s2 = 0; s2 < S2; s2++) {
      a = var[s2];
      b = -a * mean[s2];
      arg_array[0].primary = (void*)&inp[s2 * S3];
      arg_array[3].primary = (void*)&dout[s2 * S3];

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);

      eqn_param.output.primary = &db;
      db_func(&eqn_param);

      eqn_param.output.primary = (void*)dgamma;
      dgamma_func(&eqn_param);

      eqn_param.output.primary = (void*)dbeta;
      dbeta_func(&eqn_param);

      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;

      eqn_param.output.primary = (void*)&din[s2 * S3];
      din_func(&eqn_param);
    }
  }
  void ref(
      T* pdout,
      T* pinp,
      float* mean,
      float* var,
      T* pgamma,
      T* pdin,
      float* pdgamma,
      float* pdbeta) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, din, pdin, S2, S3);
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, dout, pdout, S2, S3);
    LIBXSMM_VLA_DECL(2, T, gamma, pgamma, S3);
    LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
    LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float a = var[s2], c;
      float b = -a * mean[s2];
      float ds = 0.0f;
      float db = 0.0f;
      float scale = 1.0f / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3) +=
              (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) *
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3) +=
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3);
        }
      }
      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) =
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * a *
                  LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) +
              b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
        }
      }
    }
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 6> params = {eqn_no, XsmmDtype<T>(), S1, S2, S3};
      uint64_t hash_value = string_to_hash_int<6>("layernorm_bwd_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      // auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint tmp_ld2 = 1;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_matrix_eqn_function func = NULL;
      if (eqn_no == 1) {
        /* dgamma function  */
        libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32);
        /*debug_print_eqn_tree( my_eqn1 );*/
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1);
      } else if (eqn_no == 2) {
        /* dbeta function  */
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_arg(my_eqn2, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn2, S3, S1, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32);
        /*debug_print_eqn_tree( my_eqn1 );*/
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 3) {
        /* db equation */
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_arg(my_eqn3, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn3, S3, S1, tmp_ld, 6, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3);
      } else if (eqn_no == 4) {
        /* ds equation */
        libxsmm_blasint my_eqn4 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_binary_op(my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_arg(my_eqn4, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn4, S3, S1, tmp_ld, 6, 0, in_dt);
        meqn_push_arg(my_eqn4, S3, S1, ld, 0, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4);
      } else if (eqn_no == 5) {
        /* din equation */
        libxsmm_blasint my_eqn5 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_binary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn5, S3, S1, tmp_ld, 6, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, S3, S1, ld, 3, 0, in_dt);
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn5, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32);
        func = meqn_dispatch(S3, S1, &ld, in_dt, my_eqn5);
      } else {
        PCL_ASSERT(false, "LayerNormBwdTPP: invalid eqn. number %d\n", eqn_no);
      }
      return (void*)func;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for layernorm_bwd_eqn. Params: eqn_no=%d, dt_1=%d, S1=%d, S2=%d, S3=%d",
          eqn_no,
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn dgamma_func, dbeta_func, db_func, ds_func, din_func;
};

template <typename T>
class GroupNormFwdTPP {
 public:
  GroupNormFwdTPP() {}
  GroupNormFwdTPP(int S1, int S2, int S3, float eps)
      : S1(S1),
        S2(S2),
        S3(S3),
        eps(eps),
        reduce_cols_kernel(
            S1,
            S3,
            S2 * S3,
            S3,
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD),
        reduce_rows_kernel(
            1,
            S3,
            S3,
            1,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) {}
  void operator()(T* inp, T* gamma, T* beta, float* mean, float* var, T* out) {
    LIBXSMM_ALIGNED(float tmp[2 * S3], 64);
    const float c = 1.0 / ((float)S1 * S3);
    float m, v, s, b;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[5];
    eqn_param.inputs = arg_array;
    arg_array[1].primary = &s;
    arg_array[2].primary = &b;
    arg_array[3].primary = (void*)gamma;
    arg_array[4].primary = (void*)beta;
    for (int s2 = 0; s2 < S2; s2++) {
      reduce_cols_kernel((void*)&inp[s2 * S3], (void*)tmp);
      reduce_rows_kernel((void*)tmp, (void*)&m);
      reduce_rows_kernel((void*)&tmp[S3], (void*)&v);
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      s = v;
      b = -1.0 * v * m;
      arg_array[0].primary = (void*)&inp[s2 * S3];
      eqn_param.output.primary = (void*)&out[s2 * S3];
      eqn(&eqn_param);
    }
  }
  void ref(T* pinp, T* gamma, T* beta, float* mean, float* var, T* pout) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float m = 0;
      float v = 0;
      float c = 1.0 / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        }
      }
      m = m * c;
      v = v * c;
      v = LIBXSMM_MAX(v - m * m, 0.0f);
      v = 1.0f / ((float)sqrt(v + eps));
      mean[s2] = m;
      var[s2] = v;
      float s = v;
      float b = -1.0 * v * m;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) =
              (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) *
                  gamma[s1] +
              beta[s1];
        }
      }
    }
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 4> params = {XsmmDtype<T>(), S1, S2, S3};
      uint64_t hash_value = string_to_hash_int<4>("groupnorm_fwd_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = 1;
      libxsmm_blasint tmp_ld2 = S3;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_ternary_op(
          my_eqn0,
          LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
          LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
              LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
              LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
      meqn_push_arg(my_eqn0, S3, S1, ld, 0, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32);
      meqn_push_arg(my_eqn0, 1, S1, 1, 3, 0, in_dt);
      meqn_push_arg(my_eqn0, 1, S1, 1, 4, 0, in_dt);
      debug_print_eqn_tree(my_eqn0); // printf
      return (void*)meqn_dispatch(S3, S1, &ld, out_dt, my_eqn0);
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for groupnorm_fwd_eqn. Params: dt_1=%d, S1=%d, S2=%d, S3=%d",
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
    }

   private:
    int S1, S2, S3;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  float eps;
  UnaryTPP reduce_cols_kernel;
  UnaryTPP reduce_rows_kernel;
  Eqn eqn;
};

template <typename T>
class GroupNormBwdTPP {
 public:
  GroupNormBwdTPP() {}
  GroupNormBwdTPP(int S1, int S2, int S3)
      : S1(S1),
        S2(S2),
        S3(S3),
        dgamma_func(S1, S2, S3, 1),
        dbeta_func(S1, S2, S3, 2),
        db_func(S1, S2, S3, 3),
        ds_func(S1, S2, S3, 4),
        din_func(S1, S2, S3, 5) {}
  void operator()(
      T* dout,
      T* inp,
      float* mean,
      float* var,
      T* gamma,
      T* din,
      float* dgamma,
      float* dbeta) {
    float a, b, c, db, ds;
    const float scale = 1.0f / ((float)S1 * S3);
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[8];
    eqn_param.inputs = arg_array;

    arg_array[1].primary = &a;
    arg_array[2].primary = &b;
    arg_array[4].primary = (void*)dgamma;
    arg_array[5].primary = (void*)dbeta;
    arg_array[6].primary = (void*)gamma;
    arg_array[7].primary = &c;

    for (int s2 = 0; s2 < S2; s2++) {
      a = var[s2];
      b = -a * mean[s2];
      arg_array[0].primary = (void*)&inp[s2 * S3];
      arg_array[3].primary = (void*)&dout[s2 * S3];

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);

      eqn_param.output.primary = &db;
      db_func(&eqn_param);

      eqn_param.output.primary = (void*)dgamma;
      dgamma_func(&eqn_param);

      eqn_param.output.primary = (void*)dbeta;
      dbeta_func(&eqn_param);

      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;

      eqn_param.output.primary = (void*)&din[s2 * S3];
      din_func(&eqn_param);
    }
  }
  void ref(
      T* pdout,
      T* pinp,
      float* mean,
      float* var,
      T* gamma,
      T* pdin,
      float* dgamma,
      float* dbeta) {
    int s1, s2, s3;
    LIBXSMM_VLA_DECL(3, T, din, pdin, S2, S3);
    LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
    LIBXSMM_VLA_DECL(3, T, dout, pdout, S2, S3);
    for (s2 = 0; s2 < S2; s2++) {
      float a = var[s2], c;
      float b = -a * mean[s2];
      float ds = 0.0f;
      float db = 0.0f;
      float scale = 1.0f / (S1 * S3);
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          dgamma[s1] +=
              (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) *
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          dbeta[s1] += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
          ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * gamma[s1] *
              LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
          db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * gamma[s1];
        }
      }
      b = (db * mean[s2] - ds) * a * a * a * scale;
      c = -b * mean[s2] - db * a * scale;
      for (s1 = 0; s1 < S1; s1++) {
        for (s3 = 0; s3 < S3; s3++) {
          LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) =
              LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * a * gamma[s1] +
              b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
        }
      }
    }
  }

  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(int S1, int S2, int S3, int eqn_no)
        : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 5> params = {eqn_no, XsmmDtype<T>(), S1, S2, S3};
      uint64_t hash_value = string_to_hash_int<5>("groupnorm_bwd_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      // auto out_dt = XsmmDtype<T>();
      libxsmm_blasint tmp_ld = S3;
      libxsmm_blasint tmp_ld2 = 1;
      libxsmm_blasint ld = S2 * S3;
      libxsmm_matrix_eqn_function func = NULL;
      if (eqn_no == 1) {
        /* dgamma function  */
        libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_unary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_binary_op(my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL);
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn1, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn1, S3, S1, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1);
      } else if (eqn_no == 2) {
        /* dbeta function  */
        libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
        meqn_push_arg(my_eqn2, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn2, S3, S1, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32);
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2);
      } else if (eqn_no == 3) {
        /* db equation */
        libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn3,
            LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
        meqn_push_arg(my_eqn3, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn3, 1, S1, 1, 6, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3);
      } else if (eqn_no == 4) {
        /* ds equation */
        libxsmm_blasint my_eqn4 = libxsmm_matrix_eqn_create();
        meqn_push_binary_op(
            my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD);
        meqn_push_binary_op(
            my_eqn4,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
        meqn_push_arg(my_eqn4, S3, S1, ld, 3, 0, in_dt);
        meqn_push_arg(my_eqn4, 1, S1, 1, 6, 0, in_dt);
        meqn_push_arg(my_eqn4, S3, S1, ld, 0, 0, in_dt);
        func = meqn_dispatch(1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4);
      } else if (eqn_no == 5) {
        /* din equation */
        libxsmm_blasint my_eqn5 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_binary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0 |
                LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn5, 1, S1, 1, 6, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, S3, S1, ld, 3, 0, in_dt);
        meqn_push_ternary_op(
            my_eqn5,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn5, S3, S1, ld, 0, 0, in_dt);
        meqn_push_arg(my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
        meqn_push_arg(my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32);
        func = meqn_dispatch(S3, S1, &ld, in_dt, my_eqn5);
      } else {
        PCL_ASSERT(false, "GroupNormBwdTPP: invalid eqn. number %d\n", eqn_no);
      }
      return (void*)func;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for groupnorm_bwd_eqn. Params: eqn_no=%d, dt_1=%d, S1=%d, S2=%d, S3=%d",
          eqn_no,
          XsmmDtype<T>(),
          S1,
          S2,
          S3);
    }

   private:
    int S1, S2, S3, eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int S1, S2, S3;
  Eqn dgamma_func, dbeta_func, db_func, ds_func, din_func;
};

class SplitSGDTPP : public BaseTPP {
 public:
  SplitSGDTPP() {}
  SplitSGDTPP(int N) : N(N) {
    kernel = (libxsmm_matrix_eqn_function)get_kernel();
    initialized = true;
  }
  void operator()(bfloat16* hi, bfloat16* lo, bfloat16* grad, float lr) {
    if (!initialized)
      return;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[4];
    arg_array[0].primary = (void*)lo;
    arg_array[1].primary = (void*)hi;
    arg_array[2].primary = (void*)&lr;
    arg_array[3].primary = (void*)grad;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)lo;
    auto offset = (int64_t)((char*)hi - (char*)lo);
    eqn_param.output.secondary = (void*)offset;

    kernel(&eqn_param);
  }
  void ref(bfloat16* hi, bfloat16* lo, bfloat16* grad, float lr) {
#ifndef __AVX512F__
    auto dwt = (libxsmm_bfloat16*)grad;
    auto out_hi = (libxsmm_bfloat16*)hi;
    auto out_lo = (libxsmm_bfloat16*)lo;
    for (int i = 0; i < N; i++) {
      union libxsmm_bfloat16_f32 bf16_hp;
      union libxsmm_bfloat16_f32 bf16_wt;
      bf16_wt.i[0] = 0;
      bf16_wt.i[1] = dwt[i];
      bf16_hp.i[0] = out_lo[i];
      bf16_hp.i[1] = out_hi[i];
      bf16_hp.f = bf16_wt.f * lr + bf16_hp.f;
      out_lo[i] = bf16_hp.i[0];
      out_hi[i] = bf16_hp.i[1];
    }
#else
    int64_t sz = N;
    auto vlr = _mm512_set1_ps(lr);
    int64_t i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&hi[i], &lo[i]);
      data_i = _mm512_add_ps(data_i, _mm512_mul_ps(grad_i, vlr));
      _mm512_split_storeu_ps(&hi[i], &lo[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &hi[i], &lo[i]);
      data_i = _mm512_add_ps(data_i, _mm512_mul_ps(grad_i, vlr));
      _mm512_mask_split_storeu_ps(&hi[i], &lo[i], mask, data_i);
    }
#endif
  }

 protected:
  uint64_t hash_int() override {
    std::array<int, 1> params = {N};
    uint64_t hash_value = string_to_hash_int<1>("split_sgd_eqn", params);
    return hash_value;
  }
  void* build_kernel() override {
    libxsmm_blasint ld = N;
    libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
    meqn_push_unary_op(my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_UNZIP);
    meqn_push_ternary_op(
        my_eqn0,
        LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
        LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
            LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
    /* This is the "gradient" weights   */
    meqn_push_arg(my_eqn0, N, 1, ld, 3, 0, LIBXSMM_DATATYPE_BF16);
    /* This is the scalar learning rate */
    meqn_push_arg(my_eqn0, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32);
    meqn_push_binary_op(my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ZIP);
    /* This is the tensor with lo bits  */
    meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, LIBXSMM_DATATYPE_I16);
    /* This is the tensor with hi bits  */
    meqn_push_arg(my_eqn0, N, 1, ld, 1, 0, LIBXSMM_DATATYPE_I16);
    debug_print_eqn_tree(my_eqn0);
    auto func0 = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_I16, my_eqn0);
    return (void*)func0;
  }
  void print_error() override {
    fprintf(
        stderr, "Unable to get JIT kernel for split_sgd_eqn. Params: N=%d", N);
  }

 private:
  int N = 0;
  libxsmm_matrix_eqn_function kernel = NULL;
};

template <typename Tin, typename Tout, typename Tind>
class EmbBagFwdTPP {
 public:
  EmbBagFwdTPP() {}
  EmbBagFwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            (libxsmm_meltw_unary_flags)(sizeof(Tind) == 8 ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES),
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) {}
  void operator()(Tout* output, Tin* weight, Tind* input, int N) {
    uint64_t _N = N;
    kernel((void*)weight, (void*)input, (void*)&_N, (void*)output, NULL);
  }
  void ref(Tout* output, Tin* weight, Tind* input, int N) {
    for (int64_t v = 0; v < E; v++)
      output[v] = 0;
    for (int64_t s = 0; s < N; s++) {
      auto ind = input[s];
      for (int64_t v = 0; v < E; v++)
        output[v] += weight[ind * E + v];
    }
  }

 private:
  int E;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class EmbBagBwdTPP {
 public:
  EmbBagBwdTPP() {}
  EmbBagBwdTPP(int E)
      : E(E),
        kernel(
            0,
            E,
            E,
            E,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tout>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {}
  void operator()(Tin* in, Tout* out, uint64_t N) {
    kernel((void*)in, NULL, NULL, (void*)&N, NULL, NULL, (void*)out, NULL);
  }
  void ref(Tin* in, Tout* out, uint64_t N) {
    for (uint64_t i = 0; i < N; i++) {
      for (int v = 0; v < E; v++) {
        out[i * E + v] = in[v];
      }
    }
  }

 private:
  int E;
  UnaryTPP kernel;
};

template <typename T>
class FusedAdamWTPP {
 public:
  FusedAdamWTPP() {}
  FusedAdamWTPP(int N, float beta1, float beta2, float weight_decay, float eps)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        eps(eps),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2) {}
  void operator()(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    float lrwd_1 = 1.0f - lr * weight_decay;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[6];
    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&step_size;
    arg_array[4].primary = (void*)data;
    arg_array[5].primary = (void*)&lrwd_1;
    eqn_param.output.primary = (void*)data;
    eqn2(&eqn_param);
  }

  void ref(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    int64_t sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (int64_t i = 0; i < sz; i++) {
      auto avg_i = exp_avg[i];
      auto avg_sq_i = exp_avg_sq[i];
      auto grad_i = grad[i];
      auto data_i = data[i];
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      if (weight_decay > 0.0)
        data_i = data_i - data_i * lr * weight_decay;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data[i] = data_i;
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
    int64_t i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_loadu_ps(&data[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      // if (weight_decay > 0.0)
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_storeu_ps(&data[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_loadu_ps(mask, &data[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      // if (weight_decay > 0.0)
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_storeu_ps(&data[i], mask, data_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedAdamWTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 4> params = {
          eqn_no, XsmmDtype<T>(), p->N, (p->weight_decay == 0.0 ? 0 : 1)};
      uint64_t hash_value = string_to_hash_int<4>("fused_adamw_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      int use_wd = p->weight_decay == 0.0 ? 0 : 1;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for data_i (with decay)
        auto my_eqn2 = libxsmm_matrix_eqn_create();
        if (use_wd == 1) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_arg(my_eqn2, N, 1, ld, 4, 0, in_dt); // data_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        meqn_push_arg(
            my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // step_size
        if (use_wd == 1) {
          // this scalar is (1-lr*weight_decay)
          meqn_push_arg(my_eqn2, 1, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32);
        }
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn2);
      } else {
        PCL_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for fused_adamw_eqn. Params: eqn_no=%d, dt_1=%d, N=%d, weight_decay=%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          (p->weight_decay == 0.0 ? 0 : 1));
    }

   private:
    FusedAdamWTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, weight_decay, eps;
  Eqn eqn0, eqn1, eqn2;
  friend class Eqn;
};

class FusedSplitAdamWTPP {
 public:
  typedef bfloat16 T;
  FusedSplitAdamWTPP() {}
  FusedSplitAdamWTPP(
      int N,
      float beta1,
      float beta2,
      float weight_decay,
      float eps)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        eps(eps),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2) {}
  void operator()(
      T* hi,
      T* lo,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    float lrwd_1 = 1.0f - lr * weight_decay;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[7];
    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&step_size;
    arg_array[4].primary = (void*)lo;
    arg_array[5].primary = (void*)hi;
    arg_array[6].primary = (void*)&lrwd_1;
    eqn_param.output.primary = (void*)lo;
    auto offset = (int64_t)((char*)hi - (char*)lo);
    eqn_param.output.secondary = (void*)offset;
    eqn2(&eqn_param);
  }

  void ref(
      T* hi,
      T* lo,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      float step_size,
      float lr) {
    int64_t sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (int64_t i = 0; i < sz; i++) {
      union libxsmm_bfloat16_f32 data_hp;
      float avg_i = exp_avg[i];
      float avg_sq_i = exp_avg_sq[i];
      float grad_i = grad[i];
      data_hp.i[0] = lo[i];
      data_hp.i[1] = hi[i];
      float data_i = data_hp.f;

      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      if (weight_decay > 0.0)
        data_i = data_i - data_i * lr * weight_decay;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data_hp.f = data_i;
      lo[i] = data_hp.i[0];
      hi[i] = data_hp.i[1];
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
    int64_t i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&hi[i], &lo[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_split_storeu_ps(&hi[i], &lo[i], data_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &hi[i], &lo[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(
          data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      if (weight_decay > 0.0)
        data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_split_storeu_ps(&hi[i], &lo[i], mask, data_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedSplitAdamWTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 4> params = {
          eqn_no, XsmmDtype<T>(), p->N, (p->weight_decay == 0.0 ? 0 : 1)};
      uint64_t hash_value =
          string_to_hash_int<4>("fused_split_adamw_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      int use_wd = p->weight_decay == 0.0 ? 0 : 1;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for data_i (with decay)
        auto my_eqn2 = libxsmm_matrix_eqn_create();
        meqn_push_unary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_UNARY_UNZIP,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_DATATYPE_IMPLICIT);
        if (use_wd == 1) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB);
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ZIP,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_DATATYPE_IMPLICIT);
        meqn_push_arg(
            my_eqn2, N, 1, ld, 4, 0, LIBXSMM_DATATYPE_U16); // data_i lo
        meqn_push_arg(
            my_eqn2, N, 1, ld, 5, 0, LIBXSMM_DATATYPE_U16); // data_i hi

        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        meqn_push_arg(
            my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // step_size
        if (use_wd == 1) {
          // this scalar is (1-lr*weight_decay)
          meqn_push_arg(my_eqn2, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32);
        }
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, LIBXSMM_DATATYPE_U16, my_eqn2);
      } else {
        PCL_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for fused_split_adamw_eqn. Params: eqn_no=%d, dt_1=%d, N=%d, weight_decay=%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          (p->weight_decay == 0.0 ? 0 : 1));
    }

   private:
    FusedSplitAdamWTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, weight_decay, eps;
  Eqn eqn0, eqn1, eqn2;
  friend class Eqn;
};

template <typename T>
class FusedAdamStepTPP {
 public:
  FusedAdamStepTPP() {}
  FusedAdamStepTPP(
      int N,
      float beta1,
      float beta2,
      float eps,
      bool use_weight_decay,
      bool use_bias_correction)
      : N(N),
        beta1(beta1),
        beta2(beta2),
        eps(eps),
        use_weight_decay(use_weight_decay),
        use_bias_correction(use_bias_correction),
        eqn0(this, 0),
        eqn1(this, 1),
        eqn2(this, 2) {}
  void operator()(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      T* adam_step,
      float weight_decay = 0.0,
      float exp_avg_scale = 1.0,
      float exp_avg_sq_scale = 1.0) {
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_matrix_arg arg_array[7];
    arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta1_1;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)&beta1;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = (void*)exp_avg;
    eqn0(&eqn_param);

    // arg_array[0].primary = (void*)grad;
    arg_array[1].primary = (void*)&beta2_1;
    arg_array[2].primary = (void*)exp_avg_sq;
    arg_array[3].primary = (void*)&beta2;
    eqn_param.output.primary = (void*)exp_avg_sq;
    eqn1(&eqn_param);

    arg_array[0].primary = (void*)exp_avg_sq;
    arg_array[1].primary = (void*)&eps;
    arg_array[2].primary = (void*)exp_avg;
    arg_array[3].primary = (void*)data;
    arg_array[4].primary = (void*)&weight_decay;
    arg_array[5].primary = (void*)&exp_avg_scale;
    arg_array[6].primary = (void*)&exp_avg_sq_scale;
    eqn_param.output.primary = (void*)adam_step;
    eqn2(&eqn_param);
  }

  void ref(
      T* data,
      T* grad,
      T* exp_avg,
      T* exp_avg_sq,
      T* adam_step,
      float weight_decay = 0.0,
      float exp_avg_scale = 1.0,
      float exp_avg_sq_scale = 1.0) {
    int64_t sz = N;
    float beta1_1 = 1.0f - beta1;
    float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
    for (int64_t i = 0; i < sz; i++) {
      float avg_i = exp_avg[i];
      float avg_sq_i = exp_avg_sq[i];
      float grad_i = grad[i];
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      if (use_bias_correction) {
        avg_i = avg_i * exp_avg_scale;
        avg_sq_i = avg_sq_i * exp_avg_sq_scale;
      }
      float denom = sqrtf(avg_sq_i) + eps;
      float adam_step_i = avg_i / denom;
      if (use_weight_decay) {
        float data_i = data[i];
        adam_step_i += data_i * weight_decay;
      }
      adam_step[i] = adam_step_i;
    }
#else
    auto vbeta1 = _mm512_set1_ps(beta1);
    auto vbeta1_1 = _mm512_set1_ps(beta1_1);
    auto vbeta2 = _mm512_set1_ps(beta2);
    auto vbeta2_1 = _mm512_set1_ps(beta2_1);
    auto veps = _mm512_set1_ps(eps);
    // auto vstep_size = _mm512_set1_ps(step_size);
    auto vweight_decay = _mm512_set1_ps(weight_decay);
    auto vexp_avg_scale = _mm512_set1_ps(exp_avg_scale);
    auto vexp_avg_sq_scale = _mm512_set1_ps(exp_avg_sq_scale);
    int64_t i;
    for (i = 0; i < ALIGNDOWN(sz, 16); i += 16) {
      auto avg_i = _mm512_loadu_ps_auto(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps_auto(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      _mm512_storeu_ps_auto(&exp_avg[i], avg_i);
      _mm512_storeu_ps_auto(&exp_avg_sq[i], avg_sq_i);
      if (use_bias_correction) {
        avg_i = _mm512_mul_ps(avg_i, vexp_avg_scale);
        avg_sq_i = _mm512_mul_ps(avg_sq_i, vexp_avg_sq_scale);
      }
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      auto adam_step_i = _mm512_div_ps(avg_i, denom);
      if (use_weight_decay) {
        auto data_i = _mm512_loadu_ps_auto(&data[i]);
        adam_step_i =
            _mm512_add_ps(adam_step_i, _mm512_mul_ps(data_i, vweight_decay));
      }
      _mm512_storeu_ps_auto(&adam_step[i], adam_step_i);
    }
    if (i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      avg_i = _mm512_add_ps(
          _mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(
          _mm512_mul_ps(avg_sq_i, vbeta2),
          _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      _mm512_mask_storeu_ps_auto(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps_auto(&exp_avg_sq[i], mask, avg_sq_i);
      if (use_bias_correction) {
        avg_i = _mm512_mul_ps(avg_i, vexp_avg_scale);
        avg_sq_i = _mm512_mul_ps(avg_sq_i, vexp_avg_sq_scale);
      }
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      auto adam_step_i = _mm512_div_ps(avg_i, denom);
      if (use_weight_decay) {
        auto data_i = _mm512_maskz_loadu_ps_auto(mask, &data[i]);
        adam_step_i =
            _mm512_add_ps(adam_step_i, _mm512_mul_ps(data_i, vweight_decay));
      }
      _mm512_mask_storeu_ps_auto(&adam_step[i], mask, adam_step_i);
    }
#endif
  }
  class Eqn : BaseTPP {
   public:
    Eqn() {}
    Eqn(FusedAdamStepTPP* p, int eqn_no) : p(p), eqn_no(eqn_no) {
      kernel = (libxsmm_matrix_eqn_function)get_kernel();
      initialized = true;
    }
    void operator()(libxsmm_matrix_eqn_param* eqn_param) {
      if (!initialized)
        return;
      kernel(eqn_param);
    }

   protected:
    uint64_t hash_int() override {
      std::array<int, 4> params = {
          eqn_no, XsmmDtype<T>(), p->N, p->use_weight_decay};
      uint64_t hash_value =
          string_to_hash_int<4>("fused_adam_step_eqn", params);
      return hash_value;
    }
    void* build_kernel() override {
      auto in_dt = XsmmDtype<T>();
      libxsmm_blasint ld = p->N;
      auto N = p->N;
      int use_wd = p->use_weight_decay;
      int use_bc = p->use_bias_correction;
      libxsmm_matrix_eqn_function func;
      if (eqn_no == 0) {
        // Equation for exp_avg
        auto my_eqn0 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn0, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta1
        meqn_push_binary_op(
            my_eqn0,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_arg(my_eqn0, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta1_1
        debug_print_eqn_tree(my_eqn0);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn0);
      } else if (eqn_no == 1) {
        // Equation for exp_avg_sq
        auto my_eqn1 = libxsmm_matrix_eqn_create();
        meqn_push_ternary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
            LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 |
                LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
        meqn_push_arg(my_eqn1, N, 1, ld, 2, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32); // beta2
        meqn_push_binary_op(
            my_eqn1,
            LIBXSMM_MELTW_TYPE_BINARY_MUL,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2);
        meqn_push_arg(my_eqn1, N, 1, ld, 0, 0, in_dt); // grad_i
        meqn_push_arg(my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // beta2_1
        debug_print_eqn_tree(my_eqn1);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn1);
      } else if (eqn_no == 2) {
        // Equation for adam_step_i (with decay)
        auto my_eqn2 = libxsmm_matrix_eqn_create();
        if (use_wd == 1) {
          meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD);
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
          meqn_push_arg(my_eqn2, N, 1, ld, 3, 0, in_dt); // data_i
          // weight_decay
          meqn_push_arg(my_eqn2, 1, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32);
        }
        meqn_push_binary_op(my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV);
        if (use_bc) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
          meqn_push_arg(
              my_eqn2, 1, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32); // avg_i_scale
        }
        meqn_push_arg(my_eqn2, N, 1, ld, 2, 0, in_dt); // avg_i
        meqn_push_binary_op(
            my_eqn2,
            LIBXSMM_MELTW_TYPE_BINARY_ADD,
            LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1);
        meqn_push_unary_op(my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT);
        if (use_bc) {
          meqn_push_binary_op(
              my_eqn2,
              LIBXSMM_MELTW_TYPE_BINARY_MUL,
              LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0);
          meqn_push_arg(
              my_eqn2, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32); // avg_sq_i_scale
        }
        meqn_push_arg(my_eqn2, N, 1, ld, 0, 0, in_dt); // avg_sq_i
        meqn_push_arg(my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32); // eps
        debug_print_eqn_tree(my_eqn2);
        func = meqn_dispatch(N, 1, &ld, in_dt, my_eqn2);
      } else {
        PCL_ASSERT(false, "Should not come here\n");
      }
      return (void*)func;
    }
    void print_error() override {
      fprintf(
          stderr,
          "Unable to get JIT kernel for fused_adam_step_eqn. Params: eqn_no=%d, dt_1=%d, N=%d, weight_decay=%d",
          eqn_no,
          XsmmDtype<T>(),
          p->N,
          (p->use_weight_decay == 0.0 ? 0 : 1));
    }

   private:
    FusedAdamStepTPP* p;
    int eqn_no;
    libxsmm_matrix_eqn_function kernel = NULL;
  };

 private:
  int N = 0;
  float beta1, beta2, eps;
  bool use_weight_decay, use_bias_correction;
  Eqn eqn0, eqn1, eqn2;
  friend class Eqn;
};
} // namespace tpp
} // namespace torch_ipex
#endif // _XSMM_FUNCTORS_H_
