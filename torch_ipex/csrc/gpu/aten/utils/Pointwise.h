#include <ATen/ATen.h>

#include <core/SYCL.h>

#include <core/TensorImplUtils.h>
#include <core/SYCLApplyUtils.h>
#include <ATen/aten_ipex_type_dpcpp.h>
#include <utils/Numerics.h>


#define IMPLEMENT_POINTWISE_FUNC_(NAME, CFUNC, REAL)               \
  template <typename scalar_t>                                     \
  struct Tensor_##NAME##_##REAL##_Op {                             \
    inline void operator()(scalar_t& out, scalar_t& in) const {    \
      out = CFUNC(in);                                             \
    }                                                              \
                                                                   \
    inline void operator()(scalar_t& v) const {                    \
      v = CFUNC(v);                                                \
    }                                                              \
  };                                                               \
                                                                   \
  template <typename scalar_t>                                     \
  void NAME(Tensor& self_, const Tensor& src) {                    \
    if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {      \
      at::sycl::SYCL_tensor_apply1<scalar_t>(                      \
          self_, Tensor_##NAME##_##REAL##_Op<scalar_t>());         \
    } else {                                                       \
      at::AtenIpexTypeDPCPP::resize_as_(self_, src, c10::nullopt); \
      at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(            \
          self_, src, Tensor_##NAME##_##REAL##_Op<scalar_t>());    \
    }                                                              \
  }

#define IMPLEMENT_POINTWISE_FUNC(NAME, CFUNC, REAL)                \
  IMPLEMENT_POINTWISE_FUNC_(NAME, CFUNC, REAL)


#define IMPLEMENT_POINTWISE_CALLABLE_1_(NAME, CALLABLE)            \
  template <typename scalar_t>                                     \
  void NAME(Tensor& self_, const Tensor& src, scalar_t value) {    \
    if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {      \
      at::sycl::SYCL_tensor_apply1<scalar_t>(                      \
          self_, CALLABLE<scalar_t>(value));                       \
    } else {                                                       \
      at::AtenIpexTypeDPCPP::resize_as_(self_, src, c10::nullopt); \
      at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(            \
          self_, src, CALLABLE<scalar_t>(value));                  \
    }                                                              \
  }

#define IMPLEMENT_POINTWISE_CALLABLE_1(NAME, CALLABLE)             \
  IMPLEMENT_POINTWISE_CALLABLE_1_(NAME, CALLABLE)


#define IMPLEMENT_POINTWISE_CALLABLE_2_(NAME, CALLABLE)            \
  template <typename scalar_t>                                     \
  void NAME(Tensor& self_, const Tensor& src, scalar_t val1, scalar_t val2) { \
    if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {      \
      at::sycl::SYCL_tensor_apply1<scalar_t>(                      \
          self_, CALLABLE<scalar_t>(val1, val2));                  \
    } else {                                                       \
      at::AtenIpexTypeDPCPP::resize_as_(self_, src, c10::nullopt); \
      at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(            \
          self_, src, CALLABLE<scalar_t>(val1, val2));             \
    }                                                              \
  }

#define IMPLEMENT_POINTWISE_CALLABLE_2(NAME, CALLABLE)             \
  IMPLEMENT_POINTWISE_CALLABLE_2_(NAME, CALLABLE)


template <typename T>
struct TensorATan2Op {
  void operator()(T& out, T& a, T& b) const {
    out = Numerics<T>::atan2(a, b);
  }
};

template <typename T>
struct TensorSigmoidOp {
  void operator()(T& out, T& in) const {
    T one = (T) 1.0;
    out = one / (one + Numerics<T>::exp(-static_cast<T>(in)));
  }

  void operator()(T& v) const {
    T one = (T) 1.0;
    v = one / (one + Numerics<T>::exp(-static_cast<T>(v)));
  }
};

/*
 * The following function was converted to SYCL form from code that comes
 * with the following copyright notice. It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
template <typename T, typename accreal>
struct TensorDigammaOp {
  void operator()(T& out, T& in) const {

    using compute_type = typename std::conditional<std::is_same<T, at::Half>::value, accreal, T>::type;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type PSI_10 = 2.25175258906672110764;
    static const compute_type A[] = {
       8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
       7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
       3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
       8.33333333333333333333E-2,
    };

    auto x = scalar_cast<compute_type>(in);
    if (x == 0) {
      out = scalar_cast<T>(INFINITY);
      return;
    }

    bool x_is_integer = (x == DP::floor(x));
    compute_type result = 0;

    if (x < 0) {
      if (x_is_integer) {
        out = scalar_cast<T>(INFINITY);
        return;
      }

      // Rounding errors in tan's input can really affect the output
      // for extreme values, so we always perform this computation in double.
      result = scalar_cast<compute_type>(
          - PI_f64 / DP::tan(PI_f64 * scalar_cast<double>(x)));
      x = 1 - x;
    }

    while (x < 10) {
      result -= 1 / x;
      x += 1;
    }

    if (x == 10) {
      out = scalar_cast<T>(result + PSI_10);
      return;
    }

    compute_type y = 0;
    if (x < 1.0e17) {
      compute_type z = 1.0 / (x * x);
      compute_type polevl_result = 0;
      for (int i = 0; i <= 6; i++) {
        polevl_result = polevl_result * z + A[i];
      }
      y = z * polevl_result;
    }

    out = scalar_cast<T>(DP::log(x) - (0.5 / x) - y + result);
    return;
  }

  void operator()(T& v) const {

    using compute_type = typename std::conditional<std::is_same<T, at::Half>::value, accreal, T>::type;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type PSI_10 = 2.25175258906672110764;
    static const compute_type A[] = {
       8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
       7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
       3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
       8.33333333333333333333E-2,
    };

    auto x = scalar_cast<compute_type>(v);
    if (x == 0) {
      v = scalar_cast<T>(INFINITY);
      return;
    }

    bool x_is_integer = (x == DP::floor(x));
    compute_type result = 0;

    if (x < 0) {
      if (x_is_integer) {
        v = scalar_cast<T>(INFINITY);
        return;
      }

      // Rounding errors in tan's input can really affect the output
      // for extreme values, so we always perform this computation in double.
      result = scalar_cast<compute_type>(
          - PI_f64 / DP::tan(PI_f64 * scalar_cast<double>(x)));
      x = 1 - x;
    }

    while (x < 10) {
      result -= 1 / x;
      x += 1;
    }

    if (x == 10) {
      v = scalar_cast<T>(result + PSI_10);
      return;
    }

    compute_type y = 0;
    if (x < 1.0e17) {
      compute_type z = 1.0 / (x * x);
      compute_type polevl_result = 0;
      for (int i = 0; i <= 6; i++) {
        polevl_result = polevl_result * z + A[i];
      }
      y = z * polevl_result;
    }

    v = scalar_cast<T>(DP::log(x) - (0.5 / x) - y + result);
    return;
  }


};

template <typename T, typename accreal>
struct TensorErfinvOp {
  void operator()(T& out, T& in) const {

    using compute_type = typename std::conditional<std::is_same<T, at::Half>::value, accreal, T>::type;
    compute_type z, num, dem;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type a[4]={ 0.886226899, -1.645349621,  0.914624893, -0.140543331};
    static const compute_type b[4]={-2.118377725,  1.442710462, -0.329097515,  0.012229801};
    static const compute_type c[4]={-1.970840454, -1.624906493,  3.429567803,  1.641345311};
    static const compute_type d[2]={ 3.543889200,  1.637067800};

    auto x = scalar_cast<compute_type>(in);
    if(DP::fabs(x) > 1.0) {
      out = scalar_cast<T>(NAN);
      return;
    }
    if(DP::fabs(x) == 1.0) {
      out = scalar_cast<T>((DP::copysign(1.0, scalar_cast<double>(x))) * (scalar_cast<double>(INFINITY)));
      return;
    }
    if(DP::fabs(x) <= 0.7){
      z = x * x;
      num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
      dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0])*z + scalar_cast<compute_type>(1.0));
      out = x * num / dem;
    }
    else {
      z = scalar_cast<compute_type>(DP::sqrt(-DP::log((1.0-DP::fabs(x))/2.0)));
      num = ((c[3]*z + c[2])*z + c[1])*z + c[0];
      dem = (d[1]*z + d[0])*z + scalar_cast<compute_type>(1.0);
      out = scalar_cast<T>(scalar_cast<compute_type>(DP::copysign(1.0,scalar_cast<double>(x)))*num/dem);
    }
    out = out - scalar_cast<T>((DP::erf(scalar_cast<double>(out)) - x)/((2.0/DP::sqrt(PI_f64))*DP::exp(-x*x)));
    out = out - scalar_cast<T>((DP::erf(scalar_cast<double>(out)) - x)/((2.0/DP::sqrt(PI_f64))*DP::exp(-x*x)));
    return;
  }

  void operator()(T& v) const {

    using compute_type = typename std::conditional<std::is_same<T, at::Half>::value, accreal, T>::type;
    compute_type z, num, dem;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type a[4]={ 0.886226899, -1.645349621,  0.914624893, -0.140543331};
    static const compute_type b[4]={-2.118377725,  1.442710462, -0.329097515,  0.012229801};
    static const compute_type c[4]={-1.970840454, -1.624906493,  3.429567803,  1.641345311};
    static const compute_type d[2]={ 3.543889200,  1.637067800};

    auto x = scalar_cast<compute_type>(v);
    if(DP::fabs(x) > 1.0) {
      v = scalar_cast<T>(NAN);
      return;
    }
    if(DP::fabs(x) == 1.0){
      v = scalar_cast<T>((DP::copysign(1.0, scalar_cast<double>(x))) * (scalar_cast<double>(INFINITY)));
      return;
    }
    if(DP::fabs(x) <= 0.7){
      z = x * x;
      num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
      dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0])*z + scalar_cast<compute_type>(1.0));
      v = x * num / dem;
    }
    else {
      z = scalar_cast<compute_type>(DP::sqrt(-DP::log((1.0-DP::fabs(x))/2.0)));
      num = ((c[3]*z + c[2])*z + c[1])*z + c[0];
      dem = (d[1]*z + d[0])*z + scalar_cast<compute_type>(1.0);
      v = scalar_cast<T>(scalar_cast<compute_type>(DP::copysign(1.0,scalar_cast<double>(x)))*num/dem);
    }
    v = v - scalar_cast<T>((DP::erf(scalar_cast<double>(v)) - x)/((2.0/DP::sqrt(PI_f64))*DP::exp(-x*x)));
    v = v - scalar_cast<T>((DP::erf(scalar_cast<double>(v)) - x)/((2.0/DP::sqrt(PI_f64))*DP::exp(-x*x)));
    return;
  }
};

template <typename T>
struct TensorSignOp {
  void operator()(T& out, T& in) const {
    T orig = in;
    out = (orig > 0) - (orig < 0);
  }

  void operator()(T& v) const {
    T orig = v;
    v = (orig > 0) - (orig < 0);
  }
};

template <>
struct TensorSignOp<unsigned char> {
  void operator()(unsigned char& out, unsigned char& in) const {
    unsigned char orig = in;
    out = (orig == 0) ? 0 : 1;
  }

  void operator()(unsigned char& v) const {
    unsigned char orig = v;
    v = (orig == 0) ? 0 : 1;
  }
};

template <>
struct TensorSignOp<bool> {
  void operator()(bool& out, bool& in) const {
    out = in;
  }

  void operator()(bool& v) const { }
};

template <typename T>
struct TensorClampOp {
  TensorClampOp(T min, T max) : minValue(min), maxValue(max) {}
  void operator()(T& out, T& in) const {
    T val = Numerics<T>::lt(in, maxValue) ? in : maxValue;
    out = Numerics<T>::gt(minValue, val) ? minValue : val;
  }

  void operator()(T& v) const {
    T val = Numerics<T>::lt(v, maxValue) ? v : maxValue;
    v = Numerics<T>::gt(minValue, val) ? minValue : val;
  }

  const T minValue;
  const T maxValue;
};

template <typename T>
struct TensorLerpOp {
  TensorLerpOp(T w) : w(w) {}

  void operator()(T &out, T &a, T &b) const {
    out = Numerics<T>::add(
      a,
      Numerics<T>::mul(
          w,
          Numerics<T>::sub(b, a)
        )
    );
  }

  const T w;
};


template <typename T>
struct TensorMaxOp {
  void operator()(T& out, T& in) const {
    out = Numerics<T>::gt(out, in) ? out : in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::gt(in1, in2) ? in1 : in2;
  }
};

template <typename T>
struct TensorMinOp {
  void operator()(T& out, T& in) const {
    out = Numerics<T>::lt(out, in) ? out : in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::lt(in1, in2) ? in1 : in2;
  }
};


template <typename T>
struct TensorMaxValueOp {
  TensorMaxValueOp(T v) : val(v) {}

  inline void operator()(T& out) const {
    out = Numerics<T>::lt(out, val) ? val : out;  // this order propagates NaN
  }

  inline void operator()(T& out, T& in) const {
    out = Numerics<T>::lt(in, val) ? val : in;  // this order propagates NaN
  }

  T val;
};

template <typename T>
struct TensorMinValueOp {
  TensorMinValueOp(T v) : val(v) {}

  void operator()(T& out) const {
    out = Numerics<T>::gt(out, val) ? val : out;  // this order propagates NaN
  }

  void operator()(T& out, T& in) const {
    out = Numerics<T>::gt(in, val) ? val : in;  // this order propagates NaN
  }

  T val;
};

template <typename T>
struct TensorAddCMulOp {
  TensorAddCMulOp(T v) : val(v) {}

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::add(
      out,
      Numerics<T>::mul(
        val,
        Numerics<T>::mul(in1, in2)
      )
    );
  }

  T val;
};

template <typename T>
struct TensorAddCDivOp {
  TensorAddCDivOp(T v) : val(v) {}

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::add(
      out,
      Numerics<T>::mul(
        val,
        Numerics<T>::div(in1, in2)
      )
    );
  }

  T val;
};


template <typename T>
struct TensorBitAndOp {
  void operator()(T& out, T& in) const {
    out &= in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 & in2;
  }
};

template <typename T>
struct TensorBitOrOp {
  void operator()(T& out, T& in) const {
    out |= in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 | in2;
  }
};

template <typename T>
struct TensorBitXorOp {
  void operator()(T& out, T& in) const {
    out ^= in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 ^ in2;
  }
};


template<typename T>
static typename std::enable_if<std::is_signed<T>::value, bool>::type
modulo_wrap(T a, T b) {
    return (a != 0) && (a < 0) != (b < 0);
}

template<typename T> typename std::enable_if<std::is_unsigned<T>::value, bool>::type
modulo_wrap(T a, T b) {
    return false;
}

template <typename T>
struct TensorCRemainderOp {
  void operator()(T& out, T& in) const {
    T val =  out % in;
    if (modulo_wrap(val, in)) {
      val += in;
    }
    out = val;
  }

  void operator()(T& out, T& in1, T& in2)const {
    T val = in1 % in2;
    if (modulo_wrap(val, in2)) {
      val += in2;
    }
    out = val;
  }
};

template <>
struct TensorCRemainderOp<float> {
  void operator()(float& out, float& in) const {
    out = in != 0.f ? out - in * DP::floor(out / in) : NAN;
  }

  void operator()(float& out, float& in1, float& in2) const {
    out = in2 != 0.f ? in1 - in2 * DP::floor(in1 / in2) : NAN;
  }
};

template <>
struct TensorCRemainderOp<double> {
  void operator()(double& out, double& in) const {
    out = in != 0. ? out - in * DP::floor(out / in) : NAN;
  }

  void operator()(double& out, double& in1, double& in2) const {
    out = in2 != 0. ? in1 - in2 * DP::floor(in1 / in2) : NAN;
  }
};

template <>
struct TensorCRemainderOp<at::Half> {
  void operator()(at::Half& out, at::Half& in) const {
    out = in != 0.f ? out - in * DP::floor(float(out / in)) : NAN;
  }

  void operator()(at::Half& out, at::Half& in1, at::Half& in2) const {
    out = in2 != 0.f ? in1 - in2 * DP::floor(float(in1 / in2)) : NAN;
  }
};


template <typename T>
struct TensorCFmodOp {
  void operator()(T &out, T &in) const {
    out = out % in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 % in2;
  }
};

template <>
struct TensorCFmodOp<float> {
  void operator()(float& out, float& in) const {
    out = DP::fmod(out, in);
  }

  void operator()(float& out, float& in1, float& in2) const {
    out = DP::fmod(in1, in2);
  }
};

template <>
struct TensorCFmodOp<double> {
  void operator()(double& out, double& in) const {
    out = DP::fmod(out, in);
  }

  void operator()(double& out, double& in1, double& in2) const {
    out = DP::fmod(in1, in2);
  }
};


template <>
struct TensorCFmodOp<at::Half> {
  void operator()(at::Half& out, at::Half& in) const {
    out = DP::fmod(float(out), float(in));
  }

  void operator()(at::Half& out, at::Half& in1, at::Half& in2) const {
    out = DP::fmod(float(in1), float(in2));
  }
};

template <typename T>
struct TensorCPowOp {
  void operator()(T &out, T &in) const{
    out = Numerics<T>::pow(out, in);
  }

  void operator()(T& out, T& in1, T& in2) const{
    out = Numerics<T>::pow(in1, in2);
  }
};

template <>
struct TensorCPowOp<float> {
  void operator()(float &out, float &in) const{
    out = DP::pow(out, in);
  }

  void operator()(float &out, float &in1, float &in2) const{
    out = DP::pow(in1, in2);
  }
};

template <>
struct TensorCPowOp<double> {
  void operator()(double &out, double &in) const{
    out = DP::pow(out, in);
  }

  void operator()(double &out, double &in1, double &in2) const{
    out = DP::pow(in1, in2);
  }
};

template<typename T, int StaticExp>
struct TensorPowOp {
  TensorPowOp(T v) : val(v) {}
  void operator()(T &out, T &in) const{
    if (StaticExp == 1) {
      out = in;
    } else if (StaticExp == 2) {
      out = Numerics<T>::mul(in, in);
    } else if (StaticExp == 3) {
      T square = Numerics<T>::mul(in, in);
      out = Numerics<T>::mul(square, in);
    } else {
      out = Numerics<T>::pow(in, val);
    }
  }

  void operator()(T &v) const{
    if (StaticExp == 1) {
      v = v;
    } else if (StaticExp == 2) {
      v = Numerics<T>::mul(v, v);
    } else if (StaticExp == 3) {
      v = Numerics<T>::mul(Numerics<T>::mul(v, v), v);
    } else {
      v = Numerics<T>::pow(v, val);
    }
  }

  const T val;
};

template<typename T>
struct TensorPowOp<T, -1> {
  TensorPowOp(T v) : val(v) {}
  void operator()(T &out, T &in) const{
    out = Numerics<T>::cinv(in);
  }

  void operator()(T &v) const{
    v = Numerics<T>::cinv(v);
  }

  const T val;
};

template<typename T>
struct TensorPowOp<T, -2> {
  TensorPowOp(T v) : val(v) {}
  void operator()(T &out, T &in) const{
    T square = Numerics<T>::mul(in, in);
    out = Numerics<T>::cinv(square);
  }

  void operator()(T &v) const{
    T square = Numerics<T>::mul(v, v);
    v = Numerics<T>::cinv(square);
  }

  const T val;
};

template<typename T>
struct TensorTPowOp {
  TensorTPowOp(T v) : val(v) {}

  void operator()(T &out, T &in) const{
    out = Numerics<T>::pow(val, in);
  }

  void operator()(T &v) const{
    v = Numerics<T>::pow(val, v);
  }

  const T val;
};
