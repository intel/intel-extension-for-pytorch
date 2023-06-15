#ifndef _EXT_TPP_H_
#define _EXT_TPP_H_

#include "timing.h"
#include "xsmm_functors.h"

namespace torch_ipex {
namespace tpp {

template <typename Tin, typename Tout>
class BrgemmExtTPP {
 public:
  BrgemmExtTPP() {}
  BrgemmExtTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      float beta, // = 1.0,
      XformTPP::XFORM_TYPE c_trans, // = XformTPP::XFORM_NONE_TPP,
      int a_trans, // = 0,
      int unroll_hint)
      : M(M),
        N(N),
        K(K),
        beta(beta),
        c_trans(c_trans),
        brgemm(),
        xform(),
        add() {
    // auto dt_in = XsmmDtype<Tin>();
    auto dt_out = XsmmDtype<Tout>();
    if (dt_out == LIBXSMM_DATATYPE_F32 && c_trans == XformTPP::XFORM_N2V_TPP) {
      printf(
          "Warning: reseting c_trans flag from N2V to None for FP32 output\n");
      c_trans = XformTPP::XFORM_NONE_TPP;
    }
    auto beta_ = beta;

    if (c_trans != XformTPP::XFORM_NONE_TPP) {
      beta_ = 0.0;
      xform = XformExtTPP<Tout>(M, N, c_trans);
    }
    brgemm = BrgemmTPP<Tin, Tout>(
        M, N, K, str_a, str_b, beta_, a_trans, unroll_hint);
    if (beta_ != beta) {
      add = AddTPP<Tout, Tout>(M, N);
    }
    xform_type = c_trans == XformTPP::XFORM_N2V_TPP ? VNNI : XPOSE;
  }

  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      long count,
      bool no_tile_cfg = false) {
    if (c_trans == XformTPP::XFORM_NONE_TPP) {
      ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
      brgemm(A, B, C, count, no_tile_cfg);
    } else {
      Tout tmp_C[M * N];
      {
        ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
        brgemm(A, B, tmp_C, count, no_tile_cfg);
      }
      if (beta == 0.0) {
        ScopedTimer _t(xform_type);
        xform(tmp_C, C);
      } else {
        Tout tmp[M * N];
        {
          ScopedTimer _t(xform_type);
          xform(tmp_C, tmp);
        }
        {
          ScopedTimer _t(EW_ADD);
          add(C, tmp, C);
        }
      }
    }
  }

  void ref(Tin* A, Tin* B, Tout* C, long count, bool no_tile_cfg = false) {
    if (c_trans == XformTPP::XFORM_NONE_TPP) {
      ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
      brgemm.ref(A, B, C, count, no_tile_cfg);
    } else {
      Tout tmp_C[M * N];
      {
        ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
        brgemm.ref(A, B, tmp_C, count, no_tile_cfg);
      }
      if (beta == 0.0) {
        ScopedTimer _t(xform_type);
        xform.ref(tmp_C, C);
      } else {
        Tout tmp[M * N];
        {
          ScopedTimer _t(xform_type);
          xform.ref(tmp_C, tmp);
        }
        {
          ScopedTimer _t(EW_ADD);
          add.ref(C, tmp, C);
        }
      }
    }
  }

  void config() {
    brgemm.config();
  }

  void release() {
    brgemm.release();
  }

 private:
  long M, N, K;
  float beta;
  XformTPP::XFORM_TYPE c_trans;
  BrgemmTPP<Tin, Tout> brgemm;
  XformExtTPP<Tout> xform;
  AddTPP<Tout, Tout> add;
  DebugTimer xform_type;
};

template <typename Tin, typename Tout, int impl>
class ScopedTPP<BrgemmTPP<Tin, Tout>, impl> {
 public:
  ScopedTPP(BrgemmTPP<Tin, Tout> func) : func(std::move(func)) {}
  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      long count,
      bool no_tile_cfg = false) {
    ScopedTimer _t(BRGEMM, func.flops() * count);
    if (impl == 0) {
      func(A, B, C, count, no_tile_cfg);
    } else if (impl == 1) {
      func.ref(A, B, C, count, no_tile_cfg);
    } else {
      printf("invalid impl requested\n");
      exit(1);
    }
  }

  void config() {
    func.config();
  }

  void release() {
    func.release();
  }

 private:
  BrgemmTPP<Tin, Tout> func;
};

template <typename Tin, typename Tout, int impl>
class ScopedTPP<BrgemmExtTPP<Tin, Tout>, impl> {
 public:
  ScopedTPP(BrgemmExtTPP<Tin, Tout> func) : func(std::move(func)) {}
  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      long count,
      bool no_tile_cfg = false) {
    if (impl == 0) {
      func(A, B, C, count, no_tile_cfg);
    } else if (impl == 1) {
      func.ref(A, B, C, count, no_tile_cfg);
    } else {
      printf("invalid impl requested\n");
      exit(1);
    }
  }

  void config() {
    func.config();
  }

  void release() {
    func.release();
  }

 private:
  BrgemmExtTPP<Tin, Tout> func;
};
} // namespace tpp
} // namespace torch_ipex

//#define TCBrgemmTPP BrgemmTPP

#endif // _EXT_TPP_H_
