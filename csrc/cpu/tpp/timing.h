#ifndef _BERT_TIMING_H_
#define _BERT_TIMING_H_

#include "utils.h"
namespace torch_ipex {
namespace tpp {
enum DebugTimer {
  BRGEMM,
  XPOSE,
  DROPOUT,
  LAYER_NORM,
  SOFTMAX,
  ACT,
  BIAS,
  VNNI,
  EW_COPY,
  EW_ADD,
  EW_SCL,
  EW_RECP,
  EW_RECP_SQRT,
  EW_MUL,
  EW_ZERO,
  EW_RED,
  OPTIM,
  LAST_TIMER
};

inline const char* DebugTimerName(int t) {
  const char* names[] = {
      "BRGEMM",
      "XPOSE",
      "DROPOUT",
      "LYR_NRM",
      "SOFTMAX",
      "ACT",
      "BIAS",
      "VNNI",
      "COPY",
      "ADD",
      "SCALE",
      "RECP",
      "RECP_SQRT",
      "MUL",
      "ZERO",
      "REDUCE",
      "OPTIM",
      "LAST_TIMER"};
  return names[t];
}

enum PassType { OTH, FWD, BWD, UPD };

extern PassType globalPass;
extern int globalScope;
constexpr int NUM_TIMERS = ((LAST_TIMER + 7) / 8) * 8;
extern double pass_timers[MAX_THREADS][3][NUM_TIMERS];
extern double master_pass_timers[3];
struct Scope {
  Scope(std::string const& name)
      : name(name), master_timer(0.0), detailed_timers{0.0}, flops{0.0} {}
  const std::string name;
  double master_timer;
  double detailed_timers[MAX_THREADS][NUM_TIMERS];
  double flops[MAX_THREADS][8];
};

inline std::vector<Scope>& get_scope_list() {
  static std::vector<Scope> _scope_list{Scope("Reserved")};
  return _scope_list;
}

inline std::vector<Scope>& get_pass_list() {
  static std::vector<Scope> _pass_list{
      Scope("OTH"), Scope("FWD"), Scope("BWD"), Scope("UPD")};
  return _pass_list;
}

inline int register_scope(std::string name) {
  auto& _scope_list = get_scope_list();
  _scope_list.emplace_back(name);
  int idx = _scope_list.size() - 1;
  // printf("Registering %s scope @%d\n", name.c_str(), idx);
  return idx;
}

#ifdef PROFILE_TPP
#define REGISTER_LOCAL_SCOPE(id, name) static int sc_##id = register_scope(name)
#define REGISTER_SCOPE(id, name) int sc_##id = register_scope(name)
#define USING_SCOPE(id) extern int sc_##id
#else
#define REGISTER_LOCAL_SCOPE(id, name)
#define REGISTER_SCOPE(id, name)
#define USING_SCOPE(id)
#endif

class ScopedTimer {
 public:
  ScopedTimer(DebugTimer t, long f = 0) : type(t), flops(f), start(getTime()) {}
  ~ScopedTimer() {
    auto time = getTime() - start;
    int tid = omp_get_thread_num();
    auto& pass = get_pass_list()[globalPass];
    pass.detailed_timers[tid][type] += time;
    if (type == BRGEMM)
      pass.flops[tid][0] += flops;
    if (globalPass == 0 && tid == 0)
      pass.master_timer += time;

    auto& scope = get_scope_list()[globalScope];
    scope.detailed_timers[tid][type] += time;
    if (type == BRGEMM)
      scope.flops[tid][0] += flops;
    if (globalScope == 0 && tid == 0)
      scope.master_timer += time;
  }
  DebugTimer type;
  long flops;
  double start;
};

class GlobalScope {
 public:
  GlobalScope(int t) : oldScope(globalScope), start(getTime()) {
    PCL_ASSERT(t < (int)get_scope_list().size(), "Invalid scope initialized");
    globalScope = t;
  }
  ~GlobalScope() {
    auto time = getTime() - start;
    auto& scope = get_scope_list()[globalScope];
    scope.master_timer += time;
    if (oldScope != 0) {
      // Remove time from outer scope
      auto& outer_scope = get_scope_list()[oldScope];
      outer_scope.master_timer -= time;
    }
    globalScope = oldScope;
  }
  int oldScope;
  double start;
};

class GlobalPass {
 public:
  GlobalPass(PassType p) : oldPass(globalPass), start(getTime()) {
    globalPass = p;
  }
  ~GlobalPass() {
    auto time = getTime() - start;
    auto& pass = get_pass_list()[globalPass];
    pass.master_timer += time;
    if (oldPass != 0) {
      auto& outer_pass = get_pass_list()[oldPass];
      outer_pass.master_timer -= time;
    }
    globalPass = oldPass;
  }
  PassType oldPass;
  double start;
};

#ifdef DEBUG_TRACE_TPP
static thread_local std::string prev_class_name = "";
#endif
template <typename T, int impl = 0>
class ScopedTPP {
 public:
  ScopedTPP(T func, DebugTimer t) : func(std::move(func)), t(t) {}
  template <typename... Types>
  void operator()(Types... vars) {
    ScopedTimer _t(t);
#ifdef DEBUG_TRACE_TPP
    if (omp_get_thread_num() == 0) {
      auto cur_class_name = get_class_name<T>();
      if (cur_class_name != prev_class_name) {
        std::cout << "Calling impl " << impl << " for " << cur_class_name
                  << std::endl;
        prev_class_name = cur_class_name;
      }
    }
#endif
    if (impl == 0) {
      func(vars...);
    } else if (impl == 1) {
      func.ref(vars...);
    } else {
      printf("invalid impl requested\n");
      exit(1);
    }
  }

 private:
  T func;
  DebugTimer t;
};

// Keeping below two definitions for backward compatibility for now
#define SCOPEITGEMM SCOPEIT
#define SCOPEITGEMM2 SCOPEIT

#ifdef PROFILE_TPP
#define SCOPEIT(f, ...) ScopedTPP<decltype(f), 0>(f, ##__VA_ARGS__)
#define SCOPEIT_REF(f, ...) ScopedTPP<decltype(f), 1>(f, ##__VA_ARGS__)
#define RECORD_SCOPE(scope, ...) \
  GlobalScope gs_(sc_##scope);   \
  RECORD_FUNCTION(#scope, std::vector<c10::IValue>(__VA_ARGS__))
#else
#define SCOPEIT(f, ...) f
#define RECORD_SCOPE(scope, ...)
#endif

} // namespace tpp
} // namespace torch_ipex
#endif //_BERT_TIMING_H_
