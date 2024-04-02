#include <functional>
#include <string>
#include <unordered_map>
#include "threaded_loops.h"

namespace torch_ipex {
namespace tpp {

void par_nested_loops_A(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for nowait
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      int ind[1] = {a0};
      body_func(ind);
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_AB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(2) nowait
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_aB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for nowait
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_bA(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
         b0 += loopSpecs[1].step) {
#pragma omp for nowait
      for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
           a0 += loopSpecs[0].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_Ba(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for nowait
    for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
         b0 += loopSpecs[1].step) {
      for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
           a0 += loopSpecs[0].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_BA(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(2) nowait
    for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
         b0 += loopSpecs[1].step) {
      for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
           a0 += loopSpecs[0].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_ABC(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(3) nowait
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
             c0 += loopSpecs[2].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_aBC(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for collapse(2) nowait
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
             c0 += loopSpecs[2].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_acB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
           c0 += loopSpecs[2].step) {
#pragma omp for nowait
        for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
             b0 += loopSpecs[1].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

static void par_nested_loops_ABc(
    LoopSpecs* loop_rt_spec,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> term_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(2) nowait
    for (int a0 = loop_rt_spec[0].start; a0 < loop_rt_spec[0].end;
         a0 += loop_rt_spec[0].step) {
      for (int b0 = loop_rt_spec[1].start; b0 < loop_rt_spec[1].end;
           b0 += loop_rt_spec[1].step) {
        for (int c0 = loop_rt_spec[2].start; c0 < loop_rt_spec[2].end;
             c0 += loop_rt_spec[2].step) {
          int idx[3];
          idx[0] = a0;
          idx[1] = b0;
          idx[2] = c0;
          body_func(idx);
        }
      }
    }
    if (term_func)
      term_func();
  }
}

void par_nested_loops_aCb(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for nowait
      for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
           c0 += loopSpecs[2].step) {
        for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
             b0 += loopSpecs[1].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_aCB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for collapse(2) nowait
      for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
           c0 += loopSpecs[2].step) {
        for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
             b0 += loopSpecs[1].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

static void par_nested_loops_CAB(
    LoopSpecs* loop_rt_spec,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> term_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(3) nowait
    for (int c0 = loop_rt_spec[2].start; c0 < loop_rt_spec[2].end;
         c0 += loop_rt_spec[2].step) {
      for (int a0 = loop_rt_spec[0].start; a0 < loop_rt_spec[0].end;
           a0 += loop_rt_spec[0].step) {
        for (int b0 = loop_rt_spec[1].start; b0 < loop_rt_spec[1].end;
             b0 += loop_rt_spec[1].step) {
          int idx[3];
          idx[0] = a0;
          idx[1] = b0;
          idx[2] = c0;
          body_func(idx);
        }
      }
    }
    if (term_func)
      term_func();
  }
}

static void par_nested_loops_ACb(
    LoopSpecs* loop_rt_spec,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> term_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(2) nowait
    for (int a0 = loop_rt_spec[0].start; a0 < loop_rt_spec[0].end;
         a0 += loop_rt_spec[0].step) {
      for (int c0 = loop_rt_spec[2].start; c0 < loop_rt_spec[2].end;
           c0 += loop_rt_spec[2].step) {
        for (int b0 = loop_rt_spec[1].start; b0 < loop_rt_spec[1].end;
             b0 += loop_rt_spec[1].step) {
          int idx[3];
          idx[0] = a0;
          idx[1] = b0;
          idx[2] = c0;
          body_func(idx);
        }
      }
    }
    if (term_func)
      term_func();
  }
}

static void par_nested_loops_ABCD(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(4) nowait
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
             c0 += loopSpecs[2].step) {
          for (int d0 = loopSpecs[3].start; d0 < loopSpecs[3].end;
               d0 += loopSpecs[3].step) {
            int ind[4] = {a0, b0, c0, d0};
            body_func(ind);
          }
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

std::unordered_map<std::string, par_loop_kernel> pre_defined_loops = {
    {"A", par_nested_loops_A},
    {"AB", par_nested_loops_AB},
    {"BA", par_nested_loops_BA},
    {"bA", par_nested_loops_bA},
    {"Ba", par_nested_loops_Ba},
    {"aB", par_nested_loops_aB},
    {"ABC", par_nested_loops_ABC},
    {"aBC", par_nested_loops_aBC},
    {"acB", par_nested_loops_acB},
    {"aCb", par_nested_loops_aCb},
    {"aCB", par_nested_loops_aCB},
    {"ABc", par_nested_loops_ABc},
    {"CAB", par_nested_loops_CAB},
    {"ACb", par_nested_loops_ACb},
    {"ABCD", par_nested_loops_ABCD},
};
} // namespace tpp
} // namespace torch_ipex