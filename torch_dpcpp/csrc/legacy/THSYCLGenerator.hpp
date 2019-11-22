#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <mutex>

struct THSYCLGeneratorState {
  int initf;
  uint64_t initial_seed;
};

/* A THGenerator contains all the state required for a single random number stream */
struct THSYCLGenerator {
  std::mutex mutex; /* mutex for using this generator */
  THSYCLGeneratorState state;
};
