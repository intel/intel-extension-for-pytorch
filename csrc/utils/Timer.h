#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace std::chrono;

struct ipex_timer {
 public:
  ipex_timer(int verbose_level, std::string tag);

  void now(std::string sstamp);

  void event_duration(uint64_t us);

  ~ipex_timer();

 private:
  int vlevel_;
  std::string tag_;
  time_point<high_resolution_clock> start_;
  std::vector<time_point<high_resolution_clock>> stamp_;
  std::vector<std::string> sstamp_;
  std::vector<uint64_t> event_duration_;
};

#define IPEX_TIMER(t, ...) struct ipex_timer t(__VA_ARGS__)
