#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace std::chrono;

struct ipex_timer {
  ipex_timer(int verbose_level, std::string tag) :
      vlevel_(verbose_level) {
    if (verbose_level >= 1) {
      start_ = high_resolution_clock::now();
      tag_ = tag;
    }
  }

  void now(std::string sstamp) {
    if (vlevel_ >= 1) {
      stamp_.push_back(high_resolution_clock::now());
      sstamp_.push_back(sstamp);
    }
  }

  ~ipex_timer() {
    if (vlevel_ >= 1) {
      auto pre = start_;
      std::cout << "<" << tag_ << ">";
      for (int i = 0; i < stamp_.size(); i++) {
        auto stamp = stamp_.at(i);
        auto sstamp = sstamp_.at(i);
        std::cout << " " << sstamp << "(" <<
            duration_cast<microseconds>(stamp - pre).count() << "us)";
        pre = stamp;
      }
      std::cout << " total(" <<
          duration_cast<microseconds>(stamp_.back() - start_).count() <<
          "us)" << std::endl;
    }
  }

  int vlevel_;
  std::string tag_;
  time_point<high_resolution_clock> start_;
  std::vector<time_point<high_resolution_clock>> stamp_;
  std::vector<std::string> sstamp_;
};

#define IPEX_TIMER(t, ...) struct ipex_timer t(__VA_ARGS__)
