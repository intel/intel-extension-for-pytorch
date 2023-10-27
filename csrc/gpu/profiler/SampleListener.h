#pragma once

#include <assert.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace KINETO_NAMESPACE {

class Config;

class SampleValue {
 public:
  template <class T>
  explicit SampleValue(T v) {
    init<T>(v);
  }

  SampleValue(const SampleValue&) = default;
  SampleValue& operator=(const SampleValue&) = delete;
  SampleValue(SampleValue&&) = default;
  SampleValue& operator=(SampleValue&&) = default;

  bool isInt() const {
    return type_ == INT64;
  }

  int64_t getInt() const {
    assert(isInt());
    return int_;
  }

  bool isDouble() const {
    return type_ == DOUBLE;
  }

  double getDouble() const {
    assert(isDouble());
    return dbl_;
  }

  inline void operator*=(double x) {
    assert(isDouble() || isInt());
    if (isDouble()) {
      dbl_ *= x;
    } else {
      int_ = std::round(int_ * x);
    }
  }

  inline bool operator<(const SampleValue& o) const {
    if (type_ != o.type_) {
      return type_ < o.type_;
    } else if (type_ == INT64) {
      return int_ < o.int_;
    } else if (type_ == DOUBLE) {
      return dbl_ < o.dbl_;
    }
    assert(false);
    return true;
  }

  void print(std::ostream& s) const {
    if (type_ == INT64) {
      s << int_;
    } else if (type_ == DOUBLE) {
      s << dbl_;
    } else {
      assert(false);
    }
  }

 private:
  enum Type { INT64, DOUBLE };

  template <typename T>
  void init(T v);

  Type type_{INT64};
  union {
    int64_t int_{0};
    double dbl_;
  };
};

template <>
inline void SampleValue::init(uint64_t v) {
  int_ = v, type_ = INT64;
}
template <>
inline void SampleValue::init(int64_t v) {
  int_ = v, type_ = INT64;
}
template <>
inline void SampleValue::init(int v) {
  int_ = v, type_ = INT64;
}
template <>
inline void SampleValue::init(double v) {
  dbl_ = v, type_ = DOUBLE;
}

inline std::ostream& operator<<(std::ostream& out, const SampleValue& s) {
  s.print(out);
  return out;
}

using PercentileList = std::vector<std::pair<int, SampleValue>>;

struct Stat {
  const std::string& name;
  const PercentileList percentileValues;
  SampleValue total;
};

struct Sample {
  Sample(int stats_count) {
    stats.reserve(stats_count);
  }

  int deltaMsec;
  std::vector<Stat> stats;
};

class SampleListener {
 public:
  SampleListener(const SampleListener&) = delete;
  SampleListener& operator=(const SampleListener&) = delete;

  virtual ~SampleListener(){};

  virtual void handleSample(
      int device,
      const Sample& sample,
      bool from_new_version) = 0;

  virtual void update(const Config& config) = 0;

 protected:
  SampleListener() = default;
};

} // namespace KINETO_NAMESPACE
