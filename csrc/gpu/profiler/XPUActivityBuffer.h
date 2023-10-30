#pragma once

#ifdef USE_KINETO

#include <assert.h>
#include <stdlib.h>
#include <map>
#include <memory>
#include <vector>

#include <profiler/include/kineto/ITraceActivity.h>
#include "onepti_activity_api.h"

namespace KINETO_NAMESPACE {

class XPUActivityBuffer {
 public:
  explicit XPUActivityBuffer(std::vector<uint8_t>* ipex_buf_) {
    // buf_->reserve(size);
    buf_ = std::move(ipex_buf_);
  }
  XPUActivityBuffer() = delete;
  XPUActivityBuffer& operator=(const XPUActivityBuffer&) = delete;
  XPUActivityBuffer(XPUActivityBuffer&&) = default;
  XPUActivityBuffer& operator=(XPUActivityBuffer&&) = default;

  ~XPUActivityBuffer() {
    buf_ = nullptr;
    size_ = 0;
    wrappers_.clear();
  }

  size_t size() const {
    return buf_->size();
    // return size_;
  }

  void setSize(size_t size) {
    assert(size <= buf_->capacity());
    size_ = size;
  }

  uint8_t* data() {
    return buf_->data();
  }

 private:
  std::vector<uint8_t>* buf_;
  size_t size_;
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
};

using XPUActivityBufferMap =
    std::map<uint8_t*, std::unique_ptr<XPUActivityBuffer>>;

} // namespace KINETO_NAMESPACE
#endif
