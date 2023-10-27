#pragma once

#include <fmt/format.h>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <string>

namespace KINETO_NAMESPACE {

class ActivityLogger;

class ActivityLoggerFactory {
 public:
  using FactoryFunc =
      std::function<std::unique_ptr<ActivityLogger>(const std::string& url)>;

  void addProtocol(const std::string& protocol, FactoryFunc f) {
    factories_[tolower(protocol)] = f;
  }

  std::unique_ptr<ActivityLogger> makeLogger(const std::string& url) const {
    std::string protocol = extractProtocol(url);
    auto it = factories_.find(tolower(protocol));
    if (it != factories_.end()) {
      return it->second(stripProtocol(url));
    }
    throw std::invalid_argument(fmt::format(
        "No logger registered for the {} protocol prefix", protocol));
    return nullptr;
  }

 private:
  static std::string tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    return s;
  }

  static std::string extractProtocol(std::string url) {
    return url.substr(0, url.find("://"));
  }

  static std::string stripProtocol(std::string url) {
    size_t pos = url.find("://");
    return pos == url.npos ? url : url.substr(pos + 3);
  }

  std::map<std::string, FactoryFunc> factories_;
};

} // namespace KINETO_NAMESPACE
