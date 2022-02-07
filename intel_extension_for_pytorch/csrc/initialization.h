#include <utility>

namespace torch_ipex {

class InitIPEX final {
 public:
  InitIPEX();
  ~InitIPEX();

  InitIPEX(const InitIPEX&) = delete;
  InitIPEX& operator=(const InitIPEX&) = delete;
  InitIPEX(InitIPEX&&) noexcept;
  InitIPEX& operator=(InitIPEX&&) noexcept;

  InitIPEX&& init(void (*func)()) && {
    func();
    return std::move(*this);
  }

  InitIPEX& init(void (*func)()) & {
    func();
    return *this;
  }
};

} // namespace torch_ipex
