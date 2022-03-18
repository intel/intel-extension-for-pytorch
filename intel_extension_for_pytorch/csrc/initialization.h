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
    check_pytorch_version();
    func();
    return std::move(*this);
  }

  InitIPEX& init(void (*func)()) & {
    check_pytorch_version();
    func();
    return *this;
  }

 private:
  void check_pytorch_version();
};

} // namespace torch_ipex
