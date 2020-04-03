
  // Sparse Attribute interfaces
  static int64_t _dimI(const at::Tensor & self);
  static int64_t _dimV(const at::Tensor & self);
  static int64_t sparse_dim(const at::Tensor & self);
  static int64_t dense_dim(const at::Tensor & self);
  static int64_t _nnz(const at::Tensor & self);
  static bool is_coalesced(const at::Tensor & self);
  static at::Tensor & _coalesced_(at::Tensor & self, bool coalesced);
  static at::Tensor indices(const at::Tensor& self);
  static at::Tensor values(const at::Tensor& self);
  static at::Tensor _indices(const at::Tensor & self);
  static at::Tensor _values(const at::Tensor & self);

  // Other sparse Ops
