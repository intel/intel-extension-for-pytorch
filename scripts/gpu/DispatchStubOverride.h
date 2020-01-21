static Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking);
static Tensor & fill_(Tensor & self, Scalar value);
static Tensor & fill_(Tensor & self, const Tensor & value);

static Tensor threshold(const Tensor & self, Scalar threshold, Scalar value);
static Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value);
static Tensor & threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value);


//static Tensor bitwise_not(const Tensor & self);
//static Tensor & bitwise_not_(Tensor & self);
//static Tensor & bitwise_not_out(Tensor & out, const Tensor & self);

//