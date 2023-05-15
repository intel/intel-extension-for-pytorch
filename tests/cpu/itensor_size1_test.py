import torch
import intel_extension_for_pytorch as ipex

# This script is called and tested by test_conv_reorder.py, and its purpose is:
# (1) This script is testing the case that conv grad tensor shape[n, 1, h ,w] stride[h*w, 1 , w, 1],
# where its stride can be considered as both default contiguous and channelslast by PyTorch.
# (2) The main confusing thing in this case is that since it has tensor size 1, this size's stirde
# will be ignored by PyTorch (due to meanless for size 1). But for shape [n, 1, h ,w], stride
# [h*w, h*w , w, 1] is strictly default contiguous and [h*w, 1 , w, 1] is strictly channelslast.
# (3) We consider such case to remain strictly channelslast stride (calling into oneDNN), since
# channelslast is with priority.
# (4) So we do not expect any reorder of "plainformat <-> channelslast" on conv op src/dst (fwd and bwd).
# The reoders should only have 3 on this script, which are all for weight format.
m = torch.nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
m = m.to(memory_format=torch.channels_last)
m.train()
x = torch.randn(8, 2, 224, 224).to(memory_format=torch.channels_last).requires_grad_()
origin_optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
example_input = torch.randn(8, 2, 224, 224)
ipex_model, ipex_optimizer = ipex.optimize(
    m,
    dtype=torch.float,
    optimizer=origin_optimizer,
    level="O1",
    sample_input=example_input,
)
y = ipex_model(x).sum()
ipex_optimizer.zero_grad()
y.backward()
ipex_optimizer.step()
