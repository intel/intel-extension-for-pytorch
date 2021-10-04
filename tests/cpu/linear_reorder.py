import torch
import intel_extension_for_pytorch as ipex
import torch.nn as nn
import itertools

class Model(nn.Module):
    def __init__(self, ic, oc, bias):
        super(Model, self).__init__()
        self.linear = nn.Linear(ic, oc, bias=bias)

    def forward(self, input):
        return self.linear(input)

def run_model(dtype=None):
    out_feature = [1024, 256, 1, torch.randint(3, 10, (1, )).item()]
    in_feature = [128, 479, torch.randint(3, 10, (1, )).item()]
    input_shapes=[]
    for s in in_feature:
        input_shapes += [(128, s), (2, 64, s), (2, 2, 32, s)]
    options = itertools.product(out_feature, [True, False], input_shapes)
    for out_features, bias, x_shape in options:
        in_features = x_shape[-1]
        x = torch.randn(x_shape, dtype=torch.float32).requires_grad_()
        model = Model(in_features, out_features, bias)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
        if dtype == 0 :
            conf = ipex.AmpConf(torch.float32)
            model, optimizer = ipex.optimize(model, dtype=torch.float32, optimizer=optimizer, level='O1')
            with ipex.amp.autocast(enabled=True, configure=conf):
                run_mod = model.forward(x).sum()
        elif dtype == 1 :
            conf = ipex.AmpConf(torch.bfloat16)
            model, optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer, level='O1')
            with ipex.amp.autocast(enabled=True, configure=conf):
                run_mod = model.forward(x).sum()
        else: # reserved
            pass
        optimizer.zero_grad()
        run_mod.backward()
        optimizer.step()


if __name__ == "__main__":
    print(f"fp32, {'*' * 50}")
    run_model(0)

    print(f"bf16, {'*' * 50}")
    run_model(1)
