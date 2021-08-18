import torch

org_tensor_deep_copy = torch.Tensor.__deepcopy__

def __ipex_tensor_deepcopy__(self, memo):
    if self.device.type == 'xpu':
        with torch.no_grad():
            new_tensor = self.clone()
            return new_tensor
    else:
        return org_tensor_deep_copy(self, memo)

torch.Tensor.__deepcopy__ = __ipex_tensor_deepcopy__
