import torch

def __ipex_tensor_deepcopy__(self, memo):
    if self.device.type == 'dpcpp':
        with torch.no_grad():
            new_tensor = self.clone()
            return new_tensor
    else:
        return self.__deepcopy__(memo)

torch.Tensor.__deepcopy__ = __ipex_tensor_deepcopy__
