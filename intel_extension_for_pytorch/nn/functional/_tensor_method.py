import torch
from torch.overrides import has_torch_function_unary, handle_torch_function
from ...utils._logger import logger


def _numpy(x, force=False):
    if x.dtype == torch.bfloat16:
        logger.warning(
            "calling in ipex numpy which is not share memory with torch tensor for bfloat16 input."
        )
        return torch._C._TensorBase.numpy(x.float(), force=force)
    else:
        return torch._C._TensorBase.numpy(x, force=force)


# Fix https://github.com/pytorch/pytorch/issues/82764
def __format__(self: torch.Tensor, format_spec):
    if has_torch_function_unary(self):
        return handle_torch_function(
            torch.Tensor.__format__, (self,), self, format_spec
        )
    if self.dim() == 0 and not self.is_meta and issubclass(type(self), torch.Tensor):
        return self.item().__format__(format_spec)
    return object.__format__(self, format_spec)


torch.Tensor.numpy = _numpy
torch.Tensor.__format__ = __format__
