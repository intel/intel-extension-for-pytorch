from torch._dynamo import register_backend
from .compiler import compile


@register_backend
def ipex(model, inputs):
    return compile(model, inputs)
