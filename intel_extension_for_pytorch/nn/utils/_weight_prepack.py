import torch
import copy
import intel_extension_for_pytorch._C as core

IPEX_WEIGHT_PREPACK_MODULE = {
    torch.nn.Linear: core.convert_linear_weight_layout,
    torch.nn.Conv1d: core.convert_conv_weight_layout,
    torch.nn.Conv2d: core.convert_conv_weight_layout,
    torch.nn.Conv3d: core.convert_conv_weight_layout,
    torch.nn.ConvTranspose2d: core.convert_convtranspose_weight_layout,
    torch.nn.ConvTranspose3d: core.convert_convtranspose_weight_layout,
}


def _should_prepack(module):
    if type(module) not in IPEX_WEIGHT_PREPACK_MODULE:
        return False
    # If hook is on `weight` or `bias`, will not prepack.
    if module._forward_pre_hooks is not None:
        for _, hook in module._forward_pre_hooks.items():
            if hasattr(hook, 'name') and (hook.name == 'weight' or hook.name == 'bias'):
                return False
    if module._forward_hooks is not None:
        for _, hook in module._forward_hooks.items():
            if hasattr(hook, 'name') and (hook.name == 'weight' or hook.name == 'bias'):
                return False
    if module._backward_hooks is not None:
        for _, hook in module._backward_hooks.items():
            if hasattr(hook, 'name') and (hook.name == 'weight' or hook.name == 'bias'):
                return False

    if isinstance(module, torch.nn.ConvTranspose2d):
        if module.padding[0] - module.output_padding[0] + module.stride[0] <= 0:
            return False
        if module.padding[1] - module.output_padding[1] + module.stride[1] <= 0:
            return False
    if isinstance(module, torch.nn.ConvTranspose3d):
        if module.padding[0] - module.output_padding[0] + module.stride[0] <= 0:
            return False
        if module.padding[1] - module.output_padding[1] + module.stride[1] <= 0:
            return False
        if module.padding[2] - module.output_padding[2] + module.stride[2] <= 0:
            return False
    return True


def weight_prepack_with_ipex(module):
    # for now, only weight prepack for conv and conv transpose
    if _should_prepack(module):
        # if pass the sample input, the activation shape will be recorded
        prepack_input_shape = module.input_shape if hasattr(module, "input_shape") else []
        if type(module) == torch.nn.ConvTranspose2d or type(module) == torch.nn.ConvTranspose3d:
            # Conv Transpose needs output_padding
            IPEX_WEIGHT_PREPACK_MODULE[type(module)](module.weight.data,
                                                     module.padding,
                                                     module.stride,
                                                     module.dilation,
                                                     module.output_padding,
                                                     module.groups,
                                                     prepack_input_shape)
        elif type(module) == torch.nn.Linear:
            # After prepack, the context of weight has been changed to transpose + block(BA-block),
            # while the stride of weight TensorImpl is not been changed(still AB-plain).
            # So in torch addmm shape check without transpose, it will fail.
            # If let torch now the true stride change(transpose) of the weight, the .t() is needed, it will trigger to_plain.
            # Thus, here, use return weight method
            module.weight.data = IPEX_WEIGHT_PREPACK_MODULE[type(module)](module.weight.data, prepack_input_shape)
        else:
            # For Conv1d, 2d and 3d
            IPEX_WEIGHT_PREPACK_MODULE[type(module)](module.weight.data,
                                                     module.padding,
                                                     module.stride,
                                                     module.dilation,
                                                     module.groups,
                                                     prepack_input_shape)

    for child in module.children():
        weight_prepack_with_ipex(child)
    return module


def record_input_shape_for_prepack(module, sample_input):

    def hook_function(self, input):
        # input for linear/conv/transpose conv received here will be Tuple[Tensor]
        self.input_shape = input[0].shape

    def register_hook_function(module):
        # the range is aligned with CPU team
        if type(module) in [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
            module.register_forward_pre_hook(hook_function)

    def register_hook_function_rec(module):
        register_hook_function(module)
        for child in module.children():
            register_hook_function_rec(child)

    origin_state_dict = copy.deepcopy(module.state_dict())
    register_hook_function_rec(module)
    module(*sample_input)
    module.load_state_dict(origin_state_dict)
