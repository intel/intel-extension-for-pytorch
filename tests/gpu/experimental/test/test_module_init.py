import inspect
import torch
from unittest import mock
from unittest.mock import MagicMock, patch
from torch.testing import floating_types
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.common_utils import TestCase, run_tests
from common.pytorch_test_base import TestCase, dtypesIfXPU, TEST_XPU, TEST_MULTIGPU, largeTensorTest

def build_constructor_arg_db():
    return {torch.nn.AdaptiveAvgPool1d: ((5,), {}), torch.nn.AdaptiveAvgPool2d: ((5,), {}), torch.nn.AdaptiveAvgPool3d: ((5,), {}), torch.nn.AdaptiveLogSoftmaxWithLoss: ((100, 20, [5, 10, 15]), {}), torch.nn.AdaptiveMaxPool1d: ((5,), {}), torch.nn.AdaptiveMaxPool2d: ((5,), {}), torch.nn.AdaptiveMaxPool3d: ((5,), {}), torch.nn.AlphaDropout: ((), {}), torch.nn.AvgPool1d: ((3,), {}), torch.nn.AvgPool2d: ((3,), {}), torch.nn.AvgPool3d: ((3,), {}), torch.nn.BCELoss: ((), {}), torch.nn.BCEWithLogitsLoss: ((), {}), torch.nn.BatchNorm1d: ((5,), {}), torch.nn.BatchNorm2d: ((5,), {}), torch.nn.BatchNorm3d: ((5,), {}), torch.nn.Bilinear: ((2, 3, 4), {}), torch.nn.CELU: ((), {}), torch.nn.CTCLoss: ((), {}), torch.nn.ChannelShuffle: ((4,), {}), torch.nn.ConstantPad1d: ((2, 3.5), {}), torch.nn.ConstantPad2d: ((2, 3.5), {}), torch.nn.ConstantPad3d: ((2, 3.5), {}), torch.nn.Conv1d: ((3, 3, 3), {}), torch.nn.Conv2d: ((3, 3, 3), {}), torch.nn.Conv3d: ((3, 3, 3), {}), torch.nn.ConvTranspose1d: ((3, 3, 3), {}), torch.nn.ConvTranspose2d: ((3, 3, 3), {}), torch.nn.ConvTranspose3d: ((3, 3, 3), {}), torch.nn.CosineEmbeddingLoss: ((), {}), torch.nn.CosineSimilarity: ((), {}), torch.nn.CrossEntropyLoss: ((), {}), torch.nn.CrossMapLRN2d: ((5,), {}), torch.nn.Dropout1d: ((), {}), torch.nn.Dropout2d: ((), {}), torch.nn.Dropout3d: ((), {}), torch.nn.Dropout: ((), {}), torch.nn.ELU: ((), {}), torch.nn.Embedding: ((10, 5), {}), torch.nn.EmbeddingBag: ((10, 5), {}), torch.nn.FeatureAlphaDropout: ((), {}), torch.nn.Flatten: ((), {}), torch.nn.Fold: ((5, 2), {}), torch.nn.FractionalMaxPool2d: ((5, 2), {}), torch.nn.FractionalMaxPool3d: ((5, 2), {}), torch.nn.GELU: ((), {}), torch.nn.GLU: ((), {}), torch.nn.GRU: ((5, 10), {}), torch.nn.GRUCell: ((5, 10), {}), torch.nn.GaussianNLLLoss: ((), {}), torch.nn.GroupNorm: ((3, 6, 1e-05, True), {}), torch.nn.Hardshrink: ((), {}), torch.nn.Hardsigmoid: ((), {}), torch.nn.Hardswish: ((), {}), torch.nn.Hardtanh: ((), {}), torch.nn.HingeEmbeddingLoss: ((), {}), torch.nn.HuberLoss: ((), {}), torch.nn.Identity: ((), {}), torch.nn.InstanceNorm1d: ((5, 1e-05, 0.1, True), {}), torch.nn.InstanceNorm2d: ((5, 1e-05, 0.1, True), {}), torch.nn.InstanceNorm3d: ((5, 1e-05, 0.1, True), {}), torch.nn.KLDivLoss: ((), {}), torch.nn.L1Loss: ((), {}), torch.nn.LPPool1d: ((2, 3), {}), torch.nn.LPPool2d: ((2, 3), {}), torch.nn.LSTM: ((5, 10), {}), torch.nn.LSTMCell: ((5, 10), {}), torch.nn.LayerNorm: ((2,), {}), torch.nn.LazyBatchNorm1d: ((), {}), torch.nn.LazyBatchNorm2d: ((), {}), torch.nn.LazyBatchNorm3d: ((), {}), torch.nn.LazyConv1d: ((5, 2), {}), torch.nn.LazyConv2d: ((5, 2), {}), torch.nn.LazyConv3d: ((5, 2), {}), torch.nn.LazyConvTranspose1d: ((5, 2), {}), torch.nn.LazyConvTranspose2d: ((5, 2), {}), torch.nn.LazyConvTranspose3d: ((5, 2), {}), torch.nn.LazyInstanceNorm1d: ((), {}), torch.nn.LazyInstanceNorm2d: ((), {}), torch.nn.LazyInstanceNorm3d: ((), {}), torch.nn.LazyLinear: ((5,), {}), torch.nn.LeakyReLU: ((), {}), torch.nn.Linear: ((10, 5), {}), torch.nn.LocalResponseNorm: ((2,), {}), torch.nn.LogSigmoid: ((), {}), torch.nn.LogSoftmax: ((), {}), torch.nn.MSELoss: ((), {}), torch.nn.MarginRankingLoss: ((), {}), torch.nn.MaxPool1d: ((3,), {}), torch.nn.MaxPool2d: ((3,), {}), torch.nn.MaxPool3d: ((3,), {}), torch.nn.MaxUnpool1d: ((5,), {}), torch.nn.MaxUnpool2d: ((5,), {}), torch.nn.MaxUnpool3d: ((5,), {}), torch.nn.Mish: ((), {}), torch.nn.ModuleDict: ((), {}), torch.nn.ModuleList: ((), {}), torch.nn.MultiLabelMarginLoss: ((), {}), torch.nn.MultiLabelSoftMarginLoss: ((), {}), torch.nn.MultiMarginLoss: ((), {}), torch.nn.MultiheadAttention: ((100, 2), {}), torch.nn.NLLLoss2d: ((), {}), torch.nn.NLLLoss: ((), {}), torch.nn.PReLU: ((), {}), torch.nn.PairwiseDistance: ((), {}), torch.nn.ParameterDict: ((), {}), torch.nn.ParameterList: ((), {}), torch.nn.PixelShuffle: ((2,), {}), torch.nn.PixelUnshuffle: ((2,), {}), torch.nn.PoissonNLLLoss: ((), {}), torch.nn.RNN: ((5, 10), {}), torch.nn.RNNBase: (('LSTM', 5, 10), {}), torch.nn.RNNCell: ((5, 10), {}), torch.nn.RNNCellBase: ((5, 10, True, 2), {}), torch.nn.RReLU: ((), {}), torch.nn.ReLU6: ((), {}), torch.nn.ReLU: ((), {}), torch.nn.ReflectionPad1d: ((2,), {}), torch.nn.ReflectionPad2d: ((2,), {}), torch.nn.ReflectionPad3d: ((2,), {}), torch.nn.ReplicationPad1d: ((2,), {}), torch.nn.ReplicationPad2d: ((2,), {}), torch.nn.ReplicationPad3d: ((2,), {}), torch.nn.SELU: ((), {}), torch.nn.Sequential: ((), {}), torch.nn.SiLU: ((), {}), torch.nn.Sigmoid: ((), {}), torch.nn.SmoothL1Loss: ((), {}), torch.nn.SoftMarginLoss: ((), {}), torch.nn.Softmax2d: ((), {}), torch.nn.Softmax: ((), {}), torch.nn.Softmin: ((), {}), torch.nn.Softplus: ((), {}), torch.nn.Softshrink: ((), {}), torch.nn.Softsign: ((), {}), torch.nn.SyncBatchNorm: ((5,), {}), torch.nn.Tanh: ((), {}), torch.nn.Tanhshrink: ((), {}), torch.nn.Threshold: ((0.1, 20), {}), torch.nn.Transformer: ((), {}), torch.nn.TransformerDecoder: ((torch.nn.TransformerDecoderLayer, 3), {}), torch.nn.TransformerDecoderLayer: ((10, 2), {}), torch.nn.TransformerEncoder: ((torch.nn.TransformerEncoderLayer, 3), {}), torch.nn.TransformerEncoderLayer: ((10, 2), {}), torch.nn.TripletMarginLoss: ((), {}), torch.nn.TripletMarginWithDistanceLoss: ((), {}), torch.nn.Unflatten: ((1, (2, 5, 5)), {}), torch.nn.Unfold: ((3,), {}), torch.nn.Upsample: ((), {}), torch.nn.UpsamplingBilinear2d: ((), {}), torch.nn.UpsamplingNearest2d: ((), {}), torch.nn.ZeroPad2d: ((0,), {}), torch.ao.nn.qat.Conv1d: ((3, 3, 3), {'qconfig': torch.ao.quantization.default_qconfig}), torch.ao.nn.qat.Conv2d: ((3, 3, 3), {'qconfig': torch.ao.quantization.default_qconfig}), torch.ao.nn.qat.Conv3d: ((3, 3, 3), {'qconfig': torch.ao.quantization.default_qconfig}), torch.ao.nn.qat.Linear: ((5, 2), {'qconfig': torch.ao.quantization.default_qconfig}), torch.ao.nn.qat.Embedding: ((10, 12), {'qconfig': torch.ao.quantization.float_qparams_weight_only_qconfig}), torch.ao.nn.qat.EmbeddingBag: ((10, 12), {'qconfig': torch.ao.quantization.float_qparams_weight_only_qconfig}), torch.nn.quantizable.LSTM: ((5, 6), {}), torch.nn.quantizable.LSTMCell: ((5, 6), {}), torch.nn.quantizable.MultiheadAttention: ((10, 2), {}), torch.ao.nn.quantized.BatchNorm2d: ((2,), {}), torch.ao.nn.quantized.BatchNorm3d: ((2,), {}), torch.ao.nn.quantized.Dropout: ((), {}), torch.ao.nn.quantized.Conv1d: ((3, 3, 3), {}), torch.ao.nn.quantized.Conv2d: ((3, 3, 3), {}), torch.ao.nn.quantized.Conv3d: ((3, 3, 3), {}), torch.ao.nn.quantized.ConvTranspose1d: ((3, 3, 3), {}), torch.ao.nn.quantized.ConvTranspose2d: ((3, 3, 3), {}), torch.ao.nn.quantized.ConvTranspose3d: ((16, 33, (3, 3, 5)), {'stride': (2, 1, 1), 'padding': (4, 2, 2), 'output_padding': (2, 2, 2), 'dilation': (1, 1, 1)}), torch.ao.nn.quantized.DeQuantize: ((), {}), torch.ao.nn.quantized.ELU: ((0.01, 0), {}), torch.ao.nn.quantized.Embedding: ((10, 3), {'factory_kwargs': {}}), torch.ao.nn.quantized.EmbeddingBag: ((10, 3), {'factory_kwargs': {}}), torch.ao.nn.quantized.GroupNorm: ((2, 4, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.ao.nn.quantized.Hardswish: ((0.1, 0), {}), torch.ao.nn.quantized.InstanceNorm1d: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.ao.nn.quantized.InstanceNorm2d: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.ao.nn.quantized.InstanceNorm3d: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.ao.nn.quantized.LayerNorm: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.ao.nn.quantized.LeakyReLU: ((0.01, 0), {}), torch.ao.nn.quantized.Linear: ((5, 2), {'factory_kwargs': {}}), torch.ao.nn.quantized.MaxPool2d: ((3,), {}), torch.ao.nn.quantized.Quantize: ((0.1, 0), {'dtype': torch.int16, 'factory_kwargs': {}}), torch.ao.nn.quantized.ReLU6: ((), {}), torch.ao.nn.quantized.Sigmoid: ((0.1, 0), {}), torch.ao.nn.quantized.Softmax: ((), {}), torch.ao.nn.quantized.FloatFunctional: ((), {}), torch.ao.nn.quantized.FXFloatFunctional: ((), {}), torch.ao.nn.quantized.QFunctional: ((), {}), torch.nn.qat.Conv1d: ((3, 3, 3), {'qconfig': torch.ao.quantization.default_qconfig}), torch.nn.qat.Conv2d: ((3, 3, 3), {'qconfig': torch.ao.quantization.default_qconfig}), torch.nn.qat.Conv3d: ((3, 3, 3), {'qconfig': torch.ao.quantization.default_qconfig}), torch.nn.qat.Linear: ((5, 2), {'qconfig': torch.ao.quantization.default_qconfig}), torch.nn.qat.Embedding: ((10, 12), {'qconfig': torch.ao.quantization.float_qparams_weight_only_qconfig}), torch.nn.qat.EmbeddingBag: ((10, 12), {'qconfig': torch.ao.quantization.float_qparams_weight_only_qconfig}), torch.nn.quantized.BatchNorm2d: ((2,), {}), torch.nn.quantized.BatchNorm3d: ((2,), {}), torch.nn.quantized.Dropout: ((), {}), torch.nn.quantized.Conv1d: ((3, 3, 3), {}), torch.nn.quantized.Conv2d: ((3, 3, 3), {}), torch.nn.quantized.Conv3d: ((3, 3, 3), {}), torch.nn.quantized.ConvTranspose1d: ((3, 3, 3), {}), torch.nn.quantized.ConvTranspose2d: ((3, 3, 3), {}), torch.nn.quantized.ConvTranspose3d: ((16, 33, (3, 3, 5)), {'stride': (2, 1, 1), 'padding': (4, 2, 2), 'output_padding': (2, 2, 2), 'dilation': (1, 1, 1)}), torch.nn.quantized.DeQuantize: ((), {}), torch.nn.quantized.ELU: ((0.01, 0), {}), torch.nn.quantized.Embedding: ((10, 3), {'factory_kwargs': {}}), torch.nn.quantized.EmbeddingBag: ((10, 3), {'factory_kwargs': {}}), torch.nn.quantized.GroupNorm: ((2, 4, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.nn.quantized.Hardswish: ((0.1, 0), {}), torch.nn.quantized.InstanceNorm1d: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.nn.quantized.InstanceNorm2d: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.nn.quantized.InstanceNorm3d: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.nn.quantized.LayerNorm: ((2, torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(2.0)), 0.1, 0), {}), torch.nn.quantized.LeakyReLU: ((0.01, 0), {}), torch.nn.quantized.Linear: ((5, 2), {'factory_kwargs': {}}), torch.nn.quantized.MaxPool2d: ((3,), {}), torch.nn.quantized.PReLU: ((0.01, 0), {}), torch.nn.quantized.Quantize: ((0.1, 0), {'dtype': torch.int16, 'factory_kwargs': {}}), torch.nn.quantized.ReLU6: ((), {}), torch.nn.quantized.Sigmoid: ((0.1, 0), {}), torch.nn.quantized.Softmax: ((), {}), torch.nn.quantized.FloatFunctional: ((), {}), torch.nn.quantized.FXFloatFunctional: ((), {}), torch.nn.quantized.QFunctional: ((), {})}

def instantiate_class(cls, args, kwargs, extra_kwargs):
    return cls(*args, **kwargs) if extra_kwargs is None else cls(*args, **kwargs, **extra_kwargs)

def mock_wrapper(method):
    mock = MagicMock()

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method(self, *args, **kwargs)
    wrapper.mock = mock
    return wrapper

def get_example_args(module_cls, constructor_arg_db, extra_kwargs=None):
    assert module_cls in constructor_arg_db, f'No entry for {module_cls} in the constructor arg DB. Please add it to pass these tests.'
    (args, kwargs) = constructor_arg_db[module_cls]
    extra_kwargs = {} if extra_kwargs is None else extra_kwargs
    args = [instantiate_class(arg, *get_example_args(arg, constructor_arg_db), extra_kwargs=extra_kwargs) if inspect.isclass(arg) else torch.nn.Parameter(arg.to(**extra_kwargs)) if isinstance(arg, torch.nn.Parameter) else arg for arg in args]
    kwargs = {k: instantiate_class(v, *get_example_args(v, constructor_arg_db), extra_kwargs=extra_kwargs) if inspect.isclass(v) else torch.nn.Parameter(v.to(*extra_kwargs)) if isinstance(v, torch.nn.Parameter) else v for (k, v) in kwargs.items()}
    kwargs.update(extra_kwargs)
    return (args, kwargs)

def generate_test_func(test_cls, module_cls, constructor_arg_db, verify_kwargs=True, module_is_lazy=False, check_nonexistent_arg=True):

    @dtypes(*floating_types())
    def run_test(test_cls, device, dtype, module_cls=module_cls):
        (args, kwargs) = get_example_args(module_cls, constructor_arg_db)
        module_needs_factory_kwargs = 'factory_kwargs' in kwargs
        if module_needs_factory_kwargs:
            del kwargs['factory_kwargs']
            extra_kwargs = {'factory_kwargs': {'device': device, 'dtype': dtype}}
        else:
            extra_kwargs = {'device': device, 'dtype': dtype}
        parameter_new = mock_wrapper(torch.nn.Parameter.__new__)
        with patch.object(torch.nn.Parameter, '__new__', parameter_new):
            register_buffer = mock_wrapper(torch.nn.Module.register_buffer)
            with patch.object(torch.nn.Module, 'register_buffer', register_buffer):
                m = module_cls(*args, **kwargs)
                module_creates_params_or_buffers = parameter_new.mock.called or register_buffer.mock.called
        if verify_kwargs and module_creates_params_or_buffers:
            (args, kwargs) = get_example_args(module_cls, constructor_arg_db, extra_kwargs=extra_kwargs)
            if module_is_lazy:
                uninit_param_new = mock_wrapper(torch.nn.UninitializedParameter.__new__)
                with patch.object(torch.nn.UninitializedParameter, '__new__', uninit_param_new):
                    uninit_buffer_new = mock_wrapper(torch.nn.UninitializedBuffer.__new__)
                    with patch.object(torch.nn.UninitializedBuffer, '__new__', uninit_buffer_new):
                        m = module_cls(*args, **kwargs)
                        uninit_param_new.mock.assert_has_calls([mock.call(device=device, dtype=dtype) for _ in uninit_param_new.mock.mock_calls])
                        uninit_buffer_new.mock.assert_has_calls([mock.call(device=device, dtype=dtype) for _ in uninit_buffer_new.mock.mock_calls])
            else:
                m = module_cls(*args, **kwargs)
                for (name, param) in m.named_parameters():
                    test_cls.assertEqual(str(param.device), device, f'Parameter {name} is on {param.device.type} instead of the expected device {device}')
                    if param.dtype.is_floating_point and (not module_needs_factory_kwargs):
                        test_cls.assertEqual(param.dtype, dtype, f'Parameter {name} is of dtype {param.dtype} instead of the expected dtype {dtype}')
                for (name, buffer) in m.named_buffers():
                    test_cls.assertEqual(str(buffer.device), device, f'Buffer {name} is on {buffer.device.type} instead of the expected device {device}')
                    if buffer.dtype.is_floating_point and (not module_needs_factory_kwargs):
                        test_cls.assertEqual(buffer.dtype, dtype, f'Buffer {name} is of dtype {buffer.dtype} instead of the expected dtype {dtype}')
        if check_nonexistent_arg:
            with test_cls.assertRaises(TypeError):
                m = module_cls(*args, **kwargs, nonexistent_arg='foo')
    return run_test

def generate_tests(test_cls, constructor_arg_db):
    NAMESPACES = [torch.nn, torch.ao.nn.qat, torch.ao.nn.quantized, torch.nn.qat, torch.nn.quantizable, torch.nn.quantized]
    MODULES_TO_SKIP = {torch.nn.Module, torch.nn.Container, torch.nn.NLLLoss2d, torch.ao.nn.quantized.Embedding, torch.ao.nn.quantized.EmbeddingBag, torch.nn.quantized.Embedding, torch.nn.quantized.EmbeddingBag, torch.nn.quantized.LSTM, torch.nn.quantized.MultiheadAttention}
    MODULES_WITHOUT_KWARGS_SUPPORT = {torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss, torch.nn.FractionalMaxPool2d, torch.nn.FractionalMaxPool3d, torch.nn.MultiLabelSoftMarginLoss, torch.nn.MultiMarginLoss, torch.nn.NLLLoss, torch.nn.TransformerDecoder, torch.nn.TransformerEncoder}
    MODULES_WITH_PREVIOUS_KWARGS = {torch.nn.Identity}
    LAZY_MODULES = {torch.nn.LazyBatchNorm1d, torch.nn.LazyBatchNorm2d, torch.nn.LazyBatchNorm3d, torch.nn.LazyConv1d, torch.nn.LazyConv2d, torch.nn.LazyConv3d, torch.nn.LazyConvTranspose1d, torch.nn.LazyConvTranspose2d, torch.nn.LazyConvTranspose3d, torch.nn.LazyConvTranspose3d, torch.nn.LazyInstanceNorm1d, torch.nn.LazyInstanceNorm2d, torch.nn.LazyInstanceNorm3d, torch.nn.LazyLinear}
    MODULES_THAT_REQUIRE_FBGEMM = {torch.ao.nn.quantized.Conv1d, torch.ao.nn.quantized.Conv2d, torch.ao.nn.quantized.Conv3d, torch.ao.nn.quantized.ConvTranspose1d, torch.ao.nn.quantized.ConvTranspose2d, torch.ao.nn.quantized.ConvTranspose3d, torch.ao.nn.quantized.Linear, torch.nn.quantized.Conv1d, torch.nn.quantized.Conv2d, torch.nn.quantized.Conv3d, torch.nn.quantized.ConvTranspose1d, torch.nn.quantized.ConvTranspose2d, torch.nn.quantized.ConvTranspose3d, torch.nn.quantized.Linear}
    for namespace in NAMESPACES:
        namespace_basename = namespace.__name__.split('.')[-1]
        for module_name in namespace.modules.__all__:
            module_cls = getattr(namespace.modules, module_name)
            if module_cls in MODULES_TO_SKIP:
                continue
            verify_kwargs = module_cls not in MODULES_WITHOUT_KWARGS_SUPPORT
            module_is_lazy = module_cls in LAZY_MODULES
            check_nonexistent_arg = module_cls not in MODULES_WITH_PREVIOUS_KWARGS
            run_test = generate_test_func(test_cls, module_cls, constructor_arg_db, verify_kwargs=verify_kwargs, module_is_lazy=module_is_lazy, check_nonexistent_arg=check_nonexistent_arg)
            test_name = f'test_{namespace_basename}_{module_name}'
            if module_cls in MODULES_THAT_REQUIRE_FBGEMM:
                run_test = skipIfNoFBGEMM(run_test)
            setattr(TestModuleInit, test_name, run_test)

class TestModuleInit(TestCase):
    _ignore_not_implemented_error = False
generate_tests(TestModuleInit, build_constructor_arg_db())
instantiate_device_type_tests(TestModuleInit, globals())
if __name__ == '__main__':
    run_tests()