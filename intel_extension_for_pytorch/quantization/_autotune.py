import sys
import subprocess
import copy
import time
from . import default_static_qconfig, prepare


def autotune(prepared_model, calib_dataloader, eval_func, sampling_sizes=[100], accuracy_criterion={'relative': 0.01}, tuning_time=0):
    r"""
    Automatic accuracy-driven tuning helps users quickly find out the advanced recipe for INT8 inference.

    Args:
        prepared_model (torch.nn.Module): the FP32 prepared model returned from ipex.quantization.prepare.
        calib_dataloader (generator): set a dataloader for calibration.
        eval_func (function): set a evaluation function. This function takes "model" as input parameter
            executes entire evaluation process with self contained metrics,
            and returns an accuracy value which is a scalar number. The higher the better.
        sampling_sizes (list): a list of sample sizes used in calibration, where the tuning algorithm would explore from.
            The default value is ``[100]``.
        accuracy_criterion ({accuracy_criterion_type(str, 'relative' or 'absolute') : accuracy_criterion_value(float)}):
            set the maximum allowed accuracy loss, either relative or absolute. The default value is ``{'relative': 0.01}``.
        tuning_time (seconds): tuning timeout. The default value is ``0`` which means early stop.

    Returns:
        FP32 tuned model (torch.nn.Module)
    """
    try:
        import neural_compressor
    except ImportError:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'neural_compressor==1.14.1'])
            import neural_compressor
        except:
            assert False, "Unable to import neural_compressor from the local environment."
    from neural_compressor import config
    from neural_compressor.experimental import Quantization, common
    from neural_compressor.adaptor.torch_utils.util import auto_copy
    from neural_compressor.adaptor.pytorch import get_example_inputs

    config.quantization.backend = 'pytorch_ipex'
    config.quantization.approach = 'post_training_static_quant'
    config.quantization.device = 'cpu'
    config.quantization.use_bf16 = False
    config.quantization.calibration_sampling_size = sampling_sizes
    if accuracy_criterion.get('relative'):
        config.quantization.accuracy_criterion.relative = accuracy_criterion.get('relative')
    if accuracy_criterion.get('absolute'):
        config.quantization.accuracy_criterion.absolute = accuracy_criterion.get('absolute')
    config.quantization.timeout = tuning_time
    quantizer = Quantization(config)
    fp32_model = auto_copy(prepared_model)
    quantizer.model = common.Model(fp32_model)
    quantizer.calib_dataloader = calib_dataloader
    quantizer.eval_func = eval_func
    q_model = quantizer.fit()
    dirname_str = './saved_tuning_results_'+time.strftime("%Y%m%d_%H%M%S")
    q_model.save(dirname_str)

    # This is a workaround for the bug that the auto_copy function will change its input prepared_model.
    qconfig = default_static_qconfig
    new_prepared_model = prepare(fp32_model, qconfig, example_inputs=get_example_inputs(calib_dataloader), inplace=True)
    new_prepared_model.load_qconf_summary(qconf_summary=dirname_str+'/best_configure.json')

    return new_prepared_model
