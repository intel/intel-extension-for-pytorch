# This Python file uses the following encoding: utf-8

import sys
import subprocess
import time


def autotune(
    prepared_model,
    calib_dataloader,
    eval_func,
    sampling_sizes=None,
    accuracy_criterion=None,
    tuning_time=0,
):
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
    if sampling_sizes is None:
        sampling_sizes = [100]
    if accuracy_criterion is None:
        accuracy_criterion = {"relative": 0.01}
    neural_compressor_version = "2.1"
    try:
        import neural_compressor

        if neural_compressor.__version__ != neural_compressor_version:
            raise RuntimeError(
                "Please install Intel® Neural Compressor with version {} while the current version of \
                    Intel® Neural Compressor is {}.".format(
                    neural_compressor_version, neural_compressor.__version__
                )
            )
    except ImportError:
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "neural_compressor=={}".format(neural_compressor_version),
                ]
            )
            import neural_compressor
        except BaseException:
            AssertionError(
                False
            ), "Unable to import neural_compressor from the local environment."
    from neural_compressor import PostTrainingQuantConfig
    from neural_compressor.config import TuningCriterion, AccuracyCriterion
    from neural_compressor import quantization

    conf = PostTrainingQuantConfig(
        backend="ipex",
        calibration_sampling_size=sampling_sizes,
        tuning_criterion=TuningCriterion(timeout=tuning_time),
        accuracy_criterion=AccuracyCriterion(
            criterion=list(accuracy_criterion.keys())[0],
            tolerable_loss=list(accuracy_criterion.values())[0],
        ),
        excluded_precisions=["bf16"],
    )
    q_model = quantization.fit(
        prepared_model, conf, calib_dataloader=calib_dataloader, eval_func=eval_func
    )
    dirname_str = "./saved_tuning_results_" + time.strftime("%Y%m%d_%H%M%S")
    q_model.save(dirname_str)

    prepared_model.load_qconf_summary(
        qconf_summary=dirname_str + "/best_configure.json"
    )
    return prepared_model
