# This Python file uses the following encoding: utf-8

import os
import sys
import copy
import json
from ..utils._logger import logger
import subprocess
import torch
import time
import intel_extension_for_pytorch as ipex


def autotune(
    model,
    calib_dataloader,
    calib_func=None,
    eval_func=None,
    op_type_dict=None,
    smoothquant_args=None,
    sampling_sizes=None,
    accuracy_criterion=None,
    tuning_time=0,
):
    r"""
    Automatic accuracy-driven tuning helps users quickly find out the advanced recipe for INT8 inference.

    Args:
        model (torch.nn.Module): fp32 model.
        calib_dataloader (generator): set a dataloader for calibration.
        calib_func (function): calibration function for post-training static quantization. It is optional.
            This function takes "model" as input parameter and executes entire inference process.
        eval_func (function): set a evaluation function. This function takes "model" as input parameter
            executes entire evaluation process with self contained metrics, and returns an accuracy value
            which is a scalar number. The higher the better.
        op_type_dict (dict): Tuning constraints on optype-wise for advance user to reduce tuning space.
            User can specify the quantization config by op type:
        smoothquant_args (dict): smoothquant recipes for automatic global alpha tuning, and automatic
            layer-by-layer alpha tuning for the best INT8 accuracy.
        sampling_sizes (list): a list of sample sizes used in calibration, where the tuning algorithm would explore from.
            The default value is ``[100]``.
        accuracy_criterion ({accuracy_criterion_type(str, 'relative' or 'absolute') : accuracy_criterion_value(float)}):
            set the maximum allowed accuracy loss, either relative or absolute. The default value is ``{'relative': 0.01}``.
        tuning_time (seconds): tuning timeout. The default value is ``0`` which means early stop.

    Returns:
        prepared_model (torch.nn.Module): the prepared model loaded qconfig after tuning.
    """
    if sampling_sizes is None:
        sampling_sizes = [100]
    if accuracy_criterion is None:
        accuracy_criterion = {"relative": 0.01}
    if op_type_dict is None:
        op_type_dict = {}
    if smoothquant_args is None:
        smoothquant_args = {}

    neural_compressor_version = "2.4.1"
    try:
        import neural_compressor

        if neural_compressor.__version__ < neural_compressor_version:
            raise RuntimeError(
                "Please install Intel® Neural Compressor with version >= {} \
                while the current version of Intel® Neural Compressor is {}.".format(
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
    from neural_compressor import PostTrainingQuantConfig, quantization
    from neural_compressor.config import TuningCriterion, AccuracyCriterion
    from neural_compressor.adaptor.pytorch import get_example_inputs

    recipes = {}
    op_type_dict = {}
    user_model = copy.deepcopy(model)

    try:  # dataloader with label
        for i, (model_inputs, last_ind) in enumerate(calib_dataloader):
            example_inputs = model_inputs
            break
    except Exception:
        try:  # dataloader without label
            for i, model_inputs in enumerate(calib_dataloader):
                example_inputs = model_inputs
                break
        except Exception:  # static quant
            try:
                example_inputs = get_example_inputs(model, calib_dataloader)
            except Exception:
                logger.critical(
                    "Wrong dataloader format. Please refer to autotune doc. Aborting..."
                )
                exit()

    if not smoothquant_args:  # static quantization
        qconfig = ipex.quantization.default_static_qconfig_mapping
    else:  # smoothquant
        recipes["smooth_quant"] = True
        recipes["smooth_quant_args"] = smoothquant_args
        if "auto_alpha_args" in smoothquant_args:
            recipes["default_alpha"] = smoothquant_args["auto_alpha_args"]["init_alpha"]
            del recipes["smooth_quant_args"]["auto_alpha_args"]["init_alpha"]
            recipes["smooth_quant_args"]["auto_alpha_args"]["do_blockwise"] = recipes[
                "smooth_quant_args"
            ]["auto_alpha_args"].pop("enable_blockwise_loss")
        print("smooth_quant_recipes:", recipes)
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping()

    conf = PostTrainingQuantConfig(
        backend="ipex",
        calibration_sampling_size=sampling_sizes,
        op_type_dict=op_type_dict,
        tuning_criterion=TuningCriterion(timeout=tuning_time),
        accuracy_criterion=AccuracyCriterion(
            criterion=list(accuracy_criterion.keys())[0],
            tolerable_loss=list(accuracy_criterion.values())[0],
        ),
        excluded_precisions=["bf16"],
        recipes=recipes,
    )
    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        eval_func=eval_func,
        calib_func=calib_func,
    )
    dirname_str = "./saved_tuning_results_" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(dirname_str, exist_ok=True)
    with open(os.path.join(dirname_str, "best_configure.json"), "w") as f:
        json.dump(q_model.tune_cfg, f, indent=4)

    if isinstance(example_inputs, dict):
        prepared_model = ipex.quantization.prepare(
            model, qconfig, example_kwarg_inputs=example_inputs, inplace=True
        )
    else:
        prepared_model = ipex.quantization.prepare(
            model, qconfig, example_inputs=example_inputs, inplace=True
        )
    with torch.no_grad():  # warm-up inference
        if isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
            prepared_model(*example_inputs)
        elif isinstance(example_inputs, dict):
            prepared_model(**example_inputs)
        else:
            prepared_model(example_inputs)

    prepared_model.load_qconf_summary(
        qconf_summary=dirname_str + "/best_configure.json"
    )

    try:
        for root, dirs, files in os.walk(dirname_str, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(dirname_str)
    except Exception as e:
        print(f"Failed to delete {dirname_str}. Reason: {e}")

    return prepared_model
