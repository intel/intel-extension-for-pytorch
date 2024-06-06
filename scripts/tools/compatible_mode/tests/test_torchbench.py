# runn all models usage:
# under the torchbench folder and python -u test_torchbench.py
# run single model usage:
# bs is 0 means use the default value
# python -u run.py -d xpu -t train --bs 0 --metrics None Model_name
# bs is 4 means set the bs to 4
# python -u run.py -d xpu -t train --bs 4 --metrics None Model_name

import os
import subprocess
from typing import List, Dict, Any
import sys
import time

# config
run_bs = 4 # default 4, it is used the batch size which model can be set
executor = sys.executable
device = 'xpu' # default xpu
unsupported_str_list = ["train", "notimplement", "support"]
mode = os.getenv('TEST_TORCHBENCH_MODE')
# default is testing train/inf
if mode is None:
    mode = 'all'
assert type(mode) is str and mode.lower() in ['train', 'inf', 'all'], "please specify the TEST_TORCHBENCH_MODE to be train/inf/all"
bench_file = 'run.py'
config_arg = '-u' # python -u xxxx
multi_card = False # default single card running

# log print
MODEL_BLANK_SPACE = 50
CMD_BLANK_SPACE = 150

# all models scope in torchbench
# TODO: maybe need some methods to get all models enabled in torchbench
all_models_list = [
    "BERT_pytorch",
    "Background_Matting",
    "DALLE2_pytorch",
    "LearningToPaint",
    "Super_SloMo",
    "alexnet",
    "basic_gnn_edgecnn",
    "basic_gnn_gcn",
    "basic_gnn_gin",
    "basic_gnn_sage",
    "cm3leon_generate",
    "dcgan",
    "demucs",
    "densenet121",
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_fcos_r_50_fpn",
    "detectron2_maskrcnn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    "dlrm",
    "doctr_det_predictor",
    "doctr_reco_predictor",
    "drq",
    "fastNLP_Bert",
    "functorch_dp_cifar10",
    "functorch_maml_omniglot",
    "hf_Albert",
    "hf_Bart",
    "hf_Bert",
    "hf_Bert_large",
    "hf_BigBird",
    "hf_DistilBert",
    "hf_GPT2",
    "hf_GPT2_large",
    "hf_Longformer",
    "hf_Reformer",
    "hf_T5",
    "hf_T5_base",
    "hf_T5_generate",
    "hf_T5_large",
    "hf_Whisper",
    "hf_clip",
    "hf_distil_whisper",
    "lennard_jones",
    "llama",
    "llama_v2_7b_16h",
    "maml",
    "maml_omniglot",
    "mnasnet1_0",
    "mobilenet_v2",
    "mobilenet_v2_quantized_qat",
    "mobilenet_v3_large",
    "moco",
    "nanogpt",
    "nvidia_deeprecommender",
    "opacus_cifar10",
    "phlippe_densenet",
    "phlippe_resnet",
    "pyhpc_equation_of_state",
    "pyhpc_isoneutral_mixing",
    "pyhpc_turbulent_kinetic_energy",
    "pytorch_CycleGAN_and_pix2pix",
    "pytorch_stargan",
    "pytorch_unet",
    "resnet152",
    "resnet18",
    "resnet50",
    "resnet50_quantized_qat",
    "resnext50_32x4d",
    "sam",
    "shufflenet_v2_x1_0",
    "simple_gpt",
    "simple_gpt_tp_manual",
    "soft_actor_critic",
    "speech_transformer",
    "squeezenet1_1",
    "stable_diffusion_text_encoder",
    "stable_diffusion_unet",
    "tacotron2",
    "timm_efficientdet",
    "timm_efficientnet",
    "timm_nfnet",
    "timm_regnet",
    "timm_resnest",
    "timm_vision_transformer",
    "timm_vision_transformer_large",
    "timm_vovnet",
    "torch_multimodal_clip",
    "tts_angular",
    "vgg16",
    "vision_maskrcnn",
    "yolov3"
]

# This list contains the model which is unsupported to set the maunal bs
# so set the bs 0 in command
models_list_unsupport_manual_bs = [
    "basic_gnn_edgecnn",
    "basic_gnn_gcn",
    "basic_gnn_gin",
    "basic_gnn_sage",
    "functorch_maml_omniglot",
    "maml",
    "maml_omniglot",
    "pytorch_CycleGAN_and_pix2pix",
    "pytorch_stargan",
    "soft_actor_critic",
    "speech_transformer",
    "stable_diffusion_text_encoder",
    "stable_diffusion_unet",
    "torch_multimodal_clip",
    "vision_maskrcnn",
    "pyhpc_turbulent_kinetic_energy",
    "drq",
]

# This list contains the model which is unsupported to train
# the train function is not implemented by torchbench
models_list_unsupport_eager_mode = [
    "simple_gpt",
    "simple_gpt_tp_manual",
]

# This list contains the model which needs to set/unset the specific env flag
# "model": {"set": {"env_name":"env_value"}, "unset":["env_name"]}
models_list_specific_env_flag = {
    "moco":            {"set": {"CCL_ZE_IPC_EXCHANGE": "sockets"}, "unset": ["ZE_AFFINITY_MASK"]},
    "vision_maskrcnn": {"set": {"CCL_ZE_IPC_EXCHANGE": "sockets"}, "unset": ["ZE_AFFINITY_MASK"]},
}

# contain the required bs for specific models
specific_models_with_bs: Dict[str, int] = {
    "torch_multimodal_clip": 1,
}

# example: {"BERT_pytorch": {"bs": 4, "train": True or None, "inf": True or None}}
models_list: Dict[str, Dict[str, Any]] = {}

pass_model_cnt = 0
fail_model_cnt = 0
not_supported_model_cnt = 0
# example: {"BERT_pytorch": {"result_key": {"pass"  : True}}}
#          {"BERT_pytorch": {"result_key": {"failed": "error message"}}}
#          {"BERT_pytorch": {"result_key": {"Notsupported": "not supported message"}}}
#          {"BERT_pytorch": {"duration": int}}
results_summary: Dict[str, Dict[str, Any]] = {}

def is_eligible_for_test(mode, model):
    if model in models_list_unsupport_eager_mode:
        return False
    return True

def set_the_running_mode(model):
    if mode == 'train':
        models_list[model]['train'] = True
        models_list[model]['inf'] = False
    elif mode == 'inf':
        models_list[model]['train'] = False
        models_list[model]['inf'] = True
    elif mode == 'all':
        models_list[model]['train'] = True
        models_list[model]['inf'] = True
    else:
        raise RuntimeError("[error] no mode is specified, please check env flag TEST_TORCHBENCH_MODE")

def set_the_batch_size(model):
    if model in models_list_unsupport_manual_bs:
        models_list[model]['bs'] = 0
    elif model in specific_models_with_bs:
        models_list[model]['bs'] = specific_models_with_bs[model]
    else:
        models_list[model]['bs'] = run_bs

for model in all_models_list:
    # avoid the recursive deal with model
    if model in models_list:
        continue
    # skip the model which unsupport train/inf in torchbench
    if not is_eligible_for_test(mode, model):
        continue

    models_list[model] = {}

    set_the_running_mode(model)
    set_the_batch_size(model)

# models are eligible to run
num_models = len(models_list.keys())
assert num_models >= 1, "[error] no model is collected"
print('[info] using python executor: ', executor)
print('[info] mode: ', mode)
print('[info] device: ', device)
print('[info] bs: ', run_bs)
print('[info] multi card: ', multi_card)
print('[info] running models list:')
for idx, elem in enumerate(models_list.items()):
    print('[', idx, '] ', elem)

def generate_required_env(model):
    # set the running env
    env = os.environ.copy()

    # remove the env IPEX_SHOW_OPTION for better logging
    env['IPEX_SHOW_OPTION'] = f"0"

    # single card running default
    if not multi_card:
        env['ZE_AFFINITY_MASK'] = f"0"

    if model in models_list_specific_env_flag:
        required_env = models_list_specific_env_flag[model]
        assert "set" in required_env and "unset" in required_env, "check the models_list_specific_env_flag, it is wrongly set"

        # set the required env flag
        for env_name, env_value in required_env["set"].items():
            env[env_name] = env_value

        # remove the env flag
        for env_name in required_env["unset"]:
            del env[env_name]
    return env

def generate_command(model, test_mode='train'):
    cmd: List[str] = []
    # generate the cmd
    # python -u run.py -d cuda -t train --bs 4 --model "Background_Matting"
    cmd.append(str(executor))
    cmd.append(config_arg)
    cmd.append(bench_file)
    cmd.append('-d')
    cmd.append(device)
    cmd.append('-t')
    if test_mode == 'train' and models_list[model]['train']:
        cmd.append('train')
    if test_mode == 'inf' and models_list[model]['inf']:
        cmd.append('eval')
    cmd.append('--bs')
    cmd.append(str(models_list[model]['bs']))
    # disable metrics because it will init pynvml
    cmd.append('--metrics')
    cmd.append('None')
    # specify the model
    cmd.append(model)
    str_cmd = ''
    for elem in cmd:
        str_cmd += elem + ' '
    return cmd, str_cmd.rstrip()

def testing(model, test_mode, count):
    global pass_model_cnt
    global fail_model_cnt
    global not_supported_model_cnt
    result_key = test_mode + '_result'
    if model not in results_summary:
        results_summary[model] = {}
    if result_key not in results_summary[model]:
        results_summary[model][result_key] = {}

    cmd, str_cmd = generate_command(model, test_mode)
    env = generate_required_env(model)
    stderr_msg = None
    returncode = 0
    is_not_supported = False
    time_start = time.time()
    try:
        # start run the model
        subprocess.run(cmd, encoding="utf-8", capture_output=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        returncode = e.returncode
        if returncode != 0:
            stderr_msg = e.stderr
            # check if it is not implemented
            is_not_supported = all([key.lower() in stderr_msg.lower() for key in unsupported_str_list])

    time_end = time.time()
    # set the duration
    duration = round(time_end - time_start, 2)
    results_summary[model]['duration'] = duration

    # for log alignment
    model_space = MODEL_BLANK_SPACE
    model_space = ' ' * (model_space - len(model))
    cmd_space = CMD_BLANK_SPACE
    cmd_space = ' ' * (cmd_space - len(str_cmd))
    if test_mode == 'inf':
        test_mode += ' ' * 2

    # pass
    if returncode == 0:
        print('[', count, '][', test_mode, '][success] pass model: ', model, model_space, 'cmd: ', str_cmd, cmd_space, 'time(s): ', duration)
        pass_model_cnt += 1
        results_summary[model][result_key]['pass'] = True
    else:
        print('[', count, '][', test_mode, '][error]   fail model: ', model, model_space, 'cmd: ', str_cmd, cmd_space, 'time(s): ', duration)
        # not supported train or inf
        if is_not_supported:
            not_supported_model_cnt += 1
            results_summary[model][result_key]['Notsupported'] = stderr_msg
        else:
            # real failed
            fail_model_cnt += 1
            results_summary[model][result_key]['failed'] = stderr_msg

print('[info] running models number: ', num_models)
print('[info] begin test all models......')
global_time_start = time.time()
count = 0
for model in models_list.keys():
    if models_list[model]['train']:
        testing(model, 'train', count)
        count += 1
    if models_list[model]['inf']:
        testing(model, 'inf', count)
        count += 1

global_time_end = time.time()
global_duration = global_time_end - global_time_start

# summary
print('\n' * 10 + '*' * 50)
print('[info] Detailed Summary:')
for idx, model in enumerate(results_summary.keys()):
    assert model in results_summary, "Fatal error, the model is not in the testing records"

    model_space = MODEL_BLANK_SPACE
    model_space = ' ' * (model_space - len(model))
    for test_mode in ['train', 'inf']:
        result_key = test_mode + '_result'
        if result_key in results_summary[model]:
            if 'pass' in results_summary[model][result_key]:
                print('[', idx, '] model ', model, model_space, test_mode, ' pass')
            elif 'failed' in results_summary[model][result_key]:
                print('[', idx, '] model ', model, model_space, test_mode, ' failed:\n', results_summary[model][result_key]['failed'])
            elif 'Notsupported' in results_summary[model][result_key]:
                print('[', idx, '] model ', model, model_space, test_mode, ' not supported:\n', results_summary[model][result_key]['Notsupported'])
            else:
                raise RuntimeError("model {} is not recorded into the results, check if it is running".format(model))

print('\n' * 10 + '*' * 50)
print('[info] Simplified Summary:')
for idx, model in enumerate(results_summary.keys()):
    for test_mode in ['train', 'inf']:
        result_key = test_mode + '_result'
        if result_key in results_summary[model]:
            if 'pass' in results_summary[model][result_key]:
                print('[', idx, '] model ', model, test_mode, ' pass')
            elif 'failed' in results_summary[model][result_key]:
                print('[', idx, '] model ', model, test_mode, ' failed')
            elif 'Notsupported' in results_summary[model][result_key]:
                print('[', idx, '] model ', model, test_mode, ' not supported')
            else:
                raise RuntimeError("model {} is not recorded with the results, check if it run or not".format(model))

# [watch] calculate the pass rate not includes the not supported model
if pass_model_cnt == 0 and fail_model_cnt == 0:
    print('[Error] No pass models or failed models are found')
    sys.exit()
pass_rate = pass_model_cnt / (pass_model_cnt + fail_model_cnt) * 100.0
print('[info] eligible testing model number = ', num_models)
print('[info] pass total number = ', pass_model_cnt)
print('[info] fail total number = ', fail_model_cnt)
print('[info] not supported total number = ', not_supported_model_cnt)

print('[info] pass rate = ', round(pass_rate, 3), ' %')
global_duration = global_duration / 60.0 # unit: min
print('[info] testing total duration = ', round(global_duration, 2), ' min')
