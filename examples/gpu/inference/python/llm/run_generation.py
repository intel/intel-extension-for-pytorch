import torch
import time
import json
import pathlib
import argparse
import os
import sys
from torch.nn.functional import pad
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)


# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

# args
parser = argparse.ArgumentParser("Generation script (fp32/bf16 path)", add_help=False)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="EleutherAI/gpt-j-6B",
    help="the huggingface mdoel id",
)
parser.add_argument('--sub-model-name',
    type=str,
    help="the sub model name for accuracy check"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "xpu"],
    default="xpu",
    help="xpu or cpu",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16"],
    default="bfloat16",
    help="float16, bfloat16, float32",
)
parser.add_argument(
    "--input-tokens",
    default="32",
    type=str,
    help="input tokens length if needed from prompt.json",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--lambada", action="store_true")
parser.add_argument("--dataset", default="lambada", type=str)
parser.add_argument("--num-beams", default=4, type=int, help="beam width")
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--num-warmup", default=3, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--token-latency", action="store_true", help="get token latency breakdown"
)
parser.add_argument("--print-memory", action="store_true")
parser.add_argument("--disable_optimize_transformers", action="store_true")
# WOQ related args.
parser.add_argument("--woq", action="store_true")
parser.add_argument("--calib_dataset", default="wikitext2", type=str)
parser.add_argument("--calib_group_size", default=-1, type=int)
parser.add_argument("--calib_output_dir", default="./", type=str)
parser.add_argument("--calib_checkpoint_name", default="quantized_weight.pt", type=str)
parser.add_argument("--calib_nsamples", default=128, type=int)
parser.add_argument("--calib_wbits", default=4, type=int)
parser.add_argument("--calib_seed", default=0, type=int)
parser.add_argument("--woq_checkpoint_path", default="", type=str)
parser.add_argument("--accuracy-only", action="store_true")
parser.add_argument(
    "--acc-tasks",
    nargs="+",
    default=[
        "lambada_standard",
    ],
    type=str,
    help="tasks list for accuracy validation, only enabled lambada_standard and lambada_standard at present",
)
args = parser.parse_args()
print(args)

do_profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
do_profiling = args.profile or do_profiling

# device
device = torch.device(args.device)

def get_memory_usage(name, args):
    if args.print_memory:
        if args.device == "xpu":
            memory_allocated = round(torch.xpu.memory_reserved() / 1024**3, 3)
        print(name, "memory used total:", memory_allocated, "GB")


# import ipex
if args.ipex:
    import intel_extension_for_pytorch as ipex
    '''
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass
    '''
#if args.jit:
#    torch._C._jit_set_texpr_fuser_enabled(False)

# dtype
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# load model
model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
model_class = MODEL_CLASSES[model_type]
config = AutoConfig.from_pretrained(args.model_id, torchscript=args.jit, trust_remote_code=True)
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)
if amp_dtype == torch.float16 and args.woq and args.woq_checkpoint_path == "":
    load_dtype = torch.float32
else:
    load_dtype = amp_dtype
model = model_class[0].from_pretrained(
    args.model_id, torch_dtype=load_dtype, config=config, low_cpu_mem_usage=True, trust_remote_code=True
)
tokenizer = model_class[1].from_pretrained(args.model_id)

woq_weight_path = args.woq_checkpoint_path
if args.woq and args.woq_checkpoint_path == "":
    woq_weight_path = args.calib_output_dir + args.calib_checkpoint_name
    import intel_extension_for_pytorch as ipex
    from ..dataset.calib_utils import get_loaders
    model.eval()
    dataloader, testloader = get_loaders(
        args.calib_dataset,
        tokenizer,
        nsamples=args.calib_nsamples,
        seed=args.calib_seed,
        seqlen=model.seqlen
    )
    ipex.woq(
        model,
        dataloader,
        woq_weight_path,
        wbits=args.calib_wbits,
        param_dtype=amp_dtype,
    )

model = model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# to ipex
if args.ipex:
    if args.disable_optimize_transformers:
        model = ipex.optimize(model.eval(), dtype=amp_dtype)
    else:
        if "low_precision_checkpoint" in ipex.optimize_transformers.__code__.co_varnames:
            woq_config_and_weight = None
            if args.woq and args.device == "xpu":
                woq_config = {}
                woq_config['groups'] = args.calib_group_size
                woq_config_and_weight = (torch.load(woq_weight_path), woq_config)
            model = ipex.optimize_transformers(model.eval(), dtype=amp_dtype, device=args.device, inplace=True, low_precision_checkpoint=woq_config_and_weight)
        else:
            woq_config = {}
            if args.woq:
                woq_config['is_int4'] = args.calib_wbits == 4
                woq_config['group_size'] = args.calib_group_size
                woq_config['weight_path'] = woq_weight_path
            model = ipex.optimize_transformers(model.eval(), dtype=amp_dtype, **woq_config)
    get_memory_usage("Ipex", args)


num_beams = 1 if args.greedy else args.num_beams
# generate args
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)


######################## run lm eval accuracy check ########################
def run_accuracy():
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.base import BaseLM
    from typing import Union, List, Optional
    from transformers import BatchEncoding

    TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    class HuggingFaceModel(BaseLM):
        _DEFAULT_MAX_LENGTH = 2048

        def __init__(
            self,
            config=None,
            model=None,
            tokenizer=None,
            device="xpu",
            num_beams=1,
            batch_size=1,
            max_length=None,
            dtype: Optional[Union[str, torch.dtype]] = "auto",
            tp_number=1,
        ):
            super().__init__()

            self._device = device
            self._batch_size = batch_size
            self._max_length = max_length
            self._num_beams = num_beams
            self._dtype = dtype
            self._tp_number = tp_number

            self.config = config
            self.model = model
            self.base_model = model
            self.tokenizer = tokenizer

        def _model_call(
            self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
        ) -> TokenSequence:
            with torch.inference_mode():
                output = self.base_model(inputs)

            if isinstance(output, tuple):
                return output[0]

            return output["logits"]

        @property
        def eot_token_id(self):
            # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            if self._max_length:  # if max length manually set, return it
                return self._max_length
            seqlen_config_attrs = (
                "n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self.config, attr):
                    return getattr(self.config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length

            return self._DEFAULT_MAX_LENGTH

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            # TODO: fix multi-gpu
            return self._batch_size  # * gpus

        @property
        def device(self):
            # TODO: fix multi-gpu
            return self._device

        def tok_encode(self, string: str):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def _model_generate(self, context, max_length, eos_token_id):
            generation_kwargs = {"do_sample": False, "max_length": max_length, "num_beams": self._num_beams}
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
                generation_kwargs[
                    "pad_token_id"
                ] = eos_token_id  # setting eos_token_id as pad token
            return self.model.generate(context, **generation_kwargs)

    task_dict = lm_eval.tasks.get_task_dict(args.acc_tasks)
    hfmodel = HuggingFaceModel(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device="xpu",
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )

    results = evaluator.evaluate(
        hfmodel,
        task_dict,
    )

    print(evaluator.make_table(results))


if args.accuracy_only:
    run_accuracy()
    sys.exit(0)

######################## run generation benchmark ########################
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
        
def run_generate(num_tokens, num_input_tokens, num_beams):
    print(f"*** Starting to generate {num_tokens} tokens for {num_input_tokens} tokens with num_beams={num_beams}")
    if args.prompt is not None:
        prompt = args.prompt
    elif model_type == "auto":
        raise SystemExit(
            "[ERROR] model prompt is not supported, please use --prompt for this model: "
            + args.model_id
        )
    elif int(args.input_tokens) > 8192:
        prompt = prompt_pool[model_type]["8192"] * int(int(args.input_tokens) / 8192)
    elif args.input_tokens in prompt_pool[model_type]:
        prompt = prompt_pool[model_type][args.input_tokens]
    else:
        raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)
    
    if args.token_latency:
        generate_kwargs["token_latency"] = True

    # Accuracy check, take the ref_prompt as reference for check
    f1 = open(os.path.join(os.path.dirname(__file__), "ref_prompt.json"))
    prompt_json = json.load(f1)
    f1.close()
    ref_prompt=None
    ref_prompt_cuda=None
    token_support = [(32, 32), (1024, 128)]
    if (int(num_input_tokens), num_tokens) in token_support:
        ref_prompt = prompt_json[args.sub_model_name][f"{num_input_tokens}-{num_tokens}"][f"{num_beams}"]
        try:
            ref_prompt_cuda = prompt_json[args.sub_model_name][f"{num_input_tokens}-{num_tokens}"][f"cuda-result: {num_beams}"]
        except Exception:
            pass
    acc_pass = 0
  
    # start
    total_time = 0.0
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    total_list = []
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        for i in range(num_iter):
            tic = time.time()
            with torch.autograd.profiler_legacy.profile(enabled=do_profiling, use_xpu=True, record_shapes=True) as prof:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                output = model.generate(
                    input_ids, max_new_tokens=int(args.max_new_tokens), **generate_kwargs
                )
                gen_ids = output[0] if args.token_latency else output
                gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                if args.device == "xpu":
                    torch.xpu.synchronize()
            if do_profiling:
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), "./profile.pt")
                torch.save(prof.table(sort_by="id", row_limit=-1),'./profile_id.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(), "./profile_detail.pt")
                prof.export_chrome_trace("./trace.json")
            toc = time.time()
            print("")
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o - i if model.config.model_type != "t5" else o
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                if args.token_latency:
                    total_list.append(output[1])
            if ref_prompt is not None and ref_prompt in gen_text:
                acc_pass += 1
            elif ref_prompt_cuda is not None and ref_prompt_cuda in gen_text:
                acc_pass += 1

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter - num_warmup)
    print("Inference latency: %.3f sec." % latency)

    if args.token_latency:
        import numpy as np
        from itertools import chain

        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        #p90_latency = average_2n[int(len(average_2n) * 0.9)]
        #p99_latency = average_2n[int(len(average_2n) * 0.99)]
        print("First token average latency: %.3f sec." % first_latency)
        print("Average 2... latency: %.3f sec." % average_2n_latency)
        #print("P90 2... latency: %.3f sec." % p90_latency)
        #print("P99 2... latency: %.3f sec." % p99_latency)
        
    if ref_prompt is None:
        print("Accuracy check skip")
    elif acc_pass==args.num_iter:
        print("Accuracy check pass")
    else:
        print("Accuracy check fail, the wrong iteration number is:", args.num_iter - acc_pass)

def to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj
      
for o, i, g in zip(to_list(args.max_new_tokens), to_list(args.input_tokens), to_list(args.num_beams)):
    run_generate(o, i, g)

