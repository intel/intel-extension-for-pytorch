import logging
import torch
from pathlib import Path

format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger("GPTQ")
logger.setLevel(logging.INFO)


@torch.no_grad()
def gptq(
    model,
    dataloader,
    group_size=128,
    wbits=4,
    sym=False,
    percdamp=0.01,
    act_order=False,
    nsamples=128,
    use_max_length=False,
    pad_max_length=2048,
    layer_wise=False,
    # export arguments
    compression_dtype=torch.int32,
    compression_dim=1,
    scale_dtype=torch.float16,
    save_dir="saved_results",
):
    """User API to run GPTQ; quantize and save checkpoint to designated path.

    Args:
        model (torch.nn.Module): fp32 model to quantize
        dataloader (torch.utils.data.DataLoader): an iterable containing calibration datasets.
        wbits (int): number of bits of the data type for weight.
        group_size (int): control quantization granularity along input channel (IC) dimension of weight.
                        Must be a positive power of 2 or -1.
        sym (bool): scheme. Default to be asym per checkpoint requirement.
        percdamp (float): percentage of Hessian's diagonal values' average.
        act_order (bool): whether to sort Hessian's diagonal values to rearrange channel-wise quantization order.
        nsamples (int): calibration samples' size.
        use_max_length (bool): whether to align calibration data to a fixed length.
        pad_max_length (int): whether to align calibration data to a fixed length.
        device: set to torch.device("cpu").
        layer_wise (bool): whether to do LWQ.
        compression_dtype: data type for compressed dtype, select from [torch.int8|16|32|64].
        compression_dim (int): 0 means output channel while 1 means input channel.
        scale_dtype: data type for scale and bias.
        save_dir (str): path to save checkpoint.
    """
    logger.warning(
        "The GPTQ API is deprecated. Please use the Intel(R) Neural Compressor to run GPTQ instead."
    )

    logger.info("quantizing with GPTQ algorithm")
    from ._gptq_utils import gptq_quantize, gptq_export

    for model_name in ["model", "transformer"]:
        if hasattr(model, model_name) and hasattr(
            getattr(model, model_name), "_use_sdpa"
        ):
            getattr(model, model_name)._use_sdpa = False
        if hasattr(model, model_name):
            cur_mod = getattr(model, model_name)
            for submodel_name in ["encoder", "decoder"]:
                if hasattr(cur_mod, submodel_name) and hasattr(
                    getattr(cur_mod, submodel_name), "_use_sdpa"
                ):
                    getattr(cur_mod, submodel_name)._use_sdpa = False

    model_path = None
    weight_config = {}
    for name, module in model.named_modules():
        if "lm_head" in name or "output_layer" in name or "embed_out" in name:
            continue
        if isinstance(module, torch.nn.modules.linear.Linear):
            weight_config[name] = {
                "wbits": wbits,
                "group_size": group_size,
                "sym": sym,
                "percdamp": percdamp,
                "act_order": act_order,
            }
    if use_max_length and pad_max_length is None:
        logger.warning(
            "You choose to use unified sequence length for calibration"
            + "but you have not set length value. Default sequence length"
            + "is 2048 and this might cause inference error!"
        )
    model, gptq_config = gptq_quantize(
        model,
        weight_config,
        dataloader,
        nsamples,
        use_max_length,
        pad_max_length,
        layer_wise,
        model_path,
    )
    logger.info("Exporting compressed model...")
    compressed_model = gptq_export(
        model,
        weight_config,
        gptq_config,
        compression_dtype,
        compression_dim,
        scale_dtype,
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_file_name = f"gptq_checkpoint_g{group_size}.pt"
    output_file_path = save_dir + "/" + output_file_name
    torch.save(compressed_model.state_dict(), output_file_path)
    logger.info(
        "Low-precision checkpoint generated and saved to {}.".format(output_file_path)
    )
    return compressed_model
