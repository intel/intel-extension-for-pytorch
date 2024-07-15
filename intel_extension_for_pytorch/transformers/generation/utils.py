from transformers.utils import ModelOutput
import transformers
from packaging import version

trans_version = transformers.__version__


def _extract_past_from_model_output(
    self, outputs: ModelOutput, standardize_cache_format: bool = False
):
    past_key_values = None
    cache_name = "past_key_values"
    # To use torch.jit.trace, the output is no longer a Dict. outputs[1] corresponds to "past_key_values"
    if hasattr(self, "trace_graph"):
        past_key_values = outputs[1]
    if "past_key_values" in outputs:
        past_key_values = outputs.past_key_values
    elif "mems" in outputs:
        past_key_values = outputs.mems
    elif "past_buckets_states" in outputs:
        past_key_values = outputs.past_buckets_states
    elif "cache_params" in outputs:
        past_key_values = outputs.cache_params
        cache_name = "cache_params"

    # Bloom fix: standardizes the cache format when requested
    if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
        batch_size = outputs.logits.shape[0]
        past_key_values = self._convert_to_standard_cache(
            past_key_values, batch_size=batch_size
        )
    if version.parse(trans_version) < version.parse("4.42.0"):
        return past_key_values
    return cache_name, past_key_values
