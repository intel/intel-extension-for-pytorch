from transformers.utils import ModelOutput


def _extract_past_from_model_output(
    self, outputs: ModelOutput, standardize_cache_format: bool = False
):
    past_key_values = None
    # To use torch.jit.trace, the output is no longer a Dict. outputs[1] corresponds to "past_key_values"
    if hasattr(self, "trace_graph"):
        past_key_values = outputs[1]
    if "past_key_values" in outputs:
        past_key_values = outputs.past_key_values
    elif "mems" in outputs:
        past_key_values = outputs.mems
    elif "past_buckets_states" in outputs:
        past_key_values = outputs.past_buckets_states

    # Bloom fix: standardizes the cache format when requested
    if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
        batch_size = outputs.logits.shape[0]
        past_key_values = self._convert_to_standard_cache(
            past_key_values, batch_size=batch_size
        )
    return past_key_values
