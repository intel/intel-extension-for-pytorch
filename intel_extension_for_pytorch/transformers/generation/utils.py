from transformers.utils import ModelOutput
import transformers
from packaging import version
from typing import Any, Dict
import torch

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


def _update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    # update past_key_values keeping its naming used in model code
    cache_name, cache = self._extract_past_from_model_output(
        outputs, standardize_cache_format=standardize_cache_format
    )
    model_kwargs[cache_name] = cache
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat(
            [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
        )

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [
                    decoder_attention_mask,
                    decoder_attention_mask.new_ones(
                        (decoder_attention_mask.shape[0], 1)
                    ),
                ],
                dim=-1,
            )

    if (
        model_kwargs.get("use_cache", True)
        and "cache_position" in model_kwargs
        and model_kwargs["cache_position"] is not None
    ):
        model_kwargs["cache_position"] = (
            model_kwargs["cache_position"][-1:] + num_new_tokens
        )

    return model_kwargs
