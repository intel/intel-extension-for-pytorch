import transformers.generation
from transformers.utils import ModelOutput
import transformers
import warnings
from packaging import version
from typing import Any, Dict, Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

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

    cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
    # add cross-attn mask for new token
    if cross_attention_mask_prev is not None:
        model_kwargs["cross_attention_mask"] = torch.cat(
            [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
        )

    try:
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        model_kwargs[cache_name] = cache
    except ValueError:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
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


def _get_attr_from_logit_processors(
    logits_processor, logit_processor_class, attribute_name
):
    logit_processor = next(
        (cls for cls in logits_processor if isinstance(cls, logit_processor_class)),
        None,
    )
    if logit_processor:
        return getattr(logit_processor, attribute_name, None)
    return None


def _pad_to_max_length(
    current_segments,
    pad_token_id,
    device,
    padding="right",
    bos_token_tensor=None,
    cut_off_length=None,
):
    max_total_length = 0
    sequences = []
    if padding not in ["right", "left"]:
        raise ValueError(f"`padding` must be either 'right' or 'left', not {padding}")

    for current_segment_list in current_segments:
        if (
            current_segment_list is not None
            and len([d["tokens"] for d in current_segment_list]) > 0
        ):
            sequence = torch.cat([d["tokens"] for d in current_segment_list], dim=-1)

            if cut_off_length is not None:
                sequence = sequence[-cut_off_length:]

            if bos_token_tensor is not None:
                sequence = torch.cat([bos_token_tensor, sequence])

            sequences.append(sequence)
            max_total_length = max(max_total_length, len(sequences[-1]))
        elif bos_token_tensor is not None:
            sequences.append(bos_token_tensor)
        else:
            sequences.append(torch.tensor([], device=device))

    for i in range(len(current_segments)):
        pad_length = max_total_length - len(sequences[i])
        pad = (0, pad_length) if padding == "right" else (pad_length, 0)
        sequences[i] = F.pad(sequences[i], pad=pad, value=pad_token_id)

    sequences = torch.stack(sequences, dim=0)
    return sequences


def whisper_generate(
    self,
    input_features: Optional[torch.Tensor] = None,
    generation_config=None,
    logits_processor=None,
    stopping_criteria=None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: bool = False,
    return_timestamps: Optional[bool] = None,
    task: Optional[str] = None,
    language: Optional[Union[str, List[str]]] = None,
    is_multilingual: Optional[bool] = None,
    prompt_ids: Optional[torch.Tensor] = None,
    prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
    condition_on_prev_tokens: Optional[bool] = None,
    temperature: Optional[Union[float, Tuple[float, ...]]] = None,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    num_segment_frames: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    time_precision: float = 0.02,
    return_token_timestamps: Optional[bool] = None,
    return_segments: bool = False,
    return_dict_in_generate: Optional[bool] = None,
    **kwargs,
):
    # 0. deprecate old inputs
    if "inputs" in kwargs:
        input_features = kwargs.pop("inputs")
        warnings.warn(
            "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
            FutureWarning,
        )

    # 1. prepare generation config
    generation_config, kwargs = self._prepare_generation_config(
        generation_config, **kwargs
    )

    # 2. set global generate variables
    input_stride = (
        self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
    )
    num_segment_frames = input_stride * self.config.max_source_positions
    batch_size, total_input_frames = self._retrieve_total_input_frames(
        input_features=input_features, input_stride=input_stride, kwargs=kwargs
    )
    is_shortform = total_input_frames <= num_segment_frames

    # 3. Make sure generation config is correctly set
    # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
    return_dict_in_generate = self._set_return_outputs(
        return_dict_in_generate=return_dict_in_generate,
        return_token_timestamps=return_token_timestamps,
        logprob_threshold=logprob_threshold,
        generation_config=generation_config,
    )
    timestamp_begin = self._set_return_timestamps(
        return_timestamps=return_timestamps,
        is_shortform=is_shortform,
        generation_config=generation_config,
    )
    self._set_language_and_task(
        language=language,
        task=task,
        is_multilingual=is_multilingual,
        generation_config=generation_config,
    )
    self._set_num_frames(
        return_token_timestamps=return_token_timestamps,
        generation_config=generation_config,
        kwargs=kwargs,
    )
    self._set_thresholds_and_condition(
        generation_config=generation_config,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_prev_tokens=condition_on_prev_tokens,
    )
    self._set_prompt_condition_type(
        generation_config=generation_config,
        prompt_condition_type=prompt_condition_type,
    )

    # pass self.config for backward compatibility
    init_tokens = self._retrieve_init_tokens(
        input_features,
        batch_size=batch_size,
        generation_config=generation_config,
        config=self.config,
        num_segment_frames=num_segment_frames,
        kwargs=kwargs,
    )
    # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
    # where the input ids are handled explicitly by the generate method
    self._check_decoder_input_ids(kwargs=kwargs)

    # 3. Retrieve logits processors
    device = (
        kwargs["encoder_outputs"][0].device
        if "encoder_outputs" in kwargs
        else input_features.device
    )
    begin_index = init_tokens.shape[1]
    logits_processor = self._retrieve_logit_processors(
        generation_config=generation_config,
        logits_processor=logits_processor,
        begin_index=begin_index,  # begin index is index of first generated decoder token
        num_beams=kwargs.get("num_beams", 1),
        device=device,
    )

    # 4 Set and retrieve global generation variables
    self._set_condition_on_prev_tokens(
        condition_on_prev_tokens=condition_on_prev_tokens,
        generation_config=generation_config,
    )

    temperatures = (
        [temperature] if not isinstance(temperature, (list, tuple)) else temperature
    )
    temperature = temperatures[0]

    max_frames, seek = self._retrieve_max_frames_and_seek(
        batch_size=batch_size,
        attention_mask=attention_mask,
        total_input_frames=total_input_frames,
        is_shortform=is_shortform,
    )

    # 5 Prepare running variables, list for generation
    num_return_sequences = generation_config.num_return_sequences
    (
        batch_idx_map,
        cur_bsz,
        input_features,
        seek,
        max_frames,
        init_tokens,
        do_condition_on_prev_tokens,
    ) = self._expand_variables_for_generation(
        input_features=input_features,
        seek=seek,
        max_frames=max_frames,
        init_tokens=init_tokens,
        batch_size=batch_size,
        condition_on_prev_tokens=condition_on_prev_tokens,
        generation_config=generation_config,
    )

    current_segments = self._prepare_segments(
        prompt_ids=prompt_ids,
        batch_size=cur_bsz,
        generation_config=generation_config,
    )

    # 6 Transcribe audio until we reach the end of all input audios
    while (seek < max_frames).any():
        # 6.1 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically
        # reduce the batch size during the loop in case one audio finished earlier than another one.
        # Thus, we need to keep a table of "previous-index-2-current-index" in order
        # to know which original audio is being decoded
        # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
        input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
            input_features=input_features,
            seek=seek,
            max_frames=max_frames,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
        )
        time_offset = seek * time_precision / input_stride
        seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

        # 6.2 cut out next 30s segment from input features
        segment_input = self._get_input_segment(
            input_features=input_features,
            seek=seek,
            seek_num_frames=seek_num_frames,
            num_segment_frames=num_segment_frames,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
        )

        # 6.3 prepare decoder input ids
        suppress_tokens = _get_attr_from_logit_processors(
            logits_processor,
            transformers.generation.logits_process.SuppressTokensLogitsProcessor,
            "suppress_tokens",
        )

        decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
            cur_bsz=cur_bsz,
            init_tokens=init_tokens,
            current_segments=current_segments,
            batch_idx_map=batch_idx_map,
            do_condition_on_prev_tokens=do_condition_on_prev_tokens,
            prompt_ids=prompt_ids,
            generation_config=generation_config,
            config=self.config,
            device=init_tokens.device,
            suppress_tokens=suppress_tokens,
            kwargs=kwargs,
        )

        # 6.4 set max new tokens or max length
        self._set_max_new_tokens_and_length(
            config=self.config,
            decoder_input_ids=decoder_input_ids,
            generation_config=generation_config,
        )

        # 6.5 Set current `begin_index` for all logit processors
        if logits_processor is not None:
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

        # 6.6 Run generate with fallback
        (
            seek_sequences,
            seek_outputs,
            should_skip,
            do_condition_on_prev_tokens,
            model_output_type,
        ) = self.generate_with_fallback(
            segment_input=segment_input,
            decoder_input_ids=decoder_input_ids,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
            seek=seek,
            num_segment_frames=num_segment_frames,
            max_frames=max_frames,
            temperatures=temperatures,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            return_token_timestamps=return_token_timestamps,
            do_condition_on_prev_tokens=do_condition_on_prev_tokens,
            is_shortform=is_shortform,
            batch_size=batch_size,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )

        # 6.7 In every generated sequence, split by timestamp tokens and extract segments
        for i, seek_sequence in enumerate(seek_sequences):
            prev_i = batch_idx_map[i]

            if should_skip[i]:
                seek[prev_i] += seek_num_frames[prev_i]
                continue

            segments, segment_offset = self._retrieve_segment(
                seek_sequence=seek_sequence,
                seek_outputs=seek_outputs,
                time_offset=time_offset,
                timestamp_begin=timestamp_begin,
                seek_num_frames=seek_num_frames,
                time_precision=time_precision,
                input_stride=input_stride,
                prev_idx=prev_i,
                idx=i,
                return_token_timestamps=return_token_timestamps,
            )

            current_segments[prev_i] += segments

            if is_shortform:
                seek[prev_i] += max_frames[i]
            else:
                seek[prev_i] += segment_offset

    # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
    # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
    final_segments = (
        [x[1:] for x in current_segments]
        if (
            prompt_ids is not None
            and generation_config.prompt_condition_type == "first-segment"
        )
        else current_segments
    )

    sequences = _pad_to_max_length(
        final_segments,
        generation_config.pad_token_id,
        device=self.device,
        padding="right",
    )

    # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
    if return_segments:
        return {"sequences": sequences, "segments": final_segments}

    if is_shortform:
        # add eos token:
        if (
            generation_config.max_new_tokens is None
            and generation_config.max_length is None
        ):
            eos_tokens = torch.full(
                (sequences.shape[0], 1), generation_config.eos_token_id
            )
            sequences = torch.cat([sequences, eos_tokens], dim=-1)

        if return_token_timestamps:
            outputs = {}
            outputs["sequences"] = sequences
            outputs["token_timestamps"] = torch.stack(
                [d["token_timestamps"] for d in seek_outputs], dim=0
            )
        elif hasattr(self.config, "token_latency") and self.config.token_latency:
            outputs = (sequences, seek_outputs[0])
        else:
            outputs = sequences

        if return_dict_in_generate and generation_config.return_dict_in_generate:
            dict_outputs = self._stack_split_outputs(
                seek_outputs, model_output_type, sequences.device, kwargs
            )

            if num_return_sequences > 1:
                if (
                    hasattr(dict_outputs, "encoder_attentions")
                    and dict_outputs.encoder_attentions is not None
                ):
                    dict_outputs.encoder_attentions = tuple(
                        dict_outputs.encoder_attentions[i][::num_return_sequences]
                        for i in range(len(dict_outputs.encoder_attentions))
                    )
                if (
                    hasattr(dict_outputs, "encoder_hidden_states")
                    and dict_outputs.encoder_hidden_states is not None
                ):
                    dict_outputs.encoder_hidden_states = tuple(
                        dict_outputs.encoder_hidden_states[i][::num_return_sequences]
                        for i in range(len(dict_outputs.encoder_hidden_states))
                    )
            if return_token_timestamps:
                dict_outputs["token_timestamps"] = outputs["token_timestamps"]
            return dict_outputs

        return outputs

    return sequences
