import torch
from torch import nn
from typing import Union
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.logits_process import LogitsProcessorList
import time
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
)

GenerateBeamOutput = Union[
    GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput
]


def _beam_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config,
    synced_gpus: bool,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    # 1. init beam_search values
    do_sample = generation_config.do_sample
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    early_stopping = generation_config.early_stopping
    length_penalty = generation_config.length_penalty
    max_length = generation_config.max_length
    num_beams = generation_config.num_beams
    num_return_sequences = generation_config.num_return_sequences

    batch_size_unflattened, cur_len = input_ids.shape
    batch_size = batch_size_unflattened // num_beams
    # TODO (joao): standardize special cases
    if self.__class__.__name__ == "MoshiDepthDecoder":
        vocab_size = self.config.audio_vocab_size
    elif self.__class__.__name__ == "ImageGPTForCausalImageModeling":
        vocab_size = self.get_output_embeddings().out_features
    else:
        vocab_size = self.config.get_text_config().vocab_size
    decoder_prompt_len = cur_len
    this_peer_finished = False
    token_latency = (
        self.config.token_latency if hasattr(self.config, "token_latency") else False
    )
    # At each beam search step, we want to keep top K [K = (number of EOS tokens + 1) * `num_beams`] candidates
    # with the highest log-probabilities, or sample K continuations without replacement. We gather the top K
    # (as opposed to `num_beams`, or any number lower than K) so that we have at least `num_beams` sequences
    # non-finished to continue the live beam search, in case the top `num_beams` all select an EOS token.
    n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
    beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
    top_num_beam_mask = torch.cat(
        (
            torch.ones((num_beams), dtype=torch.bool),
            torch.zeros((beams_to_keep - num_beams), dtype=torch.bool),
        ),
        dim=0,
    ).to(input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    latency_list = []
    # (joao) feature lost in the refactor. Probably won't implement, hurts readbility with minimal gains (there
    # are newer low-memory alternatives like the offloaded cache)
    sequential = generation_config.low_memory
    if sequential:
        raise ValueError(
            "`low_memory=True` is not supported after the beam search refactor. Please check the discussion in "
            "#35802 *after the PR got merged*, and add a comment there if your questions are not yet answered."
        )

    # 2. init output tuples
    all_scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # 3. init running tensors and static-shaped placeholders

    # per batch, beam-item holding current token in loop and completed sequences
    output_fill_value = (
        pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
    )
    running_sequences = torch.full(
        (batch_size, num_beams, max_length),
        fill_value=output_fill_value,
        dtype=torch.int64,
        device=input_ids.device,
    )
    running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(
        input_ids, batch_size, num_beams
    )
    sequences = running_sequences.detach().clone()

    # per batch, beam-item score, logprobs
    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    running_beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    running_beam_scores[:, 1:] = -1e9
    beam_scores = torch.full(
        (batch_size, num_beams),
        fill_value=-1e9,
        dtype=torch.float,
        device=input_ids.device,
    )

    # per batch, beam-item state bit indicating if sentence has finished.
    is_sent_finished = torch.zeros(
        (batch_size, num_beams), dtype=torch.bool, device=input_ids.device
    )

    # per batch, beam-item state bit indicating if there are valid continuations.
    next_token_hits_stopping_criteria = torch.zeros(
        (batch_size, num_beams), dtype=torch.bool, device=input_ids.device
    )

    # per batch selected beam indices
    running_beam_indices = torch.full(
        (batch_size, num_beams, max_length - cur_len),
        fill_value=-1,
        dtype=torch.int32,
        device=input_ids.device,
    )
    beam_indices = running_beam_indices.detach().clone()
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        tic = time.time()
        # a. Forward current tokens, obtain the logits
        flat_running_sequences = self._flatten_beam_dim(
            running_sequences[:, :, :cur_len]
        )
        if "past_key_values" in model_kwargs and not isinstance(
            model_kwargs["past_key_values"], tuple
        ):
            model_kwargs["past_key_values"] = None

        model_inputs = self.prepare_inputs_for_generation(
            flat_running_sequences, **model_kwargs
        )
        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        self.model_backbone = self.config.architectures[0]
        if self.model_backbone in [
            "GPTJForCausalLM",
            "LlamaForCausalLM",
            "MllamaForConditionalGeneration",
            "GPTNeoXForCausalLM",
            "OPTForCausalLM",
            "FalconForCausalLM",
            "RWForCausalLM",
            "BloomForCausalLM",
            "CodeGenForCausalLM",
            "BaichuanForCausalLM",
            "ChatGLMModel",
            "GPTBigCodeForCausalLM",
            "T5ForConditionalGeneration",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "MptForCausalLM",
            "StableLmForCausalLM",
            "QWenLMHeadModel",
            "Qwen3MoeForCausalLM",
            "Qwen3ForCausalLM",
            "GitForCausalLM",
            "LlavaLlamaForCausalLM",
            "YuanForCausalLM",
            "PhiForCausalLM",
            "Phi3ForCausalLM",
            "Phi4MMForCausalLM",
            "WhisperForConditionalGeneration",
            "Qwen2ForCausalLM",
            "Maira2ForConditionalGeneration",
            "JambaForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
        ]:
            first_token = False
            has_position_id = model_inputs.get("position_ids", None) is not None
            if hasattr(self.config, "kv_cache_dtype"):
                kv_cache_dtype = self.config.kv_cache_dtype
            elif hasattr(self, "dtype"):
                kv_cache_dtype = self.dtype
            else:
                kv_cache_dtype = torch.float
            if model_inputs["past_key_values"] is None:
                first_token = True
                if self.model_backbone == "T5ForConditionalGeneration":
                    first_token = False
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long
                    ).contiguous()
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                beam_idx_tmp,
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                self.decoder.block[i]
                                .layer[1]
                                .EncDecAttention.k(
                                    model_inputs["encoder_outputs"]["last_hidden_state"]
                                )
                                .view(
                                    int(batch_size * num_beams),
                                    -1,
                                    self.decoder.block[i]
                                    .layer[1]
                                    .EncDecAttention.n_heads,
                                    self.decoder.block[i]
                                    .layer[1]
                                    .EncDecAttention.key_value_proj_dim,
                                )
                                .transpose(0, 1),
                                self.decoder.block[i]
                                .layer[1]
                                .EncDecAttention.v(
                                    model_inputs["encoder_outputs"]["last_hidden_state"]
                                )
                                .view(
                                    int(batch_size * num_beams),
                                    -1,
                                    self.decoder.block[i]
                                    .layer[1]
                                    .EncDecAttention.n_heads,
                                    self.decoder.block[i]
                                    .layer[1]
                                    .EncDecAttention.key_value_proj_dim,
                                )
                                .transpose(0, 1),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
                elif self.model_backbone == "GitForCausalLM":
                    first_token = False
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long
                    ).contiguous()
                    num_head = self.git.encoder.layer[
                        0
                    ].attention.self.num_attention_heads
                    head_dim = int(
                        self.git.encoder.layer[0].attention.self.hidden_size / num_head
                    )
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros(
                                    [int(batch_size * num_beams), num_head, 1, head_dim]
                                )
                                .contiguous()
                                .to(kv_cache_dtype),
                                torch.zeros(
                                    [int(batch_size * num_beams), num_head, 1, head_dim]
                                )
                                .contiguous()
                                .to(kv_cache_dtype),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
                elif self.model_backbone == "WhisperForConditionalGeneration":
                    first_token = False
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long
                    ).contiguous()
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                beam_idx_tmp,
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                self.model.decoder.layers[i]
                                .encoder_attn.k_proj(
                                    model_inputs["encoder_outputs"]["last_hidden_state"]
                                )
                                .view(
                                    int(batch_size * num_beams),
                                    -1,
                                    self.model.decoder.layers[i].encoder_attn.num_heads,
                                    self.model.decoder.layers[i].encoder_attn.head_dim,
                                )
                                .contiguous(),
                                self.model.decoder.layers[i]
                                .encoder_attn.v_proj(
                                    model_inputs["encoder_outputs"]["last_hidden_state"]
                                )
                                .view(
                                    int(batch_size * num_beams),
                                    -1,
                                    self.model.decoder.layers[i].encoder_attn.num_heads,
                                    self.model.decoder.layers[i].encoder_attn.head_dim,
                                )
                                .contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
            if first_token and self.model_backbone != "YuanForCausalLM":
                if hasattr(self.config, "n_layer"):
                    num_hidden_layers = self.config.n_layer
                elif hasattr(self.config, "num_hidden_layers"):
                    num_hidden_layers = self.config.num_hidden_layers
                elif hasattr(self.config, "text_config") and hasattr(
                    self.config.text_config, "num_hidden_layers"
                ):
                    num_hidden_layers = self.config.text_config.num_hidden_layers
                elif hasattr(self.config, "num_layers"):
                    num_hidden_layers = self.config.num_layers
                elif hasattr(self.config, "n_layers"):
                    num_hidden_layers = self.config.n_layers
                beam_idx_tmp = torch.zeros(
                    (2048, int(batch_size * num_beams)), dtype=torch.long
                )

                if self.model_backbone == "MllamaForConditionalGeneration":
                    head_dim = self.config.text_config.hidden_size // (
                        self.config.text_config.num_hidden_layers
                        - len(self.config.text_config.cross_attention_layers)
                    )
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                (
                                    torch.zeros(
                                        1, 0, 0, 1, dtype=torch.long
                                    ).contiguous(),
                                    torch.zeros([1, 1, 1, 1])
                                    .contiguous()
                                    .to(kv_cache_dtype),
                                    torch.zeros([1, 1, 1, 1])
                                    .contiguous()
                                    .to(kv_cache_dtype),
                                    beam_idx_tmp,
                                )
                                if i
                                not in self.config.text_config.cross_attention_layers
                                else (
                                    torch.zeros([1, 1, 1, head_dim])
                                    .contiguous()
                                    .to(kv_cache_dtype),
                                    torch.zeros([1, 1, 1, head_dim])
                                    .contiguous()
                                    .to(kv_cache_dtype),
                                )
                            )
                            for i in range(num_hidden_layers)
                        ]
                    )
                elif self.model_backbone == "JambaForCausalLM":
                    intermediate_size = (
                        self.config.mamba_expand * self.config.hidden_size
                    )
                    conv_kernel_size = self.config.mamba_d_conv
                    ssm_state_size = self.config.mamba_d_state
                    dtype = (
                        self.config.dtype
                        if hasattr(self.config, "dtype")
                        else self.dtype
                    )
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                (
                                    torch.zeros(
                                        1, 0, 0, 1, dtype=torch.long
                                    ).contiguous(),
                                    torch.zeros([1, 1, 1, 1]).contiguous(),
                                    torch.zeros([1, 1, 1, 1]).contiguous(),
                                    beam_idx_tmp,
                                )
                                if i % self.config.attn_layer_period
                                == self.config.attn_layer_offset
                                else (
                                    torch.zeros(
                                        int(batch_size * num_beams),
                                        intermediate_size,
                                        ssm_state_size,
                                        dtype=dtype,
                                    ).contiguous(),
                                    torch.zeros(
                                        int(batch_size * num_beams),
                                        intermediate_size,
                                        conv_kernel_size,
                                        dtype=dtype,
                                    ).contiguous(),
                                    torch.tensor(False).contiguous(),
                                )
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
                elif self.model_backbone in [
                    "DeepseekV2ForCausalLM",
                    "DeepseekV3ForCausalLM",
                ]:
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),  # latent_cache
                                beam_idx_tmp,
                            )
                            for i in range(num_hidden_layers)
                        ]
                    )
                else:
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                beam_idx_tmp,
                            )
                            for i in range(num_hidden_layers)
                        ]
                    )
                if self.model_backbone not in [
                    "MllamaForConditionalGeneration",
                    "JambaForCausalLM",
                ]:
                    new_attention_mask = model_inputs["attention_mask"][
                        :batch_size
                    ].clone()
                    new_input_ids = model_inputs["input_ids"][:batch_size].clone()
                    if has_position_id:
                        new_position_ids = model_inputs["position_ids"][
                            :batch_size
                        ].clone()
                    for i in range(batch_size):
                        new_attention_mask[i] = model_inputs["attention_mask"][
                            i * num_beams
                        ]
                        new_input_ids[i] = model_inputs["input_ids"][i * num_beams]
                        if has_position_id:
                            new_position_ids[i] = model_inputs["position_ids"][
                                i * num_beams
                            ]
                    model_inputs["attention_mask"] = new_attention_mask
                    model_inputs["input_ids"] = new_input_ids
                    if has_position_id:
                        model_inputs["position_ids"] = new_position_ids
            model_inputs.pop("use_cache", None)
            model_inputs.pop("token_type_ids", None)
            if "return_last_logit" in model_inputs:
                model_inputs["return_last_logit"] = torch.tensor(
                    model_inputs["return_last_logit"]
                )
            if self.model_backbone == "T5ForConditionalGeneration":
                model_inputs.pop("head_mask", None)
                model_inputs.pop("decoder_head_mask", None)
                model_inputs.pop("decoder_attention_mask", None)
                model_inputs.pop("cross_attn_head_mask", None)
                model_inputs["encoder_outputs"] = (
                    model_inputs["encoder_outputs"]["last_hidden_state"],
                )
            if self.model_backbone == "WhisperForConditionalGeneration":
                model_inputs["encoder_outputs"] = (
                    model_inputs["encoder_outputs"]["last_hidden_state"],
                )
                model_inputs.pop("decoder_position_ids", None)
                model_inputs.pop("decoder_attention_mask", None)
            if self.model_backbone == "LlavaLlamaForCausalLM" and hasattr(
                self, "prepare_inputs_labels_for_multimodal"
            ):
                model_inputs = self.prepare_inputs_labels_for_multimodal(**model_inputs)
            if first_token and self.model_backbone == "YuanForCausalLM":
                model_inputs.pop("past_key_values", None)
            if (
                not first_token
                and self.model_backbone == "Maira2ForConditionalGeneration"
            ):
                model_inputs.pop("pixel_values", None)
            model_inputs.pop("cache_position", None)
            if self.model_backbone == "JambaForCausalLM":
                model_inputs["output_router_logits"] = torch.tensor(
                    model_inputs["output_router_logits"]
                )
                model_inputs["num_logits_to_keep"] = torch.tensor(
                    model_inputs["num_logits_to_keep"]
                )
            if self.model_backbone == "Phi3ForCausalLM":
                model_inputs.pop("inputs_embeds", None)
                model_inputs.pop("num_logits_to_keep", None)
            if hasattr(self, "trace_graph"):
                if first_token and hasattr(self, "trace_graph_first"):
                    outputs = self.trace_graph_first(**model_inputs)
                else:
                    outputs = self.trace_graph(**model_inputs)
            else:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            if (
                first_token
                and self.model_backbone != "YuanForCausalLM"
                and self.model_backbone != "MllamaForConditionalGeneration"
                and (
                    len(model_inputs["past_key_values"][0]) == 4
                    or self.model_backbone
                    in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]
                )
            ):
                if isinstance(outputs, dict):
                    outputs.logits = outputs.logits.repeat_interleave(num_beams, dim=0)
                else:
                    outputs = list(outputs)
                    outputs[0] = outputs[0].repeat_interleave(num_beams, dim=0)
                    outputs = tuple(outputs)
        else:
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        if isinstance(outputs, dict):
            next_token_logits = outputs.logits[:, -1, :]
        else:
            next_token_logits = outputs[0][:, -1, :]
        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref
        logits = next_token_logits.to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )

        # b. Compute log probs -- get log probabilities from logits, process logits with processors (*e.g.*
        # `temperature`, ...), and add new logprobs to existing running logprobs scores.
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs = logits_processor(flat_running_sequences, log_probs)

        # Store logits, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_logits:
                raw_logits += (logits.clone(),)
            if return_dict_in_generate and output_scores:
                all_scores += (log_probs.clone(),)

            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # This is needed to properly delete logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

        log_probs = self._unflatten_beam_dim(log_probs, batch_size, num_beams)
        log_probs = log_probs + running_beam_scores[:, :, None]
        log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

        # c. Retrieve top-K continuations, i.e. select the next token (greedy or sampling) and then keep the best
        # continuations among all beams based on the accumulated scores.
        topk_log_probs, topk_running_sequences, topk_running_beam_indices = (
            self._get_top_k_continuations(
                accumulated_log_probs=log_probs,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=do_sample,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )
        )

        # d. Check which running sequences have finished
        next_token_hits_stopping_criteria = stopping_criteria(
            self._flatten_beam_dim(
                topk_running_sequences[:, :, : cur_len + 1]
            ),  # remove unfilled token indexes
            all_scores,
        )
        next_token_hits_stopping_criteria = self._unflatten_beam_dim(
            next_token_hits_stopping_criteria, batch_size, beams_to_keep
        )

        # e. Get the non-finished running `num_beams` sequences for the next generation step
        running_sequences, running_beam_scores, running_beam_indices = (
            self._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )
        )

        # f. Update the completed beams if a new high score in a finished sequence is found
        sequences, beam_scores, beam_indices, is_sent_finished = (
            self._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )
        )

        # g. Prepare remaining data for the next iteration, including computing the stopping condition for
        # beam search as a whole (as opposed to individual beams, i.e. `stopping_criteria`)

        # pluck the cache from the beam indices that will be used in the next iteration
        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                past_key_values=model_kwargs["past_key_values"],
                beam_idx=self._flatten_beam_dim(
                    running_beam_indices[..., cur_len - decoder_prompt_len]
                ),
            )

        # increase cur_len
        cur_len = cur_len + 1
        latency_list.append(time.time() - tic)
        this_peer_finished = not self._beam_search_has_unfinished_sequences(
            running_beam_scores,
            beam_scores,
            is_sent_finished,
            next_token_hits_stopping_criteria,
            cur_len,
            max_length,
            decoder_prompt_len,
            early_stopping,
            length_penalty,
        )
    # 5. prepare outputs
    # Take best beams for each batch (the score is sorted in descending order)
    sequences = self._flatten_beam_dim(sequences[:, :num_return_sequences, :])
    beam_scores = self._flatten_beam_dim(beam_scores[:, :num_return_sequences])
    beam_indices = self._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

    # Crop the static-shaped tensors to the actual size.
    # `beam_indices` is initialized with -1s, and is updated with the beam index of the generated token at each
    # step. We can use it to detect the generated length, which may be != `cur_len`  (e.g. selected beam is from a
    # previous decoding iteration)
    max_generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
    output_length = decoder_prompt_len + max_generated_length
    sequences = sequences[:, :output_length]
    beam_indices = beam_indices[:, :max_generated_length]
    if return_dict_in_generate:
        if not output_scores:
            beam_scores = None

        if self.config.is_encoder_decoder:
            output_result = GenerateBeamEncoderDecoderOutput(
                sequences=sequences,
                sequences_scores=beam_scores,
                scores=all_scores,
                logits=raw_logits,
                beam_indices=beam_indices,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            output_result = GenerateBeamDecoderOnlyOutput(
                sequences=sequences,
                sequences_scores=beam_scores,
                scores=all_scores,
                logits=raw_logits,
                beam_indices=beam_indices,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        output_result = sequences
    # result
    if token_latency:
        return (output_result, latency_list)
    else:
        return output_result
