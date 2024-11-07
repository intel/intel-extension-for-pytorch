import torch
from torch import nn
import torch.distributed as dist
import warnings
from typing import Optional, Union, List
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.beam_search import BeamScorer
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
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    # init values
    token_latency = (
        self.config.token_latency if hasattr(self.config, "token_latency") else False
    )

    latency_list = []
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
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

    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
    while True:
        tic = time.time()
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        if "past_key_values" in model_kwargs and not isinstance(
            model_kwargs["past_key_values"], tuple
        ):
            model_kwargs["past_key_values"] = None
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

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
            "GitForCausalLM",
            "LlavaLlamaForCausalLM",
            "YuanForCausalLM",
            "PhiForCausalLM",
            "Phi3ForCausalLM",
            "WhisperForConditionalGeneration",
            "Qwen2ForCausalLM",
        ]:
            first_token = False
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
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
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
                                ).contiguous(),
                                torch.zeros(
                                    [int(batch_size * num_beams), num_head, 1, head_dim]
                                ).contiguous(),
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
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
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
            if first_token:
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
                ).contiguous()
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
                                    torch.zeros([1, 1, 1, 1]).contiguous(),
                                    torch.zeros([1, 1, 1, 1]).contiguous(),
                                    beam_idx_tmp,
                                )
                                if i
                                not in self.config.text_config.cross_attention_layers
                                else (
                                    torch.zeros([1, 1, 1, head_dim]).contiguous(),
                                    torch.zeros([1, 1, 1, head_dim]).contiguous(),
                                )
                            )
                            for i in range(num_hidden_layers)
                        ]
                    )
                else:
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(num_hidden_layers)
                        ]
                    )
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
            model_inputs.pop("cache_position", None)
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
        else:
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need
        if isinstance(outputs, dict):
            next_token_logits = outputs.logits[:, -1, :]
        else:
            next_token_logits = outputs[0][:, -1, :]

        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        if logits_warper is not None:
            next_token_scores_processed = logits_warper(
                input_ids, next_token_scores_processed
            )
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
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

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        probs = nn.functional.softmax(next_token_scores, dim=-1)

        next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
        next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

        next_token_scores, _indices = torch.sort(
            next_token_scores, descending=True, dim=1
        )
        next_tokens = torch.gather(next_tokens, -1, _indices)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        # increase cur_len
        cur_len = cur_len + 1
        latency_list.append(time.time() - tic)
        stopping_res = stopping_criteria(input_ids, scores)
        is_stopped = (
            stopping_res if isinstance(stopping_res, bool) else all(stopping_res)
        )
        if beam_scorer.is_done or is_stopped:
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            output_result = GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            output_result = GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        output_result = sequence_outputs["sequences"]
    # result
    if token_latency:
        return (output_result, latency_list)
    else:
        return output_result
