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
from .common import _model_forward

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

        outputs = _model_forward(
            self,
            batch_size,
            num_beams,
            model_kwargs,
            flat_running_sequences,
            output_attentions,
            output_hidden_states,
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


def _beam_sample_legacy(
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

        outputs = _model_forward(
            self,
            batch_size,
            num_beams,
            model_kwargs,
            input_ids,
            output_attentions,
            output_hidden_states,
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
