import torch
import torch.nn as nn
import torch.distributed as dist
import warnings
from dataclasses import dataclass, fields
from typing import Callable, Optional, Tuple, Union, List, Dict
from collections import OrderedDict
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP
import os
import inspect


def _ipex_prepare_model_inputs(
    self,
    inputs: Optional[torch.Tensor] = None,
    bos_token_id: Optional[int] = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
    """
    This function extracts the model-specific `inputs` for generation.
    """
    # 1. retrieve all kwargs that are non-None or non-model input related.
    # some encoder-decoder models have different names for model and encoder
    bs = inputs.shape[0]
    IPEXTransformerAtten.batch_size = bs
    IPEXTransformerMLP.batch_size = bs
    if (
        self.config.is_encoder_decoder
        and hasattr(self, "encoder")
        and self.encoder.main_input_name != self.main_input_name
    ):
        input_name = self.encoder.main_input_name
    else:
        input_name = self.main_input_name

    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

    # 2. check whether model_input_name is passed as kwarg
    # if yes and `inputs` is None use kwarg inputs
    inputs_kwarg = model_kwargs.pop(input_name, None)
    if inputs_kwarg is not None and inputs is not None:
        raise ValueError(
            f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed."
            f"Make sure to either pass {inputs} or {input_name}=..."
        )
    elif inputs_kwarg is not None:
        inputs = inputs_kwarg

    # 3. In the presence of `inputs_embeds` for text models:
    # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
    # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
    # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
    # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
    # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
    if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
        if not self.config.is_encoder_decoder:
            has_inputs_embeds_forwarding = "inputs_embeds" in set(
                inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
            )
            if not has_inputs_embeds_forwarding:
                raise ValueError(
                    f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                    "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                    "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                )
            # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
            # the attention mask) can rely on the actual model input.
            model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                inputs, bos_token_id, model_kwargs=model_kwargs
            )
        else:
            if inputs is not None:
                raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
        inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

    # 4. if `inputs` is still None, try to create `input_ids` from BOS token
    inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
    return inputs, input_name, model_kwargs

def ipex_beam_search(
    self,
    input_ids,
    beam_scorer,
    logits_processor = None,
    stopping_criteria = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.


    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }

    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )

    >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    from transformers.generation.utils import ModelOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
    from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
    print("into ipex beam search")
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, List):
        raise ValueError(f"we only support one eos token")
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    # init parameters
    finished = torch.zeros([batch_size], dtype=torch.bool, device=input_ids.device)
    length_penalty = beam_scorer.length_penalty
    do_early_stopping = beam_scorer.do_early_stopping

    index_cache = torch.zeros((0, batch_size * num_beams), dtype=torch.int, device=input_ids.device)
    max_in_seq_length = cur_len
    max_out_seq_length = stopping_criteria.max_length - max_in_seq_length
    output_token_ids = torch.empty((max_out_seq_length, batch_size * num_beams), dtype=torch.long, device=input_ids.device)
    output_beam_ids = torch.empty((max_out_seq_length, batch_size * num_beams), dtype=torch.long, device=input_ids.device)

    # init candidate pool
    candidate_num_beams = torch.zeros((batch_size), dtype=torch.long, device=input_ids.device)
    candidate_score = torch.empty((batch_size * 2 * num_beams), dtype=torch.float, device=input_ids.device)
    candidate_normed_scores = torch.empty((batch_size * 2 * num_beams), dtype=torch.float, device=input_ids.device)
    candidate_min_normed_scores = torch.empty((batch_size), dtype=torch.float, device=input_ids.device)
    candidate_output_ids = torch.empty((batch_size * num_beams, max_out_seq_length), dtype=torch.long, device=input_ids.device)
    candidate_sequence_lengths = torch.zeros((batch_size * 2 * num_beams), dtype=torch.long, device=input_ids.device)
    origin_input_ids = input_ids
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        IPEXTransformerAtten.update_beam_index(index_cache)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

        beam_scores, beam_next_tokens, beam_idx = torch.ops.torch_ipex.beam_search_topk(next_token_scores, finished, pad_token_id, eos_token_id, length_penalty, num_beams,
                                                batch_size, next_token_scores.shape[-1], cur_len-max_in_seq_length, do_early_stopping, max_in_seq_length,
                                                max_out_seq_length, output_token_ids, output_beam_ids, candidate_num_beams, candidate_normed_scores, candidate_min_normed_scores,
                                                candidate_output_ids, candidate_sequence_lengths, candidate_score)

        index_cache = torch.ops.torch_ipex.update_beam_indices_for_cache(index_cache, beam_idx, num_beams, batch_size)

        input_ids = beam_next_tokens.unsqueeze(-1)

        torch.ops.torch_ipex.update_output_indices(beam_idx, beam_next_tokens, output_beam_ids, output_token_ids, finished, cur_len-max_in_seq_length, batch_size, num_beams)
        # print(output_token_ids.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        # if model_kwargs["past_key_values"] is not None:
        #     model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if finished.all() or cur_len >= stopping_criteria.max_length:
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    out = torch.ops.torch_ipex.beam_search_finalize(candidate_num_beams, candidate_sequence_lengths, candidate_output_ids, candidate_score,
                                                    candidate_normed_scores, output_beam_ids, output_token_ids, beam_scores, finished, length_penalty, max_in_seq_length,
                                                    max_out_seq_length, batch_size, num_beams, beam_scorer.num_beam_hyps_to_keep, cur_len - max_in_seq_length, pad_token_id)

    # origin_input_ids size is [batch_size * beam_size, seq_len]
    out = torch.ops.torch_ipex.update_output_sequence(origin_input_ids, out, batch_size)
    return out


def ipex_beam_search_without_optimize(
    self,
    input_ids,
    beam_scorer,
    logits_processor = None,
    stopping_criteria = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.


    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }

    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )

    >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    from transformers.generation.utils import ModelOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
    from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
    print("into ipex beam search")
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    index_cache = torch.zeros((0, batch_size * num_beams), dtype=torch.int, device=input_ids.device)
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        IPEXTransformerAtten.update_beam_index(index_cache)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
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

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

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
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        index_cache = torch.ops.torch_ipex.update_beam_indices_for_cache(index_cache, beam_idx, num_beams, batch_size)

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
            # if model_kwargs["past_key_values"] is not None:
            #     model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
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
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]
