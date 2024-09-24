import torch
import torch.nn as nn
import torch.distributed as dist
import warnings
from functools import partial
from typing import Optional, Tuple, Union, List, Dict
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive
import time
from .utils import pad_for_gptj_lm_head, is_int4, pad_for_chatglm_output_layer
from .transformer_modules.CacheUtils import warp_cache_if_needed
from .transformer_modules.CacheUtils import IPEXStaticCache


class IPEXLLMResourceContrainer:
    container = []

    @staticmethod
    def push(resource_block):
        IPEXLLMResourceContrainer.container.append(resource_block)

    @staticmethod
    def release_resources():
        print("release resources")
        for resource_block in IPEXLLMResourceContrainer.container:
            resource_block.release_resources()


def gptj_forward_hook(model):
    import transformers

    if type(model) == transformers.models.gptj.modeling_gptj.GPTJForCausalLM:
        pad_for_gptj_lm_head(model, is_int4(model))


def llama_forward_hook(model):
    import transformers

    if type(model) == transformers.models.llama.modeling_llama.LlamaForCausalLM:
        pad_for_gptj_lm_head(model, is_int4(model))


def opt_forward_hook(model):
    import transformers

    if type(model) == transformers.models.opt.modeling_opt.OPTForCausalLM:
        pad_for_gptj_lm_head(model, is_int4(model))


def falcon_forward_hook(model):
    import transformers

    if type(model) == transformers.models.falcon.modeling_falcon.FalconForCausalLM:
        pad_for_gptj_lm_head(model, is_int4(model))


def baichuan_forward_hook(model):
    if model.__class__.__name__ == "BaichuanForCausalLM":
        pad_for_gptj_lm_head(model, is_int4(model))


def chatglm_forward_hook(model):
    if model.__class__.__name__ == "ChatGLMModel":
        pad_for_chatglm_output_layer(model, is_int4(model))


def bloom_forward_hook(model):
    import transformers

    if type(model) == transformers.models.bloom.modeling_bloom.BloomForCausalLM:
        pad_for_gptj_lm_head(model, is_int4(model))


def qwen_forward_hook(model):
    if model.__class__.__name__ == "QWenLMHeadModel":
        pad_for_gptj_lm_head(model, is_int4(model))


def _convert_to_bloom_cache_ipex(
    past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
    """
    batch_size, num_heads, seq_length, head_dim = past_key_value[0][0].shape

    return tuple(
        (
            layer_past[0].view(batch_size, num_heads, seq_length, head_dim),
            layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
        )
        for layer_past in past_key_value
    )


def ipex_convert_to_bloom_cache(model):
    if hasattr(model, "_convert_to_bloom_cache"):
        setattr(model, "_convert_to_bloom_cache", _convert_to_bloom_cache_ipex)  # noqa


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
    if (
        self.config.is_encoder_decoder
        and hasattr(self, "encoder")
        and self.encoder.main_input_name != self.main_input_name
    ):
        input_name = self.encoder.main_input_name
    else:
        input_name = self.main_input_name

    model_kwargs = {
        k: v for k, v in model_kwargs.items() if v is not None or k != input_name
    }

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
                raise ValueError(
                    "You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one."
                )
        inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

    # 4. if `inputs` is still None, try to create `input_ids` from BOS token
    inputs = self._maybe_initialize_input_ids_for_generation(
        inputs, bos_token_id, model_kwargs
    )

    bs = inputs.shape[0]
    IPEXTransformerAttn.batch_size = bs
    IPEXTransformerAttn.reset_timestamp()
    IPEXTransformerAttn.release_resources()
    return inputs, input_name, model_kwargs


def ipex_prepare_model_inputs(model):
    if hasattr(model, "_prepare_model_inputs"):
        setattr(  # noqa B010
            model, "_prepare_model_inputs", partial(_ipex_prepare_model_inputs, model)
        )


def ipex_prepare_inputs_for_generation(model):
    if hasattr(model, "prepare_inputs_for_generation"):
        func_ptr = model.prepare_inputs_for_generation

        def prepare_inputs_for_generation_ipex_wrapper(self, *args, **kwargs):
            model_inputs = func_ptr(self, *args, **kwargs)
            if "past_key_values" in model_inputs:
                model_inputs["past_key_values"] = warp_cache_if_needed(
                    model_inputs["past_key_values"]
                )
            return model_inputs

        setattr(  # noqa B010
            model,
            "prepare_inputs_for_generation",
            prepare_inputs_for_generation_ipex_wrapper,
        )


def build_bloom_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/
    a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    import math

    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    powers = torch.arange(
        1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32
    )
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src
    # /transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    if dist.is_initialized():
        num_heads_per_rank = int(num_heads / dist.get_world_size())
        offset = dist.get_rank() * num_heads_per_rank
        alibi = alibi.view(batch_size, num_heads, 1, seq_length)
        alibi = alibi[:, offset : num_heads_per_rank + offset, :, :]
        return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
    else:
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def ipex_build_bloom_alibi_tensor(model):
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "build_alibi_tensor"):
            setattr(  # noqa B010
                model.transformer, "build_alibi_tensor", build_bloom_alibi_tensor
            )


def _ipex_beam_search(
    self,
    input_ids,
    beam_scorer,
    logits_processor=None,
    stopping_criteria=None,
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
    # from transformers.generation.utils import ModelOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
    # from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        validate_stopping_criteria,
    )

    # init values
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
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn(
            "You don't have defined any stopping_criteria, this will likely loop forever",
            UserWarning,
        )
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

    if isinstance(eos_token_id, List):
        raise ValueError("we only support one eos token")
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

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

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

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    # init parameters
    finished = torch.zeros([batch_size], dtype=torch.bool, device=input_ids.device)
    length_penalty = beam_scorer.length_penalty
    do_early_stopping = beam_scorer.do_early_stopping

    index_cache = torch.zeros(
        (0, batch_size * num_beams), dtype=torch.int, device=input_ids.device
    )
    max_in_seq_length = cur_len
    max_out_seq_length = stopping_criteria.max_length - max_in_seq_length
    output_token_ids = torch.empty(
        (max_out_seq_length, batch_size * num_beams),
        dtype=torch.long,
        device=input_ids.device,
    )
    output_beam_ids = torch.empty(
        (max_out_seq_length, batch_size * num_beams),
        dtype=torch.long,
        device=input_ids.device,
    )

    # init candidate pool
    candidate_num_beams = torch.zeros(
        (batch_size), dtype=torch.long, device=input_ids.device
    )
    candidate_score = torch.empty(
        (batch_size * 2 * num_beams), dtype=torch.float, device=input_ids.device
    )
    candidate_normed_scores = torch.empty(
        (batch_size * 2 * num_beams), dtype=torch.float, device=input_ids.device
    )
    candidate_min_normed_scores = torch.empty(
        (batch_size), dtype=torch.float, device=input_ids.device
    )
    candidate_output_ids = torch.empty(
        (batch_size * 2 * num_beams, max_out_seq_length),
        dtype=torch.long,
        device=input_ids.device,
    )
    candidate_sequence_lengths = torch.zeros(
        (batch_size * 2 * num_beams), dtype=torch.long, device=input_ids.device
    )
    origin_input_ids = input_ids
    IPEXTransformerAttn.reset_timestamp()
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

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        IPEXTransformerAttnNaive.update_beam_idx(index_cache)

        # if self.method == RunMethods.JITInfer:
        #    model_inputs.pop("use_cache", None)
        #    model_inputs.pop("token_type_ids", None)
        if hasattr(self, "model_capture"):
            outputs = self.model_capture["model_capture"](
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

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        next_token_logits = None
        dummy_input_ids = torch.empty(
            (batch_size * num_beams, cur_len), dtype=torch.long, device="meta"
        )
        next_token_scores_processed = logits_processor(
            dummy_input_ids, next_token_scores
        )
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores)
        beam_scores, beam_next_tokens, beam_idx = torch.ops.torch_ipex.beam_search_topk(
            next_token_scores,
            finished,
            pad_token_id,
            eos_token_id,
            length_penalty,
            num_beams,
            batch_size,
            next_token_scores.shape[-1],
            cur_len - max_in_seq_length,
            do_early_stopping,
            max_in_seq_length,
            max_out_seq_length,
            output_token_ids,
            output_beam_ids,
            candidate_num_beams,
            candidate_normed_scores,
            candidate_min_normed_scores,
            candidate_output_ids,
            candidate_sequence_lengths,
            candidate_score,
        )
        index_cache = torch.ops.torch_ipex.update_beam_indices_for_cache(
            index_cache, beam_idx, num_beams, batch_size
        )

        input_ids = beam_next_tokens.unsqueeze(-1)
        torch.ops.torch_ipex.update_output_indices(
            beam_idx,
            beam_next_tokens,
            output_beam_ids,
            output_token_ids,
            finished,
            cur_len - max_in_seq_length,
            batch_size,
            num_beams,
        )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        outputs = None
        # if model_kwargs["past_key_values"] is not None:
        #     model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )
        # increase cur_len
        cur_len = cur_len + 1
        if hasattr(self, "token_latency") and self.token_latency:
            torch.xpu.synchronize()
            latency_list.append(time.time() - tic)
        if finished.all() or cur_len >= stopping_criteria.max_length:
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
        torch.xpu.empty_cache()

    IPEXLLMResourceContrainer.release_resources()
    out = torch.ops.torch_ipex.beam_search_finalize(
        candidate_num_beams,
        candidate_sequence_lengths,
        candidate_output_ids,
        candidate_score,
        candidate_normed_scores,
        output_beam_ids,
        output_token_ids,
        beam_scores,
        finished,
        length_penalty,
        max_in_seq_length,
        max_out_seq_length,
        batch_size,
        num_beams,
        beam_scorer.num_beam_hyps_to_keep,
        cur_len - max_in_seq_length,
        pad_token_id,
    )

    # origin_input_ids size is [batch_size * beam_size, seq_len]
    out = torch.ops.torch_ipex.update_output_sequence(origin_input_ids, out, batch_size)
    # IPEXTransformerAtten.release_all_static_cached_resources()
    reserved_mem = round(torch.xpu.memory_reserved() / 1024**3, 3)
    if reserved_mem > 50:
        torch.xpu.empty_cache()
    if hasattr(self, "token_latency") and self.token_latency:
        return out, latency_list
    return out


def ipex_beam_search(model):
    if hasattr(model, "beam_search"):
        setattr(model, "beam_search", partial(_ipex_beam_search, model))  # noqa


def ipex_beam_search_new(model):
    if hasattr(model, "_beam_search"):
        original_ptr = getattr(model, "_beam_search")  # noqa B009
        setattr(  # noqa B010
            model, "_beam_search", partial(_ipex_beam_search_, model, original_ptr)
        )  # noqa


def _ipex_beam_search_(
    self,
    original_funcion,
    input_ids,
    beam_scorer,
    logits_processor=None,
    stopping_criteria=None,
    generation_config=None,
    synced_gpus=False,
    logits_warper=None,
    **model_kwargs,
):
    r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
    ```"""
    from transformers.cache_utils import StaticCache

    if not isinstance(model_kwargs.get("past_key_values", None), StaticCache):
        return original_funcion(
            input_ids,
            beam_scorer,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            logits_warper,
            **model_kwargs,
        )
    from transformers.generation.utils import (
        _split_model_inputs,
        stack_model_outputs,
    )
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
    )

    latency_list = []
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    sequential = generation_config.low_memory
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
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

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    decoder_prompt_len = input_ids.shape[-1]

    # init parameters
    finished = torch.zeros([batch_size], dtype=torch.bool, device=input_ids.device)
    length_penalty = beam_scorer.length_penalty
    do_early_stopping = beam_scorer.do_early_stopping

    index_cache = torch.zeros(
        (0, batch_size * num_beams), dtype=torch.int, device=input_ids.device
    )
    max_in_seq_length = cur_len
    max_out_seq_length = stopping_criteria.max_length - max_in_seq_length
    output_token_ids = torch.empty(
        (max_out_seq_length, batch_size * num_beams),
        dtype=torch.long,
        device=input_ids.device,
    )
    output_beam_ids = torch.empty(
        (max_out_seq_length, batch_size * num_beams),
        dtype=torch.long,
        device=input_ids.device,
    )

    # init candidate pool
    candidate_num_beams = torch.zeros(
        (batch_size), dtype=torch.long, device=input_ids.device
    )
    candidate_score = torch.empty(
        (batch_size * 2 * num_beams), dtype=torch.float, device=input_ids.device
    )
    candidate_normed_scores = torch.empty(
        (batch_size * 2 * num_beams), dtype=torch.float, device=input_ids.device
    )
    candidate_min_normed_scores = torch.empty(
        (batch_size), dtype=torch.float, device=input_ids.device
    )
    candidate_output_ids = torch.empty(
        (batch_size * 2 * num_beams, max_out_seq_length),
        dtype=torch.long,
        device=input_ids.device,
    )
    candidate_sequence_lengths = torch.zeros(
        (batch_size * 2 * num_beams), dtype=torch.long, device=input_ids.device
    )
    origin_input_ids = input_ids
    IPEXTransformerAttn.reset_timestamp()
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        tic = time.time()

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        if sequential:
            if any(
                model_name in self.__class__.__name__.lower()
                for model_name in [
                    "fsmt",
                    "reformer",
                    "ctrl",
                    "gpt_bigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                    "jamba",
                ]
            ):
                raise RuntimeError(
                    f"Currently generation for {self.__class__.__name__} is not supported "
                    f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                )

            inputs_per_sub_batches = _split_model_inputs(
                model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
            )
            outputs_per_sub_batch = [
                self(**inputs_per_sub_batch, return_dict=True)
                for inputs_per_sub_batch in inputs_per_sub_batches
            ]

            outputs = stack_model_outputs(outputs_per_sub_batch)

        else:  # Unchanged original behavior
            input_ids_tmp = model_inputs["input_ids"].view([batch_size, num_beams, -1])
            if "position_ids" in model_inputs:
                position_ids_tmp = model_inputs["position_ids"].view(
                    [batch_size, num_beams, -1]
                )
            attention_mask_tmp = model_inputs["attention_mask"].view(
                [batch_size, num_beams, -1]
            )
            kv_cache = model_inputs.get("past_key_values", None)
            expand = False
            IPEXTransformerAttnNaive.update_beam_idx(index_cache)
            if input_ids_tmp.size(-1) > 1 and isinstance(kv_cache, IPEXStaticCache):
                model_inputs["input_ids"] = input_ids_tmp[:, 0, :]
                if "position_ids" in model_inputs:
                    model_inputs["position_ids"] = position_ids_tmp[:, 0, :]
                model_inputs["attention_mask"] = attention_mask_tmp[:, 0, :]
                expand = True
            outputs = self(**model_inputs, return_dict=True)
            if expand:
                outputs["logits"] = (
                    outputs["logits"]
                    .view([batch_size, 1, 1, -1])
                    .expand([batch_size, num_beams, 1, -1])
                    .reshape([batch_size * num_beams, 1, -1])
                )
                expand = False

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :].clone()
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        next_token_logits = None
        dummy_input_ids = torch.empty(
            (batch_size * num_beams, cur_len), dtype=torch.long, device="meta"
        )
        next_token_scores_processed = logits_processor(
            dummy_input_ids, next_token_scores
        )
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores)
        beam_scores, beam_next_tokens, beam_idx = torch.ops.torch_ipex.beam_search_topk(
            next_token_scores,
            finished,
            pad_token_id,
            eos_token_id,
            length_penalty,
            num_beams,
            batch_size,
            next_token_scores.shape[-1],
            cur_len - max_in_seq_length,
            do_early_stopping,
            max_in_seq_length,
            max_out_seq_length,
            output_token_ids,
            output_beam_ids,
            candidate_num_beams,
            candidate_normed_scores,
            candidate_min_normed_scores,
            candidate_output_ids,
            candidate_sequence_lengths,
            candidate_score,
        )
        index_cache = torch.ops.torch_ipex.update_beam_indices_for_cache(
            index_cache, beam_idx, num_beams, batch_size
        )

        input_ids = beam_next_tokens.unsqueeze(-1)
        torch.ops.torch_ipex.update_output_indices(
            beam_idx,
            beam_next_tokens,
            output_beam_ids,
            output_token_ids,
            finished,
            cur_len - max_in_seq_length,
            batch_size,
            num_beams,
        )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        outputs = None

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )
        # increase cur_len
        cur_len = cur_len + 1
        if hasattr(self, "token_latency") and self.token_latency:
            torch.xpu.synchronize()
            latency_list.append(time.time() - tic)

        # if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
        #     this_peer_finished = True
        if finished.all() or cur_len >= stopping_criteria.max_length:
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
        torch.xpu.empty_cache()

    IPEXLLMResourceContrainer.release_resources()
    out = torch.ops.torch_ipex.beam_search_finalize(
        candidate_num_beams,
        candidate_sequence_lengths,
        candidate_output_ids,
        candidate_score,
        candidate_normed_scores,
        output_beam_ids,
        output_token_ids,
        beam_scores,
        finished,
        length_penalty,
        max_in_seq_length,
        max_out_seq_length,
        batch_size,
        num_beams,
        beam_scorer.num_beam_hyps_to_keep,
        cur_len - max_in_seq_length,
        pad_token_id,
    )

    # origin_input_ids size is [batch_size * beam_size, seq_len]
    out = torch.ops.torch_ipex.update_output_sequence(origin_input_ids, out, batch_size)
    reserved_mem = round(torch.xpu.memory_reserved() / 1024**3, 3)
    if reserved_mem > 50:
        torch.xpu.empty_cache()
    if hasattr(self, "token_latency") and self.token_latency:
        return out, latency_list
    return out


def _ipex_beam_sample(
    self,
    input_ids,
    beam_scorer,
    logits_processor=None,
    stopping_criteria=None,
    logits_warper=None,
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
    Generates sequences of token ids for models with a language modeling head using **beam search multinomial
    sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.beam_sample`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
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
        [`~generation.BeamSampleDecoderOnlyOutput`], [`~generation.BeamSampleEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.BeamSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.BeamSampleEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
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
    ...     max_length=model.config.max_length,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )

    >>> outputs = model.beam_sample(
    ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    from transformers.generation.utils import (
        BeamSampleEncoderDecoderOutput,
        BeamSampleDecoderOnlyOutput,
    )
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        validate_stopping_criteria,
    )

    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
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

    index_cache = torch.zeros(
        (0, batch_size * num_beams), dtype=torch.int, device=input_ids.device
    )
    while True:
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

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        IPEXTransformerAttnNaive.update_beam_idx(index_cache)

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
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len
        )
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores)
        # Note: logits warpers are intentionally applied after adding running beam scores. On some logits warpers
        # (like top_p) this is indiferent, but on others (like temperature) it is not. For reference, see
        # https://github.com/huggingface/transformers/pull/5420#discussion_r449779867
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (logits_warper(input_ids, next_token_scores_processed),)
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
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        index_cache = torch.ops.torch_ipex.update_native_beam_indices_for_cache(
            index_cache, beam_idx, num_beams, batch_size
        )

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(
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
            return BeamSampleEncoderDecoderOutput(
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
            return BeamSampleDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]


def ipex_beam_sample(model):
    if hasattr(model, "beam_sample"):
        setattr(model, "beam_sample", partial(_ipex_beam_sample, model))  # noqa


def _ipex_beam_search_without_optimize(
    self,
    input_ids,
    beam_scorer,
    logits_processor=None,
    stopping_criteria=None,
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
    from transformers.generation.utils import (
        BeamSearchEncoderDecoderOutput,
        BeamSearchDecoderOnlyOutput,
    )
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        validate_stopping_criteria,
    )

    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn(
            "You don't have defined any stopping_criteria, this will likely loop forever",
            UserWarning,
        )
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

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

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

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    index_cache = torch.zeros(
        (0, batch_size * num_beams), dtype=torch.int, device=input_ids.device
    )
    while True:
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

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        IPEXTransformerAttnNaive.update_beam_idx(index_cache)

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
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len
        )
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores)

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

        index_cache = torch.ops.torch_ipex.update_beam_indices_for_cache(
            index_cache, beam_idx, num_beams, batch_size
        )

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        # if model_kwargs["past_key_values"] is not None:
        #     model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    self.release_resources()
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


def ipex_beam_search_without_optimize(model):
    if hasattr(model, "beam_search"):
        setattr(model, "beam_search", _ipex_beam_search_without_optimize)  # noqa
