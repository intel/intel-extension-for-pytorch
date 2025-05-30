import torch


def _model_forward(
    self,
    batch_size,
    num_beams,
    model_kwargs,
    input_ids,
    output_attentions,
    output_hidden_states,
):
    if "past_key_values" in model_kwargs and not isinstance(
        model_kwargs["past_key_values"], tuple
    ):
        model_kwargs["past_key_values"] = None
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
    # prepare variable output controls (note: some models won't accept all output controls)
    model_inputs.update(
        {"output_attentions": output_attentions} if output_attentions else {}
    )
    model_inputs.update(
        {"output_hidden_states": output_hidden_states} if output_hidden_states else {}
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
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
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
                            torch.zeros([1, 1, 1, 1]).contiguous().to(kv_cache_dtype),
                            torch.zeros([1, 1, 1, 1]).contiguous().to(kv_cache_dtype),
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
                                self.decoder.block[i].layer[1].EncDecAttention.n_heads,
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
                                self.decoder.block[i].layer[1].EncDecAttention.n_heads,
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
                num_head = self.git.encoder.layer[0].attention.self.num_attention_heads
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
                            torch.zeros([1, 1, 1, 1]).contiguous().to(kv_cache_dtype),
                            torch.zeros([1, 1, 1, 1]).contiguous().to(kv_cache_dtype),
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
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                torch.zeros([1, 1, 1, 1])
                                .contiguous()
                                .to(kv_cache_dtype),
                                beam_idx_tmp,
                            )
                            if i not in self.config.text_config.cross_attention_layers
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
                intermediate_size = self.config.mamba_expand * self.config.hidden_size
                conv_kernel_size = self.config.mamba_d_conv
                ssm_state_size = self.config.mamba_d_state
                dtype = (
                    self.config.dtype if hasattr(self.config, "dtype") else self.dtype
                )
                model_inputs["past_key_values"] = tuple(
                    [
                        (
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
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
                            torch.zeros([1, 1, 1, 1]).contiguous().to(kv_cache_dtype),
                            torch.zeros([1, 1, 1, 1]).contiguous().to(kv_cache_dtype),
                            beam_idx_tmp,
                        )
                        for i in range(num_hidden_layers)
                    ]
                )
            if self.model_backbone not in [
                "MllamaForConditionalGeneration",
                "JambaForCausalLM",
            ]:
                new_attention_mask = model_inputs["attention_mask"][:batch_size].clone()
                new_input_ids = model_inputs["input_ids"][:batch_size].clone()
                if has_position_id:
                    new_position_ids = model_inputs["position_ids"][:batch_size].clone()
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
        if not first_token and self.model_backbone == "Maira2ForConditionalGeneration":
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
    return outputs
