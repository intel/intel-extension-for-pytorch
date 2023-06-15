import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .RoPE import GPTJRotaryEmbedding

def activation_replace(module):
    from transformers.activations import NewGELUActivation

    replace_dict = {
        NewGELUActivation: torch.nn.GELU(approximate="tanh")
    }
    for m in replace_dict.keys():
        if isinstance(module, m):
            return replace_dict[m]
    return module


class IPEXGPTJAttention(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.bias = module.bias
        self.max_position = self.bias.shape[-1]
        self.masked_bias = module.masked_bias

        self.attn_dropout = module.attn_dropout
        self.resid_dropout = module.resid_dropout

        self.embed_dim = module.embed_dim
        self.num_attention_heads = module.num_attention_heads
        self.head_dim = module.head_dim
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = module.scale_attn

        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.q_proj = module.q_proj
        self.out_proj = module.out_proj
        self.rotary_dim = module.rotary_dim
        self.embed_positions = module.embed_positions
        self.total_key = None
        self.total_value = None
        self.prompt_len = None

        self.device = module.k_proj.weight.device
        self.dtype = module.k_proj.weight.dtype
        self.rotary_emb = GPTJRotaryEmbedding(self.rotary_dim, device=self.device, dtype=self.dtype)

        self.prev_len = 0
        self.cur_len = 0
        self.key_cached = None
        self.value_cached = None
        self.n_positions = 2048
        self.seq_first = False

    def get_qkv_with_greedy(self, hidden_states, layer_past, position_ids):
        if layer_past is None:
            # the first timestep
            shape = [hidden_states.shape[0], self.num_attention_heads, self.n_positions, self.head_dim]
            self.key_cached = torch.empty(shape, device=self.device, dtype=self.dtype)
            self.value_cached = torch.empty(shape, device=self.device, dtype=self.dtype)
            self.prev_len = 0
                
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        new_shape = query.size()[:-1] + (self.num_attention_heads, self.head_dim)
        query = query.view(new_shape)
        key = key.view(new_shape)
        value = value.view(new_shape)
            
        query, key = self.rotary_emb(query, key, position_ids, self.rotary_dim)
        
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        self.cur_len = self.prev_len + hidden_states.size(1) 
        self.key_cached[:, :, self.prev_len : self.cur_len, :] = key 
        self.value_cached[:, :, self.prev_len : self.cur_len, :] = value
        key = self.key_cached[:, :, : self.cur_len, :]
        value = self.value_cached[:, :, : self.cur_len, :]
    
        self.prev_len = self.cur_len
        return query, key, value

    def get_qkv_with_greedy_seq_first(self, hidden_states, layer_past, position_ids):
        # greedy search path
        if layer_past is None:
            # the first timestep
            shape = [hidden_states.shape[0], self.n_positions, self.num_attention_heads, self.head_dim]
            self.key_cached = torch.empty(shape, device=self.device, dtype=self.dtype)
            self.value_cached = torch.empty(shape, device=self.device, dtype=self.dtype)
            self.prev_len = 0
        
        self.cur_len = self.prev_len + hidden_states.size(1) 
        key = self.key_cached[:, self.prev_len : self.cur_len, :, :]
        value = self.value_cached[:, self.prev_len : self.cur_len, :, :]
            
        query = self.q_proj(hidden_states)
        shape = query.size()[:-1] + (self.embed_dim,)
        torch.matmul(hidden_states, self.k_proj.weight.t(), out=key.view(shape))
        torch.matmul(hidden_states, self.v_proj.weight.t(), out=value.view(shape))
            
        new_shape = query.size()[:-1] + (self.num_attention_heads, self.head_dim)
        query = query.view(new_shape)
        key = key.view(new_shape)
        value = value.view(new_shape)
        query, key = self.rotary_emb(query, key, position_ids, self.rotary_dim)

        key = self.key_cached[:, : self.cur_len, :, :]
        value = self.value_cached[:, : self.cur_len, :, :]
    
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        self.prev_len = self.cur_len
        return query, key, value

    def get_qkv_with_beam(self, hidden_states, layer_past, position_ids):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        new_shape = query.size()[:-1] + (self.num_attention_heads, self.head_dim)
        query = query.view(new_shape)
        key = key.view(new_shape)
        value = value.view(new_shape)
        
        query, key = self.rotary_emb(query, key, position_ids, self.rotary_dim)
    
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        return query, key, value

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):

        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = torch.where(causal_mask, attn_weights, -65504.0)

        attn_weights = attn_weights / self.scale_attn
        # attention_mask = None
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)
        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        
        if hidden_states.shape[0] == 1:
            # greedy search path
            if self.seq_first:
                query, key, value = self.get_qkv_with_greedy_seq_first(hidden_states, layer_past, position_ids)
            else:
                query, key, value = self.get_qkv_with_greedy(hidden_states, layer_past, position_ids)
        else:
            query, key, value = self.get_qkv_with_beam(hidden_states, layer_past, position_ids)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class IPEXGPTJMLP(nn.Module):
    def __init__(self, module):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()

        self.fc_in = module.fc_in
        self.fc_out = module.fc_out

        self.act = activation_replace(module.act)
        self.dropout = module.dropout

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class IPEXGPTJBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.ln_1 = module.ln_1
        self.attn = IPEXGPTJAttention(module.attn)
        self.mlp = IPEXGPTJMLP(module.mlp)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)
