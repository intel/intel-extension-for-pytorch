import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

@torch.fx.wrap
def get_embed_positions(embed_positions, position_ids):
    return embed_positions.to(position_ids.device).repeat(position_ids.shape[0], 1, 1)

def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    tmp = rotate_every_two(tensor) * sin
    tensor *= cos
    tensor += tmp

def activation_replace(module):
    from transformers.activations import NewGELUActivation

    replace_dict = {
        NewGELUActivation: torch.ops.torch_ipex.new_gelu
    }
    for m in replace_dict.keys():
        if isinstance(module, m):
            return replace_dict[m]
    return module


class IPEXGPTJAttention(nn.Module):
    pos_embdding_dim = None
    seq_len = None
    rotary_embedding_dim = None
    sin_result = None
    cos_result = None
    embedding_offset = None
    sin_embedding = None
    cos_embedding = None
    position_ids = None

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


    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

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

    def _get_embed_positions(self, position_ids):
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        return embed_positions.repeat(position_ids.shape[0], 1, 1)

    def get_rotary_emb(self, position_ids, embed_positions):
        if IPEXGPTJAttention.position_ids is None or IPEXGPTJAttention.position_ids is not position_ids:
            repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
            sincos = torch.gather(embed_positions, 1, repeated_position_ids)
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
            IPEXGPTJAttention.sin_embedding = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
            IPEXGPTJAttention.cos_embedding = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
            IPEXGPTJAttention.position_ids = position_ids
        return IPEXGPTJAttention.sin_embedding, IPEXGPTJAttention.cos_embedding

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
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        new_shape = query.size()[:-1] + (self.num_attention_heads, self.head_dim)
        query = query.view(new_shape)
        key = key.view(new_shape)
        value = value.view(new_shape)


        embed_positions = self._get_embed_positions(position_ids)

        sin, cos = self.get_rotary_emb(position_ids, embed_positions)

        if self.rotary_dim is not None:

            apply_rotary_pos_emb(key[:, :, :, : self.rotary_dim], sin, cos)
            apply_rotary_pos_emb(query[:, :, :, : self.rotary_dim], sin, cos)
        else:

            apply_rotary_pos_emb(key, sin, cos)
            apply_rotary_pos_emb(query, sin, cos)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

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
