# Below code is based on https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py.
from typing import Optional, Tuple
import torch

import transformers

from einops import rearrange
try:
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None
    print(
        "FlashAttention-2 is not installed correctly. If you want to use flash attention to inference, flash-attention >= 2.2 is needed. "
        "Please check the usage in https://github.com/Dao-AILab/flash-attention for more details."
    )


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
    )

    kv_seq_len = key_states.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[-2]
        kv_seq_len += past_kv_len

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    rotary_dim = cos.shape[-1]
    cos, sin = cos.squeeze(0,1)[:,:rotary_dim//2].contiguous(), sin.squeeze(0,1)[:,:rotary_dim//2].contiguous()

    if past_key_value is not None:
        key_cache = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
        value_cache = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
    else:
        key_cache = key_states
        value_cache = value_states

    assert not output_attentions, "output_attentions is not supported"

    q = query_states  # [bsz, q_len, nh, hd]
    k, v = key_states, value_states # [bsz, q_len, nh, hd]

    output = flash_attn_with_kvcache(
        q, key_cache, value_cache, k, v, rotary_cos=cos, rotary_sin=sin, cache_seqlens=past_kv_len, softmax_scale=None, causal=True, rotary_interleaved=False
    )
    output = rearrange(output, "b s h d -> b s (h d)", b=bsz)

    past_key_value = (key_cache[:,:kv_seq_len].transpose(1,2), value_cache[:,:kv_seq_len].transpose(1,2)) if use_cache else None

    output = self.o_proj(output)
    
    return output, None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    return attention_mask


def replace_llama_attn_with_flash_attn():
    if flash_attn_with_kvcache != None:
        print("USE_FLASH_ATTENTION: ", True)
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    else:
        print("USE_FLASH_ATTENTION: ", False)
