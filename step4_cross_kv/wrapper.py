"""Step 4 wrapper: cross K/V. Projects 4 new memory rows into the keys and values that every decoder layer's
cross-attention reads."""

import torch
from torch import nn
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingDecoder

from common.load_model import load_pytorch_model


class CrossAttentionKeyValueProjection(nn.Module):
    """The key and value projections of the 6 decoder layers' cross-attention (encoder_attn), on 4 memory rows per call.

    MoonshineStreamingAttention.forward computes them for cross-attention (lines 635-644 in Transformers 5.22.2):
        key_states = self.k_proj(current_states).view(bsz, -1, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(current_states).view(bsz, -1, num_key_value_heads, head_dim).transpose(1, 2)
    with current_states the whole memory [1, M, 320], once per transcript, then caches them. Each row's key and value
    depend only on that row: k_proj and v_proj are linear layers with no bias, applied row by row, and cross-attention
    adds no rotary positions (Step 3's adapter already put each row's position into the memory). So projecting the 4
    new rows of each adapter call and appending them along the frame axis builds the same cache.

    view and transpose change no numbers, only their grouping: each row's 320 become 8 heads of 40, and the head axis
    moves in front of the frame axis. head_dim_padding (0 in tiny) is applied later, inside attention, after the cache,
    so cached keys and values are always head_dim wide.

    Input:   memory [1,4,320] float32
    Outputs: k_new, v_new [6,1,8,4,40] float32: layer, batch, head, frame, head_dim, the order of cross_kv.ort's outputs
    """

    def __init__(self, decoder: MoonshineStreamingDecoder):
        super().__init__()
        # Each layer's own cross-attention module, for its k_proj, v_proj, head count and head size. Its q_proj and
        # o_proj are Step 5's; the traced graph never reads them, so conversion leaves their weights out.
        self.cross_attentions = nn.ModuleList(decoder_layer.encoder_attn for decoder_layer in decoder.layers)

    def forward(self, memory):
        keys_per_layer, values_per_layer = self.project_each_layer(memory)
        # A new axis in front, one entry per layer: [6,1,8,4,40]
        return torch.stack(keys_per_layer), torch.stack(values_per_layer)

    def project_each_layer(self, memory) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Each layer's keys and values for the given rows, [1,8,rows,40] each, in layer order."""
        keys_per_layer = []
        values_per_layer = []
        for cross_attention in self.cross_attentions:
            # Batch 1, written as a number. The original reads it with memory.shape[0], but a traced shape lookup
            # doesn't convert: coremltools 9.0 calls int() on a 1-element array, which NumPy 2 refuses.
            heads_shape = (1, -1, cross_attention.config.num_key_value_heads, cross_attention.head_dim)
            keys_per_layer.append(cross_attention.k_proj(memory).view(heads_shape).transpose(1, 2))
            values_per_layer.append(cross_attention.v_proj(memory).view(heads_shape).transpose(1, 2))
        return keys_per_layer, values_per_layer


def build_cross_attention_key_value_projection() -> CrossAttentionKeyValueProjection:
    """The wrapper around the real checkpoint's decoder, float32, in eval mode."""
    return CrossAttentionKeyValueProjection(load_pytorch_model().model.decoder).eval()
