"""Step 5 wrappers: one decoder step. One token in, a score for each of the 32,768 possible next tokens out, with the
self-attention cache and the cross-attention keys and values in fixed-size buffers. Part 5a passes the self-attention
cache in and out; part 5b keeps it inside the model as Core ML state."""

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional
from transformers import MoonshineStreamingForConditionalGeneration
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    MoonshineStreamingAttention,
    MoonshineStreamingDecoder,
    eager_attention_forward,
    rotate_half,
)

from common.config import (
    DECODER_HEAD_DIM,
    DECODER_PADDED_HEAD_DIM,
    MASK_BLOCKED_VALUE,
    MAXIMUM_TOKEN_COUNT,
    attention_cache_shape,
)
from common.load_model import load_pytorch_model


class DecoderStepWithExplicitCache(nn.Module):
    """MoonshineStreamingDecoder.forward and proj_out for one token, with caches passed in and out (plan part 5a).

    The original decoder.forward (lines 791-867 in Transformers 5.22.2) first turns the encoder output into memory
    (pos_emb and proj, lines 813-817: Step 3's adapter, already done), then embeds the tokens, runs the 6 layers and the
    final norm; MoonshineStreamingForConditionalGeneration applies proj_out. Its self-attention cache grows by one row
    per token, and each layer's cross-attention projects the memory into keys and values on the first token.

    This wrapper skips forward and calls the same modules directly, for one token at a time:
      - Self-attention keeps a cache of Lmax = 128 rows. The new token's key and value are written at row `position`,
        and a mask blocks the rows after it, which hold no token yet.
      - Cross-attention reads Step 4's k_cross and v_cross buffers instead of projecting the memory. cross_valid marks
        the frames that hold memory; the rest are blocked.
      - Rotary positions come from a table built once from the model's own rotary_emb (build_rotary_tables).
    Attention is written out with fixed numbers instead of calling MoonshineStreamingAttention.forward, which reads
    tensor shapes (bsz, q_len = hidden_states.shape[:-1]); a traced shape read doesn't convert (Step 4, deviation 1).
    It still uses that module's own weights, scaling, and Transformers' eager_attention_forward and rotate_half.

    proj_out is the embedding table: the checkpoint stores both, bit-identical, so the wrapper uses one (checked here).

    Nothing in the graph checks position. Swift keeps it below Lmax with its token cap: min(ceil(seconds x 6.5), Lmax)
    steps, at positions 0 ..< that.

    Inputs:  token [1,1] int32, position [1] int32 (0 for the start token),
             k_self, v_self [6,1,8,128,40] float32 (the previous step's k_self_out, v_self_out; zeros for position 0),
             k_cross, v_cross [6,1,8,768,40] (Step 4's buffers; float16 in the watch package),
             cross_valid [1,768] float32 (1 for frames that hold memory, 0 for the rest)
    Outputs: logits [1,1,32768] float32,
             k_self_out, v_self_out [6,1,8,128,40] float32: k_self and v_self with row `position` replaced
    """

    def __init__(
        self, model: MoonshineStreamingForConditionalGeneration, maximum_token_count: int = MAXIMUM_TOKEN_COUNT
    ):
        super().__init__()
        decoder = model.model.decoder
        if not torch.equal(model.proj_out.weight, decoder.embed_tokens.weight):
            raise ValueError("proj_out.weight differs from decoder.embed_tokens.weight: the wrapper can't share them")
        for layer_index, decoder_layer in enumerate(decoder.layers):
            for attention in (decoder_layer.self_attn, decoder_layer.encoder_attn):
                if attention.head_dim_padding != 0:  # the wrapper's attention leaves padding out
                    raise ValueError(f"layer {layer_index}: head_dim_padding {attention.head_dim_padding}, expected 0")

        # Only these are used; conversion keeps only the weights the traced graph reads.
        self.embed_tokens = decoder.embed_tokens  # [32768, 320]: token -> 320 numbers, and (as proj_out) back
        self.layers = decoder.layers
        self.norm = decoder.norm

        rotary_cos_by_position, rotary_sin_by_position = build_rotary_tables(decoder, maximum_token_count)
        self.register_buffer("rotary_cos_by_position", rotary_cos_by_position, persistent=False)  # [Lmax, 40]
        self.register_buffer("rotary_sin_by_position", rotary_sin_by_position, persistent=False)
        # [0, 1, ..., Lmax - 1]: each cache row's token position. int32 like position, compared with it.
        self.register_buffer(
            "token_row_positions", torch.arange(maximum_token_count, dtype=torch.int32), persistent=False
        )

    def forward(self, token, position, k_self, v_self, k_cross, v_cross, cross_valid):
        layer_cache_with_new_row = blend_new_row_into(k_self, v_self, *self.row_weights(position))
        step = self.run_step(token, position, k_cross, v_cross, cross_valid, layer_cache_with_new_row)
        return step.logits, step.k_self_out, step.v_self_out

    def row_weights(self, position) -> tuple[torch.Tensor, torch.Tensor]:
        """(is_new_row, is_kept_row), [Lmax, 1] each: 1 and 0 at row `position`, where this token's key and value go,
        and 0 and 1 at every other row. cache * is_kept_row + new_row * is_new_row writes the new row in."""
        is_new_row = (self.token_row_positions == position).float().view(-1, 1)
        return is_new_row, 1.0 - is_new_row

    def run_step(
        self, token, position, k_cross, v_cross, cross_valid, layer_cache_with_new_row: "LayerCacheWithNewRow"
    ) -> "DecoderStepResult":
        """The step itself, shared by 5a (caches in and out) and 5b (caches in MLState). layer_cache_with_new_row
        says how each layer's self-attention gets its cache, this token's row included."""
        # Additive masks, 0 where allowed and -1e4 where blocked, the same for every head: [1,1,1,keys]
        # Self-attention: rows after position hold no token yet (zeros, or tokens from an earlier transcript update).
        self_attention_mask = ((self.token_row_positions > position).float() * MASK_BLOCKED_VALUE).view(1, 1, 1, -1)
        # Cross-attention: buffer frames past the memory's end. Their keys are zeros, which score 0: not blocked.
        cross_attention_mask = ((1.0 - cross_valid) * MASK_BLOCKED_VALUE).view(1, 1, 1, -1)
        # Row `position` of each table, [1,40]: a runtime lookup (Core ML: gather), like Step 3's pos_emb.
        rotary_cos = functional.embedding(position, self.rotary_cos_by_position)
        rotary_sin = functional.embedding(position, self.rotary_sin_by_position)

        hidden_states = self.embed_tokens(token)  # [1,1,320]
        new_keys = []
        new_values = []
        k_self_out = []
        v_self_out = []
        for layer_index, decoder_layer in enumerate(self.layers):
            # MoonshineStreamingDecoderLayer.forward (lines 711-755): three blocks, each adding to hidden_states what
            # it computes from a normed copy.

            # 1. Self-attention: look back at this token and every earlier one.
            self_attention = decoder_layer.self_attn
            normed = decoder_layer.input_layernorm(hidden_states)
            query = rotate(split_heads(self_attention, self_attention.q_proj(normed)), rotary_cos, rotary_sin)
            key = rotate(split_heads(self_attention, self_attention.k_proj(normed)), rotary_cos, rotary_sin)
            value = split_heads(self_attention, self_attention.v_proj(normed))
            new_keys.append(key)
            new_values.append(value)
            # This layer's cache with row `position` holding this token's key and value, [1,8,Lmax,40]. Keys are
            # cached rotated, as the original does (rotation at line 650, cache update at 653).
            layer_keys, layer_values = layer_cache_with_new_row(layer_index, key, value)
            k_self_out.append(layer_keys)
            v_self_out.append(layer_values)
            hidden_states = hidden_states + attend(
                self_attention, query, layer_keys, layer_values, self_attention_mask
            )

            # 2. Cross-attention: look at the audio. Keys and values are Step 4's, so only the query is computed here.
            # No rotation: Step 3's adapter already put each memory row's position into it. Each row's first 40
            # numbers, in one slice per buffer: 5b's buffers are padded to 64 (5a's are 40, so all of each row).
            cross_attention = decoder_layer.encoder_attn
            normed = decoder_layer.post_attention_layernorm(hidden_states)
            query = split_heads(cross_attention, cross_attention.q_proj(normed))
            layer_cross_keys = k_cross[layer_index, :, :, :, :DECODER_HEAD_DIM]  # [1,8,768,40]
            layer_cross_values = v_cross[layer_index, :, :, :, :DECODER_HEAD_DIM]
            hidden_states = hidden_states + attend(
                cross_attention, query, layer_cross_keys, layer_cross_values, cross_attention_mask
            )

            # 3. MLP: fc1 to 2 x 1280, silu(second half) x first half, fc2 back to 320.
            hidden_states = hidden_states + decoder_layer.mlp(decoder_layer.final_layernorm(hidden_states))

        # proj_out (MoonshineStreamingForConditionalGeneration, line 1084), with the embedding table as its weight.
        logits = functional.linear(self.norm(hidden_states), self.embed_tokens.weight)  # [1,1,32768]
        return DecoderStepResult(
            logits=logits,
            new_keys=torch.stack(new_keys),
            new_values=torch.stack(new_values),
            k_self_out=torch.stack(k_self_out),
            v_self_out=torch.stack(v_self_out),
        )


class DecoderStepResult(NamedTuple):
    logits: torch.Tensor  # [1,1,32768]: a score for every possible next token
    new_keys: torch.Tensor  # [6,1,8,1,40]: this token's key in every layer (rotated), what goes into row position
    new_values: torch.Tensor  # [6,1,8,1,40]: and its value
    k_self_out: torch.Tensor  # [6,1,8,Lmax,40]: each layer's cache as attention read it, row position included
    v_self_out: torch.Tensor  # [6,1,8,Lmax,40]


# How one layer's self-attention gets its cache for this step: (layer_index, this token's key [1,8,1,40], its value)
# -> that layer's keys and values with this token's row at row position, [1,8,Lmax,40] each.
LayerCacheWithNewRow = Callable[[int, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def blend_new_row_into(
    k_self: torch.Tensor, v_self: torch.Tensor, is_new_row: torch.Tensor, is_kept_row: torch.Tensor
) -> LayerCacheWithNewRow:
    """The one-hot blend: each layer's cache from k_self and v_self [6,1,8,Lmax,40], with row position replaced,
    cache * is_kept_row + new_row * is_new_row (DecoderStepWithExplicitCache.row_weights). Plain math, no write."""

    def layer_cache_with_new_row(layer_index, key, value):
        return (
            k_self[layer_index] * is_kept_row + key * is_new_row,
            v_self[layer_index] * is_kept_row + value * is_new_row,
        )

    return layer_cache_with_new_row


class DecoderStepWithStatefulCache(nn.Module):
    """The same decoder step with the self-attention cache inside the model (plan part 5b): k_self and v_self are
    buffers here, and conversion declares them as Core ML state (MLState). Swift makes the state once per segment and
    passes it to every prediction; each layer writes this token's row into it, then reads its cache back from it.

    How the state is laid out and written decides whether it runs on the Neural Engine
    (tools/diagnose_state_on_neural_engine.py, on the Mac):
      - Rows are written with a slice assignment at the runtime position (Core ML: slice_update), as ANEMLL's stateful
        caches are. 5a's one-hot blend, as in-place mul_ then add_ on the whole state, doesn't compile for the Neural
        Engine (ANECCompile() FAILED), whatever the state's layout.
      - The state's last axis is 64 (common.config.DECODER_PADDED_HEAD_DIM): head_dim 40, padded with zeros to a
        multiple of 32. With 40, the model compiles for the Neural Engine, but every prediction fails ("ANE inference
        operation failed ... Program Inference error"). The padding is written as zeros and never read.
      - k_cross and v_cross are padded to 64 the same way: with 40, Core ML re-lays them out on every call
        (common.config.DECODER_PADDED_HEAD_DIM). run_step reads each row's first 40 numbers, so whatever the padding
        holds is never read.
      - Writing each layer's row, then reading its cache back, beat writing all 6 layers after the last one (6.1 vs
        7.5 ms per step on the Mac): every state read and write runs on the Neural Engine.
    The slice writes name every axis: with "...", coremltools 9.0 puts the slice on the first axis. torch.jit.trace
    keeps position a runtime value in them (aten::Int of the input, then aten::slice). And no shape is read
    (Step 4, deviation 1): the padding is written as numbers.

    The buffers are float32 here, so this wrapper computes exactly what 5a's does; Core ML stores state as float16.

    Inputs: token [1,1] int32, position [1] int32, k_cross, v_cross [6,1,8,768,64], cross_valid [1,768] (as in 5a)
    State:  k_self, v_self [6,1,8,128,64]
    Output: logits [1,1,32768] float32
    """

    def __init__(self, decoder_step: DecoderStepWithExplicitCache):
        super().__init__()
        self.decoder_step = decoder_step
        state_shape = attention_cache_shape(decoder_step.token_row_positions.shape[0], DECODER_PADDED_HEAD_DIM)
        # Top-level buffers: conversion matches each StateType's name with named_buffers().
        self.register_buffer("k_self", torch.zeros(state_shape))
        self.register_buffer("v_self", torch.zeros(state_shape))

    def forward(self, token, position, k_cross, v_cross, cross_valid):
        def write_then_read(layer_index, key, value):
            return (
                write_row_then_read_layer(self.k_self, layer_index, key, position),
                write_row_then_read_layer(self.v_self, layer_index, value, position),
            )

        return self.decoder_step.run_step(token, position, k_cross, v_cross, cross_valid, write_then_read).logits


def write_row_then_read_layer(state: torch.Tensor, layer_index: int, row: torch.Tensor, position) -> torch.Tensor:
    """Writes one layer's new row [1,8,1,40], zero-padded to the state's width, into state [6,1,8,Lmax,64] at row
    `position`; returns that layer's cache read back, [1,8,Lmax,40], the new row included."""
    padded_row = functional.pad(row[None], (0, DECODER_PADDED_HEAD_DIM - DECODER_HEAD_DIM))  # [1,1,8,1,64]
    state[layer_index:layer_index + 1, :, :, position:position + 1, :] = padded_row
    return state[layer_index:layer_index + 1][..., :DECODER_HEAD_DIM][0]


def split_heads(attention: MoonshineStreamingAttention, projected: torch.Tensor) -> torch.Tensor:
    """[1,1,320] -> [1,8,1,40]: one token's 320 numbers as 8 heads of 40, head axis first, as forward does (lines
    615-617). Batch 1 and 1 token are written as numbers; the original reads them from the input's shape."""
    return projected.view(1, 1, attention.config.num_key_value_heads, attention.head_dim).transpose(1, 2)


def rotate(heads: torch.Tensor, rotary_cos: torch.Tensor, rotary_sin: torch.Tensor) -> torch.Tensor:
    """apply_rotary_pos_emb (lines 524-561) for one position: each head's neighbouring pairs (0,1), (2,3), ... rotated
    by that position's angles. With build_rotary_tables' layout this is one line for all 40 numbers."""
    return heads * rotary_cos + rotate_half(heads) * rotary_sin


def attend(
    attention: MoonshineStreamingAttention,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Transformers' eager attention (query.key x scaling, + mask, softmax, weighted sum of values), then o_proj, as
    forward does (lines 666-682). query [1,8,1,40], keys and values [1,8,rows,40] -> [1,1,320]."""
    attention_output, _ = eager_attention_forward(
        attention, query, keys, values, attention_mask, scaling=attention.scaling
    )  # [1,1,8,40]
    return attention.o_proj(attention_output.reshape(1, 1, -1))


def build_rotary_tables(decoder: MoonshineStreamingDecoder, maximum_token_count: int) -> tuple[torch.Tensor, ...]:
    """cos and sin for every position 0 ..< maximum_token_count, [positions, 40] each, in float32, laid out so that
    rotate() gives what apply_rotary_pos_emb gives.

    rotary_emb returns [1, positions, 32]: 16 frequencies, written twice (torch.cat((freqs, freqs)), line 510).
    apply_rotary_pos_emb keeps the first 16 and repeats each in place (repeat_interleave), so neighbours (0,1), (2,3),
    ... share a frequency, matching rotate_half's pairs. It rotates only those 32 numbers (partial_rotary_factor 0.8)
    and passes each head's last 8 through. Here those 8 get cos 1 and sin 0: x * 1 + anything * 0, unchanged.

    A table instead of computing position x frequency in the model: on the watch that product would be float16. At
    position 127 an angle like 71.4 rounds to float16's step there (0.0625), and the frequency is rounded too. The
    table rounds only the final cos and sin values."""
    positions = torch.arange(maximum_token_count)[None]  # [1, positions], as decoder.forward passes position_ids
    rotary_cos, rotary_sin = decoder.rotary_emb(torch.zeros(1), position_ids=positions)  # x only sets dtype and device
    rotated_count = rotary_cos.shape[-1]  # 32
    if rotated_count > DECODER_HEAD_DIM:
        raise ValueError(f"rotary_emb rotates {rotated_count} numbers per head, more than head_dim {DECODER_HEAD_DIM}")
    frequency_count = rotated_count // 2  # 16
    passed_through_count = DECODER_HEAD_DIM - rotated_count  # 8

    def head_layout(values_by_position: torch.Tensor, passed_through_value: float) -> torch.Tensor:
        rotated = values_by_position[0, :, :frequency_count].repeat_interleave(2, dim=-1)  # [positions, 32]
        passed_through = torch.full((maximum_token_count, passed_through_count), passed_through_value)
        return torch.cat([rotated, passed_through], dim=-1)  # [positions, 40]

    return head_layout(rotary_cos, 1.0), head_layout(rotary_sin, 0.0)


def build_decoder_step_with_explicit_cache() -> DecoderStepWithExplicitCache:
    """The 5a wrapper around the real checkpoint, float32, in eval mode."""
    return DecoderStepWithExplicitCache(load_pytorch_model()).eval()


def build_decoder_step_with_stateful_cache() -> DecoderStepWithStatefulCache:
    """The 5b wrapper around the real checkpoint, float32, in eval mode, with an all-zero cache."""
    return DecoderStepWithStatefulCache(build_decoder_step_with_explicit_cache()).eval()
