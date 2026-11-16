"""Step 2 wrapper: Moonshine's encoder layers on one fixed window of 116 feature frames, with the masks built in."""

import torch
from torch import nn
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingEncoder

from common.config import (
    ENCODER_HELD_BACK_FRAME_COUNT,
    ENCODER_LEFT_CONTEXT_FRAME_COUNT,
    ENCODER_WINDOW_FRAME_COUNT,
    MASK_BLOCKED_VALUE,
)
from common.load_model import load_pytorch_model


class StreamingEncoder(nn.Module):
    """MoonshineStreamingEncoder.forward after the embedder, on one window, with Moonshine's masks always applied.

    The original forward builds its sliding-window masks only when it is given an attention_mask; without one, every
    frame attends to every other frame (the mask trap). This wrapper always builds them: a query frame sees a key frame
    only if the key is inside that layer's sliding window AND valid (a real frame, not padding).

    The windows are read the way Moonshine reads them: config (16, 4) means 16 frames back and 4 ahead. Transformers
    5.22.2's sliding_window_mask_function sees one frame fewer each way (dist < 16, -dist < 4: 15 back, 3 ahead).
    Moonshine's reading is the one the weights were trained with: the paper ("w_right=4 implies an algorithmic
    lookahead of 4x20 ms = 80 ms"), the runtime (left context 16 x depth = 96, total_lookahead 16 = 4 x 4), and
    encoder.ort, which it matches as closely as encoder.ort matches itself (tools/diagnose_ort.py).
    Every other operation is the encoder's own layers and final norm, with their own weights.

    Inputs:  features [1,116,320] (real frames first, zeros after), valid [1,116] (1 real, 0 padding)
    Output:  encoded [1,116,320] (rows for padding frames are meaningless)
    """

    def __init__(self, encoder: MoonshineStreamingEncoder):
        super().__init__()
        self.encoder = encoder
        # (frames back, frames ahead) per layer: (16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)
        self.sliding_windows = [tuple(window) for window in encoder.config.sliding_windows]

        # How far one encoded frame reaches through all the layers. The window math must give every kept frame at
        # least this much real context on each side, or the window's output would differ from the whole recording's.
        self.frames_seen_back = sum(frames_back for frames_back, _ in self.sliding_windows)  # 96
        self.frames_seen_ahead = sum(frames_ahead for _, frames_ahead in self.sliding_windows)  # 16
        if self.frames_seen_back > ENCODER_LEFT_CONTEXT_FRAME_COUNT or self.frames_seen_ahead > ENCODER_HELD_BACK_FRAME_COUNT:
            raise ValueError(
                f"an encoded frame sees {self.frames_seen_back} frames back and {self.frames_seen_ahead} ahead, more than "
                f"the window math keeps ({ENCODER_LEFT_CONTEXT_FRAME_COUNT} and {ENCODER_HELD_BACK_FRAME_COUNT})"
            )

        # key_in_window_by_layer[layer, query, key] is 1 where that layer's sliding window lets the query frame see
        # the key frame. A constant: only the valid part of the mask changes from call to call.
        frame_indices = torch.arange(ENCODER_WINDOW_FRAME_COUNT)
        frames_key_is_behind_query = frame_indices[:, None] - frame_indices[None, :]  # negative: the key is ahead
        key_in_window_by_layer = [
            (frames_key_is_behind_query <= frames_back) & (frames_key_is_behind_query >= -frames_ahead)
            for frames_back, frames_ahead in self.sliding_windows
        ]
        self.register_buffer("key_in_window_by_layer", torch.stack(key_in_window_by_layer).float(), persistent=False)

    def forward(self, features, valid):
        return self.run_layers(features, valid)["encoded"]

    def run_layers(self, features, valid) -> dict[str, torch.Tensor]:
        """Every layer's output, by name and in order, then the final norm's. forward returns only encoded;
        tools/diagnose_float16.py uses the rest to find where float16 loses precision."""
        key_is_valid = valid[:, None, None, :]  # [1,1,1,116]: the same for every head and every query frame
        hidden_states = features
        layer_outputs = {}
        for layer_index, encoder_layer in enumerate(self.encoder.layers):
            # Additive mask [1,1,116,116]: 0 where allowed, -1e4 where blocked, so softmax gives blocked keys ~0 weight.
            attention_mask = (1.0 - self.key_in_window_by_layer[layer_index] * key_is_valid) * MASK_BLOCKED_VALUE
            hidden_states = encoder_layer(hidden_states, attention_mask=attention_mask)
            layer_outputs[f"layer{layer_index + 1}_output"] = hidden_states
        layer_outputs["encoded"] = self.encoder.final_norm(hidden_states)
        return layer_outputs


def build_streaming_encoder() -> StreamingEncoder:
    """The wrapper around the real checkpoint's encoder, float32, in eval mode."""
    return StreamingEncoder(load_pytorch_model().model.encoder).eval()
