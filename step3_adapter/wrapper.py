"""Step 3 wrapper: the adapter. Adds each encoded frame's learned absolute position, 4 frames per call."""

import torch
from torch import nn
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingDecoder

from common.config import MEMORY_FRAMES_PER_CALL
from common.load_model import load_pytorch_model


class StreamingAdapter(nn.Module):
    """The first lines of MoonshineStreamingDecoder.forward, on 4 encoded frames per call instead of all M at once.

    The original adds rows 0 ..< M of the position table to the whole recording's encoded frames, then applies proj:
        encoder_hidden_states += self.pos_emb(torch.arange(M))
        encoder_hidden_states = self.proj(encoder_hidden_states)
    Streaming, frames arrive 4 at a time, so pos_offset says where these 4 sit in the segment: they get rows
    pos_offset ..< pos_offset + 4. Swift starts pos_offset at 0 on press and adds 4 after every call, so frame i
    always gets row i, as in the original.

    In tiny, proj is nn.Identity (encoder and decoder are both 320 wide), so this is a table lookup and an add.
    It is called anyway, so the wrapper stays the original's math.

    Nothing in the graph checks pos_offset: past 4092 the lookup reads outside the table, and past 1996 it reads
    untrained rows (common.config.ADAPTER_TRAINED_POSITION_COUNT). Swift's segment cap keeps it far below both.

    Inputs:  encoded [1,4,320] float32, pos_offset [1] int32
    Output:  memory [1,4,320] float32
    """

    def __init__(self, decoder: MoonshineStreamingDecoder):
        super().__init__()
        self.decoder = decoder  # only pos_emb and proj are used; conversion keeps only the weights the graph reads
        # [0, 1, 2, 3]: each frame's place within the call. int32 like pos_offset, so their sum stays int32.
        self.register_buffer(
            "frame_index_within_call", torch.arange(MEMORY_FRAMES_PER_CALL, dtype=torch.int32), persistent=False
        )

    def forward(self, encoded, pos_offset):
        positions = pos_offset + self.frame_index_within_call  # [4]: pos_offset, pos_offset + 1, ..., pos_offset + 3
        # A runtime index lookup (Core ML: gather), then an add. Out of place, unlike the original's +=, so the
        # caller's encoded tensor is left unchanged.
        return self.decoder.proj(encoded + self.decoder.pos_emb(positions))


def build_streaming_adapter() -> StreamingAdapter:
    """The wrapper around the real checkpoint's decoder, float32, in eval mode."""
    return StreamingAdapter(load_pytorch_model().model.decoder).eval()
