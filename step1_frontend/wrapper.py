"""Step 1 wrapper: Moonshine's frontend, one 80 ms chunk per call, with the conv history carried in and out."""

import torch
from torch import nn
from torch.nn import functional

from common.config import (
    AUDIO_SAMPLES_PER_CHUNK,
    FEATURE_FRAMES_PER_CHUNK,
    FRONTEND_CONV1_BUFFER_SHAPE,
    FRONTEND_CONV2_BUFFER_SHAPE,
)
from common.load_model import load_pytorch_model


class StreamingFrontend(nn.Module):
    """MoonshineStreamingEncoderEmbedder.forward (with padding_mask=None), one chunk at a time.

    The embedder's causal convs pad 4 zeros on the left of every call. Here those 4 positions come in
    as conv1_buffer / conv2_buffer (the last 4 conv inputs of the previous chunk), and the new last 4 go
    out for Swift to pass to the next call. Zeros at the start of a recording equal the padding.
    Every other operation is the embedder's own module, with its own weights.

    Inputs:  audio_chunk [1,1280], conv1_buffer [1,320,4], conv2_buffer [1,640,4]
    Outputs: features [1,4,320], conv1_buffer_out [1,320,4], conv2_buffer_out [1,640,4]
    """

    def __init__(self, embedder: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.frames_per_chunk = AUDIO_SAMPLES_PER_CHUNK // embedder.frame_len  # 16

        for conv, planned_buffer_shape in (
            (embedder.conv1, FRONTEND_CONV1_BUFFER_SHAPE),
            (embedder.conv2, FRONTEND_CONV2_BUFFER_SHAPE),
        ):
            module_buffer_shape = (1, conv.in_channels, conv.left_pad)
            if module_buffer_shape != planned_buffer_shape:
                raise ValueError(f"conv buffer {module_buffer_shape} != planned {planned_buffer_shape}")

        # The stride only lines up chunk to chunk if every chunk's frames divide evenly through both convs.
        total_conv_stride = embedder.conv1.stride[0] * embedder.conv2.stride[0]  # 4
        if (
            self.frames_per_chunk * embedder.frame_len != AUDIO_SAMPLES_PER_CHUNK
            or self.frames_per_chunk != FEATURE_FRAMES_PER_CHUNK * total_conv_stride
        ):
            raise ValueError(
                f"{AUDIO_SAMPLES_PER_CHUNK} samples per chunk does not give {FEATURE_FRAMES_PER_CHUNK} features "
                f"(frame_len {embedder.frame_len}, total conv stride {total_conv_stride})"
            )

    def forward(self, audio_chunk, conv1_buffer, conv2_buffer):
        stage_outputs = self.run_stages(audio_chunk, conv1_buffer, conv2_buffer)
        return stage_outputs["features"], stage_outputs["conv1_buffer_out"], stage_outputs["conv2_buffer_out"]

    def run_stages(self, audio_chunk, conv1_buffer, conv2_buffer) -> dict[str, torch.Tensor]:
        """Every intermediate result, by name and in order. forward returns only the three Swift needs;
        tools/diagnose_float16.py uses the rest to find where float16 loses precision."""
        embedder = self.embedder
        frames = audio_chunk.reshape(1, self.frames_per_chunk, embedder.frame_len)  # [1,16,80]
        cmvn_output = embedder.cmvn(frames)
        asinh_output = embedder.comp(cmvn_output)
        linear_output = embedder.linear(asinh_output)  # [1,16,320]
        linear_silu_output = functional.silu(linear_output).transpose(1, 2)  # [1,320,16]: conv1d wants channels, then time

        conv1_input = torch.cat([conv1_buffer, linear_silu_output], dim=2)  # [1,320,20]
        conv1_output = convolve_without_padding(embedder.conv1, conv1_input)  # [1,640,8]
        conv1_silu_output = functional.silu(conv1_output)

        conv2_input = torch.cat([conv2_buffer, conv1_silu_output], dim=2)  # [1,640,12]
        features = convolve_without_padding(embedder.conv2, conv2_input).transpose(1, 2)  # [1,4,320]
        return {
            "cmvn_output": cmvn_output,
            "asinh_output": asinh_output,
            "linear_output": linear_output,
            "linear_silu_output": linear_silu_output,
            "conv1_output": conv1_output,
            "conv1_silu_output": conv1_silu_output,
            "features": features,
            "conv1_buffer_out": conv1_input[:, :, -embedder.conv1.left_pad :],
            "conv2_buffer_out": conv2_input[:, :, -embedder.conv2.left_pad :],
        }


def convolve_without_padding(conv: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
    """The causal conv's own weights, stride and dilation, minus the zero padding its forward adds."""
    return functional.conv1d(values, conv.weight, conv.bias, stride=conv.stride, dilation=conv.dilation)


def build_streaming_frontend() -> StreamingFrontend:
    """The wrapper around the real checkpoint's embedder, float32, in eval mode."""
    return StreamingFrontend(load_pytorch_model().model.encoder.embedder).eval()
