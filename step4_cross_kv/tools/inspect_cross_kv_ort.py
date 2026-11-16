"""What Moonshine's cross_kv.ort computes, read from the file; nothing runs, on Core ML or onnxruntime.
Run from the top of the repo, after step3_adapter.test (it writes ref/step3.npz):

    python -m step4_cross_kv.tools.inspect_cross_kv_ort

1. Its whole graph: stored tensors, and every node with its attributes. The Reshape and Transpose nodes show the order
   of its outputs' axes.
2. Its 12 stored weight matrices, each dequantized and compared with the checkpoint's k_proj or v_proj of the same
   layer. All within half a quantization step means the layer order is right, k and v aren't swapped, and nothing
   else (such as attention's 1 / sqrt(head_dim)) is folded into the weights; the norm ratio would show such a factor.
3. The 8-bit step its DynamicQuantizeLinear picks for the memory, from ONNX's formula: one step for the whole input
   tensor, (max(0, largest value) - min(0, smallest value)) / 255. For each test recording's whole memory, which is
   what Moonshine passes, and for 4-row slices, what one of our calls would pass. Each memory value is rounded by up
   to half a step before the weights multiply it, which is the main noise check (c) will see.
"""

import numpy as np

from common.config import MEMORY_FRAMES_PER_CALL, TEST_AUDIO_PATHS
from common.load_model import load_pytorch_model
from common.ort_graph import TENSOR_DATA_TYPE_NAMES, load_ort_graph, print_graph, stored_tensor_values
from step3_adapter.test import STEP3_REFERENCE_PATH

ORT_WEIGHT_NAME_FORMAT = "cross_attn_blocks.{layer_index}.attn.{projection_name}.weight"
PROJECTION_NAMES = ("k_proj", "v_proj")
UINT8_STEP_COUNT = 255  # DynamicQuantizeLinear always quantizes to uint8: 0 to 255


def compare_weights_with_checkpoint(initializers_by_name: dict) -> None:
    print("[2. cross_kv.ort's weights vs the checkpoint's]")
    decoder = load_pytorch_model().model.decoder
    print(
        "  ONNX MatMul multiplies memory [M,320] by a stored [in,out] matrix, so it is compared with the checkpoint's "
        "nn.Linear weight [out,in] transposed.\n"
        "  error in half steps: worst |dequantized - checkpoint| divided by half that value's quantization step; "
        "<= 1 is exact up to 8-bit rounding"
    )
    for layer_index, decoder_layer in enumerate(decoder.layers):
        for projection_name in PROJECTION_NAMES:
            weight_name = ORT_WEIGHT_NAME_FORMAT.format(layer_index=layer_index, projection_name=projection_name)
            quantized_tensor = initializers_by_name[f"{weight_name}_quantized"]
            quantized_weight = stored_tensor_values(quantized_tensor)
            scale = stored_tensor_values(initializers_by_name[f"{weight_name}_scale"])
            zero_point = stored_tensor_values(initializers_by_name[f"{weight_name}_zero_point"])
            # One scale for the matrix (shape [] or [1]), or one per output column ([320]): broadcast on the last axis.
            dequantized_weight = (quantized_weight.astype(np.float32) - zero_point.astype(np.float32)) * scale

            checkpoint_weight = getattr(decoder_layer.encoder_attn, projection_name).weight.detach().numpy().T
            if dequantized_weight.shape != checkpoint_weight.shape:
                raise ValueError(
                    f"{weight_name}: stored {dequantized_weight.shape}, checkpoint {checkpoint_weight.shape}"
                )
            absolute_error = np.abs(dequantized_weight - checkpoint_weight)
            error_in_half_steps = float((absolute_error / np.broadcast_to(scale / 2, absolute_error.shape)).max())
            norm_ratio = np.linalg.norm(dequantized_weight) / np.linalg.norm(checkpoint_weight)
            if scale.size == 1:
                scale_text = f"one scale {scale.item():.4g}"
            else:
                scale_text = f"{scale.size} scales {scale.min():.3g} to {scale.max():.3g}"
            verdict = "matches" if error_in_half_steps <= 1 + 1e-3 else "DIFFERS"
            print(
                f"  layer {layer_index} {projection_name}: "
                f"{TENSOR_DATA_TYPE_NAMES[quantized_tensor.DataType()]} {list(quantized_weight.shape)}, {scale_text}, "
                f"zero points {np.unique(zero_point).tolist()[:3]}; max abs {absolute_error.max():.3g}, "
                f"error in half steps {error_in_half_steps:.4f}, norm ratio {norm_ratio:.5f}: {verdict}"
            )


def dynamic_quantization_step(values: np.ndarray) -> float:
    """The step ONNX's DynamicQuantizeLinear uses for this whole tensor."""
    return float(max(values.max(), 0.0) - min(values.min(), 0.0)) / UINT8_STEP_COUNT


def print_memory_quantization_steps() -> None:
    print("[3. the 8-bit step cross_kv.ort's DynamicQuantizeLinear picks for the memory]")
    if not STEP3_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"{STEP3_REFERENCE_PATH} (run python -m step3_adapter.test first)")
    step3_reference = np.load(STEP3_REFERENCE_PATH)
    for recording_name in TEST_AUDIO_PATHS:
        memory = step3_reference[f"{recording_name}_memory"]
        whole_memory_step = dynamic_quantization_step(memory)
        call_steps = np.array([
            dynamic_quantization_step(memory[:, first_row : first_row + MEMORY_FRAMES_PER_CALL])
            for first_row in range(0, memory.shape[1], MEMORY_FRAMES_PER_CALL)
        ])
        root_mean_square = float(np.sqrt(np.mean(memory.astype(np.float64) ** 2)))
        print(
            f"  {recording_name}: memory {memory.shape}, values {memory.min():.2f} to {memory.max():.2f}, "
            f"root mean square {root_mean_square:.3f}\n"
            f"    whole memory (as Moonshine passes it): step {whole_memory_step:.4f}, "
            f"{whole_memory_step / root_mean_square:.3f} of the root mean square\n"
            f"    {len(call_steps)} slices of {MEMORY_FRAMES_PER_CALL} rows: step median {np.median(call_steps):.4f}, "
            f"{call_steps.min():.4f} to {call_steps.max():.4f}"
        )


def main() -> None:
    print("[1. cross_kv.ort's graph]")
    initializers_by_name = print_graph(load_ort_graph("cross_kv.ort"))
    compare_weights_with_checkpoint(initializers_by_name)
    print_memory_quantization_steps()


if __name__ == "__main__":
    main()
