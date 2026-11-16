"""Step 4 checks for cross K/V. Run from conversion/, after step3_adapter.test (it writes ref/step3.npz) and
step4_cross_kv.convert:

    python -m step4_cross_kv.test

(a) the wrapper call by call, appended, vs Transformers' own cross-attention cache for the whole recording,
(b) Core ML vs wrapper, (c) Moonshine's cross_kv.ort on the whole memory vs the wrapper's appended buffer.
Exits non-zero if any check fails; writes ref/step4.npz only when all pass.
"""

import sys
from collections.abc import Callable

import coremltools
import numpy as np
import onnxruntime
import torch
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingDecoder

from common.compare import (
    COREML_FLOAT16_MIN_COSINE_SIMILARITY,
    COREML_FLOAT32_MAX_ABS_DIFF,
    MOONSHINE_ORT_MIN_COSINE_SIMILARITY,
    WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
    compare,
    compare_with_worst_frame,
)
from common.config import (
    DECODER_DIM,
    DECODER_HEAD_COUNT,
    DECODER_HEAD_DIM,
    DECODER_LAYER_COUNT,
    MAXIMUM_MEMORY_FRAME_COUNT,
    MEMORY_FRAMES_PER_CALL,
    REFERENCE_TENSORS_DIRECTORY,
    TEST_AUDIO_PATHS,
    attention_cache_shape,
    load_streaming_config,
)
from common.load_model import load_ort_session, load_pytorch_model
from step2_encoder.test import STEP2_REFERENCE_PATH
from step3_adapter.test import STEP3_REFERENCE_PATH
from step4_cross_kv.convert import CROSS_KV_FLOAT32_MODEL_PATH, CROSS_KV_MODEL_PATH
from step4_cross_kv.wrapper import CrossAttentionKeyValueProjection

STEP4_REFERENCE_PATH = REFERENCE_TENSORS_DIRECTORY / "step4.npz"
FRAME_AXIS = 3  # cross K/V are [layer, batch, head, frame, head_dim]
KEY_AND_VALUE_NAMES = ("k_cross", "v_cross")

# Runs cross K/V once: memory [1,4,320] float32 -> (k_new, v_new), each [6,1,8,4,40] float32.
# One per implementation: the wrapper, Core ML, and cross_kv.ort.
CallRunner = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def append_cross_kv(
    run_call: CallRunner, memory: np.ndarray, capacity_frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Swift's bookkeeping for one segment's memory rows [1, M, 320], as Step 3 hands them over, 4 at a time.

    Two buffers preallocated to capacity_frame_count frames (Mmax in the app) and filled with zeros, and a count of
    valid frames. Each call projects the next 4 rows and writes them on the frame axis at valid_frame_count ..<
    valid_frame_count + 4, then adds 4. Returns the valid part of both buffers, [6,1,8,M,40]: what the decoder (Step 5)
    reads, with cross_valid marking the same frames."""
    frame_count = memory.shape[1]
    if frame_count % MEMORY_FRAMES_PER_CALL:
        raise ValueError(f"{frame_count} memory rows is not a whole number of {MEMORY_FRAMES_PER_CALL}-row calls")
    if frame_count > capacity_frame_count:
        raise ValueError(f"{frame_count} memory rows don't fit in buffers of {capacity_frame_count} frames")
    k_cross = np.zeros(attention_cache_shape(capacity_frame_count), dtype=np.float32)
    v_cross = np.zeros_like(k_cross)
    valid_frame_count = 0
    for first_row in range(0, frame_count, MEMORY_FRAMES_PER_CALL):
        new_frames = slice(valid_frame_count, valid_frame_count + MEMORY_FRAMES_PER_CALL)
        k_cross[:, :, :, new_frames], v_cross[:, :, :, new_frames] = run_call(
            memory[:, first_row : first_row + MEMORY_FRAMES_PER_CALL]
        )
        valid_frame_count += MEMORY_FRAMES_PER_CALL
    return k_cross[:, :, :, :valid_frame_count], v_cross[:, :, :, :valid_frame_count]


def append_whole_segment(run_call: CallRunner, memory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """append_cross_kv with buffers exactly as long as the memory. Each test recording is one segment here, and
    two_cities (2220 rows) is longer than Mmax; Step 6 splits long recordings into segments."""
    return append_cross_kv(run_call, memory, capacity_frame_count=memory.shape[1])


def wrapper_call_runner(wrapper: CrossAttentionKeyValueProjection) -> CallRunner:
    def run_call(memory_rows):
        k_new, v_new = wrapper(torch.from_numpy(memory_rows))
        return k_new.numpy(), v_new.numpy()

    return run_call


def coreml_call_runner(coreml_model: coremltools.models.MLModel) -> CallRunner:
    def run_call(memory_rows):
        outputs = coreml_model.predict({"memory": memory_rows})
        return outputs["k_new"], outputs["v_new"]

    return run_call


def run_cross_kv_ort(cross_kv_session: onnxruntime.InferenceSession, memory: np.ndarray) -> tuple[np.ndarray, ...]:
    """cross_kv.ort on any number of memory rows [1, M, 320] -> k_cross, v_cross [6,1,8,M,40]. Moonshine passes the
    whole memory so far; its 8-bit step for the memory depends on every row passed (tools/inspect_cross_kv_ort.py)."""
    return tuple(cross_kv_session.run(list(KEY_AND_VALUE_NAMES), {"memory": memory}))


def frame_major(keys_or_values: np.ndarray) -> np.ndarray:
    """[6,1,8,M,40] -> [1, M, 1920]: each frame's numbers from every layer and head in one row, so
    compare_with_worst_frame finds the worst 20 ms frame."""
    frame_count = keys_or_values.shape[FRAME_AXIS]
    return np.moveaxis(keys_or_values, FRAME_AXIS, 0).reshape(1, frame_count, -1)


def compare_keys_and_values(
    label: str, actual: tuple[np.ndarray, np.ndarray], expected: tuple[np.ndarray, np.ndarray], **pass_bar
) -> list[bool]:
    """compare_with_worst_frame on the keys, then the values."""
    return [
        compare_with_worst_frame(f"{label} {name}", frame_major(actual_part), frame_major(expected_part), **pass_bar)
        for name, actual_part, expected_part in zip(KEY_AND_VALUE_NAMES, actual, expected)
    ]


def load_reference_by_recording(reference_path, key_suffix: str, previous_step_module: str) -> dict[str, np.ndarray]:
    if not reference_path.exists():
        raise FileNotFoundError(f"{reference_path} (run python -m {previous_step_module} first)")
    reference = np.load(reference_path)
    return {name: reference[f"{name}_{key_suffix}"] for name in TEST_AUDIO_PATHS}


def check_shapes(wrapper: CrossAttentionKeyValueProjection) -> None:
    """The planned sizes match Moonshine's, the checkpoint's and every layer's; the wrapper's outputs have the
    planned shape; and the facts the plan asks to confirm, printed."""
    streaming_config = load_streaming_config()
    cross_attentions = wrapper.cross_attentions
    decoder_config = cross_attentions[0].config
    for planned_name, planned_value, actual_value in (
        ("DECODER_LAYER_COUNT vs streaming_config depth", DECODER_LAYER_COUNT, streaming_config["depth"]),
        ("DECODER_LAYER_COUNT vs decoder layers", DECODER_LAYER_COUNT, len(cross_attentions)),
        ("DECODER_HEAD_COUNT vs streaming_config nheads", DECODER_HEAD_COUNT, streaming_config["nheads"]),
        ("DECODER_HEAD_COUNT vs num_attention_heads", DECODER_HEAD_COUNT, decoder_config.num_attention_heads),
        ("DECODER_HEAD_COUNT vs num_key_value_heads", DECODER_HEAD_COUNT, decoder_config.num_key_value_heads),
        ("DECODER_HEAD_DIM vs streaming_config head_dim", DECODER_HEAD_DIM, streaming_config["head_dim"]),
        ("DECODER_DIM vs heads x head_dim", DECODER_DIM, DECODER_HEAD_COUNT * DECODER_HEAD_DIM),
    ):
        if planned_value != actual_value:
            raise ValueError(f"{planned_name}: planned {planned_value}, actual {actual_value}")
    for layer_index, cross_attention in enumerate(cross_attentions):
        if cross_attention.head_dim != DECODER_HEAD_DIM:
            raise ValueError(f"layer {layer_index}: head_dim {cross_attention.head_dim}, planned {DECODER_HEAD_DIM}")
    print(
        f"  {DECODER_LAYER_COUNT} layers x {DECODER_HEAD_COUNT} heads x {DECODER_HEAD_DIM} numbers; "
        f"num_key_value_heads {decoder_config.num_key_value_heads} = num_attention_heads "
        f"{decoder_config.num_attention_heads}"
    )
    head_dim_paddings = [cross_attention.head_dim_padding for cross_attention in cross_attentions]
    print(f"  head_dim_padding per layer: {head_dim_paddings} (applied inside attention, after the cache)")
    biases = [
        projection.bias for cross_attention in cross_attentions
        for projection in (cross_attention.k_proj, cross_attention.v_proj)
    ]
    print(f"  k_proj and v_proj with a bias: {sum(bias is not None for bias in biases)} of {len(biases)}")

    k_new, v_new = wrapper(torch.zeros(1, MEMORY_FRAMES_PER_CALL, DECODER_DIM))
    expected_shape = attention_cache_shape(MEMORY_FRAMES_PER_CALL)
    print(
        f"  k_new {tuple(k_new.shape)}, v_new {tuple(v_new.shape)}; all-zero memory gives all-zero K and V: "
        f"{not k_new.any() and not v_new.any()} (so unused buffer frames score 0, not blocked: Step 5 must mask them)"
    )
    for name, output in (("k_new", k_new), ("v_new", v_new)):
        if tuple(output.shape) != expected_shape:
            raise ValueError(f"{name}: shape {tuple(output.shape)}, planned {expected_shape}")

    buffer_value_count = int(np.prod(attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT)))
    print(
        f"  Swift's 2 buffers at Mmax {MAXIMUM_MEMORY_FRAME_COUNT}: {attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT)}, "
        f"{buffer_value_count:,} values each; {2 * buffer_value_count * 4 / 1e6:.1f} MB together as float32, "
        f"{2 * buffer_value_count * 2 / 1e6:.1f} MB as float16"
    )


def run_original_cross_attention_cache(
    decoder: MoonshineStreamingDecoder, encoded: np.ndarray, bos_token_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """The cross K/V Transformers' own decoder caches for a whole recording: decoder.forward with an empty cache and
    one start token, as model.generate's first step. forward's first lines turn encoded into memory (what Step 3's
    adapter does), then each layer's encoder_attn projects the whole memory at once and stores the result in
    past_key_values.cross_attention_cache. forward adds the positions with +=, changing the tensor it is given, so it
    gets a copy."""
    output = decoder(
        input_ids=torch.tensor([[bos_token_id]]),
        encoder_hidden_states=torch.from_numpy(encoded).clone(),
        use_cache=True,
    )
    cache_layers = output.past_key_values.cross_attention_cache.layers
    keys = torch.stack([cache_layer.keys for cache_layer in cache_layers]).numpy()
    values = torch.stack([cache_layer.values for cache_layer in cache_layers]).numpy()
    expected_shape = attention_cache_shape(encoded.shape[1])
    for name, cached in zip(KEY_AND_VALUE_NAMES, (keys, values)):
        if cached.shape != expected_shape:
            raise ValueError(f"Transformers' cached {name}: shape {cached.shape}, expected {expected_shape}")
    return keys, values


def check_a_wrapper_vs_transformers_cache(
    decoder: MoonshineStreamingDecoder,
    encoded_by_recording: dict[str, np.ndarray],
    memory_by_recording: dict[str, np.ndarray],
    wrapper_cross_kv_by_recording: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[bool]:
    """(a) The plan's "append-only vs full recompute": Transformers projecting each whole recording's memory at once,
    vs the wrapper projecting 4 rows per call, appended."""
    bos_token_id = load_streaming_config()["bos_id"]
    results = []
    for recording_name, encoded in encoded_by_recording.items():
        original_cross_kv = run_original_cross_attention_cache(decoder, encoded, bos_token_id)
        wrapper_cross_kv = wrapper_cross_kv_by_recording[recording_name]
        frame_count = memory_by_recording[recording_name].shape[1]
        print(
            f"  {recording_name}: memory {memory_by_recording[recording_name].shape} -> "
            f"{frame_count // MEMORY_FRAMES_PER_CALL} calls -> k_cross, v_cross {wrapper_cross_kv[0].shape}"
        )
        results += compare_keys_and_values(
            f"  {recording_name} wrapper vs Transformers' cache",
            wrapper_cross_kv,
            original_cross_kv,
            maximum_allowed_max_abs_diff=WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
        )
    return results


def check_b_coreml_vs_wrapper(
    memory_by_recording: dict[str, np.ndarray],
    wrapper_cross_kv_by_recording: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[bool]:
    """(b) Core ML call by call, appended as Swift will, vs the wrapper. float16 runs twice: with ALL (the plan's
    setting) and with CPU_AND_NE (what Step 7's app loads)."""
    for model_path in (CROSS_KV_FLOAT32_MODEL_PATH, CROSS_KV_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run python -m step4_cross_kv.convert first)")

    results = []
    for description, model_path, compute_units, pass_bar in (
        ("float32, CPU only", CROSS_KV_FLOAT32_MODEL_PATH, coremltools.ComputeUnit.CPU_ONLY,
         {"maximum_allowed_max_abs_diff": COREML_FLOAT32_MAX_ABS_DIFF}),
        ("float16, all compute units", CROSS_KV_MODEL_PATH, coremltools.ComputeUnit.ALL,
         {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}),
        ("float16, CPU and Neural Engine", CROSS_KV_MODEL_PATH, coremltools.ComputeUnit.CPU_AND_NE,
         {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}),
    ):
        print(f"[(b) Core ML {description} vs wrapper]")
        run_coreml_call = coreml_call_runner(coremltools.models.MLModel(str(model_path), compute_units=compute_units))
        for recording_name, memory in memory_by_recording.items():
            coreml_cross_kv = append_whole_segment(run_coreml_call, memory)
            results += compare_keys_and_values(
                f"  {recording_name}", coreml_cross_kv, wrapper_cross_kv_by_recording[recording_name], **pass_bar
            )
    return results


def check_c_ort_vs_wrapper(
    memory_by_recording: dict[str, np.ndarray],
    wrapper_cross_kv_by_recording: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[bool]:
    """(c) Moonshine's cross_kv.ort on the whole memory, as Moonshine calls it, vs the wrapper's appended buffer.

    Two comparisons first, for scale: cross_kv.ort 4 rows at a time vs on the whole memory (its own noise: one 8-bit
    step for all rows passed together), and the wrapper's buffer with every row one call (4 frames) late vs the
    wrapper (what a misplaced row scores, which a bar must catch)."""
    print(f"  bar: cosine >= {MOONSHINE_ORT_MIN_COSINE_SIMILARITY}")
    cross_kv_session = load_ort_session("cross_kv.ort")
    results = []
    for recording_name, memory in memory_by_recording.items():
        wrapper_cross_kv = wrapper_cross_kv_by_recording[recording_name]
        ort_whole_memory_cross_kv = run_cross_kv_ort(cross_kv_session, memory)

        print(f"  {recording_name}, for scale:")
        ort_appended_cross_kv = append_whole_segment(
            lambda memory_rows: run_cross_kv_ort(cross_kv_session, memory_rows), memory
        )
        late = MEMORY_FRAMES_PER_CALL
        for label, actual, expected in (
            ("cross_kv.ort 4 rows at a time vs whole memory", ort_appended_cross_kv, ort_whole_memory_cross_kv),
            (
                "wrapper with every row one call late vs wrapper",
                tuple(part[:, :, :, :-late] for part in wrapper_cross_kv),
                tuple(part[:, :, :, late:] for part in wrapper_cross_kv),
            ),
        ):
            for name, actual_part, expected_part in zip(KEY_AND_VALUE_NAMES, actual, expected):
                compare(f"    {label} {name}", actual_part, expected_part)

        results += compare_keys_and_values(
            f"  {recording_name} cross_kv.ort whole memory vs wrapper",
            ort_whole_memory_cross_kv,
            wrapper_cross_kv,
            minimum_allowed_cosine_similarity=MOONSHINE_ORT_MIN_COSINE_SIMILARITY,
        )
    return results


def main() -> int:
    torch.set_grad_enabled(False)  # nothing here trains
    decoder = load_pytorch_model().model.decoder
    wrapper = CrossAttentionKeyValueProjection(decoder).eval()
    memory_by_recording = load_reference_by_recording(STEP3_REFERENCE_PATH, "memory", "step3_adapter.test")
    encoded_by_recording = load_reference_by_recording(STEP2_REFERENCE_PATH, "encoded", "step2_encoder.test")

    print("[shapes and sizes]")
    check_shapes(wrapper)

    run_wrapper_call = wrapper_call_runner(wrapper)
    wrapper_cross_kv_by_recording = {
        name: append_whole_segment(run_wrapper_call, memory) for name, memory in memory_by_recording.items()
    }

    print("[(a) wrapper 4 rows per call, appended, vs Transformers' cross-attention cache for the whole recording]")
    results = check_a_wrapper_vs_transformers_cache(
        decoder, encoded_by_recording, memory_by_recording, wrapper_cross_kv_by_recording
    )
    results += check_b_coreml_vs_wrapper(memory_by_recording, wrapper_cross_kv_by_recording)
    print("[(c) cross_kv.ort on the whole memory vs wrapper, appended]")
    results += check_c_ort_vs_wrapper(memory_by_recording, wrapper_cross_kv_by_recording)

    if not all(results):
        print("\nFAILED")
        return 1
    np.savez(
        STEP4_REFERENCE_PATH,
        **{
            f"{recording_name}_{name}": part
            for recording_name, cross_kv in wrapper_cross_kv_by_recording.items()
            for name, part in zip(KEY_AND_VALUE_NAMES, cross_kv)
        },
    )
    print(f"\nAll checks passed. Saved {STEP4_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
