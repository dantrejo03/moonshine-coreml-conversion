"""Step 3 checks for the adapter. Run from the top of the repo, after step2_encoder.test (it writes ref/step2.npz) and
step3_adapter.convert:

    python -m step3_adapter.test

(a) wrapper call by call vs the original decoder's first lines on the whole signal, (b) Core ML vs wrapper,
(c) Moonshine's adapter.ort vs wrapper. Exits non-zero if any check fails; writes ref/step3.npz only when all pass.
"""

import sys
from collections.abc import Callable, Iterator

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
    ADAPTER_TRAINED_POSITION_COUNT,
    DECODER_DIM,
    ENCODER_DIM,
    FEATURE_FRAME_SECONDS,
    MAXIMUM_MEMORY_FRAME_COUNT,
    MEMORY_FRAMES_PER_CALL,
    REFERENCE_TENSORS_DIRECTORY,
    TEST_AUDIO_PATHS,
    load_streaming_config,
)
from common.load_model import load_ort_session
from step2_encoder.test import STEP2_REFERENCE_PATH
from step3_adapter.convert import ADAPTER_FLOAT32_MODEL_PATH, ADAPTER_MODEL_PATH
from step3_adapter.wrapper import StreamingAdapter, build_streaming_adapter

STEP3_REFERENCE_PATH = REFERENCE_TENSORS_DIRECTORY / "step3.npz"

# One all-zero encoded frame per table row: the memory is then the position table itself, read at every offset.
POSITION_TABLE_SIGNAL_NAME = "position_table"

# Check (c)'s max abs bar is half of adapter.ort's quantization step, plus this much for float32 rounding: the
# dequantized value and the add each round by about 1e-6 when |memory| < 16.
ORT_FLOAT32_ROUNDING_MARGIN = 1e-5

# Runs the adapter once: (encoded [1,4,320] float32, pos_offset) -> memory [1,4,320] float32.
# One per implementation: the wrapper, Core ML, and adapter.ort.
CallRunner = Callable[[np.ndarray, int], np.ndarray]


def adapter_calls(encoded: np.ndarray) -> Iterator[tuple[np.ndarray, int]]:
    """The adapter calls for one segment's encoded frames [1, T, 320], in order: (4 frames, pos_offset). This is the
    bookkeeping Swift ports: pos_offset starts at 0 and goes up by 4 after each call, so frame i gets position i, as
    in the original's arange(T), and pos_offset always equals the number of memory rows so far.

    Step 2 hands over a multiple of 4 frames every time (4 per chunk, 20 on release), so the groups line up."""
    frame_count = encoded.shape[1]
    if frame_count % MEMORY_FRAMES_PER_CALL:
        raise ValueError(f"{frame_count} encoded frames is not a whole number of {MEMORY_FRAMES_PER_CALL}-frame calls")
    for pos_offset in range(0, frame_count, MEMORY_FRAMES_PER_CALL):
        yield encoded[:, pos_offset : pos_offset + MEMORY_FRAMES_PER_CALL], pos_offset


def run_adapter_streaming(run_call: CallRunner, encoded: np.ndarray) -> np.ndarray:
    """The segment's memory [1, T, 320]: every call's 4 rows, in order, as Swift appends them."""
    return np.concatenate([run_call(frames, pos_offset) for frames, pos_offset in adapter_calls(encoded)], axis=1)


def wrapper_call_runner(wrapper: StreamingAdapter) -> CallRunner:
    def run_call(encoded, pos_offset):
        return wrapper(torch.from_numpy(encoded), torch.tensor([pos_offset], dtype=torch.int32)).numpy()

    return run_call


def coreml_call_runner(coreml_model: coremltools.models.MLModel) -> CallRunner:
    def run_call(encoded, pos_offset):
        inputs = {"encoded": encoded, "pos_offset": np.array([pos_offset], dtype=np.int32)}
        return coreml_model.predict(inputs)["memory"]

    return run_call


def ort_call_runner(adapter_session: onnxruntime.InferenceSession) -> CallRunner:
    """adapter.ort takes pos_offset as int64 (the plan's gotcha 8), and any number of frames; here always 4."""

    def run_call(encoded, pos_offset):
        inputs = {"encoded": encoded, "pos_offset": np.array([pos_offset], dtype=np.int64)}
        (memory,) = adapter_session.run(["memory"], inputs)
        return memory

    return run_call


def adapter_ort_quantization_step(position_table: np.ndarray) -> float:
    """The step between adapter.ort's stored table values. onnx_shrink_ray stores the table as uint8 with one scale:
    255 steps from min(0, smallest value) to max(0, largest value). tools/inspect_position_table.py checks this
    against the scale stored in adapter.ort."""
    return float(max(position_table.max(), 0.0) - min(position_table.min(), 0.0)) / 255


def load_encoded_by_signal(position_count: int) -> dict[str, np.ndarray]:
    """What every check feeds the adapter: Step 2's kept frames for the test recordings (ref/step2.npz), and
    position_table, one all-zero frame per table row. With zero input the memory is the table itself, read at every
    offset 0, 4, ..., 4092, both ends of the table included, as each implementation stores it."""
    if not STEP2_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"{STEP2_REFERENCE_PATH} (run python -m step2_encoder.test first)")
    step2_reference = np.load(STEP2_REFERENCE_PATH)
    encoded_by_signal = {name: step2_reference[f"{name}_encoded"] for name in TEST_AUDIO_PATHS}
    encoded_by_signal[POSITION_TABLE_SIGNAL_NAME] = np.zeros((1, position_count, ENCODER_DIM), dtype=np.float32)
    return encoded_by_signal


def run_original_decoder_top(
    decoder: MoonshineStreamingDecoder, encoded: np.ndarray, bos_token_id: int
) -> tuple[np.ndarray, bool]:
    """The memory MoonshineStreamingDecoder.forward computes in its first lines, for a whole signal in one call:
    pos_emb rows 0 ..< T added, then proj. forward doesn't return it, so a hook records proj's output. The rest of
    forward (one start token through the decoder layers) runs too, and is ignored.

    forward adds pos_emb with +=, changing the tensor it is given, so it gets a copy. Also returns whether the copy
    was changed."""
    recorded_memory = []
    hook = decoder.proj.register_forward_hook(lambda module, inputs, output: recorded_memory.append(output.clone()))
    encoder_hidden_states = torch.from_numpy(encoded).clone()
    try:
        decoder(input_ids=torch.tensor([[bos_token_id]]), encoder_hidden_states=encoder_hidden_states, use_cache=False)
    finally:
        hook.remove()
    (memory,) = recorded_memory
    changed_in_place = not torch.equal(encoder_hidden_states, torch.from_numpy(encoded))
    return memory.numpy(), changed_in_place


def check_shapes(wrapper: StreamingAdapter) -> None:
    """The planned sizes match Moonshine's and the checkpoint's, a segment fits in the trained positions, and the
    wrapper's output has the planned shape."""
    streaming_config = load_streaming_config()
    position_embedding = wrapper.decoder.pos_emb
    for planned_name, planned_value, actual_value in (
        ("DECODER_DIM", DECODER_DIM, streaming_config["decoder_dim"]),
        ("ENCODER_DIM vs pos_emb row width", ENCODER_DIM, position_embedding.embedding_dim),
    ):
        if planned_value != actual_value:
            raise ValueError(f"{planned_name}: planned {planned_value}, actual {actual_value}")

    position_count = position_embedding.num_embeddings
    print(
        f"  pos_emb {tuple(position_embedding.weight.shape)}: one row per {FEATURE_FRAME_SECONDS * 1000:.0f} ms "
        f"position, {position_count * FEATURE_FRAME_SECONDS:.2f} s in all; the first {ADAPTER_TRAINED_POSITION_COUNT} "
        f"({ADAPTER_TRAINED_POSITION_COUNT * FEATURE_FRAME_SECONDS:.0f} s) trained"
    )
    print(f"  proj: {type(wrapper.decoder.proj).__name__}")
    if not MAXIMUM_MEMORY_FRAME_COUNT <= ADAPTER_TRAINED_POSITION_COUNT <= position_count:
        raise ValueError(
            f"need Mmax ({MAXIMUM_MEMORY_FRAME_COUNT}) <= trained positions ({ADAPTER_TRAINED_POSITION_COUNT}) "
            f"<= table rows ({position_count})"
        )
    print(
        f"  Mmax {MAXIMUM_MEMORY_FRAME_COUNT}: the largest pos_offset Swift sends is "
        f"{MAXIMUM_MEMORY_FRAME_COUNT - MEMORY_FRAMES_PER_CALL}"
    )

    memory = wrapper(torch.zeros(1, MEMORY_FRAMES_PER_CALL, ENCODER_DIM), torch.zeros(1, dtype=torch.int32))
    expected_shape = (1, MEMORY_FRAMES_PER_CALL, DECODER_DIM)
    print(f"  memory {tuple(memory.shape)}")
    if tuple(memory.shape) != expected_shape:
        raise ValueError(f"memory: shape {tuple(memory.shape)}, planned {expected_shape}")


def check_a_wrapper_vs_decoder(
    wrapper: StreamingAdapter,
    encoded_by_signal: dict[str, np.ndarray],
    wrapper_memory_by_signal: dict[str, np.ndarray],
) -> list[bool]:
    """(a) The original decoder's first lines on each whole signal in one call vs the wrapper call by call. On
    position_table this is the plan's "wrapper vs direct pos_emb indexing": the original reads the table with
    pos_emb(arange(4096)), the wrapper 4 rows at a time at offsets 0 to 4092."""
    bos_token_id = load_streaming_config()["bos_id"]
    results = []
    for signal_name, encoded in encoded_by_signal.items():
        original_memory, changed_in_place = run_original_decoder_top(wrapper.decoder, encoded, bos_token_id)
        wrapper_memory = wrapper_memory_by_signal[signal_name]
        frame_count = encoded.shape[1]
        print(
            f"  {signal_name}: encoded {encoded.shape} -> {frame_count // MEMORY_FRAMES_PER_CALL} adapter calls "
            f"(pos_offset 0 to {frame_count - MEMORY_FRAMES_PER_CALL}) -> memory {wrapper_memory.shape}; "
            f"decoder.forward changed its encoder_hidden_states argument in place: {changed_in_place}"
        )
        result = compare(
            f"  {signal_name} wrapper vs decoder",
            wrapper_memory,
            original_memory,
            maximum_allowed_max_abs_diff=WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
        )
        results.append(result["passed"])
    return results


def check_b_coreml_vs_wrapper(
    encoded_by_signal: dict[str, np.ndarray], wrapper_memory_by_signal: dict[str, np.ndarray]
) -> list[bool]:
    """(b) Core ML call by call, as Swift will call it, vs the wrapper. position_table checks the runtime index
    lookup at every offset. float16 runs twice: with ALL (the plan's setting) and with CPU_AND_NE (what Step 7's app
    loads)."""
    for model_path in (ADAPTER_FLOAT32_MODEL_PATH, ADAPTER_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run python -m step3_adapter.convert first)")

    results = []
    for description, model_path, compute_units, pass_bar in (
        ("float32, CPU only", ADAPTER_FLOAT32_MODEL_PATH, coremltools.ComputeUnit.CPU_ONLY,
         {"maximum_allowed_max_abs_diff": COREML_FLOAT32_MAX_ABS_DIFF}),
        ("float16, all compute units", ADAPTER_MODEL_PATH, coremltools.ComputeUnit.ALL,
         {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}),
        ("float16, CPU and Neural Engine", ADAPTER_MODEL_PATH, coremltools.ComputeUnit.CPU_AND_NE,
         {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}),
    ):
        print(f"[(b) Core ML {description} vs wrapper]")
        run_coreml_call = coreml_call_runner(coremltools.models.MLModel(str(model_path), compute_units=compute_units))
        for signal_name, encoded in encoded_by_signal.items():
            coreml_memory = run_adapter_streaming(run_coreml_call, encoded)
            results.append(
                compare_with_worst_frame(
                    f"  {signal_name}", coreml_memory, wrapper_memory_by_signal[signal_name], **pass_bar
                )
            )
    return results


def check_c_ort_vs_wrapper(
    wrapper: StreamingAdapter,
    encoded_by_signal: dict[str, np.ndarray],
    wrapper_memory_by_signal: dict[str, np.ndarray],
) -> list[bool]:
    """(c) Moonshine's adapter.ort call by call vs the wrapper.

    adapter.ort is the same lookup and add, with its table stored as uint8 (tools/inspect_position_table.py), so each
    memory value may differ by up to half a quantization step. That is the max abs bar, on every signal; the plan's
    cosine bar applies to the recordings too. Cosine alone is too loose for this step: with every position off by
    one, cosine stays near 0.99 (printed first, for scale), while max abs is hundreds of times over the bar."""
    position_table = wrapper.decoder.pos_emb.weight.detach().numpy()
    quantization_step = adapter_ort_quantization_step(position_table)
    maximum_allowed_max_abs_diff = quantization_step / 2 + ORT_FLOAT32_ROUNDING_MARGIN
    print(
        f"  adapter.ort's table is uint8 in steps of {quantization_step:.6g}: bar max abs <= "
        f"{maximum_allowed_max_abs_diff:.6g} on every signal, plus cosine >= {MOONSHINE_ORT_MIN_COSINE_SIMILARITY} "
        "on the test recordings"
    )

    print("  for scale, the wrapper with every position one too high vs the wrapper:")
    run_wrapper_call = wrapper_call_runner(wrapper)
    for recording_name in TEST_AUDIO_PATHS:
        shifted_memory = run_adapter_streaming(
            lambda frames, pos_offset: run_wrapper_call(frames, pos_offset + 1), encoded_by_signal[recording_name]
        )
        compare(f"    {recording_name}", shifted_memory, wrapper_memory_by_signal[recording_name])

    run_ort_call = ort_call_runner(load_ort_session("adapter.ort"))
    results = []
    for signal_name, encoded in encoded_by_signal.items():
        pass_bar = {"maximum_allowed_max_abs_diff": maximum_allowed_max_abs_diff}
        if signal_name in TEST_AUDIO_PATHS:
            pass_bar["minimum_allowed_cosine_similarity"] = MOONSHINE_ORT_MIN_COSINE_SIMILARITY
        ort_memory = run_adapter_streaming(run_ort_call, encoded)
        results.append(
            compare_with_worst_frame(f"  {signal_name}", ort_memory, wrapper_memory_by_signal[signal_name], **pass_bar)
        )
    return results


def main() -> int:
    torch.set_grad_enabled(False)  # nothing here trains
    wrapper = build_streaming_adapter()
    encoded_by_signal = load_encoded_by_signal(wrapper.decoder.pos_emb.num_embeddings)

    print("[shapes and sizes]")
    check_shapes(wrapper)

    run_wrapper_call = wrapper_call_runner(wrapper)
    wrapper_memory_by_signal = {
        name: run_adapter_streaming(run_wrapper_call, encoded) for name, encoded in encoded_by_signal.items()
    }

    print("[(a) wrapper call by call vs the original decoder's first lines on the whole signal, float32]")
    results = check_a_wrapper_vs_decoder(wrapper, encoded_by_signal, wrapper_memory_by_signal)
    results += check_b_coreml_vs_wrapper(encoded_by_signal, wrapper_memory_by_signal)
    print("[(c) adapter.ort call by call vs wrapper]")
    results += check_c_ort_vs_wrapper(wrapper, encoded_by_signal, wrapper_memory_by_signal)

    if not all(results):
        print("\nFAILED")
        return 1
    np.savez(
        STEP3_REFERENCE_PATH,
        **{f"{name}_memory": wrapper_memory_by_signal[name] for name in TEST_AUDIO_PATHS},
    )
    print(f"\nAll checks passed. Saved {STEP3_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
