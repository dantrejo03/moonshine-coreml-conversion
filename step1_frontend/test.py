"""Step 1 checks for the streaming frontend. Run from the top of the repo, after convert.py:

    python -m step1_frontend.test

(a) wrapper vs the original embedder, (b) Core ML vs wrapper, (c) Moonshine's frontend.ort vs wrapper.
Exits non-zero if any check fails; writes ref/step1.npz only when all of them pass.
"""

import sys
from collections.abc import Callable

import coremltools
import numpy as np
import onnxruntime
import torch

from common.audio import load_audio
from common.compare import (
    COREML_FLOAT16_MIN_COSINE_SIMILARITY,
    COREML_FLOAT32_MAX_ABS_DIFF,
    MOONSHINE_ORT_MIN_COSINE_SIMILARITY,
    WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
    compare,
    compare_with_worst_frame,
)
from common.config import (
    AUDIO_SAMPLES_PER_CHUNK,
    FEATURE_FRAMES_PER_CHUNK,
    FRONTEND_CONV1_BUFFER_SHAPE,
    FRONTEND_CONV2_BUFFER_SHAPE,
    MOONSHINE_ORT_PIECE_FILE_NAMES,
    REFERENCE_TENSORS_DIRECTORY,
    TEST_AUDIO_PATHS,
    load_streaming_config,
)
from common.load_model import load_ort_session
from step1_frontend.convert import FRONTEND_FLOAT32_MODEL_PATH, FRONTEND_MODEL_PATH
from step1_frontend.wrapper import StreamingFrontend, build_streaming_frontend

STEP1_REFERENCE_PATH = REFERENCE_TENSORS_DIRECTORY / "step1.npz"

# Near-silent signals for checks (b) and (c): recordings often start in silence, and the last chunk ends in zeros.
QUIET_SIGNAL_CHUNK_COUNT = 10
QUIET_NOISE_STANDARD_DEVIATIONS = (1e-3, 1e-4, 1e-5)  # about -60, -80, -100 dBFS

# frontend.model.ort's state inputs (moonshine-streaming-model.cpp, process_audio_chunk). Its other inputs are
# its weights, which are frontend.weights.ort's outputs.
ORT_FRONTEND_STATE_INPUT_NAMES = {"audio_chunk", "sample_buffer", "sample_len", "conv1_buffer", "conv2_buffer", "frame_count"}

# Runs one chunk: (audio_chunk, conv1_buffer, conv2_buffer) -> (features, conv1_buffer_out, conv2_buffer_out),
# all float32 numpy arrays. One per implementation: the wrapper, Core ML, and frontend.ort.
ChunkRunner = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


def pad_to_whole_chunks(samples: np.ndarray) -> np.ndarray:
    """Zeros on the end, up to a whole number of 1280-sample chunks: what Swift does with the last chunk."""
    return np.pad(samples, (0, -len(samples) % AUDIO_SAMPLES_PER_CHUNK))


def make_quiet_signals() -> dict[str, np.ndarray]:
    """Silence and faint noise, where float16 is most likely to break. CMVN squares each sample, and in
    float16 the square of anything under about 2.4e-4 rounds to 0 (the smallest positive float16 is ~6e-8)."""
    random_number_generator = np.random.default_rng(seed=0)
    sample_count = QUIET_SIGNAL_CHUNK_COUNT * AUDIO_SAMPLES_PER_CHUNK
    quiet_signals = {"silence": np.zeros(sample_count, dtype=np.float32)}
    for standard_deviation in QUIET_NOISE_STANDARD_DEVIATIONS:
        noise = random_number_generator.standard_normal(sample_count) * standard_deviation
        quiet_signals[f"noise_{standard_deviation:g}"] = noise.astype(np.float32)
    return quiet_signals


def run_chunk_by_chunk(run_chunk: ChunkRunner, padded_samples: np.ndarray) -> np.ndarray:
    """Features [1, 4 x chunk count, 320]. Each call's buffer outputs go into the next call, starting from zeros."""
    conv1_buffer = np.zeros(FRONTEND_CONV1_BUFFER_SHAPE, dtype=np.float32)
    conv2_buffer = np.zeros(FRONTEND_CONV2_BUFFER_SHAPE, dtype=np.float32)
    features_per_chunk = []
    for audio_chunk in padded_samples.reshape(-1, 1, AUDIO_SAMPLES_PER_CHUNK):
        features, conv1_buffer, conv2_buffer = run_chunk(audio_chunk, conv1_buffer, conv2_buffer)
        features_per_chunk.append(features)
    return np.concatenate(features_per_chunk, axis=1)


def wrapper_chunk_runner(wrapper: StreamingFrontend) -> ChunkRunner:
    def run_chunk(audio_chunk, conv1_buffer, conv2_buffer):
        outputs = wrapper(torch.from_numpy(audio_chunk), torch.from_numpy(conv1_buffer), torch.from_numpy(conv2_buffer))
        return tuple(output.numpy() for output in outputs)

    return run_chunk


def coreml_chunk_runner(coreml_model: coremltools.models.MLModel) -> ChunkRunner:
    def run_chunk(audio_chunk, conv1_buffer, conv2_buffer):
        outputs = coreml_model.predict(
            {"audio_chunk": audio_chunk, "conv1_buffer": conv1_buffer, "conv2_buffer": conv2_buffer}
        )
        return outputs["features"], outputs["conv1_buffer_out"], outputs["conv2_buffer_out"]

    return run_chunk


def load_ort_frontend(
    intra_op_thread_count: int | None = None,
) -> tuple[onnxruntime.InferenceSession, dict[str, np.ndarray]]:
    """frontend.model.ort, plus the weights it takes as inputs: the outputs of frontend.weights.ort, which has no inputs.
    This is the split the pinned libmoonshine.dylib loads (load_frontend_split_weights). intra_op_thread_count: as in
    load_ort_session."""
    weights_session = load_ort_session("frontend.weights.ort", intra_op_thread_count)
    weight_names = [output.name for output in weights_session.get_outputs()]
    weights_by_name = dict(zip(weight_names, weights_session.run(weight_names, {})))
    model_session = load_ort_session("frontend.model.ort", intra_op_thread_count)
    model_input_names = {model_input.name for model_input in model_session.get_inputs()}
    if model_input_names != ORT_FRONTEND_STATE_INPUT_NAMES | set(weights_by_name):
        raise ValueError(
            f"frontend.model.ort inputs {sorted(model_input_names)} are not the state inputs "
            f"{sorted(ORT_FRONTEND_STATE_INPUT_NAMES)} plus frontend.weights.ort's outputs {sorted(weights_by_name)}"
        )
    return model_session, weights_by_name


def ort_chunk_runner(model_session: onnxruntime.InferenceSession, weights_by_name: dict[str, np.ndarray]) -> ChunkRunner:
    """frontend.ort for one fresh recording. Moonshine's extra state starts at zero and is carried in here;
    the conv buffers are carried by run_chunk_by_chunk, as for the other runners."""
    frontend_state_shapes = load_streaming_config()["frontend_state_shapes"]
    extra_state = {
        "sample_buffer": np.zeros(frontend_state_shapes["sample_buffer"], dtype=np.float32),
        "sample_len": np.zeros(frontend_state_shapes["sample_len"], dtype=np.int64),
        "frame_count": np.zeros(frontend_state_shapes["frame_count"], dtype=np.int64),
    }
    output_names = [output.name for output in model_session.get_outputs()]

    def run_chunk(audio_chunk, conv1_buffer, conv2_buffer):
        inputs = {"audio_chunk": audio_chunk, "conv1_buffer": conv1_buffer, "conv2_buffer": conv2_buffer}
        outputs = dict(zip(output_names, model_session.run(output_names, inputs | extra_state | weights_by_name)))
        for state_name in extra_state:
            extra_state[state_name] = outputs[f"{state_name}_out"]
        return outputs["features"], outputs["conv1_buffer_out"], outputs["conv2_buffer_out"]

    return run_chunk


def check_shapes(wrapper: StreamingFrontend) -> None:
    """The wrapper's inputs and outputs have the planned shapes, and the buffers match Moonshine's."""
    streaming_config = load_streaming_config()
    for buffer_name, planned_shape in (
        ("conv1_buffer", FRONTEND_CONV1_BUFFER_SHAPE),
        ("conv2_buffer", FRONTEND_CONV2_BUFFER_SHAPE),
    ):
        moonshine_shape = tuple(streaming_config["frontend_state_shapes"][buffer_name])
        if planned_shape != moonshine_shape:
            raise ValueError(f"{buffer_name}: planned {planned_shape}, streaming_config.json has {moonshine_shape}")

    outputs = wrapper(
        torch.zeros(1, AUDIO_SAMPLES_PER_CHUNK),
        torch.zeros(FRONTEND_CONV1_BUFFER_SHAPE),
        torch.zeros(FRONTEND_CONV2_BUFFER_SHAPE),
    )
    expected_output_shapes = {
        "features": (1, FEATURE_FRAMES_PER_CHUNK, streaming_config["encoder_dim"]),
        "conv1_buffer_out": FRONTEND_CONV1_BUFFER_SHAPE,
        "conv2_buffer_out": FRONTEND_CONV2_BUFFER_SHAPE,
    }
    for (output_name, expected_shape), output in zip(expected_output_shapes.items(), outputs):
        print(f"  {output_name} {tuple(output.shape)}")
        if tuple(output.shape) != expected_shape:
            raise ValueError(f"{output_name}: shape {tuple(output.shape)}, planned {expected_shape}")


def check_a_wrapper_vs_embedder(
    wrapper: StreamingFrontend, padded_recordings: dict[str, np.ndarray]
) -> tuple[list[bool], dict[str, np.ndarray]]:
    """(a) The original embedder on the whole recording in one call vs the wrapper chunk by chunk."""
    results = []
    wrapper_features_by_recording = {}
    for recording_name, padded_samples in padded_recordings.items():
        original_features, _ = wrapper.embedder(torch.from_numpy(padded_samples)[None])
        wrapper_features = run_chunk_by_chunk(wrapper_chunk_runner(wrapper), padded_samples)
        print(
            f"  {recording_name}: {len(padded_samples) // AUDIO_SAMPLES_PER_CHUNK} chunks "
            f"-> features {tuple(wrapper_features.shape)}"
        )
        result = compare(
            f"  {recording_name} wrapper vs embedder",
            wrapper_features,
            original_features,
            maximum_allowed_max_abs_diff=WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
        )
        results.append(result["passed"])
        wrapper_features_by_recording[recording_name] = wrapper_features
    return results, wrapper_features_by_recording


def check_b_coreml_vs_wrapper(
    signals: dict[str, np.ndarray], wrapper_features_by_signal: dict[str, np.ndarray]
) -> list[bool]:
    """(b) Core ML chunk by chunk, carrying its own buffers as Swift will, vs the wrapper."""
    for model_path in (FRONTEND_FLOAT32_MODEL_PATH, FRONTEND_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run python -m step1_frontend.convert first)")

    results = []
    for description, model_path, compute_units, pass_bar in (
        ("float32, CPU only", FRONTEND_FLOAT32_MODEL_PATH, coremltools.ComputeUnit.CPU_ONLY,
         {"maximum_allowed_max_abs_diff": COREML_FLOAT32_MAX_ABS_DIFF}),
        ("float16, all compute units", FRONTEND_MODEL_PATH, coremltools.ComputeUnit.ALL,
         {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}),
    ):
        print(f"[(b) Core ML {description} vs wrapper]")
        run_coreml_chunk = coreml_chunk_runner(coremltools.models.MLModel(str(model_path), compute_units=compute_units))
        for signal_name, padded_samples in signals.items():
            coreml_features = run_chunk_by_chunk(run_coreml_chunk, padded_samples)
            results.append(
                compare_with_worst_frame(
                    f"  {signal_name}", coreml_features, wrapper_features_by_signal[signal_name], **pass_bar
                )
            )
    return results


def print_ort_signatures() -> None:
    """Inputs and outputs of every Moonshine .ort file, for the Findings log."""
    for file_name in MOONSHINE_ORT_PIECE_FILE_NAMES:
        session = load_ort_session(file_name)
        print(f"  {file_name}")
        for direction, values in (("in ", session.get_inputs()), ("out", session.get_outputs())):
            for value in values:
                print(f"    {direction} {value.name} {value.shape} {value.type}")


def check_c_ort_vs_wrapper(
    signals: dict[str, np.ndarray], wrapper_features_by_signal: dict[str, np.ndarray]
) -> list[bool]:
    """(c) Moonshine's quantized frontend.ort chunk by chunk from zeroed state vs the wrapper. The bar applies to
    the test recordings; the quiet signals are reported to set beside (b)."""
    model_session, weights_by_name = load_ort_frontend()
    results = []
    for signal_name, padded_samples in signals.items():
        ort_features = run_chunk_by_chunk(ort_chunk_runner(model_session, weights_by_name), padded_samples)
        pass_bar = (
            {"minimum_allowed_cosine_similarity": MOONSHINE_ORT_MIN_COSINE_SIMILARITY}
            if signal_name in TEST_AUDIO_PATHS
            else {}
        )
        results.append(
            compare_with_worst_frame(f"  {signal_name}", ort_features, wrapper_features_by_signal[signal_name], **pass_bar)
        )
    return results


def main() -> int:
    torch.set_grad_enabled(False)  # nothing here trains
    wrapper = build_streaming_frontend()
    padded_recordings = {name: pad_to_whole_chunks(load_audio(path)) for name, path in TEST_AUDIO_PATHS.items()}
    quiet_signals = make_quiet_signals()
    signals = padded_recordings | quiet_signals

    print("[shapes, one chunk]")
    check_shapes(wrapper)
    print("[(a) wrapper chunk by chunk vs original embedder on the whole recording, float32]")
    results, wrapper_features_by_recording = check_a_wrapper_vs_embedder(wrapper, padded_recordings)

    run_wrapper_chunk = wrapper_chunk_runner(wrapper)
    wrapper_features_by_signal = wrapper_features_by_recording | {
        name: run_chunk_by_chunk(run_wrapper_chunk, samples) for name, samples in quiet_signals.items()
    }
    results += check_b_coreml_vs_wrapper(signals, wrapper_features_by_signal)

    print("[Moonshine .ort files: inputs and outputs]")
    print_ort_signatures()
    print("[(c) frontend.ort chunk by chunk from zeroed state vs wrapper; bar on the test recordings only]")
    results += check_c_ort_vs_wrapper(signals, wrapper_features_by_signal)

    if not all(results):
        print("\nFAILED")
        return 1
    np.savez(
        STEP1_REFERENCE_PATH,
        **{f"{name}_features": features for name, features in wrapper_features_by_recording.items()},
    )
    print(f"\nAll checks passed. Saved {STEP1_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
