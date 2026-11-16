"""Step 2 checks for the streaming encoder. Run from conversion/, after step1_frontend.test (it writes
ref/step1.npz) and step2_encoder.convert:

    python -m step2_encoder.test

(a) wrapper window by window vs the original encoder on the whole recording, (b) Core ML vs wrapper,
(c) Moonshine's encoder.ort vs wrapper. Exits non-zero if any check fails; writes ref/step2.npz only when all pass.
"""

import dataclasses
import sys
from collections.abc import Callable, Iterator

import coremltools
import numpy as np
import onnxruntime
import torch
from torch import nn
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingEncoder

from common.compare import (
    COREML_FLOAT16_MIN_COSINE_SIMILARITY,
    COREML_FLOAT32_MAX_ABS_DIFF,
    MOONSHINE_ORT_MIN_COSINE_SIMILARITY,
    WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
    compare,
    compare_with_worst_frame,
)
from common.config import (
    ENCODER_DIM,
    ENCODER_HELD_BACK_FRAME_COUNT,
    ENCODER_LEFT_CONTEXT_FRAME_COUNT,
    ENCODER_WINDOW_FRAME_COUNT,
    FEATURE_FRAMES_PER_CHUNK,
    REFERENCE_TENSORS_DIRECTORY,
    TEST_AUDIO_PATHS,
    load_streaming_config,
)
from common.load_model import load_ort_session
from step1_frontend.test import STEP1_REFERENCE_PATH, make_quiet_signals, run_chunk_by_chunk, wrapper_chunk_runner
from step1_frontend.wrapper import StreamingFrontend
from step2_encoder.convert import ENCODER_FLOAT32_MODEL_PATH, ENCODER_MODEL_PATH
from step2_encoder.wrapper import StreamingEncoder, build_streaming_encoder

STEP2_REFERENCE_PATH = REFERENCE_TENSORS_DIRECTORY / "step2.npz"

# Runs the encoder on one window: (features [1,116,320], valid [1,116]) -> encoded [1,116,320], all float32 numpy
# arrays. One per implementation: the wrapper, Core ML, and encoder.ort.
WindowRunner = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass
class EncoderWindow:
    """One encoder call, as Swift will make it."""

    features: np.ndarray  # [1,116,320]: the window's real feature frames first, then zeros
    valid: np.ndarray  # [1,116]: 1 for each real frame, 0 for each padding frame
    new_frames: slice  # the output frames to keep, in order: emitted - windowStart ..< emitted - windowStart + new

    @property
    def new_frame_count(self) -> int:
        """How many encoded frames this call emits; the emitted count goes up by this much."""
        return self.new_frames.stop - self.new_frames.start


def encoder_window(features: np.ndarray, emitted_frame_count: int, is_final: bool) -> EncoderWindow | None:
    """One encoder call, or None when there is nothing new to emit. Mirrors encode() in moonshine-streaming-model.cpp;
    this is the window math Swift ports.

    features: every feature frame of the segment so far, [1, T, 320]. emitted_frame_count: encoded frames already
    emitted. is_final: the recording has ended, so the 16 held-back frames go out too."""
    total_feature_count = features.shape[1]
    stable_frame_count = (
        total_feature_count if is_final else max(0, total_feature_count - ENCODER_HELD_BACK_FRAME_COUNT)
    )
    new_frame_count = stable_frame_count - emitted_frame_count
    if new_frame_count <= 0:
        return None  # the first 4 chunks only fill the 16 held-back frames
    window_start = max(0, emitted_frame_count - ENCODER_LEFT_CONTEXT_FRAME_COUNT)
    window_frame_count = total_feature_count - window_start
    if window_frame_count > ENCODER_WINDOW_FRAME_COUNT:
        raise ValueError(f"window of {window_frame_count} frames is larger than {ENCODER_WINDOW_FRAME_COUNT}")

    window_features = np.zeros((1, ENCODER_WINDOW_FRAME_COUNT, ENCODER_DIM), dtype=np.float32)
    window_features[:, :window_frame_count] = features[:, window_start:total_feature_count]
    valid = np.zeros((1, ENCODER_WINDOW_FRAME_COUNT), dtype=np.float32)
    valid[:, :window_frame_count] = 1.0
    first_new_frame = emitted_frame_count - window_start
    return EncoderWindow(window_features, valid, slice(first_new_frame, first_new_frame + new_frame_count))


def streaming_windows(features: np.ndarray) -> Iterator[EncoderWindow]:
    """The encoder calls for one recording's features [1, T, 320]: encoder_window after each 80 ms chunk's 4 features
    arrive, the last one with isFinal.

    Moonshine calls encode() once per transcript update instead, after several chunks, so its windows vary in size.
    Once per chunk keeps every window at 116 frames or fewer: 96 of past context, 4 new and 16 held back (on the
    final call, 96 and 20 new).
    """
    total_feature_count = features.shape[1]
    if total_feature_count % FEATURE_FRAMES_PER_CHUNK:
        raise ValueError(f"{total_feature_count} features is not a whole number of {FEATURE_FRAMES_PER_CHUNK}-frame chunks")

    emitted_frame_count = 0
    for available_feature_count in range(FEATURE_FRAMES_PER_CHUNK, total_feature_count + 1, FEATURE_FRAMES_PER_CHUNK):
        window = encoder_window(
            features[:, :available_feature_count],
            emitted_frame_count,
            is_final=available_feature_count == total_feature_count,
        )
        if window is None:
            continue
        yield window
        emitted_frame_count += window.new_frame_count


def encode_streaming(run_window: WindowRunner, features: np.ndarray) -> np.ndarray:
    """The recording's encoded frames [1, T, 320]: each call's new frames, in order, as Swift will collect them."""
    return np.concatenate(
        [run_window(window.features, window.valid)[:, window.new_frames] for window in streaming_windows(features)],
        axis=1,
    )


def wrapper_window_runner(wrapper: StreamingEncoder) -> WindowRunner:
    def run_window(features, valid):
        return wrapper(torch.from_numpy(features), torch.from_numpy(valid)).numpy()

    return run_window


def coreml_window_runner(coreml_model: coremltools.models.MLModel) -> WindowRunner:
    def run_window(features, valid):
        return coreml_model.predict({"features": features, "valid": valid})["encoded"]

    return run_window


def ort_window_runner(encoder_session: onnxruntime.InferenceSession) -> WindowRunner:
    """encoder.ort takes any number of frames, so it gets only the window's real frames, unpadded, as Moonshine runs
    it. Its output is padded back to 116 rows so the kept frames line up with the other runners'."""

    def run_window(features, valid):
        real_frame_count = int(valid.sum())
        (encoded,) = encoder_session.run(["encoded"], {"features": features[:, :real_frame_count]})
        return np.pad(encoded, ((0, 0), (0, ENCODER_WINDOW_FRAME_COUNT - real_frame_count), (0, 0)))

    return run_window


def load_features_by_signal(frontend: StreamingFrontend) -> dict[str, np.ndarray]:
    """Step 1's output for every test signal: the recordings' features from ref/step1.npz, and the quiet signals'
    from the Step 1 wrapper, since ref/step1.npz doesn't hold them."""
    if not STEP1_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"{STEP1_REFERENCE_PATH} (run python -m step1_frontend.test first)")
    step1_reference = np.load(STEP1_REFERENCE_PATH)
    features_by_signal = {name: step1_reference[f"{name}_features"] for name in TEST_AUDIO_PATHS}
    run_frontend_chunk = wrapper_chunk_runner(frontend)
    for signal_name, samples in make_quiet_signals().items():
        features_by_signal[signal_name] = run_chunk_by_chunk(run_frontend_chunk, samples)
    return features_by_signal


class FeaturesPassThrough(nn.Module):
    """Takes the embedder's place, so the encoder's own forward can run on features that Step 1 already computed."""

    def forward(self, input_values, padding_mask=None):
        return input_values, padding_mask


def transformers_reading_of(sliding_windows) -> list[list[int]]:
    """The sliding_windows to give Transformers' mask code so it builds Moonshine's windows. Its (left, right) sees
    left - 1 frames back and right - 1 ahead (dist < left, -dist < right), so each number goes up by one."""
    return [[frames_back + 1, frames_ahead + 1] for frames_back, frames_ahead in sliding_windows]


def run_original_encoder(
    encoder: MoonshineStreamingEncoder, features: np.ndarray, attention_mask, transformers_sliding_windows
) -> np.ndarray:
    """MoonshineStreamingEncoder.forward on a whole recording's features in one call: its own mask building, layers
    and final norm. For this call only, the embedder is swapped for a pass-through, and config.sliding_windows (which
    the mask code reads on every call) for transformers_sliding_windows."""
    embedder, config_sliding_windows = encoder.embedder, encoder.config.sliding_windows
    encoder.embedder = FeaturesPassThrough()
    encoder.config.sliding_windows = transformers_sliding_windows
    try:
        return encoder(input_values=torch.from_numpy(features), attention_mask=attention_mask).last_hidden_state.numpy()
    finally:
        encoder.embedder = embedder
        encoder.config.sliding_windows = config_sliding_windows


def check_shapes(wrapper: StreamingEncoder) -> None:
    """The planned sizes match Moonshine's, and the wrapper's output has the planned shape."""
    streaming_config = load_streaming_config()
    for planned_name, planned_value, moonshine_value in (
        ("ENCODER_DIM", ENCODER_DIM, streaming_config["encoder_dim"]),
        ("ENCODER_HELD_BACK_FRAME_COUNT", ENCODER_HELD_BACK_FRAME_COUNT, streaming_config["total_lookahead"]),
        # moonshine-streaming-model.cpp, encode(): left_context_frames = 16 * config.depth
        ("ENCODER_LEFT_CONTEXT_FRAME_COUNT", ENCODER_LEFT_CONTEXT_FRAME_COUNT, 16 * streaming_config["depth"]),
        ("encoder layer count", len(wrapper.encoder.layers), streaming_config["depth"]),
    ):
        if planned_value != moonshine_value:
            raise ValueError(f"{planned_name}: planned {planned_value}, Moonshine has {moonshine_value}")

    print(
        f"  sliding_windows {wrapper.encoder.config.sliding_windows}: an encoded frame sees {wrapper.frames_seen_back} "
        f"frames back and {wrapper.frames_seen_ahead} ahead; the window math keeps {ENCODER_LEFT_CONTEXT_FRAME_COUNT} "
        f"and {ENCODER_HELD_BACK_FRAME_COUNT}"
    )
    encoded = wrapper(torch.zeros(1, ENCODER_WINDOW_FRAME_COUNT, ENCODER_DIM), torch.ones(1, ENCODER_WINDOW_FRAME_COUNT))
    expected_shape = (1, ENCODER_WINDOW_FRAME_COUNT, ENCODER_DIM)
    print(f"  encoded {tuple(encoded.shape)}")
    if tuple(encoded.shape) != expected_shape:
        raise ValueError(f"encoded: shape {tuple(encoded.shape)}, planned {expected_shape}")


def check_a_wrapper_vs_encoder(
    wrapper: StreamingEncoder,
    features_by_recording: dict[str, np.ndarray],
    wrapper_encoded_by_signal: dict[str, np.ndarray],
) -> list[bool]:
    """(a) The original encoder on the whole recording in one call vs the wrapper window by window. The all-ones
    attention_mask makes the original build its sliding-window masks, from windows given in Transformers' terms so they
    come out as Moonshine's. Two more lines, without a bar: the original with Transformers' own reading of the config
    (15 back, 3 ahead), and the original with no attention_mask at all (the mask trap)."""
    moonshine_windows_in_transformers_terms = transformers_reading_of(wrapper.sliding_windows)
    config_windows_as_transformers_reads_them = wrapper.encoder.config.sliding_windows
    results = []
    for recording_name, features in features_by_recording.items():
        windows = list(streaming_windows(features))
        padded_window_count = sum(int(window.valid.sum()) < ENCODER_WINDOW_FRAME_COUNT for window in windows)
        wrapper_encoded = wrapper_encoded_by_signal[recording_name]
        print(
            f"  {recording_name}: features {features.shape} -> {len(windows)} encoder calls "
            f"({padded_window_count} padded) -> encoded {wrapper_encoded.shape}"
        )
        all_frames_real = torch.ones(features.shape[:2], dtype=torch.long)
        original_encoded = run_original_encoder(
            wrapper.encoder, features, all_frames_real, moonshine_windows_in_transformers_terms
        )
        result = compare(
            f"  {recording_name} wrapper vs encoder",
            wrapper_encoded,
            original_encoded,
            maximum_allowed_max_abs_diff=WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
        )
        results.append(result["passed"])
        transformers_encoded = run_original_encoder(
            wrapper.encoder, features, all_frames_real, config_windows_as_transformers_reads_them
        )
        compare(
            f"  {recording_name} Transformers' own window reading (15 back, 3 ahead) vs Moonshine's",
            transformers_encoded,
            original_encoded,
        )
        trap_encoded = run_original_encoder(
            wrapper.encoder, features, None, moonshine_windows_in_transformers_terms
        )
        compare(f"  {recording_name} the mask trap: encoder without attention_mask vs with", trap_encoded, original_encoded)
    return results


def check_b_coreml_vs_wrapper(
    features_by_signal: dict[str, np.ndarray], wrapper_encoded_by_signal: dict[str, np.ndarray]
) -> list[bool]:
    """(b) Core ML window by window, as Swift will call it, vs the wrapper. float16 runs twice: with ALL (the plan's
    setting; on the Mac it picks the GPU for this model) and with CPU_AND_NE (what Step 7's app loads)."""
    for model_path in (ENCODER_FLOAT32_MODEL_PATH, ENCODER_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run python -m step2_encoder.convert first)")

    results = []
    for description, model_path, compute_units, pass_bar in (
        ("float32, CPU only", ENCODER_FLOAT32_MODEL_PATH, coremltools.ComputeUnit.CPU_ONLY,
         {"maximum_allowed_max_abs_diff": COREML_FLOAT32_MAX_ABS_DIFF}),
        ("float16, all compute units", ENCODER_MODEL_PATH, coremltools.ComputeUnit.ALL,
         {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}),
        ("float16, CPU and Neural Engine", ENCODER_MODEL_PATH, coremltools.ComputeUnit.CPU_AND_NE,
         {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}),
    ):
        print(f"[(b) Core ML {description} vs wrapper]")
        run_coreml_window = coreml_window_runner(coremltools.models.MLModel(str(model_path), compute_units=compute_units))
        for signal_name, features in features_by_signal.items():
            coreml_encoded = encode_streaming(run_coreml_window, features)
            results.append(
                compare_with_worst_frame(
                    f"  {signal_name}", coreml_encoded, wrapper_encoded_by_signal[signal_name], **pass_bar
                )
            )
    return results


def check_c_ort_vs_wrapper(
    features_by_signal: dict[str, np.ndarray], wrapper_encoded_by_signal: dict[str, np.ndarray]
) -> list[bool]:
    """(c) Moonshine's quantized encoder.ort on the same windows, unpadded, vs the wrapper. The bar applies to the
    test recordings; the quiet signals are reported to set beside (b)."""
    run_ort_window = ort_window_runner(load_ort_session("encoder.ort"))
    results = []
    for signal_name, features in features_by_signal.items():
        ort_encoded = encode_streaming(run_ort_window, features)
        pass_bar = (
            {"minimum_allowed_cosine_similarity": MOONSHINE_ORT_MIN_COSINE_SIMILARITY}
            if signal_name in TEST_AUDIO_PATHS
            else {}
        )
        results.append(
            compare_with_worst_frame(f"  {signal_name}", ort_encoded, wrapper_encoded_by_signal[signal_name], **pass_bar)
        )
    return results


def main() -> int:
    torch.set_grad_enabled(False)  # nothing here trains
    wrapper = build_streaming_encoder()
    features_by_signal = load_features_by_signal(StreamingFrontend(wrapper.encoder.embedder).eval())
    features_by_recording = {name: features_by_signal[name] for name in TEST_AUDIO_PATHS}

    print("[shapes and window sizes]")
    check_shapes(wrapper)

    run_wrapper_window = wrapper_window_runner(wrapper)
    wrapper_encoded_by_signal = {
        name: encode_streaming(run_wrapper_window, features) for name, features in features_by_signal.items()
    }

    print("[(a) wrapper window by window vs original encoder on the whole recording, float32]")
    results = check_a_wrapper_vs_encoder(wrapper, features_by_recording, wrapper_encoded_by_signal)
    results += check_b_coreml_vs_wrapper(features_by_signal, wrapper_encoded_by_signal)
    print("[(c) encoder.ort on each window, unpadded, vs wrapper; bar on the test recordings only]")
    results += check_c_ort_vs_wrapper(features_by_signal, wrapper_encoded_by_signal)

    if not all(results):
        print("\nFAILED")
        return 1
    np.savez(
        STEP2_REFERENCE_PATH,
        **{f"{name}_encoded": wrapper_encoded_by_signal[name] for name in TEST_AUDIO_PATHS},
    )
    print(f"\nAll checks passed. Saved {STEP2_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
