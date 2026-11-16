"""Step 2 diagnosis for check (c): how far is Moonshine's quantized encoder.ort from the float encoder, and why?
Run from conversion/:

    python -m step2_encoder.tools.diagnose_ort

scripts/quantize-streaming-model.sh builds every piece except the frontend with onnx_shrink_ray --method
integer_activations: int8 weights, and each matmul's other input rounded to 8 bits on the fly, with one step size per
tensor set by that tensor's smallest and largest values (DynamicQuantizeLinear). Three experiments separate that
noise from a model that computes something different:

1. Self-consistency: encoder.ort window by window vs on the whole recording in one call. A float model with local
   attention gives the same kept frames both ways (check (a): max abs 0), so any difference is encoder.ort
   disagreeing with itself: its noise floor, the best agreement any check against it can reach.
2. Far context: one second of speech put in front of every full window, more than 96 frames before any kept frame,
   beyond what an encoded frame can reach. The float encoder can't see it. If encoder.ort's kept frames change, its
   output depends on more than the frames it attends to, which no mask explains and a per-tensor step size does.
3. Window readings: the float encoder on the whole recording with each reading of sliding_windows vs encoder.ort.
   The reading encoder.ort was built with lands at experiment 1's noise floor; the others fall below it.

First run (Transformers' reading in the wrapper): experiment 3 put Moonshine's reading (16 back, 4 ahead) at 0.9916,
equal to experiment 1, and Transformers' (15 back, 3 ahead) at 0.973. The wrapper now uses Moonshine's.
"""

import numpy as np
import onnxruntime
import torch
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingEncoder

from common.compare import compare, compare_with_worst_frame
from common.config import ENCODER_LEFT_CONTEXT_FRAME_COUNT, TEST_AUDIO_PATHS
from common.load_model import load_ort_session, load_pytorch_model
from step1_frontend.wrapper import StreamingFrontend
from step2_encoder.test import (
    encode_streaming,
    load_features_by_signal,
    ort_window_runner,
    run_original_encoder,
    streaming_windows,
    transformers_reading_of,
)

# Experiment 2's far context: 1 s of speech from two_cities (20 to 21 s), put in front of every full window.
FAR_CONTEXT_RECORDING_NAME = "two_cities"
FAR_CONTEXT_FRAMES = slice(1000, 1050)

# Experiment 3's readings: (frames back, frames ahead) for each of the 6 layers. The first is the wrapper's.
SLIDING_WINDOW_READINGS = {
    "16 back, 4 ahead (Moonshine, wrapper)": [(16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)],
    "15 back, 3 ahead (Transformers)": [(15, 3), (15, 3), (15, 0), (15, 0), (15, 3), (15, 3)],
    "16 back, 3 ahead": [(16, 3), (16, 3), (16, 0), (16, 0), (16, 3), (16, 3)],
    "15 back, 4 ahead": [(15, 4), (15, 4), (15, 0), (15, 0), (15, 4), (15, 4)],
    "16 back, no lookahead": [(16, 0)] * 6,
    "16 back, 4 ahead in all 6 layers": [(16, 4)] * 6,
}
MOONSHINE_SLIDING_WINDOWS = SLIDING_WINDOW_READINGS["16 back, 4 ahead (Moonshine, wrapper)"]


def run_ort_once(encoder_session: onnxruntime.InferenceSession, features: np.ndarray) -> np.ndarray:
    """encoder.ort on these frames in one call."""
    (encoded,) = encoder_session.run(["encoded"], {"features": features})
    return encoded


def run_float_once(
    encoder: MoonshineStreamingEncoder, features: np.ndarray, sliding_windows=MOONSHINE_SLIDING_WINDOWS
) -> np.ndarray:
    """The original float encoder on these frames in one call, with sliding_windows (frames back, frames ahead)."""
    return run_original_encoder(
        encoder,
        features,
        attention_mask=torch.ones(features.shape[:2], dtype=torch.long),
        transformers_sliding_windows=transformers_reading_of(sliding_windows),
    )


def experiment_self_consistency(encoder_session, features_by_recording, ort_whole_by_recording, float_whole_by_recording):
    print("[1. encoder.ort window by window vs encoder.ort on the whole recording; a float model gives max abs 0]")
    run_ort_window = ort_window_runner(encoder_session)
    for recording_name, features in features_by_recording.items():
        ort_windowed = encode_streaming(run_ort_window, features)
        compare_with_worst_frame(f"  {recording_name}", ort_windowed, ort_whole_by_recording[recording_name])
    print("[   for scale: float encoder vs encoder.ort, both on the whole recording]")
    for recording_name in features_by_recording:
        compare_with_worst_frame(
            f"  {recording_name}", float_whole_by_recording[recording_name], ort_whole_by_recording[recording_name]
        )


def experiment_far_context(encoder, encoder_session, features_by_recording, far_context: np.ndarray):
    far_frame_count = far_context.shape[1]
    print(
        f"[2. kept frames of every full window, with {far_frame_count} frames of speech in front vs without; "
        "the float encoder can't reach them]"
    )
    runs = {
        "float encoder": lambda features: run_float_once(encoder, features),
        "encoder.ort": lambda features: run_ort_once(encoder_session, features),
    }
    for recording_name, features in features_by_recording.items():
        kept_without_by_run = {run_name: [] for run_name in runs}
        kept_with_by_run = {run_name: [] for run_name in runs}
        for window in streaming_windows(features):
            if window.new_frames.start < ENCODER_LEFT_CONTEXT_FRAME_COUNT:
                continue  # a window at the recording's start: its kept frames could reach frames put in front
            real_features = window.features[:, : int(window.valid.sum())]
            with_far_context = np.concatenate([far_context, real_features], axis=1)
            kept_after_far_context = slice(
                window.new_frames.start + far_frame_count, window.new_frames.stop + far_frame_count
            )
            for run_name, run in runs.items():
                kept_without_by_run[run_name].append(run(real_features)[:, window.new_frames])
                kept_with_by_run[run_name].append(run(with_far_context)[:, kept_after_far_context])
        for run_name in runs:
            compare(
                f"  {recording_name} {run_name}",
                np.concatenate(kept_with_by_run[run_name], axis=1),
                np.concatenate(kept_without_by_run[run_name], axis=1),
            )


def experiment_window_readings(encoder, features_by_recording, ort_whole_by_recording):
    print("[3. float encoder on the whole recording with each sliding_windows reading vs encoder.ort: cosine]")
    print(f"  {'reading':<38}" + "".join(f"{name:>12}" for name in features_by_recording))
    for reading_name, sliding_windows in SLIDING_WINDOW_READINGS.items():
        row = f"  {reading_name:<38}"
        for recording_name, features in features_by_recording.items():
            result = compare(
                reading_name,
                run_float_once(encoder, features, sliding_windows),
                ort_whole_by_recording[recording_name],
                print_result=False,
            )
            row += f"{result['cosine_similarity']:>12.6f}"
        print(row)


def main() -> None:
    torch.set_grad_enabled(False)
    encoder = load_pytorch_model().model.encoder
    all_features = load_features_by_signal(StreamingFrontend(encoder.embedder).eval())
    features_by_recording = {name: all_features[name] for name in TEST_AUDIO_PATHS}
    encoder_session = load_ort_session("encoder.ort")

    ort_whole_by_recording = {name: run_ort_once(encoder_session, f) for name, f in features_by_recording.items()}
    float_whole_by_recording = {name: run_float_once(encoder, f) for name, f in features_by_recording.items()}

    experiment_self_consistency(encoder_session, features_by_recording, ort_whole_by_recording, float_whole_by_recording)
    far_context = features_by_recording[FAR_CONTEXT_RECORDING_NAME][:, FAR_CONTEXT_FRAMES]
    experiment_far_context(encoder, encoder_session, features_by_recording, far_context)
    experiment_window_readings(encoder, features_by_recording, ort_whole_by_recording)


if __name__ == "__main__":
    main()
