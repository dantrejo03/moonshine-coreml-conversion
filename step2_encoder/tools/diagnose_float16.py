"""Step 2 float16 diagnosis: does float16 hold up on the Neural Engine, layer by layer? Run from the top of the repo:

    python -m step2_encoder.tools.diagnose_float16

Step 1 found a float16 model can match float32 on the GPU and still lose precision on the Neural Engine (Core ML's
silu op). The encoder brings new ops to doubt: GELU in every MLP, softmax over masked scores, and layer norm. This
converts a float16 copy of the wrapper that also outputs every layer, runs it window by window on CPU_AND_GPU and on
CPU_AND_NE (what the watch app uses), and compares each layer with the float32 PyTorch wrapper. If the Neural Engine
table falls behind the GPU table, the layer where the gap opens is where to look. Rerun it after OS updates.
"""

import collections

import coremltools
import numpy as np
import torch

from common.compare import compare
from common.coreml_conversion import convert_to_coreml
from step1_frontend.wrapper import StreamingFrontend
from step2_encoder.convert import INPUT_SHAPES
from step2_encoder.test import load_features_by_signal, streaming_windows
from step2_encoder.wrapper import StreamingEncoder, build_streaming_encoder

SIGNAL_NAMES_TO_DIAGNOSE = ("silence", "noise_0.0001", "noise_0.001", "beckett")
COMPUTE_UNITS_TO_COMPARE = {
    "GPU": coremltools.ComputeUnit.CPU_AND_GPU,
    "Neural Engine": coremltools.ComputeUnit.CPU_AND_NE,
}


class StreamingEncoderWithLayerOutputs(StreamingEncoder):
    """Returns every layer's output, in run_layers order, instead of only encoded."""

    def forward(self, features, valid):
        return tuple(self.run_layers(features, valid).values())


def kept_frames_by_stage(run_window, features: np.ndarray) -> dict[str, np.ndarray]:
    """Each stage's kept frames [1, T, 320] for the whole recording, window by window as in Swift.
    run_window(features, valid) returns a dict of stage outputs, each [1,116,320]."""
    frames_by_stage = collections.defaultdict(list)
    for window in streaming_windows(features):
        for stage_name, values in run_window(window.features, window.valid).items():
            frames_by_stage[stage_name].append(values[:, window.new_frames])
    return {stage_name: np.concatenate(frames, axis=1) for stage_name, frames in frames_by_stage.items()}


def largest_magnitude_by_stage(probe: StreamingEncoderWithLayerOutputs, features_by_signal) -> dict[str, float]:
    """The largest |value| each stage reaches in float32, over every window, padding rows included. float16 tops out
    at 65504, and one row that overflows ruins the real rows too: layer norm turns inf into NaN, and attention
    multiplies even a 0 weight by that row's value (0 x NaN = NaN)."""
    largest_magnitudes = collections.defaultdict(float)
    for features in features_by_signal.values():
        for window in streaming_windows(features):
            stage_outputs = probe.run_layers(torch.from_numpy(window.features), torch.from_numpy(window.valid))
            for stage_name, values in stage_outputs.items():
                largest_magnitudes[stage_name] = max(largest_magnitudes[stage_name], float(values.abs().max()))
    return largest_magnitudes


def main() -> None:
    torch.set_grad_enabled(False)
    wrapper = build_streaming_encoder()
    probe = StreamingEncoderWithLayerOutputs(wrapper.encoder).eval()
    example_inputs = (torch.randn(INPUT_SHAPES["features"]), torch.ones(INPUT_SHAPES["valid"]))
    stage_names = list(probe.run_layers(*example_inputs).keys())
    traced_probe = torch.jit.trace(probe, example_inputs)

    all_features = load_features_by_signal(StreamingFrontend(wrapper.encoder.embedder).eval())
    features_by_signal = {name: all_features[name] for name in SIGNAL_NAMES_TO_DIAGNOSE}

    def run_pytorch_window(features, valid):
        stage_outputs = probe.run_layers(torch.from_numpy(features), torch.from_numpy(valid))
        return {stage_name: values.numpy() for stage_name, values in stage_outputs.items()}

    reference_by_signal = {
        name: kept_frames_by_stage(run_pytorch_window, features) for name, features in features_by_signal.items()
    }

    print("\n[largest |value| per stage in float32, every window, padding rows included; float16's largest is 65504]")
    largest_magnitudes = largest_magnitude_by_stage(probe, features_by_signal)
    for stage_name in stage_names:
        print(f"  {stage_name:<15}{largest_magnitudes[stage_name]:>10.1f}")

    for compute_unit_name, compute_units in COMPUTE_UNITS_TO_COMPARE.items():
        coreml_probe = convert_to_coreml(
            traced_probe, INPUT_SHAPES, stage_names, coremltools.precision.FLOAT16, compute_units
        )

        def run_coreml_window(features, valid):
            return coreml_probe.predict({"features": features, "valid": valid})

        coreml_by_signal = {
            name: kept_frames_by_stage(run_coreml_window, features) for name, features in features_by_signal.items()
        }

        print(f"\n[float16 on the {compute_unit_name} vs float32 PyTorch, per layer: cosine (max abs diff)]")
        print(f"  {'stage':<15}" + "".join(f"{name:>23}" for name in features_by_signal))
        for stage_name in stage_names:
            row = f"  {stage_name:<15}"
            for signal_name in features_by_signal:
                result = compare(
                    f"{compute_unit_name} {signal_name} {stage_name}",
                    coreml_by_signal[signal_name][stage_name],
                    reference_by_signal[signal_name][stage_name],
                    print_result=False,
                )
                row += f"{result['cosine_similarity']:>13.6f} ({result['max_abs_diff']:.1e})"
            print(row)


if __name__ == "__main__":
    main()
