"""Step 1 float16 diagnosis on the Neural Engine: the evidence for the SiLU recipe in common/coreml_torch_ops.py.
Run from conversion/:

    python -m step1_frontend.tools.diagnose_float16

float16 matches float32 to 0.999998 on the GPU, but on the Neural Engine Core ML's silu op loses precision
(silu(0) comes out as up to 1.7e-3, not 0) and conv2 amplifies it. This converts a float16 copy of the wrapper
that also outputs every intermediate stage, once per SiLU recipe, runs it on CPU_AND_NE (what the watch app
uses), and compares every stage with the float32 PyTorch wrapper. Rerun it after an OS update to see whether
Core ML's own silu op has become good enough to go back to.
"""

import collections

import coremltools
import numpy as np
import torch
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import silu as coremltools_silu_translation

from common.audio import load_audio
from common.compare import compare
from common.config import AUDIO_SAMPLES_PER_CHUNK, FRONTEND_CONV1_BUFFER_SHAPE, FRONTEND_CONV2_BUFFER_SHAPE, TEST_AUDIO_PATHS
from common.coreml_conversion import convert_to_coreml, print_compute_plan
from common.coreml_torch_ops import silu as silu_from_tanh
from step1_frontend.convert import INPUT_SHAPES
from step1_frontend.test import make_quiet_signals, pad_to_whole_chunks
from step1_frontend.wrapper import StreamingFrontend, build_streaming_frontend

SIGNAL_NAMES_TO_DIAGNOSE = ("silence", "noise_0.0001", "noise_0.001", "beckett")

# Both are named silu, which is how register_torch_op knows which torch op they translate.
SILU_RECIPES = {
    "Core ML silu op (current)": coremltools_silu_translation,
    "(x/2) * (1 + tanh(x/2))": silu_from_tanh,
}


class StreamingFrontendWithStageOutputs(StreamingFrontend):
    """Returns every stage's output, in run_stages order, instead of only the three Swift needs."""

    def forward(self, audio_chunk, conv1_buffer, conv2_buffer):
        return tuple(self.run_stages(audio_chunk, conv1_buffer, conv2_buffer).values())


def run_stages_chunk_by_chunk(run_chunk, padded_samples: np.ndarray) -> dict[str, np.ndarray]:
    """Each stage's output for every chunk, stacked to [chunk count, ...], with the buffers carried as in Swift.
    run_chunk(audio_chunk, conv1_buffer, conv2_buffer) returns a dict of stage outputs."""
    conv1_buffer = np.zeros(FRONTEND_CONV1_BUFFER_SHAPE, dtype=np.float32)
    conv2_buffer = np.zeros(FRONTEND_CONV2_BUFFER_SHAPE, dtype=np.float32)
    outputs_by_stage = collections.defaultdict(list)
    for audio_chunk in padded_samples.reshape(-1, 1, AUDIO_SAMPLES_PER_CHUNK):
        stage_outputs = run_chunk(audio_chunk, conv1_buffer, conv2_buffer)
        conv1_buffer, conv2_buffer = stage_outputs["conv1_buffer_out"], stage_outputs["conv2_buffer_out"]
        for stage_name, values in stage_outputs.items():
            outputs_by_stage[stage_name].append(values)
    return {stage_name: np.stack(values) for stage_name, values in outputs_by_stage.items()}


def main() -> None:
    torch.set_grad_enabled(False)
    probe = StreamingFrontendWithStageOutputs(build_streaming_frontend().embedder).eval()
    example_inputs = tuple(torch.randn(shape) for shape in INPUT_SHAPES.values())
    stage_names = list(probe.run_stages(*example_inputs).keys())
    reported_stage_names = [name for name in stage_names if not name.endswith("_buffer_out")]  # buffers are slices
    traced_probe = torch.jit.trace(probe, example_inputs)  # records aten::silu; the recipe is picked at conversion

    all_signals = {"beckett": pad_to_whole_chunks(load_audio(TEST_AUDIO_PATHS["beckett"]))} | make_quiet_signals()
    signals = {name: all_signals[name] for name in SIGNAL_NAMES_TO_DIAGNOSE}

    def run_pytorch_chunk(audio_chunk, conv1_buffer, conv2_buffer):
        stage_outputs = probe.run_stages(
            torch.from_numpy(audio_chunk), torch.from_numpy(conv1_buffer), torch.from_numpy(conv2_buffer)
        )
        return {stage_name: values.numpy() for stage_name, values in stage_outputs.items()}

    reference_by_signal = {name: run_stages_chunk_by_chunk(run_pytorch_chunk, samples) for name, samples in signals.items()}

    for recipe_name, silu_translation in SILU_RECIPES.items():
        register_torch_op(silu_translation, override=True)
        coreml_probe = convert_to_coreml(
            traced_probe, INPUT_SHAPES, stage_names, coremltools.precision.FLOAT16, coremltools.ComputeUnit.CPU_AND_NE
        )

        def run_coreml_chunk(audio_chunk, conv1_buffer, conv2_buffer):
            return coreml_probe.predict(
                {"audio_chunk": audio_chunk, "conv1_buffer": conv1_buffer, "conv2_buffer": conv2_buffer}
            )

        coreml_by_signal = {name: run_stages_chunk_by_chunk(run_coreml_chunk, samples) for name, samples in signals.items()}

        print(f"\n[SiLU = {recipe_name}] float16 on CPU_AND_NE vs float32 PyTorch, per stage: cosine (max abs diff)")
        print_compute_plan(coreml_probe, coremltools.ComputeUnit.CPU_AND_NE)
        print(f"  {'stage':<20}" + "".join(f"{name:>23}" for name in signals))
        for stage_name in reported_stage_names:
            row = f"  {stage_name:<20}"
            for signal_name in signals:
                result = compare(
                    f"{recipe_name} {signal_name} {stage_name}",
                    coreml_by_signal[signal_name][stage_name],
                    reference_by_signal[signal_name][stage_name],
                    print_result=False,
                )
                row += f"{result['cosine_similarity']:>13.6f} ({result['max_abs_diff']:.1e})"
            print(row)


if __name__ == "__main__":
    main()
