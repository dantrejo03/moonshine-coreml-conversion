"""Checks that everything the conversion steps rely on is installed, downloaded and working.

Run from the top of the repo with the venv's Python. Exits non-zero if any check fails.
Shapes of the .ort pieces are deliberately not printed here; Step 1 does that.
"""

import sys
import traceback

import coremltools
import numpy as np
import onnxruntime
import torch
import transformers
from coremltools._deps import _TORCH_MAX_VERSION as COREMLTOOLS_MAX_TESTED_TORCH_VERSION
from packaging.version import Version

from common.audio import load_audio
from common.compare import compare
from common.config import (
    MOONSHINE_ORT_MODEL_DIRECTORY,
    MOONSHINE_ORT_PIECE_FILE_NAMES,
    SAMPLE_RATE_HZ,
    TEST_AUDIO_PATHS,
    TOKENIZER_PATH,
    load_streaming_config,
)
from common.load_model import load_pytorch_model

GIT_LFS_POINTER_MAXIMUM_BYTES = 1024


def check_versions() -> None:
    print(
        f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
        f"coremltools {coremltools.__version__}, onnxruntime {onnxruntime.__version__}, numpy {np.__version__}"
    )
    torch_release = Version(torch.__version__).release
    if torch_release > Version(COREMLTOOLS_MAX_TESTED_TORCH_VERSION).release:
        raise RuntimeError(
            f"torch {torch.__version__} is newer than {COREMLTOOLS_MAX_TESTED_TORCH_VERSION}, "
            f"the newest version coremltools {coremltools.__version__} is tested with"
        )


def check_coremltools_features() -> None:
    for feature_name, is_present in {
        "ct.target.watchOS11": hasattr(coremltools.target, "watchOS11"),
        "ct.target.watchOS26": hasattr(coremltools.target, "watchOS26"),
        "ct.StateType": hasattr(coremltools, "StateType"),
        "ct.EnumeratedShapes": hasattr(coremltools, "EnumeratedShapes"),
    }.items():
        if not is_present:
            raise RuntimeError(f"{feature_name} is missing")


def check_coreml_round_trip() -> None:
    """Convert a tiny model for watchOS 11 and run it through Core ML on this Mac."""

    class TinyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 4)

        def forward(self, values):
            return self.linear(values)

    pytorch_model = TinyLinear().eval()
    example_input = torch.randn(1, 8)
    coreml_model = coremltools.convert(
        torch.jit.trace(pytorch_model, example_input),
        inputs=[coremltools.TensorType(name="values", shape=(1, 8))],
        outputs=[coremltools.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=coremltools.target.watchOS11,
        compute_precision=coremltools.precision.FLOAT32,
        compute_units=coremltools.ComputeUnit.CPU_ONLY,
    )
    coreml_output = coreml_model.predict({"values": example_input.numpy()})["output"]
    with torch.no_grad():
        result = compare("  Core ML vs PyTorch", coreml_output, pytorch_model(example_input), maximum_allowed_max_abs_diff=1e-5)
    if not result["passed"]:
        raise RuntimeError("Core ML output does not match PyTorch")


def check_coreml_state() -> None:
    """MLState (watchOS 11): a buffer that keeps its value between predictions."""

    class Accumulator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_total", torch.zeros(1, 4, dtype=torch.float16))

        def forward(self, values):
            self.running_total += values
            return self.running_total * 1.0

    coreml_model = coremltools.convert(
        # check_trace=False: the trace check reruns forward, and the output changes on every run by design.
        torch.jit.trace(Accumulator().eval(), torch.ones(1, 4, dtype=torch.float16), check_trace=False),
        inputs=[coremltools.TensorType(name="values", shape=(1, 4), dtype=np.float16)],
        outputs=[coremltools.TensorType(name="output", dtype=np.float16)],
        states=[coremltools.StateType(wrapped_type=coremltools.TensorType(shape=(1, 4)), name="running_total")],
        convert_to="mlprogram",
        minimum_deployment_target=coremltools.target.watchOS11,
    )
    state = coreml_model.make_state()
    values = np.ones((1, 4), dtype=np.float16)
    coreml_model.predict({"values": values}, state=state)
    second_output = coreml_model.predict({"values": values}, state=state)["output"]
    if not np.allclose(second_output, 2.0):
        raise RuntimeError(f"state did not carry over: expected all 2.0, got {second_output}")


def check_moonshine_ort_pieces() -> None:
    for file_name in MOONSHINE_ORT_PIECE_FILE_NAMES:
        path = MOONSHINE_ORT_MODEL_DIRECTORY / file_name
        if not path.is_file():
            raise FileNotFoundError(f"{path} (run setup_environment.sh)")
        if path.stat().st_size <= GIT_LFS_POINTER_MAXIMUM_BYTES:
            raise RuntimeError(f"{path} is a Git LFS pointer, not a model")
        onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        print(f"  {file_name}: {path.stat().st_size / 1e6:.2f} MB, loads in onnxruntime")
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(TOKENIZER_PATH)
    streaming_config = load_streaming_config()
    print(f"  streaming_config.json: {len(streaming_config)} keys, vocab_size {streaming_config['vocab_size']}")


def check_test_audio() -> None:
    for name, path in TEST_AUDIO_PATHS.items():
        samples = load_audio(path)
        print(f"  {name}: {len(samples) / SAMPLE_RATE_HZ:.2f} s, peak {np.abs(samples).max():.3f}")


def check_pytorch_model() -> None:
    model = load_pytorch_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    sliding_windows = model.config.encoder_config.sliding_windows
    print(f"  {parameter_count / 1e6:.2f}M parameters, encoder sliding_windows {sliding_windows}")


def main() -> int:
    checks = [
        ("versions", check_versions),
        ("coremltools features", check_coremltools_features),
        ("Core ML round trip (float32, CPU only, watchOS 11)", check_coreml_round_trip),
        ("Core ML MLState (float16, watchOS 11)", check_coreml_state),
        ("Moonshine .ort reference pieces", check_moonshine_ort_pieces),
        ("test audio", check_test_audio),
        ("PyTorch model", check_pytorch_model),
    ]
    failed_check_names = []
    for name, check in checks:
        print(f"[{name}]")
        try:
            check()
            print("  OK")
        except Exception:
            traceback.print_exc()
            print("  FAILED")
            failed_check_names.append(name)
    print("\nAll checks passed." if not failed_check_names else f"\nFailed: {', '.join(failed_check_names)}")
    return 1 if failed_check_names else 0


if __name__ == "__main__":
    sys.exit(main())
