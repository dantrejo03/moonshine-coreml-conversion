"""Step 1 conversion: the StreamingFrontend wrapper to Core ML. Run from the top of the repo:

    python -m step1_frontend.convert

Writes two packages into models/, with the same inputs and outputs:
  models/frontend_float32.mlpackage  float32 math; only test.py uses it, to prove the conversion is exact
  models/frontend.mlpackage          float16 math; the one that ships to the watch
"""

import torch
from coremltools.converters.mil import Builder as mil_builder
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

import common.coreml_torch_ops  # noqa: F401  registers the tanh SiLU (common/coreml_torch_ops.py says why)
from common.config import (
    AUDIO_SAMPLES_PER_CHUNK,
    FRONTEND_CONV1_BUFFER_SHAPE,
    FRONTEND_CONV2_BUFFER_SHAPE,
    MODELS_DIRECTORY,
)
from common.coreml_conversion import save_float32_and_float16_packages
from step1_frontend.wrapper import build_streaming_frontend

# Other steps (and Step 6's pipeline) import these paths instead of building their own.
FRONTEND_FLOAT32_MODEL_PATH = MODELS_DIRECTORY / "frontend_float32.mlpackage"
FRONTEND_MODEL_PATH = MODELS_DIRECTORY / "frontend.mlpackage"

# The Architecture map's names and shapes (also frontend.ort's names), in the wrapper's argument order.
INPUT_SHAPES = {
    "audio_chunk": (1, AUDIO_SAMPLES_PER_CHUNK),
    "conv1_buffer": FRONTEND_CONV1_BUFFER_SHAPE,
    "conv2_buffer": FRONTEND_CONV2_BUFFER_SHAPE,
}
OUTPUT_NAMES = ["features", "conv1_buffer_out", "conv2_buffer_out"]  # in the wrapper's return order


@register_torch_op(override=True)
def asinh(context, node):  # the registry matches torch ops by function name, so it must be called asinh
    """Replaces coremltools 9.0's translation of torch.asinh, which emits a Core ML op (asinh) that doesn't exist.

    Built from ops Core ML has: asinh(y) = sign(y) * log(|y| + sqrt(y^2 + 1)). Working on |y| avoids the
    cancellation log(y + sqrt(y^2 + 1)) suffers for negative y in float16. Here y = k * CMVN output, with
    learned k = 0.645 and |CMVN output| <= sqrt(80), so |y| <= 5.8 and y^2 + 1 can't overflow float16.
    """
    values = _get_inputs(context, node, expected=1)[0]
    square_root = mil_builder.sqrt(x=mil_builder.add(x=mil_builder.mul(x=values, y=values), y=1.0))
    magnitude = mil_builder.log(x=mil_builder.add(x=mil_builder.abs(x=values), y=square_root))
    context.add(mil_builder.mul(x=mil_builder.sign(x=values), y=magnitude, name=node.name))


def main() -> None:
    wrapper = build_streaming_frontend()
    example_inputs = tuple(torch.randn(shape) for shape in INPUT_SHAPES.values())  # only the shapes matter
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(wrapper, example_inputs)
    save_float32_and_float16_packages(
        traced_wrapper, INPUT_SHAPES, OUTPUT_NAMES, FRONTEND_FLOAT32_MODEL_PATH, FRONTEND_MODEL_PATH
    )


if __name__ == "__main__":
    main()
