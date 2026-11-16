"""Step 2 conversion: the StreamingEncoder wrapper to Core ML. Run from the top of the repo:

    python -m step2_encoder.convert

Writes two packages into models/, with the same inputs and outputs:
  models/encoder_float32.mlpackage  float32 math; only test.py uses it, to prove the conversion is exact
  models/encoder.mlpackage          float16 math; the one that ships to the watch
"""

import torch

from common.config import ENCODER_DIM, ENCODER_WINDOW_FRAME_COUNT, MODELS_DIRECTORY
from common.coreml_conversion import save_float32_and_float16_packages
from step2_encoder.wrapper import build_streaming_encoder

# Other steps (and Step 6's pipeline) import these paths instead of building their own.
ENCODER_FLOAT32_MODEL_PATH = MODELS_DIRECTORY / "encoder_float32.mlpackage"
ENCODER_MODEL_PATH = MODELS_DIRECTORY / "encoder.mlpackage"

# The Architecture map's names and shapes, in the wrapper's argument order. valid is float32, not int32: it is a
# 0/1 multiplier in the mask math, so the whole graph stays float with no cast.
INPUT_SHAPES = {
    "features": (1, ENCODER_WINDOW_FRAME_COUNT, ENCODER_DIM),
    "valid": (1, ENCODER_WINDOW_FRAME_COUNT),
}
OUTPUT_NAMES = ["encoded"]


def main() -> None:
    wrapper = build_streaming_encoder()
    # Only the shapes matter: the graph doesn't branch on values, so any features and valid trace the same ops.
    example_inputs = (torch.randn(INPUT_SHAPES["features"]), torch.ones(INPUT_SHAPES["valid"]))
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(wrapper, example_inputs)
    save_float32_and_float16_packages(
        traced_wrapper, INPUT_SHAPES, OUTPUT_NAMES, ENCODER_FLOAT32_MODEL_PATH, ENCODER_MODEL_PATH
    )


if __name__ == "__main__":
    main()
