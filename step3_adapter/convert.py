"""Step 3 conversion: the StreamingAdapter wrapper to Core ML. Run from the top of the repo:

    python -m step3_adapter.convert

Writes two packages into models/, with the same inputs and outputs:
  models/adapter_float32.mlpackage  float32 math; only test.py uses it, to prove the conversion is exact
  models/adapter.mlpackage          float16 math; the one that ships to the watch
"""

import numpy as np
import torch

from common.config import ENCODER_DIM, MEMORY_FRAMES_PER_CALL, MODELS_DIRECTORY
from common.coreml_conversion import save_float32_and_float16_packages
from step3_adapter.wrapper import build_streaming_adapter

# Other steps (and Step 6's pipeline) import these paths instead of building their own.
ADAPTER_FLOAT32_MODEL_PATH = MODELS_DIRECTORY / "adapter_float32.mlpackage"
ADAPTER_MODEL_PATH = MODELS_DIRECTORY / "adapter.mlpackage"

# The Architecture map's names and shapes (also adapter.ort's names), in the wrapper's argument order.
INPUT_SHAPES = {
    "encoded": (1, MEMORY_FRAMES_PER_CALL, ENCODER_DIM),
    "pos_offset": (1,),
}
# pos_offset picks table rows, so it's an integer: int32, the plan's rule for Core ML (adapter.ort takes int64).
INPUT_DATA_TYPES = {"pos_offset": np.int32}
OUTPUT_NAMES = ["memory"]


def main() -> None:
    wrapper = build_streaming_adapter()
    # Only the shapes and types matter: the graph doesn't branch on values, so any encoded and pos_offset trace the
    # same ops. The rows to read are looked up when the model runs, not baked in from this example.
    example_inputs = (torch.randn(INPUT_SHAPES["encoded"]), torch.zeros(INPUT_SHAPES["pos_offset"], dtype=torch.int32))
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(wrapper, example_inputs)
    save_float32_and_float16_packages(
        traced_wrapper,
        INPUT_SHAPES,
        OUTPUT_NAMES,
        ADAPTER_FLOAT32_MODEL_PATH,
        ADAPTER_MODEL_PATH,
        INPUT_DATA_TYPES,
    )


if __name__ == "__main__":
    main()
