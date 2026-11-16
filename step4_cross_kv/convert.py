"""Step 4 conversion: the CrossAttentionKeyValueProjection wrapper to Core ML. Run from the top of the repo:

    python -m step4_cross_kv.convert

Writes two packages into models/, with the same inputs and outputs:
  models/cross_kv_float32.mlpackage  float32 math; only test.py uses it, to prove the conversion is exact
  models/cross_kv.mlpackage          float16 math; the one that ships to the watch
"""

import torch

from common.config import DECODER_DIM, MEMORY_FRAMES_PER_CALL, MODELS_DIRECTORY
from common.coreml_conversion import save_float32_and_float16_packages
from step4_cross_kv.wrapper import build_cross_attention_key_value_projection

# Other steps (and Step 6's pipeline) import these paths instead of building their own.
CROSS_KV_FLOAT32_MODEL_PATH = MODELS_DIRECTORY / "cross_kv_float32.mlpackage"
CROSS_KV_MODEL_PATH = MODELS_DIRECTORY / "cross_kv.mlpackage"

# The Architecture map's names and shapes. cross_kv.ort takes the same memory input, with any number of rows.
INPUT_SHAPES = {"memory": (1, MEMORY_FRAMES_PER_CALL, DECODER_DIM)}
# cross_kv.ort names its outputs k_cross and v_cross: the whole memory's keys and values. These are the new rows only.
OUTPUT_NAMES = ["k_new", "v_new"]


def main() -> None:
    wrapper = build_cross_attention_key_value_projection()
    # Only the shape matters: the graph has no branches, so any values trace the same ops.
    example_inputs = (torch.randn(INPUT_SHAPES["memory"]),)
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(wrapper, example_inputs)
        k_new, v_new = traced_wrapper(*example_inputs)
    print(f"[traced wrapper] memory {INPUT_SHAPES['memory']} -> k_new {tuple(k_new.shape)}, v_new {tuple(v_new.shape)}")
    save_float32_and_float16_packages(
        traced_wrapper,
        INPUT_SHAPES,
        OUTPUT_NAMES,
        CROSS_KV_FLOAT32_MODEL_PATH,
        CROSS_KV_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
