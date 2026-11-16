"""Step 5 conversion: the decoder step wrappers to Core ML. Run from the top of the repo:

    python -m step5_decoder.convert

Writes three packages into models/:
  Part 5a, DecoderStepWithExplicitCache: the self-attention cache passed in and out. Same input and output names and
  shapes in both; only test.py uses them.
    models/decoder_step_explicit_cache_float32.mlpackage  float32 math; proves the conversion is exact
    models/decoder_step_explicit_cache.mlpackage          float16 math, k_cross and v_cross taken as float16
  Part 5b, DecoderStepWithStatefulCache: the self-attention cache inside the model as Core ML state (MLState).
    models/decoder_step.mlpackage                         float16 math, like 5a's float16 package; ships to the watch.
                                                          Only float16: Core ML state is float16 either way.
"""

import coremltools
import numpy as np
import torch

import common.coreml_torch_ops  # noqa: F401  registers the tanh SiLU, for the MLPs (common/coreml_torch_ops.py)
from common.config import (
    DECODER_PADDED_HEAD_DIM,
    MAXIMUM_MEMORY_FRAME_COUNT,
    MAXIMUM_TOKEN_COUNT,
    MODELS_DIRECTORY,
    attention_cache_shape,
)
from common.coreml_conversion import print_compute_plans, save_float32_and_float16_packages, save_package
from step5_decoder.wrapper import DecoderStepWithStatefulCache, build_decoder_step_with_explicit_cache

# Other steps (and Step 6's pipeline) import these paths instead of building their own.
EXPLICIT_CACHE_FLOAT32_MODEL_PATH = MODELS_DIRECTORY / "decoder_step_explicit_cache_float32.mlpackage"
EXPLICIT_CACHE_MODEL_PATH = MODELS_DIRECTORY / "decoder_step_explicit_cache.mlpackage"
DECODER_STEP_MODEL_PATH = MODELS_DIRECTORY / "decoder_step.mlpackage"  # 5b: what Step 6 and the app use

# The Architecture map's names and shapes, in the wrapper's argument order. decoder_kv.ort's differ: its caches grow
# (k_self [6,1,8,L,40], its position is L), and it names its cross K/V inputs out_k_cross and out_v_cross.
INPUT_SHAPES = {
    "token": (1, 1),
    "position": (1,),
    "k_self": attention_cache_shape(MAXIMUM_TOKEN_COUNT),
    "v_self": attention_cache_shape(MAXIMUM_TOKEN_COUNT),
    "k_cross": attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT),
    "v_cross": attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT),
    "cross_valid": (1, MAXIMUM_MEMORY_FRAME_COUNT),
}
INPUT_DATA_TYPES = {"token": np.int32, "position": np.int32}  # integers are int32 (plan rule); the rest float32
# The watch package reads Swift's float16 cross K/V buffers as they are: 1.47 million numbers each, on every call.
FLOAT16_PACKAGE_INPUT_DATA_TYPES = {"k_cross": np.float16, "v_cross": np.float16}
OUTPUT_NAMES = ["logits", "k_self_out", "v_self_out"]  # in the wrapper's return order

# 5b: k_self and v_self move from inputs and outputs into state, named as the wrapper's buffers, each row padded to 64
# numbers for the Neural Engine (DecoderStepWithStatefulCache says why). Its k_cross and v_cross rows are padded to 64
# as well (common.config.DECODER_PADDED_HEAD_DIM); 5a's stay 40 wide.
STATE_SHAPES = {
    name: attention_cache_shape(MAXIMUM_TOKEN_COUNT, DECODER_PADDED_HEAD_DIM) for name in ("k_self", "v_self")
}
STATEFUL_INPUT_SHAPES = {  # overriding keeps INPUT_SHAPES' order, the wrapper's argument order
    **{name: shape for name, shape in INPUT_SHAPES.items() if name not in STATE_SHAPES},
    **{
        name: attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT, DECODER_PADDED_HEAD_DIM)
        for name in ("k_cross", "v_cross")
    },
}
STATEFUL_OUTPUT_NAMES = ["logits"]


def example_inputs(input_shapes: dict[str, tuple[int, ...]]) -> tuple[torch.Tensor, ...]:
    """In input_shapes' order. Only the shapes and types matter: nothing in forward branches on values, so any values
    trace the same ops."""
    example_values = {
        "token": torch.tensor([[1]], dtype=torch.int32),  # the start token
        "position": torch.tensor([3], dtype=torch.int32),
        "cross_valid": torch.ones(INPUT_SHAPES["cross_valid"]),
    }
    return tuple(example_values.get(name, torch.randn(shape)) for name, shape in input_shapes.items())


def trace(wrapper, input_shapes: dict[str, tuple[int, ...]], output_names: list[str]) -> torch.jit.ScriptModule:
    traced_inputs = example_inputs(input_shapes)
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(wrapper, traced_inputs)
        outputs = traced_wrapper(*traced_inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    output_shapes = ", ".join(f"{name} {tuple(output.shape)}" for name, output in zip(output_names, outputs))
    print(f"[traced {type(wrapper).__name__}] -> {output_shapes}")
    return traced_wrapper


def main() -> None:
    explicit_cache_wrapper = build_decoder_step_with_explicit_cache()
    save_float32_and_float16_packages(
        trace(explicit_cache_wrapper, INPUT_SHAPES, OUTPUT_NAMES),
        INPUT_SHAPES,
        OUTPUT_NAMES,
        EXPLICIT_CACHE_FLOAT32_MODEL_PATH,
        EXPLICIT_CACHE_MODEL_PATH,
        input_data_types=INPUT_DATA_TYPES,
        float16_package_input_data_types=FLOAT16_PACKAGE_INPUT_DATA_TYPES,
    )

    # Tracing runs forward, which writes into the buffers; that doesn't matter here, since the state's contents
    # aren't part of the package.
    stateful_wrapper = DecoderStepWithStatefulCache(explicit_cache_wrapper).eval()
    stateful_model = save_package(
        "5b, cache in state: float16, all compute units: ships to the watch",
        trace(stateful_wrapper, STATEFUL_INPUT_SHAPES, STATEFUL_OUTPUT_NAMES),
        STATEFUL_INPUT_SHAPES,
        STATEFUL_OUTPUT_NAMES,
        DECODER_STEP_MODEL_PATH,
        coremltools.precision.FLOAT16,
        coremltools.ComputeUnit.ALL,
        input_data_types={**INPUT_DATA_TYPES, **FLOAT16_PACKAGE_INPUT_DATA_TYPES},
        state_shapes=STATE_SHAPES,
    )
    print_compute_plans(stateful_model)


if __name__ == "__main__":
    main()
