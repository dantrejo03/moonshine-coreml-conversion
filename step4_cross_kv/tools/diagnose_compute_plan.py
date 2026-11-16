"""Why cross_kv.mlpackage runs entirely on the CPU. Converts variants of the wrapper to float16 and prints each one's
compute plan for CPU_AND_NE (what Step 7's app loads). Nothing is saved. Run from conversion/:

    python -m step4_cross_kv.tools.diagnose_compute_plan

Every variant does the same 12 projections (wrapper.project_each_layer). They differ in how the 6 layers' results are
joined, and in how many memory rows go in, which separates three suspects:
  - the stack op: "unsqueeze + concat" gives the same 5-D result another way (cross_kv.ort's way)
  - 5-D tensors: "concat on the batch axis" gives [6,8,rows,40], the same numbers in the same memory order, 4-D
  - too little work: "first layer only" is a sixth of it; and 116 rows, the size of Step 2's encoder window, which
    ran on the Neural Engine
"""

import coremltools
import torch
from torch import nn

from common.config import DECODER_DIM, ENCODER_WINDOW_FRAME_COUNT, MEMORY_FRAMES_PER_CALL
from common.coreml_conversion import convert_to_coreml, print_compute_plan
from step4_cross_kv.convert import OUTPUT_NAMES
from step4_cross_kv.wrapper import CrossAttentionKeyValueProjection, build_cross_attention_key_value_projection

LAYER_JOINS = {
    "stack (as built)": torch.stack,
    "unsqueeze + concat": lambda per_layer: torch.cat([tensor.unsqueeze(0) for tensor in per_layer]),
    "concat on the batch axis": lambda per_layer: torch.cat(per_layer),
    "first layer only": lambda per_layer: per_layer[0],
}


class LayerJoinVariant(nn.Module):
    def __init__(self, wrapper: CrossAttentionKeyValueProjection, join_layers):
        super().__init__()
        self.wrapper = wrapper
        self.join_layers = join_layers

    def forward(self, memory):
        keys_per_layer, values_per_layer = self.wrapper.project_each_layer(memory)
        return self.join_layers(keys_per_layer), self.join_layers(values_per_layer)


def main() -> None:
    torch.set_grad_enabled(False)
    wrapper = build_cross_attention_key_value_projection()
    for memory_row_count in (MEMORY_FRAMES_PER_CALL, ENCODER_WINDOW_FRAME_COUNT):
        for join_name, join_layers in LAYER_JOINS.items():
            example_memory = torch.randn(1, memory_row_count, DECODER_DIM)
            traced_variant = torch.jit.trace(LayerJoinVariant(wrapper, join_layers).eval(), (example_memory,))
            keys, _ = traced_variant(example_memory)
            coreml_model = convert_to_coreml(
                traced_variant,
                {"memory": tuple(example_memory.shape)},
                OUTPUT_NAMES,
                coremltools.precision.FLOAT16,
                coremltools.ComputeUnit.CPU_AND_NE,
            )
            print(f"[{memory_row_count} rows, {join_name}: outputs {tuple(keys.shape)}]")
            print_compute_plan(coreml_model, coremltools.ComputeUnit.CPU_AND_NE)


if __name__ == "__main__":
    main()
