"""Why decoder_step.mlpackage doesn't compile for the Neural Engine. Converts the 5b decoder step with its state laid
out and written in different ways, and for each prints whether the Neural Engine compiles it, where its ops run on
CPU_AND_NE, and, if it compiles, whether it decodes beckett as the 5a wrapper does. Nothing is saved.
Run from the top of the repo, after step5_decoder.test has passed once (it needs ref/step4.npz and ref/step2.npz):

    python -m step5_decoder.tools.diagnose_state_on_neural_engine

The 5a model, the same math with the cache passed in and out, compiles for the Neural Engine; the 5b model as built
compiles for the GPU but not the Neural Engine (ANECCompile() FAILED). So the state is the suspect:
  - its last axis: 40 numbers per row (head_dim). One report (Apple Developer Forums thread 810987, unanswered) has a
    state run on the Neural Engine only when its last axis is a multiple of 32; 64 is the next one up from 40. Stateful
    caches known to run on the Neural Engine (ANEMLL's) have head_dim 64 or 128.
  - its rank: 5 axes, [6,1,8,128,40]. ANEMLL's are 4. Without the batch axis of 1 ours has 4, the same numbers in the
    same order.
  - the write:
      blend at the end (as built): the one-hot blend over the whole state as two updates, mul_ then add_, after every
        layer has read the cache blended with its new row. A first run of this tool: the Neural Engine compile fails
        in all 4 layouts, so only the as-built layout is kept here, to show the failure.
      slice at the end: one slice write of all 6 layers' new rows at row `position`, after every layer has read the
        blended cache.
      slice per layer, then read: as ANEMLL does, each layer writes its row into the state, then reads its cache back
        from the state for attention.
    (A whole-state copy_ doesn't convert: coremltools 9.0 raises "No matching select or slice". Slice writes must name
    every axis: with "...", coremltools 9.0 puts the slice on the first axis.)
Small stateful models can't answer this: Core ML puts all of their ops on the CPU, so the Neural Engine compiler never
sees them. Each variant here is the real decoder, about 15 s to convert.

What it showed on the Mac (2026-11-10, coremltools 9.0): the blend doesn't compile for the Neural Engine in any layout.
Slice writes compile in every layout, but with a last axis of 40 every prediction fails ("ANE inference operation
failed ... Program Inference error"); with 64 all four variants decode beckett with the wrapper's tokens, and their
scores match 5a's float16 package. Rank doesn't matter. Slice per layer, then read, was fastest (6.1 ms per step vs
7.5 for slice at the end): its state reads run on the Neural Engine. DecoderStepWithStatefulCache is that variant,
[6,1,8,128,64].

Step 7c (2026-11-10) added CPU_ONLY for every variant: on an Apple Watch Series 8, whose Neural Engine rejects the
decoder, Core ML runs DecoderStepWithStatefulCache on the CPU, and there (and on the Mac's CPU) the cache row it writes
isn't the row 5a computes (cosine about 0 after one step), so it decodes garbage. 5a decodes correctly on CPU_ONLY.
"""

import tempfile
from pathlib import Path

import coremltools
import torch
from torch import nn
from torch.nn import functional

import common.coreml_torch_ops  # noqa: F401  registers the tanh SiLU, for the MLPs
from common.config import DECODER_HEAD_DIM, DECODER_LAYER_COUNT
from common.coreml_conversion import convert_to_coreml, print_compute_plan
from step5_decoder.convert import (
    FLOAT16_PACKAGE_INPUT_DATA_TYPES,
    INPUT_DATA_TYPES,
    STATEFUL_INPUT_SHAPES,
    STATEFUL_OUTPUT_NAMES,
    example_inputs,
)
from step5_decoder.test import (
    ExplicitCacheSession,
    StatefulCacheSession,
    UnusedRows,
    compare_with_reference,
    coreml_stateful_predict,
    decode,
    load_test_segments,
    wrapper_predict,
)
from step5_decoder.wrapper import (
    DecoderStepWithExplicitCache,
    blend_new_row_into,
    build_decoder_step_with_explicit_cache,
)

STATE_LAYOUTS = {
    "[6,1,8,128,40], as built": (6, 1, 8, 128, 40),
    "[6,8,128,40], no batch axis": (6, 8, 128, 40),
    "[6,1,8,128,64], rows padded to 64": (6, 1, 8, 128, 64),
    "[6,8,128,64], both": (6, 8, 128, 64),
}
WRITE_BLEND_AT_END = "blend at the end (mul_ then add_), as built"
WRITE_SLICE_AT_END = "slice at the end"
WRITE_SLICE_PER_LAYER = "slice per layer, then read"
AS_BUILT_LAYOUT_NAME = "[6,1,8,128,40], as built"
SEGMENT_NAME = "beckett"


class StateLayoutVariant(nn.Module):
    """DecoderStepWithStatefulCache with its state in another layout, or written another way. run_step still works
    with caches laid out as [6,1,8,Lmax,40]; this class converts to and from the state's layout. Padding columns are
    written as zeros and never read."""

    def __init__(self, decoder_step: DecoderStepWithExplicitCache, state_shape: tuple[int, ...], write: str):
        super().__init__()
        self.decoder_step = decoder_step
        self.has_batch_axis = len(state_shape) == 5
        self.padding_count = state_shape[-1] - DECODER_HEAD_DIM
        self.write = write
        self.register_buffer("k_self", torch.zeros(state_shape))
        self.register_buffer("v_self", torch.zeros(state_shape))

    def as_run_step_layout(self, state: torch.Tensor) -> torch.Tensor:
        cache = state[..., :DECODER_HEAD_DIM]
        return cache if self.has_batch_axis else cache.unsqueeze(1)

    def as_state_layout(self, rows: torch.Tensor) -> torch.Tensor:
        """[layers,1,8,1,40] -> the same rows in the state's layout."""
        rows = rows if self.has_batch_axis else rows.squeeze(1)
        return functional.pad(rows, (0, self.padding_count)) if self.padding_count else rows

    def write_rows(self, state: torch.Tensor, layers: slice, rows: torch.Tensor, position) -> None:
        """Writes rows [layers,1,8,1,40] into state at row `position`, for the given layers. torch.jit.trace keeps
        position a runtime value here (aten::Int, then aten::slice); coremltools turns it into slice_update. The layers
        are plain numbers: a traced shape read (rows.shape[0]) doesn't convert (Step 4, deviation 1)."""
        rows = self.as_state_layout(rows)
        if self.has_batch_axis:
            state[layers, :, :, position:position + 1, :] = rows
        else:
            state[layers, :, position:position + 1, :] = rows

    def layer_cache(self, state: torch.Tensor, layer_index: int) -> torch.Tensor:
        """One layer's cache read from the state, [1,8,Lmax,40]."""
        return self.as_run_step_layout(state[layer_index:layer_index + 1])[0]

    def forward(self, token, position, k_cross, v_cross, cross_valid):
        if self.write == WRITE_SLICE_PER_LAYER:
            def write_then_read(layer_index, key, value):
                this_layer = slice(layer_index, layer_index + 1)
                self.write_rows(self.k_self, this_layer, key[None], position)
                self.write_rows(self.v_self, this_layer, value[None], position)
                return self.layer_cache(self.k_self, layer_index), self.layer_cache(self.v_self, layer_index)

            return self.decoder_step.run_step(token, position, k_cross, v_cross, cross_valid, write_then_read).logits

        is_new_row, is_kept_row = self.decoder_step.row_weights(position)
        step = self.decoder_step.run_step(
            token, position, k_cross, v_cross, cross_valid,
            blend_new_row_into(
                self.as_run_step_layout(self.k_self), self.as_run_step_layout(self.v_self), is_new_row, is_kept_row
            ),
        )
        if self.write == WRITE_BLEND_AT_END:
            self.k_self.mul_(is_kept_row).add_(self.as_state_layout(step.new_keys) * is_new_row)
            self.v_self.mul_(is_kept_row).add_(self.as_state_layout(step.new_values) * is_new_row)
        else:
            every_layer = slice(0, DECODER_LAYER_COUNT)
            self.write_rows(self.k_self, every_layer, step.new_keys, position)
            self.write_rows(self.v_self, every_layer, step.new_values, position)
        return step.logits


def diagnose(variant: StateLayoutVariant, state_shape, segment, wrapper_transcript, model_path: Path) -> None:
    with torch.no_grad():
        traced_variant = torch.jit.trace(variant, example_inputs(STATEFUL_INPUT_SHAPES))
    try:
        # Converted for the CPU only, so conversion itself doesn't compile for the Neural Engine; tried below.
        coreml_model = convert_to_coreml(
            traced_variant,
            STATEFUL_INPUT_SHAPES,
            STATEFUL_OUTPUT_NAMES,
            coremltools.precision.FLOAT16,
            coremltools.ComputeUnit.CPU_ONLY,
            input_data_types={**INPUT_DATA_TYPES, **FLOAT16_PACKAGE_INPUT_DATA_TYPES},
            state_shapes={"k_self": state_shape, "v_self": state_shape},
        )
    except Exception as error:  # noqa: BLE001  any failure is a result here
        print(f"  conversion FAILED: {first_line(error)}")
        return
    coreml_model.save(str(model_path))
    # Step 7c: CPU_ONLY too, whether or not the Neural Engine compiles the variant (an Apple Watch Series 8's Neural
    # Engine rejects the decoder, so Core ML runs it on the CPU, where the as-built state gives garbage).
    decode_and_compare(model_path, coremltools.ComputeUnit.CPU_ONLY, segment, wrapper_transcript)
    try:
        print_compute_plan(coreml_model, coremltools.ComputeUnit.CPU_AND_NE)
    except Exception as error:  # noqa: BLE001
        print(f"  Neural Engine compile FAILED: {first_line(error)}")
        return
    decode_and_compare(model_path, coremltools.ComputeUnit.CPU_AND_NE, segment, wrapper_transcript)


def decode_and_compare(model_path: Path, compute_units, segment, wrapper_transcript) -> None:
    """Decodes the segment with the saved variant on these compute units, from a fresh state, vs the 5a wrapper."""
    coreml_model = coremltools.models.MLModel(str(model_path), compute_units=compute_units)
    float16 = FLOAT16_PACKAGE_INPUT_DATA_TYPES["k_cross"]
    try:
        compare_with_reference(
            f"  {segment.name} on {compute_units.name} vs the 5a wrapper",
            StatefulCacheSession(
                coreml_stateful_predict(coreml_model, coreml_model.make_state()), segment, float16, UnusedRows(),
            ),
            wrapper_transcript,
            segment.token_cap,
        )
    except Exception as error:  # noqa: BLE001
        print(f"  decoding on {compute_units.name} FAILED: {first_line(error)}")


def first_line(error: Exception) -> str:
    text = " ".join(str(error).split())  # Core ML's messages span several lines
    return text[:300]


def main() -> None:
    torch.set_grad_enabled(False)
    decoder_step = build_decoder_step_with_explicit_cache()
    segment = next(segment for segment in load_test_segments() if segment.name == SEGMENT_NAME)
    wrapper_transcript = decode(ExplicitCacheSession(wrapper_predict(decoder_step), segment), segment.token_cap)
    print(f"[the 5a wrapper on {segment.name}: {wrapper_transcript.describe()}]")
    variants = [(AS_BUILT_LAYOUT_NAME, WRITE_BLEND_AT_END)] + [
        (layout_name, write)
        for write in (WRITE_SLICE_AT_END, WRITE_SLICE_PER_LAYER)
        for layout_name in STATE_LAYOUTS
    ]
    with tempfile.TemporaryDirectory() as models_directory:
        for variant_index, (layout_name, write) in enumerate(variants):
            print(f"[state {layout_name}; write: {write}]")
            state_shape = STATE_LAYOUTS[layout_name]
            variant = StateLayoutVariant(decoder_step, state_shape, write).eval()
            model_path = Path(models_directory) / f"variant_{variant_index}.mlpackage"
            diagnose(variant, state_shape, segment, wrapper_transcript, model_path)


if __name__ == "__main__":
    main()
