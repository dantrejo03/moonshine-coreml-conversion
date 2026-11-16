"""Two facts about the adapter's position table (pos_emb), read from the files; nothing runs on Core ML.
Run from conversion/:

    python -m step3_adapter.tools.inspect_position_table

1. adapter.ort's whole graph and its stored table. It slices rows pos_offset ..< pos_offset + n from a uint8 copy of
   the table, dequantizes them with one scale for the whole table, and adds them to encoded. Prints the stored scale
   next to the one step3_adapter.test computes for check (c)'s bar.
2. Which rows were trained. A trained row is large and similar to its neighbors: positions 20 ms apart mean nearly
   the same thing. An untrained row is still the small random value the table started with, unrelated to its
   neighbors. Prints both measures for every 4 s of positions, and where the untrained rows begin.
"""

import numpy as np

from common.config import ADAPTER_TRAINED_POSITION_COUNT, FEATURE_FRAME_SECONDS
from common.load_model import load_pytorch_model
from common.ort_graph import load_ort_graph, print_graph, stored_tensor_values
from step3_adapter.test import adapter_ort_quantization_step

ROWS_PER_PRINTED_BLOCK = 200  # 4 s of positions


def inspect_adapter_ort(position_table: np.ndarray) -> None:
    print("[1. adapter.ort]")
    initializers_by_name = print_graph(load_ort_graph("adapter.ort"))
    scale = stored_tensor_values(initializers_by_name["pos_embed.weight_scale"]).item()
    zero_point = int(stored_tensor_values(initializers_by_name["pos_embed.weight_zero_point"]).item())
    stored_table = stored_tensor_values(initializers_by_name["pos_embed.weight_quantized"])
    stored_table = stored_table.reshape(position_table.shape)
    dequantized_table = (stored_table.astype(np.float32) - zero_point) * scale
    print(
        f"  stored scale {scale:.7g}, zero point {zero_point}; step3_adapter.test computes "
        f"{adapter_ort_quantization_step(position_table):.7g} from the float table"
    )
    absolute_error = np.abs(dequantized_table - position_table)
    print(
        f"  stored table vs float table: max abs {absolute_error.max():.6g} (half a step is {scale / 2:.6g}), "
        f"mean abs {absolute_error.mean():.6g}"
    )


def inspect_trained_rows(position_table: np.ndarray) -> None:
    print("[2. which rows of pos_emb were trained]")
    row_count = len(position_table)
    row_norms = np.linalg.norm(position_table, axis=1)
    cosine_to_next_row = (position_table[:-1] * position_table[1:]).sum(axis=1) / (row_norms[:-1] * row_norms[1:])
    for first_row in range(0, row_count, ROWS_PER_PRINTED_BLOCK):
        rows = slice(first_row, min(first_row + ROWS_PER_PRINTED_BLOCK, row_count))
        print(
            f"  rows {rows.start:4d}-{rows.stop - 1:4d} ({rows.start * FEATURE_FRAME_SECONDS:5.1f} s): "
            f"norm mean {row_norms[rows].mean():.3f}, min {row_norms[rows].min():.3f}; "
            f"cosine to next row {cosine_to_next_row[rows].mean():+.3f}"
        )

    # The untrained tail: everything after the last row whose norm is clearly above the table's final rows.
    tail_median_norm = float(np.median(row_norms[-ROWS_PER_PRINTED_BLOCK:]))
    first_untrained_row = int(np.flatnonzero(row_norms >= 2 * tail_median_norm)[-1]) + 1
    print(
        f"  last row with norm over twice the final rows' median ({tail_median_norm:.3f}): {first_untrained_row - 1}. "
        f"From row {first_untrained_row} ({first_untrained_row * FEATURE_FRAME_SECONDS:.2f} s) on: values' standard "
        f"deviation {position_table[first_untrained_row:].std():.4f}, cosine to next row "
        f"{cosine_to_next_row[first_untrained_row:].mean():+.3f}"
    )
    nearby_rows = range(first_untrained_row - 3, first_untrained_row + 3)
    print("  norms around it: " + ", ".join(f"row {row} {row_norms[row]:.3f}" for row in nearby_rows))
    verdict = "matches" if first_untrained_row == ADAPTER_TRAINED_POSITION_COUNT else "DIFFERS FROM"
    print(f"  {verdict} common.config.ADAPTER_TRAINED_POSITION_COUNT = {ADAPTER_TRAINED_POSITION_COUNT}")


def main() -> None:
    position_table = load_pytorch_model().model.decoder.pos_emb.weight.detach().numpy()
    print(f"pos_emb {position_table.shape} float32, from the checkpoint")
    inspect_adapter_ort(position_table)
    inspect_trained_rows(position_table)


if __name__ == "__main__":
    main()
