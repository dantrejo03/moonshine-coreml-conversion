# Step 3: Adapter

The adapter tells the decoder **when** each encoded frame happened. The encoder (Step 2) has no position information at all; the adapter adds a learned position code to each frame, which turns encoded frames into **memory rows**, what the decoder reads when it checks the audio. It is the only place where absolute position enters the model. It is the third of the five Core ML models. It reads Step 2's kept frames, and Step 4 (cross K/V) reads its output.

## One call

| | Name | Shape | Type | What it is |
|---|---|---|---|---|
| in | `encoded` | [1, 4, 320] | float32 | 4 of Step 2's kept frames, in order |
| in | `pos_offset` | [1] | int32 | the first frame's position in the segment: how many memory rows came before it |
| out | `memory` | [1, 4, 320] | float32 | the same 4 frames, each with its position's row added |

Every shape is fixed.

## What it computes

```
memory[i] = encoded[i] + pos_emb[pos_offset + i]      for i = 0 ..< 4
```

`pos_emb` is a learned table: one row of 320 numbers per position, one position per 20 ms frame since the segment started. The original also applies `proj`, a linear layer for models whose encoder and decoder differ in width. In tiny both are 320, so `proj` is `nn.Identity` and changes nothing.

The table has 4096 rows (81.92 s), but only rows 0–1999, the first 40 s, were trained. Rows 2000–4095 are still the small random values the table started with (`tools/inspect_position_table.py`). So a segment can't usefully run past 40 s; Mmax (768 frames, 15.36 s) is well inside.

## The state Swift carries between calls

- **On press:** `pos_offset = 0`.
- **On every encoder call:** Step 2 hands over its kept frames: none for the first 4 chunks, then 4 per chunk, and 20 on release (fewer if the recording is shorter than 5 chunks). Always a multiple of 4, so split them into groups of 4, in order:

  ```
  for each group of 4 frames:
      memory = adapter(encoded: group, pos_offset: [pos_offset])
      append memory's 4 rows to the segment's memory       // Step 4 projects them for the decoder
      pos_offset += 4
  ```

- `pos_offset` always equals the number of memory rows so far, which is also Step 2's `emitted`.
- **Nothing in the model checks `pos_offset`.** Past 1996 a call reads untrained rows, and past 4092 it reads outside the table, with no error either way. Swift's segment cap (Mmax) keeps it at 764 or below; never raise Mmax past 2000.

## Where it comes from

- **PyTorch:** the first lines of `MoonshineStreamingDecoder.forward` (`model.model.decoder`) in Transformers' `modeling_moonshine_streaming.py`: `pos_emb(arange(M))` added to the whole recording's encoder output, then `proj`. It adds with `+=`, changing the tensor it was given; calling the decoder twice on the same tensor adds the positions twice. The wrapper adds 4 rows per call, starting at `pos_offset`, and leaves its input alone.
- **Moonshine's runtime:** the adapter part of `encode` in `moonshine-streaming-model.cpp`. After each encoder call it runs `adapter.ort` on all the new frames with `adapter_pos_offset`, appends the result to `memory`, then advances `adapter_pos_offset` by the number of new frames. `adapter.ort` takes any number of frames and an int64 `pos_offset`, and stores the table as 8-bit integers with one scale (0.0131), so its memory differs from this model's by at most 0.0066 per value.

## Files

- `wrapper.py`: this stage in PyTorch, 4 frames per call
- `convert.py`: writes `models/adapter.mlpackage` (float16, ships to the watch) and `models/adapter_float32.mlpackage` (used only by the tests)
- `test.py`: the call bookkeeping (`adapter_calls`), checks (a), (b), (c); writes `ref/step3.npz` for Step 4
- `tools/inspect_position_table.py`: reads `adapter.ort`'s graph and stored table, and shows which rows of `pos_emb` were trained
