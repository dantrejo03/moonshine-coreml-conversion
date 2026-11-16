# Step 2: Encoder

The encoder turns feature frames into **encoded frames**: still 50 per second, 320 numbers each, but each one now carries context from the frames around it. It is the second of the five Core ML models. It reads Step 1's features, and Step 3's adapter reads its output.

## One call

| | Name | Shape | Type | What it is |
|---|---|---|---|---|
| in | `features` | [1, 116, 320] | float32 | a window of the recording's feature frames: real frames first, zeros after |
| in | `valid` | [1, 116] | float32 | 1 for each real frame, 0 for each padding frame |
| out | `encoded` | [1, 116, 320] | float32 | one encoded frame per window frame; Swift keeps only the new ones (below) |

Every shape is fixed. A window holds at most 116 frames: 96 of past context, 4 new and 16 held back.

## What it computes

Six transformer layers, then a final layer norm. Each layer does two things, and adds each result back onto its input:

1. Layer norm, then self-attention: each frame scores nearby frames (8 heads of 40 numbers) and mixes in what it finds.
2. Layer norm, then an MLP on each frame: 320 → 1280, GELU, → 320.

Attention is local. In every layer a frame sees itself and the 16 frames before it; in layers 1, 2, 5 and 6 it also sees the 4 frames after it (`sliding_windows` (16,4), (16,4), (16,0), (16,0), (16,4), (16,4), read as frames back and frames ahead). Through all six layers, an encoded frame depends on at most 96 frames back (1.92 s) and 16 ahead (320 ms).

There is no position information. A frame's output depends only on the frames around it, never on where it sits in the recording or the window. That's why a window gives the same encoded frames as the whole recording.

The model builds its masks itself: a frame may look at another only if it is inside that layer's window and `valid`. Padding is never looked at, so its own output rows are junk that Swift skips.

## The state Swift carries between calls

- **On press:** an empty feature list, and `emitted = 0` (how many encoded frames have been handed on).
- **On every chunk:** append Step 1's 4 features, then (with `total` = feature count, and `isFinal` true only for the last chunk on release):

  ```
  stable      = isFinal ? total : max(0, total - 16)
  new         = stable - emitted                    // skip the call when 0
  windowStart = max(0, emitted - 96)
  window      = features[windowStart ..< total]     // at most 116 frames
  pad window with zeros to 116; valid = 1 for its real frames, 0 after
  keep encoded[emitted - windowStart ..< emitted - windowStart + new]
  emitted     = stable
  ```

- **What comes out:** nothing for the first 4 chunks (they fill the 16 held-back frames), then 4 frames per chunk. The final call flushes 20: the 16 held back plus the last 4. The kept frames go to Step 3's adapter, in order.

Every kept frame has 96 real frames before it (or the start of the recording) and 16 after it (or the end): exactly the 96 and 16 it depends on, with nothing to spare. So its value equals what the whole recording in one call would give. The 16 held-back frames (320 ms) are the encoder's latency.

## Where it comes from

- **PyTorch:** `MoonshineStreamingEncoder` (`model.model.encoder`) in Transformers' `modeling_moonshine_streaming.py`: its `layers` and `final_norm`. Its own forward builds the sliding-window masks only when it's given an `attention_mask`; without one every frame attends to every other, a different model that still runs. The wrapper always builds the masks itself, and reads the windows one frame wider each way than Transformers 5.22.2 does: its `sliding_window_mask_function` gives (16, 4) as 15 back and 3 ahead. Moonshine's paper, its runtime's 96 and 16, and `encoder.ort` all say 16 and 4.
- **Moonshine's runtime:** `encode` in `moonshine-streaming-model.cpp`: the same window math. Moonshine runs it once per transcript update (several chunks at a time), so its windows vary in size, and `encoder.ort` takes any length. Swift runs it once per chunk, which keeps the window at 116.

## Files

- `wrapper.py`: this stage in PyTorch, one window per call
- `convert.py`: writes `models/encoder.mlpackage` (float16, ships to the watch) and `models/encoder_float32.mlpackage` (used only by the tests)
- `test.py`: the window math (`streaming_windows`), checks (a), (b), (c); writes `ref/step2.npz` for Step 3
- `tools/diagnose_float16.py`: compares float16 and float32 layer by layer, on the GPU and on the Neural Engine, and prints the largest value each layer reaches
- `tools/diagnose_ort.py`: three experiments on why Moonshine's quantized `encoder.ort` differs from the float model
