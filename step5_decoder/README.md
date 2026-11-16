# Step 5: Decoder step

The decoder turns memory into text, one **token** at a time. A token is a piece of a word, from a vocabulary of 32,768. Each call takes one token and its **position** (0 for the first) and returns a score, a **logit**, for every possible next token; Swift picks the highest and feeds it back in. This is the only stage that runs in a loop. It is the last of the five Core ML models. It reads Step 4's cross K/V buffers, and keeps its own cache of the earlier tokens' keys and values inside the model, as Core ML state.

## One call (`decoder_step.mlpackage`)

| | Name | Shape | Type | What it is |
|---|---|---|---|---|
| in | `token` | [1, 1] | int32 | the token at this position: 1 (start) at position 0, then the previous call's pick |
| in | `position` | [1] | int32 | 0, 1, 2, ...: this token's place in the transcript; below 128 |
| in | `k_cross` | [6, 1, 8, 768, 64] | float16 | Step 4's key buffer: layer, batch, head, frame, head_dim padded to 64 |
| in | `v_cross` | [6, 1, 8, 768, 64] | float16 | Step 4's value buffer |
| in | `cross_valid` | [1, 768] | float32 | 1 for the frames that hold memory (the first `valid_frame_count`), 0 after |
| state | `k_self` | [6, 1, 8, 128, 64] | float16 | this transcript's tokens' keys: layer, batch, head, token, head_dim padded to 64 |
| state | `v_self` | [6, 1, 8, 128, 64] | float16 | and their values |
| out | `logits` | [1, 1, 32768] | float32 | a score for every possible next token |

Every shape is fixed: 128 tokens (Lmax) and 768 memory frames (Mmax, 15.36 s).

## What it computes

```
x = embedding[token]                                      320 numbers
for each of the 6 layers:
    self-attention:  query, key, value from x; query and key turned by position (rotary positions);
                     write key and value into the state at row `position`, then attend over rows 0 ... position
    cross-attention: query from x; attend over the k_cross, v_cross frames where cross_valid is 1,
                     reading each row's first 40 numbers
    MLP:             320 -> 2 x 1280, silu(second half) x first half, -> 320
    (each block adds its result to x)
logits = embedding · norm(x)                              the same 32768 x 320 table, used backwards
```

Rows of the state after `position` are blocked by a mask, and so are buffer frames where `cross_valid` is 0; the padding of every cross K/V row is never read. Whatever they hold, the result is the same (tested with random noise in every one of them).

## The state Swift carries

- **For each segment:** Step 4's buffers, `k_cross` and `v_cross`, as float16, each row's 40 numbers followed by 24 zeros. Step 4's model outputs float32; Swift converts each call's 4 new rows as it copies them in. `cross_valid` is 1 for frames `0 ..< valid_frame_count`. And one state: `let state = decoderStep.makeState()`, passed to every prediction of the segment.
- **For each transcript update** (Moonshine re-decodes from scratch every time):

  ```
  token = 1; tokens = []
  for position in 0 ..< min(ceil(segmentSeconds x 6.5), 128):
      logits = decoderStep.prediction(token, position, k_cross, v_cross, cross_valid, using: state)
      token = argmax(logits)            // the first of equal highest scores
      tokens.append(token)
      if token == 2 { break }           // end token
  ```

  Nothing is reset between updates: each step writes its own row before reading it, and later rows are masked, so an earlier update's rows are never read. `decode` in `test.py` is this loop in Python.
- A new state starts as zeros (checked on the Mac only). Old values are harmless, but NaN would not be: a fresh state must not hold NaN.
- Predictions that share a state run one at a time.
- Nothing in the model checks `position`; the token cap keeps it below 128.
- Memory at float16: the state 1.57 MB, the cross buffers 9.44 MB (5.90 MB of numbers, the rest padding).

## On the Neural Engine

How the state is laid out and written decides whether this model runs on the Neural Engine (`tools/diagnose_state_on_neural_engine.py`):

- Each layer writes its new row into the state with a slice assignment at `position`, then reads its cache back. Rewriting the whole state as a blend doesn't compile for the Neural Engine.
- Each state row is padded from 40 numbers to 64, a multiple of 32. With 40, the model compiles but every prediction on the Neural Engine fails.
- Each `k_cross` and `v_cross` row is padded to 64 as well. With 40, Instruments on the Series 9 showed Core ML re-laying out both buffers on every call, outside the model's own operations (E5RT `PerformDataFormatChange`, mostly memory copying; `tools/list_decoder_cpu_operations.py` showed no such conversion in the program). Padded, that's gone, and the median step on the watch went from 19.7 to 15.7 ms, with identical tokens.
- Swift's `k_cross` and `v_cross` buffers are IOSurface-backed (`MLMultiArray(pixelBuffer:shape:)`, a float16 pixel buffer 64 wide and 36,864 tall). With ordinary arrays, Core ML copied both into its own buffers on every call (`MLE5BindInputBufferObjectByCopyingMultiArray`); with surfaces it doesn't, and the median step on the watch went from 15.8 to 12.4 ms.

On the Mac, 485 operations run on the Neural Engine, all state reads and writes included (12 of them cut each layer's cross K/V rows to 40 numbers); 22 small ones run on the CPU (the token and rotary-position lookups, the position comparisons, a few conversions). One step takes about 6 ms in Python on the Mac. On the Series 9, the Neural Engine works about 9.6 ms of each step (Instruments).

## Part 5a: the cache passed in and out

`decoder_step_explicit_cache.mlpackage` (and its float32 copy) is the same step with `k_self` and `v_self` [6, 1, 8, 128, 40] as inputs and `k_self_out`, `v_self_out` as outputs: the cache with row `position` replaced. Its `k_cross` and `v_cross` stay 40 wide. It is how this step was first checked against Transformers and Moonshine; on the Neural Engine its scores are bit-identical to `decoder_step.mlpackage`'s. It isn't released.

## Where it comes from

- **PyTorch:** `MoonshineStreamingDecoder` and `proj_out` in Transformers' `modeling_moonshine_streaming.py`, called layer by layer, skipping `forward`'s first lines (Step 3's adapter). `proj_out.weight` is stored in the checkpoint, bit-identical to the embedding table.
- **Moonshine's runtime:** `run_decoder_with_cross_kv` and `decode_step` in `moonshine-streaming-model.cpp` run `decoder_kv.ort`, whose cache grows by one row per call and whose position is the cache's length; the decoding loop is `transcribe_segment_with_streaming_model` in `transcriber.cpp`. `decoder_kv.ort` does the same math with 8-bit numbers; its scores match this model's to cosine 0.9999, with identical tokens.

## Files

- `wrapper.py`: this stage in PyTorch: `DecoderStepWithExplicitCache` (5a) and `DecoderStepWithStatefulCache` (5b)
- `convert.py`: writes `models/decoder_step.mlpackage` (released) and the two 5a packages (used only by the tests)
- `test.py`: the decoding loop (`decode`), Swift's bookkeeping for both parts (`ExplicitCacheSession`, `StatefulCacheSession`), checks the 5a wrapper against Transformers' `model.generate` (a), Core ML against the wrapper (b), and Moonshine's `decoder_kv.ort` against the wrapper (c), then 5b's stateful wrapper and `decoder_step.mlpackage` against 5a, token for token; writes `ref/step5.npz` (each test segment's tokens) for the pipeline
- `tools/diagnose_state_on_neural_engine.py`: converts the 5b step with its state laid out and written in different ways, to see which runs on the Neural Engine
- `tools/list_decoder_cpu_operations.py`: lists the operations Core ML runs on the CPU, and every one that reads the cross K/V, with names, types and shapes
