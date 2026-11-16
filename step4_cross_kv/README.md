# Step 4: Cross K/V

Cross K/V turns memory rows into what the decoder's **cross-attention** looks up. When the decoder (Step 5) picks the next token, each of its 6 layers compares a **query** (what it is looking for) with a **key** for every memory row, and takes a weighted average of the rows' **values**. Keys and values depend only on the audio, so each row's are computed once, as the row arrives, and kept in a buffer that every decoded token reads. It is the fourth of the five Core ML models. It reads Step 3's memory rows, and Step 5 reads its buffers.

## One call

| | Name | Shape | Type | What it is |
|---|---|---|---|---|
| in | `memory` | [1, 4, 320] | float32 | 4 new memory rows from Step 3, in order |
| out | `k_new` | [6, 1, 8, 4, 40] | float32 | their keys: layer, batch, head, frame, head_dim |
| out | `v_new` | [6, 1, 8, 4, 40] | float32 | their values, in the same order |

Every shape is fixed.

## What it computes

```
for each decoder layer l = 0 ..< 6:
    k_new[l] = k_proj_l(memory), each row's 320 numbers split into 8 heads of 40     [1, 8, 4, 40]
    v_new[l] = v_proj_l(memory), the same way
```

`k_proj` and `v_proj` are each layer's own 320 × 320 matrices, with no bias. Each row's key and value depend only on that row: no row mixes with another, and there is no position math (Step 3 already put each row's position into the memory). So projecting 4 rows per call and appending them gives exactly what projecting the whole memory at once gives. An all-zero row gives all-zero keys and values.

## The state Swift carries between calls

- **When allocating:** two buffers, `k_cross` and `v_cross`, each [6, 1, 8, Mmax, 40] with Mmax = 768, filled with zeros. Then `valid_frame_count = 0`.
- **After every adapter call** (4 new memory rows):

  ```
  (k_new, v_new) = cross_kv(memory: rows)
  for layer in 0 ..< 6, head in 0 ..< 8:
      k_cross[layer, 0, head, valid_frame_count ..< valid_frame_count + 4, :] = k_new[layer, 0, head, :, :]
      the same for v_cross and v_new
  valid_frame_count += 4
  ```

  Frames are the 4th of 5 axes, so in memory each append is 48 separate copies of 160 numbers per buffer (6 layers × 8 heads, each 4 frames × 40), not one block. `append_cross_kv` in `test.py` is this bookkeeping in Python.
- `valid_frame_count` always equals the adapter's `pos_offset`: both count the memory rows so far. Both go back to 0 when a new segment starts. It must never pass Mmax; Swift's segment cap ends the segment first.
- **Frames past `valid_frame_count` must be blocked by Step 5's `cross_valid` mask.** A zero key scores 0, which is not the same as blocked. Stale values left from an earlier segment are harmless behind the mask, but uninitialized memory could hold NaN (0 × NaN = NaN), so the buffers are zeroed once, when allocated.
- At float16, the two buffers take 5.9 MB together (11.8 MB at float32).

## Where it comes from

- **PyTorch:** the cross-attention branch of `MoonshineStreamingAttention.forward` in Transformers' `modeling_moonshine_streaming.py`, the branch taken when `key_value_states` (the memory) is given: `k_proj` and `v_proj` on the whole memory, split into heads, no rotary positions. Each `MoonshineStreamingDecoderLayer` has its own (`encoder_attn`). Transformers computes them on the first token and keeps them in `past_key_values.cross_attention_cache`.
- **Moonshine's runtime:** `compute_cross_kv` in `moonshine-streaming-model.cpp` runs `cross_kv.ort` on the whole memory before each decode that follows new rows. `cross_kv.ort` is the same graph done with 8-bit numbers: weights with one step per output column, and the memory rounded with one step for all of it (`DynamicQuantizeLinear`). Its keys and values match this model's to cosine 0.9998 or better, and its old rows shift slightly as the memory grows.

## On the Neural Engine

On the Mac, Core ML runs this model entirely on the CPU, under both `.all` and `.cpuAndNeuralEngine`. The 5-D outputs are not the reason: at 116 rows per call the same graph runs entirely on the Neural Engine, and at 4 rows even a single layer stays on the CPU (`tools/diagnose_compute_plan.py`). A call is small, about 4.9 million multiply-adds.

## Files

- `wrapper.py`: this stage in PyTorch, 4 rows per call
- `convert.py`: writes `models/cross_kv.mlpackage` (float16, released) and `models/cross_kv_float32.mlpackage` (used only by the tests)
- `test.py`: the buffer bookkeeping (`append_cross_kv`); checks the wrapper's appended buffers against Transformers' cross-attention cache for the whole recording (a), Core ML against the wrapper (b), and Moonshine's `cross_kv.ort` against the wrapper (c); writes `ref/step4.npz` for Step 5
- `tools/inspect_cross_kv_ort.py`: reads `cross_kv.ort`'s graph and weights, compares the weights with the checkpoint's, and computes the 8-bit step it uses for the memory
- `tools/diagnose_compute_plan.py`: compute plans for variants of the wrapper, to see what keeps it off the Neural Engine
