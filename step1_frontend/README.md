# Step 1: Frontend

The frontend turns raw microphone audio into **feature frames**, 80 ms at a time. It is the first of the five Core ML models, and its output is what the encoder (Step 2) reads.

## One call

| | Name | Shape | Type | What it is |
|---|---|---|---|---|
| in | `audio_chunk` | [1, 1280] | float32 | 80 ms of 16 kHz mono audio, samples in [−1, 1] |
| in | `conv1_buffer` | [1, 320, 4] | float32 | the previous call's `conv1_buffer_out` (zeros on the first call) |
| in | `conv2_buffer` | [1, 640, 4] | float32 | the previous call's `conv2_buffer_out` (zeros on the first call) |
| out | `features` | [1, 4, 320] | float32 | 4 feature frames, one per 20 ms, 320 numbers each |
| out | `conv1_buffer_out` | [1, 320, 4] | float32 | goes into the next call as `conv1_buffer` |
| out | `conv2_buffer_out` | [1, 640, 4] | float32 | goes into the next call as `conv2_buffer` |

Every shape is fixed. 1280 samples in, 4 features out, so the frontend produces 50 feature frames per second of audio.

## What it computes

1. Split the chunk into 16 frames of 80 samples (5 ms each).
2. Normalize each frame on its own: subtract its mean, then divide by its RMS. After this, loudness no longer matters.
3. Compress with asinh(k·x), where k is learned (0.645). Large values get squashed, much like a logarithm.
4. A linear layer maps each frame's 80 numbers to 320, then SiLU.
5. A causal convolution, 320 → 640 channels, kernel 5, stride 2, then SiLU. 16 time steps become 8.
6. A causal convolution, 640 → 320 channels, kernel 5, stride 2. 8 time steps become 4: the features.

There is no lookahead. Each feature frame depends only on the most recent 65 ms of audio (13 frames), never on anything after it.

## The state Swift carries between calls

Each convolution looks at its current input plus the 4 time steps before it. At the start of a chunk, those 4 steps were in the previous chunk, so the buffers hand them over.

- **On press:** both buffers start as zeros. That matches what the original model does at the start of a recording.
- **On every call:** pass the previous call's two `_out` buffers in, unchanged.
- **Features:** append each call's 4 frames to the recording's feature list. Step 2's encoder reads windows from that list.
- **On release:** pad the last, partial chunk with zeros up to 1280 samples, then make one final call.

## Where it comes from

- **PyTorch:** `MoonshineStreamingEncoderEmbedder` (`model.model.encoder.embedder`) in Transformers' `modeling_moonshine_streaming.py`. That version processes a whole recording in one call.
- **Moonshine's runtime:** `process_audio_chunk` in `moonshine-streaming-model.cpp`, running `frontend.model.ort` with weights from `frontend.weights.ort`. It also accepts chunks of any length by keeping three extra state values (`sample_buffer`, `sample_len`, `frame_count`). Swift always sends exactly 1280 samples, so this model doesn't need them.

## Files

- `wrapper.py`: this stage in PyTorch, one chunk per call
- `convert.py`: writes `models/frontend.mlpackage` (float16, released) and `models/frontend_float32.mlpackage` (used only by the tests)
- `test.py`: checks the wrapper against Transformers (a), Core ML against the wrapper (b), and Moonshine's `frontend.ort` against the wrapper (c); writes `ref/step1.npz` for Step 2
- `tools/diagnose_float16.py`: compares float16 and float32 stage by stage; the evidence for the SiLU recipe in `convert.py`
