# Step 6: The pipeline

The five Core ML models chained the way the watch app will run them: audio in 80 ms chunks while the button is held, a tentative transcript every 0.5 s, the final transcript on release. Nothing here is converted. This is the glue between the models (state, loops, text), written in Python so it can be checked against Moonshine and Transformers before Step 7 ports it to Swift.

## One press

A press is one **segment**: everything starts over on press, and a press holds at most 15.36 s (Mmax = 768 memory frames; version 1 of the app releases at that cap).

```
press()
    conv1_buffer, conv2_buffer = zeros                      Step 1
    feature_frame_count = emitted_frame_count = 0           Step 2
    memory_frame_count = 0                                  Step 3's pos_offset and Step 4's valid frame count
    cross_valid = all 0                                     k_cross, v_cross keep old contents, masked
    state = decoderStep.makeState()                         Step 5
    received_sample_count = 0

add_chunk(1280 samples)                                     each chunk as it arrives
    features, conv buffers = frontend(chunk, conv buffers)  4 new feature frames
    window = encoder_window(features so far, emitted_frame_count, isFinal: false)
    if window:                                              none for the first 4 chunks: they fill the 16 held back
        encoded = encoder(window); emitted_frame_count += new
        for each 4 new encoded frames:
            memory = adapter(frames, pos_offset: memory_frame_count)
            k_new, v_new = cross_kv(memory)
            write them into k_cross, v_cross at frames memory_frame_count ..< + 4 (float32 -> float16)
            cross_valid[those frames] = 1; memory_frame_count += 4

update()                                                    every 0.5 s while held, and once on release
    cap = min(ceil(received_sample_count / 16000 x 6.5), 128)
    token = 1; for position in 0 ..< cap: logits = decoderStep(token, position, k_cross, v_cross, cross_valid,
    state); token = argmax; stop after token 2 (Step 5's loop)
    text = tokenizer(tokens)

release(the last 0-1279 samples)
    pad them with zeros to 1280; frontend; encoder with isFinal: true
    the 16 held-back frames (20 with a padded chunk) go through adapter and cross_kv, 4 at a time
    update() -> the final text, committed
```

`push_to_talk.py` is this in Python (`PushToTalkTranscriber`). Each model is called through its own step's tested runner, and the window math is Step 2's `encoder_window`.

- **The update timer counts audio.** An update sees only the chunks that have fully arrived: at 0.5 s, 6 chunks (0.48 s), which hold 8 frames of memory (24 features, 16 held back).
- **Memory lags audio by 0.32 s while held** (the 16 held-back frames), so the token cap counts audio received, as Moonshine does, not memory.
- **Nothing in the decoder state is reset between updates**, and the cross K/V buffers are never cleared, only masked (Step 5).
- `update()` with no memory yet returns empty text without decoding, as Moonshine does.

## Recordings longer than a press

`test.py` plays a long recording as several presses, each released at a pause: from each press's start, the quietest 100 ms between 10 s and 15.36 s later, cut in its middle (`press_boundaries`). It's the plan's version-2 rule, but run on the whole file in advance, standing in for a person who releases at a pause. Doing it live, while recording, is Step 7's version 2: the cut lands in audio already processed, so the next press has to replay it.

## The tokenizer

`tokenizer.py` turns token ids into text exactly as Moonshine's runtime does (`BinTokenizer::tokens_to_text`, then `Transcriber::sanitize_text`):

- `tokenizer.bin` has 32,768 entries, one per token id in order: a length (1 byte under 128, otherwise 2: second × 128 + first − 128), then that many bytes.
- Ids 3–258 are the 256 single bytes (byte fallback: a character the vocabulary lacks is spelled as its UTF-8 bytes). Join every token's bytes first, then decode UTF-8.
- Skip every entry longer than 2 bytes that starts with `<` and ends with `>`: ids 0–2 (`<unk>`, `<s>`, `</s>`) and 32000–32767 (`<<ST_0>>` … `<<ST_767>>`).
- `▁` becomes a space; spaces and tabs are trimmed from both ends; a byte that doesn't start valid UTF-8 becomes `?`.
- One entry is empty (id 31353). Moonshine stops with "Invalid token" on it; so does the port.

## Where it comes from

- **Moonshine's runtime:** `transcribe_segment_with_streaming_model` in `transcriber.cpp` (chunks, encode, decoder reset, the decoding loop and its token cap), the state in `MoonshineStreamingState` (`moonshine-streaming-model.h`), the tokenizer in `bin-tokenizer.cpp`. Moonshine splits audio into segments with voice activity detection instead of a button, encodes once per update instead of once per chunk (Step 2 showed the frames are the same), and never processes a final partial chunk; this pipeline pads it. The pinned library (moonshine-voice 0.1.6) also never encodes with isFinal, so the last 16 encoder frames (0.32 s) of every segment are never transcribed (`tools/inspect_moonshine_library_calls.py`); this pipeline flushes them on release.
- **Transformers:** `model.generate` on the same audio is check (A)'s reference.

## Files

- `tokenizer.py`: the tokenizer port (`MoonshineTokenizer`)
- `push_to_talk.py`: the state and press, add_chunk, update, release (`PushToTalkTranscriber`), the five models loaded once (`load_pipeline_models`), timing per stage
- `test.py`: plays the test recordings press by press and prints every update. For each press: (A) checks it against `model.generate`; (B) checks Moonshine's transcriber against its `.ort` pieces run here, through our tokenizer; then compares our tokens with Moonshine's. It compares whole recordings with Moonshine's default transcriber (C) and `model.generate` on the whole file (D), prints timing, and writes `ref/step6.npz` (each press's boundaries and final tokens) for Step 7
- `tools/diagnose_moonshine_differences.py`: runs each press with the first k stages as Moonshine's `.ort` pieces and the rest in float, for k = 0 to 5, to find which stage's 8-bit rounding changes Moonshine's text
- `tools/inspect_moonshine_library_calls.py`: every onnxruntime call Moonshine's own library makes for each press, from its `log_ort_run` log
- `tools/export_reference_for_swift.py`: writes `ref/reference_transcripts.json` (`ref/step6.npz`'s utterances and final tokens, their texts, every single token's text, and each recording's sample count and SHA-256), the reference swift-moonshine-coreml's tests check the models against (RELEASING.md)
