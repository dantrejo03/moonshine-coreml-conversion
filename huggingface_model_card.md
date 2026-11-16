---
license: mit
language:
  - en
base_model: UsefulSensors/moonshine-streaming-tiny
pipeline_tag: automatic-speech-recognition
tags:
  - coreml
  - apple-watch
  - watchos
  - neural-engine
  - streaming
---

# Moonshine Streaming Tiny for Core ML

[Moonshine Streaming Tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) (English) converted to Core ML
as five float16 models, built to transcribe speech on an Apple Watch while the user is still talking. The Swift package
that runs them is [swift-moonshine-coreml](https://github.com/dantrejo03/swift-moonshine-coreml), and the code that
converted them is [moonshine-coreml-conversion](https://github.com/dantrejo03/moonshine-coreml-conversion).

This release, `{{RELEASE_TAG}}`, was made from that repo's
[`{{RELEASE_TAG}}` tag](https://github.com/dantrejo03/moonshine-coreml-conversion/tree/{{RELEASE_TAG}}) (commit
`{{GIT_COMMIT}}`).

## Files

| File | Size | Runs on | What it does |
|---|---|---|---|
| `frontend.mlpackage` | 4 MB | CPU only | 80 ms of audio → 4 feature frames |
| `encoder.mlpackage` | 14 MB | CPU + Neural Engine | a 116-frame window of features → encoded frames |
| `adapter.mlpackage` | 2.5 MB | CPU + Neural Engine | 4 encoded frames → 4 memory frames (adds position) |
| `cross_kv.mlpackage` | 2.4 MB | CPU + Neural Engine | 4 memory frames → the decoder's cross-attention keys and values for them |
| `decoder_step.mlpackage` | 41 MB | CPU + Neural Engine | one token in → scores for the next token; its self-attention cache is Core ML state |
| `tokenizer.bin` | | | Moonshine's tokenizer, token ids → text |

The models need watchOS 11, iOS 18 or macOS 15 (the decoder's cache uses `MLState`). On a watch they need a Series 9
or later: a Series 8's 2-core Neural Engine rejects them. Run the frontend on the CPU only
(`computeUnits = .cpuOnly`): on a Series 9's Neural Engine it computes wrong features.

## Versions

Each model carries its release's tag as its Core ML version (`metadata.versionString`, for example `v1.1`), which stays
with it when Xcode compiles it into an app. Code can check that all five come from one release it was tested with, and
the Swift package does. v1.1 has the same weights as v1.0, with only the versions added; v1.0's models have none.

## Usage

Download:

```sh
hf download dantrejo03/moonshine-streaming-tiny-coreml --local-dir moonshine-coreml
```

For an app, add the five `.mlpackage` folders to the Xcode target, and Xcode compiles each into a `.mlmodelc` (a watch
can't compile models itself). On a Mac or iPhone you can also compile at runtime. Loading the decoder and running one
step:

```swift
import CoreML

let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: "moonshine-coreml/decoder_step.mlpackage"))
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine  // .cpuOnly for the frontend
let decoderStep = try MLModel(contentsOf: compiledURL, configuration: configuration)
let state = decoderStep.makeState()  // one per utterance
// inputs: an MLDictionaryFeatureProvider with token, position, k_cross, v_cross and cross_valid (shapes below)
let logits = try decoderStep.prediction(from: inputs, using: state).featureValue(for: "logits")
```

The whole pipeline (all five models, the state between calls and the tokenizer) is the Swift package
[swift-moonshine-coreml](https://github.com/dantrejo03/swift-moonshine-coreml). Its README says how to add it and how to
download a release it works with, and its tests check downloaded models against reference transcripts. It doesn't
include the models: add them to your app as above, and pass the folder Xcode compiles them into
(`Bundle.main.resourceURL`) to `Moonshine.Transcriber(modelsDirectory:)`.

## Inputs and outputs

All float inputs and outputs are float32 except where noted; integers are int32.

| Model | Inputs | Outputs |
|---|---|---|
| frontend | `audio_chunk` [1, 1280] (16 kHz), `conv1_buffer` [1, 320, 4], `conv2_buffer` [1, 640, 4] | `features` [1, 4, 320], `conv1_buffer_out`, `conv2_buffer_out` (pass these back in with the next chunk) |
| encoder | `features` [1, 116, 320], `valid` [1, 116] (1 for real frames, 0 for padding) | `encoded` [1, 116, 320] |
| adapter | `encoded` [1, 4, 320], `pos_offset` [1] (memory frames so far) | `memory` [1, 4, 320] |
| cross_kv | `memory` [1, 4, 320] | `k_new`, `v_new` [6, 1, 8, 4, 40] (layer, batch, head, frame, head dim) |
| decoder_step | `token` [1, 1], `position` [1], `k_cross`, `v_cross` [6, 1, 8, 768, 64] float16, `cross_valid` [1, 768] | `logits` [1, 1, 32768]; state `k_self`, `v_self` [6, 1, 8, 128, 64] float16 |

`k_cross` and `v_cross` hold up to 768 memory frames (15.36 s of audio). Each row's 40 numbers from `cross_kv` go in
its first 40 places, and the last 24 are padding the model never reads. Rows of 64 are what the Neural Engine wants:
with 40, Core ML re-lays out both inputs on every call. `cross_valid` is 1 for the frames written so far.

## Running them

One recording (up to 15.36 s) is one utterance; everything starts over for the next.

1. At the start: zero the conv buffers and `cross_valid`, and make one decoder state (`makeState()`) for the whole
   utterance.
2. Every 1280 samples (80 ms): frontend → 4 feature frames. The encoder looks 16 frames ahead, so a frame is encoded
   once 16 later ones exist; a window is 116 frames, the new ones plus up to 96 before them and 16 after. For every
   4 new encoded frames: adapter, then cross_kv, and write the result into `k_cross`/`v_cross` as float16.
3. To transcribe (while recording, as often as you like, and once at the end): start from token 1 at position 0,
   reusing the utterance's state (each step overwrites its own position). At each step feed back the top-scoring
   token, until token 2 (end) or a cap of 6.5 tokens per second of audio (at most 128).
4. At the end: pad the last partial chunk with zeros, run it, encode the held-back frames, put them through adapter
   and cross_kv, and transcribe once more.

`tokenizer.bin` lists 32,768 tokens in id order, each as a length (1 byte under 128, otherwise 2: second × 128 + first
− 128) followed by its bytes. Join the bytes of every token, skipping special entries (over 2 bytes, starting with `<`
and ending with `>`). Turn `▁` into a space, trim spaces and tabs from both ends, then decode the bytes as UTF-8,
with each byte that doesn't begin a complete UTF-8 sequence becoming `?` (as Moonshine's runtime does).

The Swift code that does all this is [swift-moonshine-coreml](https://github.com/dantrejo03/swift-moonshine-coreml).

## On a Series 9

- The final tokens match the conversion's Python pipeline on the test recordings, on the watch and on a Mac.
- Median decoder step: 12.4 ms.
- Time from the end of a recording to the final text: 424 ms for 4.8 s of speech, 629 ms for 8.4 s, about 1.26 s for
  14–15 s.

Word error rate wasn't measured for this conversion. For the original model's (an average of 12.01% on the Open ASR
Leaderboard), see [its card](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny).

## Limitations

- English only, like the original model, and it can hallucinate or repeat words, especially on short or noisy audio.
- An utterance holds at most 15.36 s of audio (768 memory frames): the adapter's position table was only trained for the
  first 40 s, and the decoder's buffers are sized for 15 s. Split longer audio into several utterances.
- Decoding is greedy (the top-scoring token each step), and the time to the final text grows with the transcript's
  length.
- Run `decoder_step` on the CPU and Neural Engine, not the CPU only: Core ML's CPU-only path writes its state wrongly.
- Apple Watch Series 9 or later; tested on a Series 9 and an Apple Silicon Mac.

## Float32

Only float16 is released, because the Neural Engine computes in float16. The conversion scripts also write float32
copies, used to measure how much accuracy float16 loses; rebuild them with the conversion repo's `convert.py` scripts.

## References

- [Moonshine Streaming paper](https://download.moonshine.ai/docs/moonshine_streaming_paper.pdf)
- [Moonshine Streaming Tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny), the original model

## License

MIT, like the original English model. The weights and `tokenizer.bin` are Moonshine AI's
([moonshine-ai/moonshine-v2](https://github.com/moonshine-ai/moonshine-v2)); this release only converts them.
