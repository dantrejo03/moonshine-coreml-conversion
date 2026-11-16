# moonshine-coreml-conversion

Converts [Moonshine Streaming Tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) (English) to Core ML,
as five models that stream on an Apple Watch's Neural Engine, and checks each one against the original. The models are
released on Hugging Face as
[dantrejo03/moonshine-streaming-tiny-coreml](https://huggingface.co/dantrejo03/moonshine-streaming-tiny-coreml), and
the Swift package that runs them is [swift-moonshine-coreml](https://github.com/dantrejo03/swift-moonshine-coreml).
You don't need this repo to use the models: it's for rebuilding them, or changing how they're converted.

## The five models

Audio goes through them in this order, 80 ms at a time. Each step's folder converts one model, and its README says what
the model takes and returns, and what the code calling it keeps between calls.

| Step | Model | What it does |
|---|---|---|
| [step1_frontend](step1_frontend) | `frontend` | audio → feature frames |
| [step2_encoder](step2_encoder) | `encoder` | feature frames → encoded frames, each with context from the frames around it |
| [step3_adapter](step3_adapter) | `adapter` | adds each frame's position, making memory frames |
| [step4_cross_kv](step4_cross_kv) | `cross_kv` | memory frames → the keys and values the decoder's cross-attention reads |
| [step5_decoder](step5_decoder) | `decoder_step` | one token in → a score for every possible next token |

[pipeline](pipeline) chains the five into a streaming transcriber in Python, with the tokenizer, and checks it end to
end. swift-moonshine-coreml is the same thing in Swift.

## Setup

Needs a Mac with Apple silicon (the tests run the models on the Neural Engine), Python 3.11 and git.

```sh
./setup_environment.sh
source .venv/bin/activate
```

The script makes a virtual environment with pinned packages, clones Moonshine's runtime (its source and test audio) into
`third_party/`, and downloads Moonshine's own `.ort` models and the Hugging Face checkpoint. It ends by running
`check_environment.py`, which prints `All checks passed.` It's safe to re-run.

## Converting and checking

Run everything from the top of the repo, as modules. Each step's test reads what the step before it saved, so go in
order:

```sh
for step in step1_frontend step2_encoder step3_adapter step4_cross_kv step5_decoder; do
  python -m $step.convert && python -m $step.test || break
done
python -m pipeline.test
```

`convert.py` writes the step's models into `models/`: the float16 model that's released, and a float32 copy the tests
use to see how much float16 loses. `test.py` runs three checks and exits non-zero if any of them fails:

- **(a)** the step's PyTorch wrapper (`wrapper.py`: the stage with fixed shapes, as Core ML needs) against the original
  model in Transformers;
- **(b)** the Core ML models against the wrapper;
- **(c)** Moonshine's own 8-bit version of the stage (its `.ort` file) against the wrapper.

When all three pass, it saves reference tensors in `ref/` for the next step. `pipeline.test` then transcribes the test
recordings with only the Core ML models, and checks the tokens against Transformers and Moonshine's runtime.

The scripts in a step's `tools/` folder are diagnostics, run the same way: for example,
`python -m step1_frontend.tools.diagnose_float16`.

## Releases

Each release's models carry its tag (for example `v1.1`) as their Core ML version, and this repo has the same tag on the
commit the release was made from. [RELEASING.md](RELEASING.md) has the steps.

## Layout

| Path | What it holds | In git |
|---|---|---|
| `common/` | code every step shares: paths and sizes (`config.py`), loading the model and Moonshine's `.ort` files, comparisons, the Core ML conversion settings | yes |
| `step1_frontend/` … `step5_decoder/` | one stage each: `README.md`, `wrapper.py`, `convert.py`, `test.py` and `tools/` | yes |
| `pipeline/` | the five models chained (`push_to_talk.py`), the tokenizer (`tokenizer.py`) and the end-to-end checks (`test.py`) | yes |
| `models/` | every step's `.mlpackage` files | no |
| `ref/` | reference tensors, one `stepN.npz` per step | no |
| `third_party/` | Moonshine's runtime and its `.ort` models | no |
| `huggingface_release/` | a release staged for upload | no |

## License

MIT; see [LICENSE](LICENSE). Moonshine Streaming Tiny's weights are Moonshine AI's, also under the MIT License
([moonshine-ai/moonshine-v2](https://github.com/moonshine-ai/moonshine-v2)).
