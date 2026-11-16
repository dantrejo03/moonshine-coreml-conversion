# moonshine-coreml-conversion

Converts [Moonshine Streaming Tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) (English) to Core ML,
one stage per step, and checks each stage against the original. What it makes is released on Hugging Face as
[dantrejo03/moonshine-streaming-tiny-coreml](https://huggingface.co/dantrejo03/moonshine-streaming-tiny-coreml), and the
Swift package that runs those models is [swift-moonshine-coreml](https://github.com/dantrejo03/swift-moonshine-coreml).
The game plan, `to_coreml_plan.pdf`, isn't in this repo.

## Setup

```sh
./setup_environment.sh        # venv, pinned packages, moonshine-v2 clone, .ort downloads, then check_environment.py
source .venv/bin/activate
```

It's safe to re-run. It ends by running `check_environment.py`, which prints `All checks passed.` when everything is ready.

## Running scripts

Run everything from this folder, as a module, so `common` can be imported:

```sh
python -m step1_frontend.test
python -m step1_frontend.tools.diagnose_float16   # scripts in a step's tools/ folder
```

## Releasing the models

Apps don't use the packages in `models/`: they download a tagged release from Hugging Face. A release's tag is set in
every model it holds, and this repo gets the same tag on the commit the release was made from.
[RELEASING.md](RELEASING.md) has the steps, here and in swift-moonshine-coreml. The model card's source is
`huggingface_model_card.md`.

## Layout

| Path | What it holds | In git |
|---|---|---|
| `common/` | `config.py` (paths, shared sizes), `load_model.py` (PyTorch model, `.ort` sessions), `audio.py`, `compare.py`, `coreml_conversion.py` (the plan's conversion rules), `coreml_torch_ops.py` (replacements for coremltools' own op translations, such as the tanh SiLU), `ort_graph.py` (reads an `.ort` file's graph and stored tensors without running it) | yes |
| `step1_frontend/` … `step5_decoder/` | one folder per stage, all laid out the same (below) | yes |
| `models/` | every step's `.mlpackage` outputs: the five float16 packages that ship, and the float32 and explicit-cache packages only the tests use | no |
| `pipeline/` | Step 6: the five models chained as the app runs them (`push_to_talk.py`), the tokenizer port (`tokenizer.py`), and the end-to-end checks (`test.py`). No wrapper or conversion, so it doesn't follow the step folder layout below | yes |
| `ref/` | reference tensors, one `stepN.npz` per step | no |
| `third_party/moonshine-v2/` | Moonshine's C++ runtime to read (`core/`) and the test audio (`test-assets/`); its `.ort` files are LFS pointers | no |
| `third_party/moonshine_voice_models/…/tiny-streaming-en/quantized_26_11_02/` | the real `.ort` pieces, `streaming_config.json`, `tokenizer.bin` | no |

The Hugging Face checkpoint lives in the normal Hugging Face cache (`~/.cache/huggingface/hub`).

### Inside every step folder

| Path | What it holds | In git |
|---|---|---|
| `README.md` | what the stage does: inputs, outputs, shapes, and the state Swift carries (not how it's converted) | yes |
| `wrapper.py` | the stage in PyTorch, with fixed shapes and explicit inputs and outputs | yes |
| `convert.py` | writes the stage's `.mlpackage` files into the top-level `models/`, and defines their paths for other steps to import | yes |
| `test.py` | checks (a), (b), (c); writes `ref/stepN.npz` | yes |
| `tools/` | any other scripts (diagnostics, experiments) | yes |
