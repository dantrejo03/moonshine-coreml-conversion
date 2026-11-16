# Releasing models

A release passes through three repos, in this order. Each one pins the one before it by an exact version, so nothing
changes on its own:

1. **This repo** converts the models and exports their reference transcripts. The commit a release is made from gets
   the release's tag.
2. **Hugging Face**, [dantrejo03/moonshine-streaming-tiny-coreml](https://huggingface.co/dantrejo03/moonshine-streaming-tiny-coreml),
   holds the release under the same tag. Every model in it carries the tag as its Core ML version.
3. **[swift-moonshine-coreml](https://github.com/dantrejo03/swift-moonshine-coreml)** pins the release by tag and
   commit, lists it as compatible, and tests the models against the reference transcripts.

Apps then pin a version of the package and download a release that version lists. A model from a release the package
doesn't list fails to load, unless the app turns the check off.

Tags are `v<major>.<minor>`, the same here and on Hugging Face. The commands below release `v1.2`.

## 1. Here

1. Convert and check: each step's `convert.py` and `test.py`, then `python -m pipeline.test`. Commit.
2. Stage the release:

   ```sh
   ./stage_huggingface_release.sh v1.2
   ```

   It copies the five float16 packages into `huggingface_release/` and sets each one's Core ML version to `v1.2`
   (`set_release_version.py`, which leaves the weights as they are). It also adds `tokenizer.bin`, and writes the model
   card with the tag and commit filled in. To check a model's version:

   ```sh
   python -c 'import coremltools as ct; print(ct.models.MLModel("huggingface_release/decoder_step.mlpackage", skip_model_load=True).version)'
   ```

3. Export the reference transcripts, for swift-moonshine-coreml's tests:

   ```sh
   python -m pipeline.tools.export_reference_for_swift v1.2
   ```

4. Upload and tag on Hugging Face, and note the upload's commit hash, which the package pins:

   ```sh
   hf upload dantrejo03/moonshine-streaming-tiny-coreml huggingface_release . --commit-message "Release v1.2"
   hf repos tag create dantrejo03/moonshine-streaming-tiny-coreml v1.2
   ```

5. Tag this repo where the release was staged from (the card links to the tag), and push the tag:

   ```sh
   git tag -a v1.2 -m "Models released as v1.2 on Hugging Face"
   git push origin v1.2
   ```

A release that changes only metadata, as v1.1 did, skips converting. Its reference transcripts differ only in
`models.revision`.

## 2. In swift-moonshine-coreml

1. `scripts/download_models.sh`: set `release_tag` and `release_commit` (the upload's commit).
2. `Sources/MoonshineCoreML/Models/ModelRelease.swift`: add the tag to `compatibleVersions`, and remove any release this
   version of the package no longer works with.
3. If the inputs, outputs or state changed: update the expected interface (`ModelInterface.swift`) and the sizes
   (`TinyModel.swift`).
4. Copy `ref/reference_transcripts.json` from here over `Tests/MoonshineCoreMLTests/Fixtures/reference_transcripts.json`.
5. `README.md`: add the new package version to the Versions table.
6. Run `scripts/download_models.sh`, then `swift test`. With the models present, the tests check the release, the
   interface and the tokenizer, and that every reference utterance transcribes to exactly the reference tokens. A test
   also fails if the script's tag and the fixture's revision differ.
7. Commit, push, and tag a new minor version (0.2.0 after 0.1.x). Apps pin a minor version, so they move to the new
   models only when they choose to.

## 3. In apps

Move to the new package version, and download the same release it pins. WristWhisper also updates its on-watch
check's reference tokens (`OnWatchCheck.referenceTokenIDs`) from the new fixture when the weights changed, then runs on
a watch, where the first launch checks the models.
