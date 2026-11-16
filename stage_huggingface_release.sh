#!/usr/bin/env bash
# Copies what a Hugging Face release holds into huggingface_release/ (gitignored): the five float16 packages that ship,
# each with its Core ML version set to the release tag, tokenizer.bin, and huggingface_model_card.md as README.md with
# the tag and this commit filled in. RELEASING.md has the steps before and after.
#
#   ./stage_huggingface_release.sh v1.2
set -euo pipefail
cd "$(dirname "$0")"

release_tag=${1:-}
if [[ ! $release_tag =~ ^v[0-9]+\.[0-9]+$ ]]; then
  echo "usage: $0 <release tag, such as v1.2>" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes: commit first, so the release names the code it came from." >&2
  exit 1
fi

shipping_package_paths=(
  models/frontend.mlpackage
  models/encoder.mlpackage
  models/adapter.mlpackage
  models/cross_kv.mlpackage
  models/decoder_step.mlpackage
)
tokenizer_path=third_party/moonshine_voice_models/download.moonshine.ai/model/tiny-streaming-en/quantized_26_11_02/tokenizer.bin
release_directory=huggingface_release

for path in "${shipping_package_paths[@]}" "$tokenizer_path"; do
  if [ ! -e "$path" ]; then
    echo "$path is missing (run that step's convert.py, or setup_environment.sh for the tokenizer)." >&2
    exit 1
  fi
done

rm -rf "$release_directory"
mkdir "$release_directory"
for path in "${shipping_package_paths[@]}"; do
  cp -R "$path" "$release_directory/"
done
.venv/bin/python -I set_release_version.py "$release_tag" "$release_directory"/*.mlpackage
cp "$tokenizer_path" "$release_directory/"
git_commit=$(git rev-parse --short HEAD)
sed -e "s/{{RELEASE_TAG}}/$release_tag/g" -e "s/{{GIT_COMMIT}}/$git_commit/g" huggingface_model_card.md \
  > "$release_directory/README.md"

echo "Staged $release_tag from commit $git_commit:"
du -sh "$release_directory"/*
