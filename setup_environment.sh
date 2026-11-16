#!/usr/bin/env bash
# Step 0: builds everything the conversion steps need that is not in git, then checks it.
# Safe to re-run: each part is skipped when it already exists.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_FOR_VENV="${PYTHON_FOR_VENV:-python3.11}"

if [ ! -x .venv/bin/python ]; then
  "$PYTHON_FOR_VENV" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

# moonshine-v2: the C++ runtime to read (core/) and the test audio (test-assets/).
# Its .ort files are Git LFS pointers, not real models, so LFS is switched off for this clone
# (the -c values are also saved into the clone's own .git/config).
if [ ! -d third_party/moonshine-v2/.git ]; then
  git clone --depth 1 \
    -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false \
    https://github.com/moonshine-ai/moonshine-v2.git third_party/moonshine-v2
fi

# The real tiny-streaming-en .ort pieces (the answer key), downloaded the way moonshine-voice does it.
.venv/bin/python -I -c "
from pathlib import Path
from moonshine_voice.download import get_model_for_language
from moonshine_voice.moonshine_api import ModelArch
model_directory, _ = get_model_for_language(
    'en', ModelArch.TINY_STREAMING,
    cache_root=Path('third_party/moonshine_voice_models'),
    on_progress=lambda fraction, file_name: None,
)
print('Moonshine .ort pieces:', model_directory)
"

# Also downloads the Hugging Face checkpoint on first run (into the Hugging Face cache).
.venv/bin/python check_environment.py
