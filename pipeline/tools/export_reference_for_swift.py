"""Writes ref/reference_transcripts.json, what swift-moonshine-coreml's tests check the models against: copy it over
that repo's Tests/MoonshineCoreMLTests/Fixtures/reference_transcripts.json when releasing (RELEASING.md). Run after
python -m pipeline.test, with the tag the models will be released as:

    python -m pipeline.tools.export_reference_for_swift v1.2

It holds ref/step6.npz's utterance boundaries and final tokens with each final's text from tokenizer.py; the text of
every token on its own, to check the Swift tokenizer on the whole vocabulary; and each test recording's sample count and
a SHA-256 of its float32 samples, to check that Swift reads the WAV files exactly as soundfile does. The release goes in
models: the tokens are only right for models with these weights.
"""

import hashlib
import json
import re
import sys

import numpy as np

from common.audio import load_audio
from common.config import CONVERSION_DIRECTORY, REFERENCE_TENSORS_DIRECTORY, TEST_AUDIO_PATHS, VOCABULARY_SIZE
from pipeline.test import STEP6_REFERENCE_PATH
from pipeline.tokenizer import MoonshineTokenizer

REFERENCE_TRANSCRIPTS_PATH = REFERENCE_TENSORS_DIRECTORY / "reference_transcripts.json"
HUGGINGFACE_REPOSITORY = "dantrejo03/moonshine-streaming-tiny-coreml"
RELEASE_TAG_PATTERN = re.compile(r"v\d+\.\d+")


def float32_samples_sha256(samples: np.ndarray) -> str:
    """The samples' bytes as little-endian float32, the layout Swift hashes."""
    return hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest()


def single_token_texts(tokenizer: MoonshineTokenizer) -> list[str | None]:
    """Each token id's text on its own; None for an empty entry, where tokens_to_text raises (as Moonshine does)."""
    return [
        tokenizer.tokens_to_text([token_id]) if tokenizer.entries_by_token_id[token_id] else None
        for token_id in range(VOCABULARY_SIZE)
    ]


def main() -> None:
    if len(sys.argv) != 2 or not RELEASE_TAG_PATTERN.fullmatch(sys.argv[1]):
        raise SystemExit("usage: python -m pipeline.tools.export_reference_for_swift <release tag, such as v1.2>")
    release_tag = sys.argv[1]
    if not STEP6_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"{STEP6_REFERENCE_PATH} (run python -m pipeline.test first)")
    step6_reference = np.load(STEP6_REFERENCE_PATH)
    tokenizer = MoonshineTokenizer()

    recordings = []
    for recording_name, audio_path in TEST_AUDIO_PATHS.items():
        samples = load_audio(audio_path)
        utterances = []
        # step6.npz calls each utterance a press.
        for press_index, (start_sample, end_sample) in enumerate(step6_reference[f"{recording_name}_press_boundaries"]):
            final_token_ids = step6_reference[f"{recording_name}_press{press_index}_tokens"].tolist()
            utterances.append({
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "final_token_ids": final_token_ids,
                "final_text": tokenizer.tokens_to_text(final_token_ids),
            })
        recordings.append({
            "name": recording_name,
            "audio_file": audio_path.name,
            "sample_count": len(samples),
            "float32_samples_sha256": float32_samples_sha256(samples),
            "utterances": utterances,
        })
        token_counts = ", ".join(str(len(utterance["final_token_ids"])) for utterance in utterances)
        print(f"  {recording_name}: {len(samples)} samples, {len(utterances)} utterances, final tokens {token_counts}")

    texts = single_token_texts(tokenizer)
    print(f"  tokenizer: {len(texts)} single-token texts, empty entries at ids "
          f"{[token_id for token_id, text in enumerate(texts) if text is None]}")
    REFERENCE_TRANSCRIPTS_PATH.write_text(json.dumps({
        "models": {"huggingface_repository": HUGGINGFACE_REPOSITORY, "revision": release_tag},
        "single_token_texts": texts,
        "recordings": recordings,
    }, ensure_ascii=False, separators=(",", ":")))
    print(f"Saved {REFERENCE_TRANSCRIPTS_PATH.relative_to(CONVERSION_DIRECTORY)} for {release_tag}")


if __name__ == "__main__":
    main()
