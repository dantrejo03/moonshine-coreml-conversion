"""Step 6: the push-to-talk app simulated on the Mac with only the five Core ML models. Run from conversion/,
after every step's convert.py:

    python -m pipeline.test

Each test recording is played into PushToTalkTranscriber 80 ms at a time, as a microphone delivers it: the tentative
text every 0.5 s of audio, the final text on release. A recording longer than one press (15.36 s) is played as
several presses, each released at a pause (press_boundaries). Each update's text is printed, to judge by eye.

Per press, pass or fail:
(A) model.generate on the same audio (Transformers end to end, float32): identical tokens, and scores cosine >= 0.999
    (Step 5's float16 bar),
(B) Moonshine's own transcriber on the same audio, as one segment, vs Moonshine's .ort pieces run here the way its
    library runs them (whole chunks only, and no isFinal: a segment's last 16 encoder frames are dropped), their tokens
    turned into text by our tokenizer port: identical text. This checks the tokenizer port on Moonshine's own tokens,
    and that we know what Moonshine computes.
Then, reported: ours vs Moonshine's tokens, identical or where they part and by how much each side preferred its pick.
Per recording, printed with the words that differ, each difference to be explained:
(C) Moonshine's transcriber with its default settings (its own voice activity detection) on the whole file,
(D) model.generate on the whole file, as one segment.
Then the time each stage took. Exits non-zero if (A) or (B) fails; writes ref/step6.npz only when all pass.
"""

import math
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import moonshine_voice
import numpy as np
import onnxruntime
import torch
from moonshine_voice.moonshine_api import ModelArch
from moonshine_voice.transcriber import Transcriber
from transformers import MoonshineStreamingForConditionalGeneration

from common.audio import load_audio
from common.compare import COREML_FLOAT16_MIN_COSINE_SIMILARITY
from common.config import (
    AUDIO_SAMPLES_PER_CHUNK,
    ENCODER_HELD_BACK_FRAME_COUNT,
    END_TOKEN_ID,
    MAXIMUM_TOKENS_PER_SECOND,
    MOONSHINE_ORT_MODEL_DIRECTORY,
    REFERENCE_TENSORS_DIRECTORY,
    SAMPLE_RATE_HZ,
    START_TOKEN_ID,
    TEST_AUDIO_PATHS,
)
from common.load_model import load_ort_session, load_pytorch_model
from pipeline.push_to_talk import (
    CHUNK_STAGE_NAME,
    MAXIMUM_PRESS_SAMPLE_COUNT,
    UPDATE_STAGE_NAME,
    PushToTalkTranscriber,
    StageTimes,
    TranscriptUpdate,
    load_pipeline_models,
)
from pipeline.tokenizer import MoonshineTokenizer, is_special_entry
from step1_frontend.test import load_ort_frontend, ort_chunk_runner, pad_to_whole_chunks, run_chunk_by_chunk
from step2_encoder.test import transformers_reading_of
from step4_cross_kv.test import run_cross_kv_ort
from step5_decoder.test import (
    DecoderKvOrtSession,
    TestSegment,
    Transcript,
    compare_with_reference,
    decode,
    generate_greedy,
    print_first_difference,
)

STEP6_REFERENCE_PATH = REFERENCE_TENSORS_DIRECTORY / "step6.npz"
UPDATE_INTERVAL_SAMPLE_COUNT = SAMPLE_RATE_HZ // 2  # 0.5 s, the plan's default update interval, counted in samples
CHUNK_SECONDS = AUDIO_SAMPLES_PER_CHUNK / SAMPLE_RATE_HZ  # 0.08: the time budget for stages 1-4 per chunk

# Where a long recording's presses end: the plan's version-2 rule ("every 10-15 s, cut at the quietest recent moment,
# measured as the lowest loudness over short windows"), applied to the whole file in advance.
MINIMUM_PRESS_SAMPLE_COUNT = 10 * SAMPLE_RATE_HZ
LOUDNESS_WINDOW_SAMPLE_COUNT = SAMPLE_RATE_HZ // 10  # 100 ms; loudness is the samples' root mean square
LOUDNESS_WINDOW_HOP_SAMPLE_COUNT = SAMPLE_RATE_HZ // 100  # 10 ms

MOONSHINE_MAXIMUM_TOKEN_COUNT = 256  # transcriber.cpp's token cap per segment; Lmax (128) takes its place in ours
VAD_HOP_SAMPLE_COUNT = 512  # transcriber.h vad_hop_size: Moonshine's VAD passes audio on in whole hops of this size
MOONSHINE_INTRA_OP_THREAD_COUNT = 1  # moonshine-streaming-model.cpp 166: Moonshine runs each .ort op on one thread
# Moonshine's transcriber on exactly one press's audio: voice activity detection off (a threshold of 0 counts every
# moment as voice) and a maximum segment far longer than a press, so the press is one segment. Speakers off: they
# don't change the text.
MOONSHINE_ONE_SEGMENT_OPTIONS = {"vad_threshold": "0", "vad_max_segment_duration": "60", "identify_speakers": "false"}
MOONSHINE_DEFAULT_OPTIONS = {"identify_speakers": "false"}
STAGE_COUNT = 5  # frontend, encoder, adapter, cross_kv, decoder


@dataclass
class PressResult:
    start_sample: int  # in the recording
    end_sample: int
    final: TranscriptUpdate  # the text committed on release
    tentative_updates: list[TranscriptUpdate]


def press_boundaries(samples: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) samples of each press for one recording. While more than one press's worth of audio is left, the
    press ends in the middle of the quietest 100 ms between 10 s and 15.36 s after it started, and the next press
    starts there. The last press takes what's left."""
    boundaries = []
    press_start = 0
    while len(samples) - press_start > MAXIMUM_PRESS_SAMPLE_COUNT:
        search_start = press_start + MINIMUM_PRESS_SAMPLE_COUNT
        search_samples = samples[search_start : press_start + MAXIMUM_PRESS_SAMPLE_COUNT].astype(np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(search_samples, LOUDNESS_WINDOW_SAMPLE_COUNT)
        windows = windows[::LOUDNESS_WINDOW_HOP_SAMPLE_COUNT]
        loudness = np.sqrt((windows**2).mean(axis=1))
        quietest_window = int(np.argmin(loudness))
        cut = search_start + quietest_window * LOUDNESS_WINDOW_HOP_SAMPLE_COUNT + LOUDNESS_WINDOW_SAMPLE_COUNT // 2
        print(f"  cut at {cut / SAMPLE_RATE_HZ:.2f} s: the quietest 100 ms there has loudness {loudness[quietest_window]:.5f}")
        boundaries.append((press_start, cut))
        press_start = cut
    boundaries.append((press_start, len(samples)))
    return boundaries


def play_press(
    transcriber: PushToTalkTranscriber, press_samples: np.ndarray, press_start_seconds: float
) -> tuple[TranscriptUpdate, list[TranscriptUpdate]]:
    """One press, as the app runs it: button down, 80 ms chunks as they arrive, an update each time the 0.5 s timer
    fires, button up. Prints every update. Returns the final update and the tentative ones.

    The timer counts audio time: an update sees only the chunks that had fully arrived when it fired (at 0.5 s, 6
    chunks, 0.48 s). Counting samples rather than seconds keeps 25 x 0.08 from rounding past 2.0."""
    transcriber.press()
    tentative_updates = []
    next_update_sample = UPDATE_INTERVAL_SAMPLE_COUNT

    def run_updates_due_before(sample_count: int) -> None:
        nonlocal next_update_sample
        while next_update_sample < sample_count:
            update = transcriber.update()
            print_update(f"{press_start_seconds + next_update_sample / SAMPLE_RATE_HZ:6.2f} s", update)
            tentative_updates.append(update)
            next_update_sample += UPDATE_INTERVAL_SAMPLE_COUNT

    whole_chunk_count = len(press_samples) // AUDIO_SAMPLES_PER_CHUNK
    for chunk_index in range(whole_chunk_count):
        chunk_end_sample = (chunk_index + 1) * AUDIO_SAMPLES_PER_CHUNK
        run_updates_due_before(chunk_end_sample)
        transcriber.add_chunk(press_samples[chunk_end_sample - AUDIO_SAMPLES_PER_CHUNK : chunk_end_sample])
    run_updates_due_before(len(press_samples))
    final = transcriber.release(press_samples[whole_chunk_count * AUDIO_SAMPLES_PER_CHUNK :])
    print_update(f"release at {press_start_seconds + len(press_samples) / SAMPLE_RATE_HZ:.2f} s, final", final)
    return final, tentative_updates


def print_update(when: str, update: TranscriptUpdate) -> None:
    if update.text:
        text = update.text
    elif update.tokens:
        text = "(no words: the end token came first)"
    else:
        text = "(no memory yet: nothing decoded)"
    stopped_at_cap = "" if update.ended_with_end_token or not update.tokens else f" [stopped at the cap, {update.token_cap}]"
    print(f"    {when}  {update.decode_milliseconds:4.0f} ms  {text}{stopped_at_cap}")


def run_model_generate_on_audio(
    model: MoonshineStreamingForConditionalGeneration, samples: np.ndarray, max_new_tokens: int
) -> Transcript:
    """The original, end to end: model.generate on raw audio, float32, greedy. The audio is zero-padded to whole
    chunks, as release() pads it. Two settings make Transformers' encoder Moonshine's (Step 2): an all-ones
    attention_mask, without which it attends to every frame (the mask trap), and sliding_windows in Transformers'
    reading of Moonshine's windows (its off-by-one), set for this call only."""
    padded_samples = pad_to_whole_chunks(samples)
    encoder_config = model.model.encoder.config
    config_sliding_windows = encoder_config.sliding_windows
    encoder_config.sliding_windows = transformers_reading_of(config_sliding_windows)
    try:
        return generate_greedy(
            model,
            max_new_tokens,
            input_values=torch.from_numpy(padded_samples)[None],
            attention_mask=torch.ones(1, len(padded_samples), dtype=torch.long),
        )
    finally:
        encoder_config.sliding_windows = config_sliding_windows


def moonshine_transcript_lines(samples: np.ndarray, options: dict[str, str]) -> list:
    """Moonshine's own transcriber (moonshine-voice, tiny-streaming-en) on these samples all at once
    (transcribe_without_streaming): its voice activity detection splits them into segments, then each segment is
    transcribed with isFinal. Returns its lines (text, start_time, duration). A fresh transcriber per call, so nothing
    carries over from an earlier call."""
    transcriber = Transcriber(
        model_path=str(MOONSHINE_ORT_MODEL_DIRECTORY), model_arch=ModelArch.TINY_STREAMING, options=options
    )
    try:
        return transcriber.transcribe_without_streaming(samples.tolist(), SAMPLE_RATE_HZ).lines
    finally:
        transcriber.close()


def moonshine_segment(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """What Moonshine's transcriber transcribes of one segment, and the seconds its token cap counts. Its VAD passes
    audio on in whole 512-sample hops, so a last partial hop never reaches the segment; the transcriber then feeds
    only whole 1280-sample chunks of it (transcriber.cpp 527-557). The rest is dropped, not padded. The token cap
    counts the segment's whole hops."""
    segment_sample_count = len(samples) // VAD_HOP_SAMPLE_COUNT * VAD_HOP_SAMPLE_COUNT
    chunked_sample_count = segment_sample_count // AUDIO_SAMPLES_PER_CHUNK * AUDIO_SAMPLES_PER_CHUNK
    return samples[:chunked_sample_count], segment_sample_count / SAMPLE_RATE_HZ


def moonshine_token_cap(seconds: float) -> int:
    """transcriber.cpp 568-572: min(ceil(seconds x 6.5), 256)."""
    return min(math.ceil(seconds * MAXIMUM_TOKENS_PER_SECOND), MOONSHINE_MAXIMUM_TOKEN_COUNT)


class MoonshineOrtPieces:
    """Moonshine's .ort pieces, loaded once, each run the way its transcriber runs them on one fresh segment given all
    at once (transcribe_without_streaming). One method per stage, so tools/diagnose_moonshine_differences.py can swap
    them for float ones one at a time (run_five_stages)."""

    def __init__(self):
        # Step 1: the frontend is two files.
        self.frontend_session, self.frontend_weights_by_name = load_ort_frontend(MOONSHINE_INTRA_OP_THREAD_COUNT)
        self.encoder_session = load_ort_session("encoder.ort", MOONSHINE_INTRA_OP_THREAD_COUNT)
        self.adapter_session = load_ort_session("adapter.ort", MOONSHINE_INTRA_OP_THREAD_COUNT)
        self.cross_kv_session = load_ort_session("cross_kv.ort", MOONSHINE_INTRA_OP_THREAD_COUNT)
        self.decoder_session = load_ort_session("decoder_kv.ort", MOONSHINE_INTRA_OP_THREAD_COUNT)

    def features(self, samples: np.ndarray) -> np.ndarray:
        """frontend.ort chunk by chunk, its state starting from zero: [1, 4 x chunks, 320]."""
        return run_chunk_by_chunk(ort_chunk_runner(self.frontend_session, self.frontend_weights_by_name), samples)

    def encoded(self, features: np.ndarray) -> np.ndarray:
        """encoder.ort in one call on every feature frame: encode() with isFinal on a fresh segment, whose window
        starts at frame 0 and holds everything."""
        (encoded,) = self.encoder_session.run(["encoded"], {"features": features})
        return encoded

    def memory(self, encoded: np.ndarray) -> np.ndarray:
        """adapter.ort in one call on every encoded frame, from pos_offset 0 (int64)."""
        (memory,) = self.adapter_session.run(
            ["memory"], {"encoded": encoded, "pos_offset": np.array([0], dtype=np.int64)}
        )
        return memory

    def cross_kv(self, memory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """cross_kv.ort on the whole memory, as compute_cross_kv runs it before decoding."""
        return run_cross_kv_ort(self.cross_kv_session, memory)

    def decode(self, memory: np.ndarray, k_cross: np.ndarray, v_cross: np.ndarray, token_cap: int) -> Transcript:
        """decoder_kv.ort, greedy (Step 5's DecoderKvOrtSession). memory is only counted, for the frame count."""
        segment = TestSegment(name="press", encoded=memory, k_cross=k_cross, v_cross=v_cross)
        return decode(DecoderKvOrtSession(self.decoder_session, segment), token_cap)

    def transcribe(self, samples: np.ndarray) -> Transcript:
        """Moonshine's tokens for one segment, as the pinned library computes them: the audio its transcriber keeps
        (moonshine_segment), all five pieces, and the encoder without isFinal (see run_five_stages)."""
        kept_samples, seconds = moonshine_segment(samples)
        return run_five_stages([self] * STAGE_COUNT, kept_samples, moonshine_token_cap(seconds), is_final=False)


def print_onnxruntime_builds() -> None:
    """The .ort pieces here run in Python's onnxruntime; Moonshine's transcriber runs the onnxruntime library that
    moonshine-voice ships inside its package. Different builds can round the float steps around 8-bit math
    differently."""
    bundled_file_names = sorted(path.name for path in Path(moonshine_voice.__file__).parent.glob("libonnxruntime*"))
    print(
        f"  the pieces here: onnxruntime {onnxruntime.__version__} from {Path(onnxruntime.__file__).parent}, "
        f"{MOONSHINE_INTRA_OP_THREAD_COUNT} thread per op"
    )
    print(f"  Moonshine's transcriber: {', '.join(bundled_file_names)}, shipped inside moonshine-voice")


def run_five_stages(pieces_by_stage: list, samples: np.ndarray, token_cap: int, is_final: bool) -> Transcript:
    """One segment's samples through the five stages, all at once, stage i run by pieces_by_stage[i]: a
    MoonshineOrtPieces, or anything with the same five methods (features, encoded, memory, cross_kv, decode).

    is_final: whether the encoder's last 16 frames, held back for lookahead, reach the memory. The app releases with
    isFinal, so they do. The pinned Moonshine library (moonshine-voice 0.1.6) transcribes a segment without it:
    tools/inspect_moonshine_library_calls.py shows its single encoder call emitting all but the last 16 frames, which
    then never go further, so the last 0.32 s of every segment is never transcribed. The frames before them see all
    their lookahead either way, so dropping them after the encoder is exact."""
    features = pieces_by_stage[0].features(samples)
    encoded = pieces_by_stage[1].encoded(features)
    if not is_final:
        encoded = encoded[:, : encoded.shape[1] - ENCODER_HELD_BACK_FRAME_COUNT]
    memory = pieces_by_stage[2].memory(encoded)
    k_cross, v_cross = pieces_by_stage[3].cross_kv(memory)
    return pieces_by_stage[4].decode(memory, k_cross, v_cross, token_cap)


def token_text(tokenizer: MoonshineTokenizer, token_id: int) -> str:
    """One token as text, for printing: the end token and other special tokens have none."""
    return tokenizer.tokens_to_text([token_id]) or ("(end token)" if token_id == END_TOKEN_ID else f"(token {token_id})")


def print_word_differences(label: str, text: str, reference_text: str) -> None:
    """The spans of words where text and reference_text differ, each with the 3 words before it."""
    words, reference_words = text.split(), reference_text.split()
    differences = [
        opcode
        for opcode in SequenceMatcher(None, words, reference_words, autojunk=False).get_opcodes()
        if opcode[0] != "equal"
    ]
    if not differences:
        print(f"      {label}: the same {len(words)} words")
        return
    print(f"      {label}: {len(differences)} spans differ ({len(words)} words ours, {len(reference_words)} theirs)")
    for _, start, end, reference_start, reference_end in differences:
        words_before = " ".join(words[max(0, start - 3) : start])
        print(
            f'        after "{words_before}": ours "{" ".join(words[start:end])}", '
            f'theirs "{" ".join(reference_words[reference_start:reference_end])}"'
        )


def check_press(
    model: MoonshineStreamingForConditionalGeneration,
    tokenizer: MoonshineTokenizer,
    moonshine_ort_pieces: MoonshineOrtPieces,
    transcriber: PushToTalkTranscriber,
    press_samples: np.ndarray,
    final: TranscriptUpdate,
) -> list[bool]:
    """(A) and (B) for the press just released, then ours vs Moonshine's tokens. (A) decodes again with the press's
    own decoder session, so it runs before the next press resets it."""
    generated = run_model_generate_on_audio(model, press_samples, final.token_cap)
    passed_a, our_transcript = compare_with_reference(
        "    (A) Core ML pipeline vs model.generate on the same audio",
        transcriber.decoder_session,
        generated,
        final.token_cap,
        minimum_allowed_cosine_similarity=COREML_FLOAT16_MIN_COSINE_SIMILARITY,
    )
    if generated.tokens != final.tokens:
        print(f"      model.generate's text: {tokenizer.tokens_to_text(generated.tokens)}")

    lines = moonshine_transcript_lines(press_samples, MOONSHINE_ONE_SEGMENT_OPTIONS)
    moonshine_transcript = moonshine_ort_pieces.transcribe(press_samples)
    moonshine_text = tokenizer.tokens_to_text(moonshine_transcript.tokens)
    kept_samples, _ = moonshine_segment(press_samples)
    passed_b = len(lines) == 1 and lines[0].text == moonshine_text
    verdict = "identical" if passed_b else "DIFFERS: FAIL"
    print(f"    (B) Moonshine's transcriber vs its .ort pieces run here, in our tokenizer: text {verdict}")
    for line in lines:
        print(
            f"      its line: from {line.start_time:.2f} s, {line.duration:.2f} s long; the pieces here transcribe "
            f"{len(kept_samples) / SAMPLE_RATE_HZ:.2f} s"
        )
        if not passed_b:
            print(f"        transcriber: {line.text}")
    if not passed_b:
        print(f"        pieces here: {moonshine_text}")
        if len(lines) == 1:
            print_word_differences("words (ours = the pieces here)", moonshine_text, lines[0].text)

    if our_transcript.tokens == moonshine_transcript.tokens:
        print("    ours vs Moonshine's tokens: identical")
        return [passed_a, passed_b]
    print(f"    ours vs Moonshine's tokens: they differ. Moonshine: {moonshine_text or '(no words)'}")
    print("      here = ours, reference = Moonshine's .ort pieces:")
    step = print_first_difference(our_transcript, moonshine_transcript)
    if step is not None:
        print(
            f'      as text: ours "{token_text(tokenizer, our_transcript.tokens[step])}", '
            f'Moonshine "{token_text(tokenizer, moonshine_transcript.tokens[step])}"'
        )
    return [passed_a, passed_b]


def play_recording(
    name: str,
    samples: np.ndarray,
    transcriber: PushToTalkTranscriber,
    model: MoonshineStreamingForConditionalGeneration,
    tokenizer: MoonshineTokenizer,
    moonshine_ort_pieces: MoonshineOrtPieces,
) -> tuple[list[PressResult], list[bool]]:
    """Every press of one recording with its checks, then (C) and (D) on the whole recording."""
    print(f"[{name}: {len(samples) / SAMPLE_RATE_HZ:.2f} s]")
    presses = []
    results = []
    for press_index, (start_sample, end_sample) in enumerate(press_boundaries(samples)):
        press_samples = samples[start_sample:end_sample]
        start_seconds = start_sample / SAMPLE_RATE_HZ
        print(f"  press {press_index + 1}: {start_seconds:.2f} to {end_sample / SAMPLE_RATE_HZ:.2f} s")
        final, tentative_updates = play_press(transcriber, press_samples, start_seconds)
        results += check_press(model, tokenizer, moonshine_ort_pieces, transcriber, press_samples, final)
        presses.append(PressResult(start_sample, end_sample, final, tentative_updates))

    committed_text = " ".join(press.final.text for press in presses if press.final.text)
    print(f"  ours, every press's final text: {committed_text}")

    lines = moonshine_transcript_lines(samples, MOONSHINE_DEFAULT_OPTIONS)
    print(f"  (C) Moonshine's transcriber, default settings, on the whole file: {len(lines)} lines")
    for line in lines:
        print(f"      {line.start_time:6.2f} to {line.start_time + line.duration:6.2f} s: {line.text}")
    print_word_differences("words", committed_text, " ".join(line.text for line in lines if line.text))

    generated = run_model_generate_on_audio(model, samples, moonshine_token_cap(len(samples) / SAMPLE_RATE_HZ))
    generated_text = tokenizer.tokens_to_text(generated.tokens)
    print(f"  (D) model.generate on the whole file, as one segment: {generated.describe()}: {generated_text}")
    print_word_differences("words", committed_text, generated_text)
    return presses, results


def print_tokenizer_facts(tokenizer: MoonshineTokenizer) -> None:
    """What the port relies on, read from tokenizer.bin."""
    entries = tokenizer.entries_by_token_id
    special_ids = [token_id for token_id, entry in enumerate(entries) if is_special_entry(entry)]
    empty_ids = [token_id for token_id, entry in enumerate(entries) if not entry]
    byte_fallback = all(entries[3 + byte_value] == bytes([byte_value]) for byte_value in range(256))
    print(
        f"  {len(entries)} entries; special, skipped in text: ids {describe_id_ranges(special_ids)} "
        f"(start {entries[START_TOKEN_ID]}, end {entries[END_TOKEN_ID]}); empty: ids {empty_ids}; "
        f"ids 3-258 are the 256 single bytes: {byte_fallback}; longest entry {max(map(len, entries))} bytes"
    )


def describe_id_ranges(token_ids: list[int]) -> str:
    """[0, 1, 2, 7, 9, 10] -> "0-2, 7, 9-10"."""
    ranges = []
    for token_id in token_ids:
        if ranges and ranges[-1][1] == token_id - 1:
            ranges[-1][1] = token_id
        else:
            ranges.append([token_id, token_id])
    return ", ".join(f"{first}-{last}" if first != last else f"{first}" for first, last in ranges)


def print_stage_times(stage_times: StageTimes, updates: list[TranscriptUpdate]) -> None:
    """Every Core ML call by stage, then what matters for the app: stages 1-4 against the 80 ms a chunk lasts, and each
    update's decode against the 0.5 s update interval. The decoder steps include (A)'s second decodes."""
    for stage_name, milliseconds in stage_times.milliseconds_by_stage.items():
        values = np.array(milliseconds)
        print(
            f"  {stage_name}: {len(values)} calls, median {np.median(values):.1f} ms, 95th percentile "
            f"{np.percentile(values, 95):.1f} ms, max {values.max():.1f} ms, total {values.sum() / 1000:.1f} s"
        )
    chunk_milliseconds = np.array(stage_times.milliseconds_by_stage[CHUNK_STAGE_NAME])
    print(
        f"  stages 1-4 per chunk: median {np.median(chunk_milliseconds) / (CHUNK_SECONDS * 1000):.0%} of the chunk's "
        f"{CHUNK_SECONDS * 1000:.0f} ms"
    )
    decoded_updates = [update for update in updates if update.tokens]
    decode_milliseconds = np.array([update.decode_milliseconds for update in decoded_updates])
    step_counts = np.array([len(update.tokens) for update in decoded_updates])
    interval_milliseconds = UPDATE_INTERVAL_SAMPLE_COUNT / SAMPLE_RATE_HZ * 1000
    print(
        f"  {UPDATE_STAGE_NAME}: {len(decoded_updates)} updates, {step_counts.mean():.0f} steps on average "
        f"(max {step_counts.max()}); {int((decode_milliseconds > interval_milliseconds).sum())} took longer than the "
        f"{interval_milliseconds:.0f} ms interval"
    )


def main() -> int:
    torch.set_grad_enabled(False)  # nothing here trains
    tokenizer = MoonshineTokenizer()
    print("[tokenizer.bin]")
    print_tokenizer_facts(tokenizer)
    print("[the five Core ML packages, CPU and Neural Engine]")
    transcriber = PushToTalkTranscriber(load_pipeline_models(), tokenizer)
    model = load_pytorch_model()
    print("[Moonshine's .ort pieces]")
    print_onnxruntime_builds()
    moonshine_ort_pieces = MoonshineOrtPieces()

    results = []
    updates = []
    reference_arrays = {}
    for name, audio_path in TEST_AUDIO_PATHS.items():
        presses, recording_results = play_recording(
            name, load_audio(audio_path), transcriber, model, tokenizer, moonshine_ort_pieces
        )
        results += recording_results
        reference_arrays[f"{name}_press_boundaries"] = np.array(
            [(press.start_sample, press.end_sample) for press in presses], dtype=np.int64
        )
        for press_index, press in enumerate(presses):
            reference_arrays[f"{name}_press{press_index}_tokens"] = np.array(press.final.tokens, dtype=np.int32)
            updates += [*press.tentative_updates, press.final]

    print("[time per call on this Mac, in Python, with every input copied on every call]")
    print_stage_times(transcriber.stage_times, updates)

    if not all(results):
        print("\nFAILED")
        return 1
    np.savez(STEP6_REFERENCE_PATH, **reference_arrays)
    print(f"\nAll checks passed. Saved {STEP6_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
