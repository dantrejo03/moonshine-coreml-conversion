"""Which of Moonshine's 8-bit .ort pieces makes its text differ from the original model's. Run from conversion/:

    python -m pipeline.tools.diagnose_moonshine_differences

Every press of the test recordings (pipeline.test.press_boundaries) goes through the five stages the way Moonshine's
library runs a segment: only the audio its transcriber keeps (whole chunks: pipeline.test.moonshine_segment), and no
isFinal (the last 16 encoder frames never reach the memory). The first k stages are Moonshine's .ort pieces and the
rest float32 PyTorch, for k = 0 (all float: the original model) to 5 (all .ort: Moonshine's tokens, as check (B)
reproduces them). The first k whose tokens differ from k = 0 names the stage whose rounding tips the decoder. One more
line: all float the way the app runs a press, padded to whole chunks and with isFinal (check (A) matched the app to
it); where it differs from k = 0, the frames Moonshine leaves out change the text on their own.
"""

import sys

import torch

from common.audio import load_audio
from common.config import SAMPLE_RATE_HZ, TEST_AUDIO_PATHS
from common.load_model import load_pytorch_model
from pipeline.test import (
    STAGE_COUNT,
    MoonshineOrtPieces,
    moonshine_segment,
    moonshine_token_cap,
    press_boundaries,
    print_word_differences,
    run_five_stages,
)
from pipeline.tokenizer import MoonshineTokenizer
from step1_frontend.test import pad_to_whole_chunks, run_chunk_by_chunk, wrapper_chunk_runner
from step1_frontend.wrapper import StreamingFrontend
from step2_encoder.test import encode_streaming, wrapper_window_runner
from step2_encoder.wrapper import StreamingEncoder
from step3_adapter.test import run_adapter_streaming
from step3_adapter.test import wrapper_call_runner as adapter_wrapper_call_runner
from step3_adapter.wrapper import StreamingAdapter
from step4_cross_kv.test import append_whole_segment
from step4_cross_kv.test import wrapper_call_runner as cross_kv_wrapper_call_runner
from step4_cross_kv.wrapper import CrossAttentionKeyValueProjection
from step5_decoder.test import (
    ExplicitCacheSession,
    TestSegment,
    Transcript,
    decode,
    token_cap_for_seconds,
    wrapper_predict,
)
from step5_decoder.wrapper import DecoderStepWithExplicitCache

STAGE_NAMES = ("frontend", "encoder", "adapter", "cross_kv", "decoder")


class FloatPieces:
    """The same five stages as MoonshineOrtPieces in float32 PyTorch: each step's wrapper around one loaded checkpoint,
    run the streaming way that step's test runs it. Each step's check (a) showed its wrapper matches the original."""

    def __init__(self, model):
        self.run_frontend_chunk = wrapper_chunk_runner(StreamingFrontend(model.model.encoder.embedder).eval())
        self.run_encoder_window = wrapper_window_runner(StreamingEncoder(model.model.encoder).eval())
        self.run_adapter_call = adapter_wrapper_call_runner(StreamingAdapter(model.model.decoder).eval())
        self.run_cross_kv_call = cross_kv_wrapper_call_runner(
            CrossAttentionKeyValueProjection(model.model.decoder).eval()
        )
        self.decoder_predict = wrapper_predict(DecoderStepWithExplicitCache(model).eval())

    def features(self, samples):
        return run_chunk_by_chunk(self.run_frontend_chunk, samples)

    def encoded(self, features):
        return encode_streaming(self.run_encoder_window, features)

    def memory(self, encoded):
        return run_adapter_streaming(self.run_adapter_call, encoded)

    def cross_kv(self, memory):
        return append_whole_segment(self.run_cross_kv_call, memory)

    def decode(self, memory, k_cross, v_cross, token_cap) -> Transcript:
        segment = TestSegment(name="press", encoded=memory, k_cross=k_cross, v_cross=v_cross)  # memory: frame count
        return decode(ExplicitCacheSession(self.decoder_predict, segment), token_cap)


def describe_mix(ort_stage_count: int) -> str:
    if ort_stage_count == 0:
        return "all float (the original model)"
    if ort_stage_count == STAGE_COUNT:
        return "all .ort (Moonshine)"
    return f".ort through the {STAGE_NAMES[ort_stage_count - 1]}, float from the {STAGE_NAMES[ort_stage_count]}"


def print_against_all_float(label: str, transcript: Transcript, all_float: Transcript, tokenizer) -> None:
    if transcript.tokens == all_float.tokens:
        print(f"    {label}: same tokens")
        return
    print(f"    {label}: DIFFERENT tokens: {tokenizer.tokens_to_text(transcript.tokens) or '(no words)'}")
    print_word_differences("words (ours = all float)", tokenizer.tokens_to_text(all_float.tokens),
                           tokenizer.tokens_to_text(transcript.tokens))


def main() -> int:
    torch.set_grad_enabled(False)  # nothing here trains
    tokenizer = MoonshineTokenizer()
    float_pieces = FloatPieces(load_pytorch_model())
    ort_pieces = MoonshineOrtPieces()
    for name, audio_path in TEST_AUDIO_PATHS.items():
        samples = load_audio(audio_path)
        print(f"[{name}]")
        for press_index, (start_sample, end_sample) in enumerate(press_boundaries(samples)):
            press_samples = samples[start_sample:end_sample]
            kept_samples, segment_seconds = moonshine_segment(press_samples)
            token_cap = moonshine_token_cap(segment_seconds)
            print(
                f"  press {press_index + 1}: {start_sample / SAMPLE_RATE_HZ:.2f} to {end_sample / SAMPLE_RATE_HZ:.2f} s; "
                f"Moonshine keeps {len(kept_samples) / SAMPLE_RATE_HZ:.2f} s of it and transcribes all but its last "
                f"0.32 s, token cap {token_cap}"
            )
            transcripts = [
                run_five_stages(
                    [ort_pieces] * ort_stage_count + [float_pieces] * (STAGE_COUNT - ort_stage_count),
                    kept_samples,
                    token_cap,
                    is_final=False,
                )
                for ort_stage_count in range(STAGE_COUNT + 1)
            ]
            all_float = transcripts[0]
            print(f"    0, {describe_mix(0)}: {tokenizer.tokens_to_text(all_float.tokens) or '(no words)'}")
            for ort_stage_count, transcript in enumerate(transcripts[1:], start=1):
                print_against_all_float(
                    f"{ort_stage_count}, {describe_mix(ort_stage_count)}", transcript, all_float, tokenizer
                )
            app_all_float = run_five_stages(
                [float_pieces] * STAGE_COUNT,
                pad_to_whole_chunks(press_samples),
                token_cap_for_seconds(len(press_samples) / SAMPLE_RATE_HZ),
                is_final=True,
            )
            print_against_all_float(
                "all float as the app runs a press (padded, isFinal)", app_all_float, all_float, tokenizer
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
