"""Paths and fixed sizes shared by every step.

These values are contracts between steps. Change one only with a reason, and record the
change in the plan's Findings log so later steps see it.
"""

import json
from pathlib import Path

CONVERSION_DIRECTORY = Path(__file__).resolve().parent.parent
MODELS_DIRECTORY = CONVERSION_DIRECTORY / "models"  # every step's .mlpackage outputs (gitignored)
REFERENCE_TENSORS_DIRECTORY = CONVERSION_DIRECTORY / "ref"  # one stepN.npz per step

THIRD_PARTY_DIRECTORY = CONVERSION_DIRECTORY / "third_party"
MOONSHINE_V2_REPOSITORY_DIRECTORY = THIRD_PARTY_DIRECTORY / "moonshine-v2"
MOONSHINE_ORT_MODEL_DIRECTORY = (
    THIRD_PARTY_DIRECTORY
    / "moonshine_voice_models/download.moonshine.ai/model/tiny-streaming-en/quantized_26_11_02"
)
STREAMING_CONFIG_PATH = MOONSHINE_ORT_MODEL_DIRECTORY / "streaming_config.json"
TOKENIZER_PATH = MOONSHINE_ORT_MODEL_DIRECTORY / "tokenizer.bin"

# The frontend ships as two files: frontend.weights.ort (no inputs) produces tensors that
# frontend.model.ort takes as inputs. Moonshine's repo code still names a single frontend.ort.
MOONSHINE_ORT_PIECE_FILE_NAMES = [
    "frontend.model.ort",
    "frontend.weights.ort",
    "encoder.ort",
    "adapter.ort",
    "cross_kv.ort",
    "decoder_kv.ort",
]

TEST_AUDIO_PATHS = {
    "beckett": MOONSHINE_V2_REPOSITORY_DIRECTORY / "test-assets/beckett.wav",
    "two_cities": MOONSHINE_V2_REPOSITORY_DIRECTORY / "test-assets/two_cities_16k.wav",
}

HUGGING_FACE_MODEL_ID = "UsefulSensors/moonshine-streaming-tiny"

# Fixed sizes from the plan's Architecture map and Open decisions.
SAMPLE_RATE_HZ = 16_000
AUDIO_SAMPLES_PER_CHUNK = 1280  # 80 ms
FEATURE_FRAMES_PER_CHUNK = 4  # one feature frame per 20 ms
FEATURE_FRAME_SECONDS = AUDIO_SAMPLES_PER_CHUNK / FEATURE_FRAMES_PER_CHUNK / SAMPLE_RATE_HZ  # 0.02; encoded frames too
ENCODER_DIM = 320  # streaming_config.json encoder_dim: numbers per feature frame and per encoded frame
DECODER_DIM = 320  # streaming_config.json decoder_dim: numbers per memory row, what the decoder reads
DECODER_LAYER_COUNT = 6  # streaming_config.json depth
DECODER_HEAD_COUNT = 8  # nheads: attention heads per decoder layer; also num_key_value_heads (no key/value sharing)
DECODER_HEAD_DIM = 40  # head_dim: numbers per head; DECODER_HEAD_COUNT x DECODER_HEAD_DIM = DECODER_DIM
FRONTEND_CONV1_BUFFER_SHAPE = (1, 320, 4)  # conv1's last 4 inputs (kernel 5 minus 1), carried between chunks
FRONTEND_CONV2_BUFFER_SHAPE = (1, 640, 4)  # conv2's last 4 inputs
ENCODER_LEFT_CONTEXT_FRAME_COUNT = 96  # 16 frames back per layer x 6 layers: exactly how far an encoded frame reaches
ENCODER_HELD_BACK_FRAME_COUNT = 16  # streaming_config.json total_lookahead: 4 frames ahead x 4 layers, exactly
ENCODER_WINDOW_FRAME_COUNT = (
    ENCODER_LEFT_CONTEXT_FRAME_COUNT + FEATURE_FRAMES_PER_CHUNK + ENCODER_HELD_BACK_FRAME_COUNT
)  # 116
MEMORY_FRAMES_PER_CALL = FEATURE_FRAMES_PER_CHUNK  # 4: frames per adapter call (Step 3) and cross K/V call (Step 4)
# The adapter's position table (pos_emb) has 4096 rows, but only rows 0-1999 (the first 40 s of a segment) were trained;
# rows 2000-4095 are still random (step3_adapter/tools/inspect_position_table.py). A segment must fit inside them.
ADAPTER_TRAINED_POSITION_COUNT = 2000
MAXIMUM_MEMORY_FRAME_COUNT = 768  # Mmax: 15 s segment x 50 frames per second, rounded up; <= ADAPTER_TRAINED_POSITION_COUNT
MAXIMUM_TOKEN_COUNT = 128  # Lmax: 15 s x 6.5 tokens per second, rounded up
MAXIMUM_TOKENS_PER_SECOND = 6.5  # Moonshine's token cap per second of audio (transcriber.h, max_tokens_per_second)
START_TOKEN_ID = 1  # streaming_config.json bos_id: the decoder's first input, at position 0
END_TOKEN_ID = 2  # eos_id: decoding stops once the decoder picks it
VOCABULARY_SIZE = 32768  # vocab_size: scores per decoder step
MASK_BLOCKED_VALUE = -1e4  # not -inf, so float16 math stays finite


# Step 5b's model (decoder_step.mlpackage) holds each row's 40 numbers (head_dim) padded with zeros to 64, a multiple of
# 32, in its whole-row buffers. The padding is never read.
#   - Its self-attention cache, the state: with 40, the Neural Engine compiles the model but fails every prediction
#     (step5_decoder/tools/diagnose_state_on_neural_engine.py).
#   - Its k_cross and v_cross inputs: with 40, Core ML re-lays out both buffers on every call (Instruments on the
#     Series 9, 2026-11-12: E5RT PerformDataFormatChange, mostly memmove). Padded, that's gone and the median decoder
#     step went from 19.7 to 15.7 ms, with identical tokens.
DECODER_PADDED_HEAD_DIM = 64


def attention_cache_shape(row_count: int, numbers_per_row: int = DECODER_HEAD_DIM) -> tuple[int, ...]:
    """[layer, batch, head, row, head_dim]: the decoder's cached keys or values for row_count rows. Rows are memory
    frames for cross-attention (Step 4's k_cross, v_cross) and tokens for self-attention (Step 5's k_self, v_self).
    numbers_per_row is head_dim, or DECODER_PADDED_HEAD_DIM for Step 5b's state and cross K/V inputs."""
    return (DECODER_LAYER_COUNT, 1, DECODER_HEAD_COUNT, row_count, numbers_per_row)


def load_streaming_config() -> dict:
    """Moonshine's own sizes and special tokens (encoder_dim, depth, vocab_size, bos_id, ...)."""
    return json.loads(STREAMING_CONFIG_PATH.read_text())
