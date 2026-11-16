"""Step 5 checks for the decoder step. Run from the top of the repo, after step4_cross_kv.test (it writes ref/step4.npz)
and step5_decoder.convert:

    python -m step5_decoder.test

Each test segment is decoded greedily, as Moonshine's transcriber does: the start token at position 0, then each
step's highest score is the next token, until the end token or the token cap.
Part 5a (caches passed in and out):
(a) the wrapper vs model.generate on the same memory: identical tokens and scores; also the masks, by filling every
    row they block with noise,
(b) Core ML vs wrapper: identical tokens and scores,
(c) Moonshine's decoder_kv.ort vs wrapper: identical tokens, and scores on the wrapper's tokens.
Part 5b (the self-attention cache in Core ML state): the stateful wrapper and decoder_step.mlpackage vs the 5a wrapper,
token for token, including one state kept across transcript updates as Swift will.
Exits non-zero if any check fails; writes ref/step5.npz only when all pass.
"""

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import coremltools
import numpy as np
import onnxruntime
import torch
from transformers import MoonshineStreamingForConditionalGeneration
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingEncoderModelOutput

from common.compare import (
    COREML_FLOAT16_MIN_COSINE_SIMILARITY,
    COREML_FLOAT32_MAX_ABS_DIFF,
    MOONSHINE_ORT_MIN_COSINE_SIMILARITY,
    WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
    compare,
    worst_frame_cosine_similarity,
)
from common.config import (
    DECODER_HEAD_DIM,
    DECODER_LAYER_COUNT,
    DECODER_PADDED_HEAD_DIM,
    END_TOKEN_ID,
    FEATURE_FRAME_SECONDS,
    MAXIMUM_MEMORY_FRAME_COUNT,
    MAXIMUM_TOKEN_COUNT,
    MAXIMUM_TOKENS_PER_SECOND,
    REFERENCE_TENSORS_DIRECTORY,
    START_TOKEN_ID,
    VOCABULARY_SIZE,
    attention_cache_shape,
    load_streaming_config,
)
from common.load_model import load_ort_session, load_pytorch_model
from step2_encoder.test import STEP2_REFERENCE_PATH
from step4_cross_kv.test import STEP4_REFERENCE_PATH, load_reference_by_recording
from step5_decoder.convert import (
    DECODER_STEP_MODEL_PATH,
    EXPLICIT_CACHE_FLOAT32_MODEL_PATH,
    EXPLICIT_CACHE_MODEL_PATH,
    FLOAT16_PACKAGE_INPUT_DATA_TYPES,
    INPUT_SHAPES,
    OUTPUT_NAMES,
    STATE_SHAPES,
    STATEFUL_INPUT_SHAPES,
)
from step5_decoder.wrapper import DecoderStepWithExplicitCache, DecoderStepWithStatefulCache

STEP5_REFERENCE_PATH = REFERENCE_TENSORS_DIRECTORY / "step5.npz"
CACHE_ROW_AXIS = 3  # caches are [layer, batch, head, row, head_dim]; rows are tokens (self) or memory frames (cross)
UNUSED_ROWS_NOISE_SEED = 0
DECODER_KV_ORT_INPUT_NAMES = ["token", "k_self", "v_self", "out_k_cross", "out_v_cross"]
DECODER_KV_ORT_OUTPUT_NAMES = ["logits", "out_k_self", "out_v_self"]

# Runs the 5a model once: its inputs by name -> its outputs by name, as Core ML's predict does.
# One per implementation: the wrapper, and Core ML.
Predict = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def token_cap_for_seconds(audio_seconds: float) -> int:
    """How many decoder steps one transcript update may run: Moonshine's min(ceil(seconds x 6.5), 256)
    (transcriber.cpp 568-572), with Lmax in place of 256, since the self-attention cache has Lmax rows. Moonshine's
    seconds are the segment's audio so far."""
    return min(math.ceil(audio_seconds * MAXIMUM_TOKENS_PER_SECOND), MAXIMUM_TOKEN_COUNT)


def token_cap(memory_frame_count: int) -> int:
    """token_cap_for_seconds for a segment known only by its memory: its memory frames x 20 ms (the audio padded to
    whole chunks). Step 6 has the audio, and uses its seconds."""
    return token_cap_for_seconds(memory_frame_count * FEATURE_FRAME_SECONDS)


@dataclass
class TestSegment:
    """One segment's memory, as Steps 2 and 4 left it: a test recording's first min(M, Mmax) frames. A segment that
    starts at the recording's frame 0 has positions 0, 1, 2, ..., so Steps 3 and 4 computed exactly its rows."""

    name: str
    encoded: np.ndarray  # [1, M, 320]: Step 2's frames, which model.generate gets in place of its own encoder's
    k_cross: np.ndarray  # [6,1,8,M,40]: Step 4's keys for the same frames
    v_cross: np.ndarray  # [6,1,8,M,40]: and values

    @property
    def memory_frame_count(self) -> int:
        return self.encoded.shape[1]

    @property
    def token_cap(self) -> int:
        return token_cap(self.memory_frame_count)


def load_test_segments() -> list[TestSegment]:
    """beckett fits whole (500 frames, 10 s), so 268 of the buffer's 768 frames stay unused and masked. two_cities is
    2220 frames, more than Mmax, so its segment is its first 768 frames (15.36 s): a full buffer. Step 6 segments
    whole recordings."""
    encoded_by_recording = load_reference_by_recording(STEP2_REFERENCE_PATH, "encoded", "step2_encoder.test")
    k_cross_by_recording = load_reference_by_recording(STEP4_REFERENCE_PATH, "k_cross", "step4_cross_kv.test")
    v_cross_by_recording = load_reference_by_recording(STEP4_REFERENCE_PATH, "v_cross", "step4_cross_kv.test")
    segments = []
    for recording_name, encoded in encoded_by_recording.items():
        if k_cross_by_recording[recording_name].shape[CACHE_ROW_AXIS] != encoded.shape[1]:
            raise ValueError(f"{recording_name}: ref/step4.npz and ref/step2.npz have different frame counts")
        frame_count = min(encoded.shape[1], MAXIMUM_MEMORY_FRAME_COUNT)
        segments.append(TestSegment(
            name=recording_name,
            encoded=encoded[:, :frame_count],
            k_cross=np.ascontiguousarray(k_cross_by_recording[recording_name][:, :, :, :frame_count]),
            v_cross=np.ascontiguousarray(v_cross_by_recording[recording_name][:, :, :, :frame_count]),
        ))
    return segments


@dataclass
class Transcript:
    tokens: list[int]  # the token each step picked, in order; the last is the end token if it came before the cap
    logits: np.ndarray  # [steps, 32768]: each step's scores, before picking

    def describe(self) -> str:
        ending = "ends with the end token" if self.tokens[-1] == END_TOKEN_ID else "stopped at the cap"
        return f"{len(self.tokens)} tokens, {ending}"


class DecoderSession(Protocol):
    """One implementation of the decoder step, and the state it carries from step to step."""

    def start_transcript(self) -> None:
        """Moonshine's decoder_reset: forget every earlier token. Each transcript update decodes from scratch."""

    def step(self, token: int, position: int) -> np.ndarray:
        """One decoder call: `token` at `position` in, the scores for the next token out, [32768]."""


def decode(session: DecoderSession, maximum_step_count: int, tokens_to_feed: list[int] | None = None) -> Transcript:
    """Moonshine's decoding loop (transcriber.cpp 574-604), which Swift ports: the start token at position 0; each
    step's highest-scoring token is fed back at the next position; stop after the end token or maximum_step_count steps.

    tokens_to_feed (teacher forcing): feed these, one per step, instead of the session's own picks, for exactly that
    many steps. Two implementations fed the same tokens score the same prefixes, so their scores can be compared step
    by step even after their own picks part."""
    session.start_transcript()
    token = START_TOKEN_ID
    picked_tokens = []
    logits_by_step = []
    step_count = maximum_step_count if tokens_to_feed is None else len(tokens_to_feed)
    for position in range(step_count):
        logits = session.step(token, position)
        picked_token = int(np.argmax(logits))  # the first of equal highest scores, as Moonshine's loop picks
        picked_tokens.append(picked_token)
        logits_by_step.append(logits)
        if tokens_to_feed is not None:
            token = tokens_to_feed[position]
        elif picked_token == END_TOKEN_ID:
            break
        else:
            token = picked_token
    return Transcript(picked_tokens, np.stack(logits_by_step))


class UnusedRows:
    """What fills cache rows and buffer frames the masks block: zeros, or with noise=True, random values. Noise is
    what the app's buffers hold if they aren't cleared: an earlier segment's K/V past the memory's end, an earlier
    transcript update's tokens in the self-attention cache. The masks block both, so results must not change."""

    def __init__(self, noise: bool = False):
        self.noise = noise
        self.random_generator = np.random.default_rng(UNUSED_ROWS_NOISE_SEED)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        if self.noise:
            return self.random_generator.standard_normal(shape).astype(np.float32)
        return np.zeros(shape, dtype=np.float32)


def cross_attention_inputs(
    segment: TestSegment, cross_kv_data_type: type, unused_rows: UnusedRows, numbers_per_row: int = DECODER_HEAD_DIM
) -> dict[str, np.ndarray]:
    """Swift's cross-attention buffers for one segment, allocated once: k_cross and v_cross [6,1,8,Mmax,40], holding
    the segment's frames first (in the app, Step 4's appends put them there), and cross_valid [1,Mmax], 1 for those
    frames and 0 after. cross_kv_data_type is float16 for the watch packages, as Swift's buffers are.
    numbers_per_row is DECODER_PADDED_HEAD_DIM for 5b: each frame's 40 numbers, then padding filled like unused rows
    (zeros, or noise, which the model must never read)."""
    frame_count = segment.memory_frame_count
    inputs = {}
    for name, segment_rows in (("k_cross", segment.k_cross), ("v_cross", segment.v_cross)):
        buffer = unused_rows(attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT, numbers_per_row))
        buffer[:, :, :, :frame_count, :DECODER_HEAD_DIM] = segment_rows
        inputs[name] = buffer.astype(cross_kv_data_type)
    inputs["cross_valid"] = np.zeros((1, MAXIMUM_MEMORY_FRAME_COUNT), dtype=np.float32)
    inputs["cross_valid"][:, :frame_count] = 1.0
    return inputs


def token_and_position_inputs(token: int, position: int) -> dict[str, np.ndarray]:
    return {"token": np.array([[token]], dtype=np.int32), "position": np.array([position], dtype=np.int32)}


class ExplicitCacheSession:
    """Swift's bookkeeping for the 5a model (the wrapper, or Core ML) over one segment: the cross-attention buffers
    (cross_attention_inputs), and for each transcript update, k_self and v_self [6,1,8,Lmax,40], which start empty;
    every step passes them in and keeps k_self_out and v_self_out for the next step."""

    def __init__(
        self,
        predict: Predict,
        segment: TestSegment,
        cross_kv_data_type: type = np.float32,
        fill_unused_rows_with_noise: bool = False,
    ):
        self.predict = predict
        self.unused_rows = UnusedRows(noise=fill_unused_rows_with_noise)
        self.cross_inputs = cross_attention_inputs(segment, cross_kv_data_type, self.unused_rows)

    def start_transcript(self) -> None:
        self.k_self = self.unused_rows(attention_cache_shape(MAXIMUM_TOKEN_COUNT))
        self.v_self = self.unused_rows(attention_cache_shape(MAXIMUM_TOKEN_COUNT))

    def step(self, token: int, position: int) -> np.ndarray:
        outputs = self.predict({
            **token_and_position_inputs(token, position),
            "k_self": self.k_self,
            "v_self": self.v_self,
            **self.cross_inputs,
        })
        self.k_self = outputs["k_self_out"]
        self.v_self = outputs["v_self_out"]
        return outputs["logits"][0, 0]


class StatefulCacheSession:
    """Swift's bookkeeping for the 5b model (decoder_step.mlpackage, or its PyTorch wrapper) over one segment: the
    cross-attention buffers (cross_attention_inputs, rows padded to 64), and a state holding the self-attention cache,
    which predict reads and writes; Swift makes it once per segment (MLModel.makeState()) and passes it to every
    prediction.

    A new transcript update needs nothing reset: it starts again at position 0, each step writes its own row before
    reading it, and the mask blocks every later row, so an earlier update's rows are never read."""

    def __init__(self, predict: Predict, segment: TestSegment, cross_kv_data_type: type, unused_rows: UnusedRows):
        self.predict = predict
        self.cross_inputs = cross_attention_inputs(
            segment, cross_kv_data_type, unused_rows, numbers_per_row=DECODER_PADDED_HEAD_DIM
        )

    def start_transcript(self) -> None:
        pass

    def step(self, token: int, position: int) -> np.ndarray:
        return self.predict({**token_and_position_inputs(token, position), **self.cross_inputs})["logits"][0, 0]


class DecoderKvOrtSession:
    """Moonshine's run_decoder_with_cross_kv (moonshine-streaming-model.cpp 843-989) with decoder_kv.ort. The
    self-attention cache starts with 0 rows and grows by one per call (concatenation), and decoder_kv.ort takes the
    position from its length. Cross K/V are exactly the segment's frames: no buffer, so no mask. Integers are int64."""

    def __init__(self, ort_session: onnxruntime.InferenceSession, segment: TestSegment):
        self.ort_session = ort_session
        self.k_cross = segment.k_cross
        self.v_cross = segment.v_cross

    def start_transcript(self) -> None:
        self.k_self = np.zeros(attention_cache_shape(0), dtype=np.float32)
        self.v_self = np.zeros(attention_cache_shape(0), dtype=np.float32)

    def step(self, token: int, position: int) -> np.ndarray:
        if position != self.k_self.shape[CACHE_ROW_AXIS]:
            raise ValueError(f"position {position}, but decoder_kv.ort's position is its cache length")
        logits, self.k_self, self.v_self = self.ort_session.run(DECODER_KV_ORT_OUTPUT_NAMES, {
            "token": np.array([[token]], dtype=np.int64),
            "k_self": self.k_self,
            "v_self": self.v_self,
            "out_k_cross": self.k_cross,
            "out_v_cross": self.v_cross,
        })
        return logits[0, -1]


def wrapper_predict(wrapper: DecoderStepWithExplicitCache) -> Predict:
    def predict(inputs):
        outputs = wrapper(*(torch.from_numpy(inputs[name]) for name in INPUT_SHAPES))
        return {name: output.numpy() for name, output in zip(OUTPUT_NAMES, outputs)}

    return predict


def stateful_wrapper_predict(stateful_wrapper: DecoderStepWithStatefulCache) -> Predict:
    """The 5b PyTorch wrapper, whose buffers are the state; they start at zero."""
    for state_name in STATE_SHAPES:
        getattr(stateful_wrapper, state_name).zero_()

    def predict(inputs):
        logits = stateful_wrapper(*(torch.from_numpy(inputs[name]) for name in STATEFUL_INPUT_SHAPES))
        return {"logits": logits.numpy()}

    return predict


def coreml_stateful_predict(coreml_model: coremltools.models.MLModel, state) -> Predict:
    """decoder_step.mlpackage with one MLState, which every call reads and updates."""

    def predict(inputs):
        return coreml_model.predict(inputs, state=state)

    return predict


def compare_scores(label: str, logits: np.ndarray, reference_logits: np.ndarray, **pass_bar) -> bool:
    """compare(), plus the worst single step: a step whose scores are off can hide in the whole-transcript numbers."""
    result = compare(label, logits, reference_logits, **pass_bar)
    worst_cosine, worst_step = worst_frame_cosine_similarity(logits, reference_logits)
    print(f"      worst single step: cosine {worst_cosine:.6f} at step {worst_step}")
    return result["passed"]


def compare_with_reference(
    label: str, session: DecoderSession, reference: Transcript, maximum_step_count: int, **pass_bar
) -> tuple[bool, Transcript]:
    """Decodes greedily with session; passes when its tokens are the reference's and its scores meet pass_bar.
    Returns that and session's own greedy transcript.

    If the tokens are the same, the greedy run fed the reference's tokens, so its scores line up step by step. If not,
    prints the first step where they part and how close it was, then decodes again fed the reference's tokens
    (teacher forcing) and compares those scores."""
    started_seconds = time.perf_counter()
    transcript = decode(session, maximum_step_count)
    milliseconds_per_step = (time.perf_counter() - started_seconds) * 1000 / len(transcript.tokens)
    tokens_match = transcript.tokens == reference.tokens
    print(
        f"{label}: {transcript.describe()}, {milliseconds_per_step:.1f} ms per step on this Mac; "
        f"tokens {'identical' if tokens_match else 'DIFFER: FAIL'}"
    )
    scored = transcript
    if not tokens_match:
        print_first_difference(transcript, reference)
        scored = decode(session, maximum_step_count, tokens_to_feed=reference.tokens)
        print("      scores below: decoded again, fed the reference's tokens")
    scores_pass = compare_scores("      scores", scored.logits, reference.logits, **pass_bar)
    return tokens_match and scores_pass, transcript


def print_first_difference(transcript: Transcript, reference: Transcript) -> int | None:
    """Where two greedy decodes part, and by how much each preferred its own pick: a small gap is a near-tie.
    Returns that step, or None when one transcript is a prefix of the other."""
    step = next(
        (index for index, (picked, expected) in enumerate(zip(transcript.tokens, reference.tokens))
         if picked != expected),
        min(len(transcript.tokens), len(reference.tokens)),
    )
    if step == min(len(transcript.tokens), len(reference.tokens)):
        print(f"      one transcript is a prefix of the other: {transcript.describe()} vs {reference.describe()}")
        return None
    picked_token = transcript.tokens[step]
    expected_token = reference.tokens[step]
    gap_here = transcript.logits[step][picked_token] - transcript.logits[step][expected_token]
    gap_in_reference = reference.logits[step][expected_token] - reference.logits[step][picked_token]
    print(
        f"      first difference at step {step} of {len(reference.tokens)}: picked {picked_token}, reference "
        f"{expected_token}. Score gap between the two: {gap_here:.3f} here, {gap_in_reference:.3f} in the reference"
    )
    return step


def check_shapes(wrapper: DecoderStepWithExplicitCache, segments: list[TestSegment]) -> None:
    """The planned sizes match Moonshine's, decoder_kv.ort's names are the ones its session uses here, and the facts
    behind the wrapper's choices, printed."""
    streaming_config = load_streaming_config()
    for planned_name, planned_value, actual_value in (
        ("START_TOKEN_ID vs streaming_config bos_id", START_TOKEN_ID, streaming_config["bos_id"]),
        ("END_TOKEN_ID vs streaming_config eos_id", END_TOKEN_ID, streaming_config["eos_id"]),
        ("VOCABULARY_SIZE vs streaming_config vocab_size", VOCABULARY_SIZE, streaming_config["vocab_size"]),
        ("VOCABULARY_SIZE vs embedding rows", VOCABULARY_SIZE, wrapper.embed_tokens.weight.shape[0]),
        ("DECODER_LAYER_COUNT vs decoder layers", DECODER_LAYER_COUNT, len(wrapper.layers)),
        ("MAXIMUM_TOKEN_COUNT vs rotary table rows", MAXIMUM_TOKEN_COUNT, wrapper.rotary_cos_by_position.shape[0]),
    ):
        if planned_value != actual_value:
            raise ValueError(f"{planned_name}: planned {planned_value}, actual {actual_value}")

    rotated_count = int((wrapper.rotary_sin_by_position[1] != 0).sum())
    print(
        f"  proj_out is the embedding table ({tuple(wrapper.embed_tokens.weight.shape)}, checked when the wrapper is "
        f"built); rotary positions turn {rotated_count} of each head's {wrapper.rotary_cos_by_position.shape[1]} "
        f"numbers"
    )
    # Moonshine's cap for a full segment, before Lmax limits it: Lmax must not cut a full segment short.
    full_segment_token_cap = math.ceil(
        MAXIMUM_MEMORY_FRAME_COUNT * FEATURE_FRAME_SECONDS * MAXIMUM_TOKENS_PER_SECOND
    )
    if full_segment_token_cap > MAXIMUM_TOKEN_COUNT:
        raise ValueError(f"a full segment's token cap {full_segment_token_cap} is over Lmax {MAXIMUM_TOKEN_COUNT}")
    print(
        f"  Lmax {MAXIMUM_TOKEN_COUNT} tokens; Mmax {MAXIMUM_MEMORY_FRAME_COUNT} frames, whose token cap is "
        f"{full_segment_token_cap}"
    )
    for segment in segments:
        print(
            f"  {segment.name}: {segment.memory_frame_count} memory frames "
            f"({segment.memory_frame_count * FEATURE_FRAME_SECONDS:.2f} s), token cap {segment.token_cap}"
        )
    for shape, description in (
        (attention_cache_shape(MAXIMUM_TOKEN_COUNT), "self-attention cache in 5a"),
        (STATE_SHAPES["k_self"], "self-attention cache as 5b's state, rows padded to 64"),
        (attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT), "cross K/V buffers in 5a"),
        (STATEFUL_INPUT_SHAPES["k_cross"], "cross K/V buffers as 5b's inputs, rows padded to 64"),
    ):
        print(f"  {description}: 2 x {shape}, {2 * math.prod(shape) * 2 / 1e6:.2f} MB together as float16")

    ort_session = load_ort_session("decoder_kv.ort")
    ort_input_names = [ort_input.name for ort_input in ort_session.get_inputs()]
    ort_output_names = [ort_output.name for ort_output in ort_session.get_outputs()]
    # Every input is fed. Of the outputs, only these are read: it also returns out_k_cross and out_v_cross, passed
    # through unchanged (Step 1's findings).
    if ort_input_names != DECODER_KV_ORT_INPUT_NAMES or not set(DECODER_KV_ORT_OUTPUT_NAMES) <= set(ort_output_names):
        raise ValueError(f"decoder_kv.ort: inputs {ort_input_names}, outputs {ort_output_names}")


def run_model_generate(model: MoonshineStreamingForConditionalGeneration, segment: TestSegment) -> Transcript:
    """The original: model.generate, greedy, with the segment's encoder output from Step 2 instead of running
    Transformers' encoder (its windows are one frame short each way: Step 2, deviation 1), so both sides read the same
    memory. decoder.forward adds pos_emb to it in place on every step (line 816), so it gets a copy; the cross K/V
    are computed on the first step and cached, so the later additions don't reach them."""
    return generate_greedy(
        model,
        segment.token_cap,
        encoder_outputs=MoonshineStreamingEncoderModelOutput(
            last_hidden_state=torch.from_numpy(segment.encoded).clone()
        ),
    )


def generate_greedy(
    model: MoonshineStreamingForConditionalGeneration, max_new_tokens: int, **model_inputs
) -> Transcript:
    """model.generate, greedy, on model_inputs (encoder_outputs, or input_values and attention_mask), as a Transcript:
    the tokens it picked after the start token, and each step's scores."""
    output = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        output_logits=True,
        return_dict_in_generate=True,
    )
    sequence = output.sequences[0].tolist()
    if sequence[0] != START_TOKEN_ID:
        raise ValueError(f"model.generate started with token {sequence[0]}, not {START_TOKEN_ID}")
    return Transcript(sequence[1:], torch.cat(output.logits).numpy())  # logits: one [1, 32768] per step


def check_a_wrapper_vs_generate(
    model: MoonshineStreamingForConditionalGeneration,
    wrapper: DecoderStepWithExplicitCache,
    segments: list[TestSegment],
) -> tuple[list[bool], dict[str, Transcript]]:
    """(a) The plan's "wrapper greedy decode vs model.generate (greedy)", then the masks: the same decode with every
    row they block full of noise. Returns the results and the wrapper's transcripts, the reference for (b) and (c)."""
    predict = wrapper_predict(wrapper)
    results = []
    wrapper_transcripts = {}
    for segment in segments:
        generated = run_model_generate(model, segment)
        print(f"  {segment.name}: model.generate: {generated.describe()}")
        passed, wrapper_transcripts[segment.name] = compare_with_reference(
            f"  {segment.name} wrapper vs model.generate",
            ExplicitCacheSession(predict, segment),
            generated,
            segment.token_cap,
            maximum_allowed_max_abs_diff=WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
        )
        results.append(passed)
        passed, _ = compare_with_reference(
            f"  {segment.name} wrapper with noise in every masked row vs zeros",
            ExplicitCacheSession(predict, segment, fill_unused_rows_with_noise=True),
            wrapper_transcripts[segment.name],
            segment.token_cap,
            maximum_allowed_max_abs_diff=WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
        )
        results.append(passed)
    return results, wrapper_transcripts


def check_b_coreml_vs_wrapper(segments: list[TestSegment], wrapper_transcripts: dict[str, Transcript]) -> list[bool]:
    """(b) Core ML, decoding the way Swift will, vs the wrapper. float16 runs twice: with ALL (the plan's setting) and
    with CPU_AND_NE (what Step 7's app loads). The float16 package gets float16 cross K/V, as Swift's buffers are."""
    for model_path in (EXPLICIT_CACHE_FLOAT32_MODEL_PATH, EXPLICIT_CACHE_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run python -m step5_decoder.convert first)")

    float16_cross_kv_data_type = FLOAT16_PACKAGE_INPUT_DATA_TYPES["k_cross"]
    float16_pass_bar = {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}
    results = []
    for description, model_path, compute_units, cross_kv_data_type, pass_bar in (
        ("float32, CPU only", EXPLICIT_CACHE_FLOAT32_MODEL_PATH, coremltools.ComputeUnit.CPU_ONLY, np.float32,
         {"maximum_allowed_max_abs_diff": COREML_FLOAT32_MAX_ABS_DIFF}),
        ("float16, all compute units", EXPLICIT_CACHE_MODEL_PATH, coremltools.ComputeUnit.ALL,
         float16_cross_kv_data_type, float16_pass_bar),
        ("float16, CPU and Neural Engine", EXPLICIT_CACHE_MODEL_PATH, coremltools.ComputeUnit.CPU_AND_NE,
         float16_cross_kv_data_type, float16_pass_bar),
    ):
        print(f"[(b) Core ML {description} vs wrapper]")
        coreml_model = coremltools.models.MLModel(str(model_path), compute_units=compute_units)
        for segment in segments:
            passed, _ = compare_with_reference(
                f"  {segment.name}",
                ExplicitCacheSession(coreml_model.predict, segment, cross_kv_data_type=cross_kv_data_type),
                wrapper_transcripts[segment.name],
                segment.token_cap,
                **pass_bar,
            )
            results.append(passed)
    return results


def check_c_ort_vs_wrapper(segments: list[TestSegment], wrapper_transcripts: dict[str, Transcript]) -> list[bool]:
    """(c) Moonshine's decoder_kv.ort, called as Moonshine calls it, on the same cross K/V (Step 4's, float32), vs the
    wrapper. decoder_kv.ort rounds the embedding table, the weights and every matrix multiply's input to 8 bits.

    For scale first: the wrapper's scores at each step vs at the next step. Score vectors for different tokens share a
    lot, so this shows how high a cosine is for scores that aren't the same."""
    print(
        f"  bar: identical tokens, and scores cosine >= {MOONSHINE_ORT_MIN_COSINE_SIMILARITY} on the wrapper's tokens"
    )
    ort_session = load_ort_session("decoder_kv.ort")
    results = []
    for segment in segments:
        wrapper_logits = wrapper_transcripts[segment.name].logits
        for_scale_label = f"  {segment.name}, for scale: wrapper, each step vs the next"
        compare(for_scale_label, wrapper_logits[1:], wrapper_logits[:-1])
        passed, _ = compare_with_reference(
            f"  {segment.name} decoder_kv.ort vs wrapper",
            DecoderKvOrtSession(ort_session, segment),
            wrapper_transcripts[segment.name],
            segment.token_cap,
            minimum_allowed_cosine_similarity=MOONSHINE_ORT_MIN_COSINE_SIMILARITY,
        )
        results.append(passed)
    return results


def check_5b_stateful_cache(
    stateful_wrapper: DecoderStepWithStatefulCache,
    segments: list[TestSegment],
    wrapper_transcripts: dict[str, Transcript],
) -> list[bool]:
    """Part 5b, the self-attention cache in state. The plan's bar, "5b matches 5a token for token": every run must
    pick the 5a wrapper's tokens, with (b)'s score bars.

    1. The PyTorch stateful wrapper vs the 5a wrapper: the same math, so the scores should match exactly.
    2. decoder_step.mlpackage with a fresh state, under ALL and CPU_AND_NE, vs the wrapper; and, for information, vs
       5a's float16 package on the same chips, which differs only in where the cache lives.
    3. On CPU_AND_NE, what Swift does: one state per segment, kept across 2 transcript updates, here starting full of
       noise, with noise in the unused cross buffer frames too. Compared with the fresh-state run."""
    if not DECODER_STEP_MODEL_PATH.exists():
        raise FileNotFoundError(f"{DECODER_STEP_MODEL_PATH} (run python -m step5_decoder.convert first)")
    float16 = FLOAT16_PACKAGE_INPUT_DATA_TYPES["k_cross"]
    float16_pass_bar = {"minimum_allowed_cosine_similarity": COREML_FLOAT16_MIN_COSINE_SIMILARITY}
    results = []

    print("[5b PyTorch: the stateful wrapper vs the 5a wrapper]")
    for segment in segments:
        passed, _ = compare_with_reference(
            f"  {segment.name}",
            StatefulCacheSession(stateful_wrapper_predict(stateful_wrapper), segment, np.float32, UnusedRows()),
            wrapper_transcripts[segment.name],
            segment.token_cap,
            maximum_allowed_max_abs_diff=WRAPPER_VS_PYTORCH_MAX_ABS_DIFF,
        )
        results.append(passed)

    fresh_state_transcripts_on_neural_engine = {}
    for description, compute_units in (
        ("all compute units", coremltools.ComputeUnit.ALL),
        ("CPU and Neural Engine", coremltools.ComputeUnit.CPU_AND_NE),
    ):
        print(f"[5b Core ML decoder_step.mlpackage, {description}, a fresh state per segment, vs wrapper]")
        stateful_model = coremltools.models.MLModel(str(DECODER_STEP_MODEL_PATH), compute_units=compute_units)
        explicit_cache_model = coremltools.models.MLModel(str(EXPLICIT_CACHE_MODEL_PATH), compute_units=compute_units)
        fresh_state = stateful_model.make_state()
        for state_name in STATE_SHAPES:
            values = fresh_state.read_state(state_name)
            print(
                f"  a fresh state's {state_name}: {values.shape} {values.dtype}, all zeros: {not values.any()}, "
                f"all finite: {bool(np.isfinite(values).all())}"
            )
        for segment in segments:
            passed, stateful_transcript = compare_with_reference(
                f"  {segment.name}",
                StatefulCacheSession(
                    coreml_stateful_predict(stateful_model, stateful_model.make_state()), segment, float16,
                    UnusedRows(),
                ),
                wrapper_transcripts[segment.name],
                segment.token_cap,
                **float16_pass_bar,
            )
            results.append(passed)
            if compute_units == coremltools.ComputeUnit.CPU_AND_NE:
                fresh_state_transcripts_on_neural_engine[segment.name] = stateful_transcript
            explicit_cache_transcript = decode(
                ExplicitCacheSession(explicit_cache_model.predict, segment, float16), segment.token_cap
            )
            if explicit_cache_transcript.tokens == stateful_transcript.tokens:
                compare_scores(
                    "      for information, vs 5a's float16 package on the same chips",
                    stateful_transcript.logits,
                    explicit_cache_transcript.logits,
                )

    print(
        "[5b Core ML, CPU and Neural Engine: one state per segment, starting full of noise, kept across 2 transcript "
        "updates; noise in unused cross frames too; vs the fresh-state run]"
    )
    stateful_model = coremltools.models.MLModel(
        str(DECODER_STEP_MODEL_PATH), compute_units=coremltools.ComputeUnit.CPU_AND_NE
    )
    for segment in segments:
        unused_rows = UnusedRows(noise=True)
        state = stateful_model.make_state()
        for state_name, state_shape in STATE_SHAPES.items():
            # coremltools reads and writes state as float32 arrays (a float16 one is refused); Core ML keeps float16.
            state.write_state(state_name, unused_rows(state_shape))
        session = StatefulCacheSession(coreml_stateful_predict(stateful_model, state), segment, float16, unused_rows)
        for update_number in (1, 2):
            passed, _ = compare_with_reference(
                f"  {segment.name}, update {update_number}",
                session,
                fresh_state_transcripts_on_neural_engine[segment.name],
                segment.token_cap,
                **float16_pass_bar,
            )
            results.append(passed)
    return results


def main() -> int:
    torch.set_grad_enabled(False)  # nothing here trains
    model = load_pytorch_model()
    wrapper = DecoderStepWithExplicitCache(model).eval()  # shares the model's modules, which generate also runs
    stateful_wrapper = DecoderStepWithStatefulCache(wrapper).eval()
    segments = load_test_segments()

    print("[shapes and sizes]")
    check_shapes(wrapper, segments)
    print("[(a) wrapper vs model.generate, greedy, on the same memory]")
    results, wrapper_transcripts = check_a_wrapper_vs_generate(model, wrapper, segments)
    results += check_b_coreml_vs_wrapper(segments, wrapper_transcripts)
    print("[(c) decoder_kv.ort vs wrapper]")
    results += check_c_ort_vs_wrapper(segments, wrapper_transcripts)
    results += check_5b_stateful_cache(stateful_wrapper, segments, wrapper_transcripts)

    if not all(results):
        print("\nFAILED")
        return 1
    np.savez(
        STEP5_REFERENCE_PATH,
        **{
            f"{segment.name}_{key}": value
            for segment in segments
            for key, value in (
                ("tokens", np.array(wrapper_transcripts[segment.name].tokens, dtype=np.int32)),
                ("memory_frame_count", np.array(segment.memory_frame_count, dtype=np.int32)),
            )
        },
    )
    print(f"\nAll checks passed. Saved {STEP5_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
