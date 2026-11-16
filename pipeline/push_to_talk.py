"""Step 6: the push-to-talk loop, run with only the five Core ML models. This is the state and the calls Step 7 ports to
Swift (README.md says what each part keeps):

    transcriber = PushToTalkTranscriber(load_pipeline_models(), MoonshineTokenizer())
    transcriber.press()
    while the button is held:
        transcriber.add_chunk(the next 1280 samples)
        each time the 0.5 s timer fires: transcriber.update().text           tentative text
    transcriber.release(the samples left over, fewer than 1280).text          final text
"""

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import coremltools
import numpy as np

from common.config import (
    AUDIO_SAMPLES_PER_CHUNK,
    CONVERSION_DIRECTORY,
    DECODER_HEAD_DIM,
    DECODER_PADDED_HEAD_DIM,
    END_TOKEN_ID,
    ENCODER_DIM,
    FEATURE_FRAMES_PER_CHUNK,
    FRONTEND_CONV1_BUFFER_SHAPE,
    FRONTEND_CONV2_BUFFER_SHAPE,
    MAXIMUM_MEMORY_FRAME_COUNT,
    MEMORY_FRAMES_PER_CALL,
    SAMPLE_RATE_HZ,
    attention_cache_shape,
)
from pipeline.tokenizer import MoonshineTokenizer
from step1_frontend.convert import FRONTEND_MODEL_PATH
from step1_frontend.test import coreml_chunk_runner
from step2_encoder.convert import ENCODER_MODEL_PATH
from step2_encoder.test import coreml_window_runner, encoder_window
from step3_adapter.convert import ADAPTER_MODEL_PATH
from step3_adapter.test import coreml_call_runner as adapter_call_runner
from step4_cross_kv.convert import CROSS_KV_MODEL_PATH
from step4_cross_kv.test import coreml_call_runner as cross_kv_call_runner
from step5_decoder.convert import DECODER_STEP_MODEL_PATH, FLOAT16_PACKAGE_INPUT_DATA_TYPES
from step5_decoder.test import decode, token_and_position_inputs, token_cap_for_seconds

# One press is one segment, and holds at most Mmax = 768 frames: 192 chunks, 15.36 s, its last chunk padded. Version 1
# of the app releases at this cap.
MAXIMUM_PRESS_SAMPLE_COUNT = MAXIMUM_MEMORY_FRAME_COUNT // FEATURE_FRAMES_PER_CHUNK * AUDIO_SAMPLES_PER_CHUNK

# Stage names in StageTimes, besides the five models' own calls.
CHUNK_STAGE_NAME = "add_chunk: stages 1-4 for one 80 ms chunk"
UPDATE_STAGE_NAME = "update: one decode from scratch"


@dataclass
class PipelineModels:
    """The five float16 packages that ship to the watch, each loaded once."""

    frontend: coremltools.models.MLModel
    encoder: coremltools.models.MLModel
    adapter: coremltools.models.MLModel
    cross_kv: coremltools.models.MLModel
    decoder_step: coremltools.models.MLModel


def load_pipeline_models(compute_units=coremltools.ComputeUnit.CPU_AND_NE) -> PipelineModels:
    """CPU_AND_NE is what Step 7's app loads (.cpuAndNeuralEngine); on the Mac, ALL would run the encoder on the GPU
    (Step 2, deviation 5). Prints how long each took to load: Core ML compiles a package for this Mac on loading."""
    models = {}
    for name, model_path in (
        ("frontend", FRONTEND_MODEL_PATH),
        ("encoder", ENCODER_MODEL_PATH),
        ("adapter", ADAPTER_MODEL_PATH),
        ("cross_kv", CROSS_KV_MODEL_PATH),
        ("decoder_step", DECODER_STEP_MODEL_PATH),
    ):
        step_folder_name = model_path.parent.parent.name
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run python -m {step_folder_name}.convert first)")
        started_seconds = time.perf_counter()
        models[name] = coremltools.models.MLModel(str(model_path), compute_units=compute_units)
        print(
            f"  {model_path.relative_to(CONVERSION_DIRECTORY)}: loaded in "
            f"{time.perf_counter() - started_seconds:.1f} s"
        )
    return PipelineModels(**models)


class StageTimes:
    """Wall-clock milliseconds of every call, by stage, in the order the stages first ran."""

    def __init__(self):
        self.milliseconds_by_stage: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        started_seconds = time.perf_counter()
        try:
            yield
        finally:
            self.milliseconds_by_stage[stage_name].append((time.perf_counter() - started_seconds) * 1000)


@dataclass
class TranscriptUpdate:
    """One decode from scratch: what the screen shows."""

    tokens: list[int]  # the decoder's picks, in order; the last is the end token unless the token cap came first
    text: str
    token_cap: int  # the most decoder steps it was allowed
    decode_milliseconds: float

    @property
    def ended_with_end_token(self) -> bool:
        return bool(self.tokens) and self.tokens[-1] == END_TOKEN_ID


class DecoderSessionOverLiveBuffers:
    """Step 5's DecoderSession (step5_decoder/test.py) for one press: decoder_step.mlpackage with the press's MLState,
    reading the cross-attention buffers as they are at each call. As in Step 5's StatefulCacheSession, nothing is
    reset between transcript updates: each step writes its own row of the state before reading it, and the rows after
    it are masked."""

    def __init__(
        self,
        decoder_step_model: coremltools.models.MLModel,
        cross_attention_inputs: dict[str, np.ndarray],
        stage_times: StageTimes,
    ):
        self.decoder_step_model = decoder_step_model
        self.state = decoder_step_model.make_state()  # Swift: makeState(), once per press
        self.cross_attention_inputs = cross_attention_inputs
        self.stage_times = stage_times

    def start_transcript(self) -> None:
        pass

    def step(self, token: int, position: int) -> np.ndarray:
        with self.stage_times.measure("decoder_step"):
            outputs = self.decoder_step_model.predict(
                {**token_and_position_inputs(token, position), **self.cross_attention_inputs}, state=self.state
            )
        return outputs["logits"][0, 0]


class PushToTalkTranscriber:
    """Everything Swift keeps between calls (the plan's mirror of MoonshineStreamingState), and the four things the
    app does: press, add_chunk, update, release. One press is one segment, at most MAXIMUM_PRESS_SAMPLE_COUNT samples.

    Call press() before the first chunk."""

    def __init__(self, models: PipelineModels, tokenizer: MoonshineTokenizer):
        self.tokenizer = tokenizer
        self.stage_times = StageTimes()
        self.decoder_step_model = models.decoder_step
        # Each earlier step's own Core ML runner, so every model is called exactly as its step tested it.
        self.run_frontend = coreml_chunk_runner(models.frontend)
        self.run_encoder = coreml_window_runner(models.encoder)
        self.run_adapter = adapter_call_runner(models.adapter)
        self.run_cross_kv = cross_kv_call_runner(models.cross_kv)

        # Allocated and zeroed once, never cleared, as in Swift: on a later press, the frames past its memory still
        # hold an earlier press's keys and values, blocked by cross_valid. k_cross and v_cross are float16, as
        # decoder_step.mlpackage takes them; Step 4's model outputs float32, converted as it is copied in. Each row's
        # 40 numbers are padded with zeros to 64 (common.config.DECODER_PADDED_HEAD_DIM).
        cross_kv_data_type = FLOAT16_PACKAGE_INPUT_DATA_TYPES["k_cross"]
        cross_attention_shape = attention_cache_shape(MAXIMUM_MEMORY_FRAME_COUNT, DECODER_PADDED_HEAD_DIM)
        self.cross_attention_inputs = {
            "k_cross": np.zeros(cross_attention_shape, dtype=cross_kv_data_type),
            "v_cross": np.zeros(cross_attention_shape, dtype=cross_kv_data_type),
            "cross_valid": np.zeros((1, MAXIMUM_MEMORY_FRAME_COUNT), dtype=np.float32),
        }
        # Every feature frame of the press. Each becomes one encoded frame and one memory row, so Mmax rows fit a press.
        self.features = np.zeros((1, MAXIMUM_MEMORY_FRAME_COUNT, ENCODER_DIM), dtype=np.float32)

    def press(self) -> None:
        """Button down: a new segment. Everything kept per segment starts over; the cross K/V buffers keep their old
        contents behind cross_valid."""
        self.conv1_buffer = np.zeros(FRONTEND_CONV1_BUFFER_SHAPE, dtype=np.float32)  # Step 1
        self.conv2_buffer = np.zeros(FRONTEND_CONV2_BUFFER_SHAPE, dtype=np.float32)
        self.feature_frame_count = 0
        self.emitted_frame_count = 0  # encoded frames emitted so far (Step 2)
        # Memory rows so far: the adapter's pos_offset (Step 3) and the cross K/V buffers' valid frame count (Step 4).
        self.memory_frame_count = 0
        self.received_sample_count = 0  # real samples, not the last chunk's padding: sets the token cap
        self.cross_attention_inputs["cross_valid"][:] = 0.0
        self.decoder_session = DecoderSessionOverLiveBuffers(
            self.decoder_step_model, self.cross_attention_inputs, self.stage_times
        )

    def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """Button held: one 80 ms chunk, [1280] float32, through stages 1-4."""
        if audio_chunk.shape != (AUDIO_SAMPLES_PER_CHUNK,):
            raise ValueError(f"a chunk is {AUDIO_SAMPLES_PER_CHUNK} samples, got shape {audio_chunk.shape}")
        with self.stage_times.measure(CHUNK_STAGE_NAME):
            self._process_chunk(audio_chunk, is_final=False)
        self.received_sample_count += AUDIO_SAMPLES_PER_CHUNK

    def update(self) -> TranscriptUpdate:
        """The update timer fired: decode from scratch over the memory so far, as Moonshine does after every change.
        The start token at position 0, each step's pick fed back, until the end token or the token cap,
        min(ceil(seconds received x 6.5), Lmax)."""
        token_cap = token_cap_for_seconds(self.received_sample_count / SAMPLE_RATE_HZ)
        if self.memory_frame_count == 0:  # nothing to listen to yet; Moonshine returns "" here too
            return TranscriptUpdate(tokens=[], text="", token_cap=token_cap, decode_milliseconds=0.0)
        with self.stage_times.measure(UPDATE_STAGE_NAME):
            transcript = decode(self.decoder_session, token_cap)
        return TranscriptUpdate(
            tokens=transcript.tokens,
            text=self.tokenizer.tokens_to_text(transcript.tokens),
            token_cap=token_cap,
            decode_milliseconds=self.stage_times.milliseconds_by_stage[UPDATE_STAGE_NAME][-1],
        )

    def release(self, remaining_samples: np.ndarray) -> TranscriptUpdate:
        """Button up. remaining_samples: what arrived after the last whole chunk, 0 to 1279 samples. They're padded
        with zeros to a whole chunk, and the encoder runs with isFinal, so the 16 held-back frames (20 with a padded
        chunk) go through the adapter and cross K/V too. Then the final decode."""
        if len(remaining_samples) >= AUDIO_SAMPLES_PER_CHUNK:
            raise ValueError(f"{len(remaining_samples)} remaining samples: whole chunks go through add_chunk")
        if len(remaining_samples):
            padded_chunk = np.zeros(AUDIO_SAMPLES_PER_CHUNK, dtype=np.float32)
            padded_chunk[: len(remaining_samples)] = remaining_samples
            self._process_chunk(padded_chunk, is_final=True)
        else:
            self._encode(is_final=True)
        self.received_sample_count += len(remaining_samples)
        return self.update()

    def _process_chunk(self, audio_chunk: np.ndarray, is_final: bool) -> None:
        """Frontend (4 new features), then the encoder."""
        if self.feature_frame_count + FEATURE_FRAMES_PER_CHUNK > MAXIMUM_MEMORY_FRAME_COUNT:
            raise ValueError(
                f"the press is longer than {MAXIMUM_PRESS_SAMPLE_COUNT / SAMPLE_RATE_HZ} s: version 1 of the app "
                "releases at that cap"
            )
        with self.stage_times.measure("frontend"):
            features, self.conv1_buffer, self.conv2_buffer = self.run_frontend(
                audio_chunk[None], self.conv1_buffer, self.conv2_buffer
            )
        new_feature_frames = slice(self.feature_frame_count, self.feature_frame_count + FEATURE_FRAMES_PER_CHUNK)
        self.features[:, new_feature_frames] = features
        self.feature_frame_count += FEATURE_FRAMES_PER_CHUNK
        self._encode(is_final)

    def _encode(self, is_final: bool) -> None:
        """Step 2's window over the features so far; each group of 4 new encoded frames goes on to the adapter.
        Nothing comes out for the first 4 chunks: they fill the 16 held-back frames."""
        window = encoder_window(self.features[:, : self.feature_frame_count], self.emitted_frame_count, is_final)
        if window is None:
            return
        with self.stage_times.measure("encoder"):
            encoded = self.run_encoder(window.features, window.valid)
        new_encoded_frames = encoded[:, window.new_frames]
        self.emitted_frame_count += window.new_frame_count
        for first_frame in range(0, window.new_frame_count, MEMORY_FRAMES_PER_CALL):
            self._append_memory(new_encoded_frames[:, first_frame : first_frame + MEMORY_FRAMES_PER_CALL])

    def _append_memory(self, encoded_frames: np.ndarray) -> None:
        """4 encoded frames -> adapter at pos_offset = memory rows so far -> 4 memory rows -> cross K/V, written into
        the buffers at the same 4 frames, which cross_valid then lets the decoder see."""
        new_rows = slice(self.memory_frame_count, self.memory_frame_count + MEMORY_FRAMES_PER_CALL)
        with self.stage_times.measure("adapter"):
            memory = self.run_adapter(encoded_frames, self.memory_frame_count)
        with self.stage_times.measure("cross_kv"):
            k_new, v_new = self.run_cross_kv(memory)
        self.cross_attention_inputs["k_cross"][:, :, :, new_rows, :DECODER_HEAD_DIM] = k_new  # float32 -> float16
        self.cross_attention_inputs["v_cross"][:, :, :, new_rows, :DECODER_HEAD_DIM] = v_new
        self.cross_attention_inputs["cross_valid"][:, new_rows] = 1.0
        self.memory_frame_count += MEMORY_FRAMES_PER_CALL
