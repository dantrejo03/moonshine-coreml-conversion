"""What Moonshine's own library actually runs for each press: every onnxruntime call it makes, read from its own log.
Run from the top of the repo:

    python -m pipeline.tools.inspect_moonshine_library_calls

Check (B) runs Moonshine's .ort pieces in Python the way moonshine-v2 main's source says its transcriber runs one
segment (pipeline.test.MoonshineOrtPieces): the frontend once per 1280-sample chunk; the encoder, adapter and cross
K/V once each, on the whole segment; the decoder once per token. The pinned library (moonshine-voice 0.1.6) isn't
built from that source (Step 1 found its frontend split in two files). With the option log_ort_run, the library
writes each onnxruntime call and the shape of each input to stderr. This captures that log for each press, with the
same audio and options as (B), and prints per session how many calls it made and the input shapes it saw, plus its
"streaming encode" lines.
"""

import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from common.audio import load_audio
from common.config import SAMPLE_RATE_HZ, TEST_AUDIO_PATHS
from pipeline.test import MOONSHINE_ONE_SEGMENT_OPTIONS, moonshine_transcript_lines, press_boundaries

ORT_RUN_PATTERN = re.compile(r"ORT Run (\S+) took [\d.]+ ms for inputs:")
INPUT_SHAPE_PATTERN = re.compile(r"(\w+) = \[([-\d, ]*)\]\s*$")
LOG_PREFIX_END = "(): "  # every log line starts with file:line:function(): (moonshine-utils/debug-utils.h LOGF)
PRINTED_DISTINCT_SHAPES_MAXIMUM_COUNT = 3  # more distinct shapes than this (a growing cache): first and last only
PRINTED_OTHER_LINES_MAXIMUM_COUNT = 5


@contextmanager
def captured_standard_error() -> Iterator[list[str]]:
    """Everything written to file descriptor 2 inside the block, by Python or by a C library: its lines are appended
    to the yielded list when the block ends."""
    captured_lines = []
    sys.stderr.flush()
    with tempfile.TemporaryFile() as log_file:
        saved_standard_error = os.dup(2)
        os.dup2(log_file.fileno(), 2)
        try:
            yield captured_lines
        finally:
            os.dup2(saved_standard_error, 2)
            os.close(saved_standard_error)
            log_file.seek(0)
            captured_lines.extend(log_file.read().decode(errors="replace").splitlines())


def print_library_calls(log_lines: list[str]) -> None:
    """Calls per session in the order sessions first ran, the shapes each input had, then the encode lines."""
    call_counts_by_session = Counter()
    shape_counts_by_session_input = defaultdict(Counter)  # (session, input name) -> {shape text: calls}
    encode_lines = []
    other_lines = []
    current_session = None
    for line in log_lines:
        if run_match := ORT_RUN_PATTERN.search(line):
            current_session = run_match.group(1)
            call_counts_by_session[current_session] += 1
        elif current_session and (shape_match := INPUT_SHAPE_PATTERN.search(line)):
            shape_counts_by_session_input[(current_session, shape_match.group(1))][f"[{shape_match.group(2)}]"] += 1
        elif "streaming encode" in line:
            encode_lines.append(line.split(LOG_PREFIX_END, 1)[-1])
        else:
            other_lines.append(line)

    for session_name, call_count in call_counts_by_session.items():
        print(f"      {session_name}: {call_count} calls")
        for (shape_session_name, input_name), shape_counts in shape_counts_by_session_input.items():
            if shape_session_name != session_name:
                continue
            shapes = list(shape_counts)
            if len(shapes) <= PRINTED_DISTINCT_SHAPES_MAXIMUM_COUNT:
                shapes_text = ", ".join(f"{shape} x{count}" for shape, count in shape_counts.items())
            else:
                shapes_text = f"{len(shapes)} different shapes, from {shapes[0]} to {shapes[-1]}"
            print(f"        {input_name}: {shapes_text}")
    for line in encode_lines:
        print(f"      {line}")
    if other_lines:
        print(f"      {len(other_lines)} other log lines, the first {PRINTED_OTHER_LINES_MAXIMUM_COUNT}:")
        for line in other_lines[:PRINTED_OTHER_LINES_MAXIMUM_COUNT]:
            print(f"        {line}")


def main() -> int:
    library_options = {**MOONSHINE_ONE_SEGMENT_OPTIONS, "log_ort_run": "true"}
    for name, audio_path in TEST_AUDIO_PATHS.items():
        samples = load_audio(audio_path)
        print(f"[{name}]")
        for press_index, (start_sample, end_sample) in enumerate(press_boundaries(samples)):
            print(f"  press {press_index + 1}: {start_sample / SAMPLE_RATE_HZ:.2f} to {end_sample / SAMPLE_RATE_HZ:.2f} s")
            with captured_standard_error() as log_lines:
                transcript_lines = moonshine_transcript_lines(samples[start_sample:end_sample], library_options)
            for transcript_line in transcript_lines:
                print(f"    text: {transcript_line.text}")
            print_library_calls(log_lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
