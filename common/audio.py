from pathlib import Path

import numpy as np
import soundfile

from common.config import SAMPLE_RATE_HZ


def load_audio(path: Path) -> np.ndarray:
    """A 16 kHz mono WAV as float32 samples in [-1, 1], shape [sample_count]."""
    samples, sample_rate = soundfile.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE_HZ:
        raise ValueError(f"{path}: expected {SAMPLE_RATE_HZ} Hz, got {sample_rate} Hz")
    if samples.ndim != 1:
        raise ValueError(f"{path}: expected mono, got {samples.shape[1]} channels")
    return samples
