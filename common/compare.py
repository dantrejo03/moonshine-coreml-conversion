import numpy as np

from common.config import FEATURE_FRAME_SECONDS

# Starting pass bars from the plan's "Three checks per step" table.
# Log the actual numbers; tighten or loosen a bar only with evidence.
WRAPPER_VS_PYTORCH_MAX_ABS_DIFF = 1e-4  # check (a), float32
COREML_FLOAT32_MAX_ABS_DIFF = 1e-3  # check (b), float32 on CPU only
COREML_FLOAT16_MIN_COSINE_SIMILARITY = 0.999  # check (b), float16 on all compute units
MOONSHINE_ORT_MIN_COSINE_SIMILARITY = 0.99  # check (c), against the quantized .ort piece


def compare(
    label: str,
    actual,
    expected,
    maximum_allowed_max_abs_diff: float | None = None,
    minimum_allowed_cosine_similarity: float | None = None,
    print_result: bool = True,
) -> dict:
    """Prints and returns max abs diff, mean abs diff and cosine similarity of two arrays.

    Accepts numpy arrays or torch tensors. `passed` is False when a given bar is missed.
    """
    actual_values = _to_float64_numpy(actual)
    expected_values = _to_float64_numpy(expected)
    if actual_values.shape != expected_values.shape:
        raise ValueError(f"{label}: shape {actual_values.shape} != expected {expected_values.shape}")

    absolute_difference = np.abs(actual_values - expected_values)
    max_abs_diff = float(absolute_difference.max())
    mean_abs_diff = float(absolute_difference.mean())
    norm_product = np.linalg.norm(actual_values) * np.linalg.norm(expected_values)
    cosine_similarity = (
        float(np.dot(actual_values.ravel(), expected_values.ravel()) / norm_product)
        if norm_product > 0
        else float(max_abs_diff == 0)  # both all-zero counts as identical
    )

    failures = []
    if maximum_allowed_max_abs_diff is not None and not max_abs_diff <= maximum_allowed_max_abs_diff:
        failures.append(f"max abs > {maximum_allowed_max_abs_diff:g}")
    if minimum_allowed_cosine_similarity is not None and not cosine_similarity >= minimum_allowed_cosine_similarity:
        failures.append(f"cosine < {minimum_allowed_cosine_similarity:g}")
    has_bar = maximum_allowed_max_abs_diff is not None or minimum_allowed_cosine_similarity is not None
    verdict = ("FAIL: " + ", ".join(failures)) if failures else ("PASS" if has_bar else "")

    if print_result:
        print(
            f"{label}: max abs {max_abs_diff:.3e}  mean abs {mean_abs_diff:.3e}  "
            f"cosine {cosine_similarity:.6f}  {verdict}".rstrip()
        )
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_similarity": cosine_similarity,
        "passed": not failures,
    }


def worst_frame_cosine_similarity(actual: np.ndarray, expected: np.ndarray) -> tuple[float, int]:
    """Lowest cosine over single frames (the last axis holds one frame's numbers), and that frame's index."""
    actual_frames = actual.reshape(-1, actual.shape[-1]).astype(np.float64)
    expected_frames = expected.reshape(-1, expected.shape[-1]).astype(np.float64)
    cosine_per_frame = (actual_frames * expected_frames).sum(axis=-1) / (
        np.linalg.norm(actual_frames, axis=-1) * np.linalg.norm(expected_frames, axis=-1)
    )
    worst_frame_index = int(np.argmin(cosine_per_frame))  # the first NaN, if there is one
    return float(cosine_per_frame[worst_frame_index]), worst_frame_index


def compare_with_worst_frame(label: str, actual: np.ndarray, expected: np.ndarray, **pass_bar) -> bool:
    """compare(), plus the worst single frame: a whole-recording cosine is dominated by loud speech and can hide
    a few bad quiet frames. For arrays of 20 ms frames (features, encoded frames), so it prints the frame's time."""
    result = compare(label, actual, expected, **pass_bar)
    worst_cosine, worst_frame_index = worst_frame_cosine_similarity(actual, expected)
    print(f"      worst single frame: cosine {worst_cosine:.6f} at {worst_frame_index * FEATURE_FRAME_SECONDS:.2f} s")
    return result["passed"]


def _to_float64_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):  # torch tensor
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)
