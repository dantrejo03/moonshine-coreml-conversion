import onnxruntime
import torch
from transformers import MoonshineStreamingForConditionalGeneration

from common.config import HUGGING_FACE_MODEL_ID, MOONSHINE_ORT_MODEL_DIRECTORY


def load_pytorch_model() -> MoonshineStreamingForConditionalGeneration:
    """Float32, in eval mode. Eager attention keeps the masks explicit and the traced graph simple."""
    return MoonshineStreamingForConditionalGeneration.from_pretrained(
        HUGGING_FACE_MODEL_ID, attn_implementation="eager", dtype=torch.float32
    ).eval()


def load_ort_session(file_name: str, intra_op_thread_count: int | None = None) -> onnxruntime.InferenceSession:
    """One of Moonshine's .ort pieces (common.config.MOONSHINE_ORT_PIECE_FILE_NAMES), run on the CPU.

    intra_op_thread_count: None keeps onnxruntime's default, one thread per core, which Steps 1-5 used. Moonshine's
    runtime runs each op on 1 thread (moonshine-streaming-model.cpp 166); the thread count can change the order float
    numbers are added in, so Step 6 uses 1 to reproduce Moonshine's tokens."""
    session_options = onnxruntime.SessionOptions()
    if intra_op_thread_count is not None:
        session_options.intra_op_num_threads = intra_op_thread_count
    return onnxruntime.InferenceSession(
        str(MOONSHINE_ORT_MODEL_DIRECTORY / file_name), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
