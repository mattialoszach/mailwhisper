from typing import Optional, Tuple
import numpy as np
from faster_whisper import WhisperModel
import os

_MODEL: WhisperModel | None = None

def load_model() -> WhisperModel:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Defaults:
    # - CPU:  int8  (fast & ok)
    # - CUDA: float16 (fast & good)
    model_name   = os.getenv("WHISPER_LOCAL_MODEL", "medium")   # Future: Experiment with "medium" vs "small"
    device       = os.getenv("WHISPER_DEVICE", "cpu").lower()   # "cpu" or "cuda"
    compute_type = os.getenv("WHISPER_COMPUTE", None)

    if device not in ("cpu", "cuda"):
        device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    _MODEL = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )
    return _MODEL

def transcribe_array(
    audio: np.ndarray,
    *,
    samplerate: int = 16000,
    language: Optional[str] = None,
    beam_size: int = 5,
) -> Tuple[str, Optional[str]]:
    """
    Transcribes a NumPy audio array directly without an extra file.
    """
    model = load_model()

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)

    segments, info = model.transcribe(
        audio,
        beam_size=beam_size,
        language=language,         # None = auto
        # vad_filter=True,         # optional
        # temperature=0.0,         # optional
    )
    text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
    return text, getattr(info, "language", None)