"""Model loading and inference for phoneme recognition.

Resolution order for the model source:
  1. results/multitask-phoneme-model   (CTC + score regression head)
  2. results/finetuned-phoneme-model   (CTC phoneme model)
  3. PHONEME_MODEL_ID                  (Hugging Face Hub repo with the fine-tuned model)
  4. facebook/wav2vec2-base-960h       (public ASR fallback, heuristic scoring only)
"""
import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.getenv("PHONEME_RESULTS_DIR", ROOT_DIR / "results"))
MULTITASK_MODEL_PATH = RESULTS_DIR / "multitask-phoneme-model"
LOCAL_MODEL_CANDIDATES = [RESULTS_DIR / "finetuned-phoneme-model", RESULTS_DIR]
FALLBACK_ASR_MODEL_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
# wav2vec 2.0 emits one frame per 320 samples.
FRAME_SECONDS = 320 / SAMPLE_RATE
# CTC emits a short spike per phoneme, so the last one is padded to an audible length.
FINAL_PHONEME_FRAMES = 6

MODE_MULTITASK = "multitask"
MODE_FINETUNED = "finetuned"
MODE_ASR_FALLBACK = "asr-fallback"


@dataclass
class Recognition:
    phonemes: list[str]
    text: str
    confidence: float
    margin: float
    native_score: float | None = None
    # Approximate (start, end) seconds for each phoneme; None when the model decodes letters.
    spans: list[tuple[float, float]] | None = None


def _has_weights(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    )


def resolve_model_source() -> tuple[str, str]:
    """Return (mode, source) for the best available model."""
    if (MULTITASK_MODEL_PATH / "config.json").exists():
        return MODE_MULTITASK, str(MULTITASK_MODEL_PATH)
    for path in LOCAL_MODEL_CANDIDATES:
        if _has_weights(path):
            return MODE_FINETUNED, str(path)
    hub_id = os.getenv("PHONEME_MODEL_ID")
    if hub_id:
        return MODE_FINETUNED, hub_id
    return MODE_ASR_FALLBACK, FALLBACK_ASR_MODEL_ID


def pick_device() -> torch.device:
    requested = os.getenv("PHONEME_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ctc_signals(logits: torch.Tensor, pred_ids: torch.Tensor, blank_id: int) -> tuple[float, float]:
    """Estimate recognition confidence and margin from non-blank collapsed CTC token probabilities."""
    probs = torch.softmax(logits, dim=-1)[0]
    token_ids = pred_ids[0].tolist()

    confidences = []
    margins = []
    previous_token = None

    for frame_idx, token_id in enumerate(token_ids):
        if token_id == previous_token:
            continue
        previous_token = token_id

        if token_id == blank_id:
            continue

        frame_probs = probs[frame_idx]
        top_values = torch.topk(frame_probs, k=2).values
        confidences.append(frame_probs[token_id].item())
        margins.append((top_values[0] - top_values[1]).item())

    if not confidences:
        return 0.0, 0.0

    return float(sum(confidences) / len(confidences)), float(sum(margins) / len(margins))


def load_g2p():
    """Load g2p_en quietly so optional NLTK setup noise does not derail demos."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from g2p_en import G2p
            return G2p()
    except Exception as exc:
        print(f"G2P fallback unavailable: {exc}")
        return None


class PhonemeRecognizer:
    """Wraps a wav2vec2 CTC model and turns 16 kHz audio into ARPABET phonemes."""

    def __init__(self, mode: str | None = None, source: str | None = None):
        if mode is None or source is None:
            mode, source = resolve_model_source()
        self.mode = mode
        self.source = source
        self.device = pick_device()

        print(f"Loading {mode} model from {source}...")
        self.processor = Wav2Vec2Processor.from_pretrained(source)
        if mode == MODE_MULTITASK:
            from src.multitask_model import Wav2Vec2ForPronunciationAssessment
            self.model = Wav2Vec2ForPronunciationAssessment.from_pretrained(source)
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

        # The ASR fallback decodes letters, so it needs G2P to get phonemes.
        self.g2p = load_g2p() if mode == MODE_ASR_FALLBACK else None
        print("Model loaded successfully.")

    @property
    def predicts_phonemes(self) -> bool:
        return self.mode in (MODE_MULTITASK, MODE_FINETUNED)

    @property
    def native_scoring(self) -> bool:
        return self.mode == MODE_MULTITASK

    @property
    def local_path(self) -> Path | None:
        path = Path(self.source)
        return path if path.exists() else None

    def _decode_phonemes(self, pred_ids: torch.Tensor) -> tuple[list[str], list[tuple[float, float]]]:
        """Collapse CTC frame IDs into ARPABET tokens and the time span each one covers.

        A phoneme runs from the frame where it is first emitted to the frame where
        the next one starts, which is close enough to play back the right slice.
        """
        tokenizer = self.processor.tokenizer
        blank_id = tokenizer.pad_token_id
        special_ids = set(tokenizer.all_special_ids)
        special_tokens = set(tokenizer.all_special_tokens)
        phonemes = []
        start_frames = []
        previous_token = None
        token_ids = pred_ids[0].detach().cpu().tolist()

        for frame, token_id in enumerate(token_ids):
            if token_id == previous_token:
                continue
            previous_token = token_id

            if token_id == blank_id or token_id in special_ids:
                continue

            token = tokenizer.convert_ids_to_tokens(token_id)
            if token and token not in special_tokens:
                phonemes.append(str(token))
                start_frames.append(frame)

        spans = []
        for index, start in enumerate(start_frames):
            if index + 1 < len(start_frames):
                end = start_frames[index + 1]
            else:
                end = min(start + FINAL_PHONEME_FRAMES, len(token_ids))
            spans.append((round(start * FRAME_SECONDS, 3), round(end * FRAME_SECONDS, 3)))

        return phonemes, spans

    def transcribe(self, speech: np.ndarray) -> Recognition:
        input_values = self.processor(
            speech, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_values.to(self.device)

        native_score = None
        with torch.no_grad():
            if self.native_scoring:
                # Tuple return exposes the regression head output at index 1.
                model_out = self.model(input_values, return_dict=False)
                logits = model_out[0]
                native_score = float(model_out[1].item())
            else:
                logits = self.model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        confidence, margin = ctc_signals(logits, pred_ids, self.processor.tokenizer.pad_token_id)

        spans = None
        if self.predicts_phonemes:
            phonemes, spans = self._decode_phonemes(pred_ids)
            text = " ".join(phonemes)
        else:
            if self.g2p is None:
                raise RuntimeError(
                    "G2P resources are unavailable. Run python download_nltk.py once, then restart the backend."
                )
            text = self.processor.batch_decode(pred_ids)[0].lower()
            phonemes = [p for p in self.g2p(text) if p.strip() and p[0].isalnum()]

        return Recognition(
            phonemes=phonemes,
            text=text,
            confidence=confidence,
            margin=margin,
            native_score=native_score,
            spans=spans,
        )
