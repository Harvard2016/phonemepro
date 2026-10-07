import io
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Point the app at a throwaway database before db.py is imported.
_db_dir = tempfile.mkdtemp(prefix="phoneme-tests-")
os.environ["PRACTICE_DB_PATH"] = os.path.join(_db_dir, "test_history.db")


class FakeRecognizer:
    """Stands in for the wav2vec2 model so API tests do not need weights."""

    mode = "finetuned"
    source = "fake-model"
    predicts_phonemes = True
    native_scoring = False
    local_path = None

    def __init__(self):
        self.phonemes = []
        self.confidence = 0.95
        self.margin = 0.9
        self.spans = None

    def transcribe(self, speech):
        from pronunciation.model import Recognition
        return Recognition(
            phonemes=list(self.phonemes),
            text=" ".join(self.phonemes),
            confidence=self.confidence,
            margin=self.margin,
            spans=self.spans,
        )


@pytest.fixture
def recognizer():
    return FakeRecognizer()


@pytest.fixture
def client(recognizer):
    from fastapi.testclient import TestClient

    import api
    import db

    db.clear_history()
    api.app.state.recognizer = recognizer
    # Letter-per-phoneme stand-in for g2p_en, which needs NLTK data.
    api.app.state.g2p = lambda text: [c.upper() if c != " " else " " for c in text]
    # No context manager: the lifespan hook (real model load) must not run.
    yield TestClient(api.app)
    api.app.state.recognizer = None
    api.app.state.g2p = None


@pytest.fixture
def wav_bytes():
    """Half a second of a 440 Hz tone as 16 kHz mono WAV."""
    import math

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        frames = [int(12000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(8000)]
        wav.writeframes(struct.pack(f"<{len(frames)}h", *frames))
    return buffer.getvalue()
