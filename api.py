"""
FastAPI backend for the Pronunciation Practice web app.
Loads the trained wav2vec2 model once and serves pronunciation analysis.
"""
import csv
import io
import json
import os
import re
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from db import clear_history, get_history, get_today_count, save_attempt
from pronunciation.model_info import build_model_info
from pronunciation.phonemes import strip_stress
from pronunciation.scoring import score_attempt
from word_list import DIFFICULTY_LEVELS, PRACTICE_WORDS, get_all_words, get_random_word

ROOT_DIR = Path(__file__).resolve().parent
MIN_RECORDING_SECONDS = 0.3
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# A phoneme needs this many attempts before it can be called a weak spot.
MIN_ATTEMPTS_FOR_INSIGHT = 2
MAX_CUSTOM_TEXT_CHARS = 60
MAX_CUSTOM_TEXT_WORDS = 8


def get_recognizer(request: Request):
    """Return the shared recognizer, loading it on first use."""
    if getattr(request.app.state, "recognizer", None) is None:
        from pronunciation.model import PhonemeRecognizer
        request.app.state.recognizer = PhonemeRecognizer()
    return request.app.state.recognizer


def get_g2p(request: Request):
    """Return the shared grapheme-to-phoneme converter, loading it on first use."""
    if getattr(request.app.state, "g2p", None) is None:
        from pronunciation.model import load_g2p
        request.app.state.g2p = load_g2p()
    return request.app.state.g2p


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup so the first request is not slow.
    if getattr(app.state, "recognizer", None) is None:
        from pronunciation.model import PhonemeRecognizer
        app.state.recognizer = PhonemeRecognizer()
    yield


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Pronunciation Practice API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_speech(audio_bytes: bytes):
    """Decode uploaded audio to normalized 16 kHz mono samples."""
    import librosa

    # librosa needs a file path to decode compressed browser recordings.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        speech, _ = librosa.load(tmp_path, sr=16000)
    finally:
        os.remove(tmp_path)

    return librosa.util.normalize(speech)


def build_insights(attempts: list[dict]) -> dict:
    """Aggregate per-phoneme accuracy from history and recommend words to practice."""
    stats = defaultdict(lambda: {"attempts": 0, "errors": 0})

    for attempt in attempts:
        for phoneme in attempt["target_phonemes"]:
            stats[strip_stress(phoneme)]["attempts"] += 1
        for error in attempt["errors"]:
            for phoneme in error.get("expected", []):
                stats[strip_stress(phoneme)]["errors"] += 1

    phonemes = sorted(
        (
            {
                "phoneme": phoneme,
                "attempts": s["attempts"],
                "errors": s["errors"],
                "error_rate": round(min(1.0, s["errors"] / s["attempts"]), 2),
            }
            for phoneme, s in stats.items()
            if s["attempts"] >= MIN_ATTEMPTS_FOR_INSIGHT
        ),
        key=lambda item: (-item["error_rate"], -item["attempts"]),
    )
    weak = [item for item in phonemes if item["error_rate"] > 0][:5]
    weights = {item["phoneme"]: item["error_rate"] for item in weak}

    ranked = []
    for entry in PRACTICE_WORDS:
        focus = [p for p in dict.fromkeys(strip_stress(p) for p in entry["phonemes"]) if p in weights]
        if focus:
            ranked.append((sum(weights[p] for p in focus), {**entry, "focus": focus}))
    ranked.sort(key=lambda pair: -pair[0])

    return {
        "attempts": len(attempts),
        "phonemes": phonemes,
        "weak_phonemes": weak,
        "recommended_words": [entry for _, entry in ranked[:6]],
    }


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health(request: Request):
    recognizer = getattr(request.app.state, "recognizer", None)
    return {"status": "ok", "model_loaded": recognizer is not None}


@app.get("/api/word")
def random_word(level: str = Query("medium")):
    """Return a random practice word with its canonical phonemes."""
    return get_random_word(level.lower())


@app.get("/api/words")
def all_words(level: str | None = Query(None)):
    """Return practice words, optionally filtered by difficulty."""
    normalized_level = level.lower() if level else None
    return get_all_words(normalized_level)


@app.get("/api/levels")
def levels():
    """Return the supported difficulty levels."""
    return [{"id": level, "label": level.title()} for level in DIFFICULTY_LEVELS]


@app.get("/api/phonemes")
def phonemes_for_text(request: Request, text: str = Query(..., min_length=1)):
    """Return dictionary phonemes for a word or short phrase the learner typed."""
    cleaned = re.sub(r"[^a-z' ]+", " ", text.lower())
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise HTTPException(status_code=422, detail="Type a word or short phrase using letters.")
    if len(cleaned) > MAX_CUSTOM_TEXT_CHARS or len(cleaned.split()) > MAX_CUSTOM_TEXT_WORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Keep it to {MAX_CUSTOM_TEXT_WORDS} words and {MAX_CUSTOM_TEXT_CHARS} characters.",
        )

    known = next((w for w in PRACTICE_WORDS if w["word"] == cleaned), None)
    if known:
        return known

    g2p = get_g2p(request)
    if g2p is None:
        raise HTTPException(status_code=503, detail="Pronunciation lookup is unavailable on this server.")
    phonemes = [p for p in g2p(cleaned) if p.strip() and p[0].isalnum()]
    if not phonemes:
        raise HTTPException(status_code=422, detail="Could not work out how to pronounce that.")
    return {"word": cleaned, "phonemes": phonemes, "difficulty": "custom"}


@app.get("/api/history")
def history(limit: int = Query(50, ge=1, le=10000)):
    """Return recent practice attempts."""
    return get_history(limit)


@app.get("/api/history/today-count")
def history_today_count():
    """Return the number of real attempts made today."""
    return {"count": get_today_count()}


@app.delete("/api/history")
def delete_history(source: str | None = Query(None)):
    """Delete all history, or only 'real' / 'sample' rows."""
    clear_history(source)
    return {"status": "ok"}


@app.get("/api/history/export")
def export_history():
    """Return practice history as a CSV file download."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL)
    writer.writerow(["word", "score", "wer", "target_phonemes", "predicted_phonemes", "errors", "timestamp", "source"])
    for row in get_history(limit=10000):
        writer.writerow([
            row["word"],
            row["score"],
            row["wer"],
            " ".join(row["target_phonemes"]),
            " ".join(row["predicted_phonemes"]),
            len(row["errors"]),
            row["timestamp"],
            row["source"],
        ])
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=practice-history.csv"},
    )


@app.get("/api/insights")
def insights():
    """Return per-phoneme accuracy from real attempts and words that target weak sounds."""
    attempts = [a for a in get_history(limit=10000) if a["source"] == "real"]
    return build_insights(attempts)


@app.get("/api/model-info")
def model_info(request: Request):
    """Return training metrics, evaluation results, and model configuration."""
    recognizer = get_recognizer(request)
    model_path = recognizer.local_path
    # Do not leak absolute server paths to the browser.
    source = str(model_path.relative_to(ROOT_DIR)) if model_path and model_path.is_relative_to(ROOT_DIR) else recognizer.source
    return build_model_info(recognizer.mode, source, model_path)


@app.post("/api/analyze")
async def analyze(
    request: Request,
    audio: UploadFile = File(...),
    word: str = Form(...),
    phonemes: str = Form("[]"),
):
    """
    Analyze a recorded audio file against expected phonemes.
    - audio: WAV/WebM audio file
    - word: the target word (string)
    - phonemes: JSON-encoded list of canonical phonemes (used only for words outside the built-in list)
    """
    # The frontend state might be stale (e.g. user switched words rapidly),
    # so the server's canonical phonemes win for known words.
    server_word = next((w for w in PRACTICE_WORDS if w["word"].lower() == word.lower()), None)
    if server_word:
        target_phonemes = server_word["phonemes"]
    else:
        try:
            target_phonemes = json.loads(phonemes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="phonemes must be a JSON list of ARPABET symbols.")
        if not isinstance(target_phonemes, list) or not target_phonemes or not all(isinstance(p, str) for p in target_phonemes):
            raise HTTPException(status_code=422, detail="phonemes must be a non-empty JSON list of ARPABET symbols.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The recording was empty. Hold the button while you speak.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Recording is too large. Keep it under 10 MB.")

    try:
        speech = load_speech(audio_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the recording. Try recording again.")

    if len(speech) < MIN_RECORDING_SECONDS * 16000:
        raise HTTPException(status_code=400, detail="That recording was too short. Hold the button a little longer.")

    recognizer = get_recognizer(request)
    try:
        recognition = recognizer.transcribe(speech)
    except Exception as exc:
        print(f"Error analyzing audio: {exc}")
        raise HTTPException(status_code=500, detail="The model could not analyze that recording.")

    result = score_attempt(
        target_phonemes,
        recognition.phonemes,
        recognition.confidence,
        recognition.margin,
        native_score=recognition.native_score,
        phoneme_model=recognizer.predicts_phonemes,
        word=word,
        recognized_text=recognition.text,
        spans=recognition.spans,
    )

    save_attempt(word, target_phonemes, recognition.phonemes, result["score"], result["wer"], result["errors"])

    return {
        "word": word,
        "target_phonemes": target_phonemes,
        "predicted_phonemes": recognition.phonemes,
        **result,
    }


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
