"""Measure the checks that decide whether a take is speech, the right length, and confident.

    python -m scripts.evaluate_take_checks

Three questions, answered from speech already on disk:

  1. Silence. The app treats a take as "no speech" when the model hears fewer than
     2 sounds, or fewer than 1 sound per second of what cleanup called speech. How
     close does real speech get to that?
  2. Length. A practice take is kept only if the sounds heard number between 0.5 and
     1.5 times the target's. How many genuine takes does that drop?
  3. Confidence. How confident is the model on accented speech and on single words,
     clean and in noise, and what does each cutoff drop?

Sources:

  speechocean    SpeechOcean762 test utterances (cached transcriptions and their audio)
  vctk           VCTK speakers cached for the accent work (same)
  single words   the human reference recordings in web/public/audio, run through the
                 model here as the app would capture them: 0.35 s before, 0.2 s after,
                 a quiet room, then the default cleanup. Also with added noise.

Speech seconds come from the same speech finder the app uses. Sentences are not
single words, so the single-word recordings are the closest thing on disk to a
practice take; they are all native speakers.

Writes Docs/model/take_checks.json.
"""
import io
import json
import math
from pathlib import Path

import numpy as np

from pronunciation.audio_cleanup import FRAME, HIGH_PASS_HZ, HOP, SAMPLE_RATE, clean_take, frame_levels, speech_frames
from pronunciation.lexicon import UnknownWord, word_phonemes
from pronunciation.phonemes import strip_stress
from pronunciation.scoring import align
from scripts.accent_data import ACCENT_SPEAKERS, AUDIO_DIR, FEATURES_DIR
from scripts.evaluate import DEFAULT_PARQUET, PREDICTIONS_PATH
from scripts.evaluate_match import LOW_HUMAN_SCORE, as_words, load_speechocean, load_vctk

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "take_checks.json"
MODEL_DIR = ROOT_DIR / "results" / "multitask-phoneme-model"
REFERENCE_AUDIO = ROOT_DIR / "web" / "public" / "audio"

# The rules in the app (web/src/engine/takeChecks.js and contributions.js).
MIN_SOUNDS = 2
MIN_SOUNDS_PER_SECOND = 1.0
LENGTH_RATIO = (0.5, 1.5)
CONFIDENCE_CUTOFFS = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
SEED = 5


def speech_seconds(audio: np.ndarray) -> float:
    """Seconds of speech as the app's speech finder measures them (0 when it finds none)."""
    from scipy.signal import lfilter

    # The same first-order high-pass as audio_cleanup.high_pass, vectorized.
    rc = 1.0 / (2 * np.pi * HIGH_PASS_HZ)
    alpha = rc / (rc + 1.0 / SAMPLE_RATE)
    filtered = lfilter([alpha, -alpha], [1.0, -alpha], audio.astype(np.float64))
    found = speech_frames(frame_levels(filtered))
    if found is None:
        return 0.0
    indices = np.flatnonzero(found[0])
    return (min(len(audio), int(indices[-1]) * HOP + FRAME) - int(indices[0]) * HOP) / SAMPLE_RATE


def spread(values, cutoffs=(), below=True) -> dict:
    values = np.asarray(values, dtype=np.float64)
    summary = {
        "takes": int(len(values)),
        "lowest": round(float(values.min()), 3),
        "percentile_1": round(float(np.percentile(values, 1)), 3),
        "percentile_5": round(float(np.percentile(values, 5)), 3),
        "median": round(float(np.median(values)), 3),
    }
    for cutoff in cutoffs:
        summary[f"share_below_{cutoff}"] = round(float(np.mean(values < cutoff)), 4)
    return summary


def allowed_lengths(target_sounds: int) -> tuple[int, int]:
    return math.ceil(LENGTH_RATIO[0] * target_sounds), math.floor(LENGTH_RATIO[1] * target_sounds)


def length_drops(pairs: list[tuple]) -> dict:
    """Share of (target, heard) takes the practice length check would drop."""
    too_few = sum(len(heard) < allowed_lengths(len(target))[0] for target, heard in pairs)
    too_many = sum(len(heard) > allowed_lengths(len(target))[1] for target, heard in pairs)
    return {
        "takes": len(pairs),
        "dropped_too_few_sounds": round(too_few / len(pairs), 4),
        "dropped_extra_sounds": round(too_many / len(pairs), 4),
        "dropped": round((too_few + too_many) / len(pairs), 4),
    }


def speechocean_takes() -> list[dict]:
    """Cached transcription, confidence and human score of each test utterance, with its speech seconds."""
    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf

    predictions = [json.loads(line) for line in PREDICTIONS_PATH.read_text().splitlines()]
    takes = []
    parquet = pq.ParquetFile(DEFAULT_PARQUET)
    for batch in parquet.iter_batches(batch_size=64, columns=["audio"]):
        for row in batch.to_pylist():
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
            prediction = predictions[len(takes)]
            takes.append({
                "sounds": len(prediction["predicted"]), "confidence": prediction["confidence"],
                "score": prediction["human_accuracy"], "speech_seconds": speech_seconds(librosa.util.normalize(audio)),
            })
    return takes


def vctk_takes(accent: str) -> list[dict]:
    import soundfile as sf

    takes = []
    for speaker in ACCENT_SPEAKERS[accent]:
        for index, meta in enumerate(json.loads((FEATURES_DIR / f"{speaker}.json").read_text())):
            audio, _ = sf.read(AUDIO_DIR / f"{speaker}_{index:02d}.flac", dtype="float32")
            takes.append({
                "sounds": len(meta["predicted"]), "confidence": meta["confidence"],
                "speech_seconds": speech_seconds(audio),
            })
    return takes


def rate_summary(takes: list[dict]) -> dict:
    found = [take for take in takes if take["speech_seconds"] > 0]
    rates = [take["sounds"] / take["speech_seconds"] for take in found]
    summary = spread(rates, cutoffs=(MIN_SOUNDS_PER_SECOND, 2.0, 3.0))
    summary["no_speech_found"] = len(takes) - len(found)
    summary["called_silence_by_the_app"] = round(float(np.mean([
        take["speech_seconds"] == 0 or take["sounds"] < MIN_SOUNDS
        or take["sounds"] / take["speech_seconds"] < MIN_SOUNDS_PER_SECOND
        for take in takes
    ])), 4)
    return summary


def single_words() -> dict:
    """The reference recordings as practice takes: clean, and with noise 10 dB and 5 dB below the speech."""
    import librosa
    import torch
    from transformers import Wav2Vec2Processor

    from pronunciation.model import ctc_signals
    from src.multitask_model import Wav2Vec2ForPronunciationAssessment

    model = Wav2Vec2ForPronunciationAssessment.from_pretrained(str(MODEL_DIR)).eval()
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    special_ids = set(tokenizer.all_special_ids) | {blank_id}

    def hear(audio):
        inputs = processor(librosa.util.normalize(audio), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model.lm_head(model.wav2vec2(inputs).last_hidden_state)
        confidence, _ = ctc_signals(logits, logits.argmax(dim=-1), blank_id)
        ids, previous, sounds = logits[0].argmax(dim=-1).tolist(), None, []
        for token_id in ids:
            if token_id != previous and token_id not in special_ids:
                sounds.append(tokenizer.convert_ids_to_tokens(token_id))
            previous = token_id
        return sounds, confidence

    rng = np.random.default_rng(SEED)
    conditions = {"quiet_room": None, "noise_10db_below": 10.0, "noise_5db_below": 5.0}
    rows = {name: [] for name in conditions}
    for path in sorted(REFERENCE_AUDIO.glob("*/*.mp3")):
        accent, word = path.parent.name, path.stem
        try:
            target = word_phonemes(word, accent)
        except UnknownWord:
            continue
        speech, _ = librosa.load(path, sr=SAMPLE_RATE)
        level = float(np.sqrt(np.mean(speech ** 2)))
        capture = np.concatenate([np.zeros(int(0.35 * SAMPLE_RATE)), speech, np.zeros(int(0.2 * SAMPLE_RATE))])
        for name, below in conditions.items():
            noise = 10 ** (-55 / 20) if below is None else level * 10 ** (-below / 20)
            take = clean_take((capture + rng.standard_normal(len(capture)) * noise).astype(np.float32))
            sounds, confidence = hear(take.samples)
            ref, hyp = [strip_stress(p) for p in target], [strip_stress(p) for p in sounds]
            rows[name].append({
                "target": target, "heard": sounds, "confidence": confidence, "speech_seconds": take.speech_seconds,
                "sounds": len(sounds), "found": take.found_speech, "snr_db": take.snr_db,
                "matched": 100 * align(ref, hyp).hits / len(ref),
            })
    return rows


def single_word_summary(rows: list[dict]) -> dict:
    pairs = [(row["target"], row["heard"]) for row in rows]
    spoken = [row for row in rows if row["found"] and row["speech_seconds"] > 0]
    return {
        "takes": len(rows),
        "sounds_per_second": spread([row["sounds"] / row["speech_seconds"] for row in spoken],
                                    cutoffs=(MIN_SOUNDS_PER_SECOND, 2.0, 3.0)),
        "called_silence_by_the_app": round(float(np.mean([
            not row["found"] or row["sounds"] < MIN_SOUNDS
            or row["sounds"] / max(row["speech_seconds"], 1e-9) < MIN_SOUNDS_PER_SECOND for row in rows])), 4),
        "length": length_drops(pairs),
        "confidence": spread([row["confidence"] for row in rows], cutoffs=CONFIDENCE_CUTOFFS),
        "measured_snr_below_20_db": round(float(np.mean([row["snr_db"] < 20 for row in rows])), 4),
        "under_30_percent_heard_as_written": round(float(np.mean([row["matched"] < 30 for row in rows])), 4),
    }


def main():
    speechocean = speechocean_takes()
    vctk = {accent: vctk_takes(accent) for accent in ACCENT_SPEAKERS}
    low = [take for take in speechocean if take["score"] <= LOW_HUMAN_SCORE]

    so_utterances = load_speechocean()
    scottish, irish = load_vctk("scottish"), load_vctk("irish")
    words = single_words()

    results = {
        "rules": {
            "silence": f"fewer than {MIN_SOUNDS} sounds, or fewer than {MIN_SOUNDS_PER_SECOND:g} sound per second of speech",
            "length": f"sounds heard between {LENGTH_RATIO[0]} and {LENGTH_RATIO[1]} times the target's",
        },
        "sounds_per_second": {
            "speechocean_all": rate_summary(speechocean),
            "speechocean_low_score": rate_summary(low),
            **{f"vctk_{accent}": rate_summary(takes) for accent, takes in vctk.items()},
            "single_words_quiet_room": single_word_summary(words["quiet_room"])["sounds_per_second"],
        },
        "length_check_drops": {
            "words_3_to_5_sounds": {
                "speechocean_all": length_drops(as_words(so_utterances)),
                "speechocean_low_score": length_drops(
                    as_words(so_utterances, lambda u, i: u["word_scores"][i] <= LOW_HUMAN_SCORE)),
                "vctk_scottish": length_drops(as_words(scottish)),
                "vctk_irish": length_drops(as_words(irish)),
            },
            "note": "words are cut from sentences by alignment, which cannot give a word more sounds than "
                    "the transcription holds near it; the single-word recordings below are a fairer test",
        },
        "confidence": {
            "sentences": {
                "speechocean_all": spread([t["confidence"] for t in speechocean], cutoffs=CONFIDENCE_CUTOFFS),
                "speechocean_low_score": spread([t["confidence"] for t in low], cutoffs=CONFIDENCE_CUTOFFS),
                "vctk_scottish": spread([t["confidence"] for t in vctk["scottish"]], cutoffs=CONFIDENCE_CUTOFFS),
                "vctk_irish": spread([t["confidence"] for t in vctk["irish"]], cutoffs=CONFIDENCE_CUTOFFS),
            },
            "note": "confidence is how sure the model is of the sounds it heard; it does not depend on the "
                    "target, so it cannot tell a wrong word from the right one",
        },
        "single_words": {name: single_word_summary(rows) for name, rows in words.items()},
    }
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=1))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
