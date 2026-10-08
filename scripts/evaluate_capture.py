"""Measure what messy recordings do to the model, and whether cleanup fixes it.

    python -m scripts.evaluate_capture

Takes SpeechOcean762 test utterances and degrades them the way a laptop take is
degraded: silence and room noise around the speech, steady background noise,
another voice in the background, and a clipped first 100 ms (recording started
late). Each version is scored raw and after each cleanup step alone and combined,
so the defaults in pronunciation/audio_cleanup.py can be chosen from the table.

Reports phoneme error rate and how far the learned score moves from the clean
recording. Writes Docs/model/capture_results.json.
"""
import io
import json
from pathlib import Path

import numpy as np
import torch

from pronunciation.audio_cleanup import clean_take
from pronunciation.phonemes import strip_stress
from pronunciation.scoring import align
from scripts.evaluate import DEFAULT_PARQUET, iter_utterances

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "results" / "multitask-phoneme-model"
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "capture_results.json"
UTTERANCES = 120
SAMPLE_RATE = 16000
PAD_SECONDS = 0.7
ROOM_NOISE_DB = -50.0
SEED = 3


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)


def noise_at(rng, length: int, level_db: float) -> np.ndarray:
    return (rng.standard_normal(length) * 10 ** (level_db / 20)).astype(np.float32)


def degrade(speech: np.ndarray, other: np.ndarray, condition: str, rng) -> np.ndarray:
    """Return `speech` as it might arrive from a real microphone."""
    pad = int(PAD_SECONDS * SAMPLE_RATE)
    if condition == "clean":
        return speech
    if condition == "late_start":
        return speech[int(0.1 * SAMPLE_RATE):]

    padded = np.concatenate([np.zeros(pad, dtype=np.float32), speech, np.zeros(pad, dtype=np.float32)])
    padded = padded + noise_at(rng, len(padded), ROOM_NOISE_DB)
    if condition == "silence_around":
        return padded
    if condition.startswith("noise_"):
        snr = float(condition.split("_")[1].removesuffix("db"))
        return padded + noise_at(rng, len(padded), 20 * np.log10(rms(speech)) - snr)
    if condition == "second_voice":
        # Someone else talking in the room, 12 dB quieter, running the whole time.
        background = np.resize(other, len(padded)) * (rms(speech) / rms(other)) * 10 ** (-12 / 20)
        return padded + background.astype(np.float32)
    raise ValueError(condition)


def main():
    import librosa
    import soundfile as sf
    from transformers import Wav2Vec2Processor

    from src.multitask_model import Wav2Vec2ForPronunciationAssessment

    model = Wav2Vec2ForPronunciationAssessment.from_pretrained(str(MODEL_DIR)).eval()
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    special_ids = set(tokenizer.all_special_ids) | {blank_id}

    def run(audio: np.ndarray):
        inputs = processor(librosa.util.normalize(audio), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        with torch.no_grad():
            hidden = model.wav2vec2(inputs).last_hidden_state
            ids = model.lm_head(hidden)[0].argmax(dim=-1).tolist()
            score = float(model.score_head(hidden.mean(dim=1)).item()) * 10
        phonemes, previous = [], None
        for token_id in ids:
            if token_id != previous and token_id != blank_id and token_id not in special_ids:
                phonemes.append(strip_stress(tokenizer.convert_ids_to_tokens(token_id)))
            previous = token_id
        return phonemes, score

    rows = []
    for index, row in enumerate(iter_utterances(DEFAULT_PARQUET)):
        if index >= UTTERANCES + 1:
            break
        speech, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != SAMPLE_RATE:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
        rows.append((speech, [strip_stress(p) for word in row["words"] for p in word["phones"]]))

    conditions = ["clean", "silence_around", "noise_25db", "noise_15db", "noise_5db", "late_start"]
    variants = {
        "raw": lambda audio: audio,
        "filter_only": lambda audio: clean_take(audio, filter_rumble=True, trim=False, denoise=False).samples,
        "trim_0.12s": lambda audio: clean_take(audio, filter_rumble=False, pad_seconds=0.12, denoise=False).samples,
        "trim_0.5s": lambda audio: clean_take(audio, filter_rumble=False, pad_seconds=0.5, denoise=False).samples,
        "denoise_only": lambda audio: clean_take(audio, filter_rumble=False, trim=False, denoise=True).samples,
        "trim_0.5s_denoise": lambda audio: clean_take(audio, filter_rumble=False, pad_seconds=0.5, denoise=True).samples,
    }
    totals = {c: {v: {"errors": 0, "phones": 0, "score_shift": []} for v in variants} for c in conditions}
    not_found = {c: 0 for c in conditions}
    rng = np.random.default_rng(SEED)

    for index in range(UTTERANCES):
        speech, reference = rows[index]
        other = rows[index + 1][0]
        _, clean_score = run(speech)
        for condition in conditions:
            degraded = degrade(speech, other, condition, rng)
            not_found[condition] += int(not clean_take(degraded).found_speech)
            for name, variant in variants.items():
                heard, score = run(variant(degraded))
                bucket = totals[condition][name]
                bucket["errors"] += align(reference, heard).errors
                bucket["phones"] += len(reference)
                bucket["score_shift"].append(abs(score - clean_score))
        if (index + 1) % 50 == 0:
            print(f"{index + 1} utterances", flush=True)

    results = {
        "utterances": UTTERANCES,
        "conditions": {
            condition: {
                "speech_not_found": not_found[condition],
                **{
                    name: {
                        "per": round(bucket["errors"] / bucket["phones"], 4),
                        "score_shift_points": round(float(np.mean(bucket["score_shift"])), 2),
                    }
                    for name, bucket in by_variant.items()
                },
            }
            for condition, by_variant in totals.items()
        },
    }
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{'condition':16} " + " ".join(f"{name:>19}" for name in variants))
    for condition, entry in results["conditions"].items():
        cells = " ".join(
            f"{entry[name]['per'] * 100:5.1f}% {entry[name]['score_shift_points']:4.2f}pt     " for name in variants)
        print(f"{condition:16} {cells}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
