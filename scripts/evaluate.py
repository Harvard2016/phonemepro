"""Evaluate the phoneme model on the SpeechOcean762 test split.

Two stages, so scoring changes can be re-measured without re-running the model:

    python -m scripts.evaluate infer     # run the model, cache per-utterance predictions
    python -m scripts.evaluate report    # compute metrics -> Docs/model/eval_results.json

The test split is read from the Hugging Face parquet export
(mispeech/speechocean762, data/test-00000-of-00001.parquet).
"""
import argparse
import io
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = ROOT_DIR / "data" / "speechocean762-hf" / "data" / "test-00000-of-00001.parquet"
PREDICTIONS_PATH = ROOT_DIR / "results" / "eval" / "predictions.jsonl"
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "eval_results.json"

# A canonical phone counts as mispronounced when annotators scored it below this (scale 0-2).
MISPRONOUNCED_BELOW = 1.5


def iter_utterances(parquet_path: Path):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=64):
        yield from batch.to_pylist()


def infer(parquet_path: Path, limit: int | None):
    import librosa
    import soundfile as sf

    from pronunciation.model import SAMPLE_RATE, PhonemeRecognizer

    recognizer = PhonemeRecognizer()
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with open(PREDICTIONS_PATH, "w") as out:
        for index, row in enumerate(iter_utterances(parquet_path)):
            if limit is not None and index >= limit:
                break

            speech, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if speech.ndim > 1:
                speech = speech.mean(axis=1)
            if sr != SAMPLE_RATE:
                speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
            speech = librosa.util.normalize(speech)

            recognition = recognizer.transcribe(speech)

            canonical = []
            phone_scores = []
            for word in row["words"]:
                canonical.extend(word["phones"])
                phone_scores.extend(word["phones-accuracy"])

            out.write(json.dumps({
                "speaker": row["speaker"],
                "text": row["text"],
                "human_accuracy": row["accuracy"],
                "human_total": row["total"],
                "canonical": canonical,
                "phone_scores": phone_scores,
                "predicted": recognition.phonemes,
                "confidence": recognition.confidence,
                "margin": recognition.margin,
                "seconds": len(speech) / SAMPLE_RATE,
            }) + "\n")

            if (index + 1) % 250 == 0:
                print(f"{index + 1} utterances, {time.time() - started:.0f}s", flush=True)

    print(f"Wrote {PREDICTIONS_PATH} ({recognizer.mode}: {recognizer.source})")


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / math.sqrt(var_x * var_y)


def report():
    from pronunciation.phonemes import strip_stress
    from pronunciation.scoring import align, score_attempt

    rows = [json.loads(line) for line in open(PREDICTIONS_PATH)]

    strict_errors = lenient_errors = total_phones = 0
    confusions = Counter()
    per_phone = defaultdict(lambda: {"count": 0, "errors": 0})
    detection = Counter()
    app_scores, strict_scores, human_scores = [], [], []

    for row in rows:
        canonical = row["canonical"]
        predicted = row["predicted"]
        total_phones += len(canonical)

        strict_errors += align(canonical, predicted).errors

        ref = [strip_stress(p) for p in canonical]
        hyp = [strip_stress(p) for p in predicted]
        lenient = align(ref, hyp)
        lenient_errors += lenient.errors

        flagged = [True] * len(ref)
        for op in lenient.ops:
            if op.kind == "equal":
                flagged[op.ref_index] = False
            elif op.kind == "substitute":
                confusions[(ref[op.ref_index], hyp[op.hyp_index])] += 1

        for phone, is_flagged, human in zip(ref, flagged, row["phone_scores"]):
            per_phone[phone]["count"] += 1
            per_phone[phone]["errors"] += int(is_flagged)
            mispronounced = human < MISPRONOUNCED_BELOW
            if mispronounced and is_flagged:
                detection["tp"] += 1
            elif mispronounced:
                detection["fn"] += 1
            elif is_flagged:
                detection["fp"] += 1
            else:
                detection["tn"] += 1

        result = score_attempt(canonical, predicted, row["confidence"], row["margin"])
        app_scores.append(result["score"])
        strict_scores.append(max(0.0, 1 - align(canonical, predicted).errors / max(len(canonical), 1)))
        human_scores.append(row["human_accuracy"])

    tp, fp, fn, tn = (detection[k] for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    results = {
        "dataset": "speechocean762 test split",
        "utterances": len(rows),
        "speakers": len({row["speaker"] for row in rows}),
        "phones": total_phones,
        "audio_minutes": round(sum(row["seconds"] for row in rows) / 60, 1),
        "per": round(lenient_errors / total_phones, 4),
        "per_with_stress": round(strict_errors / total_phones, 4),
        "score_correlation": {
            "app_score_vs_human_accuracy": round(pearson(app_scores, human_scores), 3),
            "stress_strict_accuracy_vs_human_accuracy": round(pearson(strict_scores, human_scores), 3),
        },
        "mispronunciation_detection": {
            "definition": f"canonical phone flagged by alignment vs human phone score < {MISPRONOUNCED_BELOW} (0-2 scale)",
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "positives": tp + fn,
            "negatives": fp + tn,
            "false_alarm_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        },
        "top_confusions": [
            {"expected": expected, "heard": heard, "count": count}
            for (expected, heard), count in confusions.most_common(15)
        ],
        "per_phoneme": sorted(
            (
                {
                    "phoneme": phone,
                    "count": stats["count"],
                    "error_rate": round(stats["errors"] / stats["count"], 3),
                }
                for phone, stats in per_phone.items()
            ),
            key=lambda item: item["error_rate"],
            reverse=True,
        ),
    }

    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    summary = {k: v for k, v in results.items() if k not in ("top_confusions", "per_phoneme")}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["infer", "report"])
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.stage == "infer":
        infer(args.parquet, args.limit)
    else:
        report()


if __name__ == "__main__":
    main()
