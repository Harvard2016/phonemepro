"""Train the pronunciation score head on top of the frozen fine-tuned phoneme model.

Retraining the whole multitask model takes hours on a GPU. The encoder already
knows phonemes, so this keeps it frozen, caches one pooled embedding per
utterance, and fits only the small regression head against human accuracy
ratings. The result is saved as results/multitask-phoneme-model, which the API
loads in preference to the CTC-only model.

    python -m scripts.train_score_head features   # cache embeddings for train + test (about 10 min on CPU)
    python -m scripts.train_score_head train      # fit the head, evaluate, save the model
"""
import argparse
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.evaluate import iter_utterances, pearson

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "speechocean762-hf"
CACHE_DIR = ROOT_DIR / "results" / "score_head"
SOURCE_MODEL = ROOT_DIR / "results" / "finetuned-phoneme-model"
OUTPUT_MODEL = ROOT_DIR / "results" / "multitask-phoneme-model"
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "eval_results.json"
SEED = 7


def parquet_path(split: str) -> Path:
    path = DATA_DIR / "data" / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            "mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset", local_dir=DATA_DIR,
        )
    return path


def extract_features(split: str):
    import librosa
    import soundfile as sf
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    from pronunciation.model import SAMPLE_RATE, ctc_signals
    from pronunciation.scoring import score_attempt

    processor = Wav2Vec2Processor.from_pretrained(str(SOURCE_MODEL))
    model = Wav2Vec2ForCTC.from_pretrained(str(SOURCE_MODEL)).eval()
    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    special_ids = set(tokenizer.all_special_ids)

    embeddings, targets, speakers, heuristic = [], [], [], []
    started = time.time()

    for index, row in enumerate(iter_utterances(parquet_path(split))):
        speech, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr != SAMPLE_RATE:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
        speech = librosa.util.normalize(speech)

        inputs = processor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        with torch.no_grad():
            hidden = model.wav2vec2(inputs).last_hidden_state
            logits = model.lm_head(hidden)

        pred_ids = torch.argmax(logits, dim=-1)
        confidence, margin = ctc_signals(logits, pred_ids, blank_id)
        predicted, previous = [], None
        for token_id in pred_ids[0].tolist():
            if token_id != previous and token_id != blank_id and token_id not in special_ids:
                predicted.append(tokenizer.convert_ids_to_tokens(token_id))
            previous = token_id
        canonical = [phone for word in row["words"] for phone in word["phones"]]

        # Same pooling the multitask model applies before its score head.
        embeddings.append(hidden.mean(dim=1)[0].numpy())
        targets.append(row["accuracy"] / 10.0)
        speakers.append(row["speaker"])
        heuristic.append(score_attempt(canonical, predicted, confidence, margin)["score"] / 10.0)

        if (index + 1) % 250 == 0:
            print(f"{split}: {index + 1} utterances, {time.time() - started:.0f}s", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_DIR / f"{split}.npz",
        embeddings=np.stack(embeddings),
        targets=np.array(targets, dtype=np.float32),
        speakers=np.array(speakers),
        heuristic=np.array(heuristic, dtype=np.float32),
    )
    print(f"Saved {CACHE_DIR / f'{split}.npz'}")


def build_head(hidden_size: int, dropout: float) -> nn.Sequential:
    """Same layout as Wav2Vec2ForPronunciationAssessment.score_head."""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, 1),
    )


def fit_head(x_train, y_train, x_dev, y_dev, dropout: float, epochs: int = 400):
    """Full-batch AdamW with early stopping on the dev speakers."""
    torch.manual_seed(SEED)
    head = build_head(x_train.shape[1], dropout)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    best_state, best_loss, best_epoch = None, float("inf"), 0

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(head(x_train).squeeze(-1), y_train)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            dev_loss = nn.functional.mse_loss(head(x_dev).squeeze(-1), y_dev).item()
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        elif epoch - best_epoch >= 40:
            break

    head.load_state_dict(best_state)
    head.eval()
    return head, best_epoch


def train():
    from transformers import Wav2Vec2Processor

    from pronunciation.scoring import LEARNED_SCORE_WEIGHT, combine_scores
    from src.multitask_model import Wav2Vec2ForPronunciationAssessment

    train_data = np.load(CACHE_DIR / "train.npz")
    test_data = np.load(CACHE_DIR / "test.npz")

    # Hold out every fifth training speaker for early stopping.
    speakers = sorted(set(train_data["speakers"].tolist()))
    dev_speakers = set(speakers[::5])
    is_dev = np.array([s in dev_speakers for s in train_data["speakers"]])

    x_all = torch.tensor(train_data["embeddings"])
    y_all = torch.tensor(train_data["targets"])
    x_train, y_train = x_all[~is_dev], y_all[~is_dev]
    x_dev, y_dev = x_all[is_dev], y_all[is_dev]
    x_test = torch.tensor(test_data["embeddings"])
    y_test = test_data["targets"].tolist()

    model = Wav2Vec2ForPronunciationAssessment.from_pretrained(str(SOURCE_MODEL))
    head, stopped_at = fit_head(x_train, y_train, x_dev, y_dev, dropout=model.config.final_dropout)

    with torch.no_grad():
        dev_pred = head(x_dev).squeeze(-1).tolist()
        test_pred = head(x_test).squeeze(-1).clamp(0, 1).tolist()

    combined = [combine_scores(h * 10, p * 10) for h, p in zip(test_data["heuristic"].tolist(), test_pred)]

    results = {
        "method": "regression head on frozen encoder embeddings, trained on the train split",
        "train_utterances": int((~is_dev).sum()),
        "dev_utterances": int(is_dev.sum()),
        "stopped_at_epoch": stopped_at,
        "dev_correlation": round(pearson(dev_pred, y_dev.tolist()), 3),
        "test_correlation": round(pearson(test_pred, y_test), 3),
        "test_correlation_alignment_score": round(pearson(test_data["heuristic"].tolist(), y_test), 3),
        "blend_weight": LEARNED_SCORE_WEIGHT,
        "test_correlation_app_score": round(pearson(combined, y_test), 3),
        "test_mae_points_app_score": round(float(np.mean(np.abs(np.array(combined) / 10 - np.array(y_test)))) * 10, 2),
    }
    print(json.dumps(results, indent=2))

    model.score_head.load_state_dict(head.state_dict())
    model.save_pretrained(str(OUTPUT_MODEL))
    Wav2Vec2Processor.from_pretrained(str(SOURCE_MODEL)).save_pretrained(str(OUTPUT_MODEL))
    trainer_state = SOURCE_MODEL / "trainer_state.json"
    if trainer_state.exists():
        shutil.copy(trainer_state, OUTPUT_MODEL / "trainer_state.json")
    print(f"Saved {OUTPUT_MODEL}")

    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["learned_score"] = results
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Updated {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["features", "train"])
    args = parser.parse_args()

    if args.stage == "features":
        for split in ("train", "test"):
            extract_features(split)
    else:
        train()


if __name__ == "__main__":
    main()
