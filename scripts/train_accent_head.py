"""Train and evaluate the accent head on the cached VCTK features.

    python -m scripts.accent_data features     # transcriptions for the error-rate table
    python -m scripts.accent_data layers       # pooled encoder statistics for the accent head
    python -m scripts.train_accent_head        # train, evaluate, save

Two things are measured on speakers the head never trained on:

  1. Accent identification. A small classifier guesses which accent a recording
     is closest to, from the mean and spread of one encoder layer over time. The
     last layer is tuned to phonemes and has shed most accent cues (it scores 38%
     where chance is 25%), so the layer is chosen on held-out development
     speakers. Practice recordings are a word or two long, so the head trains on
     one-second windows and is scored on those and on whole sentences.
  2. Phoneme error rate by accent, against the American and the British
     dictionary. If the phoneme model hears accent differences, southern English
     speakers should match the British dictionary better than the American one.

Writes results/accent_head.pt and the "accents" section of Docs/model/eval_results.json.
"""
import collections
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pronunciation.lexicon import UnknownWord, text_phonemes
from pronunciation.phonemes import strip_stress
from pronunciation.scoring import align
from scripts.accent_data import ACCENT_SPEAKERS, FEATURES_DIR, LAYER_DIR, PROBE_LAYERS, PROBE_SPEAKERS, test_speakers

ROOT_DIR = Path(__file__).resolve().parent.parent
HEAD_PATH = ROOT_DIR / "results" / "accent_head.pt"
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "eval_results.json"
# Accents the head learns to tell apart. Irish was tried as a fourth class and failed:
# with five training speakers it was right on 3 of 120 held-out clips. It is kept as
# an "unseen accent" probe instead, alongside Australian.
CLASSES = ["english_southern", "american", "scottish"]
TRAINED = {accent: ACCENT_SPEAKERS[accent] for accent in CLASSES}
UNSEEN = {"irish": ACCENT_SPEAKERS["irish"], **PROBE_SPEAKERS}
CLASS_NAMES = {
    "english_southern": "Southern English",
    "american": "American",
    "scottish": "Scottish",
    "irish": "Irish",
}
DEV_SPEAKERS_PER_CLASS = 2
SEED = 11


class AccentHead(nn.Module):
    """Standardizes pooled layer statistics, then classifies them."""

    def __init__(self, features: int, classes: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.net = nn.Sequential(nn.Linear(features, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, classes))

    def forward(self, stats):
        return self.net((stats - self.mean) / self.scale)


def load_stats(speaker: str, layer_index: int) -> np.ndarray:
    """[utterances, windows, 2 * hidden]: window 0 is the whole clip, 1-3 are one-second windows."""
    stats = np.load(LAYER_DIR / f"{speaker}.npy").astype(np.float32)[:, :, layer_index]
    return stats.reshape(stats.shape[0], stats.shape[1], -1)


def fit(x_train, y_train, x_dev, y_dev, classes: int, epochs: int = 400):
    torch.manual_seed(SEED)
    head = AccentHead(x_train.shape[1], classes)
    head.mean.copy_(x_train.mean(dim=0))
    head.scale.copy_(x_train.std(dim=0).clamp_min(1e-4))
    optimizer = torch.optim.AdamW(head.net.parameters(), lr=1e-3, weight_decay=5e-2)
    # Classes have different numbers of speakers; weight the loss so none dominates.
    counts = torch.bincount(y_train, minlength=classes).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (classes * counts))
    best_state, best_accuracy, best_epoch = None, -1.0, 0

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        loss_fn(head(x_train), y_train).backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            accuracy = balanced_accuracy(head(x_dev).argmax(dim=1), y_dev, classes)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_state = {key: value.clone() for key, value in head.state_dict().items()}
        elif epoch - best_epoch >= 40:
            break

    head.load_state_dict(best_state)
    return head.eval(), best_accuracy


def balanced_accuracy(predicted, actual, classes: int) -> float:
    recalls = [
        (predicted[actual == c] == c).float().mean().item()
        for c in range(classes) if (actual == c).any()
    ]
    return float(np.mean(recalls))


def phoneme_error_rates(utterances) -> dict:
    """PER of the model's transcription against each dictionary, per accent."""
    totals = collections.defaultdict(lambda: {"errors": 0, "phones": 0, "utterances": 0})
    for accent, meta in utterances:
        try:
            references = {name: text_phonemes(meta["text"], name) for name in ("ga", "rp")}
        except UnknownWord:
            continue  # a word outside the shared lexicon: skip so both dictionaries see the same data
        heard = [strip_stress(p) for p in meta["predicted"]]
        for name, reference in references.items():
            bucket = totals[(accent, name)]
            bucket["errors"] += align([strip_stress(p) for p in reference], heard).errors
            bucket["phones"] += len(reference)
            bucket["utterances"] += 1

    table = {}
    for (accent, name), bucket in totals.items():
        table.setdefault(accent, {"utterances": bucket["utterances"]})[f"per_vs_{name}"] = round(
            bucket["errors"] / bucket["phones"], 4)
    return table


def main():
    held_out = {accent: test_speakers(speakers) for accent, speakers in ACCENT_SPEAKERS.items()}
    # More speakers per class are set aside to choose the layer and when to stop training.
    dev = {
        accent: set(sorted(set(speakers) - held_out[accent])[:DEV_SPEAKERS_PER_CLASS])
        for accent, speakers in ACCENT_SPEAKERS.items()
    }

    def tensors_for(layer_index: int, role: str):
        """(one-second windows, labels, whole clips, labels, speaker of each clip) for a split."""
        windows, window_labels, clips, clip_labels, clip_speakers = [], [], [], [], []
        for label, (accent, speakers) in enumerate(TRAINED.items()):
            for speaker in speakers:
                speaker_role = "test" if speaker in held_out[accent] else "dev" if speaker in dev[accent] else "train"
                if speaker_role != role:
                    continue
                stats = load_stats(speaker, layer_index)
                windows.append(stats[:, 1:].reshape(-1, stats.shape[-1]))
                window_labels += [label] * (stats.shape[0] * (stats.shape[1] - 1))
                clips.append(stats[:, 0])
                clip_labels += [label] * stats.shape[0]
                clip_speakers += [speaker] * stats.shape[0]
        return (torch.tensor(np.concatenate(windows)), torch.tensor(window_labels),
                torch.tensor(np.concatenate(clips)), torch.tensor(clip_labels), clip_speakers)

    # Choose the encoder layer on the development speakers.
    by_layer = {}
    for layer_index, layer in enumerate(PROBE_LAYERS):
        x_train, y_train, *_ = tensors_for(layer_index, "train")
        x_dev, y_dev, *_ = tensors_for(layer_index, "dev")
        head, dev_accuracy = fit(x_train, y_train, x_dev, y_dev, classes=len(CLASSES))
        by_layer[layer] = (dev_accuracy, layer_index, head)
        print(f"layer {layer}: dev accuracy on one-second windows {dev_accuracy:.3f}", flush=True)
    layer = max(by_layer, key=lambda candidate: by_layer[candidate][0])
    _, layer_index, head = by_layer[layer]

    x_window, y_window, x_clip, y_clip, clip_speakers = tensors_for(layer_index, "test")
    with torch.no_grad():
        window_pred = head(x_window).argmax(dim=1)
        clip_pred = head(x_clip).argmax(dim=1)

    confusion = [[int(((y_clip == a) & (clip_pred == p)).sum()) for p in range(len(CLASSES))]
                 for a in range(len(CLASSES))]

    # Speaker-level: does the majority vote over a speaker's utterances name the right accent?
    votes = collections.defaultdict(list)
    for speaker, prediction in zip(clip_speakers, clip_pred.tolist()):
        votes[speaker].append(prediction)
    speaker_label = {s: CLASSES.index(a) for a, speakers in TRAINED.items() for s in speakers}
    speakers_right = sum(collections.Counter(v).most_common(1)[0][0] == speaker_label[s] for s, v in votes.items())

    # Speakers from an accent the head never saw: what does it call them?
    probes = {}
    for accent, speakers in UNSEEN.items():
        guesses = collections.Counter()
        for speaker in speakers:
            if not (LAYER_DIR / f"{speaker}.npy").exists():
                continue
            with torch.no_grad():
                predictions = head(torch.tensor(load_stats(speaker, layer_index)[:, 0])).argmax(dim=1)
            guesses.update(CLASS_NAMES[CLASSES[i]] for i in predictions.tolist())
        total = sum(guesses.values())
        probes[accent] = {name: round(count / total, 3) for name, count in guesses.most_common()} if total else {}

    # Error rates use the first pass's transcriptions, test and probe speakers only.
    per_inputs = []
    for accent, speakers in {**{a: sorted(held_out[a]) for a in ACCENT_SPEAKERS}, **PROBE_SPEAKERS}.items():
        for speaker in speakers:
            meta_path = FEATURES_DIR / f"{speaker}.json"
            if meta_path.exists():
                per_inputs.extend((accent, row) for row in json.loads(meta_path.read_text()))

    results = {
        "dataset": "VCTK 0.92 subset (CC BY 4.0), 60 utterances per speaker",
        "unseen_accent_speakers": {accent: len(speakers) for accent, speakers in UNSEEN.items()},
        "classes": [CLASS_NAMES[c] for c in CLASSES],
        "speakers": {
            CLASS_NAMES[accent]: {
                "train": len(speakers) - len(held_out[accent]) - len(dev[accent]),
                "dev": len(dev[accent]),
                "test": len(held_out[accent]),
            }
            for accent, speakers in TRAINED.items()
        },
        "chance": round(1 / len(CLASSES), 3),
        "encoder_layer": layer,
        "dev_accuracy_by_layer": {str(k): round(v[0], 3) for k, v in by_layer.items()},
        "test_accuracy_one_second": round(balanced_accuracy(window_pred, y_window, len(CLASSES)), 3),
        "test_accuracy_full_utterance": round(balanced_accuracy(clip_pred, y_clip, len(CLASSES)), 3),
        "test_speakers_identified": f"{speakers_right} of {len(votes)}",
        "confusion_full_utterance": confusion,
        "unseen_accent_guesses": probes,
        "phoneme_error_rate_by_dictionary": phoneme_error_rates(per_inputs),
    }
    print(json.dumps(results, indent=2))

    torch.save(
        {"state_dict": head.state_dict(), "classes": CLASSES, "class_names": CLASS_NAMES, "layer": layer},
        HEAD_PATH,
    )
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["accents"] = results
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {HEAD_PATH} and updated {REPORT_PATH}")


if __name__ == "__main__":
    main()
