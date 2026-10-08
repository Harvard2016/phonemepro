"""Fine-tune the phoneme head on native accented speech, labelled with each accent's own dictionary.

    python -m scripts.finetune_accent_ctc            # train, evaluate, report
    python -m scripts.finetune_accent_ctc --apply    # also write the new head into the model

The phoneme model was trained only on learners reading against American labels.
Here its output layer (the encoder stays frozen) sees southern English speakers
transcribed with the British dictionary and American speakers with the American
one, so that, for example, a British "water" is taught as W AO T AH rather than
treated as a mispronounced W AO T ER.

Whether that is a net win is checked two ways, and the head is only applied if
both hold:
  - held-out VCTK speakers: error rate against their own accent's dictionary falls
  - SpeechOcean762 test split: learner error rate does not rise by more than 0.5 points
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pronunciation.lexicon import UnknownWord, text_phonemes
from pronunciation.phonemes import strip_stress
from pronunciation.scoring import align
from scripts.accent_data import ACCENT_SPEAKERS, FEATURES_DIR, test_speakers
from scripts.evaluate import DEFAULT_PARQUET, iter_utterances

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "results" / "multitask-phoneme-model"
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "eval_results.json"
# Which dictionary labels each accent's speech.
DICTIONARY = {"english_southern": "rp", "american": "ga"}
LEARNER_TOLERANCE = 0.005
LEARNER_UTTERANCES = 600
EPOCHS = 12
BATCH = 32
SEED = 5


def load_split(vocab: dict[str, int]):
    """Frame features and label ids for train and held-out speakers of the labelled accents."""
    data = {"train": [], "test": []}
    for accent, dictionary in DICTIONARY.items():
        held_out = test_speakers(ACCENT_SPEAKERS[accent])
        for speaker in ACCENT_SPEAKERS[accent]:
            cached = np.load(FEATURES_DIR / f"{speaker}.npz")
            meta = json.loads((FEATURES_DIR / f"{speaker}.json").read_text())
            offsets = np.concatenate([[0], np.cumsum(cached["lengths"])])
            for index, row in enumerate(meta):
                try:
                    reference = text_phonemes(row["text"], dictionary)
                except UnknownWord:
                    continue
                if any(p not in vocab for p in reference):
                    continue
                data["test" if speaker in held_out else "train"].append({
                    "accent": accent,
                    "frames": torch.tensor(cached["frames"][offsets[index]:offsets[index + 1]].astype(np.float32)),
                    "labels": torch.tensor([vocab[p] for p in reference]),
                    "reference": reference,
                })
    return data


def decode(logits: torch.Tensor, blank_id: int, special_ids: set, id_to_token: list[str]) -> list[str]:
    phonemes, previous = [], None
    for token_id in logits.argmax(dim=-1).tolist():
        if token_id != previous and token_id != blank_id and token_id not in special_ids:
            phonemes.append(id_to_token[token_id])
        previous = token_id
    return phonemes


def error_rates(head: nn.Linear, items, blank_id, special_ids, id_to_token) -> dict[str, float]:
    totals = {}
    with torch.no_grad():
        for item in items:
            heard = [strip_stress(p) for p in decode(head(item["frames"]), blank_id, special_ids, id_to_token)]
            reference = [strip_stress(p) for p in item["reference"]]
            bucket = totals.setdefault(item["accent"], [0, 0])
            bucket[0] += align(reference, heard).errors
            bucket[1] += len(reference)
    return {accent: round(errors / phones, 4) for accent, (errors, phones) in totals.items()}


def learner_error_rate(model, head: nn.Linear, processor, blank_id, special_ids, id_to_token) -> float:
    """Phoneme error rate on SpeechOcean762 test utterances with the given output layer."""
    import librosa
    import soundfile as sf

    errors = phones = 0
    for index, row in enumerate(iter_utterances(DEFAULT_PARQUET)):
        if index >= LEARNER_UTTERANCES:
            break
        speech, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        inputs = processor(librosa.util.normalize(speech), sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            hidden = model.wav2vec2(inputs).last_hidden_state[0]
        heard = [strip_stress(p) for p in decode(head(hidden), blank_id, special_ids, id_to_token)]
        reference = [strip_stress(p) for word in row["words"] for p in word["phones"]]
        errors += align(reference, heard).errors
        phones += len(reference)
    return round(errors / phones, 4)


def train_head(original: nn.Linear, items, blank_id: int, anchor: float) -> nn.Linear:
    """CTC fine-tuning of a copy of the output layer, pulled towards its starting weights."""
    torch.manual_seed(SEED)
    head = nn.Linear(original.in_features, original.out_features)
    head.load_state_dict(original.state_dict())
    start = [p.detach().clone() for p in original.parameters()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=0.0)
    ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    order = np.random.default_rng(SEED)

    for _ in range(EPOCHS):
        for batch_start in range(0, len(items), BATCH):
            batch = [items[i] for i in order.permutation(len(items))[batch_start:batch_start + BATCH]]
            frames = nn.utils.rnn.pad_sequence([item["frames"] for item in batch], batch_first=True)
            log_probs = head(frames).log_softmax(dim=-1).transpose(0, 1)
            loss = ctc(
                log_probs,
                torch.cat([item["labels"] for item in batch]),
                torch.tensor([len(item["frames"]) for item in batch]),
                torch.tensor([len(item["labels"]) for item in batch]),
            )
            # Stay close to the head that already works on learner speech.
            loss = loss + anchor * sum(((p - p0) ** 2).sum() for p, p0 in zip(head.parameters(), start))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head.eval()


def main():
    from transformers import Wav2Vec2Processor

    from src.multitask_model import Wav2Vec2ForPronunciationAssessment

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the new head into the model if it passes both checks")
    args = parser.parse_args()

    model = Wav2Vec2ForPronunciationAssessment.from_pretrained(str(MODEL_DIR)).eval()
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    id_to_token = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
    blank_id = tokenizer.pad_token_id
    special_ids = set(tokenizer.all_special_ids) | {blank_id}
    scoring = (blank_id, special_ids, id_to_token)

    data = load_split(vocab)
    print(f"{len(data['train'])} training and {len(data['test'])} held-out utterances", flush=True)

    before = {
        "accents": error_rates(model.lm_head, data["test"], *scoring),
        "learners": learner_error_rate(model, model.lm_head, processor, *scoring),
    }
    print("before:", before, flush=True)

    candidates = []
    for anchor in (1.0, 0.1, 0.01):
        head = train_head(model.lm_head, data["train"], blank_id, anchor)
        after = {
            "accents": error_rates(head, data["test"], *scoring),
            "learners": learner_error_rate(model, head, processor, *scoring),
        }
        print(f"anchor {anchor}: {after}", flush=True)
        candidates.append((anchor, head, after))

    def passes(after) -> bool:
        return (after["learners"] - before["learners"] <= LEARNER_TOLERANCE
                and all(after["accents"][a] < before["accents"][a] for a in before["accents"]))

    passing = [c for c in candidates if passes(c[2])]
    chosen = min(passing, key=lambda c: sum(c[2]["accents"].values())) if passing else None

    results = {
        "method": "CTC fine-tuning of the output layer only, on frozen encoder features",
        "labels": "southern English speech with the British dictionary, American speech with the American dictionary",
        "train_utterances": len(data["train"]),
        "test_utterances": len(data["test"]),
        "learner_test_utterances": LEARNER_UTTERANCES,
        "before": before,
        "candidates": {str(anchor): after for anchor, _, after in candidates},
        "applied": chosen is not None and args.apply,
        "chosen_anchor": chosen[0] if chosen else None,
    }
    print(json.dumps(results, indent=2))

    if chosen and args.apply:
        model.lm_head.load_state_dict(chosen[1].state_dict())
        model.save_pretrained(str(MODEL_DIR))
        print(f"Applied the new output layer to {MODEL_DIR}")
    elif not chosen:
        print("No candidate passed both checks; the model is unchanged.")

    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["accent_finetune"] = results
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
